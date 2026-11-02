"""Reading pycisTopic's pickles without running anything from them.

`pickle.load` calls whatever callable the file names, so loading a pickle
from elsewhere runs its author's code. pycisTopic saves its objects only as
pickles, and the objects hold nothing but numpy arrays, scipy sparse
matrices, pandas frames, builtins, and its own classes. So the file is read
by an unpickler that resolves an allowlist of those names and refuses any
other: an `os.system` in the file is refused by name, never called.

The allowed names are not harmless whatever they are given, so what they
are given is checked too, before numpy, scipy or pandas act on it:

* numpy's dtypes and arrays must hold the state numpy itself writes for
  them (numpy trusts a pickled dtype's flags and item size, which would let
  raw bytes from the file stand for object pointers), and each is given its
  state once, before anything else holds it;
* classes are created empty and given their state, as every pickle of them
  does; they are never constructed from a file's arguments, so pandas never
  parses a dtype or a timezone from it (`dateutil/<path>` opens that path);
* containers are filled only through `dict`, `list` and `set`, never by
  writing into an array that has been checked;
* once the whole file is read, every sparse matrix's indices must lie within
  its shape and its `indptr` must rise from 0 to its entries, every
  Categorical's and MultiIndex's codes within its categories, and every
  nullable array's mask must match its values: scipy's and pandas' compiled
  code indexes with them unchecked.

That keeps a crafted file from corrupting memory or reading files through
what it names, but it is still parsed by numpy, scipy and pandas: a pickle
is trusted input, to be read only from a source one would trust with the
data itself.

The pure-Python unpickler is used, so that each opcode can be checked; it
is a few times slower than the C one, which is small beside writing the
data.

pycisTopic itself is not needed. Its classes become inert stand-ins that
hold the restored attributes (`__dict__`); none of its code exists here to
run. numpy, scipy and pandas are resolved to whatever versions are
installed: numpy 2's `numpy._core` for a numpy 1 pickle's `numpy.core`, and
pandas' own table of renamed classes (`pandas.compat.pickle_compat`) for a
pickle written by pycisTopic's pinned pandas 1.5.

`sniff` tells a pycisTopic pickle by its first opcodes, read with
`pickletools`, which parses a pickle and executes none of it.
"""

from __future__ import annotations

import _compat_pickle
import copyreg
import gzip
import importlib
import pickle
import pickletools
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, FrozenSet, Iterable, Iterator, Optional, Tuple

import numpy as np

#: Module prefix of pycisTopic's classes, as its pickles name them.
PYCISTOPIC = "pycisTopic."

#: Opcodes read before deciding a pickle is not pycisTopic's: its object is
#: the outermost one, so its class is named within the first few.
SNIFF_OPCODES = 64

#: Globals an object's reconstruction names before its class, in protocols
#: 0 and 1: `copy_reg._reconstructor(cls, object, state)`.
_HELPERS = frozenset({
    ("copy_reg", "_reconstructor"), ("copyreg", "_reconstructor"),
    ("__builtin__", "object"), ("builtins", "object"),
})

_NUMPY = {
    "_reconstruct", "scalar", "_frombuffer", "ndarray", "dtype",
}

#: The globals a pycisTopic pickle may name: module -> names. Dates and
#: times are not among them: pandas reads a timezone given as text from a
#: file named in it, and pycisTopic's objects hold none.
ALLOWED: Dict[str, FrozenSet[str]] = {
    **{module: frozenset(_NUMPY) for module in (
        "numpy", "numpy.core.multiarray", "numpy._core.multiarray",
        "numpy.core.numeric", "numpy._core.numeric",
    )},
    **{module: frozenset({f"{kind}_matrix", f"{kind}_array"}) for kind in ("csr", "csc", "coo")
       for module in (f"scipy.sparse._{kind}", f"scipy.sparse.{kind}")},
    "pandas.core.frame": frozenset({"DataFrame"}),
    "pandas.core.series": frozenset({"Series"}),
    "pandas.core.internals.managers": frozenset({"BlockManager", "SingleBlockManager"}),
    "pandas.core.internals.blocks": frozenset({"new_block", "Block", "ObjectBlock"}),
    "pandas._libs.internals": frozenset({"_unpickle_block", "BlockPlacement"}),
    "pandas._libs.arrays": frozenset({"__pyx_unpickle_NDArrayBacked"}),
    "pandas._libs.missing": frozenset({"NA"}),
    "pandas.core.arrays.categorical": frozenset({"Categorical"}),
    "pandas.core.dtypes.dtypes": frozenset({"CategoricalDtype"}),
    "pandas.core.indexes.base": frozenset({"_new_Index", "Index"}),
    "pandas.core.indexes.range": frozenset({"RangeIndex"}),
    "pandas.core.indexes.numeric": frozenset({"Int64Index", "Float64Index", "UInt64Index"}),
    "pandas.core.indexes.category": frozenset({"CategoricalIndex"}),
    "pandas.core.indexes.multi": frozenset({"MultiIndex"}),
    "pandas.core.arrays.integer": frozenset({"IntegerArray", *(
        f"{sign}Int{bits}Dtype" for sign in ("", "U") for bits in (8, 16, 32, 64)
    )}),
    "pandas.core.arrays.floating": frozenset({"FloatingArray", "Float32Dtype", "Float64Dtype"}),
    "pandas.core.arrays.boolean": frozenset({"BooleanArray", "BooleanDtype"}),
    "pandas.core.arrays.string_": frozenset({"StringArray", "StringDtype"}),
    # pandas 3 pickles its classes under their public names.
    "pandas": frozenset({
        "DataFrame", "Series", "Index", "RangeIndex", "MultiIndex", "CategoricalIndex",
        "Categorical", "CategoricalDtype", "StringDtype", "BooleanDtype",
        "Float32Dtype", "Float64Dtype", "NA", *(
            f"{sign}Int{bits}Dtype" for sign in ("", "U") for bits in (8, 16, 32, 64)
        ),
    }),
    "pandas.arrays": frozenset({
        "BooleanArray", "IntegerArray", "FloatingArray", "StringArray", "Categorical",
    }),
    "builtins": frozenset({
        "slice", "set", "frozenset", "complex", "bytes", "bytearray", "range", "object",
    }),
    "collections": frozenset({"OrderedDict"}),
    "copyreg": frozenset({"_reconstructor"}),
    "_codecs": frozenset({"encode"}),
}

#: numpy dtype kinds an array may have: no structured or void dtypes,
#: which nothing pycisTopic holds uses.
_KINDS = frozenset("biufcOUSMm")

#: Classes a pickle calls with arguments, as the allowlist names them: none
#: of them parses a dtype from text. Any other is only ever created empty
#: and given its state.
_CONSTRUCTED = (
    *(("builtins", name) for name in ("slice", "set", "frozenset", "complex", "range", "bytes", "bytearray")),
    ("collections", "OrderedDict"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas._libs.internals", "BlockPlacement"),
    ("pandas.core.arrays.string_", "StringDtype"),  # (storage, na_value), from pandas 3
)

#: What a scipy matrix pickles: its arrays, its shape, and how it prints.
#: The flags scipy caches about its indices are dropped, so that scipy works
#: them out again from the checked arrays.
_SPARSE_STATE = frozenset({"_shape", "data", "indices", "indptr", "coords", "maxprint", "_maxprint"})
_SPARSE_CACHED = frozenset({"has_canonical_format", "_has_canonical_format",
                            "has_sorted_indices", "_has_sorted_indices"})

#: The keys pandas' `_new_Index` is pickled with, across versions.
_INDEX_KEYS = frozenset({"data", "name", "names", "levels", "codes", "labels", "sortorder",
                         "start", "stop", "step"})


class PycistopicObject:
    """A pycisTopic object, its attributes restored and nothing more.

    `kind` is its class name (`CistopicObject`, `CistopicLDAModel`, ...).
    """

    kind = "?"

    def __repr__(self) -> str:
        return f"<{self.kind}: {', '.join(sorted(vars(self)))}>"


_STAND_INS: Dict[str, type] = {}


def _stand_in(name: str) -> type:
    if name not in _STAND_INS:
        _STAND_INS[name] = type(name, (PycistopicObject,), {"kind": name})
    return _STAND_INS[name]


class RefusedPickle(pickle.UnpicklingError):
    """A pickle this does not read: it names, calls or holds something a
    pycisTopic object's pickle never does."""


class RefusedGlobal(RefusedPickle):
    pass


class _ArrayClass:
    """`numpy.ndarray` as a pickle names it: the class `_reconstruct` makes,
    and nothing a pickle may call."""

    def __repr__(self) -> str:
        return "numpy.ndarray"


_NDARRAY = _ArrayClass()

#: Not found: unlike None, never an object a pickle holds.
_MISSING = object()


def _name(value: Any) -> str:
    if value is _NDARRAY:
        return "numpy.ndarray"
    module = getattr(value, "__module__", None) or type(value).__module__
    name = getattr(value, "__qualname__", None) or type(value).__qualname__
    return f"{module}.{name}"


def _refuse(message: str) -> RefusedPickle:
    return RefusedPickle(f"the pickle {message}, which a pycisTopic object's never does, so it is not read.")


def _is_count(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0


def _shape(value: Any, what: str) -> Tuple[int, ...]:
    if type(value) is not tuple or not all(_is_count(n) for n in value):
        raise _refuse(f"gives {what} the shape {value!r}")
    return tuple(int(n) for n in value)


def _size(shape: Tuple[int, ...]) -> int:
    size = 1
    for n in shape:
        size *= n
    return size


def check_dtype(dtype: Any, what: str = "an array") -> np.dtype:
    """`dtype`, if it is one numpy builds for itself: a kind pycisTopic's
    arrays use, and the state numpy writes for it (flags, item size,
    alignment), which numpy trusts when it reads a pickle."""
    if not isinstance(dtype, np.dtype):
        raise _refuse(f"gives {what} the dtype {type(dtype).__name__}")
    if dtype.kind not in _KINDS or dtype.names is not None or dtype.subdtype is not None:
        raise _refuse(f"gives {what} the numpy dtype {dtype}")
    if dtype.__reduce__() != np.dtype(dtype.str).__reduce__():
        raise _refuse(f"gives {what} a numpy dtype {dtype.str} whose state numpy never writes for it")
    return dtype


def _vector(state: Dict[str, Any], key: str, kinds: str, what: str) -> np.ndarray:
    value = state.get(key)
    if type(value) is not np.ndarray or value.ndim != 1 or value.dtype.kind not in kinds:
        described = (
            f"a {value.ndim}-d {value.dtype} array" if type(value) is np.ndarray else type(value).__name__
        )
        raise _refuse(f"gives {what} {key} as {described}")
    return value


def _within(values: np.ndarray, lo: int, hi: int, what: str) -> None:
    """Every value in [lo, hi)."""
    if values.size and (values.min() < lo or values.max() >= hi):
        raise _refuse(f"holds {what} out of range: from {values.min()} to {values.max()}, outside [{lo}, {hi})")


def check_sparse(matrix: Any) -> None:
    """A scipy matrix's state as scipy would have built it: an array of
    each kind its format needs, a 2-D shape, indices within the shape, and
    an `indptr` rising from 0 to the number of entries. scipy's C++ code
    trusts all of it."""
    kind = type(matrix).__name__.split("_")[0]  # by its class: `format` reads its state
    what = f"a scipy {type(matrix).__name__}"
    state = vars(matrix)
    unknown = sorted(set(state) - _SPARSE_STATE)
    if unknown:
        raise _refuse(f"gives {what} {', '.join(unknown)}")
    shape = state.get("_shape")
    if type(shape) is not tuple or len(shape) != 2 or not all(_is_count(n) for n in shape):
        raise _refuse(f"gives {what} the shape {shape!r}")
    n_rows, n_cols = (int(n) for n in shape)
    data = _vector(state, "data", "biufc", what)
    if kind == "coo":
        coords = state.get("coords")
        if type(coords) is not tuple or len(coords) != 2:
            raise _refuse(f"gives {what} coords that are not a row and a column")
        rows = _vector({"row": coords[0]}, "row", "iu", what)
        cols = _vector({"col": coords[1]}, "col", "iu", what)
        if not len(rows) == len(cols) == len(data):
            raise _refuse(f"gives {what} rows, columns and values of different length")
        _within(rows, 0, n_rows, f"{what}'s rows")
        _within(cols, 0, n_cols, f"{what}'s columns")
        matrix.has_canonical_format = False  # scipy sums duplicates when it needs to
        return
    indices = _vector(state, "indices", "iu", what)
    indptr = _vector(state, "indptr", "iu", what)
    n_major, n_minor = (n_rows, n_cols) if kind == "csr" else (n_cols, n_rows)
    if len(indptr) != n_major + 1:
        raise _refuse(f"gives {what} an indptr of {len(indptr):,} for {n_major:,} rows")
    bounds = indptr.astype(np.int64)
    if bounds[0] != 0 or (np.diff(bounds) < 0).any():
        raise _refuse(f"gives {what} an indptr that does not rise from 0")
    nnz = int(bounds[-1])
    if nnz > len(indices) or nnz > len(data):
        raise _refuse(f"gives {what} an indptr past its {min(len(indices), len(data)):,} entries")
    if len(indices) != nnz or len(data) != nnz:
        matrix.indices, matrix.data = indices[:nnz], data[:nnz]
        indices = matrix.indices
    _within(indices, 0, n_minor, f"{what}'s indices")


def _check_pandas(value: Any) -> None:
    """pandas objects whose integer codes its compiled code indexes with
    unchecked: within what they index."""
    import pandas as pd
    from pandas.core.arrays.masked import BaseMaskedArray

    if isinstance(value, pd.Categorical):
        what = "a Categorical"
        codes = value._ndarray
        categories = getattr(value.dtype, "categories", None)
        if not isinstance(value.dtype, pd.CategoricalDtype) or not isinstance(categories, pd.Index):
            raise _refuse(f"gives {what} no categories")
        if type(codes) is not np.ndarray or codes.ndim != 1 or codes.dtype.kind != "i":
            raise _refuse(f"gives {what} its codes as {type(codes).__name__}")
        _within(codes, -1, len(categories), f"{what}'s codes")
    elif isinstance(value, pd.MultiIndex):
        levels, codes = list(value.levels), list(value.codes)
        if len(levels) != len(codes) or len({len(c) for c in codes}) > 1:
            raise _refuse("gives a MultiIndex levels and codes that do not match")
        for level, level_codes in zip(levels, codes):
            if type(level_codes) is not np.ndarray or level_codes.dtype.kind != "i":
                raise _refuse("gives a MultiIndex codes that are not integers")
            _within(level_codes, -1, len(level), "a MultiIndex's codes")
    elif isinstance(value, BaseMaskedArray):
        data, mask = getattr(value, "_data", None), getattr(value, "_mask", None)
        if (type(data) is not np.ndarray or type(mask) is not np.ndarray
                or mask.dtype != np.bool_ or data.shape != mask.shape):
            raise _refuse(f"gives a {type(value).__name__} a mask that does not match its values")
    elif isinstance(value, pd.arrays.StringArray):
        values = value._ndarray
        if type(values) is not np.ndarray or values.ndim != 1 or values.dtype != object:
            raise _refuse("gives a StringArray values that are not text")


class _Unpickler(pickle._Unpickler):
    """The pure-Python unpickler, every opcode that calls a name, sets an
    object's state or fills a container checked first.

    `_fresh`: objects this pickle created and has not yet given their state;
    each is given it once. `_unbuilt`: those of them that are numpy arrays
    and dtypes, which nothing may hold until they have their state.
    """

    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, file: BinaryIO) -> None:
        super().__init__(file)
        self._fresh: Dict[int, Any] = {}
        self._unbuilt: Dict[int, Any] = {}
        self._dtype_args: Dict[int, Tuple[str, bool]] = {}
        self._loaded: Dict[int, Any] = {}
        self._own: Dict[Tuple[str, int], Callable[..., Any]] = {}
        self._classes: Optional[Tuple[type, ...]] = None

    def load(self) -> Any:
        value = super().load()
        for item in self._loaded.values():
            if isinstance(item, np.ndarray):
                continue
            if type(item).__module__.startswith("scipy"):
                check_sparse(item)
            else:
                _check_pandas(item)
        return value

    # -- names ---------------------------------------------------------

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(PYCISTOPIC):
            return _stand_in(name)
        # Protocols 0-2 name Python 2's modules (`__builtin__`); the default
        # find_class maps them, so an override must too. Python 3's names
        # are not in the tables.
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        module, name = _renamed(module, name)
        if name not in ALLOWED.get(module, ()):
            raise RefusedGlobal(
                f"the pickle asks for {module}.{name}, which a pycisTopic object "
                "never holds, so it is not read."
            )
        value = _resolve(module, name)
        if value is np.ndarray:
            return _NDARRAY
        own = self._wrappers().get(_key(value))
        return own if own is not None else value

    def _constructed(self) -> Tuple[type, ...]:
        if self._classes is None:
            found = []
            for module, name in _CONSTRUCTED:
                try:
                    found.append(_resolve(module, name))
                except RefusedGlobal:  # pragma: no cover - pandas comes with the extra
                    pass
            self._classes = tuple(found)
        return self._classes

    def _wrappers(self) -> Dict[Tuple[str, int], Callable[..., Any]]:
        """The allowlisted callables given checked arguments instead, by the
        real one's identity."""
        if not self._own:
            real = {
                np.dtype: self._dtype,
                copyreg._reconstructor: self._reconstructor,
                _resolve("_codecs", "encode"): self._encode,
                _resolve("numpy._core.multiarray", "_reconstruct"): self._reconstruct,
                _resolve("numpy._core.multiarray", "scalar"): self._scalar,
                _resolve("numpy._core.numeric", "_frombuffer"): self._frombuffer,
            }
            try:
                real[_resolve("pandas.core.indexes.base", "_new_Index")] = self._new_index
                real[_ndarray_backed()] = self._ndarray_backed
            except RefusedGlobal:  # pragma: no cover - pandas comes with the extra
                pass
            self._own = {_key(value): own for value, own in real.items()}
        return self._own

    # -- what may be held, and by what ---------------------------------

    def _created(self, obj: Any, fresh: bool = True) -> Any:
        if fresh:
            self._fresh[id(obj)] = obj
        if not isinstance(obj, (PycistopicObject, np.dtype)) and (
            type(obj).__module__.startswith(("scipy", "pandas"))
        ):
            self._loaded[id(obj)] = obj
        return obj

    def _held(self, items: Iterable[Any], by: str) -> None:
        """Refuse a numpy array or dtype that is handed on before it has
        its state: what it is handed to sees it empty."""
        unbuilt = self._unbuilt
        for item in items:
            if unbuilt.get(id(item), _MISSING) is item:
                raise _refuse(f"hands a numpy {type(item).__name__} to {by} before giving it its state")

    def _operands(self, count: Optional[int] = None) -> list:
        """The items an opcode is about to consume: the last `count`, or all
        since the mark."""
        return self.stack if count is None else self.stack[len(self.stack) - count:]

    # -- calls -----------------------------------------------------------

    def _call(self, func: Any, args: Any) -> Any:
        if type(args) is not tuple:
            raise _refuse(f"calls {_name(func)} with {type(args).__name__} as its arguments")
        self._held(args, _name(func))
        if getattr(func, "__self__", None) is self:
            return func(*args)
        if isinstance(func, type):
            if not args:
                return self._created(func())
            if any(func is c for c in self._constructed()):
                if func in (bytes, bytearray) and any(not isinstance(a, (str, bytes)) for a in args):
                    raise _refuse(f"calls {func.__name__} with {', '.join(type(a).__name__ for a in args)}")
                return self._created(func(*args), fresh=False)
            raise _refuse(f"constructs {_name(func)} from arguments")
        module = getattr(func, "__module__", "") or ""
        if module.startswith("pandas") and getattr(func, "__name__", "") in ("_unpickle_block", "new_block"):
            return self._created(func(*args), fresh=False)
        raise _refuse(f"calls {_name(func)}")

    def load_reduce(self) -> None:
        stack = self.stack
        args = stack.pop()
        stack[-1] = self._call(stack[-1], args)

    dispatch[pickle.REDUCE[0]] = load_reduce

    def _new(self, cls: Any, args: Any, kwargs: Any) -> Any:
        if not isinstance(cls, type):
            raise _refuse(f"creates an instance of {cls!r}")
        if args or kwargs:
            raise _refuse(f"constructs {_name(cls)} from arguments")
        return self._created(cls.__new__(cls))

    def load_newobj(self) -> None:
        args = self.stack.pop()
        self.stack[-1] = self._new(self.stack[-1], args, None)

    dispatch[pickle.NEWOBJ[0]] = load_newobj

    def load_newobj_ex(self) -> None:
        kwargs = self.stack.pop()
        args = self.stack.pop()
        self.stack[-1] = self._new(self.stack[-1], args, kwargs)

    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj_ex

    def _instantiate(self, klass: Any, args: Any) -> None:
        self.append(self._new(klass, args, None))

    # -- state -----------------------------------------------------------

    def load_build(self) -> None:
        stack = self.stack
        state = stack.pop()
        inst = stack[-1]
        if self._fresh.pop(id(inst), _MISSING) is not inst:
            raise _refuse(f"sets the state of a {_name(type(inst))} it did not just create, or more than once")
        self._held((state,), "a state")
        if isinstance(inst, np.dtype):
            self._build_dtype(inst, state)
        elif type(inst) is np.ndarray:
            self._build_array(inst, state)
        elif isinstance(inst, PycistopicObject):
            if type(state) is not dict or not all(type(k) is str for k in state):
                raise _refuse(f"gives a {inst.kind} the state {type(state).__name__}")
            inst.__dict__.update(state)
        elif type(inst).__module__.startswith("scipy"):
            if type(state) is not dict or not all(type(k) is str for k in state):
                raise _refuse(f"gives a {type(inst).__name__} the state {type(state).__name__}")
            state = {k: v for k, v in state.items() if k not in _SPARSE_CACHED}
            setstate = getattr(inst, "__setstate__", None)
            if setstate is not None:
                setstate(state)
            else:
                inst.__dict__.update(state)
        elif type(inst).__module__.startswith("pandas"):
            stack.append(state)
            super().load_build()
        else:
            raise _refuse(f"sets the state of {_name(type(inst))}")
        self._unbuilt.pop(id(inst), None)

    dispatch[pickle.BUILD[0]] = load_build

    # -- containers: only dict, list and set are filled --------------------

    def _fill(self, target: Any, kinds: Tuple[type, ...], items: list, by: str) -> None:
        if type(target) not in kinds:
            raise _refuse(f"{by} into a {_name(type(target))}")
        self._held(items, f"a {type(target).__name__}")

    def load_append(self) -> None:
        self._fill(self.stack[-2], (list,), self.stack[-1:], "appends")
        super().load_append()

    dispatch[pickle.APPEND[0]] = load_append

    def load_appends(self) -> None:
        self._fill(self.metastack[-1][-1], (list,), self.stack, "appends")
        super().load_appends()

    dispatch[pickle.APPENDS[0]] = load_appends

    def load_setitem(self) -> None:
        self._fill(self.stack[-3], (dict, OrderedDict), self.stack[-2:], "writes")
        super().load_setitem()

    dispatch[pickle.SETITEM[0]] = load_setitem

    def load_setitems(self) -> None:
        self._fill(self.metastack[-1][-1], (dict, OrderedDict), self.stack, "writes")
        super().load_setitems()

    dispatch[pickle.SETITEMS[0]] = load_setitems

    def load_additems(self) -> None:
        self._fill(self.metastack[-1][-1], (set,), self.stack, "adds")
        super().load_additems()

    dispatch[pickle.ADDITEMS[0]] = load_additems

    def _packed(load: Callable[[Any], None], count: Optional[int]) -> Callable[[Any], None]:  # type: ignore[misc]
        def packed(self: "_Unpickler") -> None:
            if self._unbuilt:
                self._held(self._operands(count), "a tuple, list, dict or set")
            load(self)

        return packed

    for _opcode, _count in (
        (pickle.TUPLE, None), (pickle.TUPLE1, 1), (pickle.TUPLE2, 2), (pickle.TUPLE3, 3),
        (pickle.LIST, None), (pickle.DICT, None), (pickle.FROZENSET, None),
    ):
        dispatch[_opcode[0]] = _packed(dispatch[_opcode[0]], _count)
    del _opcode, _count, _packed

    # -- numpy -----------------------------------------------------------

    def _dtype(self, spec: Any, align: Any = False, copy: Any = True) -> np.dtype:
        """`numpy.dtype(spec, align, copy)`, from a type's name only; always
        a copy, so the state set on it next is set on nothing else."""
        if type(spec) is not str:
            raise _refuse(f"builds a numpy dtype from {type(spec).__name__}")
        try:
            dtype = np.dtype(spec, bool(align), True)
        except (TypeError, ValueError) as exc:
            raise _refuse(f"builds the numpy dtype {spec!r} ({exc})") from None
        check_dtype(dtype, "a dtype")
        self._dtype_args[id(dtype)] = (spec, bool(align))
        self._unbuilt[id(dtype)] = dtype
        return self._created(dtype)

    def _build_dtype(self, dtype: np.dtype, state: Any) -> None:
        """The state set first on a copy, and only on `dtype` when the copy
        comes out as numpy builds that dtype: so `dtype`, which an array may
        already use, is never wrong, even until the refusal."""
        spec, align = self._dtype_args.pop(id(dtype))
        if type(state) is not tuple:
            raise _refuse(f"gives a numpy dtype the state {type(state).__name__}")
        probe = np.dtype(spec, align, True)
        try:
            probe.__setstate__(state)
        except Exception as exc:  # numpy's own refusal of the tuple
            raise _refuse(f"gives the numpy dtype {spec!r} the state {state!r} ({exc})") from None
        check_dtype(probe, "a dtype")
        dtype.__setstate__(state)

    def _reconstruct(self, subtype: Any, shape: Any, typecode: Any = b"b") -> np.ndarray:
        """numpy's `_reconstruct`, as numpy pickles an array: an empty
        ndarray, given its state next."""
        if subtype is not _NDARRAY:
            raise _refuse(f"reconstructs {subtype!r} through numpy's _reconstruct")
        if shape != (0,) or type(shape) is not tuple:
            raise _refuse(f"reconstructs an array of shape {shape!r} through numpy's _reconstruct")
        array = np.ndarray.__new__(np.ndarray, (0,), np.dtype("b"))
        self._unbuilt[id(array)] = array
        return self._created(array)

    def _build_array(self, array: np.ndarray, state: Any) -> None:
        """An array's state: its shape, a checked dtype, and data that fills
        the shape exactly, as numpy writes it."""
        if type(state) is not tuple or len(state) not in (4, 5):
            raise _refuse(f"gives a numpy array the state {type(state).__name__}")
        shape, dtype, fortran, raw = state[-4:]
        if len(state) == 5 and not _is_count(state[0]):
            raise _refuse(f"gives a numpy array the version {state[0]!r}")
        shape = _shape(shape, "a numpy array")
        dtype = check_dtype(dtype)
        if not isinstance(fortran, (bool, int)):
            raise _refuse(f"gives a numpy array the order {fortran!r}")
        size = _size(shape)
        if dtype.hasobject:
            if type(raw) is not list or len(raw) != size:
                raise _refuse(f"gives a numpy object array of {size:,} items its data as {type(raw).__name__}")
        elif type(raw) not in (bytes, bytearray) or len(raw) != size * dtype.itemsize:
            length = len(raw) if isinstance(raw, (bytes, bytearray)) else type(raw).__name__
            raise _refuse(
                f"gives a numpy {dtype} array of shape {shape} {length} bytes of data, "
                f"not {size * dtype.itemsize:,}"
            )
        array.__setstate__(state)

    def _frombuffer(self, buffer: Any, dtype: Any, shape: Any, order: Any) -> np.ndarray:
        """numpy's `_frombuffer` (protocol 5): a checked dtype, and a buffer
        the shape fills exactly."""
        from numpy import frombuffer

        dtype = check_dtype(dtype)
        shape = _shape(shape, "a numpy array")
        if dtype.hasobject or type(buffer) not in (bytes, bytearray) or order not in ("C", "F"):
            raise _refuse(f"builds a numpy {dtype} array from {type(buffer).__name__}")
        if len(buffer) != _size(shape) * dtype.itemsize:
            raise _refuse(f"gives a numpy {dtype} array of shape {shape} {len(buffer):,} bytes of data")
        return frombuffer(buffer, dtype=dtype).reshape(shape, order=order)

    def _scalar(self, dtype: Any, raw: Any = None) -> Any:
        """numpy's `scalar`: an object as itself, or bytes as wide as the
        dtype's items."""
        dtype = check_dtype(dtype, "a scalar")
        if dtype.hasobject:
            return raw
        if type(raw) is not bytes or len(raw) != dtype.itemsize:
            raise _refuse(f"gives a numpy {dtype} scalar {type(raw).__name__} of the wrong size")
        return np.frombuffer(raw, dtype=dtype)[0]

    # -- builtins and pandas -------------------------------------------

    def _encode(self, text: Any, encoding: Any = "utf-8") -> bytes:
        """`_codecs.encode`, which protocols 0-2 write bytes with."""
        if type(text) is not str or encoding not in ("latin1", "latin-1"):
            raise _refuse(f"encodes {type(text).__name__} as {encoding!r}")
        return text.encode("latin-1")

    def _reconstructor(self, cls: Any, base: Any, state: Any) -> Any:
        """`copyreg._reconstructor` as protocols 0 and 1 write it: an empty
        instance, given its state next."""
        if base is not object or state is not None or not isinstance(cls, type):
            raise _refuse(f"calls copyreg._reconstructor for {cls!r} from {base!r}")
        return self._created(object.__new__(cls))

    def _new_index(self, cls: Any, d: Any) -> Any:
        """pandas' `_new_Index(cls, d)`, which passes `d` to the class: only
        the keys pandas pickles an index with, so never a dtype or a
        timezone to parse."""
        import pandas as pd
        from pandas.core.indexes.base import _new_Index

        if not (isinstance(cls, type) and issubclass(cls, pd.Index)) or type(d) is not dict:
            raise _refuse(f"builds an index with pandas' _new_Index from {cls!r}")
        unknown = sorted(set(d) - _INDEX_KEYS)
        if unknown:
            raise _refuse(f"builds a {cls.__name__} with {', '.join(map(str, unknown))}")
        return self._created(_new_Index(cls, d), fresh=False)

    def _ndarray_backed(self, cls: Any, checksum: Any, state: Any = None) -> Any:
        """pandas' unpickling of its Categorical and StringArray."""
        import pandas as pd

        if cls not in (pd.Categorical, pd.arrays.StringArray):
            raise _refuse(f"builds {cls!r} through pandas' __pyx_unpickle_NDArrayBacked")
        return self._created(_ndarray_backed()(cls, checksum, state), fresh=state is None)


def _ndarray_backed() -> Callable[..., Any]:
    # Looked up by name: in a class body, `__pyx_...` would be mangled.
    return _resolve("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked")


def _key(value: Any) -> Tuple[str, int]:
    return ("callable", id(value))


def _renamed(module: str, name: str) -> Tuple[str, str]:
    """pandas' table of classes renamed or moved since they were pickled."""
    if module.startswith("pandas"):
        from pandas.compat import pickle_compat

        renames = getattr(pickle_compat, "_class_locations_map", {})
        if (module, name) in renames:
            return renames[(module, name)]
    return module, name


def _resolve(module: str, name: str) -> Any:
    if module.startswith("numpy.core") or module.startswith("numpy._core"):
        # numpy 2 renamed numpy.core to numpy._core, and keeps the old name
        # only as a deprecated alias.
        suffix = module.split(".", 2)[2]
        for candidate in (f"numpy._core.{suffix}", f"numpy.core.{suffix}"):
            try:
                return getattr(importlib.import_module(candidate), name)
            except (ImportError, AttributeError):
                continue
    if module.startswith("scipy.sparse"):
        import scipy.sparse

        return getattr(scipy.sparse, name)
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        raise RefusedGlobal(
            f"the pickle asks for {module}.{name}, which this pandas or numpy "
            "does not have."
        ) from None


@contextmanager
def open_pickle(path: Path) -> Iterator[BinaryIO]:
    """The file as bytes, inflated if gzipped (by its magic bytes, not its name)."""
    with open(path, "rb") as raw:
        gzipped = raw.read(2) == b"\x1f\x8b"
    if gzipped:
        with gzip.open(path, "rb") as fh:
            yield fh
    else:
        with open(path, "rb") as fh:
            yield fh


def load(path: Path) -> Any:
    """Unpickle `path` through the allowlist and the checks."""
    with open_pickle(path) as fh:
        return _Unpickler(fh).load()


def pickled_class(path: Path) -> Optional[str]:
    """The pycisTopic class a pickle holds (`CistopicObject`, ...), from its
    first opcodes alone; None for any other file."""
    try:
        with open_pickle(path) as fh:
            strings: list = []
            for i, (op, arg, _) in enumerate(pickletools.genops(fh)):
                if i >= SNIFF_OPCODES:
                    return None
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                    strings.append(arg)
                elif op.name in ("STACK_GLOBAL", "GLOBAL"):
                    if op.name == "GLOBAL":
                        module, _, name = str(arg).partition(" ")
                    elif len(strings) >= 2:
                        module, name = strings[-2], strings[-1]
                    else:
                        return None
                    if (module, name) in _HELPERS:
                        continue  # the class itself comes next
                    return name if module.startswith(PYCISTOPIC) else None
    except Exception:  # not a pickle, or a truncated one: not this format
        return None
    return None


def sniff(path: Path) -> bool:
    path = Path(path)
    return path.is_file() and pickled_class(path) is not None

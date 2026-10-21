"""Reading pycisTopic's pickles without running anything from them.

`pickle.load` calls whatever callable the file names, so loading a pickle
from elsewhere runs its author's code. pycisTopic saves its objects only as
pickles, and the objects hold nothing but numpy arrays, scipy sparse
matrices, pandas frames, builtins, and its own classes. So the file is read
by an unpickler that resolves an allowlist of those names and refuses any
other: an `os.system` in the file is refused by name, never called.

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
import gzip
import importlib
import pickle
import pickletools
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict, FrozenSet, Iterator, Optional, Tuple

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

#: The globals a pycisTopic pickle may name: module -> names.
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
    "pandas.core.arrays.categorical": frozenset({"Categorical", "_unpickle"}),
    "pandas.core.dtypes.dtypes": frozenset({
        "CategoricalDtype", "DatetimeTZDtype", "PeriodDtype", "IntervalDtype",
    }),
    "pandas.core.indexes.base": frozenset({"_new_Index", "Index"}),
    "pandas.core.indexes.range": frozenset({"RangeIndex"}),
    "pandas.core.indexes.numeric": frozenset({"Int64Index", "Float64Index", "UInt64Index"}),
    "pandas.core.indexes.category": frozenset({"CategoricalIndex"}),
    "pandas.core.indexes.multi": frozenset({"MultiIndex"}),
    "pandas.core.indexes.datetimes": frozenset({"DatetimeIndex", "_new_DatetimeIndex"}),
    "pandas.core.arrays.integer": frozenset({"IntegerArray", *(
        f"{sign}Int{bits}Dtype" for sign in ("", "U") for bits in (8, 16, 32, 64)
    )}),
    "pandas.core.arrays.floating": frozenset({"FloatingArray", "Float32Dtype", "Float64Dtype"}),
    "pandas.core.arrays.boolean": frozenset({"BooleanArray", "BooleanDtype"}),
    "pandas.core.arrays.string_": frozenset({"StringArray", "StringDtype"}),
    "pandas.core.arrays.datetimes": frozenset({"DatetimeArray"}),
    "pandas._libs.tslibs.timestamps": frozenset({"_unpickle_timestamp", "Timestamp"}),
    "pandas._libs.tslibs.nattype": frozenset({"__nat_unpickle"}),
    "pandas._libs.tslibs.timezones": frozenset(),
    # pandas 3 pickles its classes under their public names.
    "pandas": frozenset({
        "DataFrame", "Series", "Index", "RangeIndex", "MultiIndex", "CategoricalIndex",
        "DatetimeIndex", "Categorical", "CategoricalDtype", "StringDtype", "BooleanDtype",
        "Float32Dtype", "Float64Dtype", "Timestamp", "DatetimeTZDtype", *(
            f"{sign}Int{bits}Dtype" for sign in ("", "U") for bits in (8, 16, 32, 64)
        ),
    }),
    "pandas.arrays": frozenset({
        "BooleanArray", "IntegerArray", "FloatingArray", "StringArray", "Categorical",
        "DatetimeArray",
    }),
    "builtins": frozenset({"slice", "set", "frozenset", "complex", "bytearray", "range", "object"}),
    "collections": frozenset({"OrderedDict"}),
    "copyreg": frozenset({"_reconstructor"}),
    "_codecs": frozenset({"encode"}),
}


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


class RefusedGlobal(pickle.UnpicklingError):
    pass


class _Unpickler(pickle.Unpickler):
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
        return _resolve(module, name)


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
    """Unpickle `path` through the allowlist."""
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

"""The restricted unpickler: what a crafted pickle cannot get past.

`formats/pickled.py` reads pycisTopic's pickles through an allowlist of the
names a pycisTopic object holds, and checks what numpy, scipy and pandas are
given before any of them acts on it. Every crafted pickle here names only
allowlisted globals: the checks are what refuse it.

The crafted files are loaded in a child process, because before the checks
some of them crashed the interpreter, which would have taken the test run
with it. The child loads the file and then uses what it got, as a
conversion would, so a pickle that slipped through shows as a crash or a
wrong answer rather than as nothing.
"""

from __future__ import annotations

import copyreg
import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adata.formats import pickled
from tests.conftest import _pycistopic_classes, write_cistopic_pickle, write_imputed_pickle

pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

_reconstruct = np.ndarray.__reduce__(np.zeros(1))[0]


class _Crafted:
    """Pickles as `func(*args)`, then `state` set on what that returned:
    the REDUCE and BUILD a crafted file holds."""

    def __init__(self, func, args, state=None):
        self.func, self.args, self.state = func, args, state

    def __reduce__(self):
        if self.state is None:
            return self.func, self.args
        return self.func, self.args, self.state


class _New:
    """Pickles as `cls.__new__(cls)` (NEWOBJ), then `state` set on it: how
    protocol 2 and later write a class's instance. `__class__` is what the
    pickler checks NEWOBJ's class against."""

    def __init__(self, cls, state):
        self.cls, self.state = cls, state

    @property
    def __class__(self):
        return self.cls

    def __reduce_ex__(self, protocol):
        return copyreg.__newobj__, (self.cls,), self.state


def _array(dtype, shape, raw):
    """An ndarray as numpy pickles one, with whatever state is given."""
    return _Crafted(_reconstruct, (np.ndarray, (0,), b"b"), (1, shape, dtype, False, raw))


def _dtype(spec, state):
    """A numpy dtype as numpy pickles one, with whatever state is given."""
    return _Crafted(np.dtype, (spec, False, True), state)


# The child: load, then use what was loaded as a conversion would. Exit 1
# with REFUSED when the unpickler refuses; OPENED lists the watched path if
# anything opened it.
_CHILD = r"""
import sys
watched = sys.argv[2] if len(sys.argv) > 2 else None
opened = []
if watched:
    sys.addaudithook(lambda event, args: opened.append(args[0])
                     if event == "open" and str(args[0]) == watched else None)
from adata.formats import pickled
try:
    try:
        obj = pickled.load(sys.argv[1])
    except pickled.RefusedPickle as exc:
        print("REFUSED", exc)
        sys.exit(1)
    import numpy as np
    import scipy.sparse as sp
    values = list(vars(obj).values()) if isinstance(obj, pickled.PycistopicObject) else [obj]
    for value in values:
        if sp.issparse(value):
            print(value.tocsr().toarray().sum())
        else:
            print(repr(value)[:200])
    print("LOADED")
finally:
    if opened:
        print("OPENED", opened)
"""


def _load_isolated(path: Path, watched: Path = None) -> subprocess.CompletedProcess:
    args = [sys.executable, "-c", _CHILD, str(path)] + ([str(watched)] if watched else [])
    return subprocess.run(args, capture_output=True, text=True, timeout=120)


def _write(path: Path, obj, protocol: int = 4) -> Path:
    path.write_bytes(pickle.dumps(obj, protocol=protocol))
    return path


def _refused(result: subprocess.CompletedProcess, *phrases: str) -> str:
    output = result.stdout + result.stderr
    assert result.returncode == 1 and "REFUSED" in result.stdout, (result.returncode, output[-2000:])
    for phrase in phrases:
        assert phrase in result.stdout, result.stdout
    assert "OPENED" not in result.stdout, result.stdout
    return result.stdout


# ---------------------------------------------------------------------------
# numpy: dtypes and arrays


@pytest.mark.parametrize(
    "dtype, raw",
    [
        # A structured dtype with an object field whose pickled flags were
        # cleared: numpy then treats the field's pointer as plain bytes.
        (_dtype("V8", (3, "|", None, ("f0",), {"f0": (np.dtype("O"), 0)}, 8, 1, 0)), b"\x41" * 8),
        # A plain object dtype with its flags cleared: the raw bytes become
        # the array's object pointers.
        (_dtype("O8", (3, "|", None, None, None, -1, -1, 0)), b"\x41" * 8),
        # A text dtype claiming 7 bytes an item, which UCS-4 text never is.
        (_dtype("U1", (3, "<", None, None, None, 7, 3, 8)), b"\x41" * 7),
    ],
    ids=["structured", "object", "itemsize"],
)
def test_a_dtype_whose_pickled_state_numpy_never_writes_is_refused(tmp_path, dtype, raw):
    """F040: numpy trusts the flags and item size a pickled dtype carries,
    so a dtype could claim no objects while holding them, and raw bytes
    from the file were read as object pointers: the interpreter crashed
    (or worse), with only allowlisted names in the file."""
    path = _write(tmp_path / "evil.pkl", _array(dtype, (1,), raw))
    _refused(_load_isolated(path), "dtype")


def test_the_crafted_dtype_is_refused_inside_a_cistopic_object_too(tmp_path):
    """F040, end to end: the crafted array as a CistopicObject's cell
    names, which the converter iterates while planning."""
    import scipy.sparse as sp

    classes = _pycistopic_classes()
    obj = classes["CistopicObject"]()
    obj.fragment_matrix = sp.csr_matrix(np.ones((1, 1), dtype=np.int32))
    obj.region_names = ["r0"]
    obj.cell_names = _array(_dtype("O8", (3, "|", None, None, None, -1, -1, 0)), (1,), b"\x41" * 8)
    path = _write(tmp_path / "evil.pkl", obj)
    out = tmp_path / "out.h5ad"
    result = subprocess.run(
        [sys.executable, "-c", "from adata.cli import app; app()", "convert", str(path), "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    assert "whose state numpy never writes" in " ".join((result.stdout + result.stderr).split())
    assert not out.exists()


def test_array_data_that_does_not_fill_the_shape_is_refused(tmp_path):
    path = _write(tmp_path / "short.pkl", _array(np.dtype("<i8"), (4,), b"\x00" * 8))
    _refused(_load_isolated(path), "array")


def _assemble(*ops: bytes) -> bytes:
    return b"\x80\x02" + b"".join(ops) + b"."


def _global(module: str, name: str) -> bytes:
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


# dtype('<i8') as numpy pickles it: dtype('i8', False, True), then its state.
_INT64 = (
    _global("numpy", "dtype") + b"X\x02\x00\x00\x00i8\x89\x88\x87R"
    + b"(K\x03X\x01\x00\x00\x00<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb"
)


# An int64 array of one element, its raw data `raw`: numpy's own opcodes.
def _int_array_ops(raw: bytes) -> bytes:
    return (
        _global(_reconstruct.__module__, "_reconstruct") + _global("numpy", "ndarray")
        + b"K\x00\x85" + b"C\x01b" + b"\x87R"                      # _reconstruct(ndarray, (0,), b'b')
        + b"(K\x01K\x01\x85" + _INT64
        + b"\x89C" + bytes([len(raw)]) + raw + b"tb"               # BUILD (1, (1,), dtype('i8'), False, raw)
    )


def test_an_array_cannot_be_rebuilt_after_it_was_checked(tmp_path):
    """Checks made when an array is built mean nothing if the pickle can
    build it again later with other contents (or write into it): neither
    happens in a pickle numpy writes."""
    rebuilt = _assemble(
        _int_array_ops(b"\x00" * 8), b"q\x00",
        b"(K\x01K\x01\x85" + _INT64
        + b"\x89C\x08" + b"\xff" * 8 + b"tb",                      # a second BUILD on the same array
    )
    _refused(_load_isolated(_bytes(tmp_path / "rebuilt.pkl", rebuilt)), "once")

    written = _assemble(_int_array_ops(b"\x00" * 8), b"K\x00K\x07s")  # array[0] = 7
    _refused(_load_isolated(_bytes(tmp_path / "written.pkl", written)), "ndarray")


def _bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_an_array_cannot_be_used_before_it_is_built(tmp_path):
    """An array is empty until its state is set: one handed to something
    else first, then filled, would bypass whatever that something checked."""
    used_early = _assemble(
        _global(_reconstruct.__module__, "_reconstruct") + _global("numpy", "ndarray")
        + b"K\x00\x85C\x01b\x87R" + b"q\x00"
        + b"]q\x01h\x00a"                                          # [array] before the array's BUILD
        + b"0h\x00"
        + b"(K\x01K\x01\x85" + _INT64
        + b"\x89C\x08" + b"\x00" * 8 + b"tb0h\x01",
    )
    _refused(_load_isolated(_bytes(tmp_path / "early.pkl", used_early)), "before")


def test_numpys_own_reconstruction_is_not_callable_with_other_arguments(tmp_path):
    """`numpy.ndarray` is named by a pickle only as the class
    `_reconstruct` makes; called itself, it can view any buffer."""
    called = _assemble(_global("numpy", "ndarray") + b"K\x01\x85\x85R")  # ndarray((1,))
    _refused(_load_isolated(_bytes(tmp_path / "called.pkl", called)), "numpy.ndarray")

    big = _assemble(
        _global(_reconstruct.__module__, "_reconstruct") + _global("numpy", "ndarray")
        + b"J\x00\x00\x00\x40\x85C\x01b\x87R"                       # _reconstruct(ndarray, (2**30,), b'b')
    )
    _refused(_load_isolated(_bytes(tmp_path / "big.pkl", big)), "_reconstruct")


def test_a_scalar_whose_bytes_do_not_fit_its_dtype_is_refused(tmp_path):
    scalar = np.int64(1).__reduce__()[0]  # numpy's `scalar`, as a pickle names it
    path = _write(tmp_path / "scalar.pkl", _Crafted(scalar, (np.dtype("<i8"), b"\x01\x02")))
    _refused(_load_isolated(path), "scalar")
    forged = _dtype("O8", (3, "|", None, None, None, -1, -1, 0))
    path = _write(tmp_path / "objscalar.pkl", _Crafted(scalar, (forged, b"\x41" * 8)))
    _refused(_load_isolated(path), "dtype")


# ---------------------------------------------------------------------------
# scipy: sparse matrices


def _sparse_with(kind: str, **state):
    m = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "coo": sp.coo_matrix}[kind](
        np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
    )
    for key, value in state.items():
        setattr(m, key, value)
    return m


@pytest.mark.parametrize(
    "kind, state, phrase",
    [
        # coo_tocsr wrote out of bounds on the heap for a coordinate past
        # the shape: silently for a small overshoot, SIGBUS for a large one.
        ("coo", {"coords": (np.array([0, 5, 1], dtype=np.int32), np.array([0, 2, 1], dtype=np.int32))}, "out of range"),
        ("coo", {"coords": (np.array([0, 10**9, 1]), np.array([0, 2, 1]))}, "out of range"),
        ("coo", {"coords": (np.array([0, -1, 1]), np.array([0, 2, 1]))}, "out of range"),
        ("coo", {"coords": (np.array([0, 1]), np.array([0, 2]))}, "length"),
        # A CSC's indices were written to X as they were.
        ("csc", {"indices": np.array([0, 7, 0], dtype=np.int32)}, "out of range"),
        ("csr", {"indices": np.array([0, 2, -3], dtype=np.int32)}, "out of range"),
        ("csr", {"indptr": np.array([0, 2, 1], dtype=np.int32)}, "indptr"),
        ("csr", {"indptr": np.array([0, 2, 9], dtype=np.int32)}, "indptr"),
        ("csr", {"indptr": np.array([0, 2], dtype=np.int32)}, "indptr"),
        ("csr", {"indices": np.array([0.0, 2.0, 1.0])}, "indices"),
        ("csr", {"data": np.array(["a", "b", "c"], dtype=object)}, "data"),
        ("csr", {"_shape": (2, -3)}, "shape"),
        ("csr", {"_shape": (2, 2)}, "out of range"),
        ("csr", {"_format": "csc"}, "_format"),
    ],
    ids=[
        "coo-row-past-shape", "coo-row-far-past-shape", "coo-negative", "coo-short",
        "csc-index", "csr-negative-index", "csr-indptr-falls", "csr-indptr-past-nnz",
        "csr-indptr-short", "csr-float-indices", "csr-object-data", "csr-negative-shape",
        "csr-shape-too-narrow", "csr-claims-another-format",
    ],
)
def test_a_sparse_matrix_that_is_not_well_formed_is_refused_before_scipy_uses_it(tmp_path, kind, state, phrase):
    """F038: a pickled scipy matrix is restored without any check, and
    scipy's C++ routines index with what it holds: coordinates past the
    shape wrote out of bounds on the heap, and a CSC's out-of-range indices
    reached the output."""
    path = _write(tmp_path / "evil.pkl", _sparse_with(kind, **state))
    _refused(_load_isolated(path), phrase)


@pytest.mark.parametrize("kind", ["csr", "csc", "coo", "csr_array"])
def test_well_formed_sparse_matrices_still_read(tmp_path, kind):
    dense = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
    m = getattr(sp, f"{kind}_matrix" if "_" not in kind else kind)(dense)
    loaded = pickled.load(_write(tmp_path / "ok.pkl", m))
    assert type(loaded) is type(m)
    assert np.array_equal(loaded.toarray(), dense)


# ---------------------------------------------------------------------------
# pandas


def test_a_categorical_whose_codes_pass_its_categories_is_refused(tmp_path):
    """A Categorical's codes index its categories in pandas' C code without a
    bounds check: one past the end read out of bounds (SIGBUS)."""
    cat = pd.Categorical(np.array([1.5, 2.5]))
    func, args = cat.__reduce__()[:2]
    crafted = _Crafted(func, (args[0], args[1], (cat.dtype, np.array([0, 10**8], dtype=np.int64))))
    _refused(_load_isolated(_write(tmp_path / "cat.pkl", crafted)), "codes")


def _frame_of(manager):
    """A DataFrame around `manager`, as pandas 1.5 pickles one: created
    empty, then given its manager in its state."""
    state = {"_mgr": manager, "_typ": "dataframe", "_metadata": [], "attrs": {},
             "_flags": {"allows_duplicate_labels": True}}
    return _Crafted(copyreg._reconstructor, (pd.DataFrame, object, None), state)


def _manager_state(columns, rows, blocks):
    """BlockManager's state as pandas 1.5 (pycisTopic's) pickles it."""
    return ([], [], [], {"0.14.1": {"axes": [columns, rows], "blocks": [
        {"values": values, "mgr_locs": locs} for values, locs in blocks
    ]}})


def _short_frames():
    """Frames that claim 4,000 rows but whose one block holds a single
    value, by the two routes pandas' pickles build a frame's manager."""
    from pandas._libs.internals import _unpickle_block
    from pandas.core.internals.managers import BlockManager

    with pd.option_context("future.infer_string", False):
        rows = pd.Index([f"c{k}" for k in range(4000)])
        columns = pd.Index(["score"])
    note = np.empty((1, 1), dtype=object)
    note[0, 0] = "hi"
    old = lambda values: _New(BlockManager, _manager_state(columns, rows, [(values, slice(0, 1, 1))]))
    unchecked = _Crafted(BlockManager, ((_Crafted(_unpickle_block, (np.array([[1.5]]), slice(0, 1, 1), 2)),),
                                        [columns, rows], False))
    return {"float": _frame_of(old(np.array([[7.25]]))), "object": _frame_of(old(note)),
            "unverified": _frame_of(unchecked)}


@pytest.mark.parametrize("route", ["float", "object", "unverified"])
def test_a_frame_whose_blocks_do_not_fill_its_rows_is_refused(tmp_path, route):
    """Review of #101: pandas does not check, when it unpickles a frame,
    that its blocks hold as many rows as its index; a reindex then read
    past the block: heap words written into the output for a float
    column, a crash for an object one. Both of pandas' routes to a
    manager (its pandas 1.5 state, and its constructor told not to
    verify) are checked."""
    path = _write(tmp_path / "short.pkl", _short_frames()[route])
    _refused(_load_isolated(path), "rows" if route != "unverified" else "BlockManager")


def test_a_frame_whose_blocks_do_not_cover_its_columns_once_is_refused(tmp_path):
    """Two blocks for column x and none for y: pandas' constructor counts
    the columns, not which ones."""
    from pandas._libs.internals import _unpickle_block
    from pandas.core.internals.managers import BlockManager

    with pd.option_context("future.infer_string", False):
        rows, columns = pd.Index(["a", "b"]), pd.Index(["x", "y"])
    blocks = tuple(_Crafted(_unpickle_block, (np.full((1, 2), v), slice(0, 1, 1), 2)) for v in (0.0, 1.0))
    manager = _Crafted(BlockManager, (blocks, [columns, rows]))
    _refused(_load_isolated(_write(tmp_path / "twice.pkl", _frame_of(manager))), "columns")


def test_a_frame_holds_only_what_pandas_pickles_a_frame_with(tmp_path):
    """A frame's state is set as attributes as it stands: one naming
    pandas' cache of columns could hand back any Series for a column."""
    with pd.option_context("future.infer_string", False):
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        state = {**frame.__getstate__(), "_item_cache": {"x": pd.Series([1.0])}}
    crafted = _Crafted(copyreg._reconstructor, (pd.DataFrame, object, None), state)
    _refused(_load_isolated(_write(tmp_path / "cache.pkl", crafted)), "_item_cache")


def test_a_sparse_shape_given_as_numpy_scalars_is_read_as_plain_numbers(tmp_path):
    """Review of #101: a shape of `np.uint8(255)` passed the bounds check,
    and scipy then sized its `indptr` as 255 + 1 in uint8, which is 0, and
    wrote past it converting to CSR. The shape is plain integers once
    read."""
    import warnings

    m = sp.coo_matrix(np.zeros((255, 3), dtype=np.float32))
    m.data = np.array([1.0, 2.0], dtype=np.float32)
    m.coords = (np.array([0, 254], dtype=np.int32), np.array([0, 2], dtype=np.int32))
    m._shape = (np.uint8(255), 3)
    loaded = pickled.load(_write(tmp_path / "u8.pkl", m))
    assert all(type(n) is int for n in loaded._shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        csr = loaded.tocsr()
    assert len(csr.indptr) == 256 and csr[254, 2] == 2.0


@pytest.mark.parametrize("key, value", [
    ("indices", np.array([0, 2, 1], dtype=np.uint8)),
    ("indptr", np.array([0, 2, 3], dtype=">i4")),
    ("data", np.array([1, 2, 3], dtype=">i4")),
])
def test_sparse_arrays_must_be_of_the_dtypes_scipy_makes(tmp_path, key, value):
    """scipy's compiled routines take int32 or int64 indices in the
    machine's byte order, as scipy builds them; anything else is refused
    rather than handed over."""
    _refused(_load_isolated(_write(tmp_path / "dtype.pkl", _sparse_with("csr", **{key: value}), protocol=5)), key)


def test_a_nullable_array_whose_mask_does_not_match_is_refused(tmp_path):
    arr = pd.array([1, None, 3], dtype="Int64")
    arr._mask = np.array([False])
    _refused(_load_isolated(_write(tmp_path / "masked.pkl", arr)), "mask")


@pytest.fixture
def tz_file(tmp_path):
    path = tmp_path / "zone"
    path.write_bytes(b"not a zone file")
    return path


def _tz_routes(tz: str):
    from pandas.core.indexes.base import _new_Index

    return {
        "dtype": _Crafted(pd.DatetimeTZDtype, ("ns", tz)),
        "timestamp": _Crafted(pd.Timestamp(0).__reduce__()[0], (0, None, tz, 10)),
        "index-dtype": _Crafted(_new_Index, (pd.Index, {"data": np.array(["2020-01-01"], dtype=object),
                                                        "dtype": f"datetime64[ns, {tz}]"})),
        "datetime-index": _Crafted(_new_Index, (pd.DatetimeIndex, {"data": np.array([0]), "tz": tz})),
        "interval": _Crafted(pd.IntervalDtype, (f"datetime64[ns, {tz}]",)),
        "series": _Crafted(pd.Series, (np.array(["2020-01-01"], dtype=object), None, f"datetime64[ns, {tz}]")),
    }


@pytest.mark.parametrize("route", ["dtype", "timestamp", "index-dtype", "datetime-index", "interval", "series"])
def test_a_pickle_cannot_make_pandas_open_a_file_as_a_timezone(tmp_path, tz_file, route):
    """F140: pandas reads a timezone named `dateutil/<path>` from that
    path, so a pickle naming one made the process open any local file."""
    path = _write(tmp_path / "tz.pkl", _tz_routes(f"dateutil/{tz_file}")[route])
    _refused(_load_isolated(path, watched=tz_file))


def test_a_class_is_never_constructed_from_a_pickles_arguments(tmp_path):
    """pandas' and scipy's classes are created empty and given their state;
    their constructors, which parse dtypes and build from anything, are not
    run on what a file holds."""
    path = _write(tmp_path / "frame.pkl", _Crafted(pd.DataFrame, ({"a": [1, 2]},)))
    _refused(_load_isolated(path), "pandas.DataFrame")
    path = _write(tmp_path / "csr.pkl", _Crafted(sp.csr_matrix, ((np.array([1]), np.array([5]), np.array([0, 1])),)))
    _refused(_load_isolated(path), "csr_matrix")
    path = _write(tmp_path / "recon.pkl", _Crafted(copyreg._reconstructor, (pd.DataFrame, pd.DataFrame, {"a": [1]})))
    _refused(_load_isolated(path), "_reconstructor")


# ---------------------------------------------------------------------------
# what pycisTopic writes still reads


@pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
def test_real_cistopic_pickles_read_under_every_protocol(tmp_path, protocol):
    truth = write_cistopic_pickle(tmp_path / "obj.pkl", harmony=True)
    _write(truth.path, truth.obj, protocol=protocol)
    obj = pickled.load(truth.path)
    assert np.array_equal(obj.fragment_matrix.toarray(), truth.obj.fragment_matrix.toarray())
    pd.testing.assert_frame_equal(obj.cell_data, truth.obj.cell_data)
    pd.testing.assert_frame_equal(obj.selected_model.cell_topic, truth.obj.selected_model.cell_topic)

    imputed = write_imputed_pickle(tmp_path / "imputed.pkl")
    _write(imputed.path, imputed.obj, protocol=protocol)
    assert np.array_equal(pickled.load(imputed.path).mtx, imputed.obj.mtx)


def test_what_numpy_and_pandas_pickle_besides_reads_back_equal(tmp_path):
    """Values a pycisTopic object may hold beyond the fixtures: numpy
    scalars, big-endian and Fortran-ordered arrays, text and bytes, empty
    arrays, nullable columns with missing values (pd.NA), sets."""
    with pd.option_context("future.infer_string", False):
        frame = pd.DataFrame({
            "n": pd.array([1, None], dtype="Int64"),
            "f": pd.array([0.5, None], dtype="Float64"),
            "b": pd.array([True, None], dtype="boolean"),
            "s": pd.array(["a", None], dtype=pd.StringDtype("python")),
            "c": pd.Categorical(["x", None]),
        }, index=pd.Index(["c0", "c1"], name="cell"))
        multi = pd.Series([1, 2], index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
    value = {
        "scalars": [np.int64(3), np.float32(0.5), np.bool_(True), np.str_("x")],
        "big": np.arange(4, dtype=">i4"),
        "fortran": np.asfortranarray(np.arange(6.0).reshape(2, 3)),
        "text": np.array(["a", "bb"]), "bytes": np.array([b"a", b"bb"]),
        "objects": np.array(["a", 1, None], dtype=object),
        "empty": np.zeros((0, 3)),
        "frame": frame,
        "multi": multi,
        "sets": ({1, 2}, frozenset({3})),
    }
    for protocol in (0, 2, 4, 5):
        got = pickled.load(_write(tmp_path / f"p{protocol}.pkl", value, protocol=protocol))
        expected = pickle.loads(pickle.dumps(value, protocol=protocol))  # numpy's own reading
        assert got["scalars"] == value["scalars"]
        for key in ("big", "fortran", "text", "bytes", "objects", "empty"):
            assert got[key].dtype == expected[key].dtype and np.array_equal(got[key], value[key]), key
        assert got["fortran"].flags.f_contiguous
        pd.testing.assert_frame_equal(got["frame"], frame)
        pd.testing.assert_series_equal(got["multi"], value["multi"])
        assert got["sets"] == value["sets"]


def test_a_pandas_15_pickle_reads_with_pd_na_in_its_text(tmp_path):
    """pandas 1.5 (pycisTopic's) pickles a nullable text column's missing
    values as `pandas._libs.missing.NA`, which the allowlist lacked."""
    data = _assemble(_global("pandas._libs.missing", "NA"))
    assert pickled.load(_bytes(tmp_path / "na.pkl", data)) is pd.NA

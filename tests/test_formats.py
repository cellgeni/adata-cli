"""Tests for the per-format export and import paths.

These are the surfaces that touch real file formats -- .npy, .mtx, .png, JSON
-- where a mistake produces a file that another tool rejects rather than an
exception here.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys

import numpy as np
import pytest
from rich.console import Console

from adata.elements import spec
from adata.elements import write as ew
from adata.formats.array import export_npy, import_npy
from adata.formats.image import export_image, import_image
from adata.formats.json_data import export_json, import_json
from adata.formats.mtx import read_header
from adata.formats.sparse import export_mtx, import_mtx
from adata.storage import open_store

console = Console(stderr=True)


@pytest.fixture
def store(new_store):
    path, opener = new_store()
    return path, opener


# ---------------------------------------------------------------------------
# images


@pytest.fixture
def png(temp_dir):
    from PIL import Image

    def _make(array, name="img.png"):
        path = temp_dir / name
        Image.fromarray(array).save(path)
        return path

    return _make


@pytest.mark.parametrize(
    "shape", [(4, 6), (4, 6, 3), (4, 6, 4)], ids=["gray", "rgb", "rgba"]
)
def test_image_round_trips(store, png, temp_dir, shape):
    path, opener = store
    array = (np.random.default_rng(0).random(shape) * 255).astype("uint8")
    source = png(array)

    with opener("a") as root:
        import_image(root, "uns/img", source, console)
    with opener("r") as root:
        assert root["uns"]["img"].shape == shape
        out = temp_dir / "out.png"
        export_image(root, "uns/img", out, console)

    from PIL import Image

    assert np.array_equal(np.asarray(Image.open(out)), array)


def test_image_export_scales_unit_range_floats(store, temp_dir):
    """Floats in [0, 1] are a normalised image and scale up to 0-255."""
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "img", np.array([[0.0, 0.5], [1.0, 0.25]]))
        out = temp_dir / "f.png"
        export_image(root, "uns/img", out, console)

    from PIL import Image

    got = np.asarray(Image.open(out))
    assert got[0, 0] == 0 and got[1, 0] == 255
    assert 120 <= got[0, 1] <= 135


def test_image_export_clips_floats_already_in_byte_range(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "img", np.array([[-5.0, 300.0], [10.0, 20.0]]))
        out = temp_dir / "f.png"
        export_image(root, "uns/img", out, console)

    from PIL import Image

    got = np.asarray(Image.open(out))
    assert got[0, 0] == 0, "negatives clip to 0"
    assert got[0, 1] == 255, "values above 255 clip to 255"


def test_image_export_handles_booleans(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "img", np.array([[True, False], [False, True]]))
        out = temp_dir / "b.png"
        export_image(root, "uns/img", out, console)

    from PIL import Image

    got = np.asarray(Image.open(out))
    assert got[0, 0] == 255 and got[0, 1] == 0


def test_image_export_squeezes_a_single_channel(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(
            root["uns"], "img", np.zeros((3, 4, 1), dtype="uint8")
        )
        out = temp_dir / "s.png"
        export_image(root, "uns/img", out, console)

    from PIL import Image

    assert np.asarray(Image.open(out)).shape == (3, 4)


def test_image_export_rejects_a_bad_channel_count(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "img", np.zeros((3, 4, 2), dtype="uint8"))
        with pytest.raises(ValueError, match="channels"):
            export_image(root, "uns/img", temp_dir / "x.png", console)


def test_image_export_rejects_wrong_dimensionality(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "img", np.zeros((2, 2, 2, 2), dtype="uint8"))
        with pytest.raises(ValueError, match="2D or 3D"):
            export_image(root, "uns/img", temp_dir / "x.png", console)


def test_image_export_rejects_a_group(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        with pytest.raises(ValueError, match="requires a dataset"):
            export_image(root, "uns", temp_dir / "x.png", console)


def test_image_export_rejects_an_unsupported_dtype(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(
            root["uns"], "img", np.array([[1 + 2j, 3 + 4j]], dtype="complex64")
        )
        with pytest.raises(ValueError, match="dtype"):
            export_image(root, "uns/img", temp_dir / "x.png", console)


def test_image_import_creates_intermediate_groups(store, png):
    path, opener = store
    source = png(np.zeros((2, 2, 3), dtype="uint8"))
    with opener("a") as root:
        import_image(root, "uns/spatial/sample/hires", source, console)
    with opener("r") as root:
        assert spec.encoding_type(root["uns"]["spatial"]) == spec.DICT
        assert root["uns"]["spatial"]["sample"]["hires"].shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# dense arrays


@pytest.mark.parametrize(
    "array",
    [
        np.arange(10, dtype="int32"),
        np.arange(12, dtype="float32").reshape(3, 4),
        np.arange(24, dtype="float64").reshape(2, 3, 4),
        np.array(7.5),
    ],
    ids=["1d", "2d", "3d", "scalar"],
)
def test_npy_round_trips(store, temp_dir, array):
    path, opener = store
    source = temp_dir / "in.npy"
    np.save(source, array)

    with opener("a") as root:
        import_npy(root, "uns/a", source, console)
    with opener("r") as root:
        out = temp_dir / "out.npy"
        export_npy(root, "uns/a", out, chunk_elements=5, console=console)

    assert np.array_equal(np.load(out), array)


def test_npy_export_streams_in_small_chunks(store, temp_dir):
    """A chunk size below the array length exercises the streaming loop."""
    path, opener = store
    array = np.arange(100, dtype="int64").reshape(25, 4)
    with opener("a") as root:
        ew.write_dense(root["uns"], "a", array)
        out = temp_dir / "out.npy"
        export_npy(root, "uns/a", out, chunk_elements=8, console=console)
    assert np.array_equal(np.load(out), array)


@pytest.mark.parametrize(
    "values,mask,encoding,pandas_dtype",
    [
        ([1, 2, 3], [False, True, False], spec.NULLABLE_INTEGER, "Int32"),
        ([True, False, True], [False, False, True], spec.NULLABLE_BOOLEAN, "boolean"),
    ],
    ids=["integer", "boolean"],
)
def test_npy_export_of_a_nullable_array_keeps_its_gaps_as_nan(
    store, temp_dir, values, mask, encoding, pandas_dtype
):
    """F063: the mask was dropped, so a missing entry was exported as the
    0 or False stored under it."""
    pd = pytest.importorskip("pandas")
    path, opener = store
    with opener("a") as root:
        ew.write_masked(root["uns"], "m", np.array(values, dtype="int32" if encoding == spec.NULLABLE_INTEGER else bool), mask, encoding)
        out = temp_dir / "m.npy"
        export_npy(root, "uns/m", out, chunk_elements=2, console=console)
    expected = pd.array([None if m else v for v, m in zip(values, mask)], dtype=pandas_dtype)
    got = np.load(out)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, expected.to_numpy(dtype="float64", na_value=np.nan))


def test_npy_export_of_a_nullable_array_with_no_gaps_keeps_its_dtype(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_masked(root["uns"], "m", np.array([1, 2, 3], dtype="int32"), [False] * 3, spec.NULLABLE_INTEGER)
        out = temp_dir / "m.npy"
        export_npy(root, "uns/m", out, chunk_elements=100, console=console)
    got = np.load(out)
    assert got.dtype == np.int32 and got.tolist() == [1, 2, 3]


@pytest.mark.parametrize("nullable", [False, True], ids=["string-array", "nullable-string-array"])
def test_npy_export_of_text_writes_fixed_width_unicode(store, temp_dir, nullable):
    """F064: no text array could be exported; each failed with "Array can't
    be memory-mapped: Python objects in dtype", even the
    nullable-string-array the command said it handled."""
    path, opener = store
    labels = ["a", "βeta", "", "a longer label"]
    with opener("a") as root:
        if nullable:
            ew.write_masked(root["uns"], "t", labels, [False] * 4, spec.NULLABLE_STRING_ARRAY)
        else:
            ew.write_string_array(root["uns"], "t", labels)
        out = temp_dir / "t.npy"
        export_npy(root, "uns/t", out, chunk_elements=3, console=console)
    got = np.load(out)
    assert got.dtype == np.dtype("<U14")
    assert got.tolist() == labels


def test_npy_export_refuses_text_with_gaps_before_writing(store, temp_dir):
    """A .npy of text cannot hold a missing value: refused, not written as ''."""
    path, opener = store
    with opener("a") as root:
        ew.write_masked(root["uns"], "t", ["a", "b"], [False, True], spec.NULLABLE_STRING_ARRAY)
        with pytest.raises(ValueError, match="has 1 missing values, which a .npy of text cannot hold"):
            export_npy(root, "uns/t", temp_dir / "t.npy", 100, console)
    assert not (temp_dir / "t.npy").exists()


class _Digest(io.RawIOBase):
    """A stdout that keeps only a hash of what it is given."""

    def __init__(self):
        self.hash = hashlib.sha256()

    def writable(self):
        return True

    def write(self, data):
        self.hash.update(data)
        return len(data)


def test_npy_export_streams_rather_than_mapping_the_output(store, temp_dir, monkeypatch):
    """F080: the output was a writable memmap, so the whole array ended up
    in memory as dirty pages, which tracemalloc cannot see; and the stdout
    path loaded the array whole. Both now hold a block at a time."""
    import tracemalloc

    from adata.storage import create_dataset

    def no_memmap(*args, **kwargs):
        raise AssertionError("open_memmap called")

    monkeypatch.setattr(np.lib.format, "open_memmap", no_memmap)
    digest = _Digest()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BufferedWriter(digest)))
    path, opener = store
    array = np.arange(1 << 20, dtype="float64").reshape(-1, 4)
    with opener("a") as root:
        ds = create_dataset(root["uns"], "big", shape=array.shape, dtype=array.dtype, chunks=(1024, 4))
        ds[...] = array
        spec.set_encoding(ds, spec.ARRAY)
    for out in (temp_dir / "big.npy", None):
        with opener("r") as root:
            tracemalloc.start()
            try:
                export_npy(root, "uns/big", out, chunk_elements=4096, console=console)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        assert peak < array.nbytes / 16, f"{peak:,} bytes to export {array.nbytes:,} ({out})"
    np.testing.assert_array_equal(np.load(temp_dir / "big.npy"), array)
    assert digest.hash.hexdigest() == hashlib.sha256((temp_dir / "big.npy").read_bytes()).hexdigest()


def test_npy_export_rejects_a_group_it_cannot_unwrap(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_mapping(root["uns"], "m")
        with pytest.raises(ValueError, match="cannot export as .npy"):
            export_npy(root, "uns/m", temp_dir / "x.npy", 100, console)


def test_npy_export_to_stdout(store, capsysbinary):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "a", np.arange(6, dtype="float32"))
        export_npy(root, "uns/a", None, chunk_elements=100, console=console)
    captured = capsysbinary.readouterr().out
    assert captured.startswith(b"\x93NUMPY")


def test_npy_import_creates_intermediate_groups(store, temp_dir):
    path, opener = store
    source = temp_dir / "in.npy"
    np.save(source, np.zeros((2, 2)))
    with opener("a") as root:
        import_npy(root, "uns/deep/nested/a", source, console)
    with opener("r") as root:
        assert root["uns"]["deep"]["nested"]["a"].shape == (2, 2)


# ---------------------------------------------------------------------------
# sparse matrices


def _read_mtx(path):
    """(0-based entries, shape, nnz) of an MTX file, read by scipy as the oracle."""
    mmread = pytest.importorskip("scipy.io").mmread
    coo = mmread(str(path)).tocoo()
    entries = [
        (int(r), int(c), v.item()) for r, c, v in zip(coo.row, coo.col, coo.data)
    ]
    return entries, coo.shape, read_header(path).nnz


def _write_mtx(path, header, dims, entries):
    lines = [header, "% a comment"]
    lines.append(" ".join(str(d) for d in dims))
    lines.extend(" ".join(str(v) for v in e) for e in entries)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_mtx_round_trips(store, temp_dir):
    path, opener = store
    source = _write_mtx(
        temp_dir / "m.mtx",
        "%%MatrixMarket matrix coordinate real general",
        (3, 2, 3),
        [(1, 1, 1.5), (2, 2, 2.5), (3, 1, 3.5)],
    )
    with opener("a") as root:
        import_mtx(root, "uns/m", source, console)
    with opener("r") as root:
        out = temp_dir / "out.mtx"
        export_mtx(root, "uns/m", out, None, 10, False, console)

    entries, dims, nnz = _read_mtx(out)
    assert dims == (3, 2)
    assert nnz == 3
    assert sorted(entries) == [(0, 0, 1.5), (1, 1, 2.5), (2, 0, 3.5)]


def test_mtx_pattern_field_imports_as_ones(store, temp_dir):
    path, opener = store
    source = _write_mtx(
        temp_dir / "p.mtx",
        "%%MatrixMarket matrix coordinate pattern general",
        (2, 2, 2),
        [(1, 1), (2, 2)],
    )
    with opener("a") as root:
        import_mtx(root, "uns/p", source, console)
    with opener("r") as root:
        data = np.asarray(root["uns"]["p"]["data"][...])
    assert data.tolist() == [1, 1]
    assert data.dtype == np.int32


def test_mtx_rejects_a_missing_header(temp_dir):
    bad = temp_dir / "bad.mtx"
    bad.write_text("1 1 1\n1 1 1.0\n")
    with pytest.raises(ValueError, match="not a Matrix Market file"):
        read_header(bad)


def test_mtx_export_in_memory_matches_streaming(store, temp_dir):
    """The --in-memory fast path must produce the same file as streaming."""
    path, opener = store
    with opener("a") as root:
        ew.write_sparse(
            root["uns"], "m",
            [1.0, 2.0, 3.0, 4.0], [0, 2, 1, 2], [0, 2, 3, 4], (3, 3),
        )
        streamed = temp_dir / "s.mtx"
        in_memory = temp_dir / "m.mtx"
        export_mtx(root, "uns/m", streamed, None, 1, False, console)
        export_mtx(root, "uns/m", in_memory, None, 1, True, console)

    a, dims_a, _ = _read_mtx(streamed)
    b, dims_b, _ = _read_mtx(in_memory)
    assert dims_a == dims_b
    assert sorted(a) == sorted(b)


def test_mtx_export_head_limits_entries(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_sparse(
            root["uns"], "m",
            [1.0, 2.0, 3.0, 4.0], [0, 2, 1, 2], [0, 2, 3, 4], (3, 3),
        )
        out = temp_dir / "h.mtx"
        export_mtx(root, "uns/m", out, 2, 10, False, console)
    _, _, nnz = _read_mtx(out)
    assert nnz == 2


def test_mtx_export_of_csc_uses_column_major_order(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_sparse(
            root["uns"], "m",
            [1.0, 2.0], [0, 1], [0, 1, 2], (2, 2), spec.CSC_MATRIX,
        )
        out = temp_dir / "c.mtx"
        export_mtx(root, "uns/m", out, None, 10, False, console)
    entries, dims, _ = _read_mtx(out)
    assert dims == (2, 2)
    assert sorted(entries) == [(0, 0, 1.0), (1, 1, 2.0)]


def test_mtx_export_rejects_a_dataset(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "d", np.zeros((2, 2)))
        with pytest.raises(ValueError, match="CSR/CSC matrix group"):
            export_mtx(root, "uns/d", temp_dir / "x.mtx", None, 10, False, console)


def test_mtx_export_rejects_a_non_sparse_group(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_mapping(root["uns"], "m")
        with pytest.raises(ValueError, match="expected 'csr_matrix'"):
            export_mtx(root, "uns/m", temp_dir / "x.mtx", None, 10, False, console)


def test_mtx_export_detects_inconsistent_sparse_data(store, temp_dir):
    """indptr claiming more nonzeros than data holds must not be exported."""
    path, opener = store
    with opener("a") as root:
        group = root["uns"].create_group("m")
        spec.set_encoding(group, spec.CSR_MATRIX)
        ew.set_shape_attr(group, (2, 2))
        ew.write_dense(group, "data", np.array([1.0]))
        ew.write_dense(group, "indices", np.array([0]))
        ew.write_dense(group, "indptr", np.array([0, 1, 5]))
        with pytest.raises(ValueError, match="inconsistency"):
            export_mtx(root, "uns/m", temp_dir / "x.mtx", None, 10, False, console)


def test_mtx_export_requires_a_shape_attribute(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        group = root["uns"].create_group("m")
        spec.set_encoding(group, spec.CSR_MATRIX)
        ew.write_dense(group, "data", np.array([1.0]))
        ew.write_dense(group, "indices", np.array([0]))
        ew.write_dense(group, "indptr", np.array([0, 1]))
        with pytest.raises(ValueError, match="'shape' attribute"):
            export_mtx(root, "uns/m", temp_dir / "x.mtx", None, 10, False, console)


# ---------------------------------------------------------------------------
# JSON


def test_json_round_trips_every_scalar_kind(store, temp_dir):
    path, opener = store
    source = {
        "text": "hello",
        "count": 42,
        "rate": 0.25,
        "flag": True,
        "labels": ["a", "b"],
        "numbers": [1, 2, 3],
        "grid": [["a", "b"], ["c", "d"]],
        "nested": {"deep": {"x": 1}},
    }
    payload = temp_dir / "in.json"
    payload.write_text(json.dumps(source))

    with opener("a") as root:
        import_json(root, "uns/t", payload, console)
    with opener("r") as root:
        out = temp_dir / "out.json"
        export_json(root, "uns/t", out, 1000, False, console)

    assert json.loads(out.read_text()) == source


def test_json_ragged_list_is_kept_as_text(store, temp_dir):
    """A ragged list has no array representation, so it is stored verbatim."""
    path, opener = store
    payload = temp_dir / "r.json"
    payload.write_text('{"ragged": [[1, 2], [3]]}')

    with opener("a") as root:
        import_json(root, "uns/t", payload, console)
    with opener("r") as root:
        assert spec.encoding_type(root["uns"]["t"]["ragged"]) == spec.STRING


def test_json_export_refuses_an_oversized_array(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "big", np.arange(100))
        with pytest.raises(ValueError, match="max 10"):
            export_json(root, "uns/big", temp_dir / "x.json", 10, False, console)


def test_json_export_refuses_a_sparse_matrix(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_sparse(root["uns"], "m", [1.0], [0], [0, 1], (1, 2))
        with pytest.raises(ValueError, match="Export it as .mtx"):
            export_json(root, "uns/m", temp_dir / "x.json", 1000, False, console)


def test_json_export_can_include_attributes(store, temp_dir):
    path, opener = store
    with opener("a") as root:
        ew.write_categorical(root["uns"], "c", [0, 1], ["x", "y"], ordered=True)
        out = temp_dir / "a.json"
        export_json(root, "uns/c", out, 1000, True, console)

    payload = json.loads(out.read_text())
    assert payload["__attrs__"]["encoding-type"] == "categorical"
    assert payload["__attrs__"]["ordered"] is True


def test_json_export_to_stdout(store, capsys):
    path, opener = store
    with opener("a") as root:
        ew.write_scalar(root["uns"], "s", "hi")
        export_json(root, "uns/s", None, 1000, False, console)
    assert json.loads(capsys.readouterr().out) == "hi"


def test_json_import_rejects_an_unrepresentable_value(store, temp_dir):
    from adata.formats.json_data import write_json_value

    path, opener = store
    with opener("a") as root:
        with pytest.raises(ValueError, match="Cannot convert"):
            write_json_value(root["uns"], "bad", {1, 2})


def test_json_export_writes_null_for_nan_and_infinity(store, temp_dir):
    """F128: they were written as the tokens NaN and Infinity, which are not
    JSON: jq and strict parsers refused the whole file."""
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["uns"], "v", np.array([np.nan, np.inf, -np.inf, 1.5]))
        ew.write_scalar(root["uns"], "s", float("nan"))
        out = temp_dir / "v.json"
        export_json(root, "uns", out, 1000, False, console)

    def refuse(token):
        raise ValueError(f"not JSON: {token}")

    payload = json.loads(out.read_text(), parse_constant=refuse)
    assert payload["v"] == [None, None, None, 1.5] and payload["s"] is None


# ---------------------------------------------------------------------------
# dataframes


def _frame_store(opener):
    from adata.formats.table import scan_table  # noqa: F401  (the writer import dataframe uses)

    with opener("a") as root:
        obs = root["obs"]
        ew.write_dense(obs, "n", np.array([1, 2, 3]))
        ew.write_string_array(obs, "label", ["x", "y, z", "w"])
        ew.set_column_order(obs, ["n", "label"])


def test_dataframe_export_refuses_an_unknown_column_before_writing(store, temp_dir):
    """F065: --columns was checked only while rows were written, so a typo
    left a CSV holding just the header."""
    from adata.formats.dataframe import export_dataframe

    path, opener = store
    _frame_store(opener)
    out = temp_dir / "obs.csv"
    with opener("r") as root:
        with pytest.raises(ValueError, match="'obs' has no column 'nope'; it has: n, label"):
            export_dataframe(root, "obs", ["n", "nope"], out, 100, None, console)
        with pytest.raises(ValueError, match="--head must be 0 or more"):
            export_dataframe(root, "obs", None, out, 100, -1, console)
    assert not out.exists()


def test_dataframe_export_reads_back_as_pandas_reads_it(store, temp_dir):
    """F127: lines ended in CRLF, and --head 0 wrote every row."""
    pd = pytest.importorskip("pandas")
    from adata.formats.dataframe import export_dataframe

    path, opener = store
    _frame_store(opener)
    out, empty = temp_dir / "obs.csv", temp_dir / "none.csv"
    with opener("r") as root:
        export_dataframe(root, "obs", None, out, 2, None, console)
        export_dataframe(root, "obs", None, empty, 2, 0, console)

    assert b"\r" not in out.read_bytes()
    got = pd.read_csv(out, index_col=0)
    assert got["n"].tolist() == [1, 2, 3] and got["label"].tolist() == ["x", "y, z", "w"]
    assert empty.read_text() == "_index,n,label\n"


def test_mtx_export_head_zero_writes_no_entries(store, temp_dir):
    """F127: --head 0 exported every entry, as if no limit were given."""
    from scipy import io as scipy_io

    path, opener = store
    with opener("a") as root:
        ew.write_sparse(root["uns"], "m", [1.0, 2.0], [0, 1], [0, 1, 2], (2, 2))
        out = temp_dir / "m.mtx"
        export_mtx(root, "uns/m", out, 0, 100, False, console)
        with pytest.raises(ValueError, match="--head must be 0 or more"):
            export_mtx(root, "uns/m", temp_dir / "n.mtx", -1, 100, False, console)
    assert scipy_io.mmread(out).nnz == 0


# ---------------------------------------------------------------------------
# stdout


@pytest.fixture(scope="module")
def long_store(tmp_path_factory):
    """A store whose every export is far longer than a pipe's buffer."""
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    from scipy import sparse

    n = 40_000
    obj = ad.AnnData(
        X=sparse.csr_matrix(np.ones((n, 4), dtype="float32")),
        obs=pd.DataFrame({"x": np.arange(n)}, index=[f"cell_{i}" for i in range(n)]),
    )
    obj.obsm["X_pca"] = np.ones((n, 4))
    obj.uns["v"] = np.arange(n, dtype="float64")
    path = tmp_path_factory.mktemp("long") / "long.h5ad"
    obj.write_h5ad(path)
    return path


@pytest.mark.parametrize(
    "args",
    [["dataframe", "obs"], ["array", "obsm/X_pca"], ["sparse", "X"], ["dict", "uns"]],
    ids=["dataframe", "array", "sparse", "dict"],
)
def test_an_export_ends_quietly_when_its_reader_stops(long_store, args):
    """F127: `adata export ... | head` printed an Error about a broken pipe
    and exited 1, although the reader had read all it wanted."""
    import subprocess

    command = [sys.executable, "-c", "from adata.cli import main; main()", "export", args[0], str(long_store), args[1]]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout.read(64)
    proc.stdout.close()
    err = proc.stderr.read().decode()
    proc.wait(timeout=120)
    assert proc.returncode == 0, err
    assert "Error" not in err and "BrokenPipe" not in err and "Exception ignored" not in err, err


def test_npy_export_refuses_a_record_array_with_text_fields(temp_dir):
    """Its fields are pointers to strings; written as bytes they would be
    addresses, not text."""
    import h5py

    path = temp_dir / "r.h5ad"
    with h5py.File(path, "w") as f:
        f.create_dataset("recs", data=np.array([("x", 1)], dtype=[("a", h5py.string_dtype()), ("b", "i4")]))
    with open_store(path, "r") as store:
        with pytest.raises(ValueError, match="record array with text fields"):
            export_npy(store.root, "recs", temp_dir / "r.npy", 10, console)
    assert not (temp_dir / "r.npy").exists()

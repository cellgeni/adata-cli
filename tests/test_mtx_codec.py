"""The streaming Matrix Market codec, and `import sparse` / `export sparse` on it.

scipy's `mmread` is the oracle throughout: a file this reads must give the
matrix scipy gives -- duplicates summed, coordinates 1-based -- and a file
this writes must read back in scipy as the matrix that was written.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from adata.cli import app
from adata.core.csr_build import CsrAppender
from adata.formats.mtx import (
    EntryMap,
    iter_entries,
    read_header,
    read_into,
    scan,
    value_dtype,
)
from adata.formats.sparse import export_mtx, import_mtx
from adata.formats.textio import (
    count_lines,
    format_int_rows,
    format_rows,
    iter_first_column,
    iter_line_blocks,
)

sparse = pytest.importorskip("scipy.sparse")
mmread = pytest.importorskip("scipy.io").mmread
ad = pytest.importorskip("anndata")

console = Console(stderr=True, quiet=True)
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _mtx(
    path: Path,
    entries,
    shape,
    *,
    field="integer",
    symmetry="general",
    gz=False,
    nnz=None,
    newline="\n",
    trailing_newline=True,
) -> Path:
    lines = [f"%%MatrixMarket matrix coordinate {field} {symmetry}", "% a comment"]
    lines.append(f"{shape[0]} {shape[1]} {len(entries) if nnz is None else nnz}")
    lines += [" ".join(str(v) for v in entry) for entry in entries]
    text = newline.join(lines) + (newline if trailing_newline else "")
    data = text.encode()
    path.write_bytes(gzip.compress(data) if gz else data)
    return path


def _random_entries(rng, shape, nnz, *, duplicates=0):
    flat = rng.choice(shape[0] * shape[1], size=nnz, replace=False)
    rows, cols = flat // shape[1], flat % shape[1]
    values = rng.integers(1, 50, size=nnz)
    entries = [(int(r) + 1, int(c) + 1, int(v)) for r, c, v in zip(rows, cols, values)]
    entries += entries[:duplicates]
    return entries


def _import(tmp_path, source, *, chunk=None, block_bytes=None, n_targets=1):
    """Read `source` into a fresh h5ad-like file through the two passes."""
    header = read_header(source)
    emap = EntryMap(n_rows=header.n_rows, n_cols=header.n_cols)
    kw = {"block_bytes": block_bytes} if block_bytes else {}
    scanned = scan(header, emap, **kw)
    out = tmp_path / "out.h5"
    with h5py.File(out, "w") as handle:
        appender = CsrAppender(
            handle, "X", shape=(header.n_rows, header.n_cols),
            data_dtype=value_dtype(header, scanned),
        )
        summed = read_into(
            header, emap, scanned, [[appender]], handle,
            **({"chunk": chunk} if chunk else {}), **kw,
        )
        appender.finish()
        group = handle["X"]
        matrix = sparse.csr_matrix(
            (group["data"][...], group["indices"][...], group["indptr"][...]),
            shape=(header.n_rows, header.n_cols),
        )
        dtypes = {k: group[k].dtype for k in ("data", "indices", "indptr")}
    return matrix, dtypes, summed, scanned


# ---------------------------------------------------------------------------
# reading


@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzipped"])
@pytest.mark.parametrize("order", ["sorted", "shuffled"])
def test_a_file_reads_as_scipy_reads_it(tmp_path, gz, order):
    rng = np.random.default_rng(0)
    entries = _random_entries(rng, (60, 40), 500)
    if order == "sorted":
        entries.sort()
    source = _mtx(tmp_path / ("m.mtx.gz" if gz else "m.mtx"), entries, (60, 40), gz=gz)

    matrix, dtypes, summed, scanned = _import(tmp_path, source, chunk=64, block_bytes=200)

    expected = mmread(str(source)).tocsr()
    assert (matrix != expected).nnz == 0
    assert matrix.has_sorted_indices
    assert summed == 0
    assert scanned.grouped == (order == "sorted")
    assert dtypes == {"data": np.int32, "indices": np.int32, "indptr": np.int32}


def test_duplicate_entries_are_summed_as_scipy_sums_them(tmp_path):
    rng = np.random.default_rng(1)
    entries = _random_entries(rng, (30, 20), 100, duplicates=10)
    rng.shuffle(entries)
    source = _mtx(tmp_path / "d.mtx", entries, (30, 20))

    matrix, _, summed, _ = _import(tmp_path, source, chunk=16)

    assert summed == 10
    assert (matrix != mmread(str(source)).tocsr()).nnz == 0


def test_duplicates_in_a_sorted_file_are_summed_too(tmp_path):
    source = _mtx(tmp_path / "d.mtx", [(1, 1, 2), (1, 1, 3), (2, 2, 1)], (2, 2))
    matrix, _, summed, scanned = _import(tmp_path, source)
    assert scanned.grouped and not scanned.canonical
    assert summed == 1
    assert matrix.toarray().tolist() == [[5, 0], [0, 1]]


@pytest.mark.parametrize(
    "field,value,dtype",
    [
        ("integer", 7, np.int32),
        ("integer", 2**31, np.int64),
        ("integer", -(2**31) - 1, np.int64),
        ("real", 0.1, np.float64),
        ("pattern", None, np.int32),
    ],
)
def test_the_value_dtype_follows_the_field_and_the_range(tmp_path, field, value, dtype):
    entry = (1, 1) if value is None else (1, 1, value)
    source = _mtx(tmp_path / "v.mtx", [entry], (1, 1), field=field)
    matrix, dtypes, _, _ = _import(tmp_path, source)
    assert dtypes["data"] == dtype
    assert matrix[0, 0] == (1 if value is None else value)


def test_indptr_widens_when_nnz_needs_it(tmp_path, monkeypatch):
    import adata.core.csr_build as csr_build

    monkeypatch.setattr(csr_build, "INT32_MAX", 2)
    source = _mtx(tmp_path / "w.mtx", [(1, 1, 1), (1, 2, 1), (2, 1, 1)], (2, 2))
    _, dtypes, _, _ = _import(tmp_path, source)
    assert dtypes["indptr"] == np.int64


@pytest.mark.parametrize(
    "newline,trailing", [("\r\n", True), ("\n", False)], ids=["crlf", "no final newline"]
)
def test_line_endings_do_not_matter(tmp_path, newline, trailing):
    entries = [(1, 2, 3), (2, 1, 4)]
    source = _mtx(tmp_path / "e.mtx", entries, (2, 2), newline=newline, trailing_newline=trailing)
    matrix, _, _, _ = _import(tmp_path, source, block_bytes=8)
    assert matrix.toarray().tolist() == [[0, 3], [4, 0]]


@pytest.mark.parametrize(
    "banner,message",
    [
        ("%%MatrixMarket matrix array real general", "dense"),
        ("%%MatrixMarket matrix coordinate complex general", "'complex'"),
        ("%%MatrixMarket matrix coordinate real symmetric", "'symmetric'"),
        ("%%MatrixMarket matrix coordinate real skew-symmetric", "'skew-symmetric'"),
        ("%%MatrixMarket matrix coordinate complex hermitian", "'complex'"),
        ("%%MatrixMarket matrix coordinate integer hermitian", "'hermitian'"),
        ("%MatrixMarket matrix coordinate real general", "not a Matrix Market file"),
        ("%%MatrixMarket vector coordinate real general", "unrecognised"),
    ],
)
def test_what_cannot_be_read_is_refused_by_name(tmp_path, banner, message):
    path = tmp_path / "bad.mtx"
    path.write_text(f"{banner}\n2 2 1\n1 1 1\n")
    with pytest.raises(ValueError, match=message):
        read_header(path)


@pytest.mark.parametrize(
    "body,nnz,message",
    [
        ("1 1 1\n2 2 1\n", 3, "has 2 entries; its header says 3"),
        ("1 1 1\n2 2 1\n", 1, "more than"),
        ("1 1 1\n3 1 1\n", 2, "row index is outside"),
        ("1 1 1\n1 0 1\n", 2, "column index is outside"),
        ("1 1 1\n1 2 1.5\n", 2, "not 3 numbers"),
        ("1 1 1\n1 2\n", 2, "does not hold 3 fields"),
    ],
)
def test_a_bad_body_is_refused_before_anything_is_written(tmp_path, body, nnz, message):
    path = tmp_path / "bad.mtx"
    path.write_text(f"%%MatrixMarket matrix coordinate integer general\n2 2 {nnz}\n{body}")
    header = read_header(path)
    with pytest.raises(ValueError, match=message):
        scan(header, EntryMap(n_rows=2, n_cols=2))


@pytest.mark.parametrize("block_bytes", [None, 8], ids=["one-block", "small-blocks"])
def test_a_short_line_beside_a_long_one_is_refused_by_number(tmp_path, block_bytes):
    """F129: only a block's field total was checked, so a line a field short
    and a later one a field long paired values with the wrong coordinates
    (1 1 / 2 2 3 7 read as 1 1 2 and 2 3 7), and the file read without
    complaint."""
    path = tmp_path / "ragged.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate integer general\n% a comment\n3 3 3\n"
        "1 1\n2 2 3 7\n3 3 9\n"
    )
    pd = pytest.importorskip("pandas")
    with pytest.raises(Exception, match="Expected 2 fields in line 5, saw 4"):
        pd.read_csv(path, sep=r"\s+", comment="%", header=None, skiprows=3)
    header = read_header(path)
    kw = {"block_bytes": block_bytes} if block_bytes else {}
    with pytest.raises(ValueError, match="line 4 does not hold 3 fields: it holds 2"):
        scan(header, EntryMap(n_rows=3, n_cols=3), **kw)


@pytest.mark.parametrize("block_bytes", [None, 7, 12], ids=["one-block", "7", "12"])
def test_blank_lines_among_the_entries_are_skipped_as_scipy_skips_them(tmp_path, block_bytes):
    """A blank line was refused, unless a block happened to end on it, when
    it was dropped: what a file meant depended on where blocks fell."""
    path = tmp_path / "blank.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 3 3\n"
        "1 1 1\n\n2 2 5\n   \n3 3 9\n\n"
    )
    header = read_header(path)
    kw = {"block_bytes": block_bytes} if block_bytes else {}
    rows, cols, values = (np.concatenate(a) for a in zip(*iter_entries(header, **kw)))
    got = sparse.coo_matrix((values, (rows, cols)), shape=(3, 3)).toarray()
    assert np.array_equal(got, mmread(path).toarray())


def test_lines_are_numbered_past_blank_lines_in_the_header(tmp_path):
    """Entry lines were numbered from the comments alone, so each blank line
    in the header put every number one short."""
    path = tmp_path / "bad.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate integer general\n% c\n\n3 3 2\n1 1 1\n2 2\n"
    )
    with pytest.raises(ValueError, match="line 6 does not hold 3 fields"):
        list(iter_entries(read_header(path)))


def test_a_real_file_with_fractional_coordinates_is_refused(tmp_path):
    path = tmp_path / "bad.mtx"
    path.write_text("%%MatrixMarket matrix coordinate real general\n2 2 1\n1.5 1 1\n")
    with pytest.raises(ValueError, match="whole number"):
        list(iter_entries(read_header(path)))


# ---------------------------------------------------------------------------
# import sparse


def _store(path: Path, n_obs: int, n_var: int) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix((n_obs, n_var), dtype=np.float32),
        obs={"_": np.zeros(n_obs)},
        var={"_": np.zeros(n_var)},
    ).write_h5ad(path)
    return path


def test_import_sparse_reads_a_gzipped_file_and_keeps_integers(tmp_path):
    rng = np.random.default_rng(2)
    entries = _random_entries(rng, (8, 5), 20)
    source = _mtx(tmp_path / "counts.mtx.gz", entries, (8, 5), gz=True)
    store = _store(tmp_path / "s.h5ad", 8, 5)

    result = runner.invoke(
        app, ["import", "sparse", str(store), "layers/counts", str(source), "--inplace"]
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(store).layers["counts"]
    assert got.dtype == np.int32
    assert (got != mmread(str(source)).tocsr()).nnz == 0
    with h5py.File(store, "r") as handle:
        assert handle["layers"].attrs["encoding-type"] == "dict"
        assert handle["layers/counts/data"].compression == "gzip"


def test_import_sparse_refuses_a_symmetric_file_and_leaves_the_store_alone(tmp_path):
    source = _mtx(tmp_path / "s.mtx", [(1, 1, 1)], (8, 5), symmetry="symmetric")
    store = _store(tmp_path / "s.h5ad", 8, 5)

    result = runner.invoke(
        app, ["import", "sparse", str(store), "X", str(source), "--inplace"]
    )
    assert result.exit_code == 1
    assert "'symmetric'" in _out(result)
    assert ad.read_h5ad(store).X.nnz == 0, "X must be the store's own, untouched"


def test_import_sparse_checks_the_whole_file_before_touching_the_store(tmp_path):
    path = tmp_path / "bad.mtx"
    path.write_text("%%MatrixMarket matrix coordinate integer general\n8 5 2\n1 1 1\n9 1 1\n")
    store = _store(tmp_path / "s.h5ad", 8, 5)
    before = ad.read_h5ad(store).X.toarray()

    result = runner.invoke(app, ["import", "sparse", str(store), "X", str(path), "--inplace"])
    assert result.exit_code == 1
    assert "outside" in _out(result)
    assert np.array_equal(ad.read_h5ad(store).X.toarray(), before)


# ---------------------------------------------------------------------------
# export sparse


def _store_with(path: Path, matrix) -> Path:
    ad.AnnData(X=matrix).write_h5ad(path)
    return path


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_export_sparse_reads_back_in_scipy(tmp_path, layout):
    rng = np.random.default_rng(3)
    matrix = sparse.random(40, 30, density=0.2, format=layout, random_state=rng, dtype=np.float32)
    store = _store_with(tmp_path / "s.h5ad", matrix)
    out = tmp_path / "x.mtx.gz"

    result = runner.invoke(app, ["export", "sparse", str(store), "X", "-o", str(out), "-C", "3"])
    assert result.exit_code == 0, _out(result)

    with gzip.open(out, "rb") as fh:
        assert fh.read(14) == b"%%MatrixMarket"
    back = mmread(str(out)).tocsr()
    assert back.dtype == np.float64
    assert np.array_equal(back.toarray().astype(np.float32), matrix.toarray()), (
        "float32 values must survive the text exactly"
    )


def test_export_sparse_writes_whole_numbers_as_integers(tmp_path):
    store = _store_with(tmp_path / "s.h5ad", sparse.csr_matrix(np.array([[0, 3.0], [2.0, 0]])))
    out = tmp_path / "x.mtx"
    assert runner.invoke(app, ["export", "sparse", str(store), "X", "-o", str(out)]).exit_code == 0
    assert out.read_text().splitlines()[-2:] == ["1 2 3", "2 1 2"]


def test_export_sparse_in_memory_writes_the_same_bytes(tmp_path):
    rng = np.random.default_rng(4)
    matrix = sparse.random(50, 20, density=0.3, format="csr", random_state=rng)
    store = _store_with(tmp_path / "s.h5ad", matrix)
    streamed, loaded = tmp_path / "a.mtx", tmp_path / "b.mtx"
    with h5py.File(store, "r") as handle:
        export_mtx(handle, "X", streamed, None, 2, False, console)
        export_mtx(handle, "X", loaded, None, 2, True, console)
    assert streamed.read_bytes() == loaded.read_bytes()


def test_export_then_import_is_the_identity(tmp_path):
    rng = np.random.default_rng(5)
    matrix = sparse.random(25, 15, density=0.3, format="csr", random_state=rng)
    matrix.data = np.round(matrix.data * 100).astype(np.int64)
    matrix = matrix.astype(np.int32)
    store = _store_with(tmp_path / "s.h5ad", matrix)
    out = tmp_path / "x.mtx"
    with h5py.File(store, "a") as handle:
        export_mtx(handle, "X", out, None, 4, False, console)
        import_mtx(handle, "layers/back", out, console)
    back = ad.read_h5ad(store).layers["back"]
    assert back.dtype == np.int32
    assert (back != matrix).nnz == 0


# ---------------------------------------------------------------------------
# text helpers


def test_line_blocks_end_on_whole_lines():
    text = b"".join(f"{i} {i * 7}\n".encode() for i in range(1000))
    for size in (1, 7, 64, 10_000):
        blocks = list(iter_line_blocks(io.BytesIO(text), size))
        assert b"".join(blocks) == text
        assert all(block.endswith(b"\n") for block in blocks)


def test_count_lines_and_first_column(tmp_path):
    path = tmp_path / "barcodes.tsv.gz"
    path.write_bytes(gzip.compress(b"AAA-1\textra\nCCC-1\nGGG-1\n\n"))
    assert count_lines(path) == 3
    assert [b for block in iter_first_column(path, 2) for b in block] == [
        "AAA-1", "CCC-1", "GGG-1",
    ]


def test_the_integer_formatter_matches_python():
    rng = np.random.default_rng(6)
    a = rng.integers(-10**12, 10**12, size=500)
    b = np.concatenate(([0, -1, 9, 10, 99, 100], rng.integers(0, 10**6, size=494)))
    expected = "".join(f"{x} {y}\n" for x, y in zip(a, b)).encode()
    assert format_int_rows([a, b]) == expected


def test_float_values_are_written_to_round_trip():
    values = np.array([0.1, 1 / 3, 1e-30, 2.5], dtype=np.float32)
    text = format_rows([np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1])], values)
    read = np.array([float(line.split()[2]) for line in text.decode().splitlines()], dtype=np.float32)
    assert np.array_equal(read, values)


@pytest.mark.parametrize("order", ["sorted", "shuffled"])
def test_the_file_is_read_twice_and_no_more(tmp_path, monkeypatch, order):
    """Two passes: one to count and check, one to write. A third would be a
    regression that no element counter sees, since the file is text."""
    import adata.formats.mtx as mtx_module

    rng = np.random.default_rng(7)
    entries = _random_entries(rng, (200, 50), 2000)
    if order == "sorted":
        entries.sort()
    source = _mtx(tmp_path / "m.mtx", entries, (200, 50))
    body = source.read_bytes()
    body_bytes = len(body) - len(b"".join(body.splitlines(keepends=True)[:3]))

    read = {"bytes": 0}
    original = mtx_module.iter_line_blocks

    def counting(fh, block_bytes=mtx_module.TEXT_BLOCK_BYTES):
        for block in original(fh, block_bytes):
            read["bytes"] += len(block)
            yield block

    monkeypatch.setattr(mtx_module, "iter_line_blocks", counting)
    _import(tmp_path, source, chunk=64, block_bytes=1000)
    assert read["bytes"] == 2 * body_bytes


# ---------------------------------------------------------------------------
# found in review: integer limits


def test_duplicates_that_sum_past_int32_choose_int64_before_anything_is_written(tmp_path):
    """Each value fits int32 but their sum does not.

    The dtype is chosen in pass 1, before the store is touched; choosing int32
    here would only be found out in pass 2, after `import sparse --inplace`
    had already removed the matrix being replaced.
    """
    entries = [(1, 1, 2_000_000_000), (2, 2, 1), (1, 1, 2_000_000_000)]
    source = _mtx(tmp_path / "big.mtx", entries, (2, 2))
    store = _store(tmp_path / "s.h5ad", 2, 2)

    result = runner.invoke(app, ["import", "sparse", str(store), "X", str(source), "--inplace"])
    assert result.exit_code == 0, _out(result)
    got = ad.read_h5ad(store).X
    assert got.dtype == np.int64
    assert got[0, 0] == 4_000_000_000


def test_a_file_without_duplicates_keeps_int32_however_large_its_rows(tmp_path):
    # Sorted, no repeated coordinates: nothing will be summed.
    entries = [(1, c, 2_000_000_000) for c in (1, 2, 3)]
    source = _mtx(tmp_path / "rows.mtx", entries, (1, 3))
    _, dtypes, _, scanned = _import(tmp_path, source)
    assert scanned.canonical
    assert dtypes["data"] == np.int32


def test_the_int64_limits_are_values_not_overflow(tmp_path):
    big, small = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    source = _mtx(tmp_path / "limits.mtx", [(1, 1, big), (2, 1, small)], (2, 1))
    matrix, dtypes, _, _ = _import(tmp_path, source)
    assert dtypes["data"] == np.int64
    assert matrix.toarray().ravel().tolist() == [big, small]

    beyond = _mtx(tmp_path / "beyond.mtx", [(1, 1, big + 1)], (1, 1))
    with pytest.raises(ValueError, match="does not fit int64"):
        scan(read_header(beyond), EntryMap(n_rows=1, n_cols=1))


def test_the_formatter_covers_every_int64_and_uint64():
    """`abs(INT64_MIN)` is negative in int64, and a uint64 above INT64_MAX
    wraps when cast to it; either used to hang or corrupt the export."""
    i64, u64 = np.iinfo(np.int64), np.iinfo(np.uint64)
    signed = np.array([i64.min, i64.max, -1, 0])
    unsigned = np.array([u64.max, 2**63, 0, 7], dtype=np.uint64)
    expected = "".join(f"{a} {b}\n" for a, b in zip(signed.tolist(), unsigned.tolist()))
    assert format_int_rows([signed, unsigned]) == expected.encode()

"""Tests for `adata convert` of delimited text matrices.

Fixtures are written in each layout anndata's `read_csv` and `read_text`
tell apart (see conftest), and checked against what each was built from;
anndata itself is the reference in test_external_readers.py.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_text_matrix

ad = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _dense(m) -> np.ndarray:
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


@pytest.mark.parametrize("header", ["corner", "short", "comment", "none"])
@pytest.mark.parametrize(
    "delimiter,suffix", [(",", ".csv"), ("\t", ".tsv"), (None, ".txt")], ids=["csv", "tsv", "txt"]
)
def test_each_layout_reads_as_anndata_reads_it(tmp_path, header, delimiter, suffix):
    truth = write_text_matrix(tmp_path / f"m{suffix}", delimiter=delimiter, header=header)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "row names" in _out(result) and "12 cells x 5 genes, int32" in _out(result)
    got = _read(out)
    assert sparse.issparse(got.X) and got.X.format == "csr"
    assert got.X.dtype == np.int32, "whole values stay integers"
    assert np.array_equal(got.X.toarray(), truth.counts)
    assert list(got.obs_names) == truth.cells
    expected = truth.genes if header != "none" else [str(i) for i in range(5)]
    assert list(got.var_names) == expected
    assert got.uns["text"]["header"] == {"corner": "line", "short": "line"}.get(header, header)


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_gzipped_file_without_row_names_reads(tmp_path, suffix):
    truth = write_text_matrix(tmp_path / "m.csv.gz", gz=True, row_names=False, header="short")
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out)
    assert "no row names" in _out(result)
    got = _read(out)
    assert np.array_equal(got.X.toarray(), truth.counts)
    assert list(got.obs_names) == [str(i) for i in range(12)], "numbered, as anndata numbers them"
    assert list(got.var_names) == truth.genes


@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
def test_transpose_reads_genes_as_rows(tmp_path, dense):
    truth = write_text_matrix(tmp_path / "genes.csv", transpose=True, n_cells=40, n_genes=7)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out, "--transpose", *(["--dense"] if dense else []))
    assert "transposed" in _out(result) and "40 cells x 7 genes" in _out(result)
    got = _read(out)
    assert isinstance(got.X, np.ndarray) == dense
    assert np.array_equal(_dense(got.X), truth.counts)
    assert list(got.obs_names) == truth.cells
    assert list(got.var_names) == truth.genes
    assert got.uns["text"]["transposed"]


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_dense_keeps_x_dense(tmp_path, suffix):
    truth = write_text_matrix(tmp_path / "m.tsv", delimiter="\t", n_cells=30)
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out, "--dense")
    assert "dense" in _out(result)
    got = _read(out)
    assert isinstance(got.X, np.ndarray) and got.X.dtype == np.int32
    assert np.array_equal(got.X, truth.counts)


def test_values_pick_the_dtype(tmp_path):
    counts = np.array([[1, 2], [3, 2**40]])
    _ok(write_text_matrix(tmp_path / "big.csv", counts).path, "-o", tmp_path / "big.h5ad")
    assert _read(tmp_path / "big.h5ad").X.dtype == np.int64

    values = np.array([[0.5, 0.0], [1e-9, 123456789.123]])
    _ok(write_text_matrix(tmp_path / "f.csv", values).path, "-o", tmp_path / "f.h5ad")
    got = _read(tmp_path / "f.h5ad").X
    assert got.dtype == np.float64, "float64 keeps what the text holds"
    assert np.array_equal(got.toarray(), values)


def test_quoted_names_lose_their_quotes(tmp_path):
    truth = write_text_matrix(tmp_path / "q.csv", quote=True)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert list(got.obs_names) == truth.cells
    assert list(got.var_names) == truth.genes


def test_repeated_gene_names_warn_or_are_made_unique(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(",A,A,B\nc1,1,2,3\nc2,0,1,0\n")
    result = _ok(path, "-o", tmp_path / "w.h5ad")
    assert "1 var names repeat an earlier one" in _out(result)
    _ok(path, "-o", tmp_path / "u.h5ad", "--make-unique")
    assert list(_read(tmp_path / "u.h5ad").var_names) == ["A", "A-1", "B"]


def test_r_unnamed_columns_are_a_numeric_header(tmp_path):
    """R names unnamed columns 1, 2, ...: a numeric first line a field short
    of the next is the header, as anndata reads it."""
    path = tmp_path / "r.txt"
    path.write_text("1 2 3\n1 5 0 7\n2 0 8 0\n")
    out = tmp_path / "out.h5ad"
    _ok(path, "-o", out)
    got = _read(out)
    assert list(got.var_names) == ["1", "2", "3"]
    assert list(got.obs_names) == ["1", "2"]
    assert np.array_equal(got.X.toarray(), [[5, 0, 7], [0, 8, 0]])


def test_first_column_names_overrides_detection(tmp_path):
    path = tmp_path / "ids.csv"
    path.write_text("id,a,b\n101,1,0\n102,0,2\n")
    out = tmp_path / "named.h5ad"
    _ok(path, "-o", out, "--first-column-names")
    got = _read(out)
    assert list(got.obs_names) == ["101", "102"]
    assert np.array_equal(got.X.toarray(), [[1, 0], [0, 2]])

    path.write_text("a,b\nx,1\n")
    result = _convert(path, "-o", tmp_path / "no.h5ad", "--no-first-column-names")
    assert result.exit_code == 1
    assert "line 2, column 1: 'x' is not a number" in _out(result)


def test_delimiter_overrides_the_extension(tmp_path):
    path = tmp_path / "semi.txt"
    path.write_text(";g1;g2\nc1;1;2\nc2;3;4\n")
    result = _convert(path, "-o", tmp_path / "x.h5ad")
    assert result.exit_code == 1
    out = tmp_path / "semi.h5ad"
    _ok(path, "-o", out, "--delimiter", ";")
    assert np.array_equal(_read(out).X.toarray(), [[1, 2], [3, 4]])
    _ok(write_text_matrix(tmp_path / "t.csv", delimiter="\t").path, "-o", tmp_path / "t.h5ad",
        "--delimiter", "tab")

    result = _convert(path, "-o", tmp_path / "y.h5ad", "--delimiter", "::")
    assert result.exit_code == 1 and "give one character" in _out(result)


@pytest.mark.parametrize(
    "content,message",
    [
        (",a,b\nc1,1,2\nc2,3\n", "line 3 holds 1 values, not 2"),
        (",a,b\nc1,1,2\nc2,3,NA\n", "line 3, column 3: 'NA' is not a number"),
        (",a,b,c\nc1,1,2\n", "the header holds 4 fields, but the rows hold 2 values"),
        ("a,b\n", "holds a header and no data"),
        ("# only\n\n", "holds no data"),
        ("a b\n1 2\n", "no comma in its first line"),
        (",a,b\nc1,1,2\nc2\n", "line 3 holds a name and no values"),
    ],
)
def test_a_table_that_cannot_be_read_is_refused_before_writing(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    out = tmp_path / "out.h5ad"
    result = _convert(path, "-o", out)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


def test_small_blocks_read_the_same(tmp_path, monkeypatch):
    """Lines are read a block at a time; with tiny blocks every boundary,
    and a header longer than a block, falls inside the file."""
    from adata.conversion import text

    monkeypatch.setattr(text, "BLOCK_BYTES", 16)
    truth = write_text_matrix(tmp_path / "m.csv.gz", gz=True, n_cells=50, n_genes=9)
    out = tmp_path / "out.zarr"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert np.array_equal(got.X.toarray(), truth.counts)
    assert list(got.obs_names) == truth.cells

    truth = write_text_matrix(tmp_path / "t.tsv", delimiter="\t", transpose=True, n_cells=30)
    _ok(truth.path, "-o", tmp_path / "t.h5ad", "--transpose")
    assert np.array_equal(_read(tmp_path / "t.h5ad").X.toarray(), truth.counts)


def test_a_line_longer_than_a_block_is_read_whole():
    """A line longer than the block was rebuilt from all its bytes on every
    read: quadratic in its length."""
    import io

    from adata.formats.textio import iter_line_blocks

    data = b"x" * 1000 + b"\n" + b"y" * 10 + b"\nz"
    blocks = list(iter_line_blocks(io.BytesIO(data), block_bytes=7))
    assert b"".join(blocks) == data
    assert blocks[0] == b"x" * 1000 + b"\n"
    assert all(b.endswith(b"\n") for b in blocks[:-1])


def test_text_flags_are_refused_for_other_formats(tmp_path):
    truth = write_text_matrix(tmp_path / "m.csv")
    store = tmp_path / "s.h5ad"
    _ok(truth.path, "-o", store)
    result = _convert(store, "-o", tmp_path / "o.zarr", "--transpose")
    assert result.exit_code == 1
    assert "does not apply" in _out(result)


def test_gzip_is_found_by_content_not_name(tmp_path):
    truth = write_text_matrix(tmp_path / "m.csv")
    hidden = tmp_path / "hidden.csv"
    hidden.write_bytes(gzip.compress(truth.path.read_bytes()))
    out = tmp_path / "out.h5ad"
    _ok(hidden, "-o", out)
    assert np.array_equal(_read(out).X.toarray(), truth.counts)


def test_integers_beyond_float_precision_stay_exact(tmp_path):
    """Integers of 2**53 and above parsed exactly as int64, but the dtype
    rule took them for floats, and 2**53 + 1 was written as 2**53."""
    path = tmp_path / "big.csv"
    path.write_text(",a,b\nc1,9007199254740993,1\nc2,0,-9007199254740993\n")
    out = tmp_path / "out.h5ad"
    _ok(path, "-o", out)
    got = _read(out).X
    assert got.dtype == np.int64
    assert got.toarray().tolist() == [[9007199254740993, 1], [0, -9007199254740993]]


def test_lines_of_spaces_alone_are_blank(tmp_path):
    """A line holding only spaces or tabs was kept, then refused as ragged
    (or, before the data, crashed the layout detection)."""
    path = tmp_path / "w.txt"
    path.write_text("  \n\tg1 g2\nc1 1 2\n \t \nc2 3 4\n")
    out = tmp_path / "out.h5ad"
    _ok(path, "-o", out)
    got = _read(out)
    assert list(got.var_names) == ["g1", "g2"] and list(got.obs_names) == ["c1", "c2"]
    assert np.array_equal(got.X.toarray(), [[1, 2], [3, 4]])

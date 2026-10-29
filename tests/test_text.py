"""Tests for `adata convert` of delimited text matrices.

Fixtures are written in each layout anndata's `read_csv` and `read_text`
tell apart (see conftest), and checked against what each was built from;
anndata itself is the reference in test_external_readers.py.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Every value read as the text holds it, or the file refused (#90)


def _anndata_reads(path: Path, delimiter: Optional[str], **kw):
    reader = getattr(ad, "io", ad)
    if delimiter is None:
        return reader.read_text(path, **kw)
    return reader.read_csv(path, delimiter=delimiter, **kw)


@pytest.mark.parametrize(
    "suffix,content,where",
    [
        # pandas `to_csv` writes NaN as an empty field.
        (".tsv", "\ta\tb\tc\nc1\t1.5\t\t3.0\nc2\t4.0\t5.0\t6.0\n", "line 2, column 3"),
        (".csv", ",a,b,c\nc1,1.0,2.0,3.0\nc2,4.0,5.0,\n", "line 3, column 4"),
        (".tsv", "\ta\tb\tc\nc1\t1\t\t2\nc2\t3\t4\t5\n", "line 2, column 3"),
        # GEO tables often end every line with a tab.
        (".tsv", "gene\tS1\tS2\t\nG1\t1\t2\t\nG2\t3\t4\t\n", "line 2, column 4"),
        # A field of spaces alone is as empty, to anndata and here.
        (".tsv", "\ta\tb\tc\nc1\t1\t \t2\nc2\t3\t4\t5\n", "line 2, column 3"),
        (".csv", "a,b\n1,\n2,3\n", "line 2, column 2"),
    ],
    ids=[
        "pandas-nan-tsv", "pandas-nan-last-csv", "integers", "trailing-tabs", "spaces",
        "unnamed",
    ],
)
def test_an_empty_field_is_refused_with_its_line_and_column(tmp_path, suffix, content, where):
    """F001: numpy took two delimiters in a row for one, so an empty field
    moved every later value of its block a column left, and the block's last
    values were read from uninitialised memory. The command exited 0."""
    path = tmp_path / f"empty{suffix}"
    path.write_text(content)
    # Rows without names: an empty first field would be taken for one.
    unnamed = content.startswith("a,b")
    with pytest.raises(ValueError):
        _anndata_reads(
            path, "\t" if suffix == ".tsv" else ",",
            **({"first_column_names": False} if unnamed else {}),
        )
    out = tmp_path / "out.h5ad"
    result = _convert(path, "-o", out, *(["--no-first-column-names"] if unnamed else []))
    assert result.exit_code == 1
    assert f"{where} is empty" in _out(result)
    assert not out.exists()


def test_an_empty_last_field_of_any_block_is_refused(tmp_path, monkeypatch):
    """F001: numpy stops quietly at the end of its text, so a block whose
    last field was empty was padded with whatever memory held, wherever the
    block fell in the file."""
    from adata.conversion import text

    monkeypatch.setattr(text, "BLOCK_BYTES", 16)
    path = tmp_path / "m.csv"
    path.write_text(",a,b\nc1,1.5,2\nc2,3,\nc3,5,6\nc4,7,8\n")
    with pytest.raises(ValueError):
        _anndata_reads(path, ",")
    result = _convert(path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "line 3, column 3 is empty" in _out(result)


def test_a_field_holding_two_numbers_is_refused(tmp_path):
    """numpy splits on any whitespace when the delimiter is a tab, so a
    field '1 2' read as two values, which an empty field elsewhere in the
    block could balance."""
    path = tmp_path / "m.tsv"
    path.write_text("\ta\tb\nc1\t1 2\t3\nc2\t4\t0\n")
    with pytest.raises(ValueError):
        _anndata_reads(path, "\t")
    result = _convert(path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "line 2, column 2: '1 2' is not a number" in _out(result)


@pytest.mark.parametrize("cut", [False, True], ids=["file", "block"])
def test_a_fraction_at_the_end_of_a_block_keeps_its_fraction(tmp_path, monkeypatch, cut):
    """F050: integers are tried first, and numpy stopped at the count it was
    given, so 4.5 as the last value of a block read as 4, and the matrix as
    int32."""
    from adata.conversion import text

    head = ",a,b\nc1,1,2\nc2,3,4.5\n"
    path = tmp_path / "m.csv"
    path.write_text(head + ("c3,5,6\n" if cut else ""))
    if cut:
        monkeypatch.setattr(text, "BLOCK_BYTES", len(head))  # 4.5 ends the first block
    out = tmp_path / "out.h5ad"
    _ok(path, "-o", out)
    got = _read(out).X
    assert got.dtype == np.float64
    assert np.array_equal(got.toarray(), _anndata_reads(path, ",").X)


@pytest.mark.parametrize(
    "suffix,content",
    [
        (".csv", "1,2\n3,4\n"),
        (".csv", "a,b\n1,2\n3,4\n"),
        (".csv", ",a,b\nc1,1,2\nc2,3,4\n"),
        (".txt", "# a b\n1 2\n3 4\n"),
    ],
    ids=["numbers", "header", "corner", "comment"],
)
def test_a_byte_order_mark_is_dropped(tmp_path, suffix, content):
    """F111: a UTF-8 byte order mark stayed in the first name, and before a
    number it hid that the first value was one, so the first column was
    taken for row names."""
    plain, marked = tmp_path / f"plain{suffix}", tmp_path / f"marked{suffix}"
    plain.write_text(content)
    marked.write_text(content, encoding="utf-8-sig")
    out = tmp_path / "out.h5ad"
    _ok(marked, "-o", out)
    got = _read(out)
    # anndata keeps the mark, as `open` does; the file without it is the reference.
    expected = _anndata_reads(plain, "," if suffix == ".csv" else None)
    assert np.array_equal(got.X.toarray(), expected.X)
    assert list(got.obs_names) == list(expected.obs_names)
    assert list(got.var_names) == list(expected.var_names)
    if content == "1,2\n3,4\n":
        pd = pytest.importorskip("pandas")  # which drops the mark itself
        assert np.array_equal(got.X.toarray(), pd.read_csv(marked, header=None).to_numpy())


@pytest.mark.parametrize("delimiter", [".", "e", "-", "7", "(", ")", "_"])
def test_a_delimiter_that_numbers_hold_is_refused(tmp_path, delimiter):
    """numpy reads '1.5' as one number whatever the delimiter, and 'nan(1(2'
    as one NaN, so with '.' or '(' a line held fewer values than it has
    fields, and the rest were padded."""
    path = tmp_path / "m.txt"
    path.write_text(f"a{delimiter}b\n1{delimiter}2\n")
    result = _convert(path, "-o", tmp_path / "out.h5ad", "--delimiter", delimiter)
    assert result.exit_code == 1
    assert "a number can hold it" in _out(result)


def test_nan_with_parentheses_is_refused_as_anndata_refuses_it(tmp_path):
    """numpy reads 'nan(...)' as NaN, whatever the parentheses hold, where
    Python's float and anndata refuse it."""
    path = tmp_path / "m.csv"
    path.write_text(",a,b\nc1,nan(1),2\nc2,3,4\n")
    with pytest.raises(ValueError):
        _anndata_reads(path, ",")
    result = _convert(path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "line 2, column 2: 'nan(1)' is not a number" in _out(result)

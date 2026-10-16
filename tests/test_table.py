"""Tests for the streaming CSV table reader (`formats/table.py`).

The oracle is pandas: `pd.read_csv` and then anndata's write, which turns a
text column that repeats a value into a categorical. What anndata reads back
from our dataframe must equal that, value for value and type for type.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import h5py
import numpy as np
import pytest

from adata.formats import table
from adata.formats.table import scan_table, write_table

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
from anndata.io import read_elem  # noqa: E402

CSV = """id,count,score,flag,label,note,gappy,empty
c1,3,0.5,True,a,x one,1,
c2,4,1.5,False,b,"x, two",,
c3,5,2,True,a,x three,3,
c4,6,NaN,False,b,x four,NA,
"""


def _write(tmp_path: Path, text: str, name: str = "t.csv", gz: bool = False) -> Path:
    path = tmp_path / name
    data = text.encode()
    path.write_bytes(gzip.compress(data) if gz else data)
    return path


def _expected(path: Path, index: str = "id", rows=None):
    frame = pd.read_csv(path, compression="infer").set_index(index)
    frame.index.name = None
    frame.index = frame.index.astype(str)
    if rows is not None:
        frame = frame.iloc[rows]
    obj = ad.AnnData(obs=frame)
    obj.strings_to_categoricals()
    return obj.obs


def _read_back(tmp_path: Path, scan, **kwargs):
    store = tmp_path / "t.h5"
    with h5py.File(store, "w") as f:
        write_table(f, "obs", scan, **kwargs)
    with h5py.File(store, "r") as f:
        return read_elem(f["obs"])


def _same(got, expected) -> None:
    assert list(got.index) == list(expected.index)
    assert list(got.columns) == list(expected.columns)
    for column in expected.columns:
        a, b = got[column], expected[column]
        if isinstance(b.dtype, pd.CategoricalDtype):
            assert isinstance(a.dtype, pd.CategoricalDtype), column
            assert list(a.cat.categories) == list(b.cat.categories), column
            assert a.isna().tolist() == b.isna().tolist(), column
            assert a.astype(object).fillna("<na>").tolist() == b.astype(object).fillna("<na>").tolist(), column
        else:
            assert a.dtype == b.dtype, (column, a.dtype, b.dtype)
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy(), column)


@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gzip"])
def test_each_column_is_typed_as_pandas_reads_it(tmp_path, gz):
    path = _write(tmp_path, CSV, name="t.csv.gz" if gz else "t.csv", gz=gz)
    scan = scan_table(path, index="id")
    kinds = {c.name: scan.dtype_of(c) for c in scan.columns}
    assert kinds == {
        "count": "int64", "score": "float64", "flag": "bool", "label": "category",
        "note": "str", "gappy": "float64", "empty": "float64",
    }
    _same(_read_back(tmp_path, scan), _expected(path))


def test_rows_straddling_blocks_are_read_whole(tmp_path):
    rows = ["id,n,label"] + [f"c{i},{i},{'ab'[i % 2]}" for i in range(200)]
    path = _write(tmp_path, "\n".join(rows) + "\n")
    scan = scan_table(path, index="id", block_bytes=7)
    assert scan.n_rows == 200
    _same(_read_back(tmp_path, scan, block_bytes=13), _expected(path))


def test_a_subset_of_rows_is_written_as_anndata_subsets(tmp_path):
    """anndata drops the categories no remaining row uses."""
    path = _write(tmp_path, CSV)
    scan = scan_table(path, index="id", block_bytes=30)
    rows = np.array([1, 3])
    _same(_read_back(tmp_path, scan, rows=rows, block_bytes=30), _expected(path, rows=rows))


def test_select_records_the_rows_holding_a_value(tmp_path):
    scan = scan_table(_write(tmp_path, CSV), index="id", select=("label", "b"), block_bytes=20)
    assert scan.selected.tolist() == [1, 3]


def test_a_text_column_past_the_cap_stays_strings(tmp_path, monkeypatch):
    """Tracking every distinct value would hold the column; past the cap a
    column is written as strings even if some repeat."""
    monkeypatch.setattr(table, "CATEGORY_CAP", 3)
    rows = ["id,label"] + [f"c{i},{'abcde'[i % 5]}" for i in range(20)]
    scan = scan_table(_write(tmp_path, "\n".join(rows) + "\n"), index="id")
    assert scan.dtype_of(scan.columns[0]) == "str"


def test_an_index_column_other_than_the_first_is_taken_out(tmp_path):
    path = _write(tmp_path, CSV)
    scan = scan_table(path, index="label")
    assert [c.name for c in scan.columns] == ["id", "count", "score", "flag", "note", "gappy", "empty"]


@pytest.mark.parametrize(
    "text,message",
    [
        ("id,a\nc1,1\nc2,1,2\n", "t.csv, line 3: 3 fields where the header has 2"),
        ("id,a,a\nc1,1,2\n", "repeats the column 'a'"),
        ("", "is empty"),
        ('id,a\nc1,"open\n', "A quoted field may not span lines"),
    ],
)
def test_a_malformed_table_is_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        scan_table(_write(tmp_path, text), index="id")


def test_a_missing_index_column_is_named(tmp_path):
    with pytest.raises(ValueError, match="no 'bc_wells' column"):
        scan_table(_write(tmp_path, CSV), index="bc_wells")

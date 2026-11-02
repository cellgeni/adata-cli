"""Tests for writing a pandas DataFrame in the on-disk spec (the pickle path).

Each dtype is written into uns, read back by anndata on every backend, and
compared with the frame it came from.
"""

from __future__ import annotations

import numpy as np
import pytest

from adata.formats.pandas_frame import write_frame

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")


def _read(path):
    return ad.read_zarr(path) if str(path).endswith(".zarr") else ad.read_h5ad(path)


def _frame():
    return pd.DataFrame(
        {
            "int": np.arange(4, dtype=np.int32),
            "float": [0.5, 1.5, np.nan, 2.0],
            "bool": [True, False, True, True],
            "cat": pd.Categorical(["b", "a", "b", None], categories=["b", "a"], ordered=True),
            "numcat": pd.Categorical([3, 1, 3, 1]),
            "nullint": pd.array([1, None, 3, 4], dtype="Int64"),
            "nullbool": pd.array([True, None, False, True], dtype="boolean"),
            "nullfloat": pd.array([1.0, None, 3.0, 4.0], dtype="Float64"),
            "text": ["w", "x", "y", "z"],
            "gaps": ["w", None, "y", "z"],
            "object_ints": np.array([1, 2, 3, 4], dtype=object),
            "mixed": ["a", 1, True, 2.5],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", None, "2024-01-04"]),
        },
        index=["r0", "r1", "r2", "r3"],
    )


def test_each_dtype_reads_back_as_anndata_writes_it(new_store):
    path, opener = new_store()
    warnings = []
    with opener("a") as root:
        write_frame(root["uns"], "table", _frame(), warnings=warnings, what="t")
    got = _read(path).uns["table"]
    df = _frame()
    assert list(got.index) == list(df.index)
    assert list(got.columns) == list(df.columns), "the column order is kept"
    assert got["int"].dtype == np.int32 and got["int"].tolist() == [0, 1, 2, 3]
    assert np.array_equal(got["float"], df["float"], equal_nan=True)
    assert got["bool"].dtype == bool
    assert list(got["cat"].cat.categories) == ["b", "a"] and got["cat"].cat.ordered
    assert got["cat"].isna().tolist() == [False, False, False, True]
    assert list(got["numcat"].cat.categories) == [1, 3], "numeric categories stay numbers"
    assert str(got["nullint"].dtype) == "Int64" and got["nullint"].isna().tolist() == [False, True, False, False]
    assert str(got["nullbool"].dtype) == "boolean" and pd.isna(got["nullbool"].iloc[1])
    assert np.isnan(got["nullfloat"].iloc[1]) and got["nullfloat"].iloc[2] == 3.0
    assert got["text"].tolist() == ["w", "x", "y", "z"]
    assert pd.isna(got["gaps"].iloc[1]) and got["gaps"].iloc[2] == "y"
    assert got["object_ints"].dtype == np.int64
    assert got["mixed"].tolist() == ["a", "1", "True", "2.5"]
    assert got["when"].tolist()[0] == "2024-01-01T00:00:00"
    assert any("datetimes" in w for w in warnings)
    assert not any("mixed" in w for w in warnings), "uns tables say nothing of mixed values"


def test_obs_text_with_repeats_is_categorical(new_store):
    """In obs and var, repeated text becomes a categorical, as anndata's
    strings_to_categoricals makes it; a mixed column is worth a warning."""
    path, opener = new_store()
    warnings = []
    frame = pd.DataFrame({"kind": ["a", "b", "a"], "mixed": ["a", 1, "b"]}, index=["c1", "c2", "c3"])
    with opener("a") as root:
        del root["obs"]
        write_frame(root, "obs", frame, categorize=True, warnings=warnings, what="obs")
    got = _read(path).obs
    assert got["kind"].dtype == "category" and got["kind"].tolist() == ["a", "b", "a"]
    assert any("obs['mixed'] holds mixed" in w for w in warnings)


def test_text_with_a_missing_value_in_a_table_is_a_categorical_every_release_reads(new_store):
    """F133: written as a nullable-string-array, a model table's text column
    holding a None (random_state=None) made a converted file that anndata
    0.8 to 0.10 refused whole. A categorical with code -1 is what every
    release reads, and pandas reads the gap back as missing."""
    from adata.elements import spec
    from adata.storage import open_store

    path, opener = new_store()
    frame = pd.DataFrame({"Parameter": ["lda", None, "15"]}, index=["algorithm", "random_state", "n_iter"])
    with opener("a") as root:
        write_frame(root["uns"], "parameters", frame)
    with open_store(path, "r") as store:
        column = store.root["uns/parameters/Parameter"]
        assert spec.encoding_type(column) == spec.CATEGORICAL
        assert column["codes"][1] == -1
    got = _read(path).uns["parameters"]["Parameter"]
    assert got.tolist()[0] == "lda" and pd.isna(got.iloc[1]) and got.iloc[2] == "15"

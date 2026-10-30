"""Queries and groups read from real stores, against pandas.

The column reader hands each query the column at its stored type, so these
run on every layout anndata writes (plain, categorical, nullable), on HDF5,
Zarr v2 and Zarr v3.
"""

from __future__ import annotations

import numpy as np
import pytest

from adata.core.query import QueryError
from adata.core.select import group_indices, select_indices
from adata.storage import open_store
from tests.conftest import backend_path
from tests.test_subset_oracle import ok, read_anndata, write_anndata

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")


def _obs() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "total_counts": np.array([0, 1000, 2.5, 0, 7], dtype=np.float32),
            "pct_mt": np.array([0.1, 0.3, 0.1, 0.7, 0.1], dtype=np.float32),
            "n_genes": np.array([0, 3, 7, 0, 2], dtype=np.int32),
            "is_doublet": np.array([True, False, False, True, False]),
            "score": np.array([1.0, np.nan, 2.0, np.nan, 1.0]),
            "nullable": pd.array([1, None, 3, 1, None], dtype="Int64"),
            "flag": pd.array([True, None, False, True, False], dtype="boolean"),
            "int_cat": pd.Categorical([1, 2, 1, 3, 2]),
            "cell_type": pd.Categorical(
                ["CD4-positive, alpha-beta T cell", "B cell", "alpha-beta T cell", None, "B cell"]
            ),
        },
        index=[f"c{i}" for i in range(5)],
    )


@pytest.fixture
def stored(tmp_path, backend):
    obs = _obs()
    return write_anndata(ad.AnnData(np.ones((5, 2), np.float32), obs=obs), tmp_path, backend), obs


CASES = [
    # F120: numbers and booleans compare by value, at the column's dtype.
    ("total_counts != 0", lambda f: f.total_counts != 0),
    ("total_counts == 0", lambda f: f.total_counts == 0),
    ("total_counts == 1000", lambda f: f.total_counts == 1000),
    ("pct_mt == 0.1", lambda f: f.pct_mt == 0.1),
    ("pct_mt <= 0.1", lambda f: f.pct_mt <= 0.1),
    ("pct_mt > 0.1", lambda f: f.pct_mt > 0.1),
    ("n_genes in (0, 7)", lambda f: f.n_genes.isin([0, 7])),
    ("is_doublet == true", lambda f: f.is_doublet),
    ("is_doublet != true", lambda f: ~f.is_doublet),
    ("score != 1", lambda f: f.score != 1),
    ("score == nan", lambda f: f.score.isna()),
    ("nullable != 1", lambda f: (f.nullable != 1).fillna(True)),
    ("nullable == nan", lambda f: f.nullable.isna()),
    ("nullable >= 1", lambda f: (f.nullable >= 1).fillna(False)),
    ("flag == false", lambda f: (f.flag == False).fillna(False)),  # noqa: E712
    ("flag != false", lambda f: (f.flag != False).fillna(True)),  # noqa: E712
    ("int_cat == 2", lambda f: f.int_cat == 2),
    ("int_cat in (1, 3.0)", lambda f: f.int_cat.isin([1, 3])),
    # F119: no spaces around the operators.
    ("total_counts>0", lambda f: f.total_counts > 0),
    ("n_genes<=2 and is_doublet==false", lambda f: (f.n_genes <= 2) & ~f.is_doublet),
    # F012: a quoted label with a comma is one label.
    (
        'cell_type in ("CD4-positive, alpha-beta T cell", "B cell")',
        lambda f: f.cell_type.isin(["CD4-positive, alpha-beta T cell", "B cell"]),
    ),
    (
        'cell_type not in ("CD4-positive, alpha-beta T cell", "B cell")',
        lambda f: ~f.cell_type.isin(["CD4-positive, alpha-beta T cell", "B cell"]),
    ),
]


@pytest.mark.parametrize("expr,rule", CASES, ids=[c[0] for c in CASES])
def test_a_query_on_a_store_keeps_the_rows_pandas_keeps(stored, expr, rule):
    """F012, F119, F120, read from every column layout anndata writes."""
    path, obs = stored
    with open_store(path, "r") as store:
        got = select_indices(store.root, "obs", expr, chunk_size=2)
    assert got.tolist() == np.flatnonzero(rule(obs).to_numpy(dtype=bool)).tolist()


def test_a_value_of_the_wrong_type_is_refused_on_a_store(stored):
    """F120: a value that is not a number, compared with a number column,
    matched nothing, so `!=` with a typo kept every row with exit 0."""
    path, _ = stored
    with open_store(path, "r") as store, pytest.raises(QueryError, match="holds numbers"):
        select_indices(store.root, "obs", "total_counts != zero")


def test_total_counts_not_zero_drops_the_zeros_through_the_cli(tmp_path, stored, backend):
    """F120 as reported: `-q 'total_counts != 0'` on scanpy's float32 column
    kept every cell, zeros included, with exit 0."""
    path, obs = stored
    out = backend_path(tmp_path, backend, "out")
    ok("subset", path, "-o", out, "-q", "total_counts != 0")
    assert list(read_anndata(out).obs_names) == list(obs.index[obs.total_counts != 0])


def test_a_missing_label_and_an_empty_label_are_different_groups(tmp_path, backend):
    """F122: missing values read as '', so split put the rows with no label
    and the rows labelled '' into one output. They now group apart, as
    pandas' groupby(dropna=False) groups them."""
    obs = pd.DataFrame(
        {
            "cat": pd.Categorical([None, "", None, "x", "", "x"]),
            "text": pd.array([None, "", None, "x", "", "x"], dtype="string"),
        },
        index=[f"c{i}" for i in range(6)],
    )
    path = write_anndata(ad.AnnData(np.ones((6, 1), np.float32), obs=obs), tmp_path, backend)
    for column in ("cat", "text"):
        with open_store(path, "r") as store:
            groups, order = group_indices(store.root, "obs", column, chunk_size=4)
        series = obs[column].astype(object)
        labels = [None if pd.isna(v) else v for v in series.unique()]
        want = {
            label: np.flatnonzero(series.isna() if label is None else series == label).tolist()
            for label in labels
        }
        assert {k: v.tolist() for k, v in groups.items()} == want
        assert order == labels == [None, "", "x"]

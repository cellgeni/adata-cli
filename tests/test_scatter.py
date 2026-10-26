"""Tests for writing rows in another order through scratch (`core/scatter.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from adata.core.scatter import RowScatter
from adata.storage import create_dataset


def _feed(column, values, step):
    for lo in range(0, len(values), step):
        column.add(values[lo : lo + step])


@pytest.mark.parametrize("row_step", [3, 1 << 16])
def test_every_kept_row_lands_on_its_target_and_the_rest_keep_the_fill(new_store, row_step):
    path, opener = new_store()
    rng = np.random.default_rng(0)
    n_source, n_out = 40, 30
    # A permutation of 25 output rows, five source rows dropped, five output
    # rows left unreached.
    target = np.full(n_source, -1, dtype=np.int64)
    sources = rng.choice(n_source, size=25, replace=False)
    target[sources] = rng.choice(n_out, size=25, replace=False)
    numbers = rng.random((n_source, 2))
    codes = rng.integers(0, 4, n_source)

    with opener("a") as root:
        a = create_dataset(root["uns"], "a", shape=(n_out, 2), dtype=np.float64)
        b = create_dataset(root["uns"], "b", shape=(n_out,), dtype=np.int64)
        with RowScatter(target, n_out, root, row_step=row_step) as scatter:
            first, second = scatter.column(a, np.nan), scatter.column(b, -1)
            # Fed together, in one pass, in blocks that do not line up with
            # the output blocks.
            for lo in range(0, n_source, 7):
                first.add(numbers[lo : lo + 7])
                second.add(codes[lo : lo + 7])

    with opener("r") as root:
        got_a = np.asarray(root["uns"]["a"][...])
        got_b = np.asarray(root["uns"]["b"][...])
        assert not any(k.startswith("__scatter") for k in root.keys())

    expected_a = np.full((n_out, 2), np.nan)
    expected_b = np.full(n_out, -1)
    expected_a[target[sources]] = numbers[sources]
    expected_b[target[sources]] = codes[sources]
    np.testing.assert_array_equal(got_a, expected_a)
    np.testing.assert_array_equal(got_b, expected_b)


def test_a_column_fed_fewer_rows_than_planned_is_refused(new_store):
    path, opener = new_store()
    with opener("a") as root:
        dest = create_dataset(root["uns"], "a", shape=(3,), dtype=np.float64)
        scatter = RowScatter(np.array([2, 0, 1]), 3, root)
        _feed(scatter.column(dest, np.nan), np.arange(2.0), 1)
        with pytest.raises(ValueError, match="2 source rows were given, where 3"):
            scatter.finish()


def test_a_layout_that_keeps_no_row_writes_only_the_fill(new_store):
    """HDF5 refused the scratch's chunk shape, larger than its empty data,
    when no source row was kept: an analysis table matching no cell, or an
    empty matrix beside a table, crashed the write on h5ad alone."""
    path, opener = new_store()
    with opener("a") as root:
        dest = create_dataset(root["uns"], "a", shape=(4,), dtype=np.float64)
        with RowScatter(np.full(3, -1), 4, root) as scatter:
            _feed(scatter.column(dest), np.arange(3.0), 2)
    with opener("r") as root:
        assert np.isnan(np.asarray(root["uns"]["a"][...])).all()

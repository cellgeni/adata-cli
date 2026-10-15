"""Tests for `adata.core.csr_build`, the bucket sort behind streaming transposes.

The streaming transpose is tested end to end in test_recast.py; these pin the
pieces directly, because the MTX import builds on them with inputs a
transpose never produces: blocks whose keys cover a narrow span, and entries
in no particular order.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from adata.core.csr_build import BucketSorter, bucket_bounds, count_keys, scratch_group


@pytest.mark.parametrize("span", [10, 900_000], ids=["dense span", "sparse span"])
def test_count_keys_matches_bincount(span):
    rng = np.random.default_rng(0)
    counts = np.zeros(1_000_000, dtype=np.int64)
    expected = np.zeros_like(counts)
    for _ in range(5):
        keys = rng.integers(500, 500 + span, size=1000)
        count_keys(counts, keys)
        expected += np.bincount(keys, minlength=counts.size)
    assert np.array_equal(counts, expected)


def test_count_keys_ignores_an_empty_block():
    counts = np.zeros(4, dtype=np.int64)
    count_keys(counts, np.array([], dtype=np.int64))
    assert not counts.any()


def test_bucket_bounds_cover_every_row_and_balance_nonzeros():
    # One heavy row among many light ones: bounds follow the counts.
    counts = np.ones(1000, dtype=np.int64)
    counts[10] = 5000
    indptr = np.concatenate(([0], np.cumsum(counts)))

    bounds = bucket_bounds(indptr, chunk=1000, bucket_entries=1000)

    assert bounds[0] == 0 and bounds[-1] == 1000
    assert np.all(np.diff(bounds) > 0)
    per_bucket = np.diff(indptr[bounds])
    assert per_bucket.max() <= 5000 + 1000


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
def test_the_runs_are_the_entries_in_csr_order_whatever_order_they_came_in(
    tmp_path, order
):
    rng = np.random.default_rng(1)
    n_rows, n_cols, nnz = 300, 50, 4000
    flat = rng.choice(n_rows * n_cols, size=nnz, replace=False)
    rows, cols = flat // n_cols, flat % n_cols
    values = rng.integers(1, 100, size=nnz).astype(np.int32)

    arrival = {
        "sorted": np.lexsort((cols, rows)),
        "reversed": np.lexsort((cols, rows))[::-1],
        "shuffled": rng.permutation(nnz),
    }[order]

    counts = np.bincount(rows, minlength=n_rows)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    bounds = bucket_bounds(indptr, chunk=500, bucket_entries=500)

    with h5py.File(tmp_path / "out.h5", "w") as handle:
        with scratch_group(handle, "scratch") as scratch:
            sorter = BucketSorter(scratch, bounds, np.int32)
            for block in np.array_split(arrival, 7):
                sorter.add(rows[block], cols[block], values[block])
            runs = list(sorter.runs())

    assert len(runs) > 1, "expected several buckets"
    got_rows = np.concatenate([r for r, _, _ in runs])
    got_cols = np.concatenate([c for _, c, _ in runs])
    got_values = np.concatenate([v for _, _, v in runs])

    expected = np.lexsort((cols, rows))
    assert np.array_equal(got_rows, rows[expected])
    assert np.array_equal(got_cols, cols[expected])
    assert np.array_equal(got_values, values[expected])


def test_hdf5_scratch_is_a_separate_file_removed_afterwards(tmp_path):
    with h5py.File(tmp_path / "out.h5", "w") as handle:
        with scratch_group(handle, "scratch") as scratch:
            assert scratch.file.filename != handle.filename
            during = sorted(p.name for p in tmp_path.iterdir())
        after = sorted(p.name for p in tmp_path.iterdir())

    assert any("scratch" in name for name in during)
    assert after == ["out.h5"]

"""Building a compressed sparse matrix from entries that arrive out of order.

A streaming transpose and an MTX import have the same problem: entries come
in an order unrelated to the rows they belong to, and the output cannot be
written until each row is complete. Both solve it the same way, with memory
set by the bucket size rather than by nnz:

1. count the entries per output row (`count_keys`), which gives `indptr`;
2. scatter the entries into buckets, each covering a slice of the output rows
   and holding about as many entries as one read (`bucket_bounds`,
   `BucketSorter.add`), kept on disk in a scratch group (`scratch_group`);
3. read one bucket at a time, sort it, and append it to the output
   (`BucketSorter.runs`).
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import h5py
import numpy as np

from adata.storage import is_hdf5_group

#: Smallest bucket worth making. Below this the per-bucket overhead
#: dominates and the extra passes buy nothing.
MIN_BUCKET_ENTRIES = 1 << 12


def count_keys(counts: np.ndarray, keys: np.ndarray) -> None:
    """Add one to `counts[k]` for every `k` in `keys`.

    `np.bincount(keys, minlength=len(counts))` costs the length of `counts`
    on every call, which for a block of a few thousand entries against
    millions of rows is almost all waste -- and quadratic over a matrix read
    in many blocks. Count over the span the block actually covers instead,
    or by sorting when that span is sparse.
    """
    if keys.size == 0:
        return
    lo, hi = int(keys.min()), int(keys.max())
    span = hi - lo + 1
    if span <= 4 * keys.size:
        counts[lo : hi + 1] += np.bincount(keys - lo, minlength=span)
    else:
        unique, seen = np.unique(keys, return_counts=True)
        counts[unique] += seen


def bucket_bounds(
    indptr: np.ndarray, *, chunk: int, bucket_entries: Optional[int] = None
) -> np.ndarray:
    """Row bounds splitting the output into buckets of about equal nnz.

    One bucket per slice of the output rows, holding about as many nonzeros
    as one read. Tying this to `chunk` rather than a constant is what makes
    the peak follow the setting the caller chose: with a fixed bucket size,
    anything below it went into a single bucket and the "streaming" path
    quietly held the whole matrix. An explicit `bucket_entries` is honoured
    as given; the floor applies only to the value derived from `chunk`,
    where it stops a tiny chunk producing thousands of buckets whose
    overhead outweighs the saving.

    Split by nonzero count, not by coordinate. Equal-width bounds put nearly
    everything in one bucket whenever the matrix is skewed -- and single-cell
    matrices are: a handful of genes carry most of the counts. Measured on
    one such matrix, the largest of three equal-width buckets held 88% of
    the entries, so the bucket, not the chunk, set the peak.
    """
    n_rows = int(indptr.size - 1)
    nnz = int(indptr[-1]) if indptr.size else 0
    per_bucket = (
        max(1, int(bucket_entries))
        if bucket_entries
        else max(MIN_BUCKET_ENTRIES, int(chunk))
    )
    n_buckets = max(1, int(np.ceil(nnz / per_bucket))) if nnz else 1
    n_buckets = min(n_buckets, n_rows) or 1

    targets = np.linspace(0, nnz, n_buckets + 1)[1:-1]
    bounds = np.concatenate((
        [0],
        np.searchsorted(indptr, targets, side="left").astype(np.int64),
        [n_rows],
    ))
    return np.unique(bounds)


@contextmanager
def scratch_group(dst_parent: Any, name: str) -> Iterator[Any]:
    """A group for intermediate buckets, gone afterwards.

    HDF5 never returns the space of a deleted dataset to the file, so scratch
    written into an HDF5 output stayed in it as dead space: a streamed
    transpose came out three times the size of an in-memory one. On HDF5 the
    scratch is a separate file beside the output instead. Deleting from a
    Zarr store frees the space, so there it stays in place.
    """
    if not is_hdf5_group(dst_parent):
        group = dst_parent.create_group(name)
        try:
            yield group
        finally:
            del dst_parent[name]
        return

    # Beside the output rather than in the system temp directory, which on
    # a cluster node is often far too small for a matrix's worth of entries.
    where = Path(str(dst_parent.file.filename)).parent
    fd, path = tempfile.mkstemp(
        prefix=".adata-", suffix=".transpose-scratch.h5",
        dir=str(where) if where.is_dir() else None,
    )
    os.close(fd)
    try:
        with h5py.File(path, "w") as handle:
            yield handle.create_group(name)
    finally:
        os.unlink(path)


class BucketSorter:
    """Scatter (row, column, value) entries into row-range buckets on disk.

    `add` takes entries in any order; `runs` gives them back one bucket at a
    time, sorted by row and then column, buckets in row order. Concatenated,
    the runs are the matrix in CSR order.
    """

    def __init__(self, scratch: Any, bounds: np.ndarray, value_dtype: Any) -> None:
        # Looked up through the module rather than imported by name, so a
        # test that watches `_append` sees every bucket write.
        from adata.core import subset

        self._subset = subset
        self.bounds = np.asarray(bounds, dtype=np.int64)
        self.n_buckets = len(self.bounds) - 1
        self.buckets: List[Tuple[Any, Any, Any]] = [
            (
                subset._growable(scratch, f"major{b}", np.int64),
                subset._growable(scratch, f"minor{b}", np.int64),
                subset._growable(scratch, f"value{b}", value_dtype),
            )
            for b in range(self.n_buckets)
        ]

    def add(self, major: np.ndarray, minor: np.ndarray, values: np.ndarray) -> None:
        if major.size == 0:
            return
        which = np.clip(
            np.searchsorted(self.bounds, major, side="right") - 1,
            0,
            self.n_buckets - 1,
        )
        # One stable sort by bucket, then a slice per bucket the block
        # touches, rather than a full-block comparison for every bucket.
        order = np.argsort(which, kind="stable")
        starts = np.searchsorted(which[order], np.arange(self.n_buckets + 1))
        for b in np.flatnonzero(np.diff(starts)):
            pick = order[starts[b] : starts[b + 1]]
            majors, minors, vals = self.buckets[b]
            self._subset._append(majors, major[pick])
            self._subset._append(minors, minor[pick])
            self._subset._append(vals, values[pick])

    def runs(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for majors, minors, vals in self.buckets:
            if majors.shape[0] == 0:
                continue
            major = np.asarray(majors[...], dtype=np.int64)
            minor = np.asarray(minors[...], dtype=np.int64)
            values = np.asarray(vals[...])
            order = np.lexsort((minor, major))
            yield major[order], minor[order], values[order]

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

Input that already arrives in row order -- as Cell Ranger and STARsolo write
their MTX files -- skips the buckets (`grouped_runs`). Either way the runs
go through `DuplicateSummer` and into a `CsrAppender`.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

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


def sum_keys(sums: np.ndarray, keys: np.ndarray, weights: np.ndarray) -> None:
    """Add `weights[i]` to `sums[keys[i]]`, over the span the block covers.

    As `count_keys`, and for the same reason: cost set by the block, not by
    the length of `sums`.
    """
    if keys.size == 0:
        return
    lo, hi = int(keys.min()), int(keys.max())
    span = hi - lo + 1
    if span <= 4 * keys.size:
        sums[lo : hi + 1] += np.bincount(keys - lo, weights=weights, minlength=span)
    else:
        unique, inverse = np.unique(keys, return_inverse=True)
        sums[unique] += np.bincount(inverse, weights=weights, minlength=unique.size)


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
        # A name the store does not hold: whatever was copied in beside it
        # (a modality's own top-level keys) may already have this one.
        base, n = name, 0
        while name in dst_parent:
            n += 1
            name = f"{base}_{n}"
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


Triple = Tuple[np.ndarray, np.ndarray, np.ndarray]

#: The largest nnz an int32 `indptr` can address.
INT32_MAX = int(np.iinfo(np.int32).max)


def grouped_runs(blocks: Iterator[Triple], *, sort: bool) -> Iterator[Triple]:
    """Runs in CSR order, from blocks already grouped by row.

    The fast path, for input whose rows arrive in order -- as Cell Ranger and
    STARsolo write them. A row may straddle two blocks, so the last row of
    each block is held back and joined to the next. With `sort`, entries are
    put in column order within each row; without, they are taken to be in it
    already.
    """
    carry: Optional[Triple] = None
    for rows, cols, values in blocks:
        if carry is not None:
            rows = np.concatenate((carry[0], rows))
            cols = np.concatenate((carry[1], cols))
            values = np.concatenate((carry[2], values))
            carry = None
        if rows.size == 0:
            continue
        cut = int(np.searchsorted(rows, rows[-1], side="left"))
        carry = (rows[cut:], cols[cut:], values[cut:])
        if cut:
            yield _ordered(rows[:cut], cols[:cut], values[:cut], sort)
    if carry is not None and carry[0].size:
        yield _ordered(*carry, sort)


def _ordered(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, sort: bool) -> Triple:
    if not sort:
        return rows, cols, values
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], values[order]


class DuplicateSummer:
    """Merge repeated (row, column) entries by adding their values.

    As scipy does when it builds CSR from coordinates, and so as scanpy's
    readers give them. Each run must hold whole rows, sorted, which both
    `grouped_runs` and `BucketSorter.runs` guarantee.
    """

    def __init__(self) -> None:
        self.summed = 0

    def __call__(self, runs: Iterator[Triple]) -> Iterator[Triple]:
        for rows, cols, values in runs:
            same = (rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])
            if same.any():
                starts = np.flatnonzero(np.concatenate(([True], ~same)))
                self.summed += int(rows.size - starts.size)
                values = np.add.reduceat(values, starts)
                rows, cols = rows[starts], cols[starts]
            yield rows, cols, values


class CsrAppender:
    """A CSR matrix written row by row: entries must arrive in row order.

    `data` and `indices` grow as entries are appended; `indptr` is written
    by `finish` from the rows seen, as int32 when nnz allows, as anndata and
    scipy would write it.
    """

    def __init__(
        self,
        parent: Any,
        name: str,
        *,
        shape: Tuple[int, int],
        data_dtype: Any,
        index_dtype: Any = np.int32,
        compression: Optional[str] = None,
        templates: Optional[Tuple[Any, Any, Any]] = None,
    ) -> None:
        """`templates`, a source's `data`, `indices` and `indptr`, give the
        layout to keep (chunks, codecs) and the `indptr` dtype, as copies
        keep them; without, the defaults."""
        from adata.core.subset import _growable_like
        from adata.elements import spec
        from adata.elements.write import set_shape_attr

        self.group = parent.create_group(name)
        spec.set_encoding(self.group, spec.CSR_MATRIX)
        set_shape_attr(self.group, shape)
        self.shape = shape
        self.data_dtype = np.dtype(data_dtype)
        self.index_dtype = np.dtype(index_dtype)
        self.compression = compression
        self.templates = templates or (None, None, None)
        self.data = _growable_like(
            self.group, "data", self.data_dtype, self.templates[0], compression=compression
        )
        self.indices = _growable_like(
            self.group, "indices", self.index_dtype, self.templates[1], compression=compression
        )
        self.counts = np.zeros(shape[0], dtype=np.int64)
        self.nnz = 0

    def append(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        from adata.core import subset

        if rows.size == 0:
            return
        if self.data_dtype.kind in "iu":
            info = np.iinfo(self.data_dtype)
            if values.min() < info.min or values.max() > info.max:
                raise ValueError(
                    f"A value does not fit {self.data_dtype}: summing duplicate "
                    f"entries reached {int(values.max()):,}."
                )
        count_keys(self.counts, rows)
        subset._append(self.data, values.astype(self.data_dtype, copy=False))
        subset._append(self.indices, cols.astype(self.index_dtype, copy=False))
        self.nnz += int(rows.size)

    def finish(self) -> int:
        from adata.core.subset import _sparse_dataset

        dtype = np.int32 if self.nnz <= INT32_MAX else np.int64
        template = self.templates[2]
        if template is not None and np.dtype(template.dtype).itemsize > np.dtype(dtype).itemsize:
            dtype = np.dtype(template.dtype)  # the source's wider pointers, kept
        indptr = np.concatenate(([0], np.cumsum(self.counts))).astype(dtype)
        dataset = _sparse_dataset(
            self.group, "indptr", np.dtype(dtype), indptr.size, template,
            compression=self.compression,
        )
        dataset[:] = indptr
        return self.nnz


def write_runs(
    blocks: Iterator[Triple],
    *,
    n_rows: int,
    n_targets: int,
    destinations: Sequence[Sequence[CsrAppender]],
    scratch_parent: Any,
    grouped: bool,
    canonical: bool,
    counts: np.ndarray,
    value_dtype: Any,
    chunk: int,
    sum_duplicates: bool = True,
) -> int:
    """Append entries, given in any order, to each output's CSR matrix.

    `blocks` yield `(key, column, value)` with `key = target * n_rows + row`,
    so outputs share one sort. `counts` holds entries per key, from a first
    pass, and `grouped` / `canonical` what that pass learned of the order:
    grouped entries go straight through, others through on-disk buckets.
    `destinations[t]` are the appenders for output `t`. Returns how many
    duplicate entries were summed away. `sum_duplicates=False` keeps them,
    for entries that are a stored matrix's own, moved rather than built.
    """
    summer = DuplicateSummer()

    def consume(runs: Iterator[Triple]) -> None:
        for keys, minors, values in (summer(runs) if sum_duplicates else runs):
            targets = keys // n_rows if n_targets > 1 else None
            for t in np.unique(targets) if targets is not None else (0,):
                if targets is not None:
                    pick = targets == t
                    rows, cols, vals = keys[pick] - t * n_rows, minors[pick], values[pick]
                else:
                    rows, cols, vals = keys, minors, values
                for appender in destinations[int(t)]:
                    appender.append(rows, cols, vals)

    if grouped:
        consume(grouped_runs(blocks, sort=not canonical))
    else:
        indptr = np.concatenate(([0], np.cumsum(counts)))
        bounds = bucket_bounds(indptr, chunk=chunk)
        with scratch_group(scratch_parent, "__csr_scratch__") as scratch:
            sorter = BucketSorter(scratch, bounds, value_dtype)
            for keys, minors, values in blocks:
                sorter.add(keys, minors, values)
            consume(sorter.runs())
    return summer.summed

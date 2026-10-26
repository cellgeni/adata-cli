"""Rows read in one order, written in another, a block at a time.

A table beside a matrix (Xenium's `cells.csv.gz`, an analysis projection)
lists its rows in its own order, and AnnData wants them in the matrix's. The
table can only be read front to back, and writing each row where it belongs
is a fancy-indexed write per block, which HDF5 and Zarr do some 500 times
slower than a slice.

So the rows go through scratch: each block read is sorted by the output
block its rows belong to and appended there, then each output block is read
back from scratch in one slice, placed, and written in one slice. Memory is
one block per column; the scratch costs one extra write and read of the
values. Several columns share the layout, and are fed from one read of the
table.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, List

import numpy as np

from adata.core.csr_build import scratch_group
from adata.storage import create_dataset

#: Output rows per block.
ROW_STEP = 1 << 16


class RowScatter:
    """Where each source row goes, shared by every column fed through it.

    `target[i]` is the output row of source row `i`, or -1 to drop it; no
    two source rows may share an output row. Output rows no source row
    reaches keep the column's fill value.
    """

    def __init__(self, target: np.ndarray, n_out: int, scratch_parent: Any, row_step: int = ROW_STEP) -> None:
        self.target = np.asarray(target, dtype=np.int64)
        self.n_out = n_out
        self.row_step = row_step
        kept = np.flatnonzero(self.target >= 0)
        block_of = self.target[kept] // row_step
        self.n_blocks = -(-n_out // row_step)
        order = np.argsort(block_of, kind="stable")
        self.starts = np.concatenate(
            ([0], np.cumsum(np.bincount(block_of, minlength=self.n_blocks)))
        )
        #: The output row of each scratch row.
        self.placed = self.target[kept][order]
        self.n_kept = int(kept.size)
        self._stack = ExitStack()
        self._scratch = self._stack.enter_context(scratch_group(scratch_parent, "__scatter_scratch__"))
        self._columns: List["_Column"] = []

    def column(self, dest: Any, fill: Any = None) -> "_Column":
        """A column to fill: `dest` is created at `(n_out, ...)` already.

        `fill` is what output rows no source row reaches hold; by default NaN
        for floats, -1 for integers (a categorical's missing code), False
        and the empty string."""
        column = _Column(self, dest, fill, len(self._columns))
        self._columns.append(column)
        return column

    def finish(self) -> None:
        """Write every column out of scratch, and remove the scratch."""
        try:
            for column in self._columns:
                column.finish()
        finally:
            self._stack.close()

    def __enter__(self) -> "RowScatter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.finish()
        else:
            self._stack.close()


class _Column:
    def __init__(self, layout: RowScatter, dest: Any, fill: Any, number: int) -> None:
        self.layout = layout
        self.dest = dest
        self.fill = _default_fill(dest.dtype) if fill is None else fill
        shape = (layout.n_kept,) + tuple(dest.shape[1:])
        # HDF5 refuses a chunk larger than the data, so an empty scratch --
        # no source row kept -- has none.
        chunks = {"chunks": (min(layout.n_kept, layout.row_step),) + shape[1:]} if layout.n_kept else {}
        self.scratch = create_dataset(
            layout._scratch, f"c{number}", shape=shape, dtype=dest.dtype, **chunks
        )
        self.cursor = layout.starts[:-1].copy()
        self.done = 0

    def add(self, values: np.ndarray) -> None:
        """The next block of source rows, in source order."""
        layout = self.layout
        lo, hi = self.done, self.done + len(values)
        self.done = hi
        here = layout.target[lo:hi]
        have = np.flatnonzero(here >= 0)
        if have.size == 0:
            return
        values = np.asarray(values)[have]
        blocks = here[have] // layout.row_step
        order = np.argsort(blocks, kind="stable")
        values, blocks = values[order], blocks[order]
        edges = np.flatnonzero(np.diff(blocks)) + 1
        for run, b in zip(np.split(values, edges), blocks[np.r_[0, edges]]):
            self.scratch[self.cursor[b] : self.cursor[b] + len(run)] = run
            self.cursor[b] += len(run)

    def finish(self) -> None:
        layout = self.layout
        if self.done != layout.target.size:
            raise ValueError(
                f"{self.done:,} source rows were given, where {layout.target.size:,} "
                "were planned."
            )
        dtype = self.dest.dtype
        object_dtype = dtype.kind not in "biufc"
        for b in range(layout.n_blocks):
            lo = b * layout.row_step
            rows = min(layout.row_step, layout.n_out - lo)
            block = np.full(
                (rows,) + tuple(self.dest.shape[1:]), self.fill,
                dtype=object if object_dtype else dtype,
            )
            a, z = layout.starts[b], layout.starts[b + 1]
            if a < z:
                block[layout.placed[a:z] - lo] = self.scratch[a:z]
            self.dest[lo : lo + rows] = block


def _default_fill(dtype: np.dtype) -> Any:
    kind = np.dtype(dtype).kind
    if kind in "fc":
        return np.nan
    if kind in "iu":
        return -1 if kind == "i" else 0
    if kind == "b":
        return False
    return ""

"""What the spatial formats share: tables matched to a matrix, and images.

Space Ranger's spot positions and Xenium's cell table each list their rows
in their own order, beside a 10x matrix whose barcode order AnnData keeps.
`BarcodeJoin` matches a table to the matrix while the table is scanned, and
`write_rows` writes it in the matrix's order: straight through when the two
already agree, as Xenium's do, and through `core.scatter` when not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from adata.conversion.features import STRING_STEP
from adata.core.scatter import RowScatter
from adata.elements.read import decode_str_array
from adata.formats.table import TableScan, column_values, iter_table_blocks


@dataclass
class Join:
    """Where each row of a table goes among the matrix's barcodes."""

    n_out: int
    #: The output row of each table row, -1 for one the matrix lacks; None
    #: when the table lists the matrix's barcodes exactly, in its order.
    target: Optional[np.ndarray]
    #: Table rows the matrix has no barcode for.
    n_extra: int = 0
    #: Matrix barcodes the table has no row for.
    n_missing: int = 0
    #: Matrix barcodes the table has more than one row for.
    n_repeated: int = 0
    example_missing: str = ""

    def check(self, what: str, *, allow_missing: bool = False) -> None:
        """Refuse a join that would leave a barcode with no row, or two."""
        if self.n_repeated:
            raise ValueError(
                f"{what} has more than one row for {self.n_repeated:,} of the "
                "matrix's barcodes."
            )
        if self.n_out and self.n_missing == self.n_out:
            raise ValueError(
                f"{what} has a row for none of the matrix's {self.n_out:,} barcodes; "
                "it must come from the same run as the matrix."
            )
        if self.n_missing and not allow_missing:
            raise ValueError(
                f"{self.n_missing:,} of the matrix's {self.n_out:,} barcodes have "
                f"no row in {what} (the first is {self.example_missing!r}); "
                "it must come from the same run as the matrix."
            )


class BarcodeJoin:
    """Matches a table's index to a matrix's barcodes, a block at a time.

    Given to `scan_table` as `on_index`, so the match costs no pass of its
    own. While each block equals the matrix's next barcodes it only compares
    them; at the first difference it switches to looking each value up in a
    sorted copy of the barcodes, a numpy array rather than a dict, whose
    per-string overhead is several times the barcode itself.
    """

    def __init__(self, barcodes: Any) -> None:
        self._barcodes = barcodes
        self.n_out = int(barcodes.shape[0])
        self._in_order = True
        self._done = 0
        self._targets: List[np.ndarray] = []
        self._ranked: Optional[np.ndarray] = None
        self._order: Optional[np.ndarray] = None

    def _slice(self, lo: int, hi: int) -> np.ndarray:
        return np.asarray(decode_str_array(np.asarray(self._barcodes[lo:hi])), dtype=str)

    def __call__(self, values: Sequence[str]) -> None:
        values = np.asarray(values, dtype=str)
        lo, hi = self._done, self._done + values.size
        if self._in_order:
            if hi <= self.n_out and np.array_equal(self._slice(lo, hi), values):
                self._done = hi
                return
            self._leave_order()
        self._targets.append(self._lookup(values))
        self._done = hi

    def _leave_order(self) -> None:
        # Every row so far matched the barcode at its own position.
        self._in_order = False
        self._targets = [np.arange(self._done, dtype=np.int64)]
        wanted = np.concatenate(
            [self._slice(lo, min(lo + STRING_STEP, self.n_out)) for lo in range(0, self.n_out, STRING_STEP)]
            or [np.zeros(0, dtype=str)]
        )
        self._order = np.argsort(wanted, kind="stable")
        self._ranked = wanted[self._order]

    def _lookup(self, values: np.ndarray) -> np.ndarray:
        out = np.full(values.size, -1, dtype=np.int64)
        if self._ranked.size == 0:
            return out
        at = np.minimum(np.searchsorted(self._ranked, values), self._ranked.size - 1)
        hit = self._ranked[at] == values
        out[hit] = self._order[at[hit]]
        return out

    def finish(self) -> Join:
        if self._in_order and self._done == self.n_out:
            return Join(n_out=self.n_out, target=None)
        if self._in_order:
            self._leave_order()
        target = np.concatenate(self._targets) if self._targets else np.zeros(0, dtype=np.int64)
        counts = np.bincount(target[target >= 0], minlength=self.n_out)
        missing = np.flatnonzero(counts == 0)
        return Join(
            n_out=self.n_out,
            target=target,
            n_extra=int((target < 0).sum()),
            n_missing=int(missing.size),
            n_repeated=int((counts > 1).sum()),
            example_missing=str(self._slice(missing[0], missing[0] + 1)[0]) if missing.size else "",
        )


@dataclass
class Sink:
    """One array to fill from a table: `values` turns a block's columns
    into the block's values; `fill` is what rows the table lacks get."""

    dest: Any
    values: Callable[[Dict[str, List[str]]], np.ndarray]
    fill: Any = None


def write_rows(scan: TableScan, join: Join, sinks: Sequence[Sink], scratch_parent: Any) -> None:
    """Fill each sink from one read of the table, in the matrix's order."""
    if join.target is None:
        lo = 0
        for names, block in iter_table_blocks(scan):
            hi = lo + len(names)
            for sink in sinks:
                sink.dest[lo:hi] = sink.values(block)
            lo = hi
        return
    with RowScatter(join.target, join.n_out, scratch_parent) as scatter:
        columns = [scatter.column(sink.dest, sink.fill) for sink in sinks]
        for _, block in iter_table_blocks(scan):
            for column, sink in zip(columns, sinks):
                column.add(sink.values(block))


def require_numeric(scan: TableScan, names: Sequence[str]) -> None:
    """Refuse, at plan time, a coordinate or projection column that is not
    all numbers (missing values are allowed, and become NaN)."""
    kinds = {c.name: scan.dtype_of(c) for c in scan.columns}
    absent = [n for n in names if n not in kinds]
    if absent:
        raise ValueError(f"{scan.path.name} has no {', '.join(absent)} column.")
    bad = [n for n in names if kinds[n] not in ("int64", "float64")]
    if bad:
        raise ValueError(
            f"{scan.path.name}: {', '.join(bad)} {'holds' if len(bad) == 1 else 'hold'} "
            "values that are not numbers."
        )


def numbers(scan: TableScan, names: Sequence[str], block: Dict[str, List[str]], dtype: Any = np.float64) -> np.ndarray:
    """A block's values of numeric columns, side by side, NaN where missing."""
    by_name = {c.name: c for c in scan.columns}
    return np.stack(
        [column_values(scan, by_name[n], block[n], {}).astype(dtype) for n in names], axis=1
    )


def read_image(path: Path) -> np.ndarray:
    """An image's pixels as the file holds them: uint8 for an 8-bit PNG, as
    squidpy reads them, H x W or H x W x C."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image)

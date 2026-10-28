"""What the spatial formats share: tables matched to a matrix, and images.

Space Ranger's spot positions and Xenium's cell table each list their rows
in their own order, beside a 10x matrix whose barcode order AnnData keeps.
`BarcodeJoin` matches a table to the matrix while the table is scanned, and
`write_rows` writes it in the matrix's order: straight through when the two
already agree, as Xenium's do, and through `core.scatter` when not.

Xenium and Space Ranger (spots, bins and segmented cells alike) also share
the layout of their own `analysis/` (graph-based clusters, UMAP, PCA), read
by `scan_analysis`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from adata.conversion.features import STRING_STEP
from adata.core.scatter import RowScatter
from adata.elements import spec
from adata.elements.read import decode_str_array
from adata.elements.write import create_categorical
from adata.formats.table import TableScan, column_values, drop_unused, iter_table_blocks, scan_table
from adata.storage import create_dataset
from adata.util.names import natural_sorted


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
    #: A digest of which barcodes are missing, to tell tables that lack the
    #: same ones from tables that lack as many.
    missing_digest: str = ""

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
    per-string overhead is several times the barcode itself. Barcodes are
    held as bytes, a quarter of numpy's unicode: Visium HD's 2 µm bins number
    in the millions.
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
        return _as_bytes(np.asarray(self._barcodes[lo:hi]))

    def __call__(self, values: Sequence[Any]) -> None:
        values = _as_bytes(values)
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
            or [np.zeros(0, dtype="S1")]
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
        self._targets = []  # the blocks, as big again as the target
        counts = np.bincount(target[target >= 0], minlength=self.n_out)
        missing = np.flatnonzero(counts == 0)
        return Join(
            n_out=self.n_out,
            target=target,
            n_extra=int((target < 0).sum()),
            n_missing=int(missing.size),
            n_repeated=int((counts > 1).sum()),
            example_missing=(
                self._slice(missing[0], missing[0] + 1)[0].decode("utf-8", "replace")
                if missing.size else ""
            ),
            missing_digest=hashlib.blake2b(missing.tobytes(), digest_size=16).hexdigest(),
        )


def _as_bytes(values: Any) -> np.ndarray:
    """Strings, bytes or numbers as a fixed-width bytes array, for comparing."""
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return array
    if array.dtype.kind == "O" and array.size and isinstance(array.flat[0], bytes):
        return array.astype("S")
    return np.char.encode(np.asarray(decode_str_array(array), dtype=str), "utf-8")


@dataclass
class Sink:
    """One array to fill from a table: `values` turns a block's columns
    into the block's values; `fill` is what rows the table lacks get."""

    dest: Any
    values: Callable[[Dict[str, Any]], np.ndarray]
    fill: Any = None


def table_blocks(scan: TableScan) -> Iterator[Tuple[int, Dict[str, List[str]]]]:
    """A CSV table's rows for `write_rows`: (rows in the block, its columns)."""
    for names, block in iter_table_blocks(scan):
        yield len(names), block


def write_rows(
    blocks: Iterable[Tuple[int, Dict[str, Any]]], join: Join, sinks: Sequence[Sink], scratch_parent: Any
) -> None:
    """Fill each sink from one read of a table's `blocks`, in the matrix's order."""
    if join.target is None:
        lo = 0
        for n, block in blocks:
            for sink in sinks:
                sink.dest[lo : lo + n] = sink.values(block)
            lo += n
        return
    with RowScatter(join.target, join.n_out, scratch_parent) as scatter:
        columns = [scatter.column(sink.dest, sink.fill) for sink in sinks]
        for _, block in blocks:
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


#: Image rows copied out of Pillow per write.
IMAGE_ROWS = 512


def write_image(parent: Any, name: str, path: Path) -> None:
    """An image's pixels as the file holds them, uint8 for an 8-bit PNG as
    squidpy reads them, H x W or H x W x C, written a strip of rows at a time.

    A PNG decodes only whole, but copying it out in strips spares a second
    copy of it: Visium HD's hires image is some 100 MB of pixels.
    """
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        width, height = image.size
        out = None
        for top in range(0, height, IMAGE_ROWS):
            strip = np.asarray(image.crop((0, top, width, min(top + IMAGE_ROWS, height))))
            if out is None:
                out = create_dataset(
                    parent, name, shape=(height,) + strip.shape[1:], dtype=strip.dtype,
                    chunks=(min(IMAGE_ROWS, height),) + strip.shape[1:],
                )
                spec.set_encoding(out, spec.ARRAY)
            out[top : top + strip.shape[0]] = strip


#: The analysis tables read, under `analysis/`: the graph-based clusters,
#: and each projection with the obsm key it goes to. Each is where it may
#: be, the first found read: today's names, then Space Ranger 1's, which
#: had no `gene_expression_` prefix.
ANALYSIS_CLUSTERS = (
    Path("clustering/gene_expression_graphclust/clusters.csv"),
    Path("clustering/graphclust/clusters.csv"),
)
ANALYSIS_PROJECTIONS = {
    "X_umap": (
        Path("umap/gene_expression_2_components/projection.csv"),
        Path("umap/2_components/projection.csv"),
    ),
    "X_pca": (
        Path("pca/gene_expression_10_components/projection.csv"),
        Path("pca/10_components/projection.csv"),
    ),
}


def analysis_paths() -> List[Path]:
    """Every path under `analysis/` a table is looked for at."""
    return [p for paths in (ANALYSIS_CLUSTERS, *ANALYSIS_PROJECTIONS.values()) for p in paths]


@dataclass
class Table:
    """A table beside the matrix, scanned and matched to its barcodes."""

    scan: TableScan
    join: Join


def scan_joined(path: Path, index: str, barcodes: Any, *, distinct: bool = True) -> Table:
    joiner = BarcodeJoin(barcodes)
    scan = scan_table(path, index=index, on_index=joiner, distinct=distinct)
    return Table(scan, joiner.finish())


@dataclass
class Analysis:
    """The clusters and projections found in an `analysis/` folder."""

    clusters: Optional[Table] = None
    projections: Dict[str, Table] = field(default_factory=dict)

    def found(self) -> List[str]:
        return (["clusters"] if self.clusters is not None else []) + list(self.projections)


def scan_analysis(folder: Path, barcodes: Any) -> Analysis:
    """Scan whichever analysis tables `folder` holds, matched to `barcodes`."""
    analysis = Analysis()
    if not folder.is_dir():
        return analysis
    path = _first_file(folder, ANALYSIS_CLUSTERS)
    if path is not None:
        analysis.clusters = scan_joined(path, "Barcode", barcodes)
    for key, paths in ANALYSIS_PROJECTIONS.items():
        path = _first_file(folder, paths)
        if path is not None:
            # Numbers only, so no column's distinct values are kept.
            analysis.projections[key] = scan_joined(path, "Barcode", barcodes, distinct=False)
    return analysis


def _first_file(folder: Path, paths: Sequence[Path]) -> Optional[Path]:
    return next((folder / p for p in paths if (folder / p).is_file()), None)


def check_analysis(analysis: Analysis, warnings: List[str], rows: str = "cells") -> None:
    """Refuse analysis tables that cannot be written; warn of `rows` (cells,
    spots, bins) they lack, which are left empty."""
    if analysis.clusters is not None and "Cluster" not in analysis.clusters.scan.header:
        raise ValueError(f"{analysis.clusters.scan.path.name} in analysis has no Cluster column.")
    for table in analysis.projections.values():
        require_numeric(table.scan, [c.name for c in table.scan.columns])
    # Space Ranger leaves bins with no counts out of every table alike, so
    # tables lacking the same rows share one warning rather than one each.
    lacking: Dict[Tuple[int, str], List[str]] = {}
    for label, table in [("clusters", analysis.clusters), *analysis.projections.items()]:
        if table is None:
            continue
        table.join.check(f"analysis {label}", allow_missing=True)
        if table.join.n_missing:
            lacking.setdefault((table.join.n_missing, table.join.missing_digest), []).append(label)
    for (n, _), labels in lacking.items():
        named = labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} or {labels[-1]}"
        if n == 1:
            warnings.append(f"1 {rows[:-1]} has no {named} in analysis/; it is left empty.")
        else:
            warnings.append(f"{n:,} {rows} have no {named} in analysis/; they are left empty.")


def write_analysis(root: Any, obs: Any, analysis: Analysis) -> List[str]:
    """The clusters into obs, the projections into obsm; the obs columns written."""
    columns = []
    if analysis.clusters is not None:
        _write_clusters(root, obs, analysis.clusters)
        columns.append("graphclust")
    for key, table in analysis.projections.items():
        _write_projection(root, key, table)
    return columns


def used_codes(codes: Any, n_categories: int, n: int) -> np.ndarray:
    """Which of a categorical's categories its codes use."""
    used = np.zeros(n_categories, dtype=bool)
    for lo in range(0, n, STRING_STEP):
        block = np.asarray(codes[lo : lo + STRING_STEP], dtype=np.int64)
        used[block[block >= 0]] = True
    return used


def _write_clusters(root: Any, obs: Any, table: Table) -> None:
    """The graph-based clusters, `obs["graphclust"]`, a categorical."""
    scan, join = table.scan, table.join
    column = next(c for c in scan.columns if c.name == "Cluster")
    categories = natural_sorted(column.seen or set())
    lookup = {c: i for i, c in enumerate(categories)}
    codes = create_categorical(obs, "graphclust", categories, join.n_out)
    write_rows(
        table_blocks(scan), join,
        [Sink(codes, lambda block: np.array([lookup.get(v, -1) for v in block["Cluster"]], dtype=np.int64), -1)],
        root,
    )
    if join.n_extra:
        used = used_codes(codes, len(categories), join.n_out)
        if not used.all():
            drop_unused(obs, "graphclust", used, join.n_out, block_rows=STRING_STEP)


def _write_projection(root: Any, key: str, table: Table) -> None:
    """An analysis projection (UMAP, PCA) into obsm, NaN for a cell it lacks."""
    scan, join = table.scan, table.join
    names = [c.name for c in scan.columns]
    out = create_dataset(root["obsm"], key, shape=(join.n_out, len(names)), dtype=np.float64)
    spec.set_encoding(out, spec.ARRAY)
    write_rows(
        table_blocks(scan), join,
        [Sink(out, lambda block: numbers(scan, names, block), np.nan)],
        root,
    )

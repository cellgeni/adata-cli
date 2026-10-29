"""A dense 2-D HDF5 dataset written as an AnnData matrix: a loom's matrices
(genes x cells) and a plain HDF5 file's dataset (either way round).

Memory is set by `slab_bytes`, not by the dataset or its chunks (beyond
one chunk, which HDF5 decompresses whole). How the dataset is read depends
on how many cells one of its chunks spans:

* when the budget holds a chunk's width of cells, a slab of cells at a
  time, each a whole number of chunks, so every chunk is decompressed once;
* when it holds less, but no fewer than `1 / MAX_REREADS` of it (chunks
  that split both axes, as h5py chooses them), slabs as wide as the budget
  holds, so a chunk is decompressed once for each slab that crosses it;
* otherwise -- chunks spanning the cells, one gene to a chunk -- a tile of
  whole chunks at a time, twice: once to count each cell's non-zeros, then
  to sort the entries into cells through on-disk buckets
  (`core.csr_build`), as a transposed text matrix is.

The matrix becomes CSR, or stays dense; one read of each slab or tile fills
every place that holds it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from adata.core.csr_build import CsrAppender, write_runs
from adata.elements import spec
from adata.storage import create_dataset, is_zarr_group, with_compression

#: Slabs of cells a chunk may be cut into, each decompressing it again,
#: before tiles are cheaper: they decompress it twice, and write and read
#: its entries once more through the buckets.
MAX_REREADS = 4

#: Bytes an entry takes while its bucket is sorted: its cell and gene as
#: int64, its value, the sort's order and the sorted copies.
ENTRY_BYTES = 64


def slab_size(n_genes: int, itemsize: int, chunk: int, budget: int) -> int:
    """Cells per slab: as many whole chunks as `budget` bytes hold, one at least."""
    per_cell = max(1, n_genes * itemsize)
    fit = max(1, budget // per_cell)
    return max(chunk, fit // chunk * chunk)


def _whole(chunk: int, fit: int, n: int) -> int:
    """As many whole chunks as `fit` holds, one at least, `n` at most."""
    return max(1, min(n, max(chunk, fit // chunk * chunk)))


def write_dense_matrix(
    ds: h5py.Dataset,
    places: Sequence[Tuple[Any, str]],
    *,
    cells_axis: int,
    dense: bool,
    compression: Optional[str],
    zarr_format: Optional[int],
    slab_bytes: int,
) -> None:
    """`ds` as a cells x genes matrix at each `(parent, name)` in `places`.

    `cells_axis` is 0 when the dataset's rows are cells, 1 when they are
    genes (a loom, or a transposed HDF5 matrix).
    """
    n_cells, n_genes = ds.shape[cells_axis], ds.shape[1 - cells_axis]
    dtype = np.dtype(ds.dtype.str)  # without h5py's metadata
    shape = (n_cells, n_genes)
    # Cells the budget holds, every gene of each; a contiguous dataset reads
    # any slab as cheaply, so its chunk is a cell.
    fit = max(1, slab_bytes // max(1, n_genes * dtype.itemsize))
    chunk = ds.chunks[cells_axis] if ds.chunks else 1
    step = min(max(n_cells, 1), fit // chunk * chunk if chunk <= fit else fit)

    if -(-chunk // fit) > MAX_REREADS:
        _write_sorted(
            ds, places, cells_axis=cells_axis, dense=dense, compression=compression,
            zarr_format=zarr_format, slab_bytes=slab_bytes, rows=fit,
        )
        return

    def slabs() -> Iterator[Tuple[int, np.ndarray]]:
        for lo in range(0, n_cells, step):
            if cells_axis == 0:
                yield lo, np.asarray(ds[lo : lo + step])
            else:
                yield lo, np.asarray(ds[:, lo : lo + step]).T

    if dense:
        targets = _dense_targets(places, shape, dtype, step, compression, zarr_format)
        for lo, block in slabs():
            for target in targets:
                target[lo : lo + block.shape[0]] = block
        return

    appenders: List[CsrAppender] = [
        CsrAppender(parent, name, shape=shape, data_dtype=dtype, compression=compression)
        for parent, name in places
    ]
    for lo, block in slabs():
        rows, cols = np.nonzero(block)
        values = block[rows, cols]
        for appender in appenders:
            appender.append(rows + lo, cols, values)
    for appender in appenders:
        appender.finish()


def _dense_targets(
    places: Sequence[Tuple[Any, str]],
    shape: Tuple[int, int],
    dtype: np.dtype,
    rows: int,
    compression: Optional[str],
    zarr_format: Optional[int],
) -> List[Any]:
    """A dense array at each place, chunked `rows` cells at a time."""
    targets = []
    for parent, name in places:
        kw: Dict[str, Any] = {"chunks": (max(1, min(shape[0], rows)), max(shape[1], 1))}
        if compression is not None:
            kw = with_compression(
                kw, compression, shape=shape, dtype=dtype,
                backend="zarr" if is_zarr_group(parent) else "hdf5",
                zarr_format=zarr_format,
            )
        target = create_dataset(parent, name, shape=shape, dtype=dtype, **kw)
        spec.set_encoding(target, spec.ARRAY)
        targets.append(target)
    return targets


def _tiles(
    ds: h5py.Dataset, cells_axis: int, slab_bytes: int
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """`(first cell, first gene, cells x genes block)`: the dataset a tile of
    whole chunks at a time, as many as the budget holds, one at least."""
    n_cells, n_genes = ds.shape[cells_axis], ds.shape[1 - cells_axis]
    cell_chunk, gene_chunk = ds.chunks[cells_axis], ds.chunks[1 - cells_axis]
    itemsize = ds.dtype.itemsize
    cells = _whole(cell_chunk, slab_bytes // max(1, gene_chunk * itemsize), n_cells)
    genes = _whole(gene_chunk, slab_bytes // max(1, cells * itemsize), n_genes)
    for g in range(0, n_genes, genes):
        for c in range(0, n_cells, cells):
            if cells_axis == 0:
                yield c, g, np.asarray(ds[c : c + cells, g : g + genes])
            else:
                yield c, g, np.asarray(ds[g : g + genes, c : c + cells]).T


def _write_sorted(
    ds: h5py.Dataset,
    places: Sequence[Tuple[Any, str]],
    *,
    cells_axis: int,
    dense: bool,
    compression: Optional[str],
    zarr_format: Optional[int],
    slab_bytes: int,
    rows: int,
) -> None:
    """The dataset a tile at a time, its entries sorted into cells on disk."""
    n_cells, n_genes = ds.shape[cells_axis], ds.shape[1 - cells_axis]
    dtype = np.dtype(ds.dtype.str)
    shape = (n_cells, n_genes)
    counts = np.zeros(n_cells, dtype=np.int64)
    for c, _, block in _tiles(ds, cells_axis, slab_bytes):
        counts[c : c + block.shape[0]] += np.count_nonzero(block, axis=1)

    def entries() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for c, g, block in _tiles(ds, cells_axis, slab_bytes):
            cells, genes = np.nonzero(block)
            yield cells + c, genes + g, block[cells, genes]

    sinks: List[Any]
    if dense:
        targets = _dense_targets(places, shape, dtype, rows, compression, zarr_format)
        sinks = [_DenseRows(targets, shape, dtype, rows)]
    else:
        sinks = [
            CsrAppender(parent, name, shape=shape, data_dtype=dtype, compression=compression)
            for parent, name in places
        ]
    write_runs(
        entries(), n_rows=n_cells, n_targets=1, destinations=[sinks],
        scratch_parent=places[0][0], grouped=False, canonical=True, counts=counts,
        value_dtype=dtype, chunk=max(1, slab_bytes // ENTRY_BYTES), sum_duplicates=False,
    )
    for sink in sinks:
        sink.finish()


class _DenseRows:
    """Entries arriving in row order, written to dense targets a block of
    `rows` at a time, as `CsrAppender` writes them to CSR."""

    def __init__(self, targets: List[Any], shape: Tuple[int, int], dtype: np.dtype, rows: int):
        self.targets, self.n_rows, self.rows = targets, shape[0], max(1, min(rows, shape[0]))
        self.block = np.zeros((self.rows, shape[1]), dtype=dtype)
        self.lo = 0

    def append(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        while rows.size:
            cut = int(np.searchsorted(rows, self.lo + self.rows))
            self.block[rows[:cut] - self.lo, cols[:cut]] = values[:cut]
            if cut == rows.size:
                return
            self._flush()
            rows, cols, values = rows[cut:], cols[cut:], values[cut:]

    def _flush(self) -> None:
        hi = min(self.lo + self.rows, self.n_rows)
        for target in self.targets:
            target[self.lo : hi] = self.block[: hi - self.lo]
        self.block[:] = 0
        self.lo = hi

    def finish(self) -> None:
        while self.lo < self.n_rows:
            self._flush()

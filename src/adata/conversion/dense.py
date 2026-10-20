"""A dense 2-D HDF5 dataset written as an AnnData matrix, a slab of cells
at a time: a loom's matrices (genes x cells) and a plain HDF5 file's
dataset (either way round).

Each slab is a whole number of the dataset's chunks along the cell axis,
so every chunk is decompressed once, and memory is set by the slab rather
than by the dataset. The matrix becomes CSR, or stays dense; one read of
each slab fills every place that holds it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from adata.core.csr_build import CsrAppender
from adata.elements import spec
from adata.storage import create_dataset, is_zarr_group, with_compression

#: Cells per chunk assumed for a contiguous dataset, as loompy chunks.
DEFAULT_CHUNK = 64


def slab_size(n_genes: int, itemsize: int, chunk: int, budget: int) -> int:
    """Cells per slab: as many whole chunks as `budget` bytes hold, one at least."""
    per_cell = max(1, n_genes * itemsize)
    fit = max(1, budget // per_cell)
    return max(chunk, fit // chunk * chunk)


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
    chunk = ds.chunks[cells_axis] if ds.chunks else DEFAULT_CHUNK
    slab = slab_size(n_genes, ds.dtype.itemsize, chunk, slab_bytes)
    dtype = np.dtype(ds.dtype.str)  # without h5py's metadata
    shape = (n_cells, n_genes)

    def slabs():
        for lo in range(0, n_cells, slab):
            if cells_axis == 0:
                yield lo, np.asarray(ds[lo : lo + slab])
            else:
                yield lo, np.asarray(ds[:, lo : lo + slab]).T

    if dense:
        targets = []
        for parent, name in places:
            kw: Dict[str, Any] = {"chunks": (min(max(n_cells, 1), slab), max(n_genes, 1))}
            if compression is not None:
                kw = with_compression(
                    kw, compression, shape=shape, dtype=dtype,
                    backend="zarr" if is_zarr_group(parent) else "hdf5",
                    zarr_format=zarr_format,
                )
            target = create_dataset(parent, name, shape=shape, dtype=dtype, **kw)
            spec.set_encoding(target, spec.ARRAY)
            targets.append(target)
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

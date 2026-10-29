"""Tests for `conversion/dense.py`: a dense HDF5 dataset written as X.

The loom and HDF5 importers both read their matrix through it. Here it is
driven directly, on each output backend, against datasets chunked every way
that changes how it reads them; anndata reads back what it wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pytest

from adata.conversion.dense import write_dense_matrix
from tests.conftest import backend_path, backend_zarr_format, make_skeleton, open_new

ad = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")

N_CELLS, N_GENES = 300, 6
ITEMSIZE = 4


def _source(path: Path, chunks, *, cells_axis: int) -> Tuple[h5py.File, np.ndarray]:
    counts = np.random.default_rng(0).poisson(0.6, (N_CELLS, N_GENES)).astype(np.float32)
    handle = h5py.File(path, "w")
    stored = counts if cells_axis == 0 else counts.T
    handle.create_dataset("m", data=stored, chunks=chunks, compression="gzip" if chunks else None)
    return handle, counts


def _largest_read(monkeypatch) -> List[int]:
    """Bytes of the largest single read of the source dataset, as `[n]`."""
    largest = [0]
    read = h5py.Dataset.__getitem__

    def counted(self, key):
        out = read(self, key)
        if self.name == "/m":
            largest[0] = max(largest[0], np.asarray(out).nbytes)
        return out

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counted)
    return largest


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _dense(m) -> np.ndarray:
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


#: (chunks of a genes x cells dataset, slab budget in bytes, how it is read).
CASES = {
    # A chunk's width of cells fits: slabs of whole chunks.
    "whole-chunks": ((N_GENES, 16), 16 * N_GENES * ITEMSIZE * 3),
    # Not quite: slabs narrower than a chunk, each chunk read a few times.
    "split-chunks": ((N_GENES, 40), 16 * N_GENES * ITEMSIZE),
    # One gene per chunk, every cell in it: tiles of whole chunks, sorted.
    "gene-per-chunk": ((1, N_CELLS), 2 * N_CELLS * ITEMSIZE),
    # Far wider than the budget, but narrower than the cells: a chunk a tile.
    "chunk-per-tile": ((1, 100), 120 * ITEMSIZE),
    # No chunks: any slab is as cheap.
    "contiguous": (None, 10 * N_GENES * ITEMSIZE),
}


@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
@pytest.mark.parametrize("case", list(CASES))
def test_any_chunking_is_read_within_the_budget(tmp_path, backend, monkeypatch, case, dense):
    """F026: a slab was a whole number of chunks along the cells, so a
    dataset chunked one gene per chunk was read whole in one slab, whatever
    the budget, and `--dense` made an output chunk as large."""
    chunks, budget = CASES[case]
    handle, counts = _source(tmp_path / "in.h5", chunks, cells_axis=1)
    largest = _largest_read(monkeypatch)
    path = backend_path(tmp_path, backend, "out")
    with handle, open_new(path, backend) as root:
        make_skeleton(root, [f"c{i}" for i in range(N_CELLS)], [f"g{j}" for j in range(N_GENES)])
        write_dense_matrix(
            handle["m"], [(root, "X"), (root["layers"], "again")], cells_axis=1,
            dense=dense, compression="gzip", zarr_format=backend_zarr_format(backend),
            slab_bytes=budget,
        )
    assert 0 < largest[0] <= budget
    got = _read(path)
    assert sparse.issparse(got.X) != dense
    assert np.array_equal(_dense(got.X), counts)
    assert np.array_equal(_dense(got.layers["again"]), counts), "one read fills every place"
    if dense and backend == "h5ad":
        with h5py.File(path, "r") as f:
            assert f["X"].chunks[0] * N_GENES * ITEMSIZE <= budget
            assert f["X"].compression == "gzip"


@pytest.mark.parametrize("case", ["whole-chunks", "gene-per-chunk"])
def test_cells_as_rows_are_read_within_the_budget(tmp_path, monkeypatch, case):
    """The same, with the dataset's rows the cells: chunked every cell to a
    chunk, it was one slab too."""
    chunks, budget = {
        "whole-chunks": ((16, N_GENES), 3 * 16 * N_GENES * ITEMSIZE),
        "gene-per-chunk": ((N_CELLS, 1), 2 * N_CELLS * ITEMSIZE),
    }[case]
    handle, counts = _source(tmp_path / "in.h5", chunks, cells_axis=0)
    largest = _largest_read(monkeypatch)
    path = tmp_path / "out.h5ad"
    with handle, open_new(path, "h5ad") as root:
        make_skeleton(root, [f"c{i}" for i in range(N_CELLS)], [f"g{j}" for j in range(N_GENES)])
        write_dense_matrix(
            handle["m"], [(root, "X")], cells_axis=0, dense=False, compression=None,
            zarr_format=None, slab_bytes=budget,
        )
    assert 0 < largest[0] <= budget
    assert np.array_equal(_read(path).X.toarray(), counts)


def test_an_empty_dataset_writes_an_empty_matrix(tmp_path, backend):
    with h5py.File(tmp_path / "in.h5", "w") as handle:
        ds = handle.create_dataset(
            "m", shape=(N_GENES, 0), maxshape=(N_GENES, None), dtype=np.float32, chunks=(1, 1)
        )
        path = backend_path(tmp_path, backend, "out")
        with open_new(path, backend) as root:
            make_skeleton(root, [], [f"g{j}" for j in range(N_GENES)])
            write_dense_matrix(
                ds, [(root, "X")], cells_axis=1, dense=False, compression=None,
                zarr_format=backend_zarr_format(backend), slab_bytes=1,
            )
    assert _read(path).X.shape == (0, N_GENES)

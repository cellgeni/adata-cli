"""Stores for the memory and I/O guards, built the way users' stores are.

`test_performance.py`'s `_store` builds one shape: a CSR X eight columns wide,
grown in n_obs. A path whose memory scales with the width, with the CSC
layout, or with a dense X passes every guard built on it, which is how the
recast and subset paths that `--chunk` does not bound reached a release (F091,
F146). These builders scale every axis and layout instead, in each of the
formats the tool reads:

* ``h5ad``: as anndata writes it -- a contiguous dense X, auto-chunked sparse
  arrays;
* ``zarr2``: anndata's Zarr v2;
* ``zarr3``: anndata 0.13's default Zarr v3, sharded (`auto_shard_zarr_v3`):
  inner chunks of a few hundred bytes inside shards of the whole array. Any
  read or write that does not line up with those shows here and nowhere else.

anndata writes every one, so a fixture is what a user would hand the tool,
not what the tool would write itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

#: Matrix layouts a store can hold.
LAYOUTS = ("csr", "csc", "dense")

#: Store formats, as anndata writes each.
FORMATS = ("h5ad", "zarr2", "zarr3")


def suffix(fmt: str) -> str:
    return ".h5ad" if fmt == "h5ad" else ".zarr"


def counts_matrix(
    n_obs: int,
    n_var: int,
    layout: str = "csr",
    *,
    density: float = 0.1,
    dtype: str = "float32",
    seed: int = 0,
):
    """Whole-number counts, so any float cast round-trips and needs no --force."""
    from scipy import sparse

    rng = np.random.default_rng(seed)
    matrix = sparse.random(
        n_obs, n_var, density=density, format="csr", dtype="float64", random_state=rng
    )
    matrix.data = np.ceil(matrix.data * 100)
    matrix = matrix.astype(dtype)
    if layout == "dense":
        return matrix.toarray()
    return matrix.tocsc() if layout == "csc" else matrix


def skewed_matrix(n_obs: int, n_var: int = 64, *, dense_genes: Tuple[int, int] = (8, 16),
                  layout: str = "csc", seed: int = 0):
    """Counts where a run of adjacent genes is in every cell, the rest sparse.

    Ribosomal genes sort together and are expressed nearly everywhere, so a
    real matrix has exactly this shape: a few columns hold most entries.
    """
    from scipy import sparse

    matrix = counts_matrix(n_obs, n_var, "csr", density=0.02, seed=seed).tolil()
    lo, hi = dense_genes
    matrix[:, lo:hi] = np.arange(1, n_obs * (hi - lo) + 1, dtype="float32").reshape(
        n_obs, hi - lo
    ) % 7 + 1
    matrix = sparse.csr_matrix(matrix)
    return matrix.tocsc() if layout == "csc" else matrix


def write_anndata(obj, path: Path, fmt: str) -> Path:
    """Write `obj` as anndata would in `fmt`."""
    import anndata as ad

    if fmt == "h5ad":
        obj.write_h5ad(path)
    elif fmt == "zarr2":
        with ad.settings.override(zarr_write_format=2):
            obj.write_zarr(path)
    elif fmt == "zarr3":
        with ad.settings.override(zarr_write_format=3, auto_shard_zarr_v3=True):
            obj.write_zarr(path)
    else:  # pragma: no cover - a typo in a test
        raise ValueError(f"unknown format {fmt!r}")
    return path


def matrix_store(
    directory: Path,
    *,
    n_obs: int,
    n_var: int,
    layout: str = "csr",
    fmt: str = "h5ad",
    density: float = 0.1,
    dtype: str = "float32",
    matrix=None,
    dense_chunks: Optional[Tuple[int, int]] = None,
    tag: str = "m",
) -> Path:
    """A store holding X of the given shape and layout, and a numeric obs column.

    `dense_chunks` rechunks a dense h5ad X, as a store written by another
    tool might be (anndata writes it contiguous).
    """
    import anndata as ad
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    if matrix is None:
        matrix = counts_matrix(n_obs, n_var, layout, density=density, dtype=dtype)
    n_obs, n_var = matrix.shape
    obj = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"col": np.arange(n_obs, dtype="int32")},
            index=[f"c{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_var)]),
    )
    path = directory / f"{tag}-{layout}-{n_obs}x{n_var}{suffix(fmt)}"
    write_anndata(obj, path, fmt)
    if dense_chunks is not None:
        rechunk_dense(path, dense_chunks)
    return path


def rechunk_dense(path: Path, chunks: Tuple[int, int], key: str = "X") -> None:
    """Rewrite a dense h5ad matrix with `chunks`, keeping its attributes."""
    import h5py

    with h5py.File(path, "a") as handle:
        values = handle[key][...]
        attrs = dict(handle[key].attrs)
        del handle[key]
        dataset = handle.create_dataset(key, data=values, chunks=chunks)
        for name, value in attrs.items():
            dataset.attrs[name] = value

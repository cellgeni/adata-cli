"""Synthetic inputs for the benchmark, generated deterministically.

Two choices here are not arbitrary.

**Poisson counts, not uniform floats.** Random float32 is incompressible, so
a uniform-random fixture is three times the size on disk for the same shape
and the `ci` tier stops fitting in a runner's 14 GB. Counts are also what the
tool actually sees.

**lzf, not gzip.** gzip on 10^8 values is single-threaded and costs minutes
per fixture on four vCPUs. The compression setting is pinned on *both* sides
of every comparison -- see `cases.py` -- because adata-cli forwards the
source's compression to its output while `write_h5ad` defaults to none, which
would otherwise make adata-cli's output look smaller for reasons that have
nothing to do with the tool.

**Wide variants for `recast`.** The main fixture is already float32 with
int32 indices, so narrowing it is a rewrite that changes nothing. The wide
variants hold the same matrix as float64 values and int64 indices and indptr,
which is what issue #13 was about and what a real narrowing starts from. One
stores X uncompressed, as `write_h5ad` does by default and as most float64
stores therefore are; the other keeps lzf. The difference matters: integer
counts in float64 are mostly zero bytes, which a compressor removes anyway,
so narrowing an lzf X saves a few percent on disk where an uncompressed one
halves. Both are measured, so neither number stands alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

#: Nonzeros per row block while generating, to keep the generator itself
#: from needing the memory the benchmark is about to measure.
_GEN_ROWS = 4096


@dataclass(frozen=True)
class Tier:
    """One size of input, and what it is for."""

    name: str
    n_obs: int
    n_var: int
    density: float
    note: str


TIERS: Dict[str, Tier] = {
    "smoke": Tier("smoke", 1_000, 2_000, 0.05, "fast enough to debug the harness"),
    "ci": Tier("ci", 50_000, 20_000, 0.05, "fits a 16 GB runner with the baseline"),
    "large": Tier(
        "large",
        500_000,
        20_000,
        0.05,
        "in-memory baseline is expected to hit the address-space ceiling",
    ),
}

#: The REQ-71798 shape: few rows, many variables, var columns worth merging.
#: Small enough to run in every tier, and the case the hang was reported on.
VAR_HEAVY = Tier("var-heavy", 2_000, 36_601, 0.05, "the shape that hung in 0.5.1")


def _write(path: Path, tier: Tier, seed: int, *, n_var_columns: int = 2) -> Path:
    """Write one CSR h5ad of `tier`'s shape, built a row block at a time."""
    import anndata as ad
    import h5py
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    blocks = []
    for start in range(0, tier.n_obs, _GEN_ROWS):
        rows = min(_GEN_ROWS, tier.n_obs - start)
        block = sp.random(
            rows,
            tier.n_var,
            density=tier.density,
            format="csr",
            dtype="float32",
            random_state=rng,
            data_rvs=lambda k: rng.poisson(3.0, k).astype("float32") + 1,
        )
        blocks.append(block)
    matrix = sp.vstack(blocks, format="csr")

    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(
                [f"s{i % 8}" for i in range(tier.n_obs)]
            ),
            "n_counts": np.asarray(matrix.sum(axis=1)).ravel(),
            "n_genes": matrix.getnnz(axis=1).astype("int32"),
            # A predicate on n_genes would select a different fraction at
            # every tier, since genes-per-cell tracks n_var. This one keeps
            # exactly half the rows whatever the shape, so the subset case
            # measures the same work across tiers.
            "quality": (np.arange(tier.n_obs) % 100).astype("int32"),
            "barcode": [f"bc{seed}-{i}" for i in range(tier.n_obs)],
        },
        index=[f"c{seed}-{i}" for i in range(tier.n_obs)],
    )
    var = pd.DataFrame(
        {
            "gene_symbol": [f"SYM{i}" for i in range(tier.n_var)],
            "feature_type": ["Gene Expression"] * tier.n_var,
            **{
                f"extra{c}": [f"e{c}-{i}" for i in range(tier.n_var)]
                for c in range(max(0, n_var_columns - 2))
            },
        },
        index=[f"ENSG{i:011d}" for i in range(tier.n_var)],
    )

    obj = ad.AnnData(X=matrix, obs=obs, var=var)
    # A dense element, so the `export array` / `import array` cases have
    # something of realistic width to move. 50 columns is a typical PCA.
    obj.obsm["X_pca"] = rng.standard_normal(
        (tier.n_obs, 50), dtype="float32"
    )
    obj.write_h5ad(path, compression="lzf")
    del obj, matrix, blocks

    with h5py.File(path, "r") as handle:  # cheap sanity check
        assert handle["X"].attrs["encoding-type"] == "csr_matrix"
    return path


def build(directory: Path, tier: Tier, *, count: int = 2) -> List[Path]:
    """Build (or reuse) `count` inputs of this tier's shape.

    Reused if already present: generation is the slowest part of the job and
    the content is fully determined by (tier, seed).
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in range(count):
        path = directory / f"{tier.name}-{seed}.h5ad"
        if not path.exists():
            _write(path, tier, seed)
        paths.append(path)
    return paths


#: The wide variants, by name: how each stores X.
WIDE_COMPRESSION: Dict[str, Optional[str]] = {"uncompressed": None, "lzf": "lzf"}


def _widen(src: Path, dst: Path, compression: Optional[str]) -> Path:
    """Copy `src` with X stored as float64 values and int64 indices/indptr.

    Streamed a block at a time with h5py, so the large tier does not need
    the whole matrix in memory twice over. Everything but X is copied as is.
    """
    import h5py

    block = 1 << 24
    with h5py.File(src, "r") as source, h5py.File(dst, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for key in source:
            if key != "X":
                source.copy(source[key], target, name=key)
        x_in = source["X"]
        x_out = target.create_group("X")
        for key, value in x_in.attrs.items():
            x_out.attrs[key] = value
        for name, dtype in (
            ("data", "float64"), ("indices", "int64"), ("indptr", "int64")
        ):
            column = x_in[name]
            out = x_out.create_dataset(
                name,
                shape=column.shape,
                dtype=dtype,
                chunks=column.chunks or True,
                compression=compression,
            )
            for start in range(0, column.shape[0], block):
                stop = min(start + block, column.shape[0])
                out[start:stop] = column[start:stop].astype(dtype)
    return dst


def build_wide(directory: Path, tier: Tier, variant: str) -> List[Path]:
    """The first input of `tier`, widened. Built once and reused, like `build`."""
    (source,) = build(directory, tier, count=1)
    path = directory / f"{tier.name}-0-wide-{variant}.h5ad"
    if not path.exists():
        _widen(source, path, WIDE_COMPRESSION[variant])
    return [path]

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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

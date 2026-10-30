"""Write a reference AnnData store, run inside a pinned anndata environment.

Invoked by tests/reference_stores.py through `uv run --with anndata==<v>`, so
this file must stay compatible with every anndata release under test (0.7.8
onward) and must not import anything from `adata`. Features that only exist in
later versions are left out for the releases without them.

Usage: write_reference_store.py <out_dir> <h5ad|zarr>
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

N_OBS, N_VAR = 6, 4


def _release() -> tuple:
    from importlib.metadata import version

    return tuple(int(part) for part in version("anndata").split(".")[:2])


def build():
    """An object exercising the encodings this tool has to understand."""
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["A", "B", "A", "C", "B", "A"],
                categories=["A", "B", "C"],
                ordered=True,
            ),
            "unordered": pd.Categorical(["x", "y", "x", "y", "x", "y"]),
            "n_counts": np.arange(N_OBS, dtype="int32"),
            "score": np.linspace(0, 1, N_OBS).astype("float64"),
            "free_text": ["a", "bb", "ccc", "d", "ee", "f"],
        },
        index=[f"cell_{i}" for i in range(N_OBS)],
    )

    # Nullable dtypes are written from 0.8; 0.7 fails on them part way.
    if _release() >= (0, 8):
        obs["nullable_int"] = pd.array([1, 2, None, 4, 5, None], dtype="Int32")
        obs["nullable_bool"] = pd.array(
            [True, False, None, True, False, None], dtype="boolean"
        )

    var = pd.DataFrame(
        {
            "gene_ids": [f"ENSG{i}" for i in range(N_VAR)],
            "highly_variable": [True, False, True, False],
            # Categorical in raw/var too, as scanpy leaves feature_types:
            # anndata 0.7 wrote it as codes with an object reference.
            "feature_types": pd.Categorical(
                ["Gene Expression"] * 3 + ["Antibody Capture"]
            ),
        },
        index=[f"gene_{i}" for i in range(N_VAR)],
    )

    X = sparse.csr_matrix(
        np.random.default_rng(0).poisson(1.0, (N_OBS, N_VAR)).astype("float32")
    )

    obj = ad.AnnData(X=X, obs=obs, var=var)
    obj.layers["counts"] = X.copy()
    obj.layers["dense"] = np.asarray(X.todense())
    obj.obsm["X_pca"] = np.zeros((N_OBS, 3), dtype="float32")
    obj.varm["PCs"] = np.zeros((N_VAR, 3), dtype="float32")
    obj.obsp["connectivities"] = sparse.csr_matrix(np.eye(N_OBS, dtype="float32"))
    obj.varp["corr"] = sparse.csr_matrix(np.eye(N_VAR, dtype="float32"))

    obj.uns["a_string"] = "hello"
    obj.uns["an_int"] = 42
    obj.uns["a_float"] = 3.25
    obj.uns["a_bool"] = True
    obj.uns["a_list"] = ["x", "y", "z"]
    obj.uns["numbers"] = np.arange(5)
    obj.uns["nested"] = {"deep": {"value": 1.5}}
    # scanpy's marker genes: record arrays, one field per group.
    obj.uns["rank_genes_groups"] = {
        "names": np.rec.fromarrays(
            [np.array(["gene_0", "gene_2"]), np.array(["gene_1", "gene_3"])],
            names=["A", "B"],
        ),
        "scores": np.rec.fromarrays(
            [np.array([2.5, 1.0], dtype="float32"), np.array([3.0, 0.5], dtype="float32")],
            names=["A", "B"],
        ),
    }

    obj.raw = obj
    return obj


def main(out_dir: str, fmt: str) -> None:
    import os
    from importlib.metadata import version

    os.makedirs(out_dir, exist_ok=True)
    obj = build()
    target = f"{out_dir}/reference.{fmt}"
    if fmt == "zarr":
        obj.write_zarr(target)
    else:
        obj.write_h5ad(target)

    try:
        zarr_version = version("zarr")
    except Exception:
        zarr_version = "absent"
    print(f"WROTE {target} anndata={version('anndata')} zarr={zarr_version}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

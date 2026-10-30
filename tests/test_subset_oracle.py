"""subset and split against anndata's own subsetting, element by element.

F092: the values of layers, obsp, varp and raw after subset or split were
checked only against the tool's own output (split against subset, which
shares its writer), so a writer that took the wrong rows of a layer but kept
its shape passed every test. Here every element of every output is checked
against `adata[obs, var]` as anndata computes it, read back by anndata, on
HDF5, Zarr v2 and Zarr v3.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import backend_path, backend_zarr_format

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def invoke(*args: Any):
    return runner.invoke(app, [str(a) for a in args])


def ok(*args: Any):
    result = invoke(*args)
    assert result.exit_code == 0, _out(result)
    return result


def write_anndata(obj: Any, tmp_path: Path, backend: str, stem: str = "src") -> Path:
    """`obj` written by anndata itself, on the backend asked for."""
    path = backend_path(tmp_path, backend, stem)
    if backend == "h5ad":
        obj.write_h5ad(path)
    else:
        with ad.settings.override(zarr_write_format=backend_zarr_format(backend)):
            obj.write_zarr(path)
    return path


def read_anndata(path: Path) -> Any:
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def rich_anndata(layout: str = "csr", seed: int = 0) -> Any:
    """One of every element kind subset and split narrow."""
    rng = np.random.default_rng(seed)
    n_obs, n_var = 30, 9
    dense = rng.poisson(0.8, (n_obs, n_var)).astype(np.float32)
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix}.get(layout, np.asarray)(dense)
    obs_names = [f"c{i}" for i in range(n_obs)]
    var_names = [f"v{j}" for j in range(n_var)]
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                [f"g{i % 3}" for i in range(n_obs)], categories=["g0", "g1", "g2", "never"]
            ),
            "score": rng.random(n_obs).astype(np.float32),
            "n": np.arange(n_obs, dtype=np.int64),
        },
        index=obs_names,
    )
    var = pd.DataFrame(
        {"kind": pd.Categorical([f"k{j % 2}" for j in range(n_var)]),
         "mean": rng.random(n_var)},
        index=var_names,
    )
    obj = ad.AnnData(X, obs=obs, var=var)

    def random_sparse(shape, cls):
        values = rng.random(shape).astype(np.float32)
        return cls(values * (rng.random(shape) < 0.3))

    obj.layers["csr"] = sparse.csr_matrix(dense * 2)
    obj.layers["csc"] = sparse.csc_matrix(dense * 3)
    obj.layers["dense"] = dense * 4
    obj.obsm["X_pca"] = rng.normal(size=(n_obs, 4)).astype(np.float32)
    obj.obsm["cube"] = rng.normal(size=(n_obs, 2, 3))
    obj.obsm["sparse"] = random_sparse((n_obs, 5), sparse.csr_matrix)
    obj.obsm["frame"] = pd.DataFrame(
        {
            "label": pd.Categorical(
                [f"p{i % 5}" for i in range(n_obs)], categories=["p0", "p1", "p2", "p3", "p4", "zz"]
            ),
            "x": rng.random(n_obs),
        },
        index=obs_names,
    )
    obj.varm["loadings"] = rng.normal(size=(n_var, 3)).astype(np.float32)
    obj.varm["cube"] = rng.normal(size=(n_var, 2, 2))
    obj.obsp["conn"] = random_sparse((n_obs, n_obs), sparse.csr_matrix)
    obj.obsp["dist"] = rng.random((n_obs, n_obs))
    # anndata allows trailing axes on a pairwise entry, as on obsm.
    obj.obsp["cube"] = rng.random((n_obs, n_obs, 2)).astype(np.float32)
    obj.varp["corr"] = rng.random((n_var, n_var)).astype(np.float32)
    obj.varp["sparse"] = random_sparse((n_var, n_var), sparse.csc_matrix)
    obj.varp["cube"] = rng.integers(0, 9, (n_var, n_var, 3))

    # raw holds more genes than X, in another order, so matching it by
    # position rather than by name would pick the wrong columns.
    raw_names = [f"extra{k}" for k in range(4)] + var_names[::-1]
    raw = ad.AnnData(
        random_sparse((n_obs, len(raw_names)), sparse.csr_matrix),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame({"length": np.arange(len(raw_names), dtype=np.int64)}, index=raw_names),
    )
    raw.varm["raw_loadings"] = rng.normal(size=(len(raw_names), 2))
    obj.raw = raw
    return obj


def _same(got: Any, want: Any, what: str) -> None:
    if isinstance(want, pd.DataFrame):
        pd.testing.assert_frame_equal(got, want, obj=what)
        return
    assert sparse.issparse(got) == sparse.issparse(want), what
    if sparse.issparse(want):
        assert got.format == want.format, what
        got, want = got.toarray(), want.toarray()
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype, what
    np.testing.assert_array_equal(got, want, err_msg=what)


def assert_matches_anndata(
    got: Any,
    source: Any,
    obs: Optional[Sequence[int]] = None,
    var: Optional[Sequence[int]] = None,
) -> None:
    """`got` is `source[obs, var]`, element by element, as anndata slices it.

    raw is the one documented difference: anndata keeps raw's var whole,
    while a var selection narrows it to the var names kept (raw usually
    holds more genes than X, so its own var axis is matched by name). The
    expectation is anndata's slicing of raw by those names.
    """
    obs = slice(None) if obs is None else np.asarray(obs)
    var = slice(None) if var is None else np.asarray(var)
    want = source[obs, var].copy()

    assert list(got.obs_names) == list(want.obs_names)
    assert list(got.var_names) == list(want.var_names)
    _same(got.obs, want.obs, "obs")
    _same(got.var, want.var, "var")
    _same(got.X, want.X, "X")
    for mapping in ("layers", "obsm", "varm", "obsp", "varp"):
        got_m, want_m = getattr(got, mapping), getattr(want, mapping)
        # anndata 0.13 lists X among the layers, as None.
        keys = sorted(k for k in want_m.keys() if k is not None)
        assert sorted(k for k in got_m.keys() if k is not None) == keys, mapping
        for key in keys:
            _same(got_m[key], want_m[key], f"{mapping}/{key}")

    raw = source.raw.to_adata()[obs]
    if not isinstance(var, slice):
        raw = raw[:, raw.var_names.isin(want.var_names)]
    raw = raw.copy()
    assert list(got.raw.var_names) == list(raw.var_names)
    _same(got.raw.X, raw.X, "raw/X")
    _same(got.raw.var, raw.var, "raw/var")
    _same(got.raw.varm["raw_loadings"], raw.varm["raw_loadings"], "raw/varm")


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_a_subset_on_both_axes_matches_anndatas_slicing_in_every_element(
    tmp_path, backend, layout
):
    source = rich_anndata(layout)
    src = write_anndata(source, tmp_path, backend)
    obs = [0, 2, 3, 7, 11, 12, 20, 29]
    var = [1, 4, 5, 8]
    names = tmp_path / "cells.txt"
    names.write_text("".join(f"c{i}\n" for i in obs))
    genes = tmp_path / "genes.txt"
    genes.write_text("".join(f"v{j}\n" for j in var))
    out = backend_path(tmp_path, backend, "out")

    ok("subset", src, "-o", out, "--obs", names, "--var", genes)
    assert_matches_anndata(read_anndata(out), source, obs, var)


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_a_query_subset_matches_anndatas_slicing_in_every_element(tmp_path, backend, layout):
    source = rich_anndata(layout, seed=1)
    src = write_anndata(source, tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")

    ok("subset", src, "-o", out, "-q", "score > 0.5", "--var-query", "kind == k1")
    obs = np.flatnonzero(source.obs["score"].to_numpy() > 0.5)
    var = np.flatnonzero(source.var["kind"].to_numpy() == "k1")
    assert_matches_anndata(read_anndata(out), source, obs, var)


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("axis,column", [("obs", "group"), ("var", "kind")])
def test_every_split_output_matches_anndatas_slicing_in_every_element(
    tmp_path, backend, layout, axis, column
):
    source = rich_anndata(layout, seed=2)
    src = write_anndata(source, tmp_path, backend)
    parts = tmp_path / "parts"

    ok("split", src, "--by", column, "--axis", axis, "-o", parts, "--no-manifest")
    frame = source.obs if axis == "obs" else source.var
    labels = frame[column].astype(str).to_numpy()
    suffix = src.suffix
    assert sorted(p.name for p in parts.iterdir()) == sorted(
        f"{label}{suffix}" for label in set(labels)
    )
    for label in set(labels):
        rows = np.flatnonzero(labels == label)
        got = read_anndata(parts / f"{label}{suffix}")
        if axis == "obs":
            assert_matches_anndata(got, source, obs=rows)
        else:
            assert_matches_anndata(got, source, var=rows)

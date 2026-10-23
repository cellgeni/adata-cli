"""`subset` and `split` narrow a categorical's categories as anndata does.

anndata's subsetting keeps only the categories the kept rows use
(`remove_unused_categories`), and narrows `uns["<col>_colors"]` to match.
Copies kept every category, so a subset of a 10x Multiome matrix's peaks
held all 134,905 intervals' labels for 98,319 peaks. anndata, the
reference, must read back what its own subsetting gives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

runner = CliRunner()


def _store(path: Path):
    rng = np.random.default_rng(0)
    n = 12
    obj = ad.AnnData(
        rng.random((n, 4)).astype(np.float32),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(["T"] * 4 + ["B"] * 4 + ["NK"] * 4, categories=["B", "NK", "T"]),
            "stage": pd.Categorical(["x"] * 6 + ["y"] * 6, categories=["x", "y", "z"], ordered=True),
        }, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame({"kind": pd.Categorical(["g", "g", "p", "p"])}, index=[f"f{j}" for j in range(4)]),
    )
    obj.uns["cell_type_colors"] = np.array(["#b", "#n", "#t"], dtype=object)
    obj.uns["stage_colors"] = np.array(["#x", "#y", "#z"], dtype=object)
    obj.write_h5ad(path)
    return obj


def _ok(*args):
    result = runner.invoke(app, [*map(str, args)])
    assert result.exit_code == 0, result.stdout
    return result


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_subset_keeps_the_categories_its_rows_use_as_anndata_does(tmp_path, suffix):
    obj = _store(tmp_path / "in.h5ad")
    keep = tmp_path / "keep.txt"
    keep.write_text("\n".join(f"c{i}" for i in range(4, 12)) + "\n")  # no T, no z
    out = tmp_path / f"sub{suffix}"
    _ok("subset", tmp_path / "in.h5ad", "--obs", keep, "-o", out)
    got = ad.read_zarr(out) if suffix == ".zarr" else ad.read_h5ad(out)
    ref = obj[4:].copy()
    pd.testing.assert_frame_equal(got.obs, ref.obs)
    for key in ("cell_type_colors", "stage_colors"):
        assert list(got.uns[key]) == list(ref.uns[key])
    assert got.obs["stage"].cat.ordered


def test_split_narrows_each_outputs_categories(tmp_path):
    obj = _store(tmp_path / "in.h5ad")
    _ok("split", tmp_path / "in.h5ad", "--by", "cell_type", "-o", tmp_path / "parts")
    for path in sorted((tmp_path / "parts").glob("*.h5ad")):
        got = ad.read_h5ad(path)
        (label,) = set(got.obs["cell_type"])
        ref = obj[obj.obs["cell_type"] == label].copy()
        pd.testing.assert_frame_equal(got.obs, ref.obs)
        assert list(got.uns["cell_type_colors"]) == list(ref.uns["cell_type_colors"])


def test_a_var_subset_narrows_var_and_leaves_obs_whole(tmp_path):
    obj = _store(tmp_path / "in.h5ad")
    keep = tmp_path / "keep.txt"
    keep.write_text("f0\nf1\n")
    _ok("subset", tmp_path / "in.h5ad", "--var", keep, "-o", tmp_path / "sub.h5ad")
    got = ad.read_h5ad(tmp_path / "sub.h5ad")
    ref = obj[:, :2].copy()
    pd.testing.assert_frame_equal(got.var, ref.var)
    # anndata drops a category no row uses on the axis kept whole too.
    pd.testing.assert_frame_equal(got.obs, ref.obs)
    assert list(got.uns["stage_colors"]) == list(ref.uns["stage_colors"]) == ["#x", "#y"]


def _subset_both(tmp_path, obj, obs_keep, var_keep=None):
    obj.write_h5ad(tmp_path / "in.h5ad")
    (tmp_path / "obs.txt").write_text("\n".join(obs_keep) + "\n")
    args = ["subset", tmp_path / "in.h5ad", "--obs", tmp_path / "obs.txt", "-o", tmp_path / "sub.h5ad"]
    if var_keep is not None:
        (tmp_path / "var.txt").write_text("\n".join(var_keep) + "\n")
        args += ["--var", tmp_path / "var.txt"]
    _ok(*args)
    return ad.read_h5ad(tmp_path / "sub.h5ad")


def test_colours_follow_anndatas_rule(tmp_path):
    """RGBA rows were left whole, so they no longer lined up with the
    categories; colours of the wrong length were kept, and a single colour
    left 0-d, where anndata drops the one and wraps the other."""
    obj = _store(tmp_path / "x.h5ad")
    obj.uns["cell_type_colors"] = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=float)
    obj.uns["stage_colors"] = np.array(["#a", "#b"], dtype=object)  # 2 for 3 categories
    obj.obs["one"] = pd.Categorical(["m"] * obj.n_obs)
    obj.uns["one_colors"] = "#m"
    keep = [f"c{i}" for i in range(4, 12)]
    got = _subset_both(tmp_path, obj, keep)
    ref = obj[keep].copy()
    assert np.array_equal(got.uns["cell_type_colors"], ref.uns["cell_type_colors"])
    assert "stage_colors" not in ref.uns and "stage_colors" not in got.uns
    assert list(got.uns["one_colors"]) == list(ref.uns["one_colors"]) == ["#m"]


def test_a_column_name_in_obs_and_var_has_its_colours_narrowed_obs_then_var(tmp_path):
    rng = np.random.default_rng(0)
    obj = ad.AnnData(
        rng.random((4, 3)).astype(np.float32),
        obs=pd.DataFrame({"k": pd.Categorical(["p", "q", "r", "r"])}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame({"k": pd.Categorical(["s", "t", "u"])}, index=[f"f{j}" for j in range(3)]),
    )
    obj.uns["k_colors"] = np.array(["#1", "#2", "#3"], dtype=object)
    got = _subset_both(tmp_path, obj, ["c0", "c1"], ["f0"])
    ref = obj[["c0", "c1"], ["f0"]].copy()
    # obs narrows them to p, q; then var's three categories match two
    # colours no longer, so they go.
    assert "k_colors" not in ref.uns and "k_colors" not in got.uns


def test_converting_a_whole_anndata_to_mudata_keeps_its_categories(tmp_path):
    """--mod writes the whole AnnData, no subset: an unused category stayed
    on main, and should stay."""
    import h5py

    rng = np.random.default_rng(0)
    obj = ad.AnnData(
        rng.random((3, 2)).astype(np.float32),
        var=pd.DataFrame({"kind": pd.Categorical(["g", "p"], categories=["g", "p", "z"])}, index=["f0", "f1"]),
    )
    obj.write_h5ad(tmp_path / "in.h5ad")
    _ok("convert", tmp_path / "in.h5ad", "-o", tmp_path / "out.h5mu", "--mod", "rna")
    with h5py.File(tmp_path / "out.h5mu") as f:
        assert f["mod/rna/var/kind/categories"].shape[0] == 3

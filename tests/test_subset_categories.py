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

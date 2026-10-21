"""`concat` on what a SCope or pycisTopic import leaves in obsm and varm:
dataframes in obsm (`RegulonsAUC`), and varm (`Regulons`, `topic_region`).

anndata's own `concat` is the reference: our output, read back, must equal
`anndata.concat(..., join=..., merge=...)` on the same inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from adata.core.concat import concat_on_disk

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

QUIET = Console(quiet=True, record=True)


def _store(tmp_path: Path, tag: str, n: int, genes, *, seed: int, regulons=("ATF3_(+)", "SOX2_(+)"),
           shift: float = 0.0):
    rng = np.random.default_rng(seed)
    cells = [f"{tag}{i}" for i in range(n)]
    obj = ad.AnnData(
        X=rng.poisson(1, (n, len(genes))).astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=list(genes)),
    )
    obj.obsm["RegulonsAUC"] = pd.DataFrame(rng.random((n, len(regulons))), index=cells, columns=list(regulons))
    obj.obsm["X_umap"] = rng.normal(size=(n, 2))
    membership = (np.arange(len(genes))[:, None] + np.arange(len(regulons))) % 2
    obj.varm["Regulons"] = pd.DataFrame(membership, index=list(genes), columns=list(regulons))
    obj.varm["topic_region"] = np.arange(len(genes) * 3, dtype=np.float64).reshape(len(genes), 3) + shift
    path = tmp_path / f"{tag}.h5ad"
    obj.write_h5ad(path)
    return path


def _concat(tmp_path, paths, **kw):
    out = tmp_path / f"out{len(list(tmp_path.iterdir()))}.h5ad"
    concat_on_disk(paths, out, QUIET, **kw)
    return ad.read_h5ad(out)


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_obsm_dataframes_concatenate_as_anndata_concatenates_them(tmp_path, join):
    genes = [f"g{i}" for i in range(6)]
    a = _store(tmp_path, "a", 5, genes, seed=0)
    b = _store(tmp_path, "b", 4, genes, seed=1, regulons=("ATF3_(+)", "PAX6_(-)"))
    got = _concat(tmp_path, [a, b], join=join)
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], join=join)
    frame, expected = got.obsm["RegulonsAUC"], ref.obsm["RegulonsAUC"]
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == list(got.obs_names)
    assert sorted(frame.columns) == sorted(expected.columns)
    pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False)
    assert np.array_equal(got.obsm["X_umap"], ref.obsm["X_umap"])


@pytest.mark.parametrize("merge", ["same", "unique", "first", "only"])
def test_varm_follows_merge_as_anndata_does(tmp_path, merge):
    """varm was never written: concat dropped it whatever --merge said."""
    genes = [f"g{i}" for i in range(6)]
    a = _store(tmp_path, "a", 5, genes, seed=0)
    b = _store(tmp_path, "b", 4, genes[::-1], seed=1, shift=1.0)  # topic_region differs
    got = _concat(tmp_path, [a, b], merge=merge)
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], merge=merge)
    assert sorted(got.varm.keys()) == sorted(ref.varm.keys())
    if merge == "first":
        assert sorted(got.varm.keys()) == ["Regulons", "topic_region"]
    for key in ref.varm:
        mine, theirs = got.varm[key], ref.varm[key]
        if isinstance(theirs, pd.DataFrame):
            pd.testing.assert_frame_equal(mine, theirs, check_dtype=False)
        else:
            assert np.array_equal(mine, theirs)


def test_varm_is_dropped_without_merge_with_a_note(tmp_path):
    genes = [f"g{i}" for i in range(6)]
    a, b = _store(tmp_path, "a", 5, genes, seed=0), _store(tmp_path, "b", 4, genes, seed=1)
    console = Console(record=True, width=200)
    out = tmp_path / "plain.h5ad"
    concat_on_disk([a, b], out, console)
    assert "Dropping varm: pass --merge" in console.export_text()
    assert len(ad.read_h5ad(out).varm) == 0


def test_an_outer_join_drops_varm_an_input_cannot_cover(tmp_path):
    """With genes one input lacks, that input's varm has no rows for them:
    it counts as missing, so `same` drops the entry and `first` keeps the
    input that covers every gene, if any."""
    a = _store(tmp_path, "a", 5, [f"g{i}" for i in range(6)], seed=0)
    b = _store(tmp_path, "b", 4, [f"g{i}" for i in range(3, 8)], seed=1)
    got = _concat(tmp_path, [a, b], join="outer", merge="same")
    assert len(got.varm) == 0
    got = _concat(tmp_path, [a, b], join="outer", merge="first")
    assert len(got.varm) == 0, "neither input has every gene of the union"

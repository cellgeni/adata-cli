"""Tests for `adata convert` of MuData: `.h5mu` files and MuData Zarr stores.

Fixtures are laid out as mudata's write_h5mu lays them out, every element
written with anndata's write_elem (see conftest), and checked against what
each was built from. mudata's own reader is the reference in
test_external_readers.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from adata.conversion import detect_format
from tests.conftest import write_h5mu

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy.sparse")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

INPUTS = {"h5mu": None, "zarr2": 2, "zarr3": 3}


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _mudata(tmp_path: Path, kind: str = "h5mu", **kw):
    zarr_format = INPUTS[kind]
    name = "in.h5mu" if zarr_format is None else "in.h5mu.zarr"
    return write_h5mu(tmp_path / name, zarr_format=zarr_format, **kw)


def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


# ---------------------------------------------------------------------------
# Reading


@pytest.mark.parametrize("kind", list(INPUTS))
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_each_modality_becomes_an_anndata_of_its_own(tmp_path, kind, suffix):
    truth = _mudata(tmp_path, kind)
    result = _ok(truth.path, "-o", tmp_path / f"out{suffix}")
    assert "MuData (shared cells)" in _out(result)
    for name, mod in truth.mods.items():
        got = _read(tmp_path / f"out.{name}{suffix}")
        assert list(got.obs_names) == list(mod.obs_names)
        assert list(got.var_names) == list(mod.var_names)
        assert np.array_equal(_dense(got.X), _dense(mod.X))
    assert not (tmp_path / f"out{suffix}").exists()


@pytest.mark.parametrize("kind", list(INPUTS))
def test_the_global_parts_go_onto_each_modalitys_cells(tmp_path, kind):
    """atac's cells are a reversed part of the global ones, so its rows are
    taken out of order."""
    truth = _mudata(tmp_path, kind)
    _ok(truth.path, "-o", tmp_path / "out.h5ad")
    for name in truth.mods:
        got, rows = _read(tmp_path / f"out.{name}.h5ad"), truth.rows(name)
        assert list(got.obs["leiden"]) == list(truth.obs["leiden"].iloc[rows])
        assert got.obs["sample"].isna().tolist() == truth.obs["sample"].isna().iloc[rows].tolist()
        assert np.allclose(got.obsm["X_wnn_umap"], truth.obsm["X_wnn_umap"][rows])
        graph = truth.obsp["wnn_connectivities"][rows][:, rows]
        assert np.array_equal(_dense(got.obsp["wnn_connectivities"]), _dense(graph))
        assert got.uns["wnn"]["k"] == 5


def test_a_global_name_the_modality_has_keeps_the_modalitys_with_a_warning(tmp_path):
    truth = _mudata(tmp_path)
    result = _ok(truth.path, "-o", tmp_path / "out.h5ad")
    text = _out(result)
    for where in ("obs/n_genes", "obsm/X_pca"):
        assert f"Not written: global {where} -- rna has its own {where}, which is kept." in text
    assert "Not written: global uns/rna_only -- rna has its own, which is kept." in text
    rna = _read(tmp_path / "out.rna.h5ad")
    assert list(rna.obs["n_genes"]) == list(truth.mods["rna"].obs["n_genes"])
    assert np.allclose(rna.obsm["X_pca"], truth.mods["rna"].obsm["X_pca"])
    assert rna.uns["rna_only"] == 1
    # atac has none of these, so it takes the global ones.
    atac = _read(tmp_path / "out.atac.h5ad")
    assert "n_genes" in atac.obs and atac.uns["rna_only"] == 2


def test_columns_pulled_from_a_modality_go_back_only_to_it(tmp_path):
    """`atac:tss` is atac's tss, pulled into the global obs: atac already
    has it, and rna does not get atac's."""
    truth = _mudata(tmp_path)
    result = _ok(truth.path, "-o", tmp_path / "out.h5ad")
    assert "1 global obs entry pulled from other modalities" in _out(result)
    rna, atac = _read(tmp_path / "out.rna.h5ad"), _read(tmp_path / "out.atac.h5ad")
    assert not any("tss" in c for c in rna.obs.columns)
    assert [c for c in atac.obs.columns if "tss" in c] == ["tss"]


def test_a_pulled_column_the_modality_lacks_comes_back_under_its_own_name(tmp_path):
    truth = _mudata(tmp_path)
    with h5py.File(truth.path, "a") as f:
        obs = f["mod/atac/obs"]
        del obs["tss"]
        obs.attrs["column-order"] = np.array([], dtype=h5py.string_dtype())
    _ok(truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac")
    atac = _read(tmp_path / "out.h5ad")
    rows = truth.rows("atac")
    assert np.allclose(atac.obs["tss"], truth.obs["atac:tss"].iloc[rows])


def test_mod_writes_that_modality_alone(tmp_path):
    truth = _mudata(tmp_path)
    _ok(truth.path, "-o", tmp_path / "atac.h5ad", "--mod", "atac")
    got = _read(tmp_path / "atac.h5ad")
    assert list(got.var_names) == list(truth.mods["atac"].var_names)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != truth.path.name) == ["atac.h5ad"]


def test_an_unknown_modality_is_refused_before_writing(tmp_path):
    truth = _mudata(tmp_path)
    result = _convert(truth.path, "-o", tmp_path / "out.h5ad", "--mod", "prot")
    assert result.exit_code == 1
    assert "No modality 'prot'; this file has: rna, atac." in _out(result)
    assert [p.name for p in tmp_path.iterdir()] == [truth.path.name]


def test_without_maps_cells_are_matched_by_name(tmp_path):
    """An older writer's file has no obsmap/varmap: the indexes still say
    which global cell each of the modality's is."""
    truth = _mudata(tmp_path, maps=False)
    result = _ok(truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac")
    assert "No obsmap/atac: its obs are matched to the global ones by name." in _out(result)
    got = _read(tmp_path / "out.h5ad")
    assert np.allclose(got.obsm["X_wnn_umap"], truth.obsm["X_wnn_umap"][truth.rows("atac")])


def test_a_map_that_does_not_place_each_cell_once_is_refused(tmp_path):
    truth = _mudata(tmp_path)
    with h5py.File(truth.path, "a") as f:
        where = f["obsmap/atac"][...]
        where[np.flatnonzero(where)[0]] = 0
        f["obsmap/atac"][...] = where
    result = _convert(truth.path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "obsmap/atac does not place each of the modality's 30 obs once" in _out(result)
    assert [p.name for p in tmp_path.iterdir()] == [truth.path.name]


def test_other_layouts_of_global_parts_are_taken_too(tmp_path):
    """A CSC graph stays CSC, a dense graph dense; a dataframe and a sparse
    matrix in obsm are taken row by row."""
    from anndata.io import write_elem

    truth = _mudata(tmp_path)
    n = len(truth.obs)
    rng = np.random.default_rng(3)
    csc = sp.random(n, n, density=0.3, format="csc", random_state=4, dtype=np.float32)
    dense = rng.random((n, n))
    frame = pd.DataFrame({"score": rng.random(n), "kind": pd.Categorical(rng.choice(["x", "y"], n))},
                         index=truth.obs.index)
    sparse = sp.random(n, 4, density=0.5, format="csr", random_state=5, dtype=np.float32)
    with h5py.File(truth.path, "a") as f:
        write_elem(f["obsp"], "distances", csc)
        write_elem(f["obsp"], "dense", dense)
        write_elem(f["obsm"], "frame", frame)
        write_elem(f["obsm"], "sparse", sparse)
    _ok(truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac")
    got, rows = _read(tmp_path / "out.h5ad"), truth.rows("atac")
    assert sp.isspmatrix_csc(got.obsp["distances"]) or got.obsp["distances"].format == "csc"
    assert np.array_equal(_dense(got.obsp["distances"]), csc[rows][:, rows].toarray())
    assert np.allclose(got.obsp["dense"], dense[rows][:, rows])
    pd.testing.assert_frame_equal(got.obsm["frame"], frame.iloc[rows], check_names=False)
    assert np.array_equal(_dense(got.obsm["sparse"]), sparse[rows].toarray())


def test_small_blocks_read_the_same(tmp_path, monkeypatch):
    from adata.conversion import mudata

    truth = _mudata(tmp_path)
    _ok(truth.path, "-o", tmp_path / "big.h5ad", "--mod", "atac")
    monkeypatch.setattr(mudata, "ROW_BLOCK", 3)
    _ok(truth.path, "-o", tmp_path / "small.h5ad", "--mod", "atac")
    big, small = _read(tmp_path / "big.h5ad"), _read(tmp_path / "small.h5ad")
    assert np.allclose(big.obsm["X_wnn_umap"], small.obsm["X_wnn_umap"])
    assert np.array_equal(_dense(big.obsp["wnn_connectivities"]), _dense(small.obsp["wnn_connectivities"]))


@pytest.mark.parametrize("kind", list(INPUTS))
def test_a_mudata_is_told_from_an_anndata(tmp_path, kind):
    """An .h5mu's root has obs and var groups, so it was taken for an h5ad,
    and a MuData Zarr store for a Zarr AnnData."""
    truth = _mudata(tmp_path, kind)
    assert detect_format(truth.path) == "mudata"


def test_a_mudata_without_root_attributes_is_told_by_its_userblock(tmp_path):
    truth = _mudata(tmp_path, root_attrs=False)
    assert detect_format(truth.path) == "mudata"


def test_a_zarr_output_keeps_a_zarr_inputs_version(tmp_path):
    import zarr

    truth = _mudata(tmp_path, "zarr2")
    _ok(truth.path, "-o", tmp_path / "out.zarr", "--mod", "rna")
    assert zarr.open_group(str(tmp_path / "out.zarr"), mode="r").metadata.zarr_format == 2


# ---------------------------------------------------------------------------
# Writing


def _multiome(tmp_path: Path, *, types=("Gene Expression",) * 6 + ("Peaks",) * 4, name="multi.h5ad"):
    rng = np.random.default_rng(0)
    n = 20
    obj = ad.AnnData(
        sp.random(n, len(types), density=0.3, format="csr", random_state=0, dtype=np.float32),
        obs=pd.DataFrame({"leiden": pd.Categorical(rng.choice(["x", "y"], n))},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame({"feature_types": pd.Categorical(list(types)),
                          "gene_ids": [f"id{j}" for j in range(len(types))]},
                         index=[f"f{j}" for j in range(len(types))]),
    )
    obj.layers["counts"] = obj.X.copy()
    obj.obsm["X_umap"] = rng.normal(size=(n, 2))
    obj.varm["loadings"] = rng.normal(size=(len(types), 3))
    obj.obsp["connectivities"] = sp.random(n, n, density=0.2, format="csr", random_state=1)
    obj.uns["note"] = "hi"
    obj.raw = obj
    path = tmp_path / name
    obj.write_h5ad(path)
    return path, obj


def test_an_anndata_is_split_into_modalities_by_feature_type(tmp_path):
    from anndata.io import read_elem

    source, obj = _multiome(tmp_path)
    out = tmp_path / "multi.h5mu"
    result = _ok(source, "-o", out)
    assert "rna (6 var), atac (4 var)" in _out(result)
    assert out.read_bytes()[:6] == b"MuData"
    with h5py.File(out) as f:
        assert f.userblock_size == 512
        assert f.attrs["encoding-type"] == "mudata" and int(f.attrs["axis"]) == 0
        assert [m.decode() if isinstance(m, bytes) else m for m in f["mod"].attrs["mod-order"]] == ["rna", "atac"]
        assert f["mod/rna"].attrs["encoding-type"] == "anndata"
        assert list(f["obsmap/atac"][...]) == list(range(1, 21))
        assert list(f["varmap/atac"][...]) == [0] * 6 + [1, 2, 3, 4]
        assert f["varm/rna"][...].tolist() == [True] * 6 + [False] * 4
        assert f["obsm/rna"][...].all()
        assert list(f["mod/rna/obs"].keys()) == ["_index"], "the cells' annotations are global"
    for mod, kind in (("rna", "Gene Expression"), ("atac", "Peaks")):
        with h5py.File(out) as f:
            got = read_elem(f["mod"][mod])
        sub = obj[:, obj.var["feature_types"] == kind]
        assert list(got.var_names) == list(sub.var_names)
        assert np.array_equal(_dense(got.X), _dense(sub.X))
        assert np.array_equal(_dense(got.layers["counts"]), _dense(sub.layers["counts"]))
        assert np.allclose(got.varm["loadings"], sub.varm["loadings"])
        assert list(got.raw.var_names) == list(sub.var_names)


def test_a_mudata_written_and_read_back_gives_each_modality_with_its_cells_annotations(tmp_path):
    source, obj = _multiome(tmp_path)
    _ok(source, "-o", tmp_path / "multi.h5mu")
    _ok(tmp_path / "multi.h5mu", "-o", tmp_path / "back.h5ad")
    for mod, kind in (("rna", "Gene Expression"), ("atac", "Peaks")):
        got, sub = _read(tmp_path / f"back.{mod}.h5ad"), obj[:, obj.var["feature_types"] == kind]
        assert list(got.obs["leiden"]) == list(obj.obs["leiden"])
        assert np.allclose(got.obsm["X_umap"], obj.obsm["X_umap"])
        assert np.array_equal(_dense(got.obsp["connectivities"]), _dense(obj.obsp["connectivities"]))
        assert got.uns["note"] == "hi"
        assert np.array_equal(_dense(got.X), _dense(sub.X))
        assert list(got.var.columns) == list(sub.var.columns)


def test_mod_writes_the_whole_anndata_as_one_modality(tmp_path):
    from anndata.io import read_elem

    source, obj = _multiome(tmp_path)
    _ok(source, "-o", tmp_path / "one.h5mu", "--mod", "gex")
    with h5py.File(tmp_path / "one.h5mu") as f:
        assert list(f["mod"].keys()) == ["gex"]
        assert len(read_elem(f["mod/gex/var"])) == 10


def test_without_feature_types_the_anndata_is_rna(tmp_path):
    rng = np.random.default_rng(0)
    obj = ad.AnnData(rng.random((5, 3)).astype(np.float32))
    obj.write_h5ad(tmp_path / "plain.h5ad")
    _ok(tmp_path / "plain.h5ad", "-o", tmp_path / "plain.h5mu")
    with h5py.File(tmp_path / "plain.h5mu") as f:
        assert list(f["mod"].keys()) == ["rna"]


def test_other_feature_types_are_named_after_themselves(tmp_path):
    types = ("Gene Expression",) * 3 + ("CRISPR Guide Capture",) * 2 + ("Antibody Capture",) * 2
    source, _ = _multiome(tmp_path, types=types)
    _ok(source, "-o", tmp_path / "multi.h5mu")
    with h5py.File(tmp_path / "multi.h5mu") as f:
        assert sorted(f["mod"].keys()) == ["crispr_guide_capture", "prot", "rna"]


@pytest.mark.parametrize("extra, message", [
    (["--mod", "x", "--feature-types-column", "feature_types"], "pass one"),
    (["--mod", "a/b"], "cannot be empty or hold '/'"),
    (["--feature-types-column", "nope"], "no such var column"),
])
def test_export_refusals_leave_nothing(tmp_path, extra, message):
    source, _ = _multiome(tmp_path)
    result = _convert(source, "-o", tmp_path / "multi.h5mu", *extra)
    assert result.exit_code == 1 and message in _out(result)
    assert not (tmp_path / "multi.h5mu").exists()


def test_an_obsm_entry_named_like_a_modality_is_refused(tmp_path):
    """MuData keeps obsm/<mod> for the modality's mask."""
    source, obj = _multiome(tmp_path)
    obj.obsm["rna"] = np.zeros((obj.n_obs, 2))
    obj.write_h5ad(source)
    result = _convert(source, "-o", tmp_path / "multi.h5mu")
    assert result.exit_code == 1 and "obsm/rna has the name of a modality" in _out(result)


def test_compression_applies_to_the_global_parts_too(tmp_path):
    truth = _mudata(tmp_path)
    _ok(truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac", "--compression", "gzip")
    with h5py.File(tmp_path / "out.h5ad") as f:
        assert f["X/data"].compression == "gzip"
        assert f["obsm/X_wnn_umap"].compression == "gzip"
        assert f["obsp/wnn_connectivities/data"].compression == "gzip"
        assert f["obs/leiden/codes"].compression == "gzip"


def test_a_global_graph_of_the_wrong_shape_is_refused_before_writing(tmp_path):
    from anndata.io import write_elem

    truth = _mudata(tmp_path)
    with h5py.File(truth.path, "a") as f:
        write_elem(f["obsp"], "odd", sp.random(len(truth.obs), 3, density=0.5, format="csr"))
    result = _convert(truth.path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1 and "Global obsp/odd is (40, 3), not 40 x 40." in _out(result)
    assert [p.name for p in tmp_path.iterdir()] == [truth.path.name]


def test_feature_types_that_meet_as_names_get_a_modality_each(tmp_path):
    """`Foo Bar` and `foo_bar` both name a modality foo_bar: they were given
    one name, and the second group failed with the output half written."""
    types = ("Foo Bar",) * 3 + ("foo_bar",) * 2
    source, _ = _multiome(tmp_path, types=types)
    _ok(source, "-o", tmp_path / "multi.h5mu")
    with h5py.File(tmp_path / "multi.h5mu") as f:
        assert sorted(f["mod"].keys()) == ["foo_bar", "foo_bar_1"]
        assert f["mod/foo_bar/X"].attrs["shape"][1] == 3


def test_a_remapped_sparse_element_keeps_its_layout(tmp_path):
    """Global sparse elements were rebuilt with int32 indices, uncompressed,
    whatever the source held."""
    from anndata.io import write_elem

    truth = _mudata(tmp_path)
    n = len(truth.obs)
    graph = sp.random(n, n, density=0.1, format="csr", random_state=6, dtype=np.float32)
    graph.indices = graph.indices.astype(np.int64)
    graph.indptr = graph.indptr.astype(np.int64)
    with h5py.File(truth.path, "a") as f:
        write_elem(f["obsp"], "wide", graph, dataset_kwargs={"compression": "gzip"})
    _ok(truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac")
    with h5py.File(tmp_path / "out.h5ad") as f:
        out = f["obsp/wide"]
        assert out["indices"].dtype == np.int64 and out["indptr"].dtype == np.int64
        assert out["data"].compression == "gzip" and out["indices"].compression == "gzip"

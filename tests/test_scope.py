"""Tests for `adata convert` of SCope looms: pySCENIC's, SCENIC+'s and
pycistopic's results, and R's SCopeLoomR.

Fixtures are written with h5py as pySCENIC's `export2loom` writes them (see
conftest), and checked against what each was built from. The pySCENIC
tutorial's own reading recipe, with loompy, is the reference in
test_external_readers.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_loom, write_scope_loom

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _ok(*args):
    result = runner.invoke(app, ["convert", *map(str, args)])
    assert result.exit_code == 0, _out(result)
    return result


def _set_metadata(path: Path, text: str) -> None:
    with h5py.File(path, "a") as f:
        del f["attrs/MetaData"]
        f["attrs"].create_dataset("MetaData", data=text.encode(), dtype=h5py.string_dtype("ascii"))


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_regulons_are_dataframes_named_by_regulon(tmp_path, suffix):
    truth = write_scope_loom(tmp_path / "scope.loom")
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out)
    assert "loom (SCope)" in _out(result)
    assert "3 regulons, 2 embeddings, 2 clusterings" in _out(result)
    assert "Not read" not in _out(result)

    got = _read(out)
    auc = got.obsm["RegulonsAUC"]
    assert isinstance(auc, pd.DataFrame)
    assert list(auc.columns) == truth.regulons
    assert list(auc.index) == truth.loom.cell_ids
    assert np.array_equal(auc.to_numpy(), truth.auc)
    regulons = got.varm["Regulons"]
    assert list(regulons.columns) == truth.regulons
    assert list(regulons.index) == truth.var_names
    assert (regulons.dtypes == np.int64).all()
    assert np.array_equal(regulons.to_numpy(), truth.membership)
    # The attributes SCope's own rules read are not dataframes as well.
    assert not {"Embedding", "Embeddings_X", "Embeddings_Y", "Clusterings"} & set(got.obsm)
    assert got.obs["ClusterID"].tolist() == truth.clusterings["0"].tolist()


@pytest.mark.parametrize("default_in_extra", [True, False])
def test_embeddings_are_named_from_the_metadata_and_paired_by_id(tmp_path, default_in_extra):
    """pySCENIC's export2loom writes the default embedding in Embeddings_X
    and _Y as id -1 and again in Embedding; other writers keep it in
    Embedding only."""
    truth = write_scope_loom(tmp_path / "scope.loom", default_in_extra=default_in_extra)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert sorted(k for k in got.obsm if k.startswith("X_")) == ["X_UMAP", "X_tSNE_default"]
    assert np.array_equal(got.obsm["X_tSNE_default"], truth.embeddings["-1"])
    assert got.obsm["X_UMAP"].dtype == np.float32, "each embedding keeps its dtype"
    assert np.array_equal(got.obsm["X_UMAP"], truth.embeddings["0"])
    where = got.uns["scope"]["embeddings"]
    assert where.loc["-1"].tolist() == ["tSNE (default)", "X_tSNE_default"]
    assert where.loc["0"].tolist() == ["UMAP", "X_UMAP"]


def test_clusterings_are_categoricals_labelled_by_their_clusters(tmp_path):
    truth = write_scope_loom(tmp_path / "scope.loom")
    with h5py.File(truth.path, "a") as f:
        values = f["col_attrs/Clusterings"][...]
        values["0"][0] = 7  # no such cluster
        f["col_attrs/Clusterings"][...] = values
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)

    cell_type = got.obs["Cell Type"]
    assert list(cell_type.cat.categories) == ["B cell", "T cell", "NK"], "in cluster-id order"
    labels = np.array(["B cell", "T cell", "NK"])[truth.clusterings["0"]]
    assert pd.isna(cell_type.iloc[0]), "a value with no cluster is missing"
    assert cell_type.iloc[1:].tolist() == labels[1:].tolist()

    leiden = got.obs["Leiden resolution 0.4"]
    assert list(leiden.cat.categories) == ["Leiden 0", "1"], "an undescribed cluster is its id"
    assert leiden.tolist() == np.array(["Leiden 0", "1"])[truth.clusterings["1"]].tolist()
    assert list(got.obs.columns)[-2:] == ["Cell Type", "Leiden resolution 0.4"]
    where = got.uns["scope"]["clusterings"]
    assert where.loc["1"].tolist() == ["Leiden resolution 0.4", "Leiden resolution 0.4"]


def test_a_clustering_named_like_an_obs_column_gets_a_suffix(tmp_path):
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=False)
    metadata = dict(truth.metadata)
    metadata["clusterings"] = [dict(metadata["clusterings"][0], name="Tissue")] + metadata["clusterings"][1:]
    _set_metadata(truth.path, json.dumps(metadata))
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "Clusterings written under other obs names: 'Tissue' as 'Tissue_1'" in _out(result)
    got = _read(out)
    assert got.obs["Tissue"].tolist()[:2] == ["cortex", "hippocampus"], "the attribute is kept"
    assert got.obs["Tissue_1"].tolist()[:3] == ["B cell", "T cell", "NK"]


@pytest.mark.parametrize("compressed", [True, False], ids=["compressed", "plain"])
def test_metadata_is_decoded_into_uns_scope(tmp_path, compressed):
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=compressed)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    scope = got.uns["scope"]
    assert json.loads(scope["metadata"]) == truth.metadata
    thresholds = scope["regulon_thresholds"]
    assert list(thresholds.index) == truth.regulons
    assert list(thresholds.columns) == ["defaultThresholdValue", "defaultThresholdName", "motifData"]
    assert thresholds["defaultThresholdValue"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert thresholds.loc["SOX2 (12g)", "motifData"] == "motif2.png"
    attrs = got.uns["loom"]["attrs"]
    assert "MetaData" not in attrs, "the decoded copy replaces it"
    assert attrs["Genome"] == "hg38" and attrs["SCopeTreeL1"] == "Tissue"


def test_metadata_that_cannot_be_decoded_is_kept_with_a_warning(tmp_path):
    truth = write_scope_loom(tmp_path / "scope.loom")
    _set_metadata(truth.path, "eJzLSM3JyVcozy/KSQEAGgQEXQ==!")  # not base64
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "MetaData is neither JSON nor compressed JSON" in _out(result)
    got = _read(out)
    assert got.uns["scope"]["metadata"] == "eJzLSM3JyVcozy/KSQEAGgQEXQ==!"
    assert "regulon_thresholds" not in got.uns["scope"]
    # Named by id, with the clusters found in the file.
    assert np.array_equal(got.obsm["X_embedding_-1"], truth.embeddings["-1"])
    assert np.array_equal(got.obsm["X_embedding_0"], truth.embeddings["0"])
    assert list(got.obs["clustering_0"].cat.categories) == ["0", "1", "2"]
    assert got.obs["clustering_1"].tolist() == [str(v) for v in truth.clusterings["1"]]


def test_a_clustering_named_dot_dot_gets_a_name_a_column_can_hold(tmp_path):
    """MetaData may name a clustering "..", which as an HDF5 or Zarr member
    is the parent group: writing obs failed."""
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=False)
    metadata = dict(truth.metadata)
    metadata["clusterings"] = [dict(c, name=name) for c, name in zip(metadata["clusterings"], ("..", "a/b"))]
    _set_metadata(truth.path, json.dumps(metadata))
    for suffix in (".h5ad", ".zarr"):
        out = tmp_path / f"out{suffix}"
        result = _ok(truth.path, "-o", out)
        assert "'..' as '_..', 'a/b' as 'a_b'" in _out(result)
        got = _read(out)
        assert got.obs["_.."].tolist()[:3] == ["B cell", "T cell", "NK"]
        assert got.uns["scope"]["clusterings"].loc["0"].tolist() == ["..", "_.."]


def test_a_field_name_holding_a_slash_is_renamed_with_a_warning(tmp_path):
    truth = write_scope_loom(tmp_path / "scope.loom", regulons=("HES1/HEY1_(+)", "_index", "TP53_(+)"))
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert (
        "Renamed in col_attrs RegulonsAUC: 'HES1/HEY1_(+)' as 'HES1_HEY1_(+)', '_index' as '_index_1'"
        in _out(result)
    )
    got = _read(out)
    assert list(got.obsm["RegulonsAUC"].columns) == ["HES1_HEY1_(+)", "_index_1", "TP53_(+)"]
    assert list(got.obsm["RegulonsAUC"].index) == truth.loom.cell_ids
    assert np.array_equal(got.varm["Regulons"].to_numpy(), truth.membership)


def test_a_region_loom_names_var_by_its_regions(tmp_path):
    """SCENIC+ region-based and pycistopic looms put regions where genes go."""
    truth = write_scope_loom(tmp_path / "regions.loom", region=True)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert list(got.var_names) == truth.var_names
    assert got.var_names[1] == "chr2:1000-1500"
    assert got.var["linkedGene"].tolist()[:3] == ["", "Gene0", "Gene1"]
    assert list(got.varm["Regulons"].index) == truth.var_names


def test_small_blocks_read_the_same(tmp_path, monkeypatch):
    """Attributes are read a block of rows at a time; with blocks of 7 rows
    every boundary falls inside the fixture."""
    from adata.conversion import loom

    monkeypatch.setattr(loom, "STRING_STEP", 7)
    truth = write_scope_loom(tmp_path / "scope.loom", n_cells=50, n_genes=23)
    out = tmp_path / "out.zarr"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert np.array_equal(got.obsm["RegulonsAUC"].to_numpy(), truth.auc)
    assert list(got.obsm["RegulonsAUC"].index) == truth.loom.cell_ids
    assert np.array_equal(got.varm["Regulons"].to_numpy(), truth.membership)
    assert list(got.varm["Regulons"].index) == truth.var_names
    assert np.array_equal(got.obsm["X_tSNE_default"], truth.embeddings["-1"])
    labels = np.array(["Leiden 0", "1"])[truth.clusterings["1"]]
    assert got.obs["Leiden resolution 0.4"].tolist() == labels.tolist()


def test_r_logicals_and_factors_read_as_r_meant_them(tmp_path):
    """R's hdf5r stores a factor as an enum of its levels and a logical as an
    enum of FALSE, TRUE and NA. Both came out as their codes: the CellType
    of SCENIC's example loom as 1 to 5. SCopeLoomR writes regulons as
    logicals, and old versions wrote clusterings as factors of "0", "1",
    ... over codes 1, 2, ...."""
    truth = write_scope_loom(tmp_path / "r.loom", compressed=False)
    logical = h5py.enum_dtype({"FALSE": 0, "TRUE": 1, "NA": 2}, basetype="i1")
    tissue = h5py.enum_dtype({"cortex": 1, "hippocampus": 2}, basetype="u1")
    clusters = h5py.enum_dtype({"0": 1, "1": 2, "2": 3}, basetype="u1")
    membership = truth.membership.astype(np.int8)
    membership[0, 0] = 2
    with h5py.File(truth.path, "a") as f:
        row, col = f["row_attrs"], f["col_attrs"]
        del row["Regulons"]
        records = np.zeros(membership.shape[0], dtype=[(r, logical) for r in truth.regulons])
        for i, name in enumerate(truth.regulons):
            records[name] = membership[:, i]
        row.create_dataset("Regulons", data=records)
        del col["Tissue"]
        col.create_dataset("Tissue", data=(np.arange(30) % 2 + 1).astype(np.uint8), dtype=tissue)
        del col["Passed"]
        passed = h5py.enum_dtype({"FALSE": 0, "TRUE": 1}, basetype="u1")
        col.create_dataset("Passed", data=(np.arange(30) % 2).astype(np.uint8), dtype=passed)
        strand = h5py.enum_dtype({"+": 1, "-": 2}, basetype="u1")
        row.create_dataset("Strand", data=(np.arange(10) % 2 + 1).astype(np.uint8), dtype=strand)
        del col["Clusterings"]
        records = np.zeros(30, dtype=[("0", clusters)])
        records["0"] = truth.clusterings["0"] + 1
        col.create_dataset("Clusterings", data=records)
    # SCopeLoomR records ids as text.
    metadata = dict(truth.metadata)
    metadata["clusterings"] = [
        dict(c, id=str(c["id"]), clusters=[dict(k, id=str(k["id"])) for k in c["clusters"]])
        for c in metadata["clusterings"][:1]
    ]
    metadata["embeddings"] = [dict(e, id=str(e["id"])) for e in metadata["embeddings"]]
    _set_metadata(truth.path, json.dumps(metadata))

    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    regulons = got.varm["Regulons"]
    assert str(regulons.dtypes.iloc[0]) == "boolean"
    assert pd.isna(regulons.iloc[0, 0])
    assert regulons.iloc[1:, 0].tolist() == truth.membership[1:, 0].astype(bool).tolist()
    assert list(got.obs["Tissue"].cat.categories) == ["cortex", "hippocampus"], "R's level order"
    assert got.obs["Tissue"].tolist()[:2] == ["cortex", "hippocampus"]
    assert got.obs["Passed"].tolist()[:2] == [False, True]
    assert got.var["Strand"].tolist()[:3] == ["+", "-", "+"]
    assert list(got.var.columns) == ["Gene", "Start", "Strand"]
    labels = np.array(["B cell", "T cell", "NK"])[truth.clusterings["0"]]
    assert got.obs["Cell Type"].tolist() == labels.tolist()
    assert "X_tSNE_default" in got.obsm


def test_any_compound_attribute_becomes_a_dataframe(tmp_path):
    """VSN's cluster markers, in a loom without SCope's other marks: a
    dataframe per attribute, its text decoded, and the loom no SCope loom."""
    truth = write_loom(tmp_path / "vsn.loom")
    markers = np.zeros(10, dtype=[("0", "i8"), ("1", "f4"), ("note", h5py.string_dtype("ascii"))])
    markers["0"] = np.arange(10) % 2
    markers["1"] = np.linspace(0, 1, 10)
    markers["note"] = [f"G&#232;ne{i}".encode() for i in range(10)]
    with h5py.File(truth.path, "a") as f:
        f["row_attrs"].create_dataset("ClusterMarkers_0", data=markers)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "loom (spec 3.0.0)" in _out(result)
    got = _read(out)
    frame = got.varm["ClusterMarkers_0"]
    assert list(frame.columns) == ["0", "1", "note"]
    assert list(frame.index) == truth.gene_ids
    assert frame["0"].tolist() == markers["0"].tolist()
    assert frame["1"].dtype == np.float32
    assert frame["note"].tolist()[0] == "Gène0", "text read as loompy reads it"
    assert "scope" not in got.uns


def test_a_two_dimensional_compound_attribute_is_left_out_with_a_warning(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    with h5py.File(truth.path, "a") as f:
        f["col_attrs"].create_dataset("Pairs", data=np.zeros((30, 2), dtype=[("a", "f8"), ("b", "f8")]))
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "Not read: col_attrs Pairs" in _out(result)
    assert "Pairs" not in _read(out).obsm

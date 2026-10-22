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


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


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


# ---------------------------------------------------------------------------
# export: --scope


def _scope_loom(path: Path) -> dict:
    """A SCope loom's parts, read as SCope's server reads them."""
    with h5py.File(path, "r") as f:
        cols, rows = f["col_attrs"], f["row_attrs"]
        meta = json.loads(f["attrs/MetaData"][()])
        return {
            "meta": meta,
            "cols": {k: cols[k][...] for k in cols},
            "rows": {k: rows[k][...] for k in rows},
        }


def test_a_scope_loom_round_trips_through_h5ad(tmp_path):
    truth = write_scope_loom(tmp_path / "scope.loom")
    first, back, again = tmp_path / "a.h5ad", tmp_path / "back.loom", tmp_path / "b.h5ad"
    _ok(truth.path, "-o", first)
    result = _ok(first, "-o", back, "--scope")
    # write_loom's 2-column `embedding` attribute is an embedding too.
    assert "loom 3.0.0 (SCope)" in _out(result) and "3 regulons, 3 embeddings, 2 clusterings" in _out(result)
    _ok(back, "-o", again)
    a, b = _read(first), _read(again)
    # SCope's regulon names hold no spaces: "SOX2 (12g)" comes back as "SOX2_(12g)".
    unspaced = lambda frame: frame.rename(columns=lambda c: c.replace(" ", "_"))  # noqa: E731
    pd.testing.assert_frame_equal(unspaced(a.obsm["RegulonsAUC"]), b.obsm["RegulonsAUC"])
    pd.testing.assert_frame_equal(unspaced(a.varm["Regulons"]), b.varm["Regulons"])
    for key in ("X_tSNE_default", "X_UMAP"):
        assert np.array_equal(a.obsm[key], b.obsm[key])
    assert a.obs["Cell Type"].tolist() == b.obs["Cell Type"].tolist()
    assert a.obs["Leiden resolution 0.4"].tolist() == b.obs["Leiden resolution 0.4"].tolist()
    assert list(b.uns["scope"]["embeddings"]["name"]) == ["tSNE (default)", "UMAP", "embedding"]
    thresholds = a.uns["scope"]["regulon_thresholds"].rename(index=lambda r: r.replace(" ", "_"))
    pd.testing.assert_frame_equal(thresholds, b.uns["scope"]["regulon_thresholds"])

    loom = _scope_loom(back)
    assert loom["meta"]["embeddings"] == [
        {"id": -1, "name": "tSNE (default)"}, {"id": 0, "name": "UMAP"}, {"id": 1, "name": "embedding"},
    ], "recorded ids and names kept, a new one numbered after them"
    assert [c["id"] for c in loom["meta"]["clusterings"]] == [0, 1], "ids are list positions"
    assert loom["cols"]["Embeddings_X"].dtype.names == ("-1", "0", "1")
    assert set(loom["cols"]["Embedding"].dtype.names) == {"_X", "_Y"}
    assert "X_UMAP" not in loom["cols"], "an embedding is not written twice"


def test_a_plain_anndata_becomes_a_scope_loom(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    obj = ad.AnnData(
        X=rng.poisson(1, (n, 6)).astype(np.float32),
        obs=pd.DataFrame({"leiden": pd.Categorical([str(i % 3) for i in range(n)]),
                          "n_counts": rng.random(n), "batch": pd.Categorical(["a", "b"] * 15)},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"G{j}" for j in range(6)]),
    )
    obj.obs.loc["c0", "leiden"] = np.nan
    obj.obsm["X_pca"] = rng.normal(size=(n, 5))
    obj.obsm["X_umap"] = rng.normal(size=(n, 2))
    obj.obsm["X_draw"] = rng.normal(size=(n, 2))
    obj.obsm["RegulonsAUC"] = pd.DataFrame(rng.random((n, 2)), index=obj.obs_names, columns=["TP53 (5g)", "MYC (9g)"])
    obj.varm["Regulons"] = pd.DataFrame({"TP53 (5g)": [1, 0, 1, 0, 0, 1], "MYC (9g)": [0] * 6}, index=obj.var_names)
    source = tmp_path / "plain.h5ad"
    obj.write_h5ad(source)
    out = tmp_path / "out.loom"
    result = _ok(source, "-o", out, "--scope", "--clusterings", "leiden")
    assert "'TP53 (5g)' as 'TP53_(5g)'" in _out(result), "SCope's names hold no spaces"
    loom = _scope_loom(out)
    meta = loom["meta"]
    assert meta["embeddings"] == [{"id": -1, "name": "umap"}, {"id": 0, "name": "draw"}]
    assert np.array_equal(loom["cols"]["Embedding"]["_X"], obj.obsm["X_umap"][:, 0])
    assert np.array_equal(loom["cols"]["Embeddings_Y"]["0"], obj.obsm["X_draw"][:, 1])
    assert loom["cols"]["X_pca"].shape == (n, 5), "other obsm arrays stay plain"
    assert meta["clusterings"] == [{"id": 0, "group": "", "name": "leiden",
                                    "clusters": [{"id": 0, "description": "0"}, {"id": 1, "description": "1"},
                                                 {"id": 2, "description": "2"}]}]
    codes = loom["cols"]["Clusterings"]["0"]
    assert codes[0] == -1 and codes[1:4].tolist() == [1, 2, 0]
    assert loom["cols"]["ClusterID"].tolist() == codes.tolist()
    assert loom["cols"]["RegulonsAUC"].dtype.names == ("TP53_(5g)", "MYC_(9g)")
    assert loom["rows"]["Regulons"].dtype.names == ("TP53_(5g)", "MYC_(9g)")
    assert {m["name"] for m in meta["metrics"]} == {"n_counts"}
    assert {a["name"]: a["values"] for a in meta["annotations"]}["batch"] == ["a", "b"]

    result = _convert(source, "-o", tmp_path / "x.loom", "--scope", "--default-embedding", "X_draw")
    assert result.exit_code == 0
    assert _scope_loom(tmp_path / "x.loom")["meta"]["embeddings"][0] == {"id": -1, "name": "draw"}


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--scope", "--clusterings", "n_counts"], "--clusterings n_counts: not a categorical obs column"),
        (["--scope", "--clusterings", "nope"], "--clusterings nope: no such obs column"),
        (["--scope", "--default-embedding", "X_pca"], "--default-embedding X_pca: not an obsm array of two columns"),
        (["--clusterings", "leiden"], "--clusterings shapes a SCope loom: pass --scope too"),
    ],
)
def test_scope_export_refusals_leave_nothing(tmp_path, extra, message):
    obj = ad.AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"leiden": pd.Categorical(["a", "b", "a", "b"]), "n_counts": np.arange(4.0)},
                         index=list("wxyz")),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    obj.obsm["X_pca"] = np.zeros((4, 3))
    source = tmp_path / "s.h5ad"
    obj.write_h5ad(source)
    out = tmp_path / "out.loom"
    result = _convert(source, "-o", out, *extra)
    assert result.exit_code == 1 and message in _out(result)
    assert not out.exists()


def test_too_many_regulons_for_hdf5_are_refused_before_writing(tmp_path):
    n = 4
    names = [f"REGULON_NUMBER_{i:05d}_(+)" for i in range(2500)]
    obj = ad.AnnData(X=np.ones((n, 2), dtype=np.float32), obs=pd.DataFrame(index=list("wxyz")),
                     var=pd.DataFrame(index=["g1", "g2"]))
    obj.obsm["RegulonsAUC"] = pd.DataFrame(np.zeros((n, len(names))), index=obj.obs_names, columns=names)
    source = tmp_path / "s.h5ad"
    obj.write_h5ad(source)
    out = tmp_path / "out.loom"
    result = _convert(source, "-o", out, "--scope")
    assert result.exit_code == 1
    assert "obsm/RegulonsAUC has 2,500 columns, more than HDF5 can hold" in _out(result)
    assert not out.exists()


def test_cluster_markers_need_their_metrics(tmp_path):
    """SCope crashes on ClusterMarkers_<id> when its clustering lists no
    clusterMarkerMetrics: such markers are left out, those with their
    _<accessor> tables get metrics named after them."""
    truth = write_scope_loom(tmp_path / "scope.loom")
    with h5py.File(truth.path, "a") as f:
        rows = f["row_attrs"]
        marks = np.zeros(10, dtype=[("0", "i8"), ("1", "i8"), ("2", "i8")])
        marks["0"][:3] = 1
        rows.create_dataset("ClusterMarkers_0", data=marks)
        rows.create_dataset("ClusterMarkers_0_avg_logFC", data=np.zeros(10, dtype=[("0", "f4"), ("1", "f4"), ("2", "f4")]))
        rows.create_dataset("ClusterMarkers_1", data=np.zeros(10, dtype=[("0", "i8"), ("1", "i8")]))
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    result = _ok(first, "-o", back, "--scope")
    assert "Not written: ClusterMarkers_1 -- without metrics" in _out(result)
    loom = _scope_loom(back)
    assert "ClusterMarkers_0" in loom["rows"] and "ClusterMarkers_0_avg_logFC" in loom["rows"]
    assert "ClusterMarkers_1" not in loom["rows"]
    assert loom["meta"]["clusterings"][0]["clusterMarkerMetrics"] == [
        {"accessor": "avg_logFC", "name": "avg_logFC", "description": ""}
    ]


def test_a_plain_export_of_a_scope_anndata_points_to_scope(tmp_path):
    truth = write_scope_loom(tmp_path / "scope.loom")
    first = tmp_path / "a.h5ad"
    _ok(truth.path, "-o", first)
    result = _ok(first, "-o", tmp_path / "plain.loom")
    assert "This came from a SCope loom: pass --scope" in _out(result)
    with h5py.File(tmp_path / "plain.loom", "r") as f:
        assert "RegulonsAUC" not in f["col_attrs"]


def test_scope_export_writes_in_small_blocks_the_same(tmp_path, monkeypatch):
    from adata.conversion import loom

    monkeypatch.setattr(loom, "STRING_STEP", 7)
    truth = write_scope_loom(tmp_path / "scope.loom", n_cells=50)
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    _ok(first, "-o", back, "--scope")
    loom_parts = _scope_loom(back)
    got = np.column_stack([loom_parts["cols"]["RegulonsAUC"][r.replace(" ", "_")] for r in truth.regulons])
    assert np.array_equal(got, truth.auc)
    assert np.array_equal(loom_parts["cols"]["Embeddings_X"]["0"], truth.embeddings["0"][:, 0])
    assert loom_parts["cols"]["Clusterings"]["1"].tolist() == truth.clusterings["1"].tolist()


def test_metadata_from_columns_and_a_thresholds_table(tmp_path):
    """Text and nullable obs columns are annotations and metrics too; with no
    recorded MetaData, regulon thresholds come from uns["scope"]'s table."""
    n = 6
    obj = ad.AnnData(
        X=np.ones((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"donor": ["d1", "d2", "d3", "d4", "d5", "d6"],
                          "n_genes": pd.array([1, None, 3, 4, 5, 6], dtype="Int64")},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    obj.obsm["RegulonsAUC"] = pd.DataFrame(np.zeros((n, 2)), index=obj.obs_names, columns=["A_(+)", "B (2g)"])
    obj.uns["scope"] = {"regulon_thresholds": pd.DataFrame(
        {"defaultThresholdValue": [0.5, 0.25], "motifData": ["a.png", "b.png"]}, index=["A_(+)", "B (2g)"])}
    source = tmp_path / "s.h5ad"
    obj.write_h5ad(source)
    out = tmp_path / "out.loom"
    _ok(source, "-o", out, "--scope")
    meta = _scope_loom(out)["meta"]
    assert {"name": "donor", "values": ["d1", "d2", "d3", "d4", "d5", "d6"]} in meta["annotations"]
    assert {"name": "n_genes"} in meta["metrics"]
    assert meta["regulonThresholds"] == [
        {"regulon": "A_(+)", "defaultThresholdValue": 0.5, "motifData": "a.png"},
        {"regulon": "B_(2g)", "defaultThresholdValue": 0.25, "motifData": "b.png"},
    ]


def _small_anndata(tmp_path: Path, n: int = 12, **obs):
    rng = np.random.default_rng(1)
    frame = {"leiden": pd.Categorical([str(i % 3) for i in range(n)]), **obs}
    obj = ad.AnnData(
        X=rng.poisson(1, (n, 4)).astype(np.float32),
        obs=pd.DataFrame(frame, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"G{j}" for j in range(4)]),
    )
    obj.obsm["X_umap"] = rng.normal(size=(n, 2))
    return obj


def test_regulons_that_meet_once_spaces_go_keep_one_field_in_both_tables(tmp_path):
    """`A B` and `A_B` were renamed table by table: with the columns in
    another order in Regulons, one field held one regulon's AUC and the
    other's targets."""
    obj = _small_anndata(tmp_path)
    obj.obsm["RegulonsAUC"] = pd.DataFrame({"A B": np.full(12, 0.25), "A_B": np.full(12, 0.75)},
                                           index=obj.obs_names)
    obj.varm["Regulons"] = pd.DataFrame({"A_B": [0, 0, 0, 1], "A B": [1, 1, 1, 0]}, index=obj.var_names)
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    _ok(source, "-o", out, "--scope")
    loom = _scope_loom(out)
    auc, targets = loom["cols"]["RegulonsAUC"], loom["rows"]["Regulons"]
    assert sorted(auc.dtype.names) == sorted(targets.dtype.names)
    for field in auc.dtype.names:
        spaced = auc[field][0] == 0.25
        assert list(targets[field]) == ([1, 1, 1, 0] if spaced else [0, 0, 0, 1]), field


def test_cluster_id_is_written_from_the_first_clustering_not_a_stale_column(tmp_path):
    """An obs column named ClusterID was written as it was, and SCope's own
    ClusterID left out: after the clustering was edited the two disagreed."""
    obj = _small_anndata(tmp_path, ClusterID=np.full(12, 7))
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    result = _ok(source, "-o", out, "--scope", "--clusterings", "leiden")
    assert "'ClusterID' is written afresh from the first clustering, 'leiden'" in _out(result)
    loom = _scope_loom(out)
    assert list(loom["cols"]["ClusterID"]) == [i % 3 for i in range(12)]
    assert "ClusterID" not in [m["name"] for m in loom["meta"]["metrics"]]


def test_a_text_column_with_too_many_values_is_no_annotation_and_is_not_read_whole(tmp_path, monkeypatch):
    """Annotation values were found by reading the whole column, however many
    values it had, to throw them away past 245."""
    import adata.elements.read as read
    from adata.conversion import features, scope

    n = 4000
    obj = _small_anndata(tmp_path, n=n, barcode=[f"b{i}" for i in range(n)],
                         sample=["s1", "s2"] * (n // 2))
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    _ok(source, "-o", out, "--scope")
    names = [a["name"] for a in _scope_loom(out)["meta"]["annotations"]]
    assert "barcode" not in names and "sample" in names

    monkeypatch.setattr(features, "STRING_STEP", 10)
    real, rows = read.read_str_chunk, []

    def counted(obj, lo, hi, *args, **kwargs):
        rows.append(hi - lo)
        return real(obj, lo, hi, *args, **kwargs)

    monkeypatch.setattr(read, "read_str_chunk", counted)
    with h5py.File(source) as f:
        assert scope._distinct_text(f["obs/barcode"], f["obs"], scope.MAX_ANNOTATION_VALUES) is None
        assert sum(rows) == 250, "stops at the first block past 245 values"
        assert scope._distinct_text(f["obs/sample"], f["obs"], scope.MAX_ANNOTATION_VALUES) == ["s1", "s2"]


def test_marker_fields_are_renumbered_with_their_clusters(tmp_path):
    """Clusters numbered 2, 5 and 9 were renumbered 0, 1 and 2 in MetaData,
    but the marker tables kept fields 2, 5 and 9: SCope matched no marker
    to any cluster."""
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=False)
    meta = json.loads(json.dumps(truth.metadata))
    ids = {0: 2, 1: 5, 2: 9}
    for cluster in meta["clusterings"][0]["clusters"]:
        cluster["id"] = ids[cluster["id"]]
    meta["clusterings"][0]["clusterMarkerMetrics"] = [{"accessor": "avg_logFC", "name": "logFC", "description": ""}]
    _set_metadata(truth.path, json.dumps(meta))
    with h5py.File(truth.path, "a") as f:
        cols, rows = f["col_attrs"], f["row_attrs"]
        clusterings = cols["Clusterings"][...]
        clusterings["0"] = np.vectorize(ids.get)(clusterings["0"])
        del cols["Clusterings"]
        cols.create_dataset("Clusterings", data=clusterings)
        marks = np.zeros(10, dtype=[("2", "i8"), ("5", "i8"), ("9", "i8")])
        logfc = np.zeros(10, dtype=[("2", "f4"), ("5", "f4"), ("9", "f4")])
        for k, field in enumerate(("2", "5", "9")):
            marks[field][k] = 1
            logfc[field][k] = k + 0.5
        rows.create_dataset("ClusterMarkers_0", data=marks)
        rows.create_dataset("ClusterMarkers_0_avg_logFC", data=logfc)
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    _ok(first, "-o", back, "--scope")
    loom = _scope_loom(back)
    clusters = loom["meta"]["clusterings"][0]["clusters"]
    assert [c["description"] for c in clusters] == ["B cell", "T cell", "NK"]
    marks, logfc = loom["rows"]["ClusterMarkers_0"], loom["rows"]["ClusterMarkers_0_avg_logFC"]
    assert marks.dtype.names == logfc.dtype.names == ("0", "1", "2")
    for k in range(3):
        assert list(np.flatnonzero(marks[str(k)])) == [k]
        assert logfc[str(k)][k] == k + 0.5


def test_a_clustering_named_with_clusterings_keeps_its_marker_tables(tmp_path):
    """Marker tables were kept only for a clustering an import recorded, so
    a plain AnnData's lost theirs."""
    obj = _small_anndata(tmp_path)
    marks = pd.DataFrame({str(k): [int(g == k) for g in range(4)] for k in range(3)}, index=obj.var_names)
    obj.varm["ClusterMarkers_0"] = marks
    obj.varm["ClusterMarkers_0_avg_logFC"] = marks.astype(np.float32) * 2
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    _ok(source, "-o", out, "--scope", "--clusterings", "leiden")
    loom = _scope_loom(out)
    assert loom["rows"]["ClusterMarkers_0"].dtype.names == ("0", "1", "2")
    assert "ClusterMarkers_0_avg_logFC" in loom["rows"]
    assert loom["meta"]["clusterings"][0]["clusterMarkerMetrics"][0]["accessor"] == "avg_logFC"


def test_a_categorical_with_too_many_categories_is_no_annotation_and_its_categories_are_not_read(
    tmp_path, monkeypatch
):
    import adata.elements.read as read

    n = 300
    obj = _small_anndata(tmp_path, n=n, barcode=pd.Categorical([f"b{i}" for i in range(n)]))
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    real, seen = read.read_categories, []

    def counted(column, *args, **kwargs):
        seen.append(column.name)
        return real(column, *args, **kwargs)

    monkeypatch.setattr(read, "read_categories", counted)
    from adata.conversion import scope

    with h5py.File(source) as f:
        found = scope.plan_export(f, obs_columns=["barcode", "leiden"], varm_key="varm", clusterings=None,
                                  default_embedding=None, warnings=[])
    assert [a["name"] for a in found.metadata["annotations"]] == ["leiden"]
    assert "/obs/barcode" not in seen


def test_markers_of_undescribed_clusters_are_kept_on_a_round_trip(tmp_path):
    """With no clusters in MetaData the import labels them by id, and the
    export, finding no descriptions to match, dropped every marker field."""
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=False)
    meta = json.loads(json.dumps(truth.metadata))
    meta["clusterings"][0]["clusters"] = []
    meta["clusterings"][0]["clusterMarkerMetrics"] = [{"accessor": "avg_logFC", "name": "logFC", "description": ""}]
    _set_metadata(truth.path, json.dumps(meta))
    with h5py.File(truth.path, "a") as f:
        logfc = np.zeros(10, dtype=[("0", "f4"), ("1", "f4"), ("2", "f4")])
        marks = np.zeros(10, dtype=[("0", "i8"), ("1", "i8"), ("2", "i8")])
        logfc["1"][4], marks["1"][4] = 2.5, 1
        f["row_attrs"].create_dataset("ClusterMarkers_0", data=marks)
        f["row_attrs"].create_dataset("ClusterMarkers_0_avg_logFC", data=logfc)
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    _ok(first, "-o", back, "--scope")
    loom = _scope_loom(back)
    assert loom["rows"]["ClusterMarkers_0_avg_logFC"].dtype.names == ("0", "1", "2")
    assert loom["rows"]["ClusterMarkers_0_avg_logFC"]["1"][4] == 2.5


@pytest.mark.parametrize("scope_flag", [True, False])
def test_two_elements_asking_for_one_attribute_name_keep_the_first_and_warn(tmp_path, scope_flag):
    """An obs column and an obsm entry of one name were both written to
    col_attrs: the second failed with the output half written."""
    obj = _small_anndata(tmp_path, scores=np.arange(12.0), Clusterings=pd.Categorical(["x"] * 12))
    obj.obsm["scores"] = pd.DataFrame({"A": np.ones(12)}, index=obj.obs_names) if scope_flag else np.ones((12, 3))
    obj.varm["Gene"] = np.ones((4, 3))
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    args = ["--scope", "--clusterings", "leiden"] if scope_flag else []
    result = _ok(source, "-o", out, *args)
    text = _out(result)
    assert "Not written: obsm/scores -- col_attrs/scores is obs column 'scores'." in text
    assert "Not written: varm/Gene -- row_attrs/Gene is the gene names." in text
    with h5py.File(out) as f:
        assert list(f["col_attrs/scores"][...]) == list(np.arange(12.0))
        if scope_flag:
            assert "Not written: obs column 'Clusterings' -- col_attrs/Clusterings is SCope's clusterings." in text
            assert f["col_attrs/Clusterings"].dtype.names == ("0",)


def test_an_edited_thresholds_table_wins_over_the_recorded_metadata(tmp_path):
    """The thresholds table was read only when the recorded MetaData had
    none, so editing it changed nothing in the export."""
    truth = write_scope_loom(tmp_path / "scope.loom")
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    with h5py.File(first, "a") as f:
        f["uns/scope/regulon_thresholds/defaultThresholdValue"][0] = 0.9
    _ok(first, "-o", back, "--scope")
    thresholds = _scope_loom(back)["meta"]["regulonThresholds"]
    assert thresholds[0]["defaultThresholdValue"] == 0.9
    assert [t["defaultThresholdValue"] for t in thresholds[1:]] == pytest.approx([0.2, 0.3])
    assert thresholds[0]["allThresholds"] == truth.metadata["regulonThresholds"][0]["allThresholds"]


def test_marker_metrics_list_the_tables_written(tmp_path):
    """clusterMarkerMetrics was copied as recorded, listing a metric whose
    table had gone and missing one added since."""
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=False)
    meta = json.loads(json.dumps(truth.metadata))
    meta["clusterings"][0]["clusterMarkerMetrics"] = [
        {"accessor": "avg_logFC", "name": "logFC", "description": "log fold change"},
        {"accessor": "pval", "name": "p", "description": ""},
    ]
    _set_metadata(truth.path, json.dumps(meta))
    fields = [("0", "f4"), ("1", "f4"), ("2", "f4")]
    with h5py.File(truth.path, "a") as f:
        rows = f["row_attrs"]
        rows.create_dataset("ClusterMarkers_0", data=np.zeros(10, dtype=[(n, "i8") for n, _ in fields]))
        rows.create_dataset("ClusterMarkers_0_avg_logFC", data=np.zeros(10, dtype=fields))
        rows.create_dataset("ClusterMarkers_0_score", data=np.zeros(10, dtype=fields))
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    _ok(first, "-o", back, "--scope")
    assert _scope_loom(back)["meta"]["clusterings"][0]["clusterMarkerMetrics"] == [
        {"accessor": "avg_logFC", "name": "logFC", "description": "log fold change"},
        {"accessor": "score", "name": "score", "description": ""},
    ]


def test_a_clustering_is_written_once_and_comes_back_under_its_own_name(tmp_path):
    """A clustering went out as Clusterings and as a plain attribute too; the
    loom read back gave the plain one the name, the clustering `leiden-1`,
    and each round trip added another."""
    obj = _small_anndata(tmp_path)
    source, out, back = tmp_path / "in.h5ad", tmp_path / "out.loom", tmp_path / "back.h5ad"
    obj.write_h5ad(source)
    _ok(source, "-o", out, "--scope", "--clusterings", "leiden")
    loom = _scope_loom(out)
    assert "leiden" not in loom["cols"]
    assert "leiden" not in [a["name"] for a in loom["meta"]["annotations"]]
    _ok(out, "-o", back)
    obs = _read(back).obs
    assert [c for c in obs.columns if c.startswith("leiden")] == ["leiden"]
    assert list(obs["leiden"].astype(str)) == list(obj.obs["leiden"].astype(str))


def test_markers_whose_table_lost_its_name_are_left_out_with_their_metrics(tmp_path):
    """A marker table that lost its name to a var column was left out, but
    clusterMarkerMetrics still listed it."""
    obj = _small_anndata(tmp_path)
    marks = pd.DataFrame({str(k): [int(g == k) for g in range(4)] for k in range(3)}, index=obj.var_names)
    obj.varm["ClusterMarkers_0"] = marks
    obj.varm["ClusterMarkers_0_avg_logFC"] = marks.astype(np.float32)
    obj.var["ClusterMarkers_0_avg_logFC"] = np.arange(4.0)
    source, out = tmp_path / "in.h5ad", tmp_path / "out.loom"
    obj.write_h5ad(source)
    result = _ok(source, "-o", out, "--scope", "--clusterings", "leiden")
    text = _out(result)
    assert ("Not written: varm/ClusterMarkers_0_avg_logFC -- row_attrs/ClusterMarkers_0_avg_logFC "
            "is var column 'ClusterMarkers_0_avg_logFC'.") in text
    assert "Not written: ClusterMarkers_0 -- without metrics" in text
    loom = _scope_loom(out)
    assert "ClusterMarkers_0" not in loom["rows"]
    assert loom["rows"]["ClusterMarkers_0_avg_logFC"].dtype.names is None, "the var column"
    assert "clusterMarkerMetrics" not in loom["meta"]["clusterings"][0]


def test_an_anndata_07_categorical_can_be_a_clustering(tmp_path, sample_legacy_v010_h5ad):
    """--clusterings took only the current categorical layout, and refused
    anndata 0.7's codes with a `categories` reference."""
    out = tmp_path / "out.loom"
    _ok(sample_legacy_v010_h5ad, "-o", out, "--scope", "--clusterings", "cell_type")
    loom = _scope_loom(out)
    assert list(loom["cols"]["Clusterings"]["0"]) == [0, 1, 0, 2]
    assert [c["description"] for c in loom["meta"]["clusterings"][0]["clusters"]] == ["TypeA", "TypeB", "TypeC"]


def test_an_anndata_07_categorical_with_too_many_categories_is_not_read(tmp_path, sample_legacy_v010_h5ad):
    """The count of a legacy categorical's categories was checked only for
    the current layout: anndata 0.7's were read whole to be thrown away."""
    import adata.elements.read as read
    from adata.conversion import scope

    with h5py.File(sample_legacy_v010_h5ad, "a") as f:
        obs = f["obs"]
        cats = obs["__categories"].create_dataset("barcode", data=np.array([f"b{i}" for i in range(300)], dtype="S"))
        codes = obs.create_dataset("barcode", data=np.array([0, 1, 2, 3], dtype=np.int16))
        codes.attrs["categories"] = cats.ref
    real, seen = read.read_categories, []

    def counted(column, *args, **kwargs):
        seen.append(column.name)
        return real(column, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(read, "read_categories", counted)
        with h5py.File(sample_legacy_v010_h5ad) as f:
            found = scope.plan_export(f, obs_columns=["cell_type", "barcode"], varm_key="varm",
                                      clusterings=None, default_embedding=None, warnings=[])
    assert [a["name"] for a in found.metadata["annotations"]] == ["cell_type"]
    assert "/obs/barcode" not in seen


def test_a_missing_threshold_is_left_out_of_strict_json(tmp_path):
    """A missing defaultThresholdValue went into MetaData as NaN, which no
    strict JSON reader (SCope's client) parses."""
    truth = write_scope_loom(tmp_path / "scope.loom")
    first, back = tmp_path / "a.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    with h5py.File(first, "a") as f:
        f["uns/scope/regulon_thresholds/defaultThresholdValue"][0] = np.nan
    _ok(first, "-o", back, "--scope")

    def refuse(token):
        raise ValueError(f"not JSON: {token}")

    with h5py.File(back) as f:
        meta = json.loads(f["attrs/MetaData"][()], parse_constant=refuse)
    assert "defaultThresholdValue" not in meta["regulonThresholds"][0]
    assert meta["regulonThresholds"][1]["defaultThresholdValue"] == pytest.approx(0.2)


def test_an_auc_table_that_loses_its_name_takes_its_regulon_metadata_with_it(tmp_path):
    """An obs column named RegulonsAUC kept the name, and the AUC table was
    left out, but its regulon thresholds stayed in MetaData."""
    truth = write_scope_loom(tmp_path / "scope.loom")
    first, edited, back = tmp_path / "a.h5ad", tmp_path / "b.h5ad", tmp_path / "back.loom"
    _ok(truth.path, "-o", first)
    obj = _read(first)
    obj.obs["RegulonsAUC"] = np.arange(obj.n_obs, dtype=float)
    obj.write_h5ad(edited)
    result = _ok(edited, "-o", back, "--scope")
    assert "Not written: obsm/RegulonsAUC -- col_attrs/RegulonsAUC is obs column 'RegulonsAUC'." in _out(result)
    loom = _scope_loom(back)
    assert loom["cols"]["RegulonsAUC"].dtype.names is None
    assert loom["meta"]["regulonThresholds"] == []

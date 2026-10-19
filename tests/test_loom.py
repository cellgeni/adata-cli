"""Tests for `adata convert` to and from loom.

Fixtures are written with h5py as loompy 3 and velocyto write looms (see
conftest). Import is checked against what each fixture was built from, and
against `anndata.read_loom` in test_external_readers.py; export by reading
the file back, here and with loompy there.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_loom

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

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
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _dense(m) -> np.ndarray:
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


# ---------------------------------------------------------------------------
# import


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("spec", ["3.0.0", "2.0.1"])
def test_a_loom_imports_with_its_attributes_graphs_and_globals(tmp_path, spec, suffix):
    truth = write_loom(tmp_path / "in.loom", spec=spec)
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out)
    assert f"loom (spec {spec})" in _out(result)
    assert "X = matrix, layers: counts" in _out(result)

    got = _read(out)
    assert sparse.issparse(got.X) and got.X.format == "csr"
    assert got.X.dtype == np.float32
    assert np.array_equal(got.X.toarray(), truth.counts)
    assert got.layers["counts"].dtype == np.int32, "dtypes are kept"
    assert np.array_equal(got.layers["counts"].toarray(), truth.layers["counts"])

    assert list(got.obs_names) == truth.cell_ids
    assert list(got.var_names) == truth.gene_ids, "named by gene id"
    assert list(got.var["Gene"]) == truth.gene_names, "XML references unescaped"
    assert got.obs["Tissue"].dtype == "category"
    assert got.obs["Passed"].dtype == np.uint8, "loompy's booleans stay uint8"
    assert got.obs["Clusters"].tolist() == (np.arange(30) % 3).tolist()
    assert got.obsm["embedding"].shape == (30, 2)

    knn = got.obsp["knn"]
    assert knn.shape == (30, 30) and knn.nnz == 60 and knn.has_sorted_indices
    assert knn[0, 1] == pytest.approx(0.1)

    loom = got.uns["loom"]
    assert loom["spec"] == spec
    assert loom["obs_names"] == "CellID" and loom["var_names"] == "Accession"
    assert loom["attrs"]["title"] == "Démo"


def test_a_velocyto_loom_makes_x_spliced_and_keeps_the_main_matrix(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    out = tmp_path / "velo.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "loom (velocyto)" in _out(result)
    got = _read(out)
    assert got.X.dtype == np.uint32
    assert np.array_equal(got.X.toarray(), truth.layers["spliced"])
    assert np.array_equal(got.layers["matrix"].toarray(), truth.counts)
    for name in ("spliced", "unspliced", "ambiguous"):
        assert np.array_equal(got.layers[name].toarray(), truth.layers[name])


def test_layer_picks_x(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    out = tmp_path / "u.h5ad"
    _ok(truth.path, "-o", out, "--layer", "unspliced")
    got = _read(out)
    assert np.array_equal(got.X.toarray(), truth.layers["unspliced"])
    assert "matrix" in got.layers

    out = tmp_path / "m.h5ad"
    _ok(truth.path, "-o", out, "--layer", "matrix")
    assert np.array_equal(_read(out).X.toarray(), truth.counts)


def test_dense_keeps_the_matrices_dense(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True, n_cells=150)
    out = tmp_path / "dense.h5ad"
    _ok(truth.path, "-o", out, "--dense")
    got = _read(out)
    assert isinstance(got.X, np.ndarray)
    assert np.array_equal(got.X, truth.layers["spliced"])
    assert np.array_equal(got.layers["matrix"], truth.counts)


def test_slabs_smaller_than_the_file_read_the_same(tmp_path, monkeypatch):
    """Cells are read a slab at a time; with tiny slabs every slab boundary
    falls inside the fixture."""
    from adata.conversion import loom

    monkeypatch.setattr(loom, "SLAB_BYTES", 1)
    truth = write_loom(tmp_path / "in.loom", n_cells=200)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    assert np.array_equal(_read(out).X.toarray(), truth.counts)


def test_var_names_can_be_the_symbols_or_any_attribute(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    out = tmp_path / "sym.h5ad"
    result = _ok(truth.path, "-o", out, "--var-names", "gene_symbols")
    assert "var names repeat an earlier one" in _out(result)
    got = _read(out)
    assert list(got.var_names) == truth.gene_names
    assert list(got.var["Accession"]) == truth.gene_ids

    out = tmp_path / "unique.h5ad"
    _ok(truth.path, "-o", out, "--var-names", "gene_symbols", "--make-unique")
    names = list(_read(out).var_names)
    assert len(set(names)) == len(names)

    out = tmp_path / "attr.h5ad"
    _ok(truth.path, "-o", out, "--obs-names-attr", "Tissue", "--var-names-attr", "Start")
    got = _read(out)
    assert got.obs_names[1] == "hippocampus"
    assert list(got.var_names) == [str(i * 100) for i in range(10)]


def test_a_loom_without_names_is_numbered_as_anndata_numbers_it(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    with h5py.File(truth.path, "a") as f:
        del f["col_attrs/CellID"], f["row_attrs/Accession"], f["row_attrs/Gene"]
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "they are numbered 0..29" in _out(result)
    got = _read(out)
    assert list(got.obs_names) == [str(i) for i in range(30)]
    assert list(got.var_names) == [str(i) for i in range(10)]


def test_scope_compound_attributes_are_skipped_with_a_pointer(tmp_path):
    truth = write_loom(tmp_path / "scope.loom", compound=True)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "Not read: col_attrs RegulonsAUC" in _out(result)
    assert "#27" in _out(result)
    assert "RegulonsAUC" not in _read(out).obs


@pytest.mark.parametrize(
    "spoil,extra,message",
    [
        ("layer", [], "Layer 'bad' is not a 10 x 30 matrix"),
        ("attr", [], "Column attribute 'short' has 3 entries, but the matrix has 30 cells"),
        (None, ["--layer", "nope"], "--layer 'nope': no such layer; there are matrix (the main one), counts"),
        (None, ["--var-names-attr", "nope"], "--var-names-attr 'nope': no such attribute"),
    ],
)
def test_a_loom_that_cannot_be_read_is_refused_before_writing(tmp_path, spoil, extra, message):
    truth = write_loom(tmp_path / "in.loom")
    with h5py.File(truth.path, "a") as f:
        if spoil == "layer":
            f["layers"].create_dataset("bad", data=np.zeros((3, 3)))
        elif spoil == "attr":
            f["col_attrs"].create_dataset("short", data=np.zeros(3))
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out, *extra)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


def test_loom_flags_are_refused_for_other_formats(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    store = tmp_path / "s.h5ad"
    _ok(truth.path, "-o", store)
    result = _convert(store, "-o", tmp_path / "o.zarr", "--dense")
    assert result.exit_code == 1
    assert "does not apply" in _out(result)


# ---------------------------------------------------------------------------
# export


def _anndata_source(tmp_path: Path, layout: str):
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.6, (130, 9)).astype(np.int32)
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](counts)
    obj = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(["a", "b", "c"] * 43 + ["a"]), "score": rng.normal(size=130),
             "flag": np.arange(130) % 2 == 0},
            index=[f"c{i}" for i in range(130)],
        ),
        var=pd.DataFrame({"gene_symbols": [f"Gène{i}" for i in range(9)]}, index=[f"ENSG{i}" for i in range(9)]),
        layers={"spliced": sparse.csr_matrix(counts * 2)},
        obsm={"X_umap": rng.normal(size=(130, 2))},
        obsp={"connectivities": sparse.csr_matrix(np.eye(130, dtype=np.float32))},
        uns={"title": "tést", "n": 3},
    )
    path = tmp_path / f"{layout}.h5ad"
    obj.write_h5ad(path)
    return path, counts


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_export_writes_a_loom_as_loompy_does(tmp_path, layout, monkeypatch):
    from adata.conversion import loom

    monkeypatch.setattr(loom, "SLAB_BYTES", 1)  # a slab per 64 cells: three
    source, counts = _anndata_source(tmp_path, layout)
    out = tmp_path / "out.loom"
    result = _ok(source, "-o", out)
    assert "dense" in _out(result) and "before compression" in _out(result)
    with h5py.File(out, "r") as f:
        assert f["matrix"].shape == (9, 130)
        assert f["matrix"].chunks == (9, 64) and f["matrix"].compression == "gzip"
        assert np.array_equal(f["matrix"][...], counts.T)
        assert np.array_equal(f["layers/spliced"][...], counts.T * 2)
        assert f["attrs/LOOM_SPEC_VERSION"][()] == b"3.0.0"
        assert f["col_attrs/CellID"][0] == b"c0"
        assert f["row_attrs/Accession"][0] == b"ENSG0"
        assert f["row_attrs/Gene"][0] == b"G&#232;ne0", "non-ASCII as loompy escapes it"
        assert f["col_attrs/flag"].dtype == np.uint8
        assert f["col_attrs/cluster"][1] == b"b"
        assert f["col_attrs/X_umap"].shape == (130, 2)
        assert f["col_graphs/connectivities/a"].shape == (130,)
        assert f["attrs/title"][()] == b"t&#233;st"


def test_a_round_trip_keeps_what_a_loom_holds(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    h5ad, back, again = tmp_path / "a.h5ad", tmp_path / "back.loom", tmp_path / "b.h5ad"
    _ok(truth.path, "-o", h5ad)
    _ok(h5ad, "-o", back)
    _ok(back, "-o", again)
    first, second = _read(h5ad), _read(again)
    assert np.array_equal(first.X.toarray(), second.X.toarray())
    assert list(first.obs_names) == list(second.obs_names)
    assert list(first.var_names) == list(second.var_names)
    for name in ("spliced", "unspliced", "ambiguous"):
        assert np.array_equal(first.layers[name].toarray(), second.layers[name].toarray())
    assert list(first.var["Gene"]) == list(second.var["Gene"])
    assert first.obsp["knn"].nnz == second.obsp["knn"].nnz
    assert second.uns["loom"]["attrs"]["velocyto.__version__"] == "0.17.17"


def test_a_large_export_warns_but_writes(tmp_path, monkeypatch):
    from adata.conversion import loom

    monkeypatch.setattr(loom, "LARGE_DENSE", 10)
    source, _ = _anndata_source(tmp_path, "csr")
    result = _ok(source, "-o", tmp_path / "out.loom")
    assert "A loom holds its matrices dense" in _out(result)
    assert (tmp_path / "out.loom").exists()


def test_raw_x_exports_with_raw_var(tmp_path):
    """raw/X has raw/var's genes, which filtering var leaves behind; it was
    checked against var and refused, or written with var's names."""
    counts = np.arange(40, dtype=np.float32).reshape(8, 5)
    full = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame({"gene_symbols": list("ABCDE")}, index=[f"E{i}" for i in range(5)]),
    )
    full.raw = full
    filtered = full[:, :3].copy()
    filtered.layers["counts"] = filtered.X.copy()
    source = tmp_path / "f.h5ad"
    filtered.write_h5ad(source)

    out = tmp_path / "raw.loom"
    result = _ok(source, "-o", out, "--layer", "raw/X")
    assert "Not written: X and the layers" in _out(result)
    with h5py.File(out, "r") as f:
        assert f["matrix"].shape == (5, 8)
        assert np.array_equal(f["matrix"][...], counts.T)
        assert [v.decode() for v in f["row_attrs/Accession"][...]] == [f"E{i}" for i in range(5)]
        assert [v.decode() for v in f["row_attrs/Gene"][...]] == list("ABCDE")
        assert list(f["layers"]) == []


def test_explicit_gene_ids_are_the_accession_whatever_the_var_names(tmp_path):
    """With gene_symbols and gene_ids both columns, var_names were taken for
    the ids and the gene_ids column dropped."""
    obj = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(
            {"gene_symbols": ["S1", "S2", "S3"], "gene_ids": ["E1", "E2", "E3"]},
            index=["key1", "key2", "key3"],
        ),
    )
    source = tmp_path / "s.h5ad"
    obj.write_h5ad(source)
    out = tmp_path / "s.loom"
    _ok(source, "-o", out)
    with h5py.File(out, "r") as f:
        assert [v.decode() for v in f["row_attrs/Accession"][...]] == ["E1", "E2", "E3"]
        assert [v.decode() for v in f["row_attrs/Gene"][...]] == ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------
# looms other tools write, as found on real files


def test_a_matrix_only_loom_is_read_by_its_name(tmp_path):
    """Baysor writes some looms as /matrix alone, with no attribute groups;
    these were not recognised as looms at all."""
    path = tmp_path / "ncvs.loom"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)  # genes x cells
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix", data=values)
    out = tmp_path / "out.h5ad"
    result = _ok(path, "-o", out)
    assert "loom (spec pre-2) → h5ad, 4 cells x 3 genes" in _out(result)
    got = _read(out)
    assert np.array_equal(got.X.toarray(), values.T)
    assert list(got.obs_names) == ["0", "1", "2", "3"]

    renamed = tmp_path / "ncvs.h5"
    path.rename(renamed)
    result = _convert(renamed, "-o", tmp_path / "x.h5ad")
    assert result.exit_code == 1, "without the name, a bare matrix is no evidence"


def test_an_r_loomexperiment_file_reads_its_version_and_float_graphs(tmp_path):
    """R's LoomExperiment writes LOOM_SPEC_VERSION as a one-element array
    (shown as "['2.0.1']") and graph vertices as floats."""
    truth = write_loom(tmp_path / "r.loom", spec="2.0.1")
    with h5py.File(truth.path, "a") as f:
        f.attrs["LOOM_SPEC_VERSION"] = np.array([b"2.0.1"])
        for key in ("a", "b"):
            values = f["col_graphs/knn"][key][...].astype(np.float64)
            del f["col_graphs/knn"][key]
            f["col_graphs/knn"].create_dataset(key, data=values)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "loom (spec 2.0.1)" in _out(result)
    assert _read(out).obsp["knn"].nnz == 60

    with h5py.File(truth.path, "a") as f:
        values = f["col_graphs/knn/a"][...]
        values[0] += 0.5
        f["col_graphs/knn/a"][...] = values
    result = _convert(truth.path, "-o", tmp_path / "bad.h5ad")
    assert result.exit_code == 1
    assert "not a whole number" in _out(result)


def test_baysor_names_genes_by_name(tmp_path):
    truth = write_loom(tmp_path / "b.loom")
    with h5py.File(truth.path, "a") as f:
        names = f["row_attrs/Gene"][...]
        del f["row_attrs/Gene"], f["row_attrs/Accession"]
        f["row_attrs"].create_dataset("Name", data=names)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "genes are named by 'Name'" in _out(result)
    assert list(_read(out).var_names) == truth.gene_names

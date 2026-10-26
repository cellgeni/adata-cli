"""Tests for `adata convert` of pycisTopic pickles.

Fixtures are pickled as pycisTopic pickles its objects (see conftest), with
stand-ins for its classes under its module names; the converter reads them
without pycisTopic, through an allowlist. Checked against the objects the
fixtures were built from; the real GEO pickle, written with pycisTopic's
pandas 1.5, is checked the same way in the release verification.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_cistopic_pickle, write_imputed_pickle

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


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_cistopic_object_reads_with_its_model_and_metadata(tmp_path, suffix):
    truth = write_cistopic_pickle(tmp_path / "cistopic_obj.pkl")
    obj = truth.obj
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out)
    assert "pycisTopic CistopicObject → " in _out(result)
    assert "20 cells x 30 regions, 4 topics, X_UMAP" in _out(result)

    got = _read(out)
    assert sparse.issparse(got.X) and got.X.format == "csr" and got.X.dtype == np.int32
    assert np.array_equal(got.X.toarray(), obj.fragment_matrix.T.toarray())
    assert np.array_equal(got.layers["binary"].toarray(), obj.binary_matrix.T.toarray())
    assert list(got.obs_names) == obj.cell_names and list(got.var_names) == obj.region_names

    cell_data = obj.cell_data.loc[obj.cell_names]
    assert got.obs["cisTopic_nr_frag"].dtype == np.int64, "numbers held as object are numbers"
    assert got.obs["cisTopic_nr_frag"].tolist() == cell_data["cisTopic_nr_frag"].tolist()
    assert list(got.obs["sample_id"].cat.categories) == ["sample0", "sample1"]
    assert got.obs["sample_id"].tolist() == cell_data["sample_id"].tolist()
    assert np.array_equal(got.obs["TSS_enrichment"], cell_data["TSS_enrichment"])
    assert str(got.obs["doublet"].dtype) == "boolean"
    assert got.obs["doublet"].isna().tolist() == cell_data["doublet"].isna().tolist()
    assert str(got.obs["n_peaks"].dtype) == "Int64" and pd.isna(got.obs["n_peaks"].iloc[-1]) == pd.isna(cell_data["n_peaks"].iloc[-1])
    assert got.obs["celltype"].dtype == "category", "repeated text is categorical, as anndata writes it"
    assert got.obs["note"].tolist() == cell_data["note"].tolist()
    assert got.var["Start"].dtype == np.int32 and got.var["Chromosome"].tolist()[:2] == ["chr1", "chr2"]

    model = obj.selected_model
    assert np.array_equal(got.obsm["X_topic"], model.cell_topic[obj.cell_names].to_numpy().T)
    assert np.array_equal(got.varm["topic_region"], model.region_topic.to_numpy())
    assert np.array_equal(got.obsm["X_UMAP"], obj.projections["cell"]["UMAP"].loc[obj.cell_names].to_numpy())

    meta = got.uns["pycistopic"]
    assert meta["project"] == "demo" and meta["kind"] == "CistopicObject"
    assert meta["path_to_fragments"]["sample1"] == "/data/s1/fragments.tsv.gz"
    assert list(meta["topics"]) == ["Topic1", "Topic2", "Topic3", "Topic4"]
    assert meta["projections"]["X_UMAP"] == "UMAP"
    assert meta["model"]["n_topic"] == 4
    assert meta["model"]["metrics"].loc["Metric", "Mimno_2011"] == pytest.approx(-1.3)
    assert meta["model"]["parameters"].loc["package", "Parameter"] == "lda"


def test_a_gzipped_pickle_without_a_model_reads(tmp_path):
    truth = write_cistopic_pickle(tmp_path / "obj.pkl.gz", gz=True, model=False, projections=False)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "topics" not in _out(result)
    got = _read(out)
    assert np.array_equal(got.X.toarray(), truth.obj.fragment_matrix.T.toarray())
    assert "X_topic" not in got.obsm and "model" not in got.uns["pycistopic"]


def test_harmony_topics_are_kept_too(tmp_path):
    truth = write_cistopic_pickle(tmp_path / "obj.pkl", harmony=True)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    harmony = truth.obj.selected_model.cell_topic_harmony
    assert np.array_equal(got.obsm["X_topic_harmony"], harmony[truth.obj.cell_names].to_numpy().T)


@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
def test_an_imputed_features_pickle_is_cells_by_features(tmp_path, dense, monkeypatch):
    from adata.conversion import cistopic

    monkeypatch.setattr(cistopic, "DENSE_CELLS", 4)  # blocks straddle the cells
    truth = write_imputed_pickle(tmp_path / "imputed_acc_obj.pkl")
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out, *(["--dense"] if dense else []))
    assert "pycisTopic CistopicImputedFeatures → h5ad, 15 cells x 8 features" in _out(result)
    got = _read(out)
    assert isinstance(got.X, np.ndarray) == dense
    X = got.X if dense else got.X.toarray()
    assert X.dtype == np.float32 and np.array_equal(X, truth.obj.mtx.T)
    assert list(got.var_names) == truth.obj.feature_names
    assert list(got.obs_names) == truth.obj.cell_names


def test_a_pickle_that_asks_for_code_is_refused_unrun(tmp_path):
    """The point of the allowlist: a global that a pycisTopic object never
    holds is refused by name, and never called."""
    marker = tmp_path / "ran"

    class Payload:
        def __reduce__(self):
            import os

            return (os.system, (f"touch {marker}",))

    truth = write_cistopic_pickle(tmp_path / "obj.pkl")
    truth.obj.project = Payload()
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out)
    assert result.exit_code == 1
    assert re.search(r"asks for (posix|os|nt)\.system, which a pycisTopic object never holds", _out(result))
    assert not marker.exists(), "the payload ran"
    assert not out.exists()


def test_a_pickle_that_is_not_pycistopics_needs_from(tmp_path):
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps({"a": [1, 2, 3]}))
    result = _convert(path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "Cannot tell what format" in _out(result)

    from adata.formats.pickled import sniff

    assert not sniff(tmp_path / "missing.pkl")
    (tmp_path / "junk.pkl").write_bytes(b"not a pickle at all")
    assert not sniff(tmp_path / "junk.pkl")


def test_without_the_extra_the_install_is_named(tmp_path, monkeypatch):
    import builtins

    truth = write_cistopic_pickle(tmp_path / "obj.pkl")
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    result = _convert(truth.path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "pip install 'pyadata-cli[pickle]'" in _out(result)


def test_a_mismatched_object_is_refused_before_writing(tmp_path):
    truth = write_cistopic_pickle(tmp_path / "obj.pkl")
    truth.obj.cell_names = truth.obj.cell_names[:-1]
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out)
    assert result.exit_code == 1
    assert "fragment_matrix is 30 x 20, but the object names 30 regions and 19 cells" in _out(result)
    assert not out.exists()

    result = _convert(write_cistopic_pickle(tmp_path / "ok.pkl").path, "-o", out, "--dense")
    assert result.exit_code == 1 and "--dense applies to an imputed-features pickle" in _out(result)


def test_a_models_pickle_is_named_and_refused(tmp_path):
    """pycisTopic also pickles its models alone; they hold no matrix."""
    truth = write_cistopic_pickle(tmp_path / "obj.pkl")
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(truth.obj.selected_model, protocol=2))  # GLOBAL opcodes
    result = _convert(path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "holds a CistopicLDAModel; convert reads pickles of a CistopicObject" in _out(result)


def test_a_csc_fragment_matrix_reads_the_same(tmp_path):
    truth = write_cistopic_pickle(tmp_path / "obj.pkl", model=False, projections=False)
    truth.obj.fragment_matrix = truth.obj.fragment_matrix.tocsc()
    truth.obj.binary_matrix = truth.obj.binary_matrix.tocoo()
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out, "--compression", "lzf")
    got = _read(out)
    assert np.array_equal(got.X.toarray(), truth.obj.fragment_matrix.T.toarray())
    assert np.array_equal(got.layers["binary"].toarray(), truth.obj.binary_matrix.T.toarray())

    result = _convert(truth.path, "-o", tmp_path / "x.zarr", "--compression", "lzf")
    assert result.exit_code == 1 and "not available for zarr" in _out(result)


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_csc_matrix_is_compressed_as_asked(tmp_path, suffix):
    """A CSC fragment matrix took a shortcut that wrote its arrays without
    the compression asked for."""
    import h5py
    import zarr

    truth = write_cistopic_pickle(tmp_path / "obj.pkl", model=False, projections=False)
    truth.obj.fragment_matrix = truth.obj.fragment_matrix.tocsc()
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    out = tmp_path / f"out{suffix}"
    _ok(truth.path, "-o", out, "--compression", "gzip")
    if suffix == ".h5ad":
        with h5py.File(out, "r") as f:
            assert f["X/data"].compression == "gzip"
    else:
        codecs = zarr.open_group(out, mode="r")["X/data"].compressors
        assert any("gzip" in type(c).__name__.lower() for c in codecs)
    assert np.array_equal(_read(out).X.toarray(), truth.obj.fragment_matrix.T.toarray())


def test_repeated_column_labels_keep_the_first_with_a_warning(tmp_path):
    """A frame with a repeated column label made the column lookup return a
    frame, and the conversion failed."""
    truth = write_cistopic_pickle(tmp_path / "obj.pkl", model=False, projections=False)
    cell_data = truth.obj.cell_data
    # As pycisTopic's pandas would: object strings, not pandas 3's Arrow ones.
    with pd.option_context("future.infer_string", False):
        truth.obj.cell_data = pd.concat([cell_data, cell_data[["TSS_enrichment"]] * 2], axis=1)
        truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "Not written: obs column 'TSS_enrichment', a name a dataframe reserves or repeats" in _out(result)
    got = _read(out)
    expected = cell_data.loc[truth.obj.cell_names, "TSS_enrichment"]
    assert np.array_equal(got.obs["TSS_enrichment"], expected), "the first one is kept"


@pytest.mark.parametrize("protocol", [0, 1])
def test_old_protocols_are_recognised_and_read(tmp_path, protocol):
    """Protocols 0 and 1 name copy_reg._reconstructor before the object's
    class, and the sniff stopped at it: such a .pkl was not recognised."""
    truth = write_cistopic_pickle(tmp_path / "obj.pkl", model=False, projections=False)
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=protocol))
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    assert np.array_equal(_read(out).X.toarray(), truth.obj.fragment_matrix.T.toarray())

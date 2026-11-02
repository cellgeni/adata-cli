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


@pytest.mark.parametrize("budget", [20, 5], ids=["cells", "features"])
@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr2", ".zarr3"])
def test_an_imputed_features_pickle_is_cells_by_features(tmp_path, dense, suffix, budget, monkeypatch):
    """Blocks of a few cells (8 features to a cell fit `cells`), or runs of
    one cell's features (they do not fit `features`), straddling the edges."""
    from adata.conversion import cistopic

    monkeypatch.setattr(cistopic, "BLOCK_ENTRIES", budget, raising=False)
    truth = write_imputed_pickle(tmp_path / "imputed_acc_obj.pkl")
    out = tmp_path / f"out{suffix[:5]}"
    zarr = ["--zarr-format", suffix[-1]] if suffix != ".h5ad" else []
    result = _ok(truth.path, "-o", out, *zarr, *(["--dense"] if dense else []))
    assert f"pycisTopic CistopicImputedFeatures → {suffix[1:5]}, 15 cells x 8 features" in _out(result)
    got = _read(out)
    assert isinstance(got.X, np.ndarray) == dense
    X = got.X if dense else got.X.toarray()
    assert X.dtype == np.float32 and np.array_equal(X, truth.obj.mtx.T)
    assert list(got.var_names) == truth.obj.feature_names
    assert list(got.obs_names) == truth.obj.cell_names


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_dense_output_chunks_are_bounded_in_bytes_however_wide_the_matrix(tmp_path, suffix, monkeypatch):
    """F007: `--dense` chunked X as (4096 cells, every feature), so a chunk
    grew with the features: 3.2 GB at 200,000 float32 ones, and past
    262,144 HDF5 refused it (a chunk must be under 4 GB). A chunk now holds
    a fixed number of entries, a run of one cell's features when even one
    cell is wider than that."""
    import h5py
    import zarr

    from adata.conversion import cistopic

    monkeypatch.setattr(cistopic, "BLOCK_ENTRIES", 64, raising=False)
    for n_features, expected in ((16, (4, 16)), (1000, (1, 64))):
        truth = write_imputed_pickle(tmp_path / f"wide{n_features}.pkl", n_cells=10, n_features=n_features)
        out = tmp_path / f"wide{n_features}{suffix}"
        _ok(truth.path, "-o", out, "--dense")
        if suffix == ".h5ad":
            with h5py.File(out, "r") as f:
                chunks = f["X"].chunks
        else:
            chunks = zarr.open_group(out, mode="r")["X"].chunks
        assert chunks == expected, chunks
        assert np.array_equal(_read(out).X, truth.obj.mtx.T)


def _sparse_imputed(tmp_path, fmt):
    truth = write_imputed_pickle(tmp_path / f"gene_act_{fmt}.pkl", n_cells=15, n_features=8)
    truth.obj.mtx = getattr(sparse, f"{fmt}_matrix")(truth.obj.mtx)
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    return truth


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr2", ".zarr3"])
def test_dense_is_honoured_for_a_sparse_imputed_matrix(tmp_path, fmt, suffix, monkeypatch):
    """F100: gene activity is pickled with a sparse `mtx`, and `--dense` was
    then ignored: the summary said dense and X came out CSR. It is written
    dense, a block of cells at a time from the CSR the transpose makes."""
    from adata.conversion import cistopic

    monkeypatch.setattr(cistopic, "BLOCK_ENTRIES", 20, raising=False)
    truth = _sparse_imputed(tmp_path, fmt)
    out = tmp_path / f"out{suffix[:5]}"
    zarr = ["--zarr-format", suffix[-1]] if suffix != ".h5ad" else []
    result = _ok(truth.path, "-o", out, "--dense", *zarr)
    assert ", dense" in _out(result)
    got = _read(out)
    assert isinstance(got.X, np.ndarray), type(got.X)
    assert got.X.dtype == np.float32 and np.array_equal(got.X, truth.obj.mtx.T.toarray())
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([truth.path.name, out.name])
    if suffix == ".h5ad":
        import h5py

        with h5py.File(out, "r") as f:
            assert not [k for k in f if k.startswith("__cistopic")], "scratch left behind"
    else:
        import zarr

        assert not [k for k in zarr.open_group(out, mode="r") if k.startswith("__cistopic")]


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_a_sparse_imputed_matrix_is_csr_without_dense(tmp_path, fmt):
    truth = _sparse_imputed(tmp_path, fmt)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert got.X.format == "csr" and np.array_equal(got.X.toarray(), truth.obj.mtx.T.toarray())


def test_an_imputed_matrix_that_is_not_numbers_is_refused(tmp_path):
    truth = write_imputed_pickle(tmp_path / "imputed.pkl", n_cells=3, n_features=2)
    truth.obj.mtx = truth.obj.mtx.astype(object)
    truth.path.write_bytes(pickle.dumps(truth.obj, protocol=4))
    result = _convert(truth.path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "its matrix holds object values, not numbers" in _out(result)


@pytest.mark.perf
@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
def test_writing_an_imputed_matrix_holds_a_block_not_every_feature_of_4096_cells(tmp_path, dense, monkeypatch):
    """F025 (and F007's copy): each block was 4096 cells x every feature,
    and CSR took about 9x its bytes in `np.nonzero`'s int64 coordinates: 3
    GB for 4096 x 20,000, about 50 GB at 500,000 regions, where loading the
    pickle took 360 MB. A block now holds a fixed number of entries, so
    what writing allocates does not grow with the features."""
    from adata.conversion import cistopic
    from adata.conversion.base import Options
    from adata.storage import open_store
    from tests.perf_counters import FLAT_SIZES, assert_independent_of, count_allocations

    # 256 cells of 64 features are one block; of 4096, 64 blocks of 4 cells.
    monkeypatch.setattr(cistopic, "BLOCK_ENTRIES", 1 << 14, raising=False)

    def measure(n_features: int) -> int:
        truth = write_imputed_pickle(tmp_path / f"imputed{n_features}.pkl", n_cells=256, n_features=n_features)
        out = tmp_path / f"out{n_features}{int(dense)}.h5ad"
        plan = cistopic.CistopicImport().plan(
            truth.path, out, "cistopic", "h5ad", Options(dense=dense, given=frozenset())
        )
        with open_store(out, "w") as store:
            with count_allocations() as peak:
                cistopic._write_imputed(store.root, plan.detail)
        return peak[0]

    assert_independent_of(measure, what="bytes allocated writing X", axis="n_features", sizes=FLAT_SIZES)


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

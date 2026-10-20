"""Tests for `adata convert` of one dataset of a plain HDF5 file.

Fixtures are written with h5py as anndata's `read_hdf` reads them (see
conftest), and checked against what each was built from; `read_hdf` itself
is the reference in test_external_readers.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_10x_h5, write_hdf5_matrix

ad = pytest.importorskip("anndata")
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


def test_without_a_key_the_candidates_are_listed_and_nothing_written(tmp_path):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5")
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out)
    assert result.exit_code == 1
    text = _out(result)
    assert "name the dataset that becomes X with --key" in text
    assert "The 2-D numeric datasets: data/matrix (12 x 5, int32)" in text
    assert "col_names (5, |S5), row_names (12, |S7)" in text
    assert not out.exists()


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_dataset_reads_with_read_hdfs_names(tmp_path, suffix):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5")
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out, "--key", truth.key)
    assert "hdf5 data/matrix → " in _out(result) and "12 cells x 5 genes, int32" in _out(result)
    got = _read(out)
    assert sparse.issparse(got.X) and got.X.dtype == np.int32, "the dtype is kept"
    assert np.array_equal(got.X.toarray(), truth.counts)
    assert list(got.obs_names) == truth.cells
    assert list(got.var_names) == truth.genes
    assert got.uns["hdf5"]["key"] == "data/matrix"


@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
def test_transpose_reads_genes_as_rows(tmp_path, dense):
    truth = write_hdf5_matrix(tmp_path / "t.hdf5", transpose=True, n_cells=40)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out, "--key", truth.key, "--transpose", *(["--dense"] if dense else []))
    got = _read(out)
    assert isinstance(got.X, np.ndarray) == dense
    assert np.array_equal(_dense(got.X), truth.counts)
    assert list(got.obs_names) == truth.cells, "col_names name the cells, transposed"
    assert list(got.var_names) == truth.genes


def test_names_from_other_datasets(tmp_path):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5", names="custom")
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out, "--key", truth.key,
        "--obs-names-key", "names/cells", "--var-names-key", "names/genes")
    got = _read(out)
    assert list(got.obs_names) == truth.cells and list(got.var_names) == truth.genes

    result = _convert(truth.path, "-o", tmp_path / "bad.h5ad", "--key", truth.key,
                      "--obs-names-key", "names/genes")
    assert result.exit_code == 1
    assert "--obs-names-key names/genes: not a dataset of 12 names" in _out(result)


def test_unnamed_axes_are_numbered_with_a_warning(tmp_path):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5", names="none")
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out, "--key", truth.key)
    assert "Nothing names obs; it is numbered 0..11" in _out(result)
    got = _read(out)
    assert list(got.obs_names) == [str(i) for i in range(12)]

    with h5py.File(truth.path, "a") as f:
        f["row_names"] = np.array([b"a", b"b"])
    result = _ok(truth.path, "-o", tmp_path / "short.h5ad", "--key", truth.key)
    assert "row_names holds 2 names, but obs has 12 entries" in _out(result)


@pytest.mark.parametrize(
    "key,message",
    [
        ("nope", "holds nothing there"),
        ("row_names", "holds a 1-D |S7 dataset there"),
        ("data", "holds a group there"),
    ],
)
def test_a_key_that_is_no_matrix_is_refused(tmp_path, key, message):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5")
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out, "--key", key)
    assert result.exit_code == 1
    assert message in _out(result) and "data/matrix (12 x 5" in _out(result)
    assert not out.exists()


def test_slabs_follow_the_chunks(tmp_path, monkeypatch):
    """The dataset is read a slab of whole chunks at a time; with tiny slabs
    every boundary falls inside it, along rows and, transposed, columns."""
    from adata.conversion import hdf5

    monkeypatch.setattr(hdf5, "SLAB_BYTES", 1)
    truth = write_hdf5_matrix(tmp_path / "m.hdf5", n_cells=50, chunks=(7, 3))
    _ok(truth.path, "-o", tmp_path / "a.h5ad", "--key", truth.key)
    assert np.array_equal(_read(tmp_path / "a.h5ad").X.toarray(), truth.counts)
    truth = write_hdf5_matrix(tmp_path / "t.hdf5", n_cells=50, chunks=(3, 7), transpose=True)
    _ok(truth.path, "-o", tmp_path / "t.h5ad", "--key", truth.key, "--transpose", "--dense")
    assert np.array_equal(_read(tmp_path / "t.h5ad").X, truth.counts)


def test_a_plain_h5_needs_from_and_a_10x_one_is_still_10x(tmp_path):
    truth = write_hdf5_matrix(tmp_path / "plain.h5")
    result = _convert(truth.path, "-o", tmp_path / "x.h5ad", "--key", truth.key)
    assert result.exit_code == 1 and "pass --from hdf5" in _out(result)
    out = tmp_path / "plain.h5ad"
    _ok(truth.path, "-o", out, "--from", "hdf5", "--key", truth.key)
    assert np.array_equal(_read(out).X.toarray(), truth.counts)

    tenx = write_10x_h5(
        tmp_path / "filtered_feature_bc_matrix.h5", np.ones((3, 2), dtype=np.int32),
        barcodes=["a-1", "b-1", "c-1"], ids=["E1", "E2"], names=["G1", "G2"],
    )
    result = _ok(tenx, "-o", tmp_path / "tenx.h5ad")
    assert "hdf5" not in _out(result).split("→")[0]


def test_hdf5_flags_are_refused_for_other_formats(tmp_path):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5")
    store = tmp_path / "s.h5ad"
    _ok(truth.path, "-o", store, "--key", truth.key)
    result = _convert(store, "-o", tmp_path / "o.zarr", "--key", "X")
    assert result.exit_code == 1 and "does not apply" in _out(result)


def test_repeated_gene_names_warn_or_are_made_unique(tmp_path):
    truth = write_hdf5_matrix(tmp_path / "m.hdf5")
    with h5py.File(truth.path, "a") as f:
        f["col_names"][1] = b"Gene0"
    result = _ok(truth.path, "-o", tmp_path / "w.h5ad", "--key", truth.key)
    assert "1 var names repeat an earlier one" in _out(result)
    _ok(truth.path, "-o", tmp_path / "u.h5ad", "--key", truth.key, "--make-unique")
    assert list(_read(tmp_path / "u.h5ad").var_names)[:2] == ["Gene0", "Gene0-1"]

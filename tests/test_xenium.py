"""Tests for `adata convert` of Xenium output bundles.

Fixtures are written as Xenium Onboard Analysis writes its bundle (see
conftest): XOA 1's numbered cells and later versions' named ones, the cell
table in the matrix's order or shuffled, with cells the matrix lacks, and
analysis/ with gaps. What anndata reads back is checked against what each
fixture was built from.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_xenium_bundle

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
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _refused(*args) -> str:
    result = _convert(*args)
    assert result.exit_code == 1, _out(result)
    return _out(result)


def _check_cells(adata, truth) -> None:
    assert list(adata.obs_names) == truth.cell_ids
    for name, values in truth.columns.items():
        got = adata.obs[name]
        if name == "segmentation_method":
            assert isinstance(got.dtype, pd.CategoricalDtype)
            assert got.astype(str).tolist() == list(values)
        else:
            np.testing.assert_allclose(got.to_numpy(dtype=float), values.astype(float), err_msg=name)
    np.testing.assert_allclose(adata.obsm["spatial"], truth.spatial)
    np.testing.assert_array_equal(adata.X.toarray(), truth.counts)


# ---------------------------------------------------------------------------
# the mapping


@pytest.mark.parametrize("ids", ["int", "string"])
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_cells_centroids_and_analysis_land_where_spatialdata_io_puts_them(tmp_path, ids, suffix):
    truth = write_xenium_bundle(tmp_path / "xenium", ids=ids)
    out = tmp_path / f"out{suffix}"

    result = _ok(truth.bundle, "-o", out)

    assert "Xenium →" in _out(result)
    assert "analysis: clusters, X_umap" in _out(result)
    adata = _read(out)
    _check_cells(adata, truth)
    assert adata.obs["transcript_counts"].dtype == np.int64
    assert list(adata.obs.columns) == [*truth.columns, "graphclust"]
    assert isinstance(adata.obs["graphclust"].dtype, pd.CategoricalDtype)
    assert adata.obs["graphclust"].astype(str).tolist() == list(truth.clusters)
    np.testing.assert_allclose(adata.obsm["X_umap"], truth.umap)
    assert adata.var["feature_types"].tolist()[-1] == "Negative Control Probe"


def test_a_cell_table_in_another_order_is_written_in_the_matrix_s(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium", shuffled=True)
    _ok(truth.bundle, "-o", tmp_path / "out.h5ad")
    _check_cells(_read(tmp_path / "out.h5ad"), truth)


def test_cells_the_matrix_lacks_are_left_out_with_their_categories(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium", extra_cells=4, shuffled=True)

    result = _ok(truth.bundle, "-o", tmp_path / "out.h5ad")

    assert "4 cells of cells.csv.gz are not in the matrix; they are left out" in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    _check_cells(adata, truth)
    assert "Imported cell segmentation" not in adata.obs["segmentation_method"].cat.categories


def test_cells_missing_from_the_analysis_are_left_empty(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium", missing_in_analysis=3)

    result = _ok(truth.bundle, "-o", tmp_path / "out.h5ad")

    assert "3 cells have no clusters or X_umap in analysis/" in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    assert adata.obs["graphclust"].isna().sum() == 3
    assert adata.obs["graphclust"].astype(object).fillna("").tolist() == list(truth.clusters)
    np.testing.assert_allclose(adata.obsm["X_umap"], truth.umap)


def test_no_analysis_is_a_warning(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium", analysis=False)
    result = _ok(truth.bundle, "-o", tmp_path / "out.h5ad")
    assert "No analysis/ or analysis.tar.gz" in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    assert "graphclust" not in adata.obs and "X_umap" not in adata.obsm


def test_the_analysis_is_read_from_the_archive_xenium_ships_it_in(tmp_path):
    """Xenium packs analysis/ as analysis.tar.gz, so a bundle as downloaded
    has no analysis/ folder at all."""
    truth = write_xenium_bundle(tmp_path / "xenium", missing_in_analysis=2)
    shutil.make_archive(str(truth.bundle / "analysis"), "gztar", truth.bundle, "analysis")
    shutil.rmtree(truth.bundle / "analysis")

    result = _ok(truth.bundle, "-o", tmp_path / "out.h5ad")

    assert "analysis: clusters, X_umap" in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    assert adata.obs["graphclust"].astype(object).fillna("").tolist() == list(truth.clusters)
    np.testing.assert_allclose(adata.obsm["X_umap"], truth.umap)


def test_an_archive_cannot_write_outside_its_temporary_folder(tmp_path):
    """Members are copied to paths built from the names this wants, never
    extracted where the archive says."""
    import io
    import tarfile

    truth = write_xenium_bundle(tmp_path / "xenium", analysis=False)
    with tarfile.open(truth.bundle / "analysis.tar.gz", "w:gz") as tar:
        data = b"Barcode,Cluster\n"
        info = tarfile.TarInfo("../../escaped.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    _ok(truth.bundle, "-o", tmp_path / "out.h5ad")
    assert not list(tmp_path.rglob("escaped.csv"))


def test_feature_type_keeps_the_cells_and_their_positions(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium")
    _ok(truth.bundle, "-o", tmp_path / "out.h5ad", "--feature-type", "Gene Expression")
    adata = _read(tmp_path / "out.h5ad")
    assert adata.n_vars == 4
    np.testing.assert_allclose(adata.obsm["spatial"], truth.spatial)


def test_a_bundle_is_xenium_not_a_mtx_layout(tmp_path):
    """A Xenium bundle holds cell_feature_matrix/ as a MEX folder too, which
    the 10x-mtx sniffer could take for the folder to read."""
    truth = write_xenium_bundle(tmp_path / "xenium")
    mex = truth.bundle / "cell_feature_matrix"
    mex.mkdir()
    for file in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
        (mex / file).write_bytes(b"")
    assert "Xenium →" in _out(_ok(truth.bundle, "-o", tmp_path / "out.h5ad"))


# ---------------------------------------------------------------------------
# refusals, each before anything is written


def test_a_cell_missing_from_the_table_is_refused(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium")
    path = truth.bundle / "cells.csv.gz"
    import gzip

    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    path.write_bytes(gzip.compress(("\n".join(lines[:-1]) + "\n").encode()))

    message = _refused(truth.bundle, "-o", tmp_path / "out.h5ad")

    assert "1 of the matrix's 15 barcodes have no row in cells.csv.gz" in message
    assert not list(tmp_path.glob("out*"))


def test_a_bundle_with_only_parquet_cells_says_so(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium")
    (truth.bundle / "cells.csv.gz").rename(truth.bundle / "cells.parquet")
    assert "has cells.parquet only, which is not read yet" in _refused(truth.bundle, "-o", tmp_path / "out.h5ad")


@pytest.mark.parametrize("flag", [["--unfiltered"], ["--library-id", "x"], ["--no-images"]])
def test_visium_s_flags_do_not_apply(tmp_path, flag):
    truth = write_xenium_bundle(tmp_path / "xenium")
    message = _refused(truth.bundle, "-o", tmp_path / "out.h5ad", *flag)
    assert "does not apply when converting xenium to h5ad" in message


@pytest.mark.parametrize("shuffled", [False, True])
def test_a_table_in_the_matrix_s_order_is_written_straight_through(tmp_path, shuffled):
    """Xenium writes cells.csv.gz in the matrix's order; only a table in
    another order should pay for the scatter's scratch copy."""
    from adata.conversion import Options
    from adata.conversion.xenium import XeniumImport

    truth = write_xenium_bundle(tmp_path / "xenium", shuffled=shuffled)
    plan = XeniumImport().plan(truth.bundle, tmp_path / "out.h5ad", "xenium", "h5ad", Options())
    assert (plan.detail.cells.join.target is None) is not shuffled


def _rewrite(path: Path, edit) -> None:
    path.write_text(edit(path.read_text()))


def test_clusters_matching_no_cell_are_refused_as_another_run_s(tmp_path):
    """Planned as 'every cell left empty', then crashed the h5ad write."""
    truth = write_xenium_bundle(tmp_path / "xenium")
    clusters = truth.bundle / "analysis/clustering/gene_expression_graphclust/clusters.csv"
    _rewrite(clusters, lambda text: text.replace("-1,", "-2,"))
    message = _refused(truth.bundle, "-o", tmp_path / "out.h5ad")
    assert "has a row for none of the matrix's 15 barcodes" in message


def test_a_clusters_table_without_its_column_is_refused_before_writing(tmp_path):
    """It used to fail inside the write, after staging began."""
    truth = write_xenium_bundle(tmp_path / "xenium")
    clusters = truth.bundle / "analysis/clustering/gene_expression_graphclust/clusters.csv"
    _rewrite(clusters, lambda text: text.replace("Barcode,Cluster", "Barcode,cluster", 1))
    assert "clusters.csv in analysis has no Cluster column" in _refused(truth.bundle, "-o", tmp_path / "out.h5ad")
    assert not list(tmp_path.glob("out*"))


def test_a_missing_centroid_is_nan_and_a_word_is_refused(tmp_path):
    import gzip

    truth = write_xenium_bundle(tmp_path / "xenium")
    path = truth.bundle / "cells.csv.gz"
    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    fields = lines[1].split(",")
    fields[1] = "NA"
    path.write_bytes(gzip.compress(("\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n").encode()))
    _ok(truth.bundle, "-o", tmp_path / "out.h5ad")
    assert np.isnan(_read(tmp_path / "out.h5ad").obsm["spatial"][0, 0])

    fields[1] = "left"
    path.write_bytes(gzip.compress(("\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n").encode()))
    assert "x_centroid holds values that are not numbers" in _refused(truth.bundle, "-o", tmp_path / "out2.h5ad")


def test_clusters_of_cells_the_matrix_lacks_leave_no_category(tmp_path):
    truth = write_xenium_bundle(tmp_path / "xenium")
    clusters = truth.bundle / "analysis/clustering/gene_expression_graphclust/clusters.csv"
    _rewrite(clusters, lambda text: text + "zzzzzzzz-1,99\n")
    _ok(truth.bundle, "-o", tmp_path / "out.h5ad")
    assert "99" not in _read(tmp_path / "out.h5ad").obs["graphclust"].cat.categories


def test_an_archive_without_the_tables_says_so(tmp_path):
    """Its empty unpacked folder passed for an analysis folder, and the
    import went ahead without a word."""
    import tarfile

    truth = write_xenium_bundle(tmp_path / "xenium")
    shutil.rmtree(truth.bundle / "analysis")
    with tarfile.open(truth.bundle / "analysis.tar.gz", "w:gz") as tar:
        tar.add(truth.bundle / "experiment.xenium", arcname="outs/other.txt")
    assert "holds none of the clustering, UMAP or PCA tables" in _out(_ok(truth.bundle, "-o", tmp_path / "out.h5ad"))
    assert not list(tmp_path.glob(".adata-xenium-*"))


def test_from_xenium_names_the_matrix_not_another_file(tmp_path):
    """Any file was taken for its folder, and the matrix beside it read."""
    truth = write_xenium_bundle(tmp_path / "xenium")
    message = _refused(truth.bundle / "cells.csv.gz", "-o", tmp_path / "out.h5ad", "--from", "xenium")
    assert "give its folder, or its cell_feature_matrix.h5" in message

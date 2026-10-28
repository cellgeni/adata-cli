"""Tests for `adata convert` of Space Ranger output for standard Visium.

Fixtures are written as Space Ranger writes its `outs/` folder (see
conftest): positions shuffled and including spots off the matrix, Space
Ranger 1's headerless file and the later headed one. What anndata reads
back is checked against what each fixture was built from, laid out as
`squidpy.read.visium` lays it out.
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
from tests.conftest import (
    write_segmented_outputs,
    write_space_ranger_analysis,
    write_visium_hd_outs,
    write_visium_outs,
)

ad = pytest.importorskip("anndata")

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


# ---------------------------------------------------------------------------
# the mapping


@pytest.mark.parametrize("version", ["sr1", "sr2", "sr3_float"])
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_spots_images_and_scale_factors_land_where_squidpy_puts_them(tmp_path, version, suffix):
    truth = write_visium_outs(tmp_path / "run", version=version)
    out = tmp_path / f"out{suffix}"

    result = _ok(truth.outs, "-o", out)

    assert "Space Ranger (Visium)" in _out(result)
    adata = _read(out)
    assert list(adata.obs_names) == truth.barcodes
    assert list(adata.obs.columns) == ["in_tissue", "array_row", "array_col"]
    for column in ("in_tissue", "array_row", "array_col"):
        assert adata.obs[column].dtype == np.int64, column
        np.testing.assert_array_equal(adata.obs[column], getattr(truth, column))
    spatial = adata.obsm["spatial"]
    assert spatial.dtype == (np.float64 if version == "sr3_float" else np.int64)
    np.testing.assert_array_equal(spatial, truth.spatial)

    library = adata.uns["spatial"]["V1_Test_Section"]
    for name, pixels in truth.images.items():
        assert library["images"][name].dtype == np.uint8
        np.testing.assert_array_equal(library["images"][name], pixels)
    assert library["scalefactors"] == truth.scalefactors
    assert library["metadata"] == {
        "chemistry_description": "Spatial 3' v1",
        "software_version": "spaceranger-2.1.0",
    }
    np.testing.assert_array_equal(adata.X.toarray(), truth.counts)
    assert "cellranger" in adata.uns


def test_the_run_folder_holding_outs_is_read_too(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    _ok(tmp_path / "run", "-o", tmp_path / "out.h5ad")
    assert list(_read(tmp_path / "out.h5ad").obs_names) == truth.barcodes


def test_unfiltered_reads_every_spot_with_its_position(tmp_path):
    truth = write_visium_outs(tmp_path / "run", n_spots=8, n_extra=4)
    _ok(truth.outs, "-o", tmp_path / "out.h5ad", "--unfiltered")
    adata = _read(tmp_path / "out.h5ad")
    assert adata.n_obs == 12
    assert adata.obs["in_tissue"].tolist() == [1] * 8 + [0] * 4


def test_a_matrix_given_directly_takes_its_folder_as_the_output(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    _ok(
        truth.outs / "filtered_feature_bc_matrix.h5", "-o", tmp_path / "out.h5ad",
        "--from", "visium",
    )
    assert "spatial" in _read(tmp_path / "out.h5ad").obsm


def test_a_10x_download_with_its_sample_prefix_is_found(tmp_path):
    truth = write_visium_outs(tmp_path / "run", raw=False)
    (truth.outs / "filtered_feature_bc_matrix.h5").rename(
        truth.outs / "V1_Mouse_Brain_filtered_feature_bc_matrix.h5"
    )
    _ok(truth.outs, "-o", tmp_path / "out.h5ad")
    assert list(_read(tmp_path / "out.h5ad").obs_names) == truth.barcodes


def test_library_id_and_no_images(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    _ok(truth.outs, "-o", tmp_path / "out.h5ad", "--library-id", "brain", "--no-images")
    library = _read(tmp_path / "out.h5ad").uns["spatial"]["brain"]
    assert "images" not in library
    assert library["scalefactors"] == truth.scalefactors


def test_a_missing_image_is_a_warning_not_a_refusal(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    (truth.outs / "spatial" / "tissue_hires_image.png").unlink()
    result = _ok(truth.outs, "-o", tmp_path / "out.h5ad")
    assert "no hires image" in _out(result)
    assert list(_read(tmp_path / "out.h5ad").uns["spatial"]["V1_Test_Section"]["images"]) == ["lowres"]


def test_split_feature_types_puts_the_positions_in_every_output(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    _ok(truth.outs, "-o", tmp_path / "out.h5ad", "--split-feature-types")
    for name in ("out.Gene_Expression.h5ad", "out.Antibody_Capture.h5ad"):
        adata = _read(tmp_path / name)
        np.testing.assert_array_equal(adata.obsm["spatial"], truth.spatial)
        assert "V1_Test_Section" in adata.uns["spatial"]


# ---------------------------------------------------------------------------
# detection


def test_an_outs_folder_is_visium_not_a_cell_ranger_2_mtx_layout(tmp_path):
    """A Space Ranger outs/ holds filtered_ and raw_feature_bc_matrix/ MEX
    folders, which 10x-mtx took for a Cell Ranger 2 file's genome folders
    and refused with "pick one with --genome"."""
    truth = write_visium_outs(tmp_path / "run")
    for name in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix"):
        mex = truth.outs / name
        mex.mkdir()
        for file in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
            (mex / file).write_bytes(b"")

    result = _ok(truth.outs, "-o", tmp_path / "out.h5ad")

    assert "Space Ranger (Visium)" in _out(result)


def test_a_bare_matrix_is_still_a_plain_10x_matrix(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    result = _ok(truth.outs / "filtered_feature_bc_matrix.h5", "-o", tmp_path / "out.h5ad")
    assert "10x h5" in _out(result)
    assert "spatial" not in _read(tmp_path / "out.h5ad").obsm


# ---------------------------------------------------------------------------
# refusals, each before anything is written


def test_a_barcode_with_no_position_is_refused(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    positions = truth.outs / "spatial" / "tissue_positions.csv"
    lines = positions.read_text().splitlines()
    positions.write_text("\n".join(l for l in lines if not l.startswith(truth.barcodes[3])) + "\n")

    message = _refused(truth.outs, "-o", tmp_path / "out.h5ad")

    assert f"1 of the matrix's 12 barcodes have no row in spatial/tissue_positions.csv (the first is '{truth.barcodes[3]}')" in message
    assert not list(tmp_path.glob("out*"))


def test_a_repeated_position_is_refused(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    positions = truth.outs / "spatial" / "tissue_positions.csv"
    lines = positions.read_text().splitlines()
    positions.write_text("\n".join(lines + [lines[1]]) + "\n")
    assert "more than one row for 1 of the matrix's barcodes" in _refused(truth.outs, "-o", tmp_path / "out.h5ad")




def test_aggr_output_is_refused(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    with h5py.File(truth.outs / "filtered_feature_bc_matrix.h5", "a") as f:
        f.attrs["library_ids"] = np.array([b"a", b"b"])
    assert "holds 2 libraries (a, b)" in _refused(truth.outs, "-o", tmp_path / "out.h5ad")


def test_a_bundle_with_only_mex_folders_points_at_10x_mtx(tmp_path):
    truth = write_visium_outs(tmp_path / "run", raw=False)
    (truth.outs / "filtered_feature_bc_matrix.h5").unlink()
    (truth.outs / "filtered_feature_bc_matrix").mkdir()
    message = _refused(truth.outs, "-o", tmp_path / "out.h5ad")
    assert "Convert the folder with --from 10x-mtx" in message


@pytest.mark.parametrize("flag", [["--unfiltered"], ["--library-id", "x"], ["--no-images"]])
def test_the_visium_flags_are_refused_for_other_formats(tmp_path, flag):
    truth = write_visium_outs(tmp_path / "run")
    message = _refused(truth.outs / "filtered_feature_bc_matrix.h5", "-o", tmp_path / "out.h5ad", *flag)
    assert "does not apply when converting 10x-h5 to h5ad" in message


def test_unfiltered_with_a_matrix_given_is_refused(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    message = _refused(
        truth.outs / "filtered_feature_bc_matrix.h5", "-o", tmp_path / "out.h5ad",
        "--from", "visium", "--unfiltered",
    )
    assert "is a matrix already" in message


def test_a_positions_file_saved_with_a_byte_order_mark_keeps_its_header(tmp_path):
    """Excel's CSV starts with a BOM, which made the first field
    '﻿barcode': the header was taken for a data row, and the write
    failed on 'could not convert string to float'."""
    truth = write_visium_outs(tmp_path / "run")
    positions = truth.outs / "spatial" / "tissue_positions.csv"
    positions.write_bytes(b"\xef\xbb\xbf" + positions.read_bytes())
    _ok(truth.outs, "-o", tmp_path / "out.h5ad")
    np.testing.assert_array_equal(_read(tmp_path / "out.h5ad").obsm["spatial"], truth.spatial)


def test_a_position_that_is_not_a_number_is_refused_before_writing(tmp_path):
    """It used to pass the plan and fail in the write, after staging began."""
    truth = write_visium_outs(tmp_path / "run")
    positions = truth.outs / "spatial" / "tissue_positions.csv"
    lines = positions.read_text().splitlines()
    fields = lines[1].split(",")
    fields[5] = "left"
    positions.write_text("\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n")
    message = _refused(truth.outs, "-o", tmp_path / "out.h5ad")
    assert "pxl_col_in_fullres holds values that are not numbers" in message
    assert not list(tmp_path.glob("out*"))


@pytest.mark.parametrize("name", ["", "a/b"])
def test_a_library_id_that_cannot_be_a_key_is_refused(tmp_path, name):
    """An empty one stopped with a raw h5py error; one with '/' silently
    nested as uns['spatial']['a']['b']."""
    truth = write_visium_outs(tmp_path / "run")
    assert "cannot be a key of uns" in _refused(truth.outs, "-o", tmp_path / "out.h5ad", "--library-id", name)


# ---------------------------------------------------------------------------
# Visium HD


def _check(adata, truth) -> None:
    assert list(adata.obs_names) == truth.barcodes
    for column in ("in_tissue", "array_row", "array_col"):
        assert adata.obs[column].dtype == np.int64, column
        np.testing.assert_array_equal(adata.obs[column], getattr(truth, column))
    assert adata.obsm["spatial"].dtype == np.float64
    np.testing.assert_array_equal(adata.obsm["spatial"], truth.spatial)
    np.testing.assert_array_equal(adata.X.toarray(), truth.counts)


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_an_hd_output_reads_its_8_um_bins_by_default(tmp_path, suffix):
    hd = write_visium_hd_outs(tmp_path / "run")
    out = tmp_path / f"out{suffix}"

    result = _ok(hd.outs, "-o", out)

    assert "Visium HD, 8 µm bins" in _out(result)
    adata = _read(out)
    truth = hd.bins[8]
    _check(adata, truth)
    library = adata.uns["spatial"]["tiny_mouse_brain"]
    assert library["scalefactors"]["bin_size_um"] == 8.0
    np.testing.assert_array_equal(library["images"]["hires"], truth.images["hires"])


def test_bin_size_picks_another_bin_folder_and_a_bin_folder_reads_alone(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run")
    _ok(hd.outs, "-o", tmp_path / "a.h5ad", "--bin-size", "16")
    _check(_read(tmp_path / "a.h5ad"), hd.bins[16])
    _ok(hd.bins[16].outs, "-o", tmp_path / "b.h5ad")
    _check(_read(tmp_path / "b.h5ad"), hd.bins[16])


def test_unfiltered_reads_every_bin_on_the_slide(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,))
    _ok(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "16", "--unfiltered")
    assert _read(tmp_path / "out.h5ad").n_obs == 36


def test_a_bin_size_the_output_lacks_names_those_it_has(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run")
    message = _refused(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "2")
    assert "has no 2 µm bins; it has 8 µm, 16 µm. Pick one with --bin-size" in message


def test_bin_size_on_standard_visium_is_refused(tmp_path):
    truth = write_visium_outs(tmp_path / "run")
    assert "--bin-size picks among a Visium HD output's bins" in _refused(
        truth.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "8"
    )


def test_hd_without_pyarrow_names_the_extra(tmp_path, monkeypatch):
    import builtins

    hd = write_visium_hd_outs(tmp_path / "run")
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    message = _refused(hd.outs, "-o", tmp_path / "out.h5ad")
    assert "pip install 'pyadata-cli[spatial]'" in message
    assert not list(tmp_path.glob("out*"))


def test_bin_size_with_a_matrix_given_is_refused(tmp_path):
    """It was ignored: the 8 µm matrix named was read, exit 0."""
    hd = write_visium_hd_outs(tmp_path / "run")
    message = _refused(
        hd.bins[8].outs / "filtered_feature_bc_matrix.h5", "-o", tmp_path / "out.h5ad",
        "--from", "visium", "--bin-size", "16",
    )
    assert "--bin-size picks among the matrices" in message


def test_bin_size_zero_is_a_size_the_output_lacks_not_the_default(tmp_path):
    """0 read as 'not given', and the 8 µm bins were written."""
    hd = write_visium_hd_outs(tmp_path / "run")
    assert "has no 0 µm bins" in _refused(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "0")


def test_missing_values_in_hd_positions_are_refused_before_writing(tmp_path):
    """A null in an integer column became an arbitrary number in obs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,))
    path = hd.bins[16].outs / "spatial" / "tissue_positions.parquet"
    table = pq.read_table(path)
    rows = table.column("array_row").to_pylist()
    rows[0] = None
    table = table.set_column(table.schema.get_field_index("array_row"), "array_row", pa.array(rows, pa.uint32()))
    pq.write_table(table, path)

    message = _refused(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "16")

    assert "tissue_positions.parquet has missing values in array_row (1)" in message
    assert not list(tmp_path.glob("out*"))


def test_a_bin_folder_without_scale_factors_names_what_it_lacks(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,))
    (hd.bins[16].outs / "spatial" / "scalefactors_json.json").unlink()
    message = _refused(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "16")
    assert "has no spatial/scalefactors_json.json beside the matrix" in message
    assert "tissue_positions.csv" not in message


# ---------------------------------------------------------------------------
# Space Ranger's analysis/ for spots and bins


def _check_analysis(adata, truth, *, n_extra=0) -> None:
    """graphclust, X_umap and X_pca as analysis/ lists them, empty for a
    barcode it leaves out and for the `n_extra` unfiltered ones after."""
    expected = list(truth.clusters) + [""] * n_extra
    clusters = adata.obs["graphclust"].astype(object).where(adata.obs["graphclust"].notna(), "")
    assert list(clusters) == expected
    assert list(adata.obs["graphclust"].cat.categories) == sorted(set(truth.clusters) - {""}, key=int)
    for key, values in (("X_umap", truth.umap), ("X_pca", truth.pca)):
        assert adata.obsm[key].dtype == np.float64, key
        padded = np.vstack([values, np.full((n_extra, values.shape[1]), np.nan)])
        np.testing.assert_array_equal(adata.obsm[key], padded)


@pytest.mark.parametrize("version", ["sr1", "sr2"])
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_spots_take_space_rangers_clusters_umap_and_pca(tmp_path, version, suffix):
    truth = write_visium_outs(tmp_path / "run", version=version, analysis=True)
    out = tmp_path / f"out{suffix}"

    result = _ok(truth.outs, "-o", out)

    assert "analysis: clusters, X_umap, X_pca" in _out(result)
    assert "2 spots have no clusters, X_umap or X_pca in analysis/; they are left empty" in _out(result)
    adata = _read(out)
    assert list(adata.obs.columns) == ["in_tissue", "array_row", "array_col", "graphclust"]
    _check_analysis(adata, truth.analysis)
    np.testing.assert_array_equal(adata.obsm["spatial"], truth.spatial)


@pytest.mark.parametrize("size", [8, 16])
def test_hd_bins_take_their_own_analysis(tmp_path, size):
    hd = write_visium_hd_outs(tmp_path / "run", analysis=True)
    result = _ok(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", str(size))
    assert "bins have no clusters, X_umap or X_pca in analysis/" in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    _check(adata, hd.bins[size])
    _check_analysis(adata, hd.bins[size].analysis)


def test_2_um_bins_have_no_analysis_and_need_none(tmp_path):
    """Space Ranger writes no analysis/ for 2 µm bins: nothing to read, and
    nothing to warn of."""
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(2, 8), grid=3, analysis=True)
    result = _ok(hd.outs, "-o", tmp_path / "out.h5ad", "--bin-size", "2")
    assert "analysis:" not in _out(result) and "in analysis/" not in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    assert list(adata.obs.columns) == ["in_tissue", "array_row", "array_col"]
    assert list(adata.obsm) == ["spatial"]


@pytest.mark.parametrize("missing", [0, 1])
def test_analysis_in_the_matrixs_order_lands_on_its_spots(tmp_path, missing):
    """As Space Ranger writes it: in the matrix's order, whole (written
    straight through) or with a spot left out (through the buckets)."""
    truth = write_visium_outs(tmp_path / "run")
    analysis = write_space_ranger_analysis(
        truth.outs, truth.barcodes, np.random.default_rng(1), missing=missing, shuffled=False
    )
    result = _ok(truth.outs, "-o", tmp_path / "out.h5ad")
    if missing:
        assert "1 spot has no clusters, X_umap or X_pca in analysis/; it is left empty" in _out(result)
    else:
        assert "in analysis/" not in _out(result)
    _check_analysis(_read(tmp_path / "out.h5ad"), analysis)


def test_tables_lacking_different_spots_warn_apart(tmp_path):
    """Tables lacking as many spots, but not the same ones, shared one
    warning, as if those spots lacked all of them."""
    truth = write_visium_outs(tmp_path / "run", analysis=True)
    path = truth.outs / "analysis" / "umap" / "gene_expression_2_components" / "projection.csv"
    lines = path.read_text().splitlines()
    listed = {line.split(",", 1)[0] for line in lines[1:]}
    absent = [b for b in truth.barcodes if b not in listed]
    # Swap which spots the UMAP lacks: as many, but others.
    kept = [line for line in lines[1:] if not line.startswith(lines[1].split(",", 1)[0])]
    kept = kept[1:] + [f"{b},0.5,0.5" for b in absent]
    path.write_text("\n".join([lines[0], *kept]) + "\n")

    result = _ok(truth.outs, "-o", tmp_path / "out.h5ad")

    assert "2 spots have no clusters or X_pca in analysis/" in _out(result)
    assert "2 spots have no X_umap in analysis/" in _out(result)


def test_unfiltered_spots_outside_the_analysis_are_left_empty(tmp_path):
    truth = write_visium_outs(tmp_path / "run", n_spots=8, n_extra=4, analysis=True)
    result = _ok(truth.outs, "-o", tmp_path / "out.h5ad", "--unfiltered")
    assert "6 spots have no clusters, X_umap or X_pca in analysis/" in _out(result)
    _check_analysis(_read(tmp_path / "out.h5ad"), truth.analysis, n_extra=4)


def test_split_feature_types_puts_the_analysis_in_every_output(tmp_path):
    truth = write_visium_outs(tmp_path / "run", analysis=True)
    _ok(truth.outs, "-o", tmp_path / "out.h5ad", "--split-feature-types")
    for name in ("out.Gene_Expression.h5ad", "out.Antibody_Capture.h5ad"):
        _check_analysis(_read(tmp_path / name), truth.analysis)


def test_the_newer_names_win_when_both_are_there(tmp_path):
    truth = write_visium_outs(tmp_path / "run", analysis=True)
    old = truth.outs / "analysis" / "clustering" / "graphclust"
    old.mkdir(parents=True)
    (old / "clusters.csv").write_text("Barcode,Cluster\n" + "".join(f"{b},99\n" for b in truth.barcodes))
    _ok(truth.outs, "-o", tmp_path / "out.h5ad")
    _check_analysis(_read(tmp_path / "out.h5ad"), truth.analysis)


def test_clusters_without_a_cluster_column_are_refused_before_writing(tmp_path):
    truth = write_visium_outs(tmp_path / "run", analysis=True)
    path = truth.outs / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"
    path.write_text(path.read_text().replace("Barcode,Cluster", "Barcode,Graph-based", 1))
    assert "clusters.csv in analysis has no Cluster column" in _refused(truth.outs, "-o", tmp_path / "out.h5ad")
    assert not list(tmp_path.glob("out*"))


def test_a_projection_that_is_not_numbers_is_refused_before_writing(tmp_path):
    truth = write_visium_outs(tmp_path / "run", analysis=True)
    path = truth.outs / "analysis" / "umap" / "gene_expression_2_components" / "projection.csv"
    lines = path.read_text().splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0] + ",far"
    path.write_text("\n".join(lines) + "\n")
    assert "UMAP-2 holds values that are not numbers" in _refused(truth.outs, "-o", tmp_path / "out.h5ad")
    assert not list(tmp_path.glob("out*"))


def test_an_analysis_from_another_run_is_refused(tmp_path):
    truth = write_visium_outs(tmp_path / "run", analysis=True)
    path = truth.outs / "analysis" / "pca" / "gene_expression_10_components" / "projection.csv"
    path.write_text("Barcode,PC-1\nGGGG0000-1,0.5\n")
    message = _refused(truth.outs, "-o", tmp_path / "out.h5ad")
    assert "analysis X_pca has a row for none of the matrix's 12 barcodes" in message


# ---------------------------------------------------------------------------
# Visium HD segmented cells (Space Ranger 4)


def _check_cells(adata, truth, *, unfiltered=False) -> None:
    assert list(adata.obs_names) == (truth.raw_barcodes if unfiltered else truth.barcodes)
    if unfiltered:
        return
    np.testing.assert_allclose(adata.obsm["spatial"], truth.spatial, rtol=0, atol=1e-9)
    np.testing.assert_allclose(adata.obs["cell_area"], truth.cell_area, rtol=1e-12)
    np.testing.assert_allclose(adata.obs["nucleus_area"], truth.nucleus_area, rtol=1e-12)
    clusters = adata.obs["graphclust"].astype(object).where(adata.obs["graphclust"].notna(), "")
    assert list(clusters) == list(truth.clusters)
    np.testing.assert_array_equal(adata.obsm["X_umap"], truth.umap)
    np.testing.assert_array_equal(adata.X.toarray(), truth.counts)


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_cells_land_with_their_centroids_areas_and_analysis(tmp_path, suffix):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    out = tmp_path / f"out{suffix}"

    result = _ok(hd.outs, "-o", out, "--cells")

    assert "Visium HD, segmented cells" in _out(result)
    assert "1 cells have no nucleus" in _out(result)
    assert "2 cells have no clusters or X_umap in analysis/" in _out(result)
    adata = _read(out)
    _check_cells(adata, hd.cells)
    assert list(adata.obs.columns) == ["cell_area", "nucleus_area", "graphclust"]
    library = adata.uns["spatial"]["tiny_mouse_brain"]
    assert library["scalefactors"] == hd.cells.scalefactors
    np.testing.assert_array_equal(library["images"]["hires"], hd.cells.images["hires"])
    assert library["metadata"]["software_version"] == "spaceranger-4.0.1"
    assert "cellranger" in adata.uns


def test_the_segmented_folder_and_its_matrix_read_as_cells_without_the_flag(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    _ok(hd.cells.folder, "-o", tmp_path / "a.h5ad")
    _check_cells(_read(tmp_path / "a.h5ad"), hd.cells)
    _ok(hd.cells.folder / "filtered_feature_cell_matrix.h5", "-o", tmp_path / "b.h5ad", "--from", "visium")
    _check_cells(_read(tmp_path / "b.h5ad"), hd.cells)


def test_polygons_in_another_order_than_the_matrix_land_on_their_cells(tmp_path):
    outs = tmp_path / "outs"
    truth = write_segmented_outputs(outs, shuffled=True, seed=4)
    _ok(truth.folder, "-o", tmp_path / "out.h5ad")
    _check_cells(_read(tmp_path / "out.h5ad"), truth)


def test_unfiltered_reads_every_segmented_cell(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    _ok(hd.outs, "-o", tmp_path / "out.h5ad", "--cells", "--unfiltered")
    adata = _read(tmp_path / "out.h5ad")
    _check_cells(adata, hd.cells, unfiltered=True)
    assert adata.obsm["spatial"].shape == (len(hd.cells.raw_barcodes), 2)
    assert not np.isnan(adata.obs["cell_area"]).any()


def test_bins_stay_the_default_and_the_summary_points_at_cells(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", cells=True)
    result = _ok(hd.outs, "-o", tmp_path / "out.h5ad")
    assert "8 µm bins" in _out(result)
    assert "segmented cells too: --cells" in _out(result)
    _check(_read(tmp_path / "out.h5ad"), hd.bins[8])


def test_split_feature_types_puts_the_cells_in_every_output(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    with h5py.File(hd.cells.folder / "filtered_feature_cell_matrix.h5", "a") as f:
        types = f["matrix/features/feature_type"][...].astype("S16")
        types[-1] = b"Antibody Capture"
        del f["matrix/features/feature_type"]
        f["matrix/features/feature_type"] = types
    _ok(hd.outs, "-o", tmp_path / "out.h5ad", "--cells", "--split-feature-types")
    for name in ("out.Gene_Expression.h5ad", "out.Antibody_Capture.h5ad"):
        adata = _read(tmp_path / name)
        np.testing.assert_allclose(adata.obsm["spatial"], hd.cells.spatial, rtol=0, atol=1e-9)
        assert list(adata.obs.columns) == ["cell_area", "nucleus_area", "graphclust"]


def test_without_nuclei_or_analysis_the_cells_still_convert(tmp_path):
    truth = write_segmented_outputs(tmp_path / "outs")
    (truth.folder / "nucleus_segmentations.geojson").unlink()
    import shutil
    shutil.rmtree(truth.folder / "analysis")
    result = _ok(truth.folder, "-o", tmp_path / "out.h5ad")
    assert "obs will have no nucleus_area" in _out(result)
    adata = _read(tmp_path / "out.h5ad")
    assert list(adata.obs.columns) == ["cell_area"]
    np.testing.assert_allclose(adata.obsm["spatial"], truth.spatial, rtol=0, atol=1e-9)


def test_cells_on_an_output_without_segmentation_are_refused(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,))
    assert "has no segmented cells" in _refused(hd.outs, "-o", tmp_path / "a.h5ad", "--cells")
    truth = write_visium_outs(tmp_path / "sd")
    assert "has no segmented cells" in _refused(truth.outs, "-o", tmp_path / "b.h5ad", "--cells")
    assert not list(tmp_path.glob("*.h5ad"))


def test_cells_with_a_bin_size_are_refused(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    message = _refused(hd.outs, "-o", tmp_path / "out.h5ad", "--cells", "--bin-size", "16")
    assert "give one" in message
    message = _refused(hd.cells.folder, "-o", tmp_path / "out.h5ad", "--bin-size", "16")
    assert "holds cells" in message


def test_cells_with_a_cell_matrix_reads_it_and_with_a_bin_matrix_is_refused(tmp_path):
    """--cells with the cell matrix itself was refused, though that matrix
    is read as cells anyway."""
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    _ok(hd.cells.folder / "filtered_feature_cell_matrix.h5", "-o", tmp_path / "a.h5ad", "--from", "visium", "--cells")
    _check_cells(_read(tmp_path / "a.h5ad"), hd.cells)
    message = _refused(
        hd.bins[16].outs / "filtered_feature_bc_matrix.h5", "-o", tmp_path / "b.h5ad",
        "--from", "visium", "--cells",
    )
    assert "is a matrix already" in message


def test_a_cell_with_no_polygon_is_refused_before_writing(tmp_path):
    truth = write_segmented_outputs(tmp_path / "outs")
    path = truth.folder / "cell_segmentations.geojson"
    collection = json.loads(path.read_text())
    gone = int(truth.barcodes[3][len("cellid_"):-2])
    collection["features"] = [f for f in collection["features"] if f["properties"]["cell_id"] != gone]
    path.write_text(json.dumps(collection))

    message = _refused(truth.folder, "-o", tmp_path / "out.h5ad")

    assert f"1 of the matrix's 12 barcodes have no row in cell_segmentations.geojson (by cell_id) (the first is '{gone}')" in message
    assert not list(tmp_path.glob("out.*"))


def test_a_barcode_that_names_no_cell_is_refused(tmp_path):
    truth = write_segmented_outputs(tmp_path / "outs")
    with h5py.File(truth.folder / "filtered_feature_cell_matrix.h5", "a") as f:
        f["matrix/barcodes"][0] = b"s_008um_0000_001-1"
    message = _refused(truth.folder, "-o", tmp_path / "out.h5ad")
    assert "barcode 's_008um_0000_001-1' does not name a segmented cell" in message


def test_a_broken_polygon_file_is_refused_before_writing(tmp_path):
    truth = write_segmented_outputs(tmp_path / "outs")
    path = truth.folder / "nucleus_segmentations.geojson"
    path.write_text(path.read_text()[:-40])
    message = _refused(truth.folder, "-o", tmp_path / "out.h5ad")
    assert "nucleus_segmentations.geojson: feature" in message and "is not valid JSON" in message
    assert not list(tmp_path.glob("out.*"))


@pytest.mark.parametrize("value", [None, 0, "1"])
def test_scale_factors_without_microns_per_pixel_are_refused(tmp_path, value):
    truth = write_segmented_outputs(tmp_path / "outs")
    path = truth.folder / "spatial" / "scalefactors_json.json"
    factors = json.loads(path.read_text())
    if value is None:
        del factors["microns_per_pixel"]
    else:
        factors["microns_per_pixel"] = value
    path.write_text(json.dumps(factors))
    assert "has no microns_per_pixel" in _refused(truth.folder, "-o", tmp_path / "out.h5ad")


def test_a_segmented_folder_without_polygons_is_not_visium_cells(tmp_path):
    truth = write_segmented_outputs(tmp_path / "outs")
    (truth.folder / "cell_segmentations.geojson").unlink()
    message = _refused(
        truth.folder / "filtered_feature_cell_matrix.h5", "-o", tmp_path / "out.h5ad", "--from", "visium"
    )
    assert "has no cell_segmentations.geojson beside the matrix" in message


def test_a_segmented_folder_is_visium_not_unrecognised(tmp_path):
    """segmented_outputs/ has no tissue positions, so it was not taken for a
    Space Ranger folder, and nothing else recognised it either."""
    truth = write_segmented_outputs(tmp_path / "outs")
    result = _ok(truth.folder, "-o", tmp_path / "out.h5ad")
    assert "Space Ranger (Visium HD, segmented cells)" in _out(result)

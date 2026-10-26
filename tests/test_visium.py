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
from tests.conftest import write_visium_outs

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


def test_visium_hd_is_refused_by_name(tmp_path):
    outs = tmp_path / "hd" / "outs"
    (outs / "binned_outputs" / "square_008um").mkdir(parents=True)
    message = _refused(outs, "-o", tmp_path / "out.h5ad")
    assert "Visium HD output, which is not read yet" in message


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

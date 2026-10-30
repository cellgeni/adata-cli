"""Copies of anndata 0.7's dataframes (`encoding-version` 0.1.0).

Their categoricals are datasets of codes whose `categories` attribute is an
HDF5 object reference into `__categories`. A reference cannot leave its
file, so every copy writes them as current `categorical` groups and the
frame as the current version; anndata, the reference, must read the copy
with the same categories as the source.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app

ad = pytest.importorskip("anndata")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

CELLS = ["c1", "c2", "c3", "c4"]
TYPES = ["A", "B", "A", "C"]


def _legacy_frame(parent: h5py.Group, name: str, index: list, columns: dict) -> h5py.Group:
    """A dataframe as anndata 0.7 wrote it."""
    frame = parent.create_group(name)
    frame.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.1.0"})
    frame.attrs["column-order"] = np.array(list(columns), dtype=h5py.string_dtype())
    frame.create_dataset("_index", data=np.array(index, dtype="S"))
    categories = frame.create_group("__categories")
    for key, values in columns.items():
        labels = sorted(set(values))
        cats = categories.create_dataset(key, data=np.array(labels, dtype="S"))
        codes = frame.create_dataset(key, data=np.array([labels.index(v) for v in values], dtype=np.int8))
        codes.attrs["categories"] = cats.ref
    return frame


def _legacy_h5ad(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        _legacy_frame(f, "obs", CELLS, {"cell_type": TYPES})
        _legacy_frame(f, "var", ["g1", "g2"], {"kind": ["x", "y"]})
        f.create_dataset("X", data=np.arange(8, dtype=np.float32).reshape(4, 2))
    return path


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _ok(*args):
    result = runner.invoke(app, [*map(str, args)])
    assert result.exit_code == 0, " ".join(_ANSI.sub("", result.stdout).split())
    return result


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_subset_of_a_legacy_dataframe_keeps_its_categoricals(tmp_path, suffix):
    """The copied codes kept a reference into the source file: the h5ad
    read as a corrupt object, and Zarr refused to hold it."""
    source = _legacy_h5ad(tmp_path / "legacy.h5ad")
    assert list(_read(source).obs["cell_type"]) == TYPES
    keep = tmp_path / "keep.txt"
    keep.write_text("c1\nc3\nc4\n")
    out = tmp_path / f"sub{suffix}"
    _ok("subset", source, "--obs", keep, "-o", out)
    got = _read(out)
    assert list(got.obs["cell_type"]) == ["A", "A", "C"]
    # As anndata subsets: the categories the rows use.
    assert list(got.obs["cell_type"].cat.categories) == ["A", "C"]
    assert list(got.var["kind"]) == ["x", "y"]


@pytest.mark.parametrize("zarr_format", ["2", "3"])
def test_a_legacy_dataframe_converts_to_zarr(tmp_path, zarr_format):
    source = _legacy_h5ad(tmp_path / "legacy.h5ad")
    out = tmp_path / "out.zarr"
    _ok("convert", source, "-o", out, "--zarr-format", zarr_format)
    got = _read(out)
    assert list(got.obs["cell_type"]) == TYPES and list(got.obs_names) == CELLS


def test_a_copied_legacy_frame_is_written_as_the_current_version(tmp_path):
    source = _legacy_h5ad(tmp_path / "legacy.h5ad")
    keep = tmp_path / "keep.txt"
    keep.write_text("c2\n")
    _ok("subset", source, "--obs", keep, "-o", tmp_path / "sub.h5ad")
    with h5py.File(tmp_path / "sub.h5ad") as f:
        obs = f["obs"]
        assert obs.attrs["encoding-version"] == "0.2.0"
        assert "__categories" not in obs
        assert obs["cell_type"].attrs["encoding-type"] == "categorical"
        assert list(obs["cell_type/codes"][...]) == [0], "renumbered onto B, the one used"


def test_a_mudata_with_legacy_frames_reads_with_its_categoricals(tmp_path):
    """muon's early files hold 0.1.0 frames, globally and in each modality."""
    from tests.conftest import write_h5mu

    truth = write_h5mu(tmp_path / "in.h5mu")
    leiden = [str(v) for v in truth.obs["leiden"]]
    with h5py.File(truth.path, "a") as f:
        del f["obs"]
        _legacy_frame(f, "obs", list(truth.obs.index), {"leiden": leiden})
        names = list(truth.mods["atac"].obs_names)
        del f["mod/atac/obs"]
        _legacy_frame(f["mod/atac"], "obs", names, {"batch": ["b1" if i % 2 else "b2" for i in range(len(names))]})
    _ok("convert", truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac")
    got = _read(tmp_path / "out.h5ad")
    rows = truth.rows("atac")
    assert list(got.obs["leiden"]) == [leiden[i] for i in rows]
    assert list(got.obs["batch"]) == ["b1" if i % 2 else "b2" for i in range(len(rows))]


def test_compression_applies_to_legacy_codes_taken_onto_a_modality(tmp_path):
    from tests.conftest import write_h5mu

    truth = write_h5mu(tmp_path / "in.h5mu")
    leiden = [str(v) for v in truth.obs["leiden"]]
    with h5py.File(truth.path, "a") as f:
        del f["obs"]
        _legacy_frame(f, "obs", list(truth.obs.index), {"leiden": leiden})
    _ok("convert", truth.path, "-o", tmp_path / "out.h5ad", "--mod", "atac", "--compression", "gzip")
    with h5py.File(tmp_path / "out.h5ad") as f:
        # Its labels are HDF5 strings, which are never compressed: only
        # their heap pointers would be.
        assert f["obs/leiden/codes"].compression == "gzip"


def test_a_modality_with_a_legacy_raw_var_converts_to_h5ad(tmp_path):
    """F006: a modality's raw, copied whole by h5py, kept raw/var's 0.1.0
    categoricals with null references, and anndata could not open the
    output: 'Invalid HDF5 object reference'."""
    from tests.conftest import write_h5mu

    truth = write_h5mu(tmp_path / "in.h5mu")
    rna = truth.mods["rna"]
    kinds = ["x" if i % 2 else "y" for i in range(rna.n_vars)]
    with h5py.File(truth.path, "a") as f:
        mod = f["mod/rna"]
        if "raw" in mod:
            del mod["raw"]
        raw = mod.create_group("raw")
        raw.attrs.update({"encoding-type": "raw", "encoding-version": "0.1.0"})
        raw.create_dataset("X", data=np.ones((rna.n_obs, rna.n_vars), dtype=np.float32))
        _legacy_frame(raw, "var", list(rna.var_names), {"kind": kinds})
    out = tmp_path / "out.h5ad"
    _ok("convert", truth.path, "-o", out, "--mod", "rna")
    got = _read(out)
    assert list(got.raw.var["kind"]) == kinds
    assert list(got.raw.var_names) == list(rna.var_names)


def test_a_nested_legacy_column_named_before_its_labels_leaves_no_stray_link(tmp_path):
    """F006: HDF5 copies in name order, so a column sorting before
    `__categories` ("Phase") had its labels copied ahead of their place in
    the tree, under a second name at the output's root,
    `~obj_pointed_by_<n>`."""
    path = tmp_path / "legacy.h5ad"
    with h5py.File(path, "w") as f:
        _legacy_frame(f, "obs", CELLS, {"Phase": ["G1", "S", "G1", "G2M"]})
        _legacy_frame(f, "var", ["g1", "g2"], {"kind": ["x", "y"]})
        f.create_dataset("X", data=np.arange(8, dtype=np.float32).reshape(4, 2))
        raw = f.create_group("raw")
        raw.attrs.update({"encoding-type": "raw", "encoding-version": "0.1.0"})
        raw.create_dataset("X", data=np.ones((4, 2), dtype=np.float32))
        _legacy_frame(raw, "var", ["g1", "g2"], {"Kind": ["x", "y"]})
    out = tmp_path / "out.h5ad"
    _ok("convert", path, "-o", out)
    with h5py.File(out) as f:
        assert not [k for k in f if k.startswith("~")]
    assert list(_read(out).raw.var["Kind"]) == ["x", "y"]

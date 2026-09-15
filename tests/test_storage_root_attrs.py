"""Tests for AnnData root encoding attributes enforcement/warnings."""

from pathlib import Path

import h5py
import pytest

from adata.storage import open_store


def _make_minimal_h5ad(path: Path) -> None:
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs.create_dataset("obs_names", data=[b"cell_1"])

        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var.create_dataset("var_names", data=[b"gene_1"])

        f.create_dataset("X", data=[[1.0]])


def test_open_store_read_warns_for_missing_root_attrs(temp_dir: Path) -> None:
    file_path = temp_dir / "missing_root_attrs.h5ad"
    _make_minimal_h5ad(file_path)

    with pytest.warns(UserWarning, match="missing or invalid AnnData attrs"):
        with open_store(file_path, "r"):
            pass


def test_open_store_writable_mode_sets_root_attrs(temp_dir: Path) -> None:
    file_path = temp_dir / "set_root_attrs.h5ad"
    _make_minimal_h5ad(file_path)

    with open_store(file_path, "a"):
        pass

    with h5py.File(file_path, "r") as f:
        assert f.attrs.get("encoding-type") == "anndata"
        assert f.attrs.get("encoding-version") == "0.1.0"

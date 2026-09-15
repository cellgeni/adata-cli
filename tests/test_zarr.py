"""Tests for zarr auto-detection support (v2 and v3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from typer.testing import CliRunner
from rich.console import Console

from adata.cli import app
from adata.core.subset import subset_h5ad


zarr = pytest.importorskip("zarr")

runner = CliRunner()


class UnsupportedZarrFormat(Exception):
    pass


def _open_zarr_group(path: Path, zarr_format: Optional[int]) -> Any:
    if zarr_format is None:
        return zarr.open_group(path, mode="w")

    last_exc: Exception | None = None
    for kw in ("zarr_format", "zarr_version"):
        try:
            return zarr.open_group(path, mode="w", **{kw: zarr_format})
        except (TypeError, ValueError) as exc:
            last_exc = exc
            continue

    raise UnsupportedZarrFormat(str(last_exc)) from last_exc


def _create_array(group: Any, name: str, data: np.ndarray) -> Any:
    data = np.asarray(data)
    if hasattr(group, "create_array"):
        try:
            return group.create_array(name, data=data)
        except TypeError:
            return group.create_array(
                name, data=data, shape=data.shape, dtype=data.dtype
            )
    try:
        return group.create_dataset(name, data=data, shape=data.shape)
    except TypeError:
        return group.create_dataset(name, data=data)


def _create_zarr_store(path: Path, *, zarr_format: Optional[int]) -> None:
    root = _open_zarr_group(path, zarr_format)

    obs = root.create_group("obs")
    obs.attrs["_index"] = "obs_names"
    obs_names = ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"]
    _create_array(obs, "obs_names", np.array(obs_names, dtype="S"))
    _create_array(
        obs,
        "cell_type",
        np.array(["TypeA", "TypeB", "TypeA", "TypeC", "TypeB"], dtype="S"),
    )

    var = root.create_group("var")
    var.attrs["_index"] = "var_names"
    var_names = ["gene_1", "gene_2", "gene_3", "gene_4"]
    _create_array(var, "var_names", np.array(var_names, dtype="S"))

    X = np.array(
        [
            [1.0, 0.0, 2.5, 0.0],
            [0.0, 3.2, 0.0, 1.1],
            [2.1, 0.0, 1.8, 0.0],
            [0.0, 4.5, 0.0, 2.3],
            [1.5, 0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    _create_array(root, "X", X)

    uns = root.create_group("uns")
    _create_array(uns, "description", np.array(["Test dataset"], dtype="S"))


@pytest.fixture(params=[None, 2], ids=["default", "v2"])
def zarr_format(request) -> Optional[int]:
    return request.param


def _skip_if_unsupported(exc: Exception, zarr_format: Optional[int]) -> None:
    if zarr_format == 2:
        pytest.skip("zarr v2 not supported by installed zarr")
    raise exc


def test_info_zarr_auto_detect(temp_dir, zarr_format):
    store_path = temp_dir / f"test_{zarr_format or 'default'}.zarr"
    try:
        _create_zarr_store(store_path, zarr_format=zarr_format)
    except UnsupportedZarrFormat as exc:
        _skip_if_unsupported(exc, zarr_format)

    result = runner.invoke(app, ["view", str(store_path)])
    output = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, output
    assert "5 × 4" in output


def test_export_dataframe_zarr(temp_dir, zarr_format):
    store_path = temp_dir / f"test_{zarr_format or 'default'}.zarr"
    try:
        _create_zarr_store(store_path, zarr_format=zarr_format)
    except UnsupportedZarrFormat as exc:
        _skip_if_unsupported(exc, zarr_format)
    output = temp_dir / "obs.csv"

    result = runner.invoke(
        app,
        ["export", "dataframe", str(store_path), "obs", "--output", str(output)],
    )
    if result.exit_code != 0:
        raise AssertionError(
            f"exit_code={result.exit_code} exception={result.exception!r} output={result.output}"
        )
    assert output.exists()
    assert "obs_names" in output.read_text(encoding="utf-8")


def test_export_dict_zarr(temp_dir, zarr_format):
    store_path = temp_dir / f"test_{zarr_format or 'default'}.zarr"
    try:
        _create_zarr_store(store_path, zarr_format=zarr_format)
    except UnsupportedZarrFormat as exc:
        _skip_if_unsupported(exc, zarr_format)
    output = temp_dir / "uns.json"

    result = runner.invoke(
        app, ["export", "dict", str(store_path), "uns", str(output)]
    )
    assert result.exit_code == 0
    assert output.exists()


def test_subset_zarr_output(temp_dir, zarr_format):
    store_path = temp_dir / f"test_{zarr_format or 'default'}.zarr"
    try:
        _create_zarr_store(store_path, zarr_format=zarr_format)
    except UnsupportedZarrFormat as exc:
        _skip_if_unsupported(exc, zarr_format)
    obs_file = temp_dir / "obs.txt"
    obs_file.write_text("cell_1\ncell_3\n")
    output = temp_dir / "subset.zarr"

    console = Console()
    subset_h5ad(
        file=store_path,
        output=output,
        obs_file=obs_file,
        var_file=None,
        chunk_rows=1024,
        console=console,
    )
    root = zarr.open_group(output, mode="r")
    assert root["obs"]["obs_names"].shape[0] == 2
    assert root["X"].shape == (2, 4)

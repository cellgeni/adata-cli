"""`concat` outputs, read back by the anndata release that wrote the inputs.

Uses the reference stores of `test_anndata_versions.py`, one per release
from 0.8 to 0.13, so it needs the same environments (marker `integration`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adata.cli import app
from adata.elements.read import dataframe_columns, resolve_index
from adata.elements.write import set_column_order
from adata.storage import open_store

from tests.test_anndata_versions import (  # noqa: F401 - fixtures
    N_OBS,
    _out,
    cache,
    fmt,
    read_back,
    release,
    store,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

runner = CliRunner()


def _without(source: Path, target: Path, column: str) -> Path:
    """A copy of `source` whose obs lacks `column`."""
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    with open_store(target, "a") as handle:
        obs = handle.root["obs"]
        kept = [c for c in dataframe_columns(obs, resolve_index(obs, "obs")[1]) if c != column]
        del obs[column]
        set_column_order(obs, kept)
    return target


def test_an_outer_join_missing_a_text_column_opens_in_the_release_that_wrote_the_inputs(
    store, tmp_path, fmt, read_back
):
    """F028: a plain-string column one input lacks was written as a
    nullable-string-array, and anndata 0.8, 0.9 and 0.10 could not open the
    output at all, though they had written both inputs."""
    other = _without(store, tmp_path / f"other.{fmt}", "free_text")
    out = tmp_path / f"merged.{fmt}"
    result = runner.invoke(app, [
        "concat", str(store), str(other), "-o", str(out),
        "--join", "outer", "--index-unique", "-",
    ])
    assert result.exit_code == 0, _out(result)

    assert read_back(out, "a.shape") == f"({2 * N_OBS}, 4)"
    assert read_back(out, "a.obs['free_text'].isna().tolist()") == str(
        [False] * N_OBS + [True] * N_OBS
    )
    assert read_back(out, "list(a.obs['free_text'].astype(object)[:6])") == str(
        ["a", "bb", "ccc", "d", "ee", "f"]
    )
    assert read_back(out, "str(a.obs['cell_type'].dtype)") == "category"

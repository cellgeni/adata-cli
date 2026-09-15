"""Tests for dimension validation on import.

This is the guard that stops an import leaving a store internally
inconsistent, so every axis-bearing path needs a case that passes and one
that fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from rich.console import Console

from adata.elements import write as ew
from adata.formats.validate import DATAFRAME_PATHS, validate_dimensions

console = Console(stderr=True)

N_OBS, N_VAR, N_RAW_VAR = 3, 2, 5


@pytest.fixture
def root(new_store):
    """A store with obs=3, var=2 and a raw component whose var is 5."""
    path, opener = new_store()
    with opener("a") as r:
        ew.write_sparse(r, "X", [1.0], [0], [0, 1, 1, 1], (N_OBS, N_VAR))
        raw = ew.write_mapping(r, "raw")
        ew.write_dataframe_header(
            raw, "var", [f"rg{i}" for i in range(N_RAW_VAR)], []
        )
        ew.write_sparse(
            raw, "X", [1.0], [0], [0, 1, 1, 1], (N_OBS, N_RAW_VAR)
        )
    with opener("r") as r:
        yield r


@pytest.mark.parametrize(
    "path,shape",
    [
        ("obs", (N_OBS,)),
        ("var", (N_VAR,)),
        ("X", (N_OBS, N_VAR)),
        ("layers/counts", (N_OBS, N_VAR)),
        ("obsm/X_pca", (N_OBS, 10)),
        ("varm/PCs", (N_VAR, 10)),
        ("obsp/conn", (N_OBS, N_OBS)),
        ("varp/corr", (N_VAR, N_VAR)),
        ("raw/var", (N_RAW_VAR,)),
        ("raw/X", (N_OBS, N_RAW_VAR)),
        ("raw/varm/PCs", (N_RAW_VAR, 3)),
    ],
)
def test_correct_shapes_are_accepted(root, path, shape):
    validate_dimensions(root, path, shape, console)


@pytest.mark.parametrize(
    "path,shape,message",
    [
        ("obs", (99,), "obs has 3 cells"),
        ("var", (99,), "var has 2 features"),
        ("X", (99, N_VAR), "First dimension mismatch"),
        ("X", (N_OBS, 99), "Second dimension mismatch"),
        ("X", (N_OBS,), "requires 2D"),
        ("layers/counts", (99, N_VAR), "First dimension mismatch"),
        ("obsm/X_pca", (99, 10), "First dimension mismatch"),
        ("varm/PCs", (99, 10), "First dimension mismatch"),
        ("obsp/conn", (N_OBS, 99), "must be square"),
        ("varp/corr", (N_VAR, 99), "must be square"),
        ("raw/var", (99,), "raw has 5 variables"),
        ("raw/X", (99, N_RAW_VAR), "First dimension mismatch"),
        ("raw/X", (N_OBS, 99), "raw has 5 variables"),
        ("raw/X", (N_OBS,), "raw/X requires 2D"),
        ("raw/varm/PCs", (99, 3), "raw has 5 variables"),
    ],
)
def test_wrong_shapes_are_rejected(root, path, shape, message):
    with pytest.raises(ValueError, match=message):
        validate_dimensions(root, path, shape, console)


def test_raw_var_is_measured_against_raw_x_not_the_main_var(root):
    """raw usually holds more genes than the main object; the two must not mix."""
    validate_dimensions(root, "raw/var", (N_RAW_VAR,), console)
    with pytest.raises(ValueError):
        validate_dimensions(root, "raw/var", (N_VAR,), console)


def test_unknown_paths_pass_with_a_note(root, capsys):
    validate_dimensions(root, "uns/anything", (7,), console)


def test_leading_slash_is_normalised(root):
    with pytest.raises(ValueError, match="obs has 3 cells"):
        validate_dimensions(root, "/obs", (99,), console)


def test_dataframe_paths_are_the_ones_import_image_refuses():
    assert DATAFRAME_PATHS == {"obs", "var", "raw/var"}


def test_validation_is_skipped_when_an_axis_cannot_be_read(new_store):
    """A store with no obs must not make every import fail."""
    path, opener = new_store()
    with opener("a") as root:
        del root["obs"]
    with opener("r") as root:
        validate_dimensions(root, "obsm/x", (99, 2), console)


def test_raw_without_x_falls_back_to_raw_var(new_store):
    path, opener = new_store()
    with opener("a") as root:
        raw = ew.write_mapping(root, "raw")
        ew.write_dataframe_header(raw, "var", ["a", "b", "c", "d"], [])
    with opener("r") as root:
        validate_dimensions(root, "raw/var", (4,), console)
        with pytest.raises(ValueError, match="raw has 4 variables"):
            validate_dimensions(root, "raw/var", (2,), console)

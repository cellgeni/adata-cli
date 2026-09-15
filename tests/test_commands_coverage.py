"""Tests for command surfaces the rest of the suite reaches only indirectly.

Covers the extension-dispatching `import_object` entry point, every `ls`
output mode, the concat merge strategies, and the CLI's error paths -- the
places where a user gets a message rather than a traceback.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from adata.cli import app
from adata.commands.import_data import import_object
from adata.commands.ls import list_store, walk
from adata.core.concat import MERGE_STRATEGIES, _merge_values, _MISSING
from adata.elements import spec
from adata.elements import write as ew
from adata.storage import open_store

runner = CliRunner()
console = Console(stderr=True)
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    return " ".join(_ANSI.sub("", result.stdout + (result.stderr or "")).split())


@pytest.fixture
def store(new_store):
    path, opener = new_store()
    return path, opener


# ---------------------------------------------------------------------------
# import_object: dispatch by file extension


@pytest.mark.parametrize(
    "suffix,write,target",
    [
        (".csv", lambda p: p.write_text("_index,v\nc1,1\nc2,2\nc3,3\n"), "obs"),
        (".npy", lambda p: np.save(p, np.zeros((3, 2))), "obsm/a"),
        (".json", lambda p: p.write_text('{"k": 1}'), "uns/j"),
    ],
)
def test_import_object_dispatches_on_extension(
    store, temp_dir, suffix, write, target
):
    path, opener = store
    source = temp_dir / f"in{suffix}"
    write(source)

    import_object(
        file=path,
        obj=target,
        input_file=source,
        output_file=None,
        inplace=True,
        index_column=None,
        console=console,
    )

    with opener("r") as root:
        node = root
        for part in target.split("/"):
            node = node[part]
        assert node is not None


def test_import_object_dispatches_mtx(store, temp_dir):
    path, opener = store
    source = temp_dir / "in.mtx"
    source.write_text(
        "%%MatrixMarket matrix coordinate real general\n3 2 2\n1 1 1.0\n2 2 2.0\n"
    )
    import_object(
        file=path, obj="X", input_file=source, output_file=None,
        inplace=True, index_column=None, console=console,
    )
    with opener("r") as root:
        assert spec.encoding_type(root["X"]) == spec.CSR_MATRIX


def test_import_object_rejects_an_unknown_extension(store, temp_dir):
    path, _ = store
    source = temp_dir / "in.parquet"
    source.write_text("x")
    with pytest.raises(ValueError, match="Unsupported input file extension"):
        import_object(
            file=path, obj="obs", input_file=source, output_file=None,
            inplace=True, index_column=None, console=console,
        )


def test_import_object_rejects_index_column_for_the_wrong_format(store, temp_dir):
    path, _ = store
    source = temp_dir / "in.npy"
    np.save(source, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="--index-column is only valid"):
        import_object(
            file=path, obj="obsm/a", input_file=source, output_file=None,
            inplace=True, index_column="c", console=console,
        )


def test_import_object_requires_an_output_when_not_inplace(store, temp_dir):
    path, _ = store
    source = temp_dir / "in.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="Output file is required"):
        import_object(
            file=path, obj="uns/j", input_file=source, output_file=None,
            inplace=False, index_column=None, console=console,
        )


def test_import_to_an_output_copies_the_source_first(store, temp_dir, backend):
    """Non-inplace imports must not touch the input store."""
    from tests.conftest import backend_path

    path, opener = store
    source = temp_dir / "in.json"
    source.write_text('{"k": 1}')
    out = backend_path(temp_dir, backend, "out")

    import_object(
        file=path, obj="uns/j", input_file=source, output_file=out,
        inplace=False, index_column=None, console=console,
    )

    with open_store(out, "r") as handle:
        assert "j" in handle.root["uns"]
    with opener("r") as root:
        assert "j" not in root["uns"], "the source must be left alone"


# ---------------------------------------------------------------------------
# ls


@pytest.fixture
def populated(store):
    path, opener = store
    with opener("a") as root:
        ew.write_dense(root["obsm"], "X_pca", np.zeros((3, 2)))
        group = ew.write_mapping(root["uns"], "spatial")
        ew.write_scalar(group, "scale", 1.5)
    return path


def test_ls_tree_lists_every_member(populated, capsys):
    list_store(populated, Console())
    out = capsys.readouterr().out
    for key in ("obs", "var", "obsm", "X_pca", "spatial"):
        assert key in out


def test_ls_long_shows_types_and_shapes(populated, capsys):
    list_store(populated, Console(width=200), long=True)
    out = capsys.readouterr().out
    assert "dataframe" in out
    assert "(3, 2)" in out


def test_ls_plain_emits_bare_paths(populated, capsys):
    list_store(populated, Console(), plain=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert "obsm/X_pca" in lines
    assert all(" " not in ln for ln in lines), "plain output must pipe cleanly"


def test_ls_can_start_below_a_path(populated, capsys):
    list_store(populated, Console(), entry_path="obsm", plain=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert lines == ["obsm/X_pca"]


def test_ls_of_a_dataset_shows_just_that_dataset(populated, capsys):
    list_store(populated, Console(width=200), entry_path="obsm/X_pca", long=True)
    out = capsys.readouterr().out
    assert "X_pca" in out


def test_ls_plain_of_a_dataset_prints_its_path(populated, capsys):
    list_store(populated, Console(), entry_path="obsm/X_pca", plain=True)
    assert capsys.readouterr().out.strip() == "obsm/X_pca"


def test_ls_depth_limits_recursion(populated, capsys):
    list_store(populated, Console(), depth=1, plain=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert "obsm" in lines
    assert "obsm/X_pca" not in lines


def test_ls_reports_a_missing_path(populated):
    with pytest.raises(KeyError, match="not found"):
        list_store(populated, Console(), entry_path="nope")


def test_walk_yields_nothing_for_a_dataset(populated):
    with open_store(populated, "r") as handle:
        assert list(walk(handle.root["obsm"]["X_pca"])) == []


# ---------------------------------------------------------------------------
# merge strategies


@pytest.mark.parametrize(
    "strategy,values,expected",
    [
        (None, ["a", "a"], (False, None)),
        ("same", ["a", "a"], (True, "a")),
        ("same", ["a", "b"], (False, None)),
        ("same", ["a", _MISSING], (False, None)),
        ("unique", ["a", "a"], (True, "a")),
        ("unique", ["a", _MISSING], (True, "a")),
        ("unique", ["a", "b"], (False, None)),
        ("first", ["a", "b"], (True, "a")),
        ("first", [_MISSING, "b"], (True, "b")),
        ("only", ["a", _MISSING], (True, "a")),
        ("only", ["a", "b"], (False, None)),
        ("only", ["a", "a"], (False, None)),
        ("same", [_MISSING, _MISSING], (False, None)),
    ],
)
def test_merge_strategies(strategy, values, expected):
    assert _merge_values(list(values), strategy) == expected


def test_unknown_merge_strategy_is_reported():
    with pytest.raises(ValueError, match="Unknown merge strategy"):
        _merge_values(["a"], "bogus")


def test_documented_strategies_all_work():
    for strategy in MERGE_STRATEGIES:
        _merge_values(["a", "a"], strategy)


# ---------------------------------------------------------------------------
# CLI error paths


def test_view_reports_a_missing_entry(new_store):
    path, _ = new_store()
    result = runner.invoke(app, ["view", str(path), "nope"])
    assert "not found" in _out(result)


def test_export_reports_a_missing_entry(new_store):
    path, _ = new_store()
    result = runner.invoke(app, ["export", "dict", str(path), "nope"])
    assert result.exit_code == 1
    assert "not found" in _out(result)


def test_import_reports_a_missing_output_flag(new_store, temp_dir):
    path, _ = new_store()
    source = temp_dir / "in.json"
    source.write_text("{}")
    for sub, args in (
        ("dict", ["uns/j", str(source)]),
        ("array", ["obsm/a", str(source)]),
        ("sparse", ["X", str(source)]),
        ("image", ["uns/i", str(source)]),
    ):
        result = runner.invoke(app, ["import", sub, str(path), *args])
        assert result.exit_code == 1
        assert "Output file is required" in _out(result)


def test_import_dataframe_reports_a_missing_output_flag(new_store, temp_dir):
    path, _ = new_store()
    source = temp_dir / "in.csv"
    source.write_text("a\n1\n")
    result = runner.invoke(app, ["import", "dataframe", str(path), "obs", str(source)])
    assert result.exit_code == 1
    assert "Output file is required" in _out(result)


def test_create_rejects_a_bad_zarr_format(temp_dir):
    result = runner.invoke(
        app,
        ["create", str(temp_dir / "x.zarr"), "--n-obs", "2", "--n-var", "2",
         "--zarr-format", "4"],
    )
    assert result.exit_code == 1
    assert "must be 2 or 3" in _out(result)


def test_create_requires_a_size_or_a_name_file(temp_dir):
    result = runner.invoke(app, ["create", str(temp_dir / "x.h5ad"), "--n-var", "2"])
    assert result.exit_code == 1
    assert "--n-obs" in _out(result)


def test_create_rejects_a_negative_size(temp_dir):
    result = runner.invoke(
        app, ["create", str(temp_dir / "x.h5ad"), "--n-obs", "-1", "--n-var", "2"]
    )
    assert result.exit_code == 1
    assert "must not be negative" in _out(result)


def test_create_rejects_duplicate_names(temp_dir):
    names = temp_dir / "dup.txt"
    names.write_text("a\na\n")
    result = runner.invoke(
        app,
        ["create", str(temp_dir / "x.h5ad"), "--obs-names", str(names), "--n-var", "2"],
    )
    assert result.exit_code == 1
    assert "duplicate names" in _out(result)


def test_create_rejects_an_empty_name_file(temp_dir):
    names = temp_dir / "empty.txt"
    names.write_text("\n\n")
    result = runner.invoke(
        app,
        ["create", str(temp_dir / "x.h5ad"), "--obs-names", str(names), "--n-var", "2"],
    )
    assert result.exit_code == 1
    assert "no names" in _out(result)


def test_split_rejects_a_bad_axis(new_store, temp_dir):
    path, _ = new_store()
    result = runner.invoke(
        app, ["split", str(path), "--by", "x", "-o", str(temp_dir / "o"), "--axis", "z"]
    )
    assert result.exit_code == 1
    assert "'obs' or 'var'" in _out(result)


def test_ls_reports_a_missing_store():
    result = runner.invoke(app, ["ls", "does-not-exist.h5ad"])
    assert result.exit_code != 0


def test_version_flag_prints_only_the_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert re.fullmatch(r"\d+\.\d+\.\d+\S*", result.stdout.strip())


def test_norm_path_rejects_an_empty_path():
    from adata.util.path import norm_path

    with pytest.raises(ValueError, match="non-empty"):
        norm_path("   ")

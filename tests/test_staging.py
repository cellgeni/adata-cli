"""Tests for `adata.util.staging`, the write-beside-then-rename helper.

Every command that writes a store goes through it, so its failure paths are
tested here directly rather than through each command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adata.util.staging import staged_outputs


def _names(directory: Path) -> list:
    return sorted(p.name for p in directory.iterdir())


def test_staging_replaces_an_existing_directory_rather_than_nesting_in_it(tmp_path):
    out = tmp_path / "out.zarr"
    out.mkdir()
    (out / "old").write_text("old")

    with staged_outputs([out]) as (temp,):
        temp.mkdir()
        (temp / "new").write_text("new")

    assert _names(out) == ["new"]
    assert _names(tmp_path) == ["out.zarr"]


def test_staging_moves_several_outputs_together(tmp_path):
    outs = [tmp_path / "a.h5ad", tmp_path / "b.h5ad"]
    with staged_outputs(outs) as temps:
        assert [t.suffix for t in temps] == [".h5ad", ".h5ad"]
        for temp in temps:
            temp.write_text(temp.name)

    assert [p.read_text() for p in outs] == ["a.adata-tmp.h5ad", "b.adata-tmp.h5ad"]


def test_a_failure_while_writing_keeps_the_existing_output(tmp_path):
    out = tmp_path / "out.h5ad"
    out.write_text("old")

    with pytest.raises(RuntimeError):
        with staged_outputs([out]) as (temp,):
            temp.write_text("half")
            raise RuntimeError("interrupted")

    assert out.read_text() == "old"
    assert _names(tmp_path) == ["out.h5ad"]


def test_a_leftover_backup_is_refused_not_deleted(tmp_path):
    """After a run killed mid-swap, the backup may be the only copy."""
    out = tmp_path / "out.h5ad"
    backup = tmp_path / "out.adata-old.h5ad"
    backup.write_text("the only copy")

    with pytest.raises(FileExistsError, match="adata-old"):
        with staged_outputs([out]):
            pytest.fail("the body must not run")

    assert backup.read_text() == "the only copy"


def test_a_leftover_temporary_is_refused(tmp_path):
    out = tmp_path / "out.h5ad"
    (tmp_path / "out.adata-tmp.h5ad").write_text("stale")

    with pytest.raises(FileExistsError, match="adata-tmp"):
        with staged_outputs([out]):
            pytest.fail("the body must not run")


def test_a_failed_swap_puts_back_the_outputs_already_swapped(tmp_path, monkeypatch):
    outs = [tmp_path / "a.h5ad", tmp_path / "b.h5ad"]
    for out in outs:
        out.write_text(f"old {out.stem}")

    real_rename = Path.rename

    def rename(self, target):
        if self.name == "b.adata-tmp.h5ad":
            raise OSError("disk full")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)

    with pytest.raises(OSError, match="disk full"):
        with staged_outputs(outs) as temps:
            for temp in temps:
                temp.write_text(f"new {temp.name}")

    assert [p.read_text() for p in outs] == ["old a", "old b"]
    assert _names(tmp_path) == ["a.h5ad", "b.h5ad"]


def test_a_zarr_store_named_without_the_suffix_stays_zarr(tmp_path):
    """`store.adata-tmp` would not exist yet, and so be opened as HDF5."""
    out = tmp_path / "store"
    out.mkdir()
    (out / "zarr.json").write_text("{}")

    with staged_outputs([out]) as (temp,):
        assert temp.suffix == ".zarr"
        temp.mkdir()
        (temp / "zarr.json").write_text('{"new": true}')

    assert (out / "zarr.json").read_text() == '{"new": true}'
    assert _names(tmp_path) == ["store"]

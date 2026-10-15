"""Failures in `subset`, `recast` and `split` must not cost the user data.

Each test reproduces a way these commands used to destroy or corrupt a store
while reporting success, or leave a half-written one behind (issue #35):

- `subset x.zarr -o x.zarr` emptied the store: the output was opened "w"
  before the selection was read from the source it had just cleared.
- A selection that failed still left an empty output, because the output was
  opened before the query was evaluated.
- `--inplace` deleted the source and then moved the new store in, so a crash
  between the two lost both.
- `split` wrote straight into its output directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app

ad = pytest.importorskip("anndata", reason="anndata is required for these tests")
pd = pytest.importorskip("pandas")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _store(path: Path) -> Path:
    obj = ad.AnnData(
        X=np.arange(16, dtype="float32").reshape(4, 4),
        obs=pd.DataFrame(
            {"grp": pd.Categorical(["a", "b", "a", "b"])},
            index=[f"c{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    if path.suffix == ".zarr":
        obj.write_zarr(path)
    else:
        obj.write_h5ad(path)
    return path


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _debris(directory: Path) -> list:
    return sorted(p.name for p in directory.glob("*adata-*"))


def _fail(*args, **kwargs):
    raise RuntimeError("interrupted while writing")


@pytest.fixture(params=[".h5ad", ".zarr"])
def suffix(request) -> str:
    return request.param


# ---------------------------------------------------------------------------
# subset


def test_subset_onto_its_own_input_is_refused(tmp_path, suffix):
    store = _store(tmp_path / f"in{suffix}")

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(store), "-q", "grp == a"]
    )
    assert result.exit_code == 1
    assert "Output path is the input" in _out(result)
    assert _read(store).shape == (4, 4)


def test_subset_a_failed_selection_writes_nothing(tmp_path, suffix):
    store = _store(tmp_path / f"in{suffix}")
    out = tmp_path / f"out{suffix}"

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "-q", "missing == 1"]
    )
    assert result.exit_code == 1
    assert not out.exists()
    assert _debris(tmp_path) == []


def test_subset_a_failed_selection_keeps_an_existing_output(tmp_path, suffix):
    store = _store(tmp_path / f"in{suffix}")
    out = _store(tmp_path / f"out{suffix}")

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "-q", "missing == 1"]
    )
    assert result.exit_code == 1
    assert _read(out).shape == (4, 4)


def test_subset_still_replaces_an_existing_output(tmp_path, suffix):
    store = _store(tmp_path / f"in{suffix}")
    out = _store(tmp_path / f"out{suffix}")

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "-q", "grp == a"]
    )
    assert result.exit_code == 0, _out(result)
    assert list(_read(out).obs_names) == ["c0", "c2"]
    assert _debris(tmp_path) == []


def test_subset_inplace_replaces_the_source(tmp_path, suffix):
    store = _store(tmp_path / f"in{suffix}")

    result = runner.invoke(app, ["subset", str(store), "--inplace", "-q", "grp == a"])
    assert result.exit_code == 0, _out(result)
    assert list(_read(store).obs_names) == ["c0", "c2"]
    assert _debris(tmp_path) == []


def test_subset_inplace_keeps_the_source_if_writing_fails(
    tmp_path, suffix, monkeypatch
):
    from adata.core import subset as subset_mod

    store = _store(tmp_path / f"in{suffix}")
    monkeypatch.setattr(subset_mod, "_write_targets", _fail)

    result = runner.invoke(app, ["subset", str(store), "--inplace", "-q", "grp == a"])
    assert result.exit_code == 1
    assert "interrupted" in _out(result)
    assert _read(store).shape == (4, 4)
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# recast


def test_recast_inplace_replaces_the_source(tmp_path, suffix):
    store = _store(tmp_path / f"in{suffix}")

    result = runner.invoke(
        app, ["recast", str(store), "X", "--inplace", "--dtype", "float64"]
    )
    assert result.exit_code == 0, _out(result)
    assert _read(store).X.dtype == np.dtype("float64")
    assert _debris(tmp_path) == []


def test_recast_inplace_keeps_the_source_if_writing_fails(
    tmp_path, suffix, monkeypatch
):
    from adata.commands import recast as recast_mod

    store = _store(tmp_path / f"in{suffix}")
    monkeypatch.setattr(recast_mod, "_write", _fail)

    result = runner.invoke(
        app, ["recast", str(store), "X", "--inplace", "--dtype", "float64"]
    )
    assert result.exit_code == 1
    got = _read(store)
    assert got.X.dtype == np.dtype("float32")
    assert np.array_equal(got.X, np.arange(16, dtype="float32").reshape(4, 4))
    assert _debris(tmp_path) == []


def test_recast_keeps_an_existing_output_if_writing_fails(
    tmp_path, suffix, monkeypatch
):
    from adata.commands import recast as recast_mod

    store = _store(tmp_path / f"in{suffix}")
    out = _store(tmp_path / f"out{suffix}")
    monkeypatch.setattr(recast_mod, "_write", _fail)

    result = runner.invoke(
        app, ["recast", str(store), "X", "-o", str(out), "--dtype", "float64"]
    )
    assert result.exit_code == 1
    assert _read(out).X.dtype == np.dtype("float32")
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# split


def test_split_leaves_nothing_behind_if_writing_fails(tmp_path, monkeypatch):
    from adata.core import subset as subset_mod

    store = _store(tmp_path / "in.h5ad")
    out_dir = tmp_path / "parts"
    out_dir.mkdir()
    (out_dir / "a.h5ad").write_text("an older split")
    monkeypatch.setattr(subset_mod, "_write_targets", _fail)

    result = runner.invoke(app, ["split", str(store), "--by", "grp", "-o", str(out_dir)])
    assert result.exit_code == 1
    assert sorted(p.name for p in out_dir.iterdir()) == ["a.h5ad"]
    assert (out_dir / "a.h5ad").read_text() == "an older split"


def test_split_onto_its_own_input_is_refused(tmp_path):
    # Splitting a.h5ad on a column with a value "a", into its own directory.
    store = _store(tmp_path / "a.h5ad")

    result = runner.invoke(app, ["split", str(store), "--by", "grp", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "Output path is the input" in _out(result)
    assert _read(store).shape == (4, 4)
    assert not (tmp_path / "b.h5ad").exists()

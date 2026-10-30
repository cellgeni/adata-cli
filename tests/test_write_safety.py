"""Failures in any command that writes must not cost the user data.

Each test reproduces a way a command used to destroy or corrupt a store
while reporting success, or leave a half-written one behind (issue #35, and
the write-safety findings of issue #89):

- `subset x.zarr -o x.zarr` emptied the store: the output was opened "w"
  before the selection was read from the source it had just cleared.
- A selection that failed still left an empty output, because the output was
  opened before the query was evaluated.
- `--inplace` deleted the source and then moved the new store in, so a crash
  between the two lost both.
- `split` wrote straight into its output directory.
- A failed HDF5 close or Zarr consolidation was swallowed, so an unreadable
  store was renamed into place with exit 0 (F003, F070).
- export, import and create wrote in place or over their input (F023, F024,
  F074), and -o could name a folder, which was deleted (F037).
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


def _flat(result) -> str:
    """The output with every space removed: Rich wraps a long path inside
    itself at the terminal's width (80 columns in CI)."""
    return "".join(_out(result).split())


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


# ---------------------------------------------------------------------------
# concat


def test_concat_leaves_nothing_behind_if_writing_fails(tmp_path, suffix, monkeypatch):
    from adata.core import concat as concat_mod

    a = _store(tmp_path / f"a{suffix}")
    b = _store(tmp_path / f"b{suffix}")
    out = tmp_path / f"m{suffix}"
    monkeypatch.setattr(concat_mod, "_write_matrices", _fail)

    result = runner.invoke(app, ["concat", str(a), str(b), "-o", str(out)])
    assert result.exit_code == 1
    assert not out.exists()
    assert _debris(tmp_path) == []


# ===========================================================================
# Issue #89: every writer, every failure point.

import os
import signal
import stat
import subprocess
import sys
import threading

import h5py
import zarr

from tests.conftest import backend_path, backend_zarr_format


def _rich_store(path: Path, zarr_format=None) -> Path:
    """A small store with an element of every kind the commands touch."""
    sp = pytest.importorskip("scipy.sparse")
    obj = ad.AnnData(
        X=np.arange(16, dtype="float32").reshape(4, 4),
        obs=pd.DataFrame(
            {"grp": pd.Categorical(["a", "b", "a", "b"])},
            index=[f"c{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        layers={"counts": sp.csr_matrix(np.eye(4, dtype="float32"))},
        obsm={"X_pca": np.ones((4, 2), dtype="float32")},
        uns={"k": 1, "img": np.zeros((3, 3, 3), dtype="uint8")},
    )
    if path.suffix == ".zarr":
        with ad.settings.override(zarr_write_format=zarr_format or 3):
            obj.write_zarr(path)
    else:
        obj.write_h5ad(path)
    return path


def _source(tmp_path: Path, backend: str, stem: str = "in") -> Path:
    return _rich_store(backend_path(tmp_path, backend, stem), backend_zarr_format(backend))


def _snapshot(path: Path):
    """Every byte of a file or store, to show a refusal touched nothing."""
    path = Path(path)
    if path.is_dir():
        return {
            str(p.relative_to(path)): p.read_bytes()
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
    return path.read_bytes()


def _version(path: Path) -> int:
    return zarr.open_group(str(path), mode="r").metadata.zarr_format


def _npy(tmp_path: Path, shape=(4, 2)) -> Path:
    path = tmp_path / "emb.npy"
    np.save(path, np.full(shape, 7, dtype="float32"))
    return path


#: Commands that rewrite their source with --inplace, as argument lists
#: built from (source, tmp_path). Each must leave the source byte-identical
#: whenever it fails.
INPLACE = {
    "subset": lambda s, t: ["subset", s, "--inplace", "-q", "grp == a"],
    "recast": lambda s, t: ["recast", s, "X", "--inplace", "--dtype", "float64"],
    "import": lambda s, t: ["import", "array", s, "obsm/X_new", _npy(t), "--inplace"],
}


def _args(argv) -> list:
    return [str(a) for a in argv]


# ---------------------------------------------------------------------------
# F003, F070: a failure at close


def _failing_hdf5_close(monkeypatch):
    """h5py's close raising, as it does when the final flush hits a full disk."""
    real = h5py.File.close

    def close(self):
        writing = bool(self) and self.mode != "r"
        real(self)
        if writing:
            raise RuntimeError(
                "Can't decrement id ref count (unable to extend file "
                "properly, errno = 28, error message = 'No space left on device')"
            )

    monkeypatch.setattr(h5py.File, "close", close)


def _failing_consolidation(monkeypatch):
    from adata import storage

    def consolidate(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage.zarr, "consolidate_metadata", consolidate)


def _failing_close(monkeypatch, backend: str) -> None:
    if backend == "h5ad":
        _failing_hdf5_close(monkeypatch)
    else:
        _failing_consolidation(monkeypatch)


@pytest.mark.parametrize("command", sorted(INPLACE))
def test_a_failed_close_keeps_the_inplace_source(tmp_path, backend, monkeypatch, command):
    """F003, F070: Store.close swallowed a failed HDF5 flush or Zarr
    consolidation, so staging renamed an unreadable (or, for Zarr v2, stale)
    store over the source and the command exited 0."""
    src = _source(tmp_path, backend)
    before = _snapshot(src)
    _failing_close(monkeypatch, backend)

    result = runner.invoke(app, _args(INPLACE[command](src, tmp_path)))
    assert result.exit_code == 1, _out(result)
    assert "Could not finish writing" in _out(result)
    assert _snapshot(src) == before
    assert _read(src).shape == (4, 4)
    assert _debris(tmp_path) == []


def test_a_failed_close_keeps_existing_outputs(tmp_path, backend, monkeypatch):
    """F003: subset, recast, split, import -o and create --force all
    promoted the unreadable store over whatever output already existed."""
    src = _source(tmp_path, backend)
    out = _source(tmp_path, backend, "out")
    parts = tmp_path / "parts"
    parts.mkdir()
    part = _source(parts, backend, "a")
    kept = {p: _snapshot(p) for p in (src, out, part)}
    _failing_close(monkeypatch, backend)

    for argv in (
        ["subset", src, "-o", out, "-q", "grp == a"],
        ["recast", src, "X", "-o", out, "--dtype", "float64"],
        ["import", "array", src, "obsm/X_new", _npy(tmp_path), "-o", out],
        ["create", out, "--n-obs", "2", "--n-var", "2", "--force"],
        ["split", src, "--by", "grp", "-o", parts, "--suffix", part.suffix],
    ):
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 1, (argv, _out(result))
        assert {p: _snapshot(p) for p in kept} == kept, argv
    assert _debris(tmp_path) == [] and _debris(parts) == []
    assert not (parts / f"b{part.suffix}").exists()


def test_store_close_raises_unless_an_exception_is_already_on_its_way(
    tmp_path, backend, monkeypatch
):
    """F003: the close error was dropped; and while another exception is
    propagating, that one must still be the one the user sees."""
    from adata.storage import open_store

    _failing_close(monkeypatch, backend)
    path = backend_path(tmp_path, backend)
    with pytest.raises(OSError, match="Could not finish writing"):
        with open_store(path, "w", zarr_format=backend_zarr_format(backend)):
            pass
    with pytest.raises(KeyError, match="first"):
        with open_store(path, "w", zarr_format=backend_zarr_format(backend)):
            raise KeyError("first")


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE is POSIX")
def test_a_full_disk_at_the_final_flush_fails_rather_than_replacing_the_source(
    tmp_path,
):
    """F003, reproduced without any patching: under a file-size limit that
    lets the data through but not the metadata flushed at close,
    `subset --inplace` exited 0 and left the source unreadable."""
    src = _rich_store(tmp_path / "in.h5ad")
    before = _snapshot(src)
    limit = src.stat().st_size // 2
    script = (
        "import resource, signal, sys\n"
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({limit}, {limit}))\n"
        "sys.argv = ['adata'] + sys.argv[1:]\n"
        "from adata.cli import main\n"
        "main()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, "subset", str(src), "--inplace", "-q", "grp == a"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr[-2000:]
    assert _snapshot(src) == before
    assert _read(src).shape == (4, 4)
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# F015: --chunk


def _chunk_options():
    """Every (command path, --chunk option) in the CLI, hidden ones too."""
    import typer

    found = []

    def walk(command, path):
        for param in getattr(command, "params", []):
            if "--chunk" in getattr(param, "opts", []):
                found.append((" ".join(path), param))
        for name, sub in getattr(command, "commands", {}).items():
            walk(sub, path + [name])

    walk(typer.main.get_command(app), [])
    return found


def test_every_chunk_option_refuses_zero_and_below():
    """F015: subset and split took --chunk -1, skipped the copy loop, and
    wrote an all-zero dense X with exit 0; --chunk 0 failed with range()'s
    message. Every command's --chunk now shares one bounded definition."""
    options = _chunk_options()
    commands = {path for path, _ in options}
    assert {
        "subset", "split", "concat", "recast", "convert",
        "export dataframe", "export array", "export sparse", "import array",
    } <= commands
    for path, param in options:
        assert getattr(param.type, "min", None) == 1, path


@pytest.mark.parametrize("chunk", ["0", "-1"])
@pytest.mark.parametrize("command", ["subset", "split", "recast"])
def test_a_chunk_below_one_is_refused_before_anything_is_written(
    tmp_path, backend, chunk, command
):
    """F015: with --inplace the source's dense matrices were replaced by zeros."""
    src = _source(tmp_path, backend)
    before = _snapshot(src)
    argv = {
        "subset": ["subset", src, "--inplace", "-q", "grp == a"],
        "recast": ["recast", src, "X", "--inplace", "--dtype", "float64"],
        "split": ["split", src, "--by", "grp", "-o", tmp_path / "parts"],
    }[command] + ["--chunk", chunk]

    result = runner.invoke(app, _args(argv))
    assert result.exit_code == 2
    assert "x>=1" in _out(result)
    assert _snapshot(src) == before
    assert not (tmp_path / "parts").exists()


def test_the_library_refuses_a_chunk_below_one_too(tmp_path):
    """F015: the CLI bound does not cover a direct caller."""
    from adata.commands.recast import recast_store
    from adata.commands.split import split_store
    from adata.core.subset import subset_h5ad
    from rich.console import Console

    src = _rich_store(tmp_path / "in.h5ad")
    quiet = Console(quiet=True)
    with pytest.raises(ValueError, match="--chunk"):
        subset_h5ad(src, tmp_path / "o.h5ad", None, None, obs_query="grp == a",
                    chunk_rows=0, console=quiet)
    with pytest.raises(ValueError, match="--chunk"):
        split_store(src, "grp", tmp_path / "p", quiet, chunk_rows=-1)
    with pytest.raises(ValueError, match="--chunk"):
        recast_store(src, ["X"], tmp_path / "o.h5ad", quiet, dtype="float64",
                     chunk_rows=0)
    assert _debris(tmp_path) == [] and not (tmp_path / "o.h5ad").exists()


# ---------------------------------------------------------------------------
# F016: an empty selection


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_a_name_file_matching_nothing_never_replaces_the_source(tmp_path, backend, axis):
    """F016: barcodes without the index's "-1" matched nothing; subset wrote
    a 0-row store and, with --inplace, swapped it in for the source."""
    src = _source(tmp_path, backend)
    before = _snapshot(src)
    names = tmp_path / "names.txt"
    names.write_text("c0-1\nc1-1\n" if axis == "obs" else "g0-1\n")

    result = runner.invoke(app, _args(["subset", src, "--inplace", f"--{axis}", names]))
    assert result.exit_code == 1
    assert f"No {axis} selected" in _out(result)
    assert _snapshot(src) == before
    assert _debris(tmp_path) == []


def test_an_empty_name_file_writes_nothing(tmp_path, backend):
    """F016: an empty list selected nothing and wrote an empty store."""
    src = _source(tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")
    names = tmp_path / "names.txt"
    names.write_text("\n\n")

    result = runner.invoke(app, _args(["subset", src, "-o", out, "--obs", names]))
    assert result.exit_code == 1
    assert "No obs selected" in _out(result)
    assert not out.exists()
    assert _debris(tmp_path) == []


def test_repeated_names_in_a_name_file_select_each_row_once(tmp_path, backend):
    """F093: repeated names in a list were untested; they select each row
    once, as anndata's `obs_names.isin` does."""
    src = _source(tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")
    names = tmp_path / "names.txt"
    names.write_text("c2\nc0\nc2\nc0\n")

    result = runner.invoke(app, _args(["subset", src, "-o", out, "--obs", names]))
    assert result.exit_code == 0, _out(result)
    full = _read(src)
    expected = full[full.obs_names.isin(["c0", "c2"])]
    got = _read(out)
    assert list(got.obs_names) == list(expected.obs_names)
    assert np.array_equal(got.X, expected.X)


@pytest.mark.filterwarnings("ignore:.*not unique")
def test_repeated_obs_names_in_the_store_subset_as_anndata_does(tmp_path, backend):
    """F093: a store whose index repeats a name was never subset in a test."""
    path = backend_path(tmp_path, backend)
    obj = ad.AnnData(
        X=np.arange(8, dtype="float32").reshape(4, 2),
        obs=pd.DataFrame({"grp": pd.Categorical(["a", "b", "a", "b"])},
                         index=["x", "x", "y", "y"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    if path.suffix == ".zarr":
        with ad.settings.override(zarr_write_format=backend_zarr_format(backend)):
            obj.write_zarr(path)
    else:
        obj.write_h5ad(path)
    out = backend_path(tmp_path, backend, "out")

    result = runner.invoke(app, _args(["subset", path, "-o", out, "-q", "grp == a"]))
    assert result.exit_code == 0, _out(result)
    got = _read(out)
    expected = obj[obj.obs["grp"] == "a"]
    assert list(got.obs_names) == list(expected.obs_names)
    assert np.array_equal(got.X, expected.X)


# ---------------------------------------------------------------------------
# F022: a full disk, checked for first


def _tiny_disk(monkeypatch, free: int = 1):
    import shutil as shutil_mod
    from adata.util import staging

    usage = shutil_mod.disk_usage(".")

    def disk_usage(path):
        return type(usage)(usage.total, usage.total - free, free)

    monkeypatch.setattr(staging.shutil, "disk_usage", disk_usage)


def test_every_store_writer_checks_free_space_first(tmp_path, monkeypatch):
    """F022: a full disk part way through an HDF5 write can abort the process
    inside the library, skipping all cleanup. Every writer that can produce
    a large HDF5 store now checks the disk first, and refuses before writing."""
    src = _rich_store(tmp_path / "in.h5ad")
    other = _rich_store(tmp_path / "other.h5ad")
    before = _snapshot(src)
    _tiny_disk(monkeypatch)
    out = tmp_path / "out.h5ad"

    for argv in (
        ["subset", src, "--inplace", "-q", "grp == a"],
        ["recast", src, "X", "-o", out, "--dtype", "float64"],
        ["split", src, "--by", "grp", "-o", tmp_path / "parts"],
        ["concat", src, other, "-o", out],
        ["import", "array", src, "obsm/X_new", _npy(tmp_path), "--inplace"],
    ):
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 1, (argv, _out(result))
        assert "Not enough free space" in _out(result), argv
        assert "--no-space-check" in _out(result)
        assert _snapshot(src) == before
        assert not out.exists()
        assert not list((tmp_path / "parts").glob("*.*")) if (tmp_path / "parts").exists() else True
    assert _debris(tmp_path) == []


def test_a_zarr_output_is_not_measured(tmp_path, monkeypatch):
    """A Zarr store that runs out of space fails with an ordinary error and
    is cleaned up, so it is not checked: walking a source of millions of
    chunk files to size it would cost more than it saves."""
    from adata.util import staging

    src = _rich_store(tmp_path / "in.zarr")
    _tiny_disk(monkeypatch)
    monkeypatch.setattr(staging, "disk_size", _fail)

    result = runner.invoke(app, _args(["subset", src, "-o", tmp_path / "o.zarr",
                                       "-q", "grp == a"]))
    assert result.exit_code == 0, _out(result)
    assert _read(tmp_path / "o.zarr").shape == (2, 4)


def test_no_space_check_writes_anyway(tmp_path, monkeypatch):
    """F022: the estimate can be wrong, so the check can be skipped."""
    src = _rich_store(tmp_path / "in.h5ad")
    _tiny_disk(monkeypatch)

    result = runner.invoke(
        app, _args(["subset", src, "-o", tmp_path / "o.h5ad", "-q", "grp == a",
                    "--no-space-check"])
    )
    assert result.exit_code == 0, _out(result)
    assert _read(tmp_path / "o.h5ad").shape == (2, 4)


def test_the_subset_estimate_follows_the_selection(tmp_path, monkeypatch):
    """F022: refusing a 1% subset of a store for want of the whole store's
    size would make the check useless on big inputs."""
    from adata.core import subset as subset_mod
    from adata.util.staging import disk_size

    src = _rich_store(tmp_path / "in.h5ad")
    seen = []
    monkeypatch.setattr(
        subset_mod, "check_free_space", lambda out, needed, check: seen.append(needed())
    )
    for query in ("grp == a", "grp in a,b"):
        runner.invoke(app, _args(["subset", src, "-o", tmp_path / f"{len(seen)}.h5ad",
                                  "-q", query]))
    assert seen == [pytest.approx(disk_size(src) / 2), disk_size(src)]


def test_the_recast_estimate_counts_only_the_matrix_that_grows(tmp_path, monkeypatch):
    """F022 review: the estimate multiplied the whole store by one matrix's
    growth, asking for 10 GB to densify a sparse layer into an 81 MB store."""
    from adata.commands import recast as recast_mod
    from adata.util.staging import disk_size

    sp = pytest.importorskip("scipy.sparse")
    src = tmp_path / "in.h5ad"
    ad.AnnData(
        X=np.random.default_rng(0).random((300, 200), dtype="float32"),
        layers={"small": sp.random(300, 200, density=0.001, format="csr",
                                   dtype="float32", random_state=0)},
    ).write_h5ad(src)
    seen = []
    monkeypatch.setattr(recast_mod, "check_free_space",
                        lambda out, needed, check: seen.append(needed()))

    out = tmp_path / "dense.h5ad"
    result = runner.invoke(app, _args(["recast", src, "layers/small", "-o", out,
                                       "--layout", "dense", "--force"]))
    assert result.exit_code == 0, _out(result)
    assert disk_size(out) * 0.8 < seen[0] < disk_size(out) * 1.5


# ---------------------------------------------------------------------------
# F023: export


EXPORTS = {
    "dataframe": lambda s: ["export", "dataframe", s, "obs"],
    "array": lambda s: ["export", "array", s, "obsm/X_pca"],
    "sparse": lambda s: ["export", "sparse", s, "layers/counts"],
    "dict": lambda s: ["export", "dict", s, "uns"],
    "image": lambda s: ["export", "image", s, "uns/img"],
}
SUFFIX = {"dataframe": ".csv", "array": ".npy", "sparse": ".mtx", "dict": ".json",
          "image": ".png"}


@pytest.mark.parametrize("kind", sorted(EXPORTS))
def test_export_onto_its_own_input_is_refused(tmp_path, backend, kind):
    """F023: `export sparse c.h5ad X -o c.h5ad` replaced the store with an
    80-byte MTX header and exited 0."""
    src = _source(tmp_path, backend)
    before = _snapshot(src)

    result = runner.invoke(app, _args(EXPORTS[kind](src) + ["-o", src]))
    assert result.exit_code == 1
    assert "Output path is the input" in _out(result)
    assert _snapshot(src) == before


@pytest.mark.parametrize("kind", sorted(EXPORTS))
def test_export_into_a_zarr_input_is_refused(tmp_path, kind):
    """F023: a file written inside a Zarr input lands among its chunks."""
    src = _rich_store(tmp_path / "in.zarr")
    before = _snapshot(src)

    result = runner.invoke(
        app, _args(EXPORTS[kind](src) + ["-o", src / "obs" / f"out{SUFFIX[kind]}"])
    )
    assert result.exit_code == 1
    assert "Output path is the input" in _out(result)
    assert _snapshot(src) == before


@pytest.mark.parametrize("kind", sorted(EXPORTS))
def test_a_failed_export_keeps_an_existing_output(tmp_path, kind, monkeypatch):
    """F023: an export refused part way left a one-line partial CSV where the
    user's previous file had been."""
    from adata.commands import export as export_mod

    src = _rich_store(tmp_path / "in.h5ad")
    out = tmp_path / f"out{SUFFIX[kind]}"
    out.write_bytes(b"the previous export")

    name = {"dataframe": "export_dataframe", "array": "export_npy_format",
            "sparse": "export_mtx_format", "dict": "export_json_format",
            "image": "export_image_format"}[kind]
    real = getattr(export_mod, name)

    def half_then_fail(*args, **kwargs):
        Path(kwargs["out"]).write_bytes(b"half")
        raise RuntimeError("interrupted while writing")

    monkeypatch.setattr(export_mod, name, half_then_fail)
    result = runner.invoke(app, _args(EXPORTS[kind](src) + ["-o", out]))
    assert result.exit_code == 1
    assert out.read_bytes() == b"the previous export"
    assert _debris(tmp_path) == []

    monkeypatch.setattr(export_mod, name, real)
    result = runner.invoke(app, _args(EXPORTS[kind](src) + ["-o", out]))
    assert result.exit_code == 0, _out(result)
    assert out.read_bytes() != b"the previous export"
    assert _debris(tmp_path) == []


def test_a_refused_column_keeps_an_existing_csv(tmp_path):
    """F023: `export dataframe base.h5ad obs -c grp,typo -o obs.csv` replaced
    obs.csv with its header, then refused."""
    src = _rich_store(tmp_path / "in.h5ad")
    out = tmp_path / "obs.csv"
    out.write_text("_index,grp\nc0,a\n")

    result = runner.invoke(app, _args(["export", "dataframe", src, "obs",
                                       "-c", "grp,typo", "-o", out]))
    assert result.exit_code == 1
    assert out.read_text() == "_index,grp\nc0,a\n"
    assert _debris(tmp_path) == []


def test_export_to_a_directory_is_refused(tmp_path):
    """F023, F037: -o naming a folder replaced it."""
    src = _rich_store(tmp_path / "in.h5ad")
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep me")

    result = runner.invoke(app, _args(["export", "dataframe", src, "obs", "-o", folder]))
    assert result.exit_code == 1
    assert "is a directory" in _out(result)
    assert (folder / "notes.txt").read_text() == "keep me"


def test_export_writes_through_a_staged_file(tmp_path):
    """The output appears whole: pandas reads what export wrote."""
    src = _rich_store(tmp_path / "in.h5ad")
    out = tmp_path / "obs.csv"

    result = runner.invoke(app, _args(["export", "dataframe", src, "obs", "-o", out]))
    assert result.exit_code == 0, _out(result)
    assert list(pd.read_csv(out, index_col=0).index) == ["c0", "c1", "c2", "c3"]
    assert "Wrote" in _out(result) and "adata-tmp" not in _flat(result)
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# F024: import


def test_a_failed_inplace_import_keeps_the_source(tmp_path, backend, monkeypatch):
    """F024: `import dict data.h5ad uns p.json --inplace` deleted uns, wrote
    one key and failed on the next: every original key was gone."""
    from adata.commands import import_data

    src = _source(tmp_path, backend)
    before = _snapshot(src)
    payload = tmp_path / "p.json"
    payload.write_text('{"a": 1, "b": 2}')

    def half_then_fail(root, obj, input_file, console):
        del root["uns"]
        root.create_group("uns")
        raise RuntimeError("expected bytes, str found")

    monkeypatch.setattr(import_data, "import_json", half_then_fail)
    result = runner.invoke(app, _args(["import", "dict", src, "uns", payload, "--inplace"]))
    assert result.exit_code == 1
    assert _snapshot(src) == before
    assert _read(src).uns["k"] == 1
    assert _debris(tmp_path) == []


def test_a_refused_import_keeps_an_existing_output(tmp_path, backend):
    """F024: `import sparse src.h5ad X bad.mtx -o prev.h5ad` copied the source
    over prev.h5ad before the MTX header was checked, then refused."""
    src = _source(tmp_path, backend)
    out = _source(tmp_path, backend, "prev")
    kept = _snapshot(out)
    bad = tmp_path / "bad.mtx"
    bad.write_text("not a matrix market file\n")

    result = runner.invoke(app, _args(["import", "sparse", src, "X", bad, "-o", out]))
    assert result.exit_code == 1
    assert _snapshot(out) == kept
    assert _debris(tmp_path) == []


def test_a_refused_import_leaves_no_copy_behind(tmp_path, backend):
    """F024: a refused import with -o left a full copy of the source there."""
    src = _source(tmp_path, backend)
    out = backend_path(tmp_path, backend, "new")
    bad = tmp_path / "bad.npy"
    np.save(bad, np.ones((9, 9), dtype="float32"))

    result = runner.invoke(app, _args(["import", "array", src, "obsm/X_bad", bad, "-o", out]))
    assert result.exit_code == 1
    assert not out.exists()
    assert _debris(tmp_path) == []


def test_a_failed_import_into_another_backend_keeps_the_existing_output(
    tmp_path, monkeypatch
):
    """F024: an h5ad source with `-o existing.zarr` opened it "w" and cleared
    it, unstaged."""
    from adata.commands import import_data

    src = _rich_store(tmp_path / "in.h5ad")
    out = _rich_store(tmp_path / "prev.zarr")
    kept = _snapshot(out)
    monkeypatch.setattr(import_data, "import_npy", _fail)

    result = runner.invoke(app, _args(["import", "array", src, "obsm/X_new",
                                       _npy(tmp_path), "-o", out]))
    assert result.exit_code == 1
    assert _snapshot(out) == kept
    assert _debris(tmp_path) == []


@pytest.mark.parametrize("inplace", [True, False])
def test_import_writes_a_store_anndata_reads(tmp_path, backend, inplace):
    """The staged import still lands: anndata sees the new element."""
    src = _source(tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")
    target = src if inplace else out

    result = runner.invoke(app, _args(
        ["import", "array", src, "obsm/X_new", _npy(tmp_path)]
        + (["--inplace"] if inplace else ["-o", out])
    ))
    assert result.exit_code == 0, _out(result)
    assert np.array_equal(_read(target).obsm["X_new"], np.full((4, 2), 7))
    if backend != "h5ad":
        assert _version(target) == backend_zarr_format(backend)
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# F037, F074: an output that is not a store


def _folder(tmp_path: Path, name: str = "results") -> Path:
    folder = tmp_path / name
    (folder / "sub").mkdir(parents=True)
    (folder / "notes.txt").write_text("keep me")
    (folder / "sub" / "y.txt").write_text("and me")
    return folder


def test_no_writer_replaces_a_folder_that_is_not_a_store(tmp_path, backend):
    """F037, F074: `subset data.h5ad -o results/` (as if -o were split's
    folder) deleted results/ and everything in it, and wrote an HDF5 file
    there; recast, import, create --force and convert did the same."""
    src = _source(tmp_path, backend)
    folder = _folder(tmp_path)
    kept = _snapshot(folder)

    for argv in (
        ["subset", src, "-o", folder, "-q", "grp == a"],
        ["recast", src, "X", "-o", folder, "--dtype", "float64"],
        ["import", "array", src, "obsm/X_new", _npy(tmp_path), "-o", folder],
        ["create", folder, "--n-obs", "2", "--n-var", "2", "--force"],
        ["concat", src, src, "-o", folder],
    ):
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 1, (argv, _out(result))
        assert _snapshot(folder) == kept, argv
    assert _debris(tmp_path) == []


def test_a_store_output_will_not_replace_a_text_file(tmp_path):
    """F037: `-o cells.txt` (a slip of the tab key) replaced the list."""
    src = _rich_store(tmp_path / "in.h5ad")
    cells = tmp_path / "cells.txt"
    cells.write_text("c0\nc2\n")

    result = runner.invoke(app, _args(["subset", src, "--obs", cells, "-o", cells]))
    assert result.exit_code == 1
    assert cells.read_text() == "c0\nc2\n"


def test_create_force_keeps_the_old_store_if_writing_fails(tmp_path):
    """F074: create --force deleted keep.h5ad, then failed on a name HDF5
    cannot hold, leaving a half-written store with no var."""
    keep = _rich_store(tmp_path / "keep.h5ad")
    kept = _snapshot(keep)
    names = tmp_path / "names.txt"
    names.write_text("a\x00b\nc\n")

    result = runner.invoke(app, _args(["create", keep, "--obs-names", names,
                                       "--n-var", "2", "--force"]))
    assert result.exit_code == 1
    assert _snapshot(keep) == kept
    assert _debris(tmp_path) == []


def test_create_force_replaces_a_store(tmp_path, backend):
    out = _source(tmp_path, backend, "out")

    result = runner.invoke(app, _args(["create", out, "--n-obs", "2", "--n-var", "3",
                                       "--force"]))
    assert result.exit_code == 0, _out(result)
    assert _read(out).shape == (2, 3)
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# F073: symlinks and permissions


@pytest.mark.parametrize("command", sorted(INPLACE))
def test_inplace_through_a_symlink_replaces_the_target_and_keeps_the_link(
    tmp_path, backend, command
):
    """F073: --inplace on work/link.h5ad turned the link into a private copy,
    so shared/real.h5ad, which everyone else reads, never changed."""
    shared = tmp_path / "shared"
    work = tmp_path / "work"
    shared.mkdir()
    work.mkdir()
    real = _source(shared, backend, "real")
    link = work / real.name
    link.symlink_to(Path("..") / "shared" / real.name)

    result = runner.invoke(app, _args(INPLACE[command](link, tmp_path)))
    assert result.exit_code == 0, _out(result)
    assert link.is_symlink()
    assert os.readlink(link) == str(Path("..") / "shared" / real.name)
    got = _read(real)
    if command == "subset":
        assert got.shape == (2, 4)
    elif command == "recast":
        assert got.X.dtype == np.float64
    else:
        assert "X_new" in got.obsm
    assert _debris(shared) == [] and _debris(work) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
@pytest.mark.parametrize("command", sorted(INPLACE))
def test_inplace_keeps_the_store_private(tmp_path, backend, command):
    """F073: a 0640 store (human data, say) came back 0644, world-readable."""
    src = _source(tmp_path, backend)
    os.chmod(src, 0o750 if src.is_dir() else 0o640)
    mode = src.stat().st_mode & 0o777

    result = runner.invoke(app, _args(INPLACE[command](src, tmp_path)))
    assert result.exit_code == 0, _out(result)
    assert src.stat().st_mode & 0o777 == mode


# ---------------------------------------------------------------------------
# F082: aliases of the input


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.touch()
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


def test_same_path_sees_hard_links_and_case_aliases(tmp_path):
    from adata.util.path import same_path

    src = tmp_path / "Data.h5ad"
    src.write_bytes(b"x")
    os.link(src, tmp_path / "alias.h5ad")
    assert same_path(src, tmp_path / "alias.h5ad")
    assert same_path(src, tmp_path / "data.h5ad") == _case_insensitive(tmp_path)
    assert not same_path(src, tmp_path / "other.h5ad")


def test_a_case_only_alias_of_the_input_is_refused(tmp_path, backend):
    """F082: on macOS, `subset Data.h5ad -o data.h5ad` passed the guard, and
    staging renamed the input aside and deleted it: exit 0, input gone."""
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive filesystem")
    src = _source(tmp_path, backend, "Data")
    before = _snapshot(src)
    alias = src.with_name("data" + src.suffix)

    for argv in (
        ["subset", src, "-o", alias, "-q", "grp == a"],
        ["recast", src, "X", "-o", alias, "--dtype", "float64"],
        ["import", "array", src, "obsm/X_new", _npy(tmp_path), "-o", alias],
    ):
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 1, (argv, _out(result))
        assert "Output path is the input" in _out(result)
        assert _snapshot(src) == before


def test_split_will_not_write_a_group_over_its_input_in_another_case(tmp_path):
    """F082: `split a.h5ad --by grp -o .` with a label 'A' deleted a.h5ad."""
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive filesystem")
    src = tmp_path / "a.h5ad"
    ad.AnnData(
        X=np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame({"grp": pd.Categorical(["A", "B"])}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    ).write_h5ad(src)
    before = _snapshot(src)

    result = runner.invoke(app, _args(["split", src, "--by", "grp", "-o", tmp_path]))
    assert result.exit_code == 1
    assert "Output path is the input" in _out(result)
    assert _snapshot(src) == before
    assert not (tmp_path / "B.h5ad").exists()


# ---------------------------------------------------------------------------
# F083: two runs, one output


def test_a_second_run_on_the_same_output_is_refused_while_the_first_writes(
    tmp_path, backend, monkeypatch
):
    """F083: two jobs writing one -o both passed the leftover check; for
    Zarr both wrote into the same temporary and one exited 0 with its cell
    names lost. The second is now refused and the first finishes intact."""
    from adata.core import subset as subset_mod

    src = _source(tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")
    real = subset_mod.check_free_space
    second = []

    def check(*args, **kwargs):
        # Before the first run's temporary exists: the window in which both
        # runs used to pass the leftover check.
        if not second:
            second.append(runner.invoke(
                app, _args(["subset", src, "-o", out, "-q", "grp == b"])
            ))
        return real(*args, **kwargs)

    monkeypatch.setattr(subset_mod, "check_free_space", check)
    first = runner.invoke(app, _args(["subset", src, "-o", out, "-q", "grp == a"]))
    assert first.exit_code == 0, _out(first)
    assert second[0].exit_code == 1
    assert "is writing" in _out(second[0])
    assert list(_read(out).obs_names) == ["c0", "c2"]
    assert _debris(tmp_path) == []


def test_a_lock_from_another_host_is_refused_not_removed(tmp_path):
    """F083: a lock whose run cannot be checked is never taken over: that
    run may still be writing, on another node of a cluster."""
    src = _rich_store(tmp_path / "in.h5ad")
    out = _rich_store(tmp_path / "out.h5ad")
    kept = _snapshot(out)
    lock = tmp_path / "out.h5ad.adata-lock"
    lock.write_text("node7 12345\n")

    result = runner.invoke(app, _args(["subset", src, "-o", out, "-q", "grp == a"]))
    assert result.exit_code == 1
    assert "is writing" in _out(result)
    assert lock.read_text() == "node7 12345\n"
    assert _snapshot(out) == kept


def test_a_lock_held_by_a_live_run_on_this_host_is_refused(tmp_path):
    import socket

    src = _rich_store(tmp_path / "in.h5ad")
    lock = tmp_path / "out.h5ad.adata-lock"
    lock.write_text(f"{socket.gethostname()} {os.getppid()}\n")

    result = runner.invoke(app, _args(["subset", src, "-o", tmp_path / "out.h5ad",
                                       "-q", "grp == a"]))
    assert result.exit_code == 1
    assert "is writing" in _out(result)
    assert lock.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process ids")
def test_a_lock_left_by_a_killed_run_on_this_host_is_taken_over(tmp_path):
    """F083 review: a run killed outright before its temporary existed (the
    out-of-memory killer during a long query) left only its lock, and every
    rerun was refused, where before the lock nothing at all was left."""
    import socket

    src = _rich_store(tmp_path / "in.h5ad")
    out = tmp_path / "out.h5ad"
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    lock = tmp_path / "out.h5ad.adata-lock"
    lock.write_text(f"{socket.gethostname()} {gone.pid}\n")

    result = runner.invoke(app, _args(["subset", src, "-o", out, "-q", "grp == a"]))
    assert result.exit_code == 0, _out(result)
    assert _read(out).shape == (2, 4)
    assert _debris(tmp_path) == []


# ---------------------------------------------------------------------------
# F088: Zarr versions


@pytest.mark.parametrize("version", [2, 3])
def test_every_writer_keeps_the_source_zarr_version(tmp_path, version):
    """F088: nothing caught subset or split writing a v2 source as v3. The
    split test meant to could not fail: anndata 0.13 writes v3 by default."""
    src = _rich_store(tmp_path / "in.zarr", zarr_format=version)
    assert _version(src) == version

    runs = {
        tmp_path / "subset.zarr": ["subset", src, "-o", tmp_path / "subset.zarr",
                                   "-q", "grp == a"],
        tmp_path / "parts" / "a.zarr": ["split", src, "--by", "grp",
                                        "-o", tmp_path / "parts"],
        tmp_path / "recast.zarr": ["recast", src, "X", "-o", tmp_path / "recast.zarr",
                                   "--dtype", "float64"],
        tmp_path / "import.zarr": ["import", "array", src, "obsm/X_new",
                                   _npy(tmp_path), "-o", tmp_path / "import.zarr"],
    }
    for out, argv in runs.items():
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 0, (argv, _out(result))
        assert _version(out) == version, argv
        assert _read(out).n_vars == 4

    for argv in (
        ["subset", src, "--inplace", "-q", "grp in a,b"],
        ["import", "array", src, "obsm/X_new", _npy(tmp_path), "--inplace"],
    ):
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 0, (argv, _out(result))
        assert _version(src) == version, argv


# ---------------------------------------------------------------------------
# F095: --inplace with -o


@pytest.mark.parametrize("command", sorted(INPLACE))
def test_inplace_and_output_together_are_refused(tmp_path, backend, command):
    """F095: the source was rewritten and -o silently ignored."""
    src = _source(tmp_path, backend)
    before = _snapshot(src)
    out = backend_path(tmp_path, backend, "out")

    result = runner.invoke(app, _args(INPLACE[command](src, tmp_path) + ["-o", out]))
    assert result.exit_code == 1
    assert "not both" in _out(result)
    assert _snapshot(src) == before
    assert not out.exists()


# ---------------------------------------------------------------------------
# F137: fsync


@pytest.mark.skipif(sys.platform == "win32", reason="os.sync is POSIX")
@pytest.mark.parametrize("suffix_", [".h5ad", ".zarr"])
def test_a_replaced_output_is_synced_before_the_old_one_goes(tmp_path, monkeypatch, suffix_):
    """F137 (plausible; needs a power cut to see): --inplace deleted the
    original while the new store might still be only in the page cache. The
    successor is now synced (an HDF5 file by fsync, a Zarr tree by one
    sync) before the original is renamed aside, and the directory after."""
    from adata.util import staging

    src = _rich_store(tmp_path / f"in{suffix_}")
    events = []
    real_fsync, real_sync, real_rename = os.fsync, os.sync, Path.rename

    def fsync(fd):
        events.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync")
        return real_fsync(fd)

    def rename(self, target):
        events.append(f"rename {Path(self).name} -> {Path(target).name}")
        return real_rename(self, target)

    monkeypatch.setattr(staging.os, "fsync", fsync)
    monkeypatch.setattr(staging.os, "sync", lambda: (events.append("sync"), real_sync())[1])
    monkeypatch.setattr(Path, "rename", rename)
    result = runner.invoke(app, _args(["subset", src, "--inplace", "-q", "grp == a"]))
    assert result.exit_code == 0, _out(result)
    renames = [i for i, e in enumerate(events) if e.startswith("rename")]
    flush = "fsync" if suffix_ == ".h5ad" else "sync"
    assert flush in events[: renames[0]]
    assert events[renames[0]] == f"rename in{suffix_} -> in.adata-old{suffix_}"
    assert "dir" in events[renames[-1]:]


# ---------------------------------------------------------------------------
# F145: SIGTERM, and leftovers


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("suffix_", [".h5ad", ".zarr"])
def test_sigterm_while_writing_cleans_up_like_any_failure(tmp_path, suffix_):
    """F145: a scheduler's SIGTERM (time limit, scancel, bkill) ended the
    process on the spot: the partial temporary stayed, and blocked reruns."""
    src = _rich_store(tmp_path / f"in{suffix_}")
    before = _snapshot(src)
    script = (
        "import os, signal, sys\n"
        "from adata.core import subset\n"
        "real = subset._write_targets\n"
        "def write(*a, **k):\n"
        "    real(*a, **k)\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "subset._write_targets = write\n"
        "sys.argv = ['adata'] + sys.argv[1:]\n"
        "from adata.cli import main\n"
        "main()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, "subset", str(src), "--inplace", "-q", "grp == a"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 128 + signal.SIGTERM, proc.stderr[-2000:]
    assert _snapshot(src) == before
    assert _debris(tmp_path) == []


def test_every_leftover_is_named_at_once(tmp_path):
    """F145: split refused on the first leftover it found, so N groups meant
    N reruns to clear them."""
    src = _rich_store(tmp_path / "in.h5ad")
    parts = tmp_path / "parts"
    parts.mkdir()
    for name in ("a.adata-tmp.h5ad", "b.adata-tmp.h5ad"):
        (parts / name).write_text("partial")

    result = runner.invoke(app, _args(["split", src, "--by", "grp", "-o", parts]))
    assert result.exit_code == 1
    text = _flat(result)
    assert "a.adata-tmp.h5ad" in text and "b.adata-tmp.h5ad" in text
    assert sorted(p.name for p in parts.iterdir()) == ["a.adata-tmp.h5ad",
                                                       "b.adata-tmp.h5ad"]


# ---------------------------------------------------------------------------
# convert: the same output check and free-space check


def test_convert_overwrite_will_not_replace_a_folder_that_is_not_a_store(tmp_path):
    """F037: `convert base.h5ad -o proj --to h5ad --overwrite` replaced the
    folder proj/, which held other files, with an HDF5 file."""
    src = _rich_store(tmp_path / "in.h5ad")
    folder = _folder(tmp_path, "proj")
    kept = _snapshot(folder)

    for argv in (
        ["convert", src, "-o", folder, "--to", "h5ad", "--overwrite"],
        ["convert", src, "-o", folder, "--to", "10x-mtx", "--overwrite"],
    ):
        result = runner.invoke(app, _args(argv))
        assert result.exit_code == 1, (argv, _out(result))
        assert _snapshot(folder) == kept
    assert _debris(tmp_path) == []


def test_convert_checks_free_space_first(tmp_path, monkeypatch):
    """F022: the HDF5 copy convert makes is where a full disk was seen to
    abort the process (SIGABRT in H5Ocopy), leaving the temporary behind."""
    src = _rich_store(tmp_path / "in.zarr")
    _tiny_disk(monkeypatch)

    result = runner.invoke(app, _args(["convert", src, "-o", tmp_path / "o.h5ad"]))
    assert result.exit_code == 1
    assert "Not enough free space" in _out(result)
    assert not (tmp_path / "o.h5ad").exists()

    result = runner.invoke(app, _args(["convert", src, "-o", tmp_path / "o.h5ad",
                                       "--no-space-check"]))
    assert result.exit_code == 0, _out(result)
    assert _read(tmp_path / "o.h5ad").shape == (4, 4)


def test_a_file_written_into_a_folder_being_read_is_allowed(tmp_path):
    """Only a Zarr input is guarded against outputs inside it: writing next
    to a Space Ranger outs/ or into the MTX folder being read is fine."""
    src = _rich_store(tmp_path / "in.h5ad")
    folder = tmp_path / "mtx"
    made = runner.invoke(app, _args(["convert", src, "-o", folder,
                                     "--to", "10x-mtx", "--force"]))
    assert made.exit_code == 0, _out(made)

    result = runner.invoke(app, _args(["convert", folder, "-o", folder / "c.h5ad"]))
    assert result.exit_code == 0, _out(result)
    assert _read(folder / "c.h5ad").shape == (4, 4)


# ---------------------------------------------------------------------------
# From the review of #111


@pytest.mark.parametrize("command", ["import", "export"])
def test_an_output_that_is_a_symlink_is_written_through(tmp_path, command):
    """F073 review: `import ... -o work/link.h5ad` and `export ... -o link.csv`
    wrote through the link before staging; staged, they replaced the link
    with a private file and left its target unchanged."""
    src = _rich_store(tmp_path / "in.h5ad")
    shared = tmp_path / "shared"
    shared.mkdir()
    if command == "import":
        real = _rich_store(shared / "real.h5ad")
        argv = ["import", "array", src, "obsm/X_new", _npy(tmp_path)]
    else:
        real = shared / "real.csv"
        real.write_text("old")
        argv = ["export", "dataframe", src, "obs"]
    link = tmp_path / f"link{real.suffix}"
    link.symlink_to(real)

    result = runner.invoke(app, _args(argv + ["-o", link]))
    assert result.exit_code == 0, _out(result)
    assert link.is_symlink() and link.resolve() == real.resolve()
    if command == "import":
        assert "X_new" in _read(real).obsm
    else:
        assert list(pd.read_csv(real, index_col=0).index) == ["c0", "c1", "c2", "c3"]
    assert _debris(tmp_path) == [] and _debris(shared) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX devices and FIFOs")
def test_export_writes_to_devices_and_pipes_as_they_are(tmp_path):
    """F023 review: staged, `-o /dev/null` failed for want of a lock beside
    it, and `-o fifo` renamed the pipe away and left a regular file there,
    while the reader waited on the pipe for ever."""
    src = _rich_store(tmp_path / "in.h5ad")

    result = runner.invoke(app, _args(["export", "dataframe", src, "obs", "-o", os.devnull]))
    assert result.exit_code == 0, _out(result)
    assert stat.S_ISCHR(os.stat(os.devnull).st_mode)

    fifo = tmp_path / "obs.fifo"
    os.mkfifo(fifo)
    got = []
    reader = threading.Thread(target=lambda: got.append(fifo.read_text()))
    reader.start()
    result = runner.invoke(app, _args(["export", "dataframe", src, "obs", "-o", fifo]))
    reader.join(timeout=30)
    assert result.exit_code == 0, _out(result)
    assert got and got[0].startswith("_index,grp")
    assert stat.S_ISFIFO(os.stat(fifo).st_mode)
    assert _debris(tmp_path) == []


def test_export_will_not_replace_a_store(tmp_path):
    """F023 review: `export dataframe a.h5ad obs -o b.h5ad` (for b.csv)
    replaced the store b.h5ad with 36 bytes of CSV and exited 0."""
    src = _rich_store(tmp_path / "a.h5ad")
    other = _rich_store(tmp_path / "b.h5ad")
    kept = _snapshot(other)

    for target in (other, _rich_store(tmp_path / "c.zarr")):
        result = runner.invoke(app, _args(["export", "dataframe", src, "obs", "-o", target]))
        assert result.exit_code == 1, _out(result)
    assert _snapshot(other) == kept


def test_an_empty_folder_can_be_the_output(tmp_path, backend):
    """F037 review: a pre-created empty `out.zarr/` was refused as "not a
    Zarr store"; before the check it was replaced, as it should be."""
    src = _source(tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")
    out.mkdir()

    result = runner.invoke(app, _args(["subset", src, "-o", out, "-q", "grp == a"]))
    assert result.exit_code == 0, _out(result)
    assert _read(out).shape == (2, 4)


def test_split_labels_that_differ_only_in_case_get_their_own_stores(tmp_path):
    """Review of #111: labels `B` and `b` are one file on macOS and Windows,
    so the two groups were written to one path and the split refused with
    a misleading message. They now get distinct names everywhere."""
    src = tmp_path / "in.h5ad"
    obj = ad.AnnData(
        X=np.arange(6, dtype="float32").reshape(3, 2),
        obs=pd.DataFrame({"grp": pd.Categorical(["B", "b", "b"])},
                         index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    obj.write_h5ad(src)
    parts = tmp_path / "parts"

    result = runner.invoke(app, _args(["split", src, "--by", "grp", "-o", parts]))
    assert result.exit_code == 0, _out(result)
    assert list(_read(parts / "B.h5ad").obs_names) == ["c0"]
    assert list(_read(parts / "b_1.h5ad").obs_names) == ["c1", "c2"]


def test_an_mtx_output_will_not_replace_a_file(tmp_path):
    """Review of #111: `convert in.h5ad -o notes.txt --to 10x-mtx
    --overwrite` replaced the file with a folder it then could not enter
    (the file's mode, copied onto it, had no execute bit)."""
    src = _rich_store(tmp_path / "in.h5ad")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep me")

    result = runner.invoke(app, _args(["convert", src, "-o", notes, "--to", "10x-mtx",
                                       "--overwrite", "--force"]))
    assert result.exit_code == 1
    assert notes.read_text() == "keep me"
    assert _debris(tmp_path) == []


def test_a_signal_held_during_the_swap_is_raised_even_if_the_swap_fails():
    """Review of #111: a SIGTERM held while the swap ran was dropped if the
    swap then failed, so the run exited 1 rather than 143."""
    from adata.util.staging import Terminated, _Signals

    with _Signals() as signals:
        with pytest.raises(Terminated) as raised:
            with signals.hold():
                os.kill(os.getpid(), signal.SIGTERM)
                raise OSError("the rename failed")
    assert raised.value.code == 128 + signal.SIGTERM


def test_an_old_copy_that_cannot_be_deleted_does_not_fail_the_run(tmp_path, monkeypatch):
    """Review of #111: every output was in place, but a backup that could
    not be removed (a read-only subtree, NFS's .nfs files) exited 1."""
    from adata.util import staging

    out = tmp_path / "out.h5ad"
    out.write_text("old")
    real = staging._remove

    def remove(path):
        if "adata-old" in path.name:
            raise PermissionError(13, "Permission denied", str(path))
        real(path)

    monkeypatch.setattr(staging, "_remove", remove)
    with pytest.warns(UserWarning, match="could not delete the old copy"):
        with staging.staged_outputs([out]) as (temp,):
            temp.write_text("new")
    assert out.read_text() == "new"


@pytest.mark.parametrize("kind", ["array", "sparse"])
def test_an_input_of_the_wrong_shape_is_refused_before_the_store_is_copied(
    tmp_path, backend, monkeypatch, kind
):
    """F024 review: `import sparse big.h5ad X bad.mtx --inplace` copied the
    whole store before reading the MTX header, then refused: as long, and
    as much room, as the store is large, to say the file did not fit."""
    from adata.commands import import_data

    src = _source(tmp_path, backend)
    before = _snapshot(src)
    if kind == "array":
        data = tmp_path / "bad.npy"
        np.save(data, np.ones((9, 2), dtype="float32"))
    else:
        data = tmp_path / "bad.mtx"
        data.write_text("%%MatrixMarket matrix coordinate real general\n9 4 1\n1 1 1.0\n")
    monkeypatch.setattr(import_data, "copy_path", _fail)
    monkeypatch.setattr(import_data, "copy_store_contents", _fail)

    result = runner.invoke(app, _args(["import", kind, src, "X", data, "--inplace"]))
    assert result.exit_code == 1
    assert "mismatch" in _out(result).lower()
    assert "interrupted" not in _out(result)
    assert _snapshot(src) == before
    assert _debris(tmp_path) == []


def test_import_array_copies_in_blocks_of_chunk_elements(tmp_path, backend):
    """F015: `import array` now takes the shared --chunk."""
    src = _source(tmp_path, backend)
    data = tmp_path / "pca.npy"
    values = np.arange(8, dtype="float32").reshape(4, 2)
    np.save(data, values)

    result = runner.invoke(app, _args(["import", "array", src, "obsm/X_new", data,
                                       "--inplace", "--chunk", "3"]))
    assert result.exit_code == 0, _out(result)
    assert np.array_equal(_read(src).obsm["X_new"], values)

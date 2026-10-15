"""Tests for `adata convert`: changing a store's format.

anndata is the oracle. It writes every source and reads every output, so a
conversion is correct only if what anndata reads back equals what it wrote.
(The matrix command that used to be called `convert` is `recast` now, and is
tested in test_recast.py; the forwarding from the old spelling is tested
here.)
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
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")
zarr = pytest.importorskip("zarr")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

BACKENDS = ["h5ad", "zarr2", "zarr3"]


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _path(tmp_path: Path, backend: str, stem: str) -> Path:
    return tmp_path / (f"{stem}.h5ad" if backend == "h5ad" else f"{stem}.zarr")


def _version(backend: str):
    return {"zarr2": 2, "zarr3": 3}.get(backend)


def _object():
    n_obs, n_var = 30, 12
    rng = np.random.default_rng(0)
    X = sparse.random(
        n_obs, n_var, density=0.3, format="csr", dtype="float32", random_state=rng
    )
    obj = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(rng.choice(["a", "b", "c"], n_obs)),
                "n_counts": np.arange(n_obs, dtype="int32"),
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(
            {"gene_ids": [f"ENSG{i}" for i in range(n_var)]},
            index=[f"gene_{i}" for i in range(n_var)],
        ),
    )
    obj.layers["dense"] = X.toarray()
    obj.obsm["X_umap"] = rng.normal(size=(n_obs, 2)).astype("float32")
    obj.uns["params"] = {"layer": None, "k": 5, "names": ["x", "y"]}
    obj.raw = obj
    return obj


def _write(obj, path: Path, backend: str) -> Path:
    if backend == "h5ad":
        obj.write_h5ad(path)
        return path
    previous = ad.settings.zarr_write_format
    ad.settings.zarr_write_format = _version(backend)
    try:
        obj.write_zarr(path)
    finally:
        ad.settings.zarr_write_format = previous
    return path


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _zarr_format(path: Path):
    return zarr.open_group(str(path), mode="r").metadata.zarr_format


def _leftovers(tmp_path: Path):
    return sorted(p.name for p in tmp_path.iterdir() if "adata-tmp" in p.name)


@pytest.fixture
def h5ad(tmp_path) -> Path:
    return _write(_object(), tmp_path / "in.h5ad", "h5ad")


# ---------------------------------------------------------------------------
# h5ad <-> zarr


@pytest.mark.parametrize("src", BACKENDS)
@pytest.mark.parametrize("dst", BACKENDS)
def test_every_backend_pair_round_trips_through_anndata(tmp_path, src, dst):
    expected = _object()
    source = _write(expected, _path(tmp_path, src, "in"), src)
    out = _path(tmp_path, dst, "out")

    args = ["convert", str(source), "-o", str(out)]
    if _version(dst):
        args += ["--zarr-format", str(_version(dst))]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, _out(result)

    got = _read(out)
    assert np.array_equal(got.X.toarray(), expected.X.toarray())
    assert got.X.format == "csr"
    pd.testing.assert_frame_equal(got.obs, expected.obs)
    pd.testing.assert_frame_equal(got.var, expected.var)
    assert np.array_equal(got.layers["dense"], expected.layers["dense"])
    assert np.array_equal(got.obsm["X_umap"], expected.obsm["X_umap"])
    assert got.uns["params"]["layer"] is None
    assert got.uns["params"]["k"] == 5
    assert list(got.uns["params"]["names"]) == ["x", "y"]
    assert np.array_equal(got.raw.X.toarray(), expected.raw.X.toarray())
    if _version(dst):
        assert _zarr_format(out) == _version(dst)
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("src,expected", [("zarr2", 2), ("zarr3", 3), ("h5ad", 3)])
def test_zarr_output_keeps_the_source_version_unless_asked(tmp_path, src, expected):
    """A v2 store must not be silently upgraded; an h5ad gets anndata's v3."""
    source = _write(_object(), _path(tmp_path, src, "in"), src)
    out = tmp_path / "out.zarr"

    result = runner.invoke(app, ["convert", str(source), "-o", str(out)])
    assert result.exit_code == 0, _out(result)
    assert _zarr_format(out) == expected


def test_zarr_output_is_consolidated(h5ad, tmp_path):
    """anndata reads through the consolidated index when there is one."""
    out = tmp_path / "out.zarr"
    assert runner.invoke(app, ["convert", str(h5ad), "-o", str(out)]).exit_code == 0
    assert (out / "zarr.json").exists()
    consolidated = zarr.open_consolidated(str(out), mode="r")
    assert "X" in consolidated


def test_compressed_zarr_stays_compressed_in_h5ad(tmp_path):
    source = _write(_object(), tmp_path / "in.zarr", "zarr3")
    out = tmp_path / "out.h5ad"
    assert runner.invoke(app, ["convert", str(source), "-o", str(out)]).exit_code == 0
    with h5py.File(out) as handle:
        assert handle["X"]["data"].compression == "gzip"


@pytest.mark.parametrize("choice", ["gzip", "lzf"])
def test_compression_can_be_chosen_for_h5ad(h5ad, tmp_path, choice):
    out = tmp_path / "out.h5ad"
    result = runner.invoke(
        app, ["convert", str(h5ad), "-o", str(out), "--compression", choice]
    )
    assert result.exit_code == 0, _out(result)
    with h5py.File(out) as handle:
        assert handle["X"]["data"].compression == choice
        assert handle["layers"]["dense"].compression == choice
    assert np.array_equal(_read(out).X.toarray(), _read(h5ad).X.toarray())


@pytest.mark.parametrize("version", [2, 3])
def test_compression_none_writes_uncompressed_zarr(h5ad, tmp_path, version):
    out = tmp_path / "out.zarr"
    result = runner.invoke(
        app,
        ["convert", str(h5ad), "-o", str(out), "--compression", "none",
         "--zarr-format", str(version)],
    )
    assert result.exit_code == 0, _out(result)
    root = zarr.open_group(str(out), mode="r")
    assert root["X"]["data"].compressors == ()
    assert np.array_equal(_read(out).X.toarray(), _read(h5ad).X.toarray())


def test_an_anndata_07_store_is_recognised_by_its_structure(sample_h5ad_file, tmp_path):
    """anndata 0.7 wrote no root attributes, and the name proves nothing."""
    renamed = sample_h5ad_file.rename(tmp_path / "old.h5")
    out = tmp_path / "out.zarr"
    result = runner.invoke(app, ["convert", str(renamed), "-o", str(out)])
    assert result.exit_code == 0, _out(result)
    assert _read(out).shape == (5, 4)


def test_an_unrecognised_input_needs_from(tmp_path):
    plain = tmp_path / "plain.h5"
    with h5py.File(plain, "w") as handle:
        handle["x"] = np.arange(3)

    result = runner.invoke(app, ["convert", str(plain), "-o", str(tmp_path / "o.h5ad")])
    assert result.exit_code == 1
    assert "Cannot tell what format" in _out(result) and "--from" in _out(result)

    # Named as a 10x file would be, it is still not one: content decides.
    result = runner.invoke(
        app, ["convert", str(plain), "-o", str(tmp_path / "o.h5ad"), "--from", "10x-h5"]
    )
    assert result.exit_code == 1
    assert "Not a 10x matrix" in _out(result)


def test_to_names_the_format_of_an_unfamiliar_extension(h5ad, tmp_path):
    out = tmp_path / "out.anndata"
    result = runner.invoke(app, ["convert", str(h5ad), "-o", str(out), "--to", "h5ad"])
    assert result.exit_code == 0, _out(result)
    assert _read(out.rename(tmp_path / "out.h5ad")).shape == (30, 12)


# ---------------------------------------------------------------------------
# refusing, and leaving nothing behind


@pytest.mark.parametrize(
    "extra,message",
    [
        (["-o", "{tmp}/o.h5ad", "--zarr-format", "2"], "--zarr-format does not apply"),
        (["-o", "{tmp}/o.zarr", "--zarr-format", "4"], "--zarr-format must be 2 or 3"),
        (["-o", "{tmp}/o.zarr", "--compression", "lzf"], "not available for zarr"),
        (["-o", "{tmp}/o.h5ad", "--compression", "zstd"], "not available for h5ad"),
        (["-o", "{tmp}/o.h5ad", "--to", "zarr"], "contradicts the output's extension"),
        (["-o", "{tmp}/o.data", "--to", "zarr"], "needs a name ending in .zarr"),
        (["-o", "{tmp}/o.data"], "Cannot tell the output format"),
        (["-o", "{tmp}/o.h5ad", "--to", "loom"], "Unknown format 'loom'"),
        (["-o", "{tmp}/o.h5ad", "--from", "loom"], "Unknown format 'loom'"),
        (["-o", "-"], "cannot go to stdout"),
        ([], "Output path is required"),
        (["-o", "{tmp}/o.h5ad", "--in-memory"], "--in-memory applies to `adata recast`"),
        (["-o", "{tmp}/o.h5ad", "-C", "64"], "--chunk applies to `adata recast`"),
        (["{tmp}/o.zarr"], "To write it, pass -o"),
    ],
)
def test_bad_arguments_are_refused_before_anything_is_written(
    h5ad, tmp_path, extra, message
):
    args = [a.replace("{tmp}", str(tmp_path)) for a in extra]
    before = sorted(p.name for p in tmp_path.iterdir())

    result = runner.invoke(app, ["convert", str(h5ad), *args])
    assert result.exit_code == 1, _out(result)
    assert message in _out(result)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_existing_output_is_kept_unless_overwrite(h5ad, tmp_path, backend):
    out = _path(tmp_path, backend, "out")
    _write(ad.AnnData(np.zeros((2, 2), dtype="float32")), out, backend)

    result = runner.invoke(app, ["convert", str(h5ad), "-o", str(out)])
    assert result.exit_code == 1
    assert "already exists" in _out(result) and "--overwrite" in _out(result)
    assert _read(out).shape == (2, 2)

    args = ["convert", str(h5ad), "-o", str(out), "--overwrite"]
    if _version(backend):
        args += ["--zarr-format", str(_version(backend))]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, _out(result)
    assert _read(out).shape == (30, 12)
    assert [p.name for p in tmp_path.iterdir() if "adata-" in p.name] == []


@pytest.mark.parametrize("backend", ["h5ad", "zarr3"])
def test_converting_onto_the_input_is_refused(tmp_path, backend):
    """On Zarr this would empty the source before reading a single element."""
    source = _write(_object(), _path(tmp_path, backend, "in"), backend)
    result = runner.invoke(
        app, ["convert", str(source), "-o", str(source), "--overwrite"]
    )
    assert result.exit_code == 1
    assert "Output path is the input" in _out(result)
    assert _read(source).shape == (30, 12)


def test_a_failure_part_way_leaves_the_old_output_and_no_debris(
    h5ad, tmp_path, monkeypatch
):
    """The point of staging: whatever half got written is thrown away."""
    from adata.conversion import anndata_copy

    out = tmp_path / "out.zarr"
    _write(ad.AnnData(np.zeros((2, 2), dtype="float32")), out, "zarr3")

    real = anndata_copy.copy_tree

    def failing(src, dst, name, **kwargs):
        if name == "var":
            raise RuntimeError("disk full")
        return real(src, dst, name, **kwargs)

    monkeypatch.setattr(anndata_copy, "copy_tree", failing)
    result = runner.invoke(app, ["convert", str(h5ad), "-o", str(out), "--overwrite"])
    assert result.exit_code == 1
    assert "disk full" in _out(result)
    assert _read(out).shape == (2, 2), "the previous output must survive"
    assert _leftovers(tmp_path) == []


def test_a_stale_temporary_from_an_interrupted_run_is_reported(h5ad, tmp_path):
    (tmp_path / "out.adata-tmp.zarr").mkdir()
    result = runner.invoke(app, ["convert", str(h5ad), "-o", str(tmp_path / "out.zarr")])
    assert result.exit_code == 1
    assert "interrupted run" in _out(result)
    assert not (tmp_path / "out.zarr").exists()


# ---------------------------------------------------------------------------
# the old meaning: convert's matrix flags forward to recast until 1.0.0


def test_matrix_flags_forward_to_recast_with_a_warning(h5ad, tmp_path):
    old, new = tmp_path / "old.h5ad", tmp_path / "new.h5ad"

    result = runner.invoke(
        app, ["convert", str(h5ad), "X", "-o", str(old), "--dtype", "float64"]
    )
    assert result.exit_code == 0, _out(result)
    text = _out(result)
    assert "deprecated" in text and "'recast'" in text and "--dtype" in text

    result = runner.invoke(
        app, ["recast", str(h5ad), "X", "-o", str(new), "--dtype", "float64"]
    )
    assert result.exit_code == 0, _out(result)
    assert "deprecated" not in _out(result)

    got_old, got_new = _read(old).X, _read(new).X
    assert got_old.dtype == got_new.dtype == np.dtype("float64")
    assert np.array_equal(got_old.toarray(), got_new.toarray())


def test_forwarding_keeps_every_recast_flag(h5ad, tmp_path):
    """--all, --layout, --in-memory, -C and --zarr-format all still reach recast."""
    out = tmp_path / "out.zarr"
    result = runner.invoke(
        app,
        ["convert", str(h5ad), "--all", "-o", str(out), "--layout", "csc",
         "--in-memory", "-C", "16", "--zarr-format", "2"],
    )
    assert result.exit_code == 0, _out(result)
    got = _read(out)
    assert got.X.format == "csc" and got.raw.X.format == "csc"
    assert _zarr_format(out) == 2


def test_forwarding_works_in_place(h5ad):
    result = runner.invoke(
        app, ["convert", str(h5ad), "X", "--inplace", "--dtype", "float64"]
    )
    assert result.exit_code == 0, _out(result)
    assert _read(h5ad).X.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "extra", [["--to", "zarr"], ["--compression", "gzip"], ["--overwrite"]]
)
def test_format_flags_and_matrix_flags_do_not_mix(h5ad, tmp_path, extra):
    result = runner.invoke(
        app,
        ["convert", str(h5ad), "X", "-o", str(tmp_path / "o.zarr"),
         "--dtype", "float64", *extra],
    )
    assert result.exit_code == 1
    assert "cannot be combined" in _out(result) and "adata recast" in _out(result)


def test_forwarding_still_validates_like_recast(h5ad, tmp_path):
    result = runner.invoke(app, ["convert", str(h5ad), "X", "--dtype", "float64"])
    assert result.exit_code == 1
    assert "Output file is required" in _out(result)

    result = runner.invoke(
        app, ["convert", str(h5ad), "X", "-o", str(tmp_path / "o.h5ad"), "--layout", "coo"]
    )
    assert result.exit_code == 1
    assert "--layout must be one of" in _out(result)


# ---------------------------------------------------------------------------
# staging, directly


def test_staging_replaces_an_existing_directory_rather_than_nesting_in_it(tmp_path):
    from adata.util.staging import staged_outputs

    out = tmp_path / "out.zarr"
    out.mkdir()
    (out / "old").write_text("old")

    with staged_outputs([out]) as (temp,):
        temp.mkdir()
        (temp / "new").write_text("new")

    assert sorted(p.name for p in out.iterdir()) == ["new"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.zarr"]


def test_staging_moves_several_outputs_together(tmp_path):
    from adata.util.staging import staged_outputs

    outs = [tmp_path / "a.h5ad", tmp_path / "b.h5ad"]
    with staged_outputs(outs) as temps:
        assert [t.suffix for t in temps] == [".h5ad", ".h5ad"]
        for temp in temps:
            temp.write_text(temp.name)

    assert [p.read_text() for p in outs] == ["a.adata-tmp.h5ad", "b.adata-tmp.h5ad"]

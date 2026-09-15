"""Compatibility against stores written by real anndata releases, 0.8 to 0.13.

The rest of the suite checks this tool against stores it built itself, which
cannot catch the thing that actually broke it: anndata changing how it writes
a file. Here each fixture is produced by running a pinned anndata release in
its own environment, so the assertions are about what those releases really
wrote rather than about this repo's understanding of the format.

Marked `integration` because building the environments needs `uv` and, the
first time, the network:

    pytest -m integration                  # just these
    pytest -m "not integration"            # skip them
    ADATA_SKIP_VERSION_FIXTURES=1 pytest   # skip them via the environment
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from adata.core.info import axis_len, get_entry_type
from adata.elements import spec
from adata.elements.read import element_len, read_str_all, resolve_index
from adata.storage import is_group, open_store

from tests.reference_stores import (
    RELEASES,
    ReferenceCache,
    ReferenceUnavailable,
    must_build,
    offline,
    uv_available,
)

pytestmark = pytest.mark.integration

runner = CliRunner()

N_OBS, N_VAR = 6, 4
OBS_NAMES = [f"cell_{i}" for i in range(N_OBS)]
VAR_NAMES = [f"gene_{i}" for i in range(N_VAR)]


@pytest.fixture(scope="session")
def cache(tmp_path_factory) -> ReferenceCache:
    if offline():
        pytest.skip("ADATA_SKIP_VERSION_FIXTURES is set")
    if not uv_available():
        pytest.skip("uv is needed to build the reference stores")
    return ReferenceCache(tmp_path_factory.mktemp("anndata-versions"))


@pytest.fixture(params=RELEASES, ids=[r.label for r in RELEASES])
def release(request):
    return request.param


@pytest.fixture(params=["h5ad", "zarr"])
def fmt(request) -> str:
    return request.param


@pytest.fixture
def store(cache, release, fmt) -> Path:
    """A reference store for this release and format.

    A build failure fails the test wherever these fixtures are required -- CI,
    or ADATA_REQUIRE_VERSION_FIXTURES=1 -- so a broken pin cannot quietly
    reduce the whole job to skips. Elsewhere it degrades to a skip, since a
    local environment problem should not block unrelated work.
    """
    try:
        return cache.get(release, fmt)
    except ReferenceUnavailable as exc:
        message = f"could not build {release.label} ({fmt}): {exc}"
        if must_build():
            pytest.fail(message)
        pytest.skip(message)


def _out(result) -> str:
    return result.stdout + (result.stderr or "")


# ---------------------------------------------------------------------------
# reading


def test_view_reports_the_right_shape(store):
    result = runner.invoke(app, ["view", str(store)])
    assert result.exit_code == 0, _out(result)
    assert f"{N_OBS} × {N_VAR}" in result.stdout


def test_ls_walks_the_whole_store(store):
    result = runner.invoke(app, ["ls", str(store), "--long"])
    assert result.exit_code == 0, _out(result)
    for key in ("obs", "var", "X", "uns", "layers"):
        assert key in result.stdout


def test_axis_lengths_are_readable_whatever_the_index_layout(store):
    """The index is a dataset in older releases and a group from 0.11."""
    with open_store(store, "r") as handle:
        assert axis_len(handle.root, "obs") == N_OBS
        assert axis_len(handle.root, "var") == N_VAR


def test_index_names_read_back_correctly(store):
    with open_store(store, "r") as handle:
        for axis, expected in (("obs", OBS_NAMES), ("var", VAR_NAMES)):
            index, _ = resolve_index(handle.root[axis], axis)
            assert read_str_all(index) == expected
            assert element_len(index) == len(expected)


def test_every_obs_column_exports(store):
    result = runner.invoke(app, ["export", "dataframe", str(store), "obs"])
    assert result.exit_code == 0, _out(result)

    lines = [ln for ln in result.stdout.splitlines() if ln]
    header = lines[0].split(",")
    assert len(lines) == N_OBS + 1
    for column in ("cell_type", "n_counts", "score", "free_text"):
        assert column in header, f"{column} missing from {header}"

    # The categorical must render as its labels, not its codes.
    cell_type = lines[1].split(",")[header.index("cell_type")]
    assert cell_type == "A"


def test_nullable_columns_render_missing_values_as_empty(store):
    result = runner.invoke(app, ["export", "dataframe", str(store), "obs"])
    assert result.exit_code == 0, _out(result)

    lines = [ln for ln in result.stdout.splitlines() if ln]
    header = lines[0].split(",")
    if "nullable_int" not in header:
        pytest.skip("this release did not write a nullable column")
    column = header.index("nullable_int")
    assert lines[3].split(",")[column] == "", "row 3 is NA in the fixture"


def test_raw_var_exports_from_its_own_path(store):
    result = runner.invoke(app, ["export", "dataframe", str(store), "raw/var"])
    assert result.exit_code == 0, _out(result)
    assert "gene_ids" in result.stdout


def test_sparse_x_exports_to_matrix_market(store):
    result = runner.invoke(app, ["export", "sparse", str(store), "X"])
    assert result.exit_code == 0, _out(result)
    lines = result.stdout.splitlines()
    assert lines[0].startswith("%%MatrixMarket")
    dims = [ln for ln in lines if not ln.startswith("%")][0].split()
    assert dims[:2] == [str(N_OBS), str(N_VAR)]


def test_uns_exports_to_json(store):
    import json

    result = runner.invoke(app, ["export", "dict", str(store), "uns"])
    assert result.exit_code == 0, _out(result)
    payload = json.loads(result.stdout)
    assert payload["a_string"] == "hello"
    assert int(payload["an_int"]) == 42
    assert payload["nested"]["deep"]["value"] == pytest.approx(1.5)


def test_categorical_is_detected_whatever_the_layout(store):
    """0.7-era categoricals are codes plus a reference; later ones are groups."""
    with open_store(store, "r") as handle:
        info = get_entry_type(handle.root["obs"]["cell_type"])
        assert info["type"] == "categorical"


def test_encodings_are_recognised_not_guessed(store):
    with open_store(store, "r") as handle:
        root = handle.root
        assert get_entry_type(root["X"])["type"] == "sparse-matrix"
        assert get_entry_type(root["obs"])["type"] == "dataframe"
        assert get_entry_type(root["obsm"]["X_pca"])["type"] in (
            "array",
            "dense-matrix",
        )


# ---------------------------------------------------------------------------
# writing, checked by reading the result back with the same anndata release


@pytest.fixture
def read_back(release, fmt):
    """Read a store using the same anndata release that wrote the fixture.

    Checking our output with the *current* anndata would not show whether an
    older release can still open what we wrote.
    """
    import subprocess

    from tests.reference_stores import Release

    def _read(path: Path, expression: str):
        script = (
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import anndata as ad, sys\n"
            f"a = ad.read_{'zarr' if fmt == 'zarr' else 'h5ad'}(sys.argv[1])\n"
            f"print('RESULT', {expression})\n"
        )
        cmd = [
            "uv", "run", "--no-project", "--python", release.python,
            "--with", release.spec, "--with", "scipy",
        ]
        for extra in release.extras:
            cmd += ["--with", extra]
        cmd += ["python", "-c", script, str(path)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        line = [
            ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")
        ]
        if not line:
            pytest.fail(f"reading back failed:\n{out.stdout}\n{out.stderr}")
        return line[0][len("RESULT ") :]

    return _read


def test_subset_output_opens_in_the_release_that_wrote_the_input(
    store, tmp_path, fmt, read_back
):
    """A store we write must be readable by the anndata that made the source."""
    names = tmp_path / "keep.txt"
    names.write_text("cell_0\ncell_2\n")
    out = tmp_path / f"subset.{fmt}"

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "--obs", str(names)]
    )
    assert result.exit_code == 0, _out(result)

    assert read_back(out, "(a.n_obs, a.n_vars)") == "(2, 4)"
    assert read_back(out, "list(a.obs_names)") == "['cell_0', 'cell_2']"
    assert read_back(out, "str(a.obs['cell_type'].dtype)") == "category"


def test_subset_preserves_raw_across_versions(store, tmp_path, fmt, read_back):
    names = tmp_path / "keep.txt"
    names.write_text("cell_0\ncell_2\n")
    out = tmp_path / f"subset.{fmt}"
    assert runner.invoke(
        app, ["subset", str(store), "-o", str(out), "--obs", str(names)]
    ).exit_code == 0

    assert read_back(out, "a.raw is not None") == "True"
    assert read_back(out, "(a.raw.shape)") == "(2, 4)"


@pytest.mark.parametrize("target", ["h5ad", "zarr"])
def test_conversion_between_backends_keeps_the_data(store, tmp_path, target):
    """Reading with the current anndata is enough to prove the crossing works."""
    ad = pytest.importorskip("anndata")

    names = tmp_path / "keep.txt"
    names.write_text("\n".join(OBS_NAMES))
    out = tmp_path / f"converted.{target}"

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "--obs", str(names)]
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_zarr(out) if target == "zarr" else ad.read_h5ad(out)
    assert got.shape == (N_OBS, N_VAR)
    assert list(got.obs_names) == OBS_NAMES
    assert str(got.obs["cell_type"].dtype) == "category"
    assert sorted(got.obs["cell_type"].cat.categories) == ["A", "B", "C"]


def test_split_then_concat_round_trips_across_versions(store, tmp_path, fmt):
    ad = pytest.importorskip("anndata")

    parts = tmp_path / "parts"
    assert runner.invoke(
        app, ["split", str(store), "--by", "cell_type", "-o", str(parts)]
    ).exit_code == 0

    pieces = sorted(str(p) for p in parts.glob(f"*.{fmt}"))
    assert len(pieces) == 3, f"expected one store per category, got {pieces}"

    merged = tmp_path / f"merged.{fmt}"
    result = runner.invoke(app, ["concat", *pieces, "-o", str(merged)])
    assert result.exit_code == 0, _out(result)

    got = ad.read_zarr(merged) if fmt == "zarr" else ad.read_h5ad(merged)
    assert got.shape == (N_OBS, N_VAR)
    assert sorted(got.obs_names) == sorted(OBS_NAMES)


def test_query_filtering_works_on_every_version(store, tmp_path, fmt):
    ad = pytest.importorskip("anndata")

    out = tmp_path / f"filtered.{fmt}"
    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "-q", "cell_type == A"]
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_zarr(out) if fmt == "zarr" else ad.read_h5ad(out)
    assert list(got.obs_names) == ["cell_0", "cell_2", "cell_5"]

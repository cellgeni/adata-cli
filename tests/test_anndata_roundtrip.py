"""Interoperability tests against the real anndata library.

These are the tests that matter most. Everything else in the suite checks the
CLI against stores the suite itself built, which cannot catch the case that
actually broke: anndata changing how it writes a file. Here anndata writes the
fixtures and reads the results back, in both directions and on both backends.

Skipped when anndata is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app

ad = pytest.importorskip("anndata", reason="anndata is required for round-trip tests")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()

N_OBS, N_VAR = 6, 4


def _build(tmp_path: Path, fmt: str, *, nullable_strings: bool) -> Path:
    """Write a reference store with anndata covering every common encoding.

    `nullable_strings` selects between the two on-disk shapes anndata can give
    a string column -- a plain `string-array` dataset, or the
    `nullable-string-array` group it writes by default since 0.11. The index
    itself follows the same choice, which is what broke the reader.
    """
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["A", "B", "A", "C", "B", "A"], categories=["A", "B", "C"], ordered=True
            ),
            "n_counts": np.arange(N_OBS, dtype="int32"),
            "nullable_int": pd.array([1, 2, None, 4, 5, None], dtype="Int32"),
            "nullable_bool": pd.array(
                [True, False, None, True, False, None], dtype="boolean"
            ),
            "free_text": ["a", "bb", "ccc", "d", "ee", "f"],
        },
        index=[f"cell_{i}" for i in range(N_OBS)],
    )
    var = pd.DataFrame(
        {
            "gene_ids": [f"ENSG{i}" for i in range(N_VAR)],
            "highly_variable": [True, False, True, False],
        },
        index=[f"gene_{i}" for i in range(N_VAR)],
    )
    X = sparse.csr_matrix(
        np.random.default_rng(0).poisson(1.0, (N_OBS, N_VAR)).astype("float32")
    )

    obj = ad.AnnData(X=X, obs=obs, var=var)
    obj.layers["counts"] = X.copy()
    obj.obsm["X_pca"] = np.zeros((N_OBS, 3), dtype="float32")
    obj.varm["PCs"] = np.zeros((N_VAR, 3), dtype="float32")
    obj.obsp["connectivities"] = sparse.csr_matrix(np.eye(N_OBS, dtype="float32"))
    obj.uns["a_string"] = "hello"
    obj.uns["an_int"] = 42
    obj.uns["nested"] = {"k": np.arange(5)}
    obj.raw = obj

    previous = ad.settings.allow_write_nullable_strings
    ad.settings.allow_write_nullable_strings = nullable_strings
    try:
        path = tmp_path / f"ref.{fmt}"
        if fmt == "h5ad":
            obj.write_h5ad(path)
        else:
            obj.write_zarr(path)
    finally:
        ad.settings.allow_write_nullable_strings = previous
    return path


def _read(path: Path):
    """Read a store back with anndata, treating its format complaints as errors.

    anndata raises OldFormatWarning for elements missing encoding metadata, so
    promoting it here means a store we write that is not fully spec-compliant
    fails the test rather than merely warning.
    """
    import warnings

    from anndata._warnings import OldFormatWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", OldFormatWarning)
        return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


@pytest.fixture(params=["h5ad", "zarr"])
def fmt(request) -> str:
    return request.param


@pytest.fixture(params=[True, False], ids=["nullable-strings", "string-arrays"])
def reference(request, fmt, tmp_path) -> Path:
    return _build(tmp_path, fmt, nullable_strings=request.param)


def test_view_reads_anndata_output(reference):
    """`view` must not choke on a file anndata just wrote."""
    result = runner.invoke(app, ["view", str(reference)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert f"{N_OBS} × {N_VAR}" in result.stdout


def test_ls_reads_anndata_output(reference):
    result = runner.invoke(app, ["ls", str(reference), "--long"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")


def test_export_dataframe_reads_every_column(reference):
    """Every obs column must render, whatever encoding anndata chose for it."""
    result = runner.invoke(app, ["export", "dataframe", str(reference), "obs"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    header, *rows = [ln for ln in result.stdout.splitlines() if ln]
    assert header.split(",") == [
        "_index",
        "cell_type",
        "n_counts",
        "nullable_int",
        "nullable_bool",
        "free_text",
    ], "column-order should be honoured, not the backend's own ordering"
    assert len(rows) == N_OBS
    # Masked entries render empty rather than as a sentinel.
    assert rows[2].split(",")[3] == ""


def test_export_dataframe_accepts_arbitrary_paths(reference):
    """Dataframes outside obs/var are exportable too (issue #4)."""
    result = runner.invoke(app, ["export", "dataframe", str(reference), "raw/var"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "gene_ids" in result.stdout


@pytest.mark.parametrize("out_fmt", ["h5ad", "zarr"])
def test_subset_roundtrips_through_anndata(reference, out_fmt, tmp_path):
    """A subset must be readable by anndata with every dtype intact.

    Covers all four backend pairings, including the HDF5<->Zarr crossings
    where a string dtype has to be translated rather than copied.
    """
    names = tmp_path / "keep.txt"
    names.write_text("cell_0\ncell_2\n")
    out = tmp_path / f"subset.{out_fmt}"

    result = runner.invoke(
        app, ["subset", str(reference), "-o", str(out), "--obs", str(names)]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    got = _read(out)
    assert got.shape == (2, N_VAR)
    assert list(got.obs_names) == ["cell_0", "cell_2"]

    dtypes = got.obs.dtypes.astype(str).to_dict()
    assert dtypes["cell_type"] == "category"
    assert dtypes["nullable_int"] == "Int32"
    assert dtypes["nullable_bool"] == "boolean"
    assert got.obs["cell_type"].cat.ordered is True
    assert list(got.obs["cell_type"].cat.categories) == ["A", "B", "C"]

    assert got.obs["nullable_int"].isna().tolist() == [False, True]
    assert got.raw is not None, "raw/ must survive subsetting"
    assert got.raw.shape == (2, N_VAR)
    assert "counts" in got.layers


def test_subset_matches_var_on_raws_own_axis(reference, tmp_path):
    """raw/ carries its own var axis and must be matched against it."""
    names = tmp_path / "vkeep.txt"
    names.write_text("gene_0\ngene_2\n")
    out = tmp_path / "subset_var.h5ad"

    result = runner.invoke(
        app, ["subset", str(reference), "-o", str(out), "--var", str(names)]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    got = _read(out)
    assert list(got.var_names) == ["gene_0", "gene_2"]
    assert list(got.raw.var_names) == ["gene_0", "gene_2"]


def test_csv_roundtrip_preserves_categoricals(reference, tmp_path):
    """obs -> CSV -> obs must not degrade a categorical into free text."""
    csv = tmp_path / "obs.csv"
    out = tmp_path / "reimported.h5ad"

    assert (
        runner.invoke(
            app, ["export", "dataframe", str(reference), "obs", "-o", str(csv)]
        ).exit_code
        == 0
    )
    result = runner.invoke(
        app,
        ["import", "dataframe", str(reference), "obs", str(csv),
         "-o", str(out), "-i", "_index"],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    got = _read(out)
    assert str(got.obs["cell_type"].dtype) == "category"
    assert list(got.obs["cell_type"]) == ["A", "B", "A", "C", "B", "A"]


def test_json_roundtrip_covers_every_scalar_kind(reference, tmp_path):
    """uns values must come back as the Python types they went in as."""
    payload = tmp_path / "payload.json"
    payload.write_text(
        '{"title":"run","n":100,"rate":0.5,"flag":true,"nothing":null,'
        '"labels":["a","b"],"weights":[1.5,2.5]}'
    )
    out = tmp_path / "with_uns.h5ad"

    result = runner.invoke(
        app,
        ["import", "dict", str(reference), "uns/run", str(payload), "-o", str(out)],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    run = _read(out).uns["run"]
    assert run["title"] == "run"
    assert int(run["n"]) == 100
    assert float(run["rate"]) == pytest.approx(0.5)
    assert bool(run["flag"]) is True
    assert run["nothing"] is None
    assert list(run["labels"]) == ["a", "b"]
    assert list(run["weights"]) == pytest.approx([1.5, 2.5])


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_sparse_subset_matches_scipy(tmp_path, layout):
    """Block-streamed sparse subsetting must equal scipy's own result exactly.

    Uses a matrix large enough to span several blocks, so the block boundary
    handling and the minor-axis remap are actually exercised.
    """
    rng = np.random.default_rng(7)
    n, m = 2000, 800
    X = sparse.random(n, m, density=0.05, format="csr", dtype="float32", random_state=7)

    obj = ad.AnnData(
        X=X if layout == "csr" else X.tocsc(),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(m)]),
    )
    src = tmp_path / "big.h5ad"
    obj.write_h5ad(src)

    keep_obs = sorted(rng.choice(n, 500, replace=False))
    keep_var = sorted(rng.choice(m, 300, replace=False))
    (tmp_path / "obs.txt").write_text("\n".join(f"c{i}" for i in keep_obs))
    (tmp_path / "var.txt").write_text("\n".join(f"g{i}" for i in keep_var))

    out = tmp_path / "big_subset.h5ad"
    result = runner.invoke(
        app,
        ["subset", str(src), "-o", str(out),
         "--obs", str(tmp_path / "obs.txt"), "--var", str(tmp_path / "var.txt")],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    got = ad.read_h5ad(out)
    expected = obj.X[keep_obs][:, keep_var]
    assert got.shape == (len(keep_obs), len(keep_var))
    assert abs(got.X - expected).nnz == 0
    assert type(got.X).__name__ == f"{layout}_matrix"


# ---------------------------------------------------------------------------
# regressions from review of #6


@pytest.mark.parametrize("fmt", ["h5ad", "zarr"])
def test_json_import_keeps_nested_string_arrays_rectangular(tmp_path, fmt):
    """A nested string list was flattened into a vector on import."""
    store = tmp_path / f"base.{fmt}"
    obj = ad.AnnData(
        X=np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    obj.write_zarr(store) if fmt == "zarr" else obj.write_h5ad(store)

    payload = tmp_path / "nested.json"
    payload.write_text('{"grid": [["a","b"],["c","d"]], "nums": [[1,2],[3,4]]}')

    result = runner.invoke(
        app, ["import", "dict", str(store), "uns/t", str(payload), "--inplace"]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    uns = _read(store).uns["t"]
    assert uns["grid"].shape == (2, 2), "a 2x2 string grid must not flatten"
    assert uns["nums"].shape == (2, 2)
    assert [list(row) for row in uns["grid"]] == [["a", "b"], ["c", "d"]]


@pytest.mark.parametrize("fmt", ["h5ad", "zarr"])
def test_json_null_survives_a_full_round_trip(tmp_path, fmt):
    """Exporting a `null` element emitted its storage placeholder, not None.

    The placeholder differs per backend -- an h5py.Empty, or a 0-d zarr bool --
    so neither serialised back to JSON null.
    """
    store = tmp_path / f"base.{fmt}"
    obj = ad.AnnData(
        X=np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    obj.write_zarr(store) if fmt == "zarr" else obj.write_h5ad(store)

    source = {
        "grid": [["a", "b"], ["c", "d"]],
        "nums": [[1, 2], [3, 4]],
        "nothing": None,
        "title": "run",
    }
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(source))

    assert runner.invoke(
        app, ["import", "dict", str(store), "uns/t", str(payload), "--inplace"]
    ).exit_code == 0

    out = tmp_path / "out.json"
    result = runner.invoke(
        app, ["export", "dict", str(store), "uns/t", "-o", str(out)]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    assert json.loads(out.read_text()) == source, "JSON must round-trip exactly"
    assert _read(store).uns["t"]["nothing"] is None


def test_writes_into_a_consolidated_zarr_store_are_visible(tmp_path):
    """Edits to an anndata-written .zarr reported success but vanished.

    anndata writes a consolidated metadata index at the root. New members do
    land on disk, but every reader that honours the index -- anndata included
    -- keeps reading the stale snapshot, so the write looks like a no-op.
    Stores the CLI writes itself are not consolidated, which is why this only
    showed up against anndata's output.
    """
    zarr = pytest.importorskip("zarr")

    store = tmp_path / "base.zarr"
    ad.AnnData(
        X=np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_zarr(store)

    root = json.loads((store / "zarr.json").read_text())
    assert root.get("consolidated_metadata"), "fixture must be consolidated"

    payload = tmp_path / "p.json"
    payload.write_text('{"answer": 42}')
    result = runner.invoke(
        app, ["import", "dict", str(store), "uns/t", str(payload), "--inplace"]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    # Visible through the consolidated index, not just on disk.
    assert "t" in list(zarr.open_group(str(store))["uns"].keys())
    assert int(ad.read_zarr(store).uns["t"]["answer"]) == 42


def test_subset_of_a_consolidated_zarr_store_round_trips(tmp_path):
    """The same staleness would affect any command that writes a .zarr."""
    store = tmp_path / "base.zarr"
    ad.AnnData(
        X=np.ones((4, 2), dtype="float32"),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_zarr(store)

    names = tmp_path / "keep.txt"
    names.write_text("c0\nc2\n")
    out = tmp_path / "sub.zarr"

    result = runner.invoke(
        app, ["subset", str(store), "-o", str(out), "--obs", str(names)]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    got = ad.read_zarr(out)
    assert got.shape == (2, 2)
    assert list(got.obs_names) == ["c0", "c2"]

"""Tests for `adata convert`.

scipy is the oracle throughout. A transpose or a cast is only correct if it
agrees exactly with what scipy would have produced from the same matrix, and
"looks plausible" is not a standard worth having for something that rewrites
people's data.
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
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _matrix(n_obs=40, n_var=25, density=0.2, layout="csr", seed=0, integral=True):
    rng = np.random.default_rng(seed)
    matrix = sparse.random(
        n_obs, n_var, density=density, format="csr", dtype="float64",
        random_state=rng,
    )
    if integral:
        # Counts, which is what issue #13 is about: float32 holds them exactly.
        matrix.data = np.round(matrix.data * 100)
    return matrix.tocsc() if layout == "csc" else matrix


def _store(path: Path, matrix, *, layers=None, raw=False) -> Path:
    obj = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(matrix.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(matrix.shape[1])]),
    )
    for name, value in (layers or {}).items():
        obj.layers[name] = value
    if raw:
        obj.raw = obj
    obj.write_h5ad(path)
    return path


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


# ---------------------------------------------------------------------------
# layout


@pytest.mark.parametrize("in_memory", [False, True], ids=["streaming", "in-memory"])
@pytest.mark.parametrize("source,target", [("csr", "csc"), ("csc", "csr")])
def test_transpose_matches_scipy_exactly(tmp_path, source, target, in_memory):
    """Both paths, both directions, against scipy's own conversion.

    Having two implementations is only worth it if they agree; that is the
    whole reason the streaming one is not checked merely against itself.
    """
    matrix = _matrix(layout=source)
    store = _store(tmp_path / "in.h5ad", matrix)

    out = tmp_path / f"{source}-{target}-{in_memory}.h5ad"
    argv = ["convert", str(store), "X", "-o", str(out), "--layout", target]
    if in_memory:
        argv.append("--in-memory")
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(out).X
    expected = matrix.tocsc() if target == "csc" else matrix.tocsr()
    assert got.format == target
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.data, expected.data)


def test_transposing_twice_is_the_identity(tmp_path):
    matrix = _matrix()
    store = _store(tmp_path / "in.h5ad", matrix)

    mid, back = tmp_path / "mid.h5ad", tmp_path / "back.h5ad"
    assert runner.invoke(
        app, ["convert", str(store), "X", "-o", str(mid), "--layout", "csc"]
    ).exit_code == 0
    assert runner.invoke(
        app, ["convert", str(mid), "X", "-o", str(back), "--layout", "csr"]
    ).exit_code == 0

    result = ad.read_h5ad(back).X
    assert result.format == "csr"
    assert np.array_equal(result.indptr, matrix.indptr)
    assert np.array_equal(result.indices, matrix.indices)
    assert np.array_equal(result.data, matrix.data)


def test_the_two_transpose_paths_agree_on_an_awkward_matrix(tmp_path):
    """Empty rows, a full row, and a single-entry column all in one."""
    dense = np.zeros((6, 5))
    dense[0, :] = [1, 2, 3, 4, 5]   # full row
    dense[3, 2] = 7                  # lone entry
    # rows 1, 2, 4, 5 stay empty
    matrix = sparse.csr_matrix(dense)
    store = _store(tmp_path / "in.h5ad", matrix)

    outputs = {}
    for tag, extra in (("stream", []), ("memory", ["--in-memory"])):
        out = tmp_path / f"{tag}.h5ad"
        assert runner.invoke(
            app,
            ["convert", str(store), "X", "-o", str(out), "--layout", "csc", *extra],
        ).exit_code == 0
        outputs[tag] = ad.read_h5ad(out).X

    expected = matrix.tocsc()
    for tag, got in outputs.items():
        assert np.array_equal(got.indptr, expected.indptr), tag
        assert np.array_equal(got.indices, expected.indices), tag
        assert np.array_equal(got.data, expected.data), tag


# ---------------------------------------------------------------------------
# dtype -- what issue #13 asked for


def test_counts_stored_as_float64_convert_to_float32_and_shrink(tmp_path):
    """The reported case, end to end, including that the file gets smaller.

    An early version produced an output eight times the input, because it
    created the arrays with a fixed 65,536-element chunk and promoted int32
    indices to int64. A conversion asked for to halve a file must not
    enlarge it, so the size is part of the test.
    """
    import h5py

    matrix = _matrix(n_obs=200, n_var=120, density=0.1)
    store = _store(tmp_path / "in.h5ad", matrix)
    out = tmp_path / "f32.h5ad"

    result = runner.invoke(
        app, ["convert", str(store), "X", "-o", str(out), "--dtype", "float32"]
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(out).X
    assert got.dtype == np.dtype("float32")
    assert np.array_equal(got.toarray(), matrix.toarray().astype("float32"))

    def stored(path):
        with h5py.File(path) as handle:
            return sum(
                handle["X"][key].id.get_storage_size()
                for key in ("data", "indices", "indptr")
            )

    assert stored(out) < stored(store), (
        f"X grew from {stored(store):,} to {stored(out):,} bytes on a "
        "float64 -> float32 conversion"
    )


def test_the_index_dtype_is_preserved_unless_asked(tmp_path):
    """Defaulting to int64 silently doubled every int32 store's indices."""
    import h5py

    store = _store(tmp_path / "in.h5ad", _matrix())
    with h5py.File(store) as handle:
        source_dtype = handle["X/indices"].dtype

    out = tmp_path / "out.h5ad"
    assert runner.invoke(
        app, ["convert", str(store), "X", "-o", str(out), "--dtype", "float32"]
    ).exit_code == 0

    with h5py.File(out) as handle:
        assert handle["X/indices"].dtype == source_dtype


def test_indices_can_be_narrowed_when_asked(tmp_path):
    import h5py

    store = _store(tmp_path / "in.h5ad", _matrix())
    out = tmp_path / "out.h5ad"
    result = runner.invoke(
        app,
        ["convert", str(store), "X", "-o", str(out), "--indices-dtype", "int32"],
    )
    assert result.exit_code == 0, _out(result)

    with h5py.File(out) as handle:
        assert handle["X/indices"].dtype == np.dtype("int32")
        assert handle["X/indptr"].dtype == np.dtype("int32")
    assert np.array_equal(ad.read_h5ad(out).X.toarray(), _matrix().toarray())


# ---------------------------------------------------------------------------
# refusing before writing


def test_a_lossy_cast_is_refused_and_nothing_is_written(tmp_path):
    """The store must be left alone, not half converted."""
    matrix = _matrix(integral=False)
    matrix.data = matrix.data + 0.123456789012345
    store = _store(tmp_path / "in.h5ad", matrix)
    out = tmp_path / "out.h5ad"

    result = runner.invoke(
        app, ["convert", str(store), "X", "-o", str(out), "--dtype", "float32"]
    )
    assert result.exit_code == 1
    text = _out(result)
    assert "lossy" in text and "--force" in text
    assert "do not round-trip" in text
    assert not out.exists(), "a refused conversion must leave no output behind"


def test_force_converts_anyway(tmp_path):
    matrix = _matrix(integral=False)
    matrix.data = matrix.data + 0.123456789012345
    store = _store(tmp_path / "in.h5ad", matrix)
    out = tmp_path / "out.h5ad"

    result = runner.invoke(
        app,
        ["convert", str(store), "X", "-o", str(out), "--dtype", "float32", "--force"],
    )
    assert result.exit_code == 0, _out(result)
    assert ad.read_h5ad(out).X.dtype == np.dtype("float32")


def test_a_lossless_cast_says_so(tmp_path):
    """Worth telling the user: it is the question they were asking."""
    store = _store(tmp_path / "in.h5ad", _matrix())
    result = runner.invoke(
        app,
        ["convert", str(store), "X", "-o", str(tmp_path / "o.h5ad"),
         "--dtype", "float32"],
    )
    assert result.exit_code == 0
    assert "round-trips" in _out(result)


def test_an_index_dtype_too_small_to_address_the_matrix_is_refused(tmp_path):
    from adata.core.convert import check_index_dtype

    store = _store(tmp_path / "in.h5ad", _matrix())
    with pytest.raises(ValueError, match="cannot index this matrix"):
        # 3 billion columns cannot be addressed by int32, whatever the nnz.
        check_index_dtype(
            _FakeGroup(nnz=10), np.dtype("int32"), (10, 3_000_000_000)
        )
    assert store.exists()


class _FakeGroup:
    """Just enough of a sparse group for the index-range check."""

    def __init__(self, nnz: int):
        self._nnz = nnz

    def __getitem__(self, key):
        assert key == "indices"
        return type("D", (), {"shape": (self._nnz,)})()


# ---------------------------------------------------------------------------
# density


def test_densify_then_sparsify_round_trips(tmp_path):
    matrix = _matrix(density=0.3)
    store = _store(tmp_path / "in.h5ad", matrix)

    dense_path, sparse_path = tmp_path / "dense.h5ad", tmp_path / "sparse.h5ad"
    assert runner.invoke(
        app,
        ["convert", str(store), "X", "-o", str(dense_path), "--layout", "dense",
         "--force"],
    ).exit_code == 0
    got_dense = ad.read_h5ad(dense_path).X
    assert not sparse.issparse(got_dense)
    assert np.array_equal(_dense(got_dense), matrix.toarray())

    assert runner.invoke(
        app,
        ["convert", str(dense_path), "X", "-o", str(sparse_path),
         "--layout", "csr"],
    ).exit_code == 0
    got = ad.read_h5ad(sparse_path).X
    assert got.format == "csr"
    assert np.array_equal(got.toarray(), matrix.toarray())


def test_densifying_a_sparse_matrix_is_refused_when_it_would_explode(tmp_path):
    """0.5% dense over a wide matrix is exactly the case that hurts."""
    matrix = _matrix(n_obs=200, n_var=2000, density=0.005)
    store = _store(tmp_path / "in.h5ad", matrix)
    out = tmp_path / "out.h5ad"

    result = runner.invoke(
        app, ["convert", str(store), "X", "-o", str(out), "--layout", "dense"]
    )
    assert result.exit_code == 1
    text = _out(result)
    assert "would grow" in text and "--force" in text
    assert not out.exists()


def test_sparsifying_a_mostly_dense_matrix_warns(tmp_path):
    dense = np.ones((20, 10))
    store = _store(tmp_path / "in.h5ad", sparse.csr_matrix(dense))
    dense_path = tmp_path / "dense.h5ad"
    assert runner.invoke(
        app,
        ["convert", str(store), "X", "-o", str(dense_path), "--layout", "dense",
         "--force"],
    ).exit_code == 0

    result = runner.invoke(
        app,
        ["convert", str(dense_path), "X", "-o", str(tmp_path / "s.h5ad"),
         "--layout", "csr"],
    )
    assert result.exit_code == 0, _out(result)
    assert "nonzero" in _out(result)


# ---------------------------------------------------------------------------
# selection and plumbing


def test_all_converts_x_layers_and_raw(tmp_path):
    matrix = _matrix()
    store = _store(
        tmp_path / "in.h5ad", matrix, layers={"counts": matrix.copy()}, raw=True
    )
    out = tmp_path / "out.h5ad"

    result = runner.invoke(
        app, ["convert", str(store), "--all", "-o", str(out), "--layout", "csc"]
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(out)
    assert got.X.format == "csc"
    assert got.layers["counts"].format == "csc"
    assert got.raw is not None and got.raw.X.format == "csc"
    assert list(got.obs_names) == [f"c{i}" for i in range(matrix.shape[0])]


def test_untargeted_elements_are_carried_over_untouched(tmp_path):
    matrix = _matrix()
    obj = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "b"] * (matrix.shape[0] // 2))},
            index=[f"c{i}" for i in range(matrix.shape[0])],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(matrix.shape[1])]),
    )
    obj.layers["untouched"] = matrix.copy()
    obj.obsm["X_pca"] = np.arange(matrix.shape[0] * 3, dtype="float32").reshape(-1, 3)
    obj.uns["note"] = "keep me"
    store = tmp_path / "in.h5ad"
    obj.write_h5ad(store)

    out = tmp_path / "out.h5ad"
    assert runner.invoke(
        app, ["convert", str(store), "X", "-o", str(out), "--layout", "csc"]
    ).exit_code == 0

    got = ad.read_h5ad(out)
    assert got.X.format == "csc"
    assert got.layers["untouched"].format == "csr", "an untargeted layer changed"
    assert got.uns["note"] == "keep me"
    assert np.array_equal(got.obsm["X_pca"], obj.obsm["X_pca"])
    assert list(got.obs["group"]) == list(obj.obs["group"])


def test_inplace_replaces_the_source(tmp_path):
    matrix = _matrix()
    store = _store(tmp_path / "in.h5ad", matrix)

    result = runner.invoke(
        app, ["convert", str(store), "X", "--inplace", "--dtype", "float32"]
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(store)
    assert got.X.dtype == np.dtype("float32")
    assert np.array_equal(got.X.toarray(), matrix.toarray().astype("float32"))
    assert not list(tmp_path.glob("*convert-tmp*")), "temp file left behind"


def test_converting_to_zarr_keeps_the_values(tmp_path):
    matrix = _matrix()
    store = _store(tmp_path / "in.h5ad", matrix)
    out = tmp_path / "out.zarr"

    result = runner.invoke(
        app, ["convert", str(store), "X", "-o", str(out), "--layout", "csc"]
    )
    assert result.exit_code == 0, _out(result)
    got = ad.read_zarr(out).X
    assert got.format == "csc"
    assert np.array_equal(got.toarray(), matrix.toarray())


# ---------------------------------------------------------------------------
# argument handling


def test_convert_needs_something_to_do(tmp_path):
    store = _store(tmp_path / "in.h5ad", _matrix())
    result = runner.invoke(
        app, ["convert", str(store), "X", "-o", str(tmp_path / "o.h5ad")]
    )
    assert result.exit_code == 1
    assert "at least one of --dtype" in _out(result)


def test_convert_needs_an_output(tmp_path):
    store = _store(tmp_path / "in.h5ad", _matrix())
    result = runner.invoke(app, ["convert", str(store), "X", "--dtype", "float32"])
    assert result.exit_code == 1
    assert "Output file is required" in _out(result)


@pytest.mark.parametrize(
    "flag,value,expected",
    [
        ("--dtype", "complex128", "Unknown dtype"),
        ("--indices-dtype", "float32", "Unknown dtype"),
        ("--layout", "coo", "--layout must be one of"),
    ],
)
def test_bad_values_are_rejected_by_name(tmp_path, flag, value, expected):
    store = _store(tmp_path / "in.h5ad", _matrix())
    result = runner.invoke(
        app,
        ["convert", str(store), "X", "-o", str(tmp_path / "o.h5ad"), flag, value],
    )
    assert result.exit_code == 1
    assert expected in _out(result)


def test_converting_something_that_is_not_a_matrix_says_so(tmp_path):
    store = _store(tmp_path / "in.h5ad", _matrix())
    result = runner.invoke(
        app,
        ["convert", str(store), "obs", "-o", str(tmp_path / "o.h5ad"),
         "--dtype", "float32"],
    )
    assert result.exit_code == 1
    assert "Not a matrix" in _out(result)


def test_a_missing_entry_names_the_path(tmp_path):
    store = _store(tmp_path / "in.h5ad", _matrix())
    result = runner.invoke(
        app,
        ["convert", str(store), "layers/nope", "-o", str(tmp_path / "o.h5ad"),
         "--dtype", "float32"],
    )
    assert result.exit_code == 1
    assert "layers/nope" in _out(result)

"""The streamed sparse writers: dtypes, storage settings, and one-pass split.

Two defects shared this code. Every sparse output of subset, split and
concat widened `indices`/`indptr` to int64 and dropped the source's
compression, so outputs came out about twice the size anndata writes; and
split subset once per group, reading all of X once per group whenever the
groups were interleaved. These tests pin the fixes against anndata and
scipy rather than against the code itself.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from rich.console import Console

from adata.commands.split import split_store
from adata.core import subset as subset_mod
from adata.core.concat import _concat_index_dtypes, concat_on_disk
from adata.core.subset import (
    fan_out_sparse_matrix,
    split_h5ad,
    subset_h5ad,
    subset_sparse_matrix_group,
)
from adata.storage import open_store

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

QUIET = Console(quiet=True)


def _anndata(n_obs=40, n_var=12, *, layout="csr", seed=0, raw=True):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.6, (n_obs, n_var)).astype("float32")
    X = sparse.csr_matrix(dense) if layout == "csr" else (
        sparse.csc_matrix(dense) if layout == "csc" else dense
    )
    obs = pd.DataFrame(
        {
            # Interleaved, the case that made split read X once per group.
            "group": pd.Categorical([f"g{i % 3}" for i in range(n_obs)]),
            "score": np.arange(n_obs, dtype="float32"),
        },
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"kind": pd.Categorical([f"k{i % 2}" for i in range(n_var)])},
        index=[f"v{i}" for i in range(n_var)],
    )
    obj = ad.AnnData(X=X, obs=obs, var=var)
    obj.layers["counts"] = sparse.csc_matrix(dense * 2)
    obj.obsm["X_pca"] = rng.normal(size=(n_obs, 3)).astype("float32")
    obj.varm["loadings"] = rng.normal(size=(n_var, 2)).astype("float32")
    obj.obsp["conn"] = sparse.csr_matrix(
        rng.random((n_obs, n_obs)).astype("float32") * (rng.random((n_obs, n_obs)) < 0.2)
    )
    if raw:
        obj.raw = obj
    return obj


def _write(obj, path: Path) -> Path:
    if path.suffix == ".zarr":
        obj.write_zarr(path)
    else:
        obj.write_h5ad(path, compression="lzf")
    return path


def _dense(m):
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


def _assert_same(a, b):
    assert list(a.obs_names) == list(b.obs_names)
    assert list(a.var_names) == list(b.var_names)
    np.testing.assert_array_equal(_dense(a.X), _dense(b.X))
    np.testing.assert_array_equal(_dense(a.layers["counts"]), _dense(b.layers["counts"]))
    np.testing.assert_array_equal(a.obsm["X_pca"], b.obsm["X_pca"])
    np.testing.assert_array_equal(a.varm["loadings"], b.varm["loadings"])
    np.testing.assert_array_equal(_dense(a.obsp["conn"]), _dense(b.obsp["conn"]))
    pd.testing.assert_frame_equal(a.obs, b.obs)
    if a.raw is not None or b.raw is not None:
        np.testing.assert_array_equal(_dense(a.raw.X), _dense(b.raw.X))
        assert list(a.raw.var_names) == list(b.raw.var_names)


# ---------------------------------------------------------------------------
# dtypes and storage settings


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_subset_keeps_the_source_index_dtypes(tmp_path, suffix, layout):
    src = _write(_anndata(layout=layout), tmp_path / f"src{suffix}")
    out = tmp_path / f"out{suffix}"
    subset_h5ad(src, out, None, None, console=QUIET, obs_indices=np.arange(0, 40, 3))

    with open_store(src, "r") as s, open_store(out, "r") as o:
        for key in ("indices", "indptr", "data"):
            assert o.root["X"][key].dtype == s.root["X"][key].dtype, key
        # scipy writes int32 for a matrix this small; the defect was int64.
        assert o.root["X"]["indices"].dtype == np.int32


def test_subset_keeps_int64_indices_int64(tmp_path):
    obj = _anndata(raw=False)
    obj.X = sparse.csr_matrix(obj.X)
    obj.X.indices = obj.X.indices.astype(np.int64)
    obj.X.indptr = obj.X.indptr.astype(np.int64)
    src = _write(obj, tmp_path / "wide.h5ad")
    with h5py.File(src, "r") as f:
        assert f["X/indices"].dtype == np.int64, "fixture must really be int64"

    out = tmp_path / "out.h5ad"
    subset_h5ad(src, out, None, None, console=QUIET, obs_indices=np.arange(5))
    with h5py.File(out, "r") as f:
        assert f["X/indices"].dtype == np.int64
        assert f["X/indptr"].dtype == np.int64


def test_subset_keeps_the_source_compression(tmp_path):
    src = _write(_anndata(), tmp_path / "src.h5ad")
    out = tmp_path / "out.h5ad"
    subset_h5ad(src, out, None, None, console=QUIET, obs_indices=np.arange(0, 40, 2))

    with h5py.File(src, "r") as s, h5py.File(out, "r") as o:
        for key in ("data", "indices", "indptr"):
            assert s[f"X/{key}"].compression == "lzf"
            assert o[f"X/{key}"].compression == "lzf", key


# ---------------------------------------------------------------------------
# the vectorised gather, against scipy


@pytest.mark.parametrize("layout", ["csr", "csc"])
@pytest.mark.parametrize("seed", range(6))
def test_sparse_subset_matches_scipy(tmp_path, layout, seed):
    rng = np.random.default_rng(seed)
    n_obs, n_var = 57, 23
    dense = rng.poisson(0.5, (n_obs, n_var)).astype("float32")
    dense[rng.random(n_obs) < 0.3] = 0  # empty rows
    matrix = sparse.csr_matrix(dense) if layout == "csr" else sparse.csc_matrix(dense)
    src = _write(ad.AnnData(X=matrix), tmp_path / "m.h5ad")

    def pick(n):
        choice = rng.integers(0, 4)
        if choice == 0:
            return None
        if choice == 1:
            return np.array([], dtype=np.int64)
        return np.sort(rng.choice(n, size=rng.integers(1, n), replace=False))

    targets = [(pick(n_obs), pick(n_var)) for _ in range(4)]
    out = tmp_path / "out.h5ad"
    with h5py.File(src, "r") as s, h5py.File(out, "w") as o:
        fan_out_sparse_matrix(
            s["X"],
            "X",
            [(o.create_group(f"t{i}"), ob, va) for i, (ob, va) in enumerate(targets)],
            # Small blocks, so selections straddle block boundaries and some
            # blocks hold nothing a target wants.
            chunk_major=5,
        )
    for i, (ob, va) in enumerate(targets):
        expected = dense
        if ob is not None:
            expected = expected[ob]
        if va is not None:
            expected = expected[:, va]
        with h5py.File(out, "r") as o:
            g = o[f"t{i}/X"]
            cls = sparse.csr_matrix if layout == "csr" else sparse.csc_matrix
            got = cls(
                (g["data"][...], g["indices"][...], g["indptr"][...]),
                shape=tuple(g.attrs["shape"]),
            )
        np.testing.assert_array_equal(got.toarray(), expected)
        assert got.has_sorted_indices


def test_unsorted_selection_is_refused(tmp_path):
    src = _write(_anndata(raw=False), tmp_path / "s.h5ad")
    with h5py.File(src, "r") as s, h5py.File(tmp_path / "o.h5ad", "w") as o:
        with pytest.raises(ValueError, match="sorted and unique"):
            subset_sparse_matrix_group(s["X"], o, "X", np.array([3, 1]), None)


# ---------------------------------------------------------------------------
# split in one pass


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_split_matches_subset_per_group(tmp_path, suffix, layout, axis):
    src = _write(_anndata(layout=layout), tmp_path / f"src{suffix}")
    column = "group" if axis == "obs" else "kind"
    planned = split_store(
        src, column, tmp_path / "parts", QUIET, axis=axis, manifest=False
    )
    assert len(planned) == (3 if axis == "obs" else 2)

    source = ad.read_h5ad(src) if suffix == ".h5ad" else ad.read_zarr(src)
    frame = source.obs if axis == "obs" else source.var
    for label, path, count in planned:
        indices = np.flatnonzero(frame[column].astype(str).to_numpy() == label)
        expected_path = tmp_path / f"expected-{label}{suffix}"
        subset_h5ad(
            src,
            expected_path,
            None,
            None,
            console=QUIET,
            obs_indices=indices if axis == "obs" else None,
            var_indices=indices if axis == "var" else None,
        )
        read = ad.read_h5ad if suffix == ".h5ad" else ad.read_zarr
        got, expected = read(path), read(expected_path)
        assert (got.n_obs if axis == "obs" else got.n_vars) == count
        _assert_same(got, expected)
        # And against anndata's own slicing, not just our other code path.
        want = source[indices] if axis == "obs" else source[:, indices]
        np.testing.assert_array_equal(_dense(got.X), _dense(want.X))


def test_split_in_batches_matches_split_in_one(tmp_path, monkeypatch):
    src = _write(_anndata(n_obs=50), tmp_path / "src.h5ad")
    source = ad.read_h5ad(src)
    groups = [np.arange(i, 50, 5) for i in range(5)]

    whole = [(tmp_path / f"w{i}.h5ad", g, None) for i, g in enumerate(groups)]
    split_h5ad(src, whole, console=QUIET)

    monkeypatch.setattr(subset_mod, "MAX_OPEN_OUTPUTS", 2)
    batched = [(tmp_path / f"b{i}.h5ad", g, None) for i, g in enumerate(groups)]
    split_h5ad(src, batched, console=QUIET)

    for (w, g, _), (b, _, _) in zip(whole, batched):
        _assert_same(ad.read_h5ad(b), ad.read_h5ad(w))
        np.testing.assert_array_equal(_dense(ad.read_h5ad(b).X), _dense(source[g].X))


def test_split_outputs_keep_dtypes_and_compression(tmp_path):
    src = _write(_anndata(), tmp_path / "src.h5ad")
    planned = split_store(src, "group", tmp_path / "parts", QUIET, manifest=False)
    for _, path, _ in planned:
        with h5py.File(path, "r") as f:
            assert f["X/indices"].dtype == np.int32
            assert f["X/indptr"].dtype == np.int32
            assert f["X/data"].compression == "lzf"


# ---------------------------------------------------------------------------
# concat


def test_concat_keeps_int32_indices_and_compression(tmp_path):
    parts = [
        _write(_anndata(n_obs=10, seed=s, raw=False), tmp_path / f"p{s}.h5ad")
        for s in range(3)
    ]
    for s, p in enumerate(parts):
        obj = ad.read_h5ad(p)
        obj.obs_names = [f"p{s}-{n}" for n in obj.obs_names]
        del obj.obsp["conn"]
        obj.write_h5ad(p, compression="lzf")
    out = tmp_path / "all.h5ad"
    concat_on_disk(parts, out, QUIET)

    with h5py.File(out, "r") as f:
        for element in ("X", "layers/counts"):
            assert f[f"{element}/indices"].dtype == np.int32, element
            assert f[f"{element}/indptr"].dtype == np.int32, element
            assert f[f"{element}/data"].compression == "lzf", element

    expected = ad.concat([ad.read_h5ad(p) for p in parts])
    got = ad.read_h5ad(out)
    np.testing.assert_array_equal(_dense(got.X), _dense(expected.X))
    np.testing.assert_array_equal(
        _dense(got.layers["counts"]), _dense(expected.layers["counts"])
    )


class _Fake:
    """Just enough of a sparse group for `_concat_index_dtypes`."""

    def __init__(self, nnz, indices=np.int32, indptr=np.int32):
        self._d = {
            "data": np.empty(0, dtype=np.float32),
            "indices": np.empty(0, dtype=indices),
            "indptr": np.empty(0, dtype=indptr),
        }
        self._nnz = nnz

    def __getitem__(self, key):
        if key == "data":
            return type("D", (), {"shape": (self._nnz,)})()
        return self._d[key]


def test_concat_index_dtypes_widen_only_when_needed():
    limit = int(np.iinfo(np.int32).max)
    small = [_Fake(10), _Fake(20)]
    assert _concat_index_dtypes(small, 1000) == (np.int32, np.int32)

    # Each input fits int32 on its own; together their nonzeros do not.
    big = [_Fake(limit // 2 + 1), _Fake(limit // 2 + 1)]
    assert _concat_index_dtypes(big, 1000) == (np.int32, np.int64)

    assert _concat_index_dtypes(small, limit + 1) == (np.int64, np.int32)

    # A wider input is never narrowed.
    mixed = [_Fake(10), _Fake(10, indices=np.int64, indptr=np.int64)]
    assert _concat_index_dtypes(mixed, 1000) == (np.int64, np.int64)


# ---------------------------------------------------------------------------
# row reads for dataframes and obsm


@pytest.mark.parametrize("block", [1, 3, 1 << 20])
def test_take_rows_reads_blocks_not_a_fancy_index(tmp_path, monkeypatch, block):
    monkeypatch.setattr(subset_mod, "TAKE_BLOCK_ROWS", block)
    rng = np.random.default_rng(0)
    numbers = rng.normal(size=(30, 2))
    strings = np.array([f"s{i}" for i in range(30)], dtype=object)
    with h5py.File(tmp_path / "t.h5", "w") as f:
        f["n"] = numbers
        f.create_dataset("s", data=strings, dtype=h5py.string_dtype())
        for indices in (
            np.array([], dtype=np.int64),
            np.array([7]),
            np.array([0, 1, 2, 29]),
            np.sort(rng.choice(30, 11, replace=False)),
        ):
            np.testing.assert_array_equal(
                subset_mod._take_rows(f["n"], indices), numbers[indices]
            )
            got = subset_mod._take_rows(f["s"], indices)
            assert [x.decode() if isinstance(x, bytes) else x for x in got] == list(
                strings[indices]
            )
        with pytest.raises(ValueError, match="sorted and unique"):
            subset_mod._take_rows(f["n"], np.array([2, 1]))

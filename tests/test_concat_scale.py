"""`concat` at scale: CSC inputs, and more inputs than can be open at once.

What these check is the result, against `anndata.concat`, on the paths that
scale: CSC matrices rebuilt through buckets on disk (F011), and inputs opened
as they are needed rather than all at once (F076). How much they cost is
`test_performance.py`'s business.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest
from rich.console import Console

from adata.core import concat as concat_mod
from adata.core.concat import concat_on_disk
from adata.storage import open_store

from tests.conftest import backend_path, backend_zarr_format

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

QUIET = Console(quiet=True)


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _write(obj, path: Path, backend: str) -> Path:
    if path.suffix == ".zarr":
        with ad.settings.override(zarr_write_format=backend_zarr_format(backend)):
            obj.write_zarr(path)
    else:
        obj.write_h5ad(path)
    return path


def _csc_input(path: Path, backend: str, tag: str, n_cells: int, genes, *, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    values = rng.poisson(1.2, (n_cells, len(genes))).astype(np.float32)
    values[:, 0] = 0  # an empty column
    obj = ad.AnnData(
        X=sparse.csc_matrix(values),
        obs=pd.DataFrame(index=[f"{tag}{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=list(genes)),
    )
    obj.layers["counts"] = sparse.csc_matrix(values * 2)
    return _write(obj, path, backend)


def _stored_csc(path: Path, element: str):
    with open_store(path, "r") as handle:
        group = handle.root[element]
        assert group.attrs["encoding-type"] == "csc_matrix"
        return sparse.csc_matrix(
            (group["data"][...], group["indices"][...], group["indptr"][...]),
            shape=tuple(int(n) for n in group.attrs["shape"]),
        )


# ---------------------------------------------------------------------------
# F011: CSC inputs


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_csc_inputs_concatenate_as_anndata_concatenates_them(tmp_path, backend, join):
    """F011: CSC concat read each column of each input separately, 2 x n_var x
    n_inputs reads that each decompressed a whole chunk: 37 times the input
    on h5ad, 572 times on Zarr. Each input is now read once, in blocks, and
    the output's columns put together through buckets on disk. Tiny blocks
    and enough entries for several buckets exercise every seam; one input
    orders its genes differently and lacks some, so columns are remapped."""
    genes = [f"g{i:02d}" for i in range(60)]
    shuffled = list(np.random.default_rng(9).permutation(genes[:50])) + ["x1", "x2"]
    a = _csc_input(backend_path(tmp_path, backend, "a"), backend, "a", 150, genes, seed=0)
    b = _csc_input(backend_path(tmp_path, backend, "b"), backend, "b", 120, shuffled, seed=1)
    c = _csc_input(backend_path(tmp_path, backend, "c"), backend, "c", 90, genes, seed=2)
    out = backend_path(tmp_path, backend, "out")
    concat_on_disk([a, b, c], out, QUIET, join=join, chunk_rows=2)

    got = _read(out)
    ref = ad.concat([_read(a), _read(b), _read(c)], join=join)
    assert list(got.var_names) == list(ref.var_names)
    for element, mine, theirs in (("X", got.X, ref.X), ("layers/counts", got.layers["counts"], ref.layers["counts"])):
        stored = _stored_csc(out, element)
        assert stored.has_sorted_indices, element
        assert stored.nnz >= 8192, "enough entries for more than one bucket"
        assert np.array_equal(mine.toarray(), theirs.toarray()), element


def test_csc_concat_keeps_the_index_dtypes_and_layout(tmp_path):
    """F011: the rewrite keeps what the per-column version kept: int32
    indices and pointers from int32 inputs, and the source's compression."""
    genes = [f"g{i}" for i in range(20)]
    paths = []
    for tag, seed in (("a", 0), ("b", 1)):
        obj = ad.AnnData(
            X=sparse.csc_matrix(np.random.default_rng(seed).poisson(1, (30, 20)).astype(np.float32)),
            obs=pd.DataFrame(index=[f"{tag}{i}" for i in range(30)]),
            var=pd.DataFrame(index=genes),
        )
        obj.write_h5ad(tmp_path / f"{tag}.h5ad", compression="gzip")
        paths.append(tmp_path / f"{tag}.h5ad")
    out = tmp_path / "out.h5ad"
    concat_on_disk(paths, out, QUIET, chunk_rows=4)
    with h5py.File(out) as f, h5py.File(paths[0]) as src:
        for name in ("data", "indices", "indptr"):
            assert f[f"X/{name}"].dtype == src[f"X/{name}"].dtype, name
        assert f["X/data"].compression == src["X/data"].compression == "gzip"
    ref = ad.concat([ad.read_h5ad(p) for p in paths])
    assert np.array_equal(ad.read_h5ad(out).X.toarray(), ref.X.toarray())


def test_csc_inputs_with_nothing_stored_concatenate(tmp_path, backend):
    """F011: no entries at all needs no buckets."""
    genes = ["g0", "g1"]
    paths = []
    for tag in "ab":
        obj = ad.AnnData(X=sparse.csc_matrix((3, 2), dtype=np.float32),
                         obs=pd.DataFrame(index=[f"{tag}{i}" for i in range(3)]),
                         var=pd.DataFrame(index=genes))
        paths.append(_write(obj, backend_path(tmp_path, backend, tag), backend))
    out = backend_path(tmp_path, backend, "out")
    concat_on_disk(paths, out, QUIET)
    got = _read(out)
    assert got.shape == (6, 2) and got.X.nnz == 0 and got.X.format == "csc"


# ---------------------------------------------------------------------------
# F076: inputs opened as they are needed


def _rich_input(path: Path, tag: str, n: int, genes, *, seed: int) -> Path:
    """An h5ad holding something of every kind concat reads per input."""
    rng = np.random.default_rng(seed)
    cells = [f"{tag}{i}" for i in range(n)]
    obs = pd.DataFrame(index=cells)
    obs["cell_type"] = pd.Categorical(rng.choice(["T", "B", f"NK{tag}"], n))
    obs["n_counts"] = rng.integers(0, 100, n)
    obs["score"] = pd.array([None if i % 3 == 0 else i for i in range(n)], dtype="Int64")
    obs["note"] = [f"{tag}-{i}" for i in range(n)]
    var = pd.DataFrame({"gene_ids": [f"E{g}" for g in genes]}, index=list(genes))
    obj = ad.AnnData(X=sparse.csc_matrix(rng.poisson(1, (n, len(genes))).astype(np.float32)),
                     obs=obs, var=var)
    obj.layers["dense"] = rng.normal(size=(n, len(genes)))
    obj.layers["csr"] = sparse.csr_matrix(rng.poisson(1, (n, len(genes))).astype(np.float32))
    obj.obsm["X_pca"] = rng.normal(size=(n, 3))
    obj.obsm["frame"] = pd.DataFrame({"r0": rng.random(n), "kind": pd.Categorical(["x"] * n)}, index=cells)
    obj.varm["loadings"] = np.arange(len(genes) * 2, dtype=np.float64).reshape(len(genes), 2)
    obj.varp["corr"] = sparse.csr_matrix(np.eye(len(genes)))
    obj.uns["title"] = "atlas"
    obj.uns["nested"] = {"a": np.arange(3), "b": "x"}
    obj.write_h5ad(path)
    return path


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_with_one_input_open_at_a_time_matches_anndata(tmp_path, monkeypatch, join):
    """F076: concat opened every input up front and held them all, so it ran
    out of file descriptors at about 253 HDF5 inputs. Inputs are now opened
    as they are needed, at most `MAX_OPEN_INPUTS` at once. With the limit at
    one, an object read from an input closed since -- which h5py invalidates --
    would fail here, so every element has to be looked up again."""
    genes = [f"g{i}" for i in range(8)]
    paths = [
        _rich_input(tmp_path / "a.h5ad", "a", 6, genes, seed=0),
        _rich_input(tmp_path / "b.h5ad", "b", 5, genes[2:] + ["x"], seed=1),
        _rich_input(tmp_path / "c.h5ad", "c", 4, genes[::-1], seed=2),
    ]
    kwargs = dict(join=join, merge="first", uns_merge="same", label="batch", keys=["a", "b", "c"],
                  chunk_rows=2)
    monkeypatch.setattr(concat_mod, "MAX_OPEN_INPUTS", 1)
    concat_on_disk(paths, tmp_path / "one.h5ad", QUIET, **kwargs)
    monkeypatch.setattr(concat_mod, "MAX_OPEN_INPUTS", 64)
    concat_on_disk(paths, tmp_path / "all.h5ad", QUIET, **kwargs)

    one, every = ad.read_h5ad(tmp_path / "one.h5ad"), ad.read_h5ad(tmp_path / "all.h5ad")
    ref = ad.concat([ad.read_h5ad(p) for p in paths], join=join, merge="first",
                    uns_merge="same", label="batch", keys=["a", "b", "c"], fill_value=0)
    for got in (one, every):
        assert list(got.var_names) == list(ref.var_names)
        assert np.array_equal(got.X.toarray(), ref.X.toarray())
        assert np.allclose(got.layers["dense"], ref.layers["dense"])
        assert np.array_equal(got.layers["csr"].toarray(), ref.layers["csr"].toarray())
        assert np.allclose(got.obsm["X_pca"], ref.obsm["X_pca"])
        assert list(got.obs.columns) == list(ref.obs.columns)
        for column in ref.obs.columns:
            assert [str(v) for v in got.obs[column]] == [str(v) for v in ref.obs[column]], column
        assert got.uns["title"] == ref.uns["title"] == "atlas"
        if join == "inner":
            # Under an outer join anndata pads an input's var columns, varm
            # and varp for the genes it lacks; concat counts that input as
            # lacking them, so `first` finds none here.
            assert sorted(got.varm) == sorted(ref.varm) == ["loadings"]
            assert sorted(got.varp) == sorted(ref.varp) == ["corr"]
            assert list(got.var["gene_ids"]) == list(ref.var["gene_ids"])
    pd.testing.assert_frame_equal(one.obs, every.obs)
    assert np.array_equal(one.X.toarray(), every.X.toarray())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file descriptor limits")
def test_more_inputs_than_the_file_descriptor_limit_concatenate(tmp_path):
    """F076: with every input held open, concat stopped with 'Too many open
    files' at about 253 HDF5 inputs under macOS's default limit of 256. Here
    the limit is 64 and there are 100 inputs."""
    paths = []
    for i in range(100):
        obj = ad.AnnData(X=np.full((2, 3), i, dtype=np.float32),
                         obs=pd.DataFrame(index=[f"c{i}-0", f"c{i}-1"]),
                         var=pd.DataFrame(index=["g0", "g1", "g2"]))
        obj.write_h5ad(tmp_path / f"in{i:03d}.h5ad")
        paths.append(str(tmp_path / f"in{i:03d}.h5ad"))
    out = tmp_path / "out.h5ad"
    script = textwrap.dedent(
        """
        import resource, sys
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
        from adata.cli import app
        app(sys.argv[1:])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "concat", *paths, "-o", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    got = ad.read_h5ad(out)
    assert got.shape == (200, 3)
    assert np.array_equal(got.X[::2, 0], np.arange(100, dtype=np.float32))


def _tiny_disk(monkeypatch, free: int = 1) -> None:
    import shutil

    from adata.util import staging

    usage = shutil.disk_usage(".")
    monkeypatch.setattr(staging.shutil, "disk_usage",
                        lambda path: type(usage)(usage.total, usage.total - free, free))


def test_the_csc_scratch_is_checked_for_space_even_for_a_zarr_output(tmp_path, monkeypatch):
    """F011: a CSC matrix's buckets go in an HDF5 file beside the output, and
    a full disk part way through an HDF5 write can abort with no cleanup.
    A Zarr output is not itself measured, but its scratch is, without
    walking the inputs to size them."""
    from adata.util import staging

    genes = [f"g{i}" for i in range(10)]
    a = _csc_input(tmp_path / "a.zarr", "zarr3", "a", 20, genes, seed=0)
    b = _csc_input(tmp_path / "b.zarr", "zarr3", "b", 20, genes, seed=1)
    _tiny_disk(monkeypatch)
    monkeypatch.setattr(staging, "disk_size", lambda path: pytest.fail("walked an input"))
    out = tmp_path / "out.zarr"
    with pytest.raises(OSError, match="Not enough free space"):
        concat_on_disk([a, b], out, QUIET)
    assert not out.exists() and not list(tmp_path.glob(".*scratch*"))
    concat_on_disk([a, b], out, QUIET, space_check=False)
    assert _read(out).shape == (40, 10)


def test_an_h5ad_outputs_estimate_adds_the_csc_scratch(tmp_path, monkeypatch):
    """F011: the scratch is there while the output is written, so an HDF5
    output needs room for both: the inputs' size, and for the largest CSC
    matrix 16 bytes of row and column plus the value per stored entry."""
    from adata.util.staging import disk_size

    genes = [f"g{i}" for i in range(10)]
    a = _csc_input(tmp_path / "a.h5ad", "h5ad", "a", 20, genes, seed=0)
    b = _csc_input(tmp_path / "b.h5ad", "h5ad", "b", 20, genes, seed=1)
    seen = []
    monkeypatch.setattr(concat_mod, "check_free_space",
                        lambda out, needed, check=True: seen.append(needed() if callable(needed) else needed))
    concat_on_disk([a, b], tmp_path / "out.h5ad", QUIET)
    nnz = sum(int(_read(p).layers["counts"].nnz) for p in (a, b))
    assert seen == [disk_size(a) + disk_size(b) + nnz * (16 + 4)]


def _csc_store(path: Path, cells, genes, data, indices, indptr) -> Path:
    """A CSC X stored exactly as given, unsorted rows within a column too."""
    from adata.elements import spec
    from adata.elements import write as ew

    with open_store(path, "w") as handle:
        root = handle.root
        ew.write_dataframe_header(root, "obs", list(cells), [])
        ew.write_dataframe_header(root, "var", list(genes), [])
        ew.ensure_anndata_skeleton(root)
        ew.write_sparse(root, "X", np.asarray(data, dtype=np.float32), np.asarray(indices),
                        np.asarray(indptr), (len(cells), len(genes)), spec.CSC_MATRIX)
    return path


@pytest.mark.parametrize("unsorted", [False, True], ids=["sorted", "an-input-unsorted"])
def test_a_column_bigger_than_a_block_is_streamed_and_comes_out_sorted(tmp_path, unsorted):
    """F011: a gene stored in most cells fills a bucket on its own, which is
    streamed a block at a time rather than read whole -- its entries arrive
    input by input, each input's in row order. If an input stored that
    column out of order, the bucket is sorted after all."""
    n = 2600
    paths = []
    for tag, seed in (("a", 0), ("b", 1)):
        rng = np.random.default_rng(seed)
        rows = np.arange(n)
        if unsorted and tag == "b":
            rows = rng.permutation(rows)
        sparse_rows = np.flatnonzero(rng.random(n) < 0.1)
        data = np.concatenate((rng.integers(1, 9, n), rng.integers(1, 9, sparse_rows.size)))
        indices = np.concatenate((rows, sparse_rows))
        indptr = [0, n, n + sparse_rows.size]
        paths.append(_csc_store(tmp_path / f"{tag}.h5ad", [f"{tag}{i}" for i in range(n)],
                                ["dense", "sparse"], data, indices, indptr))
    out = tmp_path / "out.h5ad"
    concat_on_disk(paths, out, QUIET, chunk_rows=64)
    stored = _stored_csc(out, "X")
    assert stored.has_sorted_indices
    ref = ad.concat([ad.read_h5ad(p) for p in paths])
    assert np.array_equal(stored.toarray(), ref.X.toarray())


def test_a_matrix_out_of_step_with_obs_is_refused_before_writing(tmp_path):
    """The CSR path now fills one int64 array of row pointers sized by obs,
    so a matrix whose rows are not its input's obs is refused up front,
    naming the input, rather than failing half way through the write."""
    genes = ["g0", "g1"]
    good = _csc_store(tmp_path / "a.h5ad", ["a0", "a1"], genes, [1, 2], [0, 1], [0, 1, 2])
    bad = _csc_store(tmp_path / "b.h5ad", ["b0", "b1", "b2"], genes, [1], [0], [0, 1, 1])
    with h5py.File(bad, "a") as f:
        f["X"].attrs["shape"] = np.array([2, 2])
    out = tmp_path / "out.h5ad"
    with pytest.raises(ValueError, match=r"b\.h5ad.* 2 rows of it but 3 obs"):
        concat_on_disk([good, bad], out, QUIET)
    assert not out.exists()


def test_running_out_of_file_descriptors_says_how_to_raise_the_limit(tmp_path, monkeypatch):
    """F076: if the process has no descriptors left even for the inputs
    concat holds open, the error names the limit and `ulimit -n`, not just
    'Too many open files'."""
    import errno

    genes = ["g0"]
    paths = [_csc_store(tmp_path / f"{t}.h5ad", [f"{t}0"], genes, [1], [0], [0, 1]) for t in "ab"]

    def exhausted(path, mode, **kwargs):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(concat_mod, "open_store", exhausted)
    with pytest.raises(OSError, match=r"ulimit -n") as raised:
        concat_on_disk(paths, tmp_path / "out.h5ad", QUIET)
    assert "a.h5ad" in str(raised.value)

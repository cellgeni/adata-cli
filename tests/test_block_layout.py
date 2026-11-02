"""Blocks and shards: outputs keep their source's chunks and shards, are
written a whole unit at a time, and still equal anndata's own subsetting
(B6b, #97).

The cost of each of these is guarded in `test_performance.py`; here the
outputs are checked against anndata, on h5ad, Zarr v2, Zarr v3 as anndata
writes it today, and Zarr v3 in anndata 0.13's layout under zarr 3.1
(`zarr31`: inner chunks of a kilobyte inside shards of a whole array), at
blocks small enough that every boundary is crossed: a major read in parts,
a unit cut to a sliver of a split's write budget, a matrix read in tiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import backend_path
from tests.perf_counters import count_io
from tests.perf_fixtures import counts_matrix, zarr31_layout
from tests.test_subset_oracle import (
    assert_matches_anndata,
    ok,
    read_anndata,
    rich_anndata,
    write_anndata,
)

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")
zarr = pytest.importorskip("zarr")

FORMATS = ("h5ad", "zarr2", "zarr3", "zarr31")


def write_store(obj: Any, tmp_path: Path, fmt: str, stem: str = "src") -> Path:
    """`obj` as anndata writes it in `fmt`; `zarr31` then rewrites X."""
    path = write_anndata(obj, tmp_path, "zarr3" if fmt == "zarr31" else fmt, stem)
    return zarr31_layout(path) if fmt == "zarr31" else path


def _names(tmp_path: Path, stem: str, prefix: str, picked) -> Path:
    path = tmp_path / f"{stem}.txt"
    path.write_text("".join(f"{prefix}{i}\n" for i in picked))
    return path


# ---------------------------------------------------------------------------
# anndata as the oracle, at blocks smaller than every unit


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_a_subset_in_tiny_blocks_matches_anndata(tmp_path, fmt, layout):
    """At --chunk 2 a sparse block holds two rows' worth of entries, fewer
    than a column of a CSC layer, which is read in parts (F032); every
    output unit is filled over many blocks before it is written (F031)."""
    chunk = 2
    source = rich_anndata(layout, seed=3)
    src = write_store(source, tmp_path, fmt)
    obs = [0, 1, 2, 5, 9, 10, 11, 17, 23, 28, 29]
    var = [0, 3, 4, 8]
    out = backend_path(tmp_path, "h5ad" if fmt == "h5ad" else "zarr3", "out")

    ok("subset", src, "-o", out, "--obs", _names(tmp_path, "c", "c", obs),
       "--var", _names(tmp_path, "g", "v", var), "-C", chunk)
    assert_matches_anndata(read_anndata(out), source, obs, var)


# h5ad's chunks and anndata 0.13's shards: the other formats' units are
# cut by the same code, and `test_subset_oracle` splits them.
@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("fmt", ["h5ad", "zarr31"])
def test_split_in_tiny_blocks_and_a_tiny_write_budget_matches_anndata(
    tmp_path, monkeypatch, fmt, layout
):
    """Each output's units are cut to its share of the pass's write budget
    (F079); here the budget is a few hundred bytes, so units are a row or
    two, and every output still equals anndata's slice."""
    import adata.storage as storage

    monkeypatch.setattr(storage, "WRITE_BUFFER_BYTES", 512)
    source = rich_anndata(layout, seed=4)
    src = write_store(source, tmp_path, fmt)
    parts = tmp_path / "parts"

    ok("split", src, "--by", "group", "-o", parts, "--no-manifest", "-C", 2)
    labels = source.obs["group"].astype(str).to_numpy()
    for label in set(labels):
        got = read_anndata(parts / f"{label}{src.suffix}")
        assert_matches_anndata(got, source, obs=np.flatnonzero(labels == label))


def test_a_matrix_read_in_tiles_matches_anndata(tmp_path, monkeypatch):
    """F061, F081: where one chunk height across the matrix is over the read
    budget -- chunks of (n_obs, 2) -- a band is read in tiles of whole
    chunks, each output column band written by its own writer. Made to
    happen at this size by a read budget of a few bytes."""
    import h5py

    import adata.storage as storage
    from adata.core import subset

    monkeypatch.setattr(storage, "TARGET_READ_BYTES", 64)
    monkeypatch.setattr(subset, "TARGET_READ_BYTES", 64)
    source = rich_anndata("dense", seed=5)
    src = write_store(source, tmp_path, "h5ad")
    with h5py.File(src, "a") as handle:
        for key in ("X", "layers/dense"):
            values, attrs = handle[key][...], dict(handle[key].attrs)
            del handle[key]
            ds = handle.create_dataset(key, data=values, chunks=(values.shape[0], 2),
                                       compression="gzip")
            ds.attrs.update(attrs)
    obs, var = [1, 2, 3, 8, 13, 21, 22, 27], [0, 2, 3, 6, 7]
    out = tmp_path / "out.h5ad"
    ok("subset", src, "-o", out, "--obs", _names(tmp_path, "c", "c", obs),
       "--var", _names(tmp_path, "g", "v", var), "-C", 1)
    assert_matches_anndata(read_anndata(out), source, obs, var)

    parts = tmp_path / "parts"
    ok("split", src, "--by", "group", "-o", parts, "--no-manifest", "-C", 1)
    labels = source.obs["group"].astype(str).to_numpy()
    for label in set(labels):
        assert_matches_anndata(read_anndata(parts / f"{label}.h5ad"), source,
                               obs=np.flatnonzero(labels == label))

    zout = tmp_path / "copy.zarr"
    ok("convert", src, "-o", zout)
    got = read_anndata(zout)
    np.testing.assert_array_equal(got.X, source.X)
    np.testing.assert_array_equal(got.layers["dense"], source.layers["dense"])


# ---------------------------------------------------------------------------
# outputs keep their source's layout (F033)


def _sharded_counts(tmp_path: Path, layout: str, n_obs: int = 3000, n_var: int = 40,
                    stem: str = "src") -> tuple:
    matrix = counts_matrix(n_obs, n_var, layout, density=0.2)
    obj = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"n": np.arange(n_obs, dtype=np.int64),
                          "group": pd.Categorical([f"g{i % 3}" for i in range(n_obs)])},
                         index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"v{j}" for j in range(n_var)]),
    )
    return obj, write_store(obj, tmp_path, "zarr31", stem)


def _arrays(group: Any) -> dict:
    x = group["X"]
    if isinstance(x, zarr.Array):
        return {"X": x}
    return {f"X/{k}": x[k] for k in ("data", "indices")}


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("command", ["subset", "split", "concat", "recast"])
def test_outputs_keep_the_sources_inner_chunks_and_shards(tmp_path, command, layout):
    """F033: outputs of anndata 0.13's layout dropped the shard and kept the
    245-element inner chunk as their own, a file each: 4,083 files for one
    array the source holds in 3. Each output array has the source's inner
    chunk and shard now, the shard clamped to the output, in whole chunks."""
    obj, src = _sharded_counts(tmp_path, layout)
    source = _arrays(zarr.open_group(str(src), mode="r"))
    if command == "subset":
        outputs = [tmp_path / "out.zarr"]
        ok("subset", src, "-o", outputs[0], "-q", "n >= 100", "-C", 64)
    elif command == "split":
        ok("split", src, "--by", "group", "-o", tmp_path / "parts", "--no-manifest", "-C", 64)
        outputs = sorted((tmp_path / "parts").iterdir())
    elif command == "concat":
        other = write_store(obj, tmp_path, "zarr31", "other")
        outputs = [tmp_path / "out.zarr"]
        ok("concat", src, other, "-o", outputs[0], "--index-unique", "-", "-C", 64)
    else:
        outputs = [tmp_path / "out.zarr"]
        ok("recast", src, "X", "-o", outputs[0], "--dtype", "float64", "-C", 64)

    for path in outputs:
        group = zarr.open_group(str(path), mode="r")
        for name, array in _arrays(group).items():
            want = source[name]
            assert array.chunks == want.chunks, (path.name, name)
            shards = tuple(
                min(s, max(c, -(-n // c) * c)) if array.ndim > 1 or name == "X" else s
                for s, c, n in zip(want.shards, want.chunks, array.shape)
            )
            assert array.shards == shards, (path.name, name, array.shards, shards)
            files = [f for f in (path / name).rglob("*")
                     if f.is_file() and f.name not in ("zarr.json", ".zarray", ".zattrs")]
            assert len(files) <= 4, f"{name} of {path.name} is {len(files)} files"
        got = read_anndata(path)
        rows = got.obs["n"].to_numpy() % len(obj)
        want_x = obj.X[rows]
        got_x = got.X
        if sparse.issparse(want_x):
            want_x, got_x = want_x.toarray(), got_x.toarray()
        np.testing.assert_array_equal(got_x, want_x)


def test_converting_a_sharded_store_to_h5ad_lets_hdf5_choose_its_chunks(tmp_path):
    """F033: a shard's 245-element inner chunk became the HDF5 chunk, 16,327
    chunks where anndata's own h5ad has 512, and a file twice the size. HDF5
    picks the chunks now, and X takes what anndata's own takes."""
    import h5py

    obj, src = _sharded_counts(tmp_path, "csr", n_obs=20_000, n_var=100)
    out = tmp_path / "out.h5ad"
    ok("convert", src, "-o", out)
    theirs = tmp_path / "theirs.h5ad"
    ad.read_zarr(src).write_h5ad(theirs, compression="gzip")

    def stored(path: Path) -> int:
        with h5py.File(path, "r") as handle:
            for name in ("data", "indices"):
                assert handle[f"X/{name}"].chunks[0] > 4096, handle[f"X/{name}"].chunks
            return sum(handle[f"X/{n}"].id.get_storage_size() for n in ("data", "indices"))

    assert stored(out) <= 1.1 * stored(theirs), (stored(out), stored(theirs))
    np.testing.assert_array_equal(read_anndata(out).X.toarray(), obj.X.toarray())


def test_a_sharded_store_written_as_zarr_v2_takes_its_shard_as_the_chunk(tmp_path):
    """F033: Zarr v2 has no shards, so `--zarr-format 2` kept only the inner
    chunk, a 245-element file each. The shard is the chunk now."""
    obj, src = _sharded_counts(tmp_path, "csr")
    out = tmp_path / "v2.zarr"
    ok("convert", src, "-o", out, "--zarr-format", "2")
    source = zarr.open_group(str(src), mode="r")
    group = zarr.open_group(str(out), mode="r")
    for name in ("data", "indices"):
        array, shards = group[f"X/{name}"], source[f"X/{name}"].shards
        # The shard, clamped to the array as any chunk is.
        assert array.chunks == tuple(min(s, n) for s, n in zip(shards, array.shape))
    np.testing.assert_array_equal(read_anndata(out).X.toarray(), obj.X.toarray())


# ---------------------------------------------------------------------------
# the pieces


@pytest.mark.parametrize("sharded", [False, True])
def test_a_row_writer_writes_each_unit_once_whatever_the_blocks(new_store, backend, sharded):
    """Blocks of every size, none of them a unit, land where they belong, and
    each unit (chunk, or shard) is written once."""
    from adata.storage import RowWriter, create_dataset

    if sharded and backend != "zarr3":
        pytest.skip("only Zarr v3 shards")
    path, opener = new_store()
    values = np.arange(103 * 6, dtype="float32").reshape(103, 6)
    kw = {"chunks": (7, 3)}
    if sharded:
        kw["shards"] = (14, 6)
    with opener("a") as root:
        ds = create_dataset(root["uns"], "m", shape=values.shape, dtype=values.dtype, **kw)
        writer = RowWriter(ds)
        with count_io() as io:
            at = 0
            for size in (1, 5, 2, 13, 30, 4, 48):
                writer.write(values[at:at + size])
                at += size
            writer.flush()
        np.testing.assert_array_equal(ds[...], values)
    unit = 14 if sharded else 7
    # The last unit is part-filled, and counted whole.
    assert io.encoded_bytes <= io.written_bytes + unit * 6 * 4, io


def test_a_row_writer_grows_an_array_and_writes_columns(new_store):
    from adata.storage import RowWriter, create_dataset

    path, opener = new_store()
    with opener("a") as root:
        grown = create_dataset(root["uns"], "g", shape=(0,), dtype="int64", chunks=(10,),
                               **({} if "zarr" in str(path) else {"maxshape": (None,)}))
        writer = RowWriter(grown, grow=True)
        for start in range(0, 57, 4):
            writer.write(np.arange(start, min(start + 4, 57)))
            assert grown.shape[0] % 10 == 0, "grown a whole unit at a time"
        writer.flush()
        np.testing.assert_array_equal(grown[...], np.arange(57))

        wide = create_dataset(root["uns"], "w", shape=(20, 9), dtype="int32", chunks=(4, 3))
        left, right = RowWriter(wide, cols=(0, 4)), RowWriter(wide, cols=(4, 9))
        values = np.arange(180, dtype="int32").reshape(20, 9)
        for start in range(0, 20, 3):
            left.write(values[start:start + 3, :4])
            right.write(values[start:start + 3, 4:])
        left.flush(), right.flush()
        np.testing.assert_array_equal(wide[...], values)


def test_a_row_writer_does_not_hold_a_unit_over_its_budget(new_store):
    """A unit the backend chose, too large to hold, is written as rows come."""
    from adata.storage import RowWriter, create_dataset

    path, opener = new_store()
    with opener("a") as root:
        ds = create_dataset(root["uns"], "m", shape=(100, 4), dtype="float64", chunks=(50, 4))
        assert RowWriter(ds).unit == 50
        assert RowWriter(ds, budget=50 * 4 * 8 - 1).unit == 1


def test_read_blocks_round_to_whole_chunks_within_the_budget():
    from adata.storage import read_blocks

    # Contiguous: --chunk rows, as before.
    assert read_blocks((1000, 10), None, 4, rows=64, budget=10**6) == (64, None)
    # Rounded up to whole chunk heights.
    assert read_blocks((1000, 10), (50, 10), 4, rows=64, budget=10**6) == (100, None)
    # A chunk taller than --chunk is read whole, within the budget.
    assert read_blocks((100_000, 10), (5000, 10), 4, rows=64, budget=10**6) == (5000, None)
    # More than the budget allows: fewer whole chunk heights.
    assert read_blocks((100_000, 10), (50, 10), 4, rows=10_000, budget=50 * 40 * 3) == (150, None)
    # One chunk height across the matrix is over the budget: tiles of whole chunks.
    assert read_blocks((100_000, 64), (100_000, 8), 4, rows=64, budget=100_000 * 4 * 16) == (
        100_000, 16)
    # Never more rows than the array, in whole chunks.
    assert read_blocks((10, 4), (4, 4), 4, rows=1000, budget=10**6) == (12, None)


@pytest.mark.parametrize("seed", range(20))
def test_sparse_blocks_cover_every_kept_major_once_within_the_budget(seed):
    """F032's blocks: whole majors while they fit, a major over the budget in
    parts, majors no target keeps skipped."""
    from adata.core.subset import _selected, _sparse_blocks

    rng = np.random.default_rng(seed)
    n_major = int(rng.integers(1, 40))
    sizes = rng.integers(0, 12, n_major) * (rng.random(n_major) < 0.8)
    indptr = np.concatenate(([0], np.cumsum(sizes)))
    budget = int(rng.integers(1, 15))
    selections = [None if rng.random() < 0.3 else
                  np.sort(rng.choice(n_major, int(rng.integers(0, n_major + 1)), replace=False))
                  for _ in range(int(rng.integers(1, 4)))]
    if all(s is not None and s.size == 0 for s in selections):
        selections[0] = None

    got = {i: np.zeros(n_major, dtype=np.int64) for i in range(len(selections))}
    last = -1
    for first, stop, lo, hi in _sparse_blocks(indptr, budget, selections):
        assert hi - lo <= budget
        assert lo >= last, "blocks arrive in order"
        last = hi
        for i, sel in enumerate(selections):
            _, rows = _selected(sel, first, stop)
            for r in rows:
                got[i][r] += max(0, min(indptr[r + 1], hi) - max(indptr[r], lo))
    for i, sel in enumerate(selections):
        kept = np.arange(n_major) if sel is None else sel
        np.testing.assert_array_equal(got[i][kept], sizes[kept])


def test_a_shard_index_is_read_once_and_the_cache_is_bounded(tmp_path, monkeypatch):
    """F062: each read of part of a shard fetched its index; a store opened
    for reading keeps the indexes now, up to `SHARD_INDEX_CACHE_BYTES`."""
    from zarr.abc.store import SuffixByteRequest
    from zarr.storage import LocalStore

    import adata.storage as storage
    from adata.storage import open_store

    _, src = _sharded_counts(tmp_path, "csr")
    fetched = []
    original = LocalStore.get

    async def get(self, key, prototype, byte_range=None):
        if isinstance(byte_range, SuffixByteRequest):
            fetched.append(key)
        return await original(self, key, prototype, byte_range)

    monkeypatch.setattr(LocalStore, "get", get)

    def read_in_blocks() -> int:
        fetched.clear()
        with open_store(src, "r") as store:
            data = store.root["X/data"]
            for start in range(0, data.shape[0], 1000):
                data[start:start + 1000]
        return len(fetched)

    assert read_in_blocks() == 1
    monkeypatch.setattr(storage, "SHARD_INDEX_CACHE_BYTES", 0)
    assert read_in_blocks() > 10, "a cache too small for an index keeps none"

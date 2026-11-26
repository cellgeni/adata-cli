"""`recast` on every backend, with blocks small enough to cross every boundary.

`test_recast.py` checks recast on h5ad with its default blocks, where most
matrices fit in one. These run on h5ad, Zarr v2 and Zarr v3 (`new_store`),
at a `--chunk` of a few rows and a few entries, so every path splits the
matrix into many blocks, and check the result two ways: against scipy's own
conversion of the same matrix, and by reading it back with anndata.

Memory is checked in `test_performance.py`; these are the correctness half of
the same fixes (issue #96).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from adata.cli import app
from adata.commands.recast import recast_store
from adata.core import recast as core
from adata.elements import spec
from adata.elements import write as ew

ad = pytest.importorskip("anndata", reason="anndata reads every result back")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()
QUIET = Console(quiet=True)
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _matrix(n_obs: int = 23, n_var: int = 17, *, density: float = 0.3, seed: int = 0,
            integral: bool = True):
    rng = np.random.default_rng(seed)
    matrix = sparse.random(n_obs, n_var, density=density, format="csr",
                           dtype="float64", random_state=rng)
    # Counts from 1, so no stored entry is an explicit zero.
    matrix.data = np.ceil(matrix.data * 100) if integral else matrix.data
    return matrix


def _as(matrix, layout: str):
    if layout == "dense":
        return matrix.toarray()
    return matrix.tocsc() if layout == "csc" else sparse.csr_matrix(matrix)


def _write_matrix(parent: Any, name: str, value: Any) -> None:
    if sparse.issparse(value):
        ew.write_sparse(
            parent, name, value.data, value.indices, value.indptr, value.shape,
            enc_type=spec.CSC_MATRIX if value.format == "csc" else spec.CSR_MATRIX,
        )
    else:
        ew.write_dense(parent, name, value)


def _put(new_store, value: Any, at: str = "X", *, stem: str = "in") -> Path:
    """A store on this test's backend holding `value` at `at`, obs and var to match."""
    path, opener = new_store(stem)
    n_obs, n_var = value.shape
    with opener("a") as root:
        ew.write_dataframe_header(root, "obs", [f"c{i}" for i in range(n_obs)], [])
        ew.write_dataframe_header(root, "var", [f"g{i}" for i in range(n_var)], [])
        parent = root
        *groups, leaf = at.split("/")
        for name in groups:
            parent = parent[name] if name in parent else ew.write_mapping(parent, name)
        _write_matrix(parent, leaf, value)
    return path


def _output(source: Path, stem: str = "out") -> Path:
    return source.with_name(stem + source.suffix)


def _read(path: Path):
    return ad.read_h5ad(path) if path.suffix == ".h5ad" else ad.read_zarr(path)


def _at(obj: Any, path: str) -> Any:
    """`X`, `obsm/X_pca` or `uns/a/b` of an AnnData read back."""
    if path == "uns/neighbors/connectivities":
        # anndata moves the graph older versions kept in uns to obsp on reading.
        return obj.obsp["connectivities"]
    first, *rest = path.split("/")
    value = getattr(obj, first)
    for part in rest:
        value = value[part]
    return value


def _recast(source: Path, *, out: Optional[Path] = None, entries=("X",), **kwargs) -> Path:
    out = out or _output(source)
    kwargs.setdefault("chunk", 7)
    kwargs.setdefault("chunk_rows", 3)
    recast_store(source, list(entries), out, QUIET, **kwargs)
    return out


def _same(got, expected) -> None:
    """Equal values, equal dtype, and the layout scipy would give."""
    if sparse.issparse(expected):
        assert sparse.issparse(got), f"expected {expected.format}, got a dense array"
        assert got.format == expected.format
        assert got.has_sorted_indices, "indices out of order within a major"
        assert got.nnz == expected.nnz, f"nnz {got.nnz}, scipy has {expected.nnz}"
        got, expected = got.toarray(), expected.toarray()
    else:
        assert not sparse.issparse(got), f"expected a dense array, got {got.format}"
        got = np.asarray(got)
    assert got.dtype == expected.dtype, f"{got.dtype} against {expected.dtype}"
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# every conversion, in many blocks


@pytest.mark.parametrize("target", ["csr", "csc", "dense"])
@pytest.mark.parametrize("source", ["csr", "csc", "dense"])
def test_every_layout_conversion_matches_scipy_in_blocks(new_store, source, target):
    """F030, F077: CSC densify and dense-to-CSC sparsify now go through a CSR
    scratch and the streaming transpose, and the transpose steps by entries.
    Whatever route a conversion takes, in blocks of three rows and seven
    entries the result must be what scipy makes of the same matrix."""
    matrix = _matrix()
    source_path = _put(new_store, _as(matrix, source))

    out = _recast(source_path, layout=target, force=True)

    expected = _as(matrix, target)
    if target != "dense":
        expected.sort_indices()
    _same(_read(out).X, expected)


@pytest.mark.parametrize("source", ["csr", "csc"])
def test_a_transpose_splitting_and_streaming_majors_matches_scipy(new_store, source):
    """F077: pass 2 now reads blocks of entries, splitting a major across
    them, and pass 3 streams a major larger than the chunk from a bucket of
    its own instead of sorting it whole. Empty majors at both ends, a full
    row and a full column, in blocks of five entries and buckets of four."""
    from adata.storage import open_store

    dense = np.zeros((30, 12))
    dense[1:29, 4] = np.arange(1, 29)  # a full column; rows 0 and 29 empty
    dense[3, 1:11] = 2                 # a full row; columns 0 and 11 empty
    dense[7, 9] = 5
    matrix = _as(sparse.csr_matrix(dense), source)
    path = _put(new_store, matrix)
    out = _output(path)

    with open_store(path, "r") as src, open_store(out, "w") as dst:
        core.transpose_sparse_streaming(
            core.describe(src.root["X"]), dst.root, "X",
            data_dtype=np.dtype("float64"), index_dtype=np.dtype("int64"),
            chunk=5, bucket_entries=4,
        )
        group = dst.root["X"]
        got = (sparse.csc_matrix if source == "csr" else sparse.csr_matrix)(
            (group["data"][...], group["indices"][...], group["indptr"][...]),
            shape=dense.shape,
        )

    expected = _as(sparse.csr_matrix(dense), "csc" if source == "csr" else "csr")
    _same(got, expected)


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_a_cast_in_many_blocks_matches_scipy(new_store, layout):
    """F029: the round-trip check now reads a dense matrix a block of rows at a
    time. It must still see every value: one lossy value in the last block is
    enough to refuse."""
    matrix = _matrix()
    value = _as(matrix, layout)
    source_path = _put(new_store, value)

    out = _recast(source_path, dtype="float32")
    _same(_read(out).X, value.astype("float32"))

    lossy = _matrix(integral=False)
    lossy_path = _put(new_store, _as(lossy, layout), stem="lossy")
    with pytest.raises(ValueError, match="is lossy"):
        _recast(lossy_path, dtype="float32")


def test_the_cast_check_reads_one_block_of_rows_at_a_time(new_store):
    """F029: check_cast read `dataset[0:1048576]` of a dense matrix, every row
    and column at once, whatever --chunk said, then held two more copies of
    it. It reads --chunk rows now."""
    matrix = _matrix(n_obs=40, n_var=9).toarray()
    source_path = _put(new_store, matrix)
    reads = []

    from adata.storage import open_store

    with open_store(source_path, "r") as store:
        dataset = store.root["X"]

        class Watched:
            dtype, shape, ndim = dataset.dtype, dataset.shape, dataset.ndim

            def __getitem__(self, key):
                block = dataset[key]
                reads.append(np.asarray(block).shape)
                return block

        report = core.check_cast(Watched(), np.dtype("float32"), rows=4)

    assert report.lossless and report.total == matrix.size
    assert max(shape[0] for shape in reads) == 4, reads


# ---------------------------------------------------------------------------
# index ranges
#
# A matrix needing int64 offsets has more than 2^31 nonzeros, which a test
# cannot build. int16 stands in for int32: the checks compare against the
# dtype's own range, so the logic is the same at 32,767 as at 2^31 - 1.


@pytest.fixture
def int16_indices(monkeypatch):
    """Let `--indices-dtype` take int16, as a stand-in for int32."""
    monkeypatch.setattr(core, "INDEX_DTYPES", core.INDEX_DTYPES + ("int16",))


@pytest.mark.parametrize("force", [False, True], ids=["plain", "forced"])
def test_force_does_not_waive_the_index_range_check(new_store, int16_indices, force):
    """F013: `--force` skipped check_index_dtype altogether. Given only to
    accept a lossy float cast, it also let indptr be cast to a dtype that
    cannot hold nnz: offsets wrapped negative, a corrupt matrix with exit 0.
    The range check is not a judgement about values, so nothing waives it."""
    matrix = sparse.csr_matrix(np.ones((300, 200)))  # 60,000 nonzeros
    source_path = _put(new_store, matrix)
    out = _output(source_path)

    with pytest.raises(ValueError, match="cannot hold this matrix's offsets"):
        recast_store(source_path, ["X"], out, QUIET, indices_dtype="int16",
                     dtype="float32", force=force)
    assert not out.exists()


def test_a_dense_source_has_its_coordinates_checked_before_writing(new_store, int16_indices):
    """F013: dense to sparse never ran the range check, so coordinates past
    the index dtype's range wrapped. They are known from the shape, so the
    refusal comes before anything is written."""
    source_path = _put(new_store, np.ones((2, 40_000), dtype="float32"))
    out = _output(source_path)

    with pytest.raises(ValueError, match="cannot hold this matrix's coordinates"):
        recast_store(source_path, ["X"], out, QUIET, layout="csr", indices_dtype="int16")
    assert not out.exists()


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_a_dense_source_whose_nonzeros_overflow_the_offsets_is_refused(
    new_store, int16_indices, layout
):
    """F013: sparsify cast indptr with a plain astype, so once nnz passed the
    dtype's range the offsets wrapped and the command still succeeded. nnz is
    only known by reading, so this one is refused while writing; staging then
    removes the half-written output."""
    from tests.perf_counters import count_io

    from adata.storage import copy_attrs, create_dataset, open_store

    source_path = _put(new_store, np.ones((300, 200), dtype="float32"))
    # Chunks 16 rows tall, as --chunk: reads take whole chunks of the source
    # (F061), so a source held in one chunk is read in one piece.
    with open_store(source_path, "a") as store:
        root = store.root
        values, attrs = root["X"][...], dict(root["X"].attrs)
        del root["X"]
        rechunked = create_dataset(root, "X", data=values, chunks=(16, 200))
        copy_attrs(attrs, rechunked.attrs,
                   target_backend="zarr" if source_path.suffix == ".zarr" else "hdf5")
    out = _output(source_path)

    with count_io() as io, pytest.raises(ValueError, match="cannot hold this matrix's offsets"):
        recast_store(source_path, ["X"], out, QUIET, layout=layout,
                     indices_dtype="int16", chunk_rows=16)
    assert not out.exists()
    # As soon as the count passes the range, CSC's scratch included: not
    # after writing all of it.
    assert io.read_bytes < 300 * 200 * 4, f"read {io.read_bytes:,} bytes first"
    leftovers = [p.name for p in source_path.parent.iterdir() if "adata-" in p.name]
    assert leftovers == [], leftovers


@pytest.mark.parametrize("layout,refused", [("csr", False), ("csc", True)])
def test_the_coordinates_checked_are_those_of_the_output_minor_axis(
    new_store, int16_indices, layout, refused
):
    """With F013's check always running, it must check the right thing: CSR
    indices are column coordinates and CSC ones row coordinates. Checking
    the larger dimension refused int16 indices for a CSR matrix with more
    rows than int16 holds but two columns."""
    rows = np.arange(0, 40_000, 800)
    matrix = sparse.csr_matrix(
        (np.ones(rows.size), (rows, np.zeros(rows.size, dtype=int))), shape=(40_000, 2)
    )
    source_path = _put(new_store, matrix)

    if refused:
        with pytest.raises(ValueError, match="cannot hold this matrix's coordinates"):
            _recast(source_path, layout=layout, indices_dtype="int16")
        assert not _output(source_path).exists()
        return
    out = _recast(source_path, layout=layout, indices_dtype="int16")
    _same(_read(out).X, matrix)


def test_offsets_that_fit_are_written_in_the_narrow_dtype(new_store, int16_indices):
    """The other side of F013's check: a dense source whose nonzeros fit the
    narrow dtype is converted, with both arrays in it."""
    dense = np.eye(100, dtype="float32")
    source_path = _put(new_store, dense)

    out = _recast(source_path, layout="csc", indices_dtype="int16")
    got = _read(out).X
    _same(got, sparse.csc_matrix(dense))

    from adata.storage import open_store

    with open_store(out, "r") as store:
        assert store.root["X/indices"].dtype == np.dtype("int16")
        assert store.root["X/indptr"].dtype == np.dtype("int16")


# ---------------------------------------------------------------------------
# float16


@pytest.mark.parametrize("layout", ["csr", "csc"])
@pytest.mark.parametrize("source", ["csr", "csc", "dense"])
def test_float16_is_refused_for_a_sparse_output(new_store, source, layout):
    """F058: `--dtype float16` to a sparse layout said every value round-trips
    and wrote a matrix that scipy, and so anndata, cannot read: 'scipy.sparse
    does not support dtype float16'. With --inplace it replaced a readable
    source with an unreadable one. It is refused now, before anything is
    written, with what to do instead."""
    source_path = _put(new_store, _as(_matrix(), source))
    out = _output(source_path)
    args = ["recast", str(source_path), "X", "-o", str(out), "--dtype", "float16"]
    if source != layout:
        args += ["--layout", layout]

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    text = _out(result)
    assert "float16" in text and "float32" in text and "--layout dense" in text, text
    assert not out.exists()


def test_float16_is_refused_for_every_sparse_matrix_all_selects(new_store):
    """F058, through --all: one sparse layer is enough to refuse the lot."""
    path, opener = new_store()
    matrix = _matrix(n_obs=3, n_var=2)
    with opener("a") as root:
        _write_matrix(root, "X", matrix.toarray())
        _write_matrix(root["layers"], "counts", matrix)
    out = _output(path)

    result = runner.invoke(app, ["recast", str(path), "--all", "-o", str(out),
                                 "--dtype", "float16"])

    assert result.exit_code == 1 and "layers/counts" in _out(result)
    assert not out.exists()


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("dtype", [d for d in core.DTYPES if d != "float16"] + ["float16"])
def test_anndata_reads_back_every_dtype_and_layout_recast_writes(new_store, dtype, layout):
    """F058's general rule: every dtype `--dtype` accepts, in every layout it
    can be written in, gives a store anndata reads back with the values
    scipy gives. float16 only dense, the one layout scipy can hold it in."""
    matrix = _matrix(n_obs=6, n_var=5)
    matrix.data = np.minimum(matrix.data, 1) if dtype == "bool" else matrix.data
    source_path = _put(new_store, matrix)
    if dtype == "float16" and layout != "dense":
        pytest.skip("refused, see test_float16_is_refused_for_a_sparse_output")

    # Blocks do not matter here; the dtype and layout do.
    out = _recast(source_path, dtype=dtype, layout=layout, force=True,
                  chunk=4096, chunk_rows=64)

    if layout == "dense":
        expected = matrix.toarray().astype(dtype)
    else:
        expected = _as(matrix.astype(dtype), layout)
    _same(_read(out).X, expected)


# ---------------------------------------------------------------------------
# a lossy cast drops the zeros it makes


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_a_lossy_cast_from_dense_keeps_no_entries_that_became_zero(new_store, layout):
    """F121: sparsify found the nonzeros before casting, so 0.3 became an
    explicit 0 entry: 616 stored entries where scipy has 194. The values read
    the same, but the store is larger and nnz-based QC (n_genes_by_counts) is
    wrong. The block is cast first now, and its zeros dropped."""
    rng = np.random.default_rng(0)
    dense = np.where(rng.random((30, 20)) < 0.6, rng.random((30, 20)) * 3, 0.0)
    source_path = _put(new_store, dense)

    out = _recast(source_path, layout=layout, dtype="int32", force=True)

    expected = _as(sparse.csr_matrix(dense.astype("int32")), layout)
    got = _read(out).X
    assert (got.data != 0).all(), "explicit zeros were written"
    _same(got, expected)


# ---------------------------------------------------------------------------
# nested paths


@pytest.mark.parametrize(
    "at",
    ["uns/neighbors/connectivities", "uns/a/b/c", "obsm/X_pca"],
)
def test_a_matrix_at_any_depth_is_converted_not_copied(new_store, at):
    """F043: only a matrix one level below the root was converted. One at
    `uns/neighbors/connectivities`, where older anndata keeps the neighbour
    graph, was copied unconverted while the command reported 'Converting'
    and 'Wrote', and even warned about the cast it had not made."""
    from adata.storage import open_store

    matrix = _matrix(n_obs=12, n_var=12, density=0.2, integral=False)
    path = _put(new_store, matrix, at=at)
    parent = at.rpartition("/")[0]
    with open_store(path, "a") as store:
        # A sibling at the same level and an entry above it, both of which
        # must come through untouched.
        _write_matrix(store.root[parent], "sibling", matrix)
        ew.write_scalar(store.root["uns"], "note", 3)

    out = _recast(path, entries=[at], dtype="float32", force=True)

    got = _read(out)
    _same(_at(got, at), matrix.astype("float32"))
    _same(_at(got, f"{parent}/sibling"), matrix)
    assert got.uns["note"] == 3


# ---------------------------------------------------------------------------
# output layout


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_a_sparse_matrix_from_a_dense_one_has_element_sized_chunks(new_store, layout):
    """F078: sparsify gave `data` and `indices` the dense source's chunk
    height as their element chunk: a source chunked 16 rows tall gave 16-
    element chunks, 281,188 of them for 4.5M nonzeros and a file three times
    the size. A row height is not an element count."""
    dense = _as(_matrix(n_obs=400, n_var=50, density=0.3), "dense")
    path, opener = new_store()
    with opener("a") as root:
        ew.write_dataframe_header(root, "obs", [f"c{i}" for i in range(400)], [])
        ew.write_dataframe_header(root, "var", [f"g{i}" for i in range(50)], [])
        ew.write_dense(root, "X", dense, chunks=(4, 50))

    out = _recast(path, layout=layout, chunk_rows=64, chunk=4096)

    from adata.storage import open_store

    with open_store(out, "r") as store:
        for name in ("data", "indices"):
            chunks = store.root[f"X/{name}"].chunks
            assert chunks[0] >= min(4096, store.root[f"X/{name}"].shape[0]), (
                f"X/{name} chunked {chunks} for {store.root[f'X/{name}'].shape} "
                "elements: the source's 4-row chunk height became an element count"
            )
    _same(_read(out).X, _as(sparse.csr_matrix(dense), layout))


# ---------------------------------------------------------------------------
# free space


@pytest.mark.parametrize(
    "layout,source,streamed,in_memory",
    [
        # bucket entries: an int64 major, an int64 minor and a float64 value
        ("csc", "csr", 24, 0),
        # and the CSR the CSC goes through first, with its indptr
        ("dense", "csc", 24 + 16, 16),
        ("dense", "csr", 0, 0),
        ("csr", "csr", 0, 0),
    ],
)
def test_the_free_space_estimate_counts_the_scratch(
    tmp_path, layout, source, streamed, in_memory
):
    """A streaming transpose writes every entry to scratch beside the output,
    and a CSC matrix being densified goes through a CSR first, in memory or
    not. The estimate counted only the output, so a disk with room for that
    could fill part way through the scratch."""
    import adata.commands.recast as command
    from adata.storage import open_store

    matrix = _matrix(n_obs=50, n_var=40)
    path = tmp_path / "in.h5ad"
    ad.AnnData(X=_as(matrix, source)).write_h5ad(path)

    with open_store(path, "r") as store:
        plan = core.plan_conversion(store.root["X"], "X", layout=layout, force=True)
        rows = 8 * (matrix.shape[0] + 1) if layout == "dense" and source == "csc" else 0
        for flag, per_entry in ((False, streamed), (True, in_memory)):
            expected = matrix.nnz * per_entry + (rows if per_entry else 0)
            assert command._scratch(plan, flag) == expected, (flag, per_entry)


def test_dense_to_csc_checks_free_space_again_once_it_knows_its_nonzeros(
    tmp_path, monkeypatch
):
    """Dense to CSC goes through a CSR in scratch and the transpose's buckets,
    sized by nnz, which is known only once the matrix has been read. The
    up-front estimate cannot count them, so `sparsify` asks again before
    transposing, and a disk too small for them is refused there."""
    import adata.commands.recast as command

    dense = _matrix(n_obs=60, n_var=30).toarray()
    path = tmp_path / "in.h5ad"
    ad.AnnData(X=dense).write_h5ad(path)
    asked = []

    def check(output, needed, check=True):
        asked.append(needed() if callable(needed) else needed)
        if len(asked) == 2:
            raise OSError("Not enough free space")

    monkeypatch.setattr(command, "check_free_space", check)
    out = tmp_path / "out.h5ad"
    with pytest.raises(OSError, match="Not enough free space"):
        recast_store(path, ["X"], out, QUIET, layout="csc", chunk_rows=7)

    nnz = int(np.count_nonzero(dense))
    assert asked[1] == nnz * (16 + 8 + 8 + 8), asked
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir() if "adata-" in p.name] == []

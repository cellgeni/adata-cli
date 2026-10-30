"""`concat` on what a SCope or pycisTopic import leaves in obsm and varm:
dataframes in obsm (`RegulonsAUC`), and varm (`Regulons`, `topic_region`).

anndata's own `concat` is the reference: our output, read back, must equal
`anndata.concat(..., join=..., merge=...)` on the same inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from adata.core.concat import concat_on_disk

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

QUIET = Console(quiet=True, record=True)


def _store(tmp_path: Path, tag: str, n: int, genes, *, seed: int, regulons=("ATF3_(+)", "SOX2_(+)"),
           shift: float = 0.0):
    rng = np.random.default_rng(seed)
    cells = [f"{tag}{i}" for i in range(n)]
    obj = ad.AnnData(
        X=rng.poisson(1, (n, len(genes))).astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=list(genes)),
    )
    obj.obsm["RegulonsAUC"] = pd.DataFrame(rng.random((n, len(regulons))), index=cells, columns=list(regulons))
    obj.obsm["X_umap"] = rng.normal(size=(n, 2))
    membership = (np.arange(len(genes))[:, None] + np.arange(len(regulons))) % 2
    obj.varm["Regulons"] = pd.DataFrame(membership, index=list(genes), columns=list(regulons))
    obj.varm["topic_region"] = np.arange(len(genes) * 3, dtype=np.float64).reshape(len(genes), 3) + shift
    path = tmp_path / f"{tag}.h5ad"
    obj.write_h5ad(path)
    return path


def _concat(tmp_path, paths, **kw):
    out = tmp_path / f"out{len(list(tmp_path.iterdir()))}.h5ad"
    concat_on_disk(paths, out, QUIET, **kw)
    return ad.read_h5ad(out)


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_obsm_dataframes_concatenate_as_anndata_concatenates_them(tmp_path, join):
    genes = [f"g{i}" for i in range(6)]
    a = _store(tmp_path, "a", 5, genes, seed=0)
    b = _store(tmp_path, "b", 4, genes, seed=1, regulons=("ATF3_(+)", "PAX6_(-)"))
    got = _concat(tmp_path, [a, b], join=join)
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], join=join)
    frame, expected = got.obsm["RegulonsAUC"], ref.obsm["RegulonsAUC"]
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == list(got.obs_names)
    assert sorted(frame.columns) == sorted(expected.columns)
    pd.testing.assert_frame_equal(frame[expected.columns], expected, check_dtype=False)
    assert np.array_equal(got.obsm["X_umap"], ref.obsm["X_umap"])


@pytest.mark.parametrize("merge", ["same", "unique", "first", "only"])
def test_varm_follows_merge_as_anndata_does(tmp_path, merge):
    """varm was never written: concat dropped it whatever --merge said."""
    genes = [f"g{i}" for i in range(6)]
    a = _store(tmp_path, "a", 5, genes, seed=0)
    b = _store(tmp_path, "b", 4, genes[::-1], seed=1, shift=1.0)  # topic_region differs
    got = _concat(tmp_path, [a, b], merge=merge)
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], merge=merge)
    assert sorted(got.varm.keys()) == sorted(ref.varm.keys())
    if merge == "first":
        assert sorted(got.varm.keys()) == ["Regulons", "topic_region"]
    for key in ref.varm:
        mine, theirs = got.varm[key], ref.varm[key]
        if isinstance(theirs, pd.DataFrame):
            pd.testing.assert_frame_equal(mine, theirs, check_dtype=False)
        else:
            assert np.array_equal(mine, theirs)


def test_varm_is_dropped_without_merge_with_a_note(tmp_path):
    genes = [f"g{i}" for i in range(6)]
    a, b = _store(tmp_path, "a", 5, genes, seed=0), _store(tmp_path, "b", 4, genes, seed=1)
    console = Console(record=True, width=200)
    out = tmp_path / "plain.h5ad"
    concat_on_disk([a, b], out, console)
    assert "Dropping varm: pass --merge" in console.export_text()
    assert len(ad.read_h5ad(out).varm) == 0


def test_an_outer_join_drops_varm_an_input_cannot_cover(tmp_path):
    """With genes one input lacks, that input's varm has no rows for them:
    it counts as missing, so `same` drops the entry and `first` keeps the
    input that covers every gene, if any."""
    a = _store(tmp_path, "a", 5, [f"g{i}" for i in range(6)], seed=0)
    b = _store(tmp_path, "b", 4, [f"g{i}" for i in range(3, 8)], seed=1)
    got = _concat(tmp_path, [a, b], join="outer", merge="same")
    assert len(got.varm) == 0
    got = _concat(tmp_path, [a, b], join="outer", merge="first")
    assert len(got.varm) == 0, "neither input has every gene of the union"


def _varm_store(path: Path, genes, entries, *, compress: bool = False):
    obj = ad.AnnData(
        X=np.zeros((2, len(genes)), dtype=np.float32),
        obs=pd.DataFrame(index=[f"{path.stem}{i}" for i in range(2)]),
        var=pd.DataFrame(index=list(genes)),
    )
    for key, value in entries.items():
        obj.varm[key] = value
    if path.suffix == ".zarr":
        obj.write_zarr(path)
    else:
        obj.write_h5ad(path, compression="gzip" if compress else None)
    return path


@pytest.mark.parametrize("ext", [".h5ad", ".zarr"])
def test_varm_on_genes_in_another_order_is_compared_and_written_a_block_at_a_time(tmp_path, ext):
    """varm was read whole, every input's at once, to compare it: memory grew
    with the entry and the number of inputs rather than with --chunk."""
    genes = [f"g{i}" for i in range(40)]
    shuffled = list(np.random.default_rng(0).permutation(genes))
    values = np.arange(40 * 3, dtype=np.float64).reshape(40, 3)
    by_gene = dict(zip(genes, values))
    a = _varm_store(tmp_path / f"a{ext}", genes, {"topic_region": values})
    b = _varm_store(tmp_path / f"b{ext}", shuffled,
                    {"topic_region": np.stack([by_gene[g] for g in shuffled])})
    c = _varm_store(tmp_path / f"c{ext}", shuffled,
                    {"topic_region": np.stack([by_gene[g] for g in shuffled]) + 1})
    for inputs, kept in (([a, b], True), ([a, c], False)):
        out = tmp_path / f"out{len(list(tmp_path.iterdir()))}{ext}"
        concat_on_disk(inputs, out, QUIET, merge="same", chunk_rows=4)
        got = ad.read_zarr(out) if ext == ".zarr" else ad.read_h5ad(out)
        assert ("topic_region" in got.varm) is kept
        if kept:
            assert np.array_equal(got.varm["topic_region"], values)
    # The first input has no varm, so `first` takes c's, onto the first's genes.
    bare = _varm_store(tmp_path / f"bare{ext}", genes, {})
    out = tmp_path / f"first{ext}"
    concat_on_disk([bare, c], out, QUIET, merge="first", chunk_rows=4)
    got = ad.read_zarr(out) if ext == ".zarr" else ad.read_h5ad(out)
    assert np.array_equal(got.varm["topic_region"], values + 1)


def test_varm_text_arrays_are_compared_by_their_text(tmp_path):
    """Text arrays were compared by their bytes, which for strings read from
    HDF5 are pointers to Python objects: equal text compared unequal."""
    genes = [f"g{i}" for i in range(5)]
    labels = np.asarray([[f"t{i}", f"u{i}"] for i in range(5)], dtype=object)
    a = _varm_store(tmp_path / "a.h5ad", genes, {"labels": labels})
    b = _varm_store(tmp_path / "b.h5ad", genes, {"labels": labels.copy()})
    got = _concat(tmp_path, [a, b], merge="same")
    assert [list(row) for row in got.varm["labels"]] == labels.tolist()


def test_a_kept_varm_array_keeps_its_compression(tmp_path):
    """varm was written with the writer's defaults, uncompressed whatever the
    input was."""
    import h5py

    genes = [f"g{i}" for i in range(50)]
    values = np.tile(np.arange(50, dtype=np.float64)[:, None], (1, 4))
    a = _varm_store(tmp_path / "a.h5ad", genes, {"topic_region": values}, compress=True)
    b = _varm_store(tmp_path / "b.h5ad", genes, {"topic_region": values}, compress=True)
    out = tmp_path / "out.h5ad"
    concat_on_disk([a, b], out, QUIET, merge="same", chunk_rows=8)
    with h5py.File(out) as f, h5py.File(a) as src:
        assert f["varm/topic_region"].compression == src["varm/topic_region"].compression == "gzip"
    assert np.array_equal(ad.read_h5ad(out).varm["topic_region"], values)


@pytest.mark.parametrize("other", [np.int32, np.int64, np.float32, np.float64])
def test_varm_in_another_dtype_merges_as_anndata_merges_it(tmp_path, other):
    """anndata's `same` keeps an array only when the dtype matches as well as
    the values (0.13: int64 against int32, float32 or float64 is dropped), so
    the dtype stays part of what is compared."""
    genes = [f"g{i}" for i in range(6)]
    values = np.arange(12, dtype=np.int64).reshape(6, 2)
    a = _varm_store(tmp_path / "a.h5ad", genes, {"counts": values})
    b = _varm_store(tmp_path / "b.h5ad", genes, {"counts": values.astype(other)})
    got = _concat(tmp_path, [a, b], merge="same")
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], merge="same")
    assert sorted(got.varm) == sorted(ref.varm)


def test_varm_differing_by_a_fraction_is_not_the_same(tmp_path):
    genes = [f"g{i}" for i in range(6)]
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    a = _varm_store(tmp_path / "a.h5ad", genes, {"x": values})
    b = _varm_store(tmp_path / "b.h5ad", genes, {"x": values + 0.5})
    assert "x" not in _concat(tmp_path, [a, b], merge="same").varm


# ---------------------------------------------------------------------------
# F118: varp


def _varp_store(path: Path, genes, entries, *, n_cells: int = 2):
    obj = ad.AnnData(
        X=np.zeros((n_cells, len(genes)), dtype=np.float32),
        obs=pd.DataFrame(index=[f"{path.stem}{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=list(genes)),
    )
    for key, value in entries.items():
        obj.varp[key] = value
    if path.suffix == ".zarr":
        obj.write_zarr(path)
    else:
        obj.write_h5ad(path)
    return path


def _pairwise(genes, seed: int = 0, density: float = 0.4) -> "pd.DataFrame":
    """A gene x gene matrix, labelled so it can be put in any gene order."""
    rng = np.random.default_rng(seed)
    values = rng.integers(1, 9, (len(genes), len(genes))) * (rng.random((len(genes),) * 2) < density)
    return pd.DataFrame(values.astype(np.float64), index=list(genes), columns=list(genes))


def _as(layout: str, frame: "pd.DataFrame", genes=None):
    values = (frame.loc[list(genes), list(genes)] if genes is not None else frame).to_numpy()
    return {"dense": np.asarray, "csr": sparse.csr_matrix, "csc": sparse.csc_matrix}[layout](values)


def _dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)



@pytest.mark.parametrize("layout", ["dense", "csr", "csc"])
@pytest.mark.parametrize("merge", ["same", "unique", "first", "only"])
def test_varp_follows_merge_as_anndata_does(tmp_path, merge, layout):
    """F118: varp was dropped whatever --merge said, with a reason ('pairwise
    values have no meaning between cells from different inputs') that holds
    for obsp only: varp relates genes, and anndata merges it as it merges
    varm. b's genes are in another order, so its entries are compared and
    written on a's."""
    genes = [f"g{i}" for i in range(7)]
    shuffled = list(np.random.default_rng(1).permutation(genes))
    shared, own_a, own_b = _pairwise(genes, 0), _pairwise(genes, 1), _pairwise(genes, 2)
    a = _varp_store(tmp_path / "a.h5ad", genes, {"corr": _as(layout, shared), "own": _as(layout, own_a)})
    b = _varp_store(tmp_path / "b.h5ad", shuffled, {
        "corr": _as(layout, shared, shuffled), "own": _as(layout, own_b, shuffled),
        "only_b": _as(layout, own_b, shuffled),
    })
    got = _concat(tmp_path, [a, b], merge=merge, chunk_rows=2)
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], merge=merge)
    assert sorted(got.varp) == sorted(ref.varp)
    assert got.varp, "every strategy keeps something here"
    for key in ref.varp:
        assert np.array_equal(_dense(got.varp[key]), _dense(ref.varp[key])), key


@pytest.mark.parametrize("ext", [".h5ad", ".zarr"])
def test_varp_in_another_gene_order_is_written_on_the_output_genes(tmp_path, ext):
    """F118: the entry kept comes from an input whose genes are in another
    order, so a sparse one is rebuilt on the output's genes through buckets
    on disk, and a dense one a block of rows at a time."""
    genes = [f"g{i}" for i in range(30)]
    shuffled = list(np.random.default_rng(3).permutation(genes))
    frame = _pairwise(genes, 4)
    entries = {layout: _as(layout, frame, shuffled) for layout in ("dense", "csr", "csc")}
    bare = _varp_store(tmp_path / f"bare{ext}", genes, {})
    b = _varp_store(tmp_path / f"b{ext}", shuffled, entries)
    out = tmp_path / f"out{ext}"
    concat_on_disk([bare, b], out, QUIET, merge="first", chunk_rows=3)
    got = ad.read_zarr(out) if ext == ".zarr" else ad.read_h5ad(out)
    ref = ad.concat([ad.read_h5ad(bare) if ext == ".h5ad" else ad.read_zarr(bare),
                     ad.read_h5ad(b) if ext == ".h5ad" else ad.read_zarr(b)], merge="first")
    assert list(got.var_names) == genes
    for layout in ("dense", "csr", "csc"):
        mine = got.varp[layout]
        assert np.array_equal(_dense(mine), _dense(ref.varp[layout])), layout
        assert np.array_equal(_dense(mine), frame.to_numpy()), layout
        if layout != "dense":
            assert mine.format == layout and mine.has_sorted_indices


def test_sparse_varp_is_compared_by_value_as_anndata_compares_it(tmp_path):
    """F118: anndata's `same` finds a CSR and a CSC entry with the same
    values equal, and float32 equal to float64, and ignores stored zeros;
    a dense entry must match its dtype as well, as varm's does."""
    genes = [f"g{i}" for i in range(5)]
    frame = _pairwise(genes, 5)
    with_zero = sparse.csr_matrix(frame.to_numpy())
    with_zero.data[0] = 0.0
    without = frame.to_numpy().copy()
    without[np.unravel_index(np.flatnonzero(without)[0], without.shape)] = 0.0
    a = _varp_store(tmp_path / "a.h5ad", genes, {
        "formats": sparse.csr_matrix(frame.to_numpy()), "dtypes": sparse.csr_matrix(frame.to_numpy()),
        "zeros": with_zero, "dense_dtypes": frame.to_numpy(),
    })
    b = _varp_store(tmp_path / "b.h5ad", genes, {
        "formats": sparse.csc_matrix(frame.to_numpy()),
        "dtypes": sparse.csr_matrix(frame.to_numpy().astype(np.float32)),
        "zeros": sparse.csr_matrix(without), "dense_dtypes": frame.to_numpy().astype(np.float32),
    })
    got = _concat(tmp_path, [a, b], merge="same")
    ref = ad.concat([ad.read_h5ad(a), ad.read_h5ad(b)], merge="same")
    assert sorted(got.varp) == sorted(ref.varp) == ["dtypes", "formats", "zeros"]


def test_an_outer_join_drops_varp_an_input_cannot_cover(tmp_path):
    """F118, as varm: an input lacking some of the output's genes has no
    entry for them, so it counts as not having the entry."""
    a = _varp_store(tmp_path / "a.h5ad", [f"g{i}" for i in range(6)],
                    {"corr": _pairwise([f"g{i}" for i in range(6)]).to_numpy()})
    b = _varp_store(tmp_path / "b.h5ad", [f"g{i}" for i in range(3, 8)],
                    {"corr": _pairwise([f"g{i}" for i in range(3, 8)]).to_numpy()})
    assert len(_concat(tmp_path, [a, b], join="outer", merge="same").varp) == 0
    assert len(_concat(tmp_path, [a, b], join="outer", merge="first").varp) == 0


def test_varp_is_dropped_without_merge_with_a_note_and_obsp_always(tmp_path):
    """F118: without --merge varp is dropped as varm is, with a note that
    says how to keep it; obsp is dropped whatever --merge says."""
    genes = [f"g{i}" for i in range(4)]
    paths = []
    for tag in "ab":
        obj = ad.AnnData(X=np.ones((3, 4), dtype=np.float32),
                         obs=pd.DataFrame(index=[f"{tag}{i}" for i in range(3)]),
                         var=pd.DataFrame(index=genes))
        obj.varp["corr"] = np.eye(4)
        obj.obsp["distances"] = sparse.csr_matrix(np.eye(3))
        obj.write_h5ad(tmp_path / f"{tag}.h5ad")
        paths.append(tmp_path / f"{tag}.h5ad")
    console = Console(record=True, width=200)
    concat_on_disk(paths, tmp_path / "plain.h5ad", console)
    text = console.export_text()
    assert "Dropping varp: pass --merge" in text and "Dropping obsp" in text
    plain = ad.read_h5ad(tmp_path / "plain.h5ad")
    assert len(plain.varp) == 0 and len(plain.obsp) == 0
    console = Console(record=True, width=200)
    concat_on_disk(paths, tmp_path / "merged.h5ad", console, merge="same")
    merged = ad.read_h5ad(tmp_path / "merged.h5ad")
    assert list(merged.varp) == ["corr"] and len(merged.obsp) == 0
    assert "Dropping obsp" in console.export_text()

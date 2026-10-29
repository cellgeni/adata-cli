"""Tests for `adata convert` to and from loom.

Fixtures are written with h5py as loompy 3 and velocyto write looms (see
conftest). Import is checked against what each fixture was built from, and
against `anndata.read_loom` in test_external_readers.py; export by reading
the file back, here and with loompy there.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_loom

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _dense(m) -> np.ndarray:
    return m.toarray() if sparse.issparse(m) else np.asarray(m)


# ---------------------------------------------------------------------------
# import


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("spec", ["3.0.0", "2.0.1"])
def test_a_loom_imports_with_its_attributes_graphs_and_globals(tmp_path, spec, suffix):
    truth = write_loom(tmp_path / "in.loom", spec=spec)
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out)
    assert f"loom (spec {spec})" in _out(result)
    assert "X = matrix, layers: counts" in _out(result)

    got = _read(out)
    assert sparse.issparse(got.X) and got.X.format == "csr"
    assert got.X.dtype == np.float32
    assert np.array_equal(got.X.toarray(), truth.counts)
    assert got.layers["counts"].dtype == np.int32, "dtypes are kept"
    assert np.array_equal(got.layers["counts"].toarray(), truth.layers["counts"])

    assert list(got.obs_names) == truth.cell_ids
    assert list(got.var_names) == truth.gene_ids, "named by gene id"
    assert list(got.var["Gene"]) == truth.gene_names, "XML references unescaped"
    assert got.obs["Tissue"].dtype == "category"
    assert got.obs["Passed"].dtype == np.uint8, "loompy's booleans stay uint8"
    assert got.obs["Clusters"].tolist() == (np.arange(30) % 3).tolist()
    assert got.obsm["embedding"].shape == (30, 2)

    knn = got.obsp["knn"]
    assert knn.shape == (30, 30) and knn.nnz == 60 and knn.has_sorted_indices
    assert knn[0, 1] == pytest.approx(0.1)

    loom = got.uns["loom"]
    assert loom["spec"] == spec
    assert loom["obs_names"] == "CellID" and loom["var_names"] == "Accession"
    assert loom["attrs"]["title"] == "Démo"


def test_a_velocyto_loom_makes_x_spliced_and_keeps_the_main_matrix(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    out = tmp_path / "velo.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "loom (velocyto)" in _out(result)
    got = _read(out)
    assert got.X.dtype == np.uint32
    assert np.array_equal(got.X.toarray(), truth.layers["spliced"])
    assert np.array_equal(got.layers["matrix"].toarray(), truth.counts)
    for name in ("spliced", "unspliced", "ambiguous"):
        assert np.array_equal(got.layers[name].toarray(), truth.layers[name])


def test_the_main_matrix_is_kept_when_a_layer_is_also_named_matrix(tmp_path):
    """F047: with X a layer (spliced), the main matrix goes to
    layers["matrix"]; when the loom had a layer of that name too, the main
    matrix was not written at all."""
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    with h5py.File(truth.path, "a") as f:
        f["layers"].create_dataset("matrix", data=f["matrix"][...] + 100)
        main, named = f["matrix"][...], f["layers/matrix"][...]

    # A layer named matrix: written from an AnnData, whose X is the main one.
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "X = matrix, layers: " in _out(result)
    got = _read(out)
    assert np.array_equal(_dense(got.X), main.T)
    assert np.array_equal(_dense(got.layers["matrix"]), named.T)
    assert np.array_equal(_dense(got.layers["spliced"]), truth.layers["spliced"])

    # spliced asked for: the main matrix takes the next free name.
    out = tmp_path / "spliced.h5ad"
    result = _ok(truth.path, "-o", out, "--layer", "spliced")
    assert "The main matrix is written as layers['matrix_1']: a layer is already named 'matrix'." in _out(result)
    assert "X = spliced, layers: matrix_1, " in _out(result)
    got = _read(out)
    assert np.array_equal(_dense(got.layers["matrix_1"]), main.T)
    assert np.array_equal(_dense(got.layers["matrix"]), named.T)
    assert np.array_equal(_dense(got.X), truth.layers["spliced"])


def test_a_velocyto_round_trip_keeps_the_same_matrices_however_often_it_is_made(tmp_path):
    """F047: keeping the main matrix as layers["matrix_1"] made each loom ->
    h5ad -> loom cycle of a velocyto file add a copy of X; the main matrix
    of a loom written from an AnnData is that AnnData's X."""
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    first = tmp_path / "0.h5ad"
    _ok(truth.path, "-o", first)
    path = first
    for cycle in (1, 2):
        loom, path = tmp_path / f"{cycle}.loom", tmp_path / f"{cycle}.h5ad"
        _ok(tmp_path / f"{cycle - 1}.h5ad", "-o", loom)
        result = _ok(loom, "-o", path)
        assert "Warning" not in _out(result)
    ref, got = ad.read_h5ad(first), ad.read_h5ad(path)
    # anndata 0.13 lists a None key among the layers; the integration tests skip it too.
    names = lambda a: sorted(k for k in a.layers.keys() if k)  # noqa: E731
    assert names(got) == names(ref) == ["ambiguous", "matrix", "spliced", "unspliced"]
    assert np.array_equal(_dense(got.X), _dense(ref.X))
    for name in names(ref):
        assert np.array_equal(_dense(got.layers[name]), _dense(ref.layers[name])), name


def test_layer_picks_x(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    out = tmp_path / "u.h5ad"
    _ok(truth.path, "-o", out, "--layer", "unspliced")
    got = _read(out)
    assert np.array_equal(got.X.toarray(), truth.layers["unspliced"])
    assert "matrix" in got.layers

    out = tmp_path / "m.h5ad"
    _ok(truth.path, "-o", out, "--layer", "matrix")
    assert np.array_equal(_read(out).X.toarray(), truth.counts)


def test_dense_keeps_the_matrices_dense(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True, n_cells=150)
    out = tmp_path / "dense.h5ad"
    _ok(truth.path, "-o", out, "--dense")
    got = _read(out)
    assert isinstance(got.X, np.ndarray)
    assert np.array_equal(got.X, truth.layers["spliced"])
    assert np.array_equal(got.layers["matrix"], truth.counts)


def test_slabs_smaller_than_the_file_read_the_same(tmp_path, monkeypatch):
    """Cells are read a slab at a time; with tiny slabs every slab boundary
    falls inside the fixture."""
    from adata.conversion import loom

    monkeypatch.setattr(loom, "SLAB_BYTES", 1)
    truth = write_loom(tmp_path / "in.loom", n_cells=200)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    assert np.array_equal(_read(out).X.toarray(), truth.counts)


def test_var_names_can_be_the_symbols_or_any_attribute(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    out = tmp_path / "sym.h5ad"
    result = _ok(truth.path, "-o", out, "--var-names", "gene_symbols")
    assert "var names repeat an earlier one" in _out(result)
    got = _read(out)
    assert list(got.var_names) == truth.gene_names
    assert list(got.var["Accession"]) == truth.gene_ids

    out = tmp_path / "unique.h5ad"
    _ok(truth.path, "-o", out, "--var-names", "gene_symbols", "--make-unique")
    names = list(_read(out).var_names)
    assert len(set(names)) == len(names)

    out = tmp_path / "attr.h5ad"
    _ok(truth.path, "-o", out, "--obs-names-attr", "Tissue", "--var-names-attr", "Start")
    got = _read(out)
    assert got.obs_names[1] == "hippocampus"
    assert list(got.var_names) == [str(i * 100) for i in range(10)]


def test_a_loom_without_names_is_numbered_as_anndata_numbers_it(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    with h5py.File(truth.path, "a") as f:
        del f["col_attrs/CellID"], f["row_attrs/Accession"], f["row_attrs/Gene"]
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "they are numbered 0..29" in _out(result)
    got = _read(out)
    assert list(got.obs_names) == [str(i) for i in range(30)]
    assert list(got.var_names) == [str(i) for i in range(10)]


@pytest.mark.parametrize(
    "spoil,extra,message",
    [
        ("layer", [], "Layer 'bad' is not a 10 x 30 matrix"),
        ("attr", [], "Column attribute 'short' has 3 entries, but the matrix has 30 cells"),
        (None, ["--layer", "nope"], "--layer 'nope': no such layer; there are matrix (the main one), counts"),
        (None, ["--var-names-attr", "nope"], "--var-names-attr 'nope': no such attribute"),
    ],
)
def test_a_loom_that_cannot_be_read_is_refused_before_writing(tmp_path, spoil, extra, message):
    truth = write_loom(tmp_path / "in.loom")
    with h5py.File(truth.path, "a") as f:
        if spoil == "layer":
            f["layers"].create_dataset("bad", data=np.zeros((3, 3)))
        elif spoil == "attr":
            f["col_attrs"].create_dataset("short", data=np.zeros(3))
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out, *extra)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


def test_loom_flags_are_refused_for_other_formats(tmp_path):
    truth = write_loom(tmp_path / "in.loom")
    store = tmp_path / "s.h5ad"
    _ok(truth.path, "-o", store)
    result = _convert(store, "-o", tmp_path / "o.zarr", "--dense")
    assert result.exit_code == 1
    assert "does not apply" in _out(result)


# ---------------------------------------------------------------------------
# export


def _anndata_source(tmp_path: Path, layout: str):
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.6, (130, 9)).astype(np.int32)
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](counts)
    obj = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(["a", "b", "c"] * 43 + ["a"]), "score": rng.normal(size=130),
             "flag": np.arange(130) % 2 == 0},
            index=[f"c{i}" for i in range(130)],
        ),
        var=pd.DataFrame({"gene_symbols": [f"Gène{i}" for i in range(9)]}, index=[f"ENSG{i}" for i in range(9)]),
        layers={"spliced": sparse.csr_matrix(counts * 2)},
        obsm={"X_umap": rng.normal(size=(130, 2))},
        obsp={"connectivities": sparse.csr_matrix(np.eye(130, dtype=np.float32))},
        uns={"title": "tést", "n": 3},
    )
    path = tmp_path / f"{layout}.h5ad"
    obj.write_h5ad(path)
    return path, counts


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_export_writes_a_loom_as_loompy_does(tmp_path, layout, monkeypatch):
    from adata.conversion import loom

    monkeypatch.setattr(loom, "SLAB_BYTES", 1)  # a slab per 64 cells: three
    source, counts = _anndata_source(tmp_path, layout)
    out = tmp_path / "out.loom"
    result = _ok(source, "-o", out)
    assert "dense" in _out(result) and "before compression" in _out(result)
    with h5py.File(out, "r") as f:
        assert f["matrix"].shape == (9, 130)
        assert f["matrix"].chunks == (9, 64) and f["matrix"].compression == "gzip"
        assert np.array_equal(f["matrix"][...], counts.T)
        assert np.array_equal(f["layers/spliced"][...], counts.T * 2)
        assert f["attrs/LOOM_SPEC_VERSION"][()] == b"3.0.0"
        assert f["col_attrs/CellID"][0] == b"c0"
        assert f["row_attrs/Accession"][0] == b"ENSG0"
        assert f["row_attrs/Gene"][0] == b"G&#232;ne0", "non-ASCII as loompy escapes it"
        assert f["col_attrs/flag"].dtype == np.uint8
        assert f["col_attrs/cluster"][1] == b"b"
        assert f["col_attrs/X_umap"].shape == (130, 2)
        assert f["col_graphs/connectivities/a"].shape == (130,)
        assert f["attrs/title"][()] == b"t&#233;st"


def test_a_round_trip_keeps_what_a_loom_holds(tmp_path):
    truth = write_loom(tmp_path / "velo.loom", velocyto=True)
    h5ad, back, again = tmp_path / "a.h5ad", tmp_path / "back.loom", tmp_path / "b.h5ad"
    _ok(truth.path, "-o", h5ad)
    _ok(h5ad, "-o", back)
    _ok(back, "-o", again)
    first, second = _read(h5ad), _read(again)
    assert np.array_equal(first.X.toarray(), second.X.toarray())
    assert list(first.obs_names) == list(second.obs_names)
    assert list(first.var_names) == list(second.var_names)
    for name in ("spliced", "unspliced", "ambiguous"):
        assert np.array_equal(first.layers[name].toarray(), second.layers[name].toarray())
    assert list(first.var["Gene"]) == list(second.var["Gene"])
    assert first.obsp["knn"].nnz == second.obsp["knn"].nnz
    assert second.uns["loom"]["attrs"]["velocyto.__version__"] == "0.17.17"


def test_a_large_export_warns_but_writes(tmp_path, monkeypatch):
    from adata.conversion import loom

    monkeypatch.setattr(loom, "LARGE_DENSE", 10)
    source, _ = _anndata_source(tmp_path, "csr")
    result = _ok(source, "-o", tmp_path / "out.loom")
    assert "A loom holds its matrices dense" in _out(result)
    assert (tmp_path / "out.loom").exists()


def test_raw_x_exports_with_raw_var(tmp_path):
    """raw/X has raw/var's genes, which filtering var leaves behind; it was
    checked against var and refused, or written with var's names."""
    counts = np.arange(40, dtype=np.float32).reshape(8, 5)
    full = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame({"gene_symbols": list("ABCDE")}, index=[f"E{i}" for i in range(5)]),
    )
    full.raw = full
    filtered = full[:, :3].copy()
    filtered.layers["counts"] = filtered.X.copy()
    source = tmp_path / "f.h5ad"
    filtered.write_h5ad(source)

    out = tmp_path / "raw.loom"
    result = _ok(source, "-o", out, "--layer", "raw/X")
    assert "Not written: X and the layers" in _out(result)
    with h5py.File(out, "r") as f:
        assert f["matrix"].shape == (5, 8)
        assert np.array_equal(f["matrix"][...], counts.T)
        assert [v.decode() for v in f["row_attrs/Accession"][...]] == [f"E{i}" for i in range(5)]
        assert [v.decode() for v in f["row_attrs/Gene"][...]] == list("ABCDE")
        assert list(f["layers"]) == []


def test_explicit_gene_ids_are_the_accession_whatever_the_var_names(tmp_path):
    """With gene_symbols and gene_ids both columns, var_names were taken for
    the ids and the gene_ids column dropped."""
    obj = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(
            {"gene_symbols": ["S1", "S2", "S3"], "gene_ids": ["E1", "E2", "E3"]},
            index=["key1", "key2", "key3"],
        ),
    )
    source = tmp_path / "s.h5ad"
    obj.write_h5ad(source)
    out = tmp_path / "s.loom"
    _ok(source, "-o", out)
    with h5py.File(out, "r") as f:
        assert [v.decode() for v in f["row_attrs/Accession"][...]] == ["E1", "E2", "E3"]
        assert [v.decode() for v in f["row_attrs/Gene"][...]] == ["S1", "S2", "S3"]


def _text(ds) -> list:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in ds[...]]


def _legacy_h5ad(path: Path) -> Path:
    """An h5ad as anndata 0.7 wrote it: dataframes at encoding-version
    0.1.0, a categorical as its codes with a `categories` reference into
    `__categories`, -1 for a missing value."""
    with h5py.File(path, "w") as f:
        for axis, names in (("obs", ["c1", "c2", "c3", "c4"]), ("var", ["E1", "E2"])):
            frame = f.create_group(axis)
            frame.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.1.0"})
            frame.create_dataset("_index", data=np.array(names, dtype="S"))
        for axis, column, labels, codes in (
            ("obs", "cell_type", ["TypeA", "TypeB", "TypeC"], [0, 1, -1, 2]),
            ("var", "gene_symbols", ["A", "B"], [1, 0]),
        ):
            frame = f[axis]
            labels_ds = frame.require_group("__categories").create_dataset(column, data=np.array(labels, dtype="S"))
            frame.create_dataset(column, data=np.array(codes, dtype=np.int8)).attrs["categories"] = labels_ds.ref
            frame.attrs["column-order"] = np.array([column], dtype="S")
        f.create_dataset("X", data=np.arange(8, dtype=np.float32).reshape(4, 2))
    return path


def test_anndata_07_categoricals_export_as_their_labels(tmp_path):
    """F008: an anndata 0.7 categorical (codes with a `categories`
    reference) was taken for a number column, and its codes written in
    place of its labels."""
    source = _legacy_h5ad(tmp_path / "legacy.h5ad")
    ref = ad.read_h5ad(source)
    out = tmp_path / "out.loom"
    _ok(source, "-o", out)
    with h5py.File(out, "r") as f:
        assert h5py.check_string_dtype(f["col_attrs/cell_type"].dtype) is not None
        assert _text(f["col_attrs/cell_type"]) == ["" if pd.isna(v) else v for v in ref.obs["cell_type"]]
        assert _text(f["row_attrs/Gene"]) == list(ref.var["gene_symbols"])
        assert _text(f["row_attrs/Accession"]) == list(ref.var_names)


def test_a_store_without_x_writes_the_layer_named_and_leaves_x_out(tmp_path):
    """F102: with --layer, X was listed as a layer to write even when the
    store had none, and the write failed half way with HDF5's own error."""
    counts = np.arange(12, dtype=np.float32).reshape(4, 3)
    obj = ad.AnnData(
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        layers={"counts": sparse.csr_matrix(counts), "spliced": counts * 2},
    )
    source = tmp_path / "nox.h5ad"
    obj.write_h5ad(source)
    ref = ad.read_h5ad(source)
    assert ref.X is None

    out = tmp_path / "out.loom"
    result = _ok(source, "-o", out, "--layer", "counts")
    assert "from layers/counts, layers: spliced;" in _out(result)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["matrix"][...], _dense(ref.layers["counts"]).T)
        assert list(f["layers"]) == ["spliced"]
        assert np.array_equal(f["layers/spliced"][...], _dense(ref.layers["spliced"]).T)

    result = _convert(source, "-o", tmp_path / "plain.loom")
    assert result.exit_code == 1
    assert "This store has no X: name the matrix to write with --layer (there are counts, spliced)." in _out(result)
    assert not (tmp_path / "plain.loom").exists()


def test_a_layer_named_x_is_written_from_itself_and_x_beside_it(tmp_path):
    """F102 (review): a layer named X was written from the top-level X,
    and with --layer naming another matrix it was listed twice, failing
    half way with HDF5's `name already exists`."""
    obj = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        layers={"X": np.full((3, 2), 7, dtype=np.float32), "counts": np.full((3, 2), 3, dtype=np.float32)},
    )
    source = tmp_path / "x.h5ad"
    obj.write_h5ad(source)
    ref = ad.read_h5ad(source)

    plain = tmp_path / "plain.loom"
    _ok(source, "-o", plain)
    with h5py.File(plain, "r") as f:
        assert np.array_equal(f["matrix"][...], _dense(ref.X).T)
        assert np.array_equal(f["layers/X"][...], _dense(ref.layers["X"]).T)

    named = tmp_path / "counts.loom"
    result = _ok(source, "-o", named, "--layer", "counts")
    assert "X is written as layers/X_1: a layer is already named 'X'." in _out(result)
    with h5py.File(named, "r") as f:
        assert np.array_equal(f["matrix"][...], _dense(ref.layers["counts"]).T)
        assert sorted(f["layers"]) == ["X", "X_1"]
        assert np.array_equal(f["layers/X"][...], _dense(ref.layers["X"]).T)
        assert np.array_equal(f["layers/X_1"][...], _dense(ref.X).T)


def test_columns_named_like_a_looms_own_attributes_are_left_out_with_a_warning(tmp_path):
    """F103: an obs column named CellID, and var columns named Gene or
    Accession beside the columns those are written from, were dropped
    without a word, though the docs promise a warning."""
    obj = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame({"CellID": ["x", "y", "z"], "n": [1, 2, 3]}, index=["a", "b", "c"]),
        var=pd.DataFrame(
            {"gene_symbols": ["S1", "S2"], "Gene": ["T1", "T2"], "gene_ids": ["E1", "E2"], "Accession": ["Z1", "Z2"]},
            index=["g1", "g2"],
        ),
    )
    source, out = tmp_path / "clash.h5ad", tmp_path / "clash.loom"
    obj.write_h5ad(source)
    ref = ad.read_h5ad(source)
    text = _out(_ok(source, "-o", out))
    assert "Not written: obs column 'CellID' -- col_attrs/CellID is the cell names." in text
    assert "Not written: var column 'Gene' -- row_attrs/Gene is the gene names." in text
    assert "Not written: var column 'Accession' -- row_attrs/Accession is the gene ids." in text
    with h5py.File(out, "r") as f:
        assert _text(f["col_attrs/CellID"]) == list(ref.obs_names)
        assert list(f["col_attrs/n"][...]) == list(ref.obs["n"])
        assert _text(f["row_attrs/Gene"]) == list(ref.var["gene_symbols"])
        assert _text(f["row_attrs/Accession"]) == list(ref.var["gene_ids"])
        assert sorted(f["row_attrs"]) == ["Accession", "Gene"]


@pytest.mark.parametrize("extra", [[], ["--scope"]], ids=["plain", "scope-without-clusterings"])
def test_an_obs_column_and_an_obsm_entry_named_clusterid_keep_the_column(tmp_path, extra):
    """F104: ClusterID was left unclaimed for SCope to decide, even in a
    plain export, so an obsm entry of that name was written beside the obs
    column, and the second write failed with the output half written."""
    obj = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame({"ClusterID": [4, 5, 6]}, index=["a", "b", "c"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        obsm={"ClusterID": np.zeros((3, 3))},
    )
    source, out = tmp_path / "cid.h5ad", tmp_path / "cid.loom"
    obj.write_h5ad(source)
    text = _out(_ok(source, "-o", out, *extra))
    assert "Not written: obsm/ClusterID -- col_attrs/ClusterID is obs column 'ClusterID'." in text
    with h5py.File(out, "r") as f:
        assert list(f["col_attrs/ClusterID"][...]) == list(ad.read_h5ad(source).obs["ClusterID"])


def test_with_clusterings_clusterid_is_scopes_and_an_obsm_entry_of_that_name_is_left_out(tmp_path):
    """F104, with --scope: ClusterID is the first clustering's, so neither
    the obs column nor the obsm entry of that name is written."""
    obj = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame({"ClusterID": [4, 5, 6], "leiden": pd.Categorical(["x", "y", "x"])}, index=["a", "b", "c"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        obsm={"ClusterID": np.zeros((3, 3))},
    )
    source, out = tmp_path / "cid.h5ad", tmp_path / "cid.loom"
    obj.write_h5ad(source)
    text = _out(_ok(source, "-o", out, "--scope", "--clusterings", "leiden"))
    assert "obs column 'ClusterID' is written afresh from the first clustering, 'leiden'." in text
    assert "Not written: obsm/ClusterID -- col_attrs/ClusterID is SCope's first clustering." in text
    with h5py.File(out, "r") as f:
        assert list(f["col_attrs/ClusterID"][...]) == list(ad.read_h5ad(source).obs["leiden"].cat.codes)


# ---------------------------------------------------------------------------
# looms other tools write, as found on real files


def test_a_matrix_only_loom_is_read_by_its_name(tmp_path):
    """Baysor writes some looms as /matrix alone, with no attribute groups;
    these were not recognised as looms at all."""
    path = tmp_path / "ncvs.loom"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)  # genes x cells
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix", data=values)
    out = tmp_path / "out.h5ad"
    result = _ok(path, "-o", out)
    assert "loom (spec pre-2) → h5ad, 4 cells x 3 genes" in _out(result)
    got = _read(out)
    assert np.array_equal(got.X.toarray(), values.T)
    assert list(got.obs_names) == ["0", "1", "2", "3"]

    renamed = tmp_path / "ncvs.h5"
    path.rename(renamed)
    result = _convert(renamed, "-o", tmp_path / "x.h5ad")
    assert result.exit_code == 1, "without the name, a bare matrix is no evidence"


def test_an_r_loomexperiment_file_reads_its_version_and_float_graphs(tmp_path):
    """R's LoomExperiment writes LOOM_SPEC_VERSION as a one-element array
    (shown as "['2.0.1']") and graph vertices as floats."""
    truth = write_loom(tmp_path / "r.loom", spec="2.0.1")
    with h5py.File(truth.path, "a") as f:
        f.attrs["LOOM_SPEC_VERSION"] = np.array([b"2.0.1"])
        for key in ("a", "b"):
            values = f["col_graphs/knn"][key][...].astype(np.float64)
            del f["col_graphs/knn"][key]
            f["col_graphs/knn"].create_dataset(key, data=values)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "loom (spec 2.0.1)" in _out(result)
    assert _read(out).obsp["knn"].nnz == 60

    with h5py.File(truth.path, "a") as f:
        values = f["col_graphs/knn/a"][...]
        values[0] += 0.5
        f["col_graphs/knn/a"][...] = values
    result = _convert(truth.path, "-o", tmp_path / "bad.h5ad")
    assert result.exit_code == 1
    assert "not a whole number" in _out(result)


def test_baysor_names_genes_by_name(tmp_path):
    truth = write_loom(tmp_path / "b.loom")
    with h5py.File(truth.path, "a") as f:
        names = f["row_attrs/Gene"][...]
        del f["row_attrs/Gene"], f["row_attrs/Accession"]
        f["row_attrs"].create_dataset("Name", data=names)
    out = tmp_path / "out.h5ad"
    result = _ok(truth.path, "-o", out)
    assert "genes are named by 'Name'" in _out(result)
    assert list(_read(out).var_names) == truth.gene_names


def test_the_loom_page_lists_every_attribute_names_are_looked_for_in():
    """F153: the page's flag table left out some of the attributes a loom's
    cells and genes are named from, so a file named by `barcodes` or
    `gene_ids` seemed to need a flag."""
    from adata.conversion import loom

    page = (Path(__file__).parents[1] / "docs/reference/formats/loom.md").read_text()
    rows = {line.split("|")[1].strip(): line for line in page.splitlines() if line.startswith("| `--")}
    for flag, names in (
        ("`--obs-names-attr`", loom.OBS_NAMES),
        ("`--var-names-attr`", loom.GENE_IDS + loom.GENE_SYMBOLS),
    ):
        missing = [name for name in names if f"`{name}`" not in rows[flag]]
        assert not missing, f"{flag} does not list {missing}"


_CHECK_FIXED_EXPORTS = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad
from loompy.loom_validator import LoomValidator

problems = []
dense = lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m)
for source, loom, matrix in zip(*[iter(sys.argv[1:])] * 3):
    src = ad.read_h5ad(source)
    validator = LoomValidator()
    if not validator.validate(loom):
        problems.append(f"{loom}: validator: {validator.errors}")
    got = ad.io.read_loom(loom)
    ref = src.X if matrix == "X" else src.layers[matrix.split("/", 1)[1]]
    if not np.array_equal(dense(got.X), dense(ref).astype(np.float32)):
        problems.append(f"{loom}: X differs from {matrix}")
    if list(got.obs_names) != list(src.obs_names):
        problems.append(f"{loom}: obs_names differ")
    for column in src.obs.columns:
        if column == "CellID":
            continue
        want = ["" if pd.isna(v) else str(v) for v in src.obs[column]]
        if column not in got.obs or list(got.obs[column].astype(str)) != want:
            problems.append(f"{loom}: obs[{column!r}] differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_loompy_validates_and_anndata_reads_the_looms_the_audit_fixes_write(tmp_path):
    """F008, F102, F103, F104 against loompy's validator and anndata.read_loom:
    an anndata 0.7 store's labels, a store without X, and names that clash."""
    from tests.test_external_readers import _run

    legacy = _legacy_h5ad(tmp_path / "legacy.h5ad")
    nox, clash, cid = tmp_path / "nox.h5ad", tmp_path / "clash.h5ad", tmp_path / "cid.h5ad"
    counts = np.arange(12, dtype=np.float32).reshape(4, 3)
    ad.AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]), var=pd.DataFrame(index=["g0", "g1", "g2"]),
               layers={"counts": sparse.csr_matrix(counts)}).write_h5ad(nox)
    ad.AnnData(X=np.ones((3, 2), dtype=np.float32),
               obs=pd.DataFrame({"CellID": ["x", "y", "z"]}, index=["a", "b", "c"]),
               var=pd.DataFrame({"gene_symbols": ["S1", "S2"], "Gene": ["T1", "T2"]}, index=["g1", "g2"])
               ).write_h5ad(clash)
    ad.AnnData(X=np.ones((3, 2), dtype=np.float32), obs=pd.DataFrame({"ClusterID": [4, 5, 6]}, index=["a", "b", "c"]),
               var=pd.DataFrame(index=["g1", "g2"]), obsm={"ClusterID": np.zeros((3, 3))}).write_h5ad(cid)
    args = []
    for source, matrix in ((legacy, "X"), (nox, "layers/counts"), (clash, "X"), (cid, "X")):
        out = source.with_suffix(".loom")
        _ok(source, "-o", out, *(["--layer", matrix.split("/", 1)[1]] if matrix != "X" else []))
        args += [str(source), str(out), matrix]
    problems = _run(["anndata", "loompy", "scipy"], _CHECK_FIXED_EXPORTS, *args)
    assert problems == [], problems

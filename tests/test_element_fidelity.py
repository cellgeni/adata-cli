"""Elements anndata and scanpy write, through every copy path.

A copy is HDF5 to HDF5, HDF5 to Zarr or Zarr to HDF5, and Zarr v2 or v3 on
either side. Each source here is written by anndata itself -- or, for anndata
0.7's layout, is a store anndata 0.7.8 wrote (`tests/fixtures/anndata-0.7.8`)
-- and anndata, the oracle, reads every copy back.
"""

from __future__ import annotations

import re
import subprocess
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.reference_stores import legacy_store

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


def _ok(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, _out(result)
    return result


def _path(tmp: Path, backend: str, stem: str) -> Path:
    return tmp / (f"{stem}.h5ad" if backend == "h5ad" else f"{stem}.zarr")


def _format_flag(backend: str) -> list:
    return {"zarr2": ["--zarr-format", "2"], "zarr3": ["--zarr-format", "3"]}.get(backend, [])


def _write(obj, path: Path, backend: str) -> Path:
    if backend == "h5ad":
        obj.write_h5ad(path)
        return path
    previous = ad.settings.zarr_write_format
    ad.settings.zarr_write_format = 2 if backend == "zarr2" else 3
    try:
        obj.write_zarr(path)
    finally:
        ad.settings.zarr_write_format = previous
    return path


def _read(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _same_frame(got, want) -> None:
    """Equal values, column by column, whatever text dtype each backend
    reads back (anndata reads h5ad text as pandas strings, Zarr as object)."""
    assert list(got.index) == list(want.index)
    assert list(got.columns) == list(want.columns)
    for column in want.columns:
        a, b = got[column], want[column]
        if isinstance(b.dtype, pd.CategoricalDtype):
            assert isinstance(a.dtype, pd.CategoricalDtype), column
            assert list(a.cat.categories) == list(b.cat.categories), column
            assert a.cat.ordered == b.cat.ordered, column
        assert a.astype(object).tolist() == b.astype(object).tolist(), column


# ---------------------------------------------------------------------------
# scanpy's rank_genes_groups: record arrays


def _marker_genes(n_obs: int = 4, n_var: int = 3):
    obj = ad.AnnData(np.arange(n_obs * n_var, dtype=np.float32).reshape(n_obs, n_var))
    obj.obs_names = [f"c{i}" for i in range(n_obs)]
    obj.uns["rank_genes_groups"] = {
        "params": {"groupby": "leiden", "method": "wilcoxon"},
        # As scanpy builds them: one field per group, text of any width.
        "names": np.rec.fromarrays(
            [np.array(["g1", "g22", "gene_333"]), np.array(["ß", "g2", ""])],
            names=["0", "1"],
        ),
        "scores": np.rec.fromarrays(
            [np.array([2.5, 1.0, -0.5], dtype="float32"), np.array([3.0, 0.5, 0.0], dtype="float32")],
            names=["0", "1"],
        ),
        "pvals": np.rec.fromarrays(
            [np.array([0.01, 0.2, 0.9]), np.array([0.001, 0.5, 1.0])], names=["0", "1"],
        ),
    }
    return obj


def _same_markers(got, want) -> None:
    for key in ("names", "scores", "pvals"):
        a, b = got.uns["rank_genes_groups"][key], want.uns["rank_genes_groups"][key]
        assert a.dtype.names == b.dtype.names, key
        for field in b.dtype.names:
            assert a[field].tolist() == b[field].tolist(), (key, field)
            if b[field].dtype.kind == "f":
                assert a[field].dtype == b[field].dtype, (key, field)


@pytest.mark.parametrize("target", BACKENDS)
@pytest.mark.parametrize("source", BACKENDS)
def test_a_rank_genes_groups_result_converts_to_every_backend(tmp_path, source, target):
    """F017: a record array crossed no backend. To Zarr, its h5py text fields
    (an `object` dtype) failed as 'ambiguous'; to HDF5, Zarr's `<U` fields
    failed with 'No conversion path for dtype'."""
    truth = _marker_genes()
    src = _write(truth, _path(tmp_path, source, "src"), source)
    out = _path(tmp_path, target, "out")
    _ok("convert", src, "-o", out, *_format_flag(target))
    _same_markers(_read(out), truth)


@pytest.mark.parametrize("target", BACKENDS)
@pytest.mark.parametrize("source", BACKENDS)
def test_a_rank_genes_groups_result_survives_a_subset_to_every_backend(tmp_path, source, target):
    """F017: subset copies uns whole, through the same copy as convert."""
    truth = _marker_genes()
    src = _write(truth, _path(tmp_path, source, "src"), source)
    keep = tmp_path / "keep.txt"
    keep.write_text("c0\nc2\n")
    out = _path(tmp_path, target, "out")
    _ok("subset", src, "--obs", keep, "-o", out)
    got = _read(out)
    assert list(got.obs_names) == ["c0", "c2"]
    _same_markers(got, truth)


@pytest.mark.parametrize("compression", ["gzip", "lzf", "none"])
def test_a_rank_genes_groups_result_copies_between_h5ad_files_with_compression(tmp_path, compression):
    """F021: copying it dataset by dataset (as --compression does) gave h5py
    the record's fill value, a value for each text field, and h5py crashed
    the process creating the dataset, leaving a staged file behind. Run in
    a process of its own, so a crash fails this test rather than the run."""
    truth = _marker_genes()
    src = _write(truth, tmp_path / "src.h5ad", "h5ad")
    out = tmp_path / "out.h5ad"
    result = subprocess.run(
        [sys.executable, "-c", "from adata.cli import main; main()",
         "convert", str(src), "-o", str(out), "--compression", compression],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.glob("*adata-*")), "nothing staged is left behind"
    _same_markers(_read(out), truth)
    if compression != "none":
        with h5py.File(out) as f:
            assert f["uns/rank_genes_groups/scores"].compression == compression


# ---------------------------------------------------------------------------
# Stores anndata 0.7.8 wrote


@pytest.mark.parametrize("target", BACKENDS)
@pytest.mark.parametrize("fmt", ["h5ad", "zarr"])
@pytest.mark.parametrize("command", ["convert", "subset"])
def test_a_store_anndata_0_7_wrote_copies_as_anndata_reads_it(tmp_path, fmt, target, command):
    """F006, F072, F087: anndata 0.7's layout was only ever tested on stores
    built by hand. On the real thing, an h5ad copy nulled the references of
    raw/var's categoricals (anndata could not open it), every copy made the
    ordered categorical unordered, and its rank_genes_groups crossed no
    backend."""
    src = legacy_store(fmt, tmp_path / "legacy")
    truth = _read(src)
    assert truth.obs["cell_type"].cat.ordered, "the fixture is what 0.7.8 wrote"
    out = _path(tmp_path, target, "out")
    if command == "convert":
        _ok("convert", src, "-o", out, *_format_flag(target))
    else:
        keep = tmp_path / "keep.txt"
        keep.write_text("\n".join(truth.obs_names))
        _ok("subset", src, "--obs", keep, "-o", out)
    got = _read(out)
    _same_frame(got.obs, truth.obs)
    _same_frame(got.var, truth.var)
    _same_frame(got.raw.var, truth.raw.var)
    assert (got.X != truth.X).nnz == 0
    assert (got.raw.X != truth.raw.X).nnz == 0
    np.testing.assert_array_equal(got.obsm["X_pca"], truth.obsm["X_pca"])
    for key in ("names", "scores"):
        a, b = got.uns["rank_genes_groups"][key], truth.uns["rank_genes_groups"][key]
        for field in b.dtype.names:
            assert a[field].tolist() == b[field].tolist(), (key, field)


@pytest.mark.parametrize("fmt", ["h5ad", "zarr"])
def test_a_store_anndata_0_7_wrote_opens_without_a_warning(tmp_path, fmt):
    """F164: 0.7 never stamped the root, and every command on one of its
    stores said the store's attributes were 'missing or invalid'."""
    from adata.storage import open_store

    src = legacy_store(fmt, tmp_path / "legacy")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with open_store(src, "r") as store:
            assert "obs" in store.root
    result = _ok("view", src)
    assert "missing or invalid" not in _out(result)


def test_a_legacy_frame_nested_in_raw_copies_between_h5ad_files(tmp_path):
    """F006: h5py's copy of `raw` nulled the reference from each codes
    dataset of raw/var to its labels; anndata failed with 'Invalid HDF5
    object reference'."""
    src = legacy_store("h5ad", tmp_path / "legacy")
    out = tmp_path / "out.h5ad"
    _ok("convert", src, "-o", out)
    with h5py.File(out) as f:
        ref = f["raw/var/feature_types"].attrs["categories"]
        assert ref, "a live reference"
        assert f[ref].name == "/raw/var/__categories/feature_types"
    _same_frame(_read(out).raw.var, _read(src).raw.var)


def test_legacy_frames_in_obsm_and_uns_copy_between_h5ad_files(tmp_path):
    """F006: the same for 0.7 dataframes below obsm and uns, which h5py
    copied whole, references and all."""
    src = legacy_store("h5ad", tmp_path / "legacy")
    with h5py.File(src, "a") as f:
        f.copy(f["obs"], f["obsm"], "frame", expand_refs=True)
        f.copy(f["var"], f["uns"], "frame", expand_refs=True)
    truth = _read(src)
    assert truth.obsm["frame"]["cell_type"].cat.ordered
    assert list(truth.uns["frame"]["feature_types"]) == list(truth.var["feature_types"])
    out = tmp_path / "out.h5ad"
    _ok("convert", src, "-o", out)
    got = _read(out)
    _same_frame(got.obsm["frame"], truth.obsm["frame"])
    _same_frame(got.uns["frame"], truth.uns["frame"])


# ---------------------------------------------------------------------------
# h5py's other object dtypes


def _with_uns(path: Path, build) -> Path:
    obj = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    obj.write_h5ad(path)
    with h5py.File(path, "a") as f:
        build(f)
    return path


def test_variable_length_numbers_are_refused_by_zarr_not_turned_into_text(tmp_path):
    """F126: any `object` dtype was taken for text, so ragged int lists
    came out of a Zarr conversion as the strings '[1 2 3]' and '[4]'."""
    def build(f):
        ds = f["uns"].create_dataset("ragged", shape=(2,), dtype=h5py.vlen_dtype(np.int32))
        ds[0], ds[1] = [1, 2, 3], [4]

    src = _with_uns(tmp_path / "src.h5ad", build)
    out = tmp_path / "out.zarr"
    result = runner.invoke(app, ["convert", str(src), "-o", str(out)])
    assert result.exit_code == 1
    assert "/uns/ragged holds variable-length sequences of int32" in _out(result)
    assert not out.exists()


def test_variable_length_numbers_copy_between_h5ad_files_with_compression(tmp_path):
    """F126: the copy dataset by dataset tried to write them as text and
    failed with 'Can't implicitly convert non-string objects to strings'."""
    def build(f):
        ds = f["uns"].create_dataset("ragged", shape=(2,), dtype=h5py.vlen_dtype(np.int32))
        ds[0], ds[1] = [1, 2, 3], [4]

    src = _with_uns(tmp_path / "src.h5ad", build)
    out = tmp_path / "out.h5ad"
    _ok("convert", src, "-o", out, "--compression", "gzip")
    with h5py.File(out) as f:
        assert h5py.check_vlen_dtype(f["uns/ragged"].dtype) == np.int32
        assert [list(v) for v in f["uns/ragged"][...]] == [[1, 2, 3], [4]]
    assert [list(v) for v in _read(out).uns["ragged"]] == [[1, 2, 3], [4]]


def test_a_dataset_of_references_is_refused_by_zarr(tmp_path):
    """F126: object references took the text path as well."""
    def build(f):
        f["uns"].create_dataset("pointers", data=np.array([f["X"].ref], dtype=h5py.ref_dtype))

    src = _with_uns(tmp_path / "src.h5ad", build)
    result = runner.invoke(app, ["convert", str(src), "-o", str(tmp_path / "out.zarr")])
    assert result.exit_code == 1
    assert "/uns/pointers holds HDF5 object references" in _out(result)


def test_a_dataset_of_references_copies_between_h5ad_files_with_compression(tmp_path):
    """F126: copied dataset by dataset, its references were stringified."""
    def build(f):
        f["uns"].create_dataset("pointers", data=np.array([f["X"].ref], dtype=h5py.ref_dtype))

    src = _with_uns(tmp_path / "src.h5ad", build)
    out = tmp_path / "out.h5ad"
    _ok("convert", src, "-o", out, "--compression", "gzip")
    with h5py.File(out) as f:
        target = f[f["uns/pointers"][0]]
        np.testing.assert_array_equal(target[...], np.ones((2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Attributes


def test_unusual_hdf5_attributes_convert_to_zarr(tmp_path):
    """F130: an attribute holding h5py.Empty, or a 2-D array of bytes, failed
    the conversion with a JSON error that named no element."""
    def build(f):
        attrs = f["uns"].create_dataset("v", data=np.arange(3)).attrs
        attrs["encoding-type"], attrs["encoding-version"] = "array", "0.2.0"
        attrs["nothing"] = h5py.Empty("f4")
        attrs["grid"] = np.array([[b"a", b"b"], [b"c", b"d"]])

    src = _with_uns(tmp_path / "src.h5ad", build)
    out = tmp_path / "out.zarr"
    _ok("convert", src, "-o", out)
    attrs = dict(zarr.open_group(out, mode="r")["uns/v"].attrs)
    assert attrs["nothing"] is None
    assert attrs["grid"] == [["a", "b"], ["c", "d"]]
    np.testing.assert_array_equal(_read(out).uns["v"], np.arange(3))


def test_a_null_attribute_converts_back_to_hdf5(tmp_path):
    """F130: the way back, Zarr's JSON null became an attribute h5py
    could not write."""
    src = _write(ad.AnnData(np.ones((2, 2), dtype=np.float32), uns={"v": np.arange(3)}),
                 tmp_path / "src.zarr", "zarr3")
    group = zarr.open_group(src, mode="a")
    group["uns/v"].attrs["nothing"] = None
    zarr.consolidate_metadata(src)
    out = tmp_path / "out.h5ad"
    _ok("convert", src, "-o", out)
    with h5py.File(out) as f:
        assert isinstance(f["uns/v"].attrs["nothing"], h5py.Empty)
    np.testing.assert_array_equal(_read(out).uns["v"], np.arange(3))


@pytest.mark.parametrize("target", ["out.zarr", "out.h5ad"])
def test_a_reference_attribute_is_refused_naming_its_element(tmp_path, target):
    """F130: a reference attribute failed a Zarr conversion with 'Object of
    type Reference is not JSON serializable', naming nothing; between HDF5
    files it was written as a pointer into the wrong file."""
    def build(f):
        f["uns"].create_dataset("v", data=np.arange(3)).attrs["link"] = f["X"].ref

    src = _with_uns(tmp_path / "src.h5ad", build)
    extra = ["--compression", "gzip"] if target.endswith("h5ad") else []
    result = runner.invoke(app, ["convert", str(src), "-o", str(tmp_path / target), *extra])
    assert result.exit_code == 1
    assert "Attribute 'link' of /uns/v is an HDF5 object reference" in _out(result)


# ---------------------------------------------------------------------------
# Zarr v2 codecs


def _v2_store(path: Path) -> Path:
    import numcodecs

    obj = ad.AnnData(np.arange(20, dtype=np.float32).reshape(5, 4), obs={"n": np.arange(5)})
    obj.obs_names = [f"c{i}" for i in range(5)]
    obj.obsm["X_pca"] = np.arange(10, dtype=np.float64).reshape(5, 2)
    _write(obj, path, "zarr2")
    group = zarr.open_group(path, mode="a")
    for key, codec in (("X", numcodecs.Zstd(level=5)), ("obsm/X_pca", None)):
        values, attrs = group[key][...], dict(group[key].attrs)
        del group[key]
        group.create_array(key, data=values, compressors=codec).attrs.update(attrs)
    zarr.consolidate_metadata(path)
    return path


@pytest.mark.parametrize("command", ["convert", "subset"])
def test_a_zarr_v2_store_keeps_each_arrays_compression(tmp_path, command):
    """F134, F135: an uncompressed v2 array came out Blosc-compressed, and
    the codec of a compressed one was read through zarr's deprecated
    `compressor`, behind an `except` that turned any error -- a deprecation
    made an error, as some pipelines do -- into Blosc as well."""
    src = _v2_store(tmp_path / "src.zarr")
    out = tmp_path / "out.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        if command == "convert":
            _ok("convert", src, "-o", out)
        else:
            _ok("subset", src, "-q", "n < 3", "-o", out)
    group = zarr.open_group(out, mode="r")
    assert group.metadata.zarr_format == 2
    assert [c.get_config() for c in group["X"].compressors] == [{"id": "zstd", "level": 5, "checksum": False}]
    assert group["obsm/X_pca"].compressors == ()
    got, truth = _read(out), _read(src)
    rows = slice(None) if command == "convert" else slice(0, 3)
    np.testing.assert_array_equal(got.X, truth.X[rows])
    np.testing.assert_array_equal(got.obsm["X_pca"], truth.obsm["X_pca"][rows])


# ---------------------------------------------------------------------------
# Empty dataspaces


def _blank(f):
    f["uns"].create_dataset("blank", data=h5py.Empty("f4"))


def test_an_untagged_empty_dataspace_is_not_made_a_null_in_zarr(tmp_path):
    """F136: an empty dataspace another HDF5 writer left, with no encoding,
    was tagged `null` on its way to Zarr, and anndata 0.8 to 0.11 refuse a
    `null` whole. Zarr has nothing to hold it as, so it is left out."""
    src = _with_uns(tmp_path / "src.h5ad", _blank)
    out = tmp_path / "out.zarr"
    with pytest.warns(UserWarning, match="Not copied: /uns/blank"):
        _ok("convert", src, "-o", out, "--zarr-format", "2")
    assert "blank" not in zarr.open_group(out, mode="r")["uns"]
    assert "blank" not in _read(out).uns


def test_an_untagged_empty_dataspace_stays_untagged_between_h5ad_files(tmp_path):
    """F136: copied dataset by dataset, it was given `encoding-type: null`."""
    src = _with_uns(tmp_path / "src.h5ad", _blank)
    out = tmp_path / "out.h5ad"
    _ok("convert", src, "-o", out, "--compression", "gzip")
    with h5py.File(out) as f:
        assert f["uns/blank"].shape is None
        assert "encoding-type" not in f["uns/blank"].attrs
    _read(out)


def test_a_null_anndata_wrote_stays_a_null_in_every_backend(tmp_path):
    """anndata's own None: a tagged empty dataspace, still copied as a null."""
    pytest.importorskip("anndata", minversion="0.12")
    src = _write(ad.AnnData(np.ones((2, 2), dtype=np.float32), uns={"none": None}),
                 tmp_path / "src.h5ad", "h5ad")
    for out in (tmp_path / "out.zarr", tmp_path / "out.h5ad"):
        _ok("convert", src, "-o", out, *(["--compression", "gzip"] if out.suffix == ".h5ad" else []))
        assert _read(out).uns["none"] is None


# ---------------------------------------------------------------------------
# Fill values


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_recast_to_bool_works_on_a_float_h5ad_matrix(tmp_path, layout):
    """F071: the float source's fill value, 0.0, was passed on to the new
    bool dataset, and h5py has no conversion for it: 'no appropriate
    function for conversion path'."""
    values = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    matrix = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix}.get(layout, np.asarray)(values)
    src = _write(ad.AnnData(matrix), tmp_path / "src.h5ad", "h5ad")
    out = tmp_path / "out.h5ad"
    _ok("recast", src, "X", "--dtype", "bool", "-o", out)
    got = _read(out).X
    got = got.toarray() if sparse.issparse(got) else got
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, values.astype(bool))

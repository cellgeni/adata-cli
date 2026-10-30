"""`concat` against `anndata.concat`, on inputs whose encodings the test sets.

The inputs are written with h5py and zarr through this package's element
writers rather than by anndata, so that each test controls how a column is
stored: plain strings, a nullable column, a legacy 0.7 categorical. anndata
reads every input and concatenates them in memory; that is the reference our
output, read back by anndata, must equal.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import h5py
import numpy as np
import pytest
from rich.console import Console
from scipy import sparse
from typer.testing import CliRunner

from adata.cli import app
from adata.core.concat import concat_on_disk
from adata.elements import spec
from adata.elements import write as ew
from adata.storage import open_store

from tests.conftest import backend_path, backend_zarr_format

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

QUIET = Console(quiet=True)
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

Writer = Callable[[Any, str], Any]


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


# ---------------------------------------------------------------------------
# building inputs


def _store(
    path: Path,
    cells: Sequence[str],
    genes: Sequence[str],
    *,
    X: Any = None,
    obs: Optional[Dict[str, Writer]] = None,
    var: Optional[Dict[str, Writer]] = None,
    obsm: Optional[Dict[str, Writer]] = None,
    layers: Optional[Dict[str, Writer]] = None,
    zarr_format: Optional[int] = None,
) -> Path:
    """A store with these names, whose columns, layers and obsm entries are
    each written by the function given for them; X too, if given one."""
    with open_store(path, "w", zarr_format=zarr_format) as handle:
        root = handle.root
        for axis, names, columns in (("obs", cells, obs), ("var", genes, var)):
            frame = ew.write_dataframe_header(root, axis, list(names), list(columns or {}))
            for name, write in (columns or {}).items():
                write(frame, name)
        ew.ensure_anndata_skeleton(root)
        if X is None:
            X = np.arange(1, len(cells) * len(genes) + 1, dtype=np.float32).reshape(len(cells), len(genes))
        X(root, "X") if callable(X) else _matrix(root, "X", X)
        for group, entries in (("obsm", obsm), ("layers", layers)):
            for name, write in (entries or {}).items():
                write(root[group], name)
    return path


def _matrix(parent: Any, name: str, value: Any) -> None:
    if sparse.issparse(value):
        enc = spec.CSC_MATRIX if value.format == "csc" else spec.CSR_MATRIX
        ew.write_sparse(parent, name, value.data, value.indices, value.indptr, value.shape, enc)
    else:
        ew.write_dense(parent, name, np.asarray(value))


def _on(backend: str, tmp_path: Path, stem: str, cells, genes, **kw) -> Path:
    return _store(backend_path(tmp_path, backend, stem), cells, genes,
                  zarr_format=backend_zarr_format(backend), **kw)


def strings(*values: str) -> Writer:
    return lambda g, n: ew.write_string_array(g, n, list(values))


def categorical(categories: Sequence[Any], codes: Sequence[int], ordered: bool = False) -> Writer:
    return lambda g, n: ew.write_categorical(g, n, list(codes), list(categories), ordered=ordered)


def masked(enc: str, values: Sequence[Any], mask: Sequence[bool]) -> Writer:
    return lambda g, n: ew.write_masked(g, n, np.asarray(values) if enc != spec.NULLABLE_STRING_ARRAY
                                        else list(values), list(mask), enc)


def dense(*values: Any, dtype: Any = None) -> Writer:
    return lambda g, n: ew.write_dense(g, n, np.asarray(values, dtype=dtype))


def matrix(value: Any) -> Writer:
    return lambda g, n: _matrix(g, n, value)


def chunked(value: Any, chunks: tuple) -> Writer:
    """A dense array stored in `chunks` and compressed with gzip, which no
    backend's defaults give: a copy that keeps them kept them on purpose."""
    def write(parent: Any, name: str) -> None:
        value_ = np.asarray(value)
        if isinstance(parent, h5py.Group):
            out = parent.create_dataset(name, data=value_, chunks=chunks,
                                        compression="gzip", compression_opts=6)
        elif parent.metadata.zarr_format == 3:
            from zarr.codecs import GzipCodec

            out = parent.create_array(name, data=value_, chunks=chunks,
                                      compressors=[GzipCodec(level=6)])
        else:
            from numcodecs import GZip

            out = parent.create_array(name, data=value_, chunks=chunks, compressors=GZip(level=6))
        spec.set_encoding(out, spec.ARRAY)
    return write


# ---------------------------------------------------------------------------
# reading results


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _reference(paths: Sequence[Path], **kw):
    return ad.concat([_read(p) for p in paths], **kw)


def _concat(paths: Sequence[Path], out: Path, **kw):
    concat_on_disk(list(paths), out, QUIET, **kw)
    return _read(out)


def _cells(values: Any) -> list:
    """Values as plain Python, every spelling of missing as None."""
    out = []
    for v in pd.Series(values).astype(object).tolist():
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(v.item() if isinstance(v, np.generic) else v)
    return out


def _encoding(path: Path, element: str) -> Optional[str]:
    with open_store(path, "r") as handle:
        return spec.encoding_type(handle.root[element])


def _no_output(out: Path) -> None:
    assert not out.exists()
    assert not list(out.parent.glob(f"{out.stem}.adata-*")), "staging left something behind"


# ---------------------------------------------------------------------------
# F002: repeated var names


@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize(
    "genes_b",
    [["G1", "DUP", "G3", "DUP"], ["DUP", "G1", "G3", "DUP"], ["G1", "G3"]],
    ids=["same-order", "other-order", "unique-in-b"],
)
def test_repeated_var_names_are_refused_before_the_output_exists(tmp_path, join, layout, genes_b):
    """F002: an input whose var names repeat (gene symbols never made unique)
    sent two source columns to one target column. concat exited 0 having
    zeroed, summed or overwritten them -- CSR wrote duplicate entries, dense
    kept only the last column, an outer join lost a var -- where
    anndata.concat refuses the same inputs."""
    genes_a = ["G1", "DUP", "G3", "DUP"]

    def X(n_genes):
        values = np.arange(1, 2 * n_genes + 1, dtype=np.float32).reshape(2, n_genes)
        return {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](values)

    a = _store(tmp_path / "a.h5ad", ["a0", "a1"], genes_a, X=X(len(genes_a)))
    b = _store(tmp_path / "b.h5ad", ["b0", "b1"], genes_b, X=X(len(genes_b)))
    # anndata refuses them in its own words, or, at the dependency floors,
    # in pandas' as it reindexes.
    with pytest.raises(Exception, match="uniquely valued|duplicate labels"):
        _reference([a, b], join=join)

    out = tmp_path / "out.h5ad"
    with pytest.raises(ValueError, match="DUP") as raised:
        concat_on_disk([a, b], out, QUIET, join=join)
    assert "a.h5ad" in str(raised.value)
    _no_output(out)


def test_repeated_var_names_in_a_later_input_are_refused_too(tmp_path):
    """F002: duplicates only in the second input still mapped two of its
    columns onto the first input's single G1."""
    a = _store(tmp_path / "a.h5ad", ["a0"], ["G1", "G2"])
    b = _store(tmp_path / "b.h5ad", ["b0"], ["G1", "G1"])
    out = tmp_path / "out.h5ad"
    with pytest.raises(ValueError, match="G1") as raised:
        concat_on_disk([a, b], out, QUIET, join="outer")
    assert "b.h5ad" in str(raised.value)
    _no_output(out)


def test_the_cli_names_repeated_var_names_and_how_to_fix_them(tmp_path):
    """F002, from the command line: exit 1 with an `Error:` naming the
    repeated names and pointing at making them unique."""
    a = _store(tmp_path / "a.h5ad", ["a0"], ["G1", "DUP", "DUP"])
    b = _store(tmp_path / "b.h5ad", ["b0"], ["G1", "DUP", "DUP"])
    out = tmp_path / "out.h5ad"
    result = runner.invoke(app, ["concat", str(a), str(b), "-o", str(out)])
    assert result.exit_code == 1
    text = _out(result)
    assert "Error:" in text and "DUP" in text and "unique" in text
    _no_output(out)


# ---------------------------------------------------------------------------
# F010: anndata 0.7 categoricals


def _legacy_store(path: Path, cells, genes, columns: Dict[str, tuple], var_columns=None) -> Path:
    """An h5ad as anndata 0.7 wrote it: each categorical is a dataset of codes
    whose `categories` attribute references a dataset in `__categories`.
    Every frame also gets a plain column and at least two rows: anndata
    cannot read a 0.7 frame with no columns, or with one row."""
    def frame(f, name, index, cols):
        group = f.create_group(name)
        group.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.1.0"})
        group.attrs["column-order"] = np.array(["n", *cols], dtype=h5py.string_dtype())
        group.create_dataset("_index", data=np.array(index, dtype="S"))
        group.create_dataset("n", data=np.arange(len(index)))
        holder = group.create_group("__categories")
        for key, (labels, codes) in cols.items():
            cats = holder.create_dataset(key, data=np.array(labels, dtype="S"))
            codes_ds = group.create_dataset(key, data=np.asarray(codes, dtype=np.int8))
            codes_ds.attrs["categories"] = cats.ref

    with h5py.File(path, "w") as f:
        frame(f, "obs", cells, columns)
        frame(f, "var", genes, var_columns or {})
        f.create_dataset("X", data=np.ones((len(cells), len(genes)), dtype=np.float32))
    return path


def test_anndata_07_categoricals_concatenate_by_label_not_by_code(tmp_path):
    """F010: a 0.7 categorical (codes plus a reference to its categories) was
    taken for a plain integer column, so the output held int8 codes, and code
    0 meant T for one input's cells and NK for the other's."""
    a = _legacy_store(tmp_path / "a.h5ad", ["a0", "a1", "a2"], ["g1", "g2"],
                      {"cell_type": (["T", "B"], [0, 1, 0])})
    b = _legacy_store(tmp_path / "b.h5ad", ["b0", "b1"], ["g1", "g2"],
                      {"cell_type": (["NK", "T"], [0, 1])})
    got = _concat([a, b], tmp_path / "out.h5ad")
    ref = _reference([a, b])
    assert isinstance(got.obs["cell_type"].dtype, pd.CategoricalDtype)
    assert _cells(got.obs["cell_type"]) == _cells(ref.obs["cell_type"]) == ["T", "B", "T", "NK", "T"]
    assert sorted(got.obs["cell_type"].cat.categories) == ["B", "NK", "T"]


def test_anndata_07_categoricals_mix_with_current_ones(tmp_path):
    """F010: a 0.7 categorical next to a current one of the same name."""
    a = _legacy_store(tmp_path / "a.h5ad", ["a0", "a1"], ["g1", "g2"], {"cell_type": (["T", "B"], [1, 0])})
    b = _store(tmp_path / "b.h5ad", ["b0", "b1"], ["g1", "g2"],
               obs={"cell_type": categorical(["NK", "T"], [0, -1])})
    got = _concat([a, b], tmp_path / "out.h5ad")
    assert _cells(got.obs["cell_type"]) == _cells(_reference([a, b]).obs["cell_type"]) == ["B", "T", "NK", None]


def _legacy_zarr(path: Path, cells, genes, labels, codes) -> Path:
    """A Zarr v2 store as anndata 0.7 wrote it: the codes dataset's
    `categories` attribute is the path of its labels, not a reference."""
    import zarr

    def text(group, name, values):
        group.create_array(name, shape=(len(values),), dtype=str)[:] = np.array(values, dtype=object)

    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    for name, index, column in (("obs", cells, (labels, codes)), ("var", genes, None)):
        group = root.create_group(name)
        group.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.1.0",
                            "column-order": ["n", "ct"] if column else ["n"]})
        text(group, "_index", index)
        group.create_array("n", data=np.arange(len(index)))
        if column:
            text(group.create_group("__categories"), "ct", column[0])
            codes_ds = group.create_array("ct", data=np.asarray(column[1], dtype=np.int8))
            codes_ds.attrs["categories"] = "__categories/ct"
    root.create_array("X", data=np.ones((len(cells), len(genes)), dtype=np.float32))
    zarr.consolidate_metadata(str(path))
    return path


def test_anndata_07_categoricals_on_zarr_are_found_through_their_frame(tmp_path):
    """F010 on Zarr, where a 0.7 categorical names its labels by path."""
    a = _legacy_zarr(tmp_path / "a.zarr", ["a0", "a1", "a2"], ["g1", "g2"], ["T", "B"], [0, 1, 0])
    b = _legacy_zarr(tmp_path / "b.zarr", ["b0", "b1"], ["g1", "g2"], ["NK", "T"], [0, 1])
    got = _concat([a, b], tmp_path / "out.zarr")
    assert _cells(got.obs["ct"]) == _cells(_reference([a, b]).obs["ct"]) == ["T", "B", "T", "NK", "T"]


@pytest.mark.parametrize("merge", ["same", "first"])
def test_anndata_07_var_categoricals_are_merged_by_label(tmp_path, merge):
    """F010: --merge read a 0.7 var categorical's `codes` member, which a
    codes dataset does not have."""
    genes = ["g1", "g2", "g3"]
    a = _legacy_store(tmp_path / "a.h5ad", ["a0", "a1"], genes, {}, {"kind": (["x", "y"], [0, 1, 0])})
    b = _legacy_store(tmp_path / "b.h5ad", ["b0", "b1"], genes[::-1], {}, {"kind": (["y", "x"], [1, 0, 1])})
    got = _concat([a, b], tmp_path / "out.h5ad", merge=merge)
    ref = _reference([a, b], merge=merge)
    assert _cells(got.var["kind"]) == _cells(ref.var["kind"]) == ["x", "y", "x"]


# ---------------------------------------------------------------------------
# F028, F053: text columns with missing values


def test_an_outer_join_writes_a_text_column_an_input_lacks_so_every_release_reads_it(tmp_path, backend):
    """F028: a plain-string column that one input lacks came out as a
    nullable-string-array, which anndata 0.8, 0.9 and 0.10 cannot read at
    all, even when they wrote both inputs. anndata itself writes a text
    column with missing values as a categorical whose missing rows have
    code -1, which every release reads."""
    a = _on(backend, tmp_path, "a", ["a0", "a1", "a2"], ["g1"], obs={"free_text": strings("x", "yy", "zzz")})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g1"])
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out, join="outer")
    assert _encoding(out, "obs/free_text") == spec.CATEGORICAL
    with open_store(out, "r") as handle:
        assert list(handle.root["obs/free_text/codes"][...]) == [0, 1, 2, -1, -1]
    expected = _cells(_reference([a, b], join="outer").obs["free_text"])
    assert _cells(got.obs["free_text"]) == expected == ["x", "yy", "zzz", None, None]


def test_an_outer_join_of_obsm_dataframes_writes_missing_text_the_same_way(tmp_path, backend):
    """F028: obsm dataframes are joined as obs is, and had the same fault."""
    def frame(columns):
        def write(g, n):
            group = ew.write_dataframe_header(g, n, ["c0", "c1"], list(columns))
            for key, w in columns.items():
                w(group, key)
        return write

    a = _on(backend, tmp_path, "a", ["c0", "c1"], ["g1"], obsm={"meta": frame({"label": strings("p", "q")})})
    b = _on(backend, tmp_path, "b", ["c0", "c1"], ["g1"], obsm={"meta": frame({"score": dense(1.0, 2.0)})})
    out = backend_path(tmp_path, backend, "out")
    concat_on_disk([a, b], out, QUIET, join="outer", index_unique="-")
    assert _encoding(out, "obsm/meta/label") == spec.CATEGORICAL
    assert _cells(_read(out).obsm["meta"]["label"]) == ["p", "q", None, None]


def test_a_nullable_string_column_stays_nullable_when_rows_are_missing(tmp_path, backend):
    """F028: the choice of encoding keeps nullable-string-array where an input
    already used it, so nothing an input could hold is lost."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"],
            obs={"label": masked(spec.NULLABLE_STRING_ARRAY, ["x", ""], [False, True])})
    b = _on(backend, tmp_path, "b", ["b0"], ["g1"])
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out, join="outer")
    assert _encoding(out, "obs/label") == spec.NULLABLE_STRING_ARRAY
    assert _cells(got.obs["label"]) == _cells(_reference([a, b], join="outer").obs["label"]) == ["x", None, None]


@pytest.mark.parametrize(
    "first",
    [
        categorical(["x", "y"], [0, -1, 1]),
        masked(spec.NULLABLE_STRING_ARRAY, ["x", "", "y"], [False, True, False]),
    ],
    ids=["categorical", "nullable-string"],
)
def test_mixed_text_encodings_keep_their_missing_values(tmp_path, backend, first):
    """F053: a categorical or nullable column concatenated with a plain-string
    one fell back to plain strings, and its missing values became empty
    strings."""
    a = _on(backend, tmp_path, "a", ["a0", "a1", "a2"], ["g1"], obs={"label": first})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g1"], obs={"label": strings("z", "")})
    got = _concat([a, b], backend_path(tmp_path, backend, "out"))
    expected = _cells(_reference([a, b]).obs["label"])
    assert _cells(got.obs["label"]) == expected == ["x", None, "y", "z", ""]


def test_a_mixed_text_column_an_input_lacks_keeps_both_kinds_of_missing(tmp_path, backend):
    """F053: rows missing in a categorical and rows of an input without the
    column are both missing in the output, and an empty string stays one."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"], obs={"label": categorical(["x"], [0, -1])})
    b = _on(backend, tmp_path, "b", ["b0"], ["g1"], obs={"label": strings("")})
    c = _on(backend, tmp_path, "c", ["c0"], ["g1"])
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b, c], out, join="outer")
    assert _encoding(out, "obs/label") == spec.CATEGORICAL
    expected = _cells(_reference([a, b, c], join="outer").obs["label"])
    assert _cells(got.obs["label"]) == expected == ["x", None, "", None]


@pytest.mark.parametrize(
    "text",
    [categorical(["x", "y"], [0, 1]), masked(spec.NULLABLE_STRING_ARRAY, ["a", ""], [False, True])],
    ids=["categorical", "nullable-string"],
)
def test_text_meeting_a_nullable_number_column_stays_text(tmp_path, backend, text):
    """F052-F054 review: once nullable numbers were concatenated as numbers,
    a column that was text in one input and a nullable number in another
    was dropped, where it used to be kept as text. It is kept again, its
    numbers rendered and its missing values still missing."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"], obs={"c": text})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g1"],
            obs={"c": masked(spec.NULLABLE_INTEGER, [5, 0], [False, True])})
    got = _concat([a, b], backend_path(tmp_path, backend, "out"))
    expected = [None if v is None else str(v) for v in _cells(_reference([a, b]).obs["c"])]
    assert _cells(got.obs["c"]) == expected


# ---------------------------------------------------------------------------
# F051: categories that are not text


@pytest.mark.parametrize(
    "first, second",
    [([1, 2, 1], [3, 2]), ([1.5, 2.5], [2.5, 0.5]), ([True, False], [True])],
    ids=["int", "float", "bool"],
)
def test_categories_that_are_not_text_keep_their_dtype(tmp_path, first, second):
    """F051: integer categories came out as the strings '1', '2', ..."""
    def store(name, values):
        cells = [f"{name}{i}" for i in range(len(values))]
        obj = ad.AnnData(
            X=np.ones((len(values), 1), dtype=np.float32),
            obs=pd.DataFrame({"group": pd.Categorical(values)}, index=cells),
            var=pd.DataFrame({"tier": pd.Categorical([7])}, index=["g1"]),
        )
        obj.write_h5ad(tmp_path / f"{name}.h5ad")
        return tmp_path / f"{name}.h5ad"

    a, b = store("a", first), store("b", second)
    got = _concat([a, b], tmp_path / "out.h5ad", merge="same")
    ref = _reference([a, b], merge="same")
    assert got.obs["group"].cat.categories.dtype.kind == ref.obs["group"].cat.categories.dtype.kind
    assert sorted(got.obs["group"].cat.categories) == sorted(ref.obs["group"].cat.categories)
    assert _cells(got.obs["group"]) == _cells(ref.obs["group"])
    assert list(got.var["tier"].cat.categories) == [7]


@pytest.mark.parametrize(
    "values, empty",
    [([True, False], "anndata"), ([1, 2], "anndata"), ([1, 2], "adata")],
    ids=["bool-and-anndata-empty", "int-and-anndata-empty", "int-and-adata-empty"],
)
def test_an_empty_category_set_does_not_turn_categories_into_text(tmp_path, values, empty):
    """F051: an all-missing categorical has no categories, stored as float64
    by anndata and as text by this tool. Its dtype decided the union's, so
    the other input's booleans or integers came out as strings.

    Values are checked against anndata. The dtype is the other input's:
    anndata itself makes integers float64 here, an artefact of appending its
    empty float64 categories, which an integer label has no need of."""
    a = ad.AnnData(X=np.ones((len(values), 1), dtype=np.float32),
                   obs=pd.DataFrame({"g": pd.Categorical(values)}, index=[f"a{i}" for i in range(len(values))]),
                   var=pd.DataFrame(index=["g1"]))
    a.write_h5ad(tmp_path / "a.h5ad")
    if empty == "anndata":
        ad.AnnData(X=np.ones((1, 1), dtype=np.float32),
                   obs=pd.DataFrame({"g": pd.Categorical([np.nan])}, index=["b0"]),
                   var=pd.DataFrame(index=["g1"])).write_h5ad(tmp_path / "b.h5ad")
    else:
        _store(tmp_path / "b.h5ad", ["b0"], ["g1"], obs={"g": categorical([], [-1])})
    paths = [tmp_path / "a.h5ad", tmp_path / "b.h5ad"]
    got = _concat(paths, tmp_path / "out.h5ad")
    assert got.obs["g"].cat.categories.dtype.kind == np.asarray(values).dtype.kind
    assert _cells(got.obs["g"]) == values + [None]
    if empty == "anndata":
        assert _cells(got.obs["g"]) == _cells(_reference(paths).obs["g"])


def test_categories_of_different_kinds_fall_back_to_text(tmp_path):
    """F051: numbers and words share no dtype but text."""
    a = _store(tmp_path / "a.h5ad", ["a0"], ["g1"], obs={"group": categorical(["x"], [0])})
    b = ad.AnnData(X=np.ones((1, 1), dtype=np.float32),
                   obs=pd.DataFrame({"group": pd.Categorical([4])}, index=["b0"]),
                   var=pd.DataFrame(index=["g1"]))
    b.write_h5ad(tmp_path / "b.h5ad")
    got = _concat([a, tmp_path / "b.h5ad"], tmp_path / "out.h5ad")
    assert _cells(got.obs["group"]) == ["x", "4"]


def test_numeric_categories_meeting_text_are_written_as_text(tmp_path, backend):
    """F051, F053: integer categories with a missing code, concatenated with
    a plain-string column, become text and keep the gap."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"], obs={"group": lambda g, n: ew.write_categorical(
        g, n, [1, -1], np.array([3, 4]))})
    b = _on(backend, tmp_path, "b", ["b0"], ["g1"], obs={"group": strings("x")})
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out)
    assert _encoding(out, "obs/group") == spec.CATEGORICAL
    assert _cells(got.obs["group"]) == ["4", None, "x"]
    assert [str(v) if v is not None else None for v in _cells(_reference([a, b]).obs["group"])] == ["4", None, "x"]


# ---------------------------------------------------------------------------
# F052, F054: nullable and plain numbers


def test_a_nullable_integer_and_a_nullable_boolean_promote_to_nullable_integer(tmp_path, backend):
    """F052: the two masked kinds were written as nullable-string-array, so the
    numbers became text and anndata 0.8 to 0.10 could not open the file."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"],
            obs={"n": masked(spec.NULLABLE_INTEGER, [5, 0], [False, True])})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g1"],
            obs={"n": masked(spec.NULLABLE_BOOLEAN, [True, False], [False, True])})
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out)
    assert _encoding(out, "obs/n") == spec.NULLABLE_INTEGER
    assert str(got.obs["n"].dtype) == "Int64"
    # pandas keeps them apart as objects; True == 1 all the same.
    assert _cells(got.obs["n"]) == _cells(_reference([a, b]).obs["n"]) == [5, None, 1, None]


@pytest.mark.parametrize(
    "plain, nullable, enc, dtype",
    [
        (dense(1, 2, dtype=np.int64), masked(spec.NULLABLE_INTEGER, [3, 0], [False, True]),
         spec.NULLABLE_INTEGER, "Int64"),
        (dense(True, False), masked(spec.NULLABLE_BOOLEAN, [True, False], [False, True]),
         spec.NULLABLE_BOOLEAN, "boolean"),
    ],
    ids=["int", "bool"],
)
def test_a_plain_column_and_a_nullable_one_concatenate_as_nullable(tmp_path, backend, plain, nullable, enc, dtype):
    """F054: a column plain in one input and nullable in another (int64 and
    Int64, bool and boolean) was dropped as 'cannot concatenate encodings'.
    pandas promotes the pair to the nullable dtype."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"], obs={"n": plain})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g1"], obs={"n": nullable})
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out)
    ref = _reference([a, b])
    assert _encoding(out, "obs/n") == enc
    assert str(got.obs["n"].dtype) == str(ref.obs["n"].dtype) == dtype
    assert _cells(got.obs["n"]) == _cells(ref.obs["n"])


def test_a_plain_float_column_and_a_nullable_integer_one_stay_float(tmp_path, backend):
    """F054: there is no nullable float encoding, so float wins and the masked
    rows are NaN, which pandas' Float64 holds as missing too."""
    a = _on(backend, tmp_path, "a", ["a0"], ["g1"], obs={"n": dense(1.5)})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g1"], obs={"n": masked(spec.NULLABLE_INTEGER, [3, 0], [False, True])})
    got = _concat([a, b], backend_path(tmp_path, backend, "out"))
    assert got.obs["n"].dtype == np.float64
    assert _cells(got.obs["n"]) == _cells(_reference([a, b]).obs["n"]) == [1.5, 3.0, None]


def test_a_boolean_column_an_input_lacks_becomes_nullable_boolean(tmp_path, backend):
    """F054: a bool column missing from one input became float64 (1.0, 0.0,
    NaN). pandas keeps the booleans as objects; nullable-boolean is the
    encoding that keeps both them and the gaps."""
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g1"], obs={"flag": dense(True, False)})
    b = _on(backend, tmp_path, "b", ["b0"], ["g1"])
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out, join="outer")
    assert _encoding(out, "obs/flag") == spec.NULLABLE_BOOLEAN
    assert _cells(got.obs["flag"]) == _cells(_reference([a, b], join="outer").obs["flag"]) == [True, False, None]


def test_an_integer_column_an_input_lacks_still_becomes_float(tmp_path, backend):
    """F054 leaves this as pandas does it: NaN in float64."""
    a = _on(backend, tmp_path, "a", ["a0"], ["g1"], obs={"n": dense(4, dtype=np.int32)})
    b = _on(backend, tmp_path, "b", ["b0"], ["g1"])
    got = _concat([a, b], backend_path(tmp_path, backend, "out"), join="outer")
    assert got.obs["n"].dtype == np.float64
    assert _cells(got.obs["n"]) == _cells(_reference([a, b], join="outer").obs["n"]) == [4.0, None]


# ---------------------------------------------------------------------------
# F055: sparse obsm


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_obsm_entries_are_stacked(tmp_path, backend, fmt):
    """F055: a sparse obsm entry present in every input was dropped as 'not an
    array'. anndata.concat stacks it."""
    def entry(seed):
        values = sparse.random(3, 4, density=0.5, format=fmt, random_state=seed, dtype=np.float32)
        return matrix(values)

    a = _on(backend, tmp_path, "a", ["a0", "a1", "a2"], ["g1"], obsm={"spots": entry(0)})
    b = _on(backend, tmp_path, "b", ["b0", "b1", "b2"], ["g1"], obsm={"spots": entry(1)})
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out)
    ref = _reference([a, b])
    assert _encoding(out, "obsm/spots") == f"{fmt}_matrix"
    assert sparse.issparse(got.obsm["spots"])
    assert np.array_equal(got.obsm["spots"].toarray(), ref.obsm["spots"].toarray())


def test_obsm_entries_with_different_encodings_are_refused_before_writing(tmp_path):
    """F055: refused up front, as X is, rather than dropped or half written."""
    a = _store(tmp_path / "a.h5ad", ["a0", "a1"], ["g1"],
               obsm={"spots": matrix(sparse.csr_matrix(np.eye(2, 3, dtype=np.float32)))})
    b = _store(tmp_path / "b.h5ad", ["b0", "b1"], ["g1"], obsm={"spots": dense(*np.eye(2, 3))})
    out = tmp_path / "out.h5ad"
    with pytest.raises(ValueError, match="obsm/spots") as raised:
        concat_on_disk([a, b], out, QUIET)
    assert "adata recast" in str(raised.value)
    _no_output(out)


# ---------------------------------------------------------------------------
# F056, F057: merging uns and var


def _plain(value: Any) -> Any:
    """A merged mapping with numpy containers turned into Python ones."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@pytest.mark.parametrize("strategy", ["same", "unique", "first", "only"])
def test_nested_uns_merges_as_anndata_merges_it(tmp_path, strategy):
    """F056: first and only took or dropped a nested mapping whole instead of
    deciding key by key, and same and unique left behind the empty mappings
    whose members all disagreed."""
    uns = [
        {"nested": {"x": 1, "deep": {"y": "a", "z": 2}}, "conflict": {"k": 1}, "empty": {}, "solo": {"s": 1}},
        {"nested": {"w": 3, "deep": {"y": "b", "z": 2}}, "conflict": {"k": 2}, "empty": {}},
    ]
    paths = []
    for i, mapping in enumerate(uns):
        obj = ad.AnnData(X=np.ones((1, 1), dtype=np.float32),
                         obs=pd.DataFrame(index=[f"c{i}"]), var=pd.DataFrame(index=["g1"]))
        obj.uns.update(mapping)
        obj.write_h5ad(tmp_path / f"{i}.h5ad")
        paths.append(tmp_path / f"{i}.h5ad")
    got = _concat(paths, tmp_path / "out.h5ad", uns_merge=strategy)
    ref = _reference(paths, uns_merge=strategy)
    assert _plain(dict(got.uns)) == _plain(dict(ref.uns))


def test_same_and_unique_match_text_across_an_h5ad_and_a_zarr_input(tmp_path):
    """F057: text read from HDF5 is bytes and from Zarr is str, and the two
    were compared undecoded, so a string in uns or var never matched across
    backends and --merge/--uns-merge same dropped it."""
    paths = []
    for stem, suffix in (("h5", ".h5ad"), ("zr", ".zarr")):
        obj = ad.AnnData(X=np.ones((1, 2), dtype=np.float32),
                         obs=pd.DataFrame(index=[f"c{suffix}"]),
                         var=pd.DataFrame({"gene_ids": ["E1", "E2"]}, index=["g1", "g2"]))
        obj.uns["title"] = "atlas"
        obj.uns["labels"] = np.array(["x", "y"])
        obj.uns["frame"] = pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"])
        path = tmp_path / f"{stem}{suffix}"
        obj.write_zarr(path) if suffix == ".zarr" else obj.write_h5ad(path)
        paths.append(path)
    inputs = [_read(p) for p in paths]
    for strategy in ("same", "unique"):
        got = _concat(paths, tmp_path / f"out-{strategy}.h5ad", merge=strategy, uns_merge=strategy)
        ref = _reference(paths, merge=strategy, uns_merge=strategy)
        assert {"frame", "title"} <= set(ref.uns)
        # anndata reads the two `labels` as object and StringDType arrays and
        # its own comparison then calls them different; their values are
        # what decides, and both inputs hold the same ones.
        assert [list(i.uns["labels"]) for i in inputs] == [["x", "y"]] * 2
        assert sorted(got.uns) == ["frame", "labels", "title"]
        assert got.uns["title"] == ref.uns["title"] == "atlas"
        assert list(got.uns["labels"]) == ["x", "y"]
        pd.testing.assert_frame_equal(got.uns["frame"], ref.uns["frame"], check_dtype=False)
        assert list(got.var["gene_ids"]) == list(ref.var["gene_ids"]) == ["E1", "E2"]


# ---------------------------------------------------------------------------
# F115: CSR column order


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_csr_rows_stay_sorted_when_an_input_orders_var_differently(tmp_path, backend, join):
    """F115: the remapped column indices were appended in source order, so a
    row of an input whose var order differs from the target's came out
    unsorted -- a matrix scipy flags as non-canonical."""
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(6)]
    a = _on(backend, tmp_path, "a", ["a0", "a1", "a2"], genes,
            X=sparse.csr_matrix(rng.poisson(1, (3, 6)).astype(np.float32)))
    b = _on(backend, tmp_path, "b", ["b0", "b1", "b2", "b3"], genes[::-1] + ["extra"],
            X=sparse.csr_matrix(rng.poisson(1, (4, 7)).astype(np.float32)))
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out, join=join, chunk_rows=2)
    with open_store(out, "r") as handle:
        x = handle.root["X"]
        matrix = sparse.csr_matrix((x["data"][...], x["indices"][...], x["indptr"][...]),
                                   shape=tuple(x.attrs["shape"]))
    assert matrix.has_sorted_indices
    ref = _reference([a, b], join=join)
    assert list(got.var_names) == list(ref.var_names) or join == "outer"
    order = [list(got.var_names).index(g) for g in ref.var_names]
    assert np.array_equal(got.X.toarray()[:, order], ref.X.toarray())


# ---------------------------------------------------------------------------
# F117: no variables at all


@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_inputs_without_variables_concatenate(tmp_path, backend, join, layout):
    """F117: inputs with no var at all were refused with 'Use --join outer',
    even under --join outer. anndata concatenates them to n_obs x 0."""
    def empty(n):
        values = np.zeros((n, 0), dtype=np.float32)
        return {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix}.get(layout, np.asarray)(values)

    a = _on(backend, tmp_path, "a", ["a0", "a1"], [], X=empty(2), obs={"n": dense(1, 2)})
    b = _on(backend, tmp_path, "b", ["b0"], [], X=empty(1), obs={"n": dense(3)})
    got = _concat([a, b], backend_path(tmp_path, backend, "out"), join=join)
    ref = _reference([a, b], join=join)
    assert got.shape == ref.shape == (3, 0)
    assert list(got.obs["n"]) == list(ref.obs["n"]) == [1, 2, 3]


def test_inputs_sharing_no_variables_are_still_refused_under_an_inner_join(tmp_path):
    """F117 keeps the refusal it was worded for: variables exist, none shared."""
    a = _store(tmp_path / "a.h5ad", ["a0"], ["g1"])
    b = _store(tmp_path / "b.h5ad", ["b0"], ["g2"])
    with pytest.raises(ValueError, match="--join outer"):
        concat_on_disk([a, b], tmp_path / "out.h5ad", QUIET)


# ---------------------------------------------------------------------------
# F075: dense layout


def _layout(path: Path, element: str) -> tuple:
    """A dense array's chunks and compression, as its backend spells them."""
    with open_store(path, "r") as handle:
        array = handle.root[element]
        if isinstance(array, h5py.Dataset):
            return array.chunks, array.compression
        return tuple(array.chunks), [type(c).__name__ for c in array.compressors]


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_dense_matrices_and_obsm_arrays_keep_the_sources_chunking_and_compression(tmp_path, backend, join):
    """F075: dense X, layers and obsm arrays were written contiguous and
    uncompressed whatever the inputs were, so 109 KB of gzip-compressed
    input came out as 3.7 MB. The sparse path and varm kept theirs."""
    rng = np.random.default_rng(0)

    def make(stem, cells, genes):
        x = rng.poisson(1, (len(cells), len(genes))).astype(np.float32)
        return _on(backend, tmp_path, stem, cells, genes, X=chunked(x, (3, 2)),
                   layers={"counts": chunked(x * 2, (3, 2))},
                   obsm={"X_pca": chunked(rng.normal(size=(len(cells), 4)), (3, 4))})

    a = make("a", [f"a{i}" for i in range(7)], ["g0", "g1", "g2", "g3"])
    b = make("b", [f"b{i}" for i in range(5)], ["g0", "g1", "g2", "g4"])
    out = backend_path(tmp_path, backend, "out")
    got = _concat([a, b], out, join=join, chunk_rows=4)
    for element in ("X", "layers/counts", "obsm/X_pca"):
        assert _layout(out, element) == _layout(a, element), element
    ref = _reference([a, b], join=join, fill_value=0)
    assert np.array_equal(got.X, ref.X)
    assert np.array_equal(got.layers["counts"], ref.layers["counts"])
    assert np.array_equal(got.obsm["X_pca"], ref.obsm["X_pca"])


# ---------------------------------------------------------------------------
# F113: an outer join's var order


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize(
    "genes_b",
    [["g2", "g0", "g1"], ["g0", "g1", "g2"], ["g3", "g0"], ["g2", "g0"]],
    ids=["same-order", "reordered", "new-genes", "a-subset"],
)
def test_an_outer_join_orders_var_as_anndata_does(tmp_path, backend, layout, genes_b):
    """F113: an outer join's var came out in order of first appearance, where
    anndata.concat sorts the union unless every input has the same var in
    the same order. Values matched once aligned, but code that aligns by
    position against an anndata-made reference saw another matrix."""
    genes_a = ["g2", "g0", "g1"]
    as_layout = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout]

    def X(n_cells, n_genes, start):
        return as_layout(np.arange(start, start + n_cells * n_genes, dtype=np.float32).reshape(n_cells, n_genes))

    a = _on(backend, tmp_path, "a", ["a0", "a1"], genes_a, X=X(2, 3, 1))
    b = _on(backend, tmp_path, "b", ["b0", "b1", "b2"], genes_b, X=X(3, len(genes_b), 10))
    got = _concat([a, b], backend_path(tmp_path, backend, "out"), join="outer")
    ref = _reference([a, b], join="outer", fill_value=0)
    assert list(got.var_names) == list(ref.var_names)
    dense = lambda m: m.toarray() if sparse.issparse(m) else np.asarray(m)
    assert np.array_equal(dense(got.X), dense(ref.X))


def test_an_outer_join_keeps_a_shared_var_order_beside_inputs_without_variables(tmp_path):
    """F113: an input with no variables has no say in the order, as in
    anndata, where a union with an empty index is the other index as is."""
    genes = ["g2", "g0", "g1"]
    a = _store(tmp_path / "a.h5ad", ["a0"], genes)
    empty = _store(tmp_path / "e.h5ad", ["e0"], [], X=np.zeros((1, 0), dtype=np.float32))
    b = _store(tmp_path / "b.h5ad", ["b0"], genes)
    got = _concat([a, empty, b], tmp_path / "out.h5ad", join="outer")
    ref = _reference([a, empty, b], join="outer", fill_value=0)
    assert list(got.var_names) == list(ref.var_names) == genes
    assert np.array_equal(got.X, ref.X)


# ---------------------------------------------------------------------------
# F114: the order of merged categories


@pytest.mark.parametrize(
    "first, second",
    [
        (["t2", "t10", "b"], ["t1", "b"]),
        (["b", "a"], ["a", "b"]),
        (["c", "b", "a"], ["a"]),
        (["a"], ["c", "b", "a"]),
        (["b", "a"], ["b", "a"]),
        ([10, 2], [3, 10]),
        ([2.5, -1.0], [0.5]),
    ],
    ids=["new-categories", "same-set-reordered", "first-covers-the-union",
         "second-covers-the-union", "identical", "integers", "floats"],
)
def test_merged_categories_are_ordered_as_anndata_orders_them(tmp_path, backend, first, second):
    """F114: merged categories came out in order of first appearance, where
    anndata natsorts the union -- unless the first input already holds every
    category, whose order pandas then keeps. `.cat.codes` and plotting order
    differed from anndata.concat's."""
    a = _on(backend, tmp_path, "a", [f"a{i}" for i in range(len(first))], ["g"],
            obs={"c": categorical(first, list(range(len(first))))})
    b = _on(backend, tmp_path, "b", [f"b{i}" for i in range(len(second))], ["g"],
            obs={"c": categorical(second, list(range(len(second))))})
    got = _concat([a, b], backend_path(tmp_path, backend, "out"))
    ref = _reference([a, b])
    assert list(got.obs["c"].cat.categories) == list(ref.obs["c"].cat.categories)
    assert list(got.obs["c"]) == list(ref.obs["c"])
    assert not got.obs["c"].cat.ordered


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_merged_categories_follow_the_first_input_that_has_the_column(tmp_path, join):
    """F114 under an outer join: the first input without the column has no
    say, as in anndata."""
    a = _store(tmp_path / "a.h5ad", ["a0"], ["g"], obs={"other": strings("x")})
    b = _store(tmp_path / "b.h5ad", ["b0", "b1", "b2"], ["g"],
               obs={"other": strings("y", "y", "y"), "c": categorical(["c", "b", "a"], [0, 1, 2])})
    c = _store(tmp_path / "c.h5ad", ["c0"], ["g"],
               obs={"other": strings("z"), "c": categorical(["a"], [0])})
    got = _concat([a, b, c], tmp_path / "out.h5ad", join=join)
    ref = _reference([a, b, c], join=join)
    if join == "outer":
        assert list(got.obs["c"].cat.categories) == list(ref.obs["c"].cat.categories)
        assert _cells(got.obs["c"]) == _cells(ref.obs["c"])
    else:
        assert "c" not in got.obs and "c" not in ref.obs


@pytest.mark.parametrize(
    "second, keeps_order",
    [
        ((["lo", "mid", "hi"], True), True),
        ((["hi", "mid", "lo"], True), False),
        ((["lo", "mid", "hi", "max"], True), False),
        ((["lo", "mid", "hi"], False), False),
        (([], True), True),
    ],
    ids=["same-order", "reversed", "another-set", "second-unordered", "second-all-missing"],
)
def test_ordered_categoricals_stay_ordered_only_when_every_input_agrees(tmp_path, backend, second, keeps_order):
    """F114: 'ordered' categoricals whose orders conflict kept the first
    input's order, silently reversing what the second input's meant.
    anndata gives up on the order (plain values); concat now writes them
    unordered, with natsorted categories and a warning."""
    first = ["lo", "mid", "hi"]
    labels, ordered = second
    a = _on(backend, tmp_path, "a", ["a0", "a1", "a2"], ["g"],
            obs={"c": categorical(first, [0, 1, 2], ordered=True)})
    codes = [len(labels) - 1, 0] if labels else [-1, -1]
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g"],
            obs={"c": categorical(labels, codes, ordered=ordered)})
    console = Console(record=True, width=200)
    out = backend_path(tmp_path, backend, "out")
    concat_on_disk([a, b], out, console)
    got, ref = _read(out), _reference([a, b])
    assert _cells(got.obs["c"]) == _cells(ref.obs["c"])
    warned = "written unordered" in " ".join(console.export_text().split())
    if keeps_order:
        assert ref.obs["c"].cat.ordered and got.obs["c"].cat.ordered
        assert list(got.obs["c"].cat.categories) == list(ref.obs["c"].cat.categories) == first
        assert not warned
    else:
        assert ref.obs["c"].dtype == object
        assert not got.obs["c"].cat.ordered
        union = sorted(set(first) | set(labels))
        assert list(got.obs["c"].cat.categories) == union
        assert warned


# ---------------------------------------------------------------------------
# F116: --fill-value and integer matrices


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_fill_value_leaves_integer_matrices_alone_when_nothing_is_filled(tmp_path, backend, layout):
    """F116: --fill-value made integer X and layers float64 on an inner join,
    where no cell is filled, doubling X; anndata.concat keeps them int32."""
    as_layout = {"csr": sparse.csr_matrix, "dense": np.asarray}[layout]
    x = np.arange(6, dtype=np.int32).reshape(2, 3)
    a = _on(backend, tmp_path, "a", ["a0", "a1"], ["g0", "g1", "g2"], X=as_layout(x),
            layers={"counts": matrix(as_layout(x * 2))})
    b = _on(backend, tmp_path, "b", ["b0", "b1"], ["g2", "g1", "g3"], X=as_layout(x + 10),
            layers={"counts": matrix(as_layout(x * 3))})
    got = _concat([a, b], backend_path(tmp_path, backend, "out"), fill_value=-1)
    ref = _reference([a, b], fill_value=-1)
    dense = lambda m: m.toarray() if sparse.issparse(m) else np.asarray(m)
    for mine, theirs in ((got.X, ref.X), (got.layers["counts"], ref.layers["counts"])):
        assert mine.dtype == theirs.dtype == np.int32
        assert np.array_equal(dense(mine), dense(theirs))


@pytest.mark.parametrize(
    "dtype, fill, expected",
    [
        (np.int32, -1, np.int32),
        (np.int32, 7, np.int32),
        (np.int32, 0.5, np.float64),
        (np.int32, np.nan, np.float64),
        (np.uint8, -1, np.float64),
        (np.uint8, 300, np.float64),
        (np.bool_, 1, np.bool_),
        (np.bool_, 2, np.float64),
        (np.float32, 0.5, np.float32),
    ],
)
def test_an_outer_join_widens_a_matrix_only_for_a_fill_value_it_cannot_hold(tmp_path, dtype, fill, expected):
    """F116: when an outer join does fill cells, an integer matrix that can
    hold the fill value keeps its dtype; one that cannot becomes float64.
    The values are anndata.concat's either way."""
    x = np.array([[1, 0], [1, 1]], dtype=dtype)
    a = _store(tmp_path / "a.h5ad", ["a0", "a1"], ["g0", "g1"], X=x)
    b = _store(tmp_path / "b.h5ad", ["b0", "b1"], ["g0", "g2"], X=x)
    got = _concat([a, b], tmp_path / "out.h5ad", join="outer", fill_value=fill)
    ref = _reference([a, b], join="outer", fill_value=fill)
    assert got.X.dtype == expected
    assert np.array_equal(got.X.astype(np.float64), ref.X.astype(np.float64), equal_nan=True)

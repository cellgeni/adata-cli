"""Tests for the import command."""

import json
import re
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import io as scipy_io
from scipy import sparse
from typer.testing import CliRunner

from rich.console import Console

from adata.cli import app
from tests.conftest import backend_path, backend_zarr_format


runner = CliRunner()
QUIET = Console(quiet=True)


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestImportDataframe:
    def test_import_dataframe_obs_inplace(self, sample_h5ad_file, temp_dir):
        """Test importing CSV into obs with --inplace."""
        csv_file = temp_dir / "new_obs.csv"
        csv_file.write_text(
            "cell_id,score,label\n"
            "cell_1,1.5,A\n"
            "cell_2,2.5,B\n"
            "cell_3,3.5,A\n"
            "cell_4,4.5,C\n"
            "cell_5,5.5,B\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "--inplace",
                "-i",
                "cell_id",
            ],
        )
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "5 rows" in output
        assert "2 columns" in output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "obs" in f
            obs = f["obs"]
            assert "score" in obs
            assert "label" in obs

    def test_import_dataframe_obs_output(self, sample_h5ad_file, temp_dir):
        """Test importing CSV into obs with output file."""
        csv_file = temp_dir / "new_obs.csv"
        csv_file.write_text(
            "cell_id,score,label\n"
            "cell_1,1.5,A\n"
            "cell_2,2.5,B\n"
            "cell_3,3.5,A\n"
            "cell_4,4.5,C\n"
            "cell_5,5.5,B\n"
        )
        output_file = temp_dir / "output.h5ad"

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "-o",
                str(output_file),
                "-i",
                "cell_id",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output file has the new data
        with h5py.File(output_file, "r") as f:
            assert "obs" in f
            obs = f["obs"]
            assert "score" in obs

        # Verify source file is unchanged
        with h5py.File(sample_h5ad_file, "r") as f:
            obs = f["obs"]
            assert "score" not in obs

    def test_import_dataframe_var(self, sample_h5ad_file, temp_dir):
        """Test importing CSV into var."""
        csv_file = temp_dir / "new_var.csv"
        csv_file.write_text(
            "gene_id,mean,std\n"
            "gene_1,0.1,0.01\n"
            "gene_2,0.2,0.02\n"
            "gene_3,0.3,0.03\n"
            "gene_4,0.4,0.04\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "var",
                str(csv_file),
                "--inplace",
                "-i",
                "gene_id",
            ],
        )
        assert result.exit_code == 0
        assert "4 rows" in strip_ansi(result.output)

    def test_import_dataframe_dimension_mismatch(self, sample_h5ad_file, temp_dir):
        """Test that dimension mismatch is rejected."""
        csv_file = temp_dir / "wrong_obs.csv"
        csv_file.write_text("cell_id,score\ncell_1,1.0\ncell_2,2.0\n")

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "--inplace",
                "-i",
                "cell_id",
            ],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()

    def test_import_dataframe_invalid_index_column(self, sample_h5ad_file, temp_dir):
        """Test that invalid index column is rejected."""
        csv_file = temp_dir / "obs.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "--inplace",
                "-i",
                "nonexistent",
            ],
        )
        assert result.exit_code == 1
        assert "has no 'nonexistent' column; it has: a, b, c" in strip_ansi(result.output)

    def test_import_dataframe_at_arbitrary_path(self, sample_h5ad_file, temp_dir):
        """Dataframes can be imported anywhere, not just obs/var (issue #4)."""
        csv_file = temp_dir / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "uns/data",
                str(csv_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0, result.output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert f["uns/data"].attrs["encoding-type"] == "dataframe"
            assert "b" in f["uns/data"]

    def test_import_dataframe_requires_output_or_inplace(
        self, sample_h5ad_file, temp_dir
    ):
        """Test that either --output or --inplace is required."""
        csv_file = temp_dir / "obs.csv"
        csv_file.write_text("a,b\n1,2\n")

        result = runner.invoke(
            app,
            [
                "import",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
            ],
        )
        assert result.exit_code == 1
        assert "Output file is required" in result.output


class TestImportArray:
    def test_import_array_obsm(self, sample_h5ad_file, temp_dir):
        """Test importing NPY into obsm."""
        npy_file = temp_dir / "pca.npy"
        arr = np.random.randn(5, 10).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "obsm/X_pca",
                str(npy_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0
        assert "5×10" in strip_ansi(result.output)

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "obsm/X_pca" in f
            np.testing.assert_allclose(f["obsm/X_pca"][...], arr)

    def test_import_array_varm(self, sample_h5ad_file, temp_dir):
        """Test importing NPY into varm."""
        npy_file = temp_dir / "pcs.npy"
        arr = np.random.randn(4, 5).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "varm/PCs",
                str(npy_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "varm/PCs" in f

    def test_import_array_X(self, sample_h5ad_file, temp_dir):
        """Test importing NPY into X."""
        npy_file = temp_dir / "X.npy"
        arr = np.random.randn(5, 4).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "X",
                str(npy_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "X" in f
            np.testing.assert_allclose(f["X"][...], arr)

    def test_import_array_dimension_mismatch_obsm(self, sample_h5ad_file, temp_dir):
        """Test that obsm dimension mismatch is rejected."""
        npy_file = temp_dir / "bad_pca.npy"
        arr = np.random.randn(10, 5).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "obsm/X_pca",
                str(npy_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()

    def test_import_array_dimension_mismatch_X(self, sample_h5ad_file, temp_dir):
        """Test that X dimension mismatch is rejected."""
        npy_file = temp_dir / "bad_X.npy"
        arr = np.random.randn(5, 10).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "X",
                str(npy_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()

    def test_import_array_requires_output_or_inplace(self, sample_h5ad_file, temp_dir):
        """Test that either --output or --inplace is required."""
        npy_file = temp_dir / "data.npy"
        np.save(npy_file, np.array([1, 2, 3]))

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "obsm/X_pca",
                str(npy_file),
            ],
        )
        assert result.exit_code == 1
        assert "Output file is required" in result.output


class TestImportSparse:
    def test_import_sparse_X(self, sample_h5ad_file, temp_dir):
        """Test importing MTX into X."""
        mtx_file = temp_dir / "X.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n"
            "% test matrix\n"
            "5 4 5\n"
            "1 1 1.0\n"
            "2 2 2.0\n"
            "3 3 3.0\n"
            "4 4 4.0\n"
            "5 1 5.0\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                "sparse",
                str(sample_h5ad_file),
                "X",
                str(mtx_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "5×4" in output
        assert "5 non-zero" in output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "X" in f
            X = f["X"]
            enc = X.attrs.get("encoding-type")
            if isinstance(enc, bytes):
                enc = enc.decode("utf-8")
            assert enc == "csr_matrix"

    def test_import_sparse_layer(self, sample_h5ad_file, temp_dir):
        """Test importing MTX into layers."""
        mtx_file = temp_dir / "layer.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n"
            "5 4 3\n"
            "1 1 1.0\n"
            "3 2 2.0\n"
            "5 4 3.0\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                "sparse",
                str(sample_h5ad_file),
                "layers/counts",
                str(mtx_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "layers/counts" in f

    def test_import_sparse_dimension_mismatch(self, sample_h5ad_file, temp_dir):
        """Test that MTX dimension mismatch is rejected."""
        mtx_file = temp_dir / "bad.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n" "10 4 1\n" "1 1 1.0\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                "sparse",
                str(sample_h5ad_file),
                "X",
                str(mtx_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()

    def test_import_sparse_requires_output_or_inplace(self, sample_h5ad_file, temp_dir):
        """Test that either --output or --inplace is required."""
        mtx_file = temp_dir / "data.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n" "5 4 1\n" "1 1 1.0\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                "sparse",
                str(sample_h5ad_file),
                "X",
                str(mtx_file),
            ],
        )
        assert result.exit_code == 1
        assert "Output file is required" in result.output


class TestImportDict:
    def test_import_dict_uns(self, sample_h5ad_file, temp_dir):
        """Test importing JSON into uns."""
        json_file = temp_dir / "metadata.json"
        json_file.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "colors": ["red", "green", "blue"],
                    "n_pcs": 50,
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "import",
                "dict",
                str(sample_h5ad_file),
                "uns/metadata",
                str(json_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0
        assert "JSON data" in result.output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "uns/metadata" in f
            assert "colors" in f["uns/metadata"]
            assert "n_pcs" in f["uns/metadata"]

    def test_import_dict_nested(self, sample_h5ad_file, temp_dir):
        """Test importing nested JSON."""
        json_file = temp_dir / "config.json"
        json_file.write_text(
            json.dumps(
                {
                    "settings": {
                        "threshold": 0.5,
                        "enabled": True,
                    },
                    "labels": ["A", "B", "C"],
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "import",
                "dict",
                str(sample_h5ad_file),
                "uns/config",
                str(json_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "uns/config/settings" in f
            assert "uns/config/labels" in f

    def test_import_dict_requires_output_or_inplace(self, sample_h5ad_file, temp_dir):
        """Test that either --output or --inplace is required."""
        json_file = temp_dir / "data.json"
        json_file.write_text('{"key": "value"}')

        result = runner.invoke(
            app,
            [
                "import",
                "dict",
                str(sample_h5ad_file),
                "uns/data",
                str(json_file),
            ],
        )
        assert result.exit_code == 1
        assert "Output file is required" in result.output


class TestImportValidation:
    def test_replace_existing_object(self, sample_h5ad_file, temp_dir):
        """Test that existing objects can be replaced."""
        with h5py.File(sample_h5ad_file, "r") as f:
            original_X = np.array(f["X"][...])

        npy_file = temp_dir / "new_X.npy"
        new_arr = np.ones((5, 4), dtype=np.float32) * 999
        np.save(npy_file, new_arr)

        result = runner.invoke(
            app,
            [
                "import",
                "array",
                str(sample_h5ad_file),
                "X",
                str(npy_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            np.testing.assert_allclose(f["X"][...], new_arr)
            assert not np.allclose(f["X"][...], original_X)


# ---------------------------------------------------------------------------
# Read back with anndata and pandas
#
# The tests above look for the written elements with h5py. These read the
# whole store back with anndata, on every backend, and compare what it holds
# with what pandas, numpy and scipy read from the input (F089).

N_OBS = 6


def _out(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _read(path: Path):
    """anndata's reading of the store, its complaints about format as errors."""
    from anndata._warnings import OldFormatWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", OldFormatWarning)
        return ad.read_h5ad(path) if path.suffix == ".h5ad" else ad.read_zarr(path)


@pytest.fixture
def written(temp_dir, backend):
    """A store anndata wrote, holding every slot an import can aim at."""
    obj = ad.AnnData(
        X=np.arange(N_OBS * 3, dtype="float32").reshape(N_OBS, 3),
        obs=pd.DataFrame({"n": np.arange(N_OBS)}, index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame({"m": [1.0, 2.0, 3.0]}, index=["g1", "g2", "g3"]),
    )
    obj.obsm["X_pca"] = np.ones((N_OBS, 2))
    obj.varm["PCs"] = np.ones((3, 2))
    obj.layers["counts"] = obj.X.copy()
    obj.obsp["distances"] = sparse.csr_matrix(np.eye(N_OBS, dtype="float32"))
    obj.uns["keep"] = {"a": 1}
    obj.raw = obj
    path = backend_path(temp_dir, backend, "written")
    previous = ad.settings.zarr_write_format
    ad.settings.zarr_write_format = backend_zarr_format(backend) or previous
    try:
        obj.write_h5ad(path) if backend == "h5ad" else obj.write_zarr(path)
    finally:
        ad.settings.zarr_write_format = previous
    return path


def _import(kind, store, entry, source, *extra):
    out = store.with_name(f"out{len(list(store.parent.glob('out*')))}{store.suffix}")
    result = runner.invoke(app, ["import", kind, str(store), entry, str(source), "-o", str(out), *extra])
    return result, out


def _pandas_frame(csv: Path, index: str):
    """What anndata writes for `pd.read_csv`'s frame: text that repeats a
    value becomes a categorical."""
    frame = pd.read_csv(csv).set_index(index)
    frame.index.name = None
    frame.index = frame.index.astype(str)
    obj = ad.AnnData(obs=frame)
    obj.strings_to_categoricals()
    return obj.obs


def _labels(column) -> list:
    return column.astype(object).where(column.notna(), "<na>").map(str).tolist()


def _same_frame(got, expected) -> None:
    assert list(got.index) == list(expected.index)
    assert list(got.columns) == list(expected.columns)
    for name in expected.columns:
        a, b = got[name], expected[name]
        if isinstance(b.dtype, pd.CategoricalDtype):
            assert isinstance(a.dtype, pd.CategoricalDtype), name
            assert list(a.cat.categories) == list(b.cat.categories), name
            assert a.isna().tolist() == b.isna().tolist(), name
            assert _labels(a) == _labels(b), name
        elif b.dtype == object:
            # pandas reads True/False with gaps as Python objects, a column
            # anndata cannot write: it stays the text it was, a categorical.
            assert isinstance(a.dtype, pd.CategoricalDtype), name
            assert _labels(a) == _labels(b), name
        elif pd.api.types.is_string_dtype(b.dtype):
            # anndata reads a Zarr string array as objects, an HDF5 one as str.
            assert pd.api.types.is_string_dtype(a.dtype), (name, a.dtype)
            assert _labels(a) == _labels(b), name
        else:
            assert a.dtype == b.dtype, (name, a.dtype, b.dtype)
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy(), name)


CELLS = (
    "﻿cell,score,count,gappy,flag,flags,label,note,sample\n"
    "c0,1.5,1,1,True,True,A,one,1_1\n"
    "c1,,2,,False,,B,two,1_10\n"
    "c2,2.5,3,3,True,False,A,three,11_0\n"
    "c3,NA,4,4,False,True,C,four,2_1\n"
    "c4,3.5,5,5,True,,B,five,2_2\n"
    "c5,4.5,6,6,True,True,A,six,2_3\n"
)


def test_imported_columns_are_typed_as_pandas_reads_them(written, temp_dir):
    """F018: a numeric column with a gap was stored as strings ('1.5', '',
    'NA'), True/False as a categorical of strings, labels like 1_1 as
    integers (F020), and the BOM Excel writes stayed in the first name."""
    csv = temp_dir / "cells.csv"
    csv.write_text(CELLS, encoding="utf-8")

    result, out = _import("dataframe", written, "obs", csv, "-i", "cell")

    assert result.exit_code == 0, _out(result)
    got = _read(out).obs
    expected = _pandas_frame(csv, "cell")
    assert {k: str(v) for k, v in expected.dtypes.items()} == {
        "score": "float64", "count": "int64", "gappy": "float64", "flag": "bool",
        "flags": "object", "label": "category", "note": "str", "sample": "str",
    }
    _same_frame(got, expected)


def test_a_short_row_is_refused_before_the_store_is_touched(written, temp_dir):
    """F018: a row with too few fields was taken, its missing cells NaN."""
    csv = temp_dir / "cells.csv"
    csv.write_text("cell,a,b\n" + "".join(f"c{i},{i},{i}\n" for i in range(N_OBS - 1)) + f"c{N_OBS - 1},1\n")

    result = runner.invoke(app, ["import", "dataframe", str(written), "obs", str(csv), "--inplace"])

    assert result.exit_code == 1
    assert f"line {N_OBS + 1}: 2 fields where the header has 3" in _out(result)
    assert list(_read(written).obs.columns) == ["n"]


def test_the_export_of_a_dataframe_imports_back_as_pandas_reads_it(temp_dir, backend):
    """F018: nullable integers and categoricals with gaps are exported as
    empty fields, and came back as columns of strings."""
    obs = pd.DataFrame(
        {
            "real": [0.5, np.nan, 1.5, 2.5],
            "whole": pd.array([1, None, 3, 4], dtype="Int64"),
            "flag": [True, False, True, True],
            "maybe": pd.array([True, None, False, True], dtype="boolean"),
            "kind": pd.Categorical(["a", None, "b", "a"]),
            "name": ["w", "x", "y", "z"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    obj = ad.AnnData(obs=obs)
    path = backend_path(temp_dir, backend, "frames")
    obj.write_h5ad(path) if backend == "h5ad" else obj.write_zarr(path)
    csv, out = temp_dir / "obs.csv", temp_dir / ("back" + path.suffix)

    exported = runner.invoke(app, ["export", "dataframe", str(path), "obs", "-o", str(csv)])
    assert exported.exit_code == 0, _out(exported)
    result = runner.invoke(app, ["import", "dataframe", str(path), "obs", str(csv), "-o", str(out), "-i", "_index"])

    assert result.exit_code == 0, _out(result)
    got = _read(out).obs
    _same_frame(got, _pandas_frame(csv, "_index"))
    assert str(got["whole"].dtype) == "float64" and got["whole"].isna().tolist() == [False, True, False, False]
    assert got["kind"].isna().tolist() == [False, True, False, False]


def test_categorical_overrides_still_apply(written, temp_dir):
    """--categorical makes a text column a categorical whatever its values,
    and --no-auto-categorical keeps the others as strings."""
    csv = temp_dir / "cells.csv"
    csv.write_text(CELLS, encoding="utf-8")

    forced, out = _import("dataframe", written, "obs", csv, "-i", "cell", "--categorical", "note,count")
    assert forced.exit_code == 0, _out(forced)
    assert "'count' holds int64 values, so it stays as it is" in _out(forced)
    got = _read(out).obs
    assert got["note"].dtype == "category" and got["count"].dtype == "int64"

    plain, out = _import("dataframe", written, "obs", csv, "-i", "cell", "--no-auto-categorical")
    assert plain.exit_code == 0, _out(plain)
    got = _read(out).obs
    assert not any(isinstance(t, pd.CategoricalDtype) for t in got.dtypes)
    assert got["label"].tolist() == ["A", "B", "A", "C", "B", "A"]

    unknown, _ = _import("dataframe", written, "obs", csv, "-i", "cell", "--categorical", "nope")
    assert unknown.exit_code == 1
    assert "--categorical names 'nope', which cells.csv does not have" in _out(unknown)


def test_a_dataframe_import_holds_a_block_not_the_file(new_store, temp_dir, monkeypatch):
    """F034: the whole CSV was read into a list of dicts and then a list per
    column, so memory grew with the file. Eight times the rows must not
    take much more memory."""
    import tracemalloc

    from adata.formats import dataframe, table

    def peak(n: int) -> int:
        path, opener = new_store(f"s{n}")
        with opener("a") as root:
            del root["obs"]
            from adata.elements.write import write_dataframe_header

            write_dataframe_header(root, "obs", [f"c{i}" for i in range(n)], [])
        csv = temp_dir / f"t{n}.csv"
        csv.write_text("id,a,b,c\n" + "".join(f"c{i},{i % 97},{i % 89 / 4},l{i % 7}\n" for i in range(n)))
        real = table.scan_table
        monkeypatch.setattr(
            dataframe, "scan_table", lambda *a, **k: real(*a, block_bytes=1 << 16, **k)
        )
        real_write = table.write_table
        monkeypatch.setattr(
            dataframe, "write_table", lambda *a, **k: real_write(*a, block_bytes=1 << 16, **k)
        )
        with opener("a") as root:
            tracemalloc.start()
            try:
                dataframe.import_dataframe(root, "obs", csv, "id", QUIET)
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

    small, large = peak(5_000), peak(40_000)
    assert large < 2 * small, f"{small:,} bytes at 5,000 rows, {large:,} at 40,000"


@pytest.mark.parametrize(
    "kind,entry,source",
    [
        ("array", "obs", "rows.npy"),
        ("array", "var", "genes.npy"),
        ("array", "obsm", "rows.npy"),
        ("array", "layers", "matrix.npy"),
        ("array", "uns", "rows.npy"),
        ("array", "raw", "matrix.npy"),
        ("array", "raw/var", "genes.npy"),
        ("sparse", "obs", "matrix.mtx"),
        ("sparse", "obsm", "matrix.mtx"),
        ("sparse", "layers", "matrix.mtx"),
        ("sparse", "obs/col", "matrix.mtx"),
        ("dict", "obs", "params.json"),
        ("dict", "obsm", "params.json"),
        ("dict", "X", "params.json"),
        ("dict", "layers/counts", "params.json"),
        ("dataframe", "obsm", "rows.csv"),
        ("dataframe", "X", "rows.csv"),
        ("dataframe", "uns", "rows.csv"),
    ],
)
def test_an_element_of_the_wrong_kind_is_refused_where_the_layout_fixes_one(
    written, temp_dir, kind, entry, source
):
    """F019: `import array data obs a.npy` (and sparse and dict, into obs,
    var, obsm or layers) exited 0, and anndata could then not read the
    store at all; with --inplace that destroyed the source's obs."""
    np.save(temp_dir / "rows.npy", np.ones((N_OBS, 2)))
    np.save(temp_dir / "genes.npy", np.ones((3, 2)))
    np.save(temp_dir / "matrix.npy", np.ones((N_OBS, 3)))
    scipy_io.mmwrite(temp_dir / "matrix.mtx", sparse.csr_matrix(np.ones((N_OBS, 3))))
    (temp_dir / "params.json").write_text('{"a": 1}')
    (temp_dir / "rows.csv").write_text("id,a\n" + "".join(f"c{i},{i}\n" for i in range(N_OBS)))
    before = _read(written)

    result = runner.invoke(app, ["import", kind, str(written), entry, str(temp_dir / source), "--inplace"])

    assert result.exit_code == 1, _out(result)
    assert "would leave a store anndata cannot read" in _out(result) or "is a group of elements" in _out(result)
    after = _read(written)
    assert list(after.obs.columns) == list(before.obs.columns)
    assert set(after.obsm) == set(before.obsm) and set(after.layers) == set(before.layers)


def test_an_entry_inside_a_mapping_is_still_replaced_with_any_kind(written, temp_dir):
    """Naming the entry is how to replace it: a dataframe in obsm with an
    array, or a dict in uns with an array."""
    np.save(temp_dir / "rows.npy", np.full((N_OBS, 2), 7.0))
    result, out = _import("array", written, "obsm/X_pca", temp_dir / "rows.npy")
    assert result.exit_code == 0, _out(result)
    np.testing.assert_array_equal(_read(out).obsm["X_pca"], np.full((N_OBS, 2), 7.0))

    result, out = _import("array", written, "uns/keep", temp_dir / "rows.npy")
    assert result.exit_code == 0, _out(result)
    np.testing.assert_array_equal(_read(out).uns["keep"], np.full((N_OBS, 2), 7.0))


@pytest.mark.parametrize(
    "array",
    [
        np.arange(N_OBS * 4, dtype="int16").reshape(N_OBS, 4),
        np.linspace(0, 1, N_OBS * 2).reshape(N_OBS, 2).astype("float32"),
        np.asfortranarray(np.arange(N_OBS * 3, dtype="float64").reshape(N_OBS, 3)),
        np.array([[f"label {i}", f"é{i}"] for i in range(N_OBS)]),
    ],
    ids=["int16", "float32", "fortran", "text"],
)
def test_an_imported_array_reads_back_as_numpy_saved_it(written, temp_dir, array):
    source = temp_dir / "a.npy"
    np.save(source, array)

    result, out = _import("array", written, "obsm/new", source)

    assert result.exit_code == 0, _out(result)
    got = np.asarray(_read(out).obsm["new"])
    np.testing.assert_array_equal(got, np.load(source))
    if array.dtype.kind != "U":
        assert got.dtype == array.dtype


def test_an_array_import_holds_a_block_not_the_array(new_store, temp_dir):
    """F034: `np.load` read the whole .npy (40 GB for a large X) before
    writing it in one call."""
    import tracemalloc

    from adata.formats.array import import_npy

    path, opener = new_store()
    source = temp_dir / "big.npy"
    np.save(source, np.arange(1 << 20, dtype="float64").reshape(-1, 8))
    with opener("a") as root:
        tracemalloc.start()
        try:
            import_npy(root, "uns/big", source, QUIET, chunk_elements=1 << 12)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert peak < (1 << 20) * 8 / 8, f"{peak:,} bytes for an 8 MiB array"
    with opener("r") as root:
        np.testing.assert_array_equal(root["uns"]["big"][...], np.load(source))


def test_an_imported_matrix_reads_back_as_scipy_reads_it(written, temp_dir):
    source = temp_dir / "m.mtx"
    matrix = sparse.random(N_OBS, 3, density=0.5, format="csr", random_state=0, dtype="float64")
    scipy_io.mmwrite(source, matrix)

    result, out = _import("sparse", written, "layers/new", source)

    assert result.exit_code == 0, _out(result)
    got = _read(out).layers["new"]
    expected = scipy_io.mmread(source).tocsr()
    assert sparse.issparse(got) and (got != expected).nnz == 0


def test_a_json_null_is_left_out_so_older_anndata_can_read_the_store(written, temp_dir):
    """F035: a null was written as `encoding-type: null`, which anndata 0.8
    to 0.11 cannot read, so one null in params.json made the whole store
    unreadable to the release that wrote it."""
    source = temp_dir / "params.json"
    source.write_text('{"x": null, "kept": 1, "deep": {"y": null, "z": "s"}}')

    result, out = _import("dict", written, "uns/meta", source)

    assert result.exit_code == 0, _out(result)
    assert "2 null values were left out, as anndata leaves out None: uns/meta/x, uns/meta/deep/y" in _out(result)
    meta = _read(out).uns["meta"]
    assert meta == {"kept": 1, "deep": {"z": "s"}}
    nulls = []
    if out.suffix == ".h5ad":
        with h5py.File(out) as f:
            f.visititems(lambda name, obj: nulls.append(name) if obj.attrs.get("encoding-type") == "null" else None)
    else:
        nulls = [str(p) for p in out.rglob("*") if p.name in ("zarr.json", ".zattrs") and '"null"' in p.read_text()]
    assert nulls == []

    (temp_dir / "null.json").write_text("null")
    refused, _ = _import("dict", written, "uns/meta", temp_dir / "null.json")
    assert refused.exit_code == 1 and "holds only null" in _out(refused)


def test_json_arrays_are_typed_where_their_values_share_a_type(written, temp_dir):
    """F067: a mixed or nullable array was stringified value by value with
    Python's str() (`[1, null]` became ['1', 'None'])."""
    source = temp_dir / "params.json"
    payload = {
        "gaps": [1, None, 3],
        "grid": [[1.5, None], [2, 3]],
        "wide": [2**63, 1],
        "flags": [True, False],
        "mixed": [1, "x"],
        "texts": ["a", None],
        "bools_and_numbers": [True, 1],
        "nulls": [None, None],
    }
    source.write_text(json.dumps(payload))

    result, out = _import("dict", written, "uns/meta", source)

    assert result.exit_code == 0, _out(result)
    meta = _read(out).uns["meta"]
    np.testing.assert_array_equal(meta["gaps"], [1.0, np.nan, 3.0])
    np.testing.assert_array_equal(meta["grid"], [[1.5, np.nan], [2.0, 3.0]])
    assert meta["wide"].dtype == np.uint64 and meta["wide"].tolist() == [2**63, 1]
    assert meta["flags"].dtype == bool
    for name in ("mixed", "texts", "bools_and_numbers", "nulls"):
        assert json.loads(meta[name]) == payload[name], name


@pytest.mark.parametrize(
    "text,message",
    [
        ('{"n": 99999999999999999999999}', "uns/meta/n holds the whole number 99999999999999999999999"),
        ('{"n": [1, -99999999999999999999]}', "which no 64-bit integer holds"),
        ('{"a/b": 1}', "has the key 'a/b', which cannot name an element"),
        ('{"ok": {"": 1}}', "has the key '', which cannot name an element"),
        ('[1, 2]', None),
    ],
)
def test_json_that_no_element_can_hold_is_refused_before_writing(written, temp_dir, text, message):
    """F067: an integer past 64 bits was written as an object array that
    made the store unreadable, and a key holding '/' made nested groups."""
    source = temp_dir / "params.json"
    source.write_text(text)
    entry = "uns" if message is None else "uns/meta"

    result = runner.invoke(app, ["import", "dict", str(written), entry, str(source), "--inplace"])

    assert result.exit_code == 1
    assert (message or "'uns' must stay a mapping") in _out(result)
    assert _read(written).uns == {"keep": {"a": 1}}

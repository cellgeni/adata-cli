"""Tests for info.py and read.py modules."""

import pytest
import h5py
import numpy as np
from adata.info import axis_len, get_axis_group, get_entry_type, format_type_info
from adata.read import decode_str_array, read_categorical_column, col_chunk_as_strings


class TestGetEntryType:
    """Tests for get_entry_type function."""

    def test_get_entry_type_dataframe(self, sample_h5ad_file):
        """Test type detection for dataframe (obs/var)."""
        with h5py.File(sample_h5ad_file, "r") as f:
            info = get_entry_type(f["obs"])
            assert info["type"] == "dataframe"
            assert info["export_as"] == "csv"

    def test_get_entry_type_dense_matrix(self, sample_h5ad_file):
        """Test type detection for dense matrix."""
        with h5py.File(sample_h5ad_file, "r") as f:
            info = get_entry_type(f["X"])
            assert info["type"] == "dense-matrix"
            assert info["export_as"] == "npy"
            assert info["shape"] == (5, 4)

    def test_get_entry_type_sparse_matrix(self, sample_sparse_csr_h5ad):
        """Test type detection for sparse matrix."""
        with h5py.File(sample_sparse_csr_h5ad, "r") as f:
            info = get_entry_type(f["X"])
            assert info["type"] == "sparse-matrix"
            assert info["export_as"] == "mtx"
            assert info["encoding"] == "csr_matrix"

    def test_get_entry_type_dict(self, sample_h5ad_file):
        """Test type detection for dict/group."""
        with h5py.File(sample_h5ad_file, "r") as f:
            info = get_entry_type(f["uns"])
            assert info["type"] == "dict"
            assert info["export_as"] == "json"

    def test_get_entry_type_1d_array(self, temp_dir):
        """Test type detection for 1D array."""
        file_path = temp_dir / "test.h5ad"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("arr", data=np.array([1, 2, 3, 4, 5]))
        with h5py.File(file_path, "r") as f:
            info = get_entry_type(f["arr"])
            assert info["type"] == "array"
            assert info["export_as"] == "npy"

    def test_get_entry_type_scalar(self, temp_dir):
        """Test type detection for scalar."""
        file_path = temp_dir / "test.h5ad"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("scalar", data=42)
        with h5py.File(file_path, "r") as f:
            info = get_entry_type(f["scalar"])
            assert info["type"] == "scalar"
            assert info["export_as"] == "json"


class TestFormatTypeInfo:
    """Tests for format_type_info function."""

    def test_format_type_info_dataframe(self):
        """Test formatting dataframe type info."""
        info = {"type": "dataframe", "export_as": "csv"}
        result = format_type_info(info)
        assert "<dataframe>" in result
        assert "green" in result

    def test_format_type_info_sparse(self):
        """Test formatting sparse matrix type info."""
        info = {"type": "sparse-matrix", "export_as": "mtx"}
        result = format_type_info(info)
        assert "<sparse-matrix>" in result
        assert "magenta" in result

    def test_format_type_info_unknown(self):
        """Test formatting unknown type info."""
        info = {"type": "unknown", "export_as": None}
        result = format_type_info(info)
        assert "<unknown>" in result
        assert "red" in result


class TestAxisLen:
    """Tests for axis_len function."""

    def test_axis_len_obs(self, sample_h5ad_file):
        """Test getting length of obs axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            length = axis_len(f, "obs")
            assert length == 5

    def test_axis_len_var(self, sample_h5ad_file):
        """Test getting length of var axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            length = axis_len(f, "var")
            assert length == 4

    def test_axis_len_nonexistent(self, sample_h5ad_file):
        """Test getting length of non-existent axis raises KeyError."""
        with h5py.File(sample_h5ad_file, "r") as f:
            with pytest.raises(KeyError, match="'nonexistent' not found"):
                axis_len(f, "nonexistent")

    def test_axis_len_not_a_group(self, temp_dir):
        """Test that axis_len raises TypeError when axis is not a group."""
        file_path = temp_dir / "test.h5ad"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("obs", data=np.array([1, 2, 3]))
        with h5py.File(file_path, "r") as f:
            with pytest.raises(TypeError, match="'obs' is not a group"):
                axis_len(f, "obs")

    def test_axis_len_missing_index(self, temp_dir):
        """Test that axis_len raises KeyError when index dataset is missing."""
        file_path = temp_dir / "test.h5ad"
        with h5py.File(file_path, "w") as f:
            f.create_group("obs")
        with h5py.File(file_path, "r") as f:
            with pytest.raises(KeyError, match="Could not find an index"):
                axis_len(f, "obs")


class TestGetAxisGroup:
    """Tests for get_axis_group function."""

    def test_get_axis_group_obs(self, sample_h5ad_file):
        """Test getting obs axis group."""
        with h5py.File(sample_h5ad_file, "r") as f:
            group, length, index_name = get_axis_group(f, "obs")
            assert isinstance(group, h5py.Group)
            assert length == 5
            assert index_name == "obs_names"

    def test_get_axis_group_var(self, sample_h5ad_file):
        """Test getting var axis group."""
        with h5py.File(sample_h5ad_file, "r") as f:
            group, length, index_name = get_axis_group(f, "var")
            assert isinstance(group, h5py.Group)
            assert length == 4
            assert index_name == "var_names"

    def test_get_axis_group_invalid(self, sample_h5ad_file):
        """Test getting invalid axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            with pytest.raises(ValueError, match="axis must be 'obs' or 'var'"):
                get_axis_group(f, "invalid")

    def test_get_axis_group_missing(self, temp_dir):
        """Test getting missing axis."""
        file_path = temp_dir / "empty.h5ad"
        with h5py.File(file_path, "w") as f:
            f.create_group("obs")

        with h5py.File(file_path, "r") as f:
            with pytest.raises(KeyError, match="'var' not found"):
                get_axis_group(f, "var")


class TestDecodeStrArray:
    """Tests for decode_str_array function."""

    def test_decode_bytes_array(self):
        """Test decoding bytes array."""
        arr = np.array([b"hello", b"world"], dtype="S")
        result = decode_str_array(arr)
        assert result.dtype.kind == "U"
        assert list(result) == ["hello", "world"]

    def test_decode_object_array(self):
        """Test decoding object array."""
        arr = np.array(["hello", "world"], dtype=object)
        result = decode_str_array(arr)
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_decode_unicode_array(self):
        """Test decoding unicode array."""
        arr = np.array(["hello", "world"], dtype="U10")
        result = decode_str_array(arr)
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_decode_non_ascii_bytes_array(self):
        """Test decoding non-ASCII UTF-8 bytes without crashing."""
        arr = np.array(
            ["β-cell".encode("utf-8"), "μglia".encode("utf-8")], dtype=object
        )
        result = decode_str_array(arr)
        assert list(result) == ["β-cell", "μglia"]


class TestReadCategoricalColumn:
    """Tests for read_categorical_column function."""

    def test_read_categorical_column(self, sample_categorical_h5ad):
        """Test reading categorical column."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            col_group = f["obs"]["cell_type"]
            cache = {}
            result = read_categorical_column(col_group, 0, 4, cache)
            assert result == ["TypeA", "TypeB", "TypeA", "TypeC"]

    def test_read_categorical_column_slice(self, sample_categorical_h5ad):
        """Test reading categorical column slice."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            col_group = f["obs"]["cell_type"]
            cache = {}
            result = read_categorical_column(col_group, 1, 3, cache)
            assert result == ["TypeB", "TypeA"]

    def test_read_categorical_column_caching(self, sample_categorical_h5ad):
        """Test that categorical column uses cache."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            col_group = f["obs"]["cell_type"]
            cache = {}

            # First call should populate cache
            read_categorical_column(col_group, 0, 2, cache)
            assert len(cache) == 1

            # Second call should reuse cache
            read_categorical_column(col_group, 2, 4, cache)
            assert len(cache) == 1


class TestColChunkAsStrings:
    """Tests for col_chunk_as_strings function."""

    def test_col_chunk_dataset(self, sample_h5ad_file):
        """Test reading dataset column as strings."""
        with h5py.File(sample_h5ad_file, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "cell_type", 0, 3, cache)
            assert result == ["TypeA", "TypeB", "TypeA"]

    def test_col_chunk_numeric(self, sample_h5ad_file):
        """Test reading numeric column as strings."""
        with h5py.File(sample_h5ad_file, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "n_counts", 0, 3, cache)
            assert result == ["100", "200", "150"]

    def test_col_chunk_categorical(self, sample_categorical_h5ad):
        """Test reading categorical column as strings."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "cell_type", 0, 4, cache)
            assert result == ["TypeA", "TypeB", "TypeA", "TypeC"]

    def test_col_chunk_not_found(self, sample_h5ad_file):
        """Test reading non-existent column."""
        with h5py.File(sample_h5ad_file, "r") as f:
            cache = {}
            with pytest.raises(RuntimeError, match="not found in group"):
                col_chunk_as_strings(f["obs"], "nonexistent", 0, 5, cache)

    def test_col_chunk_multiple_categorical_columns_keep_values(self, temp_dir):
        """Test categorical cache does not leak across columns."""
        file_path = temp_dir / "multi_categorical.h5ad"

        with h5py.File(file_path, "w") as f:
            obs = f.create_group("obs")
            obs.attrs["_index"] = "obs_names"
            obs.create_dataset(
                "obs_names", data=np.array(["c1", "c2", "c3"], dtype="S")
            )

            age = obs.create_group("age")
            age.attrs["encoding-type"] = "categorical"
            age.create_dataset("categories", data=np.array(["5.0", "6.0"], dtype="S"))
            age.create_dataset("codes", data=np.array([0, 1, 0], dtype=np.int8))

            cell_type = obs.create_group("cell_type")
            cell_type.attrs["encoding-type"] = "categorical"
            cell_type.create_dataset(
                "categories",
                data=np.array(["Neuron", "β-cell"], dtype=object),
            )
            cell_type.create_dataset("codes", data=np.array([1, 0, 1], dtype=np.int8))

        with h5py.File(file_path, "r") as f:
            cache = {}
            age_values = col_chunk_as_strings(f["obs"], "age", 0, 3, cache)
            cell_type_values = col_chunk_as_strings(f["obs"], "cell_type", 0, 3, cache)

            assert age_values == ["5.0", "6.0", "5.0"]
            assert cell_type_values == ["β-cell", "Neuron", "β-cell"]


class TestLegacyV010Support:
    """Tests for legacy v0.1.0 format support."""

    def test_get_entry_type_legacy_categorical(self, sample_legacy_v010_h5ad):
        """Test type detection for legacy categorical column (v0.1.0)."""
        with h5py.File(sample_legacy_v010_h5ad, "r") as f:
            info = get_entry_type(f["obs"]["cell_type"])
            assert info["type"] == "categorical"
            assert info["version"] == "0.1.0"
            assert "Legacy" in info["details"]

    def test_get_entry_type_legacy_dataframe(self, sample_legacy_v010_h5ad):
        """Test type detection for legacy dataframe (v0.1.0)."""
        with h5py.File(sample_legacy_v010_h5ad, "r") as f:
            info = get_entry_type(f["obs"])
            assert info["type"] == "dataframe"
            assert info["version"] == "0.1.0"
            assert "legacy" in info["details"].lower()

    def test_read_legacy_categorical_column(self, sample_legacy_v010_h5ad):
        """Test reading legacy categorical column."""
        with h5py.File(sample_legacy_v010_h5ad, "r") as f:
            cache = {}
            result = read_categorical_column(
                f["obs"]["cell_type"], 0, 4, cache, f["obs"]
            )
            assert result == ["TypeA", "TypeB", "TypeA", "TypeC"]

    def test_col_chunk_legacy_categorical(self, sample_legacy_v010_h5ad):
        """Test col_chunk_as_strings with legacy categorical column."""
        with h5py.File(sample_legacy_v010_h5ad, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "cell_type", 0, 4, cache)
            assert result == ["TypeA", "TypeB", "TypeA", "TypeC"]

    def test_col_chunk_legacy_numeric(self, sample_legacy_v010_h5ad):
        """Test col_chunk_as_strings with legacy numeric column."""
        with h5py.File(sample_legacy_v010_h5ad, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "n_counts", 0, 4, cache)
            assert result == ["100", "200", "150", "300"]

    def test_legacy_categorical_slice(self, sample_legacy_v010_h5ad):
        """Test reading slice of legacy categorical column."""
        with h5py.File(sample_legacy_v010_h5ad, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "cell_type", 1, 3, cache)
            assert result == ["TypeB", "TypeA"]

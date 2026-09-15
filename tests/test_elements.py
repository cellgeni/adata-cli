"""Unit tests for the element layer, across HDF5, Zarr v2 and Zarr v3.

Everything else in the codebase reads and writes through these functions, so
a gap here shows up as a subtle corruption several layers away. Each test runs
against all three backends, because the two most expensive bugs in this
project's history -- variable-length strings and group-valued indices -- were
both cases where one backend behaved differently from the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from adata.elements import spec
from adata.elements import write as ew
from adata.elements.read import (
    dataframe_columns,
    decode_str_array,
    element_len,
    is_ordered,
    read_categories,
    read_masked_column,
    read_str_all,
    read_str_chunk,
    resolve_index,
)
from adata.elements.strings import (
    as_str_array,
    is_string_dtype,
    is_string_element,
    string_dtype_for,
    target_dtype,
)
from adata.storage import is_dataset, is_group


# ---------------------------------------------------------------------------
# spec


def test_every_encoding_has_a_declared_version():
    """A type missing from the table would be written without a version."""
    names = {
        v
        for k, v in vars(spec).items()
        if k.isupper() and isinstance(v, str) and not k.endswith("_TYPES")
    }
    for name in names:
        if name in spec.CURRENT_VERSION:
            assert spec.CURRENT_VERSION[name].count(".") == 2


def test_decode_attr_normalises_both_backends_spellings():
    assert spec.decode_attr(b"dataframe") == "dataframe"
    assert spec.decode_attr("dataframe") == "dataframe"
    assert spec.decode_attr(np.str_("x")) == "x"
    assert spec.decode_attr(np.True_) is True
    assert spec.decode_attr(None) is None


def test_encoding_of_reports_none_for_untagged_elements():
    class Bare:
        attrs: dict = {}

    assert spec.encoding_of(Bare()) == (None, None)
    assert spec.encoding_type(Bare()) is None


# ---------------------------------------------------------------------------
# strings


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.dtype("S5"), True),
        (np.dtype("<U3"), True),
        (np.dtype(object), True),
        (np.dtypes.StringDType(), True),
        (np.dtype("float32"), False),
        (np.dtype("int64"), False),
        (np.dtype(bool), False),
        (None, False),
        (str, True),
    ],
)
def test_is_string_dtype(dtype, expected):
    assert is_string_dtype(dtype) is expected


def test_target_dtype_maps_text_per_backend_and_passes_others_through():
    assert target_dtype(np.dtype("S5"), "zarr") is str
    assert target_dtype(np.dtype("S5"), "hdf5") == string_dtype_for("hdf5")
    assert target_dtype(np.dtype("float32"), "zarr") == np.dtype("float32")


def test_as_str_array_decodes_and_keeps_shape():
    out = as_str_array(np.array([[b"a", b"bb"], [b"ccc", b"d"]], dtype=object))
    assert out.shape == (2, 2)
    assert out[0, 1] == "bb"


def test_as_str_array_handles_non_ascii():
    assert as_str_array([b"caf\xc3\xa9"])[0] == "café"


# ---------------------------------------------------------------------------
# write -> read, on every backend


def test_string_array_round_trips(new_store):
    path, opener = new_store()
    values = ["alpha", "", "café", "a longer one"]
    with opener("a") as root:
        ew.write_string_array(root["uns"], "s", values)
    with opener("r") as root:
        ds = root["uns"]["s"]
        assert spec.encoding_of(ds) == (spec.STRING_ARRAY, "0.2.0")
        assert read_str_all(ds) == values
        assert is_string_element(ds)


def test_string_array_keeps_multidimensional_shape(new_store):
    path, opener = new_store()
    grid = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with opener("a") as root:
        ew.write_string_array(root["uns"], "grid", grid)
    with opener("r") as root:
        assert root["uns"]["grid"].shape == (2, 2)


@pytest.mark.parametrize("ordered", [True, False])
def test_categorical_round_trips_with_order(new_store, ordered):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_categorical(
            root["uns"], "c", [0, 2, -1, 1], ["x", "y", "z"], ordered=ordered
        )
    with opener("r") as root:
        col = root["uns"]["c"]
        assert spec.encoding_of(col) == (spec.CATEGORICAL, "0.2.0")
        assert is_ordered(col) is ordered
        assert list(read_categories(col)) == ["x", "y", "z"]
        # -1 denotes missing and must render empty, not index from the end.
        assert read_str_chunk(col, 0, 4) == ["x", "z", "", "y"]


def test_categorical_codes_are_tagged_as_arrays(new_store):
    """Untagged codes make anndata fall back to its legacy reader."""
    path, opener = new_store()
    with opener("a") as root:
        ew.write_categorical(root["uns"], "c", [0, 1], ["x", "y"])
    with opener("r") as root:
        assert spec.encoding_type(root["uns"]["c"]["codes"]) == spec.ARRAY
        assert spec.encoding_type(root["uns"]["c"]["categories"]) == spec.STRING_ARRAY


@pytest.mark.parametrize(
    "n_categories,expected",
    [(2, np.int8), (200, np.int16), (40_000, np.int32)],
)
def test_categorical_code_dtype_widens_with_category_count(n_categories, expected):
    from adata.elements.write import _codes_dtype

    assert _codes_dtype(n_categories) is expected


@pytest.mark.parametrize(
    "enc,values,mask",
    [
        (spec.NULLABLE_INTEGER, np.array([1, 2, 3], dtype="int32"), [False, True, False]),
        (spec.NULLABLE_BOOLEAN, np.array([True, False, True]), [False, False, True]),
        (spec.NULLABLE_STRING_ARRAY, ["a", "b", "c"], [True, False, False]),
    ],
)
def test_masked_elements_round_trip(new_store, enc, values, mask):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_masked(root["uns"], "m", values, mask, enc)
    with opener("r") as root:
        col = root["uns"]["m"]
        assert spec.encoding_type(col) == enc
        assert element_len(col) == 3
        rendered = read_masked_column(col, 0, 3, na_repr="NA")
        for i, is_missing in enumerate(mask):
            assert (rendered[i] == "NA") is bool(is_missing)


def test_masked_write_records_na_value(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_masked(
            root["uns"], "m", ["a"], [True], spec.NULLABLE_STRING_ARRAY,
            na_value="NaN",
        )
    with opener("r") as root:
        assert spec.decode_attr(root["uns"]["m"].attrs["na-value"]) == "NaN"


def test_write_masked_rejects_a_non_masked_encoding(new_store):
    path, opener = new_store()
    with opener("a") as root:
        with pytest.raises(ValueError, match="not a masked encoding"):
            ew.write_masked(root["uns"], "m", [1], [False], spec.ARRAY)


@pytest.mark.parametrize(
    "value,enc",
    [
        ("hello", spec.STRING),
        (42, spec.NUMERIC_SCALAR),
        (3.5, spec.NUMERIC_SCALAR),
        (True, spec.NUMERIC_SCALAR),
        (np.float32(1.5), spec.NUMERIC_SCALAR),
    ],
)
def test_scalars_are_written_as_the_right_encoding(new_store, value, enc):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_scalar(root["uns"], "s", value)
    with opener("r") as root:
        ds = root["uns"]["s"]
        assert spec.encoding_type(ds) == enc
        assert ds.shape == ()


def test_null_round_trips(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_null(root["uns"], "n")
    with opener("r") as root:
        assert spec.encoding_type(root["uns"]["n"]) == spec.NULL


def test_sparse_shape_attr_is_written_in_the_backends_own_form(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_sparse(root["uns"], "m", [1.0], [0], [0, 1], (1, 2))
    with opener("r") as root:
        group = root["uns"]["m"]
        assert spec.encoding_type(group) == spec.CSR_MATRIX
        assert [int(d) for d in group.attrs["shape"]] == [1, 2]


def test_write_sparse_rejects_a_non_sparse_encoding(new_store):
    path, opener = new_store()
    with opener("a") as root:
        with pytest.raises(ValueError, match="not a sparse encoding"):
            ew.write_sparse(
                root["uns"], "m", [1.0], [0], [0, 1], (1, 2), spec.ARRAY
            )


def test_mappings_are_tagged_and_reused(new_store):
    path, opener = new_store()
    with opener("a") as root:
        first = ew.write_mapping(root["uns"], "m")
        ew.write_scalar(first, "x", 1)
        again = ew.write_mapping(root["uns"], "m")
        assert "x" in again, "an existing mapping must be reused, not cleared"
    with opener("r") as root:
        assert spec.encoding_of(root["uns"]["m"]) == (spec.DICT, "0.1.0")


def test_write_mapping_replace_clears_existing_contents(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_scalar(ew.write_mapping(root["uns"], "m"), "x", 1)
        ew.write_mapping(root["uns"], "m", replace=True)
    with opener("r") as root:
        assert list(root["uns"]["m"].keys()) == []


def test_skeleton_creates_every_optional_mapping(new_store):
    path, opener = new_store()
    with opener("r") as root:
        for key in ("layers", "obsm", "obsp", "varm", "varp", "uns"):
            assert key in root
            assert spec.encoding_type(root[key]) == spec.DICT


# ---------------------------------------------------------------------------
# read helpers


def test_element_len_handles_every_column_layout(new_store):
    path, opener = new_store()
    with opener("a") as root:
        uns = root["uns"]
        ew.write_string_array(uns, "plain", ["a", "b", "c"])
        ew.write_categorical(uns, "cat", [0, 1, 0], ["x", "y"])
        ew.write_masked(
            uns, "nul", ["a", "b", "c"], [False] * 3, spec.NULLABLE_STRING_ARRAY
        )
    with opener("r") as root:
        for key in ("plain", "cat", "nul"):
            assert element_len(root["uns"][key]) == 3


def test_element_len_rejects_a_scalar(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_scalar(root["uns"], "s", 1)
    with opener("r") as root:
        with pytest.raises(ValueError, match="scalar"):
            element_len(root["uns"]["s"])


def test_element_len_rejects_an_unrecognised_group(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_mapping(root["uns"], "m")
    with opener("r") as root:
        with pytest.raises(TypeError, match="expected 'values' or 'codes'"):
            element_len(root["uns"]["m"])


def test_resolve_index_prefers_the_declared_index(new_store):
    """A store carrying both `_index` and a stale `obs_names` must not guess."""
    path, opener = new_store()
    with opener("a") as root:
        ew.write_string_array(root["obs"], "obs_names", ["wrong", "wrong", "wrong"])
    with opener("r") as root:
        index, name = resolve_index(root["obs"], "obs")
        assert name == "_index"
        assert read_str_all(index) == ["c1", "c2", "c3"]


def test_resolve_index_falls_back_to_the_naming_convention(new_store):
    path, opener = new_store()
    with opener("a") as root:
        group = ew.write_mapping(root["uns"], "frame")
        ew.write_string_array(group, "obs_names", ["a", "b"])
    with opener("r") as root:
        _, name = resolve_index(root["uns"]["frame"], "obs")
        assert name == "obs_names"


def test_resolve_index_reports_what_it_looked_for(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_mapping(root["uns"], "frame")
    with opener("r") as root:
        with pytest.raises(KeyError, match="tried"):
            resolve_index(root["uns"]["frame"], "obs")


def test_dataframe_columns_follows_column_order(new_store):
    path, opener = new_store()
    with opener("a") as root:
        del root["obs"]
        group = ew.write_dataframe_header(
            root, "obs", ["c1", "c2"], ["zebra", "apple"]
        )
        ew.write_dense(group, "apple", [1, 2])
        ew.write_dense(group, "zebra", [3, 4])
    with opener("r") as root:
        # Not alphabetical, which is what the backend would otherwise give.
        assert dataframe_columns(root["obs"], "_index") == ["zebra", "apple"]


def test_dataframe_columns_appends_undeclared_columns(new_store):
    """A column on disk but missing from column-order must not vanish."""
    path, opener = new_store()
    with opener("a") as root:
        del root["obs"]
        group = ew.write_dataframe_header(root, "obs", ["c1", "c2"], ["declared"])
        ew.write_dense(group, "declared", [1, 2])
        ew.write_dense(group, "extra", [5, 6])
    with opener("r") as root:
        assert dataframe_columns(root["obs"], "_index") == ["declared", "extra"]


def test_read_str_chunk_rejects_an_unsupported_group(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_mapping(root["uns"], "m")
    with opener("r") as root:
        with pytest.raises(ValueError, match="Unsupported group encoding"):
            read_str_chunk(root["uns"]["m"], 0, 1)


def test_read_str_chunk_renders_numbers_without_a_string_detour(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_dense(root["uns"], "n", np.array([1, 2, 3], dtype="int64"))
    with opener("r") as root:
        assert read_str_chunk(root["uns"]["n"], 0, 3) == ["1", "2", "3"]


def test_read_str_chunk_reads_a_window(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_string_array(root["uns"], "s", [f"v{i}" for i in range(10)])
    with opener("r") as root:
        assert read_str_chunk(root["uns"]["s"], 3, 6) == ["v3", "v4", "v5"]


def test_read_str_all_spans_chunk_boundaries(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_string_array(root["uns"], "s", [f"v{i}" for i in range(25)])
    with opener("r") as root:
        assert read_str_all(root["uns"]["s"], chunk_size=4) == [
            f"v{i}" for i in range(25)
        ]


def test_read_categories_reports_a_column_with_none(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_dense(root["uns"], "plain", [1, 2])
    with opener("r") as root:
        with pytest.raises(KeyError, match="Cannot find categories"):
            read_categories(root["uns"]["plain"])


@pytest.mark.parametrize(
    "given,expected",
    [
        (np.array([b"a", b"bb"]), ["a", "bb"]),
        (np.array(["a", "bb"], dtype=object), ["a", "bb"]),
        (np.array([1, 2]), ["1", "2"]),
        (np.array([b"caf\xc3\xa9"], dtype=object), ["café"]),
    ],
)
def test_decode_str_array(given, expected):
    assert decode_str_array(given).tolist() == expected


def test_decode_str_array_preserves_shape():
    assert decode_str_array(np.array([[b"a", b"b"]])).shape == (1, 2)


def test_decode_str_array_replaces_undecodable_bytes():
    assert "�" in decode_str_array(np.array([b"\xff\xfe"]))[0]

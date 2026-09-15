"""Writing AnnData elements in the current on-disk spec.

Every element this tool creates is tagged with its `encoding-type` and
`encoding-version` here, so a store the CLI writes is indistinguishable from
one anndata wrote. Text always goes out as variable-length UTF-8 -- fixed-width
bytes are readable by anndata but are not what `string-array` specifies, and on
Zarr v3 they produce a dtype other libraries cannot read.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from adata.elements import spec
from adata.storage import create_dataset, is_zarr_group


def _backend_of(parent: Any) -> str:
    return "zarr" if is_zarr_group(parent) else "hdf5"


def _replace(parent: Any, name: str) -> None:
    """Remove an existing member so it can be rewritten."""
    if name in parent:
        del parent[name]


def write_mapping(parent: Any, name: str, replace: bool = False) -> Any:
    """Create (or fetch) a group tagged as a `dict` mapping."""
    if replace:
        _replace(parent, name)
    group = parent[name] if name in parent else parent.create_group(name)
    spec.set_encoding(group, spec.DICT)
    return group


def write_string_array(
    parent: Any, name: str, values: Iterable[Any], replace: bool = False
) -> Any:
    """Write a `string-array`: variable-length UTF-8 on either backend."""
    if replace:
        _replace(parent, name)
    ds = create_dataset(parent, name, data=np.asarray(list(values), dtype=object))
    spec.set_encoding(ds, spec.STRING_ARRAY)
    return ds


def write_dense(
    parent: Any, name: str, data: Any, replace: bool = False, **kwargs: Any
) -> Any:
    """Write a dense numeric `array`."""
    if replace:
        _replace(parent, name)
    ds = create_dataset(parent, name, data=np.asarray(data), **kwargs)
    spec.set_encoding(ds, spec.ARRAY)
    return ds


def write_scalar(parent: Any, name: str, value: Any, replace: bool = False) -> Any:
    """Write a 0-d scalar as `string` or `numeric-scalar`, per its type."""
    if replace:
        _replace(parent, name)

    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        ds = create_dataset(parent, name, data=np.asarray(text, dtype=object))
        spec.set_encoding(ds, spec.STRING)
        return ds

    arr = np.asarray(value)
    ds = create_dataset(parent, name, data=arr)
    spec.set_encoding(ds, spec.NUMERIC_SCALAR)
    return ds


def write_null(parent: Any, name: str, replace: bool = False) -> Any:
    """Write an explicit null, matching how anndata >= 0.12 encodes `None`.

    HDF5 uses a null dataspace (`h5py.Empty`); Zarr uses a 0-d boolean array.
    Both carry `encoding-type: null`, so `None` round-trips instead of needing
    an invented marker attribute.
    """
    if replace:
        _replace(parent, name)

    if is_zarr_group(parent):
        ds = parent.create_array(name, shape=(), dtype=bool)
    else:
        import h5py

        ds = parent.create_dataset(name, data=h5py.Empty("f4"))
    spec.set_encoding(ds, spec.NULL)
    return ds


def write_categorical(
    parent: Any,
    name: str,
    codes: Any,
    categories: Sequence[Any],
    ordered: bool = False,
    replace: bool = False,
) -> Any:
    """Write a `categorical` group of `categories` + `codes`.

    A code of -1 denotes a missing value, as the spec requires.
    """
    if replace:
        _replace(parent, name)
    group = parent.create_group(name)
    spec.set_encoding(group, spec.CATEGORICAL)
    group.attrs["ordered"] = bool(ordered)

    write_string_array(group, "categories", categories)
    write_dense(group, "codes", np.asarray(codes, dtype=_codes_dtype(len(categories))))
    return group


def _codes_dtype(n_categories: int) -> Any:
    """Smallest signed integer dtype that can index `n_categories` plus -1."""
    if n_categories < 128:
        return np.int8
    if n_categories < 32_768:
        return np.int16
    return np.int32


def write_masked(
    parent: Any,
    name: str,
    values: Any,
    mask: Any,
    enc_type: str,
    na_value: Optional[str] = None,
    replace: bool = False,
) -> Any:
    """Write a nullable element as a `values` + `mask` group.

    A True entry in `mask` marks a missing value.
    """
    if enc_type not in spec.MASKED_TYPES:
        raise ValueError(f"{enc_type!r} is not a masked encoding.")
    if replace:
        _replace(parent, name)

    group = parent.create_group(name)
    spec.set_encoding(group, enc_type)
    if na_value is not None:
        group.attrs["na-value"] = na_value

    if enc_type == spec.NULLABLE_STRING_ARRAY:
        write_string_array(group, "values", values)
    else:
        write_dense(group, "values", values)
    write_dense(group, "mask", np.asarray(mask, dtype=bool))
    return group


def write_sparse(
    parent: Any,
    name: str,
    data: Any,
    indices: Any,
    indptr: Any,
    shape: Sequence[int],
    enc_type: str = spec.CSR_MATRIX,
    replace: bool = False,
) -> Any:
    """Write a CSR/CSC sparse matrix group."""
    if enc_type not in spec.SPARSE_TYPES:
        raise ValueError(f"{enc_type!r} is not a sparse encoding.")
    if replace:
        _replace(parent, name)

    group = parent.create_group(name)
    spec.set_encoding(group, enc_type)
    set_shape_attr(group, shape)

    create_dataset(group, "data", data=np.asarray(data))
    create_dataset(group, "indices", data=np.asarray(indices))
    create_dataset(group, "indptr", data=np.asarray(indptr))
    return group


def set_shape_attr(group: Any, shape: Sequence[int]) -> None:
    """Set a sparse group's `shape`, in the form the backend's attrs accept.

    Zarr attributes must be JSON-serialisable, so a numpy array cannot be
    stored there; HDF5 conventionally holds an integer array.
    """
    dims = [int(d) for d in shape]
    if is_zarr_group(group):
        group.attrs["shape"] = dims
    else:
        group.attrs["shape"] = np.array(dims, dtype=np.int64)


def write_dataframe_header(
    parent: Any,
    name: str,
    index_values: Iterable[Any],
    column_order: Sequence[str],
    index_name: str = "_index",
    replace: bool = True,
) -> Any:
    """Create a `dataframe` group with its index and declared column order.

    Columns themselves are written afterwards by the caller; `column-order`
    records the authored order so readers do not fall back to the backend's
    own (alphabetical, on HDF5) enumeration.
    """
    if replace:
        _replace(parent, name)

    group = parent.create_group(name)
    spec.set_encoding(group, spec.DATAFRAME)
    group.attrs["_index"] = index_name
    set_column_order(group, column_order)
    write_string_array(group, index_name, index_values)
    return group


def set_column_order(group: Any, columns: Sequence[str]) -> None:
    """Record a dataframe's column order in the form the backend accepts.

    Zarr attributes must be JSON, so a plain list is used. HDF5 needs an
    explicit variable-length UTF-8 dtype -- an object array has no native
    HDF5 equivalent, and an empty one cannot be inferred at all.
    """
    names = [str(c) for c in columns]
    if is_zarr_group(group):
        group.attrs["column-order"] = names
    else:
        import h5py

        group.attrs["column-order"] = np.array(
            names, dtype=h5py.string_dtype(encoding="utf-8")
        )


def ensure_anndata_skeleton(root: Any) -> None:
    """Create the optional mapping groups an AnnData store is expected to have."""
    for key in ("layers", "obsm", "obsp", "varm", "varp", "uns"):
        write_mapping(root, key)

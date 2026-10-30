"""Backend-neutral handling of string elements.

The AnnData spec requires `string-array` to be variable-length UTF-8: a vlen
str dtype in HDF5, `VariableLengthUTF8` in Zarr. Neither backend accepts the
other's spelling, and a dtype round-tripped verbatim across backends fails
outright -- an h5py vlen dataset reports `dtype == object`, which Zarr rejects
as ambiguous. Everything that creates or copies a string element therefore
resolves the dtype through this module rather than reusing the source's.
"""

from __future__ import annotations

from typing import Any, Optional

import h5py
import numpy as np

from adata.storage import is_dataset, is_zarr_array


def is_string_dtype(dtype: Any) -> bool:
    """True if `dtype` holds text, in any spelling either backend produces.

    Covers h5py vlen-UTF-8 (object), fixed-width bytes ('S'), numpy unicode
    ('U'), numpy StringDType ('T', what zarr-python 3 reports for both
    `VariableLengthUTF8` and v2 `VLenUTF8`), and bare object arrays.

    Not h5py's other `object` dtypes, variable-length numbers and object
    references: taken for text, they were copied as the strings '[1 2 3]'.
    """
    if dtype is None:
        return False
    if dtype is str:
        return True
    try:
        if h5py.check_string_dtype(dtype) is not None:
            return True
        if (
            h5py.check_vlen_dtype(dtype) is not None
            or h5py.check_ref_dtype(dtype) is not None
        ):
            return False
    except (TypeError, AttributeError):
        pass
    kind = getattr(dtype, "kind", None)
    return kind in ("O", "S", "U", "T")


def is_string_element(obj: Any) -> bool:
    """True if `obj` is a dataset/array holding text."""
    return is_dataset(obj) and is_string_dtype(getattr(obj, "dtype", None))


def string_dtype_for(backend: str, zarr_format: Optional[int] = None) -> Any:
    """Return the spec-compliant variable-length UTF-8 dtype for `backend`.

    `zarr_format` is accepted for symmetry; zarr-python 3 maps `str` to
    `VariableLengthUTF8` under both v2 and v3, so it does not currently affect
    the result.
    """
    if backend == "zarr":
        return str
    return h5py.string_dtype(encoding="utf-8")


def target_dtype(src_dtype: Any, target_backend: str, zarr_format: Optional[int] = None) -> Any:
    """Map a source dtype onto one the target backend can actually create.

    String dtypes are normalised to variable-length UTF-8; everything else is
    passed through unchanged.
    """
    if is_string_dtype(src_dtype):
        return string_dtype_for(target_backend, zarr_format)
    return src_dtype


def as_str_array(values: Any) -> np.ndarray:
    """Coerce a sequence to a 1-D numpy array of Python str, decoding bytes."""
    arr = np.asarray(values, dtype=object)
    flat = [
        v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in arr.reshape(-1)
    ]
    return np.asarray(flat, dtype=object).reshape(arr.shape)


def text_field_widths(records: np.ndarray) -> dict:
    """The longest value, in characters, of each variable-length text field."""
    widths = {}
    for field in records.dtype.names or ():
        sub = records.dtype.fields[field][0]
        if is_string_dtype(sub) and sub.kind in ("O", "T"):
            values = as_str_array(records[field]).reshape(-1)
            widths[field] = max((len(v) for v in values), default=0)
    return widths


def record_dtype(src_dtype: Any, target_backend: str, widths: Optional[dict] = None) -> np.dtype:
    """A record dtype the target can create, field by field, as anndata
    writes a `rec-array`: text as variable-length str on HDF5, fixed-width
    `<U` on Zarr (`widths` gives the fields of variable length, measured).
    Fixed-width bytes and every other field are kept."""
    fields = []
    for field in src_dtype.names:
        sub = src_dtype.fields[field][0]
        if not is_string_dtype(sub) or sub.kind == "S":
            fields.append((field, sub))
        elif target_backend == "hdf5":
            fields.append((field, string_dtype_for("hdf5")))
        elif sub.kind == "U":
            fields.append((field, sub))
        else:
            fields.append((field, f"<U{max(1, (widths or {}).get(field, 1))}"))
    return np.dtype(fields)


def as_records(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Records read from one backend, respelt in `dtype` (`record_dtype`)."""
    values = np.asarray(values)
    out = np.empty(values.shape, dtype=dtype)
    for field in dtype.names:
        column = values[field]
        target = dtype.fields[field][0]
        if is_string_dtype(target) and target.kind != column.dtype.kind or target.kind == "O":
            column = as_str_array(column)
        out[field] = column
    return out

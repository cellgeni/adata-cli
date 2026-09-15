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
    """
    if dtype is None:
        return False
    if dtype is str:
        return True
    try:
        if h5py.check_string_dtype(dtype) is not None:
            return True
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


def strip_string_filters(kwargs: dict, target_backend: str, zarr_format: Optional[int]) -> dict:
    """Drop codecs that cannot survive the jump to `target_backend`.

    A Zarr v2 string array carries `VLenUTF8` in `filters`; forwarding it to a
    v3 array raises `TypeError: Expected an ArrayArrayCodec`. The v3 string
    dtype encodes variable length itself, so the filter is simply dropped.
    """
    if target_backend != "zarr" or zarr_format != 3:
        return kwargs
    out = dict(kwargs)
    out.pop("filters", None)
    return out

"""Reading AnnData elements, in every layout the spec has ever used.

Elements that logically hold one column of text are stored four different ways
depending on which anndata wrote the file:

* a plain dataset of bytes or vlen str (`string-array`, or untagged in 0.7.x),
* a group of `categories` + `codes` (`categorical`),
* a dataset of codes plus a `categories` attribute (legacy 0.7.x categorical),
* a group of `values` + `mask` (`nullable-string-array`, what anndata >= 0.11
  writes for every string column *including the dataframe index*).

The last case is why this module exists: code that assumed an index was always
a dataset fails outright on any recent file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adata.elements import spec
from adata.storage import is_dataset, is_group, is_hdf5_dataset


def decode_str_array(array: np.ndarray) -> np.ndarray:
    """Decode an array of bytes/objects to a numpy array of str."""
    arr = np.asarray(array)

    if np.issubdtype(arr.dtype, np.bytes_):
        flat = arr.reshape(-1)
        decoded = [
            (
                b.decode("utf-8", errors="replace")
                if isinstance(b, (bytes, np.bytes_))
                else str(b)
            )
            for b in flat
        ]
        return np.asarray(decoded, dtype=str).reshape(arr.shape)

    # "T" is numpy's StringDType, which zarr-python 3 reports for both
    # VariableLengthUTF8 (v3) and VLenUTF8 (v2); it refuses a direct astype(str).
    if arr.dtype.kind in ("O", "T"):
        flat = arr.reshape(-1)
        decoded = [
            (
                v.decode("utf-8", errors="replace")
                if isinstance(v, (bytes, np.bytes_))
                else str(v)
            )
            for v in flat
        ]
        return np.asarray(decoded, dtype=str).reshape(arr.shape)

    return arr.astype(str)


def element_len(obj: Any) -> int:
    """Length along the first axis, whether `obj` is a dataset or a group.

    Handles the masked (`values`/`mask`) and categorical group layouts, so
    callers never need to know how a column happens to be stored.
    """
    if is_dataset(obj):
        shape = getattr(obj, "shape", None)
        if not shape:
            raise ValueError("Element is a scalar and has no length.")
        return int(shape[0])

    if is_group(obj):
        enc = spec.encoding_type(obj)
        if enc in spec.MASKED_TYPES or "values" in obj:
            return int(obj["values"].shape[0])
        if enc == spec.CATEGORICAL or "codes" in obj:
            return int(obj["codes"].shape[0])
        raise TypeError(
            f"Cannot determine length of group with encoding {enc!r}: "
            "expected 'values' or 'codes'."
        )

    raise TypeError(f"Unsupported element type: {type(obj)}")


def resolve_index(group: Any, axis: Optional[str] = None) -> Tuple[Any, str]:
    """Return the ``(index_element, index_name)`` of a dataframe group.

    Honours the declared ``_index`` attribute first and only then falls back to
    the ``obs_names``/``var_names`` convention -- the reverse of that order
    picks the wrong column on a file carrying both.
    """
    index_name = spec.decode_attr(group.attrs.get("_index", None))

    candidates: List[str] = []
    if index_name:
        candidates.append(str(index_name))
    if axis == "obs":
        candidates.append("obs_names")
    elif axis == "var":
        candidates.append("var_names")
    else:
        candidates.extend(("obs_names", "var_names"))
    candidates.append("_index")

    for name in candidates:
        if name in group:
            return group[name], name

    raise KeyError(
        f"Could not find an index in group {getattr(group, 'name', '?')!r}; "
        f"tried {candidates}."
    )


def dataframe_columns(group: Any, index_name: Optional[str] = None) -> List[str]:
    """Return a dataframe group's data columns in their authored order.

    The spec records original column order in the `column-order` attribute;
    without it the order would be whatever the backend enumerates, which is
    alphabetical on HDF5. Columns present on disk but absent from the
    attribute are appended so nothing is ever dropped.
    """
    if index_name is None:
        try:
            _, index_name = resolve_index(group)
        except KeyError:
            index_name = None

    skip = set(spec.RESERVED_DATAFRAME_KEYS)
    if index_name:
        skip.add(index_name)

    present = [k for k in group.keys() if k not in skip]

    declared = group.attrs.get("column-order", None)
    if declared is None:
        return present

    ordered = [
        str(spec.decode_attr(c))
        for c in np.asarray(declared).reshape(-1).tolist()
    ]
    seen = set()
    out = [c for c in ordered if c in present and not (c in seen or seen.add(c))]
    out.extend(c for c in present if c not in seen)
    return out


def _categorical_cache_key(col: Any, parent_group: Any | None = None) -> str:
    col_name = getattr(col, "name", None)
    if isinstance(col_name, str) and col_name:
        return col_name

    if parent_group is not None:
        parent_name = getattr(parent_group, "name", "")
        rel_name = getattr(col, "path", "")
        if parent_name or rel_name:
            return f"{parent_name}/{rel_name}"

    return repr(col)


def read_categories(col: Any, parent_group: Any | None = None) -> np.ndarray:
    """Read the category labels of a categorical column, modern or legacy."""
    if is_group(col):
        return np.asarray(decode_str_array(col["categories"][...]), dtype=str)

    cats_ref = col.attrs.get("categories", None)
    if cats_ref is not None and is_hdf5_dataset(col):
        # Legacy 0.7.x HDF5: an object reference into __categories.
        return np.asarray(decode_str_array(col.file[cats_ref][...]), dtype=str)

    if parent_group is not None and "__categories" in parent_group:
        col_name = str(getattr(col, "name", "")).split("/")[-1]
        cats_grp = parent_group["__categories"]
        if col_name in cats_grp:
            return np.asarray(decode_str_array(cats_grp[col_name][...]), dtype=str)

    if cats_ref is not None and isinstance(cats_ref, (str, bytes)):
        # Legacy Zarr: `categories` is an absolute path within the store.
        path = spec.decode_attr(cats_ref).lstrip("/")
        root = getattr(col, "store_path", None)
        if parent_group is not None and path.split("/")[-1] in parent_group:
            return np.asarray(
                decode_str_array(parent_group[path.split("/")[-1]][...]), dtype=str
            )
        del root

    raise KeyError(
        f"Cannot find categories for categorical column "
        f"{getattr(col, 'name', '?')!r}."
    )


def is_ordered(col: Any) -> bool:
    """Whether a categorical column declares an order over its categories."""
    return bool(spec.decode_attr(col.attrs.get("ordered", False)))


def read_categorical_column(
    col: Any,
    start: int,
    end: int,
    cache: Dict[str, np.ndarray],
    parent_group: Any | None = None,
) -> List[str]:
    """Read rows [start, end) of a categorical column as strings.

    Codes outside the category range -- notably -1 -- render as empty, which is
    how the spec denotes a missing value.
    """
    key = _categorical_cache_key(col, parent_group)
    if key not in cache:
        cache[key] = read_categories(col, parent_group)
    cats = cache[key]

    codes_ds = col["codes"] if is_group(col) else col
    codes = np.asarray(codes_ds[start:end], dtype=np.int64)
    return [cats[c] if 0 <= c < len(cats) else "" for c in codes]


def _format_values(values: np.ndarray) -> List[str]:
    """Render a numeric or text array as strings without a lossy detour."""
    arr = np.asarray(values)
    if arr.dtype.kind in ("O", "S", "U", "T"):
        return decode_str_array(arr).tolist()
    return [str(v) for v in arr.tolist()]


def read_masked_column(
    col: Any, start: int, end: int, na_repr: str = ""
) -> List[str]:
    """Read rows [start, end) of a `values`/`mask` group as strings.

    A True mask marks a missing entry, rendered as `na_repr`.
    """
    values = col["values"][start:end]
    mask = np.asarray(col["mask"][start:end], dtype=bool)
    rendered = _format_values(values)
    return [na_repr if m else v for v, m in zip(rendered, mask)]


def read_str_chunk(
    obj: Any,
    start: int,
    end: int,
    cache: Optional[Dict[str, np.ndarray]] = None,
    parent_group: Any | None = None,
    na_repr: str = "",
) -> List[str]:
    """Read rows [start, end) of any column-like element as strings.

    This is the one entry point for turning a column into text, whatever
    layout it uses on disk.
    """
    if cache is None:
        cache = {}

    if is_dataset(obj):
        if "categories" in obj.attrs:
            return read_categorical_column(obj, start, end, cache, parent_group)
        chunk = obj[start:end]
        arr = np.asarray(chunk)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return _format_values(arr)

    if is_group(obj):
        enc = spec.encoding_type(obj)

        if enc == spec.CATEGORICAL or (enc is None and "codes" in obj):
            return read_categorical_column(obj, start, end, cache, parent_group)

        if enc in spec.MASKED_TYPES or (enc is None and "values" in obj and "mask" in obj):
            return read_masked_column(obj, start, end, na_repr=na_repr)

        raise ValueError(
            f"Unsupported group encoding {enc!r} for element "
            f"{getattr(obj, 'name', '?')!r}."
        )

    raise TypeError(f"Unsupported element type: {type(obj)}")


def read_str_all(obj: Any, chunk_size: int = 200_000, **kwargs: Any) -> List[str]:
    """Read an entire column-like element as strings, in chunks."""
    n = element_len(obj)
    cache: Dict[str, np.ndarray] = {}
    out: List[str] = []
    for start in range(0, n, chunk_size):
        out.extend(
            read_str_chunk(obj, start, min(start + chunk_size, n), cache, **kwargs)
        )
    return out


def col_chunk_as_strings(
    group: Any,
    col_name: str,
    start: int,
    end: int,
    cat_cache: Dict[str, np.ndarray],
) -> List[str]:
    """Read rows [start, end) of a named column within a dataframe group."""
    if col_name not in group:
        raise RuntimeError(
            f"Column {col_name!r} not found in group "
            f"{getattr(group, 'name', '?')!r}"
        )
    return read_str_chunk(group[col_name], start, end, cat_cache, parent_group=group)

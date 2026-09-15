from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union

import numpy as np

from adata.elements import spec
from adata.elements.read import dataframe_columns, element_len, resolve_index
from adata.storage import is_dataset, is_group, is_hdf5_dataset


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _base_result() -> Dict[str, Any]:
    return {
        "type": "unknown",
        "export_as": None,
        "encoding": None,
        "shape": None,
        "dtype": None,
        "details": "",
        "version": None,
        "ordered": None,
    }


#: Declared encoding-type -> (reported type, suggested export format).
_ENCODING_TYPES = {
    spec.ANNDATA: ("anndata", None),
    spec.RAW: ("raw", None),
    spec.DICT: ("dict", "json"),
    spec.DATAFRAME: ("dataframe", "csv"),
    spec.ARRAY: ("array", "npy"),
    spec.STRING_ARRAY: ("string-array", "csv"),
    spec.CATEGORICAL: ("categorical", "csv"),
    spec.CSR_MATRIX: ("sparse-matrix", "mtx"),
    spec.CSC_MATRIX: ("sparse-matrix", "mtx"),
    spec.NULLABLE_INTEGER: ("nullable-array", "csv"),
    spec.NULLABLE_BOOLEAN: ("nullable-array", "csv"),
    spec.NULLABLE_STRING_ARRAY: ("nullable-array", "csv"),
    spec.NUMERIC_SCALAR: ("scalar", "json"),
    spec.STRING: ("scalar", "json"),
    spec.NULL: ("null", "json"),
    spec.AWKWARD_ARRAY: ("awkward-array", "json"),
}


def get_entry_type(entry: Any) -> Dict[str, Any]:
    """Describe an element: its type, shape, and how it can be exported.

    The declared `encoding-type` is authoritative and is consulted first;
    structural inspection is only a fallback, for files written by anndata
    0.7.x and earlier which carry no encoding attributes. Inferring first
    would let a group merely *containing* a member called `obs_names` be
    misreported as a dataframe.
    """
    result = _base_result()

    enc, enc_ver = spec.encoding_of(entry)
    result["encoding"] = enc
    result["version"] = enc_ver

    if is_dataset(entry):
        result["shape"] = entry.shape
        result["dtype"] = str(entry.dtype)

    if enc in _ENCODING_TYPES:
        result["type"], result["export_as"] = _ENCODING_TYPES[enc]
        _describe(entry, result, enc)
        return result

    return _infer_untagged(entry, result)


def _describe(entry: Any, result: Dict[str, Any], enc: str) -> None:
    """Fill in the human-readable detail line for a tagged element."""
    if enc in spec.SPARSE_TYPES:
        shape = entry.attrs.get("shape", None)
        shape_str = f"{int(shape[0])}x{int(shape[1])}" if shape is not None else "?"
        kind = enc.replace("_matrix", "").upper()
        result["details"] = f"Sparse {kind} matrix {shape_str}"
        return

    if enc == spec.CATEGORICAL:
        result["ordered"] = bool(spec.decode_attr(entry.attrs.get("ordered", False)))
        n_codes = _safe_len(entry.get("codes"))
        n_cats = _safe_len(entry.get("categories"))
        order = ", ordered" if result["ordered"] else ""
        result["details"] = f"Categorical [{n_codes} values, {n_cats} categories{order}]"
        return

    if enc == spec.DATAFRAME:
        cols = dataframe_columns(entry)
        legacy = " (legacy __categories)" if "__categories" in entry else ""
        result["details"] = f"DataFrame with {len(cols)} columns{legacy}"
        return

    if enc in spec.MASKED_TYPES:
        n = _safe_len(entry.get("values"))
        na = spec.decode_attr(entry.attrs.get("na-value", None))
        na_str = f", na-value={na}" if na is not None else ""
        result["details"] = f"Nullable array [{n} values] ({enc}{na_str})"
        return

    if enc == spec.STRING:
        result["details"] = "String scalar"
        return

    if enc == spec.NUMERIC_SCALAR:
        result["details"] = f"Numeric scalar ({entry.dtype})"
        return

    if enc == spec.NULL:
        result["details"] = "Null value"
        return

    if enc == spec.AWKWARD_ARRAY:
        result["details"] = f"Awkward array (length={entry.attrs.get('length', '?')})"
        return

    if enc == spec.STRING_ARRAY:
        n = entry.shape[0] if getattr(entry, "shape", None) else "?"
        result["details"] = f"String array [{n}]"
        return

    if enc in spec.MAPPING_TYPES:
        result["details"] = f"Group with {len(list(entry.keys()))} keys"
        return

    if enc == spec.ARRAY:
        result["details"] = _array_details(entry)


def _safe_len(obj: Any) -> Any:
    if obj is None:
        return "?"
    shape = getattr(obj, "shape", None)
    return shape[0] if shape else "?"


def _array_details(entry: Any) -> str:
    if entry.shape == ():
        return f"Scalar value ({entry.dtype})"
    if entry.ndim == 1:
        return f"1D array [{entry.shape[0]}] ({entry.dtype})"
    if entry.ndim == 2:
        return f"Dense matrix {entry.shape[0]}x{entry.shape[1]} ({entry.dtype})"
    return f"{entry.ndim}D array {entry.shape} ({entry.dtype})"


def _infer_untagged(entry: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """Classify an element that carries no encoding attributes (anndata <= 0.7)."""
    if is_dataset(entry):
        if "categories" in entry.attrs:
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["version"] = result["version"] or "0.1.0"
            n_cats = "?"
            if is_hdf5_dataset(entry):
                try:
                    n_cats = entry.file[entry.attrs["categories"]].shape[0]
                except Exception:
                    n_cats = "?"
            result["details"] = (
                f"Legacy categorical [{entry.shape[0]} values, {n_cats} categories]"
            )
            return result

        if entry.shape == ():
            result["type"] = "scalar"
            result["export_as"] = "json"
            result["details"] = f"Scalar value ({entry.dtype})"
            return result

        result["type"] = "dense-matrix" if entry.ndim == 2 else "array"
        result["export_as"] = "npy"
        result["details"] = _array_details(entry)
        return result

    if is_group(entry):
        if "codes" in entry and "categories" in entry:
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["details"] = (
                f"Categorical [{_safe_len(entry.get('codes'))} values, "
                f"{_safe_len(entry.get('categories'))} categories]"
            )
            return result

        if "values" in entry and "mask" in entry:
            result["type"] = "nullable-array"
            result["export_as"] = "csv"
            result["details"] = f"Nullable array [{_safe_len(entry.get('values'))} values]"
            return result

        if "_index" in entry.attrs or "obs_names" in entry or "var_names" in entry:
            result["type"] = "dataframe"
            result["export_as"] = "csv"
            result["version"] = result["version"] or "0.1.0"
            legacy = " (legacy v0.1.0)" if "__categories" in entry else ""
            result["details"] = (
                f"DataFrame with {len(dataframe_columns(entry))} columns{legacy}"
            )
            return result

        result["type"] = "dict"
        result["export_as"] = "json"
        result["details"] = f"Group with {len(list(entry.keys()))} keys"
        return result

    return result


def format_type_info(info: Dict[str, Any]) -> str:
    type_colors = {
        "anndata": "cyan",
        "raw": "cyan",
        "nullable-array": "blue",
        "string-array": "green",
        "null": "dim",
        "awkward-array": "magenta",
        "dataframe": "green",
        "sparse-matrix": "magenta",
        "dense-matrix": "blue",
        "array": "blue",
        "dict": "yellow",
        "categorical": "green",
        "scalar": "white",
        "unknown": "red",
    }

    color = type_colors.get(info["type"], "white")
    return f"[{color}]<{info['type']}>[/]"


def axis_len(file: Any, axis: str) -> int:
    """Number of rows along `axis` ("obs" or "var").

    Resolves the index through :func:`resolve_index`, so this works whether the
    index is a plain dataset or -- as anndata >= 0.11 writes it -- a
    `nullable-string-array` group of `values` and `mask`.
    """
    if axis not in file:
        raise KeyError(f"'{axis}' not found in the file.")

    group = file[axis]
    if not is_group(group):
        raise TypeError(f"'{axis}' is not a group.")

    if axis not in ("obs", "var"):
        raise ValueError(f"Invalid axis '{axis}'. Must be 'obs' or 'var'.")

    index, index_name = resolve_index(group, axis)
    try:
        return element_len(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot determine length of '{axis}': index {index_name!r} "
            f"has no usable length ({exc})."
        ) from exc


def get_axis_group(file: Any, axis: str) -> Tuple[Any, int, str]:
    """Return the ``(group, length, index_name)`` triple for an axis."""
    if axis not in ("obs", "var"):
        raise ValueError("axis must be 'obs' or 'var'.")
    if axis not in file:
        raise KeyError(f"'{axis}' not found in the file.")

    group = file[axis]
    _, index_name = resolve_index(group, axis)
    return group, axis_len(file, axis), index_name

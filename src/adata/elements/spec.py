"""AnnData on-disk encoding constants and attribute access.

This is the single source of truth for `encoding-type` / `encoding-version`
pairs. Every read that inspects an encoding and every write that stamps one
goes through here, so the spec version the tool targets is stated in exactly
one place.

See docs/ELEMENTS_h5ad.md and docs/ELEMENTS_zarr.md for the full spec.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

ANNDATA = "anndata"
RAW = "raw"
DICT = "dict"
DATAFRAME = "dataframe"
ARRAY = "array"
STRING_ARRAY = "string-array"
CATEGORICAL = "categorical"
CSR_MATRIX = "csr_matrix"
CSC_MATRIX = "csc_matrix"
NULLABLE_INTEGER = "nullable-integer"
NULLABLE_BOOLEAN = "nullable-boolean"
NULLABLE_STRING_ARRAY = "nullable-string-array"
NUMERIC_SCALAR = "numeric-scalar"
STRING = "string"
AWKWARD_ARRAY = "awkward-array"
NULL = "null"

#: Encoding type -> the version this tool writes.
CURRENT_VERSION = {
    ANNDATA: "0.1.0",
    RAW: "0.1.0",
    DICT: "0.1.0",
    DATAFRAME: "0.2.0",
    ARRAY: "0.2.0",
    STRING_ARRAY: "0.2.0",
    CATEGORICAL: "0.2.0",
    CSR_MATRIX: "0.1.0",
    CSC_MATRIX: "0.1.0",
    NULLABLE_INTEGER: "0.1.0",
    NULLABLE_BOOLEAN: "0.1.0",
    NULLABLE_STRING_ARRAY: "0.1.0",
    NUMERIC_SCALAR: "0.2.0",
    STRING: "0.2.0",
    AWKWARD_ARRAY: "0.1.0",
    NULL: "0.1.0",
}

SPARSE_TYPES = frozenset({CSR_MATRIX, CSC_MATRIX})
NULLABLE_TYPES = frozenset(
    {NULLABLE_INTEGER, NULLABLE_BOOLEAN, NULLABLE_STRING_ARRAY}
)
#: Encodings stored as a group holding `values` and `mask`.
MASKED_TYPES = NULLABLE_TYPES
#: Encodings that yield text when read as strings.
STRING_TYPES = frozenset({STRING_ARRAY, NULLABLE_STRING_ARRAY, STRING, CATEGORICAL})
#: Group encodings that are plain mappings of further elements.
MAPPING_TYPES = frozenset({DICT, ANNDATA, RAW})

#: Keys that are structural rather than dataframe columns.
RESERVED_DATAFRAME_KEYS = frozenset({"_index", "__categories"})


def decode_attr(value: Any) -> Any:
    """Decode a raw attribute value to a plain Python value.

    HDF5 hands back `bytes` for string attributes while Zarr hands back `str`;
    numpy scalars appear on both. Normalising here is what lets the rest of the
    codebase compare attributes with `==`.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        return value.item()
    return value


def encoding_of(obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return the declared ``(encoding-type, encoding-version)`` of an element.

    Either component is None when the element is untagged, which is the normal
    case for files written by anndata 0.7.x and earlier.
    """
    attrs = getattr(obj, "attrs", {})
    enc = decode_attr(attrs.get("encoding-type", None))
    ver = decode_attr(attrs.get("encoding-version", None))
    return (enc or None, ver or None)


def encoding_type(obj: Any) -> Optional[str]:
    """Return just the declared ``encoding-type``, or None if untagged."""
    return encoding_of(obj)[0]


def set_encoding(obj: Any, enc_type: str, version: Optional[str] = None) -> None:
    """Stamp ``encoding-type``/``encoding-version`` on a group or dataset.

    The version defaults to the current spec version for that type, so callers
    never hardcode one.
    """
    if version is None:
        version = CURRENT_VERSION[enc_type]
    obj.attrs["encoding-type"] = enc_type
    obj.attrs["encoding-version"] = version

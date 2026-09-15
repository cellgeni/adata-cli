"""Column reading. The implementation now lives in :mod:`adata.elements.read`.

Kept as a stable import path for existing callers and tests.
"""

from __future__ import annotations

from adata.elements.read import (
    col_chunk_as_strings,
    decode_str_array,
    element_len,
    read_categorical_column,
    read_masked_column,
    read_str_all,
    read_str_chunk,
    resolve_index,
)

__all__ = [
    "col_chunk_as_strings",
    "decode_str_array",
    "element_len",
    "read_categorical_column",
    "read_masked_column",
    "read_str_all",
    "read_str_chunk",
    "resolve_index",
]

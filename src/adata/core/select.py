"""Selecting rows of an axis by column value, without loading the store.

Both `subset --query` and `split --by` need the same thing: walk an axis's
annotation columns in chunks and decide which rows to keep. Only the columns a
query actually mentions are read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adata.core.info import get_axis_group
from adata.core.query import Column, QueryError, compile_query, referenced_columns
from adata.elements import spec
from adata.elements.read import (
    categories_dataset,
    col_chunk_as_strings,
    dataframe_columns,
    decode_str_array,
)
from adata.storage import is_dataset, is_group


def _available_columns(group: Any, index_name: str) -> List[str]:
    cols = dataframe_columns(group, index_name)
    return cols + [index_name]


def _plain(values: np.ndarray) -> Column:
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind in "iufb":
        return Column(values)
    return Column(decode_str_array(values))


def _categorical(categories: Tuple[np.ndarray, np.ndarray], codes: Any) -> Column:
    """Rows of a categorical, from its categories (as stored, and as text)
    and codes. A code outside the categories (-1) is a missing entry."""
    stored, text = categories
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    present = (codes >= 0) & (codes < len(stored))
    if not len(stored):
        return Column(np.full(len(codes), "", dtype=str), ~present)
    picked = np.where(present, codes, 0)
    if stored is text:
        return Column(np.where(present, text[picked], ""), ~present)
    return Column(stored[picked], ~present, np.where(present, text[picked], ""))


def _categories(obj: Any, group: Any, name: str, cache: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if name not in cache:
        stored = np.asarray(categories_dataset(obj, group)[...]).reshape(-1)
        text = decode_str_array(stored)
        cache[name] = (stored if stored.dtype.kind in "iufb" else text, text)
    return cache[name]


def read_column(group: Any, name: str, start: int, end: int, cache: Dict) -> Column:
    """Rows [start, end) of a dataframe column, at the type it is stored as.

    The query compares numbers and booleans by value and `split` keeps a
    missing entry apart from an empty one, so both need more than the text
    `col_chunk_as_strings` gives: a missing entry reads there as "", and a
    float32 0.1 as 0.10000000149011612. `cache` keeps each categorical's
    categories across chunks.
    """
    if name not in group:
        raise RuntimeError(
            f"Column {name!r} not found in group {getattr(group, 'name', '?')!r}"
        )
    obj = group[name]
    if is_dataset(obj):
        if "categories" in obj.attrs:
            # anndata 0.7: the codes, with the categories by reference.
            return _categorical(_categories(obj, group, name, cache), obj[start:end])
        return _plain(obj[start:end])

    if is_group(obj):
        enc = spec.encoding_type(obj)
        if enc == spec.CATEGORICAL or (enc is None and "codes" in obj):
            return _categorical(
                _categories(obj, group, name, cache), obj["codes"][start:end]
            )
        if enc in spec.MASKED_TYPES or (enc is None and "values" in obj and "mask" in obj):
            column = _plain(obj["values"][start:end])
            mask = np.asarray(obj["mask"][start:end], dtype=bool).reshape(-1)
            values = column.values
            if values.dtype.kind == "U":
                values = np.where(mask, "", values)
            return Column(values, mask)
        raise ValueError(
            f"Unsupported group encoding {enc!r} for element "
            f"{getattr(obj, 'name', '?')!r}."
        )

    raise TypeError(f"Unsupported element type: {type(obj)}")


def _read_chunk(
    group: Any,
    columns: List[str], start: int, end: int, cache: Dict
) -> Dict[str, Column]:
    return {name: read_column(group, name, start, end, cache) for name in columns}


def select_indices(
    root: Any,
    axis: str,
    expr: str,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Row indices of `axis` matching the query `expr`.

    Only the columns the query references are read, so filtering on one
    annotation does not touch the rest of the frame.
    """
    group, n_rows, index_name = get_axis_group(root, axis)
    predicate = compile_query(expr)

    wanted = referenced_columns(expr)
    available = _available_columns(group, index_name)
    missing = [c for c in wanted if c not in available]
    if missing:
        raise QueryError(
            f"Column(s) {', '.join(missing)} not found in '{axis}'. "
            f"Available: {', '.join(sorted(available))}"
        )

    cache: Dict = {}
    kept: List[np.ndarray] = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        mask = np.asarray(
            predicate(_read_chunk(group, wanted, start, end, cache)), dtype=bool
        )
        if mask.any():
            kept.append(np.nonzero(mask)[0] + start)

    if not kept:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(kept).astype(np.int64)


def group_indices(
    root: Any,
    axis: str,
    column: str,
    *,
    chunk_size: int = 100_000,
) -> Tuple[Dict[Optional[str], np.ndarray], List[Optional[str]]]:
    """Group `axis` row indices by the value of `column`.

    Returns the groups and the distinct labels in order of first appearance,
    so output ordering follows the data rather than the hash of the labels.
    Rows whose value is missing form a group of their own, under None: a
    missing value is not the empty string, which can be a label too.
    """
    group, n_rows, index_name = get_axis_group(root, axis)

    available = _available_columns(group, index_name)
    if column not in available:
        raise KeyError(
            f"Column {column!r} not found in '{axis}'. "
            f"Available: {', '.join(sorted(available))}"
        )

    cache: Dict = {}
    buckets: Dict[Optional[str], List[np.ndarray]] = {}
    order: List[Optional[str]] = []

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = read_column(group, column, start, end, cache)
        values = chunk.labels()
        gaps = chunk.is_missing()
        # Positions in the chunk of the rows that have a value.
        present = np.flatnonzero(~gaps) if gaps.any() else None
        if present is not None:
            values = values[present]

        # One pass over the chunk, whatever the number of distinct labels.
        # Taking `np.nonzero(values == label)` per label -- as an earlier
        # version did -- rescanned the whole chunk once per label, so the
        # cost was O(n_rows * n_groups): a million cells split by a thousand
        # samples came to 10^9 comparisons, and `split` looked like a hang.
        uniques, first_seen, codes = np.unique(
            values, return_index=True, return_inverse=True
        )
        codes = codes.ravel()

        # Sorting the codes puts each label's positions in one contiguous
        # run, so every group is a slice rather than a search.
        by_code = np.argsort(codes, kind="stable")
        run_starts = np.searchsorted(codes[by_code], np.arange(len(uniques)), "left")
        run_ends = np.searchsorted(codes[by_code], np.arange(len(uniques)), "right")
        if present is not None:
            by_code, first_seen = present[by_code], present[first_seen]

        found: List[Tuple[int, Optional[str], np.ndarray]] = [
            (int(first_seen[code]), str(uniques[code]),
             by_code[run_starts[code] : run_ends[code]])
            for code in range(len(uniques))
        ]
        if present is not None:
            rows = np.flatnonzero(gaps)
            found.append((int(rows[0]), None, rows))

        # `np.unique` sorts; `order` has to stay in order of first appearance,
        # because it names the output files.
        for _, label, rows in sorted(found, key=lambda item: item[0]):
            if label not in buckets:
                buckets[label] = []
                order.append(label)
            buckets[label].append(rows + start)

    return (
        {k: np.concatenate(v).astype(np.int64) for k, v in buckets.items()},
        order,
    )


def read_names(root: Any, axis: str, indices: Optional[np.ndarray] = None) -> List[str]:
    """Read an axis's index labels, optionally only at `indices`."""
    group, n_rows, index_name = get_axis_group(root, axis)
    cache: Dict = {}
    names = col_chunk_as_strings(group, index_name, 0, n_rows, cache)
    if indices is None:
        return names
    return [names[i] for i in indices]

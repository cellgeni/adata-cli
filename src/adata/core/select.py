"""Selecting rows of an axis by column value, without loading the store.

Both `subset --query` and `split --by` need the same thing: walk an axis's
annotation columns in chunks and decide which rows to keep. Only the columns a
query actually mentions are read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adata.core.info import get_axis_group
from adata.core.query import QueryError, compile_query, referenced_columns
from adata.elements.read import col_chunk_as_strings, dataframe_columns


def _available_columns(group: Any, index_name: str) -> List[str]:
    cols = dataframe_columns(group, index_name)
    return cols + [index_name]


def _read_chunk(
    group: Any,
    columns: List[str], start: int, end: int, cache: Dict
) -> Dict[str, List[str]]:
    return {
        name: col_chunk_as_strings(group, name, start, end, cache)
        for name in columns
    }


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
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Group `axis` row indices by the value of `column`.

    Returns the groups and the distinct labels in order of first appearance,
    so output ordering follows the data rather than the hash of the labels.
    """
    group, n_rows, index_name = get_axis_group(root, axis)

    available = _available_columns(group, index_name)
    if column not in available:
        raise KeyError(
            f"Column {column!r} not found in '{axis}'. "
            f"Available: {', '.join(sorted(available))}"
        )

    cache: Dict = {}
    buckets: Dict[str, List[np.ndarray]] = {}
    order: List[str] = []

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        values = np.asarray(
            col_chunk_as_strings(group, column, start, end, cache), dtype=str
        )

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

        # `np.unique` sorts; `order` has to stay in order of first appearance,
        # because it names the output files.
        for code in np.argsort(first_seen, kind="stable"):
            label = str(uniques[code])
            if label not in buckets:
                buckets[label] = []
                order.append(label)
            buckets[label].append(by_code[run_starts[code] : run_ends[code]] + start)

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

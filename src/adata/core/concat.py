"""Concatenating AnnData stores on disk, along the obs axis.

Mirrors the useful core of `anndata.experimental.concat_on_disk`: inputs are
streamed a block of rows at a time and written straight into the output, so
peak memory is set by the block size rather than by the total size of the
inputs.

Columns of the concatenated axis follow `join`, in obs and in obsm's
dataframes alike; elements of the alternative axis (var's columns, varm) and
of uns follow a `merge` strategy, since there is no single correct way to
reconcile values that differ between inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console

from adata.core.info import axis_len
from adata.core.subset import aligned_rows
from adata.core.select import read_names
from adata.elements import spec
from adata.elements.read import (
    categories_dataset,
    dataframe_columns,
    decode_str_array,
    read_str_all,
    is_ordered,
    resolve_index,
)
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_categorical,
    write_dataframe_header,
    write_dense,
    write_masked,
    write_mapping,
    write_string_array,
)
from adata.elements.strings import is_string_dtype
from adata.storage import (
    copy_tree,
    create_dataset,
    dataset_create_kwargs,
    fit_chunks,
    is_dataset,
    is_zarr_group,
    Store,
    is_group,
    open_store,
)
from adata.util.names import natural_sorted
from adata.util.staging import staged_outputs

MERGE_STRATEGIES = ("same", "unique", "first", "only")
#: What the CLI accepts. "drop" is the default and maps to no merge at all.
MERGE_CHOICES = ("drop",) + MERGE_STRATEGIES


def _index_union(per_input: Sequence[List[str]]) -> List[str]:
    """Union of indices, in order of first appearance."""
    seen: Dict[str, None] = {}
    for names in per_input:
        for name in names:
            seen.setdefault(name, None)
    return list(seen)


def _index_intersection(per_input: Sequence[List[str]]) -> List[str]:
    """Intersection of indices, ordered by the first input."""
    common = set(per_input[0])
    for names in per_input[1:]:
        common &= set(names)
    return [name for name in per_input[0] if name in common]


def _column_map(source: List[str], target: List[str]) -> np.ndarray:
    """Map each source position to its target position, or -1 if dropped."""
    lookup = {name: i for i, name in enumerate(target)}
    return np.fromiter(
        (lookup.get(name, -1) for name in source), dtype=np.int64, count=len(source)
    )


def _apply_index_unique(
    names: List[str], key: Optional[str], delimiter: Optional[str]
) -> List[str]:
    if delimiter is None or key is None:
        return names
    return [f"{name}{delimiter}{key}" for name in names]


def _resolve_keys(
    files: Sequence[Path], keys: Optional[Sequence[str]]
) -> List[str]:
    """Name each input, defaulting to its filename stem.

    Keys must be distinct: `--label` writes them as categorical categories,
    which have to be unique, and `--index-unique` uses them to disambiguate obs
    names. Duplicates would otherwise produce an output that reports success
    but cannot be read back.
    """
    if keys is None:
        resolved = [f.stem if f.suffix else f.name for f in files]
        duplicates = _duplicates(resolved)
        if duplicates:
            raise ValueError(
                f"Inputs in different directories share the filename(s) "
                f"{', '.join(duplicates)}, so the default keys are not unique. "
                "Pass --keys to name them explicitly."
            )
        return resolved

    if len(keys) != len(files):
        raise ValueError(
            f"--keys has {len(keys)} entries but {len(files)} inputs were given."
        )
    duplicates = _duplicates(keys)
    if duplicates:
        raise ValueError(
            f"--keys must be unique; repeated: {', '.join(duplicates)}."
        )
    return list(keys)


def _refuse_repeated_var_names(files: Sequence[Path], var_names: List[List[str]]) -> None:
    """Refuse inputs whose var names repeat, naming them.

    Inputs are aligned by var name, so a name that occurs twice sends two
    source columns to one target column: CSR wrote both entries into it,
    dense kept the last, and an outer join lost a variable, all with exit 0.
    anndata.concat refuses the same inputs.
    """
    for path, names in zip(files, var_names):
        repeated = _duplicates(names)
        if not repeated:
            continue
        shown = ", ".join(repeated[:10])
        if len(repeated) > 10:
            shown += f" and {len(repeated) - 10:,} more"
        raise ValueError(
            f"'{path}' repeats var names: {shown}. concat matches inputs by "
            "var name, so each name must be unique. Make them unique first, "
            "e.g. with anndata's var_names_make_unique()."
        )


def _duplicates(values: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    for value in values:
        seen[value] = seen.get(value, 0) + 1
    return sorted(v for v, n in seen.items() if n > 1)


# ---------------------------------------------------------------------------
# merge strategies


def _merge_values(values: List[Any], strategy: Optional[str]) -> Tuple[bool, Any]:
    """Reconcile one element's value across inputs.

    Returns ``(keep, value)``. The strategies match anndata's: "same" keeps a
    value only when every input agrees, "unique" when there is exactly one
    distinct value among those present, "first" takes the earliest present,
    and "only" keeps it when exactly one input has it at all.
    """
    if strategy is None:
        return False, None

    present = [v for v in values if v is not _MISSING]
    if not present:
        return False, None

    if strategy == "first":
        return True, present[0]

    if strategy == "only":
        return (len(present) == 1), (present[0] if len(present) == 1 else None)

    distinct: List[Any] = []
    for value in present:
        if not any(_equal(value, seen) for seen in distinct):
            distinct.append(value)

    if strategy == "unique":
        return (len(distinct) == 1), (distinct[0] if len(distinct) == 1 else None)

    if strategy == "same":
        keep = len(distinct) == 1 and len(present) == len(values)
        return keep, (distinct[0] if keep else None)

    raise ValueError(
        f"Unknown merge strategy {strategy!r}. "
        f"Choose from: {', '.join(MERGE_STRATEGIES)}"
    )


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


class _Present:
    """Stands for a column whose value was not read, only its presence."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<present>"


_PRESENT = _Present()


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, _Incomparable) or isinstance(b, _Incomparable):
        return False
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        return bool(a == b)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# obs columns


def _column_kind(obj: Any) -> str:
    """Classify a column by how it must be concatenated.

    A dataset with a `categories` attribute is a categorical as anndata 0.7
    wrote it: the dataset holds the codes and the attribute points at the
    labels. Taking it for a plain integer column mixed codes from different
    category sets.
    """
    enc = spec.encoding_type(obj)
    if enc == spec.CATEGORICAL or (is_group(obj) and "codes" in obj):
        return "categorical"
    if is_dataset(obj) and "categories" in obj.attrs:
        return "categorical"
    if enc in spec.MASKED_TYPES or (is_group(obj) and "values" in obj):
        return "masked"
    if is_dataset(obj):
        return "string" if is_string_dtype(obj.dtype) else "numeric"
    return "other"


def _masked_encoding(obj: Any) -> str:
    """A `values`/`mask` group's encoding, inferred from its values if untagged."""
    enc = spec.encoding_type(obj)
    if enc in spec.MASKED_TYPES:
        return enc
    dtype = obj["values"].dtype
    if is_string_dtype(dtype):
        return spec.NULLABLE_STRING_ARRAY
    return spec.NULLABLE_BOOLEAN if dtype.kind == "b" else spec.NULLABLE_INTEGER


def _family(obj: Any) -> str:
    """What a column holds, whatever its encoding: categories, text or numbers."""
    kind = _column_kind(obj)
    if kind == "masked":
        return "text" if _masked_encoding(obj) == spec.NULLABLE_STRING_ARRAY else "number"
    return {"string": "text", "numeric": "number"}.get(kind, kind)


def _codes(col: Any) -> Any:
    """A categorical's codes: its `codes` member, or the 0.7 dataset itself."""
    return col["codes"] if is_group(col) else col


def _categories(col: Any, frame: Any = None) -> np.ndarray:
    """A categorical's categories in their stored dtype: numbers stay numbers,
    and text is decoded to str, whichever backend spelled it."""
    values = np.asarray(categories_dataset(col, frame)[...])
    if is_string_dtype(values.dtype):
        return np.asarray(decode_str_array(values).tolist(), dtype=object)
    return values


def _ordered(col: Any, frame: Any = None) -> bool:
    """Whether a categorical is ordered. anndata 0.7 kept the flag on the
    categories dataset rather than on the codes."""
    if is_group(col):
        return is_ordered(col)
    return is_ordered(col) or is_ordered(categories_dataset(col, frame))


def _category_labels(per_input: List[np.ndarray]) -> List[np.ndarray]:
    """Category sets cast to one dtype: their own when they are all numbers,
    or all booleans; text otherwise, the one form every value has."""
    kinds = {c.dtype.kind for c in per_input}
    if kinds and kinds <= set("iuf"):
        dtype = np.result_type(*[c.dtype for c in per_input])
        return [c.astype(dtype) for c in per_input]
    if kinds == {"b"}:
        return per_input
    return [
        c if c.dtype == object else np.asarray(decode_str_array(c).tolist(), dtype=object)
        for c in per_input
    ]


def _concat_categorical(
    parent: Any,
    name: str,
    columns: List[Optional[Any]],
    frames: List[Any],
    lengths: List[int],
) -> None:
    """Concatenate categorical columns, unioning their category sets.

    Inputs rarely share a category order, so codes are remapped rather than
    concatenated directly. A row from an input lacking the column gets code
    -1, which anndata reads back as a missing value. Categories keep their
    dtype: integer categories stay integers rather than becoming '1', '2'.
    """
    present = [(c, f) for c, f in zip(columns, frames) if c is not None]
    labels = _category_labels([_categories(c, f) for c, f in present])

    # Hash lookup, not list membership. `if category not in categories` on a
    # list is O(k) per probe and so O(k^2) over the union -- around 2 million
    # string comparisons at 1024 categories, and 5 billion at 100k. It reads
    # the same number of bytes either way, so only a comparison count sees it.
    lookup: Dict[Any, int] = {}
    for source in labels:
        for category in source.tolist():
            lookup.setdefault(category, len(lookup))
    dtype = labels[0].dtype if labels else np.dtype(object)
    categories = np.asarray(list(lookup), dtype=dtype)
    ordered = all(_ordered(c, f) for c, f in present)

    codes = np.full(sum(lengths), -1, dtype=np.int64)
    offset = 0
    source_labels = iter(labels)
    for col, length in zip(columns, lengths):
        if col is not None:
            source = next(source_labels).tolist()
            remap = np.array([lookup[c] for c in source] + [-1], dtype=np.int64)
            source_codes = np.asarray(_codes(col)[...], dtype=np.int64)
            source_codes[(source_codes < 0) | (source_codes >= len(source))] = len(source)
            codes[offset : offset + length] = remap[source_codes]
        offset += length

    write_categorical(parent, name, codes, categories, ordered=ordered)


def _concat_numbers(
    parent: Any, name: str, columns: List[Optional[Any]], lengths: List[int]
) -> None:
    """Concatenate number columns, plain or nullable, as pandas does.

    A nullable input makes the result nullable: int64 with Int64 is Int64,
    bool with boolean is boolean, and Int64 with boolean promotes to Int64,
    as numpy promotes bool to int. A column missing from some input needs a
    way to say "absent": booleans become nullable-boolean, integers float
    padded with NaN, as pandas pads them. Floats, which have no nullable
    encoding, hold missing values as NaN.
    """
    present = [c for c in columns if c is not None]
    nullable = any(_column_kind(c) == "masked" for c in present)
    dtype = np.result_type(
        *[(c["values"] if _column_kind(c) == "masked" else c).dtype for c in present]
    )
    gaps = nullable or len(present) < len(columns)

    total = sum(lengths)
    values = np.zeros(total, dtype=dtype)
    missing = np.ones(total, dtype=bool)
    offset = 0
    for col, length in zip(columns, lengths):
        if col is not None:
            if _column_kind(col) == "masked":
                values[offset : offset + length] = np.asarray(col["values"][...], dtype=dtype)
                missing[offset : offset + length] = np.asarray(col["mask"][...], dtype=bool)
            else:
                values[offset : offset + length] = np.asarray(col[...], dtype=dtype)
                missing[offset : offset + length] = False
        offset += length

    if not gaps:
        write_dense(parent, name, values)
    elif dtype.kind == "b":
        write_masked(parent, name, values, missing, spec.NULLABLE_BOOLEAN)
    elif dtype.kind in "iu" and nullable:
        write_masked(parent, name, values, missing, spec.NULLABLE_INTEGER)
    elif dtype.kind in "iufc":
        values = values.astype(np.float64) if dtype.kind in "iu" else values
        values[missing] = np.nan
        write_dense(parent, name, values)
    else:
        write_dense(parent, name, values)


def _text(col: Any, frame: Any) -> Tuple[np.ndarray, np.ndarray]:
    """A text-like column as `(values, missing)`: str values, and True where
    a categorical's code is -1 or a nullable column's mask is set."""
    kind = _column_kind(col)
    if kind == "categorical":
        # Rendered as text, numeric categories too: this column is meeting
        # text in another input.
        labels = decode_str_array(_categories(col, frame))
        codes = np.asarray(_codes(col)[...], dtype=np.int64)
        missing = (codes < 0) | (codes >= len(labels))
        table = np.append(labels.astype(object), "")
        return table[np.where(missing, len(labels), codes)], missing
    if kind == "masked":
        values = decode_str_array(np.asarray(col["values"][...])).astype(object)
        return values, np.asarray(col["mask"][...], dtype=bool)
    values = decode_str_array(np.asarray(col[...])).astype(object)
    return values, np.zeros(len(values), dtype=bool)


def _concat_text(
    parent: Any,
    name: str,
    columns: List[Optional[Any]],
    frames: List[Any],
    lengths: List[int],
) -> None:
    """Concatenate text columns, whatever mix of plain strings, categoricals
    and nullable strings encodes them, keeping every missing value missing:
    an input's own (code -1, a set mask) and the rows of inputs that lack
    the column. Rendering them all as strings, as an earlier version did,
    turned the missing values into empty strings."""
    total = sum(lengths)
    values = np.empty(total, dtype=object)
    values[:] = ""
    missing = np.ones(total, dtype=bool)
    nullable = False
    offset = 0
    for col, frame, length in zip(columns, frames, lengths):
        if col is not None:
            values[offset : offset + length], missing[offset : offset + length] = _text(col, frame)
            nullable = nullable or _column_kind(col) == "masked"
        offset += length
    _write_text(parent, name, values, missing, nullable=nullable)


def _write_text(
    parent: Any, name: str, values: np.ndarray, missing: np.ndarray, *, nullable: bool
) -> None:
    """Write a concatenated text column, choosing the encoding that holds it.

    The one place that choice is made. With nothing missing: plain strings.
    With missing values: a nullable-string-array if an input already used
    that encoding, so its readers need anndata 0.11 or later anyway; else a
    categorical whose missing rows have code -1, with categories in natural
    order. That is what anndata itself writes for a text column with missing
    values, and every release reads it, where 0.8, 0.9 and 0.10 cannot open
    a file holding a nullable-string-array at all.
    """
    if nullable:
        values = values.copy()
        values[missing] = ""
        write_masked(parent, name, values, missing, spec.NULLABLE_STRING_ARRAY)
    elif missing.any():
        # Coded through np.unique, so the Python work is per category, not
        # per row.
        distinct, inverse = np.unique(values[~missing], return_inverse=True)
        categories = natural_sorted(distinct.tolist())
        rank = {c: i for i, c in enumerate(categories)}
        codes = np.full(len(values), -1, dtype=np.int64)
        codes[~missing] = np.array([rank[c] for c in distinct.tolist()], dtype=np.int64)[inverse]
        write_categorical(parent, name, codes, categories)
    else:
        write_string_array(parent, name, values)


def _concat_obs_column(
    parent: Any,
    name: str,
    columns: List[Optional[Any]],
    frames: List[Any],
    lengths: List[int],
    console: Console,
) -> bool:
    """Write one concatenated annotation column. Returns whether it was kept.

    `frames` are the dataframes the columns come from, which a 0.7
    categorical needs to find its categories on Zarr.
    """
    families = {_family(c) for c in columns if c is not None}

    if families == {"categorical"}:
        _concat_categorical(parent, name, columns, frames, lengths)
    elif families == {"number"}:
        _concat_numbers(parent, name, columns, lengths)
    elif families <= {"text", "categorical"}:
        _concat_text(parent, name, columns, frames, lengths)
    else:
        kinds = sorted({_column_kind(c) for c in columns if c is not None})
        console.print(
            f"[yellow]Dropping column {name!r}: "
            f"cannot concatenate encodings {kinds}[/]"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# matrices


def _matrix_kind(obj: Any) -> str:
    enc = spec.encoding_type(obj)
    if enc in spec.SPARSE_TYPES:
        return enc
    if is_dataset(obj):
        return "dense"
    return "other"


def _widen_to_hold(dtype: np.dtype, largest: int) -> np.dtype:
    """`dtype`, or the narrowest of int32/int64 wider than it that holds `largest`."""
    dtype = np.dtype(dtype)
    if largest <= np.iinfo(dtype).max:
        return dtype
    for candidate in (np.int32, np.int64):
        if largest <= np.iinfo(candidate).max:
            return np.dtype(np.result_type(dtype, candidate))
    return np.dtype(np.int64)


def _concat_index_dtypes(
    sources: List[Any], n_minor: int
) -> Tuple[np.dtype, np.dtype]:
    """The `indices` and `indptr` dtypes for a concatenation of `sources`.

    The widest the inputs used, so int32 inputs give an int32 output rather
    than the int64 every output used to get. That is widened only when the
    result could not be addressed otherwise: `indices` must reach the
    output's minor dimension, and `indptr` its total nonzero count, which is
    known before writing as the sum of the inputs'. Both are checked against
    the chosen dtype's own limit, not int32's, since the spec allows
    narrower index arrays and an int16 `indptr` would otherwise wrap.
    """
    indices = np.result_type(*[s["indices"].dtype for s in sources])
    indptr = np.result_type(*[s["indptr"].dtype for s in sources])
    nnz = sum(int(s["data"].shape[0]) for s in sources)
    return _widen_to_hold(indices, n_minor), _widen_to_hold(indptr, nnz)


def _write_indptr(
    group: Any, indptr: List[int], dtype: np.dtype, template: Any
) -> None:
    from adata.core.subset import _sparse_dataset

    values = np.asarray(indptr, dtype=np.int64).astype(dtype, copy=False)
    _sparse_dataset(group, "indptr", dtype, values.size, template)[:] = values


def _concat_sparse(
    dst_parent: Any,
    name: str,
    sources: List[Any],
    col_maps: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    chunk_rows: int,
) -> None:
    """Concatenate CSR matrices row-wise, remapping columns as they stream.

    Absent columns need no fill: a sparse matrix's zeros are implicit, so
    dropped entries simply do not appear in the output. An input whose var
    order differs from the target's has its rows re-sorted by target column,
    so the output stays canonical; appending them in source order wrote
    column indices out of order within a row.
    """
    from adata.core.subset import _append, _growable_like

    group = dst_parent.create_group(name)
    spec.set_encoding(group, spec.CSR_MATRIX)
    from adata.elements.write import set_shape_attr

    set_shape_attr(group, (n_rows, n_cols))

    dtype = np.result_type(*[s["data"].dtype for s in sources])
    index_dtype, pointer_dtype = _concat_index_dtypes(sources, n_cols)
    out_data = _growable_like(group, "data", dtype, sources[0]["data"])
    out_indices = _growable_like(
        group, "indices", index_dtype, sources[0]["indices"]
    )
    indptr = [0]
    nnz = 0

    for source, col_map in zip(sources, col_maps):
        src_indptr = np.asarray(source["indptr"][...], dtype=np.int64)
        src_data, src_indices = source["data"], source["indices"]
        n_src_rows = len(src_indptr) - 1
        kept_cols = col_map[col_map >= 0]
        increasing = bool(np.all(kept_cols[1:] > kept_cols[:-1]))

        for start in range(0, n_src_rows, chunk_rows):
            end = min(start + chunk_rows, n_src_rows)
            lo, hi = int(src_indptr[start]), int(src_indptr[end])
            block_idx = (
                np.asarray(src_indices[lo:hi], dtype=np.int64)
                if hi > lo
                else np.empty(0, dtype=np.int64)
            )
            block_data = (
                np.asarray(src_data[lo:hi]) if hi > lo else np.empty(0, dtype=dtype)
            )

            rows = np.repeat(
                np.arange(end - start), np.diff(src_indptr[start : end + 1])
            )
            mapped = col_map[block_idx]
            keep = mapped >= 0
            rows, mapped, values = rows[keep], mapped[keep], block_data[keep]
            if not increasing:
                # Rows stay in order and each row's entries sort by target
                # column: one sort per block, not one per row.
                order = np.lexsort((mapped, rows))
                mapped, values = mapped[order], values[order]

            counts = np.bincount(rows, minlength=end - start)
            indptr.extend((nnz + np.cumsum(counts)).tolist())
            nnz += int(counts.sum())
            _append(out_indices, mapped)
            _append(out_data, values.astype(dtype))

    _write_indptr(group, indptr, pointer_dtype, sources[0]["indptr"])


def _concat_dense(
    dst_parent: Any,
    name: str,
    sources: List[Any],
    col_maps: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    chunk_rows: int,
    fill_value: float,
) -> None:
    """Concatenate dense matrices row-wise, scattering columns into place."""
    dtype = np.result_type(*[s.dtype for s in sources])
    if fill_value != 0 and dtype.kind in ("i", "u", "b"):
        dtype = np.dtype("float64")

    out = create_dataset(dst_parent, name, shape=(n_rows, n_cols), dtype=dtype)
    spec.set_encoding(out, spec.ARRAY)

    row_offset = 0
    for source, col_map in zip(sources, col_maps):
        keep = col_map >= 0
        targets = col_map[keep]
        n_src_rows = source.shape[0]

        for start in range(0, n_src_rows, chunk_rows):
            end = min(start + chunk_rows, n_src_rows)
            block = np.full((end - start, n_cols), fill_value, dtype=dtype)
            block[:, targets] = np.asarray(source[start:end, :])[:, keep]
            out[row_offset + start : row_offset + end, :] = block

        row_offset += n_src_rows


def _inverse_column_map(col_map: np.ndarray, n_cols: int) -> np.ndarray:
    """Invert a source->target column map into target->source, -1 where absent."""
    inverse = np.full(n_cols, -1, dtype=np.int64)
    present = col_map >= 0
    inverse[col_map[present]] = np.nonzero(present)[0]
    return inverse


def _concat_csc(
    dst_parent: Any,
    name: str,
    sources: List[Any],
    col_maps: List[np.ndarray],
    row_counts: List[int],
    n_rows: int,
    n_cols: int,
) -> None:
    """Concatenate CSC matrices row-wise, keeping the CSC encoding.

    A CSC matrix is stored by column, so concatenating along obs means, for
    each target column, appending each input's row indices in turn with that
    input's row offset added. Because every input's column is already sorted
    and the offsets increase, the result is sorted without a re-sort.
    """
    from adata.core.subset import _append, _growable_like
    from adata.elements.write import set_shape_attr

    group = dst_parent.create_group(name)
    spec.set_encoding(group, spec.CSC_MATRIX)
    set_shape_attr(group, (n_rows, n_cols))

    dtype = np.result_type(*[s["data"].dtype for s in sources])
    index_dtype, pointer_dtype = _concat_index_dtypes(sources, n_rows)
    out_data = _growable_like(group, "data", dtype, sources[0]["data"])
    out_indices = _growable_like(
        group, "indices", index_dtype, sources[0]["indices"]
    )
    indptr = [0]
    nnz = 0

    inverses = [_inverse_column_map(cm, n_cols) for cm in col_maps]
    indptrs = [np.asarray(s["indptr"][...], dtype=np.int64) for s in sources]
    row_offsets = np.cumsum([0] + list(row_counts[:-1]))

    for target_col in range(n_cols):
        rows: List[np.ndarray] = []
        values: List[np.ndarray] = []
        for source, src_indptr, inverse, offset in zip(
            sources, indptrs, inverses, row_offsets
        ):
            src_col = int(inverse[target_col])
            if src_col < 0:
                continue
            lo, hi = int(src_indptr[src_col]), int(src_indptr[src_col + 1])
            if hi <= lo:
                continue
            rows.append(np.asarray(source["indices"][lo:hi], dtype=np.int64) + offset)
            values.append(np.asarray(source["data"][lo:hi]))

        if rows:
            _append(out_indices, np.concatenate(rows))
            _append(out_data, np.concatenate(values).astype(dtype))
            nnz += int(sum(len(r) for r in rows))
        indptr.append(nnz)

    _write_indptr(group, indptr, pointer_dtype, sources[0]["indptr"])


def check_matrix_encodings(roots: List[Any], console: Console) -> None:
    """Fail before writing anything if a matrix cannot be concatenated.

    Checked up front rather than mid-write: discovering this half way through
    would leave a partial store behind, and skipping the matrix would produce
    an output silently missing X.
    """
    def _check(label: str, sources: List[Any]) -> None:
        kinds = {_matrix_kind(s) for s in sources}
        if kinds in ({spec.CSR_MATRIX}, {spec.CSC_MATRIX}, {"dense"}):
            return
        wanted = sorted(kinds)[0]
        raise ValueError(
            f"Cannot concatenate {label!r}: inputs use "
            f"{', '.join(sorted(kinds))}. Every input must use the same "
            f"encoding. Recast them to match first, e.g. "
            f"`adata recast INPUT {label} -o converted.h5ad --layout "
            f"{'dense' if wanted == 'dense' else wanted.removesuffix('_matrix')}`."
        )

    if all("X" in r for r in roots):
        _check("X", [r["X"] for r in roots])
    elif any("X" in r for r in roots):
        console.print("[yellow]Skipping X: not present in every input[/]")

    names = _index_union(
        [list(r["layers"].keys()) if "layers" in r else [] for r in roots]
    )
    for name in names:
        if all("layers" in r and name in r["layers"] for r in roots):
            _check(f"layers/{name}", [r["layers"][name] for r in roots])

    # obsm entries are stacked only when every input stores them alike; a
    # sparse entry dense elsewhere is refused here, as X would be, rather
    # than dropped. Entries of different widths are skipped later, whatever
    # their encodings, and dataframes are not matrices at all.
    names = _index_union([list(r["obsm"].keys()) if "obsm" in r else [] for r in roots])
    for name in names:
        if not all("obsm" in r and name in r["obsm"] for r in roots):
            continue
        sources = [r["obsm"][name] for r in roots]
        kinds = {_matrix_kind(s) for s in sources}
        if (kinds & spec.SPARSE_TYPES and kinds <= spec.SPARSE_TYPES | {"dense"}
                and len({_trailing_shape(s) for s in sources}) == 1):
            _check(f"obsm/{name}", sources)


def _shape(obj: Any) -> Tuple[int, ...]:
    """A matrix's shape, sparse (from its `shape` attribute) or dense."""
    if spec.encoding_type(obj) in spec.SPARSE_TYPES:
        return tuple(int(d) for d in np.asarray(obj.attrs["shape"]).reshape(-1))
    return tuple(int(d) for d in obj.shape)


def _trailing_shape(obj: Any) -> Tuple[int, ...]:
    """An obsm entry's shape past the obs axis."""
    return _shape(obj)[1:]


def _concat_matrix(
    dst_parent: Any,
    name: str,
    sources: List[Any],
    col_maps: List[np.ndarray],
    row_counts: List[int],
    n_rows: int,
    n_cols: int,
    chunk_rows: int,
    fill_value: float,
    console: Console,
) -> bool:
    """Concatenate X or one layer across inputs. Returns whether it was written.

    Encodings are validated by :func:`check_matrix_encodings` before the output
    store exists, so anything reaching here is concatenable.
    """
    kinds = {_matrix_kind(s) for s in sources}

    if kinds == {spec.CSR_MATRIX}:
        _concat_sparse(
            dst_parent, name, sources, col_maps, n_rows, n_cols, chunk_rows
        )
        return True

    if kinds == {spec.CSC_MATRIX}:
        _concat_csc(
            dst_parent, name, sources, col_maps, row_counts, n_rows, n_cols
        )
        return True

    if kinds == {"dense"}:
        _concat_dense(
            dst_parent,
            name,
            sources,
            col_maps,
            n_rows,
            n_cols,
            chunk_rows,
            fill_value,
        )
        return True

    # A backstop. `check_matrix_encodings` runs before the output store is
    # created and should have raised already; this catches an element it
    # does not cover, where failing late still beats writing nonsense.
    raise ValueError(
        f"Cannot concatenate {name!r}: inputs use {', '.join(sorted(kinds))}."
    )


def _concat_obsm(
    dst_parent: Any,
    name: str,
    sources: List[Any],
    n_rows: int,
    chunk_rows: int,
    console: Console,
    *,
    lengths: Optional[List[int]] = None,
    join: str = "inner",
    target_obs: Optional[List[str]] = None,
) -> bool:
    """Concatenate an obsm entry row-wise; its columns are not an axis.

    A dataframe in every input (a SCope import's `RegulonsAUC`) is
    concatenated as obs is, its columns following `join`.
    """
    if all(spec.encoding_type(s) == spec.DATAFRAME for s in sources) and target_obs is not None:
        _concat_frame(dst_parent, name, sources, lengths or [], join, target_obs, console)
        return True
    kinds = {_matrix_kind(s) for s in sources}
    if "other" in kinds:
        console.print(f"[yellow]Skipping obsm/{name}: not an array, or not a dataframe, in every input[/]")
        return False

    widths = {_trailing_shape(s) for s in sources}
    if len(widths) != 1:
        console.print(
            f"[yellow]Skipping obsm/{name}: inputs disagree on shape {sorted(widths)}[/]"
        )
        return False
    if len(kinds) != 1:
        # A backstop: `check_matrix_encodings` refuses this before writing.
        raise ValueError(f"Cannot concatenate 'obsm/{name}': inputs use {', '.join(sorted(kinds))}.")

    trailing = widths.pop()
    if kinds != {"dense"}:
        # Sparse in every input, with one encoding (checked before writing):
        # stacked as X is, every column kept where it is.
        same = [np.arange(trailing[0], dtype=np.int64)] * len(sources)
        if kinds == {spec.CSR_MATRIX}:
            _concat_sparse(dst_parent, name, sources, same, n_rows, trailing[0], chunk_rows)
        else:
            _concat_csc(dst_parent, name, sources, same, [_shape(s)[0] for s in sources],
                        n_rows, trailing[0])
        return True
    dtype = np.result_type(*[s.dtype for s in sources])
    out = create_dataset(
        dst_parent, name, shape=(n_rows,) + trailing, dtype=dtype
    )
    spec.set_encoding(out, spec.ARRAY)

    offset = 0
    for source in sources:
        for start in range(0, source.shape[0], chunk_rows):
            end = min(start + chunk_rows, source.shape[0])
            out[offset + start : offset + end, ...] = source[start:end, ...]
        offset += source.shape[0]
    return True


def _concat_frame(
    dst_parent: Any,
    name: str,
    sources: List[Any],
    lengths: List[int],
    join: str,
    target_obs: List[str],
    console: Console,
) -> None:
    """Dataframes stacked row-wise, as obs is: their columns follow `join`."""
    per_input = [dataframe_columns(s, resolve_index(s, "obs")[1]) for s in sources]
    if join == "inner":
        shared = set(per_input[0]).intersection(*map(set, per_input[1:]))
        names = [c for c in per_input[0] if c in shared]
    else:
        names = _index_union(per_input)
    frame = write_dataframe_header(dst_parent, name, target_obs, [])
    written = [
        column for column in names
        if _concat_obs_column(frame, column, [s[column] if column in s else None for s in sources],
                              sources, lengths, console)
    ]
    set_column_order(frame, written)


# ---------------------------------------------------------------------------
# varm


def _digest(hasher: Any, values: np.ndarray) -> None:
    """Feed values to `hasher` exactly: bytes for numbers, so NaN equals NaN,
    and the text itself for strings, never object pointers."""
    if values.dtype.kind in "OSUT":
        for text in decode_str_array(values).reshape(-1).tolist():
            data = text.encode("utf-8", "surrogatepass")
            hasher.update(len(data).to_bytes(8, "little") + data)
    else:
        hasher.update(np.ascontiguousarray(values).tobytes())


def _varm_snapshot(obj: Any, where: np.ndarray, block_rows: int) -> Any:
    """A comparable value of a varm entry, on the target var: a digest of an
    array, read a block of rows at a time; of each column, for a dataframe.

    A digest, not the values, so that comparing an entry holds one block
    (or one var-length column) in memory rather than every input's whole
    entry at once.
    """
    if is_dataset(obj):
        hasher = hashlib.blake2b(digest_size=32)
        for start in range(0, len(where), block_rows):
            _digest(hasher, aligned_rows(obj, where[start:start + block_rows]))
        kind = "text" if is_string_dtype(obj.dtype) else str(obj.dtype)
        return ("array", kind, tuple(obj.shape[1:]), hasher.digest())
    if spec.encoding_type(obj) != spec.DATAFRAME:
        return _Incomparable()
    columns = []
    for name in dataframe_columns(obj, resolve_index(obj, "var")[1]):
        column = obj[name]
        kind = _column_kind(column)
        hasher = hashlib.blake2b(digest_size=32)
        if kind == "numeric":
            values = np.asarray(column[...])[where]
            _digest(hasher, values)
            columns.append((name, str(values.dtype), hasher.digest()))
        elif kind == "categorical":
            values, missing = _text(column, obj)
            _digest(hasher, missing[where])
            _digest(hasher, values[where])
            columns.append((name, "categorical", hasher.digest()))
        elif kind == "masked":
            _digest(hasher, np.asarray(column["mask"][...], dtype=bool)[where])
            _digest(hasher, np.asarray(column["values"][...])[where])
            columns.append((name, "masked", hasher.digest()))
        else:
            _digest(hasher, np.asarray(read_str_all(column), dtype=object)[where])
            columns.append((name, "text", hasher.digest()))
    return ("dataframe", tuple(columns))


def _write_varm_array(parent: Any, name: str, obj: Any, take: np.ndarray, block_rows: int) -> None:
    """A varm array's rows on the target var, a block at a time, its
    chunking and compression kept as every other copied dataset's are."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    shape = (len(take),) + tuple(obj.shape[1:])
    kw = fit_chunks(dataset_create_kwargs(obj, target_backend=backend, dst_parent=parent),
                    shape, backend=backend)
    text = is_string_dtype(obj.dtype)
    out = create_dataset(parent, name, shape=shape, dtype=obj.dtype, **kw)
    spec.set_encoding(out, spec.STRING_ARRAY if text else spec.ARRAY)
    for start in range(0, len(take), block_rows):
        out[start:start + block_rows, ...] = aligned_rows(obj, take[start:start + block_rows])


def _write_varm(
    dst: Any,
    roots: List[Any],
    var_names: List[List[str]],
    target_var: List[str],
    merge: Optional[str],
    console: Console,
    chunk_rows: int = 1024,
) -> None:
    """varm entries the merge strategy keeps, on the target var.

    As var's columns: an input lacking the entry, or some of the target's
    genes (an outer join), counts as missing; `same` and `unique` compare
    digests read once per input, `first` and `only` presence alone. Arrays
    are read and written `chunk_rows` genes at a time.
    """
    names = _index_union([list(r["varm"].keys()) if "varm" in r else [] for r in roots])
    if not names:
        return
    if merge is None:
        console.print(
            "[yellow]Dropping varm: pass --merge to keep the entries the inputs agree on.[/]"
        )
        return
    varm = write_mapping(dst, "varm")
    positions = [_column_map(target_var, names_i) for names_i in var_names]
    compares = merge in ("same", "unique")
    for name in names:
        aligned: List[Any] = []
        for root, where in zip(roots, positions):
            if "varm" not in root or name not in root["varm"] or (where < 0).any():
                aligned.append(_MISSING)
            elif compares:
                aligned.append(_varm_snapshot(root["varm"][name], where, chunk_rows))
            else:
                aligned.append(_PRESENT)
        keep, _ = _merge_values(aligned, merge)
        if not keep:
            continue
        source = next(i for i, v in enumerate(aligned) if v is not _MISSING)
        obj, take = roots[source]["varm"][name], positions[source]
        if is_dataset(obj):
            _write_varm_array(varm, name, obj, take, chunk_rows)
        elif spec.encoding_type(obj) == spec.DATAFRAME:
            frame = write_dataframe_header(varm, name, target_var, [])
            columns = dataframe_columns(obj, resolve_index(obj, "var")[1])
            for column in columns:
                _write_var_column(frame, column, obj[column], take, obj)
            set_column_order(frame, columns)
        else:
            console.print(f"[yellow]Skipping varm/{name}: neither an array nor a dataframe[/]")


# ---------------------------------------------------------------------------
# orchestration


def _merge_group(
    dst_parent: Any,
    name: str,
    groups: List[Optional[Any]],
    strategy: Optional[str],
    console: Console,
) -> None:
    """Merge a mapping (uns) across inputs under `strategy`, as anndata's
    `merge_nested` does: mappings every input holds under a key are merged
    member by member, whatever the strategy, and a mapping left with no
    members is dropped rather than written empty."""
    target = write_mapping(dst_parent, name)
    if strategy is None:
        return
    present = [g for g in groups if g is not None]
    # An input without the mapping holds none of its keys, so `same` keeps
    # nothing, as anndata's intersection with an empty `uns` does.
    if strategy == "same" and len(present) < len(groups):
        present = []
    _write_merged(target, _merge_plan(present, strategy))


def _is_mapping(obj: Any) -> bool:
    """A group anndata reads as a dict: tagged so, or untagged (0.7)."""
    return (
        is_group(obj)
        and spec.encoding_type(obj) in (spec.DICT, None)
        and "h5sparse_format" not in obj.attrs
    )


def _merge_plan(groups: List[Any], strategy: str) -> Dict[str, Any]:
    """What to keep of `groups` merged: for each key, the member to copy, or
    a nested plan for a mapping. `same` considers the keys every group has,
    the other strategies the keys any group has; only `same` and `unique`
    compare values, `first` and `only` decide by presence."""
    keys: Dict[str, None] = {}
    for group in groups:
        for key in group.keys():
            keys.setdefault(key, None)
    if strategy == "same":
        keys = {k: None for k in keys if all(k in g for g in groups)}

    plan: Dict[str, Any] = {}
    for key in keys:
        members = [g[key] for g in groups if key in g]
        if all(_is_mapping(m) for m in members):
            nested = _merge_plan(members, strategy)
            if nested:
                plan[key] = nested
            continue
        compares = strategy in ("same", "unique")
        keep, _ = _merge_values(
            [_readable_value(m) if compares else _PRESENT for m in members], strategy
        )
        if keep:
            plan[key] = members[0]
    return plan


def _write_merged(target: Any, plan: Dict[str, Any]) -> None:
    for key, value in plan.items():
        if isinstance(value, dict):
            _write_merged(write_mapping(target, key), value)
        else:
            copy_tree(value, target, key)


#: Elements larger than this are not compared value-by-value.
MAX_COMPARABLE_ELEMENTS = 1_000_000


class _Incomparable:
    """Stands for a value too large to compare, and equal to nothing."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<incomparable>"


def _readable_value(obj: Any, budget: Optional[List[int]] = None) -> Any:
    """A comparable snapshot of an element, for the merge strategies.

    Recurses into groups and reads their datasets, including attributes. Using
    only child key names -- as an earlier version did -- would make two
    dataframes with the same columns but different values compare equal, so
    `same` and `unique` would keep conflicting metadata.

    Very large elements are reported as incomparable, which equals nothing and
    so is never kept by `same` or `unique`.
    """
    if budget is None:
        budget = [MAX_COMPARABLE_ELEMENTS]

    try:
        attrs = tuple(
            sorted(
                (str(k), _hashable(spec.decode_attr(v)))
                for k, v in obj.attrs.items()
                if k not in ("encoding-version",)
            )
        )

        if is_dataset(obj):
            size = int(np.prod(obj.shape)) if obj.shape else 1
            budget[0] -= size
            if budget[0] < 0:
                return _Incomparable()
            return ("dataset", attrs, _hashable(np.asarray(obj[...])))

        children = []
        for key in sorted(obj.keys()):
            value = _readable_value(obj[key], budget)
            if isinstance(value, _Incomparable):
                return value
            children.append((str(key), value))
        return ("group", attrs, tuple(children))
    except Exception:
        return _Incomparable()


def _hashable(value: Any) -> Any:
    """Reduce a value to something `_equal` can compare reliably.

    Text is compared decoded, as a tuple of str, whatever spelled it: HDF5
    hands back bytes, or an object array of bytes, where Zarr hands back str
    or StringDType, and comparing those undecoded meant a string never
    matched across an h5ad and a Zarr input. A list (how Zarr returns an
    array attribute) is compared as the array HDF5 returns for it.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "surrogateescape")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        try:
            value = np.asarray(value)
        except ValueError:
            return tuple(_hashable(v) for v in value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OSUT":
            return ("text", value.shape, tuple(decode_str_array(value).reshape(-1).tolist()))
        return (value.shape, value.dtype.kind, value.tobytes())
    return value


def _inputs_zarr_format(stores: Sequence[Store]) -> Optional[int]:
    """The Zarr version the inputs agree on, or None to take the default.

    A Zarr output follows its inputs, as every other command's follows its
    source, so v2 inputs are not silently upgraded. h5ad inputs have no
    version and do not count; Zarr inputs of both versions get the default.
    """
    versions = {s.zarr_format for s in stores} - {None}
    return versions.pop() if len(versions) == 1 else None


def concat_on_disk(
    files: Sequence[Path],
    output: Path,
    console: Console,
    *,
    join: str = "inner",
    label: Optional[str] = None,
    keys: Optional[Sequence[str]] = None,
    index_unique: Optional[str] = None,
    merge: Optional[str] = None,
    uns_merge: Optional[str] = None,
    fill_value: float = 0.0,
    chunk_rows: int = 1024,
    zarr_format: Optional[int] = None,
) -> None:
    """Concatenate stores along the obs axis, streaming each one in turn.

    `join` aligns var: "inner" keeps only variables present in every input,
    "outer" keeps their union. obs columns follow the same rule. Elements of
    the var axis and of uns are reconciled by `merge`/`uns_merge`, since
    inputs may legitimately disagree about them.

    obsp and varp are not concatenated -- a pairwise matrix has no meaningful
    value between cells that came from different inputs. anndata's own
    implementation refuses these too.
    """
    if len(files) < 2:
        raise ValueError("Concatenation needs at least two input stores.")
    if join not in ("inner", "outer"):
        raise ValueError("--join must be 'inner' or 'outer'.")
    if output.exists():
        raise FileExistsError(f"'{output}' already exists.")

    input_keys = _resolve_keys(files, keys)
    stores = [open_store(f, "r") for f in files]

    try:
        roots = [s.root for s in stores]

        var_names = [read_names(r, "var") for r in roots]
        obs_names = [read_names(r, "obs") for r in roots]
        obs_counts = [axis_len(r, "obs") for r in roots]
        _refuse_repeated_var_names(files, var_names)

        target_var = (
            _index_intersection(var_names)
            if join == "inner"
            else _index_union(var_names)
        )
        # Inputs with no variables at all concatenate to n_obs x 0, as in
        # anndata; the refusal is for inputs that have some but share none.
        if not target_var and any(var_names):
            raise ValueError(
                "No variables are shared by all inputs. Use --join outer to "
                "keep their union instead."
            )

        col_maps = [_column_map(names, target_var) for names in var_names]
        n_obs = sum(obs_counts)
        n_var = len(target_var)

        console.print(
            f"[cyan]Concatenating {len(files)} stores: "
            f"{n_obs} obs x {n_var} var ({join} join)[/]"
        )
        for f, key, count in zip(files, input_keys, obs_counts):
            console.print(f"  [dim]{key}[/]: {count} obs from {f.name}")

        target_obs: List[str] = []
        for names, key in zip(obs_names, input_keys):
            target_obs.extend(_apply_index_unique(names, key, index_unique))

        if len(set(target_obs)) != len(target_obs):
            console.print(
                "[yellow]Warning: obs names are not unique across inputs. "
                "Pass --index-unique to disambiguate them.[/]"
            )

        # Validated before the output exists, so a mismatch cannot leave a
        # half-written store behind.
        check_matrix_encodings(roots, console)

        if zarr_format is None:
            zarr_format = _inputs_zarr_format(stores)

        # Written beside the output and renamed into place once complete, so
        # a failure part way leaves nothing that reads as a finished store.
        with staged_outputs([output]) as (temp,), open_store(
            temp, "w", zarr_format=zarr_format
        ) as dst_store:
            dst = dst_store.root
            _write_obs(
                dst, roots, target_obs, obs_counts, join, label, input_keys, console
            )
            _write_var(dst, roots, var_names, target_var, merge, console)
            ensure_anndata_skeleton(dst)

            _write_matrices(
                dst,
                roots,
                col_maps,
                obs_counts,
                n_obs,
                n_var,
                chunk_rows,
                fill_value,
                console,
            )
            _write_obsm(dst, roots, n_obs, chunk_rows, console,
                        lengths=obs_counts, join=join, target_obs=target_obs)
            _write_varm(dst, roots, var_names, target_var, merge, console, chunk_rows)

            uns_groups = [r["uns"] if "uns" in r else None for r in roots]
            if any(g is not None for g in uns_groups):
                if uns_merge is None:
                    write_mapping(dst, "uns")
                else:
                    _merge_group(dst, "uns", uns_groups, uns_merge, console)

            for name in ("obsp", "varp"):
                if any(name in r and len(list(r[name].keys())) for r in roots):
                    console.print(
                        f"[yellow]Dropping {name}: pairwise values have no "
                        f"meaning across concatenated inputs.[/]"
                    )
            if any("raw" in r for r in roots):
                console.print("[yellow]Dropping raw/: not concatenated.[/]")

        console.print(f"[green]Wrote[/] {output} ({n_obs} obs x {n_var} var)")
    finally:
        for store in stores:
            store.close()


def _write_obs(
    dst: Any,
    roots: List[Any],
    target_obs: List[str],
    obs_counts: List[int],
    join: str,
    label: Optional[str],
    input_keys: List[str],
    console: Console,
) -> None:
    """Write the concatenated obs frame, plus the batch column if requested."""
    groups = [r["obs"] for r in roots]
    per_input_cols = [
        dataframe_columns(g, resolve_index(g, "obs")[1]) for g in groups
    ]

    if join == "inner":
        shared = set(per_input_cols[0]).intersection(*map(set, per_input_cols[1:]))
        names = [c for c in per_input_cols[0] if c in shared]
    else:
        names = _index_union(per_input_cols)

    written: List[str] = []
    obs_group = write_dataframe_header(dst, "obs", target_obs, [])

    for name in names:
        columns = [g[name] if name in g else None for g in groups]
        if _concat_obs_column(obs_group, name, columns, groups, obs_counts, console):
            written.append(name)

    if label:
        if label in written:
            raise ValueError(
                f"--label {label!r} collides with an existing obs column."
            )
        codes = np.concatenate(
            [np.full(n, i, dtype=np.int64) for i, n in enumerate(obs_counts)]
        )
        write_categorical(obs_group, label, codes, input_keys, ordered=False)
        written.append(label)

    set_column_order(obs_group, written)


def _write_var(
    dst: Any,
    roots: List[Any],
    var_names: List[List[str]],
    target_var: List[str],
    merge: Optional[str],
    console: Console,
) -> None:
    """Write the aligned var frame, keeping columns the merge strategy allows."""
    var_group = write_dataframe_header(dst, "var", target_var, [])
    if merge is None:
        return

    groups = [r["var"] for r in roots]
    per_input_cols = [
        dataframe_columns(g, resolve_index(g, "var")[1]) for g in groups
    ]
    candidates = _index_union(per_input_cols)

    positions = [_column_map(target_var, names) for names in var_names]
    written: List[str] = []

    # "first" and "only" decide on presence alone, so the column values are
    # never read for them.
    compares = merge in ("same", "unique")

    for name in candidates:
        aligned: List[Any] = []
        for group, where in zip(groups, positions):
            if name not in group or (where < 0).any():
                aligned.append(_MISSING)
                continue
            if not compares:
                aligned.append(_PRESENT)
                continue
            # Read the column once and index the result. Reading it inside the
            # generator -- as an earlier version did -- re-read the whole
            # column for every target variable, which is quadratic and turns a
            # 36k-var merge into hours of pure CPU.
            values = read_str_all(group[name], parent_group=group)
            aligned.append(tuple(values[i] for i in where))

        keep, _ = _merge_values(aligned, merge)
        if not keep:
            continue

        source_i = next(
            i for i, v in enumerate(aligned) if v is not _MISSING
        )
        column = groups[source_i][name]
        take = positions[source_i]
        _write_var_column(var_group, name, column, take, groups[source_i])
        written.append(name)

    set_column_order(var_group, written)


def _write_var_column(
    parent: Any, name: str, column: Any, take: np.ndarray, frame: Any = None
) -> None:
    """Write one var column, reordered onto the target var index.

    Each encoding is rewritten as itself. Rendering everything as text -- as an
    earlier version did for nullable columns -- turned missing values into
    empty strings and lost the numeric and boolean dtypes. A categorical keeps
    its categories' dtype, and one written by anndata 0.7 (a dataset of
    codes) is found through `frame`, the dataframe it belongs to.
    """
    kind = _column_kind(column)

    if kind == "categorical":
        categories = _categories(column, frame)
        codes = np.asarray(_codes(column)[...], dtype=np.int64)[take]
        write_categorical(parent, name, codes, categories, ordered=_ordered(column, frame))
        return

    if kind == "numeric":
        write_dense(parent, name, np.asarray(column[...])[take])
        return

    if kind == "masked":
        enc = spec.encoding_type(column) or spec.NULLABLE_STRING_ARRAY
        mask = np.asarray(column["mask"][...], dtype=bool)[take]
        raw_values = np.asarray(column["values"][...])
        if enc == spec.NULLABLE_STRING_ARRAY:
            from adata.elements.read import decode_str_array

            values = decode_str_array(raw_values)[take].tolist()
        else:
            values = raw_values[take]
        na_value = spec.decode_attr(column.attrs.get("na-value", None))
        write_masked(parent, name, values, mask, enc, na_value=na_value)
        return

    values = read_str_all(column)
    write_string_array(parent, name, [values[i] for i in take])


def _write_matrices(
    dst: Any,
    roots: List[Any],
    col_maps: List[np.ndarray],
    row_counts: List[int],
    n_obs: int,
    n_var: int,
    chunk_rows: int,
    fill_value: float,
    console: Console,
) -> None:
    """Concatenate X and every layer shared by all inputs."""
    if all("X" in r for r in roots):
        _concat_matrix(
            dst,
            "X",
            [r["X"] for r in roots],
            col_maps,
            row_counts,
            n_obs,
            n_var,
            chunk_rows,
            fill_value,
            console,
        )

    layer_names = _index_union(
        [list(r["layers"].keys()) if "layers" in r else [] for r in roots]
    )
    if not layer_names:
        return

    layers = write_mapping(dst, "layers")
    for name in layer_names:
        if not all("layers" in r and name in r["layers"] for r in roots):
            console.print(
                f"[yellow]Skipping layer {name!r}: not present in every input[/]"
            )
            continue
        _concat_matrix(
            layers,
            name,
            [r["layers"][name] for r in roots],
            col_maps,
            row_counts,
            n_obs,
            n_var,
            chunk_rows,
            fill_value,
            console,
        )


def _write_obsm(
    dst: Any,
    roots: List[Any],
    n_obs: int,
    chunk_rows: int,
    console: Console,
    *,
    lengths: Optional[List[int]] = None,
    join: str = "inner",
    target_obs: Optional[List[str]] = None,
) -> None:
    """Concatenate obsm entries shared by all inputs."""
    names = _index_union(
        [list(r["obsm"].keys()) if "obsm" in r else [] for r in roots]
    )
    if not names:
        return

    obsm = write_mapping(dst, "obsm")
    for name in names:
        if not all("obsm" in r and name in r["obsm"] for r in roots):
            console.print(
                f"[yellow]Skipping obsm/{name}: not present in every input[/]"
            )
            continue
        _concat_obsm(
            obsm, name, [r["obsm"][name] for r in roots], n_obs, chunk_rows, console,
            lengths=lengths, join=join, target_obs=target_obs,
        )

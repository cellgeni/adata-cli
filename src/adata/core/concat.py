"""Concatenating AnnData stores on disk, along the obs axis.

Mirrors the useful core of `anndata.experimental.concat_on_disk`: inputs are
streamed a block of rows at a time and written straight into the output, so
peak memory is set by the block size rather than by the total size of the
inputs.

Columns of the concatenated axis follow `join`; elements of the alternative
axis and of uns follow a `merge` strategy, since there is no single correct
way to reconcile values that differ between inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console

from adata.core.info import axis_len
from adata.core.select import read_names
from adata.elements import spec
from adata.elements.read import (
    dataframe_columns,
    element_len,
    read_str_all,
    read_categories,
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
from adata.storage import (
    copy_tree,
    create_dataset,
    is_dataset,
    is_group,
    open_store,
)

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
    """Classify a column by how it must be concatenated."""
    enc = spec.encoding_type(obj)
    if enc == spec.CATEGORICAL or (is_group(obj) and "codes" in obj):
        return "categorical"
    if enc in spec.MASKED_TYPES or (is_group(obj) and "values" in obj):
        return "masked"
    if is_dataset(obj):
        from adata.elements.strings import is_string_dtype

        return "string" if is_string_dtype(obj.dtype) else "numeric"
    return "other"


def _concat_categorical(
    parent: Any, name: str, columns: List[Optional[Any]], lengths: List[int]
) -> None:
    """Concatenate categorical columns, unioning their category sets.

    Inputs rarely share a category order, so codes are remapped rather than
    concatenated directly. A row from an input lacking the column gets code
    -1, which anndata reads back as a missing value.
    """
    # Hash lookup, not list membership. `if category not in categories` on a
    # list is O(k) per probe and so O(k^2) over the union -- around 2 million
    # string comparisons at 1024 categories, and 5 billion at 100k. It reads
    # the same number of bytes either way, so only a comparison count sees it.
    lookup: Dict[str, int] = {}
    for col in columns:
        if col is None:
            continue
        for category in read_categories(col):
            text = str(category)
            if text not in lookup:
                lookup[text] = len(lookup)

    categories = list(lookup)
    ordered = all(is_ordered(c) for c in columns if c is not None)

    codes = np.full(sum(lengths), -1, dtype=np.int64)
    offset = 0
    for col, length in zip(columns, lengths):
        if col is not None:
            source_cats = [str(c) for c in read_categories(col)]
            remap = np.array(
                [lookup[c] for c in source_cats] + [-1], dtype=np.int64
            )
            source_codes = np.asarray(col["codes"][...], dtype=np.int64)
            source_codes[source_codes < 0] = len(source_cats)
            codes[offset : offset + length] = remap[source_codes]
        offset += length

    write_categorical(parent, name, codes, categories, ordered=ordered)


def _concat_numeric(
    parent: Any, name: str, columns: List[Optional[Any]], lengths: List[int]
) -> None:
    """Concatenate numeric columns, promoting to float when rows are missing.

    An integer column cannot represent "absent", so a column missing from some
    input is promoted to float and padded with NaN -- the same thing pandas
    does on an outer join.
    """
    present = [c for c in columns if c is not None]
    dtype = np.result_type(*[c.dtype for c in present])
    complete = len(present) == len(columns)

    if not complete and dtype.kind in ("i", "u", "b"):
        dtype = np.dtype("float64")

    out = np.empty(sum(lengths), dtype=dtype)
    if not complete:
        out[:] = np.nan if dtype.kind == "f" else 0

    offset = 0
    for col, length in zip(columns, lengths):
        if col is not None:
            out[offset : offset + length] = np.asarray(col[...], dtype=dtype)
        offset += length

    write_dense(parent, name, out)


def _concat_masked(
    parent: Any, name: str, columns: List[Optional[Any]], lengths: List[int]
) -> None:
    """Concatenate nullable columns, masking rows from inputs that lack them."""
    kinds = {spec.encoding_type(c) for c in columns if c is not None}
    enc = kinds.pop() if len(kinds) == 1 else spec.NULLABLE_STRING_ARRAY

    total = sum(lengths)
    mask = np.ones(total, dtype=bool)
    present = [c for c in columns if c is not None]

    # Fill a typed buffer by slice, rather than a Python list of length
    # n_obs one element at a time. The list cost an object per row and three
    # full passes over it, which is what made obs concatenation the most
    # allocation-hungry part of a streamed concat.
    if enc == spec.NULLABLE_STRING_ARRAY:
        from adata.elements.read import decode_str_array

        filled = np.empty(total, dtype=object)
        filled[:] = ""
    else:
        decode_str_array = None
        dtype = np.result_type(*[c["values"].dtype for c in present])
        filled = np.zeros(total, dtype=dtype)

    offset = 0
    for col, length in zip(columns, lengths):
        if col is not None:
            chunk = np.asarray(col["values"][...])
            if decode_str_array is not None:
                chunk = decode_str_array(chunk)
            filled[offset : offset + length] = chunk
            mask[offset : offset + length] = np.asarray(
                col["mask"][...], dtype=bool
            )
        offset += length

    write_masked(parent, name, filled, mask, enc)


def _concat_string(
    parent: Any, name: str, columns: List[Optional[Any]], lengths: List[int]
) -> None:
    """Concatenate string columns, masking rows from inputs that lack them."""
    total = sum(lengths)
    values: List[str] = [""] * total
    mask = np.zeros(total, dtype=bool)

    offset = 0
    for col, length in zip(columns, lengths):
        if col is None:
            mask[offset : offset + length] = True
        else:
            values[offset : offset + length] = read_str_all(col)
        offset += length

    if mask.any():
        write_masked(parent, name, values, mask, spec.NULLABLE_STRING_ARRAY)
    else:
        write_string_array(parent, name, values)


def _concat_obs_column(
    parent: Any,
    name: str,
    columns: List[Optional[Any]],
    lengths: List[int],
    console: Console,
) -> bool:
    """Write one concatenated annotation column. Returns whether it was kept."""
    kinds = {_column_kind(c) for c in columns if c is not None}

    if kinds == {"categorical"}:
        _concat_categorical(parent, name, columns, lengths)
    elif kinds == {"numeric"}:
        _concat_numeric(parent, name, columns, lengths)
    elif kinds == {"masked"}:
        _concat_masked(parent, name, columns, lengths)
    elif kinds <= {"string", "categorical", "masked"}:
        # Mixed text-like encodings: fall back to plain strings, which every
        # one of them can represent.
        _concat_string(parent, name, columns, lengths)
    else:
        console.print(
            f"[yellow]Dropping column {name!r}: "
            f"cannot concatenate encodings {sorted(kinds)}[/]"
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
    dropped entries simply do not appear in the output.
    """
    from adata.core.subset import _append, _growable

    group = dst_parent.create_group(name)
    spec.set_encoding(group, spec.CSR_MATRIX)
    from adata.elements.write import set_shape_attr

    set_shape_attr(group, (n_rows, n_cols))

    dtype = np.result_type(*[s["data"].dtype for s in sources])
    out_data = _growable(group, "data", dtype)
    out_indices = _growable(group, "indices", np.int64)
    indptr = [0]
    nnz = 0

    for source, col_map in zip(sources, col_maps):
        src_indptr = np.asarray(source["indptr"][...], dtype=np.int64)
        src_data, src_indices = source["data"], source["indices"]
        n_src_rows = len(src_indptr) - 1

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

            kept_idx: List[np.ndarray] = []
            kept_data: List[np.ndarray] = []
            for row in range(start, end):
                sl = slice(int(src_indptr[row]) - lo, int(src_indptr[row + 1]) - lo)
                mapped = col_map[block_idx[sl]]
                keep = mapped >= 0
                kept_idx.append(mapped[keep])
                kept_data.append(block_data[sl][keep])
                nnz += int(keep.sum())
                indptr.append(nnz)

            if kept_idx:
                _append(out_indices, np.concatenate(kept_idx))
                _append(out_data, np.concatenate(kept_data).astype(dtype))

    create_dataset(group, "indptr", data=np.asarray(indptr, dtype=np.int64))


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
    from adata.core.subset import _append, _growable
    from adata.elements.write import set_shape_attr

    group = dst_parent.create_group(name)
    spec.set_encoding(group, spec.CSC_MATRIX)
    set_shape_attr(group, (n_rows, n_cols))

    dtype = np.result_type(*[s["data"].dtype for s in sources])
    out_data = _growable(group, "data", dtype)
    out_indices = _growable(group, "indices", np.int64)
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

    create_dataset(group, "indptr", data=np.asarray(indptr, dtype=np.int64))


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
        raise ValueError(
            f"Cannot concatenate {label!r}: inputs use "
            f"{', '.join(sorted(kinds))}. Every input must use the same "
            "encoding -- convert them to match first."
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
) -> bool:
    """Concatenate an obsm entry row-wise; its columns are not an axis."""
    if not all(is_dataset(s) for s in sources):
        console.print(f"[yellow]Skipping obsm/{name}: not a dense array in every input[/]")
        return False

    widths = {tuple(s.shape[1:]) for s in sources}
    if len(widths) != 1:
        console.print(
            f"[yellow]Skipping obsm/{name}: inputs disagree on shape {sorted(widths)}[/]"
        )
        return False

    trailing = widths.pop()
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


# ---------------------------------------------------------------------------
# orchestration


def _merge_group(
    dst_parent: Any,
    name: str,
    groups: List[Optional[Any]],
    strategy: Optional[str],
    console: Console,
) -> None:
    """Merge a mapping (uns, varm, ...) across inputs under `strategy`."""
    target = write_mapping(dst_parent, name)
    if strategy is None:
        return

    keys: List[str] = []
    for group in groups:
        if group is None:
            continue
        for key in group.keys():
            if key not in keys:
                keys.append(key)

    for key in keys:
        members = [
            g[key] if g is not None and key in g else _MISSING for g in groups
        ]
        present = [m for m in members if m is not _MISSING]

        if all(is_group(m) for m in present) and strategy in ("same", "unique"):
            # Nested mappings are merged member-wise rather than compared whole.
            if all(spec.encoding_type(m) == spec.DICT for m in present):
                _merge_group(target, key, [
                    m if m is not _MISSING else None for m in members
                ], strategy, console)
                continue

        comparable = [
            _MISSING if m is _MISSING else _readable_value(m) for m in members
        ]
        keep, _ = _merge_values(comparable, strategy)
        if keep:
            source = next(m for m in members if m is not _MISSING)
            copy_tree(source, target, key)


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
    """Reduce a value to something `_equal` can compare reliably."""
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.kind, value.tobytes()
                if value.dtype.kind not in ("O", "T") else tuple(map(str, value.reshape(-1).tolist())))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


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

        target_var = (
            _index_intersection(var_names)
            if join == "inner"
            else _index_union(var_names)
        )
        if not target_var:
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

        with open_store(output, "w", zarr_format=zarr_format) as dst_store:
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
            _write_obsm(dst, roots, n_obs, chunk_rows, console)

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
        if _concat_obs_column(obs_group, name, columns, obs_counts, console):
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
            values = read_str_all(group[name])
            aligned.append(tuple(values[i] for i in where))

        keep, _ = _merge_values(aligned, merge)
        if not keep:
            continue

        source_i = next(
            i for i, v in enumerate(aligned) if v is not _MISSING
        )
        column = groups[source_i][name]
        take = positions[source_i]
        _write_var_column(var_group, name, column, take)
        written.append(name)

    set_column_order(var_group, written)


def _write_var_column(
    parent: Any, name: str, column: Any, take: np.ndarray
) -> None:
    """Write one var column, reordered onto the target var index.

    Each encoding is rewritten as itself. Rendering everything as text -- as an
    earlier version did for nullable columns -- turned missing values into
    empty strings and lost the numeric and boolean dtypes.
    """
    kind = _column_kind(column)

    if kind == "categorical":
        categories = [str(c) for c in read_categories(column)]
        codes = np.asarray(column["codes"][...], dtype=np.int64)[take]
        write_categorical(
            parent, name, codes, categories, ordered=is_ordered(column)
        )
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
    dst: Any, roots: List[Any], n_obs: int, chunk_rows: int, console: Console
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
            obsm, name, [r["obsm"][name] for r in roots], n_obs, chunk_rows, console
        )

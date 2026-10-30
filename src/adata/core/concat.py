"""Concatenating AnnData stores on disk, along the obs axis.

Mirrors the useful core of `anndata.experimental.concat_on_disk`: inputs are
streamed a block of rows at a time and written straight into the output, so
peak memory is set by the block size rather than by the total size of the
inputs. Inputs are opened as they are needed, a bounded number at a time
(`_Inputs`), so neither file descriptors nor memory grow with how many there
are.

Columns of the concatenated axis follow `join`, in obs and in obsm's
dataframes alike; elements of the alternative axis (var's columns, varm,
varp) and of uns follow a `merge` strategy, since there is no single correct
way to reconcile values that differ between inputs.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.core.info import axis_len
from adata.core.subset import _open_output_limit, aligned_rows
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
    set_shape_attr,
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
    detect_backend,
    open_store,
    zarr_format_of,
)
from adata.util.names import natural_sorted
from adata.util.staging import check_free_space, disk_size, staged_outputs

MERGE_STRATEGIES = ("same", "unique", "first", "only")
#: What the CLI accepts. "drop" is the default and maps to no merge at all.
MERGE_CHOICES = ("drop",) + MERGE_STRATEGIES

#: How many inputs concat holds open at once: a quarter of the file
#: descriptor limit, as split caps its open outputs. Beyond that, inputs are
#: closed and opened again as they are needed.
MAX_OPEN_INPUTS = _open_output_limit()


# ---------------------------------------------------------------------------
# inputs


class _Inputs:
    """The input stores, opened as they are needed, at most
    `MAX_OPEN_INPUTS` of them at once.

    Opening every input up front and holding all of them to the end ran out
    of file descriptors at about 253 HDF5 inputs under macOS's default limit
    (about 1,020 on Linux), and each open input's caches made memory grow
    with the number of inputs whatever --chunk was. An input is opened when
    it is next needed, and another closed to make room: the one used most
    recently. Every pass over the inputs goes from the first to the last, so
    the next pass then finds the first ones still open, where closing the
    one used longest ago would close each input just before it is needed.

    So an object read from an input stays valid only until another input is
    opened: code that comes back to an input looks its element up again,
    through `get`, `each` or a `_Column`, rather than holding on to it.
    """

    def __init__(self, files: Sequence[Path]) -> None:
        self.files = list(files)
        self._open: "OrderedDict[int, Store]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.files)

    def __enter__(self) -> "_Inputs":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(suppress=exc_type is not None)

    def root(self, i: int) -> Any:
        store = self._open.get(i)
        if store is not None:
            self._open.move_to_end(i)
            return store.root
        while len(self._open) >= max(1, MAX_OPEN_INPUTS):
            self._open.popitem(last=True)[1].close()
        store = self._open[i] = _open_input(self.files[i])
        return store.root

    def get(self, i: int, path: str) -> Optional[Any]:
        """Input `i`'s element at `path`, or None if it has none."""
        root = self.root(i)
        return root[path] if path in root else None

    def each(self, path: str) -> Iterator[Optional[Any]]:
        """Each input's element at `path` in turn, None where it has none."""
        for i in range(len(self)):
            yield self.get(i, path)

    def keys(self, i: int, path: str) -> List[str]:
        """The members of input `i`'s group at `path`; none if it lacks it."""
        group = self.get(i, path)
        return list(group.keys()) if group is not None else []

    def close(self, suppress: bool = False) -> None:
        while self._open:
            self._open.popitem(last=False)[1].close(suppress=suppress)


def _open_input(path: Path) -> Store:
    try:
        return open_store(path, "r")
    except OSError as error:
        if error.errno != errno.EMFILE:
            raise
        raise OSError(
            f"Could not open '{path}': too many files are open. concat holds "
            f"at most {MAX_OPEN_INPUTS} inputs open at once, but the process "
            "has no descriptors left for them; raise its limit with `ulimit -n`."
        ) from error


class _Column:
    """One dataframe column across the inputs, looked up afresh on each pass.

    Iterating gives `(column, frame)` for each input in turn: `column` is
    None where the input lacks it, and `frame` is the dataframe it belongs
    to, which an anndata 0.7 categorical needs to find its categories.
    """

    def __init__(self, inputs: _Inputs, frame: str, name: str) -> None:
        self.inputs, self.frame, self.name = inputs, frame, name

    def __iter__(self) -> Iterator[Tuple[Optional[Any], Any]]:
        for i in range(len(self.inputs)):
            frame = self.inputs.get(i, self.frame)
            present = frame is not None and self.name in frame
            yield (frame[self.name] if present else None), frame

    def present(self) -> Iterator[Tuple[Any, Any]]:
        """`(column, frame)` for the inputs that have the column."""
        return ((column, frame) for column, frame in self if column is not None)


# ---------------------------------------------------------------------------
# indices


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


def _var_union(per_input: Sequence[List[str]]) -> List[str]:
    """An outer join's var, in the order anndata gives it.

    anndata unions the inputs' indexes with pandas, which keeps the order
    when every input has the same var in the same order and sorts the union
    otherwise; an input with no variables has no say. First appearance, as
    an earlier version used, put the same genes in another order, so code
    aligning by position against an anndata-made reference saw another
    matrix.
    """
    named = [names for names in per_input if names]
    if not named:
        return []
    if all(names is named[0] or names == named[0] for names in named[1:]):
        return list(named[0])
    return sorted(_index_union(named))


def _column_map(source: List[str], target: List[str]) -> np.ndarray:
    """Map each source position to its target position, or -1 if dropped."""
    lookup = {name: i for i, name in enumerate(target)}
    return np.fromiter(
        (lookup.get(name, -1) for name in source), dtype=np.int64, count=len(source)
    )


def _per_var_list(var_names: List[List[str]], compute: Callable[[List[str]], Any]) -> List[Any]:
    """`compute(names)` for each input, once for inputs sharing a var list."""
    done: Dict[int, Any] = {}
    for names in var_names:
        if id(names) not in done:
            done[id(names)] = compute(names)
    return [done[id(names)] for names in var_names]


def _apply_index_unique(
    names: List[str], key: Optional[str], delimiter: Optional[str]
) -> List[str]:
    if delimiter is None or key is None:
        return names
    return [f"{name}{delimiter}{key}" for name in names]


def _read_axes(
    inputs: _Inputs, input_keys: List[str], index_unique: Optional[str]
) -> Tuple[List[List[str]], List[int], List[str], List[Optional[int]]]:
    """Each input's var names, obs count and Zarr version, and the output's
    obs names, reading the inputs one at a time.

    Inputs with the same var names share one list. An atlas's inputs
    usually do, and a copy each made memory grow by a gene list per input.
    """
    shared: Dict[Tuple[str, ...], List[str]] = {}
    var_names: List[List[str]] = []
    obs_counts: List[int] = []
    formats: List[Optional[int]] = []
    target_obs: List[str] = []
    for i, key in enumerate(input_keys):
        root = inputs.root(i)
        names = read_names(root, "var")
        var_names.append(shared.setdefault(tuple(names), names))
        obs_counts.append(axis_len(root, "obs"))
        formats.append(zarr_format_of(root))
        target_obs.extend(_apply_index_unique(read_names(root, "obs"), key, index_unique))
    return var_names, obs_counts, target_obs, formats


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
    checked = set()
    for path, names in zip(files, var_names):
        if id(names) in checked:
            continue
        checked.add(id(names))
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


def _merge_choice(values: Iterable[Any], strategy: Optional[str]) -> Optional[int]:
    """Which input's value of an element to keep, or None to drop it.

    The strategies match anndata's: "same" keeps a value only when every
    input agrees, "unique" when there is exactly one distinct value among
    those present, "first" takes the earliest present, and "only" keeps it
    when exactly one input has it at all.

    `values` is read lazily and at most two are held at once, a reference
    and the one compared with it, so that comparing an element across many
    inputs does not hold every input's copy; and the reading stops as soon
    as the answer is known.
    """
    if strategy is None:
        return None
    if strategy not in MERGE_STRATEGIES:
        raise ValueError(
            f"Unknown merge strategy {strategy!r}. "
            f"Choose from: {', '.join(MERGE_STRATEGIES)}"
        )
    chosen: Optional[int] = None
    reference: Any = _MISSING
    for i, value in enumerate(values):
        if value is _MISSING:
            if strategy == "same":
                return None
            continue
        if chosen is None:
            if strategy == "first":
                return i
            chosen, reference = i, value
        elif strategy == "only" or not _equal(value, reference):
            return None
    return chosen


def _merge_values(values: Iterable[Any], strategy: Optional[str]) -> Tuple[bool, Any]:
    """Reconcile one element's value across inputs: `(keep, value)`, as
    `_merge_choice` decides it."""
    values = list(values)
    chosen = _merge_choice(values, strategy)
    return (chosen is not None), (values[chosen] if chosen is not None else None)


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
    or all booleans; text otherwise, the one form every value has. An empty
    set has no say: anndata writes an all-missing categorical's categories
    as float64, and this tool as text."""
    voters = [c for c in per_input if c.size]
    kinds = {c.dtype.kind for c in voters}
    if kinds and kinds <= set("iuf"):
        dtype = np.result_type(*[c.dtype for c in voters])
        return [c.astype(dtype) for c in per_input]
    if kinds == {"b"}:
        return [c.astype(bool) for c in per_input]
    return [
        c if c.dtype == object else np.asarray(decode_str_array(c).tolist(), dtype=object)
        for c in per_input
    ]


def _natural_order(values: List[Any], dtype: np.dtype) -> List[Any]:
    """`values` as natsort orders them: numbers by value, text naturally."""
    return sorted(values) if dtype.kind in "biuf" else natural_sorted(values)


def _merged_categories(labels: List[np.ndarray], ordered: List[bool]) -> Tuple[np.ndarray, bool]:
    """A merged categorical's categories, and whether it stays ordered, as
    anndata decides them.

    Unordered, the first input's categories keep their order when they
    already hold every input's, as pandas keeps them; otherwise the union is
    natsorted, as anndata sorts it. First appearance, as an earlier version
    used, gave other codes and another plotting order than anndata.concat.

    Ordered only when every input with categories is ordered and has the
    same ones in the same order. Otherwise the first input's order was kept,
    silently reversing what another input's meant; anndata gives up the
    order (plain values), and so the union is written unordered.
    """
    dtype = labels[0].dtype if labels else np.dtype(object)
    # Hash lookup, not list membership. `if category not in categories` on a
    # list is O(k) per probe and so O(k^2) over the union -- around 2 million
    # string comparisons at 1024 categories, and 5 billion at 100k. It reads
    # the same number of bytes either way, so only a comparison count sees it.
    union: Dict[Any, None] = {}
    for source in labels:
        for category in source.tolist():
            union.setdefault(category, None)
    if any(ordered):
        stated = [(c, o) for c, o in zip(labels, ordered) if c.size]
        if stated and all(
            o and c.size == len(union) and np.array_equal(c, stated[0][0]) for c, o in stated
        ):
            return stated[0][0], True
    elif labels and labels[0].size == len(union):
        return labels[0], False
    return np.asarray(_natural_order(list(union), dtype), dtype=dtype), False


def _concat_categorical(
    parent: Any, name: str, column: _Column, lengths: List[int], console: Console
) -> None:
    """Concatenate categorical columns, unioning their category sets.

    Inputs rarely share a category order, so codes are remapped rather than
    concatenated directly. A row from an input lacking the column gets code
    -1, which anndata reads back as a missing value. Categories keep their
    dtype: integer categories stay integers rather than becoming '1', '2'.
    """
    labels: List[np.ndarray] = []
    ordered: List[bool] = []
    for col, frame in column.present():
        labels.append(_categories(col, frame))
        ordered.append(_ordered(col, frame))
    labels = _category_labels(labels)
    categories, keeps_order = _merged_categories(labels, ordered)
    if any(ordered) and not keeps_order:
        console.print(
            f"[yellow]Column {name!r}: the inputs disagree on its category order, "
            "so it is written unordered (anndata keeps no order either).[/]"
        )
    lookup = {category: i for i, category in enumerate(categories.tolist())}

    codes = np.full(sum(lengths), -1, dtype=np.int64)
    offset = 0
    source_labels = iter(labels)
    for (col, _), length in zip(column, lengths):
        if col is not None:
            source = next(source_labels).tolist()
            remap = np.array([lookup[c] for c in source] + [-1], dtype=np.int64)
            source_codes = np.asarray(_codes(col)[...], dtype=np.int64)
            source_codes[(source_codes < 0) | (source_codes >= len(source))] = len(source)
            codes[offset : offset + length] = remap[source_codes]
        offset += length

    write_categorical(parent, name, codes, categories, ordered=keeps_order)


def _concat_numbers(
    parent: Any, name: str, column: _Column, lengths: List[int]
) -> None:
    """Concatenate number columns, plain or nullable, as pandas does.

    A nullable input makes the result nullable: int64 with Int64 is Int64,
    bool with boolean is boolean, and Int64 with boolean promotes to Int64,
    as numpy promotes bool to int. A column missing from some input needs a
    way to say "absent": booleans become nullable-boolean, integers float
    padded with NaN, as pandas pads them. Floats, which have no nullable
    encoding, hold missing values as NaN.
    """
    kinds = [
        (_column_kind(c), (c["values"] if _column_kind(c) == "masked" else c).dtype)
        for c, _ in column.present()
    ]
    nullable = any(kind == "masked" for kind, _ in kinds)
    dtype = np.result_type(*[d for _, d in kinds])
    gaps = nullable or len(kinds) < len(lengths)

    total = sum(lengths)
    values = np.zeros(total, dtype=dtype)
    missing = np.ones(total, dtype=bool)
    offset = 0
    for (col, _), length in zip(column, lengths):
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
    parent: Any, name: str, column: _Column, lengths: List[int]
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
    for (col, frame), length in zip(column, lengths):
        if col is not None:
            values[offset : offset + length], missing[offset : offset + length] = _text(col, frame)
            nullable = nullable or (
                _column_kind(col) == "masked"
                and _masked_encoding(col) == spec.NULLABLE_STRING_ARRAY
            )
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
    column: _Column,
    lengths: List[int],
    console: Console,
) -> bool:
    """Write one concatenated annotation column. Returns whether it was kept."""
    kinds = [(_family(c), _column_kind(c)) for c, _ in column.present()]
    families = {family for family, _ in kinds}
    # Text meeting nullable numbers is kept as text, the numbers rendered and
    # their mask kept; meeting plain numbers it is dropped below. Both as
    # before, since anndata cannot write either mix itself.
    nullable_numbers = all(kind == "masked" for family, kind in kinds if family == "number")

    if families == {"categorical"}:
        _concat_categorical(parent, name, column, lengths, console)
    elif families == {"number"}:
        _concat_numbers(parent, name, column, lengths)
    elif families <= {"text", "categorical", "number"} and nullable_numbers:
        _concat_text(parent, name, column, lengths)
    else:
        console.print(
            f"[yellow]Dropping column {name!r}: "
            f"cannot concatenate encodings {sorted({kind for _, kind in kinds})}[/]"
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
    sources: Iterable[Any], n_minor: int
) -> Tuple[np.dtype, np.dtype]:
    """The `indices` and `indptr` dtypes for a concatenation of `sources`.

    The widest the inputs used, so int32 inputs give an int32 output rather
    than the int64 every output used to get. That is widened only when the
    result could not be addressed otherwise: `indices` must reach the
    output's minor dimension, and `indptr` its total nonzero count, which is
    known before writing as the sum of the inputs'. Both are checked against
    the chosen dtype's own limit, not int32's, since the spec allows
    narrower index arrays and an int16 `indptr` would otherwise wrap.
    `sources` is read once, so it may be a lazy pass over the inputs.
    """
    indices: List[np.dtype] = []
    indptr: List[np.dtype] = []
    nnz = 0
    for source in sources:
        indices.append(source["indices"].dtype)
        indptr.append(source["indptr"].dtype)
        nnz += int(source["data"].shape[0])
    return (
        _widen_to_hold(np.result_type(*indices), n_minor),
        _widen_to_hold(np.result_type(*indptr), nnz),
    )


def _write_indptr(
    group: Any, indptr: np.ndarray, dtype: np.dtype, template: Any
) -> None:
    from adata.core.subset import _sparse_dataset

    values = np.asarray(indptr, dtype=np.int64).astype(dtype, copy=False)
    _sparse_dataset(group, "indptr", dtype, values.size, template)[:] = values


def _sparse_group(dst_parent: Any, name: str, enc: str, shape: Tuple[int, int]) -> Any:
    group = dst_parent.create_group(name)
    spec.set_encoding(group, enc)
    set_shape_attr(group, shape)
    return group


def _entries_per_block(nnz: int, n_major: int, chunk_rows: int) -> int:
    """How many stored entries make a block: as many as `chunk_rows` rows
    hold on average, so a sparse matrix read by entries costs about what one
    read by rows does."""
    return max(1, chunk_rows) * max(1, -(-nnz // max(1, n_major)))


def _concat_sparse(
    dst_parent: Any,
    name: str,
    inputs: _Inputs,
    path: str,
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

    group = _sparse_group(dst_parent, name, spec.CSR_MATRIX, (n_rows, n_cols))
    dtype = np.result_type(*[s["data"].dtype for s in inputs.each(path)])
    index_dtype, pointer_dtype = _concat_index_dtypes(inputs.each(path), n_cols)
    template = inputs.get(0, path)
    out_data = _growable_like(group, "data", dtype, template["data"])
    out_indices = _growable_like(group, "indices", index_dtype, template["indices"])
    # One int64 array for the row pointers. A Python list of int objects
    # took 36 to 40 bytes a row against 8: 400 MB at 10M cells.
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    row = nnz = 0

    for i, col_map in enumerate(col_maps):
        source = inputs.get(i, path)
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
            indptr[row + 1 : row + 1 + end - start] = nnz + np.cumsum(counts)
            row += end - start
            nnz += int(counts.sum())
            _append(out_indices, mapped)
            _append(out_data, values.astype(dtype))
        # Let go of this input's pointers before reading the next one's,
        # rather than hold both: 8 bytes a row more at the peak.
        del src_indptr

    _write_indptr(group, indptr, pointer_dtype, inputs.get(0, path)["indptr"])


def _dtype_holding(dtype: np.dtype, fill_value: float) -> np.dtype:
    """`dtype`, or float64 if it cannot hold `fill_value` exactly."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "biu":
        return dtype
    if not np.isfinite(fill_value) or fill_value != int(fill_value):
        return np.dtype(np.float64)
    if dtype.kind == "b":
        return dtype if fill_value in (0, 1) else np.dtype(np.float64)
    info = np.iinfo(dtype)
    return dtype if info.min <= fill_value <= info.max else np.dtype(np.float64)


def _layout_like(src: Any, dst_parent: Any, shape: Tuple[int, ...], dtype: Any) -> Dict[str, Any]:
    """Creation kwargs that give a new array `src`'s chunking and
    compression, the chunks fitted to its own shape, as every copied
    dataset's are kept."""
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    kw = fit_chunks(dataset_create_kwargs(src, target_backend=backend, dst_parent=dst_parent),
                    shape, backend=backend)
    if np.dtype(dtype) != np.dtype(src.dtype):
        # The source's fill value is of its own dtype, not the output's.
        kw.pop("fillvalue", None)
        kw.pop("fill_value", None)
    return kw


def _concat_dense(
    dst_parent: Any,
    name: str,
    inputs: _Inputs,
    path: str,
    col_maps: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    chunk_rows: int,
    fill_value: float,
) -> None:
    """Concatenate dense matrices row-wise, scattering columns into place.

    The output keeps the first input's chunking and compression; written
    with the defaults, gzip-compressed inputs came out contiguous and
    uncompressed, 34 times their size for a matrix of zeros. Cells an input
    lacks (an outer join) get `fill_value`, and only then may the dtype
    change: to float64, if the inputs' cannot hold it. Promoting whenever a
    fill value was given made integer X float64 on an inner join, where
    nothing is filled.
    """
    dtype = np.result_type(*[s.dtype for s in inputs.each(path)])
    if any(np.count_nonzero(m >= 0) < n_cols for m in col_maps):
        dtype = _dtype_holding(dtype, fill_value)

    shape = (n_rows, n_cols)
    out = create_dataset(dst_parent, name, shape=shape, dtype=dtype,
                         **_layout_like(inputs.get(0, path), dst_parent, shape, dtype))
    spec.set_encoding(out, spec.ARRAY)

    row_offset = 0
    for i, col_map in enumerate(col_maps):
        source = inputs.get(i, path)
        keep = col_map >= 0
        targets = col_map[keep]
        gaps = targets.size < n_cols
        n_src_rows = source.shape[0]

        for start in range(0, n_src_rows, chunk_rows):
            end = min(start + chunk_rows, n_src_rows)
            block = (np.full((end - start, n_cols), fill_value, dtype=dtype) if gaps
                     else np.empty((end - start, n_cols), dtype=dtype))
            block[:, targets] = np.asarray(source[start:end, :])[:, keep]
            out[row_offset + start : row_offset + end, :] = block

        row_offset += n_src_rows


@contextmanager
def _bucket_scratch(dst_parent: Any) -> Iterator[Any]:
    """Where the buckets go: an HDF5 file beside the output, whatever the
    output is. `scratch_group` keeps a Zarr output's scratch inside the
    output, where every small append to a bucket rewrites that array's
    metadata: putting a CSC matrix together there took sixty times as long
    as in HDF5."""
    from adata.core.csr_build import scratch_group

    root = getattr(getattr(dst_parent, "store", None), "root", None)
    if not is_zarr_group(dst_parent) or root is None:
        with scratch_group(dst_parent, "__concat_scratch__") as group:
            yield group
        return
    fd, path = tempfile.mkstemp(
        prefix=".adata-", suffix=".concat-scratch.h5", dir=str(Path(root).parent)
    )
    os.close(fd)
    try:
        # No chunk cache, as `scratch_group`'s: each bucket would hold a
        # partly written chunk in one.
        with h5py.File(path, "w", rdcc_nbytes=0) as handle:
            yield handle.create_group("buckets")
    finally:
        os.unlink(path)


def _csc_bounds(indptr: np.ndarray, entries: int) -> np.ndarray:
    """Column bounds of the buckets: about `entries` entries each, as a
    transpose's are, and a column holding more than that alone in a bucket
    of its own, so that it can be streamed rather than read whole."""
    from adata.core.csr_build import MIN_BUCKET_ENTRIES, bucket_bounds

    bounds = bucket_bounds(indptr, chunk=entries)
    big = np.flatnonzero(np.diff(indptr) > max(MIN_BUCKET_ENTRIES, entries))
    return np.unique(np.concatenate((bounds, big, big + 1)))


def _in_row_order(minors: Any, n: int, entries: int) -> bool:
    """Whether a bucket's rows are in order already, read a piece at a time."""
    last = -1
    for lo in range(0, n, entries):
        piece = np.asarray(minors[lo : lo + entries], dtype=np.int64)
        if piece.size and (piece[0] < last or bool((piece[1:] < piece[:-1]).any())):
            return False
        last = int(piece[-1]) if piece.size else last
    return True


def _csc_runs(sorter: Any, entries: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """The buckets' `(rows, values)`, column by column, each in row order.

    A bucket of several columns is read whole and sorted. A bucket of one
    column holding more than `entries` is streamed instead: its entries
    arrived input by input, each input's in row order, so they are sorted
    already -- unless some input stored that column out of order, which a
    first pass over the rows finds, and then it is sorted whole after all.
    Otherwise a gene expressed in most of a large atlas's cells would put
    millions of entries in memory at once.
    """
    for b, (majors, minors, values) in enumerate(sorter.buckets):
        n = int(majors.shape[0])
        if n == 0:
            continue
        one_column = int(sorter.bounds[b + 1] - sorter.bounds[b]) == 1
        if one_column and n > entries and _in_row_order(minors, n, entries):
            for lo in range(0, n, entries):
                yield (np.asarray(minors[lo : lo + entries], dtype=np.int64),
                       np.asarray(values[lo : lo + entries]))
            continue
        major = np.asarray(majors[...], dtype=np.int64)
        minor = np.asarray(minors[...], dtype=np.int64)
        order = np.lexsort((minor, major))
        yield minor[order], np.asarray(values[...])[order]


def _concat_csc(
    dst_parent: Any,
    name: str,
    inputs: _Inputs,
    path: str,
    col_maps: List[np.ndarray],
    row_counts: List[int],
    n_rows: int,
    n_cols: int,
    chunk_rows: int,
) -> None:
    """Concatenate CSC matrices row-wise, keeping the CSC encoding.

    Each output column holds that gene's entries from every input, so none
    is complete until every input has been read. Reading one column of
    every input at a time, as an earlier version did, made 2 x n_var x
    n_inputs reads, each decompressing a whole chunk (on sharded Zarr,
    fetching the shard index too) to take a few hundred entries: 37 times
    the input's size read on h5ad, 572 times on Zarr. Instead each input is
    read once, in blocks of entries sized by `chunk_rows`, and its entries
    scattered into column-range buckets on disk, as a streaming transpose
    does (`core/csr_build.py`); the buckets are then read back in turn,
    sorted and appended. The output's column pointers are known before any
    entry is read, from the inputs' own.
    """
    from adata.core.csr_build import BucketSorter
    from adata.core.subset import _append, _growable_like

    group = _sparse_group(dst_parent, name, spec.CSC_MATRIX, (n_rows, n_cols))
    dtype = np.result_type(*[s["data"].dtype for s in inputs.each(path)])
    index_dtype, pointer_dtype = _concat_index_dtypes(inputs.each(path), n_rows)
    template = inputs.get(0, path)
    out_data = _growable_like(group, "data", dtype, template["data"])
    out_indices = _growable_like(group, "indices", index_dtype, template["indices"])

    counts = np.zeros(n_cols, dtype=np.int64)
    for source, col_map in zip(inputs.each(path), col_maps):
        kept = col_map >= 0
        counts[col_map[kept]] += np.diff(np.asarray(source["indptr"][...], dtype=np.int64))[kept]
    indptr = np.concatenate(([0], np.cumsum(counts)))
    total = int(indptr[-1])

    if total:
        entries = _entries_per_block(total, n_rows, chunk_rows)
        with _bucket_scratch(dst_parent) as scratch:
            sorter = BucketSorter(scratch, _csc_bounds(indptr, entries), dtype)
            offset = 0
            for i, (col_map, n_src_rows) in enumerate(zip(col_maps, row_counts)):
                source = inputs.get(i, path)
                src_indptr = np.asarray(source["indptr"][...], dtype=np.int64)
                src_data, src_indices = source["data"], source["indices"]
                stored = int(src_indptr[-1]) if src_indptr.size else 0
                for lo in range(0, stored, entries):
                    hi = min(lo + entries, stored)
                    cols = np.searchsorted(src_indptr, np.arange(lo, hi), side="right") - 1
                    target = col_map[cols]
                    keep = target >= 0
                    rows = np.asarray(src_indices[lo:hi], dtype=np.int64)[keep] + offset
                    values = np.asarray(src_data[lo:hi])[keep].astype(dtype, copy=False)
                    sorter.add(target[keep], rows, values)
                offset += n_src_rows
            for rows, values in _csc_runs(sorter, entries):
                _append(out_indices, rows.astype(index_dtype, copy=False))
                _append(out_data, values)

    _write_indptr(group, indptr, pointer_dtype, inputs.get(0, path)["indptr"])


def check_matrix_encodings(
    inputs: _Inputs, console: Console, row_counts: Optional[Sequence[int]] = None
) -> None:
    """Fail before writing anything if a matrix cannot be concatenated.

    Checked up front rather than mid-write: discovering this half way through
    would leave a partial store behind, and skipping the matrix would produce
    an output silently missing X. A matrix whose rows are not its input's
    obs is refused here too, rather than written out of step with obs.
    """
    def _check(label: str, path: str) -> None:
        kinds: Dict[str, None] = {}
        rows: List[int] = []
        for source in inputs.each(path):
            kind = _matrix_kind(source)
            kinds.setdefault(kind, None)
            rows.append(_shape(source)[0] if kind != "other" else -1)
        if set(kinds) in ({spec.CSR_MATRIX}, {spec.CSC_MATRIX}, {"dense"}):
            for file, got, expected in zip(inputs.files, rows, row_counts or rows):
                if got != expected:
                    raise ValueError(
                        f"Cannot concatenate {label!r}: '{file}' holds {got:,} rows "
                        f"of it but {expected:,} obs."
                    )
            return
        wanted = sorted(kinds)[0]
        raise ValueError(
            f"Cannot concatenate {label!r}: inputs use "
            f"{', '.join(sorted(kinds))}. Every input must use the same "
            f"encoding. Recast them to match first, e.g. "
            f"`adata recast INPUT {label} -o converted.h5ad --layout "
            f"{'dense' if wanted == 'dense' else wanted.removesuffix('_matrix')}`."
        )

    n = len(inputs)
    has_x = [inputs.get(i, "X") is not None for i in range(n)]
    if all(has_x):
        _check("X", "X")
    elif any(has_x):
        console.print("[yellow]Skipping X: not present in every input[/]")

    for name in _index_union([inputs.keys(i, "layers") for i in range(n)]):
        path = f"layers/{name}"
        if all(source is not None for source in inputs.each(path)):
            _check(path, path)

    # obsm entries are stacked only when every input stores them alike; a
    # sparse entry dense elsewhere is refused here, as X would be, rather
    # than dropped. Entries of different widths are skipped later, whatever
    # their encodings, and dataframes are not matrices at all.
    for name in _index_union([inputs.keys(i, "obsm") for i in range(n)]):
        path = f"obsm/{name}"
        if not all(source is not None for source in inputs.each(path)):
            continue
        kinds, widths = set(), set()
        for source in inputs.each(path):
            kinds.add(_matrix_kind(source))
            if _matrix_kind(source) != "other":
                widths.add(_trailing_shape(source))
        if kinds & spec.SPARSE_TYPES and kinds <= spec.SPARSE_TYPES | {"dense"} and len(widths) == 1:
            _check(path, path)


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
    inputs: _Inputs,
    path: str,
    col_maps: List[np.ndarray],
    row_counts: List[int],
    n_rows: int,
    n_cols: int,
    chunk_rows: int,
    fill_value: float,
) -> None:
    """Concatenate X or one layer across inputs.

    Encodings are validated by :func:`check_matrix_encodings` before the output
    store exists, so anything reaching here is concatenable.
    """
    kinds = {_matrix_kind(s) for s in inputs.each(path)}

    if kinds == {spec.CSR_MATRIX}:
        _concat_sparse(dst_parent, name, inputs, path, col_maps, n_rows, n_cols, chunk_rows)
    elif kinds == {spec.CSC_MATRIX}:
        _concat_csc(dst_parent, name, inputs, path, col_maps, row_counts, n_rows, n_cols, chunk_rows)
    elif kinds == {"dense"}:
        _concat_dense(dst_parent, name, inputs, path, col_maps, n_rows, n_cols, chunk_rows,
                      fill_value)
    else:
        # A backstop. `check_matrix_encodings` runs before the output store is
        # created and should have raised already; this catches an element it
        # does not cover, where failing late still beats writing nonsense.
        raise ValueError(
            f"Cannot concatenate {name!r}: inputs use {', '.join(sorted(kinds))}."
        )


def _concat_obsm(
    dst_parent: Any,
    name: str,
    inputs: _Inputs,
    path: str,
    n_rows: int,
    chunk_rows: int,
    console: Console,
    *,
    lengths: List[int],
    join: str = "inner",
    target_obs: Optional[List[str]] = None,
) -> bool:
    """Concatenate an obsm entry row-wise; its columns are not an axis.

    A dataframe in every input (a SCope import's `RegulonsAUC`) is
    concatenated as obs is, its columns following `join`.
    """
    if target_obs is not None and all(
        spec.encoding_type(s) == spec.DATAFRAME for s in inputs.each(path)
    ):
        _concat_frame(dst_parent, name, inputs, path, lengths, join, target_obs, console)
        return True
    kinds = {_matrix_kind(s) for s in inputs.each(path)}
    if "other" in kinds:
        console.print(f"[yellow]Skipping obsm/{name}: not an array, or not a dataframe, in every input[/]")
        return False

    widths = {_trailing_shape(s) for s in inputs.each(path)}
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
        same = [np.arange(trailing[0], dtype=np.int64)] * len(inputs)
        if kinds == {spec.CSR_MATRIX}:
            _concat_sparse(dst_parent, name, inputs, path, same, n_rows, trailing[0], chunk_rows)
        else:
            _concat_csc(dst_parent, name, inputs, path, same, lengths, n_rows, trailing[0],
                        chunk_rows)
        return True

    dtype = np.result_type(*[s.dtype for s in inputs.each(path)])
    shape = (n_rows,) + trailing
    out = create_dataset(dst_parent, name, shape=shape, dtype=dtype,
                         **_layout_like(inputs.get(0, path), dst_parent, shape, dtype))
    spec.set_encoding(out, spec.ARRAY)

    offset = 0
    for source in inputs.each(path):
        for start in range(0, source.shape[0], chunk_rows):
            end = min(start + chunk_rows, source.shape[0])
            out[offset + start : offset + end, ...] = source[start:end, ...]
        offset += source.shape[0]
    return True


def _concat_frame(
    dst_parent: Any,
    name: str,
    inputs: _Inputs,
    path: str,
    lengths: List[int],
    join: str,
    target_obs: List[str],
    console: Console,
) -> None:
    """Dataframes stacked row-wise, as obs is: their columns follow `join`."""
    per_input = [dataframe_columns(s, resolve_index(s, "obs")[1]) for s in inputs.each(path)]
    if join == "inner":
        shared = set(per_input[0]).intersection(*map(set, per_input[1:]))
        names = [c for c in per_input[0] if c in shared]
    else:
        names = _index_union(per_input)
    frame = write_dataframe_header(dst_parent, name, target_obs, [])
    written = [
        column for column in names
        if _concat_obs_column(frame, column, _Column(inputs, path, column), lengths, console)
    ]
    set_column_order(frame, written)


# ---------------------------------------------------------------------------
# varm and varp


def _digest(hasher: Any, values: np.ndarray) -> None:
    """Feed values to `hasher` exactly: bytes for numbers, so NaN equals NaN,
    and the text itself for strings, never object pointers."""
    if values.dtype.kind in "OSUT":
        for text in decode_str_array(values).reshape(-1).tolist():
            data = text.encode("utf-8", "surrogatepass")
            hasher.update(len(data).to_bytes(8, "little") + data)
    else:
        hasher.update(np.ascontiguousarray(values).tobytes())


def _array_snapshot(
    obj: Any, rows: np.ndarray, block_rows: int, columns: Optional[np.ndarray] = None
) -> Any:
    """A digest of an array on the target var, read a block of rows at a
    time: its rows put in the target's order, and for a pairwise entry its
    columns too. A digest, not the values, so that comparing an entry holds
    one block in memory rather than every input's whole entry at once."""
    hasher = hashlib.blake2b(digest_size=32)
    for start in range(0, len(rows), block_rows):
        block = aligned_rows(obj, rows[start:start + block_rows])
        _digest(hasher, block if columns is None else block[:, columns])
    kind = "text" if is_string_dtype(obj.dtype) else str(obj.dtype)
    shape = tuple(obj.shape[1:]) if columns is None else (len(columns),)
    return ("array", kind, shape, hasher.digest())


def _varm_snapshot(obj: Any, where: np.ndarray, block_rows: int) -> Any:
    """A comparable value of a varm entry, on the target var: a digest of an
    array, read a block of rows at a time; of each column, for a dataframe."""
    if is_dataset(obj):
        return _array_snapshot(obj, where, block_rows)
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
    shape = (len(take),) + tuple(obj.shape[1:])
    text = is_string_dtype(obj.dtype)
    out = create_dataset(parent, name, shape=shape, dtype=obj.dtype,
                         **_layout_like(obj, parent, shape, obj.dtype))
    spec.set_encoding(out, spec.STRING_ARRAY if text else spec.ARRAY)
    for start in range(0, len(take), block_rows):
        out[start:start + block_rows, ...] = aligned_rows(obj, take[start:start + block_rows])


def _entries_on(obj: Any, col_map: np.ndarray, entries: int) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A sparse pairwise entry's stored entries on the target var, a block
    of `entries` at a time in the order they are stored: `(major, minor,
    values)`, both renumbered by `col_map`, and those of genes the target
    lacks left out."""
    indptr = np.asarray(obj["indptr"][...], dtype=np.int64)
    stored = int(indptr[-1]) if indptr.size else 0
    for lo in range(0, stored, entries):
        hi = min(lo + entries, stored)
        major = col_map[np.searchsorted(indptr, np.arange(lo, hi), side="right") - 1]
        minor = col_map[np.asarray(obj["indices"][lo:hi], dtype=np.int64)]
        values = np.asarray(obj["data"][lo:hi])
        keep = (major >= 0) & (minor >= 0)
        yield major[keep], minor[keep], values[keep]


#: Seeds for the two halves of a sparse entry's 128-bit digest.
_SEEDS = (0x243F6A8885A308D3, 0x13198A2E03707344)


def _splitmix(x: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, which spreads every input bit over the output."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _sparse_snapshot(obj: Any, col_map: np.ndarray, block_rows: int) -> Any:
    """A comparable value of a sparse pairwise entry on the target var.

    anndata compares sparse entries by value: a CSR and a CSC entry holding
    the same values are equal, float32 equals float64, and a stored zero is
    no different from an absent one. So the digest is of the nonzero
    `(row, column, value)` triples, each value as float64, and it is the sum
    of each triple's hash: the order the entries are stored in, and the
    input's gene order, do not matter, and they are read a block at a time.
    """
    csr = spec.encoding_type(obj) == spec.CSR_MATRIX
    shape = _shape(obj)
    nnz = int(obj["data"].shape[0])
    entries = _entries_per_block(nnz, shape[0], block_rows)
    totals = np.zeros(len(_SEEDS), dtype=np.uint64)
    count = 0
    for major, minor, values in _entries_on(obj, col_map, entries):
        nonzero = values != 0
        rows, cols = (major, minor) if csr else (minor, major)
        rows, cols = rows[nonzero].astype(np.uint64), cols[nonzero].astype(np.uint64)
        bits = values[nonzero].astype(np.float64)
        bits[np.isnan(bits)] = np.nan  # one NaN, whatever its payload
        bits = bits.view(np.uint64)
        for k, seed in enumerate(_SEEDS):
            mixed = _splitmix(_splitmix(_splitmix(rows ^ np.uint64(seed)) ^ cols) ^ bits)
            totals[k:k + 1] += mixed.sum(dtype=np.uint64)
        count += int(rows.size)
    return ("sparse", count, tuple(int(t) for t in totals))


def _pairwise_snapshot(obj: Any, take: np.ndarray, col_map: np.ndarray, block_rows: int) -> Any:
    """A comparable value of a varp entry on the target var."""
    if is_dataset(obj):
        return _array_snapshot(obj, take, block_rows, columns=take)
    if spec.encoding_type(obj) in spec.SPARSE_TYPES:
        return _sparse_snapshot(obj, col_map, block_rows)
    return _Incomparable()


def _write_pairwise(
    parent: Any, name: str, obj: Any, take: np.ndarray, col_map: np.ndarray, block_rows: int,
    console: Console,
) -> None:
    """A varp entry on the target var, rows and columns alike, a block at a
    time. A sparse entry already on the target's genes is copied as it is;
    one in another order is rebuilt through buckets on disk, as a
    transpose is, its entries read a block at a time in stored order."""
    from adata.core.csr_build import CsrAppender, write_runs

    n = len(take)
    if is_dataset(obj):
        out = create_dataset(parent, name, shape=(n, n), dtype=obj.dtype,
                             **_layout_like(obj, parent, (n, n), obj.dtype))
        spec.set_encoding(out, spec.STRING_ARRAY if is_string_dtype(obj.dtype) else spec.ARRAY)
        for start in range(0, n, block_rows):
            out[start:start + block_rows, :] = aligned_rows(obj, take[start:start + block_rows])[:, take]
        return
    enc = spec.encoding_type(obj)
    if enc not in spec.SPARSE_TYPES:
        console.print(f"[yellow]Skipping varp/{name}: neither an array nor a sparse matrix[/]")
        return
    if len(take) == _shape(obj)[0] and np.array_equal(take, np.arange(len(take))):
        copy_tree(obj, parent, name)
        return

    indptr = np.asarray(obj["indptr"][...], dtype=np.int64)
    kept = col_map >= 0
    # Entries a major keeps, counted before those of genes the target lacks
    # are dropped: an estimate, which only balances the buckets.
    counts = np.zeros(n, dtype=np.int64)
    counts[col_map[kept]] = np.diff(indptr)[kept]
    entries = _entries_per_block(int(indptr[-1]) if indptr.size else 0, n, block_rows)
    appender = CsrAppender(
        parent, name, shape=(n, n), data_dtype=obj["data"].dtype,
        index_dtype=_widen_to_hold(obj["indices"].dtype, n),
        templates=(obj["data"], obj["indices"], obj["indptr"]),
    )
    write_runs(
        _entries_on(obj, col_map, entries), n_rows=n, n_targets=1,
        destinations=[[appender]], scratch_parent=parent, grouped=False, canonical=False,
        counts=counts, value_dtype=obj["data"].dtype, chunk=entries, sum_duplicates=False,
    )
    appender.finish()
    # Built by major, which is what CSC's columns are too: n x n either way.
    spec.set_encoding(appender.group, enc)


def _write_varm(
    dst: Any,
    inputs: _Inputs,
    positions: List[np.ndarray],
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
    names = _index_union([inputs.keys(i, "varm") for i in range(len(inputs))])
    if not names:
        return
    if merge is None:
        console.print(
            "[yellow]Dropping varm: pass --merge to keep the entries the inputs agree on.[/]"
        )
        return
    varm = write_mapping(dst, "varm")
    complete = _per_var_list(positions, lambda where: not (where < 0).any())
    compares = merge in ("same", "unique")
    for name in names:
        path = f"varm/{name}"

        def aligned() -> Iterator[Any]:
            for obj, where, whole in zip(inputs.each(path), positions, complete):
                if obj is None or not whole:
                    yield _MISSING
                elif compares:
                    yield _varm_snapshot(obj, where, chunk_rows)
                else:
                    yield _PRESENT

        source = _merge_choice(aligned(), merge)
        if source is None:
            continue
        obj, take = inputs.get(source, path), positions[source]
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


def _write_varp(
    dst: Any,
    inputs: _Inputs,
    positions: List[np.ndarray],
    col_maps: List[np.ndarray],
    merge: Optional[str],
    console: Console,
    chunk_rows: int = 1024,
) -> None:
    """varp entries the merge strategy keeps, on the target var, as varm's.

    varp relates genes, not cells, so it is merged as anndata merges it,
    like varm. It used to be dropped with obsp, whose pairwise values do
    have no meaning between cells from different inputs.
    """
    names = _index_union([inputs.keys(i, "varp") for i in range(len(inputs))])
    if not names:
        return
    if merge is None:
        console.print(
            "[yellow]Dropping varp: pass --merge to keep the entries the inputs agree on.[/]"
        )
        return
    varp = write_mapping(dst, "varp")
    complete = _per_var_list(positions, lambda where: not (where < 0).any())
    compares = merge in ("same", "unique")
    for name in names:
        path = f"varp/{name}"

        def aligned() -> Iterator[Any]:
            for obj, where, col_map, whole in zip(inputs.each(path), positions, col_maps, complete):
                if obj is None or not whole:
                    yield _MISSING
                elif compares:
                    yield _pairwise_snapshot(obj, where, col_map, chunk_rows)
                else:
                    yield _PRESENT

        source = _merge_choice(aligned(), merge)
        if source is not None:
            _write_pairwise(varp, name, inputs.get(source, path), positions[source],
                            col_maps[source], chunk_rows, console)


# ---------------------------------------------------------------------------
# orchestration


def _merge_group(
    dst_parent: Any,
    name: str,
    inputs: _Inputs,
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
    holders = [i for i in range(len(inputs)) if inputs.get(i, name) is not None]
    # An input without the mapping holds none of its keys, so `same` keeps
    # nothing, as anndata's intersection with an empty `uns` does.
    if strategy == "same" and len(holders) < len(inputs):
        holders = []
    _write_merged(target, _merge_plan(inputs, holders, name, strategy), inputs)


def _is_mapping(obj: Any) -> bool:
    """A group anndata reads as a dict: tagged so, or untagged (0.7)."""
    return (
        is_group(obj)
        and spec.encoding_type(obj) in (spec.DICT, None)
        and "h5sparse_format" not in obj.attrs
    )


def _merge_plan(inputs: _Inputs, holders: List[int], path: str, strategy: str) -> Dict[str, Any]:
    """What to keep of the mapping at `path` in the inputs `holders`, merged:
    for each key, `(input, path)` of the member to copy, or a nested plan for
    a mapping. `same` considers the keys every holder has, the other
    strategies the keys any holder has; only `same` and `unique` compare
    values, `first` and `only` decide by presence."""
    owners: Dict[str, List[int]] = {}
    for i in holders:
        for key in inputs.keys(i, path):
            owners.setdefault(key, []).append(i)
    if strategy == "same":
        owners = {k: v for k, v in owners.items() if len(v) == len(holders)}

    compares = strategy in ("same", "unique")
    plan: Dict[str, Any] = {}
    for key, members in owners.items():
        member = f"{path}/{key}"
        if all(_is_mapping(inputs.get(i, member)) for i in members):
            nested = _merge_plan(inputs, members, member, strategy)
            if nested:
                plan[key] = nested
            continue
        chosen = _merge_choice(
            (_readable_value(inputs.get(i, member)) if compares else _PRESENT for i in members),
            strategy,
        )
        if chosen is not None:
            plan[key] = (members[chosen], member)
    return plan


def _write_merged(target: Any, plan: Dict[str, Any], inputs: _Inputs) -> None:
    for key, value in plan.items():
        if isinstance(value, dict):
            _write_merged(write_mapping(target, key), value, inputs)
        else:
            copy_tree(inputs.get(*value), target, key)


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


def _scratch_size(inputs: _Inputs) -> int:
    """The most scratch space concat needs at once: for the largest CSC
    matrix, each stored entry's row and column as int64 and its value."""
    n = len(inputs)
    paths = ["X"] + [f"{group}/{name}" for group in ("layers", "obsm")
                     for name in _index_union([inputs.keys(i, group) for i in range(n)])]
    largest = 0
    for path in paths:
        nnz, dtypes = 0, []
        for source in inputs.each(path):
            if source is None or spec.encoding_type(source) != spec.CSC_MATRIX:
                break
            nnz += int(source["data"].shape[0])
            dtypes.append(source["data"].dtype)
        else:
            largest = max(largest, nnz * (16 + np.result_type(*dtypes).itemsize))
    return largest


def _check_space(inputs: _Inputs, output: Path, check: bool) -> None:
    """Refuse to start if the disk cannot hold the output and the scratch.

    The output needs about the sum of the inputs (an inner join keeps less,
    an outer join's dense fill can add some), and a CSC matrix's buckets go
    in an HDF5 file beside it. A full disk part way through an HDF5 write
    can abort inside the library with no cleanup, so the scratch is checked
    whatever the output is; a Zarr output itself is not, nor its inputs
    walked to size it, as `check_free_space` explains.
    """
    if not check:
        return
    if detect_backend(output) == "hdf5":
        check_free_space(
            output, lambda: _scratch_size(inputs) + sum(disk_size(f) for f in inputs.files)
        )
        return
    scratch = _scratch_size(inputs)
    if scratch:
        check_free_space(output.with_name(f".{output.name}.scratch.h5"), scratch)


def _inputs_zarr_format(formats: Sequence[Optional[int]]) -> Optional[int]:
    """The Zarr version the inputs agree on, or None to take the default.

    A Zarr output follows its inputs, as every other command's follows its
    source, so v2 inputs are not silently upgraded. h5ad inputs have no
    version and do not count; Zarr inputs of both versions get the default.
    """
    versions = set(formats) - {None}
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
    space_check: bool = True,
) -> None:
    """Concatenate stores along the obs axis, streaming each one in turn.

    `join` aligns var: "inner" keeps only variables present in every input,
    "outer" keeps their union. obs columns follow the same rule. Elements of
    the var axis (var's columns, varm, varp) and of uns are reconciled by
    `merge`/`uns_merge`, since inputs may legitimately disagree about them.

    obsp is not concatenated -- a pairwise value has no meaningful value
    between cells that came from different inputs, and anndata's own
    implementation refuses it too.
    """
    if len(files) < 2:
        raise ValueError("Concatenation needs at least two input stores.")
    if join not in ("inner", "outer"):
        raise ValueError("--join must be 'inner' or 'outer'.")
    if output.exists():
        raise FileExistsError(f"'{output}' already exists.")

    input_keys = _resolve_keys(files, keys)

    with _Inputs(files) as inputs:
        var_names, obs_counts, target_obs, formats = _read_axes(inputs, input_keys, index_unique)
        _refuse_repeated_var_names(files, var_names)

        target_var = (
            _index_intersection(var_names) if join == "inner" else _var_union(var_names)
        )
        # Inputs with no variables at all concatenate to n_obs x 0, as in
        # anndata; the refusal is for inputs that have some but share none.
        if not target_var and any(var_names):
            raise ValueError(
                "No variables are shared by all inputs. Use --join outer to "
                "keep their union instead."
            )

        col_maps = _per_var_list(var_names, lambda names: _column_map(names, target_var))
        positions = _per_var_list(var_names, lambda names: _column_map(target_var, names))
        n_obs = sum(obs_counts)
        n_var = len(target_var)

        console.print(
            f"[cyan]Concatenating {len(files)} stores: "
            f"{n_obs} obs x {n_var} var ({join} join)[/]"
        )
        for f, key, count in zip(files, input_keys, obs_counts):
            console.print(f"  [dim]{key}[/]: {count} obs from {f.name}")

        if len(set(target_obs)) != len(target_obs):
            console.print(
                "[yellow]Warning: obs names are not unique across inputs. "
                "Pass --index-unique to disambiguate them.[/]"
            )

        # Validated before the output exists, so a mismatch cannot leave a
        # half-written store behind.
        check_matrix_encodings(inputs, console, obs_counts)
        _check_space(inputs, output, space_check)

        if zarr_format is None:
            zarr_format = _inputs_zarr_format(formats)

        # Written beside the output and renamed into place once complete, so
        # a failure part way leaves nothing that reads as a finished store.
        with staged_outputs([output]) as (temp,), open_store(
            temp, "w", zarr_format=zarr_format
        ) as dst_store:
            dst = dst_store.root
            _write_obs(
                dst, inputs, target_obs, obs_counts, join, label, input_keys, console
            )
            _write_var(dst, inputs, positions, target_var, merge, console)
            ensure_anndata_skeleton(dst)

            _write_matrices(
                dst,
                inputs,
                col_maps,
                obs_counts,
                n_obs,
                n_var,
                chunk_rows,
                fill_value,
                console,
            )
            _write_obsm(dst, inputs, n_obs, chunk_rows, console,
                        lengths=obs_counts, join=join, target_obs=target_obs)
            _write_varm(dst, inputs, positions, target_var, merge, console, chunk_rows)
            _write_varp(dst, inputs, positions, col_maps, merge, console, chunk_rows)

            if any(inputs.get(i, "uns") is not None for i in range(len(inputs))):
                if uns_merge is None:
                    write_mapping(dst, "uns")
                else:
                    _merge_group(dst, "uns", inputs, uns_merge, console)

            if any(inputs.keys(i, "obsp") for i in range(len(inputs))):
                console.print(
                    "[yellow]Dropping obsp: pairwise values have no "
                    "meaning across concatenated inputs.[/]"
                )
            if any(inputs.get(i, "raw") is not None for i in range(len(inputs))):
                console.print("[yellow]Dropping raw/: not concatenated.[/]")

    console.print(f"[green]Wrote[/] {output} ({n_obs} obs x {n_var} var)")


def _write_obs(
    dst: Any,
    inputs: _Inputs,
    target_obs: List[str],
    obs_counts: List[int],
    join: str,
    label: Optional[str],
    input_keys: List[str],
    console: Console,
) -> None:
    """Write the concatenated obs frame, plus the batch column if requested."""
    per_input_cols = [
        dataframe_columns(g, resolve_index(g, "obs")[1]) for g in inputs.each("obs")
    ]

    if join == "inner":
        shared = set(per_input_cols[0]).intersection(*map(set, per_input_cols[1:]))
        names = [c for c in per_input_cols[0] if c in shared]
    else:
        names = _index_union(per_input_cols)

    written: List[str] = []
    obs_group = write_dataframe_header(dst, "obs", target_obs, [])

    for name in names:
        if _concat_obs_column(obs_group, name, _Column(inputs, "obs", name), obs_counts, console):
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
    inputs: _Inputs,
    positions: List[np.ndarray],
    target_var: List[str],
    merge: Optional[str],
    console: Console,
) -> None:
    """Write the aligned var frame, keeping columns the merge strategy allows."""
    var_group = write_dataframe_header(dst, "var", target_var, [])
    if merge is None:
        return

    per_input_cols = [
        dataframe_columns(g, resolve_index(g, "var")[1]) for g in inputs.each("var")
    ]
    candidates = _index_union(per_input_cols)
    complete = _per_var_list(positions, lambda where: not (where < 0).any())
    written: List[str] = []

    # "first" and "only" decide on presence alone, so the column values are
    # never read for them.
    compares = merge in ("same", "unique")

    for name in candidates:

        def aligned() -> Iterator[Any]:
            for (column, group), where, whole in zip(_Column(inputs, "var", name), positions, complete):
                if column is None or not whole:
                    yield _MISSING
                elif not compares:
                    yield _PRESENT
                else:
                    # Read the column once and index the result. Reading it
                    # inside the generator -- as an earlier version did --
                    # re-read the whole column for every target variable,
                    # which is quadratic and turns a 36k-var merge into hours
                    # of pure CPU.
                    values = read_str_all(column, parent_group=group)
                    yield tuple(values[i] for i in where)

        source = _merge_choice(aligned(), merge)
        if source is None:
            continue
        group = inputs.get(source, "var")
        _write_var_column(var_group, name, group[name], positions[source], group)
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
    inputs: _Inputs,
    col_maps: List[np.ndarray],
    row_counts: List[int],
    n_obs: int,
    n_var: int,
    chunk_rows: int,
    fill_value: float,
    console: Console,
) -> None:
    """Concatenate X and every layer shared by all inputs."""
    if all(x is not None for x in inputs.each("X")):
        _concat_matrix(dst, "X", inputs, "X", col_maps, row_counts, n_obs, n_var,
                       chunk_rows, fill_value)

    layer_names = _index_union([inputs.keys(i, "layers") for i in range(len(inputs))])
    if not layer_names:
        return

    layers = write_mapping(dst, "layers")
    for name in layer_names:
        path = f"layers/{name}"
        if not all(layer is not None for layer in inputs.each(path)):
            console.print(
                f"[yellow]Skipping layer {name!r}: not present in every input[/]"
            )
            continue
        _concat_matrix(layers, name, inputs, path, col_maps, row_counts, n_obs, n_var,
                       chunk_rows, fill_value)


def _write_obsm(
    dst: Any,
    inputs: _Inputs,
    n_obs: int,
    chunk_rows: int,
    console: Console,
    *,
    lengths: List[int],
    join: str = "inner",
    target_obs: Optional[List[str]] = None,
) -> None:
    """Concatenate obsm entries shared by all inputs."""
    names = _index_union([inputs.keys(i, "obsm") for i in range(len(inputs))])
    if not names:
        return

    obsm = write_mapping(dst, "obsm")
    for name in names:
        path = f"obsm/{name}"
        if not all(entry is not None for entry in inputs.each(path)):
            console.print(
                f"[yellow]Skipping obsm/{name}: not present in every input[/]"
            )
            continue
        _concat_obsm(
            obsm, name, inputs, path, n_obs, chunk_rows, console,
            lengths=lengths, join=join, target_obs=target_obs,
        )

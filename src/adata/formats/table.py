"""Delimited text tables, read in two streaming passes.

Per-cell metadata arrives as CSV (Parse's `cell_metadata.csv`) with a row
per cell, so a mega run's table has a million rows. `pd.read_csv` would hold
it whole; here the file is read a block of lines at a time, twice.

The first pass finds each column's type as `pd.read_csv` would: `int64`,
`float64` when a number is missing or fractional, `bool` for a column of
True/False, and text otherwise, with pandas' own list of strings that mean
"missing". The second writes the columns as anndata would write that pandas
frame: a text column becomes a categorical when it repeats a value
(anndata's `strings_to_categoricals`), and a string array when every value
is distinct.

Two limits, both deliberate:

* A text column with more than `CATEGORY_CAP` distinct values is written as
  strings, even when some repeat, since tracking its values would hold the
  column in memory.
* A quoted field may not contain a newline: the file is cut into blocks at
  line ends. Such a row is refused with its line number.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from adata.elements import spec
from adata.elements.strings import string_dtype_for
from adata.elements.write import create_categorical, set_column_order, write_string_array
from adata.formats.textio import iter_line_blocks, open_binary
from adata.storage import create_dataset, is_zarr_group
from adata.util.names import natural_sorted

#: Distinct values tracked per text column before it is left as strings.
CATEGORY_CAP = 1 << 16

#: Bytes of a table parsed at once. Smaller than for MTX, because a parsed
#: row is a list of Python strings, several times the size of its text.
TABLE_BLOCK_BYTES = 1 << 20

#: What `pd.read_csv` reads as missing, by default.
NA_VALUES = frozenset({
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
    "nan", "null",
})

_TRUE = frozenset({"True", "TRUE", "true"})
_FALSE = frozenset({"False", "FALSE", "false"})

# A column's kind narrows as values are seen: bool -> int -> float -> text.
_ORDER = ("bool", "int", "float", "text")


@dataclass
class Column:
    name: str
    kind: str = "bool"
    has_missing: bool = False
    #: Distinct values of a text column, None once past the cap.
    seen: Optional[Set[str]] = field(default_factory=set)

    @property
    def categories(self) -> Optional[List[str]]:
        """The categories, in the order anndata gives them."""
        return None if self.seen is None else natural_sorted(self.seen)


@dataclass
class TableScan:
    path: Path
    delimiter: str
    header: List[str]
    index: str
    n_rows: int
    columns: List[Column]
    #: Rows (0-based, after the header) where `select` matched, if asked.
    selected: Optional[np.ndarray] = None

    def dtype_of(self, column: Column) -> str:
        """What the column is written as: int64, float64, bool, category or str."""
        if column.kind == "int":
            return "float64" if column.has_missing else "int64"
        if column.kind == "float":
            return "float64"
        if column.kind == "bool":
            if not column.seen:
                return "float64"  # nothing but missing values, as pandas reads it
            # pandas reads True/False with gaps as text.
            return "category" if column.has_missing else "bool"
        return "category" if self.is_categorical(column) else "str"

    def is_categorical(self, column: Column) -> bool:
        if column.kind not in ("text", "bool") or column.seen is None:
            return False
        # anndata's rule: fewer categories (missing values are none) than
        # rows. So a column of distinct values and one gap is categorical.
        return len(column.seen) < self.n_rows


def _rows(
    path: Path, delimiter: str, block_bytes: int
) -> Iterator[Tuple[int, List[List[str]]]]:
    """Blocks of parsed rows, each with the file line number of its first."""
    line = 1
    with open_binary(path) as fh:
        for block in iter_line_blocks(fh, block_bytes):
            reader = csv.reader(io.StringIO(block.decode("utf-8")), delimiter=delimiter, strict=True)
            rows: List[List[str]] = []
            start = line
            try:
                for row in reader:
                    rows.append(row)
            except csv.Error as exc:
                raise ValueError(
                    f"{path.name}, line {start + reader.line_num - 1}: {exc}. "
                    "A quoted field may not span lines."
                ) from exc
            line += block.count(b"\n")
            yield start, rows


def _narrow(column: Column, values: List[str]) -> None:
    """Narrow `column`'s kind to cover `values`."""
    present = [v for v in values if v not in NA_VALUES]
    if len(present) < len(values):
        column.has_missing = True
    kind = _ORDER.index(column.kind)
    while present and kind < 3:
        try:
            if kind == 0:
                if not all(v in _TRUE or v in _FALSE for v in present):
                    raise ValueError
            elif kind == 1:
                np.array(present, dtype=np.int64)
            else:
                np.array(present, dtype=np.float64)
            break
        except (ValueError, OverflowError):
            kind += 1
    column.kind = _ORDER[kind]
    if column.seen is not None:
        column.seen.update(present)
        if len(column.seen) > CATEGORY_CAP:
            column.seen = None


def scan_table(
    path: Path,
    *,
    index: Optional[str] = None,
    delimiter: str = ",",
    select: Optional[Tuple[str, str]] = None,
    block_bytes: int = TABLE_BLOCK_BYTES,
) -> TableScan:
    """Pass 1: the header, the row count and each column's type.

    `index` names the column that becomes the row names (the first, if not
    given). `select=(column, value)` also records the rows where that column
    holds that value.
    """
    path = Path(path)
    header: Optional[List[str]] = None
    columns: List[Column] = []
    positions: List[int] = []
    pick = -1
    selected: List[np.ndarray] = []
    n_rows = 0
    for start, rows in _rows(path, delimiter, block_bytes):
        if header is None:
            rows = [row for row in rows if row]
            if not rows:
                continue
            header, rows = rows[0], rows[1:]
            if len(set(header)) != len(header):
                repeated = sorted({h for h in header if header.count(h) > 1})
                raise ValueError(f"{path.name} repeats the column {', '.join(map(repr, repeated))}.")
            index = header[0] if index is None else index
            if index not in header:
                raise ValueError(f"{path.name} has no {index!r} column; it has: {', '.join(header)}.")
            positions = [i for i, name in enumerate(header) if name != index]
            columns = [Column(header[i]) for i in positions]
            if select is not None:
                if select[0] not in header:
                    raise ValueError(f"{path.name} has no {select[0]!r} column.")
                pick = header.index(select[0])
            start += 1
        _check_widths(path, rows, len(header), start)
        rows = [row for row in rows if row]
        if not rows:
            continue
        by_field = list(zip(*rows))
        for column, i in zip(columns, positions):
            _narrow(column, list(by_field[i]))
        if pick >= 0:
            hits = np.flatnonzero(np.asarray(by_field[pick], dtype=object) == select[1])
            selected.append(hits + n_rows)
        n_rows += len(rows)
    if header is None:
        raise ValueError(f"{path.name} is empty: it has no header line.")
    return TableScan(
        path=path,
        delimiter=delimiter,
        header=header,
        index=index,
        n_rows=n_rows,
        columns=columns,
        selected=(
            np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)
        ) if select is not None else None,
    )


def _check_widths(path: Path, rows: List[List[str]], width: int, start: int) -> None:
    for offset, row in enumerate(rows):
        if row and len(row) != width:
            raise ValueError(
                f"{path.name}, line {start + offset}: {len(row)} fields where "
                f"the header has {width}."
            )


def iter_table_blocks(
    scan: TableScan,
    *,
    rows: Optional[np.ndarray] = None,
    block_bytes: int = TABLE_BLOCK_BYTES,
) -> Iterator[Tuple[List[str], Dict[str, List[str]]]]:
    """Pass 2: (index values, {column: values}) a block of rows at a time.

    `rows`, ascending, keeps only those rows.
    """
    at = scan.header.index(scan.index)
    done = 0
    first = True
    for _, block in _rows(scan.path, scan.delimiter, block_bytes):
        block = [row for row in block if row]
        if first and block:
            block, first = block[1:], False
        if not block:
            continue
        if rows is not None:
            lo, hi = np.searchsorted(rows, (done, done + len(block)))
            picked = [block[i] for i in rows[lo:hi] - done]
            done += len(block)
            block = picked
            if not block:
                continue
        else:
            done += len(block)
        by_field = list(zip(*block))
        yield list(by_field[at]), {
            name: list(by_field[i]) for i, name in enumerate(scan.header) if name != scan.index
        }


def _values(scan: TableScan, column: Column, values: List[str], lookup: Dict[str, int]) -> np.ndarray:
    kind = scan.dtype_of(column)
    if kind == "int64":
        return np.array(values, dtype=np.int64)
    if kind == "float64":
        return np.array([np.nan if v in NA_VALUES else v for v in values], dtype=np.float64)
    if kind == "bool":
        return np.array([v in _TRUE for v in values], dtype=bool)
    if kind == "category":
        return np.array([lookup.get(v, -1) for v in values], dtype=np.int64)
    return np.array(["" if v in NA_VALUES else v for v in values], dtype=object)


def write_table(
    parent: Any,
    name: str,
    scan: TableScan,
    *,
    rows: Optional[np.ndarray] = None,
    block_bytes: int = TABLE_BLOCK_BYTES,
) -> Any:
    """Pass 2: the table as a `dataframe` named `name` under `parent`.

    Every array is created at its full length first and filled a block of
    rows at a time, so memory is set by the block, not by the table.
    """
    n = scan.n_rows if rows is None else int(rows.size)
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    group = parent.create_group(name)
    spec.set_encoding(group, spec.DATAFRAME)
    group.attrs["_index"] = "_index"
    set_column_order(group, [c.name for c in scan.columns])

    index = create_dataset(group, "_index", shape=(n,), dtype=string_dtype_for(backend))
    spec.set_encoding(index, spec.STRING_ARRAY)
    targets: Dict[str, Any] = {}
    lookups: Dict[str, Dict[str, int]] = {}
    for column in scan.columns:
        kind = scan.dtype_of(column)
        if kind == "category":
            categories = column.categories or []
            targets[column.name] = create_categorical(group, column.name, categories, n)
            lookups[column.name] = {c: i for i, c in enumerate(categories)}
            continue
        dtype = string_dtype_for(backend) if kind == "str" else np.dtype(kind)
        targets[column.name] = create_dataset(group, column.name, shape=(n,), dtype=dtype)
        spec.set_encoding(
            targets[column.name], spec.STRING_ARRAY if kind == "str" else spec.ARRAY
        )

    used = {name: np.zeros(len(lookup), dtype=bool) for name, lookup in lookups.items()}
    written = 0
    for names, block in iter_table_blocks(scan, rows=rows, block_bytes=block_bytes):
        end = written + len(names)
        index[written:end] = np.asarray(names, dtype=object)
        for column in scan.columns:
            values = _values(scan, column, block[column.name], lookups.get(column.name, {}))
            targets[column.name][written:end] = values
            if column.name in used:
                used[column.name][values[values >= 0]] = True
        written = end
    if written != n:
        raise ValueError(f"{scan.path.name} changed while it was read: {written:,} rows, not {n:,}.")
    if rows is not None:
        for name, seen in used.items():
            if not seen.all():
                _drop_unused(group, name, seen, n, block_rows=max(1, block_bytes // 8))
    return group


def _drop_unused(group: Any, name: str, used: np.ndarray, n: int, block_rows: int) -> None:
    """Remove categories no written row uses, as anndata does on subsetting."""
    remap = np.full(used.size + 1, -1, dtype=np.int64)  # the last slot maps -1
    remap[:-1][used] = np.arange(int(used.sum()))
    categorical = group[name]
    categories = np.asarray(categorical["categories"][...], dtype=object)[used]
    del categorical["categories"]
    write_string_array(categorical, "categories", categories)
    codes = categorical["codes"]
    for lo in range(0, n, block_rows):
        block = np.asarray(codes[lo : lo + block_rows], dtype=np.int64)
        codes[lo : lo + block.size] = remap[block]


def read_small_table(path: Path, delimiter: str = ",") -> Tuple[List[str], List[List[str]]]:
    """The header and every row of a table small enough to hold, such as a
    gene list."""
    header: Optional[List[str]] = None
    body: List[List[str]] = []
    for start, rows in _rows(Path(path), delimiter, TABLE_BLOCK_BYTES):
        if header is None:
            rows = [row for row in rows if row]
            if not rows:
                continue
            header, rows = rows[0], rows[1:]
            start += 1
        _check_widths(Path(path), rows, len(header), start)
        body.extend(row for row in rows if row)
    if header is None:
        raise ValueError(f"{Path(path).name} is empty: it has no header line.")
    return header, body


def write_text_column(
    parent: Any,
    name: str,
    n: int,
    blocks: Callable[[], Iterable[Sequence[str]]],
    *,
    missing: Sequence[str] = (),
) -> Any:
    """A text column of `n` values, from `blocks()` called twice, as anndata
    would write it: a categorical when it repeats a value, else strings.

    Pass 1 collects the distinct values, up to `CATEGORY_CAP`; pass 2 writes
    codes in anndata's (natural) category order, or the strings themselves.
    Values in `missing` are gaps: no category, code -1.
    """
    gaps = frozenset(missing)
    seen: Optional[Set[str]] = set()
    for block in blocks():
        seen.update(v for v in block if v not in gaps)
        if len(seen) > CATEGORY_CAP:
            seen = None
            break
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    if seen is not None and len(seen) < n:
        categories = natural_sorted(seen)
        lookup = {c: i for i, c in enumerate(categories)}
        target = create_categorical(parent, name, categories, n)
        convert = lambda block: np.array([lookup.get(v, -1) for v in block], dtype=np.int64)  # noqa: E731
    else:
        target = create_dataset(parent, name, shape=(n,), dtype=string_dtype_for(backend))
        spec.set_encoding(target, spec.STRING_ARRAY)
        convert = lambda block: np.asarray(list(block), dtype=object)  # noqa: E731
    written = 0
    for block in blocks():
        values = convert(block)
        target[written : written + values.size] = values
        written += values.size
    if written != n:
        raise ValueError(f"Column {name!r} has {written:,} values, not {n:,}.")
    return parent[name]

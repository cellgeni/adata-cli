"""Delimited text matrices, as anndata's `read_csv` and `read_text` read them.

A dense matrix in text, one line per row: GEO count tables, R's
`write.table` and `write.csv` output, hand-made CSVs. The layout is found
as anndata's `_read_text` finds it:

* lines starting with `#` before the data are comments; with no header
  line, the last one names the columns;
* the first line is a header when its last field is not a number;
* the first column names the rows when its first value is not a number. A
  header with a field more than the values has a corner cell, and one with
  as many (R's `write.table`) names only the columns;
* names have `"` stripped; blank lines are skipped.

`--transpose` reads a file that holds genes as rows, as count tables often
do, and `--first-column-names` / `--no-first-column-names` overrides the
row-name detection.

Values keep integers: a file whose every value is whole gives int32 or
int64, as MTX import chooses; any other gives float64, which holds what the
text holds. `recast` can narrow either.

The file is read twice, a block of lines at a time, each block parsed by
numpy with Python work only per line, to cut off its name:

1. in `plan`: the layout, every row's width and values checked, the dtype,
   and with `--transpose` each output row's non-zeros;
2. in `write`: X, CSR (`--dense` keeps it dense), straight from the blocks;
   transposed, through `core.csr_build`'s on-disk buckets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import Target, write_obs_index, write_var
from adata.core.csr_build import CsrAppender, scratch_group, write_runs
from adata.core.recast import densify, describe
from adata.elements import spec
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_mapping,
    write_scalar,
    write_string_array,
)
from adata.formats.textio import TEXT_BLOCK_BYTES, iter_line_blocks, open_binary
from adata.storage import (
    COMPRESSIONS,
    create_dataset,
    is_zarr_group,
    open_store,
    with_compression,
)
from adata.util.names import count_duplicates, make_index_unique

#: Extensions read as text, each also gzipped; and the delimiter each implies.
EXTENSIONS = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".txt": None, ".data": None}

#: `--delimiter` spellings. None is anndata's `delimiter=None`: runs of
#: whitespace.
DELIMITERS = {",": ",", "comma": ",", "tab": "\t", "\t": "\t", "\\t": "\t",
              ";": ";", "whitespace": None}

#: Bytes of text read at once. Parsed, a block takes several times its size
#: (floats, their coordinates), so it is smaller than MTX's.
BLOCK_BYTES = TEXT_BLOCK_BYTES >> 2

#: Bytes of a dense X written per chunk of rows.
DENSE_CHUNK_BYTES = 64 << 20

#: Entries per bucket read when sorting a transposed file into rows.
BUCKET_ENTRIES = 1 << 20


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def extension_delimiter(path: Path) -> Optional[str]:
    """The delimiter a file's extension implies (`.gz` aside); `,` when none does."""
    name = Path(path).name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    for extension, delimiter in EXTENSIONS.items():
        if name.endswith(extension):
            return delimiter
    return ","


def _describe_delimiter(delimiter: Optional[str]) -> str:
    return {None: "whitespace", "\t": "tab", ",": "comma"}.get(delimiter, repr(delimiter))


@dataclass
class Layout:
    """Where the data starts in a file, and what names it."""

    delimiter: Optional[str]
    #: Non-blank lines before the data: comments and the header.
    skip: int
    #: Column names, as given (corner cell dropped); None to number them.
    columns: Optional[List[str]]
    #: "line", "comment" or "none": where the column names came from.
    header: str
    row_names: bool
    #: Values per line, the row name not counted.
    width: int
    comments: List[str] = field(default_factory=list)

    def split(self, text: str) -> List[str]:
        return text.split(self.delimiter) if self.delimiter is not None else text.split()


def _lines(path: Path, block_bytes: int) -> Iterator[Tuple[List[int], List[bytes]]]:
    """Each block's non-blank lines, without their line ends, and their numbers."""
    number = 0
    with open_binary(path) as fh:
        for block in iter_line_blocks(fh, block_bytes):
            numbers, lines = [], []
            for line in block.split(b"\n"):
                number += 1
                line = line[:-1] if line.endswith(b"\r") else line
                if line:
                    numbers.append(number)
                    lines.append(line)
            if block.endswith(b"\n"):
                number -= 1  # the empty piece after the last newline is no line
            if lines:
                yield numbers, lines


def _first_lines(path: Path, n: int) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for numbers, lines in _lines(path, BLOCK_BYTES):
        for number, line in zip(numbers, lines):
            text = line.decode("utf-8", "replace")
            found.append((number, text))
            # Comments do not count towards the lines wanted.
            if sum(not t.startswith("#") for _, t in found) >= n:
                return found
    return found


def detect(path: Path, delimiter: Optional[str], first_column_names: Optional[bool]) -> Layout:
    """The layout of a text matrix, from its first lines, as anndata finds it."""
    name = Path(path).name
    head = _first_lines(path, 2)
    comments, skip = [], 0
    for _, text in head:
        if not text.startswith("#"):
            break
        comment = text.lstrip("# ")
        if comment:
            comments.append(comment)
        skip += 1
    data = [text for _, text in head[skip:]]
    if not data:
        raise ValueError(f"{name} holds no data: no line but comments or blank ones.")
    first = data[0]
    if delimiter is not None and delimiter not in first:
        raise ValueError(
            f"{name}: no {_describe_delimiter(delimiter)} in its first line. "
            "Name the delimiter with --delimiter."
        )
    layout = Layout(delimiter, skip, None, "none", False, 0, comments)
    one = layout.split(first)
    two = layout.split(data[1]) if len(data) > 1 else None
    numeric_header = (
        two is not None and len(two) == len(one) + 1 and all(_is_float(f) for f in one)
    )
    if not _is_float(one[-1]) or numeric_header:
        # anndata reads a numeric first line one field short of the next as
        # column names: the integers R gives unnamed columns.
        layout.header, layout.skip = "line", skip + 1
        columns = [
            str(int(float(f))) if numeric_header else f for f in one
        ]
        values = two
    else:
        columns, values = None, one
    if values is None:
        raise ValueError(f"{name} holds a header and no data.")
    if first_column_names is None:
        layout.row_names = numeric_header or not _is_float(values[0])
    else:
        layout.row_names = first_column_names
    layout.width = len(values) - layout.row_names
    if layout.width < 1:
        raise ValueError(f"{name}: its first data line holds no values.")
    if columns is None and comments:
        columns, layout.header = comments[-1].split(), "comment"
    if columns is not None:
        if len(columns) == layout.width + 1:
            columns = columns[1:]  # the corner cell, over the row names
        if len(columns) != layout.width:
            raise ValueError(
                f"{name}: the header holds {len(columns):,} fields, but the rows "
                f"hold {layout.width:,} values: it should hold {layout.width:,}, or "
                f"{layout.width + 1:,} with a corner over the row names."
            )
        layout.columns = [c.strip('"') for c in columns]
    return layout


@dataclass
class Scan:
    """What the first pass learned."""

    n_rows: int
    whole: bool
    vmin: float
    vmax: float
    #: Row names, when they are held (the genes of a transposed file).
    names: Optional[List[str]]
    #: Non-zeros per output row, for a transposed file.
    counts: Optional[np.ndarray]

    def dtype(self) -> np.dtype:
        """int32 or int64 when every value is whole (and exact), else float64."""
        if self.whole and self.n_rows and max(abs(self.vmin), abs(self.vmax)) < 2.0**53:
            info = np.iinfo(np.int32)
            if info.min <= self.vmin and self.vmax <= info.max:
                return np.dtype(np.int32)
            return np.dtype(np.int64)
        return np.dtype(np.float64)


class _Reader:
    """Parses a text matrix a block of lines at a time."""

    def __init__(self, path: Path, layout: Layout, *, floats: bool = False) -> None:
        self.path = Path(path)
        self.layout = layout
        delimiter = layout.delimiter
        self.raw = None if delimiter is None else delimiter.encode()
        # np.fromstring's `sep`: a space matches any run of whitespace.
        self.sep = " " if delimiter is None else delimiter
        #: Whether to parse floats from the start. Integers parse three
        #: times as fast, so they are tried until a block holds another value.
        self.floats = floats

    def blocks(self) -> Iterator[Tuple[List[str], np.ndarray]]:
        """(row names, values as an n x width float64 array), a block at a time."""
        skip = self.layout.skip
        for numbers, lines in _lines(self.path, BLOCK_BYTES):
            if skip:
                taken = min(skip, len(lines))
                numbers, lines, skip = numbers[taken:], lines[taken:], skip - taken
                if not lines:
                    continue
            yield self._parse(numbers, lines)

    def _parse(self, numbers: List[int], lines: List[bytes]) -> Tuple[List[str], np.ndarray]:
        names: List[str] = []
        if self.layout.row_names:
            rests = []
            for number, line in zip(numbers, lines):
                parts = line.split(self.raw, 1)
                if len(parts) < 2:
                    raise ValueError(
                        f"{self.path.name}: line {number:,} holds a name and no values."
                    )
                names.append(parts[0].decode("utf-8", "replace").strip('"'))
                rests.append(parts[1])
            lines = rests
        width = self.layout.width
        text = b"\n".join(lines)
        if (_fields(text, self.raw, len(lines)) != width).any():
            self._diagnose(numbers, lines)
        if self.raw is not None:
            text = text.replace(b"\n", self.raw)
        values = None
        if not self.floats:
            values = self._numbers(text, np.int64, len(lines) * width)
            info = np.iinfo(np.int64)
            if values is None or values.size != len(lines) * width or (
                # fromstring saturates at the limits rather than failing.
                values.size and (values.max() == info.max or values.min() == info.min)
            ):
                values, self.floats = None, True
        if values is None:
            values = self._numbers(text, np.float64, len(lines) * width)
        if values is None or values.size != len(lines) * width:
            self._diagnose(numbers, lines)
        return names, values.reshape(len(lines), width)

    def _numbers(self, text: bytes, dtype: Any, count: int) -> Optional[np.ndarray]:
        """`count` numbers from `text`, whose fields have been counted.

        The count matters for memory as well as speed: without one, numpy
        grows its buffer as it parses, and the fragments kept the process
        hundreds of megabytes above the block.
        """
        with warnings.catch_warnings():
            # A token that is no number ends the parse early, with a warning.
            warnings.simplefilter("error", DeprecationWarning)
            try:
                return np.fromstring(text, dtype=dtype, sep=self.sep, count=count)
            except (ValueError, DeprecationWarning):
                return None

    def _diagnose(self, numbers: List[int], lines: List[bytes]) -> None:
        """Find the line that would not parse, and refuse it by number."""
        name, width = self.path.name, self.layout.width
        for number, line in zip(numbers, lines):
            fields = self.layout.split(line.decode("utf-8", "replace"))
            if len(fields) != width:
                raise ValueError(
                    f"{name}: line {number:,} holds {len(fields):,} values, not "
                    f"{width:,} as the first data line does."
                )
            for column, value in enumerate(fields, start=1 + self.layout.row_names):
                if not _is_float(value):
                    raise ValueError(
                        f"{name}: line {number:,}, column {column:,}: {value!r} is "
                        "not a number."
                    )
        raise ValueError(f"{name}: a line from line {numbers[0]:,} on is not numbers.")


_SPACE = np.zeros(256, dtype=bool)
_SPACE[[9, 10, 11, 12, 13, 32]] = True


def _fields(text: bytes, delimiter: Optional[bytes], n_lines: int) -> np.ndarray:
    """Fields on each line of `text`, every line at once: the delimiters
    between line ends, or, for whitespace, the starts of runs of others."""
    buffer = np.frombuffer(text, dtype=np.uint8)
    ends = np.concatenate(([0], np.flatnonzero(buffer == 10), [buffer.size]))
    if delimiter is not None:
        marks, extra = np.flatnonzero(buffer == delimiter[0]), 1
    else:
        space = _SPACE[buffer]
        starts = ~space
        starts[1:] &= space[:-1]
        marks, extra = np.flatnonzero(starts), 0
    return np.diff(np.searchsorted(marks, ends)) + extra


def scan(path: Path, layout: Layout, transpose: bool) -> Scan:
    """Pass 1: every line checked, the value range, and what pass 2 needs."""
    reader = _Reader(path, layout)
    n_rows, whole = 0, True
    vmin, vmax = np.inf, -np.inf
    names: Optional[List[str]] = [] if transpose else None
    counts = np.zeros(layout.width, dtype=np.int64) if transpose else None
    for block_names, values in reader.blocks():
        n_rows += values.shape[0]
        if values.size:
            vmin, vmax = min(vmin, float(values.min())), max(vmax, float(values.max()))
            if whole and values.dtype.kind == "f":
                whole = bool(np.isfinite(values).all() and np.array_equal(values, np.trunc(values)))
        if names is not None:
            names.extend(block_names or [str(i) for i in range(n_rows - values.shape[0], n_rows)])
        if counts is not None:
            counts += np.count_nonzero(values, axis=0)
    return Scan(n_rows, whole, vmin, vmax, names, counts)


@dataclass
class _Import:
    layout: Layout
    scanned: Scan
    transpose: bool
    dense: bool
    obs_names: Optional[List[str]]
    var_names: List[str]
    shape: Tuple[int, int]
    dtype: np.dtype
    compression: Optional[str]
    zarr_format: Optional[int]


class TextImport(Converter):
    options = frozenset({
        "--delimiter", "--transpose", "--first-column-names", "--dense",
        "--compression", "--make-unique",
    })

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )
        if options.delimiter is None:
            delimiter = extension_delimiter(source)
        elif options.delimiter in DELIMITERS:
            delimiter = DELIMITERS[options.delimiter]
        elif len(options.delimiter) == 1:
            delimiter = options.delimiter
        else:
            raise ValueError(
                f"--delimiter {options.delimiter!r}: give one character, or "
                "tab, comma or whitespace."
            )
        layout = detect(source, delimiter, options.first_column_names)
        scanned = scan(source, layout, options.transpose)

        warnings: List[str] = []
        columns = layout.columns or [str(i) for i in range(layout.width)]
        if layout.columns is None:
            warnings.append(
                f"No header names the columns; they are numbered 0..{layout.width - 1}."
            )
        if options.transpose:
            obs_names, var_names = columns, scanned.names
            shape = (layout.width, scanned.n_rows)
        else:
            obs_names, var_names = None, columns
            shape = (scanned.n_rows, layout.width)
        if options.make_unique:
            var_names = make_index_unique(var_names)
        elif count_duplicates(var_names):
            warnings.append(
                f"{count_duplicates(var_names):,} var names repeat an earlier one. "
                "Pass --make-unique to suffix them as anndata's "
                "var_names_make_unique does."
            )
        dtype = scanned.dtype()
        detail = _Import(
            layout=layout,
            scanned=scanned,
            transpose=options.transpose,
            dense=options.dense,
            obs_names=obs_names,
            var_names=var_names,
            shape=shape,
            dtype=dtype,
            compression=options.compression or ("gzip" if backend == "hdf5" else None),
            zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
        )
        names = "row names" if layout.row_names else "no row names"
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"text ({_describe_delimiter(delimiter)}, {names}"
                f"{', transposed' if options.transpose else ''}) → {dst_format}, "
                f"{shape[0]:,} cells x {shape[1]:,} genes, {dtype}"
                f"{', dense' if options.dense else ''}"
            ),
            warnings=warnings,
            detail=detail,
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Import = plan.detail
        reader = _Reader(plan.source, detail.layout, floats=detail.dtype.kind == "f")
        with open_store(staged[0], "w", zarr_format=detail.zarr_format) as store:
            root = store.root
            write_var(root, Target(label=None, keep=None, var_names=detail.var_names, columns={}))
            ensure_anndata_skeleton(root)
            _write_uns(root["uns"], detail)
            with console.status(f"[cyan]Reading {Path(plan.source).name}...[/]"):
                if detail.transpose:
                    obs = _write_names(root, detail.obs_names)
                    _write_transposed(root, reader, detail)
                else:
                    obs = write_obs_index(root, detail.shape[0], _rows(root, reader, detail))
            set_column_order(obs, [])


def _write_names(root: Any, names: List[str]) -> Any:
    step = 1 << 18
    return write_obs_index(
        root, len(names), (names[lo : lo + step] for lo in range(0, len(names), step))
    )


def _dense_target(parent: Any, name: str, detail: _Import) -> Any:
    n_rows, n_cols = detail.shape
    rows = max(1, min(n_rows, DENSE_CHUNK_BYTES // max(1, n_cols * detail.dtype.itemsize)))
    kw: dict = {"chunks": (rows, max(n_cols, 1))}
    if detail.compression is not None:
        backend = "zarr" if is_zarr_group(parent) else "hdf5"
        kw = with_compression(
            kw, detail.compression, shape=detail.shape, dtype=detail.dtype,
            backend=backend, zarr_format=detail.zarr_format,
        )
    target = create_dataset(parent, name, shape=detail.shape, dtype=detail.dtype, **kw)
    spec.set_encoding(target, spec.ARRAY)
    return target


def _rows(root: Any, reader: _Reader, detail: _Import) -> Iterator[List[str]]:
    """Cells as rows: each block's names, for obs, with X written as they go."""
    append: Callable[[int, np.ndarray], None]
    if detail.dense:
        target = _dense_target(root, "X", detail)

        def append(lo: int, block: np.ndarray) -> None:
            target[lo : lo + block.shape[0]] = block
    else:
        appender = CsrAppender(
            root, "X", shape=detail.shape, data_dtype=detail.dtype,
            compression=detail.compression,
        )

        def append(lo: int, block: np.ndarray) -> None:
            rows, cols = np.nonzero(block)
            appender.append(rows + lo, cols, block[rows, cols])

    lo = 0
    for names, values in reader.blocks():
        append(lo, values.astype(detail.dtype))
        yield names or [str(i) for i in range(lo, lo + values.shape[0])]
        lo += values.shape[0]
    if not detail.dense:
        appender.finish()


def _write_transposed(root: Any, reader: _Reader, detail: _Import) -> None:
    """Genes as rows: each block's entries sorted into cell rows through buckets."""

    def entries() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        lo = 0
        for _, values in reader.blocks():
            genes, cells = np.nonzero(values)
            yield cells, genes + lo, values[genes, cells].astype(detail.dtype)
            lo += values.shape[0]

    def build(parent: Any, name: str) -> None:
        appender = CsrAppender(
            parent, name, shape=detail.shape, data_dtype=detail.dtype,
            compression=detail.compression,
        )
        write_runs(
            entries(), n_rows=detail.shape[0], n_targets=1, destinations=[[appender]],
            scratch_parent=root, grouped=False, canonical=True,
            counts=detail.scanned.counts, value_dtype=detail.dtype, chunk=BUCKET_ENTRIES,
        )
        appender.finish()

    if not detail.dense:
        build(root, "X")
        return
    with scratch_group(root, "__text_scratch__") as scratch:
        build(scratch, "csr")
        rows = max(1, DENSE_CHUNK_BYTES // max(1, detail.shape[1] * detail.dtype.itemsize))
        densify(describe(scratch["csr"]), root, "X", data_dtype=detail.dtype, chunk_rows=rows)


def _write_uns(uns: Any, detail: _Import) -> None:
    layout = detail.layout
    meta = write_mapping(uns, "text")
    write_scalar(meta, "delimiter", _describe_delimiter(layout.delimiter))
    write_scalar(meta, "header", layout.header)
    write_scalar(meta, "row_names", bool(layout.row_names))
    write_scalar(meta, "transposed", bool(detail.transpose))
    if layout.comments:
        write_string_array(meta, "comments", layout.comments)

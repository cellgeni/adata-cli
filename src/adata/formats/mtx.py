"""Matrix Market (`.mtx`) files, read and written a block at a time.

Only the coordinate format with `general` symmetry, the one every
single-cell tool writes, and the `integer`, `real` and `pattern` fields.
Everything else is refused by name rather than half read: a `symmetric` file
stores one triangle and means both, so reading it as `general` silently
loses half the matrix.

Reading is two passes over the file, with memory set by the block rather
than by nnz:

1. `scan` counts the entries per output row, notes whether they arrive in
   row order, and finds the value range, which picks the dtype;
2. `read_into` reads the entries again and appends them to the output in CSR
   order -- directly when they arrived in order, through on-disk buckets
   (`core.csr_build`) when they did not.

`EntryMap` describes how entries in the file become entries of the output:
10x and STARsolo store genes x cells, so a 10x import reads each file column
as an output row, and can split the features among several outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from adata.core.csr_build import (
    CsrAppender,
    count_keys,
    sum_keys,
    write_runs,
)
from adata.core.recast import DEFAULT_CHUNK, Matrix
from adata.elements import spec
from adata.formats.textio import (
    TEXT_BLOCK_BYTES,
    format_rows,
    iter_line_blocks,
    open_binary,
)

BANNER = "%%MatrixMarket"
FIELDS = ("integer", "real", "pattern")

#: Entries read from a stored matrix per block, when writing one out.
EXPORT_BLOCK_ENTRIES = 1 << 20

#: The bytes numpy's `sep=" "` takes for space: tab, newline, vertical tab,
#: form feed, carriage return and space.
_SPACE = np.zeros(256, dtype=bool)
_SPACE[[9, 10, 11, 12, 13, 32]] = True


@dataclass(frozen=True)
class MtxHeader:
    path: Path
    field: str
    n_rows: int
    n_cols: int
    nnz: int
    comments: Tuple[str, ...] = ()

    @property
    def columns(self) -> int:
        """Fields per line: row, column, and a value unless the field is `pattern`."""
        return 2 if self.field == "pattern" else 3


def _skip_header(fh: BinaryIO) -> Tuple[str, List[str], str]:
    """The banner, the comment lines, and the size line, read from `fh`."""
    banner = fh.readline().decode("utf-8", "replace").strip()
    comments: List[str] = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("The MTX file ends before its size line.")
        text = line.decode("utf-8", "replace").strip()
        if text.startswith("%"):
            comments.append(text)
            continue
        if text:
            return banner, comments, text


def read_header(path: Path) -> MtxHeader:
    """Parse and check a Matrix Market header; refuse what this cannot read."""
    path = Path(path)
    with open_binary(path) as fh:
        banner, comments, size = _skip_header(fh)

    parts = banner.split()
    if not parts or parts[0] != BANNER:
        raise ValueError(
            f"{path.name} is not a Matrix Market file: it does not start with "
            f"{BANNER}."
        )
    if len(parts) != 5 or parts[1].lower() != "matrix":
        raise ValueError(f"{path.name}: unrecognised Matrix Market banner {banner!r}.")
    fmt, fld, symmetry = (p.lower() for p in parts[2:])
    if fmt == "array":
        raise ValueError(
            f"{path.name} is a dense ('array') Matrix Market file; only the "
            "sparse 'coordinate' format is supported."
        )
    if fmt != "coordinate":
        raise ValueError(f"{path.name}: unknown Matrix Market format {fmt!r}.")
    if fld not in FIELDS:
        raise ValueError(
            f"{path.name} holds {fld!r} values; supported fields are "
            f"{', '.join(FIELDS)}."
        )
    if symmetry != "general":
        raise ValueError(
            f"{path.name} is {symmetry!r}: it stores one triangle and means "
            "both, which is not supported. Only 'general' files are."
        )

    try:
        n_rows, n_cols, nnz = (int(v) for v in size.split())
    except ValueError:
        raise ValueError(
            f"{path.name}: the size line should be 'rows columns entries', "
            f"not {size!r}."
        ) from None
    return MtxHeader(path, fld, n_rows, n_cols, nnz, tuple(comments))


def _fields_per_line(text: bytes) -> np.ndarray:
    """The fields on each line of `text`, every line at once: the starts of
    runs of other bytes than space, between one newline and the next."""
    buffer = np.frombuffer(text, dtype=np.uint8)
    space = _SPACE[buffer]
    keep = ~space
    keep[1:] &= space[:-1]  # where a field starts
    newline = np.equal(buffer, 10, out=space)  # space is not needed again
    keep |= newline
    # Field starts and newlines in order: a line's fields lie between two.
    kinds = newline[keep]
    del space, keep
    ends = np.append(np.flatnonzero(kinds), kinds.size)
    return np.diff(ends, prepend=-1) - 1


def _parse(block: bytes, header: MtxHeader, first_line: int) -> np.ndarray:
    """One block of entry lines as an (n, columns) array: int64, or float64 for `real`."""
    dtype = np.float64 if header.field == "real" else np.int64
    text = block.rstrip()
    if not text:
        return np.empty((0, header.columns), dtype=dtype)
    # Line by line, not the block's total: a line a field short and a later
    # one a field long add up, and pair values with the wrong coordinates.
    fields = _fields_per_line(text)
    wrong = np.flatnonzero(fields != header.columns)
    if wrong.size:
        line, found = first_line + int(wrong[0]), int(fields[wrong[0]])
        if not found:
            raise ValueError(
                f"{header.path.name}: line {line:,} is a blank line among the entries."
            )
        raise ValueError(
            f"{header.path.name}: line {line:,} does not hold {header.columns} "
            f"fields: it holds {found}."
        )
    try:
        flat = np.fromstring(text, dtype=dtype, sep=" ")
    except ValueError:
        raise ValueError(
            f"{header.path.name}: a line from line {first_line:,} on is not "
            f"{header.columns} numbers{' (integers)' if dtype is np.int64 else ''}."
        ) from None
    lines = fields.size
    if flat.size != lines * header.columns:  # pragma: no cover - counted above
        raise ValueError(
            f"{header.path.name}: a line from line {first_line:,} on does not "
            f"hold {header.columns} numbers."
        )
    if dtype is np.int64 and flat.size:
        info = np.iinfo(np.int64)
        if flat.max() == info.max or flat.min() == info.min:
            # fromstring saturates rather than failing, so a value at either
            # limit may be one beyond it. Rare enough to check the slow way.
            if any(not info.min <= int(token) <= info.max for token in text.split()):
                raise ValueError(
                    f"{header.path.name}: a value from line {first_line:,} on "
                    "does not fit int64."
                )
    return flat.reshape(lines, header.columns)


def iter_entries(
    header: MtxHeader, *, block_bytes: int = TEXT_BLOCK_BYTES
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The file's entries, a block at a time: 0-based rows and columns, and values.

    Values are int64 for `integer` and `pattern` (all ones) files and float64
    for `real` ones. Coordinates are checked against the size line, and the
    number of entries against the count it gives.
    """
    seen = 0
    with open_binary(header.path) as fh:
        _, comments, _ = _skip_header(fh)
        line = len(comments) + 3
        for block in iter_line_blocks(fh, block_bytes):
            table = _parse(block, header, line)
            n = table.shape[0]
            line += block.count(b"\n")
            if n == 0:
                continue
            if header.field == "real":
                coords = table[:, :2]
                if not np.array_equal(coords, np.trunc(coords)):
                    raise ValueError(
                        f"{header.path.name}: a row or column index is not a "
                        "whole number."
                    )
                rows = coords[:, 0].astype(np.int64) - 1
                cols = coords[:, 1].astype(np.int64) - 1
                values = table[:, 2]
            else:
                rows = table[:, 0] - 1
                cols = table[:, 1] - 1
                values = table[:, 2] if header.columns == 3 else np.ones(n, dtype=np.int64)
            for axis, index, limit in (("row", rows, header.n_rows), ("column", cols, header.n_cols)):
                if index.min() < 0 or index.max() >= limit:
                    raise ValueError(
                        f"{header.path.name}: a {axis} index is outside 1..{limit:,}, "
                        "the size its header gives."
                    )
            seen += n
            if seen > header.nnz:
                break
            yield rows, cols, values
    if seen != header.nnz:
        raise ValueError(
            f"{header.path.name} has {'more than ' if seen > header.nnz else ''}"
            f"{seen:,} entries; its header says {header.nnz:,}."
        )


@dataclass
class EntryMap:
    """How an entry of the file becomes an entry of an output.

    `transpose` reads the file's columns as output rows, as for 10x and
    STARsolo files, which store genes x cells. `owner` and `local`, indexed by
    feature (the output's column axis), split the features among several
    outputs: `owner` gives each feature's output, or -1 to drop it, and
    `local` its column there. Outputs are numbered within one key space,
    `key = output * n_rows + row`, so one sort orders them all. `row_map`,
    indexed by the file's row (after `transpose`), gives each its output row,
    or -1 to drop it; `n_rows` then counts output rows.
    """

    n_rows: int
    n_cols: int
    transpose: bool = False
    owner: Optional[np.ndarray] = None
    local: Optional[np.ndarray] = None
    n_targets: int = 1
    target_cols: List[int] = field(default_factory=list)
    row_map: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.target_cols:
            self.target_cols = [self.n_cols]

    def apply(
        self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_rows, features = (cols, rows) if self.transpose else (rows, cols)
        if self.row_map is not None:
            out_rows = self.row_map[out_rows]
            kept = out_rows >= 0
            out_rows, features, values = out_rows[kept], features[kept], values[kept]
        if self.owner is None:
            return out_rows, features, values
        owner = self.owner[features]
        kept = owner >= 0
        keys = owner[kept] * self.n_rows + out_rows[kept]
        return keys, self.local[features[kept]], values[kept]


@dataclass
class Scan:
    """What the first pass learned."""

    counts: np.ndarray
    grouped: bool
    canonical: bool
    vmin: float
    vmax: float
    #: The largest sum of absolute values in any one output row: a bound on
    #: what summing duplicate entries can reach.
    row_bound: float = 0.0


def scan(
    header: MtxHeader, emap: EntryMap, *, block_bytes: int = TEXT_BLOCK_BYTES
) -> Scan:
    """Pass 1: entries per output row, whether they arrive in order, value range."""
    counts = np.zeros(emap.n_targets * emap.n_rows, dtype=np.int64)
    magnitude = np.zeros(counts.size, dtype=np.float64)
    grouped = canonical = True
    last_key, last_col = -1, -1
    vmin, vmax = np.inf, -np.inf
    for rows, cols, values in iter_entries(header, block_bytes=block_bytes):
        keys, minors, kept = emap.apply(rows, cols, values)
        if keys.size == 0:
            continue
        count_keys(counts, keys)
        sum_keys(magnitude, keys, np.abs(kept).astype(np.float64))
        vmin = min(vmin, float(kept.min()))
        vmax = max(vmax, float(kept.max()))
        if grouped:
            steps = np.diff(keys)
            if keys[0] < last_key or (steps < 0).any():
                grouped = canonical = False
            elif canonical:
                same = steps == 0
                if (keys[0] == last_key and minors[0] <= last_col) or (
                    np.diff(minors)[same] <= 0
                ).any():
                    canonical = False
        last_key, last_col = int(keys[-1]), int(minors[-1])
    row_bound = float(magnitude.max()) if magnitude.size else 0.0
    return Scan(counts, grouped, canonical, vmin, vmax, row_bound)


def value_dtype(header: MtxHeader, scanned: Scan) -> np.dtype:
    """int32 for counts that fit it, int64 for those that do not; float64 for reals.

    float64 rather than float32 because the text may hold more precision than
    float32 keeps, and narrowing is `recast`'s job, which checks it first.

    The choice is made before anything is written, so it must hold after
    duplicate entries are summed. A file with no repeated coordinates
    (`canonical`) has nothing to sum, and its value range decides. Otherwise
    no summed entry can exceed its row's total magnitude, so that decides:
    a sound bound, if sometimes a wider dtype than the sums needed.
    """
    if header.field == "real":
        return np.dtype(np.float64)
    low, high = scanned.vmin, scanned.vmax
    if not scanned.canonical:
        # Every value already fits int64 (the parser checks); only a sum
        # can leave it. Floats, so compare with 2**63 itself: the int64
        # maximum rounds up to it.
        if scanned.row_bound >= 2.0**63:
            raise ValueError(
                f"{header.path.name}: summing its duplicate entries could "
                "exceed int64."
            )
        low, high = min(low, -scanned.row_bound), max(high, scanned.row_bound)
    info = np.iinfo(np.int32)
    if low >= info.min and high <= info.max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def read_into(
    header: MtxHeader,
    emap: EntryMap,
    scanned: Scan,
    destinations: Sequence[Sequence[CsrAppender]],
    scratch_parent: Any,
    *,
    chunk: Optional[int] = None,
    block_bytes: int = TEXT_BLOCK_BYTES,
) -> int:
    """Pass 2: append the file's entries to each output, in CSR order.

    `destinations[t]` are the appenders for output `t`, each of which gets
    every entry of that output. Returns how many duplicate entries were
    summed away.
    """
    return write_runs(
        (emap.apply(*entry) for entry in iter_entries(header, block_bytes=block_bytes)),
        n_rows=emap.n_rows,
        n_targets=emap.n_targets,
        destinations=destinations,
        scratch_parent=scratch_parent,
        grouped=scanned.grouped,
        canonical=scanned.canonical,
        counts=scanned.counts,
        value_dtype=np.float64 if header.field == "real" else np.int64,
        chunk=chunk or DEFAULT_CHUNK,
    )


# ---------------------------------------------------------------------------
# writing


def mtx_field(dtype: np.dtype) -> str:
    kind = np.dtype(dtype).kind
    if kind in "iub":
        return "integer"
    if kind == "f":
        return "real"
    raise ValueError(f"A {dtype} matrix cannot be written as Matrix Market.")


def write_header(
    fh: BinaryIO,
    *,
    field: str,
    n_rows: int,
    n_cols: int,
    nnz: int,
    comments: Sequence[str] = (),
) -> None:
    lines = [f"{BANNER} matrix coordinate {field} general"]
    lines += [c if c.startswith("%") else f"% {c}" for c in comments]
    lines.append(f"{n_rows} {n_cols} {nnz}")
    fh.write(("\n".join(lines) + "\n").encode("utf-8"))


def _major_blocks(
    indptr: np.ndarray, budget: int, max_majors: Optional[int] = None
) -> Iterator[Tuple[int, int]]:
    """(lo, hi) ranges of the major axis holding about `budget` entries each.

    At least one major per block, however many entries it holds, and at most
    `max_majors` when given.
    """
    n_major = indptr.size - 1
    lo = 0
    while lo < n_major:
        target = indptr[lo] + budget
        hi = int(np.searchsorted(indptr, target, side="right")) - 1
        hi = min(max(hi, lo + 1), n_major)
        if max_majors is not None:
            hi = min(hi, lo + max(1, max_majors))
        yield lo, hi
        lo = hi


def iter_matrix_entries(
    matrix: Matrix,
    *,
    transpose: bool,
    block_entries: int = EXPORT_BLOCK_ENTRIES,
    chunk_rows: int = 1024,
    max_majors: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A stored matrix's entries, 0-based, as (rows, columns, values) blocks.

    One read of `indices` and of `data` per block of rows, never one per row.
    With `transpose`, the matrix's columns become the rows -- AnnData's
    cells x genes written as 10x's genes x cells.
    """
    if matrix.sparse:
        indptr = np.asarray(matrix.obj["indptr"][...], dtype=np.int64)
        indices, data = matrix.obj["indices"], matrix.obj["data"]
        for lo, hi in _major_blocks(indptr, block_entries, max_majors):
            start, end = int(indptr[lo]), int(indptr[hi])
            if end <= start:
                continue
            minor = np.asarray(indices[start:end], dtype=np.int64)
            values = np.asarray(data[start:end])
            major = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(indptr[lo : hi + 1]))
            rows, cols = (major, minor) if matrix.kind == spec.CSR_MATRIX else (minor, major)
            yield (cols, rows, values) if transpose else (rows, cols, values)
        return

    n_rows = matrix.shape[0]
    for lo in range(0, n_rows, chunk_rows):
        block = np.asarray(matrix.obj[lo : min(lo + chunk_rows, n_rows)])
        r, c = np.nonzero(block)
        rows = r.astype(np.int64) + lo
        cols = c.astype(np.int64)
        values = block[r, c]
        yield (cols, rows, values) if transpose else (rows, cols, values)


def check_sparse(matrix: Matrix) -> None:
    """Refuse a sparse matrix whose `indptr`, `indices` and `data` disagree."""
    if not matrix.sparse:
        return
    indptr = matrix.obj["indptr"]
    nnz = int(indptr[indptr.shape[0] - 1]) if indptr.shape[0] else 0
    n_data = int(matrix.obj["data"].shape[0])
    if not nnz == n_data == matrix.nnz:
        raise ValueError(
            f"Sparse matrix data inconsistency: indptr implies {nnz} nonzeros, "
            f"but data has {n_data} and indices has {matrix.nnz}."
        )


def count_entries(matrix: Matrix, *, chunk_rows: int = 1024) -> int:
    """The entries a matrix writes: its stored nonzeros, or a dense one's nonzeros."""
    if matrix.sparse:
        return matrix.nnz
    total = 0
    n_rows = matrix.shape[0]
    for lo in range(0, n_rows, chunk_rows):
        total += int(np.count_nonzero(np.asarray(matrix.obj[lo : min(lo + chunk_rows, n_rows)])))
    return total


def write_entries(
    fh: BinaryIO,
    entries: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    pattern: bool = False,
    limit: Optional[int] = None,
) -> int:
    """Write 1-based entry lines; stop after `limit` if given. Returns how many."""
    written = 0
    for rows, cols, values in entries:
        if limit is not None:
            room = limit - written
            if room <= 0:
                break
            rows, cols, values = rows[:room], cols[:room], values[:room]
        fh.write(format_rows([rows + 1, cols + 1], None if pattern else values))
        written += rows.size
    return written

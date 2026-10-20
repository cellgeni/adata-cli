"""Reading and writing large text files a block at a time, gzipped or not.

MTX and TSV files from Cell Ranger and STARsolo run to tens of gigabytes,
and the obvious ways of reading them -- a Python object per line, or
`np.loadtxt` -- cost more in memory or time than the matrix they describe.
Everything here works on blocks of about `TEXT_BLOCK_BYTES`: read a block,
cut it at its last newline, and parse or format the whole block with numpy.

Whether a file is gzipped is decided by its first two bytes, not its name:
Cell Ranger gzips its files and STARsolo does not, and files are renamed in
transit often enough that the suffix is not evidence.
"""

from __future__ import annotations

import gzip
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Sequence, Tuple

import numpy as np

GZIP_MAGIC = b"\x1f\x8b"

#: Bytes of text handled at once: about a million MTX lines.
TEXT_BLOCK_BYTES = 1 << 24


def is_gzip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == GZIP_MAGIC


@contextmanager
def open_binary(path: Path) -> Iterator[BinaryIO]:
    """Open a file for reading as bytes, inflating it if it is gzipped."""
    if is_gzip(path):
        with gzip.open(path, "rb") as fh:
            yield fh
    else:
        with open(path, "rb") as fh:
            yield fh


@contextmanager
def create_binary(path: Path, *, gzipped: bool, level: int = 4) -> Iterator[BinaryIO]:
    """Create a file for writing bytes, gzipped if asked.

    The gzip header's timestamp and file name are left empty, so the same
    content always gives the same bytes.
    """
    with open(path, "wb") as raw:
        if not gzipped:
            yield raw
            return
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=level, mtime=0
        ) as fh:
            yield fh


def iter_line_blocks(
    fh: BinaryIO, block_bytes: int = TEXT_BLOCK_BYTES
) -> Iterator[bytes]:
    """Blocks of whole lines, each ending in a newline but perhaps the last.

    A line longer than a block is gathered in pieces and joined once: adding
    each read to the bytes before it copied them all again, which for a
    header naming a million cells read a block at a time was quadratic.
    """
    carry: List[bytes] = []
    while True:
        chunk = fh.read(block_bytes)
        if not chunk:
            break
        cut = chunk.rfind(b"\n")
        if cut < 0:
            carry.append(chunk)
            continue
        carry.append(chunk[: cut + 1])
        yield b"".join(carry)
        carry = [chunk[cut + 1 :]] if cut + 1 < len(chunk) else []
    if carry:
        yield b"".join(carry)


def _lines(block: bytes) -> List[bytes]:
    lines = block.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return [line[:-1] if line.endswith(b"\r") else line for line in lines]


def count_lines(path: Path, block_bytes: int = TEXT_BLOCK_BYTES) -> int:
    """Lines in a file, not counting blank ones at its end."""
    total = 0
    trailing_blank = 0
    with open_binary(path) as fh:
        for block in iter_line_blocks(fh, block_bytes):
            for line in _lines(block):
                if line.strip():
                    total += trailing_blank + 1
                    trailing_blank = 0
                else:
                    trailing_blank += 1
    return total


def iter_first_column(
    path: Path, rows_per_block: int, block_bytes: int = TEXT_BLOCK_BYTES
) -> Iterator[List[str]]:
    """The first tab-separated field of each line, `rows_per_block` at a time.

    A barcodes file has one column, but some tools add more; only the first
    is the barcode. Blank lines at the end are ignored, as `count_lines` does.
    """
    pending: List[str] = []
    blank = 0
    with open_binary(path) as fh:
        for block in iter_line_blocks(fh, block_bytes):
            for line in _lines(block):
                if not line.strip():
                    blank += 1
                    continue
                if blank:
                    pending.extend([""] * blank)
                    blank = 0
                pending.append(line.split(b"\t", 1)[0].decode("utf-8"))
                if len(pending) >= rows_per_block:
                    yield pending[:rows_per_block]
                    pending = pending[rows_per_block:]
    if pending:
        yield pending


def read_tsv(path: Path) -> List[List[str]]:
    """Every line of a small tab-separated file, split into fields."""
    with open_binary(path) as fh:
        lines = _lines(fh.read())
    while lines and not lines[-1].strip():
        lines.pop()
    return [line.decode("utf-8").split("\t") for line in lines]


# ---------------------------------------------------------------------------
# formatting


def _magnitude(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each integer's absolute value as uint64, and whether it is negative.

    In uint64 so that every int64 and every uint64 has one: `abs` of the
    int64 minimum is itself, and casting a uint64 above the int64 maximum to
    int64 wraps it negative.
    """
    if values.dtype.kind == "u":
        return values.astype(np.uint64), np.zeros(values.shape, dtype=bool)
    signed = values.astype(np.int64)
    negative = signed < 0
    magnitude = signed.astype(np.uint64)
    # Two's complement: the magnitude of a negative value is ~v + 1.
    magnitude[negative] = ~magnitude[negative] + np.uint64(1)
    return magnitude, negative


def _digits(magnitude: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """How many characters each integer takes, its sign included."""
    count = np.ones(magnitude.shape, dtype=np.int64)
    rest = magnitude // np.uint64(10)
    while rest.any():
        count += rest > 0
        rest //= np.uint64(10)
    return count + negative


def format_int_rows(columns: Sequence[np.ndarray], sep: bytes = b" ") -> bytes:
    """Integer columns as text: one line per row, fields separated by `sep`.

    Vectorised: each column becomes a matrix of digit characters, right
    aligned, with a mask dropping the leading padding, so no Python object
    is made per value. Measured at about ten million lines a second, the
    same as %-formatting a tuple per line but without the tuples.
    """
    if len(sep) != 1:
        raise ValueError("format_int_rows needs a one-byte separator.")
    n = int(columns[0].shape[0]) if columns else 0
    if n == 0:
        return b""
    pieces: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for i, column in enumerate(columns):
        magnitude, negative = _magnitude(np.asarray(column))
        widths = _digits(magnitude, negative)
        width = int(widths.max())
        chars = np.empty((n, width), dtype=np.uint8)
        rest = magnitude.copy()
        for k in range(width - 1, -1, -1):
            chars[:, k] = (rest % np.uint64(10)).astype(np.uint8) + ord("0")
            rest //= np.uint64(10)
        if negative.any():
            chars[negative, width - widths[negative]] = ord("-")
        pieces.append(chars)
        masks.append(np.arange(width) >= (width - widths)[:, None])
        last = i == len(columns) - 1
        end = np.full((n, 1), ord("\n") if last else sep[0], dtype=np.uint8)
        pieces.append(end)
        masks.append(np.ones((n, 1), dtype=bool))
    return np.hstack(pieces)[np.hstack(masks)].tobytes()


def float_format(dtype: np.dtype) -> str:
    """The shortest %-format that reads back to exactly the same value."""
    return "%.9g" if np.dtype(dtype).itemsize <= 4 else "%.17g"


def format_rows(
    columns: Sequence[np.ndarray], last: Optional[np.ndarray], sep: bytes = b" "
) -> bytes:
    """Integer `columns`, then an optional value column of any numeric dtype.

    Integer values -- and floats that all hold whole numbers, which is how
    counts usually arrive -- take the vectorised path. Other floats are
    formatted per value, exactly: `%.9g` round-trips a float32, `%.17g` a
    float64.
    """
    if last is None:
        return format_int_rows(columns, sep)
    values = np.asarray(last)
    if values.dtype.kind in "iub":
        return format_int_rows([*columns, values], sep)
    if values.dtype.kind == "f":
        whole = np.isfinite(values).all() and (np.abs(values) < 2**53).all()
        if whole and np.array_equal(values, np.trunc(values)):
            return format_int_rows([*columns, values.astype(np.int64)], sep)
    fmt = float_format(values.dtype)
    sep_text = sep.decode()
    head = format_int_rows(columns, sep).split(b"\n")[:-1]
    return b"".join(
        prefix + (sep_text + (fmt % value) + "\n").encode()
        for prefix, value in zip(head, values.tolist())
    )

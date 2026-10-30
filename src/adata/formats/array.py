"""NumPy `.npy` import and export for `adata import array` / `export array`.

Both stream: the header of a `.npy` names its dtype and shape, and the data
after it is the array's bytes in row order, so an export writes the header
and then one block of rows at a time (to a file or to a pipe alike), and an
import maps the file and copies a block of rows at a time. Memory is set by
the block, never by the array.
"""

from __future__ import annotations

import io
import struct
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.elements import spec
from adata.elements.read import decode_str_array
from adata.elements.write import write_dense, write_mapping, write_scalar
from adata.formats.common import _get_encoding_type, _resolve, stdout_reader_may_stop
from adata.formats.validate import check_target, validate_dimensions
from adata.storage import create_dataset, is_dataset, is_group, is_zarr_group
from adata.elements.strings import is_string_dtype

#: Elements of a `.npy` copied into the store at once by `import array`.
IMPORT_CHUNK_ELEMENTS = 1 << 20


def _row_blocks(shape: Tuple[int, ...], chunk_elements: int) -> Iterator[Tuple[int, int]]:
    """[start, end) row ranges of about `chunk_elements` elements each."""
    n = int(shape[0])
    row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    step = max(1, int(chunk_elements) // max(1, row))
    for start in range(0, n, step):
        yield start, min(start + step, n)


def _source(h5obj: Any, obj: str) -> Tuple[Any, Optional[Any]]:
    """The values to export and, for a nullable array, its mask."""
    if is_dataset(h5obj):
        return h5obj, None
    if not is_group(h5obj):
        raise ValueError("Target is not an array-like object.")
    enc = _get_encoding_type(h5obj)
    if enc not in spec.MASKED_TYPES:
        raise ValueError(
            f"Target '{obj}' is a group with encoding '{enc}'; cannot export as .npy directly."
        )
    if "values" not in h5obj or "mask" not in h5obj:
        raise ValueError(f"Encoded group '{obj}' is missing its 'values' or 'mask' dataset.")
    return h5obj["values"], h5obj["mask"]


def _count_missing(mask: Any, chunk_elements: int) -> int:
    return sum(
        int(np.count_nonzero(mask[start:end]))
        for start, end in _row_blocks(mask.shape, chunk_elements)
    )


def _text_width(values: Any, chunk_elements: int) -> int:
    """The longest value, in characters: the width of the `<U` dtype."""
    if values.shape == ():
        return len(decode_str_array(np.asarray(values[()])).item())
    width = 0
    for start, end in _row_blocks(values.shape, chunk_elements):
        block = decode_str_array(np.asarray(values[start:end]))
        if block.size:
            width = max(width, int(np.char.str_len(block).max()))
    return width


def _npy_header(dtype: np.dtype, shape: Tuple[int, ...]) -> bytes:
    """A .npy header, in the earliest version that can hold it, as np.save
    picks: 1.0, 2.0 past 64 KiB, and 3.0 for field names latin-1 cannot
    spell (`rank_genes_groups` has fields named after cell types)."""
    header = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": tuple(int(d) for d in shape),
    }
    for write in (np.lib.format.write_array_header_1_0, np.lib.format.write_array_header_2_0):
        buffer = io.BytesIO()
        try:
            write(buffer, header)
            return buffer.getvalue()
        except ValueError:  # too long for 1.0; not latin-1 (UnicodeEncodeError)
            continue
    # 3.0 is 2.0 in UTF-8. numpy writes it only through a private function,
    # so it is spelled out: magic, version, a 4-byte length, and the dict,
    # padded with spaces to a multiple of 64 bytes and ended by a newline.
    text = ("{" + "".join(f"{k!r}: {v!r}, " for k, v in header.items()) + "}").encode("utf-8")
    text += b" " * (-(12 + len(text) + 1) % 64) + b"\n"
    return b"\x93NUMPY\x03\x00" + struct.pack("<I", len(text)) + text


def _holds_objects(path: Path) -> bool:
    """Does the .npy at `path` hold Python objects? Read from its header."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        read = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        return read is not None and read(fh)[2].hasobject


def export_npy(
    root: Any,
    obj: str,
    out: Optional[Path],
    chunk_elements: int,
    console: Console,
) -> None:
    """Write a dense array to a .npy file, or to stdout when `out` is None.

    Blocks of about `chunk_elements` elements are read and written in turn,
    to a file or a pipe alike. A nullable integer or boolean array with
    missing values is written as float64 with NaN where they are missing,
    since a .npy has no mask; without any, it keeps its dtype. Text is
    written as fixed-width unicode, as wide as its longest value.
    """
    h5obj = _resolve(root, obj)
    values, mask = _source(h5obj, obj)
    chunk_elements = max(1, int(chunk_elements))
    shape = tuple(values.shape)

    text = is_string_dtype(values.dtype) and values.dtype.kind != "S"
    missing = _count_missing(mask, chunk_elements) if mask is not None else 0
    if text:
        if missing:
            raise ValueError(
                f"'{obj}' has {missing:,} missing values, which a .npy of text "
                "cannot hold. `adata export dataframe` writes them as empty fields."
            )
        dtype = np.dtype(f"<U{max(1, _text_width(values, chunk_elements))}")
    elif values.dtype.hasobject:
        raise ValueError(
            f"'{obj}' is a record array with text fields, which a .npy holds "
            "only pickled."
        )
    elif missing:
        dtype = np.dtype(np.float64)
        console.print(
            f"[yellow]Note:[/] '{obj}' has {missing:,} missing values, written "
            "as NaN, so the array is float64."
        )
    else:
        # Without h5py's metadata (it marks fixed-width bytes as text),
        # which a .npy cannot hold.
        dtype = np.dtype(values.dtype.descr) if values.dtype.names else np.dtype(values.dtype.str)

    def convert(start: Optional[int], end: Optional[int]) -> np.ndarray:
        if start is None:
            block = np.asarray(values[()])
        else:
            block = np.asarray(values[start:end])
        if text:
            block = decode_str_array(block)
        block = np.ascontiguousarray(block, dtype=dtype)
        if missing and start is not None:
            gaps = np.asarray(mask[start:end], dtype=bool)
            block[gaps] = np.nan
        return block

    header = _npy_header(dtype, shape)
    to_stdout = out is None or str(out) == "-"
    if not to_stdout:
        out.parent.mkdir(parents=True, exist_ok=True)
    target = nullcontext(sys.stdout.buffer) if to_stdout else open(out, "wb")
    status = (
        nullcontext()
        if to_stdout
        else console.status(f"[magenta]Exporting {obj} to {out}...[/]")
    )
    with target as fh, status as progress, stdout_reader_may_stop():
        fh.write(header)
        if shape == ():
            fh.write(convert(None, None).tobytes())
        else:
            n = shape[0]
            for start, end in _row_blocks(shape, chunk_elements):
                if progress:
                    progress.update(f"[magenta]Exporting {obj}: {start}-{end} of {n}...[/]")
                fh.write(convert(start, end).tobytes())
    if not to_stdout:
        console.print(f"[green]Wrote[/] {out}")


def import_npy(
    root: Any,
    obj: str,
    input_file: Path,
    console: Console,
    chunk_elements: int = IMPORT_CHUNK_ELEMENTS,
) -> None:
    """Import a .npy file as a dense array, a block of rows at a time.

    The file is mapped, not loaded, so only the rows being copied are held.
    Text (`<U` or `S`) is written as a string array.
    """
    obj = check_target(obj, "array")
    if _holds_objects(input_file):
        raise ValueError(
            f"{Path(input_file).name} holds Python objects, which a .npy stores "
            "pickled; adata does not unpickle them."
        )
    arr = np.load(input_file, mmap_mode="r")

    validate_dimensions(root, obj, arr.shape, console)

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else write_mapping(parent, part)
    name = parts[-1]
    if name in parent:
        del parent[name]

    text = arr.dtype.kind in ("U", "S")
    if arr.shape == ():
        value = arr[()]
        if text:
            write_scalar(parent, name, decode_str_array(np.asarray(value)).item())
        else:
            write_dense(parent, name, np.asarray(value))
    else:
        blocks = list(_row_blocks(arr.shape, chunk_elements))
        rows = max(1, blocks[0][1] - blocks[0][0]) if blocks else 1
        # Zarr's own guess at chunks can span far more rows than a block,
        # and each block would then rewrite whole chunks: a chunk is a block.
        kwargs = {"chunks": (rows, *arr.shape[1:])} if is_zarr_group(parent) else {}
        dtype = str if text else arr.dtype
        target = create_dataset(parent, name, shape=arr.shape, dtype=dtype, **kwargs)
        spec.set_encoding(target, spec.STRING_ARRAY if text else spec.ARRAY)
        for start, end in blocks:
            block = np.asarray(arr[start:end])
            if text:
                block = decode_str_array(block).astype(object)
            target[start:end] = block

    shape_str = "×".join(str(d) for d in arr.shape)
    console.print(f"[green]Imported[/] {shape_str} array into '{obj}'")

"""Import command helpers for creating/replacing objects in h5ad/zarr stores.

An import never edits a store in place. The source is copied beside its
destination (the source itself with --inplace, or -o), the element is
written into the copy, and the copy is moved into place only once that has
succeeded: an import that is refused or fails part way leaves the source and
any existing output as they were.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.formats.array import IMPORT_CHUNK_ELEMENTS, import_npy
from adata.formats.dataframe import import_dataframe
from adata.formats.image import import_image
from adata.formats.json_data import import_json
from adata.formats.mtx import read_header
from adata.formats.sparse import import_mtx
from adata.formats.validate import check_target, validate_dimensions
from adata.storage import copy_path, copy_store_contents, detect_backend, open_store
from adata.util.path import output_target
from adata.util.staging import check_free_space, disk_size, staged_outputs


EXTENSION_FORMAT = {
    ".csv": "csv",
    ".npy": "npy",
    ".mtx": "mtx",
    ".json": "json",
}


def _copy_source(
    file: Path, temp: Path, output: Path, inplace: bool, console: Console
) -> None:
    """Copy `file` to `temp`, converting it if `output` is the other backend."""
    src_backend = detect_backend(file)
    dst_backend = detect_backend(temp)
    if src_backend == dst_backend:
        copy_path(file, temp)
        if not inplace:
            console.print(f"[dim]Copied {file} → {output}[/]")
        return

    with open_store(file, "r") as src_store, open_store(temp, "w") as dst_store:
        copy_store_contents(src_store.root, dst_store.root)
    console.print(
        f"[dim]Converted {file} ({src_backend}) → {output} ({dst_backend})[/]"
    )


def _npy_shape(path: Path) -> Optional[Tuple[int, ...]]:
    """A .npy's shape, read as the import maps it, or None to leave a file
    the import refuses (Python objects, a truncated file) to say so there."""
    try:
        return tuple(np.load(path, mmap_mode="r").shape)
    except (OSError, ValueError):
        return None


@contextmanager
def _source_root(file: Path) -> Iterator[Any]:
    """The source as the import will see it in its copy: a Zarr store's
    members as they are on disk, not as a consolidated index (possibly
    stale) lists them, and no warning about the root's attributes."""
    if detect_backend(file) == "zarr":
        import zarr

        yield zarr.open_group(str(file), mode="r", use_consolidated=False)
        return
    with h5py.File(file, "r") as handle:
        yield handle


def _precheck(file: Path, obj: str, input_file: Path, kind: str) -> None:
    """The refusals an import makes from its input's header, made first.

    Against the source, before it is copied: a matrix of the wrong shape
    used to be found only after a copy of the whole store, which can take
    as long, and as much room, as the store is large. The same checks, in
    the same order, as the import itself, so the message is the same.
    """
    obj = check_target(obj, kind)
    if kind == "sparse":
        header = read_header(input_file)
        shape: Optional[Tuple[int, ...]] = (header.n_rows, header.n_cols)
    else:
        shape = _npy_shape(input_file)
    if shape is None:
        return
    with _source_root(file) as root:
        validate_dimensions(root, obj, shape, Console(quiet=True))


def import_into(
    file: Path,
    output: Optional[Path],
    inplace: bool,
    write: Callable[[Path], None],
    console: Console,
    *,
    input_file: Optional[Path] = None,
    obj: Optional[str] = None,
    kind: Optional[str] = None,
    space_check: bool = True,
) -> Path:
    """Copy `file`, call `write` on the copy's path, and move it into place.

    The destination is `output`, or with `inplace` the source itself.
    `kind` ("array" for a .npy, "sparse" for an MTX) has `obj`'s shape
    checked against the source from `input_file`'s header before anything
    is copied. Returns the path written.
    """
    others = [input_file] if input_file is not None else []
    target = output_target(file, output, inplace, others=others)
    if kind is not None and obj is not None and input_file is not None:
        _precheck(file, obj, input_file, kind)
    with staged_outputs([target]) as (temp,):
        check_free_space(
            target,
            lambda: disk_size(file) + (disk_size(input_file) if input_file else 0),
            check=space_check,
        )
        _copy_source(file, temp, target, inplace, console)
        write(temp)
    return target


def import_object(
    file: Path,
    obj: str,
    input_file: Path,
    output_file: Optional[Path],
    inplace: bool,
    index_column: Optional[str],
    console: Console,
) -> None:
    suffixes = [x.lower() for x in input_file.suffixes]
    # A gzipped matrix is still a matrix: `counts.mtx.gz` imports as `.mtx`.
    ext = suffixes[-2] if len(suffixes) > 1 and suffixes[-1] == ".gz" else input_file.suffix.lower()

    if ext not in EXTENSION_FORMAT:
        raise ValueError(
            f"Unsupported input file extension '{ext}'. "
            f"Supported: {', '.join(sorted(EXTENSION_FORMAT.keys()))}"
        )

    fmt = EXTENSION_FORMAT[ext]

    if index_column and (fmt != "csv" or obj not in ("obs", "var")):
        raise ValueError("--index-column is only valid for CSV import into 'obs' or 'var'.")

    def write(target: Path) -> None:
        if fmt == "csv":
            _import_csv(target, obj, input_file, index_column, console)
        elif fmt == "npy":
            _import_npy(target, obj, input_file, console)
        elif fmt == "mtx":
            _import_mtx(target, obj, input_file, console)
        elif fmt == "json":
            _import_json(target, obj, input_file, console)

    import_into(
        file, output_file, inplace, write, console,
        input_file=input_file, obj=obj,
        kind={"npy": "array", "mtx": "sparse"}.get(fmt),
    )


def _import_csv(
    file: Path,
    obj: str,
    input_file: Path,
    index_column: Optional[str],
    console: Console,
    categorical: Optional[list] = None,
    auto_categorical: bool = True,
) -> None:
    with open_store(file, "a") as store:
        import_dataframe(
            store.root,
            obj=obj,
            input_file=input_file,
            index_column=index_column,
            console=console,
            categorical=categorical,
            auto_categorical=auto_categorical,
        )


def _import_npy(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
    chunk_elements: int = IMPORT_CHUNK_ELEMENTS,
) -> None:
    with open_store(file, "a") as store:
        import_npy(
            store.root, obj=obj, input_file=input_file, console=console,
            chunk_elements=chunk_elements,
        )


def _import_mtx(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    with open_store(file, "a") as store:
        import_mtx(store.root, obj=obj, input_file=input_file, console=console)


def _import_image(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    with open_store(file, "a") as store:
        import_image(store.root, obj=obj, input_file=input_file, console=console)


def _import_json(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    with open_store(file, "a") as store:
        import_json(store.root, obj=obj, input_file=input_file, console=console)

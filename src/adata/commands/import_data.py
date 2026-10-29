"""Import command helpers for creating/replacing objects in h5ad/zarr stores.

An import never edits a store in place. The source is copied beside its
destination (the source itself with --inplace, or -o), the element is
written into the copy, and the copy is moved into place only once that has
succeeded: an import that is refused or fails part way leaves the source and
any existing output as they were.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from rich.console import Console

from adata.formats.array import import_npy
from adata.formats.dataframe import import_dataframe
from adata.formats.image import import_image
from adata.formats.json_data import import_json
from adata.formats.sparse import import_mtx
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


def import_into(
    file: Path,
    output: Optional[Path],
    inplace: bool,
    write: Callable[[Path], None],
    console: Console,
    *,
    input_file: Optional[Path] = None,
    space_check: bool = True,
) -> Path:
    """Copy `file`, call `write` on the copy's path, and move it into place.

    The destination is `output`, or with `inplace` the source itself.
    Returns the path written.
    """
    others = [input_file] if input_file is not None else []
    target = output_target(file, output, inplace, others=others)
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

    import_into(file, output_file, inplace, write, console, input_file=input_file)


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
) -> None:
    with open_store(file, "a") as store:
        import_npy(store.root, obj=obj, input_file=input_file, console=console)


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

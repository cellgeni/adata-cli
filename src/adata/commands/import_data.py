"""Import command helpers for creating/replacing objects in h5ad/zarr stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from adata.formats.array import import_npy
from adata.formats.dataframe import import_dataframe
from adata.formats.image import import_image
from adata.formats.json_data import import_json
from adata.formats.sparse import import_mtx
from adata.storage import copy_path, copy_store_contents, detect_backend, open_store


EXTENSION_FORMAT = {
    ".csv": "csv",
    ".npy": "npy",
    ".mtx": "mtx",
    ".json": "json",
}


def _prepare_target_path(
    file: Path,
    output_file: Optional[Path],
    inplace: bool,
    console: Console,
) -> Path:
    if inplace:
        return file
    if output_file is None:
        raise ValueError("Output file is required unless --inplace is specified.")

    src_backend = detect_backend(file)
    dst_backend = detect_backend(output_file)

    if src_backend == dst_backend:
        copy_path(file, output_file)
        console.print(f"[dim]Copied {file} → {output_file}[/]")
        return output_file

    with open_store(file, "r") as src_store, open_store(output_file, "w") as dst_store:
        copy_store_contents(src_store.root, dst_store.root)
    console.print(
        f"[dim]Converted {file} ({src_backend}) → {output_file} ({dst_backend})[/]"
    )
    return output_file


def import_object(
    file: Path,
    obj: str,
    input_file: Path,
    output_file: Optional[Path],
    inplace: bool,
    index_column: Optional[str],
    console: Console,
) -> None:
    target_file = _prepare_target_path(file, output_file, inplace, console)
    ext = input_file.suffix.lower()

    if ext not in EXTENSION_FORMAT:
        raise ValueError(
            f"Unsupported input file extension '{ext}'. "
            f"Supported: {', '.join(sorted(EXTENSION_FORMAT.keys()))}"
        )

    fmt = EXTENSION_FORMAT[ext]

    if index_column and (fmt != "csv" or obj not in ("obs", "var")):
        raise ValueError("--index-column is only valid for CSV import into 'obs' or 'var'.")

    if fmt == "csv":
        _import_csv(target_file, obj, input_file, index_column, console)
    elif fmt == "npy":
        _import_npy(target_file, obj, input_file, console)
    elif fmt == "mtx":
        _import_mtx(target_file, obj, input_file, console)
    elif fmt == "json":
        _import_json(target_file, obj, input_file, console)


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

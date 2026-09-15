from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.console import Console

from adata.formats.array import export_npy as export_npy_format
from adata.formats.dataframe import export_dataframe
from adata.formats.image import export_image as export_image_format
from adata.formats.json_data import export_json as export_json_format
from adata.formats.sparse import export_mtx as export_mtx_format
from adata.storage import open_store


def export_table(
    file: Path,
    axis: str,
    columns: Optional[List[str]],
    out: Optional[Path],
    chunk_rows: int,
    head: Optional[int],
    console: Console,
) -> None:
    with open_store(file, "r") as store:
        export_dataframe(
            store.root,
            axis=axis,
            columns=columns,
            out=out,
            chunk_rows=chunk_rows,
            head=head,
            console=console,
        )


def export_npy(
    file: Path,
    obj: str,
    out: Optional[Path],
    chunk_elements: int,
    console: Console,
) -> None:
    with open_store(file, "r") as store:
        export_npy_format(
            store.root,
            obj=obj,
            out=out,
            chunk_elements=chunk_elements,
            console=console,
        )


def export_mtx(
    file: Path,
    obj: str,
    out: Optional[Path],
    head: Optional[int],
    chunk_elements: int,
    in_memory: bool,
    console: Console,
) -> None:
    with open_store(file, "r") as store:
        export_mtx_format(
            store.root,
            obj=obj,
            out=out,
            head=head,
            chunk_elements=chunk_elements,
            in_memory=in_memory,
            console=console,
        )


def export_json(
    file: Path,
    obj: str,
    out: Optional[Path],
    max_elements: int,
    include_attrs: bool,
    console: Console,
) -> None:
    with open_store(file, "r") as store:
        export_json_format(
            store.root,
            obj=obj,
            out=out,
            max_elements=max_elements,
            include_attrs=include_attrs,
            console=console,
        )


def export_image(file: Path, obj: str, out: Path, console: Console) -> None:
    with open_store(file, "r") as store:
        export_image_format(store.root, obj=obj, out=out, console=console)


__all__ = [
    "export_image",
    "export_json",
    "export_mtx",
    "export_npy",
    "export_table",
]

"""`export`: write one element of a store out as a file, or to stdout.

A file is written beside its destination and renamed into place once
complete, as a store is (see `adata.util.staging`), so an export that fails
or is refused part way leaves an existing file as it was. The output is
checked first: it may not be the input store, lie inside a Zarr input, or
name a folder.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from adata.formats.array import export_npy as export_npy_format
from adata.formats.dataframe import export_dataframe
from adata.formats.image import export_image as export_image_format
from adata.formats.json_data import export_json as export_json_format
from adata.formats.sparse import export_mtx as export_mtx_format
from adata.storage import open_store
from adata.util.path import check_output
from adata.util.staging import staged_outputs


@contextmanager
def _output(file: Path, out: Optional[Path], console: Console) -> Iterator[Optional[Path]]:
    """The path to export to: a staged temporary for a file, as is for stdout."""
    if out is None or str(out) == "-":
        yield out
        return
    out = Path(out)
    check_output(out, [file], store=False)
    try:
        mode = os.stat(out).st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None and not stat.S_ISREG(mode):
        # A device or a pipe (/dev/null, a FIFO, `>(gzip > x)`): written as
        # it is. Staging would rename it aside and put a file in its place.
        yield out
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with staged_outputs([out]) as (temp,):
        yield temp
    console.print(f"[green]Wrote[/] {out}")


def export_table(
    file: Path,
    axis: str,
    columns: Optional[List[str]],
    out: Optional[Path],
    chunk_rows: int,
    head: Optional[int],
    console: Console,
) -> None:
    with _output(file, out, console) as target, open_store(file, "r") as store:
        export_dataframe(
            store.root,
            axis=axis,
            columns=columns,
            out=target,
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
    with _output(file, out, console) as target, open_store(file, "r") as store:
        export_npy_format(
            store.root,
            obj=obj,
            out=target,
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
    with _output(file, out, console) as target, open_store(file, "r") as store:
        export_mtx_format(
            store.root,
            obj=obj,
            out=target,
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
    with _output(file, out, console) as target, open_store(file, "r") as store:
        export_json_format(
            store.root,
            obj=obj,
            out=target,
            max_elements=max_elements,
            include_attrs=include_attrs,
            console=console,
        )


def export_image(file: Path, obj: str, out: Optional[Path], console: Console) -> None:
    with _output(file, out, console) as target, open_store(file, "r") as store:
        export_image_format(store.root, obj=obj, out=target, console=console)


__all__ = [
    "export_image",
    "export_json",
    "export_mtx",
    "export_npy",
    "export_table",
]

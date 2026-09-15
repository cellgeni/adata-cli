from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
from rich.console import Console

from adata.formats.common import _get_encoding_type, _resolve
from adata.formats.validate import validate_dimensions
from adata.elements.write import write_dense
from adata.storage import create_dataset, is_dataset, is_group
from adata.util.path import norm_path


def export_npy(
    root: Any,
    obj: str,
    out: Optional[Path],
    chunk_elements: int,
    console: Console,
) -> None:
    """Write a dense array to a .npy file, or to stdout when `out` is None.

    Writing to a file streams in chunks through a memory-mapped output. Stdout
    is not seekable, so that path materialises the array first and is only
    suitable for arrays that fit in memory.
    """
    h5obj = _resolve(root, obj)

    if is_group(h5obj):
        enc = _get_encoding_type(h5obj)
        if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
            if "values" not in h5obj:
                raise ValueError(f"Encoded group '{obj}' is missing 'values' dataset.")
            ds = h5obj["values"]
            console.print(f"[dim]Exporting nullable array values from '{obj}'[/]")
        else:
            raise ValueError(
                f"Target '{obj}' is a group with encoding '{enc}'; cannot export as .npy directly."
            )
    elif is_dataset(h5obj):
        ds = h5obj
    else:
        raise ValueError("Target is not an array-like object.")

    if out is None or str(out) == "-":
        np.save(sys.stdout.buffer, np.asarray(ds[...]), allow_pickle=False)
        sys.stdout.buffer.flush()
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=ds.dtype, shape=ds.shape)
    try:
        if ds.shape == ():
            mm[...] = ds[()]
            console.print(f"[green]Wrote[/] {out}")
            return

        if ds.ndim == 1:
            n = int(ds.shape[0])
            step = max(1, int(chunk_elements))
            with console.status(f"[magenta]Exporting {obj} to {out}...[/]") as status:
                for start in range(0, n, step):
                    end = min(start + step, n)
                    status.update(
                        f"[magenta]Exporting {obj}: {start}-{end} of {n}...[/]"
                    )
                    mm[start:end] = ds[start:end]
            console.print(f"[green]Wrote[/] {out}")
            return

        n0 = int(ds.shape[0])
        row_elems = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
        step0 = max(1, int(chunk_elements) // max(1, row_elems))
        with console.status(f"[magenta]Exporting {obj} to {out}...[/]") as status:
            for start in range(0, n0, step0):
                end = min(start + step0, n0)
                status.update(
                    f"[magenta]Exporting {obj}: {start}-{end} of {n0}...[/]"
                )
                mm[start:end, ...] = ds[start:end, ...]
        console.print(f"[green]Wrote[/] {out}")
    finally:
        del mm


def import_npy(
    root: Any,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    obj = norm_path(obj)
    arr = np.load(input_file)

    validate_dimensions(root, obj, arr.shape, console)

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else parent.create_group(part)
    name = parts[-1]

    write_dense(parent, name, arr, replace=True)

    shape_str = "×".join(str(d) for d in arr.shape)
    console.print(f"[green]Imported[/] {shape_str} array into '{obj}'")

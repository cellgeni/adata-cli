"""The `convert` command: rewrite matrices, copy everything else.

Writes a whole new store rather than editing one in place, for the same
reason `subset` does: a conversion that fails half way leaves the original
untouched, and `--inplace` becomes an atomic swap of a finished file rather
than a partial edit of a live one.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console

from adata.core.convert import (
    DEFAULT_CHUNK,
    LAYOUTS,
    Plan,
    convert_matrix,
    parse_dtype,
    plan_conversion,
)
from adata.elements import spec
from adata.elements.write import ensure_anndata_skeleton
from adata.storage import copy_tree, detect_backend, is_group, open_store

#: What `--all` means. The matrices an AnnData object is built from, not
#: every 2-D array in the file: obsm/varm hold embeddings, whose dtype is
#: rarely what anyone is trying to shrink. Those still convert by path.
ALL_PREFIXES = ("X", "layers", "raw/X")


def discover_matrices(root: Any) -> List[str]:
    """Entry paths `--all` resolves to, in a stable order."""
    found: List[str] = []
    if "X" in root:
        found.append("X")
    if "layers" in root and is_group(root["layers"]):
        found.extend(f"layers/{key}" for key in sorted(root["layers"].keys()))
    if "raw" in root and is_group(root["raw"]) and "X" in root["raw"]:
        found.append("raw/X")
    return found


def _resolve(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("/"):
        if part not in obj:
            raise KeyError(f"{path!r} not found in the store.")
        obj = obj[part]
    return obj


def convert_store(
    file: Path,
    entries: Optional[List[str]],
    output: Optional[Path],
    console: Console,
    *,
    dtype: Optional[str] = None,
    indices_dtype: Optional[str] = None,
    layout: Optional[str] = None,
    convert_all: bool = False,
    inplace: bool = False,
    chunk: int = DEFAULT_CHUNK,
    chunk_rows: int = 1024,
    in_memory: bool = False,
    force: bool = False,
    zarr_format: Optional[int] = None,
) -> None:
    """Write a copy of `file` with the named matrices converted."""
    if not inplace and output is None:
        raise ValueError("Output file is required unless --inplace is specified.")
    if dtype is None and indices_dtype is None and layout is None:
        raise ValueError(
            "Nothing to do: pass at least one of --dtype, --indices-dtype "
            "or --layout."
        )
    if layout is not None and layout not in LAYOUTS:
        raise ValueError(f"--layout must be one of: {', '.join(LAYOUTS)}.")

    data_dtype = parse_dtype(dtype) if dtype else None
    index_dtype = (
        parse_dtype(indices_dtype, allowed=("int32", "int64"))
        if indices_dtype
        else None
    )

    if inplace:
        backend = detect_backend(file)
        if backend == "zarr":
            base = file.stem if file.suffix else file.name
            dst_path = file.with_name(f"{base}.convert-tmp.zarr")
        else:
            dst_path = file.with_name(f"{file.name}.convert-tmp")
        if dst_path.exists():
            raise FileExistsError(f"Temporary path already exists: {dst_path}")
    else:
        dst_path = output

    if zarr_format is None and detect_backend(file) == "zarr":
        with open_store(file, "r") as probe:
            zarr_format = probe.zarr_format

    with open_store(file, "r") as src_store:
        src = src_store.root
        targets = discover_matrices(src) if convert_all else list(entries or [])
        if not targets:
            raise ValueError(
                "No matrices selected. Name one (e.g. `X`) or pass --all."
            )
        # Every check first, before the destination exists. A refusal must
        # not leave behind a store holding a copy of obs and var and nothing
        # else -- the caller cannot tell that from a finished conversion.
        plans = {
            path: plan_conversion(
                _resolve(src, path),
                path,
                dtype=data_dtype,
                index_dtype=index_dtype,
                layout=layout,
                chunk=chunk,
                force=force,
                console=console,
            )
            for path in targets
        }

        console.print(
            f"[cyan]Converting {len(targets)} matrix/matrices:[/] "
            + ", ".join(targets)
        )

        with open_store(dst_path, "w", zarr_format=zarr_format) as dst_store:
            dst = dst_store.root
            _write(
                src, dst, plans,
                chunk=chunk, chunk_rows=chunk_rows, in_memory=in_memory,
                console=console,
            )
            ensure_anndata_skeleton(dst)

    if inplace:
        if file.is_dir():
            shutil.rmtree(file)
        elif file.exists():
            file.unlink()
        if dst_path.is_dir():
            shutil.move(str(dst_path), str(file))
        else:
            dst_path.replace(file)
        console.print(f"[green]Converted[/] {file}")
    else:
        console.print(f"[green]Wrote[/] {dst_path}")


def _write(src: Any, dst: Any, plans: dict, **options: Any) -> None:
    """Copy the store across, converting the targeted entries as they pass."""
    by_parent: dict = {}
    for path in plans:
        parent, _, leaf = path.rpartition("/")
        by_parent.setdefault(parent, {})[leaf] = plans[path]

    for key in src.keys():
        if key in ("layers", "raw") and any(
            p == key or p.startswith(f"{key}/") for p in by_parent
        ):
            _write_group(src[key], dst, key, by_parent, **options)
        elif key in by_parent.get("", {}):
            convert_matrix(
                src[key], dst, key, plan=by_parent[""][key], **options
            )
        else:
            copy_tree(src[key], dst, key)


def _write_group(
    group: Any, dst: Any, name: str, by_parent: dict, **options: Any
) -> None:
    """Recreate one container, converting the members that were named."""
    from adata.storage import copy_attrs, is_zarr_group

    out = dst.create_group(name)
    copy_attrs(
        group.attrs,
        out.attrs,
        target_backend="zarr" if is_zarr_group(dst) else "hdf5",
    )
    if not spec.encoding_type(out):
        spec.set_encoding(out, spec.RAW if name == "raw" else spec.DICT)

    wanted = by_parent.get(name, {})
    for key in group.keys():
        if key in wanted:
            convert_matrix(group[key], out, key, plan=wanted[key], **options)
        else:
            copy_tree(group[key], out, key)

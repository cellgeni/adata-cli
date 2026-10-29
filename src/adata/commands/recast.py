"""The `recast` command: rewrite matrices, copy everything else.

Writes a whole new store rather than editing one in place, for the same
reason `subset` does: a conversion that fails half way leaves the original
untouched. The new store is written beside its destination and renamed into
place only once complete (see `adata.util.staging`), so `--inplace` replaces a
finished file rather than editing a live one, and the original is deleted only
after its successor is in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console

from adata.core.recast import (
    DEFAULT_CHUNK,
    LAYOUTS,
    Plan,
    _stored_bytes,
    convert_matrix,
    parse_dtype,
    plan_conversion,
)
from adata.elements import spec
from adata.elements.write import ensure_anndata_skeleton
from adata.storage import copy_tree, detect_backend, is_group, open_store
from adata.util.path import output_target
from adata.util.staging import check_free_space, disk_size, staged_outputs

def discover_matrices(root: Any) -> List[str]:
    """Entry paths `--all` resolves to, in a stable order.

    The matrices an AnnData object is built from, not every 2-D array in the
    file: obsm/varm hold embeddings, whose dtype is rarely what anyone is
    trying to shrink. Those still convert by path.
    """
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


def recast_store(
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
    space_check: bool = True,
) -> None:
    """Write a copy of `file` with the named matrices converted."""
    if dtype is None and indices_dtype is None and layout is None:
        raise ValueError(
            "Nothing to do: pass at least one of --dtype, --indices-dtype "
            "or --layout."
        )
    if layout is not None and layout not in LAYOUTS:
        raise ValueError(f"--layout must be one of: {', '.join(LAYOUTS)}.")
    if chunk < 1 or chunk_rows < 1:
        raise ValueError("--chunk must be at least 1.")

    data_dtype = parse_dtype(dtype) if dtype else None
    index_dtype = (
        parse_dtype(indices_dtype, allowed=("int32", "int64"))
        if indices_dtype
        else None
    )

    # Refuses -o naming the input: opening the destination "w" clears it
    # while the source is still being read from it. HDF5 refuses; Zarr does
    # not, and quietly produced an empty store where the data used to be.
    dst_path = output_target(file, output, inplace)

    if zarr_format is None and detect_backend(file) == "zarr":
        with open_store(file, "r") as probe:
            zarr_format = probe.zarr_format

    # The source is opened inside the staging block, so it is closed before
    # `--inplace` renames it aside.
    with staged_outputs([dst_path]) as (temp,):
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

            check_free_space(
                dst_path,
                disk_size(file) * max([1.0] + [_growth(p) for p in plans.values()]),
                check=space_check,
            )

            console.print(
                f"[cyan]Converting {len(targets)} matrix/matrices:[/] "
                + ", ".join(targets)
            )

            with open_store(temp, "w", zarr_format=zarr_format) as dst_store:
                dst = dst_store.root
                _write(
                    src, dst, plans,
                    chunk=chunk, chunk_rows=chunk_rows, in_memory=in_memory,
                    console=console,
                )
                ensure_anndata_skeleton(dst)

    if inplace:
        console.print(f"[green]Converted[/] {file}")
    else:
        console.print(f"[green]Wrote[/] {dst_path}")


def _growth(plan: Plan) -> float:
    """How many times its current size a matrix will be once converted.

    From the uncompressed sizes, so a guess for compressed stores, but it
    catches the cases that matter: a wider dtype, and densifying.
    """
    src = plan.source
    current = _stored_bytes(src.obj)
    if current <= 0:
        return 1.0
    if plan.layout == "dense":
        cells = 1
        for extent in src.shape:
            cells *= int(extent)
        return cells * plan.data_dtype.itemsize / current
    if not src.sparse:
        return 1.0  # sparsifying is done to shrink; its nnz is not known yet
    nnz = int(src.obj["data"].shape[0])
    pointers = int(src.obj["indptr"].shape[0])
    projected = (
        nnz * (plan.data_dtype.itemsize + plan.index_dtype.itemsize)
        + pointers * plan.pointer_dtype.itemsize
    )
    return projected / current


def _write(src: Any, dst: Any, plans: dict, **options: Any) -> None:
    """Copy the store across, converting the targeted entries as they pass."""
    by_parent: dict = {}
    for path in plans:
        parent, _, leaf = path.rpartition("/")
        by_parent.setdefault(parent, {})[leaf] = plans[path]

    for key in src.keys():
        if key in by_parent.get("", {}):
            convert_matrix(
                src[key], dst, key, plan=by_parent[""][key], **options
            )
        elif key in by_parent:
            # Any parent, not just layers and raw. Restricting it to those
            # two meant an explicitly named `obsm/X_pca` was copied
            # unconverted and the command still reported success.
            _write_group(src[key], dst, key, by_parent, **options)
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

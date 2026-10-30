"""`split` -- write one store per distinct value of an annotation column.

Requested in issue #2, modelled on cellgeni/scraft's `split_h5ad` but
streaming: the source is never loaded into memory, only the column being split
on is read, and each output is produced by the same subset machinery used by
`adata subset`, fanned out so every matrix is read once for all groups.

Every output is written beside its destination and renamed into place only
once all of them are complete, so a split that fails leaves no partial stores
(and existing outputs as they were). The manifest is written last, so it
exists only when everything it lists does.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.core.select import group_indices
from adata.core.subset import split_h5ad
from adata.elements.read import element_len, resolve_index
from adata.storage import detect_backend, open_store
from adata.util.names import unique_names
from adata.util.path import check_output
from adata.util.staging import check_free_space, disk_size, staged_outputs


def split_store(
    file: Path,
    column: str,
    output_dir: Path,
    console: Console,
    axis: str = "obs",
    suffix: Optional[str] = None,
    dry_run: bool = False,
    manifest: bool = True,
    min_size: int = 1,
    chunk_rows: int = 1024,
    zarr_format: Optional[int] = None,
    space_check: bool = True,
) -> List[Tuple[str, Path, int]]:
    """Split `file` into one store per distinct value of `column`.

    Returns ``(label, path, n_rows)`` for each group written. Groups smaller
    than `min_size` are skipped with a warning rather than producing tiny
    stores nobody asked for.

    `zarr_format` overrides the Zarr version of the outputs; without it they
    follow the source store's.
    """
    if axis not in ("obs", "var"):
        raise ValueError("--axis must be 'obs' or 'var'.")
    if chunk_rows < 1:
        raise ValueError("--chunk must be at least 1.")

    if suffix is None:
        suffix = ".zarr" if detect_backend(file) == "zarr" else ".h5ad"

    with open_store(file, "r") as store:
        groups, order = group_indices(store.root, axis, column)
        if zarr_format is None:
            zarr_format = store.zarr_format
        n_rows = element_len(resolve_index(store.root[axis], axis)[0])

    if not groups:
        raise ValueError(f"Column {column!r} produced no groups.")

    names = _distinct(unique_names(order), order)
    console.print(
        f"[cyan]Splitting on {axis}[{column}]: "
        f"{len(order)} group{'s' if len(order) != 1 else ''}[/]"
    )

    planned: List[Tuple[str, Path, int]] = []
    for label in order:
        indices = groups[label]
        out_path = output_dir / f"{names[label]}{suffix}"
        if len(indices) < min_size:
            console.print(
                f"[yellow]Skipping {label!r}: {len(indices)} "
                f"{axis} < --min-size {min_size}[/]"
            )
            continue
        planned.append((label, out_path, len(indices)))

    for label, out_path, count in planned:
        console.print(f"  {label!r} -> {out_path} ({count} {axis})")

    for _, out_path, _ in planned:
        try:
            check_output(out_path, [file])
        except ValueError as error:
            raise ValueError(f"{error} Choose another --output-dir.") from None
    if manifest:
        # Checked with the stores, so it cannot be refused once they exist.
        check_output(_manifest_path(file, output_dir), [file], store=False)

    if dry_run:
        console.print("[yellow]Dry run: nothing written.[/]")
        return planned

    output_dir.mkdir(parents=True, exist_ok=True)
    if planned:
        kept = sum(count for _, _, count in planned)
        check_free_space(
            planned[0][1],
            lambda: disk_size(file) * (kept / n_rows if n_rows else 1.0),
            check=space_check,
        )

    # One call for every group, so each matrix is read once rather than once
    # per group.
    with staged_outputs([out_path for _, out_path, _ in planned]) as temps:
        split_h5ad(
            file,
            [
                (
                    temp,
                    np.sort(groups[label]) if axis == "obs" else None,
                    np.sort(groups[label]) if axis == "var" else None,
                )
                for (label, _, _), temp in zip(planned, temps)
            ],
            chunk_rows=chunk_rows,
            console=console,
            zarr_format=zarr_format,
        )

    if manifest:
        _write_manifest(file, output_dir, column, planned, console)

    console.print(f"[green]Wrote[/] {len(planned)} stores to {output_dir}")
    return planned


def _distinct(names: Dict[str, str], order: List[str]) -> Dict[str, str]:
    """Stems no two of which name one file, whatever the filesystem.

    `B` and `b` are one file on macOS and Windows, and `unique_names` can
    give `a_1` to a label `a ` beside one that is `a_1` already: either way
    two groups were written to one path.
    """
    taken: set = set()
    distinct: Dict[str, str] = {}
    for label in order:
        stem = candidate = names[label]
        count = 0
        while candidate.casefold() in taken:
            count += 1
            candidate = f"{stem}_{count}"
        taken.add(candidate.casefold())
        distinct[label] = candidate
    return distinct


def _manifest_path(file: Path, output_dir: Path) -> Path:
    return output_dir / f"{file.stem if file.suffix else file.name}_manifest.csv"


def _write_manifest(
    file: Path,
    output_dir: Path,
    column: str,
    planned: List[Tuple[str, Path, int]],
    console: Console,
) -> Path:
    """Record what was written, for downstream pipelines to consume."""
    source_id = file.stem if file.suffix else file.name
    path = _manifest_path(file, output_dir)
    with staged_outputs([path]) as (temp,):
        with open(temp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "column", "value", "anndatas", "n"])
            for label, out_path, count in planned:
                writer.writerow([source_id, column, label, str(out_path), count])
    console.print(f"[dim]Manifest written to {path}[/]")
    return path

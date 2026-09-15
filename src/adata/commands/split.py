"""`split` -- write one store per distinct value of an annotation column.

Requested in issue #2, modelled on cellgeni/scraft's `split_h5ad` but
streaming: the source is never loaded into memory, only the column being split
on is read, and each output is produced by the same subset machinery used by
`adata subset`.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.core.select import group_indices
from adata.core.subset import subset_h5ad
from adata.storage import detect_backend, open_store


def sanitize_label(label: str) -> str:
    """Turn a column value into something safe to use as a filename."""
    text = (label or "").strip()
    if not text:
        return "NA"
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "", text)
    return text or "NA"


def unique_names(labels: List[str]) -> Dict[str, str]:
    """Map each label to a distinct filename stem.

    Sanitising can map two different labels onto the same stem ("a/b" and
    "a b"), so collisions get a numeric suffix rather than silently
    overwriting one another.
    """
    used: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for label in labels:
        base = sanitize_label(label)
        count = used.get(base, 0)
        used[base] = count + 1
        mapping[label] = base if count == 0 else f"{base}_{count}"
    return mapping


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

    if suffix is None:
        suffix = ".zarr" if detect_backend(file) == "zarr" else ".h5ad"

    with open_store(file, "r") as store:
        groups, order = group_indices(store.root, axis, column)
        if zarr_format is None:
            zarr_format = store.zarr_format

    if not groups:
        raise ValueError(f"Column {column!r} produced no groups.")

    names = unique_names(order)
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

    if dry_run:
        console.print("[yellow]Dry run: nothing written.[/]")
        return planned

    output_dir.mkdir(parents=True, exist_ok=True)

    for label, out_path, _ in planned:
        indices = np.sort(groups[label])
        subset_h5ad(
            file=file,
            output=out_path,
            obs_file=None,
            var_file=None,
            chunk_rows=chunk_rows,
            console=console,
            obs_indices=indices if axis == "obs" else None,
            var_indices=indices if axis == "var" else None,
            zarr_format=zarr_format,
        )

    if manifest:
        _write_manifest(file, output_dir, column, planned, console)

    console.print(f"[green]Wrote[/] {len(planned)} stores to {output_dir}")
    return planned


def _write_manifest(
    file: Path,
    output_dir: Path,
    column: str,
    planned: List[Tuple[str, Path, int]],
    console: Console,
) -> Path:
    """Record what was written, for downstream pipelines to consume."""
    source_id = file.stem if file.suffix else file.name
    path = output_dir / f"{source_id}_manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "column", "value", "anndatas", "n"])
        for label, out_path, count in planned:
            writer.writerow([source_id, column, label, str(out_path), count])
    console.print(f"[dim]Manifest written to {path}[/]")
    return path

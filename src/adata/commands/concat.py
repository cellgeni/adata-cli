"""Command wrapper for `concat`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from adata.core.concat import MERGE_CHOICES, MERGE_STRATEGIES, concat_on_disk
from adata.util.path import check_output

__all__ = ["MERGE_CHOICES", "MERGE_STRATEGIES", "concat_stores"]


def concat_stores(
    files: Sequence[Path],
    output: Path,
    console: Console,
    space_check: bool = True,
    **kwargs: object,
) -> None:
    if not output.exists():
        # concat never replaces an output (it refuses one that exists), but
        # a new one may still name a path inside a Zarr input.
        check_output(output, files)
    # The free space is checked by concat_on_disk, once it knows whether a
    # CSC matrix needs scratch space beside the output too.
    concat_on_disk(files, output, console, space_check=space_check, **kwargs)  # type: ignore[arg-type]

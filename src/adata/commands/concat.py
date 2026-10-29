"""Command wrapper for `concat`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from adata.core.concat import MERGE_CHOICES, MERGE_STRATEGIES, concat_on_disk
from adata.util.path import check_output
from adata.util.staging import check_free_space, disk_size

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
    # About the sum of the inputs: an inner join keeps less, an outer join's
    # dense fill can add some.
    check_free_space(
        output, sum(disk_size(f) for f in files), check=space_check
    )
    concat_on_disk(files, output, console, **kwargs)  # type: ignore[arg-type]

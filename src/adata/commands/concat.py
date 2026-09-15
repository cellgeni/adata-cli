"""Command wrapper for `concat`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from adata.core.concat import MERGE_STRATEGIES, concat_on_disk

__all__ = ["MERGE_STRATEGIES", "concat_stores"]


def concat_stores(
    files: Sequence[Path],
    output: Path,
    console: Console,
    **kwargs: object,
) -> None:
    concat_on_disk(files, output, console, **kwargs)  # type: ignore[arg-type]

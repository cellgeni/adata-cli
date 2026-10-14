"""Write outputs beside their destination, and move them into place at the end.

A conversion that fails half way must not leave a half-written store where
the output should be: the next command to open it would read whatever got
written as if it were the whole thing. Writing to a temporary sibling and
renaming it only on success means the output either appears whole or not at
all, and an existing output is replaced only once its successor is complete.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Sequence


def _sibling(path: Path, tag: str) -> Path:
    """`out.zarr` -> `out.<tag>.zarr`, keeping the suffix that picks the backend."""
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


@contextmanager
def staged_outputs(outputs: Sequence[Path]) -> Iterator[List[Path]]:
    """Yield temporary paths to write `outputs` to; move them into place after.

    On success each output is swapped in: an existing one is set aside first
    and deleted only once every new output is in place. On failure the
    temporary paths are deleted and the outputs are left as they were.
    """
    outputs = [Path(p) for p in outputs]
    temps = [_sibling(p, "adata-tmp") for p in outputs]
    for temp in temps:
        if temp.exists():
            raise FileExistsError(
                f"'{temp}' already exists, probably left by an interrupted run. "
                "Remove it and try again."
            )

    try:
        yield temps
    except BaseException:
        for temp in temps:
            _remove(temp)
        raise

    backups = []
    for output, temp in zip(outputs, temps):
        if output.exists():
            backup = _sibling(output, "adata-old")
            _remove(backup)
            output.rename(backup)
            backups.append(backup)
        temp.rename(output)
    for backup in backups:
        _remove(backup)

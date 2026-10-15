"""Write outputs beside their destination, and move them into place at the end.

A command that fails half way must not leave a half-written store where the
output should be: the next command to open it would read whatever got written
as if it were the whole thing. Writing to a temporary sibling and renaming it
only on success means the output either appears whole or not at all.

An existing output is renamed aside (`out.adata-old.zarr`) before its
successor is renamed in, and deleted only once every output is in place. If a
rename fails part way, the outputs already swapped are put back. A run killed
between the two renames leaves the old data under the `adata-old` name, never
nowhere, and the next run refuses to start until that has been dealt with.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from adata.util.path import is_zarr_path


def _sibling(path: Path, tag: str) -> Path:
    """`out.zarr` -> `out.<tag>.zarr`, keeping the suffix that picks the backend."""
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def _temp_for(path: Path) -> Path:
    temp = _sibling(path, "adata-tmp")
    if path.suffix != ".zarr" and is_zarr_path(path):
        # A Zarr store named without the suffix. A temporary named the same
        # way would not exist yet, and so be opened as HDF5. Only a Zarr
        # store: any other directory, such as an MTX folder, keeps its name.
        temp = temp.with_name(f"{temp.name}.zarr")
    return temp


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


@contextmanager
def staged_outputs(outputs: Sequence[Path]) -> Iterator[List[Path]]:
    """Yield temporary paths to write `outputs` to; move them into place after.

    On success each output is swapped in: an existing one is set aside first
    and deleted only once every new output is in place. On failure the
    temporary paths are deleted and the outputs are left as they were.
    """
    outputs = [Path(p) for p in outputs]
    temps = [_temp_for(p) for p in outputs]
    backups = [_sibling(p, "adata-old") for p in outputs]
    for leftover in temps + backups:
        if _exists(leftover):
            raise FileExistsError(
                f"'{leftover}' already exists, probably left by an interrupted "
                "run. Check what it holds, remove it, and try again."
            )

    try:
        yield temps
    except BaseException:
        for temp in temps:
            _remove(temp)
        raise

    swapped: List[Tuple[Path, Path, bool]] = []
    try:
        for output, temp, backup in zip(outputs, temps, backups):
            had_output = _exists(output)
            if had_output:
                output.rename(backup)
            try:
                temp.rename(output)
            except BaseException:
                if had_output:
                    backup.rename(output)
                raise
            swapped.append((output, backup, had_output))
    except BaseException:
        # Put back what was already swapped, newest first, and drop the
        # temporaries that never made it.
        for output, backup, had_output in reversed(swapped):
            _remove(output)
            if had_output:
                backup.rename(output)
        for temp in temps:
            _remove(temp)
        raise

    for _, backup, had_output in swapped:
        if had_output:
            _remove(backup)

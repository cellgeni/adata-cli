from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def norm_path(path: str) -> str:
    """Normalize object paths used inside h5ad/zarr stores."""
    value = path.strip()
    if not value:
        raise ValueError("Object path must be non-empty.")
    return value.lstrip("/")


def same_path(left: Path, right: Path) -> bool:
    """Do these name the same store, through symlinks and `..` alike?

    When both exist, the filesystem decides: on a case-insensitive one
    (macOS, Windows) `Data.h5ad` and `data.h5ad` are one file, and so are
    two hard links, though neither resolves to the same string.
    """
    left, right = Path(left), Path(right)
    try:
        if left.exists() and right.exists():
            return os.path.samefile(left, right)
        return left.resolve() == right.resolve()
    except OSError:  # pragma: no cover - unresolvable path
        return left.absolute() == right.absolute()


def is_zarr_path(path: Path) -> bool:
    """Is this an existing Zarr store (v2 or v3), judged by its metadata files?"""
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    if (path / "zarr.json").exists():
        return True
    if (path / ".zgroup").exists() or (path / ".zattrs").exists():
        return True
    return False


def _inside(path: Path, store: Path) -> bool:
    """Does `path` lie inside `store`, a Zarr store, among its chunks?

    Only a Zarr store: a file written into a Space Ranger `outs/` or an MTX
    folder that is being read does no harm.
    """
    directory = Path(store)
    if not is_zarr_path(directory):
        return False
    return any(
        parent.exists() and same_path(parent, directory)
        for parent in Path(path).absolute().parents
    )


def _is_hdf5(path: Path) -> bool:
    import h5py

    try:
        return bool(h5py.is_hdf5(str(path)))
    except OSError:  # pragma: no cover - unreadable
        return False


def check_output(
    output: Path,
    inputs: Iterable[Path] = (),
    *,
    store: bool = True,
    inplace_hint: bool = False,
) -> None:
    """Refuse an output that would destroy something it should not.

    Every command that writes calls this before it opens anything:

    - an output that is one of the inputs, under any alias, is refused:
      opening it to write empties a Zarr input before it has been read, and
      truncates an HDF5 one;
    - a store output (`store=True`) that names an existing directory which
      is neither a Zarr store nor empty, or an existing non-empty file which
      is not HDF5, is refused: replacing it would delete a folder of files,
      or a text file, that `-o` named by mistake;
    - a file output (`store=False`, an export) that names a directory or an
      existing store is refused for the same reason: an export never writes
      a store.

    An existing store is still replaced by a store; that is what `-o` asks
    for.
    """
    output = Path(output)
    for source in inputs:
        if same_path(source, output) or _inside(output, source):
            raise ValueError(
                f"Output path is the input: {output}."
                + (
                    " Use --inplace to replace it, which writes to a "
                    "temporary file first."
                    if inplace_hint
                    else ""
                )
            )
    if not (output.exists() or output.is_symlink()):
        return
    if output.is_dir():
        if store and (is_zarr_path(output) or not any(output.iterdir())):
            return
        raise ValueError(
            f"'{output}' is a directory"
            + (", not a Zarr store" if store else "")
            + ": refusing to replace it. Name a file"
            + (" or store" if store else "")
            + " to write instead."
        )
    if not output.is_file() or not output.stat().st_size:
        return
    if store and not _is_hdf5(output):
        raise ValueError(
            f"'{output}' exists and is not an HDF5 file or a Zarr store: "
            "refusing to replace it. Remove it first if it should go."
        )
    if not store and _is_hdf5(output):
        raise ValueError(
            f"'{output}' is an HDF5 file: refusing to replace it with an "
            "export. Name the file to write."
        )


def output_target(
    source: Path,
    output: Optional[Path],
    inplace: bool,
    *,
    others: Iterable[Path] = (),
) -> Path:
    """Where a command that can rewrite its input should write.

    `--inplace` and `-o` are exclusive: given both, the source used to be
    rewritten and the output quietly ignored. With `--inplace` a symlinked
    source is followed, so the store it points at is replaced and the link
    itself kept; otherwise the output is checked with `check_output`,
    against the source and any `others` the command reads.
    """
    if inplace and output is not None:
        raise ValueError(
            "Give --output/-o or --inplace, not both: --inplace replaces the "
            "input."
        )
    if not inplace:
        if output is None:
            raise ValueError(
                "Output file is required. Use --output/-o or --inplace."
            )
        check_output(output, [source, *others], inplace_hint=True)
        return Path(output)
    return Path(os.path.realpath(source))

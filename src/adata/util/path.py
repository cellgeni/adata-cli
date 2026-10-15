from __future__ import annotations

from pathlib import Path


def norm_path(path: str) -> str:
    """Normalize object paths used inside h5ad/zarr stores."""
    value = path.strip()
    if not value:
        raise ValueError("Object path must be non-empty.")
    return value.lstrip("/")


def same_path(left: Path, right: Path) -> bool:
    """Do these name the same store, through symlinks and `..` alike?"""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:  # pragma: no cover - unresolvable path
        return Path(left).absolute() == Path(right).absolute()


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

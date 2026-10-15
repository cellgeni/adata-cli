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

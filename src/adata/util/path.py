from __future__ import annotations


def norm_path(path: str) -> str:
    """Normalize object paths used inside h5ad/zarr stores."""
    value = path.strip()
    if not value:
        raise ValueError("Object path must be non-empty.")
    return value.lstrip("/")

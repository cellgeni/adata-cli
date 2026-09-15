from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.console import Console

from adata.formats.common import _resolve
from adata.storage import is_dataset


def export_image(root: Any, obj: str, out: Path, console: Console) -> None:
    h5obj = _resolve(root, obj)
    if not is_dataset(h5obj):
        raise ValueError("Image export requires a dataset.")
    arr = np.asarray(h5obj[...])

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image array; got shape {arr.shape}.")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Expected last dimension (channels) to be 1, 3, or 4; got {arr.shape}."
        )

    if np.issubdtype(arr.dtype, np.floating):
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported image dtype: {arr.dtype}")

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    img = Image.fromarray(arr)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]Wrote[/] {out}")

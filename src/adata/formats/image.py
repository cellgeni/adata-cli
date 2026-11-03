from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from rich.console import Console

from adata.elements.write import write_dense, write_mapping
from adata.formats.validate import DATAFRAME_PATHS, check_target, validate_dimensions
from adata.formats.common import _resolve
from adata.storage import is_dataset
from adata.util.markup import literal
from adata.util.path import norm_path


#: Formats Pillow writes a channel of 16 bits to (mode "I;16"), for an
#: image of one channel; a colour image it writes with 8 bits a channel.
SIXTEEN_BIT = frozenset({"PNG", "TIFF"})


def export_image(
    root: Any, obj: str, out: Optional[Path], console: Console, *, scale: bool = False
) -> None:
    """Write a 2-D or 3-D array as an image, or as PNG to stdout when `out` is None.

    The format follows the output's extension; stdout has none, so it is PNG.
    Values are kept as they are (`pixels`), or with `scale` stretched onto
    0-255.
    """
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
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    to_stdout = out is None or str(out) == "-"
    fmt = "PNG" if to_stdout else Image.registered_extensions().get(out.suffix.lower())
    img = Image.fromarray(pixels(arr, obj, fmt, scale=scale))
    if to_stdout:
        img.save(sys.stdout.buffer, format="PNG")
        sys.stdout.buffer.flush()
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def pixels(arr: np.ndarray, obj: str, fmt: Optional[str], *, scale: bool = False) -> np.ndarray:
    """`arr` as an image holds it, its values unchanged: whole numbers from
    0 to 255 as 8 bits, and one channel of uint16 (or of whole numbers to
    65535) as 16 bits where `fmt` keeps them (PNG, TIFF). Floats from 0 to 1
    are a normalised image and scale up to 0-255; booleans are 0 and 255.
    Other values are refused rather than clipped, unless `scale` stretches
    them onto 0-255: the smallest to 0, the largest to 255."""
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"Unsupported image dtype: {arr.dtype}")
    if scale:
        return _stretch(arr)
    if arr.dtype == np.uint8:
        return arr
    deep = arr.ndim == 2 and fmt in SIXTEEN_BIT
    if arr.dtype == np.uint16 and deep:
        return arr
    finite = bool(np.isfinite(arr).all()) if arr.dtype.kind == "f" else True
    values = arr[np.isfinite(arr)] if not finite else arr
    lo, hi = (values.min(), values.max()) if values.size else (0, 0)
    if arr.dtype.kind == "f" and finite and lo >= 0:
        if hi <= 1:
            return (arr * 255.0).astype(np.uint8)
        if hi <= 255:
            return arr.astype(np.uint8)
    elif arr.dtype.kind in "iu" and lo >= 0:
        if hi <= 255:
            return arr.astype(np.uint8)
        if hi <= 65535 and deep:
            return arr.astype(np.uint16)

    if not finite:
        why = "some are not finite"
    elif arr.dtype.kind in "iu" and lo >= 0 and hi <= 65535 and arr.ndim == 2:
        why = f"{fmt or 'that format'} keeps 8 bits a pixel; a .png or .tiff keeps 16"
    elif arr.dtype.kind in "iu" and lo >= 0 and hi <= 65535:
        why = "a colour image keeps 8 bits a channel (one channel keeps 16, as .png or .tiff)"
    else:
        top = 65535 if deep else 255
        why = f"an image keeps whole numbers from 0 to {top}" + (
            ", or floats from 0 to 1" if arr.dtype.kind == "f" else ""
        )
    raise ValueError(
        f"'{obj}' holds {arr.dtype} values from {lo:g} to {hi:g}, and {why}. Pass "
        "--scale to stretch them onto 0-255, or export the values as they are with "
        "`adata export array`."
    )


def _stretch(arr: np.ndarray) -> np.ndarray:
    """Values stretched onto 0-255: the smallest finite one to 0, the
    largest to 255; any that is not finite to 0."""
    values = arr.astype(np.float64)
    finite = np.isfinite(values)
    out = np.zeros(arr.shape, dtype=np.float64)
    if finite.any():
        lo, hi = values[finite].min(), values[finite].max()
        if hi > lo:
            out[finite] = (values[finite] - lo) / (hi - lo) * 255.0
    return np.rint(out).astype(np.uint8)


def import_image(root: Any, obj: str, input_file: Path, console: Console) -> None:
    """Read an image file into the store as a dense array.

    Kept as raw pixels (H, W) or (H, W, C) with no encoding beyond `array`,
    which is how spatial tooling stores tissue images under uns/spatial.
    """
    obj = norm_path(obj)
    arr = np.asarray(Image.open(input_file))

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image; got shape {arr.shape}.")

    if obj in DATAFRAME_PATHS:
        raise ValueError(
            f"'{obj}' must hold a dataframe; writing an image there would "
            "corrupt the store. Images belong somewhere unstructured, such as "
            "'uns/spatial/hires'."
        )
    obj = check_target(obj, "array")

    # An image's height is not an axis, so a path that implies one is almost
    # certainly a mistake -- but check it rather than assume.
    validate_dimensions(root, obj, arr.shape, console)

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else write_mapping(parent, part)

    write_dense(parent, parts[-1], arr, replace=True)
    console.print(
        f"[green]Imported[/] {'x'.join(str(d) for d in arr.shape)} "
        f"image ({arr.dtype}) into '{literal(obj)}'"
    )

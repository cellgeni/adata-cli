from __future__ import annotations

from typing import Optional, Tuple, Any

from rich.console import Console

from adata.core.info import axis_len
from adata.util.path import norm_path


OBS_AXIS_PREFIXES = ("obs", "obsm/", "obsp/")
VAR_AXIS_PREFIXES = ("var", "varm/", "varp/")
MATRIX_PREFIXES = ("X", "layers/")

#: Paths that must hold a dataframe, so writing an array there would corrupt
#: the store rather than merely mis-size it.
DATAFRAME_PATHS = frozenset({"obs", "var", "raw/var"})


def _get_axis_length(root: Any, axis: str) -> Optional[int]:
    try:
        return axis_len(root, axis)
    except Exception:
        return None


def _raw_var_length(root: Any) -> Optional[int]:
    """Number of variables in `raw`, which is its own axis.

    Taken from `raw/X`'s declared width where possible: that is the invariant
    a replacement `raw/var` has to keep, and it usually differs from the main
    object's var count.
    """
    if "raw" not in root:
        return None
    raw = root["raw"]

    if "X" in raw:
        shape = raw["X"].attrs.get("shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
        x_shape = getattr(raw["X"], "shape", None)
        if x_shape is not None and len(x_shape) >= 2:
            return int(x_shape[1])

    if "var" in raw:
        try:
            return axis_len(raw, "var")
        except Exception:
            return None
    return None


def _validate_raw(
    root: Any, obj_path: str, data_shape: Tuple[int, ...], console: Console
) -> bool:
    """Validate a path under `raw/`. Returns whether the path was recognised."""
    if obj_path == "raw" or not obj_path.startswith("raw/"):
        return False

    n_raw_var = _raw_var_length(root)
    n_obs = _get_axis_length(root, "obs")
    rest = obj_path[len("raw/"):]

    if rest == "var":
        if n_raw_var is not None and data_shape[0] != n_raw_var:
            raise ValueError(
                f"Row count mismatch: input has {data_shape[0]} rows, but raw "
                f"has {n_raw_var} variables. Replacing raw/var with a "
                "different length would leave raw/X inconsistent."
            )
        return True

    if rest == "X":
        if len(data_shape) < 2:
            raise ValueError(
                f"raw/X requires 2D data, got {len(data_shape)}D."
            )
        if n_obs is not None and data_shape[0] != n_obs:
            raise ValueError(
                f"First dimension mismatch: input has {data_shape[0]} rows, "
                f"but obs has {n_obs} cells."
            )
        if n_raw_var is not None and data_shape[1] != n_raw_var:
            raise ValueError(
                f"Second dimension mismatch: input has {data_shape[1]} columns, "
                f"but raw has {n_raw_var} variables."
            )
        return True

    if rest.startswith("varm/"):
        if n_raw_var is not None and data_shape[0] != n_raw_var:
            raise ValueError(
                f"First dimension mismatch: input has {data_shape[0]} rows, "
                f"but raw has {n_raw_var} variables."
            )
        return True

    return False


def validate_dimensions(
    root: Any,
    obj_path: str,
    data_shape: Tuple[int, ...],
    console: Console,
) -> None:
    obj_path = norm_path(obj_path)

    if _validate_raw(root, obj_path, data_shape, console):
        return

    n_obs = _get_axis_length(root, "obs")
    n_var = _get_axis_length(root, "var")

    if obj_path == "obs":
        if n_obs is not None and data_shape[0] != n_obs:
            raise ValueError(
                f"Row count mismatch: input has {data_shape[0]} rows, "
                f"but obs has {n_obs} cells."
            )
        return
    if obj_path == "var":
        if n_var is not None and data_shape[0] != n_var:
            raise ValueError(
                f"Row count mismatch: input has {data_shape[0]} rows, "
                f"but var has {n_var} features."
            )
        return

    for prefix in MATRIX_PREFIXES:
        if obj_path == prefix or obj_path.startswith(prefix + "/") or obj_path.startswith(prefix):
            if obj_path == "X" or obj_path.startswith("layers/"):
                if len(data_shape) < 2:
                    raise ValueError(
                        f"Matrix data requires 2D shape, got {len(data_shape)}D."
                    )
                if n_obs is not None and data_shape[0] != n_obs:
                    raise ValueError(
                        f"First dimension mismatch: input has {data_shape[0]} rows, "
                        f"but obs has {n_obs} cells."
                    )
                if n_var is not None and data_shape[1] != n_var:
                    raise ValueError(
                        f"Second dimension mismatch: input has {data_shape[1]} columns, "
                        f"but var has {n_var} features."
                    )
                return

    for prefix in OBS_AXIS_PREFIXES:
        if obj_path.startswith(prefix) and obj_path != "obs":
            if n_obs is not None and data_shape[0] != n_obs:
                raise ValueError(
                    f"First dimension mismatch: input has {data_shape[0]} rows, "
                    f"but obs has {n_obs} cells."
                )
            if obj_path.startswith("obsp/") and len(data_shape) >= 2:
                if data_shape[1] != n_obs:
                    raise ValueError(
                        "obsp matrix must be square (n_obs × n_obs): "
                        f"got {data_shape[0]}×{data_shape[1]}, expected {n_obs}×{n_obs}."
                    )
            return

    for prefix in VAR_AXIS_PREFIXES:
        if obj_path.startswith(prefix) and obj_path != "var":
            if n_var is not None and data_shape[0] != n_var:
                raise ValueError(
                    f"First dimension mismatch: input has {data_shape[0]} rows, "
                    f"but var has {n_var} features."
                )
            if obj_path.startswith("varp/") and len(data_shape) >= 2:
                if data_shape[1] != n_var:
                    raise ValueError(
                        "varp matrix must be square (n_var × n_var): "
                        f"got {data_shape[0]}×{data_shape[1]}, expected {n_var}×{n_var}."
                    )
            return

    console.print(f"[dim]Note: No dimension validation for path '{obj_path}'[/]")

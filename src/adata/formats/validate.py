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

#: Groups that hold one element per key: each key is replaceable, the group
#: itself is not.
MAPPING_PATHS = frozenset({"obsm", "varm", "obsp", "varp", "layers", "uns", "raw/varm"})

#: Where an entry may not be of a given kind: a dataframe's columns are 1-D
#: arrays, a matrix is no mapping, and obsp, varp and layers hold matrices.
_REFUSED_UNDER = {
    "obs": {"sparse", "dict", "dataframe"},
    "var": {"sparse", "dict", "dataframe"},
    "raw/var": {"sparse", "dict", "dataframe"},
    "obsm": {"dict"},
    "varm": {"dict"},
    "raw/varm": {"dict"},
    "obsp": {"dict", "dataframe"},
    "varp": {"dict", "dataframe"},
    "layers": {"dict", "dataframe"},
}

_WHAT = {
    "array": "an array",
    "sparse": "a sparse matrix",
    "dict": "JSON",
    "dataframe": "a dataframe",
}

_UNREADABLE = "would leave a store anndata cannot read"

#: The files Zarr keeps a group's or an array's metadata in: a member of
#: that name would overwrite them, or vanish behind them.
_ZARR_METADATA = frozenset({".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"})


def check_element_name(name: str, where: str) -> None:
    """Refuse a name no element can have in every backend, before writing.

    `where` says what holds the name, for the message: a JSON object, or a
    CSV header. A name holding '/' would make nested groups, and an empty
    one, '.' or '..' names no member at all.
    """
    if name in ("", ".", "..") or "/" in name or name in _ZARR_METADATA:
        raise ValueError(
            f"{where} has the name {name!r}, which cannot name an element: names "
            "may not be empty, '.' or '..', hold '/', or be one of Zarr's "
            f"metadata files ({', '.join(sorted(_ZARR_METADATA))})."
        )


def parent_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def check_target(obj_path: str, kind: str) -> str:
    """The normalised path, once it is known that a `kind` element may go
    there: "array", "sparse", "dict" or "dataframe".

    Run before anything is read or written. An element of the wrong kind
    where the layout fixes one (an array in place of `obs`, a matrix in
    place of the whole of `obsm`) used to be written, and anndata could
    then not read the store at all. An entry the user names inside such a
    group (`obsm/X_pca`, `uns/params`) is theirs to replace.
    """
    path = norm_path(obj_path).strip("/")
    what = _WHAT[kind]
    if path in ("", "raw"):
        raise ValueError(
            f"'{path or '/'}' is a group of elements, not one; name the entry "
            "to write, such as 'X', 'obsm/X_pca' or 'uns/params'."
        )
    if path in DATAFRAME_PATHS:
        if kind != "dataframe":
            raise ValueError(
                f"'{path}' must hold a dataframe; writing {what} there {_UNREADABLE}. "
                "Use `adata import dataframe` for a table."
            )
        return path
    if path == "uns" and kind == "dict":
        return path
    if path in MAPPING_PATHS:
        hint = "a JSON object, with `adata import dict`, or " if path == "uns" else ""
        raise ValueError(
            f"'{path}' holds one element per key; writing {what} in its place "
            f"{_UNREADABLE}. Write {hint}an entry inside it, such as '{path}/name'."
        )
    if path in ("X", "raw/X") and kind in ("dict", "dataframe"):
        raise ValueError(f"'{path}' must hold a matrix; writing {what} there {_UNREADABLE}.")
    parent = parent_of(path)
    if kind in _REFUSED_UNDER.get(parent, ()):
        raise ValueError(
            f"'{path}' is an entry of '{parent}', which cannot hold {what}: "
            f"writing it {_UNREADABLE}."
        )
    return path


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

    if parent_of(obj_path.strip("/")) in DATAFRAME_PATHS and len(data_shape) != 1:
        raise ValueError(
            f"'{obj_path}' would be a column of '{parent_of(obj_path.strip('/'))}', "
            f"which holds 1-D arrays, but the input's shape is "
            f"{'x'.join(map(str, data_shape)) or 'a scalar'}."
        )

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

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from rich.console import Console

from adata.core.read import decode_str_array
from adata.formats.common import _check_json_exportable, _resolve
from adata.elements.write import (
    write_dense,
    write_mapping,
    write_null,
    write_scalar,
    write_string_array,
)
from adata.storage import create_dataset, is_dataset, is_group
from adata.util.path import norm_path


def export_json(
    root: Any,
    obj: str,
    out: Path | None,
    max_elements: int,
    include_attrs: bool,
    console: Console,
) -> None:
    h5obj = _resolve(root, obj)
    _check_json_exportable(h5obj, max_elements=max_elements)

    payload = _to_jsonable(
        h5obj, max_elements=max_elements, include_attrs=include_attrs
    )
    if out is None or str(out) == "-":
        out_fh = sys.stdout
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out, "w", encoding="utf-8")
    try:
        json.dump(payload, out_fh, indent=2, ensure_ascii=False, sort_keys=True)
        out_fh.write("\n")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()
    if out_fh is not sys.stdout:
        console.print(f"[green]Wrote[/] {out}")


def _attrs_to_jsonable(attrs: Any, max_elements: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in attrs.keys():
        v = attrs.get(k)
        out[str(k)] = _pyify(v, max_elements=max_elements)
    return out


def _pyify(value: Any, max_elements: int) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > max_elements:
            raise ValueError(
                f"Refusing to convert array of size {value.size} (> {max_elements}) to JSON."
            )
        if np.issubdtype(value.dtype, np.bytes_) or value.dtype.kind == "O":
            value = decode_str_array(value)
        return value.tolist()
    return value


def _dataset_to_jsonable(ds: Any, max_elements: int) -> Any:
    if ds.shape == ():
        v = ds[()]
        return _pyify(v, max_elements=max_elements)
    n = int(np.prod(ds.shape)) if ds.shape else 0
    if n > max_elements:
        ds_name = getattr(ds, "name", "<dataset>")
        raise ValueError(
            f"Refusing to convert dataset {ds_name!r} with {n} elements (> {max_elements}) to JSON."
        )
    arr = np.asarray(ds[...])
    return _pyify(arr, max_elements=max_elements)


def _to_jsonable(h5obj: Any, max_elements: int, include_attrs: bool) -> Any:
    if is_dataset(h5obj):
        return _dataset_to_jsonable(h5obj, max_elements=max_elements)

    d: Dict[str, Any] = {}
    if include_attrs and len(h5obj.attrs):
        d["__attrs__"] = _attrs_to_jsonable(h5obj.attrs, max_elements=max_elements)

    for key in h5obj.keys():
        child = h5obj[key]
        if is_group(child) or is_dataset(child):
            d[str(key)] = _to_jsonable(
                child,
                max_elements=max_elements,
                include_attrs=include_attrs,
            )
    return d


def import_json(
    root: Any,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    obj = norm_path(obj)
    with open(input_file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else parent.create_group(part)
    name = parts[-1]

    if name in parent:
        del parent[name]

    _write_json_to_group(parent, name, payload)

    console.print(f"[green]Imported[/] JSON data into '{obj}'")


def _write_json_to_group(parent: Any, name: str, value: Any) -> None:
    """Write one JSON value as the AnnData element that best represents it."""
    if isinstance(value, dict):
        group = write_mapping(parent, name, replace=True)
        for k, v in value.items():
            _write_json_to_group(group, k, v)
        return

    if value is None:
        write_null(parent, name, replace=True)
        return

    if isinstance(value, str):
        write_scalar(parent, name, value, replace=True)
        return

    if isinstance(value, (bool, int, float)):
        write_scalar(parent, name, value, replace=True)
        return

    if isinstance(value, list):
        _write_json_list(parent, name, value)
        return

    raise ValueError(f"Cannot convert JSON value of type {type(value).__name__}")


def _write_json_list(parent: Any, name: str, value: list) -> None:
    """Write a JSON array as a string or numeric array where it is uniform.

    Ragged or mixed lists have no array representation in the spec, so they
    are stored as their JSON text rather than silently reshaped.
    """
    if all(isinstance(v, str) for v in value):
        write_string_array(parent, name, value, replace=True)
        return

    try:
        arr = np.array(value)
    except (ValueError, TypeError):
        arr = None

    if arr is not None and arr.dtype.kind in ("b", "i", "u", "f"):
        write_dense(parent, name, arr, replace=True)
        return

    if arr is not None and arr.dtype.kind in ("U", "S", "O", "T"):
        write_string_array(parent, name, arr.reshape(-1).tolist(), replace=True)
        return

    write_scalar(parent, name, json.dumps(value), replace=True)

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.core.read import decode_str_array
from adata.elements import spec
from adata.formats.common import _check_json_exportable, _resolve, stdout_reader_may_stop
from adata.formats.validate import check_target
from adata.elements.write import (
    write_dense,
    write_mapping,
    write_scalar,
    write_string_array,
)
from adata.storage import is_dataset, is_group


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
    nonfinite = [0]
    payload = _finite(payload, nonfinite)
    if nonfinite[0]:
        console.print(
            f"[yellow]Note:[/] {nonfinite[0]:,} NaN or infinite "
            f"{'value is' if nonfinite[0] == 1 else 'values are'} written as null, "
            "since JSON has no number for them."
        )
    if out is None or str(out) == "-":
        out_fh = sys.stdout
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out, "w", encoding="utf-8")
    try:
        with stdout_reader_may_stop():
            json.dump(
                payload, out_fh, indent=2, ensure_ascii=False, sort_keys=True,
                allow_nan=False,
            )
            out_fh.write("\n")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()
    if out_fh is not sys.stdout:
        console.print(f"[green]Wrote[/] {out}")


def _finite(value: Any, count: List[int]) -> Any:
    """`value` with every NaN and infinity made None, counted in `count`.

    `json.dump` writes them as `NaN` and `Infinity`, which are not JSON:
    jq and most parsers refuse the whole file.
    """
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        count[0] += 1
        return None
    if isinstance(value, list):
        return [_finite(v, count) for v in value]
    if isinstance(value, dict):
        return {k: _finite(v, count) for k, v in value.items()}
    return value


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
    if spec.encoding_type(ds) == spec.NULL:
        # A null is a placeholder -- an h5py.Empty or a 0-d zarr bool -- whose
        # stored value is meaningless. Reading it would emit that placeholder
        # rather than restoring the None it represents.
        return None

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
    """Write a JSON file's value at `obj`, as the elements anndata would.

    Everything is checked before the store is touched. A `null` has no
    encoding anndata 0.8 to 0.11 can read, so a key holding one is left
    out, as those releases leave out a `None`, and named in a warning.
    """
    obj = check_target(obj, "dict")
    with open(input_file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if obj == "uns" and not isinstance(payload, dict):
        raise ValueError(
            f"'uns' must stay a mapping, but {Path(input_file).name} holds a "
            f"JSON {_json_type(payload)}. Name an entry inside it, such as 'uns/name'."
        )
    if payload is None:
        raise ValueError(f"{Path(input_file).name} holds only null: there is nothing to write.")
    plan = _plan(payload, obj)

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else write_mapping(parent, part)
    name = parts[-1]

    if name in parent:
        del parent[name]

    skipped = _write_plan(parent, name, plan, obj)
    if skipped:
        console.print(
            f"[yellow]Warning:[/] {len(skipped)} null "
            f"{'value was' if len(skipped) == 1 else 'values were'} left out, as "
            f"anndata leaves out None: {', '.join(skipped[:5])}"
            f"{', ...' if len(skipped) > 5 else ''}"
        )

    console.print(f"[green]Imported[/] JSON data into '{obj}'")


def write_json_value(parent: Any, name: str, value: Any) -> List[str]:
    """Write one JSON value as the AnnData element that best represents it.

    Returns the paths of the nulls left out (see `import_json`).
    """
    if value is None:
        return [name]
    return _write_plan(parent, name, _plan(value, name), name)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return {str: "string", list: "array", dict: "object"}.get(type(value), type(value).__name__)


_INT64 = np.iinfo(np.int64)
_UINT64 = np.iinfo(np.uint64)


def _integer_dtype(values: List[int], where: str) -> Any:
    """int64, or uint64 for whole numbers past int64; refused past both."""
    low, high = min(values), max(values)
    if low >= _INT64.min and high <= _INT64.max:
        return np.int64
    if low >= 0 and high <= _UINT64.max:
        return np.uint64
    where = where or "The value"
    if low < _INT64.min or high > _UINT64.max:
        number = low if low < _INT64.min else high
        raise ValueError(f"{where} holds the whole number {number}, which no 64-bit integer holds.")
    raise ValueError(
        f"{where} holds both {low} and {high}, which no one 64-bit integer type holds."
    )


def _leaves(value: list) -> Optional[Tuple[Tuple[int, ...], List[Any]]]:
    """The shape and the values, in row order, of a JSON array of arrays
    that is rectangular all the way down; None for a ragged one."""
    if not value or not isinstance(value[0], list):
        if any(isinstance(v, list) for v in value):
            return None
        return (len(value),), list(value)
    inner = [_leaves(v) if isinstance(v, list) else None for v in value]
    if any(i is None for i in inner) or len({i[0] for i in inner}) != 1:
        return None
    return (len(value),) + inner[0][0], [leaf for i in inner for leaf in i[1]]


def _plan(value: Any, where: str) -> Tuple[str, Any]:
    """What to write for `value`, checked whole before anything is written:
    ("skip", None) for a null, ("dict", {key: plan}), ("scalar", value),
    ("strings", array), ("array", array) or ("text", its JSON).
    """
    if value is None:
        return "skip", None
    if isinstance(value, dict):
        plans = {}
        for key, item in value.items():
            if key in ("", ".", "..") or "/" in key:
                raise ValueError(
                    f"{where or 'The JSON'} has the key {key!r}, which cannot name "
                    "an element: keys may not be empty, '.' or '..', or hold '/'."
                )
            plans[key] = _plan(item, f"{where}/{key}" if where else key)
        return "dict", plans
    if isinstance(value, bool) or isinstance(value, (str, float)):
        return "scalar", value
    if isinstance(value, int):
        return "scalar", np.asarray(value, dtype=_integer_dtype([value], where))
    if isinstance(value, list):
        return _plan_list(value, where)
    raise ValueError(f"Cannot convert JSON value of type {type(value).__name__}")


def _plan_list(value: list, where: str) -> Tuple[str, Any]:
    """A JSON array as a typed array where its values share a type.

    Strings make a string array, booleans a bool array, whole numbers an
    integer one and numbers with a fraction float64, with NaN for a null
    among numbers. Anything else (strings beside numbers, nulls beside
    strings, a ragged array) has no array encoding, so it is kept as its
    JSON text rather than stringified value by value.
    """
    found = _leaves(value)
    if found is not None:
        shape, leaves = found
        kinds = {_json_type(v) for v in leaves}
        if not leaves:
            return "array", np.zeros(shape, dtype=np.float64)
        if kinds == {"string"}:
            return "strings", np.asarray(leaves, dtype=object).reshape(shape)
        if kinds == {"boolean"}:
            return "array", np.asarray(leaves, dtype=bool).reshape(shape)
        if kinds <= {"number", "null"} and "number" in kinds:
            whole = all(isinstance(v, int) for v in leaves)
            if whole:
                dtype = _integer_dtype(leaves, where)
                return "array", np.asarray(leaves, dtype=dtype).reshape(shape)
            try:
                numbers = [np.nan if v is None else float(v) for v in leaves]
            except OverflowError:
                raise ValueError(
                    f"{where or 'The value'} holds a whole number too large for float64."
                ) from None
            return "array", np.asarray(numbers, dtype=np.float64).reshape(shape)
    return "text", json.dumps(value)


def _write_plan(parent: Any, name: str, plan: Tuple[str, Any], where: str) -> List[str]:
    kind, value = plan
    if kind == "skip":
        return [where]
    if kind == "dict":
        group = write_mapping(parent, name, replace=True)
        skipped: List[str] = []
        for key, item in value.items():
            skipped += _write_plan(group, key, item, f"{where}/{key}")
        return skipped
    if kind in ("scalar", "text"):
        write_scalar(parent, name, value, replace=True)
    elif kind == "strings":
        write_string_array(parent, name, value, replace=True)
    else:
        write_dense(parent, name, value, replace=True)
    return []

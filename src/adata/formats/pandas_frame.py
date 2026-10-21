"""A pandas DataFrame written in the AnnData on-disk spec, column by column.

Only the pickle path uses this: pandas is an optional extra here, and every
other reader builds its columns from numpy arrays and text. Each column is
written as anndata's writer would write it:

* numbers and booleans as arrays, their dtype kept;
* a Categorical as a `categorical`, its categories (text or numbers) and
  order kept;
* pandas' nullable integers and booleans as `nullable-integer` and
  `nullable-boolean`; nullable floats as floats with NaN;
* text as a `string-array`, or a `nullable-string-array` when a value is
  missing; in obs and var (`categorize=True`) text with a repeated value
  becomes a categorical, as anndata's `strings_to_categoricals` makes it;
* an `object` column by what it holds (`pd.api.types.infer_dtype`): numbers
  as numbers, text as text, anything mixed as text with a warning;
* datetimes as ISO text, with a warning. Anything else is left out, with a
  warning.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from adata.elements import spec
from adata.elements.write import (
    _codes_dtype,
    set_column_order,
    write_categorical,
    write_dataframe_header,
    write_dense,
    write_masked,
    write_string_array,
)
from adata.util.names import natural_sorted


def write_frame(
    parent: Any,
    name: str,
    df: Any,
    *,
    index: Optional[Sequence[str]] = None,
    categorize: bool = False,
    warnings: Optional[List[str]] = None,
    what: str = "",
) -> Any:
    """`df` as a `dataframe` group `parent[name]`; returns the group.

    `index` replaces the frame's own index, which is written as text. `what`
    names the frame in warnings (`obs`, `uns/pycistopic/metrics`).
    """
    warnings = warnings if warnings is not None else []
    labels = [str(v) for v in (df.index if index is None else index)]
    group = write_dataframe_header(parent, name, labels, [])
    written = []
    for position, column in enumerate(df.columns):
        key = str(column)
        if key in spec.RESERVED_DATAFRAME_KEYS or key in written:
            warnings.append(f"Not written: {what or name} column {key!r}, a name a dataframe reserves or repeats.")
            continue
        # By position: a repeated label selects every column it names.
        if write_column(group, key, df.iloc[:, position], categorize=categorize,
                        warnings=warnings, what=f"{what or name}[{key!r}]"):
            written.append(key)
    set_column_order(group, written)
    return group


def write_column(
    parent: Any, name: str, column: Any, *, categorize: bool, warnings: List[str], what: str
) -> bool:
    """One pandas Series as a column; False when it is left out."""
    import pandas as pd

    dtype = column.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        _categorical(parent, name, np.asarray(column.cat.codes), dtype.categories, dtype.ordered)
        return True
    if isinstance(dtype, pd.BooleanDtype) or (
        isinstance(dtype, pd.api.extensions.ExtensionDtype) and dtype.kind in "iu"
        and not isinstance(dtype, pd.CategoricalDtype)
    ):
        array = column.array
        mask = np.asarray(array.isna() if hasattr(array, "isna") else column.isna(), dtype=bool)
        fill = False if isinstance(dtype, pd.BooleanDtype) else 0
        values = column.to_numpy(dtype=dtype.numpy_dtype, na_value=fill)
        enc = spec.NULLABLE_BOOLEAN if isinstance(dtype, pd.BooleanDtype) else spec.NULLABLE_INTEGER
        write_masked(parent, name, values, mask, enc)
        return True
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and dtype.kind == "f":
        write_dense(parent, name, column.to_numpy(dtype=dtype.numpy_dtype, na_value=np.nan))
        return True
    if dtype.kind in "biufc":
        write_dense(parent, name, column.to_numpy())
        return True
    if dtype.kind == "M":
        warnings.append(f"{what} holds datetimes; written as ISO text.")
        _text(parent, name, [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in column],
              categorize=categorize)
        return True
    if dtype.kind in "OSU" or isinstance(dtype, pd.StringDtype):
        inferred = pd.api.types.infer_dtype(column, skipna=True)
        if inferred in ("integer", "floating", "mixed-integer-float", "decimal"):
            values = pd.to_numeric(column, errors="coerce")
            if inferred == "integer" and not values.isna().any():
                write_dense(parent, name, values.to_numpy(dtype=np.int64))
            else:
                write_dense(parent, name, values.to_numpy(dtype=np.float64))
            return True
        if inferred == "boolean" and not column.isna().any():
            write_dense(parent, name, column.to_numpy(dtype=bool))
            return True
        if categorize and inferred not in ("string", "empty", "unicode", "bytes"):
            # In obs and var a column of mixed values is worth a word; in a
            # table of parameters ("lda", 15, True) it is what is expected.
            warnings.append(f"{what} holds {inferred} values; written as text.")
        _text(
            parent, name,
            [None if _missing(v) else (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
             for v in column],
            categorize=categorize,
        )
        return True
    warnings.append(f"Not written: {what}, of dtype {dtype}, which no column encoding holds.")
    return False


def _missing(value: Any) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(parent: Any, name: str, values: List[Optional[str]], *, categorize: bool) -> None:
    present = [v for v in values if v is not None]
    if categorize:
        categories = natural_sorted(set(present))
        if len(categories) < len(values):
            lookup = {c: i for i, c in enumerate(categories)}
            write_categorical(parent, name, [lookup.get(v, -1) if v is not None else -1
                                             for v in values], categories)
            return
    if len(present) < len(values):
        write_masked(
            parent, name, np.asarray(["" if v is None else v for v in values], dtype=object),
            np.asarray([v is None for v in values], dtype=bool), spec.NULLABLE_STRING_ARRAY,
        )
        return
    write_string_array(parent, name, np.asarray(values, dtype=object))


def _categorical(parent: Any, name: str, codes: np.ndarray, categories: Any, ordered: bool) -> None:
    values = np.asarray(categories)
    if values.dtype.kind in "biuf":
        group = parent.create_group(name)
        spec.set_encoding(group, spec.CATEGORICAL)
        group.attrs["ordered"] = bool(ordered)
        write_dense(group, "categories", values)
        write_dense(group, "codes", codes.astype(_codes_dtype(len(values))))
        return
    write_categorical(parent, name, codes, [str(v) for v in values], ordered=bool(ordered))

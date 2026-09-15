from __future__ import annotations

import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.core.read import col_chunk_as_strings
from adata.formats.common import _resolve
from adata.elements.read import dataframe_columns, element_len, resolve_index
from adata.formats.validate import validate_dimensions
from adata.elements.write import (
    write_categorical,
    write_dataframe_header,
    write_dense,
    write_string_array,
)
from adata.storage import create_dataset, is_group, is_zarr_group


def export_dataframe(
    root: Any,
    axis: str,
    columns: Optional[List[str]],
    out: Optional[Path],
    chunk_rows: int,
    head: Optional[int],
    console: Console,
) -> None:
    """Stream a dataframe group out as CSV.

    `axis` is any path to a dataframe-encoded group, not only "obs" or "var" --
    a store may hold dataframes in obsm, varm, uns or raw as well.
    """
    group = _resolve(root, axis)
    if not is_group(group):
        raise ValueError(f"'{axis}' is not a group and cannot be exported as CSV.")

    try:
        index, index_name = resolve_index(group)
    except KeyError as exc:
        raise ValueError(
            f"'{axis}' is not a dataframe: it has no index. "
            "CSV export needs a dataframe-encoded group such as 'obs', 'var' "
            "or 'raw/var'."
        ) from exc
    n_rows = element_len(index)

    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")

    if columns:
        col_names = list(columns)
    else:
        col_names = dataframe_columns(group, index_name)

    if index_name not in col_names:
        col_names.insert(0, index_name)
    else:
        col_names = [index_name] + [c for c in col_names if c != index_name]

    if head is not None and head > 0:
        n_rows = min(n_rows, head)

    if out is None or str(out) == "-":
        out_fh = sys.stdout
    else:
        out_fh = open(out, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_fh)

    try:
        writer.writerow(col_names)
        cat_cache = {}

        use_status = out_fh is not sys.stdout
        status_ctx = (
            console.status(f"[magenta]Exporting {axis} table to {out}...[/]")
            if use_status
            else nullcontext()
        )

        with status_ctx as status:
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                if use_status and status:
                    status.update(
                        f"[magenta]Exporting rows {start}-{end} of {n_rows}...[/]"
                    )
                cols_data: List[List[str]] = []
                for col in col_names:
                    cols_data.append(
                        col_chunk_as_strings(group, col, start, end, cat_cache)
                    )
                for row_idx in range(end - start):
                    row = [
                        cols_data[col_idx][row_idx]
                        for col_idx in range(len(col_names))
                    ]
                    writer.writerow(row)
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


def _read_csv(
    input_file: Path,
    index_column: Optional[str],
) -> Tuple[List[dict], List[str], List[str], str]:
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")
        fieldnames = list(reader.fieldnames)

        if index_column:
            if index_column not in fieldnames:
                raise ValueError(
                    f"Index column '{index_column}' not found in CSV. "
                    f"Available columns: {', '.join(fieldnames)}"
                )
            idx_col = index_column
        else:
            idx_col = fieldnames[0]

        rows = list(reader)

    index_values = [row[idx_col] for row in rows]
    data_columns = [c for c in fieldnames if c != idx_col]

    return rows, data_columns, index_values, idx_col


def _looks_categorical(values: List[str], n_rows: int) -> bool:
    """Heuristic: few enough repeated labels that a category is the better fit.

    Mirrors the usual pandas rule of thumb -- a column whose distinct values
    number well under half its rows is almost always a label, not free text.
    Callers can always override explicitly.
    """
    n_distinct = len(set(values))
    return 0 < n_distinct <= min(1000, max(1, n_rows // 2))


def _write_column(
    group: Any,
    name: str,
    values: List[str],
    n_rows: int,
    categorical: bool,
) -> None:
    """Write one CSV column, choosing the narrowest faithful encoding."""
    try:
        write_dense(group, name, np.array(values, dtype=np.int64))
        return
    except (ValueError, TypeError, OverflowError):
        pass

    try:
        write_dense(group, name, np.array(values, dtype=np.float64))
        return
    except (ValueError, TypeError):
        pass

    if categorical:
        categories = sorted(set(values))
        lookup = {c: i for i, c in enumerate(categories)}
        write_categorical(
            group, name, [lookup[v] for v in values], categories, ordered=False
        )
        return

    write_string_array(group, name, values)


def import_dataframe(
    root: Any,
    obj: str,
    input_file: Path,
    index_column: Optional[str],
    console: Console,
    categorical: Optional[List[str]] = None,
    auto_categorical: bool = True,
) -> None:
    """Replace `obs` or `var` with the contents of a CSV file.

    Columns that parse cleanly as integers or floats become numeric arrays;
    the rest become either a `categorical` or a `string-array`. `categorical`
    names columns to force, and `auto_categorical` applies the heuristic to
    the remainder.
    """
    if obj not in ("obs", "var"):
        raise ValueError(
            f"CSV import is only supported for 'obs' or 'var', not '{obj}'."
        )

    rows, data_columns, index_values, _ = _read_csv(input_file, index_column)
    n_rows = len(rows)

    validate_dimensions(root, obj, (n_rows,), console)

    index_name = "_index"
    group = write_dataframe_header(
        root, obj, index_values, data_columns, index_name=index_name
    )

    forced = set(categorical or ())
    n_categorical = 0
    for col in data_columns:
        values = [row[col] for row in rows]
        as_categorical = col in forced or (
            auto_categorical and _looks_categorical(values, n_rows)
        )
        _write_column(group, col, values, n_rows, as_categorical)
        if as_categorical and not _is_numeric(group, col):
            n_categorical += 1

    detail = f" ({n_categorical} categorical)" if n_categorical else ""
    console.print(
        f"[green]Imported[/] {n_rows} rows x {len(data_columns)} columns "
        f"into '{obj}'{detail}"
    )


def _is_numeric(group: Any, col: str) -> bool:
    """True when the written column ended up as a numeric array."""
    from adata.elements import spec

    return spec.encoding_type(group[col]) == spec.ARRAY

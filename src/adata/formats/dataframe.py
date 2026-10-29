from __future__ import annotations

import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console

from adata.core.read import col_chunk_as_strings
from adata.elements.read import dataframe_columns, element_len, resolve_index
from adata.elements.write import write_mapping
from adata.formats.common import _resolve, stdout_reader_may_stop
from adata.formats.table import CATEGORY_CAP, scan_table, write_table
from adata.formats.validate import check_target, validate_dimensions
from adata.storage import is_group


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

    Everything that can be refused is checked before `out` is opened, so a
    refusal leaves no partial file behind. Lines end in `\n` alone, and
    `head=0` writes the header and no rows.
    """
    group = _resolve(root, axis)
    if not is_group(group):
        raise ValueError(f"'{axis}' is not a group and cannot be exported as CSV.")
    if head is not None and head < 0:
        raise ValueError(f"--head must be 0 or more, not {head}.")

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

    available = dataframe_columns(group, index_name)
    if columns:
        col_names = list(columns)
        absent = [c for c in col_names if c != index_name and c not in group]
        if absent:
            raise ValueError(
                f"'{axis}' has no column {', '.join(map(repr, absent))}; it has: "
                f"{', '.join(available) or 'only its index'}."
            )
    else:
        col_names = available

    if index_name not in col_names:
        col_names.insert(0, index_name)
    else:
        col_names = [index_name] + [c for c in col_names if c != index_name]

    if head is not None:
        n_rows = min(n_rows, head)

    to_stdout = out is None or str(out) == "-"
    out_fh = sys.stdout if to_stdout else open(out, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_fh, lineterminator="\n")

    try:
        with stdout_reader_may_stop():
            writer.writerow(col_names)
            cat_cache = {}

            status_ctx = (
                nullcontext()
                if to_stdout
                else console.status(f"[magenta]Exporting {axis} table to {out}...[/]")
            )

            with status_ctx as status:
                for start in range(0, n_rows, chunk_rows):
                    end = min(start + chunk_rows, n_rows)
                    if status:
                        status.update(
                            f"[magenta]Exporting rows {start}-{end} of {n_rows}...[/]"
                        )
                    cols_data: List[List[str]] = []
                    for col in col_names:
                        cols_data.append(
                            col_chunk_as_strings(group, col, start, end, cat_cache)
                        )
                    writer.writerows(zip(*cols_data))
    finally:
        if not to_stdout:
            out_fh.close()


def import_dataframe(
    root: Any,
    obj: str,
    input_file: Path,
    index_column: Optional[str],
    console: Console,
    categorical: Optional[List[str]] = None,
    auto_categorical: bool = True,
) -> None:
    """Replace a dataframe at `obj` with the contents of a CSV file.

    `obj` is any dataframe path, not only "obs" or "var". The file is read
    in two passes through `formats.table`, so memory is set by its block,
    not by the file: the first types each column as `pd.read_csv` would
    (missing numbers are NaN, True/False is bool) and checks every row's
    width; the second writes the frame as anndata writes it, a text column
    that repeats a value as a categorical. `categorical` names text columns
    to make categoricals whatever their values, and `auto_categorical=False`
    keeps every other text column as strings.
    """
    obj = check_target(obj, "dataframe")
    input_file = Path(input_file)
    scan = scan_table(input_file, index=index_column)
    by_name = {column.name: column for column in scan.columns}
    if "_index" in by_name:
        raise ValueError(
            f"{input_file.name} has a column named '_index' besides its index "
            f"column {scan.index!r}; anndata keeps that name for the index."
        )
    forced = list(dict.fromkeys(categorical or ()))
    unknown = [name for name in forced if name not in by_name]
    if unknown:
        raise ValueError(
            f"--categorical names {', '.join(map(repr, unknown))}, which "
            f"{input_file.name} does not have as a column (other than its index)."
        )

    validate_dimensions(root, obj, (scan.n_rows,), console)

    for column in scan.columns:
        kind = scan.dtype_of(column)
        if column.name in forced:
            if kind not in ("str", "category"):
                console.print(
                    f"[yellow]Note:[/] {column.name!r} holds {kind} values, so it "
                    "stays as it is rather than a categorical."
                )
            elif column.seen is None:
                console.print(
                    f"[yellow]Warning:[/] {column.name!r} has more than "
                    f"{CATEGORY_CAP:,} distinct values, so it is written as strings."
                )
            else:
                scan.overrides[column.name] = "category"
        elif kind == "category" and not auto_categorical:
            scan.overrides[column.name] = "str"

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else write_mapping(parent, part)
    if parts[-1] in parent:
        del parent[parts[-1]]
    write_table(parent, parts[-1], scan)

    n_categorical = sum(scan.dtype_of(c) == "category" for c in scan.columns)
    detail = f" ({n_categorical} categorical)" if n_categorical else ""
    console.print(
        f"[green]Imported[/] {scan.n_rows} rows x {len(scan.columns)} columns "
        f"into '{obj}'{detail}"
    )

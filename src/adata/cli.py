"""CLI for AnnData stores (.h5ad / .zarr) with export and import subcommands."""

from pathlib import Path
from typing import Optional, Sequence, List

from rich.console import Console
import typer

from adata.commands import (
    list_store,
    show_info,
    subset_h5ad,
    export_mtx,
    export_npy,
    export_json,
    export_table,
)

from adata.commands import export_image as export_image_cmd

app = typer.Typer(
    help="Streaming CLI for huge AnnData .h5ad and .zarr stores "
    "(view, ls, subset, export, import)."
)
# Use stderr for status/progress to keep stdout clean for data output
# force_terminal=True ensures Rich output is visible even in non-TTY environments
console = Console(stderr=True, force_terminal=True)
# Results go to stdout so they can be piped; status and errors stay on stderr.
out_console = Console()

# Create sub-apps for export and import
export_app = typer.Typer(help="Export objects from an AnnData store.")
import_app = typer.Typer(help="Import objects into an AnnData store.")
app.add_typer(export_app, name="export")
app.add_typer(import_app, name="import")


# ============================================================================
# INFO command
# ============================================================================
@app.command("view")
def view(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: Optional[str] = typer.Argument(
        None,
        help="Entry path to inspect (e.g., 'obsm/X_pca', 'X', 'uns')",
    ),
    tree: bool = typer.Option(
        False,
        "--tree",
        "-t",
        help="Show a tree of all entries",
    ),
    depth: int = typer.Option(
        None,
        "--depth",
        "-d",
        help="Maximum recursion depth for tree display (only with --tree)",
    ),
) -> None:
    """
    Show high-level information about an AnnData store.

    Use --tree to see a tree of all entries.
    Pass an entry path to inspect a specific entry in detail.

    Examples:
        adata view data.h5ad
        adata view --tree data.h5ad
        adata view data.h5ad obsm/X_pca
    """
    try:
        show_info(
            file,
            console,
            show_types=tree,
            depth=depth,
            entry_path=entry,
            out_console=out_console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@app.command("info", hidden=True)
def info(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: Optional[str] = typer.Argument(
        None,
        help="Entry path to inspect (e.g., 'obsm/X_pca', 'X', 'uns')",
    ),
    tree: bool = typer.Option(False, "--tree", "-t", help="Show a tree of all entries"),
    depth: int = typer.Option(
        None, "--depth", "-d", help="Maximum recursion depth for tree display"
    ),
) -> None:
    """Deprecated alias for 'view'. Removed in 1.0.0."""
    console.print(
        "[yellow]Warning:[/] 'info' is deprecated and will be removed in 1.0.0; "
        "use 'view' instead.",
    )
    view(file=file, entry=entry, tree=tree, depth=depth)


# ============================================================================
# LS command
# ============================================================================
@app.command("ls")
def ls(
    file: Path = typer.Argument(
        ...,
        help="Path to any .h5ad/.zarr/.h5 store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: Optional[str] = typer.Argument(
        None, help="Only list below this path (e.g. 'obsm', 'uns/spatial')"
    ),
    depth: Optional[int] = typer.Option(
        None, "--depth", "-d", help="Maximum depth to descend"
    ),
    long: bool = typer.Option(
        False, "--long", "-l", help="Show type, shape, dtype and encoding"
    ),
    plain: bool = typer.Option(
        False, "-1", "--plain", help="One bare path per line, for piping"
    ),
) -> None:
    """
    List the contents of any HDF5 or Zarr store.

    Makes no AnnData assumptions, so it also works on .loom and plain .h5
    files. Use -1 to pipe paths into other tools.

    Examples:
        adata ls data.h5ad
        adata ls data.h5ad --long
        adata ls data.h5ad obsm --depth 1
        adata ls data.h5ad -1 | grep spatial
    """
    try:
        list_store(
            file,
            out_console,
            entry_path=entry,
            depth=depth,
            long=long,
            plain=plain,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# SUBSET command
# ============================================================================
@app.command()
def subset(
    file: Path = typer.Argument(
        ...,
        help="Input .h5ad/.zarr",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad/.zarr path. Required unless --inplace.",
        dir_okay=True,
        file_okay=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
    obs: Optional[Path] = typer.Option(
        None,
        "--obs",
        help="File with obs names (one per line)",
        exists=True,
        readable=True,
    ),
    var: Optional[Path] = typer.Option(
        None,
        "--var",
        help="File with var names (one per line)",
        exists=True,
        readable=True,
    ),
    chunk_rows: int = typer.Option(
        1024,
        "--chunk",
        "-C",
        "--chunk-rows",
        "-r",
        help="Row chunk size for dense matrices",
    ),
) -> None:
    """Subset an AnnData store by obs and/or var names."""
    if obs is None and var is None:
        console.print(
            "[bold red]Error:[/] At least one of --obs or --var must be provided.",
        )
        raise typer.Exit(code=1)

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        subset_h5ad(
            file=file,
            output=output,
            obs_file=obs,
            var_file=var,
            chunk_rows=chunk_rows,
            console=console,
            inplace=inplace,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# EXPORT subcommands
# ============================================================================
@export_app.command("dataframe")
def export_dataframe(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Path of the dataframe to export (e.g. 'obs', 'var', 'raw/var')"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", writable=True, help="Output CSV file path"
    ),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Comma separated column names to include",
    ),
    chunk_rows: int = typer.Option(
        10_000,
        "--chunk",
        "-C",
        "--chunk-rows",
        "-r",
        help="Number of rows to read per chunk",
    ),
    head: Optional[int] = typer.Option(
        None, "--head", "-n", help="Output only the first n entries"
    ),
) -> None:
    """
    Export any dataframe-encoded group to CSV.

    Not limited to obs/var -- any path holding a dataframe works, including
    ones under raw/, obsm/ or uns/.

    Examples:
        adata export dataframe data.h5ad obs --output obs.csv
        adata export dataframe data.h5ad var --columns gene_id,mean
        adata export dataframe data.h5ad raw/var --head 100
    """

    col_list: Optional[List[str]] = None
    if columns:
        col_list = [col.strip() for col in columns.split(",") if col.strip()]

    try:
        export_table(
            file=file,
            axis=entry,
            columns=col_list,
            out=output,
            chunk_rows=chunk_rows,
            head=head,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("array")
def export_array(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to export (e.g., 'obsm/X_pca', 'varm/PCs', 'X')"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .npy file path (defaults to stdout)",
        writable=True,
    ),
    chunk_elements: int = typer.Option(
        100_000,
        "--chunk",
        "-C",
        help="Number of elements to read per chunk",
    ),
) -> None:
    """
    Export a dense array or matrix to NumPy .npy format.

    Writes to stdout when no --output is given. Writing to a file streams in
    chunks; stdout is not seekable, so that path holds the array in memory.

    Examples:
        adata export array data.h5ad obsm/X_pca -o pca.npy
        adata export array data.h5ad X -o matrix.npy
        adata export array data.h5ad obsm/X_umap > umap.npy
    """

    try:
        export_npy(
            file=file,
            obj=entry,
            out=output,
            chunk_elements=chunk_elements,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("sparse")
def export_sparse(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to export (e.g., 'X', 'layers/counts')"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        writable=True,
        help="Output .mtx file path (defaults to stdout)",
    ),
    head: Optional[int] = typer.Option(
        None, "--head", "-n", help="Output only the first n entries of mtx file"
    ),
    chunk_elements: int = typer.Option(
        1_000,
        "--chunk",
        "-C",
        help="Number of rows/columns (depends on compression format) to process per chunk",
    ),
    in_memory: bool = typer.Option(
        False,
        "--in-memory",
        "-m",
        help="Load the entire sparse matrix into memory before exporting (may be faster for small matrices)",
    ),
) -> None:
    """
    Export a sparse matrix (CSR/CSC) to Matrix Market (.mtx) format.

    Examples:
        h5ad export sparse data.h5ad X matrix.mtx
        h5ad export sparse data.h5ad layers/counts counts.mtx
        h5ad export sparse data.h5ad X --head 100
    """

    try:
        export_mtx(
            file=file,
            obj=entry,
            out=output,
            head=head,
            chunk_elements=chunk_elements,
            in_memory=in_memory,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("dict")
def export_dict(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to export (e.g., 'uns', 'uns/colors')"
    ),
    output_arg: Optional[Path] = typer.Argument(None, help="Output .json file path"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output .json file path"
    ),
    max_elements: int = typer.Option(
        100_000,
        "--max-elements",
        help="Maximum array elements for JSON export",
    ),
    include_attrs: bool = typer.Option(
        False, "--include-attrs", help="Include HDF5 attributes in JSON export"
    ),
) -> None:
    """
    Export a dict/group or scalar to JSON format.

    Examples:
        h5ad export dict data.h5ad uns metadata.json
        h5ad export dict data.h5ad uns/colors colors.json
    """

    try:
        out_path = output if output is not None else output_arg
        export_json(
            file=file,
            obj=entry,
            out=out_path,
            max_elements=max_elements,
            include_attrs=include_attrs,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("image")
def export_image(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(..., help="Entry path to export (2D or 3D array)"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output image file (.png, .jpg, .tiff)"
    ),
) -> None:
    """
    Export an image-like array to PNG/JPG/TIFF format.

    The array should be 2D (H,W) or 3D (H,W,C) with C in {1,3,4}.

    Examples:
        h5ad export image data.h5ad uns/spatial/image tissue.png
    """

    try:
        export_image_cmd(file=file, obj=entry, out=output, console=console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# IMPORT subcommands
# ============================================================================
def _get_target_file(file: Path, output: Optional[Path], inplace: bool) -> Path:
    """Determine target path and copy/convert if needed."""
    from adata.commands.import_data import _prepare_target_path

    return _prepare_target_path(file, output, inplace, console)


@import_app.command("dataframe")
def import_dataframe(
    file: Path = typer.Argument(
        ...,
        help="Path to the source .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to create/replace ('obs' or 'var')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input CSV file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad/.zarr path. Required unless --inplace.",
        dir_okay=True,
        file_okay=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
    index_column: Optional[str] = typer.Option(
        None,
        "--index-column",
        "-i",
        help="Column to use as index. Defaults to first column.",
    ),
) -> None:
    """
    Import a CSV file into obs or var.

    Examples:
        h5ad import dataframe data.h5ad obs cells.csv -o output.h5ad -i cell_id
        h5ad import dataframe data.h5ad var genes.csv --inplace -i gene_id
    """
    from adata.commands.import_data import _import_csv

    if entry not in ("obs", "var"):
        console.print(
            f"[bold red]Error:[/] Entry must be 'obs' or 'var', not '{entry}'.",
        )
        raise typer.Exit(code=1)

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_csv(target, entry, input_file, index_column, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@import_app.command("array")
def import_array(
    file: Path = typer.Argument(
        ...,
        help="Path to the source .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to create/replace (e.g., 'X', 'obsm/X_pca')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input .npy file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad/.zarr path. Required unless --inplace.",
        dir_okay=True,
        file_okay=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
) -> None:
    """
    Import a NumPy .npy file as a dense array.

    Dimensions are validated against existing obs/var.

    Examples:
        h5ad import array data.h5ad obsm/X_pca pca.npy -o output.h5ad
        h5ad import array data.h5ad X matrix.npy --inplace
    """
    from adata.commands.import_data import _import_npy

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_npy(target, entry, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@import_app.command("sparse")
def import_sparse(
    file: Path = typer.Argument(
        ...,
        help="Path to the source .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    obj: str = typer.Argument(
        ..., help="Object path to create/replace (e.g., 'X', 'layers/counts')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input .mtx file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad/.zarr path. Required unless --inplace.",
        dir_okay=True,
        file_okay=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
) -> None:
    """
    Import a Matrix Market (.mtx) file as a CSR sparse matrix.

    Dimensions are validated against existing obs/var.

    Examples:
        h5ad import sparse data.h5ad X matrix.mtx -o output.h5ad
        h5ad import sparse data.h5ad layers/counts counts.mtx --inplace
    """
    from adata.commands.import_data import _import_mtx

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_mtx(target, obj, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@import_app.command("dict")
def import_dict(
    file: Path = typer.Argument(
        ...,
        help="Path to the source .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    obj: str = typer.Argument(
        ..., help="Object path to create/replace (e.g., 'uns', 'uns/metadata')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input .json file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad/.zarr path. Required unless --inplace.",
        dir_okay=True,
        file_okay=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
) -> None:
    """
    Import a JSON file into uns or other dict-like groups.

    Examples:
        h5ad import dict data.h5ad uns/metadata config.json -o output.h5ad
        h5ad import dict data.h5ad uns settings.json --inplace
    """
    from adata.commands.import_data import _import_json

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_json(target, obj, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    app(standalone_mode=True)


def main_deprecated(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the deprecated `h5ad` command name."""
    console.print(
        "[yellow]Warning:[/] the 'h5ad' command is deprecated and will be removed "
        "in 1.0.0; use 'adata' instead.",
    )
    main(argv)

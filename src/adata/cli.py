"""CLI for AnnData stores (.h5ad / .zarr) with export and import subcommands."""

from pathlib import Path
from typing import Optional, Sequence, List

from rich.console import Console
import typer
from rich.markup import escape

from adata.conversion import FORMATS, Options
from adata.core.recast import LAYOUTS
from adata.commands import (
    MERGE_CHOICES,
    concat_stores,
    convert_store,
    recast_store,
    create_store,
    split_store,
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
    "(view, ls, create, subset, split, concat, convert, recast, export, import)."
)
# Use stderr for status/progress to keep stdout clean for data output
# force_terminal=True ensures Rich output is visible even in non-TTY environments
console = Console(stderr=True, force_terminal=True)
# Results go to stdout so they can be piped; status and errors stay on stderr.
out_console = Console()


def _version_callback(value: bool) -> None:
    if value:
        from adata import __version__

        out_console.print(__version__, highlight=False)
        raise typer.Exit()


# Create sub-apps for export and import
export_app = typer.Typer(help="Export objects from an AnnData store.")
import_app = typer.Typer(help="Import objects into an AnnData store.")
app.add_typer(export_app, name="export")
app.add_typer(import_app, name="import")


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the installed version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Streaming CLI for huge AnnData .h5ad and .zarr stores."""


# ============================================================================
# VIEW command
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
# CREATE command
# ============================================================================
@app.command("create")
def create(
    output: Path = typer.Argument(
        ...,
        help="Path of the store to create (.h5ad or .zarr)",
        dir_okay=True,
        file_okay=True,
    ),
    n_obs: Optional[int] = typer.Option(
        None, "--n-obs", help="Number of observations (cells)"
    ),
    n_var: Optional[int] = typer.Option(
        None, "--n-var", help="Number of variables (genes)"
    ),
    obs_names: Optional[Path] = typer.Option(
        None,
        "--obs-names",
        help="File of obs names, one per line (sets --n-obs)",
        exists=True,
        readable=True,
    ),
    var_names: Optional[Path] = typer.Option(
        None,
        "--var-names",
        help="File of var names, one per line (sets --n-var)",
        exists=True,
        readable=True,
    ),
    zarr_format: Optional[int] = typer.Option(
        None, "--zarr-format", help="Zarr spec version to write (2 or 3)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite an existing store"
    ),
) -> None:
    """
    Create a new, empty AnnData store.

    The result is a valid AnnData object straight away; fill it in with
    `adata import`. Give each axis either a size or a file of names.

    Examples:
        adata create out.h5ad --n-obs 5000 --n-var 2000
        adata create out.zarr --obs-names cells.txt --var-names genes.txt
        adata import sparse out.h5ad X counts.mtx --inplace
    """
    if zarr_format is not None and zarr_format not in (2, 3):
        console.print("[bold red]Error:[/] --zarr-format must be 2 or 3.")
        raise typer.Exit(code=1)

    try:
        create_store(
            output,
            console,
            n_obs=n_obs,
            n_var=n_var,
            obs_names=obs_names,
            var_names=var_names,
            zarr_format=zarr_format,
            force=force,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1)


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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
    obs_query: Optional[str] = typer.Option(
        None,
        "--obs-query",
        "-q",
        help="Keep obs matching an expression, e.g. \"cluster == Cortex_2\"",
    ),
    var_query: Optional[str] = typer.Option(
        None,
        "--var-query",
        help="Keep var matching an expression, e.g. \"highly_variable == True\"",
    ),
    chunk_rows: int = typer.Option(
        1024,
        "--chunk",
        "-C",
        "--chunk-rows",
        "-r",
        help="Row chunk size for dense matrices",
    ),
    zarr_format: Optional[int] = typer.Option(
        None,
        "--zarr-format",
        help="Zarr spec version to write (defaults to the source store's)",
    ),
) -> None:
    """
    Subset an AnnData store by obs and/or var.

    Select either by name list (--obs/--var) or by expression
    (--obs-query/--var-query). Expressions support ==, !=, <, <=, >, >=, in,
    not in, and, or, not, and parentheses.

    Examples:
        adata subset data.h5ad -o out.h5ad --obs barcodes.txt
        adata subset data.h5ad -o out.h5ad --obs-query "cluster == Cortex_2"
        adata subset data.h5ad -o out.h5ad -q "n_counts > 1000 and cluster in A,B"
        adata subset data.h5ad -o out.h5ad --var-query "highly_variable == True"
    """
    if zarr_format is not None and zarr_format not in (2, 3):
        console.print("[bold red]Error:[/] --zarr-format must be 2 or 3.")
        raise typer.Exit(code=1)

    if obs is None and var is None and obs_query is None and var_query is None:
        console.print(
            "[bold red]Error:[/] Provide at least one of --obs, --var, "
            "--obs-query or --var-query.",
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
            obs_query=obs_query,
            var_query=var_query,
            zarr_format=zarr_format,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1)


# ============================================================================
# CONCAT command
# ============================================================================
@app.command("concat")
def concat(
    files: List[Path] = typer.Argument(
        ...,
        help="Two or more .h5ad/.zarr stores to concatenate",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output .h5ad/.zarr path", dir_okay=True
    ),
    join: str = typer.Option(
        "inner",
        "--join",
        "-j",
        help="Align var by intersection ('inner') or union ('outer')",
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Add an obs column recording which input each cell came from"
    ),
    keys: Optional[str] = typer.Option(
        None,
        "--keys",
        help="Comma separated names for the inputs (defaults to their filenames)",
    ),
    index_unique: Optional[str] = typer.Option(
        None,
        "--index-unique",
        help="Delimiter used to suffix obs names with their key, e.g. '-'",
    ),
    merge: Optional[str] = typer.Option(
        None,
        "--merge",
        help="How to reconcile var columns: drop, same, unique, first, only "
        "(default: drop)",
    ),
    uns_merge: Optional[str] = typer.Option(
        None,
        "--uns-merge",
        help="How to reconcile uns: drop, same, unique, first, only "
        "(default: drop)",
    ),
    fill_value: float = typer.Option(
        0.0, "--fill-value", help="Value for dense cells introduced by an outer join"
    ),
    chunk_rows: int = typer.Option(
        1024, "--chunk", "-C", help="Row chunk size while streaming"
    ),
    zarr_format: Optional[int] = typer.Option(
        None,
        "--zarr-format",
        help="Zarr spec version to write (defaults to the inputs' when they agree)",
    ),
) -> None:
    """
    Concatenate stores along the obs axis.

    Streams each input in turn, so memory use is set by --chunk rather than by
    the size of the inputs. obsp, varp and raw are not carried over.

    Examples:
        adata concat a.h5ad b.h5ad -o merged.h5ad
        adata concat *.h5ad -o merged.h5ad --join outer --label sample
        adata concat a.h5ad b.h5ad -o m.h5ad --keys a,b --index-unique - --uns-merge same
    """
    if zarr_format is not None and zarr_format not in (2, 3):
        console.print("[bold red]Error:[/] --zarr-format must be 2 or 3.")
        raise typer.Exit(code=1)

    for name, value in (("--merge", merge), ("--uns-merge", uns_merge)):
        if value is not None and value not in MERGE_CHOICES:
            console.print(
                f"[bold red]Error:[/] {name} must be one of: "
                f"{', '.join(MERGE_CHOICES)}"
            )
            raise typer.Exit(code=1)

    # "drop" is the default, and naming it explicitly has to be allowed: a
    # config that spells out the default should not be rejected.
    merge = None if merge == "drop" else merge
    uns_merge = None if uns_merge == "drop" else uns_merge

    try:
        concat_stores(
            files,
            output,
            console,
            join=join,
            label=label,
            keys=[k.strip() for k in keys.split(",")] if keys else None,
            index_unique=index_unique,
            merge=merge,
            uns_merge=uns_merge,
            fill_value=fill_value,
            chunk_rows=chunk_rows,
            zarr_format=zarr_format,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1)


# ============================================================================
# CONVERT command
# ============================================================================
#: The flags `recast` has and format conversion does not. Until 1.0.0, any of
#: them on `convert` means the old meaning of the command, and it forwards.
_RECAST_FLAGS = {
    "inplace": "--inplace",
    "convert_all": "--all",
    "dtype": "--dtype",
    "indices_dtype": "--indices-dtype",
    "layout": "--layout",
}

#: Flags that only shape a recast; given without one of the above they have
#: nothing to act on.
_RECAST_ONLY_FLAGS = {"in_memory": "--in-memory", "chunk_rows": "--chunk"}

#: Format-conversion flags, by parameter name, as `Options.given` spells them.
_CONVERT_FLAGS = {
    "zarr_format": "--zarr-format",
    "compression": "--compression",
    "force": "--force",
    "overwrite": "--overwrite",
    "genome": "--genome",
    "var_names": "--var-names",
    "make_unique": "--make-unique",
    "feature_type": "--feature-type",
    "split_feature_types": "--split-feature-types",
    "layer": "--layer",
    "gene_ids_column": "--gene-ids-column",
    "gene_names_column": "--gene-names-column",
    "feature_types_column": "--feature-types-column",
    "raw": "--raw",
    "min_cell_probability": "--min-cell-probability",
    "sample": "--sample",
    "all_samples": "--all-samples",
    "unfiltered": "--unfiltered",
    "library_id": "--library-id",
    "no_images": "--no-images",
    "bin_size": "--bin-size",
    "dense": "--dense",
    "obs_names_attr": "--obs-names-attr",
    "var_names_attr": "--var-names-attr",
    "delimiter": "--delimiter",
    "transpose": "--transpose",
    "first_column_names": "--first-column-names",
    "key": "--key",
    "obs_names_key": "--obs-names-key",
    "var_names_key": "--var-names-key",
    "scope": "--scope",
    "clusterings": "--clusterings",
    "default_embedding": "--default-embedding",
    "mod": "--mod",
}

#: Flags that only make sense for format conversion; with recast's flags,
#: they mean two commands' worth of request at once.
_FORMAT_ONLY_FLAGS = {
    "from_format": "--from",
    "to_format": "--to",
    **{
        name: flag
        for name, flag in _CONVERT_FLAGS.items()
        if flag not in ("--force", "--zarr-format")
    },
}


def _given(ctx: typer.Context, flags: dict) -> List[str]:
    """The flags among `flags` that were actually typed, not defaulted."""
    from click.core import ParameterSource

    return [
        flag
        for name, flag in flags.items()
        if ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT)
    ]


@app.command("convert", context_settings={"allow_extra_args": True})
def convert(
    ctx: typer.Context,
    file: Path = typer.Argument(
        ...,
        help="Input file, store, or 10x/STARsolo matrix folder",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path; its extension picks the format (.h5ad, .zarr, .h5, .loom, .h5mu), "
        "or a folder with --to 10x-mtx",
        dir_okay=True,
        file_okay=True,
    ),
    from_format: Optional[str] = typer.Option(
        None,
        "--from",
        help=f"Input format, if it cannot be detected: {', '.join(FORMATS)}",
    ),
    to_format: Optional[str] = typer.Option(
        None,
        "--to",
        help=f"Output format, if the extension does not say: {', '.join(FORMATS)}",
    ),
    zarr_format: Optional[int] = typer.Option(
        None,
        "--zarr-format",
        help="Zarr spec version to write (defaults to the source store's, else 3)",
    ),
    compression: Optional[str] = typer.Option(
        None,
        "--compression",
        help="none, gzip or lzf for h5ad; none, gzip, zstd or blosc for zarr "
        "(defaults to the source's); gzip or none for a 10x-mtx folder",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Convert despite a lossy or inflating result",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing output",
    ),
    genome: Optional[str] = typer.Option(
        None,
        "--genome",
        help="10x input: the genome to keep (picks the group or subfolder of a "
        "Cell Ranger 2 matrix)",
    ),
    var_names: str = typer.Option(
        "gene_ids",
        "--var-names",
        help="10x input: name features by gene_ids (default; unique) or "
        "gene_symbols (scanpy's default)",
    ),
    make_unique: bool = typer.Option(
        False,
        "--make-unique",
        help="10x input: suffix repeated var names, as var_names_make_unique does",
    ),
    feature_type: Optional[str] = typer.Option(
        None,
        "--feature-type",
        help="10x input: keep only this feature type, e.g. 'Gene Expression'",
    ),
    split_feature_types: bool = typer.Option(
        False,
        "--split-feature-types",
        help="10x input: write one output per feature type, OUT.<type>.h5ad",
    ),
    layer: Optional[str] = typer.Option(
        None,
        "--layer",
        help="10x or loom output: the matrix to write: X (default), raw/X, or "
        "a layer; loom input: the loom layer that becomes X (default spliced, "
        "else the main matrix)",
    ),
    gene_ids_column: Optional[str] = typer.Option(
        None,
        "--gene-ids-column",
        help="10x output: var column holding gene ids (default gene_ids)",
    ),
    gene_names_column: Optional[str] = typer.Option(
        None,
        "--gene-names-column",
        help="10x output: var column holding gene names (default: var_names)",
    ),
    feature_types_column: Optional[str] = typer.Option(
        None,
        "--feature-types-column",
        help="10x output: var column holding feature types (default feature_types)",
    ),
    raw: Optional[Path] = typer.Option(
        None,
        "--raw",
        help="CellBender input: the raw Cell Ranger .h5 it ran on, added as "
        "layers['cellranger']",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    min_cell_probability: Optional[float] = typer.Option(
        None,
        "--min-cell-probability",
        help="CellBender input: keep droplets with cell_probability at least "
        "this (CellBender's own cutoff is 0.5)",
    ),
    sample: Optional[str] = typer.Option(
        None,
        "--sample",
        help="Parse input: the sample folder to read (default: all-sample)",
    ),
    all_samples: bool = typer.Option(
        False,
        "--all-samples",
        help="Parse input: write one output per sample, OUT.<sample>.h5ad",
    ),
    unfiltered: bool = typer.Option(
        False,
        "--unfiltered",
        help="Parse or Visium input: read every barcode (DGE_unfiltered, "
        "raw_feature_bc_matrix.h5), not only cells",
    ),
    library_id: Optional[str] = typer.Option(
        None,
        "--library-id",
        help="Visium input: the key under uns['spatial'] (default: the "
        "matrix's library id, as scanpy and squidpy use)",
    ),
    no_images: bool = typer.Option(
        False,
        "--no-images",
        help="Visium input: leave the tissue images out of uns['spatial'] "
        "(the scale factors stay)",
    ),
    bin_size: Optional[int] = typer.Option(
        None,
        "--bin-size",
        help="Visium HD input: the bin size to read, in µm (default 8; "
        "Space Ranger writes 2, 8 and 16)",
    ),
    dense: bool = typer.Option(
        False,
        "--dense",
        help="loom, text or HDF5 input: keep X (and the layers) dense (default: sparse CSR)",
    ),
    obs_names_attr: Optional[str] = typer.Option(
        None,
        "--obs-names-attr",
        help="loom input: the column attribute naming cells (default CellID)",
    ),
    var_names_attr: Optional[str] = typer.Option(
        None,
        "--var-names-attr",
        help="loom input: the row attribute naming genes (default Accession, "
        "or Gene with --var-names gene_symbols)",
    ),
    delimiter: Optional[str] = typer.Option(
        None,
        "--delimiter",
        help="text input: the delimiter: a character, tab or whitespace "
        "(default from the extension: , for .csv, tab for .tsv, whitespace for .txt)",
    ),
    transpose: bool = typer.Option(
        False,
        "--transpose",
        help="text or HDF5 input: the matrix holds genes as rows and cells as columns",
    ),
    first_column_names: Optional[bool] = typer.Option(
        None,
        "--first-column-names/--no-first-column-names",
        help="text input: whether the first column names the rows (default: "
        "detected, as anndata detects it)",
    ),
    key: Optional[str] = typer.Option(
        None,
        "--key",
        help="HDF5 input: the 2-D dataset that becomes X (without it, the "
        "candidates are listed)",
    ),
    obs_names_key: Optional[str] = typer.Option(
        None,
        "--obs-names-key",
        help="HDF5 input: the dataset naming the cells (default row_names, "
        "col_names with --transpose)",
    ),
    scope: bool = typer.Option(
        False,
        "--scope",
        help="loom output: write a SCope loom: regulons, embeddings, clusterings "
        "and MetaData, as the SCope viewer reads them",
    ),
    clusterings: Optional[str] = typer.Option(
        None,
        "--clusterings",
        help="SCope output: categorical obs columns written as clusterings, "
        "comma separated (besides those a SCope import recorded)",
    ),
    default_embedding: Optional[str] = typer.Option(
        None,
        "--default-embedding",
        help="SCope output: the obsm key SCope shows first (default: the "
        "recorded default, else X_umap, else X_tsne, else the first)",
    ),
    mod: Optional[str] = typer.Option(
        None,
        "--mod",
        help="MuData: the modality to read (default: each, to OUT.<mod>.<ext>), "
        "or the name to write the whole AnnData under (default: split by "
        "feature_types)",
    ),
    var_names_key: Optional[str] = typer.Option(
        None,
        "--var-names-key",
        help="HDF5 input: the dataset naming the genes (default col_names, "
        "row_names with --transpose)",
    ),
    # Deprecated: recast's flags, so `convert` keeps working as it did until
    # 1.0.0. Hidden from --help; the old matrix paths arrive in ctx.args.
    inplace: bool = typer.Option(False, "--inplace", hidden=True),
    convert_all: bool = typer.Option(False, "--all", hidden=True),
    dtype: Optional[str] = typer.Option(None, "--dtype", hidden=True),
    indices_dtype: Optional[str] = typer.Option(None, "--indices-dtype", hidden=True),
    layout: Optional[str] = typer.Option(None, "--layout", hidden=True),
    in_memory: bool = typer.Option(False, "--in-memory", hidden=True),
    chunk_rows: int = typer.Option(1024, "--chunk", "-C", hidden=True),
) -> None:
    """
    Convert a store to another format.

    The output's extension picks the format: .h5ad, .zarr, .loom, or .h5 for
    a 10x Genomics matrix (Cell Ranger's *_feature_bc_matrix.h5); --to
    10x-mtx writes a Matrix Market folder. Between h5ad and zarr every element is
    copied, streamed, keeping the source's chunking and compression where the
    target can express them. CellBender output keeps its latent variables, as
    CellBender's own loader names them. A 10x matrix is read as scanpy's
    read_10x_h5 or read_10x_mtx (gex_only=False) reads it, keeping integer
    counts, and written as Cell Ranger writes it. A Space Ranger outs/
    folder keeps its spot positions, images and scale factors where squidpy
    puts them; a Visium HD one is read at the bin size --bin-size names
    (needs pyadata-cli[spatial]); and a Xenium bundle its cell table,
    centroids and Xenium's own clusters, UMAP and PCA. MTX folders from Cell
    Ranger and STARsolo, velocyto's included, and Parse split-pipe output
    are read in two streaming passes, as are text tables (.csv, .tsv, .txt,
    gzipped too), read as anndata's read_csv and read_text read them; a plain
    HDF5 dataset is read as read_hdf reads it (--key names it); a pycisTopic
    pickle without pycisTopic (needs pyadata-cli[pickle]). A loom
    is read as anndata.read_loom reads it; a SCope loom (pySCENIC, SCENIC+,
    pycistopic) also gives its regulons as dataframes in obsm and varm, its
    embeddings and clusterings named from its MetaData, and the MetaData in
    uns["scope"]; --scope writes them back as SCope reads them. Features are named by gene id, unique but in probe-level
    (Flex) matrices; --var-names gene_symbols names them as scanpy does.

    Nothing is written until every check has passed, and the output is moved
    into place only once complete: a failed conversion leaves nothing
    behind. An existing output is refused unless --overwrite is given.

    To change a matrix's dtype, layout or density, use `recast`. Until 1.0.0,
    `convert` given recast's flags still forwards to it, with a warning.

    Examples:
        adata convert data.h5ad -o data.zarr
        adata convert data.zarr -o data.h5ad --compression gzip
        adata convert data.zarr -o v3.zarr --zarr-format 3
        adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
        adata convert raw_feature_bc_matrix.h5 -o rna.h5ad --feature-type "Gene Expression"
        adata convert counts.h5ad -o for_cellbender.h5 --layer counts
        adata convert Solo.out/Gene/filtered -o counts.h5ad
        adata convert Solo.out/Velocyto/raw -o velocyto.h5ad
        adata convert spaceranger_run/outs -o section.h5ad
        adata convert visium_hd_run/outs -o bins.h5ad --bin-size 8
        adata convert xenium_bundle -o cells.h5ad
        adata convert counts.h5ad -o counts_mtx --to 10x-mtx --layer counts
        adata convert cellbender_output.h5 -o cb.h5ad --raw raw_feature_bc_matrix.h5
        adata convert cellbender_output.h5 -o cells.h5ad --min-cell-probability 0.5
        adata convert splitpipe_out -o parse.h5ad
        adata convert splitpipe_out -o parse.h5ad --all-samples --unfiltered
        adata convert velocyto.loom -o velocyto.h5ad
        adata convert pyscenic_output.loom -o scenic.h5ad
        adata convert counts.h5ad -o counts.loom
        adata convert scenic.h5ad -o scenic.loom --scope --clusterings leiden
        adata convert GSE123_counts.txt.gz -o counts.h5ad --transpose
        adata convert matrix.hdf5 -o matrix.h5ad --key data/counts
        adata convert cistopic_obj.pkl -o atac.h5ad
        adata convert multiome.h5mu -o multiome.h5ad
        adata convert scenicplus.h5mu -o rna.h5ad --mod scRNA_counts
        adata convert multiome.h5ad -o multiome.h5mu
    """
    entries = list(ctx.args)
    recast_flags = _given(ctx, _RECAST_FLAGS)

    try:
        if recast_flags:
            mixed = _given(ctx, _FORMAT_ONLY_FLAGS)
            if mixed:
                raise ValueError(
                    f"{', '.join(mixed)} cannot be combined with "
                    f"{', '.join(recast_flags)}: the first change a store's "
                    "format, the second its matrices. Run `adata recast` for "
                    "the matrices."
                )
            console.print(
                "[yellow]Warning:[/] 'convert' with "
                f"{', '.join(recast_flags)} is deprecated and will be removed "
                "in 1.0.0; use 'recast' instead.",
            )
            if not inplace and output is None:
                raise ValueError("Output file is required. Use --output/-o or --inplace.")
            if layout is not None and layout not in LAYOUTS:
                raise ValueError(f"--layout must be one of: {', '.join(LAYOUTS)}.")
            if zarr_format is not None and zarr_format not in (2, 3):
                raise ValueError("--zarr-format must be 2 or 3.")
            recast_store(
                file=file,
                entries=entries or None,
                output=output,
                console=console,
                dtype=dtype,
                indices_dtype=indices_dtype,
                layout=layout,
                convert_all=convert_all,
                inplace=inplace,
                chunk_rows=chunk_rows,
                in_memory=in_memory,
                force=force,
                zarr_format=zarr_format,
            )
            return

        if entries:
            hint = (
                f" To write it, pass -o {entries[0]}."
                if output is None and len(entries) == 1
                else ""
            )
            raise ValueError(
                f"Unexpected argument{'s' if len(entries) > 1 else ''}: "
                f"{' '.join(entries)}. convert takes one input.{hint} To change "
                "a matrix's dtype or layout, use `adata recast`."
            )
        stray = _given(ctx, _RECAST_ONLY_FLAGS)
        if stray:
            raise ValueError(
                f"{', '.join(stray)} {'applies' if len(stray) == 1 else 'apply'} "
                "to `adata recast`, not to format conversion."
            )
        if output is None:
            raise ValueError("Output path is required: pass -o/--output.")

        convert_store(
            file,
            output,
            console,
            src_format=from_format,
            dst_format=to_format,
            options=Options(
                given=frozenset(_given(ctx, _CONVERT_FLAGS)),
                zarr_format=zarr_format,
                compression=compression,
                force=force,
                overwrite=overwrite,
                genome=genome,
                var_names=var_names,
                make_unique=make_unique,
                feature_type=feature_type,
                split_feature_types=split_feature_types,
                layer=layer,
                gene_ids_column=gene_ids_column,
                gene_names_column=gene_names_column,
                feature_types_column=feature_types_column,
                raw=raw,
                min_cell_probability=min_cell_probability,
                sample=sample,
                all_samples=all_samples,
                unfiltered=unfiltered,
                library_id=library_id,
                no_images=no_images,
                bin_size=bin_size,
                dense=dense,
                obs_names_attr=obs_names_attr,
                var_names_attr=var_names_attr,
                delimiter=delimiter,
                transpose=transpose,
                first_column_names=first_column_names,
                key=key,
                obs_names_key=obs_names_key,
                var_names_key=var_names_key,
                scope=scope,
                clusterings=clusterings,
                default_embedding=default_embedding,
                mod=mod,
            ),
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1)


# ============================================================================
# RECAST command
# ============================================================================
@app.command("recast")
def recast(
    file: Path = typer.Argument(
        ...,
        help="Input .h5ad/.zarr",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entries: Optional[List[str]] = typer.Argument(
        None,
        help="Matrix paths to recast, e.g. 'X', 'layers/counts', 'raw/X'",
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
    convert_all: bool = typer.Option(
        False,
        "--all",
        help="Recast X, every layer, and raw/X",
    ),
    dtype: Optional[str] = typer.Option(
        None,
        "--dtype",
        help="New dtype for the values, e.g. float32",
    ),
    indices_dtype: Optional[str] = typer.Option(
        None,
        "--indices-dtype",
        help="New dtype for sparse indices: int32 or int64",
    ),
    layout: Optional[str] = typer.Option(
        None,
        "--layout",
        help="Target layout: csr, csc, dense or sparse",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Recast despite a lossy cast or a large size increase",
    ),
    in_memory: bool = typer.Option(
        False,
        "--in-memory",
        help="Transpose in memory instead of streaming (faster if it fits)",
    ),
    chunk_rows: int = typer.Option(
        1024,
        "--chunk",
        "-C",
        help="Row chunk size for dense matrices",
    ),
    zarr_format: Optional[int] = typer.Option(
        None,
        "--zarr-format",
        help="Zarr spec version to write (defaults to the source store's)",
    ),
) -> None:
    """
    Change a matrix's dtype, layout or density.

    Counts held as float64 cost twice the disk and twice the read for no
    information; a tool that wants CSC cannot use a CSR store; and concat
    refuses inputs whose encodings disagree. All three are this command.

    A cast that would not round-trip is refused before anything is written,
    as is a densification that would inflate the store; --force overrides
    both. Transposing streams by default, so it works on matrices too large
    to load.

    This was `convert` until 0.7.0; `convert` now changes a store's format.

    Examples:
        adata recast data.h5ad X -o out.h5ad --dtype float32
        adata recast data.h5ad X -o out.h5ad --layout csc
        adata recast data.h5ad X --inplace --dtype float32 --indices-dtype int32
        adata recast data.h5ad --all -o out.h5ad --dtype float32
    """
    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    if layout is not None and layout not in LAYOUTS:
        console.print(
            f"[bold red]Error:[/] --layout must be one of: {', '.join(LAYOUTS)}."
        )
        raise typer.Exit(code=1)

    if zarr_format is not None and zarr_format not in (2, 3):
        console.print("[bold red]Error:[/] --zarr-format must be 2 or 3.")
        raise typer.Exit(code=1)

    try:
        recast_store(
            file=file,
            entries=list(entries) if entries else None,
            output=output,
            console=console,
            dtype=dtype,
            indices_dtype=indices_dtype,
            layout=layout,
            convert_all=convert_all,
            inplace=inplace,
            chunk_rows=chunk_rows,
            in_memory=in_memory,
            force=force,
            zarr_format=zarr_format,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1)


# ============================================================================
# SPLIT command
# ============================================================================
@app.command("split")
def split(
    file: Path = typer.Argument(
        ...,
        help="Input .h5ad/.zarr",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    by: str = typer.Option(
        ..., "--by", "-b", help="Column to split on (e.g. 'sample', 'cell_type')"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to write the split stores into"
    ),
    axis: str = typer.Option(
        "obs", "--axis", help="Axis the column belongs to ('obs' or 'var')"
    ),
    suffix: Optional[str] = typer.Option(
        None,
        "--suffix",
        help="Output extension (defaults to the source store's format)",
    ),
    min_size: int = typer.Option(
        1, "--min-size", help="Skip groups with fewer rows than this"
    ),
    manifest: bool = typer.Option(
        True, "--manifest/--no-manifest", help="Write a CSV manifest of the outputs"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be written without writing it"
    ),
    chunk_rows: int = typer.Option(
        1024, "--chunk", "-C", help="Row chunk size for dense matrices"
    ),
    zarr_format: Optional[int] = typer.Option(
        None,
        "--zarr-format",
        help="Zarr spec version to write (defaults to the source store's)",
    ),
) -> None:
    """
    Split a store into one file per distinct value of a column.

    Streams the source, so it works on stores far larger than memory.

    Examples:
        adata split data.h5ad --by sample -o per_sample/
        adata split data.h5ad --by cell_type -o clusters/ --dry-run
        adata split data.h5ad --by cluster -o out/ --min-size 50
    """
    if axis not in ("obs", "var"):
        console.print("[bold red]Error:[/] --axis must be 'obs' or 'var'.")
        raise typer.Exit(code=1)

    if zarr_format is not None and zarr_format not in (2, 3):
        console.print("[bold red]Error:[/] --zarr-format must be 2 or 3.")
        raise typer.Exit(code=1)

    try:
        split_store(
            file=file,
            column=by,
            output_dir=output_dir,
            console=console,
            axis=axis,
            suffix=suffix,
            dry_run=dry_run,
            manifest=manifest,
            min_size=min_size,
            chunk_rows=chunk_rows,
            zarr_format=zarr_format,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        help="Output .mtx file path; .mtx.gz gzips it (defaults to stdout)",
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
        adata export sparse data.h5ad X -o matrix.mtx
        adata export sparse data.h5ad layers/counts -o counts.mtx
        adata export sparse data.h5ad X --head 100
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        adata export dict data.h5ad uns metadata.json
        adata export dict data.h5ad uns/colors colors.json
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        None,
        "--output",
        "-o",
        help="Output image file (.png, .jpg, .tiff); defaults to PNG on stdout",
    ),
) -> None:
    """
    Export an image-like array to PNG/JPG/TIFF format.

    The array should be 2D (H,W) or 3D (H,W,C) with C in {1,3,4}. The
    format follows the output's extension; without --output the image goes
    to stdout as PNG.

    Examples:
        adata export image data.h5ad uns/spatial/image -o tissue.png
        adata export image data.h5ad uns/spatial/image > tissue.png
    """

    try:
        export_image_cmd(file=file, obj=entry, out=output, console=console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        ..., help="Dataframe path to create/replace (e.g. 'obs', 'var', 'raw/var')"
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
    categorical: Optional[str] = typer.Option(
        None,
        "--categorical",
        help="Comma separated columns to force to categorical",
    ),
    auto_categorical: bool = typer.Option(
        True,
        "--auto-categorical/--no-auto-categorical",
        help="Infer categoricals from low-cardinality string columns",
    ),
) -> None:
    """
    Import a CSV file as a dataframe.

    String columns with few distinct values become categoricals; pass
    --no-auto-categorical to keep them as plain strings, or --categorical to
    force specific ones.

    Examples:
        adata import dataframe data.h5ad obs cells.csv -o out.h5ad -i cell_id
        adata import dataframe data.h5ad var genes.csv --inplace -i gene_id
        adata import dataframe data.h5ad obs cells.csv --inplace --categorical batch
    """
    from adata.commands.import_data import _import_csv

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        cat_list = (
            [c.strip() for c in categorical.split(",") if c.strip()]
            if categorical
            else None
        )
        _import_csv(
            target,
            entry,
            input_file,
            index_column,
            console,
            categorical=cat_list,
            auto_categorical=auto_categorical,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        adata import array data.h5ad obsm/X_pca pca.npy -o output.h5ad
        adata import array data.h5ad X matrix.npy --inplace
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        ..., help="Input .mtx or .mtx.gz file", exists=True, readable=True
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
        adata import sparse data.h5ad X matrix.mtx -o output.h5ad
        adata import sparse data.h5ad layers/counts counts.mtx --inplace
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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
        adata import dict data.h5ad uns/metadata config.json -o output.h5ad
        adata import dict data.h5ad uns settings.json --inplace
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
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
        raise typer.Exit(code=1)


@import_app.command("image")
def import_image_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to the source .h5ad/.zarr store",
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=True,
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to create/replace (e.g. 'uns/spatial/hires')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input image file (.png, .jpg, .tiff)", exists=True, readable=True
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
        False, "--inplace", help="Modify source file directly."
    ),
) -> None:
    """
    Import an image file as a dense array.

    Examples:
        adata import image data.h5ad uns/spatial/hires tissue.png --inplace
    """
    from adata.commands.import_data import _import_image

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_image(target, entry, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {escape(str(e))}")
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

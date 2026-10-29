---
title: Troubleshooting
nav_order: 5
---

# Troubleshooting
{: .no_toc }

Find the message you got, then the fix. Errors start `Error:` and end the
command with exit code 1; a usage error (an unknown option, a missing
argument, an input that does not exist) is exit code 2. Messages are shown
here with real paths replaced by short ones.

1. TOC
{:toc}

## Before anything is written

Every command resolves its selection, checks its casts and plans its outputs
before it creates anything, so after one of these errors there is nothing to
clean up, and an existing output is untouched.

### `Cannot tell what format 'x' is. Pass --from, one of: ...`

`convert` recognises its input by content, and this file matched no format it
reads. Check that the path is the file or folder you meant: a 10x MTX input is
the folder holding `matrix.mtx`, not the file, and a Parse input is the run,
sample or `DGE_*` folder. If it is one of the formats listed, `--from` names it.

Two variants say more:

| Message | Fix |
|---|---|
| `'x.h5' is HDF5, but no format this reads: pass --from hdf5 with --key to read one of its datasets as X.` | A plain HDF5 file, not a 10x matrix: see [HDF5 matrices](reference/formats/hdf5.md) |
| `'matrix.mtx.gz' is a single Matrix Market file, which convert does not read on its own` | Give the folder holding it with its barcodes and features, or use `adata import sparse` to put the matrix into a store |

### `Cannot tell the output format from 'x'. Use a known extension or pass --to`

The output's extension picks its format. Use `.h5ad`, `.zarr`, `.h5`, `.loom`
or `.h5mu`, or `--to 10x-mtx` for an MTX folder, which has no extension.

### `--to loom contradicts the output's extension, which means h5ad`

`--to` and the extension disagree, and the extension is what picks the
backend. Rename the output, or drop `--to`.

### `cellbender can be read but not written.`

Some formats are import-only: CellBender, Parse, text tables, HDF5 matrices
and pycisTopic pickles. [Formats](reference/formats/index.md) has the full
read and write table.

### `Cannot convert loom to 10x-h5 directly`

AnnData is the hub: convert to `.h5ad` first, then from that to the other
format.

### `--layer does not apply when converting 10x-h5 to h5ad.`

A flag that the conversion would ignore is refused rather than ignored. Each
format's own flags are on its page under [Formats](reference/formats/index.md);
here, `--layer` picks the matrix to *export*, so it applies from `.h5ad` to
`.h5`, not the other way.

### `X: float64 -> int32 is lossy: 2 of 2 values (100.00%) do not round-trip, largest change 0.7. Pass --force to convert anyway.`

`recast` cast every value and cast it back, and some changed: fractions cast
to an integer, or values beyond the new type's range. Pick a wider dtype, or
pass `--force` if the loss is what you want. See
[Change dtype, layout or density](guides/recast.md).

### `X would grow from 800 MB to 6,400 MB (8.0x). That is above the 4x limit; pass --force if it is what you want.`

Densifying a sparse matrix would more than quadruple it. `--force` writes it
anyway; check the size first.

### `int32 cannot hold this matrix's offsets: indptr must reach ...`

The matrix has more nonzeros than int32 can count. Keep `--indices-dtype
int64`.

### `Output path is the input: data.h5ad. Use --inplace to replace it`

Reading a store and writing it at once would destroy it: a Zarr store opened
for writing is emptied before a single element has been read. Use `--inplace`,
which writes beside the source and swaps the result in at the end, or give
another output path. `split` says `Choose another --output-dir`.

### `'out.zarr' already exists. Pass --overwrite to replace it.`

`convert` does not replace an existing output unless asked; `create` takes
`--force` for the same. The old output is replaced only once the new one is
complete.

### `var has no feature ids ... which a 10x matrix records for every feature. Pass --force to write placeholders`

A 10x export needs an id and a type per feature, read from `var["gene_ids"]`
and `var["feature_types"]` or the columns `--gene-ids-column` and
`--feature-types-column` name. Point the flags at the right columns, or pass
`--force` to write the var names as ids and `Gene Expression` as the type.

### `Cannot concatenate 'X': inputs use csc_matrix, csr_matrix.`

Every input's `X` (and each layer) must have the same encoding. The message
gives the `recast` command that converts an input to match; run it on the odd
ones out.

### `Column(s) clster not found in 'obs'. Available: ...`

A query named a column the axis does not have; `split --by` says the same as
`Column 'clster' not found in 'obs'`. The message lists the columns there are;
names are case-sensitive.

### `The obs query matched no rows.`

Nothing to write, so nothing is written. Check the values with `adata export
dataframe data.h5ad obs --columns cluster`. Equality compares each value's
text, so in a float column `n == 1` matches nothing where `n == 1.0` matches;
`>=` and `<=` compare numbers.

### `Operator '>' needs a number, but got 'high'.`

The ordering operators compare numbers. For text, use `==`, `!=` or `in`. See
[Query expressions](reference/query.md).

### `First dimension mismatch: input has 2000 rows, but obs has 2688 cells.`

`import` checks an array's length against the axis its path implies:
`obsm/...` against obs, `varm/...` against var, `X` and layers against both.
The input belongs to another store, or is transposed.

### `Provide at least one of --obs, --var, --obs-query or --var-query.`

`subset` with no selection would copy the store. Use `convert` to copy, or add
a selection.

## Visium and Xenium

These are refused before anything is written too.

### `12 of the matrix's 2,702 barcodes have no row in spatial/tissue_positions.csv (the first is 'AAAC...-1')`

The positions and the matrix come from different runs, or the positions
were edited. Every spot of the matrix needs a position; spots in the
positions but not the matrix are fine. A Xenium cell missing from
`cells.csv.gz` is refused the same way, as is a Visium HD cell with no
outline in `cell_segmentations.geojson` (named there by its integer
`cell_id`), and so is a table (positions, cells, outlines or an analysis
table) with a row for none of the matrix's barcodes. An `analysis/` folder
copied in from another run is refused this way; move it out of the folder
to convert without it.

### `tissue_positions.csv: pxl_col_in_fullres holds values that are not numbers.`

A coordinate, centroid or projection column holds text. Missing values are
fine (they become NaN); a word is not. The file has been edited, or is not
what its name says.

### `This Visium HD output has no 2 µm bins; it has 8 µm, 16 µm. Pick one with --bin-size.`

`--bin-size` names a folder under `binned_outputs/` (`square_002um` for 2).
Space Ranger writes 2, 8 and 16 µm bins, but a copied or trimmed output may
hold fewer; the message lists those there are.

### `outs has no segmented cells: --cells reads the segmented_outputs/ folder Space Ranger 4 writes for Visium HD.`

Space Ranger segments cells from version 4, for Visium HD only, and only
when the run had the image it segments from. Standard Visium and Space
Ranger 3 output have bins or spots only; leave `--cells` out.

### `cell_segmentations.geojson: feature 1234 is not valid JSON (Expecting ',' delimiter).`

The GeoJSON was cut short, usually by a copy that did not finish. Copy it
again from the Space Ranger output.

### `This matrix holds 2 libraries (A1, B1), as spaceranger aggr writes`

An aggregated matrix's spots come from several slides, with no one image to
place them on. Convert each library's own `outs/`, and `adata concat` the
results if you need them together.

### `... has filtered_feature_bc_matrix/ as a Matrix Market folder but no filtered_feature_bc_matrix.h5`

The positions are read only beside the `.h5` matrix. Convert the folder with
`--from 10x-mtx` for the counts alone. A Xenium bundle with only
`cell_feature_matrix/` says the same.

### `... has no cells.csv.gz. It has cells.parquet only, which is not read yet.`

Xenium writes `cells.csv.gz` beside `cells.parquet` in every version so far;
a bundle trimmed to Parquet cannot be read yet.

### `--unfiltered does not apply when converting xenium to h5ad.`

Xenium has one matrix. `--unfiltered`, `--library-id` and `--no-images` are
Visium's.

### `10,225 bins have no clusters, X_umap or X_pca in analysis/; they are left empty.`

A warning, not a refusal. Space Ranger's own analysis leaves out Visium HD
bins with no counts, and analyses only the filtered spots or bins, so with
`--unfiltered` the rest have none either. Those rows get no cluster and NaN
in `obsm["X_umap"]` and `obsm["X_pca"]`; filter on
`adata.obs["graphclust"].notna()` to keep the analysed ones. Tables that
lack different rows warn apart.

## Leftovers from an interrupted run

### `'out.adata-tmp.h5ad' already exists, probably left by an interrupted run. Check what it holds, remove it, and try again.`

A run was killed outright, by the out-of-memory killer or a scheduler's time
limit, before it could clean up. Commands write to `out.adata-tmp.<ext>` and
rename it into place at the end, moving an existing output aside to
`out.adata-old.<ext>` first:

- **`*.adata-tmp.*`** is always incomplete. Delete it.
- **`*.adata-old.*`** is the previous output, left because the run died
  between the two renames. If the output itself is missing, rename the
  `adata-old` store back to it; otherwise delete it.

{: .important }
The command refuses rather than deleting either itself, since an `adata-old`
store may be the only copy of your data. See
[Pipelines and containers](guides/pipelines.md#outputs-appear-whole-or-not-at-all).

## Zarr

### anndata cannot open a Zarr store `adata` wrote

A Zarr store written from an input that is not Zarr is v3, which anndata
reads from 0.12 with zarr-python 3. For an older anndata, write v2:

```bash
adata convert data.h5ad -o data.zarr --zarr-format 2
```

A Zarr source keeps its own version, so this arises from any input that is
not Zarr (an `.h5ad`, a 10x matrix, a loom, ...), and from `concat` of Zarr
inputs of different versions.

### A member added by another tool does not show in anndata

anndata reads a Zarr store's consolidated metadata, an index of every member
written at the root, and a tool that adds a member without rewriting it
leaves the member invisible. `adata` rewrites the index whenever it closes a
store it wrote; for a store another tool changed, run
`zarr.consolidate_metadata("data.zarr")` in Python.

## Memory and speed

### The command was killed for running out of memory

Lower `--chunk`: a dense block holds `--chunk` × n_var values, 1,024 rows by
default. Three things are held whole by design: obs and var columns, an
`export array` written to stdout (give `-o` instead), and a pycisTopic pickle,
which cannot be read in part. See
[Pipelines and containers](guides/pipelines.md#memory-and---chunk).

### `A loom holds its matrices dense: 54.0 GB to compress and write. It is written, but may take a while`

A loom has no sparse matrices, so a sparse store's zeros are all written.
Chunks of zeros compress to almost nothing, but writing them takes time. If
the reader can take `.h5ad`, `.zarr` or a 10x `.h5`, use that instead.

### It is slower than loading the file with anndata

For data that fits in memory, it often is: streaming trades time for memory.
[Benchmarks](reference/benchmarks.md) publishes the rows where the baseline
wins alongside those where it loses.

## Installing

### `ModuleNotFoundError: No module named 'click'` from `adata convert`

0.7.0 to 0.17.2 imported click, which typer stopped installing in 0.26, so
`convert` crashed on any new install of them. Where click was installed
anyway, they refused every convert with `... cannot be combined with
--inplace, --all, --dtype, --indices-dtype, --layout` instead. 0.17.3 fixes
both: `pip install -U pyadata-cli`. To stay on an older release, install
`'typer<0.26'` beside it, which brings the click it expects.

## Optional extras

### `Reading a pycisTopic pickle needs pandas and scipy, which rebuild its tables and matrices: pip install 'pyadata-cli[pickle]'.`

pandas and scipy are not dependencies of `adata`; the `pickle` extra adds
them, and only pycisTopic pickles need them. The container image has them.

### `Reading Visium HD positions needs pyarrow, which reads its Parquet tables: pip install 'pyadata-cli[spatial]'.`

Visium HD records where each bin sits only as Parquet. The `spatial` extra
adds pyarrow, which only Visium HD needs; the container image has it.

## Reading the output elsewhere

### anndata warns `OldFormatWarning`

An element has no encoding metadata, which is how anndata 0.7 wrote stores,
and anndata read it with its legacy reader. The data is read correctly. If
`adata` wrote the store from a current source and anndata still warns, that
is a bug: please [report it](https://github.com/cellgeni/adata-cli/issues).

### Genes are named by id, not by symbol

10x, MTX, Parse and loom imports name features by gene id, which is unique,
with the symbols in a var column. `--var-names gene_symbols` names them as
scanpy does, and `--make-unique` suffixes the symbols that repeat.

### A tool wants floats, and `X` holds integers

Imports keep Cell Ranger's integer counts, where scanpy makes them float32.
`adata recast out.h5ad X --inplace --dtype float32` converts them.

## Warnings worth reading

Warnings go to stderr and do not change the exit code.

| Warning | Meaning |
|---|---|
| `N obs names not found in file` | Names in a `--obs` list the store does not have were skipped |
| `obs names are not unique across inputs. Pass --index-unique to disambiguate them.` | `concat` inputs share barcodes; `--index-unique -` appends each input's key |
| `'convert' with --dtype is deprecated and will be removed in 1.0.0; use 'recast' instead.` | The command ran as `recast`; change the script before 1.0.0 |
| `the 'h5ad' command is deprecated and will be removed in 1.0.0; use 'adata' instead.` | The command was renamed in 0.5.0 |
| `'info' is deprecated and will be removed in 1.0.0; use 'view' instead.` | `info` is now `view` |

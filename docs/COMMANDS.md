# Command reference

Every command takes a `.h5ad` file or a `.zarr` directory; the backend is
detected from the path. Results go to **stdout** and status to **stderr**, so
output pipes cleanly:

```bash
adata export dataframe data.h5ad obs 2>/dev/null | head
adata ls data.h5ad -1 | grep spatial
```

Run `adata <command> --help` for the authoritative flag list.

## Table of contents

- [view](#view)
- [ls](#ls)
- [subset](#subset)
- [split](#split)
- [concat](#concat)
- [convert](#convert)
- [recast](#recast)
- [create](#create)
- [export](#export)
- [import](#import)
- [Query expressions](#query-expressions)
- [Zarr format selection](#zarr-format-selection)

---

## `view`

AnnData-aware inspection.

```bash
adata view data.h5ad                  # n_obs x n_var and the top-level keys
adata view data.h5ad --tree           # a tree with each element's type
adata view data.h5ad --tree --depth 3
adata view data.h5ad obsm/X_pca       # detail for one entry
```

| Flag | Meaning |
|---|---|
| `--tree`, `-t` | Show a tree of all entries |
| `--depth N`, `-d` | Maximum recursion depth (with `--tree`) |

`info` is a deprecated alias, removed in 1.0.0.

## `ls`

Format-agnostic listing. Makes no AnnData assumptions, so it also works on
`.loom` and plain `.h5` files.

```bash
adata ls data.h5ad
adata ls data.h5ad --long             # type, shape, dtype, encoding
adata ls data.h5ad uns --depth 2      # only below a path
adata ls data.h5ad -1 | grep spatial  # bare paths, for piping
```

## `subset`

Write a filtered copy. Select by name list, by expression, or both axes at
once.

```bash
adata subset data.h5ad -o out.h5ad --obs barcodes.txt
adata subset data.h5ad -o out.h5ad --obs-query "cluster == Cortex_2"
adata subset data.h5ad -o out.h5ad -q "n_counts > 1000 and cluster in A,B"
adata subset data.h5ad -o out.h5ad --var-query "highly_variable == True"
adata subset data.h5ad --inplace --obs barcodes.txt
```

| Flag | Meaning |
|---|---|
| `--output`, `-o` | Output path. Required unless `--inplace` |
| `--inplace` | Replace the source (written to a temporary path first) |
| `--obs` / `--var` | File of names to keep, one per line |
| `--obs-query`, `-q` / `--var-query` | Keep rows matching an expression |
| `--chunk`, `-C` | Row chunk size for dense matrices |
| `--zarr-format` | Zarr version to write; defaults to the source's |

`raw/` is carried over and matched against its **own** var axis, which usually
holds more genes than the main object.

The selection is resolved before anything is written, and the output is
written beside its destination (`out.adata-tmp.h5ad`) and moved into place
only once complete. A query that fails therefore leaves nothing behind, and an
existing output (or, with `--inplace`, the source) is replaced only once its
successor is finished. The output path may not name the input; use
`--inplace`.

## `convert`

Change a store's format. The output's extension picks the format; `--to`
names it when the extension does not say, and `--from` names the input's
when it cannot be detected from its content.

```bash
adata convert data.h5ad -o data.zarr
adata convert data.zarr -o data.h5ad --compression gzip
adata convert old.zarr -o new.zarr --zarr-format 3
adata convert data.h5ad -o data.zarr --overwrite
adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
adata convert counts.h5ad -o for_cellbender.h5 --layer counts
adata convert Solo.out/Gene/filtered -o counts.h5ad
adata convert counts.h5ad -o counts_mtx --to 10x-mtx
adata convert splitpipe_out -o parse.h5ad
adata convert velocyto.loom -o velocyto.h5ad
adata convert counts.h5ad -o counts.loom
adata convert GSE123_counts.txt.gz -o counts.h5ad --transpose
```

| Flag | Meaning |
|---|---|
| `--output`, `-o` | Output path (required) |
| `--from` | Input format, if it cannot be detected |
| `--to` | Output format, if the extension does not say |
| `--zarr-format` | Zarr version to write; defaults to the source's, else 3 |
| `--compression` | `none`, `gzip` or `lzf` for h5ad; `none`, `gzip`, `zstd` or `blosc` for Zarr; `gzip` (default) or `none` for an MTX folder |
| `--overwrite` | Replace an existing output |
| `--force` | Convert despite a lossy, inflating or placeholder result |

| Format | Extension | Read | Write |
|---|---|---|---|
| `h5ad` | `.h5ad` | yes | yes |
| `zarr` | `.zarr` (v2 or v3) | yes | yes |
| `10x-h5` | `.h5` | Cell Ranger 2 and 3+, Space Ranger, Xenium | Cell Ranger 3+ layout |
| `cellbender` | `.h5` | CellBender `remove-background` 0.2 and later | no |
| `10x-mtx` | a folder (`--to 10x-mtx`) | Cell Ranger 2 and 3+, STARsolo (velocyto too) | Cell Ranger 3+ layout, gzipped |
| `parse` | a folder | Parse Biosciences split-pipe: a run, a sample or a `DGE_*` folder | no |
| `loom` | `.loom` | loompy, velocyto, kallisto\|bustools, SCope (pySCENIC, SCENIC+, pycistopic); spec 2 and 3 | spec 3.0.0, as loompy 3 writes it |
| `text` | `.csv`, `.tsv`, `.tab`, `.txt`, `.data`, each also `.gz` | a dense matrix, as anndata's `read_csv` and `read_text` read it | no |

Formats other than h5ad and Zarr convert to and from AnnData only: to turn a
10x file into another format, convert it to `.h5ad` first.

Between h5ad and Zarr every element is copied as it is, streamed, so the
store's size is not a limit. Chunking carries over, and so does compression
where the target can express it: Zarr codecs travel between stores of the same
Zarr version, a compressed Zarr array becomes gzip in an h5ad, and an h5ad
written to Zarr gets zarr-python's default codec. `--compression` replaces all
of that with one choice. Elements are not re-encoded, so a store written by
anndata 0.7 keeps its legacy layout.

Every check runs before anything is written. The output is written beside
its destination (`out.adata-tmp.zarr`) and moved into place only once
complete, so a failed conversion leaves nothing behind. An existing output is
refused unless `--overwrite` is given, and is replaced only once its
successor is finished.

To change a matrix's dtype, layout or density instead, use
[`recast`](#recast).

### 10x Genomics HDF5

Cell Ranger's `filtered_feature_bc_matrix.h5` and `raw_feature_bc_matrix.h5`,
and the same layout from Space Ranger, Xenium and CellBender. The file is
recognised by its content, whatever its name.

```bash
adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
adata convert raw_feature_bc_matrix.h5 -o rna.h5ad --feature-type "Gene Expression"
adata convert multiome.h5 -o sample.h5ad --split-feature-types
adata convert cellranger2.h5 -o mouse.h5ad --genome mm10
adata convert counts.h5ad -o out.h5 --layer counts
```

| Flag | Meaning |
|---|---|
| `--genome` | Keep this genome's features; picks the group in a Cell Ranger 2 file holding several |
| `--var-names` | `gene_ids` (default) or `gene_symbols`, scanpy's default |
| `--make-unique` | Suffix repeated var names as `var_names_make_unique` does |
| `--feature-type` | Keep one feature type, e.g. `"Gene Expression"` |
| `--split-feature-types` | Write one output per type, `OUT.<type>.h5ad` |
| `--layer` | Export: the matrix to write, `X` (default), `raw/X` or a layer |
| `--gene-ids-column` | Export: var column with gene ids (default `gene_ids`) |
| `--gene-names-column` | Export: var column with gene names (default the var names) |
| `--feature-types-column` | Export: var column with feature types (default `feature_types`) |

Features are named by gene id, with the symbols in `var["gene_symbols"]`:
the layout `scanpy.read_10x_mtx(var_names="gene_ids")` gives. Ids are
unique where symbols repeat (Cell Ranger's GRCh38-2020-A reference repeats
10), so naming by them, as scanpy does by default, needs `--make-unique` to
give unique names; `--var-names gene_symbols` does that when wanted. The
exception is a probe-level (Flex) matrix, whose several probes per gene
share a gene id: pass `--make-unique` there, or use `var["probe_ids"]`.
Either way, repeated names are reported when they occur. With it,
reading gives what `scanpy.read_10x_h5(gex_only=False)` gives, column for
column: `gene_ids`, `feature_types`, every other per-feature field Cell Ranger
recorded (`genome`, and `pattern`, `read` and so on for antibody and CRISPR
features), `probe_ids` for a probe-based (Flex) matrix, and
`filtered_barcodes` in obs where present. Unlike scanpy, every feature type
is kept unless `--feature-type` says otherwise, and `X` keeps Cell Ranger's
integer counts, where scanpy converts them to float32; run `adata recast
out.h5ad X --inplace --dtype float32` if a tool needs floats. Repeated var
names are kept and reported, as scanpy does; `--make-unique` suffixes them. The run's metadata, which scanpy
drops, goes in `uns["cellranger"]`.

The matrix is stored genes x cells in CSC, which is byte for byte cells x
genes in CSR, so `data`, `indices` and `indptr` stream straight across;
`--split-feature-types` reads them once for every output.

Writing produces the Cell Ranger 3+ layout for CellBender, SoupX,
DropletUtils, Seurat and anything else that reads Cell Ranger output: gzip
compression and fixed-width strings, as Cell Ranger writes them, since
PyTables (which CellBender reads with) cannot read the variable-length kind.
A CSC or dense matrix is transposed or sparsified on the way, streaming.
Each feature needs an id and a type: if var has no `gene_ids` and no
`feature_types` column (or the columns named by the flags above), the export
is refused, and `--force` writes the names as ids and `Gene Expression` as
the type. A matrix of floats is written as it is, with a warning. A file
imported by `convert` exports with its run metadata and tag fields intact.

### CellBender

The output of CellBender's `remove-background` (`output.h5` and
`output_filtered.h5`, 0.2 and later) is recognised by its latent variables,
whatever its name. Its matrix is a 10x one and is read as above; the latents
are kept, under the names CellBender's own loaders give them, so code written
against those keeps working. Those loaders name features by symbol; pass
`--var-names gene_symbols` for the same var names.

| CellBender | AnnData |
|---|---|
| denoised counts | `X` |
| per-droplet latents (`cell_probability`, `cell_size`, `background_fraction`, `droplet_efficiency`; `latent_*` in 0.2) | obs columns, NaN for droplets CellBender did not analyse |
| which droplets were analysed | `obs["cellbender_analyzed"]` |
| `gene_expression_encoding` (`latent_gene_encoding` in 0.2) | obsm, same name |
| `ambient_expression` | `var["ambient_expression"]` |
| which features were analysed (`features_analyzed_inds`, 0.3+) | `var["cellbender_analyzed"]` |
| everything else (fitted constants, learning curve, settings) | `uns["cellbender"]` |

```bash
adata convert output.h5 -o cellbender.h5ad
adata convert output.h5 -o cells.h5ad --min-cell-probability 0.5
adata convert output_filtered.h5 -o cb.h5ad --raw raw_feature_bc_matrix.h5
```

| Flag | Meaning |
|---|---|
| `--min-cell-probability` | Keep droplets whose `cell_probability` is at least this; CellBender's own cutoff is 0.5 |
| `--raw` | Add the raw Cell Ranger matrix CellBender ran on as `layers["cellranger"]`, and the denoised counts as `layers["cellbender"]`, as `load_anndata_from_input_and_output` does |

The full `output.h5` keeps every droplet, as `read_10x_h5` would. `--raw`
matches barcodes, so it works for `output_filtered.h5` too, whose cells
CellBender orders by count; the raw file must list the same features in the
same order, and every output barcode, or it is refused. The 10x flags above
apply as well. `--from 10x-h5` reads a CellBender file as a plain 10x matrix,
without the latents.

### 10x Genomics and STARsolo MTX folders

A folder holding `matrix.mtx`, `barcodes.tsv` and `features.tsv` (or
Cell Ranger 2's `genes.tsv`), each plain or gzipped, is recognised as one.
Every shape Cell Ranger and STARsolo write is read:

* Cell Ranger 3+ (`filtered_feature_bc_matrix/`), with id, name and feature
  type per feature;
* Cell Ranger 2 (`filtered_gene_bc_matrices/`), one subfolder per genome:
  give the parent and `--genome` picks one, or the only one is taken;
* STARsolo `Solo.out/Gene/filtered/`, `GeneFull/` and so on, uncompressed,
  with two or three feature columns;
* STARsolo `Solo.out/Velocyto/raw/`: `spliced.mtx`, `unspliced.mtx` and
  `ambiguous.mtx` become layers of those names, and X is a copy of
  `spliced`, the layout scvelo expects;
* any of these with every file under one prefix, as GEO serves them
  (`GSM123_matrix.mtx.gz`, ...).

```bash
adata convert filtered_feature_bc_matrix -o counts.h5ad
adata convert Solo.out/Gene/filtered -o counts.h5ad
adata convert Solo.out/Velocyto/raw -o velocyto.h5ad
adata convert filtered_gene_bc_matrices -o mouse.h5ad --genome mm10
adata convert GSE000_RAW -o sample.h5ad --split-feature-types
adata convert counts.h5ad -o counts_mtx --to 10x-mtx --layer counts
```

The import flags are those of the HDF5 format above (`--genome`,
`--var-names`, `--make-unique`, `--feature-type`, `--split-feature-types`),
plus `--compression` for the output store, which is gzip on h5ad unless
given. Features are named by gene id unless `--var-names gene_symbols` is
given; with it, reading gives what `scanpy.read_10x_mtx(gex_only=False)`
gives, column for column, except that integer counts stay integers (int32, int64
when a value needs it) where scanpy converts them to float32; a `real` matrix
is read as float64. Duplicate entries are summed, as scipy does.

The matrix is read in two streaming passes, so its size is not a limit: the
first checks the whole folder -- barcode and feature counts against the
matrix, every entry against its size -- before any output exists, and the
second writes X. Files in barcode order, as Cell Ranger and STARsolo write
them, are written straight through; others are sorted through scratch space
beside the output.

Exporting needs `--to 10x-mtx`, since a folder has no extension to say so,
and writes a Cell Ranger 3 folder: gzipped (`--compression none` for plain
files), three feature columns, entries in barcode order, from X or the
matrix `--layer` names. Feature ids and types come from var as for the HDF5
export, with the same `--force` placeholders. A barcode or feature holding a
tab or a line break is refused, since it would break the TSV file.

### Parse Biosciences split-pipe

split-pipe writes, for the combined run and for each sample it was given,
`DGE_filtered/` (cells) and `DGE_unfiltered/` (every barcode), each holding
`count_matrix.mtx` (cells x genes), `all_genes.csv` and
`cell_metadata.csv`:

```
run/all-sample/DGE_filtered/   (all-well/ in older versions)
run/all-sample/DGE_unfiltered/
run/<sample>/DGE_filtered/ ...
```

Give the run folder, a sample folder or a `DGE_*` folder:

```bash
adata convert run -o parse.h5ad
adata convert run -o s1.h5ad --sample s1
adata convert run -o parse.h5ad --all-samples --unfiltered
adata convert run -o mouse.h5ad --genome mm10
adata convert run/all-sample/DGE_filtered -o parse.zarr
```

| Flag | Meaning |
|---|---|
| `--sample` | The sample folder to read; the combined `all-sample` by default |
| `--all-samples` | Write one output per sample, `OUT.<sample>.h5ad` |
| `--unfiltered` | Read `DGE_unfiltered/` rather than `DGE_filtered/` |
| `--genome` | Keep that genome's genes, and the cells split-pipe called as that species |

Reading gives what Parse's scanpy recipe gives (`read_mtx`, then both tables
with pandas, indexed by `bc_wells` and `gene_name`), checked against it in
CI, except that integer counts stay integers and features are named by gene
id unless `--var-names gene_symbols` is given. obs holds every
`cell_metadata.csv` column, typed as pandas reads it and stored as anndata
writes it: a text column that repeats a value is a categorical. var holds
`gene_symbols` (or `gene_ids`) and `genome`. `--var-names` and
`--make-unique` apply as for 10x input.

`--genome` works on runs made with a reference of several genomes built by
split-pipe, where each gene and cell names its own. A combined reference
built for Cell Ranger (`GRCh38-mm10-2020A`) is one genome to split-pipe, so
every gene and cell names it, and the species shows only in the gene ids'
prefixes.

Both files are read in two streaming passes, as for 10x MTX folders, so a
run's size is not a limit: the first checks the matrix against both tables
before any output exists, and the second writes obs and X.

### loom

A loom holds a dense genes x cells matrix, layers of the same shape, an
attribute per cell (`col_attrs`) and per gene (`row_attrs`), graphs, and
global attributes. velocyto writes its counts this way, as do loompy and
kallisto|bustools, and pySCENIC, SCENIC+ and pycistopic their results for
the SCope viewer.

```bash
adata convert velocyto.loom -o velocyto.h5ad
adata convert velocyto.loom -o unspliced.h5ad --layer unspliced
adata convert pyscenic_output.loom -o scenic.h5ad
adata convert counts.loom -o dense.h5ad --dense --var-names gene_symbols
adata convert counts.h5ad -o counts.loom
```

| loom | AnnData |
|---|---|
| `/matrix`, or `/layers/spliced` when there is one | `X`, CSR (`--dense` keeps it dense) |
| `/layers/<name>`, and `/matrix` when X is `spliced` | `layers[<name>]`, `layers["matrix"]` |
| `col_attrs` with one value per cell / several | obs columns / obsm |
| `row_attrs`, the same way | var columns / varm |
| compound `col_attrs` and `row_attrs`, a field per column | dataframes in obsm and varm |
| `col_graphs`, `row_graphs` | obsp, varp |
| global attributes | `uns["loom"]["attrs"]` |

| Flag | Meaning |
|---|---|
| `--layer` | The loom layer that becomes X: `matrix` (the main one) or a layer name |
| `--dense` | Keep X and the layers dense |
| `--obs-names-attr` | The attribute naming cells (default `CellID`, else `cell_id`, `obs_names` or `barcode`) |
| `--var-names-attr` | The attribute naming genes (default the ids: `Accession` or `gene_id`) |

Reading gives what `anndata.read_loom` gives, checked against it in CI, with
three differences: values keep their dtype (velocyto's integer counts stay
integers, where anndata makes X float32); graphs and global attributes are
kept, where anndata drops them; and genes are named by id (`Accession`),
the symbols (`Gene`) staying a column, unless `--var-names gene_symbols` is
given. Strings are read as loompy reads them, with XML references such as
`&#233;` turned back into characters. Looms from other writers read too:
R's LoomExperiment (graph vertices stored as floats), Baysor (genes in a
`Name` attribute, and a `.loom` holding the matrix alone, which is
recognised by its name), and R's writers' factors and logicals, which
become categoricals and booleans rather than their codes.

The matrix is read a slab of cells at a time, a whole number of the file's
64-cell chunks, so each chunk is decompressed once and memory is set by the
slab, not the file.

Writing produces a spec 3.0.0 loom that loompy reads and validates: X (or
the matrix `--layer` names) as `/matrix`, every layer, obs and var as
attributes (`CellID`; `Gene` for symbols and `Accession` for ids), 2-D obsm
and varm, obsp and varp as graphs, and uns scalars as global attributes.
Strings are written as loompy writes them, ASCII with XML references. A
loom's matrices are dense by definition: the summary gives their size before
compression, and above 10 GB a warning says it will take a while. Chunks
holding only zeros compress to almost nothing.

#### SCope looms

pySCENIC, SCENIC+ and pycistopic (and VSN, and R's SCopeLoomR) write their
results into the loom for the SCope viewer: each regulon's AUC per cell and
its target genes (or regions), the embeddings and clusterings shown in the
viewer, and a `MetaData` attribute naming them. `anndata.read_loom`
refuses these looms, since loompy's validator rejects compound attributes.
Here such a loom reads as `loom (SCope)`, the summary naming what was found
(`298 regulons, 5 embeddings, 1 clustering`):

| SCope loom | AnnData |
|---|---|
| `col_attrs/RegulonsAUC`, a field per regulon | `obsm["RegulonsAUC"]`, a dataframe of cells x regulons |
| `row_attrs/Regulons`, a field per regulon, 0 or 1 | `varm["Regulons"]`, a dataframe of genes x regulons |
| `col_attrs/Embeddings_X` and `Embeddings_Y`, a field per embedding id; `Embedding` | `obsm["X_<name>"]`, n x 2, named from `MetaData` (`tSNE (default)` becomes `X_tSNE_default`); `Embedding` is the default one (id -1) when the others lack it |
| `col_attrs/Clusterings`, a field per clustering id | a categorical obs column named from `MetaData`, its categories the clusters' descriptions in cluster-id order |
| other compound attributes (VSN's `ClusterMarkers_<i>`, SCopeLoomR's `MotifRegulonsAUC`) | dataframes in obsm or varm, under their own names |
| global `MetaData`, JSON, plain or compressed | `uns["scope"]["metadata"]`, the JSON text |
| `regulonThresholds` in `MetaData` | `uns["scope"]["regulon_thresholds"]`, a dataframe by regulon |
| the embeddings and clusterings in `MetaData` | `uns["scope"]["embeddings"]` and `["clusterings"]`: each id's name, and where it went |

The dataframes keep the regulons' names, `ATF3_(+)` and the like, as their
columns, and the obs or var names as their index, which is how anndata
expects them; `pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)`, the
pySCENIC tutorial's recipe, gives the same table, as CI checks. A field
name holding `/`, which HDF5 and Zarr cannot store, has it replaced by `_`,
with a warning. A clustering named like an obs column already there gets a
suffix (`Tissue_1`), with a note; one `MetaData` does not name is called
`clustering_<id>`, and an embedding `X_embedding_<id>`. A value with no
cluster in `MetaData` is missing. `MetaData` that is neither JSON nor
base64 of compressed JSON is kept as text, with a warning; the embeddings
and clusterings are then named by id, and a clustering's categories are the
ids found.

Region-based looms (SCENIC+, pycistopic) hold regions (`chr1:100-200`)
where genes go, so var is named by region, and SCENIC+'s `linkedGene` is a
var column. The compound attributes are read a block of cells at a time,
so memory is one block across all regulons, never the table.

Writing a loom writes a plain one: an AnnData's regulons, embeddings and
clusterings are not yet written in SCope's form (#61).

### Text tables

A dense matrix in text, one line per row: GEO count tables, R's
`write.table` and `write.csv` output, a pandas `to_csv`.

```bash
adata convert counts.csv -o counts.h5ad
adata convert GSE123_counts.txt.gz -o counts.h5ad --transpose
adata convert table.txt -o table.h5ad --delimiter ";" --dense
```

| Text table | AnnData |
|---|---|
| each line (each column with `--transpose`) | a cell: X's row, named by the first column |
| the header, or the last `#` comment line | var names |
| no header; no names in the first column | var and obs numbered from 0, as anndata numbers them |
| the delimiter and layout found | `uns["text"]` |

| Flag | Meaning |
|---|---|
| `--delimiter` | A character, `tab` or `whitespace` (runs of spaces and tabs). Default from the extension: `,` for `.csv`, tab for `.tsv` and `.tab`, whitespace for `.txt` and `.data`, as `read_text` splits |
| `--transpose` | The file holds genes as rows and cells as columns, as count tables often do |
| `--first-column-names`, `--no-first-column-names` | Whether the first column names the rows, when detection gets it wrong |
| `--dense` | Keep X dense (default: CSR) |
| `--make-unique` | Suffix repeated var names, as `var_names_make_unique` does |

The layout is found as anndata's `read_text` finds it, checked against it
in CI. The first line is a header when its last field is not a number, and
the first column names the rows when its first value is not a number. A
header with a field more than the values has a corner cell over the row
names; one with as many (R's `write.table`) names only the columns. Lines
starting with `#` before the data are comments, the last of them naming
the columns when there is no header. Names have `"` stripped, and blank
lines are skipped.

Values keep integers: when every value is whole, X is int32 (int64 if a
value needs it), as for MTX folders; otherwise it is float64, which holds
what the text holds where anndata gives float32. A line with a different
number of values, or a value that is not a number (`NA`), is refused with
its line and column, before any output exists.

The file is read twice, a block of lines at a time, each block parsed by
numpy, so memory is set by the block rather than by the file: once to find
the layout, check every line and pick the dtype, and once to write X. A
transposed file's entries are sorted into cells through on-disk buckets.
Gzip is found by content, whatever the name.

## `recast`

Change a matrix's dtype, layout or density. Counts held as float64 cost twice
the disk and twice the read for no information; a tool that wants CSC cannot
use a CSR store; and `concat` refuses inputs whose encodings disagree.

```bash
adata recast data.h5ad X -o out.h5ad --dtype float32
adata recast data.h5ad X -o out.h5ad --layout csc
adata recast data.h5ad X --inplace --dtype float32 --indices-dtype int32
adata recast data.h5ad --all -o out.h5ad --dtype float32
adata recast data.h5ad layers/counts -o out.h5ad --layout dense --force
```

| Flag | Meaning |
|---|---|
| `--output`, `-o` | Output path. Required unless `--inplace` |
| `--inplace` | Replace the source (written to a temporary path first) |
| `--all` | Recast `X`, every layer and `raw/X` |
| `--dtype` | New dtype for the values, e.g. `float32` |
| `--indices-dtype` | New dtype for sparse indices: `int32` or `int64` |
| `--layout` | `csr`, `csc`, `dense` or `sparse` |
| `--force` | Recast despite a lossy cast or a large size increase |
| `--in-memory` | Transpose in memory rather than streaming |
| `--chunk`, `-C` | Row chunk size for dense matrices |
| `--zarr-format` | Zarr version to write; defaults to the source's |

Two things are refused before anything is written. A cast that would not
round-trip -- checked by casting every value and casting it back, not by
comparing dtypes -- and a densification that would inflate the store beyond
four times its size. `--force` overrides either, and the message says how
many values would change or how large the result would be.

The index dtype is **preserved** unless `--indices-dtype` asks otherwise, so
narrowing the values does not silently widen the indices and leave the file
bigger than it started. `indices` and `indptr` keep their own widths, which
can differ: a narrow matrix with more than 2^31 nonzeros needs int64 offsets
over int32 coordinates, and both are range-checked before writing.

The output path may not name the input; use `--inplace`. Either way the new
store is written beside its destination and moved into place only once the
conversion has finished, so a failure leaves the original untouched.

Transposing streams by default and works on matrices too large to load, at
the cost of two extra passes over the nonzeros. `--in-memory` is faster when
the matrix fits.

Until 0.7.0 this command was called `convert`. Until 1.0.0, `convert` given
any of `--dtype`, `--indices-dtype`, `--layout`, `--all` or `--inplace` still
runs `recast`, with a warning.

## `split`

One store per distinct value of a column.

```bash
adata split data.h5ad --by sample -o per_sample/
adata split data.h5ad --by cell_type -o clusters/ --dry-run
adata split data.h5ad --by cluster -o out/ --min-size 50
```

| Flag | Meaning |
|---|---|
| `--by`, `-b` | Column to split on |
| `--output-dir`, `-o` | Where to write the stores |
| `--axis` | `obs` (default) or `var` |
| `--min-size` | Skip groups smaller than this |
| `--dry-run` | Print the plan without writing |
| `--manifest / --no-manifest` | Write a CSV listing the outputs (default on) |

Labels are sanitised for use as filenames; collisions get a numeric suffix.
Every store is written beside its destination and moved into place only once
all of them are complete, and the manifest is written last, so a split that
fails leaves no partial stores. An output that would land on the input is
refused.

## `concat`

Concatenate along the obs axis, streaming each input in turn.

```bash
adata concat a.h5ad b.h5ad -o merged.h5ad
adata concat *.h5ad -o merged.h5ad --join outer --label sample
adata concat a.h5ad b.h5ad -o m.h5ad --keys a,b --index-unique - --uns-merge same
```

| Flag | Meaning |
|---|---|
| `--join`, `-j` | `inner` (shared vars only) or `outer` (their union) |
| `--label` | Add an obs column recording each cell's source |
| `--keys` | Names for the inputs; defaults to their filenames |
| `--index-unique` | Delimiter for suffixing obs names with their key |
| `--merge` / `--uns-merge` | `drop` (default), `same`, `unique`, `first`, `only` |
| `--fill-value` | Value for dense cells introduced by an outer join |
| `--chunk`, `-C` | Row chunk size while streaming |
| `--zarr-format` | Zarr version to write; defaults to the inputs' when they agree, otherwise v3 |

obs columns keep their dtypes: categoricals union their category sets, nullable
columns keep their masks, and a column missing from one input is padded rather
than dropped.

`obsp`, `varp` and `raw` are **not** carried over — a pairwise value has no
meaning between cells from different inputs. anndata's own `concat_on_disk`
refuses these too.

### Merge strategies

| Strategy | Keeps a value when |
|---|---|
| *(unset)* | never — the element is dropped |
| `same` | every input has it and they all agree |
| `unique` | exactly one distinct value among the inputs that have it |
| `first` | always; takes the earliest input that has it |
| `only` | exactly one input has it at all |

## `create`

Write a new, empty store for `import` to fill in.

```bash
adata create out.h5ad --n-obs 5000 --n-var 2000
adata create out.zarr --obs-names cells.txt --var-names genes.txt
```

Give each axis either a size (names are generated as `cell_0000`, `gene_0000`)
or a file of names. `--force` overwrites an existing store.

## `export`

| Subcommand | Produces | Accepts |
|---|---|---|
| `dataframe` | CSV | any dataframe path (`obs`, `var`, `raw/var`, …) |
| `array` | `.npy` | any dense array |
| `sparse` | `.mtx`, or `.mtx.gz` | CSR/CSC groups |
| `dict` | JSON | any group or scalar |
| `image` | PNG/JPEG/TIFF | 2D or 3D arrays |

```bash
adata export dataframe data.h5ad obs -o obs.csv
adata export dataframe data.h5ad obs --columns cluster,n_counts --head 100
adata export sparse data.h5ad X -o matrix.mtx
adata export array data.h5ad obsm/X_umap > umap.npy
adata export dict data.h5ad uns -o metadata.json
```

Omit `--output` to write to stdout. Writing to a file streams in chunks;
stdout is not seekable, so `export array` holds the array in memory on that
path.

`export sparse` gzips its output when the name ends in `.gz`. Whole-number
values are written as integers, and other floats with enough digits to read
back exactly.

## `import`

Write data into a store at any path. All subcommands need `--output`/`-o` or
`--inplace`.

```bash
adata import dataframe data.h5ad obs cells.csv --inplace -i cell_id
adata import array     data.h5ad obsm/X_umap umap.npy --inplace
adata import sparse    data.h5ad X counts.mtx --inplace
adata import dict      data.h5ad uns/params params.json --inplace
adata import image     data.h5ad uns/spatial/hires tissue.png --inplace
```

Dimensions are validated against the existing obs/var where the path implies
an axis.

For `sparse`: the file is read in two streaming passes, so its size is not a
limit, and it may be gzipped (`counts.mtx.gz`). The first pass checks the
whole file before the store is touched. `integer` values stay integers
(int32, or int64 when a value needs it), `real` ones become float64, and
`pattern` entries become ones. Duplicate entries are summed, as scipy does.
Only `coordinate general` files are read: `symmetric`, `skew-symmetric`,
`hermitian` and `complex` files, and dense `array` ones, are refused. An
input whose entries are not in row order is sorted through scratch space
beside the store.

For `dataframe`: columns that parse as integers or floats become numeric
arrays; string columns with few distinct values become categoricals. Use
`--categorical col1,col2` to force specific columns, or
`--no-auto-categorical` to keep all strings as `string-array`.

Giving `-o out.zarr` for an `.h5ad` source (or the reverse) converts the store
as it writes.

## Query expressions

Used by `subset --obs-query` and `--var-query`.

| Operator | Example |
|---|---|
| `==`, `!=` | `cluster == Cortex_2` |
| `<`, `<=`, `>`, `>=` | `n_counts > 1000` |
| `in`, `not in` | `cluster in A,B,C` |
| `and`, `or`, `not` | `n_counts > 500 and not cluster == Fiber_tract` |
| parentheses | `(a == 1 or b == 2) and c > 3` |

Values with spaces need quoting: `label == "cell type A"`. Equality and
membership compare the column's string form; the ordering operators parse it as
a number, and entries that will not parse simply do not match.

Only the columns a query mentions are read, so filtering on one annotation does
not touch the rest of the frame.

This is deliberately not SQL. For anything more involved — joins, aggregates,
window functions — export to CSV and use [duckdb](https://duckdb.org):

```bash
adata export dataframe data.h5ad obs -o cells.csv
duckdb -noheader -list -c \
  "SELECT _index FROM 'cells.csv' WHERE cluster='Cortex_2' AND n_counts > 1000" \
  > barcodes.txt
adata subset data.h5ad -o cortex.h5ad --obs barcodes.txt
```

## Zarr format selection

New Zarr stores follow the source store's version, so a v2 input is not
silently upgraded. `--zarr-format 2|3` on `create`, `subset`, `split`,
`concat`, `convert` and `recast` overrides that. Writing a Zarr store from
an `.h5ad` source defaults to v3. `concat` follows its Zarr inputs when they
share one version (h5ad inputs do not count) and writes v3 when they differ.

```bash
adata create out.zarr --n-obs 100 --n-var 50 --zarr-format 2
adata subset data.zarr -o out.zarr --obs keep.txt --zarr-format 3
adata split data.zarr --by sample -o parts/ --suffix .zarr --zarr-format 3
adata concat a.zarr b.zarr -o merged.zarr --zarr-format 3
adata convert data.zarr -o v3.zarr --zarr-format 3
```

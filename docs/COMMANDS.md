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
| `10x-mtx` | a folder (`--to 10x-mtx`) | Cell Ranger 2 and 3+, STARsolo (velocyto too) | Cell Ranger 3+ layout, gzipped |

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
| `--var-names` | `gene_symbols` (default) or `gene_ids` |
| `--make-unique` | Suffix repeated var names as `var_names_make_unique` does |
| `--feature-type` | Keep one feature type, e.g. `"Gene Expression"` |
| `--split-feature-types` | Write one output per type, `OUT.<type>.h5ad` |
| `--layer` | Export: the matrix to write, `X` (default), `raw/X` or a layer |
| `--gene-ids-column` | Export: var column with gene ids (default `gene_ids`) |
| `--gene-names-column` | Export: var column with gene names (default the var names) |
| `--feature-types-column` | Export: var column with feature types (default `feature_types`) |

Reading gives what `scanpy.read_10x_h5(gex_only=False)` gives, column for
column: `gene_ids`, `feature_types`, every other per-feature field Cell Ranger
recorded (`genome`, and `pattern`, `read` and so on for antibody and CRISPR
features), `probe_ids` for a probe-based (Flex) matrix, and
`filtered_barcodes` in obs where present. Unlike scanpy, every feature type
is kept unless `--feature-type` says otherwise, and `X` keeps Cell Ranger's
integer counts, where scanpy converts them to float32; run `adata recast
out.h5ad X --inplace --dtype float32` if a tool needs floats. Repeated var
names are kept and reported, as scanpy does. The run's metadata, which scanpy
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
given. Reading gives what `scanpy.read_10x_mtx(gex_only=False)` gives,
column for column, except that integer counts stay integers (int32, int64
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

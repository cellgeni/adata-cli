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
| `sparse` | `.mtx` | CSR/CSC groups |
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
silently upgraded. `--zarr-format 2|3` on `create`, `subset`, `split` and
`concat` overrides that. Writing a Zarr store from an `.h5ad` source defaults
to v3.

```bash
adata create out.zarr --n-obs 100 --n-var 50 --zarr-format 2
adata subset data.zarr -o out.zarr --obs keep.txt --zarr-format 3
adata split data.zarr --by sample -o parts/ --suffix .zarr --zarr-format 3
adata concat a.zarr b.zarr -o merged.zarr --zarr-format 3
```

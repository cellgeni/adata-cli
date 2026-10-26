---
title: Inspect and export
parent: Guides
nav_order: 1
---

# Inspect and export
{: .no_toc }

1. TOC
{:toc}

## What is in a store

`view` reads a store as AnnData: its shape, then the keys of each part.

```bash
adata view data.h5ad
adata view data.h5ad --tree --depth 2   # every entry, with its type
adata view data.h5ad obsm/X_pca         # one entry: type, shape, dtype
```

`ls` makes no AnnData assumptions, so it also lists a `.loom`, a Cell Ranger
`.h5` or any other HDF5 file or Zarr store:

```bash
adata ls data.h5ad --long               # type, shape, dtype and encoding of each path
adata ls data.h5ad uns --depth 2        # only below one path
adata ls filtered_feature_bc_matrix.h5  # not AnnData; listed all the same
```

Both read metadata, not data, so they are as quick on a store of hundreds of
gigabytes as on a small one.

## Tables

`export dataframe` writes any dataframe as CSV: `obs`, `var`, `raw/var`, or
one stored under `obsm` or `uns`.

```bash
adata export dataframe data.h5ad obs -o cells.csv
adata export dataframe data.h5ad var --columns gene_ids,highly_variable
adata export dataframe data.h5ad obs --head 20
```

The index comes first, as `_index`. Categoricals are written as their labels,
and missing values of nullable columns as empty fields. Rows are read
`--chunk` (10,000 by default) at a time.

## Matrices and arrays

| You want | Command |
|---|---|
| A dense array (`obsm/X_umap`, `varm/PCs`, a dense `X`) as `.npy` | `adata export array data.h5ad obsm/X_umap -o umap.npy` |
| A sparse matrix (`X`, a layer, `obsp/distances`) as Matrix Market | `adata export sparse data.h5ad X -o counts.mtx.gz` |
| A whole Cell Ranger-style folder or `.h5` | `adata convert`; see [Convert formats](convert.md) |

`export sparse` gzips when the name ends in `.gz`, writes whole numbers as
integers and other floats with enough digits to read back exactly.
`export array` streams to a file; to stdout it holds the array in memory,
since a pipe cannot be seeked.

## Metadata and images

`export dict` writes a group or scalar as JSON, and `export image` writes a
2-D or 3-D array as PNG, JPEG or TIFF, such as a Visium tissue image:

```bash
adata export dict data.h5ad uns/neighbors -o neighbors.json
adata export image data.h5ad uns/spatial/V1_Adult_Mouse_Brain/images/hires -o hires.png
```

## Piping to other tools

Without `--output`, every export but `export image` writes to stdout, and
status and progress go to stderr, so the data stream stays clean:

```bash
adata export dataframe data.h5ad obs | duckdb -c "SELECT cluster, count(*) FROM read_csv('/dev/stdin') GROUP BY 1"
adata ls data.h5ad -1 | grep spatial
adata export dict data.h5ad uns/pca | jq '.variance_ratio[:5]'
```

For a filter `--obs-query` cannot express, export `obs`, query it with
duckdb, and give the names back to `subset`:

```bash
adata export dataframe data.h5ad obs -o cells.csv
duckdb -noheader -list -c \
  "SELECT _index FROM 'cells.csv' WHERE cluster='Cortex_2' AND n_counts > 1000" \
  > barcodes.txt
adata subset data.h5ad -o cortex.h5ad --obs barcodes.txt
```

## Putting things back

Each export has an `import` counterpart that writes the element into a store,
checking its length against obs or var where the path implies one:

```bash
adata import dataframe data.h5ad obs cells.csv --index-column _index --inplace
adata import array     data.h5ad obsm/X_umap umap.npy --inplace
adata import sparse    data.h5ad layers/counts counts.mtx.gz --inplace
```

`adata create` makes an empty store to import into, for building one from
scratch. [Commands](../reference/commands.md#import) has the details of each.

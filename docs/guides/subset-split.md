---
title: Subset and split
parent: Guides
nav_order: 2
---

# Subset and split
{: .no_toc }

1. TOC
{:toc}

## Keep some cells or genes

`subset` writes a copy holding the rows you select, on either axis or both.
Select by a file of names, one per line:

```bash
adata subset data.h5ad -o out.h5ad --obs barcodes.txt
adata subset data.h5ad -o out.h5ad --var genes.txt
```

or by an expression over obs or var columns:

```bash
adata subset data.h5ad -o cortex.h5ad --obs-query "cluster == Cortex_2"
adata subset data.h5ad -o big.h5ad -q "n_counts > 1000 and cluster in A,B"
adata subset data.h5ad -o hvg.h5ad --var-query "highly_variable == True"
```

[Query expressions](../reference/query.md) lists the operators. Only the
columns a query names are read. Name lists and queries combine: `--obs`
with `--var-query` keeps those cells and the genes that match.

Rows keep the source's order, not the order of a name list.

{: .note }
Names the store does not have are skipped, and counted in a warning (`1 obs
names not found in file`). Check it when a list came from another dataset. A query that matches no rows
is refused, and nothing is written.

## What a subset keeps

Everything, narrowed to the rows kept:

- `X`, layers, obsm, varm, obsp and varp, on the axes they span;
- `raw`, matched against its **own** var, which usually holds more genes than
  the main object, so a `--var` subset keeps raw's genes as they were;
- uns, as it is, apart from the colours below.

Categorical columns keep only the categories their rows use, as anndata's own
subsetting does, and `uns["<column>_colors"]` keeps the colours of those
categories. The index dtypes, chunking and compression of every matrix stay
as they were; [recast](recast.md) changes them.

## One store per sample, cluster or donor

`split` writes one store per distinct value of a column, reading each matrix
once for all of them:

```bash
adata split data.h5ad --by sample -o per_sample/
adata split data.h5ad --by cell_type -o clusters/ --dry-run
adata split data.h5ad --by cluster -o out/ --min-size 50
```

Each output is named after its value, made safe for a filename, with a
numeric suffix where two would collide. `--dry-run` prints the plan without
writing. `--min-size` skips small groups. `<input>_manifest.csv` in the
output folder lists each value, its file and its number of rows (columns `id`,
`column`, `value`, `anndatas` and `n`); `--no-manifest` leaves it out.
`--axis var` splits genes instead, by a var column.

Every store is written beside its destination and moved into place only when
all of them are done, so a split that fails leaves no partial outputs.
`split`, then [concat](concat.md), gives back the same cells.

## Replacing the source

`--inplace` replaces the source with the subset:

```bash
adata subset data.h5ad --inplace --obs barcodes.txt
```

The new store is written beside the source and swapped in only once it is
complete, so a failure leaves the source as it was. An output path that
names the input is refused rather than read and overwritten at once:

```
Error: Output path is the input: data.h5ad. Use --inplace to replace it, which
writes to a temporary file first.
```

## Into the other backend

The output's extension picks its backend, so a subset converts as it writes:

```bash
adata subset data.h5ad -o cortex.zarr --obs-query "cluster == Cortex_2"
adata split  data.zarr --by sample -o parts/ --suffix .h5ad
```

A Zarr output keeps the Zarr version of a Zarr source, and is v3 from an
`.h5ad`; `--zarr-format` chooses.

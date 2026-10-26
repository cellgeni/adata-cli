---
title: Concatenate
parent: Guides
nav_order: 3
---

# Concatenate
{: .no_toc }

1. TOC
{:toc}

## Joining stores

`concat` stacks stores along obs, streaming each input in turn, so memory is
set by `--chunk` and not by the inputs:

```bash
adata concat a.h5ad b.h5ad -o merged.h5ad
adata concat per_sample/*.h5ad -o merged.h5ad --join outer --label sample
```

Inputs may mix `.h5ad` and `.zarr`, and the output's extension picks its
backend.

## Which genes: `--join`

| `--join` | var of the output | Cells' values for genes an input lacks |
|---|---|---|
| `inner` (default) | the genes every input has | none: those genes are dropped |
| `outer` | every gene any input has | zero in a sparse matrix; `--fill-value` (default 0) in a dense one |

## Where each cell came from: labels and keys

Each input has a key: its filename without the extension, or the names
`--keys` gives, in order. `--label` adds an obs column holding each cell's key:

```bash
adata concat s1.h5ad s2.h5ad -o merged.h5ad --label sample --keys donorA,donorB
```

Barcodes repeat between samples. When obs names are not unique across the
inputs, `concat` says so; `--index-unique -` appends each cell's key to its
name (`AAACCTG-1-donorA`), as anndata's `index_unique` does.

## obs, obsm and the matrices

obs columns keep their dtypes: categoricals take the union of their
categories, nullable columns keep their masks, and a column one input lacks is
padded with missing values rather than dropped. obsm entries every input has
are stacked, dataframes among them, their columns following `--join`. `X` and
the layers every input has are stacked.

Matrices must agree on their encoding: inputs where one `X` is CSR and
another CSC are refused before anything is written, with the
[recast](recast.md) command that makes them agree.

## var, varm and uns: `--merge` and `--uns-merge`

Each input has its own var columns, varm and uns, and they may disagree. The
merge strategy says what the output keeps:

| Strategy | Keeps a value when |
|---|---|
| *(unset)* | never: the element is dropped |
| `same` | every input has it and they all agree |
| `unique` | exactly one distinct value among the inputs that have it |
| `first` | always; takes the earliest input that has it |
| `only` | exactly one input has it at all |

`--merge` applies to var's columns and to varm, compared on the output's
genes: in an outer join, an input lacking some of them counts as not having
the entry. `--uns-merge` applies to uns.

```bash
adata concat a.h5ad b.h5ad -o m.h5ad --merge same --uns-merge first
```

## What is not carried over

`obsp`, `varp` and `raw` are left out: a pairwise value has no meaning between
cells of different inputs, and `raw` would need its own join. anndata's
`concat_on_disk` leaves them out too. Recompute neighbours on the merged
store. A layer every input has is stacked like `X`, so counts kept in a layer
come through.

## Zarr inputs

The output keeps the Zarr version its Zarr inputs share; `.h5ad` inputs do not
count. Inputs of different versions give a v3 output. `--zarr-format` chooses
either way.

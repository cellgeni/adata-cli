---
title: Change dtype, layout or density
parent: Guides
nav_order: 4
---

# Change dtype, layout or density
{: .no_toc }

1. TOC
{:toc}

`recast` rewrites a matrix (`X`, a layer, `raw/X` or any 2-D array) with a
different value dtype, index dtype, layout or density, streaming, and copies
everything else as it is.

## Make a store smaller

Counts are often written as float64, which costs twice the disk and twice the
read of float32 for no information. Check that every value survives, then
write:

```bash
adata recast data.h5ad X -o small.h5ad --dtype float32
adata recast data.h5ad --all -o small.h5ad --dtype float32 --indices-dtype int32
```

`--all` takes `X`, every layer and `raw/X`. Before writing, `recast` casts
every value and casts it back, and says what it found:

```
X: float64 -> float32: every value round-trips
```

Index dtypes stay as they are unless `--indices-dtype` changes them, so
narrowing the values does not quietly widen the indices. `indices` and
`indptr` are checked separately: a matrix with more than 2<sup>31</sup>
nonzeros needs int64 `indptr`, and is refused int32 with the reason.

## Change the layout

```bash
adata recast data.h5ad X -o csc.h5ad --layout csc
adata recast data.h5ad layers/counts -o out.h5ad --layout dense --force
```

`--layout` is `csr`, `csc`, `dense` or `sparse` (sparse in the layout the
matrix already has, or CSR from dense). CSR to CSC is a transpose, which
streams through on-disk buckets by default and works on matrices too large to
load, at the cost of two more passes over the nonzeros. `--in-memory` is
faster when the matrix fits.

This is also how inputs that `concat` refuses for disagreeing on CSR and CSC
are made to agree.

## Refusals, and `--force`

Two things are refused before anything is written:

- **A cast that loses values.** Integers out of range, or fractions cast to an
  integer type:

  ```
  Error: X: float64 -> int32 is lossy: 2 of 2 values (100.00%) do not
  round-trip, largest change 0.7. Pass --force to convert anyway.
  ```
- **A densification that more than quadruples a matrix**, with the size it
  would reach.

{: .warning }
`--force` writes anyway. Use it when the loss is what you want, such as
rounding normalised values, and never to get past a message you have not read.

## In place

```bash
adata recast data.h5ad X --inplace --dtype float32
```

The new store is written beside the source and swapped in once complete, so
an interrupted run leaves the source as it was. Naming the source as `-o` is
refused, since the output would be emptied while it is still being read.

Until 0.7.0 this command was `convert`. Until 1.0.0, `convert` given any of
`--dtype`, `--indices-dtype`, `--layout`, `--all` or `--inplace` runs `recast`,
with a warning.

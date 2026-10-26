---
title: HDF5 matrices
parent: Formats
grand_parent: Reference
nav_order: 7
---

# HDF5 matrices

One 2-D numeric dataset of a plain HDF5 file, as anndata's `read_hdf`
reads it. `--key` names it; without `--key` nothing is written, and the
refusal lists the file's 2-D numeric datasets with their shapes and
dtypes, and the 1-D ones that could name their rows and columns.

```bash
adata convert matrix.hdf5 -o matrix.h5ad
adata convert matrix.hdf5 -o matrix.h5ad --key data/counts
adata convert export.h5 -o export.h5ad --from hdf5 --key exprs --transpose
adata convert matrix.hdf5 -o matrix.h5ad --key counts --obs-names-key barcodes --var-names-key genes
```

| HDF5 file | AnnData |
|---|---|
| the dataset `--key` names (its rows are cells; genes with `--transpose`) | X, CSR (`--dense` keeps it dense), its dtype kept |
| `row_names` and `col_names` at the root, naming the dataset's rows and columns | obs and var names |
| the key and the names' datasets | `uns["hdf5"]` |

| Flag | Meaning |
|---|---|
| `--key` | The 2-D dataset that becomes X |
| `--transpose` | The dataset holds genes as rows and cells as columns |
| `--obs-names-key`, `--var-names-key` | The 1-D datasets naming the cells and the genes, in place of `row_names` and `col_names` |
| `--dense` | Keep X dense (default: CSR) |
| `--make-unique` | Suffix repeated var names, as `var_names_make_unique` does |

A plain `.h5` is claimed by 10x Genomics files, so one that is no 10x
matrix needs `--from hdf5`; `convert` says so. A `row_names` or `col_names`
whose length does not fit is left out with a warning, and that axis is
numbered. The dataset is read a slab of cells at a time, a whole number of
its chunks along the cell axis (columns, with `--transpose`), as a
[loom's matrix](loom.md) is, so each chunk is decompressed once and memory is set by the
slab.

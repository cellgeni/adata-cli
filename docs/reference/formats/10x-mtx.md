---
title: 10x and STARsolo MTX folders
parent: Formats
grand_parent: Reference
nav_order: 3
---

# 10x Genomics and STARsolo MTX folders

A folder holding `matrix.mtx`, `barcodes.tsv` and `features.tsv` (or
Cell Ranger 2's `genes.tsv`), each plain or gzipped, is recognised as one.
Every shape Cell Ranger and STARsolo write is read:

* Cell Ranger 3+ (`filtered_feature_bc_matrix/`), with id, name and feature
  type per feature;
* Cell Ranger 2 (`filtered_gene_bc_matrices/`), one subfolder per genome:
  give the parent and `--genome` picks one, or the only one is taken. A
  folder holding a run's `filtered/` and `raw/` matrices (STARsolo's
  `Gene/`, Cell Ranger's `outs/`) is not taken for genomes: give the one to
  convert;
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

The import flags are those of the [10x HDF5 format](10x-h5.md) (`--genome`,
`--var-names`, `--make-unique`, `--feature-type`, `--split-feature-types`),
plus `--compression` for the output store, which is gzip on h5ad unless
given. Features are named by gene id unless `--var-names gene_symbols` is
given; with it, reading gives the var names and columns
`scanpy.read_10x_mtx(gex_only=False)` gives, checked against scanpy in CI,
except that integer counts stay integers (int32, int64 when a value needs
it) where scanpy converts them to float32; a `real` matrix is read as
float64. Duplicate entries are summed, as scipy does.

The matrix is read in two streaming passes, so its size is not a limit: the
first checks the whole folder -- barcode and feature counts against the
matrix, every entry against its size -- before any output exists, and the
second writes X. Files in barcode order, as Cell Ranger and STARsolo write
them, are written straight through; others are sorted through scratch space
beside the output.

Exporting needs `--to 10x-mtx`, since a folder has no extension to say so,
and writes a Cell Ranger 3 folder: gzipped (`--compression none` for plain
files), three feature columns, entries in barcode order, from X or the
matrix `--layer` names. Feature ids, names and types come from var as for
the [HDF5 export](10x-h5.md), with the same `--force` placeholders. Names
that the TSV readers would not read back are refused: one holding a tab or a
line break, which would break the file; one starting with a double quote,
which pandas (and so scanpy) and R take for a quoted field; and an empty
barcode, whose blank line they skip.

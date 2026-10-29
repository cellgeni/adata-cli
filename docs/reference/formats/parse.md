---
title: Parse Biosciences
parent: Formats
grand_parent: Reference
nav_order: 6
---

# Parse Biosciences split-pipe

split-pipe writes, for the combined run and for each sample it was given,
`DGE_filtered/` (cells) and `DGE_unfiltered/` (every barcode), each holding
`count_matrix.mtx` (cells x genes), `all_genes.csv` and
`cell_metadata.csv`, each plain or gzipped:

```
run/all-sample/DGE_filtered/
run/all-sample/DGE_unfiltered/
run/<sample>/DGE_filtered/ ...
```

This is the layout split-pipe 1.6.1 writes, which the import was checked
against. A run laid out otherwise, as older versions' may be, is not
recognised.

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
`--make-unique` apply as for [10x input](10x-h5.md).

`--genome` works on runs made with a reference of several genomes built by
split-pipe, where each gene and cell names its own. A combined reference
built for Cell Ranger (`GRCh38-mm10-2020A`) is one genome to split-pipe, so
every gene and cell names it, and the species shows only in the gene ids'
prefixes.

Both files are read in two streaming passes, as for [10x MTX folders](10x-mtx.md), so a
run's size is not a limit: the first checks the matrix against both tables
before any output exists, and the second writes obs and X.

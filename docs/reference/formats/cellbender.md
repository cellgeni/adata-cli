---
title: CellBender
parent: Formats
grand_parent: Reference
nav_order: 2
---

# CellBender

The output of CellBender's `remove-background` (`output.h5` and
`output_filtered.h5`, 0.2 and later) is recognised by its latent variables,
whatever its name. Its matrix is a 10x one and is read as a
[10x HDF5 file](10x-h5.md) is; the latents
are kept, under the names CellBender's own loaders give them, so code written
against those keeps working. Those loaders name features by symbol; pass
`--var-names gene_symbols` for the same var names.

| CellBender | AnnData |
|---|---|
| denoised counts | `X` |
| per-droplet latents (`cell_probability`, `cell_size`, `background_fraction`, `droplet_efficiency`; `latent_*` in 0.2) | obs columns, NaN for droplets CellBender did not analyse |
| which droplets were analysed | `obs["cellbender_analyzed"]` |
| `gene_expression_encoding` (`latent_gene_encoding` in 0.2) | obsm, same name |
| `ambient_expression` (and, 0.3+, any other per-feature global) | `var["ambient_expression"]` |
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

The full `output.h5` keeps every droplet, as `read_10x_h5` would. A
`--min-cell-probability` that keeps no droplet is refused, with the highest
probability the file holds. `--raw` matches barcodes, so it works for
`output_filtered.h5` too, whose cells CellBender orders by count; the raw
file must be a 10x HDF5 matrix listing the same features in the same order,
and every output barcode, or it is refused. `layers["cellranger"]` is laid
out as `X` is: the same index dtypes, chunks and compression. The
[10x HDF5 flags](10x-h5.md) apply as well. `--from 10x-h5` reads a
CellBender file as a plain 10x matrix, without the latents; `--from
cellbender` refuses a file with no latents, which is not CellBender output.

In 0.2, the latents sit in the matrix group beside the loss per epoch and
other fitted values; only `ambient_expression` goes to var, and the rest to
`uns["cellbender"]` whatever their length.

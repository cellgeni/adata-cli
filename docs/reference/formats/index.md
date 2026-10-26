---
title: Formats
parent: Reference
nav_order: 2
has_children: true
---

# Formats

What `adata convert` reads and writes besides `.h5ad` and `.zarr`. Each page
gives the format's mapping to AnnData, its own flags, and how it streams. The
flags every conversion takes (`--output`, `--from`, `--to`, `--compression`,
`--overwrite`, `--force`) are under [`convert`](../commands.md#convert), and
[Convert formats](../../guides/convert.md) says which to use when.

| Format | `--from` / `--to` | Read | Write |
|---|---|---|---|
| [10x Genomics HDF5](10x-h5.md) | `10x-h5` | Cell Ranger 2 and 3+; the matrix alone of Space Ranger and Xenium | Cell Ranger 3+ layout |
| [CellBender](cellbender.md) | `cellbender` | `remove-background` 0.2 and later, with its latents | no |
| [10x and STARsolo MTX folders](10x-mtx.md) | `10x-mtx` | Cell Ranger 2 and 3+, STARsolo (velocyto too), GEO prefixes | Cell Ranger 3+ layout, gzipped |
| [Visium (Space Ranger)](visium.md) | `visium` | an `outs/` folder: counts, spot or bin positions, tissue images, scale factors (Space Ranger 1–4; Visium HD needs `pyadata-cli[spatial]`) | no |
| [Xenium](xenium.md) | `xenium` | an output bundle: counts, cells and centroids, Xenium's clusters, UMAP and PCA | no |
| [Parse Biosciences](parse.md) | `parse` | split-pipe: a run, a sample or a `DGE_*` folder | no |
| [loom and SCope](loom.md) | `loom` | loompy, velocyto, kallisto\|bustools, SCope; spec 2 and 3 | spec 3.0.0; a SCope loom with `--scope` |
| [Text tables](text.md) | `text` | a dense matrix in `.csv`, `.tsv`, `.txt` and the like, gzipped too | no |
| [HDF5 matrices](hdf5.md) | `hdf5` | one 2-D dataset of a plain HDF5 file | no |
| [pycisTopic pickles](cistopic.md) | `cistopic` | `CistopicObject`, `CistopicImputedFeatures` (needs `pyadata-cli[pickle]`) | no |
| [MuData](mudata.md) | `mudata` | each modality of an `.h5mu` or MuData Zarr store | `.h5mu`, var split by feature type |

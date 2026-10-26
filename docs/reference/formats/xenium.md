---
title: Xenium
parent: Formats
grand_parent: Reference
nav_order: 5
---

# Xenium

A Xenium Onboard Analysis output bundle (XOA 1 to 4): the cell x feature
counts, the table of cells with their centroids, and the clustering, UMAP
and PCA Xenium computed. Give the bundle folder; it is recognised by
`experiment.xenium` beside `cell_feature_matrix.h5`.

```bash
adata convert output-XETG00000__0001234__Region_1 -o region1.h5ad
adata convert xenium_bundle -o region1.h5ad --feature-type "Gene Expression"
```

| Xenium | AnnData |
|---|---|
| `cell_feature_matrix.h5` | `X`, var, as a [10x HDF5 file](10x-h5.md) is read: Gene Expression and the control probes and codewords, each a feature type |
| `cells.csv.gz`: `cell_id` | the obs names (the matrix's barcodes) |
| `cells.csv.gz`: `x_centroid`, `y_centroid` | `obsm["spatial"]`, `[x, y]` in microns |
| `cells.csv.gz`: every other column (`transcript_counts`, `cell_area`, `nucleus_area`, `segmentation_method`, ...) | obs columns, typed as pandas reads them; text as a categorical when it repeats a value, as anndata writes it |
| `analysis/clustering/gene_expression_graphclust/clusters.csv` | `obs["graphclust"]`, a categorical |
| `analysis/umap/gene_expression_2_components/projection.csv` | `obsm["X_umap"]` |
| `analysis/pca/gene_expression_10_components/projection.csv` | `obsm["X_pca"]` |
| the matrix's run metadata | `uns["cellranger"]` |

Xenium ships `analysis/` packed as `analysis.tar.gz`. When it has not been
unpacked, the three tables above are copied out of it, by name, into a
temporary folder beside the output, removed when the conversion ends. A cell an
analysis table lacks is left empty (no cluster, NaN), with a warning
giving the count.

The [10x HDF5 flags](10x-h5.md) (`--var-names`, `--make-unique`,
`--feature-type`, `--split-feature-types`) apply. Cell boundaries,
transcripts and the morphology images are not read.

## Differences from spatialdata-io

spatialdata-io's `xenium` reader returns a SpatialData, whose table holds
the same cells and `obsm["spatial"]`. Its obs also has `cell_id` as a
column, which here is the obs names, and a `region` column naming the
SpatialData element the table annotates, which an AnnData on its own has
no use for. CI checks the import against 10x's own recipe for scanpy:
`read_10x_h5`, the cell table reindexed to it, the analysis tables.

## How it is read

The cell table can hold a million rows. It is matched to the matrix's
barcodes while it is scanned, and Xenium writes the two in the same order,
so it is then written straight through, a block at a time. A table in
another order is written in the matrix's order through scratch space beside
the output; cells the matrix lacks are left out, with a warning. A cell of
the matrix with no row in the table is refused before anything is written.

A bundle with `cells.parquet` and no `cells.csv.gz` is refused: Parquet is
not read yet.

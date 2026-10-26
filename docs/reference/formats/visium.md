---
title: Visium (Space Ranger)
parent: Formats
grand_parent: Reference
nav_order: 4
---

# Visium (Space Ranger)

A Space Ranger output folder for standard Visium (55 µm spots), from Space
Ranger 1 to 4: the counts, where each spot sits on the slide, the tissue
images and the scale factors between them. The result is laid out as
`squidpy.read.visium` lays it out, so scanpy's and squidpy's spatial plots
work on it, and CI checks it against squidpy.

Give the `outs/` folder, or the run folder holding it. It is recognised by
`spatial/` (the positions and `scalefactors_json.json`) beside a
`*_feature_bc_matrix.h5`, including one named with its sample, as 10x's
downloads are (`V1_Mouse_Brain_filtered_feature_bc_matrix.h5`).

```bash
adata convert outs -o section.h5ad
adata convert outs -o every_spot.h5ad --unfiltered
adata convert outs -o section.h5ad --library-id brain_A1 --no-images
adata convert outs/filtered_feature_bc_matrix.h5 -o section.h5ad --from visium
```

| Space Ranger | AnnData |
|---|---|
| `filtered_feature_bc_matrix.h5` (`raw_` with `--unfiltered`) | `X`, var, as a [10x HDF5 file](10x-h5.md) is read |
| `spatial/tissue_positions.csv` (`tissue_positions_list.csv`, no header, in Space Ranger 1): `in_tissue`, `array_row`, `array_col` | obs columns of those names, int64 |
| the same: `pxl_col_in_fullres`, `pxl_row_in_fullres` | `obsm["spatial"]`, `[x, y]` in full-resolution pixels; int64, or float64 when the file has decimals |
| `spatial/tissue_hires_image.png`, `tissue_lowres_image.png` | `uns["spatial"][library]["images"]["hires"]`, `["lowres"]`: the PNGs' pixels, uint8 |
| `spatial/scalefactors_json.json` | `uns["spatial"][library]["scalefactors"]`, as it is |
| the matrix's `chemistry_description`, `software_version` | `uns["spatial"][library]["metadata"]` |
| the matrix's other run metadata | `uns["cellranger"]`, so an export to `.h5` keeps it |

| Flag | Meaning |
|---|---|
| `--unfiltered` | Read `raw_feature_bc_matrix.h5`: every spot on the slide, not only those under tissue |
| `--library-id` | The key under `uns["spatial"]` (default: the matrix's library id, as scanpy and squidpy use) |
| `--no-images` | Leave the images out; the scale factors and metadata stay |

The [10x HDF5 flags](10x-h5.md) (`--genome`, `--var-names`,
`--make-unique`, `--feature-type`, `--split-feature-types`) apply as well;
with `--split-feature-types` every output gets the positions and images.

## Differences from squidpy

- Features are named by gene id, with the symbols in `var["gene_symbols"]`,
  and every feature type is kept; squidpy names them by symbol and keeps
  Gene Expression only. `--var-names gene_symbols --feature-type "Gene
  Expression"` gives squidpy's.
- `X` keeps Space Ranger's integer counts, where squidpy (through scanpy)
  makes them float32.

## How it is read

obs follows the matrix's barcode order. The positions list every spot on
the slide, in their own order, so they are matched to the matrix's barcodes
as they are scanned and written in its order, a block at a time; they are a
few thousand rows (some 14,000 on an 11 mm slide) however deep the
sequencing. A barcode with no position, or two, is refused before anything
is written. The images are read whole: a hires PNG is 2,000 pixels on its
longest side.

Refused, with the reason:

- output of `spaceranger aggr`, whose matrix holds several libraries;
- Visium HD (`binned_outputs/`), whose positions are Parquet and are not
  read yet. Each bin's `filtered_feature_bc_matrix.h5` converts as a
  [10x HDF5 file](10x-h5.md) meanwhile;
- a folder with the matrix as Matrix Market only, with no `.h5`.

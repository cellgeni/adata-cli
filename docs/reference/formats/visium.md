---
title: Visium (Space Ranger)
parent: Formats
grand_parent: Reference
nav_order: 4
---

# Visium (Space Ranger)

A Space Ranger output folder, for standard Visium (55 µm spots, Space Ranger
1 to 4) and for [Visium HD](#visium-hd) (2, 8 and 16 µm bins, Space Ranger 3
and 4, and [segmented cells](#segmented-cells), Space Ranger 4): the counts, where each spot or bin sits on the slide, the tissue
images and the scale factors between them. The result is laid out as
`squidpy.read.visium` lays it out, so scanpy's and squidpy's spatial plots
work on it. CI checks standard Visium against squidpy, and Visium HD, which
squidpy does not read, against scanpy with pandas reading the positions and
geopandas the cells' outlines.

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
- a folder with the matrix as Matrix Market only, with no `.h5`.

## Visium HD

Space Ranger 3 and 4 write a Visium HD run's counts at three bin sizes, each
in its own folder under `binned_outputs/` (`square_002um`, `square_008um`,
`square_016um`), laid out as a standard Visium `outs/` but with the
positions as `spatial/tissue_positions.parquet`. Give the `outs/` folder and
`--bin-size` picks the bins, 8 µm by default as 10x recommends for
analysis; or give a bin folder itself.

```bash
pip install 'pyadata-cli[spatial]'    # pyarrow, which reads the Parquet positions

adata convert outs -o bins_8um.h5ad
adata convert outs -o bins_2um.h5ad --bin-size 2
adata convert outs/binned_outputs/square_016um -o bins_16um.h5ad
```

| Flag | Meaning |
|---|---|
| `--bin-size` | The bins to read, in µm (default 8; refused for standard Visium) |

The result is laid out as above: `in_tissue`, `array_row` and `array_col` in
obs, `obsm["spatial"]` in full-resolution pixels (float64, as Space Ranger
records them), and the bin folder's own images and scale factors, which
carry `bin_size_um` and `microns_per_pixel`, in `uns["spatial"]`.

The positions list every bin on the slide, about 11 million at 2 µm, where
the filtered matrix holds the bins under tissue. They are read twice, a
buffered stream of only the columns needed: the barcodes, matched to the
matrix's as they come, then the numbers, written in the matrix's order. On
10x's Visium HD tiny sample, the 2 µm bins (330,035 in the matrix) convert in
3.5 seconds with a peak of 470 MB with `--no-images`, and 610 MB with the
100 MB hires image; pandas alone takes about 1 GB to read the positions. `--no-images` leaves the
images out.

### Segmented cells

Space Ranger 4 also segments a Visium HD run into cells, in
`outs/segmented_outputs/`: a cells x features matrix
(`filtered_feature_cell_matrix.h5`, barcodes like `cellid_000000001-1`),
each cell's outline and its nucleus's as GeoJSON polygons in
full-resolution pixels, the images and scale factors, and Space Ranger's own
`analysis/`. Give the `outs/` folder with `--cells`, or the
`segmented_outputs/` folder itself. Without `--cells` an HD `outs/` is still
read as 8 µm bins, and the summary says the cells are there too.

```bash
adata convert outs -o cells.h5ad --cells
adata convert outs/segmented_outputs -o cells.h5ad
adata convert outs -o every_cell.h5ad --cells --unfiltered
```

| Space Ranger | AnnData |
|---|---|
| `filtered_feature_cell_matrix.h5` (`raw_` with `--unfiltered`) | `X`, var, as a [10x HDF5 file](10x-h5.md) is read |
| `cell_segmentations.geojson` | `obsm["spatial"]`: each outline's centroid, `[x, y]` in full-resolution pixels, weighted by area as shapely's is; `obs["cell_area"]`, in µm² |
| `nucleus_segmentations.geojson` | `obs["nucleus_area"]`, in µm²; NaN for a cell with no nucleus |
| `analysis/clustering/gene_expression_graphclust/clusters.csv` | `obs["graphclust"]`, categorical |
| `analysis/umap/…/projection.csv`, `analysis/pca/…/projection.csv` | `obsm["X_umap"]`, `obsm["X_pca"]` |
| `spatial/` images, scale factors; the matrix's metadata | `uns["spatial"][library]`, as above |

| Flag | Meaning |
|---|---|
| `--cells` | Read the segmented cells, not bins (refused with `--bin-size`, and for an output with no `segmented_outputs/`; a cell matrix given itself is read as cells without it) |

Areas are the polygons' areas in pixels times `microns_per_pixel` squared,
the unit and names Xenium uses. The outlines themselves are not kept: an
AnnData has no place squidpy or scanpy would read them from. Nor are
`barcode_mappings.parquet` (which 2 µm bins fall in which cell) and the
`graphclust_annotated_*.geojson` copies of the outlines.

The cells' scale factors, kept as Space Ranger wrote them, carry no
`spot_diameter_fullres`, so plotting needs a size: scanpy takes one,
`sc.pl.spatial(adata, color="graphclust", spot_size=10)`, but squidpy's
`spatial_scatter` looks it up in the scale factors whatever else it is
given, so set one first, in pixels:
`adata.uns["spatial"][library]["scalefactors"]["spot_diameter_fullres"] = 10`.

The GeoJSON is one line of some hundreds of bytes a cell, a few hundred MB
for a full slide, which `json.load` and geopandas hold whole. It is read
here a feature at a time; what is kept is each cell's centroid and area, its
nucleus's area and where each goes in the matrix, as a spot's position is:
about 50 bytes a cell, some 35 MB for 700,000 cells, so that a cell with no
outline is refused before anything is written. Polygons whose cell the matrix
lacks, the background the filtered matrix leaves out, are passed over.

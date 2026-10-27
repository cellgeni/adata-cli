# Status

Working notes on what is in progress and what comes next. Updated as work
moves; the plan of record for the current item is in the section below.

## Released

| Version | What | Issues |
|---|---|---|
| 0.17.1 | Visium HD, one bin size at a time, with the new `spatial` extra (pyarrow) | #81 |
| 0.17.0 | `convert` reads Visium (Space Ranger 1–4) and Xenium (XOA 1–4) with positions, images and Xenium's analysis | #79 |
| 0.16.2 | Documentation site (guides, troubleshooting, a page per format); `export image` writes PNG to stdout without `-o` | #76, #77 |
| 0.16.1 | `subset`, `split` and the MuData export narrow categoricals (and their colours) as anndata's subsetting does | #74 |
| 0.16.0 | MuData `.h5mu` read (each modality, global parts on its cells) and written (split by feature type); copies of anndata 0.7 dataframes readable | #31, #72 |
| 0.15.0 | `convert --scope` writes SCope looms; `concat` stacks obsm dataframes and merges varm; loom attribute-name clashes resolved before writing | #61, #69, #70 |
| 0.14.0 | pycisTopic pickles (optional `pickle` extra) | #28 |
| 0.13.0 | Delimited text tables and plain HDF5 matrices | #25 |
| 0.12.0 | SCope loom import | #27 |

The conversion epic's sub-issues (#21–#30) are all done and ticked in #20.

## In progress

### Visium HD segmented cells (Space Ranger 4), for 0.17.2

`adata convert outs --cells` (or `segmented_outputs/` given itself) reads
the cell matrix, each cell's centroid into `obsm["spatial"]`, `cell_area`
and `nucleus_area` in µm² into obs, and the cells' `analysis/` (graphclust,
UMAP, PCA). Without `--cells` an HD output is still read as 8 µm bins.

Decisions:
- Centroids weighted by area (shapely's), in full-resolution pixels, so
  squidpy plots them over the images; areas in µm², Xenium's names. The
  outlines are not kept.
- The GeoJSON (one line, a few hundred MB on a full slide) is read a
  feature at a time (`formats/geojson.py`); each polygon's three numbers are
  kept from the one read, so a cell with no outline is refused in the plan.
- `analysis/` reading moved from `xenium.py` to `spatial.py` and is shared.
- Spot and bin `analysis/`, and `cell_id` on 2 µm bins, stay follow-ups.

Steps:
- [x] `formats/geojson.py`, checked against shapely (and `json.load`).
- [x] Analysis reading shared in `spatial.py`; Xenium unchanged.
- [x] Cells in `visium.py`, the `segmented_outputs/` fixture, tests.
- [x] `--cells`; perf guards; geopandas comparison; `import-visium-cells`
  benchmark case.
- [x] Docs (Visium page, commands, convert guide, troubleshooting).
- [ ] PR, independent review, CI green; then release 0.17.2.

## Follow-ups noticed

- `cell_id` (with `in_cell`, `in_nucleus`) on 2 µm bins, from Space Ranger
  4's `barcode_mappings.parquet`.
- Spatial bundles whose matrix is a MEX folder only (refused for now, with a
  pointer to `--from 10x-mtx`).
- Space Ranger's own `analysis/` (clusters, UMAP) for Visium spots and HD
  bins, as the segmented cells now have it (`spatial.scan_analysis`).
- CellBender's `_write_embeddings` onto `core/scatter.py`, which generalises
  it.

- Fixed in 0.16.2 (#77): `export image` without `-o` crashed (`'NoneType'
  object has no attribute 'parent'`); it now writes PNG to stdout, as every
  other export writes there. Found by the docs review.
- Fixed in 0.16.1 (#74): `subset`, `split` and the MuData export kept
  every category of a categorical column, where anndata's subsetting keeps
  only those used (and narrows `uns["<col>_colors"]`). The ~50% "slack"
  noted here before was mostly a measuring error -- `get_storage_size()`
  leaves out the variable-length string heap -- but the unpruned categories
  were real: the pbmc3k Multiome `.h5mu` drops from 84.7 to 74.2 MB, and a
  subset's obs and var now equal anndata's own subset.

## Next candidates (#31)

- alevin-fry USA mode, kallisto | bustools
- Seurat `.h5Seurat`

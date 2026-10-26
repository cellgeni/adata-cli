# Status

Working notes on what is in progress and what comes next. Updated as work
moves; the plan of record for the current item is in the section below.

## Released

| Version | What | Issues |
|---|---|---|
| 0.16.2 | Documentation site (guides, troubleshooting, a page per format); `export image` writes PNG to stdout without `-o` | #76, #77 |
| 0.16.1 | `subset`, `split` and the MuData export narrow categoricals (and their colours) as anndata's subsetting does | #74 |
| 0.16.0 | MuData `.h5mu` read (each modality, global parts on its cells) and written (split by feature type); copies of anndata 0.7 dataframes readable | #31, #72 |
| 0.15.0 | `convert --scope` writes SCope looms; `concat` stacks obsm dataframes and merges varm; loom attribute-name clashes resolved before writing | #61, #69, #70 |
| 0.14.0 | pycisTopic pickles (optional `pickle` extra) | #28 |
| 0.13.0 | Delimited text tables and plain HDF5 matrices | #25 |
| 0.12.0 | SCope loom import | #27 |

The conversion epic's sub-issues (#21–#30) are all done and ticked in #20.

## In progress

### Spatial import: standard Visium and Xenium (`feat/spatial`, for 0.17.0)

Decisions:
- Standard Visium (Space Ranger 1–4) and Xenium (XOA 1–4) first, with no new
  dependency: both come as CSV, JSON and PNG. Visium HD, whose positions are
  Parquet only, waits for a `spatial` extra with pyarrow.
- Laid out as squidpy (Visium) and spatialdata-io's table (Xenium) lay
  them out; images stay the PNGs' uint8 pixels. Xenium brings its own
  clusters, UMAP and PCA, read from `analysis.tar.gz` as it ships.
- A table beside the matrix is matched to its barcodes while scanned
  (`conversion/spatial.py`), and written straight through when in the
  matrix's order (Xenium's cells, on every real bundle checked) or through
  `core/scatter.py` when not.

Steps:
- [x] `table.py`: headerless files, `on_index`, reusable columns
- [x] `core/scatter.py`
- [x] Visium: `visium.py`, flags `--library-id`, `--no-images`, `--unfiltered`
- [x] Xenium: `xenium.py`, analysis from the folder or the archive
- [x] Performance guards; squidpy and 10x-recipe comparisons; `import-xenium` benchmark
- [x] Real data by hand: Visium SR 1.1 and SR 2.1, Xenium XOA 3.0 and 4.0 tiny
- [x] Docs: two format pages, tables, guide, Troubleshooting, Developers
- [ ] PR, review, merge; release 0.17.0

## Follow-ups noticed

- Visium HD bins and Space Ranger 4's cell segmentation, behind a `spatial`
  extra (pyarrow) for their Parquet positions.
- Spatial bundles whose matrix is a MEX folder only (refused for now, with a
  pointer to `--from 10x-mtx`).
- Space Ranger's own `analysis/` (clusters, UMAP) for Visium, as Xenium's.
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

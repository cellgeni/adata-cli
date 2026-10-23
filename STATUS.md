# Status

Working notes on what is in progress and what comes next. Updated as work
moves; the plan of record for the current item is in the section below.

## Released

| Version | What | Issues |
|---|---|---|
| 0.16.0 | MuData `.h5mu` read (each modality, global parts on its cells) and written (split by feature type); copies of anndata 0.7 dataframes readable | #31, #72 |
| 0.15.0 | `convert --scope` writes SCope looms; `concat` stacks obsm dataframes and merges varm; loom attribute-name clashes resolved before writing | #61, #69, #70 |
| 0.14.0 | pycisTopic pickles (optional `pickle` extra) | #28 |
| 0.13.0 | Delimited text tables and plain HDF5 matrices | #25 |
| 0.12.0 | SCope loom import | #27 |

The conversion epic's sub-issues (#21–#30) are all done and ticked in #20.

## In progress

Nothing. #20's sub-issues are all ticked; #31 stays open for the formats below.

## Follow-ups noticed

- Fixed (`fix/subset-categories`): `subset`, `split` and the MuData export kept
  every category of a categorical column, where anndata's subsetting keeps
  only those used (and narrows `uns["<col>_colors"]`). The ~50% "slack"
  noted here before was mostly a measuring error -- `get_storage_size()`
  leaves out the variable-length string heap -- but the unpruned categories
  were real: the pbmc3k Multiome `.h5mu` drops from 84.7 to 74.2 MB, and a
  subset's obs and var now equal anndata's own subset.

## Next candidates (#31)

- Spatial: Visium / Xenium (matrix via 10x h5, coordinates to
  `obsm["spatial"]`)
- alevin-fry USA mode, kallisto | bustools
- Seurat `.h5Seurat`

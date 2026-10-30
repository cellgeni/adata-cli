# Status

Working notes on what is in progress and what comes next. Updated as work
moves; the plan of record for the current item is in the section below.

## Released

| Version | What | Issues |
|---|---|---|
| 0.17.4 | Audit fixes, wave 1 (data safety): every writer staged and checked, close errors fail the command, `--chunk` at least 1, a free-space check; text/MTX/dense imports refuse what they cannot read; concat refuses repeated var names and keeps missing values and dtypes; import/export type values as pandas does; Pillow 13 | #89, #90, #91, #92, #98 |
| 0.17.3 | Space Ranger's analysis (clusters, UMAP, PCA) for Visium spots and HD bins; `convert` works on fresh installs (broken since 0.7.0) | #85, #88 |
| 0.17.2 | Visium HD segmented cells (Space Ranger 4): centroids, areas and Space Ranger's analysis | #83 |
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

The fixes from the October 2026 audit, epic #87: 164 findings in 16
batches, one PR and one sub-issue each. Batches in a wave own disjoint files
and run in parallel; a batch that shares files with another starts once that
one has merged.

| Wave | Batches | Release |
|---|---|---|
| 0 | B0 #88 | 0.17.3 |
| 1 | B1 #89, B2a #90, B2b #91, B3a #92, B7 #98 | 0.17.4: data safety |
| 2 | B3b #93, B4 #94, B5 #95, B8a #99, B8b #100, B9 #102 | 0.17.5: fidelity |
| 3 | B6a #96, B6b #97, B8c #101 | 0.17.6: streaming |
| 4 | B10 #103 | 0.17.6 |

Decisions:
- Every batch ships as a 0.17.x patch.
- `concat` orders an outer join's var and merged categories as anndata does
  (sorted), and the changelog says so (B3b).
- Pickle reading is hardened, and the docs say pickles are trusted input
  (B8c).
- A full disk aborts inside HDF5, beyond the tool's reach, so writers check
  free space against the output's estimated size first (B1).
- Each PR carries its changelog entry in its description, and the release PR
  collects them, so parallel batches do not conflict in `docs/changelog.md`.

Steps:
- [x] B0: `convert` on fresh installs (#88), in 0.17.3
- [x] Wave 1, then 0.17.4
- [ ] Wave 2, then 0.17.5
- [ ] Waves 3 and 4, then 0.17.6

## Follow-ups noticed

- From wave 1 of the audit fixes:
  - `core/csr_build.py`'s on-disk sort makes many small appends, slow for
    a Zarr scratch at small budgets; buffer bucket writes in
    `BucketSorter` (B2a).
  - The dense tile path's temporaries run to about 10-17x the budget, a
    constant factor rather than growth with the input (B2a).
  - `import array obs/<new>` doesn't add the column to `column-order`, so
    anndata doesn't show it (pre-existing, B2b).
  - The free-space estimate runs low for a var-only subset, a dense loom
    output and the transpose's scratch: no false refusals, a weaker check
    (B1).
  - The floors job lets indirect dependencies float, so a new upstream
    release can turn it red, and publish.yml reuses tests.yml (B7).
- `cell_id` (with `in_cell`, `in_nucleus`) on 2 µm bins, from Space Ranger
  4's `barcode_mappings.parquet` (a 0.17.x after the audit fixes).
- Dropped: spatial bundles whose matrix is a MEX folder only. 10x always
  writes the `.h5` beside it, and the refusal points to `--from 10x-mtx`.

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

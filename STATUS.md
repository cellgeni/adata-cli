# Status

Working notes on what is in progress and what comes next. Updated as work
moves; the plan of record for the current item is in the section below.

## Released

| Version | What | Issues |
|---|---|---|
| 0.17.6 | Audit fixes, waves 3 and 4 (streaming): reads and writes follow the source's chunks and Zarr v3 shards, each shard written once; `recast` checks and converts in `--chunk` blocks; `split`'s memory no longer grows with its groups; pickle arrays checked before use; 16-bit images export as they are; plain-line logs and names printed as stored; h5py >= 3.16 | #96, #97, #101, #103 |
| 0.17.5 | Audit fixes, wave 2 (fidelity): scanpy's marker genes and anndata 0.7 files survive every copy, data held outside an h5ad refused; subset/split keep every named row and subset every element as anndata; concat takes any number of inputs in anndata's order; 10x, MTX, Parse, CellBender, loom, SCope and MuData fixes; release pipeline hardened, image non-root; zarr >= 3.2 | #93, #94, #95, #99, #100, #102 |
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

Nothing. The fixes from the October 2026 audit (epic #87, 164 findings in
16 batches) finished with 0.17.6; the sub-issues (#88–#103) are all done
and ticked in #87. The batch names in the follow-ups below (B1 to B10) are
theirs.

## Follow-ups noticed

- From waves 3 and 4 of the audit fixes:
  - The streaming transpose and other users of `csr_build.BucketSorter`
    append every input block to every bucket, (nnz/chunk)² appends, and
    each bucket holds three growable datasets (1.8 GB at 30,000 buckets).
    The transpose knows its row sizes, so fixed regions as in concat's
    `_ColumnBuckets` would do; concat's `_csc_runs` and recast's
    `_bucket_runs` could then move into `csr_build` (B3b, B6a).
  - The transpose's scratch lives inside a Zarr output and is appended to
    unbuffered, about 135 times its writes; an HDF5 scratch beside the
    output, as concat has, would fix it. It is also missing from recast's
    free-space estimate (B3b, B6a, B6b).
  - recast's sparse reads are a fixed million entries, not `--chunk`; and
    `concat`, `recast` and `export` read `--chunk` rows where one chunk
    height is over the read budget, with column tiles only in `subset`,
    `split` and `convert` (B6a, B6b).
  - `copy_dataset` writes HDF5 text uncompressed: a Zarr to h5ad
    conversion's obs index is about nine times anndata's (B6b).
  - `convert` has no `--chunk`: the cisTopic and MuData converters use a
    fixed byte budget (B8b, B8c).
  - A pickle holding pandas 3's Arrow strings is refused; pycisTopic's
    pandas 1.5 never writes one (B8c).
  - Python warnings reach stderr with their source line; printing them as
    `Warning: ...` lines would match the rest of the output (B5).
  - CI's `external-readers` job is named "10x h5 against scanpy and
    PyTables" and its `compatibility` job "anndata 0.8-0.13", though the
    first checks every converter's reader and the second reads 0.7.8 too.
    Renaming a job changes its required-check name, so it waits for the
    repository settings (B10). `test_concat_versions.py` is an
    `integration` test no CI job runs (B10).
  - `test_write_safety.py::test_export_writes_to_devices_and_pipes_as_they_are`
    failed once in a dozen local runs on macOS: its FIFO reader stayed
    blocked with no writer left. The reader thread is not a daemon, so it
    also kept pytest from exiting; a daemon thread would at least let the
    run end (B10).
- From wave 2 of the audit fixes:
  - CI: a Python 3.14 leg; Dependabot for the pinned actions, uv and base
    image (the pinned actions run on Node 20, which GitHub is retiring); the
    base image to Debian trixie; a Codecov token (B9).
  - Repository settings: the default workflow token can write, and the
    `pypi` environment has no protection rules (B9).
  - Zarr stores need a counterpart to the HDF5 hardening in 0.17.5, outside
    that fix's scope (B5).
  - Flex's `target_sets` group is not carried, and `--make-unique` names
    export with their suffixes (B8a).
  - `formats/textio.py` does not unquote a field that starts with `"`, as
    pandas does; if it did, MTX export could quote such names rather than
    refuse them (B8a).
  - A fixed-width text dataset that HDF5 compressed can inflate far beyond
    its stored size when read whole, as the loom import reads var columns;
    only loom's global attributes are bounded (B8b).
  - Under an outer join, an input lacking some genes counts as lacking var
    columns, varm and varp entirely; anndata pads them (B3b).
  - A numeric `in` list with thousands of values is slow (B4).
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

## Next candidates (#31)

- alevin-fry USA mode, kallisto | bustools
- Seurat `.h5Seurat`

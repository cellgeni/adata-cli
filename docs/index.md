---
title: Home
nav_order: 1
permalink: /
---

# adata-cli

`adata` explores and edits AnnData stores (`.h5ad` and `.zarr`) from the
command line, without loading them into memory and without a Python session.
It inspects, filters, splits, concatenates, recasts and converts them, and
moves single elements in and out as CSV, JSON, `.npy`, `.mtx` or images.

It implements the AnnData on-disk spec directly against h5py and zarr, with no
dependency on anndata, pandas or scipy. Commands stream: memory follows
`--chunk` rather than the size of the store, so stores larger than RAM are
fine (the [benchmarks](reference/benchmarks.md) show how closely, and where
loading the whole file is faster). The tool is small enough to drop into a
container or a pipeline step where the scientific stack would be overkill.

```
.h5ad / .zarr ──► view, ls ──► export (CSV, JSON, npy, mtx, image) ──► your tools
      │                                                                    │
      ├──► subset, split, concat, recast ──► .h5ad / .zarr                 │
      └──► convert ◄──► 10x, MTX, loom, MuData, ...        import ◄────────┘
```

## Install

```bash
pip install pyadata-cli
pip install 'pyadata-cli[pickle]'   # also read pycisTopic pickles (adds pandas, scipy)
pip install 'pyadata-cli[spatial]'  # also read Visium HD (adds pyarrow)
```

The command is `adata`. The distribution is named `pyadata-cli` because
`adata-cli` was already taken on PyPI by an unrelated project. Python 3.12 or
later.

Or use the container, which has both extras and duckdb in it:

```bash
docker run --rm -v "$PWD":/data quay.io/cellgeni/adata-cli:latest \
    adata view /data/data.h5ad
```

## Quick start

```bash
adata view data.h5ad                   # what's in it
adata ls data.h5ad --long              # every path, with shapes and encodings
adata export dataframe data.h5ad obs   # obs as CSV, on stdout

adata subset data.h5ad -o cortex.h5ad --obs-query "cluster == Cortex_2"
adata split  data.h5ad --by sample -o per_sample/
adata concat per_sample/*.h5ad -o merged.h5ad --join outer --label sample

adata convert data.h5ad -o data.zarr
adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
adata recast  data.h5ad X -o small.h5ad --dtype float32
```

Status goes to stderr and data to stdout, so every command pipes cleanly.

## What it reads and writes

- **Every AnnData on-disk layout**, from anndata 0.7 to the current spec, in
  HDF5 and in Zarr v2 and v3; everything written is in the current spec. CI
  checks this against stores written by anndata 0.8 to 0.13, and checks that
  each of those releases reads what `adata` writes.
- **Other formats, through `convert`**: 10x Genomics HDF5 and MTX folders
  (Cell Ranger, STARsolo, velocyto), Visium and Xenium spatial outputs with
  their coordinates and images, CellBender output, Parse Biosciences runs,
  looms (velocyto's and SCope's), text tables, plain HDF5 matrices,
  pycisTopic pickles and MuData. CI checks them against the readers people
  use for them: scanpy, squidpy, loompy, anndata, mudata.
- **Outputs that keep the source's choices**: Zarr version, index dtypes,
  chunking and compression carry over unless a flag changes them.
- **Outputs that appear whole or not at all**: every store is written beside
  its destination and moved into place only when complete.

## Where next

| Page | Read it when |
|---|---|
| [Getting started](getting-started.md) | You are using `adata` for the first time |
| [Inspect and export](guides/inspect-export.md) | You want to see what a store holds, or get a table or matrix out of it |
| [Subset and split](guides/subset-split.md) | You want some of the cells or genes, or one store per sample |
| [Concatenate](guides/concat.md) | You have several stores to join into one |
| [Change dtype, layout or density](guides/recast.md) | A store is larger than it needs to be, or a tool wants CSC |
| [Convert formats](guides/convert.md) | Your data is in, or needs to be in, another format |
| [Pipelines and containers](guides/pipelines.md) | You are running `adata` in Nextflow, Docker or Singularity |
| [Commands](reference/commands.md) | You need every flag of a command |
| [Formats](reference/formats/index.md) | You need to know what each format maps to |
| [Troubleshooting](troubleshooting.md) | A command stopped with an error |
| [Developers](developers.md) | You are changing the code |
| [Changelog](changelog.md) | You want to know what changed between releases |

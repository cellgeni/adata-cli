---
title: Convert formats
parent: Guides
nav_order: 5
---

# Convert formats
{: .no_toc }

1. TOC
{:toc}

## AnnData is the hub

`convert` reads any format it knows into AnnData, and writes AnnData out to
any format it can write. There are no converters between two other formats:
to turn a loom into a 10x matrix, convert it to `.h5ad` first, then from that.

```
10x HDF5, 10x/STARsolo MTX ─┐                       ┌─► 10x HDF5, 10x MTX
Visium, Xenium, CellBender  │                       │
Parse, text, HDF5           ├─► .h5ad ◄──► .zarr ◄──┤
loom, pycisTopic            │                       ├─► loom (SCope too)
MuData ─────────────────────┘                       └─► MuData
```

```bash
adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
adata convert counts.h5ad -o counts.loom
```

The input's format is found from its content, since extensions lie: an `.h5`
can be a Cell Ranger matrix, CellBender output or a plain HDF5 file. The
output's extension picks the output format. `--from` and `--to` name either
when that is not enough, such as `--to 10x-mtx` for an MTX folder, which has
no extension.

## Which format, when

| You have | Convert with | Read the page |
|---|---|---|
| Cell Ranger's `*_feature_bc_matrix.h5` | `adata convert in.h5 -o out.h5ad` | [10x HDF5](../reference/formats/10x-h5.md) |
| A Space Ranger `outs/` folder (standard Visium) | `adata convert outs -o out.h5ad` | [Visium](../reference/formats/visium.md) |
| A Visium HD `outs/` folder | `adata convert outs -o out.h5ad --bin-size 8` (needs `pyadata-cli[spatial]`) | [Visium HD](../reference/formats/visium.md#visium-hd) |
| Space Ranger 4's segmented cells | `adata convert outs -o out.h5ad --cells` | [Segmented cells](../reference/formats/visium.md#segmented-cells) |
| A Xenium output bundle | `adata convert bundle -o out.h5ad` | [Xenium](../reference/formats/xenium.md) |
| CellBender `output.h5` | `adata convert output.h5 -o out.h5ad` | [CellBender](../reference/formats/cellbender.md) |
| A `matrix.mtx` folder from Cell Ranger, STARsolo or GEO | `adata convert folder -o out.h5ad` | [MTX folders](../reference/formats/10x-mtx.md) |
| A Parse Biosciences split-pipe run | `adata convert run -o out.h5ad` | [Parse](../reference/formats/parse.md) |
| A velocyto, loompy or SCope (pySCENIC) loom | `adata convert in.loom -o out.h5ad` | [loom](../reference/formats/loom.md) |
| A count table from GEO, R or pandas | `adata convert in.csv.gz -o out.h5ad` | [Text tables](../reference/formats/text.md) |
| A matrix in a plain HDF5 file | `adata convert in.hdf5 -o out.h5ad --key path` | [HDF5 matrices](../reference/formats/hdf5.md) |
| A pycisTopic `cistopic_obj.pkl` | `adata convert in.pkl -o out.h5ad` | [pycisTopic](../reference/formats/cistopic.md) |
| A muon or SCENIC+ `.h5mu` | `adata convert in.h5mu -o out.h5ad` | [MuData](../reference/formats/mudata.md) |

| You need | Convert with |
|---|---|
| A matrix for CellBender, SoupX, DropletUtils or Seurat | `adata convert in.h5ad -o out.h5 --layer counts` |
| An MTX folder for a tool that reads Cell Ranger output | `adata convert in.h5ad -o out_mtx --to 10x-mtx` |
| A loom for velocyto tools or the SCope viewer | `adata convert in.h5ad -o out.loom` (`--scope` for SCope) |
| A MuData, one modality per feature type | `adata convert in.h5ad -o out.h5mu` |
| Zarr, for cloud storage or Zarr-native tools | `adata convert in.h5ad -o out.zarr` |

## How it differs from scanpy's readers

{: .note }
The importers read what scanpy's and anndata's readers read, and CI checks
each against them, with a few deliberate differences, listed below.

- **Features are named by gene id**, where scanpy's 10x readers name them by
  symbol. Ids are unique and symbols repeat; `--var-names gene_symbols` gives
  scanpy's names.
- **Integer counts stay integers**, where scanpy makes them float32. `adata
  recast out.h5ad X --inplace --dtype float32` if a tool needs floats.
- **Nothing is dropped**: every feature type unless `--feature-type` picks
  one, the run's metadata in `uns`, a loom's graphs and global attributes.

## h5ad and Zarr

Between `.h5ad` and `.zarr` every element is copied as it is, streaming.
Chunking carries over, and so does compression where the target can express
it; `--compression` replaces it with one choice. A Zarr output keeps a Zarr
source's version and is v3 from an `.h5ad`:

```bash
adata convert data.h5ad -o data.zarr                     # Zarr v3
adata convert data.h5ad -o data.zarr --zarr-format 2     # for anndata before 0.12
adata convert data.zarr -o data.h5ad --compression gzip
```

`subset`, `split`, `concat` and `import` convert as they write too, when the
output's extension names the other backend.

## Safe by construction

Every check runs before the output exists: an unknown format, a flag that
does not apply to this pair of formats, a 10x export with no feature ids.
The output is written beside its destination and moved into place only when
complete, and an existing output is replaced only with `--overwrite`.
[Troubleshooting](../troubleshooting.md#before-anything-is-written) lists the
refusals.

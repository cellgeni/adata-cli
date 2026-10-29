---
title: loom and SCope
parent: Formats
grand_parent: Reference
nav_order: 7
---

# loom and SCope looms
{: .no_toc }

1. TOC
{:toc}

A loom holds a dense genes x cells matrix, layers of the same shape, an
attribute per cell (`col_attrs`) and per gene (`row_attrs`), graphs, and
global attributes. velocyto writes its counts this way, as do loompy and
kallisto|bustools, and pySCENIC, SCENIC+ and pycistopic their results for
the SCope viewer.

```bash
adata convert velocyto.loom -o velocyto.h5ad
adata convert velocyto.loom -o unspliced.h5ad --layer unspliced
adata convert pyscenic_output.loom -o scenic.h5ad
adata convert counts.loom -o dense.h5ad --dense --var-names gene_symbols
adata convert counts.h5ad -o counts.loom
```

| loom | AnnData |
|---|---|
| `/matrix`, or `/layers/spliced` when there is one and no layer is named `matrix` | `X`, CSR (`--dense` keeps it dense) |
| `/layers/<name>`, and `/matrix` when X is a layer | `layers[<name>]`, `layers["matrix"]` (`layers["matrix_1"]` when a layer is named `matrix`) |
| `col_attrs` with one value per cell / several | obs columns / obsm |
| `row_attrs`, the same way | var columns / varm |
| compound `col_attrs` and `row_attrs`, a field per column | dataframes in obsm and varm |
| `col_graphs`, `row_graphs` | obsp, varp |
| global attributes | `uns["loom"]["attrs"]` |

| Flag | Meaning |
|---|---|
| `--layer` | The loom layer that becomes X: `matrix` (the main one) or a layer name (default `spliced` when there is one and no layer is named `matrix`, else `matrix`) |
| `--dense` | Keep X and the layers dense |
| `--var-names` | `gene_ids` (default) to name genes by id, `gene_symbols` by symbol; each falls back to the other when the loom records only one |
| `--make-unique` | Suffix repeated gene names (`-1`, `-2`, ...) as anndata's `var_names_make_unique` does |
| `--obs-names-attr` | The attribute naming cells (default the first there of `CellID`, `cell_id`, `obs_names`, `barcode` and `barcodes`; without one, cells are numbered) |
| `--var-names-attr` | The attribute naming genes (default the first there of the ids, `Accession`, `gene_id` and `gene_ids`, then of the symbols, `Gene`, `gene_name`, `gene_names`, `gene_symbols` and `Name`; symbols first with `--var-names gene_symbols`; without one, genes are numbered) |

Reading gives what `anndata.read_loom` gives, checked against it in CI, with
four differences: values keep their dtype (velocyto's integer counts stay
integers, where anndata makes X float32); graphs and global attributes are
kept, where anndata drops them, and so is the main matrix: a loom with a
layer named `matrix` was written from an AnnData that had one, so its main
matrix, that AnnData's X, is X again, where anndata takes `spliced` and
lets the layer replace the main matrix; genes are named by id (`Accession`), the symbols (`Gene`)
staying a column, unless `--var-names gene_symbols` is given; and compound
attributes, which loompy's validator rejects, so that anndata refuses the
file, become dataframes (see [SCope looms](#scope-looms)). Strings are read as loompy reads them, with XML references such as
`&#233;` turned back into characters. Looms from other writers read too:
R's LoomExperiment (graph vertices stored as floats), Baysor (genes in a
`Name` attribute, and a `.loom` holding the matrix alone, which is
recognised by its name), and R's writers' factors and logicals, which
become categoricals and booleans rather than their codes.

The matrix is read a slab of cells at a time, a whole number of the file's
chunks along the cells (loompy writes 64 x 64 chunks), so each chunk is
decompressed once and memory is set by the slab, not the file. A loom
chunked otherwise is read as a [plain HDF5 matrix](hdf5.md) is: slabs as
wide as memory allows cut through a chunk that spans more cells than a slab
holds, and a chunk that spans the cells is read a few chunks at a time, its
entries sorted into cells through on-disk buckets.

Writing produces a spec 3.0.0 loom that loompy reads and validates: X (or
the matrix `--layer` names) as `/matrix`, every layer, obs and var as
attributes (`CellID`; `Gene` for symbols and `Accession` for ids), 2-D obsm
and varm, obsp and varp as graphs, and uns scalars as global attributes.
anndata 0.7's categoricals are written as their labels, as current ones are.
Strings are written as loompy writes them, ASCII with XML references. A
loom's matrices are dense by definition: the summary gives their size before
compression, and above 10 GB a warning says it will take a while. Chunks
holding only zeros compress to almost nothing.

| Flag | Meaning |
|---|---|
| `--layer` | The matrix that becomes `/matrix` (default X): a layer's name, with X, when there is one, written as a layer (`X_1` if a layer is named `X`); or `raw/X`, written with raw/var's genes and without X and the layers |

A store without X needs `--layer`, and is refused without it before
anything is written. A loom's attributes share one flat namespace per
axis: where two elements ask for one name, the loom's own attributes keep
theirs (`CellID`, `Gene`, `Accession`), then the obs and var columns, then
the obsm and varm arrays, and the others are left out with a warning. So an
obs column named `CellID`, or a var column named `Gene` beside the
`gene_symbols` that `Gene` is written from, is left out, as is an obsm
entry named like an obs column.

## SCope looms

pySCENIC, SCENIC+ and pycistopic (and VSN, and R's SCopeLoomR) write their
results into the loom for the SCope viewer: each regulon's AUC per cell and
its target genes (or regions), the embeddings and clusterings shown in the
viewer, and a `MetaData` attribute naming them. `anndata.read_loom`
refuses these looms, since loompy's validator rejects compound attributes.
Here such a loom reads as `loom (SCope)`, the summary naming what was found
(`298 regulons, 5 embeddings, 1 clustering`):

| SCope loom | AnnData |
|---|---|
| `col_attrs/RegulonsAUC`, a field per regulon | `obsm["RegulonsAUC"]`, a dataframe of cells x regulons |
| `row_attrs/Regulons`, a field per regulon, 0 or 1 | `varm["Regulons"]`, a dataframe of genes x regulons |
| `col_attrs/Embeddings_X` and `Embeddings_Y`, a field per embedding id; `Embedding` | `obsm["X_<name>"]`, n x 2, named from `MetaData` (`tSNE (default)` becomes `X_tSNE_default`); `Embedding` is the default one (id -1) when the others lack it |
| `col_attrs/Clusterings`, a field per clustering id | a categorical obs column named from `MetaData`, its categories the clusters' descriptions in cluster-id order |
| other compound attributes (VSN's `ClusterMarkers_<i>`, SCopeLoomR's `MotifRegulonsAUC`) | dataframes in obsm or varm, under their own names |
| global `MetaData`, JSON, plain or compressed | `uns["scope"]["metadata"]`, the JSON text |
| `regulonThresholds` in `MetaData` | `uns["scope"]["regulon_thresholds"]`, a dataframe by regulon |
| the embeddings and clusterings in `MetaData` | `uns["scope"]["embeddings"]` and `["clusterings"]`: each id's name, and where it went |

The dataframes keep the regulons' names, `ATF3_(+)` and the like, as their
columns, and the obs or var names as their index, which is how anndata
expects them; `pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)`, the
pySCENIC tutorial's recipe, gives the same table, as CI checks. A field
name holding `/`, which HDF5 and Zarr cannot store, has it replaced by `_`,
with a warning. A clustering named like an obs column already there gets a
suffix (`Tissue_1`), with a note; one `MetaData` does not name is called
`clustering_<id>`, and an embedding `X_embedding_<id>`. A value with no
cluster in `MetaData` is missing. A global attribute that holds more than
64 MiB once read is not read, with a warning. `MetaData` that is neither
JSON nor base64 of compressed JSON, or that decompresses to more than
64 MiB, is kept as text, with a warning; the embeddings
and clusterings are then named by id, and a clustering's categories are the
ids found.

Region-based looms (SCENIC+, pycistopic) hold regions (`chr1:100-200`)
where genes go, so var is named by region, and SCENIC+'s `linkedGene` is a
var column. The compound attributes are read a block of cells at a time,
so memory is one block across all regulons, never the table.

## Writing SCope looms

`--scope` writes a loom the SCope viewer reads: an AnnData's regulons,
embeddings and clusterings in SCope's form, with a `MetaData` naming them.
Without it a loom export is plain, and one of an AnnData that came from a
SCope loom says that `--scope` would keep its regulons.

```bash
adata convert scenic.h5ad -o scenic.loom --scope
adata convert scenic.h5ad -o scenic.loom --scope --clusterings leiden,cell_type
adata convert scenic.h5ad -o scenic.loom --scope --default-embedding X_umap
```

| AnnData | SCope loom |
|---|---|
| a dataframe in obsm or varm (`RegulonsAUC`, `Regulons`, `ClusterMarkers_<id>`) | a compound `col_attrs` or `row_attrs` of its numeric columns |
| `obsm` arrays of two columns | `Embeddings_X` and `Embeddings_Y`, a field per id; the default (id -1) as `Embedding` too |
| the categorical obs columns a SCope import recorded, and those `--clusterings` names | `Clusterings`, int codes per clustering (and not a plain attribute too), with `ClusterID` for the first; their clusters described in `MetaData` |
| other obs columns | plain attributes, as ever, listed in `MetaData` as metrics (numbers) and annotations (up to 245 values) |
| `uns["scope"]` | `MetaData`'s regulon thresholds (from the `regulon_thresholds` table, as edited), and the recorded ids and names |

| Flag | Meaning |
|---|---|
| `--scope` | Write a SCope loom |
| `--clusterings` | Categorical obs columns to write as clusterings, comma separated |
| `--default-embedding` | The obsm key SCope shows first (default: the recorded default, else `X_umap`, else `X_tsne`, else the first) |

An AnnData that came from a SCope loom keeps its embeddings' and
clusterings' ids and names, recorded in `uns["scope"]`, so a SCope loom
converts to h5ad and back with the same attributes. As SCope expects,
clusterings and their clusters are numbered 0..n-1 in order, regulon names
lose their spaces (`TP53 (5g)` becomes `TP53_(5g)`, alike in the AUC and
the Regulons, with a warning), and cluster markers are written only as
`ClusterMarkers_<id>` with their metrics (`ClusterMarkers_<id>_avg_logFC` and the like, which
`clusterMarkerMetrics` then lists, under the names a SCope import recorded), their fields renumbered with the clusters; a
clustering `--clusterings` names owns `ClusterMarkers_<its id>`. When there are clusterings, `ClusterID` is written afresh from the first
one, in place of an obs column of that name, so the two cannot
disagree; without any, an obs column `ClusterID` is written as any other. `MetaData` is plain JSON, which SCope reads as it
reads compressed. A clustering that is not categorical is refused, as is a
table with more columns than HDF5 can hold in one compound attribute (its
type must fit 64 KiB, a limit pySCENIC's own export shares), before any
output exists. Attributes are written a block of cells at a time. The
flat namespace holds SCope's parts too: where an obs column, an obsm entry
or one of SCope's parts ask for one name, the first keeps it (loom's and
SCope's own attributes, then columns, SCope's tables, arrays) and the
others are left out with a warning.

SCope opens looms without loompy's validation, which rejects compound
attributes; so do `loompy.connect(..., validate=False)` and the pySCENIC
tutorial's recipe, which CI uses to read our SCope looms back.

---
title: MuData
parent: Formats
grand_parent: Reference
nav_order: 9
---

# MuData

A MuData (muon, mudata, SCENIC+) holds one AnnData per modality under
`mod/`, beside global elements that span them: a joint clustering in obs,
a WNN embedding in obsm, the joint graph in obsp. `obsmap` and `varmap` say
which global rows each modality has. Both an `.h5mu` and a MuData Zarr
store are read; an output is an `.h5mu`.

```bash
adata convert multiome.h5mu -o multiome.h5ad
adata convert scenicplus.h5mu -o rna.h5ad --mod scRNA_counts
adata convert multiome.h5ad -o multiome.h5mu
adata convert multiome.h5ad -o one.h5mu --mod rna
```

| Flag | Meaning |
|---|---|
| `--mod` | Reading: the modality to write (default: each, to `OUT.<mod>.<ext>`; a file with one modality goes to `OUT`). Writing: the name to write the whole AnnData under, instead of splitting it |
| `--feature-types-column` | Writing: the var column to split by (default `feature_types`, when there is one) |

Reading writes each modality as it is stored, then adds the global parts
onto its cells (or features), in the modality's order, through the maps:

| MuData | Each modality's AnnData |
|---|---|
| `mod/<name>` | the AnnData itself: X, layers, obs, var, obsm, varm, obsp, varp, uns, raw |
| global obs and var columns | obs and var columns, on the modality's rows |
| `<name>:col` (a column pulled from this modality) | `col`, when the modality has no `col` of its own |
| `<other>:col` (pulled from another modality) | left out, with a note |
| global obsm, varm | obsm, varm, on the modality's rows; the per-modality masks are left out |
| global obsp, varp | obsp, varp, both axes on the modality's rows |
| global uns | uns |

Where a global element has a name the modality already has, the
modality's own is kept and the global one is left out, with a warning.
Without `obsmap`/`varmap` (an older writer), rows are matched by name. A map
that does not place each of the modality's rows once is refused before
anything is written. Global rows are read a block at a time, sorted for the
read when the modality's order differs; a graph's entries are remapped as
they stream and rebuilt through on-disk buckets, keeping CSR or CSC.

Writing splits var by its feature types, as muon reads a 10x Multiome
matrix: `Gene Expression` becomes `rna`, `Peaks` `atac`, `Antibody Capture`
`prot`, and any other type a modality named after it (`CRISPR Guide
Capture` → `crispr_guide_capture`). Without a feature-types column the
AnnData is one modality, `rna`.

| AnnData | MuData |
|---|---|
| X, layers, var, varm, varp, raw | each modality's, for its features; every matrix read once for all of them |
| obs | global obs; each modality's obs holds the cell names only |
| obsm, obsp, uns | global obsm, obsp, uns |
| | `obsmap`, `varmap` and the `obsm`/`varm` masks, root `encoding-type: mudata`, `axis: 0`, and the 512-byte `MuData` header mudata reads first |

An obsm entry named like a modality is refused, since a MuData keeps that
name for the modality's mask. Converting the `.h5mu` back gives each
modality with the cells' annotations and embeddings again.

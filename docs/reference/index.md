---
title: Reference
nav_order: 4
has_children: true
---

# Reference

| Page | Covers |
|---|---|
| [Commands](commands.md) | Every command and flag |
| [Formats](formats/index.md) | Each format `convert` reads or writes: what maps to what, its flags, how it streams |
| [Query expressions](query.md) | The language of `--obs-query` and `--var-query` |
| [On-disk spec: h5ad](spec-h5ad.md) | How each AnnData element is laid out in HDF5, and what `adata` does with it |
| [On-disk spec: Zarr](spec-zarr.md) | The same for Zarr, including the v2 and v3 differences |
| [Benchmarks](benchmarks.md) | Peak memory and wall time against anndata and scanpy, remeasured on every release |
| [Testing](testing.md) | How the suite is organised, the compatibility tests against anndata releases, and the complexity guards |

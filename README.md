# adata-cli

Explore and edit huge AnnData stores (`.h5ad` and `.zarr`) from the command line,
without loading them into memory and without a Python session. `adata` inspects,
subsets, splits, concatenates, recasts and converts stores, and moves elements in
and out as CSV, JSON, `.npy`, `.mtx` or images. It implements the AnnData on-disk
spec directly against h5py and zarr, so it streams stores larger than RAM, and
converts to and from 10x, STARsolo, Visium, Xenium, CellBender, Parse, loom, MuData
and more.

**Documentation: <https://cellgeni.github.io/adata-cli/>**

## Quick start

```bash
pip install pyadata-cli

adata view data.h5ad
adata subset data.h5ad -o cortex.h5ad --obs-query "cluster == Cortex_2"
adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
```

The command is `adata`; the distribution is `pyadata-cli`, because `adata-cli` was
taken on PyPI. `pip install 'pyadata-cli[pickle]'` adds pycisTopic pickles, and
`'pyadata-cli[spatial]'` Visium HD. The container image, with both, is
`quay.io/cellgeni/adata-cli`:

```bash
docker run --rm -v "$PWD":/data quay.io/cellgeni/adata-cli:latest adata view /data/data.h5ad
```

## Documentation

| Page | |
|---|---|
| [Getting started](https://cellgeni.github.io/adata-cli/getting-started/) | Install, then a first session: inspect, export, subset, import, convert |
| [Guides](https://cellgeni.github.io/adata-cli/guides/) | Inspect and export, subset and split, concatenate, recast, convert formats, pipelines and containers |
| [Reference](https://cellgeni.github.io/adata-cli/reference/) | Every command and flag, each format, query expressions, the on-disk spec, benchmarks |
| [Troubleshooting](https://cellgeni.github.io/adata-cli/troubleshooting/) | Error messages and their fixes |
| [Developers](https://cellgeni.github.io/adata-cli/developers/) | Layout, layers, the rules the code relies on, tests, releases |

The pages are also readable on GitHub in [`docs/`](https://github.com/cellgeni/adata-cli/tree/main/docs).

## Repository

| Path | Contents |
|---|---|
| `src/adata/cli.py`, `commands/` | The command-line app, and one module per command |
| `src/adata/core/` | The streaming algorithms: subset, concat, recast, the out-of-core transpose, queries |
| `src/adata/elements/`, `storage/` | The AnnData on-disk spec, over h5py and zarr alike |
| `src/adata/conversion/`, `formats/` | `convert`'s formats, and the file types of `export` and `import` |
| `tests/` | The test suite, including anndata compatibility and complexity guards |
| `benchmarks/` | The comparison against anndata and scanpy, rerun on every release |
| `docs/` | Documentation site |
| `Dockerfile` | The `quay.io/cellgeni/adata-cli` image |
| `.github/workflows/` | Tests, PyPI and image publishing, benchmarks |

## Changes

See the [changelog](https://cellgeni.github.io/adata-cli/changelog/).

## Licence

[MIT](https://github.com/cellgeni/adata-cli/blob/main/LICENSE)

# adata CLI

A command-line tool for exploring huge AnnData stores (`.h5ad` and `.zarr`) without loading them fully into memory. Streams data directly from disk for efficient inspection of structure, metadata, and matrices.

## Features

- Streaming access to very large `.h5ad` and `.zarr` stores — see [the benchmarks](docs/BENCHMARKS.md) for what that costs in practice, including where loading the file outright is faster
- Auto-detects `.h5ad` files vs `.zarr` directories
- Chunked processing for dense and sparse matrices (CSR/CSC)
- Reads every AnnData on-disk layout, from 0.7.x through the current spec, and always writes the current one
- Converts between HDF5 and Zarr (v2 and v3) in either direction
- Rich terminal output with progress indicators, kept on stderr so results pipe cleanly

**Documentation: [cellgeni.github.io/adata-cli](https://cellgeni.github.io/adata-cli/)**

## Installation

```bash
pip install pyadata-cli
```

The command is `adata`. The distribution is named `pyadata-cli` because
`adata-cli` was already taken on PyPI by an unrelated project.

From source with [uv](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/cellgeni/adata-cli.git
cd adata-cli
uv sync
```

For development and testing:
```bash
uv sync --extra dev
```

Alternative with pip:
```bash
git clone https://github.com/cellgeni/adata-cli.git
cd adata-cli
pip install .
```

For development and testing with pip:
```bash
pip install -e ".[dev]"
```

## Commands (Overview)

Run help at any level (e.g. `adata --help`, `adata export --help`).

- `view` – AnnData-aware inspection: store layout, shapes, and encodings; supports drilling into paths like `obsm/X_pca` or `uns`.
- `ls` – list the contents of any HDF5 or Zarr store as a tree, with no AnnData assumptions (works on `.loom` and plain `.h5`); `-1` emits bare paths for piping.
- `create` – write a new, empty AnnData store for `import` to fill in.
- `subset` – stream and write a filtered copy, selected by obs/var name lists (`--obs`/`--var`) or by expression (`--obs-query`/`--var-query`).
- `split` – write one store per distinct value of an annotation column, with a CSV manifest.
- `concat` – concatenate stores along the obs axis, with `--join inner|outer` and merge strategies for var and uns.
- `export` – extract data from a store; subcommands: `dataframe` (any dataframe group to CSV), `array` (dense to `.npy`), `sparse` (CSR/CSC to `.mtx`), `dict` (JSON), `image` (PNG). Results go to stdout when no `--output` is given.
- `import` – write new data into a store at any path; subcommands: `dataframe` (CSV), `array` (`.npy`), `sparse` (`.mtx`), `dict` (JSON), `image` (PNG/JPEG/TIFF).

### Building a store from scratch

```bash
adata create out.h5ad --obs-names cells.txt --var-names genes.txt
adata import sparse    out.h5ad X            counts.mtx --inplace
adata import dataframe out.h5ad obs          cells.csv  --inplace -i cell_id
adata import array     out.h5ad obsm/X_umap  umap.npy   --inplace
adata import dict      out.h5ad uns/params   params.json --inplace
```

### Filtering without a name list

```bash
adata subset data.h5ad -o cortex.h5ad --obs-query "cluster == Cortex_2"
adata subset data.h5ad -o big.h5ad    -q "n_counts > 1000 and cluster in A,B"
adata split  data.h5ad --by sample -o per_sample/
adata concat per_sample/*.h5ad -o merged.h5ad --join outer --label sample
```

## Documentation

- [Get started](docs/GET_STARTED.md) — a short tutorial
- [Command reference](docs/COMMANDS.md) — every command and flag
- [Element spec: HDF5](docs/ELEMENTS_h5ad.md) / [Zarr](docs/ELEMENTS_zarr.md) — the on-disk format, and what this tool does with it
- [Testing](docs/TESTING.md) — how the suite is organised, how compatibility is verified against six anndata releases, and the complexity guards that keep cost regressions out
- [Benchmarks](docs/BENCHMARKS.md) — peak memory and wall time against anndata and scanpy, remeasured on every release
- [Changelog](CHANGELOG.md)

## Docker

A docker image is available on QUAY: `quay.io/cellgeni/adata-cli:latest`. Pull and run with:

```bash
docker run --rm -it -v /path/to/data:/data quay.io/cellgeni/adata-cli:latest adata view /data/your_file.h5ad
```
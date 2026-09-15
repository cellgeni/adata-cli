# adata CLI

A command-line tool for exploring huge AnnData stores (`.h5ad` and `.zarr`) without loading them fully into memory. Streams data directly from disk for efficient inspection of structure, metadata, and matrices.

## Features

- Streaming access to very large `.h5ad` and `.zarr` stores
- Auto-detects `.h5ad` files vs `.zarr` directories
- Chunked processing for dense and sparse matrices (CSR/CSC)
- Reads every AnnData on-disk layout, from 0.7.x through the current spec, and always writes the current one
- Converts between HDF5 and Zarr (v2 and v3) in either direction
- Rich terminal output with progress indicators, kept on stderr so results pipe cleanly

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):
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
- `subset` – stream and write a filtered copy based on obs/var name lists, preserving dense and sparse matrix encodings.
- `export` – extract data from a store; subcommands: `dataframe` (any dataframe group to CSV), `array` (dense to `.npy`), `sparse` (CSR/CSC to `.mtx`), `dict` (JSON), `image` (PNG). Results go to stdout when no `--output` is given.
- `import` – write new data into a store; subcommands: `dataframe` (CSV → obs/var), `array` (`.npy`), `sparse` (`.mtx`), `dict` (JSON).

See [docs/GET_STARTED.md](docs/GET_STARTED.md) for a short tutorial.

## Docker

A docker image is available on QUAY: `quay.io/cellgeni/adata-cli:latest`. Pull and run with:

```bash
docker run --rm -it -v /path/to/data:/data quay.io/cellgeni/adata-cli:latest adata view /data/your_file.h5ad
```
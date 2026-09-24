# adata-cli

Explore and edit huge AnnData stores (`.h5ad` and `.zarr`) from the command
line, without loading them into memory and without a Python session.

`adata-cli` implements the AnnData on-disk spec directly against `h5py` and
`zarr`. It has no dependency on `anndata`, `pandas` or `scipy`, which is what
lets it stream stores far larger than RAM — and what makes it useful in a
container or a pipeline step where installing the scientific stack would be
overkill.

## Install

```bash
pip install pyadata-cli
```

The command is `adata`. The distribution is named `pyadata-cli` because
`adata-cli` was already taken on PyPI by an unrelated project.

Or run it without installing anything:

```bash
docker run --rm -it -v /path/to/data:/data \
    quay.io/cellgeni/adata-cli:latest adata view /data/your_file.h5ad
```

## Documentation

- **[Get started](GET_STARTED.md)** — a hands-on walkthrough: inspect a store,
  export metadata, filter it, write a subset.
- **[Command reference](COMMANDS.md)** — every command and flag.
- **[Element spec: HDF5](ELEMENTS_h5ad.md)** — how each AnnData element is laid
  out in `.h5ad`, and what this tool does with it.
- **[Element spec: Zarr](ELEMENTS_zarr.md)** — the same for `.zarr`, including
  the v2/v3 differences.
- **[Testing](TESTING.md)** — how the suite is organised, how compatibility is
  verified against six real anndata releases, and the complexity guards that
  keep cost regressions out.
- **[Benchmarks](BENCHMARKS.md)** — peak memory and wall time against anndata
  and scanpy, remeasured and republished on every release.

## At a glance

```bash
adata view data.h5ad                   # what's in it
adata ls data.h5ad --long              # every path, with shapes and encodings
adata export dataframe data.h5ad obs   # obs as CSV, on stdout

adata subset data.h5ad -o cortex.h5ad --obs-query "cluster == Cortex_2"
adata split  data.h5ad --by sample -o per_sample/
adata concat per_sample/*.h5ad -o merged.h5ad --join outer --label sample

adata convert data.h5ad X -o small.h5ad --dtype float32

adata create new.h5ad --n-obs 5000 --n-var 2000
adata import sparse new.h5ad X counts.mtx --inplace
```

## Format support

Every AnnData on-disk layout is readable, from anndata 0.7.x through the
current spec; everything written is in the current spec. This is verified in
CI against stores written by anndata 0.8, 0.9, 0.10, 0.11, 0.12 and 0.13, in
both formats. Both backends are
supported in both directions, so `adata subset in.h5ad -o out.zarr` converts as
it filters.

| Element | Read | Write |
|---|---|---|
| `anndata`, `raw`, `dict` | yes | yes |
| `dataframe` (0.2.0 and legacy 0.1.0) | yes | 0.2.0 |
| `array` | yes | yes |
| `csr_matrix` / `csc_matrix` | yes | yes |
| `categorical` (incl. `ordered`, legacy layouts) | yes | yes |
| `string-array` | yes | yes |
| `nullable-integer` / `-boolean` / `-string-array` | yes | yes |
| `numeric-scalar`, `string`, `null` | yes | yes |
| `awkward-array` | shown by `view`/`ls` | no |

## Source

[github.com/cellgeni/adata-cli](https://github.com/cellgeni/adata-cli)

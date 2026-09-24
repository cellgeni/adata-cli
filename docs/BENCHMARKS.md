# Benchmarks

adata-cli measured against anndata, and against scanpy wherever scanpy has a real
equivalent. The [`Benchmark`](../.github/workflows/benchmark.yml) workflow rewrites
this page on every tag and keeps each run's raw numbers in `docs/benchmarks/`.

**Peak RSS is the headline, not wall time.** This tool exists so that memory is set
by `--chunk` rather than by the size of the input. For data that fits in RAM, loading
the whole thing is often faster — so the tables below carry the rows where adata-cli
is the slower of the two, because those are the same measurement. The trade is memory
for time, and a table that hid the cost would not be worth publishing.

## What is measured

Fifteen cases on a GitHub-hosted runner: concat (inner, outer, `--merge same`),
subset by query, h5ad→zarr, `view`, `ls`, `create`, `export` dataframe / array /
sparse, `import dataframe`, `split`, and a peak-RSS-against-input-size sweep. Inputs
are 50,000 × 20,000 at 5% density, plus the 2,000 × 36,601 shape that hung in 0.5.1.

Every run also reports a startup floor (`--version` for each contender), because the
CLI costs 0.3–1 s to import and `import scanpy` costs 3–8 s; on small inputs that is
the entire measurement.

`export image`, `export dict`, `import image` and `import dict` are not here. No
library offers an equivalent, so the rows would only ever read `n/a`. They are covered
by the complexity guards instead.

## Results — `0.6.0`

| | |
|---|---|
| Tier | `ci`, 50,000 obs x 20,000 var, 5% dense CSR |
| Var-heavy shape | 2,000 obs x 36,601 var |
| Page cache | dropped between runs |
| Address-space ceiling | 12.9 GB per process |
| Repeats | 3 (fastest shown) |
| Platform | Linux x86_64, Python 3.12.3 |
| Versions | adata-cli 0.6.0, anndata 0.13.4, scanpy 1.12.4 |
| Generated | 2026-09-24T10:35:13Z |

### `startup`

What does each contender cost before it does any work?

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.61 s | 69.2 MB | - | - |
| anndata | 1.65 s | 132.5 MB | - | - |
| scanpy | 3.65 s | 264.8 MB | - | - |

### `concat-inner`

Concatenate two stores on the obs axis, inner join.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 5.90 s | 176.2 MB | 1.2 GB | - |
| anndata (concat_on_disk) | 6.40 s | 450.6 MB | 831.9 MB | - |
| anndata (in memory) | 12.27 s | 1.8 GB | 580.3 MB | - |
| scanpy | n/a | n/a | n/a | - |
| | *scanpy has no concat of its own; it re-exports anndata's* | | | |

### `concat-merge-same`

Concatenate carrying var columns forward -- the REQ-71798 case.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 1.24 s | 193.5 MB | 92.6 MB | - |
| anndata (concat_on_disk) | 2.37 s | 247.3 MB | 63.2 MB | - |
| anndata (in memory) | 2.84 s | 286.7 MB | 44.6 MB | - |

### `subset-query`

Keep the obs rows matching a predicate on an existing column.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 3.25 s | 345.2 MB | 309.7 MB | - |
| anndata (backed) | 5.16 s | 674.7 MB | 146.6 MB | - |
| anndata (in memory) | 5.42 s | 984.2 MB | 146.6 MB | - |
| scanpy (filter_cells) | 9.31 s | 1.6 GB | 291.2 MB | - |

### `h5ad-to-zarr`

Convert a store to Zarr.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 6.04 s | 246.0 MB | 115.3 MB in 2133 files | - |
| anndata (in memory) | 6.28 s | 778.6 MB | 136.8 MB in 55 files | - |
| scanpy | n/a | n/a | n/a | - |
| | *no streaming converter; scanpy defers to anndata* | | | |

### `inspect`

Report what is in the store, without reading the matrix.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.62 s | 71.5 MB | - | - |
| anndata (read_elem on obs) | 1.82 s | 159.9 MB | - | - |
| anndata (full load) | 3.31 s | 583.7 MB | - | - |

### `export-obs`

Write the obs table out as CSV.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.77 s | 85.6 MB | 1.8 MB | - |
| anndata (read_elem on obs) | 1.88 s | 159.7 MB | 1.8 MB | - |
| anndata (full load) | 3.45 s | 583.6 MB | 1.8 MB | - |

### `split-by-sample`

Write one store per distinct value of an obs column.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 56.72 s | 753.9 MB | 630.9 MB in 9 files | - |
| anndata (hand-written loop) | 7.82 s | 685.2 MB | 301.7 MB in 8 files | - |

### `ls`

List everything in the store.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.59 s | 70.6 MB | - | - |
| h5py (visit) | 1.65 s | 136.4 MB | - | - |
| h5ls -r | n/a | n/a | n/a | - |
| | *h5ls is not installed* | | | |

### `create`

Write an empty store with a given obs/var shape.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.65 s | 84.2 MB | 3.4 MB | - |
| anndata | 8.83 s | 228.0 MB | 49.2 MB | - |

### `export-array`

Write a dense element out as .npy.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.65 s | 82.8 MB | 10.0 MB | - |
| anndata (read_elem) | 1.82 s | 154.5 MB | 10.0 MB | - |
| anndata (full load) | 3.42 s | 583.7 MB | 10.0 MB | - |

### `export-sparse`

Write X out as Matrix Market text.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 25.59 s | 76.1 MB | 761.3 MB | - |
| anndata (sparse_dataset) | 7.14 s | 967.9 MB | 661.6 MB | - |
| anndata (full load) | 7.56 s | 773.6 MB | 661.6 MB | - |

### `import-dataframe`

Replace obs from a CSV.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 1.27 s | 113.8 MB | 296.0 MB | - |
| anndata | 7.02 s | 596.3 MB | 291.2 MB | - |

### `concat-outer`

Concatenate two stores keeping the union of variables.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 5.91 s | 176.6 MB | 1.2 GB | - |
| anndata (concat_on_disk) | 6.49 s | 450.7 MB | 831.9 MB | - |
| anndata (in memory) | 12.46 s | 1.8 GB | 580.3 MB | - |

### `convert-dtype`

Rewrite X as float32.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 5.68 s | 90.4 MB | 291.1 MB | - |
| anndata (in memory) | 7.00 s | 983.8 MB | 291.2 MB | - |
| scanpy | n/a | n/a | n/a | - |
| | *no dtype rewrite; scanpy defers to anndata* | | | |

### `convert-layout`

Transpose X from CSR to CSC.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli (streaming) | 14.00 s | 366.0 MB | 1.2 GB | - |
| adata-cli (--in-memory) | 9.75 s | 2.3 GB | 297.0 MB | - |
| anndata (in memory) | 7.40 s | 983.8 MB | 297.0 MB | - |

### `rss-vs-size`

Does peak memory track input size, at a fixed --chunk? This is the README's claim, stated as a measurement.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 5.93 s | 171.1 MB | 1.2 GB | - |
| anndata (in memory) | 12.56 s | 1.8 GB | 580.3 MB | - |

---

Rows marked `n/a` are operations the baseline does not offer; that is a result, not a gap in the measurement. Where a baseline is faster, the row stands as it is -- the trade this tool makes is memory for time, and hiding the cost would make the table useless.

**Output size** is the file as the filesystem reports it, which for HDF5 includes allocation slack. On a small tier that slack can be several times the stored bytes and the column says more about the writer's allocation strategy than about the data; at `ci` and above it is noise. Zarr stores report a file count alongside, because a directory of small chunks measures much larger than it holds.

### History

`concat-inner` on adata-cli, run by run. Full results for each are in [`docs/benchmarks/`](benchmarks/).

| Run | Ref | Tier | Wall time | Peak RSS | Raw |
|---|---|---|---|---|---|
| 2026-09-24T10:35:13Z | `0.6.0` | ci | 5.90 s | 176.2 MB | [json](benchmarks/0.6.0.json) |

## How peak memory is measured

Each measured command is forked from a small shim process, not from the benchmark
runner itself. On Linux a forked child inherits its parent's resident pages and
`execve` folds that into the `maxrss` the kernel reports, so a child of a fat parent
cannot appear small: with the runner holding 330 MB, a process allocating nothing
measured 326 MB. The shim brings that floor down to about 8 MB, uniform across every
contender and visible in the `startup` row.

## A caveat on the streaming claim

Peak memory is not flat in input size. Over a 256× span at a fixed chunk, peak
allocation grows 1.6× for `export array`, 2.2× for `export dataframe`, 6.5× for
`export sparse` and 46× for `subset` — far below the input curve, but not constant.
obs columns are read whole and a dense block is `--chunk` × n_var. The benchmark
exists to keep that curve visible rather than to assert a guarantee the code does
not yet meet.

## Running it yourself

```bash
uv run python -m benchmarks.run --tier smoke --out results.json
uv run python -m benchmarks.report results.json
```

Tiers are `smoke` (seconds), `ci` (50,000 × 20,000) and `large` (dispatch only, where
the in-memory baseline is expected to hit the address-space ceiling).

See [TESTING.md](TESTING.md#performance) for how the baselines are kept honest, and
how this differs from the complexity guards that gate every pull request.

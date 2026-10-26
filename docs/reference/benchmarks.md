---
title: Benchmarks
parent: Reference
nav_order: 6
redirect_from:
  - /BENCHMARKS.html
---

# Benchmarks

adata-cli measured against anndata, and against scanpy wherever scanpy has a real
equivalent. The [`Benchmark`](https://github.com/cellgeni/adata-cli/blob/main/.github/workflows/benchmark.yml) workflow rewrites
this page on every tag and keeps each run's raw numbers in `docs/benchmarks/`.

**Peak RSS is the headline, not wall time.** This tool exists so that memory is set
by `--chunk` rather than by the size of the input. For data that fits in RAM, loading
the whole thing is often faster — so the tables below carry the rows where adata-cli
is the slower of the two, because those are the same measurement. The trade is memory
for time, and a table that hid the cost would not be worth publishing.

## What is measured

Nineteen cases on a GitHub-hosted runner: concat (inner, outer, `--merge same`),
subset by query, h5ad→zarr (`convert`), `view`, `ls`, `create`, `export` dataframe /
array / sparse, `import dataframe`, MTX folder import (`convert`), `split`, `recast` narrowing and layout, and a
peak-RSS-against-input-size sweep. Inputs
are 50,000 × 20,000 at 5% density, plus the 2,000 × 36,601 shape that hung in 0.5.1.
`recast` narrowing starts from a copy of the main input stored as float64 values
with int64 indices, once with X uncompressed (as `write_h5ad` leaves it) and once
with lzf, and its tables report the file's size before and after.

Every run also reports a startup floor (`--version` for each contender), because the
CLI costs 0.3–1 s to import and `import scanpy` costs 3–8 s; on small inputs that is
the entire measurement.

`export image`, `export dict`, `import image` and `import dict` are not here. No
library offers an equivalent, so the rows would only ever read `n/a`. They are covered
by the complexity guards instead.

## Results — `0.16.2`

| | |
|---|---|
| Tier | `ci`, 50,000 obs x 20,000 var, 5% dense CSR |
| Var-heavy shape | 2,000 obs x 36,601 var |
| Page cache | dropped between runs |
| Address-space ceiling | 12.9 GB per process |
| Repeats | 3 (fastest shown) |
| Platform | Linux x86_64, Python 3.12.3 |
| Versions | adata-cli 0.16.2, anndata 0.13.4, scanpy 1.12.4 |
| Generated | 2026-10-26T14:36:31Z |

### `startup`

What does each contender cost before it does any work?

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.50 s | 70.0 MB | - | time -12% |
| anndata | 1.24 s | 135.6 MB | - | time -17% |
| scanpy | 2.88 s | 262.4 MB | - | time -10% |

### `concat-inner`

Concatenate two stores on the obs axis, inner join.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 7.95 s | 170.5 MB | 583.3 MB | time -34% |
| anndata (concat_on_disk) | 5.29 s | 452.7 MB | 831.9 MB | time -13% |
| anndata (in memory) | 7.67 s | 1.8 GB | 580.3 MB | time -37% |
| scanpy | n/a | n/a | n/a | - |
| | *scanpy has no concat of its own; it re-exports anndata's* | | | |

### `concat-merge-same`

Concatenate carrying var columns forward -- the REQ-71798 case.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 1.34 s | 195.4 MB | 45.7 MB | time -23% |
| anndata (concat_on_disk) | 1.84 s | 248.8 MB | 63.2 MB | time -17% |
| anndata (in memory) | 1.99 s | 290.3 MB | 44.6 MB | time -24% |

### `subset-query`

Keep the obs rows matching a predicate on an existing column.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 3.20 s | 213.4 MB | 147.1 MB | time -22% |
| anndata (backed) | 3.42 s | 630.1 MB | 146.6 MB | time -31% |
| anndata (in memory) | 3.49 s | 986.6 MB | 146.6 MB | time -34% |
| scanpy (filter_cells) | 6.96 s | 1.6 GB | 291.2 MB | time -22% |

### `h5ad-to-zarr`

Convert a store to Zarr.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 4.87 s | 155.6 MB | 151.6 MB in 4717 files | time -26%, RSS +25% |
| anndata (in memory) | 5.93 s | 776.5 MB | 136.8 MB in 55 files | no change |
| scanpy | n/a | n/a | n/a | - |
| | *no streaming converter; scanpy defers to anndata* | | | |

### `inspect`

Report what is in the store, without reading the matrix.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.87 s | 71.7 MB | - | time +47% |
| anndata (read_elem on obs) | 2.42 s | 162.4 MB | - | time +43% |
| anndata (full load) | 2.56 s | 586.3 MB | - | time -19% |

### `export-obs`

Write the obs table out as CSV.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.81 s | 85.5 MB | 1.8 MB | no change |
| anndata (read_elem on obs) | 2.14 s | 162.6 MB | 1.8 MB | time +24% |
| anndata (full load) | 2.58 s | 586.2 MB | 1.8 MB | time -20% |

### `split-by-sample`

Write one store per distinct value of an obs column.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 4.58 s | 199.7 MB | 302.6 MB in 9 files | time -30% |
| anndata (hand-written loop) | 6.34 s | 687.2 MB | 301.7 MB in 8 files | time -19% |

### `ls`

List everything in the store.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.55 s | 71.2 MB | - | time -10% |
| h5py (visit) | 1.64 s | 138.8 MB | - | no change |
| h5ls -r | n/a | n/a | n/a | - |
| | *h5ls is not installed* | | | |

### `create`

Write an empty store with a given obs/var shape.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.54 s | 84.6 MB | 3.4 MB | time -19% |
| anndata | 7.50 s | 231.1 MB | 49.2 MB | time -14% |

### `export-array`

Write a dense element out as .npy.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.62 s | 83.4 MB | 10.0 MB | no change |
| anndata (read_elem) | 1.38 s | 156.6 MB | 10.0 MB | time -15% |
| anndata (full load) | 4.63 s | 585.9 MB | 10.0 MB | time +43% |

### `export-sparse`

Write X out as Matrix Market text.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 9.42 s | 252.0 MB | 661.3 MB | time -15% |
| anndata (sparse_dataset) | 5.64 s | 971.5 MB | 661.6 MB | time -21% |
| anndata (full load) | 6.24 s | 777.5 MB | 661.6 MB | time -13% |

### `import-dataframe`

Replace obs from a CSV.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 1.85 s | 114.6 MB | 296.0 MB | time +50% |
| anndata | 5.35 s | 596.6 MB | 291.2 MB | time -24% |

### `import-10x-mtx`

Read a gzipped Cell Ranger 3 MTX folder of the same counts into an h5ad. adata-cli streams it in two passes and keeps integer counts; scanpy loads it and converts them to float32.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 12.17 s | 344.2 MB | 279.2 MB | time -25% |
| scanpy | 11.88 s | 3.3 GB | 284.6 MB | time -15% |

### `import-csv`

Read a dense count table, a CSV of the first 10,000 cells, into an h5ad. adata-cli reads it twice, a block of lines at a time, and writes CSR int32; anndata's read_csv loads it as dense float32.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 7.05 s | 179.5 MB | 56.5 MB | - |
| anndata | 37.60 s | 1.8 GB | 44.7 MB | - |

### `concat-outer`

Concatenate two stores keeping the union of variables.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 9.79 s | 173.0 MB | 583.3 MB | time -21% |
| anndata (concat_on_disk) | 5.26 s | 452.9 MB | 831.9 MB | time -15% |
| anndata (in memory) | 7.57 s | 1.8 GB | 580.3 MB | time -38% |

### `recast-narrow`

Narrow X from float64 values with int64 indices to float32 and int32, as issue #13 asked. X is uncompressed, as `write_h5ad` leaves it by default, so the output keeps that and the baseline writes it the same way.

| Contender | Wall time | Peak RSS | Input → output | vs previous |
|---|---|---|---|---|
| adata-cli | 2.92 s | 119.6 MB | 815.6 MB → 415.4 MB (-49%) | time +24% |
| anndata (in memory) | 3.55 s | 1.8 GB | 815.6 MB → 417.3 MB (-49%) | time -10% |
| scanpy | n/a | n/a | n/a | - |
| | *no dtype rewrite; scanpy defers to anndata* | | | |

### `recast-narrow-lzf`

The same narrowing on an lzf-compressed X. Integer counts in float64 are mostly zero bytes, which lzf already removes, so expect a small saving on disk; the halving is in memory, once the matrix is loaded.

| Contender | Wall time | Peak RSS | Input → output | vs previous |
|---|---|---|---|---|
| adata-cli | 5.55 s | 128.7 MB | 302.5 MB → 291.1 MB (-4%) | time -30% |
| anndata (in memory) | 5.95 s | 1.8 GB | 302.5 MB → 291.2 MB (-4%) | time -27% |

### `convert-layout`

Transpose X from CSR to CSC. The output should be the size of the input: 0.7.0 fixed a `--layout` that wrote it three times larger.

| Contender | Wall time | Peak RSS | Input → output | vs previous |
|---|---|---|---|---|
| adata-cli (streaming) | 14.29 s | 372.5 MB | 291.2 MB → 297.0 MB (+2%) | no change |
| adata-cli (--in-memory) | 7.80 s | 2.3 GB | 291.2 MB → 297.0 MB (+2%) | time -17% |
| anndata (in memory) | 4.74 s | 986.2 MB | 291.2 MB → 297.0 MB (+2%) | time -33% |

### `rss-vs-size`

Does peak memory track input size, at a fixed --chunk? This is the README's claim, stated as a measurement.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 9.33 s | 176.7 MB | 583.3 MB | time -23% |
| anndata (in memory) | 10.49 s | 1.8 GB | 580.3 MB | time -14% |

---

Rows marked `n/a` are operations the baseline does not offer; that is a result, not a gap in the measurement. Where a baseline is faster, the row stands as it is -- the trade this tool makes is memory for time, and hiding the cost would make the table useless.

**Output size** is the file as the filesystem reports it, which for HDF5 includes allocation slack. On a small tier that slack can be several times the stored bytes and the column says more about the writer's allocation strategy than about the data; at `ci` and above it is noise. Zarr stores report a file count alongside, because a directory of small chunks measures much larger than it holds. Where a case exists to change a file's size, the column reads input → output, with the change.

### History

`concat-inner` on adata-cli, run by run. Full results for each are in [`docs/benchmarks/`](https://github.com/cellgeni/adata-cli/tree/main/docs/benchmarks).

| Run | Ref | Tier | Wall time | Peak RSS | Raw |
|---|---|---|---|---|---|
| 2026-10-26T14:36:31Z | `0.16.2` | ci | 7.95 s | 170.5 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.16.2.json) |
| 2026-10-23T17:54:37Z | `0.16.1` | ci | 10.88 s | 180.9 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.16.1.json) |
| 2026-10-23T15:09:41Z | `0.16.0` | ci | 10.94 s | 177.8 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.16.0.json) |
| 2026-10-22T19:08:11Z | `0.15.0` | ci | 12.59 s | 177.9 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.15.0.json) |
| 2026-10-21T16:33:57Z | `0.14.0` | ci | 10.63 s | 172.3 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.14.0.json) |
| 2026-10-20T23:25:54Z | `0.13.0` | ci | 11.79 s | 179.3 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.13.0.json) |
| 2026-10-20T15:49:10Z | `0.12.0` | ci | 12.61 s | 172.8 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.12.0.json) |
| 2026-10-19T19:34:47Z | `0.11.1` | ci | 7.60 s | 174.9 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.11.1.json) |
| 2026-10-19T17:02:30Z | `0.11.0` | ci | 12.24 s | 176.9 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.11.0.json) |
| 2026-10-19T10:48:49Z | `0.10.0` | ci | 12.13 s | 172.9 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.10.0.json) |
| 2026-10-16T11:55:36Z | `0.9.0` | ci | 12.28 s | 176.2 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.9.0.json) |
| 2026-10-15T23:13:08Z | `0.8.0` | ci | 12.11 s | 172.8 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.8.0.json) |
| 2026-10-15T14:55:32Z | `0.7.1` | ci | 15.37 s | 172.4 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.7.1.json) |
| 2026-10-15T12:48:08Z | `0.7.0` | ci | 9.29 s | 170.3 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.7.0.json) |
| 2026-09-24T15:41:03Z | `0.6.1` | ci | 12.26 s | 177.2 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.6.1.json) |
| 2026-09-24T10:35:13Z | `0.6.0` | ci | 5.90 s | 176.2 MB | [json](https://github.com/cellgeni/adata-cli/blob/main/docs/benchmarks/0.6.0.json) |

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

See [Testing](testing.md#performance) for how the baselines are kept honest, and
how this differs from the complexity guards that gate every pull request.

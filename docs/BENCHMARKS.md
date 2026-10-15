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

Seventeen cases on a GitHub-hosted runner: concat (inner, outer, `--merge same`),
subset by query, h5ad→zarr (`convert`), `view`, `ls`, `create`, `export` dataframe /
array / sparse, `import dataframe`, `split`, `recast` dtype and layout, and a
peak-RSS-against-input-size sweep. Inputs
are 50,000 × 20,000 at 5% density, plus the 2,000 × 36,601 shape that hung in 0.5.1.

Every run also reports a startup floor (`--version` for each contender), because the
CLI costs 0.3–1 s to import and `import scanpy` costs 3–8 s; on small inputs that is
the entire measurement.

`export image`, `export dict`, `import image` and `import dict` are not here. No
library offers an equivalent, so the rows would only ever read `n/a`. They are covered
by the complexity guards instead.

## Results — `0.7.0`

| | |
|---|---|
| Tier | `ci`, 50,000 obs x 20,000 var, 5% dense CSR |
| Var-heavy shape | 2,000 obs x 36,601 var |
| Page cache | dropped between runs |
| Address-space ceiling | 12.9 GB per process |
| Repeats | 3 (fastest shown) |
| Platform | Linux x86_64, Python 3.12.3 |
| Versions | adata-cli 0.7.0, anndata 0.13.4, scanpy 1.12.4 |
| Generated | 2026-10-15T12:48:08Z |

### `startup`

What does each contender cost before it does any work?

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.63 s | 67.9 MB | - | time +13% |
| anndata | 1.65 s | 133.5 MB | - | no change |
| scanpy | 3.57 s | 260.3 MB | - | no change |

### `concat-inner`

Concatenate two stores on the obs axis, inner join.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 9.29 s | 170.3 MB | 583.3 MB | time -24% |
| anndata (concat_on_disk) | 5.94 s | 447.3 MB | 831.9 MB | no change |
| anndata (in memory) | 9.09 s | 1.8 GB | 580.3 MB | time -26% |
| scanpy | n/a | n/a | n/a | - |
| | *scanpy has no concat of its own; it re-exports anndata's* | | | |

### `concat-merge-same`

Concatenate carrying var columns forward -- the REQ-71798 case.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 1.50 s | 193.1 MB | 45.7 MB | time -14% |
| anndata (concat_on_disk) | 2.24 s | 246.3 MB | 63.2 MB | no change |
| anndata (in memory) | 2.42 s | 288.9 MB | 44.6 MB | no change |

### `subset-query`

Keep the obs rows matching a predicate on an existing column.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 3.35 s | 209.5 MB | 147.1 MB | time -18% |
| anndata (backed) | 4.26 s | 683.4 MB | 146.6 MB | time -14% |
| anndata (in memory) | 4.49 s | 985.2 MB | 146.6 MB | time -14% |
| scanpy (filter_cells) | 7.69 s | 1.6 GB | 291.2 MB | time -15% |

### `h5ad-to-zarr`

Convert a store to Zarr.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 6.14 s | 154.2 MB | 151.6 MB in 4717 files | time -21%, RSS -34% |
| anndata (in memory) | 5.70 s | 778.4 MB | 136.8 MB in 55 files | no change |
| scanpy | n/a | n/a | n/a | - |
| | *no streaming converter; scanpy defers to anndata* | | | |

### `inspect`

Report what is in the store, without reading the matrix.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.61 s | 69.8 MB | - | no change |
| anndata (read_elem on obs) | 1.76 s | 160.3 MB | - | no change |
| anndata (full load) | 3.19 s | 584.3 MB | - | no change |

### `export-obs`

Write the obs table out as CSV.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.77 s | 83.3 MB | 1.8 MB | no change |
| anndata (read_elem on obs) | 1.78 s | 161.1 MB | 1.8 MB | no change |
| anndata (full load) | 3.22 s | 584.4 MB | 1.8 MB | no change |

### `split-by-sample`

Write one store per distinct value of an obs column.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 4.99 s | 196.5 MB | 302.6 MB in 9 files | time -25% |
| anndata (hand-written loop) | 6.34 s | 684.4 MB | 301.7 MB in 8 files | time -19% |

### `ls`

List everything in the store.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.63 s | 68.9 MB | - | no change |
| h5py (visit) | 1.67 s | 136.4 MB | - | no change |
| h5ls -r | n/a | n/a | n/a | - |
| | *h5ls is not installed* | | | |

### `create`

Write an empty store with a given obs/var shape.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.66 s | 82.5 MB | 3.4 MB | no change |
| anndata | 9.00 s | 229.1 MB | 49.2 MB | no change |

### `export-array`

Write a dense element out as .npy.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 0.66 s | 81.3 MB | 10.0 MB | time +10% |
| anndata (read_elem) | 1.68 s | 154.6 MB | 10.0 MB | no change |
| anndata (full load) | 3.22 s | 584.3 MB | 10.0 MB | no change |

### `export-sparse`

Write X out as Matrix Market text.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 21.22 s | 74.7 MB | 761.3 MB | time -17% |
| anndata (sparse_dataset) | 6.53 s | 968.5 MB | 661.6 MB | no change |
| anndata (full load) | 6.72 s | 774.4 MB | 661.6 MB | no change |

### `import-dataframe`

Replace obs from a CSV.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 1.96 s | 112.4 MB | 296.0 MB | time +63% |
| anndata | 5.36 s | 600.5 MB | 291.2 MB | time -22% |

### `concat-outer`

Concatenate two stores keeping the union of variables.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 9.15 s | 175.6 MB | 583.3 MB | time -25% |
| anndata (concat_on_disk) | 5.93 s | 449.6 MB | 831.9 MB | no change |
| anndata (in memory) | 9.07 s | 1.8 GB | 580.3 MB | time -26% |

### `convert-dtype`

Rewrite X as float32.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 4.02 s | 88.2 MB | 291.1 MB | time -28% |
| anndata (in memory) | 5.35 s | 984.0 MB | 291.2 MB | time -23% |
| scanpy | n/a | n/a | n/a | - |
| | *no dtype rewrite; scanpy defers to anndata* | | | |

### `convert-layout`

Transpose X from CSR to CSC.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli (streaming) | 18.18 s | 362.2 MB | 297.0 MB | time +33% |
| adata-cli (--in-memory) | 8.85 s | 2.3 GB | 297.0 MB | no change |
| anndata (in memory) | 6.04 s | 984.7 MB | 297.0 MB | time -16% |

### `rss-vs-size`

Does peak memory track input size, at a fixed --chunk? This is the README's claim, stated as a measurement.

| Contender | Wall time | Peak RSS | Output | vs previous |
|---|---|---|---|---|
| adata-cli | 9.19 s | 180.6 MB | 583.3 MB | time -24% |
| anndata (in memory) | 8.99 s | 1.8 GB | 580.3 MB | time -26% |

---

Rows marked `n/a` are operations the baseline does not offer; that is a result, not a gap in the measurement. Where a baseline is faster, the row stands as it is -- the trade this tool makes is memory for time, and hiding the cost would make the table useless.

**Output size** is the file as the filesystem reports it, which for HDF5 includes allocation slack. On a small tier that slack can be several times the stored bytes and the column says more about the writer's allocation strategy than about the data; at `ci` and above it is noise. Zarr stores report a file count alongside, because a directory of small chunks measures much larger than it holds.

### History

`concat-inner` on adata-cli, run by run. Full results for each are in [`docs/benchmarks/`](https://github.com/cellgeni/adata-cli/tree/main/docs/benchmarks).

| Run | Ref | Tier | Wall time | Peak RSS | Raw |
|---|---|---|---|---|---|
| 2026-10-15T12:48:08Z | `0.7.0` | ci | 9.29 s | 170.3 MB | [json](benchmarks/0.7.0.json) |
| 2026-09-24T15:41:03Z | `0.6.1` | ci | 12.26 s | 177.2 MB | [json](benchmarks/0.6.1.json) |
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

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

<!-- results -->

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

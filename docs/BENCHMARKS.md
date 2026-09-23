# Benchmarks

This page is written by the [`Benchmark`](../.github/workflows/benchmark.yml)
workflow on every tag. Until the next release it stands empty.

What will appear here: adata-cli against anndata, and against scanpy wherever
scanpy has a real equivalent, measured on a GitHub-hosted runner at 50,000
obs x 20,000 var plus the 2,000 x 36,601 shape that hung in 0.5.1.

**Peak RSS is the headline, not wall time.** This tool exists so that memory
is set by `--chunk` rather than by input size. For data that fits in RAM,
loading the whole thing is often faster, and the tables say so where it is
true -- the trade is memory for time, and a table that hid the cost would be
worth nothing.

A `ci` run taken during development, as an indication: `adata concat` of two
50,000 x 20,000 stores peaked at 202 MB against 1,778 MB for `ad.concat` in
memory and 439 MB for `anndata.experimental.concat_on_disk`, and was faster
than both. The published tables will say where it is slower, too.

To produce one locally:

```bash
uv run python -m benchmarks.run --tier smoke --out results.json
uv run python -m benchmarks.report results.json
```

See [TESTING.md](TESTING.md#performance) for what is measured, how the
baselines are kept honest, and how this differs from the complexity guards
that run on every pull request.

# Testing

```bash
uv sync --extra dev
uv run pytest                      # everything
uv run pytest -m "not integration" # fast: no environment building, ~40s
uv run pytest -m perf              # just the tracing complexity guards
uv run pytest -m integration       # compatibility across anndata releases
```

CI runs the unit suite on Python 3.12 and 3.13 with a **90% coverage floor**,
and the compatibility suite as a separate job.

## How the suite is organised

| File | What it covers |
|---|---|
| `test_elements.py` | The element layer — encodings, string dtypes, readers and writers — run against HDF5, Zarr v2 and Zarr v3 |
| `test_storage.py` | Backend detection, copying, Zarr versions, consolidated metadata |
| `test_formats.py` | `.npy`, `.mtx`, image and JSON export/import, including their failure modes |
| `test_validate.py` | Dimension validation for every axis-bearing path, including `raw/` |
| `test_invariants.py` | Relationships that must hold for any data: subsetting everything is the identity, split partitions exactly, concat undoes split |
| `test_anndata_roundtrip.py` | anndata writes the fixtures, reads back our output |
| `test_anndata_versions.py` | Compatibility with six real anndata releases (see below) |
| `test_commands_phase2.py`, `test_commands_coverage.py`, `test_cli.py` | Command surfaces and error paths |
| `test_performance.py` | Complexity guards -- what an operation costs, not how long it takes (see below) |
| `test_benchmark_harness.py` | That the benchmark's measurement is trustworthy |
| `test_subset.py`, `test_export.py`, `test_import.py`, `test_info_read.py`, `test_zarr.py`, `test_query.py` | Per-feature unit tests |

## Writing a test

Use the `new_store` fixture to get a store on each backend in turn — it is
parametrised over `h5ad`, `zarr2` and `zarr3`, so one test body covers all
three:

```python
def test_something(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_string_array(root["uns"], "s", ["a", "b"])
    with opener("r") as root:          # reopen: Zarr rewrites its index on close
        assert read_str_all(root["uns"]["s"]) == ["a", "b"]
```

For CLI surfaces, use the module-level `CliRunner` and assert on
`result.stdout + (result.stderr or "")`, since status goes to stderr. Strip
ANSI before matching message text — Rich also wraps long lines, so collapse
whitespace.

## Performance

Two mechanisms, and confusing them is the main way this gets misused.

| | `tests/test_performance.py` | `benchmarks/` |
|---|---|---|
| Measures | operation **counts** | wall time, peak RSS, output size |
| Runs | every CI job, both interpreters | tags, and `workflow_dispatch` |
| Gates | **yes** -- it fails the build | never |
| Data | 64-4096 elements | 50,000 x 20,000 |

### Why the guards do not use a clock

A test that can fail because a runner was busy does not belong in a merge
gate. Every number in `test_performance.py` is deterministic: the same input
gives the same count on every machine. That is what lets it block a merge.

This is not new -- `test_commands_phase2.py` already counts `read_str_all`
calls for exactly this reason. The performance file generalises the idea; it
does not replace those tests, and where an exact count is derivable an exact
count is still better than a ratio, because it says what the number *should*
be rather than only that it did not grow.

### What is counted

`tests/perf_counters.py` patches four seams, all verified against the pinned
h5py 3.15.1 and zarr 3.1.5:

| Hook | Sees |
|---|---|
| `h5py.Dataset.__getitem__` | HDF5 reads |
| `h5py.Dataset.__setitem__` | HDF5 writes |
| `h5py.Group.create_dataset` | HDF5 writes made at creation |
| `zarr.Array.__getitem__` | Zarr reads, v2 and v3 alike |
| `zarr.storage.LocalStore.get` | chunk and metadata fetches |
| `LocalStore.set` / `.delete` | write storms |

Both write seams are needed. `create_dataset(name, data=...)` writes its payload
during creation and never touches `__setitem__`, so hooking only the latter leaves
every `import` and `create` guard measuring zero — which is how they were first
written, and what the lower bound below caught.

Use `io.reads` for a read path, `io.work` (reads plus writes) for `import` and
`create`.

The libraries are patched rather than anything in `src/adata/`. An in-repo
seam would only see the call sites that remembered to use it, which is the
wrong property for a guard meant to catch the read nobody thought of -- and
the matrix paths slice the backend objects directly anyway.

Elements are counted, not just calls: the `--merge` bug was n calls each
reading n elements, and counting calls alone would miss a vectorised variant
that reads n elements n times in one call.

**Known bypasses.** `Dataset.read_direct`, `np.asarray(dataset)` (it goes via
`__array__`), `dataset.asstr()[...]` and `.fields()` reach the file without
passing any hook. None are used today. If you add one, the counters will
quietly report less work than happened -- which is why every guard also
asserts a lower bound, and why there are two canary tests with known absolute
counts. If a canary fails, fix the hooks before trusting anything else here.

### The invariant

Three sizes at 4x spacing, comparing successive **increments**:

```python
d1 = c(4n) - c(n)
d2 = c(16n) - c(4n)
assert d2 <= 6 * d1     # grows no faster than linearly
assert d1 >= 4n         # and the counters actually saw something
```

The increment form is the point. Comparing raw counts needs an additive
constant to absorb fixed setup cost, and there is no principled value for
one: too small and it is flaky, too large and a quadratic with a small
coefficient hides underneath at n=64. Any constant appears in both
differences and cancels exactly.

Why 6: at 4x spacing the increment ratio is 4.0 for linear work, about 4.4
for n log n, 8 for n^1.5 and 16 for quadratic. 6 sits in the gap with room on
both sides. `test_growth_limit_separates_linear_from_super_linear` asserts
that calibration rather than leaving it as a comment, and it fails if anyone
changes `SIZES` without recomputing the limit.

**Scale one axis per test, and name it in the test id.** Scaling two at once
makes legitimate work look quadratic -- an outer-join concat really does
produce n_obs x n_var_union cells. The axes worth separate coverage are
`n_var`, `n_obs`, `n_inputs`, `n_obs_columns`, `n_var_columns`,
`n_categories` and `n_groups`.

### Three invariants, not one

`assert_grows_linearly` only catches super-linear growth. Most commands need exactly
that, but two claims this tool makes are stronger, and a linear guard would happily
accept a 64x increase in a command that is supposed to read nothing.

| Helper | Claim | Used for |
|---|---|---|
| `assert_grows_linearly` | no worse than linear in one axis | most commands |
| `assert_grows_slower_than_input` | grows at least N times slower than the input | streaming at a fixed chunk |
| `assert_independent_of` | does not grow at all | inspection, and grouping work per row |

**Inspection is free, and that is an exact number.** `view` and `ls` reach only
`.shape`, `.dtype` and `.attrs`: `axis_len` goes through `element_len`, which reads a
shape, and `_array_details` and `_infer_untagged` never touch a value. So the guard
asserts **zero** data elements read rather than a ratio — 0 against 0 is not a
meaningful ratio, and the moment inspection reads one column the answer stops being
zero however the store scales. A companion test exports the same fixture to prove
there was data there to read, so the zero cannot pass because the fixture was empty.

**Streaming is bounded well below the input, which is weaker than it sounds and is
what the measurements support.** Over a 256x span at a fixed chunk:

| | peak allocation growth |
|---|---|
| `export array` | 1.6x |
| `export dataframe` | 2.2x |
| `export sparse` | 6.5x |
| `subset` | 46x |

Only `export array` is close to flat, so only it is held to a near-flat bound. None is
asserted as flat outright. `subset` is the weakest because obs columns are
materialised one at a time — a known gap that `benchmarks/` reports rather than these
guards conceal. Streamed `export sparse` is separately asserted to stay under a
quarter of what `--in-memory` costs, both measured in the same run so the factor holds
on any machine.

### Where counting is not enough

Two real defects in `concat.py` were invisible to all of the above, and both
needed their own instrument:

- **Category merging** used `if category not in categories` on a list. The
  category lists are read once either way, so no read counter sees it, and
  `x not in lst` is a single bytecode, so the line tracer does not either --
  the quadratic is inside C-level list membership. Counting string
  comparisons through a `str` subclass is what made it visible: 2,096,128
  comparisons at k=1024.
- **A per-element loop over obs rows.** Linear, just with a fat constant, so
  no ratio catches it. The guard compares executed Python lines per row
  against the numeric column path measured in the same run -- self-calibrating,
  so it needs no hand-tuned budget and holds across interpreters.

A third case needed a third instrument. `split --by` grouped rows with
`np.nonzero(values == label)` inside a loop over distinct labels, rescanning each
chunk once per label: O(n_rows x n_groups), which at a million cells and a thousand
samples is 10^9 comparisons. No read counter moves, because the chunk is already in
memory; nothing lasting is allocated; and the Python line count per label is constant.
`count_scanned_elements` counts what is handed to numpy's scanning primitives and
makes it visible — 16,384 elements at 4 groups against 1,048,576 at 256.

It is a **floor, not a measurement**: an operator such as `values == label` dispatches
to the ufunc in C and never passes the patched `np.equal`, and `arr.argsort()` is
invisible for the same reason. That is the right property for a guard — it can only
under-report — but it means a guard using it must also assert a lower bound, so that
under-reporting to nothing fails instead of passing.

`count_allocations` (tracemalloc), `count_lines` (`sys.settrace`) and
`count_scanned_elements` are the tools for all of this. The line tracer costs a 10-50x
slowdown, so its tests carry the `perf` marker and stay small.

### What the guards deliberately do not claim

Obs columns are read whole, so peak allocation for a concat is O(n_obs), not
O(`--chunk`). The streaming guarantee holds for X, not for obs annotation.
The guards only stop that getting worse than linear; `benchmarks/` reports
the actual curve.

### What is covered

Every subcommand has a guard. When you add a command, add one: pick the axis its cost
should scale with, scale only that, and hold everything else fixed.

| Command | Axis scaled | Invariant |
|---|---|---|
| `view`, `view --types`, `ls`, `ls --long`, `ls --plain` | n_obs, n_var | **zero** data reads |
| `create` | n_obs | linear in writes, both generated names and a name file |
| `concat` | n_var, n_obs, n_inputs, n_obs_columns, n_var_columns | linear; every `--merge` strategy and both joins |
| `concat --label`, `--index-unique` | n_obs | linear |
| `concat` category union | n_categories | bounded string comparisons |
| `subset` by name, by query | n_obs, n_var | linear |
| `split --by`, `--axis var` | n_groups | linear reads, **flat** scan work |
| `export dataframe` | n_obs, n_obs_columns | linear |
| `export array`, `sparse` (both paths), `dict`, `image` | elements, nnz, keys, pixels | linear |
| `import dataframe`, `array`, `sparse`, `dict`, `image` | rows, elements, nnz, keys, pixels | linear in writes |
| h5ad to zarr | n_obs | linear |

The `slow` marker is on the three guards that build a 65,536-row store; they still gate
merges, and `-m "not slow"` skips them locally. `perf` is on the tracing guards.

### Reading a failure

It says cost grew super-linearly on the named axis. The assertion message
carries the whole measured series and the computed ratio. Usually the code is
wrong. Occasionally the expectation is -- an operation legitimately gained
work -- and then the new number needs a comment saying why, in the same style
as the exact counts in `test_commands_phase2.py`.

## The comparative benchmark

```bash
uv run python -m benchmarks.run --tier smoke --out results.json
uv run python -m benchmarks.report results.json
```

Tiers are `smoke` (1,000 x 2,000, seconds), `ci` (50,000 x 20,000) and
`large` (500,000 x 20,000, dispatch only, where the in-memory baseline is
expected to hit the ceiling). Every tier also builds the 2,000 x 36,601
var-heavy shape, which is what hung in 0.5.1.

`ci` is cheaper than it looks: 580 MB of fixtures and three cases took 43
seconds end to end on a laptop, so the full set with three repeats is minutes
rather than the hour the workflow allows. The 90-minute timeout is headroom
for `large`, not an estimate.

A `ci` run measured while writing this, for a sense of what the tables say — and of
what they are for. Peak RSS first, wall time second:

| Case | adata-cli | best baseline |
|---|---|---|
| `concat-inner` | **202 MB**, 2.35 s | 439 MB, 16.44 s (`concat_on_disk`) |
| `concat-outer` | **202 MB**, 2.60 s | 436 MB, **2.24 s** (`concat_on_disk`) |
| `inspect` | **63 MB**, 0.31 s | 138 MB, 0.79 s (`read_elem`) |
| `create` | **78 MB**, 0.28 s | 2,225 MB, 2.85 s |
| `import-dataframe` | **107 MB**, 0.41 s | 571 MB, 1.84 s |
| `export-sparse` | **68 MB**, 10.79 s | 759 MB, **1.90 s** (full load) |
| `ls` | 63 MB, 0.25 s | **7 MB, 0.01 s** (`h5ls -r`) |

The last two rows are the reason the report is not a leaderboard. `export sparse`
streams in a tenth of the memory and takes five times as long; `h5ls` walks the file
in a hundredth of our time and a ninth of our memory, because it is C and does not
start a Python interpreter. Both belong in the table. A benchmark that only published
the rows we win would not be worth running.

Note also that 202 MB is not flat in input size — obs columns are read whole, and a
dense block is `--chunk` x n_var. The benchmark exists to keep that curve visible
rather than to assert a claim the code does not yet meet.

Results are published to `docs/BENCHMARKS.md` and `docs/benchmarks/<tag>.json`
on every tag, and appended to the GitHub release notes if a release exists.
Artifacts expire; the docs page is the durable series.

### How it measures

- **`os.wait4`, not `resource.getrusage`.** `RUSAGE_CHILDREN` is a running
  maximum over every child ever reaped, so one large case would poison every
  later row. Output goes to temporary files rather than pipes, because
  `communicate()` reaps the child and loses its rusage.
- **`RLIMIT_AS` at 12 GiB on every child.** Cases where the in-memory
  baseline cannot cope are the whole point, but an uncontained OOM kills the
  runner agent and the job ends with no report. With a ceiling it is a row
  that says `out of memory` at a limit we can state. macOS refuses
  `RLIMIT_AS`, so a local run is unbounded.
- **Baselines run from venvs built up front**, not `uv run --with`. The
  latter is right for building fixtures, as `reference_stores.py` does, and
  wrong here: the first invocation would put several hundred megabytes of
  wheel downloads into the measured wall time and uv's own memory into the
  measured peak. scanpy therefore never enters `uv.lock` or the image.

### Rules that keep the comparison honest

These live in `benchmarks/cases.py` as well, because this is the part that
decays fastest.

- **Use the best idiom the baseline has.** Comparing `export dataframe`
  against a full `read_h5ad()` is a strawman -- anndata reads just `obs` via
  `read_elem`. The good idiom is the primary row; the naive load may appear
  only as a clearly labelled second row.
- **Pin compression on both sides.** adata-cli forwards the source's
  settings; `write_h5ad` defaults to none.
- **`n/a` is a result.** Where scanpy has no equivalent, or
  `concat_on_disk` refuses, say which and why. An omitted row reads as an
  oversight.
- **Include the startup floor.** The CLI costs 0.3-1 s to import, and
  `import scanpy` 3-8 s; on a small tier that is the entire measurement.
- **Do not hide the rows where the baseline wins.** `_concat_csr` loops per
  row in Python and scipy's C `vstack` will often beat it on time at many
  times the memory. That trade is the argument for the tool.
- **Give the baseline everything it needs.** `concat_on_disk` imports `dask` to
  concatenate a dense element and fails outright without it, so the baseline
  environments install it. Measuring a library crippled by a missing optional
  dependency would be measuring our own setup.
- **Four commands are deliberately not benchmarked.** `export image`, `export dict`,
  `import image` and `import dict` have no library equivalent, so their rows would only
  ever read `n/a` while adding runtime to every tag. The complexity guards cover them.
- **Say whether the page cache was dropped.** A fixture written seconds ago
  is entirely in RAM, which understates streaming. The runner can drop it; a
  laptop usually cannot, and the report states which happened.

## Compatibility testing against real anndata releases

`test_anndata_versions.py` does not trust this repo's idea of the format. For
each release below it builds an environment with `uv`, writes a reference
store with that exact anndata, and then checks that the CLI reads it, and that
what the CLI writes can be reopened **by that same release**.

| Release | Interpreter | Pins | Why it is in the list |
|---|---|---|---|
| 0.8.0 | 3.11 | `pandas<2`, `numpy<2`, `zarr<3` | Introduced `encoding-type`/`encoding-version` |
| 0.9.2 | 3.11 | `pandas<2`, `numpy<2`, `zarr<3` | |
| 0.10.9 | 3.12 | `pandas<3`, `numpy<2`, `zarr<3` | |
| 0.11.4 | 3.12 | `pandas<3`, `zarr<3` | The index became a `nullable-string-array` group |
| 0.12.2 | 3.12 | `pandas<3`, `zarr>=3` | Zarr v3 |
| 0.13.3 | 3.12 | `zarr>=3` | Zarr v3 by default |

The pins matter: a modern pandas makes string columns a type the older
releases cannot write, and pandas 1.x has no wheels for Python 3.12. To add a
release, append to `RELEASES` in `tests/reference_stores.py`.

First run downloads those environments; afterwards `uv` serves them from cache
and the suite takes about a minute. To skip it without the marker:

```bash
ADATA_SKIP_VERSION_FIXTURES=1 uv run pytest
```

## Coverage

```bash
uv run pytest -m "not integration" --cov=adata --cov-report=term-missing
```

The floor is 90%. What remains uncovered is mostly defensive `except` branches
around backend calls that do not fail in practice.

---
title: "Testing"
parent: Reference
nav_order: 7
redirect_from:
  - /TESTING.html
---

# Testing

```bash
uv sync --extra dev
uv run pytest                      # everything
uv run pytest -m "not integration" # no environment building, ~2.5 min
uv run pytest -m perf              # just the cost guards
uv run pytest -m integration       # anndata releases, scanpy and PyTables
```

CI runs the unit suite on Python 3.12 and 3.13 with a **90% coverage floor**,
and the cost guards (`-m perf`), the anndata compatibility suite and the 10x
external-reader suite as separate jobs. A fresh-install job installs the package alone from a fresh
resolution, without `uv.lock`, as `pip install pyadata-cli` would. It
imports every module on the base dependencies alone, runs `adata convert`
with and without click beside typer, and runs the docs-accuracy suite with no
click installed. The locked suite cannot see a dependency the lock brings in
by accident: 0.7.0 to 0.17.2 imported click, which typer had stopped
installing, so `convert` crashed on every fresh install. A floors job runs
the unit suite once more, on both interpreters, with each direct dependency
at the lowest version `pyproject.toml` allows, after checking that each one
is at its floor and imports.

Each leg of the unit suite checks that it runs the Python it is named after
before testing anything. uv takes its interpreter from `.python-version`
(3.12) ahead of the one on `PATH`, so up to 0.17.4 the 3.13 leg ran the
suite on 3.12; the workflow now sets `UV_PYTHON` for each leg.

The workflows are tested too. Publishing, the image and the benchmark run
only on a tag, so a pull request never runs them. `test_release_pipeline.py`
reads them, works out which jobs publish in each kind of run, and runs their
scripts against stand-ins. The `workflow lint` job runs
[actionlint](https://github.com/rhysd/actionlint) and
[zizmor](https://docs.zizmor.sh) over them on every pull request, though not
inside a release, where an advisory published that day would stop it. The
`container image` job builds the Docker image and runs
`.github/scripts/smoke-image.sh` against it, on every pull request and inside
every release before PyPI has it.

## How the suite is organised

| File | What it covers |
|---|---|
| `test_elements.py` | The element layer — encodings, string dtypes, readers and writers — run against HDF5, Zarr v2 and Zarr v3 |
| `test_storage.py` | Backend detection, copying, Zarr versions, consolidated metadata |
| `test_formats.py` | `.npy`, `.mtx`, image and JSON export/import, including their failure modes |
| `test_validate.py` | Dimension validation for every axis-bearing path, including `raw/` |
| `test_invariants.py` | Relationships that must hold for any data: subsetting everything is the identity, split partitions exactly, concat undoes split |
| `test_anndata_roundtrip.py` | anndata writes the fixtures, reads back our output |
| `test_anndata_versions.py` | Compatibility with seven real anndata releases (see below) |
| `test_element_fidelity.py` | Elements anndata and scanpy write through every copy path (HDF5 to HDF5, HDF5 to Zarr, Zarr to HDF5, Zarr v2 and v3): scanpy's `rank_genes_groups` record arrays, the stores anndata 0.7.8 wrote (nested 0.1.0 frames, `ordered` on the labels), variable-length numbers and references, unusual attributes, Zarr v2 codecs, empty dataspaces and fill values; read back by anndata |
| `test_commands_phase2.py`, `test_commands_coverage.py`, `test_cli.py` | Command surfaces and error paths |
| `test_convert.py` | `convert` between every backend pair, read back by anndata; its refusals, staging, and the forwarding of recast's old flags |
| `test_recast.py` | `recast` against scipy: casts, transposes, density changes and every refusal |
| `test_recast_streaming.py` | `recast` on HDF5, Zarr v2 and Zarr v3 in blocks of a few rows and entries: every layout to every other, every dtype in every layout, matrices at any depth, index ranges with and without `--force`; against scipy and read back by anndata |
| `test_tenx.py` | 10x HDF5 import and export against Cell Ranger-shaped fixtures (v2, v3, multimodal, probe-based): mapping, filters, round trips, refusals |
| `test_mtx.py` | 10x / STARsolo MTX folders against hand-written fixtures (Cell Ranger 2 and 3, STARsolo, velocyto, GEO prefixes): mapping, entry orders, refusals that leave nothing, round trips |
| `test_cellbender.py` | CellBender 0.2 and 0.3+ output, full and filtered: each latent at its barcode, `--min-cell-probability`, `--raw` alignment, refusals |
| `test_parse.py` | Parse split-pipe runs, as v1.6.1 writes them: run, sample and DGE folders, `--sample`, `--all-samples`, `--unfiltered`, `--genome`, refusals |
| `test_loom.py` | loom import and export against loompy- and velocyto-shaped fixtures (spec 2 and 3): X choice, names, attributes, graphs, `--dense`, round trips |
| `test_text.py` | Text matrices in every layout anndata's `read_text` tells apart (corner, short and comment headers, no header, no row names), each delimiter, gzip, `--transpose` CSR and dense, dtypes, refusals by line, small blocks |
| `test_hdf5.py` | One dataset of a plain HDF5 file: the listing without `--key`, `read_hdf`'s `row_names` / `col_names` and other names, `--transpose` CSR and dense, slabs of chunks, `--from hdf5` on `.h5` |
| `test_cistopic.py` | pycisTopic pickles read through the allowlist: a CistopicObject's matrices, frames, model and projections, an imputed-features pickle, a pickle naming `os.system` refused unrun, the missing-extra hint |
| `test_pandas_frame.py` | Each pandas dtype written as anndata writes it, on every backend |
| `test_concat_axis_elements.py` | `concat` on obsm dataframes and varm, against `anndata.concat` for each join and merge strategy |
| `test_legacy_copies.py` | anndata 0.7's dataframes (categoricals as codes with an object reference into `__categories`) copied by `subset` (h5ad and Zarr), `convert` to Zarr v2/v3 and a MuData import: written as current `categorical` groups in a 0.2.0 frame, read back by anndata with their categories |
| `test_subset_categories.py` | `subset` and `split` narrow categoricals as anndata's subsetting does: unused categories dropped (on an axis kept whole too), order and `ordered` kept, `uns["<col>_colors"]` narrowed with them; compared with anndata's own subset |
| `test_mudata.py` | MuData read and written, as mudata lays it out (anndata's `write_elem`): each modality, `--mod`, global obs/obsm/obsp/uns onto a modality's cells out of order, clashes kept, pulled `<mod>:` columns, no maps (by name), a refused map, CSC/dense graphs and obsm frames, `.h5mu` and MuData Zarr sniffed apart from AnnData; export split by feature type, userblock, maps and masks, round trip |
| `test_scope.py` | SCope looms, read as pySCENIC's `export2loom` writes them and written with `--scope` (round trips, plain AnnData, refusals, markers): regulons as dataframes, embeddings and clusterings named from `MetaData`, plain or compressed `MetaData`, region looms, R's logicals and factors, renamed fields, small blocks |
| `test_table.py` | The streaming CSV reader against pandas: column types, quoting, row subsets, blocks, refusals |
| `test_mtx_codec.py`, `test_csr_build.py` | The streaming Matrix Market reader and writer against scipy, and the bucket sort they and `recast` share |
| `test_external_readers.py` | 10x conversions, HDF5 and MTX, Parse runs and looms, checked by scanpy (Parse's own recipe for Parse), anndata with loompy (the pySCENIC tutorial's recipe for SCope looms), anndata's `read_csv`, `read_text` and `read_hdf`, a pandas 1.5 pycisTopic pickle read back there, mudata (a file it writes read as it reads it; ours read by it), and PyTables in their own uv environments (`integration`) |
| `test_performance.py` | Complexity guards -- what an operation costs, not how long it takes (see below); its fixtures are in `perf_fixtures.py` |
| `test_benchmark_harness.py` | That the benchmark's measurement is trustworthy, and that each run compares with the release before it |
| `test_release_pipeline.py` | The workflows and the Dockerfile: which jobs publish in each kind of run, which hold a write token and what they run, pinned actions and base image, no untrusted value pasted into a script, each Python leg on its own Python |
| `test_subset_oracle.py` | `subset` and `split` against anndata's `adata[obs, var]`, element by element: X, every layer layout, obsm and varm of any rank and as dataframes, obsp, varp, raw; on HDF5, Zarr v2 and Zarr v3 |
| `test_subset_selection.py`, `test_select.py`, `test_split_names.py` | Name lists that repeat a name, queries on every column layout against pandas, missing labels apart from empty ones, and output names that never share a file |
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

`tests/perf_counters.py` patches these seams, all verified against the pinned
h5py 3.16.0 and zarr 3.4.1:

| Hook | Sees |
|---|---|
| `h5py.Dataset.__getitem__` | HDF5 reads |
| `h5py.Dataset.__setitem__` | HDF5 writes |
| `h5py.Group.create_dataset` | HDF5 writes made at creation |
| `zarr.Array.__getitem__`, `.oindex[...]` | Zarr reads, v2 and v3 alike |
| `zarr.Array.__setitem__`, `zarr.Group.create_array` | Zarr writes (bytes only) |
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

**Bytes, three ways.** Elements alone hid a 70x read amplification and a
shard rewritten on every block, so `IOCounts` also counts bytes:

| Field | Counts | Catches |
|---|---|---|
| `read_bytes`, `largest_read`, `largest_by_name` | bytes the reads returned; the largest one read | a block that `--chunk` does not bound, with no allocator involved |
| `decoded_bytes` | bytes of every whole chunk the reads touched | reads that decode a chunk to use a row of it |
| `written_bytes`, `largest_write` | bytes handed to writes | |
| `encoded_bytes` | bytes of every whole chunk the writes touched, and every whole shard of a sharded Zarr array | writes smaller than the unit, each rewriting all of it |
| `written_by_name`, `encoded_by_name` | the two above, per array | a writer's own output apart from its scratch |
| `store_get_bytes`, `store_set_bytes` | bytes Zarr's store fetched and stored | |

Decoded and encoded bytes are a model worked out from the selection, the chunk
grid and the codecs, not a measurement of the library, which is what keeps them
deterministic. It follows HDF5's chunk cache (per dataset, least recently used,
at the dataset's own size: 1 MiB in the stores `open_store` opens, 8 MiB by
default since HDF5 2.0), so appending a block at a time to a growing dataset
costs each chunk once. A chunk is read into the cache whole, filtered or not;
only an unfiltered chunk too big for the cache is read and written in place,
just the bytes asked for. Zarr caches nothing, and writing part of a shard
rewrites the shard.
`test_the_byte_counters_see_whole_chunks_decoded_and_rewritten` checks the
model against numbers worked out by hand.

**Resident memory.** `count_allocations` uses tracemalloc, which sees numpy's
buffers but not what HDF5, blosc or zarr's codecs allocate in C.
`peak_rss(argv)` runs `adata <argv>` in a fresh interpreter and returns how far
the command raised the operating system's high-water mark (`ru_maxrss`, read
once the CLI is imported and again when the command ends; on Linux `VmHWM`,
since `ru_maxrss` there starts at the parent's peak). A subprocess, because
the high-water mark only rises. It has a few megabytes of noise, so
guard it with generous ratios against inputs of tens of megabytes, as
`test_a_dense_cast_adds_a_block_of_resident_memory_not_the_matrix` does;
`test_peak_rss_sees_a_command_that_holds_the_matrix` checks that it sees
`--in-memory` hold the matrix.

**Known bypasses.** `Dataset.read_direct`, `np.asarray(dataset)` (it goes via
`__array__`), `dataset.asstr()[...]`, `.fields()` and Zarr's `vindex` reach
the file without passing any hook. None are used today. If you add one, the counters will
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
| `assert_memory_is_a_fraction_of_the_input` | adds at most a share (half) of what the input adds, in bytes | memory, along either axis |
| `assert_independent_of` | does not grow at all | inspection, grouping work per row, the largest read |

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

### Memory along every axis and layout

The guards above grow n_obs of an eight-column CSR store, the one shape every
path handles well. A CSC densify holding n_obs x `--chunk` values, a dense cast
check reading the whole matrix and a transpose stepping by the average major
all passed them. `tests/perf_fixtures.py` builds stores the way users have
them -- every layout, written by anndata as h5ad, Zarr v2 and anndata 0.13's
default sharded Zarr v3 -- and `test_peak_memory_is_a_small_share_of_the_input` runs
`recast` (each dtype and layout change), `subset` and `concat` over each
layout, growing n_obs and then n_var. zarr 3.4 gives anndata's shards inner
chunks of about a megabyte; `zarr31_layout` rewrites a store's X as zarr 3.1
laid it out, inner chunks of a kilobyte in shards of the whole array, which
is what users' stores hold and where a read or write that splits a unit
costs most.

A growth ratio cannot be the claim along n_var: widen a matrix and a block of
`--chunk` rows widens with it, as fast as the input, so a ratio cannot tell it
from the whole matrix. Peak memory added per byte of input added can: a block
of 32 rows of a matrix thousands of rows tall adds a few hundredths, and a path
holding the matrix adds a whole byte and more for each copy. The limit is half;
bounded paths came to at most 0.21. Growing n_obs, `--chunk` is 256 rows, so a
block of `--chunk` columns over every row (CSC densify before F030) is plain.
Each fixture is built once and shared by the commands.

Sparse `subset` was the known gap, read 4,096 majors a block whatever
`--chunk` said (F032); 0.17.6 closed it, and `GRID_GAPS` is empty.

The recast guards beside it pin each block in bytes, on h5ad and sharded Zarr
v3 (Zarr v2 is covered by the correctness tests): the cast
check reads `--chunk` rows of a dense matrix, CSC densify and sparsify read
blocks set by the chunk, and a skewed transpose never reads more than about two
buckets of entries at once.

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
| `export sparse` | n_obs | **flat** read calls: one per block, not per row |
| `import dataframe`, `array`, `sparse`, `dict`, `image` | rows, elements, nnz, keys, pixels | linear in writes |
| `import sparse` | n_obs | **flat** write calls |
| h5ad to zarr (`subset`), `convert` both directions | n_obs | linear |
| `convert` 10x import (whole, `--split-feature-types`), export | n_barcodes | linear |
| `convert` 10x MTX import (in order, shuffled, `--split-feature-types`) | n_barcodes | linear; **flat** write calls |
| `convert` to 10x MTX | n_obs | **flat** read calls |
| `convert` CellBender (plain, `--min-cell-probability`, `--raw` full and filtered) | n_barcodes | linear; **flat** write calls |
| `recast --dtype`, `--layout csc`, `--layout dense` | nnz, n_obs | linear; the streaming transpose's **peak memory** is bounded |
| `recast` (every dtype and layout change), `subset`, `concat` | n_obs, n_var; CSR, CSC, dense | **peak memory** a small share of the input |
| `recast --dtype` (dense), a skewed transpose | n_obs; h5ad, sharded Zarr v3 | the **largest read** set by the chunk |
| CSC `--layout dense`, dense `--layout csc` | n_obs; h5ad | the **largest read** set by the chunk, and memory per row |
| `subset`, `split`, `concat`, `recast`, `convert` into anndata 0.13's shards | Zarr v3 as zarr 3.1 wrote it | each shard **encoded once** |
| dense `subset`, `concat`, `recast`, `export array` | tall gzip chunks, zarr's own chunks | each chunk **decoded once** |
| `subset` into a sharded output | 1 MiB units; dense, CSR | **peak allocation** a few units, not the shard |
| `split` into chunked HDF5 | eight groups | each output chunk **encoded once** |
| `subset`, `concat` of a sharded store | Zarr v3 as zarr 3.1 wrote it | each shard index **fetched once** |
| sparse `subset` | n_obs; CSR, CSC, a gene in every cell | the **largest read** about `--chunk` rows of entries |
| `convert` of a matrix chunked (n_obs, 8) | n_obs | the **largest read** a tile within the budget |
| `split` | 128 against 512 groups | **resident memory** flat past 128 outputs a pass |

The `slow` marker is on the guards that build a 65,536-row store or run `peak_rss`;
they still gate merges, and `-m "not slow"` skips them locally. `perf` is on
every guard in the file, and on the few beside the code they guard; CI runs
them as a job of their own, beside the unit suite's coverage job.

h5py 3.15 kept 32 to 160 bytes of every dataset read, write and `.shape` for good, so
any long streaming loop grew with its block count;
`test_h5py_calls_keep_no_memory_once_they_return` keeps the floor at 3.16.0, which
does not. 3.16.0 brings HDF5 2.0, whose chunk cache is 8 MiB a dataset rather than 1;
`open_store` keeps 1 MiB, which
`test_stores_give_each_hdf5_dataset_a_one_mebibyte_chunk_cache` checks.

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

Results are published to [Benchmarks](benchmarks.md) and `docs/benchmarks/<tag>.json`
on every tag, and appended to the GitHub release notes if a release exists.
Artifacts expire; the docs page is the durable series. The "vs previous"
column compares with the release just before, at the same tier; a run that
is not a release compares with the newest one.

The job that measures holds a read-only token, because the baselines are the
newest anndata, scanpy, dask and geopandas from PyPI rather than anything
`uv.lock` pins. A second job, which installs nothing, renders the results
with the standard library and commits them.

### How it measures

- **Every command is forked from a small shim, not from the runner.** On Linux a
  forked child inherits its parent's resident pages and `execve` folds that into the
  reported `maxrss`, so a child of a process that has imported anndata cannot measure
  below roughly 200 MB — which would have flattened every row in the table. Measured
  with a 330 MB parent: 326 MB for a no-op child directly, 8 MB through the shim.
  `posix_spawn` does not help (329 MB), so dropping `preexec_fn` is not a fix.
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
what the CLI writes can be reopened **by that same release**: the output of
`subset`, of `convert` (to the same backend and to the other, `--zarr-format
2` for the releases on zarr-python 2), of `split` and of `concat` over its
pieces. The reference store holds a categorical in `raw/var` and scanpy's
`rank_genes_groups` record arrays besides the common encodings.

| Release | Interpreter | Pins (exact) | Why it is in the list |
|---|---|---|---|
| 0.7.8 | 3.10 | pandas 1.3.5, numpy 1.23.5, zarr 2.10.3, h5py 3.7.0 | The legacy layout: no root attributes, categoricals as codes with a reference, `ordered` on `__categories`. Read only: it predates the encoding `adata` writes |
| 0.8.0 | 3.11 | pandas 1.5.3, numpy 1.26.4, zarr 2.18.7 | Introduced `encoding-type`/`encoding-version`; the oldest release that reads what `adata` writes |
| 0.9.2 | 3.11 | pandas 1.5.3, numpy 1.26.4, zarr 2.18.7 | |
| 0.10.9 | 3.12 | pandas 2.3.3, numpy 1.26.4, zarr 2.18.7 | |
| 0.11.4 | 3.12 | pandas 2.3.3, numpy 2.5.4, zarr 2.18.7 | The first that reads a `nullable-string-array` (it still writes text as `string-array`) |
| 0.12.2 | 3.12 | pandas 2.3.3, numpy 2.5.4, zarr 3.4.2 | Zarr v3; the first that reads a `null` |
| 0.13.4 | 3.12 | pandas 3.1.0, numpy 2.5.4, zarr 3.4.2 | Zarr v3 by default; with pandas 3, text as `nullable-string-array` |

Every release runs on exact versions (scipy and h5py too). A modern pandas
makes string columns a type the older releases cannot write, pandas 1.x has
no wheels for Python 3.12, and a floating pin would change what the job
checks, or turn it red, without a change here. To add a release, append to
`RELEASES` in `tests/reference_stores.py`. The readers
`test_external_readers.py` compares against (scanpy, PyTables, loompy,
mudata, squidpy and the rest) are pinned the same way, in `ORACLES` there.

The unit suite needs no network to read anndata 0.7's layout:
`tests/fixtures/anndata-0.7.8/` holds the two stores the 0.7.8 environment
wrote (the Zarr one zipped), which `test_element_fidelity.py` copies through
every path. To write them again, run the fixture writer in that environment:

```bash
uv run --no-project --python 3.10 --with anndata==0.7.8 --with scipy==1.9.3 --with pandas==1.3.5 --with numpy==1.23.5 --with zarr==2.10.3 --with h5py==3.7.0 --with numcodecs==0.13.1 tests/fixtures/write_reference_store.py <dir> h5ad
```

and the same with `zarr` for the Zarr store.

Under pandas 3 anndata 0.13 writes every text column as a
`nullable-string-array`, while most files in the wild hold `string-array`
datasets. `test_anndata_roundtrip.py` builds its fixtures both ways, and
concatenates one of each against `anndata.concat`.

First run downloads those environments; afterwards `uv` serves them from cache
and the suite takes about a minute. To skip it without the marker:

```bash
ADATA_SKIP_VERSION_FIXTURES=1 uv run pytest
```

## Coverage

```bash
uv run pytest -m "not integration and not perf" --cov=adata --cov-report=term-missing
```

The floor is 90%. What remains uncovered is mostly defensive `except` branches
around backend calls that do not fail in practice.

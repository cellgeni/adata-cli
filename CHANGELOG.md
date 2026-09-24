# Changelog

Notable changes to `adata-cli`. Versions are `MAJOR.MINOR.PATCH`; tags carry no
`v` prefix.

## Unreleased

### Fixed

- **`subset`, `split` and `concat` wrote sparse matrices about twice as large
  as they needed to be.** Every sparse output widened `indices` and `indptr`
  to int64, whatever the source used, and was written without the source's
  compression, so with an lzf input the `data` and `indices` came out
  uncompressed. Outputs now keep the source's index dtypes and storage
  settings. `concat` widens to int64 only when the combined matrix could not
  be addressed otherwise. `convert` already did this; the code next to it
  did not.

- **`split` read the whole matrix once per group.** It ran a separate subset
  for each group. When groups were interleaved through the file, as samples
  usually are, each subset read nearly all of X, so a split into k groups
  read and decompressed X k times: 56.7 s against 7.8 s for a hand-written
  anndata loop in the 0.6.0 benchmark. `split` now reads each matrix once
  and shares every block out among all the outputs. It holds up to a quarter
  of the file-descriptor limit open at once, and makes one pass per batch
  beyond that. `subset` shares the same writer, which now gathers rows in
  one vectorised step instead of a Python loop over rows, and skips blocks
  that hold none of the selected rows. A new guard in
  `tests/test_performance.py` checks that split's reads of X do not grow
  with the number of groups.

  Dataframe columns are no longer read through an h5py fancy index, which
  cost 0.38 s per column per group on 50,000 cells. They are now read as
  contiguous blocks and selected in memory. With both changes, the ci-tier
  benchmark on a laptop splits in 1.6 s against anndata's 2.3 s.

## 0.6.0

Adds `adata convert`, and fixes four quadratic paths that made `concat` and
`split` appear to hang on real data. Three of the four were found by a new
suite of complexity guards, which now run on every pull request; the first
was reported from a pipeline that had to kill twelve tasks after 98 minutes.

### Added

- **`adata convert` changes a matrix's dtype, layout or density on disk**
  ([#13](https://github.com/cellgeni/adata-cli/issues/13)). Counts stored as
  float64 halve with `--dtype float32`; `--indices-dtype int32` halves the
  index arrays of a matrix small enough to address that way; `--layout
  csr|csc` transposes between the sparse encodings; `--layout dense|sparse`
  changes the density. Works on `X`, any layer, `raw/X` or any 2-D array by
  path, and on all of them at once with `--all`.

  A cast that would not round-trip is refused before anything is written --
  every value is cast and cast back, because whether float64 counts survive
  float32 depends on the counts, not on the dtypes. So is a densification
  that would inflate the store beyond four times its size. `--force`
  overrides either, and both messages say what they measured.

  Transposing streams by default, holding one bucket of nonzeros rather than
  the matrix, so it works on files too large to load; `--in-memory` is
  faster when the matrix fits. Buckets are balanced by nonzero count rather
  than by coordinate range, because a single-cell matrix is skewed -- a few
  genes carry most of the counts -- and equal-width bounds put most of one
  in a single bucket.

  `--indices-dtype` is checked against both what `indices` must address and
  what `indptr` must reach, which differ: a narrow matrix with more than
  2^31 nonzeros needs int64 offsets over int32 coordinates. Both are
  preserved from the source when not specified.

- **`concat` now names the command to run** when inputs disagree about a
  matrix encoding. The check itself is not new, but nothing tested it and it
  could not suggest a fix, because there was none.

- **Complexity guards in the test suite** (`tests/test_performance.py`).
  Cost regressions now fail at merge time. They count operations rather than
  seconds -- h5py and zarr reads, Zarr store traffic, Python allocation and
  executed lines -- and assert that successive increments grow no faster than
  linearly, so nothing here can fail because a CI runner was busy. See
  [docs/TESTING.md](docs/TESTING.md#performance). Every subcommand is covered:
  `ls`, `view`, `create`, all five `export` and all five `import` variants,
  `split` on both axes, and the `concat` options nothing else reached.
  Two claims are now enforced rather than described -- `view` and `ls` read
  **zero** data elements at any store size, and streaming stays far below the
  input curve at a fixed `--chunk`.

- **A comparative benchmark** (`benchmarks/`), run on every tag against
  anndata and against scanpy where scanpy has a real equivalent. Reports peak
  RSS, wall time and output size; publishes to
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and the release notes. Report-only
  -- it never fails a build. Fifteen cases, covering every command with a real
  baseline, including `h5ls -r` for `ls` and the rows where adata-cli is the
  slower of the two.

- **`--merge drop` and `--uns-merge drop` are accepted.** `drop` was already
  the documented default behaviour but was rejected as a value, so a config
  could not state it explicitly.

### Fixed

- **`concat --merge` never finished on a real store.** Aligning a var column
  onto the target index re-read the whole column from disk once per target
  variable, so the cost was quadratic: at 36,601 variables a merge that should
  take a fraction of a second ran for hours at 100% CPU with the output file
  never growing past its header. Reported against 0.5.1 (REQ-71798), where 12
  of 13 pipeline tasks had to be killed after 98 minutes. The column is now
  read once per input, and `--merge first` / `--merge only`, which decide on
  presence alone, read no column values at all.

- **`concat` was quadratic in the number of categories** in an obs column.
  Category merging probed a list rather than a dict: 2,096,128 string
  comparisons to union 1,024 categories, and around 5e9 for a 100k-category
  column. Found by the new guards.

- **`split --by` was quadratic**, O(n_rows x n_groups). `group_indices` grouped
  rows with `np.nonzero(values == label)` inside a loop over distinct labels,
  rescanning each chunk once per label: at 4,096 rows, 16,384 elements scanned
  for 4 groups and 1,048,576 for 256. A million cells split by a thousand
  samples is ~10^9 comparisons. One `np.unique` pass per chunk makes it flat
  in the group count. Found by the new guards; order of first appearance,
  which names the output files, is unchanged.

- **`concat` built a Python object per row** for nullable and string obs
  columns, then walked the list twice more. Filling a typed buffer by slice
  removes three full passes over every such column.

- **Copying variable-length strings ignored its own read budget.** The width
  of a vlen element was assumed to be 64 bytes, because h5py reports the
  itemsize of a pointer, so the step was the same 524,288 elements whatever
  the data held: 2 GiB per read at 4 KiB elements against a stated 32 MiB
  budget, and for any array shorter than that step, the whole array in one
  go. Copying 200,000 strings of 4 KiB peaked at 827 MB. The width is now
  sampled from the first 256 elements. Reported by an automated review on
  PR #14 and confirmed by measurement; `uns` can hold arbitrary text, so this
  was not a width the layer could assume.

- **Peak RSS in the benchmark was floored by the runner's own memory on Linux.**
  A forked child inherits its parent's resident pages and `execve` folds that
  into the `maxrss` the kernel reports, so every contender would have measured
  at least what `benchmarks/run.py` used to build the fixtures — around
  200 MB — and the tables would have read "everything costs about the same".
  Commands are now forked from a small shim: with a 330 MB parent, a no-op
  child goes from 326 MB to 8 MB. Caught by `test_benchmark_harness.py`, which
  exists for exactly this. The published figures were measured on macOS, which
  resets the high-water mark at exec, and are unchanged.

## 0.5.1

Makes the container image usable from Nextflow, and stops `copy_dataset`
reading one row at a time from row-chunked stores.

### Fixed

- **Copying a row-chunked store was dominated by read latency.** The read step
  was the source's chunk height verbatim, so a store chunked `(1, n_cols)` was
  copied one row per read. On a network filesystem (Lustre, NFS) each read is a
  round-trip, so a million-row copy spent nearly all of its time waiting. Reads
  are now sized to a 32 MiB budget, rounded down to a whole number of source
  chunks. A `(1_000_000, 30_000)` float32 store chunked `(1, 30_000)` goes from
  1 row per read to 279.
- Read sizing no longer trusts `itemsize` for variable-length strings. h5py
  reports 8 there because the value is a pointer, which overestimated the row
  count by an order of magnitude and broke the memory bound.

### Container

- **The image could not be used from a Nextflow process.** Nextflow requires
  `/bin/bash` to be the container entrypoint, so `ENTRYPOINT ["adata"]` made
  every Docker- and Podman-backed task fail with `No such command
  '/bin/bash'`. Apptainer was unaffected, as `singularity exec` ignores the
  entrypoint.
- **Task metrics were silently lost.** `procps` is absent from the base image,
  so Nextflow could not run `ps` to collect them. The required tool set
  (`bash`, `ps`, `awk`, `date`, `grep`, `sed`, `tail`, `tee`) is now installed
  and asserted at build time.
- `PYTHONNOUSERSITE` is set, so a bind-mounted `$HOME` under Apptainer can no
  longer shadow the image's virtualenv with the user's `~/.local` packages.
- `XDG_CACHE_HOME` points at `/tmp`, so the image tolerates being run under an
  arbitrary UID with no writable `$HOME`.
- Added a `.dockerignore`. Local builds were copying the host's `.venv`,
  `.git` and `.pytest_cache` into the image.

### Changed

- **The image no longer sets an entrypoint, so the command must be named
  explicitly:** `docker run IMAGE adata view file.h5ad`, where `docker run
  IMAGE view file.h5ad` previously worked.

## 0.5.0

Renamed from `h5ad` to `adata-cli`, restored compatibility with current
AnnData files, and added four commands.

### Renamed

- Distribution `pyadata-cli` on PyPI (the short name was already taken by
  an unrelated project), import package `adata`, command `adata`.
- `info` is now `view`.
- The `h5ad` command and the `info` subcommand remain as aliases that warn and
  then run normally. **Both are removed in 1.0.0.**

### Fixed

- **The CLI could not read any file written by anndata >= 0.11.** Since pandas'
  `future.infer_string` became the default, anndata writes `obs/_index` as a
  `nullable-string-array` group rather than a dataset, and `axis_len` required
  a dataset — so `info`, `export dataframe` and `subset` all failed outright.
- **HDF5 -> Zarr conversion failed on any store with string columns**, i.e. all
  of them: an h5py variable-length string dataset reports `dtype == object`,
  which Zarr rejects. All four backend pairings now convert.
- **`subset` silently dropped `raw/`.** It is now carried over and matched
  against its own var axis. Unrecognised top-level keys are copied with a
  warning rather than dropped.
- **`subset` corrupted group-valued columns**, copying them whole while
  narrowing everything else. This was invisible before, because files
  containing such columns could not be read at all.
- Categoricals no longer degrade to plain strings on a CSV round-trip.
- `None` round-trips via anndata's `null` encoding instead of an invented
  `_is_none` marker attribute.
- `column-order` is honoured on export; previously columns came out in
  whatever order the backend enumerated (alphabetical on HDF5).
- Type detection dispatches on `encoding-type` before falling back to
  structure. A group merely *containing* a member named `obs_names` is no
  longer misreported as a dataframe.
- `subset` prefers the declared `_index` over the `obs_names`/`var_names`
  convention, which had been backwards.
- A Zarr v2 store is no longer silently upgraded to v3.
- Chunk shapes are clamped on the axis-column path, which could raise from
  h5py when subsetting below a column's chunk size.

### Added

- `adata ls` — tree listing for any HDF5 or Zarr store, with no AnnData
  assumptions, so `.loom` and plain `.h5` work. `--long`, `--depth`, and `-1`
  for bare paths that pipe into other tools.
- `adata create` — write a new, empty store for `import` to fill in.
- `adata split --by <column>` — one store per distinct value, with a CSV
  manifest. (#2)
- `adata concat` — concatenate along obs with `--join inner|outer`,
  `--label`/`--keys`/`--index-unique` and merge strategies for var and uns.
  Verified to agree with `anndata.concat`.
- `adata subset --obs-query` / `--var-query` — a small predicate language
  (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `and`, `or`, `not`,
  parentheses) evaluated while streaming. No new dependency.
- `adata import image`, and `import` at any path rather than only `obs`/`var`.
- `--categorical` / `--no-auto-categorical` on `import dataframe`.
- `export dataframe` from any dataframe path, not only `obs`/`var`. (#4)
- `export array` can write to stdout. (#4)
- `adata --version`.
- `--zarr-format` on the commands that create stores.

### Changed

- Results go to stdout and status to stderr throughout. `view --tree`
  previously wrote its tree to stderr and only the header to stdout. (#4)
- Everything written is tagged with its `encoding-type` and
  `encoding-version`; text is always variable-length UTF-8, as the spec
  requires, rather than fixed-width bytes.
- Sparse subsetting streams in blocks instead of loading `data`/`indices`/
  `indptr` whole.
- The Docker image ships `duckdb` in place of `csvkit`.
- CI runs the whole suite on Python 3.12 and 3.13 rather than naming test
  files individually — which is why `test_storage_root_attrs.py` had never
  run.

## 0.3.2 and earlier

See the git history. These releases were published to Quay only.

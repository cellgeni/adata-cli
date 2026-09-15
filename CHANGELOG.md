# Changelog

Notable changes to `adata-cli`. Versions are `MAJOR.MINOR.PATCH`; tags carry no
`v` prefix.

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

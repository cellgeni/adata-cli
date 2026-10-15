# AGENTS.md

Guidance for coding agents working in this repository. `CLAUDE.md` imports this
file. Edit this one, not that.

`adata` is a streaming CLI for AnnData stores (`.h5ad` and `.zarr`). It reads
and writes the on-disk format with h5py and zarr directly. It does not depend on
anndata: anndata appears only in tests and benchmarks, as the reference. The
point of the tool is that memory is set by `--chunk`, not by the size of the
input, so code that loads a whole matrix or column is a regression unless it is
behind an explicit `--in-memory`.

Naming:
- The PyPI distribution is `pyadata-cli`.
- The import package and the command are `adata`.
- The repo directory is still called `h5ad-cli`.
- `h5ad` is a deprecated alias of the command (`main_deprecated` in `cli.py`).
- `convert` given recast's old flags forwards to `recast` with a warning.

Both of these deprecated paths stay until 1.0.0.

## Commands

```bash
uv sync --extra dev                                   # set up
uv run adata --help                                   # run the CLI from source
uv run pytest -m "not integration"                    # unit suite, ~2.5 min; what CI gates on
uv run pytest tests/test_recast.py::test_inplace_replaces_the_source   # one test
uv run pytest -m "not integration" --cov=adata --cov-fail-under=90     # CI's coverage floor
uv run pytest -m perf                                 # tracing complexity guards only
ADATA_REQUIRE_VERSION_FIXTURES=1 uv run pytest tests/test_anndata_versions.py -m integration
uv run python -m benchmarks.run --tier smoke --case concat-inner --out results.json
uv run python -m benchmarks.report results.json
```

Test markers:
- `integration` tests build uv environments with real anndata releases (0.8 to
  0.13), scanpy and PyTables, so they need the network on the first run.
- `perf` tests are slow, but CI runs them in the main job.

Neither CI nor the repo runs a linter or formatter. Match the style of the
surrounding code.

## Architecture

Each layer calls only the layers below it:

- **`cli.py`**: the whole Typer app, including the `export` and `import` sub-apps.
  - Commands parse options, validate cheap things such as `--zarr-format`, call
    into `commands/`, and turn exceptions into `Error:` plus exit code 1.
  - `console` writes to **stderr**. Only `out_console` and data go to stdout,
    so output pipes cleanly.
- **`commands/`**: one module per command. Each orchestrates its command:
  - planning and refusals;
  - progress output;
  - choosing the output Zarr version;
  - staging outputs.
- **`core/`**: the streaming algorithms:
  - `subset.py` holds a fan-out writer shared by `subset` and `split`, which
    reads each matrix once for every output;
  - `concat.py` concatenates on disk;
  - `recast.py` handles dtype, layout and density changes, including an
    out-of-core transpose;
  - `csr_build.py` builds a CSR matrix from entries in any order, through
    row-range buckets on disk. The transpose uses it, and so does anything
    that reads entries out of order;
  - `query.py` and `select.py` hold the expression language behind
    `--obs-query`.
- **`elements/`**: the AnnData on-disk spec.
  - `spec.py` is the single source of truth for `encoding-type` and
    `encoding-version`.
  - `read.py` accepts every layout from anndata 0.7 to the current one.
  - `write.py` always writes the current one.
  - `strings.py` reconciles how HDF5 and Zarr spell text.
  - The full spec is in `docs/ELEMENTS_h5ad.md` and `docs/ELEMENTS_zarr.md`.
- **`storage/`**: hides whether a store is h5py or zarr.
  - `open_store(path, mode, zarr_format=...)` detects the backend from the
    path and returns a `Store`.
  - On close, a writable Zarr `Store` rewrites the consolidated metadata.
    Without that, anndata reads a stale index and new members look missing.
  - Use the `is_group`/`is_dataset`/`create_dataset`/`copy_tree` helpers here
    rather than branching on the backend yourself.
- **`conversion/`**: `adata convert`. AnnData is the hub, so each format has an
  importer, an exporter, or both, and there are no converters between pairs of
  other formats. To add a format:
  - add one `Format` entry in `conversion/__init__.py::FORMATS` with a content
    sniffer;
  - write a converter module (`tenx.py` is the model). The feature table that
    every 10x-style format shares (var names, feature-type splits, var to
    id/name/type on export) is in `features.py`; reuse it;
  - add Cell Ranger-style h5py fixtures in `tests/conftest.py`;
  - add a scanpy comparison in `tests/test_external_readers.py`.

  Content sniffing runs before extensions, because extensions lie (`.h5`).
  A format with no extension, such as an MTX folder (`10x-mtx`), is chosen as
  an output only with `--to`.
- **`formats/`**: the per-file-type halves of `export` and `import`: npy, mtx,
  csv, json and images. `mtx.py` is a streaming Matrix Market codec (two
  passes in, block writes out) shared by `import`/`export sparse` and the
  `10x-mtx` converter; `textio.py` holds its gzip-aware block I/O.
- **`util/`**: `staging.py` and `path.py` (below), plus `names.py`.

## Rules the code depends on

These are easy to break without any test noticing straight away:

- **Never write a store in place.** Every command that writes a store wraps the
  write in `util/staging.staged_outputs([...])`.
  - It writes to `out.adata-tmp.<ext>` and renames into place only when the
    write succeeds.
  - It moves an existing output aside to `*.adata-old.*` and deletes it only
    once every new output is in place.
  - It refuses to start if leftovers from an interrupted run exist.
  - `--inplace` passes the source itself as the output. Open the source
    *inside* the staging block, so it is closed before it is renamed aside.
- **Refuse an output that names the input** with `util/path.same_path`. With
  Zarr, opening the output `"w"` clears the directory that is still being read.
- **Validate before the output exists.** Resolve selections, check casts and
  build plans first, so that a refusal writes nothing.
- **Zarr outputs keep the source's version.**
  - Pass `zarr_format=` to `open_store`. Without it zarr-python writes v3, and a
    v2 input gets silently upgraded.
  - Commands read the source's version from `Store.zarr_format`.
  - `concat` uses the version its Zarr inputs share.
- **Outputs keep the source's index dtypes, chunking and compression.**
  Changing them is `recast`'s job, or that of an explicit flag.

## Tests

- `tests/conftest.py::new_store` is parametrised over `h5ad`, `zarr2` and
  `zarr3`. Write element-level tests against it, so one test body covers all
  three backends.
- CLI tests use a module-level `CliRunner`. Strip ANSI codes and collapse
  whitespace before matching messages, because Rich wraps them. The CLI test
  files define an `_out(result)` helper for this.
- Correctness is checked against an oracle: anndata reads back what we write,
  and scipy checks `recast`. Self-consistency alone is not accepted.
- `tests/test_performance.py` guards complexity by **counting** elements and
  bytes read through hooks on h5py and zarr (`tests/perf_counters.py`). It never
  uses a clock. A change that adds reads per row or per group will fail it.
- `tests/test_docs_are_accurate.py` parses every `adata …` line in the README,
  `docs/` and the `--help` examples, and checks it against the real Click tree.
  A new or renamed option has to be documented correctly in
  `docs/COMMANDS.md`. A new command also needs a section there, an entry in the
  table of contents and a mention in the app help.
- House style:
  - test names read as sentences
    (`test_a_failed_selection_keeps_an_existing_output`);
  - a regression test's docstring says what used to go wrong;
  - comments explain *why*.

## Releases

- Use a branch `release/X.Y.Z`.
- Set the version in `pyproject.toml` and in the `pyadata-cli` entry of `uv.lock`.
- Add a `## X.Y.Z` section to `CHANGELOG.md`, in its style:
  - a summary paragraph;
  - `### Added`, `### Changed` and `### Fixed` sections;
  - entries that start with a bold one-line lead and an issue link.
- Merge the release branch, then push a tag with no `v` prefix. The tag
  triggers:
  - `publish.yml`, which checks that the tag matches pyproject, tests, builds
    and publishes to PyPI;
  - `quay-on-tag.yml`, which pushes the Docker image to
    `quay.io/cellgeni/adata-cli`;
  - `benchmark.yml`, which reruns `benchmarks/` and commits the results to
    `docs/BENCHMARKS.md` and `docs/benchmarks/<tag>.json`.
- `docs/BENCHMARKS.md` is generated. To change its wording, edit
  `benchmarks/page_template.md`.

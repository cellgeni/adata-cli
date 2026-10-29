---
title: Developers
nav_order: 6
---

# Developers
{: .no_toc }

1. TOC
{:toc}

## Setting up

```bash
git clone https://github.com/cellgeni/adata-cli.git
cd adata-cli
uv sync --extra dev
uv run adata --help
uv run pytest -m "not integration"     # the unit suite CI gates on, a few minutes
```

The tool itself depends on h5py, zarr, numcodecs, numpy, Pillow, Typer and
Rich; the `pickle` extra adds pandas and scipy, and the `spatial` extra
pyarrow. A module `src/adata` imports gets a line in `pyproject.toml`, with a
floor: CI's floors job installs each direct dependency at its floor and runs
the unit suite on them.
anndata, pandas, scipy and pyarrow come with the `dev` extra, as the
reference the tests check against; scanpy, loompy, mudata and older anndata releases are
installed by the `integration` tests in environments of their own.

## Repository layout

| Path | Contents |
|---|---|
| `src/adata/cli.py` | The Typer app: every command's options and help, and errors turned into `Error:` and exit code 1 |
| `src/adata/commands/` | One module per command: planning, refusals, progress, staging the outputs |
| `src/adata/core/` | The streaming algorithms: the subset fan-out writer, on-disk concat, recast and its out-of-core transpose, the CSR bucket builder, the query language |
| `src/adata/elements/` | The AnnData on-disk spec: encodings, readers for every layout since anndata 0.7, writers of the current one |
| `src/adata/storage/` | One interface over h5py and zarr |
| `src/adata/conversion/` | `adata convert`: one module per format, through AnnData |
| `src/adata/formats/` | The file types of `export` and `import`: npy, Matrix Market, CSV, JSON, images; and what the converters read: Parquet, GeoJSON polygons |
| `src/adata/util/` | Output staging, path checks, names |
| `tests/` | The test suite; [Testing](reference/testing.md) describes it |
| `benchmarks/` | The comparative benchmark against anndata and scanpy |
| `docs/` | This site |
| `Dockerfile` | The image published to `quay.io/cellgeni/adata-cli` |
| `.github/workflows/` | Tests, PyPI publishing, the image, the benchmark |
| `AGENTS.md` | The same ground as this page, written for coding agents |
| `STATUS.md` | What is released, in progress and next |

## The layers

Each layer calls only those below it:

```
cli.py ──► commands/ ──► core/, conversion/, formats/ ──► elements/ ──► storage/
```

- **`cli.py`** parses options and checks cheap things such as `--zarr-format`.
  Its `console` writes to stderr; only data and `out_console` go to stdout,
  which is what lets output pipe cleanly.
- **`commands/`** orchestrates one command: what to read, what to refuse, what
  to write where, and which Zarr version.
- **`core/`** holds the algorithms that stream. `subset.py`'s writer serves
  `subset` and `split`, and reads each matrix once for every output.
  `csr_build.py` builds CSR from entries in any order through row-range buckets
  on disk; the transpose and every out-of-order reader use it. `scatter.py`
  does the same for the rows of a table that comes in another order than the
  matrix beside it.
- **`elements/spec.py`** is the single source of truth for `encoding-type` and
  `encoding-version`. `read.py` accepts every layout from anndata 0.7 on;
  `write.py` always writes the current one; `strings.py` reconciles how HDF5
  and Zarr spell text.
- **`storage/`** hides the backend. `open_store(path, mode, zarr_format=...)`
  returns a `Store`; use its `is_group`, `is_dataset`, `create_dataset` and
  `copy_tree` rather than branching on h5py or zarr.

### Adding a format to `convert`

AnnData is the hub, so a format needs an importer, an exporter or both, never
a converter to another format.

1. Add a `Format` to `FORMATS` in `conversion/__init__.py`, with a content
   sniffer. The first sniffer to match wins, so a specific format
   (`cellbender`, which is also a 10x matrix) goes before a general one, and
   the folder formats (`visium`, `xenium`) go before `10x-mtx`, which would
   otherwise claim their MEX subfolders.
2. Write the converter module; `tenx.py` is the model, and `features.py` holds
   the feature table every 10x-style format shares. A format that pairs a 10x
   matrix with a table (`visium.py`, `xenium.py`) uses `spatial.py` to match
   the table to the matrix's barcodes and write it in their order, and to
   read the `analysis/` folder Xenium and Space Ranger (spots, bins and
   cells) share.
3. Add fixtures shaped like the real tool's output to `tests/conftest.py`, and
   a comparison against the reader people use in
   `tests/test_external_readers.py`.
4. Give it a page under `docs/reference/formats/` and a row in the tables on
   the formats index and the `convert` section of
   [Commands](reference/commands.md#convert).

## Rules the code relies on

None of these fails a test at once when broken, so they are worth knowing
before changing anything that writes:

- **Never write a store in place.** Every write goes through
  `util/staging.staged_outputs`: write to `out.adata-tmp.<ext>`, move an
  existing output aside to `out.adata-old.<ext>`, rename into place, then
  delete the old one. It refuses to start over leftovers. With `--inplace`
  the source is the output, so open it *inside* the staging block, where it
  is closed before being renamed aside.
- **Refuse an output that names the input**, with `util/path.same_path`.
  Opening a Zarr output for writing empties the directory being read.
- **Validate before the output exists.** Resolve selections, check casts and
  plan, so that a refusal writes nothing.
- **Zarr outputs keep the source's version.** Pass `zarr_format=` to
  `open_store`; without it zarr-python writes v3, upgrading a v2 input.
  `concat` uses the version its Zarr inputs share.
- **Outputs keep the source's index dtypes, chunking and compression.**
  Changing them is `recast`'s job, or an explicit flag's.
- **Memory is set by `--chunk`.** Loading a whole matrix or column is a
  regression unless it is behind an explicit `--in-memory`.
- **A writable Zarr store rewrites its consolidated metadata on close.**
  anndata trusts that index, and new members are invisible without it.

## Tests

- **Every backend in one test body.** `tests/conftest.py::new_store` is
  parametrised over `h5ad`, `zarr2` and `zarr3`.
- **An oracle, not self-consistency.** anndata reads back what is written,
  scipy checks `recast`, scanpy, loompy and mudata check the conversions.
  `test_anndata_versions.py` builds environments with anndata 0.8 to 0.13 and
  checks both directions.
- **Complexity guards that count, not time.** `tests/test_performance.py`
  counts the elements and bytes read through hooks on h5py and zarr. A change
  that adds reads per row or per group fails it on any machine.
- **The docs are tested.** `tests/test_docs_are_accurate.py` checks every
  `adata …` line in the README, these pages and the `--help` examples against
  the real Click tree; that each command has a section and a contents entry
  in [Commands](reference/commands.md) and a mention in the app help; that
  every relative link names a page and an anchor that exist; and that every
  page has the front matter the sidebar needs.

```bash
uv run pytest -m "not integration" --cov=adata --cov-fail-under=90   # CI's coverage floor
uv run pytest -m perf                                                # the complexity guards
ADATA_REQUIRE_VERSION_FIXTURES=1 uv run pytest tests/test_anndata_versions.py -m integration
```

`integration` tests need the network on their first run, to build their
environments. House style: test names read as sentences, a regression test's
docstring says what used to go wrong, and comments say why. No linter or
formatter runs; match the code around you.

[Testing](reference/testing.md) has the detail: what each file covers, what the
guards count and why, and the anndata releases checked.

## The documentation site

The site is [Just the Docs](https://just-the-docs.com), built by GitHub Pages
from `docs/` on `main`. Each page needs `title` and `nav_order` front matter,
and `parent` (and `grand_parent` under Formats) to sit in the sidebar. Link
between pages as `page.md`, relative to the page; anything else needs an
absolute URL, since pages are served at `/page/`. A page that moves keeps its
old URL with `redirect_from`.

To preview it, with Ruby and Bundler:

```bash
cd docs
bundle install
bundle exec jekyll serve      # http://localhost:4000/adata-cli/
```

The benchmark page, `reference/benchmarks.md`, is generated: edit
`benchmarks/page_template.md` instead.

## Benchmarks

```bash
uv run python -m benchmarks.run --tier smoke --case concat-inner --out results.json
uv run python -m benchmarks.report results.json
```

The `Benchmark` workflow runs the `ci` tier on every tag and commits the
results to [Benchmarks](reference/benchmarks.md) and `docs/benchmarks/<tag>.json`.
It reports and never gates; the complexity guards gate.

## Releases

1. Branch `release/X.Y.Z`.
2. Set the version in `pyproject.toml` and in the `pyadata-cli` entry of
   `uv.lock`.
3. Add a `## X.Y.Z` section at the top of `docs/changelog.md`: a summary
   paragraph, then whichever of `### Added`, `### Changed` and `### Fixed`
   apply, each entry led by a bold line and an issue link.
4. Merge, then push a tag with no `v` prefix. The tag runs `publish.yml` (checks
   the tag against `pyproject.toml`, tests, publishes to PyPI),
   `quay-on-tag.yml` (the image) and `benchmark.yml` (the benchmark page).

`h5ad` as a command name, `info` and `convert` with `recast`'s flags are
deprecated and stay until 1.0.0.

# Testing

```bash
uv sync --extra dev
uv run pytest                      # everything
uv run pytest -m "not integration" # fast: no environment building, ~40s
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

---
title: "On-disk spec: Zarr"
parent: Reference
nav_order: 5
redirect_from:
  - /ELEMENTS_zarr.html
---

# AnnData on-disk element specifications — Zarr (`.zarr`)

This document describes how *elements* are encoded inside an AnnData **Zarr** container (`.zarr`).  
It is intended to be GitHub-renderable Markdown (no Sphinx/MyST directives).

> **Scope**
>
> - “Modern” encoding metadata (`encoding-type`, `encoding-version`) is the convention used by **anndata ≥ 0.8**.
> - “Legacy” conventions (notably DataFrame categorical handling) are described for **anndata 0.7.x** files, which are still commonly encountered.

## Table of contents

- [Encoding metadata](#encoding-metadata)
- [AnnData group](#anndata-group)
- [Dense arrays](#dense-arrays)
- [Sparse arrays (CSR/CSC)](#sparse-arrays-csrcsc)
- [DataFrames](#dataframes)
  - [DataFrame v0.2.0](#dataframe-v020)
  - [DataFrame v0.1.0 (legacy: anndata 0.7.x)](#dataframe-v010-legacy-anndata-07x)
  - [Legacy categorical columns (Series-level)](#legacy-categorical-columns-series-level)
- [Mappings](#mappings)
  - [The `dict` encoding (v0.1.0)](#the-dict-encoding-v010)
- [Scalars](#scalars)
- [Categorical arrays](#categorical-arrays)
- [String arrays](#string-arrays)
- [Nullable arrays](#nullable-arrays)
  - [Missing value semantics](#missing-value-semantics)
- [Awkward arrays (experimental)](#awkward-arrays-experimental)
- [Sources](#sources)

## Encoding metadata

**Modern convention (anndata ≥ 0.8):**

- Any element (Zarr *group* or *array*) that participates in the element-dispatch system:
  - **MUST** have attribute `encoding-type` (string)
  - **MUST** have attribute `encoding-version` (string, parseable as a version)

Readers should dispatch first on `encoding-type`, then on `encoding-version`.

**Legacy convention (anndata ≤ 0.7.x):**

- Many objects do *not* have `encoding-type`/`encoding-version`.
- Some elements (e.g. CSR/CSC sparse matrices, legacy DataFrames) *do* use `encoding-type`/`encoding-version`.
- Readers typically infer element kinds from:
  - known AnnData keys (`X`, `obs`, `var`, …),
  - group structure, and/or
  - legacy attributes (e.g. the `categories` attribute on categorical columns).

## AnnData group

### `encoding-type: anndata`, `encoding-version: 0.1.0`

An `AnnData` object **MUST** be stored as a Zarr **group** with attributes:

- `encoding-type: "anndata"`
- `encoding-version: "0.1.0"`

Required members:

- `obs` — a [DataFrame](#dataframes)
- `var` — a [DataFrame](#dataframes)

Optional members (if present, they must satisfy these constraints):

- `X` — dense array or sparse array; shape `(n_obs, n_var)`
- `layers` — mapping; values dense or sparse arrays; each shape `(n_obs, n_var)`
- `obsm` — mapping; values dense arrays, sparse arrays, or dataframes; first dim `n_obs`
- `varm` — mapping; values dense arrays, sparse arrays, or dataframes; first dim `n_var`
- `obsp` — mapping; values dense or sparse arrays; first two dims `n_obs`
- `varp` — mapping; values dense or sparse arrays; first two dims `n_var`
- `uns` — mapping/dict-like container (recursive)

## Dense arrays

### `encoding-type: array`, `encoding-version: 0.2.0`

- A dense array **MUST** be stored as a Zarr **array**.
- The array **MUST** have attributes:
  - `encoding-type: "array"`
  - `encoding-version: "0.2.0"`

> **Legacy note**
>
> In anndata 0.7.x, dense arrays were typically stored as plain Zarr arrays *without* `encoding-type`/`encoding-version`.

## Sparse arrays (CSR/CSC)

### `encoding-type: csr_matrix|csc_matrix`, `encoding-version: 0.1.0`

A sparse matrix **MUST** be stored as a Zarr **group**.

- Group attributes:
  - `encoding-type: "csr_matrix"` **or** `"csc_matrix"`
  - `encoding-version: "0.1.0"`
  - `shape`: integer array of length 2 (matrix shape)
- Group members (arrays):
  - `data`
  - `indices`
  - `indptr`

The exact CSR/CSC semantics follow SciPy’s conventions.

## DataFrames

DataFrames are stored column-wise: each column is stored as a Zarr array (or group, if the column itself is an encoded element).

<a id="dataframe-v020"></a>
### DataFrame v0.2.0

#### `encoding-type: dataframe`, `encoding-version: 0.2.0`

A dataframe **MUST** be stored as a Zarr **group**.

- Group attributes:
  - `_index`: string — the key of the array to be used as the row index
  - `column-order`: array of strings — original column order
  - `encoding-type: "dataframe"`
  - `encoding-version: "0.2.0"`
- Group members:
  - the index array (named by `_index`)
  - one member per column
- All column entries **MUST** have the same length in their first dimension.
- Columns **SHOULD** share chunking along the first dimension.

Columns are independently encoded:
- simple numeric/bool columns are commonly `encoding-type: array`
- categorical columns are commonly `encoding-type: categorical`

<a id="dataframe-v010-legacy-anndata-07x"></a>
### DataFrame v0.1.0 (legacy: anndata 0.7.x)

#### `encoding-type: dataframe`, `encoding-version: 0.1.0`

A legacy dataframe is stored as a Zarr **group** where:

- Group attributes include:
  - `_index`
  - `column-order`
  - `encoding-type: "dataframe"`
  - `encoding-version: "0.1.0"`
- Each column is an array.
- Categorical columns are stored as **integer code arrays**, and their category labels are stored in a reserved subgroup named `__categories`.

**Reserved subgroup:**

- `__categories/<colname>` stores the array of category labels for column `<colname>`.

<a id="legacy-categorical-columns-series-level"></a>
### Legacy categorical columns (Series-level)

In v0.1.0 DataFrames, a categorical column array (e.g. `obs/cell_type`) can be identified by the presence of an attribute:

- `categories`: a **path string, relative to the dataframe group**, to the corresponding labels: `__categories/<colname>`, as anndata 0.7.8 writes it.

(This differs from HDF5, which stores an object reference.)

Whether the categories are ordered is not on the column. It is the
`ordered` attribute of the labels array, `__categories/<colname>`.

## Mappings

Mappings are stored as Zarr **groups** on disk.

- This includes standard AnnData mappings such as `layers`, `obsm`, `varm`, `obsp`, `varp`, and `uns`.
- Mappings are distinct from DataFrames and sparse arrays and do not require special mapping-specific attributes.
- Mapping semantics are recursive: entries in `uns` can themselves be groups containing additional encoded elements.

### The `dict` encoding (v0.1.0)

anndata 0.8 and later tag a mapping group with:

- `encoding-type: "dict"`
- `encoding-version: "0.1.0"`

This is the current encoding, not a legacy one: anndata 0.13 writes it, and
so does this tool, on every mapping group. Files from anndata 0.7.x leave
mappings untagged, and readers accept those too.

## Scalars

### `encoding-version: 0.2.0`

Scalars are stored as **0-dimensional Zarr arrays**.

These should typically only occur inside `uns` and are commonly used for saved parameters.

- Numeric scalars:
  - `encoding-type: "numeric-scalar"`
  - `encoding-version: "0.2.0"`
  - value is numeric (including boolean, ints, floats, complex)
- String scalars:
  - `encoding-type: "string"`
  - `encoding-version: "0.2.0"`
  - a 0-d array of variable-length UTF-8 text (`VariableLengthUTF8`, `VLenUTF8` on v2), as anndata 0.13 writes it; anndata 0.8 to 0.11 wrote a fixed-width `<U`, and readers accept both

> **Legacy note**
>
> In anndata 0.7.x, scalar strings were commonly stored without `encoding-type`/`encoding-version`.

## Categorical arrays

### `encoding-type: categorical`, `encoding-version: 0.2.0`

Categorical arrays are stored as a Zarr **group** with members:

- `codes`: integer array  
  - values are zero-based indices into `categories`
  - signed integer arrays **MAY** use `-1` to denote missing values
- `categories`: array of labels

Group attributes:

- `encoding-type: "categorical"`
- `encoding-version: "0.2.0"`
- `ordered`: boolean (whether the categories are ordered)

## String arrays

### `encoding-type: string-array`, `encoding-version: 0.2.0`

- String arrays **MUST** be stored as Zarr arrays.
- Array attributes:
  - `encoding-type: "string-array"`
  - `encoding-version: "0.2.0"`
- **Zarr requirement:** the array **MUST** be stored using `numcodecs.VLenUTF8` for variable-length UTF-8 strings.

## Nullable arrays

These encodings support Pandas nullable integer/boolean/string arrays by storing a `values` array plus a boolean `mask` array.

### `encoding-type: nullable-integer`, `encoding-version: 0.1.0`

- Stored as a Zarr group with arrays:
  - `values` (integer)
  - `mask` (boolean)

### `encoding-type: nullable-boolean`, `encoding-version: 0.1.0`

- Stored as a Zarr group with arrays:
  - `values` (boolean)
  - `mask` (boolean)
- `values` and `mask` **MUST** have the same shape.

### `encoding-type: nullable-string-array`, `encoding-version: 0.1.0`

- Stored as a Zarr group with arrays:
  - `values` (string array)
  - `mask` (boolean)
- Group attributes:
  - `encoding-type: "nullable-string-array"`
  - `encoding-version: "0.1.0"`
  - optional `na-value`: `"NA"` or `"NaN"` (default `"NA"`)

<a id="missing-value-semantics"></a>
#### Missing value semantics

For elements supporting a `na-value` attribute:

- `"NA"`: comparisons propagate missingness (e.g. `"x" == NA` → `NA`)
- `"NaN"`: comparisons yield boolean results (e.g. `"x" == NaN` → `false`)

Readers should preserve semantics when the runtime model supports it.

## Record arrays

### `encoding-type: rec-array`, `encoding-version: 0.2.0`

A numpy record array, as scanpy's `rank_genes_groups` keeps its results in
`uns` (`names`, `scores`, `pvals` and the rest: one field per group). It is a
single Zarr array with a structured dtype:

- a text field is fixed-width unicode, `<U<n>` for the longest value, as
  anndata writes it to Zarr (HDF5 holds the same field as variable-length
  UTF-8);
- every other field keeps its dtype.

anndata 0.7.x wrote the same arrays untagged.

zarr-python warns that a structured dtype has no Zarr v3 specification yet;
anndata writes rec-arrays to v3 regardless, and this tool does too.

## Awkward arrays (experimental)

### `encoding-type: awkward-array`, `encoding-version: 0.1.0`

Ragged arrays are stored by decomposing an Awkward Array into constituent buffers (via `ak.to_buffers`), then storing those buffers as Zarr arrays within a group.

Group attributes:

- `encoding-type: "awkward-array"`
- `encoding-version: "0.1.0"`
- `form`: string — serialized Awkward “form”
- `length`: integer — logical length

Group members: arrays for the buffers (often named like `nodeX-*`).

> **Experimental**
>
> This encoding is considered experimental in the anndata 0.9.x series and later.

## What `adata-cli` does with these elements

This tool reads every layout listed above, including the legacy 0.7.x forms,
and always writes the current spec version shown in each section.

| Element | Read | Written |
|---|---|---|
| `anndata` | yes | yes (0.1.0, stamped on every store it creates) |
| `raw` | yes | yes (0.1.0; subset against its own var axis) |
| `dict` | yes | yes (0.1.0, on every mapping group) |
| `dataframe` | 0.2.0 and legacy 0.1.0 | 0.2.0, with `column-order` |
| `array` | yes | yes (0.2.0) |
| `csr_matrix` / `csc_matrix` | yes | yes (0.1.0); both are streamed, never loaded whole |
| `categorical` | 0.2.0, plus both legacy layouts | 0.2.0, preserving `ordered` |
| `string-array` | yes | yes (0.2.0), variable-length UTF-8 |
| `nullable-integer` / `-boolean` / `-string-array` | yes | yes (0.1.0) |
| `numeric-scalar` | yes | yes (0.2.0) |
| `string` | yes | yes (0.2.0), as a 0-d dataset |
| `null` | yes | yes (0.1.0) |
| `rec-array` | yes, tagged or not | copied field by field: text as `<U` to Zarr, as variable-length UTF-8 to HDF5 |
| `awkward-array` | reported by `view` and `ls` | not written |

### `null` (`encoding-version: 0.1.0`)

Not in the upstream prose spec, but written by anndata 0.12+ for a `None`
value in `uns`. In HDF5 it is a dataset with a null dataspace (`h5py.Empty`);
in Zarr it is a 0-d boolean array. Both carry `encoding-type: null`.

anndata 0.8 to 0.11 cannot read a `null`, so this tool writes one only where
the source had one. An HDF5 empty dataspace *without* the tag (anndata never
writes one; other HDF5 writers do) has no Zarr equivalent: copied to Zarr it
is left out, with a warning, and between HDF5 files it is copied as it is.

### What cannot be copied to Zarr

Two HDF5 dtypes have no Zarr equivalent, and a copy to Zarr refuses them
before writing anything, naming the element: variable-length sequences of
numbers (`h5py.vlen_dtype(int)`) and object references, whether a dataset of
them or an attribute holding one. Neither is anything anndata writes.
Between HDF5 files, variable-length numbers are copied. A reference is
copied only inside an element HDF5 copies whole, pointing at the copy of
its target (as anndata 0.7's categoricals in a nested dataframe are); one
that leads out of the element, or is copied on its own (`--compression`
copies dataset by dataset), is refused, naming it.

### Elements with no `encoding-type`

Files written by anndata 0.7.x tag only their dataframes, sparse matrices and
`raw`; the root, mappings, dense arrays, scalars and record arrays carry no
encoding attributes. A store whose root is untagged while its `obs` and `var`
are 0.1.0 dataframes is taken for one of these, and opens without a warning
about its root. Untagged elements are classified structurally: a group with `codes` and `categories` is a
categorical, one with `values` and `mask` is a nullable array, one with
`_index` in its attributes is a dataframe, and anything else is a mapping.
Structural inference is only ever a fallback -- a declared `encoding-type`
always wins.

### Zarr v2 versus v3

`zarr-python` 3 defaults to writing **v3**, and anndata 0.13 writes v3 by
default too. Both versions are readable here, and a derived store keeps the
source store's version rather than being silently upgraded; pass
`--zarr-format 2|3` to choose explicitly.

The differences that matter when copying between stores:

| | v2 | v3 |
|---|---|---|
| Group metadata | `.zgroup` / `.zattrs` | `zarr.json` |
| Array metadata | `.zarray` | `zarr.json` |
| Variable-length text | `VLenUTF8` in `filters` | `VariableLengthUTF8` data type |
| Compression | `compressor` (single) | `compressors` (a sequence) |
| Sharding | not available | `shards` |

A v2 string array's `VLenUTF8` filter cannot be forwarded to a v3 array -- it
raises `Expected an ArrayArrayCodec`, because the v3 string data type encodes
variable length itself. This tool drops the filter and resolves the dtype for
the target rather than reusing the source's.

Codecs travel only between stores of the same Zarr version, and with them
the absence of one: an uncompressed array stays uncompressed, on v2 as on v3
(read through zarr's `compressors`, on both). Across backends none do: an
HDF5 dataset written to Zarr gets zarr-python's default codec, and a
compressed Zarr array written to HDF5 becomes a gzip dataset at level 4
(h5py's default). An array that was not compressed stays uncompressed.

Note also that `zarr-python` flags both `NullTerminatedBytes` (what `dtype="S"`
produces) and `FixedLengthUTF32` (what `<U` produces) as having no v3
specification, and warns that other Zarr libraries may not be able to read
them. This is why text is always written as variable-length UTF-8.

## Sources

- AnnData “on-disk format” prose docs (modern, ≥0.8): https://anndata.readthedocs.io/en/stable/fileformat-prose.html
- AnnData 0.7.8 “on-disk format” prose docs (legacy): https://dokk.org/documentation/anndata/0.7.8/fileformat-prose/

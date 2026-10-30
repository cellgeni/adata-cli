---
title: Commands
parent: Reference
nav_order: 1
redirect_from:
  - /COMMANDS.html
---

# Command reference

Every command takes a `.h5ad` file or a `.zarr` directory; the backend is
detected from the path. Results go to **stdout** and status to **stderr**, so
output pipes cleanly:

```bash
adata export dataframe data.h5ad obs 2>/dev/null | head
adata ls data.h5ad -1 | grep spatial
```

Run `adata <command> --help` for the authoritative flag list. Each format
`convert` reads or writes has its own page under [Formats](formats/index.md).

## Table of contents

- [view](#view)
- [ls](#ls)
- [subset](#subset)
- [split](#split)
- [concat](#concat)
- [convert](#convert)
- [recast](#recast)
- [create](#create)
- [export](#export)
- [import](#import)
- [Query expressions](query.md), on their own page
- [Writing outputs](#writing-outputs): what every command that writes checks
- [Zarr format selection](#zarr-format-selection)

---

## `view`

AnnData-aware inspection.

```bash
adata view data.h5ad                  # n_obs x n_var and the top-level keys
adata view data.h5ad --tree           # a tree with each element's type
adata view data.h5ad --tree --depth 3
adata view data.h5ad obsm/X_pca       # detail for one entry
```

| Flag | Meaning |
|---|---|
| `--tree`, `-t` | Show a tree of all entries |
| `--depth N`, `-d` | Maximum recursion depth (with `--tree`) |

`info` is a deprecated alias, removed in 1.0.0.

## `ls`

Format-agnostic listing. Makes no AnnData assumptions, so it also works on
`.loom` and plain `.h5` files.

```bash
adata ls data.h5ad
adata ls data.h5ad --long             # type, shape, dtype, encoding
adata ls data.h5ad uns --depth 2      # only below a path
adata ls data.h5ad -1 | grep spatial  # bare paths, for piping
```

## `subset`

Write a filtered copy. Select by name list, by expression, or both axes at
once.

```bash
adata subset data.h5ad -o out.h5ad --obs barcodes.txt
adata subset data.h5ad -o out.h5ad --obs-query "cluster == Cortex_2"
adata subset data.h5ad -o out.h5ad -q "n_counts > 1000 and cluster in A,B"
adata subset data.h5ad -o out.h5ad --var-query "highly_variable == True"
adata subset data.h5ad --inplace --obs barcodes.txt
```

| Flag | Meaning |
|---|---|
| `--output`, `-o` | Output path. Required unless `--inplace` |
| `--inplace` | Replace the source (written to a temporary path first) |
| `--obs` / `--var` | File of names to keep, one per line |
| `--obs-query`, `-q` / `--var-query` | Keep rows matching an [expression](query.md) |
| `--chunk`, `-C` | Row chunk size for dense matrices (at least 1) |
| `--zarr-format` | Zarr version to write; defaults to the source's |
| `--no-space-check` | Write even if the disk looks too small; see [Writing outputs](#writing-outputs) |

`raw/` is carried over and matched against its **own** var axis, which usually
holds more genes than the main object.

A categorical obs or var column keeps only the categories its rows use, as
anndata's own subsetting keeps them (on an axis kept whole too), and
`uns["<column>_colors"]` keeps the colours of those categories. `split`
does the same for each output.

The selection is resolved before anything is written, and the output is
written beside its destination (`out.adata-tmp.h5ad`) and moved into place
only once complete. A query that fails therefore leaves nothing behind, and an
existing output (or, with `--inplace`, the source) is replaced only once its
successor is finished. The output path may not name the input; use
`--inplace`.

A selection that keeps no rows is refused, whether a query or a name list
that matches nothing (barcodes without the `-1` the index carries, say).
Nothing is written, and `--inplace` leaves the source as it was.

## `convert`

Change a store's format. The output's extension picks the format; `--to`
names it when the extension does not say, and `--from` names the input's
when it cannot be detected from its content.

```bash
adata convert data.h5ad -o data.zarr
adata convert data.zarr -o data.h5ad --compression gzip
adata convert old.zarr -o new.zarr --zarr-format 3
adata convert data.h5ad -o data.zarr --overwrite
adata convert filtered_feature_bc_matrix.h5 -o counts.h5ad
adata convert counts.h5ad -o for_cellbender.h5 --layer counts
adata convert Solo.out/Gene/filtered -o counts.h5ad
adata convert counts.h5ad -o counts_mtx --to 10x-mtx
adata convert spaceranger_run/outs -o section.h5ad
adata convert visium_hd_run/outs -o bins.h5ad --bin-size 8
adata convert visium_hd_run/outs -o cells.h5ad --cells
adata convert xenium_bundle -o cells.h5ad
adata convert splitpipe_out -o parse.h5ad
adata convert velocyto.loom -o velocyto.h5ad
adata convert counts.h5ad -o counts.loom
adata convert GSE123_counts.txt.gz -o counts.h5ad --transpose
adata convert matrix.hdf5 -o matrix.h5ad --key data/counts
adata convert cistopic_obj.pkl -o atac.h5ad
adata convert multiome.h5mu -o multiome.h5ad
adata convert multiome.h5ad -o multiome.h5mu
```

| Flag | Meaning |
|---|---|
| `--output`, `-o` | Output path (required) |
| `--from` | Input format, if it cannot be detected |
| `--to` | Output format, if the extension does not say |
| `--zarr-format` | Zarr version to write; defaults to the source's, else 3 |
| `--compression` | `none`, `gzip` or `lzf` for h5ad; `none`, `gzip`, `zstd` or `blosc` for Zarr; `gzip` (default) or `none` for an MTX folder |
| `--overwrite` | Replace an existing output |
| `--force` | Convert despite a lossy, inflating or placeholder result |
| `--no-space-check` | Write even if the disk looks too small; see [Writing outputs](#writing-outputs) |

| Format | Extension | Read | Write |
|---|---|---|---|
| `h5ad` | `.h5ad` | yes | yes |
| `zarr` | `.zarr` (v2 or v3) | yes | yes |
| [`10x-h5`](formats/10x-h5.md) | `.h5` | Cell Ranger 2 and 3+; the matrix alone of Space Ranger and Xenium | Cell Ranger 3+ layout |
| [`visium`](formats/visium.md) | a folder | Space Ranger `outs/`: counts, spot or bin positions or segmented cells, images, Space Ranger's clusters, UMAP and PCA (standard Visium; Visium HD bins with `pyadata-cli[spatial]`; Space Ranger 4's cells with `--cells`) | no |
| [`xenium`](formats/xenium.md) | a folder | a Xenium bundle: counts, cells and centroids, Xenium's analysis | no |
| [`cellbender`](formats/cellbender.md) | `.h5` | CellBender `remove-background` 0.2 and later | no |
| [`10x-mtx`](formats/10x-mtx.md) | a folder (`--to 10x-mtx`) | Cell Ranger 2 and 3+, STARsolo (velocyto too) | Cell Ranger 3+ layout, gzipped |
| [`parse`](formats/parse.md) | a folder | Parse Biosciences split-pipe: a run, a sample or a `DGE_*` folder | no |
| [`loom`](formats/loom.md) | `.loom` | loompy, velocyto, kallisto\|bustools, SCope (pySCENIC, SCENIC+, pycistopic); spec 2 and 3 | spec 3.0.0, as loompy 3 writes it; a SCope loom with `--scope` |
| [`text`](formats/text.md) | `.csv`, `.tsv`, `.tab`, `.txt`, `.data`, each also `.gz` | a dense matrix, as anndata's `read_csv` and `read_text` read it | no |
| [`hdf5`](formats/hdf5.md) | `.hdf5`, `.he5`, `.hdf`; a plain `.h5` with `--from hdf5` | one 2-D dataset, as anndata's `read_hdf` reads it | no |
| [`cistopic`](formats/cistopic.md) | `.pkl`, `.pickle`, gzipped too | a pycisTopic `CistopicObject` or `CistopicImputedFeatures`, without pycisTopic (needs `pyadata-cli[pickle]`) | no |
| [`mudata`](formats/mudata.md) | `.h5mu`; a MuData Zarr store is read by its content | each modality (or the one `--mod` names), with the global parts on its cells | `.h5mu`, var split by feature type |

Formats other than h5ad and Zarr convert to and from AnnData only: to turn a
10x file into another format, convert it to `.h5ad` first. Each format's
page gives what maps to what, its own flags, and how it streams.

Between h5ad and Zarr every element is copied as it is, streamed, so the
store's size is not a limit. Chunking carries over, and so does compression
where the target can express it: Zarr codecs travel between stores of the same
Zarr version, a compressed Zarr array becomes gzip in an h5ad, and an h5ad
written to Zarr gets zarr-python's default codec. `--compression` replaces all
of that with one choice. Elements are not re-encoded, so a store written by
anndata 0.7 keeps its legacy layout.

Every check runs before anything is written. The output is written beside
its destination (`out.adata-tmp.zarr`) and moved into place only once
complete, so a failed conversion leaves nothing behind. An existing output is
refused unless `--overwrite` is given, and is replaced only once its
successor is finished. `--overwrite` replaces a store, or a Matrix Market
folder holding only its own files, never a folder holding anything else.

To change a matrix's dtype, layout or density instead, use
[`recast`](#recast).

## `recast`

Change a matrix's dtype, layout or density. Counts held as float64 cost twice
the disk and twice the read for no information; a tool that wants CSC cannot
use a CSR store; and `concat` refuses inputs whose encodings disagree.

```bash
adata recast data.h5ad X -o out.h5ad --dtype float32
adata recast data.h5ad X -o out.h5ad --layout csc
adata recast data.h5ad X --inplace --dtype float32 --indices-dtype int32
adata recast data.h5ad --all -o out.h5ad --dtype float32
adata recast data.h5ad layers/counts -o out.h5ad --layout dense --force
```

| Flag | Meaning |
|---|---|
| `--output`, `-o` | Output path. Required unless `--inplace` |
| `--inplace` | Replace the source (written to a temporary path first) |
| `--all` | Recast `X`, every layer and `raw/X` |
| `--dtype` | New dtype for the values, e.g. `float32` |
| `--indices-dtype` | New dtype for sparse indices: `int32` or `int64` |
| `--layout` | `csr`, `csc`, `dense` or `sparse` |
| `--force` | Recast despite a lossy cast or a large size increase |
| `--in-memory` | Transpose in memory rather than streaming |
| `--chunk`, `-C` | Row chunk size for dense matrices (at least 1) |
| `--zarr-format` | Zarr version to write; defaults to the source's |
| `--no-space-check` | Write even if the disk looks too small; see [Writing outputs](#writing-outputs) |

Two things are refused before anything is written. A cast that would not
round-trip -- checked by casting every value and casting it back, not by
comparing dtypes -- and a densification that would inflate the store beyond
four times its size. `--force` overrides either, and the message says how
many values would change or how large the result would be.

The index dtype is **preserved** unless `--indices-dtype` asks otherwise, so
narrowing the values does not silently widen the indices and leave the file
bigger than it started. `indices` and `indptr` keep their own widths, which
can differ: a narrow matrix with more than 2^31 nonzeros needs int64 offsets
over int32 coordinates, and both are range-checked before writing.

The output path may not name the input; use `--inplace`. Either way the new
store is written beside its destination and moved into place only once the
conversion has finished, so a failure leaves the original untouched.

Transposing streams by default and works on matrices too large to load, at
the cost of two extra passes over the nonzeros. `--in-memory` is faster when
the matrix fits.

Until 0.7.0 this command was called `convert`. Until 1.0.0, `convert` given
any of `--dtype`, `--indices-dtype`, `--layout`, `--all` or `--inplace` still
runs `recast`, with a warning.

## `split`

One store per distinct value of a column.

```bash
adata split data.h5ad --by sample -o per_sample/
adata split data.h5ad --by cell_type -o clusters/ --dry-run
adata split data.h5ad --by cluster -o out/ --min-size 50
```

| Flag | Meaning |
|---|---|
| `--by`, `-b` | Column to split on |
| `--output-dir`, `-o` | Where to write the stores |
| `--axis` | `obs` (default) or `var` |
| `--min-size` | Skip groups smaller than this |
| `--dry-run` | Print the plan without writing |
| `--manifest / --no-manifest` | Write a CSV listing the outputs (default on) |
| `--chunk`, `-C` | Row chunk size for dense matrices (at least 1) |
| `--zarr-format` | Zarr version to write; defaults to the source's |
| `--no-space-check` | Write even if the disk looks too small; see [Writing outputs](#writing-outputs) |

Labels are sanitised for use as filenames; collisions get a numeric suffix,
and so do labels that differ only in case (`B`, `b_1`), which would be one
file on macOS and Windows.
Every store is written beside its destination and moved into place only once
all of them are complete, and the manifest is written last, so a split that
fails leaves no partial stores. An output that would land on the input is
refused, on a case-insensitive filesystem too (a label `A` beside `a.h5ad`).

## `concat`

Concatenate along the obs axis, streaming each input in turn.

```bash
adata concat a.h5ad b.h5ad -o merged.h5ad
adata concat *.h5ad -o merged.h5ad --join outer --label sample
adata concat a.h5ad b.h5ad -o m.h5ad --keys a,b --index-unique - --uns-merge same
```

| Flag | Meaning |
|---|---|
| `--join`, `-j` | `inner` (shared vars only) or `outer` (their union) |
| `--label` | Add an obs column recording each cell's source |
| `--keys` | Names for the inputs; defaults to their filenames |
| `--index-unique` | Delimiter for suffixing obs names with their key |
| `--merge` / `--uns-merge` | `drop` (default), `same`, `unique`, `first`, `only` |
| `--fill-value` | Value for dense cells introduced by an outer join |
| `--chunk`, `-C` | Row chunk size while streaming (at least 1) |
| `--zarr-format` | Zarr version to write; defaults to the inputs' when they agree, otherwise v3 |
| `--no-space-check` | Write even if the disk looks too small; see [Writing outputs](#writing-outputs) |

obs columns keep their dtypes: categoricals union their category sets, nullable
columns keep their masks, and a column missing from one input is padded rather
than dropped.

obsm entries present in every input are stacked: arrays as they are, and
dataframes (a SCope import's `RegulonsAUC`) as obs is, their columns
following `--join`. var's columns and varm (a SCope import's `Regulons`,
pycisTopic's `topic_region`) follow `--merge`, compared on the output's
genes: an input lacking some of them, as in an outer join, counts as not
having the entry. Without `--merge`, varm is dropped with a note.

`obsp`, `varp` and `raw` are **not** carried over — a pairwise value has no
meaning between cells from different inputs. anndata's own `concat_on_disk`
refuses these too.

### Merge strategies

| Strategy | Keeps a value when |
|---|---|
| *(unset)* | never — the element is dropped |
| `same` | every input has it and they all agree |
| `unique` | exactly one distinct value among the inputs that have it |
| `first` | always; takes the earliest input that has it |
| `only` | exactly one input has it at all |

## `create`

Write a new, empty store for `import` to fill in.

```bash
adata create out.h5ad --n-obs 5000 --n-var 2000
adata create out.zarr --obs-names cells.txt --var-names genes.txt
```

Give each axis either a size (names are generated as `cell_0000`, `gene_0000`)
or a file of names. `--force` overwrites an existing store, and only a store:
a folder that is not a Zarr store, or a file that is not HDF5, is refused.
The new store is written beside the old one and swapped in once complete, so
a failed `create --force` leaves the old store as it was.

## `export`

| Subcommand | Produces | Accepts |
|---|---|---|
| `dataframe` | CSV | any dataframe path (`obs`, `var`, `raw/var`, …) |
| `array` | `.npy` | any dense array |
| `sparse` | `.mtx`, or `.mtx.gz` | CSR/CSC groups |
| `dict` | JSON | any group or scalar |
| `image` | PNG/JPEG/TIFF | 2D or 3D arrays |

```bash
adata export dataframe data.h5ad obs -o obs.csv
adata export dataframe data.h5ad obs --columns cluster,n_counts --head 100
adata export sparse data.h5ad X -o matrix.mtx
adata export array data.h5ad obsm/X_umap > umap.npy
adata export dict data.h5ad uns -o metadata.json
```

Omit `--output` to write to stdout; `export image` writes PNG there, and
otherwise takes its format from the output's extension. `export dataframe`,
`export array` and `export sparse` stream in chunks, to a file or to stdout
alike. An export whose reader stops early (`| head`) ends quietly, with exit
code 0.

`export dataframe` checks `--columns` before it writes anything, and ends
each line in `\n` alone. On `export dataframe` and `export sparse`,
`--head 0` writes the header and nothing else; a negative `--head` is refused.

`export array` writes a nullable integer or boolean array that has missing
values as float64, with NaN where they are missing, since a `.npy` has no
mask; one with none keeps its dtype. Text is written as fixed-width unicode,
as wide as its longest value. Text with missing values is refused: export it
with `export dataframe`, which writes them as empty fields.

`export dict` writes NaN and infinite values as `null`, since JSON has no
number for them, and says how many it wrote.

A file output is written beside its destination and renamed into place once
complete, so an export that fails or is refused leaves an existing file as it
was. An output that names the input store, lies inside a Zarr input, or names
a folder is refused before anything is read.

`export sparse` gzips its output when the name ends in `.gz`. Whole-number
values are written as integers, and other floats with enough digits to read
back exactly.

## `import`

Write data into a store at any path. All subcommands need `--output`/`-o` or
`--inplace`, and refuse both at once.

An import never edits a store in place. The source is copied beside the
destination (beside itself with `--inplace`), the element is written into
the copy, and the copy is moved into place once that has worked, so an
import that is refused or fails part way leaves the source and any existing
output as they were. The copy needs room for the whole store: each
subcommand checks the free space first and takes `--no-space-check` to skip
that (see [Writing outputs](#writing-outputs)).

```bash
adata import dataframe data.h5ad obs cells.csv --inplace -i cell_id
adata import array     data.h5ad obsm/X_umap umap.npy --inplace
adata import sparse    data.h5ad X counts.mtx --inplace
adata import dict      data.h5ad uns/params params.json --inplace
adata import image     data.h5ad uns/spatial/hires tissue.png --inplace
```

Dimensions are validated against the existing obs/var where the path implies
an axis. For `array` and `sparse` that is done from the file's header,
before the store is copied, so a matrix of the wrong shape is refused at
once rather than after a copy of a large store.

An element of a kind the AnnData layout does not allow where it would go is
refused before anything is written, since anndata could not then read the
store: an array, a matrix, an image or JSON in place of `obs`, `var` or
`raw/var`; anything in place of the whole of `obsm`, `varm`, `obsp`, `varp`
or `layers`, or of `uns` but a JSON object; JSON or a dataframe as `X`; an
array that is not 1-D as a column of a dataframe. Name the entry to write,
such as `obsm/X_pca` or `uns/params`: an entry is replaced whatever it held.

For `array`: the `.npy` is read a block of rows at a time, so its size is not
a limit; `--chunk`, `-C` sets how many elements a block holds (1,048,576 by
default, at least 1). Text (`<U` or `S`) becomes a string array.

For `sparse`: the file is read in two streaming passes, so its size is not a
limit, and it may be gzipped (`counts.mtx.gz`). The first pass checks the
whole file before the element is written. `integer` values stay integers
(int32, or int64 when a value needs it), `real` ones become float64, and
`pattern` entries become ones. Duplicate entries are summed, as scipy does.
Only `coordinate general` files are read: `symmetric`, `skew-symmetric`,
`hermitian` and `complex` files, and dense `array` ones, are refused. An
input whose entries are not in row order is sorted through scratch space
beside the store.

For `dataframe`: the CSV is read in two streaming passes, and each column is
typed as `pandas.read_csv` types it: integers, floats (a missing value makes
an integer column float64, with NaN), True/False as bool, and text
otherwise. A number is what pandas reads as one, so a label such as `1_1`
stays text, and whole numbers past int64 become uint64, or text past uint64.
A text column that repeats a value becomes a categorical, as anndata makes
it. Use `--categorical col1,col2` to make specific text columns categoricals
whatever their values (numeric columns stay numeric), or
`--no-auto-categorical` to keep all text as `string-array`. A row with more
or fewer fields than the header, a header that repeats a name, a column with
no name (as pandas writes an unnamed index) or named `_index` besides the
index, and a `--categorical` name the file does not have are refused before
the store is touched.

For `dict`: an object becomes a mapping, and an array a typed array where its
values share a type: strings, booleans, whole numbers (int64, or uint64 past
it), or numbers (float64, with NaN for a `null` among them). An array that
mixes types, or is ragged, is kept as its JSON text. A key holding `null` is
left out, with a warning, as anndata 0.8 to 0.11 leave out `None`: they
cannot read the `null` encoding, and would refuse the whole store. A whole
number past 64 bits is refused.

Names, of a CSV's columns or a JSON object's keys, may not be empty, `.` or
`..`, hold `/`, or be one of Zarr's metadata files (`.zarray`, `.zattrs`,
`.zgroup`, `.zmetadata`, `zarr.json`).

Giving `-o out.zarr` for an `.h5ad` source (or the reverse) converts the store
as it writes.

## Writing outputs

What every command that writes a store or file checks, and how it writes.

**The output is checked before anything is opened.** It may not be the
input, under any name that reaches it: a symlink, `..`, a hard link, or a
change of case on a case-insensitive filesystem (macOS, Windows). A store
output (`-o` on `subset`, `recast`, `concat`, `import`, and `create`'s path)
may replace an existing store (or an empty folder), but not a folder that is
not a Zarr store or a file that is not HDF5: `-o results/`, as if it were
`split`'s folder, used to delete `results/` and everything in it. An
`export` output may not be a folder or a store. `--inplace` and `-o`
together are refused, rather than `-o` being ignored.

**It is written beside its destination** (`out.adata-tmp.h5ad`) and renamed
into place only once complete; an existing output is renamed aside to
`out.adata-old.h5ad` and deleted once the new one is in place. A failure at
any point, including the last flush to disk when the store is closed, leaves
the existing output (or, with `--inplace`, the source) as it was. Before an
existing output is replaced, its successor is synced to disk, so a power cut
just after cannot lose both, and it takes over the old one's permissions and
group: a private `0640` store stays private. An output that is a symlink,
or a symlinked source with `--inplace`, is followed: the store or file it
points at is replaced, and the link kept. An `export` to a device or a pipe
(`/dev/null`, a FIFO, `>(gzip > out.gz)`) is written as it is.

**One run at a time per output.** A run claims each output with a lock file
beside it (`out.h5ad.adata-lock`, naming its host and process) for as long
as it writes, so a second run writing the same output, a pipeline's retry
while the first attempt still runs, is refused instead of mixing its data
into the first's. A lock left by a run that was killed on the same host is
taken over; one from another host is refused, since that run may still be
writing.

**A scheduler's SIGTERM** (a time limit, `scancel`, `bkill`), or SIGHUP from
a closed terminal, gets the same cleanup as a failure: the partial output is
removed and the command exits with code 143 (129 for SIGHUP). Only SIGKILL
leaves anything behind; see
[Leftovers](../troubleshooting.md#leftovers-from-an-interrupted-run).

**Free space is checked first, for an HDF5 output.** A disk that fills part
way through an HDF5 write can stop the process inside the HDF5 library,
where nothing can clean up. So `subset`, `recast`, `split`, `concat`,
`convert` and each `import` subcommand, writing an HDF5 store (`.h5ad`,
`.loom`, `.h5`, `.h5mu`), estimate its size and refuse to start when the
output's filesystem has less free than that. A Zarr store that runs out of
space fails with an ordinary error and is removed, so it is not checked. The
estimate follows the source's size on disk: the fraction a `subset` or
`split` keeps, the whole store for `recast` plus what the converted matrices
gain (a wider dtype, a densified matrix), the inputs together for `concat`,
the store plus the file for `import`, the input file or store for `convert`
(a folder input, such as Space Ranger's `outs/`, holds images and BAMs the
output never carries, so it is not measured). It is an estimate, and a
quota can be smaller than the free space it sees, so `--no-space-check`
skips it.

**`--chunk` is at least 1** wherever it appears. A zero or negative value is
a usage error (exit 2), where it used to write an all-zero dense matrix.

## Zarr format selection

New Zarr stores follow the source store's version, so a v2 input is not
silently upgraded. `--zarr-format 2|3` on `create`, `subset`, `split`,
`concat`, `convert` and `recast` overrides that. Writing a Zarr store from
an `.h5ad` source defaults to v3. `concat` follows its Zarr inputs when they
share one version (h5ad inputs do not count) and writes v3 when they differ.

```bash
adata create out.zarr --n-obs 100 --n-var 50 --zarr-format 2
adata subset data.zarr -o out.zarr --obs keep.txt --zarr-format 3
adata split data.zarr --by sample -o parts/ --suffix .zarr --zarr-format 3
adata concat a.zarr b.zarr -o merged.zarr --zarr-format 3
adata convert data.zarr -o v3.zarr --zarr-format 3
```

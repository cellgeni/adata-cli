---
title: Changelog
nav_order: 7
---

# Changelog

Notable changes to `adata-cli`. Versions are `MAJOR.MINOR.PATCH`; tags carry no
`v` prefix.

## 0.17.6

The last wave of fixes from the October 2026 audit
([#87](https://github.com/cellgeni/adata-cli/issues/87)), on streaming, which
closes it. Reads and writes follow the source's chunks and shards, so
anndata 0.13's sharded Zarr v3 stores stream at the cost `--chunk` implies
and each shard is written once; `recast` checks and converts in `--chunk`
blocks; `split`'s memory no longer grows with its groups. Pickle reading is
hardened, 16-bit images export as they are, and the CLI's output is plain
lines in logs and pipes, with names printed as the store spells them.
h5py 3.16 or later is now required; its Linux wheels need glibc 2.28.

### Changed

- **h5py 3.16.0 or later is required**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). h5py 3.15 kept 32
  to 160 bytes of memory for good on every dataset read and write, so long
  streaming runs grew with their number of blocks. h5py 3.16's HDF5 2.0 gives
  each dataset an 8 MiB chunk cache; stores are opened with 1 MiB, as before.
  Its Linux wheels need glibc 2.28 or later.

### Fixed

- **Subset, split, concat, recast and convert write each Zarr v3 shard once**
  ([#97](https://github.com/cellgeni/adata-cli/issues/97)). anndata 0.13 writes
  Zarr v3 sharded, a whole array in one shard. Commands wrote outputs a
  `--chunk` block at a time, and zarr rewrites a whole shard for a write of any
  part of it: subset and concat wrote 15 times the bytes they meant to, split
  121 times, and a dense subset held 1.7 GB. Outputs are written a whole shard
  (or chunk) at a time now. A shard is cut along its rows to at most 8 MiB, 128
  MiB shared across the outputs `split` writes at once, and about four times
  that is held while zarr encodes it.
- **Outputs keep a sharded source's shards and inner chunks**
  ([#97](https://github.com/cellgeni/adata-cli/issues/97)). A sparse matrix's
  `data` and `indices` dropped the shard and kept the inner chunk, 245 elements
  a file under anndata 0.13 with zarr 3.1: thousands of files per array. An
  h5ad output took the inner chunk as its HDF5 chunk, making a file twice
  anndata's own. Zarr v3 outputs keep the shard. HDF5 outputs let HDF5 choose
  their chunks, and `--zarr-format 2` takes the shard as the chunk.
- **Sparse `subset` and `split` follow `--chunk`**
  ([#97](https://github.com/cellgeni/adata-cli/issues/97)). Blocks were 4,096
  rows or columns whatever `--chunk` said, so a CSC block (4,096 genes across
  every cell) grew with the number of cells. A block holds about as many
  entries as `--chunk` rows do now, and a gene expressed in every cell is read
  in parts.
- **Reads follow the source's chunks**
  ([#97](https://github.com/cellgeni/adata-cli/issues/97)). `subset`, `split`,
  `concat`, `recast`, `convert` and `export` read a dense matrix in whole
  chunks of the source, so a chunk taller than `--chunk` is no longer decoded
  once for every block: 7 to 60 times the matrix for zarr's own chunks or
  chunks of (n_obs, 8). Past 32 MiB, `subset`, `split` and `convert` read a
  band of columns, so `convert` no longer reads a column-chunked matrix whole.
  A Zarr v3 shard's index is read once per command, not once per block.
- **`split` memory no longer grows with the number of groups**
  ([#97](https://github.com/cellgeni/adata-cli/issues/97)). An open HDF5 file
  costs about 0.6 MB whatever is in it, and split held up to 1,024 open at
  once: 581 MB of RSS at 512 groups, against 143 MB now. It writes at most 128
  outputs per pass over the matrices, so past 128 groups the matrices are read
  once per 128. Each output is written a whole chunk at a time, so a chunk 512
  rows tall is no longer decoded and encoded once for every block.
- **`recast --dtype` checks a dense matrix a block at a time**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). The round-trip
  check read the whole matrix at once, and two more copies of it, whatever
  `--chunk` said. It reads `--chunk` rows now, as the conversion does.
- **Densifying a CSC matrix, and writing a dense one as CSC, hold `--chunk`
  rows** ([#96](https://github.com/cellgeni/adata-cli/issues/96)). Each block
  was `--chunk` columns of every row, so memory followed the number of cells: 8
  GB a block at 2M cells. Both now go through a CSR in scratch and the
  streaming transpose.
- **The streaming transpose reads about a chunk at a time on skewed matrices**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). It stepped by the
  average row, so a run of genes expressed in every cell was read many chunks
  at once, and a gene expressed in every cell was sorted whole. It steps by
  entries now, and streams such a gene from a bucket of its own.
- **A sparse matrix made from a dense one keeps element-sized chunks**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). `recast --layout
  csr`/`csc` and the 10x-h5 export gave `data` and `indices` the dense source's
  chunk height as their chunk, down to 16 elements: hundreds of thousands of
  chunks and a file three times the size.
- **`--force` no longer skips the index range check**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). Given to accept a
  lossy cast, it also let `indptr` overflow `--indices-dtype` and wrap
  negative, a corrupt matrix with exit 0. A dense matrix made sparse is
  range-checked too.
- **float16 is refused for a sparse matrix**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). scipy has no
  float16 sparse matrix, so the store `recast` wrote could not be read by
  anndata or scanpy. Use `--dtype float32`, or `--layout dense`.
- **A dense matrix made sparse by a lossy cast drops the zeros the cast makes**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). Values that became
  0 were stored as explicit zeros, three times scipy's nnz in one case, which
  misleads nnz-based QC.
- **Matrices nested two or more groups deep are converted**
  ([#96](https://github.com/cellgeni/adata-cli/issues/96)). `recast` of
  `uns/neighbors/connectivities` reported converting it and copied it
  unchanged.
- **Pickle arrays are checked before use**
  ([#101](https://github.com/cellgeni/adata-cli/issues/101)). Reading a
  pycisTopic pickle checks each array it rebuilds (dtype, shape, and a sparse
  matrix's indices) before numpy, scipy or pandas uses it, and refuses a pickle
  that fails before anything is written, so a crafted pickle can no longer make
  scipy write out of bounds or have numpy read memory as another type. Dates
  and times are no longer read from pickles, which also stops a crafted
  timezone from opening a local file; pycisTopic's objects hold none. Pickles
  are still trusted input: read them only from a source you would trust with
  the data. Pickles that pandas 1.5 writes with missing values in a nullable
  text column now read.
- **Imputed pycisTopic matrices convert whatever their width**
  ([#101](https://github.com/cellgeni/adata-cli/issues/101)). `--dense`
  chunked X as 4096 cells by every feature, which HDF5 refused past 262,144
  float32 features, and CSR output held 4096 cells' coordinates at once,
  about 50 GB at 500,000 regions. Both now work about a million entries at
  a time, and a dense X is chunked the same way.
- **`--dense` is honoured for gene activity**
  ([#101](https://github.com/cellgeni/adata-cli/issues/101)). An imputed
  object holding a sparse matrix was written as CSR although `--dense` was
  given and the summary said dense.
- **A Xenium bundle converts into a new folder**
  ([#101](https://github.com/cellgeni/adata-cli/issues/101)). With only
  `analysis.tar.gz`, converting to an output whose folder did not exist yet
  failed with a bare "No such file or directory" naming a temporary path.
- **`export image` keeps 16-bit images**
  ([#101](https://github.com/cellgeni/adata-cli/issues/101)). Every value
  from 255 up was clipped to 255, silently. One channel of 16 bits is now
  written as a 16-bit PNG or TIFF, and values an image cannot keep are
  refused, naming their range; the new `--scale` stretches them onto 0-255.
- **Names with square brackets print as the store spells them**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)). Rich read the
  `[mt]` of scanpy's `pct_counts[mt]` as a style tag. `view`, `ls`, `split` and
  messages showed `pct_counts`, a name the next command could not find, and a
  name, label or attribute holding `[/x]` stopped the command. Every name,
  label, path and value from a store or the command line is escaped now.
- **`view FILE ENTRY` on a missing entry fails**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)). It printed its
  `Error:` line on stdout and exited 0, so a script took the message for the
  entry's description. It goes to stderr with exit code 1 now, as `ls` does.
- **Logs hold plain lines**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)). Written to a file
  or a pipe, stderr held colour and cursor codes and wrapped messages at 80
  columns, so `grep '^Error:' adata.log` found nothing. Off a terminal there
  are no codes now and an error is one line. On a terminal nothing changes.
- **`convert --help` names the extras to install**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)):
  `pyadata-cli[spatial]` and `pyadata-cli[pickle]` showed as `pyadata-cli`.
- **Help says what the commands do**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)):
  - `--to` lists only the formats `convert` can write;
  - `--inplace` says the source is replaced by a store written beside it, not
    modified directly;
  - `convert --help` says which formats are only read.
- **`adata.cli.main(argv)` runs `argv`**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)). It parsed the
  process's own command line instead.
- **The documentation matches the code**
  ([#103](https://github.com/cellgeni/adata-cli/issues/103)):
  - the command reference has every option;
  - troubleshooting and the pipelines guide list everything held whole in
    memory, images among them;
  - the testing page maps every test file and CI job;
  - the 0.17.1 entry below no longer says images are not loaded whole.

## 0.17.5

The second wave of fixes from the October 2026 audit
([#87](https://github.com/cellgeni/adata-cli/issues/87)), on fidelity. What
anndata and scanpy write survives every copy path, scanpy's marker genes and
anndata 0.7 files included; `subset` and `split` keep exactly the rows named
and subset every element as anndata does; `concat` takes any number of
inputs and orders an outer join as anndata does; the 10x, MTX, Parse,
CellBender, loom, SCope and MuData converters round-trip or refuse clearly;
and a release publishes only what passed. Two changes need attention: zarr
3.2 or later is required, and the container image runs as an unprivileged
user, so a Docker run that writes to a mounted folder needs `--user`.

### Changed

- **The container image runs as an unprivileged user**
  ([#102](https://github.com/cellgeni/adata-cli/issues/102)). It runs as uid
  10001 rather than root, and its base is pinned by digest. duckdb's download
  is checked against a pinned SHA-256. Singularity, Apptainer and Docker with
  `-u` still run it as the caller. A plain `docker run` on a bind mount needs
  `--user "$(id -u):$(id -g)"`.
- **zarr 3.2 or later is required**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). zarr 3.1 cannot
  open a v3 store that newer zarr wrote with a record array, as anndata 0.13
  does for `rank_genes_groups`.
- **An outer join's var and merged categories are in anndata's order**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  `concat --join outer` keeps the inputs' var order when they all share it
  and sorts the union otherwise, as `anndata.concat` does; it used to be the
  order of first appearance. Merged categories keep the first input's order
  when it already holds every category, and are natsorted otherwise
  (accented letters as natsort orders them, "Müller" after "Muscle"). An
  ordered categorical whose inputs disagree on the order is written
  unordered, with a warning, where it used to keep the first input's order.
- **A quoted value in an `in` list is one label**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). `cluster in
  "1,2,3"` used to be split at the commas. A quoted value is now taken as
  written, so write `cluster in 1,2,3` or `cluster in ("1", "2", "3")` for
  three labels.
- **split names a label that is a filename already after itself**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). `CD4`, `CD4_1` and
  `CD4+` now give `CD4.h5ad`, `CD4_1.h5ad` and `CD4_2.h5ad`; the label that
  needed sanitising takes the suffix. Missing values, NaN included, claim `NA`
  first, so a literal `NA` label moves to `NA_1` when the column also has
  missing values, and NaN in a float column goes to `NA` rather than `nan`.
- **Removed the `adata.info` and `adata.read` modules**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). They only
  re-exported `adata.core.info` and `adata.core.read`. `subset_h5ad` lost its
  unused `quiet` argument, and `indices_from_name_set` now also returns how
  many names matched more than one row.
- **10x export takes feature names from `var["gene_symbols"]` when there is
  one** ([#99](https://github.com/cellgeni/adata-cli/issues/99)). The ids come
  from `var["gene_ids"]`, and the var names stand in for the ids when var is
  laid out as `--var-names gene_ids` leaves it, as the loom export already
  does. A store with both columns used to export its var names as the names.
- **A probe-level (Flex) matrix exports its probe ids as the feature ids**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). This holds in 10x
  HDF5, with the gene ids in `features/gene_id`, and in 10x MTX `features.tsv`.
- **10x HDF5 imports read Cell Ranger's XML character references as characters,
  and record which feature datasets they read**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). The record is in
  `uns["cellranger"]["feature_keys"]`, so an export writes the datasets back.
  `--var-names gene_ids --make-unique` keeps the ids it had to suffix in
  `var["gene_ids"]`.

### Fixed

- **Datasets held outside the file are refused**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). An .h5ad whose
  datasets keep their data in other files (HDF5 external storage or virtual
  datasets) is refused when it is opened or copied, so a crafted file can no
  longer pull other local files into an output.
- **scanpy's marker genes cross backends**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). `rank_genes_groups`
  (record arrays in `uns`) failed every conversion and subset between h5ad and
  Zarr, and `convert --compression` of an h5ad holding them crashed the
  process. They are now copied field by field, as anndata writes them, `concat
  --uns-merge same` keeps them (it dropped every text one read from an h5ad),
  and `export dict` writes them as JSON.
- **anndata 0.7 files copy whole**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). An h5ad copy nulled
  the references of 0.7 dataframes nested in `raw`, `obsm` or `uns` (anndata
  could not open the output), every copy made an ordered categorical unordered,
  and every command warned that such a file's root attributes were invalid. CI
  now reads stores written by anndata 0.7.8.
- **Elements other writers leave behind**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). Variable-length
  numbers and object references were turned into text on their way to Zarr;
  they are now refused there by name, and copied between h5ad files. An
  attribute holding nothing, or a 2-D array of bytes, no longer fails a
  conversion to Zarr, and an untagged empty dataspace is no longer tagged
  `null` (which anndata 0.8 to 0.11 refuse).
- **Zarr v2 keeps its compression**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). An uncompressed v2
  array came out Blosc-compressed, and so could a compressed one when zarr's
  deprecated API failed.
- **recast to bool on h5ad**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). It failed on every
  float matrix with "no appropriate function for conversion path".
- **float32 exports as written**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). `export dataframe`
  printed 0.1 in a float32 column as 0.10000000149011612; it now writes what
  pandas writes.
- **pycisTopic tables open in older anndata**
  ([#95](https://github.com/cellgeni/adata-cli/issues/95)). A model table's
  text with a missing value is a categorical, which anndata 0.8 to 0.10 read,
  not a `nullable-string-array`, which they refuse.
- **Name lists keep every row with a listed name**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). `subset
  --obs`/`--var` kept only the first row with each name, so after a `concat`
  without `--index-unique`, a barcode list kept only the first sample's cells.
  It now keeps what anndata's `obs_names.isin(names)` keeps, raw included, and
  says how many names matched more than one row.
- **Quoted labels with commas work in `in` lists**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). `cell_type in
  ("CD4-positive, alpha-beta T cell", "B cell")` split the first label at its
  comma, missing the CD4 cells and matching others. A quoted value is now one
  label; bare words are still separated by commas.
- **Queries compare numbers and booleans by value**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). Equality compared
  each value's text, so `total_counts != 0` kept the zeros of a float32 column,
  `is_doublet == true` matched nothing, and float32 values of 0.1 missed
  `pct_mt <= 0.1`. Values now compare at the column's dtype, as pandas compares
  them. A missing number equals no value, and `nan` selects it. A value of the
  wrong type is refused.
- **Queries need no spaces around operators**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). `n_counts>1000`
  failed with "Unexpected end of query."; it now parses. A lone `=` or `!` says
  which operators there are.
- **obsm and varm arrays of any rank are subset**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). An obsm or varm
  array that was not 2-D was copied whole, so anndata could not read the
  output, and with `--inplace` the source became unreadable. They are now
  subset on their first axis. An element subset cannot narrow, such as an
  awkward array, is refused before anything is written.
- **obsm and varm dataframes drop unused categories, as anndata does**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). `subset` and
  `split` now narrow their categorical columns as they narrow obs and var. A
  categorical column stored without encoding attributes is now narrowed with
  the rest of its frame instead of being copied whole.
- **split keeps missing labels apart from empty ones**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). Rows with no value
  and rows labelled `""` went into one `NA` store. Missing values, NaN
  included, now get their own `NA` store, and the manifest lists their value as
  empty.
- **Output names never share a file**
  ([#94](https://github.com/cellgeni/adata-cli/issues/94)). Names for `split`,
  `convert --split-feature-types` and `--all-samples` could collide: labels
  `A+`, `A` and `A_1` gave `A`, `A_1` and `A_1`, and labels differing only in
  case shared a file on macOS and Windows. The conversion then failed with a
  false "another adata run is writing" message. Each name is now distinct,
  case-insensitively.
- **`concat` reads CSC inputs once, not once per column**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  Each column of each input was a separate read that decompressed a whole
  chunk: 37 times the inputs' size was read on h5ad and 572 times on Zarr,
  so atlas-sized CSC inputs never finished. Inputs are now read in blocks
  and the columns put together in a scratch file beside the output, which
  the free-space check counts.
- **`concat` takes any number of inputs**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  Every input was held open, so concat failed with "Too many open files" at
  about 253 h5ad inputs on macOS (1,020 on Linux), and memory grew with the
  number of inputs. Inputs are now opened as they are needed, at most 64 at
  once.
- **Dense matrices keep their compression in `concat`**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  Dense X, layers and obsm arrays were written uncompressed and unchunked,
  whatever the inputs were: 34 times larger for a matrix of zeros.
- **`concat --merge` keeps varp**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  varp was always dropped, with a reason that holds only for obsp. It now
  follows `--merge` as varm does, sparse entries compared by value as
  anndata compares them.
- **`--fill-value` no longer makes integer matrices float64 for nothing**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  An integer X became float64 even on an inner join, where nothing is
  filled. It now widens only when an outer join fills cells with a value
  the dtype cannot hold.
- **Less memory per row in `concat` and `recast --layout csr`**
  ([#93](https://github.com/cellgeni/adata-cli/issues/93)).
  Row pointers and per-row counts were Python lists, about 36 bytes a row
  against 8: 400 MB more at 10M cells.
- **A probe-level (Flex) matrix exports as Cell Ranger writes one**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). The export used to
  write gene ids (suffixed ones after `--make-unique`) as `features/id` and
  drop the probe ids. Now:
  - boolean fields such as `filtered_probes` stay boolean, where they were
    written as `'True'`/`'False'` strings that read back as always true;
  - untagged fields, Cell Ranger's tag keys and `filtered_barcodes` are written
    back, from scanpy-made stores too.
- **Naming one column on 10x export keeps var names that are ids**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)).
  `--gene-names-column` no longer gives a false "no gene ids" refusal on a
  store named by id, and `--gene-ids-column` no longer writes ids as names.
- **float16 counts are written to and read from 10x HDF5 as float32**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). scipy has no
  float16 sparse matrix, so anndata could not read such an import.
- **Text outside ASCII is written to 10x HDF5 as Cell Ranger writes it, as XML
  character references, with a warning**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). UTF-8 bytes in
  ASCII-typed strings made `scanpy.read_10x_h5` fail.
- **A malformed 10x matrix is refused wherever it is malformed**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). Every feature
  array, `filtered_barcodes`, the start, end and order of `indptr`, and each
  entry's feature index are now checked. Only the ids, names, barcodes and the
  length of `indptr` were before.
- **`--genome` with `--feature-type` is refused when that genome has no feature
  of the type** ([#99](https://github.com/cellgeni/adata-cli/issues/99)). It
  used to write an output with no features.
- **CellBender: a `--min-cell-probability` that keeps no droplet is refused**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). It crashed an h5ad
  output part-way and wrote a Zarr store of no cells.
- **CellBender: `--from cellbender` refuses a file without latents, and a
  `--raw` that is not a 10x matrix is refused clearly**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). A plain 10x file
  used to give a fake "CellBender 0.2" result, and a non-HDF5 `--raw` gave
  h5py's "file signature not found".
- **CellBender 0.2: global arrays stay in `uns["cellbender"]` whatever their
  length** ([#99](https://github.com/cellgeni/adata-cli/issues/99)). The loss
  per epoch went into var whenever it happened to be as long as the features.
- **CellBender: `layers["cellranger"]` is laid out as `X`**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). It has the same
  index dtypes, chunks and compression, where it used to get int32 indices and
  the defaults.
- **MTX: matrix folders that are not Cell Ranger 2 genomes are no longer taken
  for genomes** ([#99](https://github.com/cellgeni/adata-cli/issues/99)). A
  run's `filtered/` and `raw/` (STARsolo's `Gene/`, Cell Ranger's `outs/`) and
  Cell Ranger multi's `sample_filtered_feature_bc_matrix/` are refused, and the
  message names the folder to give.
- **MTX export refuses names pandas, R or scanpy would misread**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). The checks run
  before the folder exists. Refused are:
  - a field starting with a double quote;
  - an empty id, type or barcode;
  - a barcode containing a comma.
- **Parse: gzipped `all_genes.csv` and `cell_metadata.csv` are read**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). A gzipped matrix
  already was.
- **Docs no longer claim more than holds**
  ([#99](https://github.com/cellgeni/adata-cli/issues/99)). The 10x imports
  match scanpy's columns with `--var-names gene_symbols`, and the differences
  are listed. The Parse page no longer claims split-pipe's older `all-well`
  layout.
- **Loom export keeps anndata 0.7 categoricals' labels**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  A categorical stored as anndata 0.7 wrote it (integer codes pointing at
  `__categories`) was written to the loom as its codes; it is now written
  as its labels, as a current categorical is, and left out of a SCope
  table, which holds numbers only.
- **`--scope` exports from Zarr stores**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  A SCope loom export from any Zarr store, v2 or v3, failed before writing
  with `object of type 'Array' has no len()`; it now writes the same loom
  as from an h5ad.
- **MuData export reads wide dense matrices in bounded blocks**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  Dense X, layers and raw were read 16,384 rows at a time however wide,
  so a wide matrix was held whole or nearly; each block now holds at most
  64 MiB.
- **Loom import never drops the main matrix**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  When X was a layer (`spliced`) and the loom also had a layer named
  `matrix`, the main matrix was not written. Such a loom was written from
  an AnnData, whose X the main matrix holds, so X is now that matrix and
  every layer keeps its name; a loom round trip gives back the same
  matrices. With `--layer spliced` the main matrix is `layers["matrix_1"]`.
- **A float16 loom imports as anndata can read it**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  A float16 loom matrix, or one exported from a float16 X, became a
  float16 sparse matrix, which anndata cannot read; matrices that go
  sparse are now float32, with a warning, and `--dense` keeps float16.
- **Loom export from a store without X**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  `--layer` on a store without X failed half way with an HDF5 error; the
  layer is now written and X left out. Without `--layer` such a store is
  refused before anything is written, naming the layers there are. A layer
  named `X` is written from itself, not from X, which goes beside it as
  `X_1`.
- **Loom export warns about every name it leaves out**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  An obs column named `CellID`, or var columns named `Gene` or `Accession`
  beside the columns those come from, were dropped without a word; they
  are left out with a warning, as the docs say. An obs column and an obsm
  entry both named `ClusterID` no longer fail mid-write: the column is
  kept and the entry left out with a warning.
- **Loom metadata reading is bounded in size**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)). A SCope loom's
  compressed `MetaData` is decompressed only up to a limit, and a larger one is
  refused as malformed, so a crafted loom cannot exhaust memory.
- **The loom page matches the code**
  ([#100](https://github.com/cellgeni/adata-cli/issues/100)).
  The flag table gives every default, and the page counts four differences
  from `anndata.read_loom`, as the code does.
- **The `py3.13` CI leg runs Python 3.13**
  ([#102](https://github.com/cellgeni/adata-cli/issues/102)). uv took its
  interpreter from `.python-version` before the one on `PATH`, so the leg ran
  the suite on 3.12. Each leg now sets `UV_PYTHON` and checks what it got.
- **The image is pushed only when the release passed**
  ([#102](https://github.com/cellgeni/adata-cli/issues/102)). It was built and
  pushed on any tag, moving `:latest` even when tests failed or the tag named
  the wrong version. It is now built and smoke-tested with the tests, before
  PyPI, and pushed after PyPI has the release. `:latest` moves only to the
  newest final version.
- **A TestPyPI dispatch no longer publishes to PyPI**
  ([#102](https://github.com/cellgeni/adata-cli/issues/102)). Run from a tag,
  it published to both. A PyPI dispatch now has to be run from a tag, and each
  Publish job asks only for the permissions it uses.
- **The benchmark's "vs previous" is the release before**
  ([#102](https://github.com/cellgeni/adata-cli/issues/102)). It compared with
  an arbitrary earlier run (0.17.0 with 0.8.0); it now takes the previous
  version at the same tier, and the page names it.
- **Least privilege in the release workflows**
  ([#102](https://github.com/cellgeni/adata-cli/issues/102)). The benchmark no
  longer runs unpinned packages with a token that can push to main. Actions and
  uv are pinned, and tags reach scripts through `env:`.

## 0.17.4

The first wave of fixes from the October 2026 audit
([#87](https://github.com/cellgeni/adata-cli/issues/87)), on data safety. No
command can now lose your data and still exit 0: every writer stages its
output and checks it first, and a write that fails as the store is closed
fails the command. Text, MTX and dense imports refuse what they cannot read
rather than shift values; `concat` refuses repeated var names and keeps
missing values and column dtypes, as anndata does; `import` and `export` type
values as pandas does. Pillow moves to 13.0.0 for its decoder advisories.

### Changed

- **Dependencies are declared, capped and tested at their floors**
  ([#98](https://github.com/cellgeni/adata-cli/issues/98)).
  numcodecs, which the Zarr v2 compressors import, is declared. h5py,
  numpy and zarr are capped below their next major, and typer below 1.0.
  A CI job installs each direct dependency at its floor, on Python 3.12
  and 3.13, and runs the suite on them. Floors that could not install or
  work were raised: the `pickle` extra now needs pandas 2.2.3 and scipy
  1.15, and the `spatial` extra pyarrow 18.

### Fixed

- **A failed final write no longer replaces your data with an unreadable
  store** ([#89](https://github.com/cellgeni/adata-cli/issues/89)). When the
  last flush to disk failed as a store was closed (a full disk or quota), the
  error was swallowed, the unreadable store was moved into place with exit code
  0, and `--inplace` deleted the source. A Zarr v2 store whose index failed to
  rewrite hid what had just been imported. Both now fail the command and leave
  the source and any existing output as they were.
- **`export`, `import` and `create` write as safely as the other commands**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). `export … -o`
  naming the input store truncated it; a failed export or import left a partial
  file or store in place of an existing one, or destroyed elements of the
  source with `--inplace`; `create --force` deleted the old store before
  writing. All three now write beside the destination and move the result in
  only once complete. `import --inplace` therefore copies the store first, and
  needs the time and the room for that copy.
- **An output that is not a store is never replaced**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). `subset data.h5ad
  -o results/` deleted the folder and everything in it; so did recast, `create
  --force` and `convert --overwrite`. A folder that is not a Zarr store, or a
  file that is not HDF5, is now refused, and so is an output that is the input
  under a case-only alias or a hard link.
- **An empty selection is refused**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). A name list that
  matched nothing (barcodes without their `-1`, say) wrote an empty store, and
  with `--inplace` replaced the source with it.
- **`--chunk` must be at least 1**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). A zero or negative
  value wrote an all-zero dense matrix with exit code 0; it is now a usage
  error on every command.
- **`--inplace` with `-o` is refused**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). `-o` was silently
  ignored and the source rewritten.
- **`--inplace` keeps symlinks and permissions**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). A symlinked store
  had its link replaced by a private copy, leaving the real store unchanged,
  and a `0640` store came back world-readable.
- **Two runs writing one output can no longer mix their data**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). Each run now claims
  its output with a lock file (`out.h5ad.adata-lock`); a second run is refused
  while the first writes.
- **A scheduler's SIGTERM cleans up**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). A time limit,
  `scancel` or `bkill` left a partial `*.adata-tmp.*` that blocked the next
  run; the run now removes it and exits with code 143. When leftovers do remain
  (after SIGKILL), the refusal names all of them at once.
- **Free space is checked before writing**
  ([#89](https://github.com/cellgeni/adata-cli/issues/89)). A disk that fills
  during an HDF5 write can stop the process inside the HDF5 library, past any
  cleanup. Commands that write a store now refuse to start when the output's
  filesystem has less free space than the estimated output; `--no-space-check`
  skips the check. A replaced output is also synced to disk before the old one
  is deleted.
- **`concat` refuses inputs whose var names repeat**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). Two columns with
  one name were merged into one target column: CSR summed them into the second,
  dense kept the last, an outer join lost a variable, and the command exited 0.
  The error names the repeated names; make them unique first, as anndata.concat
  also requires.
- **`concat` reads anndata 0.7 categoricals by label**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). They were
  concatenated as their integer codes, so code 0 could mean a different label
  in each input.
- **A text column with missing values opens in anndata 0.8, 0.9 and 0.10**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). An outer join wrote
  a text column that an input lacked as a nullable-string-array, which those
  releases cannot read at all. It is now a categorical with the missing rows
  coded -1, as anndata writes it, unless an input already used a
  nullable-string-array.
- **`concat` keeps missing values, and the dtypes of mixed columns**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). Missing values in a
  column that is categorical or nullable in one input and plain text in another
  became empty strings. A column plain in one input and nullable in another
  (int64 and Int64, bool and boolean) was dropped; it is now nullable. A
  nullable integer with a nullable boolean is a nullable integer, not text. A
  boolean column an input lacks is nullable-boolean, not float. Integer
  categories stay integers.
- **`concat` stacks sparse obsm entries**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). They were dropped.
  An entry stored differently across inputs is refused before writing, as X is.
- **`--uns-merge` merges nested mappings as anndata does**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). `first` and `only`
  now decide key by key inside nested mappings, `same` and `unique` no longer
  leave empty mappings behind, and text matches across an h5ad and a Zarr input
  under `--merge`/`--uns-merge same` and `unique`.
- **CSR output stays sorted, and inputs without variables concatenate**
  ([#92](https://github.com/cellgeni/adata-cli/issues/92)). An input whose var
  order differed from the output's left column indices unsorted within rows.
  Inputs with no variables were refused with "Use --join outer" even under
  `--join outer`.
- **Text import refuses an empty field instead of shifting the values after
  it**
  ([#90](https://github.com/cellgeni/adata-cli/issues/90)). pandas writes NaN
  as an
  empty field, and GEO tables often end every line with a tab. Such a file was
  read with every later value of its block one column to the left, and the
  block's last values taken from uninitialised memory, and the command exited
  0.
  It is now refused with the field's line and column, as anndata refuses it.
- **Text import keeps a fraction that ends a block**
  ([#90](https://github.com/cellgeni/adata-cli/issues/90)). When a block's only
  non-integer was its last value, the block was read as integers, so 4.5
  became 4.
- **Text import drops a UTF-8 byte order mark**
  ([#90](https://github.com/cellgeni/adata-cli/issues/90)). It stayed in the
  first name, and before a number it made the first column read as row names.
- **MTX import checks each line's field count**
  ([#90](https://github.com/cellgeni/adata-cli/issues/90)). A line a field
  short
  and a later one a field long were paired up silently, putting values at the
  wrong coordinates. The first wrong line is now refused by number, counted
  past any blank lines in the header. Blank lines among the entries are
  skipped, as scipy skips them; before, one was refused unless a block
  happened to end on it.
- **HDF5 and loom import keep memory bounded whatever the chunking**
  ([#90](https://github.com/cellgeni/adata-cli/issues/90)). A dataset chunked
  one
  gene to a chunk, or with every cell in one chunk, was read whole in one slab.
  It is now read a few chunks at a time and its entries sorted into cells on
  disk. Chunks a little wider than a slab are cut into slabs. The on-disk
  buckets that sort entries (also used by shuffled MTX files and
  `--transpose` text) no longer hold a chunk cache each, which made their
  memory grow with the number of buckets.
- **`import dataframe` typed columns unlike pandas, and held the whole CSV**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). A number column
  with a gap was stored as strings, True/False as a categorical of strings, and
  a row with too few fields was taken with NaN in its missing cells. The CSV is
  now read in two streaming passes and typed as `pandas.read_csv` types it, so
  a gap makes a number column float64. A short row, a repeated or missing
  column name, and a `--categorical` name the file does not have are refused
  before the store is touched.
- **CSV labels such as `1_1` were read as numbers**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). `1_1`, `1_10` and
  `11_0` became 11, 110 and 110, in `import dataframe` and in the obs `convert`
  writes from Parse, Visium and Xenium tables. A value is now a number only
  when pandas reads it as one. Whole numbers past int64 are uint64 instead of
  rounded floats, and a column of True/False followed by integers is text
  instead of crashing the import.
- **`import array`, `sparse` and `dict` could replace `obs`, `obsm` or `layers`
  with an array** ([#91](https://github.com/cellgeni/adata-cli/issues/91)). The
  command exited 0 and anndata could then not read the store; with `--inplace`,
  that store was the source. An element of the wrong kind for its path is now
  refused before anything is written. Name an entry inside the group, such as
  `obsm/X_pca`.
- **A JSON `null` made a store unreadable to anndata 0.8 to 0.11**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). `import dict`, and
  `convert` of a Space Ranger scale factors file that holds one, wrote
  `encoding-type: null`. A key holding null is now left out with a warning, as
  those releases leave out `None`.
- **`import dict` turned mixed arrays into strings value by value**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). `[1, null]` became
  `['1', 'None']`, and an integer past 64 bits made the store unreadable. An
  array whose values share a type is now typed, with NaN for a null among
  numbers; any other array is kept as its JSON text. Integers past 64 bits, and
  keys that are empty, hold `/` or are named like Zarr's metadata files, are
  refused.
- **`import array` held the whole `.npy` in memory**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). The file is now
  mapped and copied a block of rows at a time.
- **`export array` dropped missing values, could not export text, and held its
  output in memory** ([#91](https://github.com/cellgeni/adata-cli/issues/91)).
  A nullable integer or boolean array with gaps is now float64 with NaN in
  them, and text is written as fixed-width unicode. The output streams to a
  file or to stdout, instead of going through a memory map or a copy of the
  whole array.
- **An export piped into `head` reported an error**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). `adata export ... |
  head` now ends quietly with exit code 0. `export dataframe` also checks
  `--columns` before creating its output and ends lines in `\n`, not `\r\n`.
  `--head 0` there and in `export sparse` writes only the header.
- **`export dict` wrote `NaN` and `Infinity`, which are not JSON**
  ([#91](https://github.com/cellgeni/adata-cli/issues/91)). They are now
  written as `null`, with a note that says how many.
- **Pillow 13.0.0, for the image decoder advisories**
  ([#98](https://github.com/cellgeni/adata-cli/issues/98)).
  `adata import image` and the Visium and Xenium tissue images open
  user-supplied files with Pillow. 12.1.0, which the lock and the container
  image pinned, has published heap-overflow and out-of-bounds-write
  advisories in its decoders. The floor is now `pillow>=13.0.0`, and the
  lock also takes pygments 2.20.0 and pytest 9.0.3 (dev) for theirs.

## 0.17.3

Standard Visium spots and Visium HD's 8 and 16 µm bins now bring in Space
Ranger's own clusters, UMAP and PCA, as the segmented cells and Xenium
already did, from today's `analysis/` layout and Space Ranger 1's. `adata
convert` now works on a fresh install, which it did not from 0.7.0 to 0.17.2.

### Added

- **Space Ranger's analysis for spots and bins** ([#85](https://github.com/cellgeni/adata-cli/pull/85)).
  `adata convert outs -o section.h5ad` reads `analysis/` beside the matrix
  when it is there:
  - `obs["graphclust"]`, a categorical;
  - `obsm["X_umap"]` and `obsm["X_pca"]`.

  Space Ranger 1's folder names (`graphclust/`, `2_components/`,
  `10_components/`) are read too. The 2 µm bins have no `analysis/`.
  - Space Ranger leaves bins with no counts out of its analysis (half the
    8 µm bins of 10x's tiny sample) and analyses only the filtered spots
    and bins, so with `--unfiltered` the rest have none. Those rows are
    left empty, with one warning for tables that lack the same rows.
  - A table from another run, with a barcode twice, or with a projection
    that is not all numbers is refused before anything is written.
  - Checked in CI against squidpy and scanpy with pandas reading the
    tables. On 10x's tiny HD sample and SR 1.1 Adult Mouse Brain, the
    values equal pandas' exact parse; the 8 µm import peaks 13 MB higher
    than in 0.17.2.

### Changed

- **Analysis projections are scanned without keeping their values.** The
  scan that types a table's columns kept each column's distinct values for
  categories, up to 65,536 strings a column: some 60 MB for a PCA's ten
  columns, which can never be categories. Xenium and the segmented cells
  gain this too.
- **CellBender's embeddings go through the shared row scatter**
  (`core/scatter.py`) rather than a copy of it; the output is unchanged.

### Fixed

- **`adata convert` works on a fresh install** ([#88](https://github.com/cellgeni/adata-cli/issues/88)).
  typer 0.26 and later carry their own copy of click and no longer install
  it, so the click import that told typed flags from defaults crashed every
  convert on a new install of any release from 0.7.0 to 0.17.2. Where click
  was installed anyway, its enum never matched typer's, and every convert
  was refused for mixing format and matrix flags nobody had typed. Flags are
  now told apart by name, whichever click typer brings. The container image
  installs from the lock, which has click, so it was not affected.
  - A new CI job installs the package from a fresh resolution rather than
    the lock, imports every module on the base dependencies alone, and runs
    `convert` with and without click.

## 0.17.2

Space Ranger 4's segmented cells join Visium HD's bins: `convert --cells`
reads the cell matrix with each cell's centroid, its area and its nucleus's,
and Space Ranger's clusters, UMAP and PCA, reading the cells' GeoJSON
outlines a feature at a time.

### Added

- **Visium HD segmented cells** ([#83](https://github.com/cellgeni/adata-cli/pull/83)).
  `adata convert outs -o cells.h5ad --cells`, or the `segmented_outputs/`
  folder given itself, reads `filtered_feature_cell_matrix.h5` (`raw_` with
  `--unfiltered`) with:
  - `obsm["spatial"]`: each outline's area-weighted centroid in
    full-resolution pixels, as shapely computes it, so scanpy and squidpy
    plot the cells over the images;
  - `obs["cell_area"]` and `obs["nucleus_area"]`, in µm², as Xenium names
    them;
  - `obs["graphclust"]`, `obsm["X_umap"]` and `obsm["X_pca"]` from the
    cells' `analysis/`;
  - the images, scale factors and metadata in `uns["spatial"]`.

  Without `--cells` an HD output is still read as 8 µm bins, and the
  summary says the cells are there. Checked in CI against 10x's scanpy and
  geopandas recipe, and on 10x's SR 4.0.1 tiny sample against shapely.
  - The GeoJSON, one line of a few hundred MB on a full slide, is read a
    feature at a time; about 50 bytes a cell are kept, so that a cell with
    no outline is refused before anything is written. At 50,000 cells the
    import peaks at 190 MB, against 771 MB for the recipe.
- **An `import-visium-cells` benchmark case**, against that recipe.

### Changed

- **Xenium's `analysis/` reading is shared** with the segmented cells,
  which lay theirs out the same way; Xenium's output is unchanged.

## 0.17.1

Visium HD joins standard Visium and Xenium: `convert` reads a Space Ranger
3 or 4 HD output at one bin size, with its bins' positions, images and
scale factors, through a new optional `spatial` extra (pyarrow) for the
Parquet positions.

### Added

- **Visium HD** ([#81](https://github.com/cellgeni/adata-cli/pull/81)).
  `adata convert outs -o bins.h5ad` reads the 8 µm bins, as 10x recommends
  for analysis; `--bin-size 2` or `16` picks another size, and a bin folder
  given directly reads as it is. The layout is standard Visium's: obs
  `in_tissue`, `array_row`, `array_col`; `obsm["spatial"]` in
  full-resolution pixels; the bin folder's images and scale factors in
  `uns["spatial"]`. Checked in CI against scanpy with pandas reading the
  positions, and on 10x's SR 4.0.1 tiny sample at every bin size.
  - The positions are read as a buffered stream, the barcodes and the
    numbers in separate passes. On that sample the 2 µm bins (11.2 million
    positions, 330,035 in the matrix) convert in 3.5 s with a 470 MB peak
    without images, where pandas alone takes about 1 GB to read them.
- **The `spatial` extra**: `pip install 'pyadata-cli[spatial]'` adds
  pyarrow, which only Visium HD needs; the container image has it.

### Changed

- **Images are copied into the store a strip of rows at a time**, rather
  than as a second whole copy: Visium HD's hires image is 6,000 pixels, some
  100 MB. The image itself is still decoded whole, as a PNG can only be.
- **Barcodes are matched as bytes**, a quarter of the memory of numpy's
  unicode, for Visium, Visium HD and Xenium alike.

## 0.17.0

`convert` reads 10x's spatial outputs with their coordinates: a Space
Ranger folder for standard Visium, with its spot positions, tissue images
and scale factors where squidpy puts them, and a Xenium bundle, with its
cell table, centroids and Xenium's own clusters, UMAP and PCA. No new
dependency; Visium HD, whose positions are Parquet only, comes later.

### Added

- **Visium (Space Ranger 1–4)**
  ([#79](https://github.com/cellgeni/adata-cli/pull/79)). `adata convert
  outs -o section.h5ad` gives what `squidpy.read.visium` gives, checked
  against it in CI: obs `in_tissue`, `array_row`, `array_col`;
  `obsm["spatial"]` in full-resolution pixels; `uns["spatial"][library]`
  with the images (uint8, as the PNGs hold them), the scale factors and the
  metadata.
  - New flags `--library-id` and `--no-images`; `--unfiltered` reads
    `raw_feature_bc_matrix.h5`, as it reads Parse's unfiltered matrix.
  - Space Ranger 1's headerless `tissue_positions_list.csv`, decimal
    positions, and 10x's downloads named with their sample are read.
  - Refused before anything is written: a barcode with no position, aggr
    output, Visium HD, and a folder whose matrix is Matrix Market only.
- **Xenium (XOA 1–4)**
  ([#79](https://github.com/cellgeni/adata-cli/pull/79)). `adata convert
  bundle -o cells.h5ad` puts `cells.csv.gz`'s columns in obs and the
  centroids in `obsm["spatial"]` (µm), and Xenium's graph-based clusters,
  UMAP and PCA in `obs["graphclust"]`, `obsm["X_umap"]` and `obsm["X_pca"]`,
  read from `analysis.tar.gz` as Xenium ships it. Checked against 10x's
  scanpy recipe in CI.
  - The cell table is matched to the matrix while it is scanned and
    written straight through, as Xenium writes both in one order; a table
    in another order goes through scratch buckets (`core/scatter.py`), so
    memory stays at a block either way.
- **An `import-xenium` benchmark case**, against 10x's scanpy recipe.

### Fixed

- **A Space Ranger `outs/` folder was refused**
  ([#79](https://github.com/cellgeni/adata-cli/pull/79)) as a Cell Ranger 2
  MTX layout ("pick one with --genome"): its two MEX folders were taken for
  genomes. A Xenium bundle likewise.
- **A CSV saved with a byte order mark**, as Excel saves one, kept the
  mark in its first column's name, so a table looked up by that name was
  refused as lacking it, or a Visium positions file taken for one with no
  header.

## 0.16.2

`export image` writes to stdout without `--output`, as every other export
does, and the documentation is a site with a sidebar and search, guides
and a troubleshooting page.

### Changed

- **The documentation is a Just the Docs site**
  ([#76](https://github.com/cellgeni/adata-cli/pull/76)), laid out as
  cellgeni's pipeline docs: Home, Getting started, six guides (inspect and
  export, subset and split, concatenate, recast, convert formats,
  pipelines and containers), Reference, Troubleshooting, Developers and
  this changelog, at <https://cellgeni.github.io/adata-cli/>.
  - The command reference keeps one section per command; each format
    `convert` reads or writes has a page of its own.
  - Troubleshooting quotes the tool's own error messages, each with its fix.
  - Old URLs (`/COMMANDS.html` and five more) redirect, `#anchor` included.
  - The changelog moved to `docs/changelog.md`, and the README is half
    its length.

### Fixed

- **`export image` without `--output` writes PNG to stdout**
  ([#77](https://github.com/cellgeni/adata-cli/pull/77)). It crashed with
  `'NoneType' object has no attribute 'parent'`, where every other export
  writes to stdout. With `-o`, the format still follows the extension.

## 0.16.1

`subset`, `split` and the MuData export now narrow categorical columns as
anndata's own subsetting does, so their obs and var equal what anndata
gives, and outputs that kept every label of a large categorical are smaller.

### Fixed

- **A subset keeps only the categories its rows use**
  ([#74](https://github.com/cellgeni/adata-cli/pull/74)). anndata drops a
  category no kept row uses, on obs and var (the axis kept whole too), and
  narrows `uns["<col>_colors"]` with it; `subset`, `split` and the MuData
  export kept every category. A peaks subset of a 10x Multiome matrix held
  all 134,905 interval labels for 98,319 peaks, and its `.h5mu` was 84.7 MB
  where it is now 74.2 MB.
  - Colours follow anndata's rule: those of the categories kept, along the
    first axis (RGBA rows too); a list of the wrong length is dropped, a
    single colour becomes a list of one; obs before var.
  - Order and `ordered` are kept. A whole AnnData written to `.h5mu` with
    `--mod` keeps its categories, as nothing is subset.

## 0.16.0

Reads and writes MuData, the container muon, mudata and SCENIC+ keep
multimodal data in: an `.h5mu` (or a MuData Zarr store) converts to one
AnnData per modality, with the joint results on its cells, and an AnnData
converts back to an `.h5mu`, split by feature type. Copies of anndata 0.7's
dataframes, which every copy broke, now come out readable.

### Added

- **`adata convert x.h5mu -o x.h5ad` writes each modality as an AnnData**
  ([#31](https://github.com/cellgeni/adata-cli/issues/31)), to
  `x.<mod>.h5ad`, or the one `--mod` names:
  - the modality as it is stored;
  - the global obs columns, obsm and obsp (a joint clustering, a WNN
    embedding, the joint graph) taken onto its cells through `obsmap`, in
    its order, a block at a time; graphs keep CSR or CSC;
  - a global name the modality already has keeps the modality's own, with
    a warning; a column pulled from a modality (`<mod>:col`) goes back
    only to that modality.

  Without maps, cells are matched by name; a map that does not place each
  cell once is refused before anything is written. A global sparse element
  keeps its index dtypes, compression and duplicate entries. An `.h5mu` is no longer
  taken for an h5ad, nor a MuData Zarr store for a Zarr AnnData. On muon's
  `minipbcite.h5mu`, both modalities equal what mudata reads.
- **`adata convert x.h5ad -o x.h5mu` writes a MuData**
  ([#31](https://github.com/cellgeni/adata-cli/issues/31)). var is split
  by feature type as muon does (Gene Expression → `rna`, Peaks → `atac`,
  Antibody Capture → `prot`), or the whole AnnData is written under
  `--mod`:
  - each modality keeps X, layers, var, varm, varp and raw for its
    features, every matrix read once for all of them; raw's features are
    shared out by raw/var's own feature types, so genes and peaks var was
    filtered of still land in their modality (with `--mod`, raw is kept
    whole);
  - obs, obsm, obsp and uns are written once, globally;
  - the maps, masks and userblock mudata expects are written too.

  mudata reads the result without a warning; a 10x Multiome matrix goes to
  `.h5mu` in two steps.

### Fixed

- **Copies of anndata 0.7's dataframes are readable**
  ([#72](https://github.com/cellgeni/adata-cli/pull/72)). Their categoricals
  point into `__categories` by an HDF5 object reference, which cannot
  leave the file: `subset` wrote an h5ad that anndata could not read, and
  `subset` or `convert` to Zarr stopped with "Object of type Reference is
  not JSON serializable". Every copy now writes them as current
  `categorical` groups, in a dataframe of the current version.

## 0.15.0

Writes SCope looms, so results from pySCENIC, SCENIC+ or pycistopic can go
back to the SCope viewer after editing as AnnData. A SCope loom now
converts to h5ad and back with its regulons, embeddings and clusterings.
`concat` keeps what those imports leave in obsm and varm.

### Added

- **`adata convert ... -o x.loom --scope` writes a SCope loom** ([#61](https://github.com/cellgeni/adata-cli/issues/61)):
  - obsm and varm dataframes (`RegulonsAUC`, `Regulons`,
    `ClusterMarkers_<id>`) become compound attributes;
  - two-column embeddings become `Embeddings_X`/`_Y`, the default also as
    `Embedding`;
  - the clusterings a SCope import recorded, plus those `--clusterings`
    names, become `Clusterings`;
  - `MetaData` describes the embeddings, clusterings, regulon thresholds,
    metrics and annotations. Ids and names recorded on import are kept.

  The rules SCope's server reads by are kept:
  - clustering and cluster ids are 0..n-1 in order;
  - regulon names have no spaces, in both the AUC and the Regulons
    attributes;
  - cluster markers are written only with their metrics, since SCope
    crashes on them otherwise.

  A loom round trip keeps what was edited in between: `ClusterID` comes
  from the first clustering, the thresholds from the `regulon_thresholds`
  table, the marker metrics from the marker tables written. A
  non-categorical clustering, or embeddings, clusterings or a table too
  many for one HDF5 compound attribute, are refused before writing. On the public pySCENIC loom
  (Zenodo 10944066), loom → h5ad → SCope loom gives back the same AUC,
  regulons, embeddings, clusterings and thresholds through the pySCENIC
  tutorial's recipe.

### Fixed

- **`concat` stacks dataframes in obsm and merges varm**
  ([#61](https://github.com/cellgeni/adata-cli/issues/61)). It skipped
  obsm dataframes (a SCope import's `RegulonsAUC`) as "not a dense array",
  and never wrote varm, though `--merge` was meant to cover it, as in
  anndata.
  - obsm dataframes are now stacked as obs is, their columns following
    `--join`.
  - varm is merged as var's columns are, on the output's genes, compared
    and copied a `--chunk` of genes at a time, its chunking and
    compression kept.

  Both match `anndata.concat` for each join and merge strategy.
- **A loom export no longer stops half written when two elements ask for
  one attribute name.** An obs column and an obsm entry of one name (or a
  var column named like an attribute loom writes itself) were both written
  to the same `col_attrs`/`row_attrs` name, and the second write failed.
  The first now keeps the name, and the others are left out with a
  warning, decided before anything is written.

## 0.14.0

Reads pycisTopic's pickles, the only way pycisTopic saves its objects,
without pycisTopic and without running anything from the file. pandas and
scipy, which rebuild the pickle's tables and matrices, come with a new
optional extra: `pip install 'pyadata-cli[pickle]'`. The Docker image
includes the extra.

### Added

- **`adata convert` reads pycisTopic pickles** ([#28](https://github.com/cellgeni/adata-cli/issues/28)),
  for pycisTopic's `CistopicObject` and its `CistopicImputedFeatures`.
  - **CistopicObject:**
    - `fragment_matrix` becomes X, CSR cells x regions with its dtype kept,
      and `binary_matrix` becomes a layer;
    - `cell_data` and `region_data` become obs and var, column dtypes kept
      (categoricals, nullable integers and booleans);
    - the selected model's topics become `obsm["X_topic"]` and
      `varm["topic_region"]`;
    - projections become `obsm["X_<name>"]`;
    - the project, fragment files and the model's tables go into
      `uns["pycistopic"]`.
  - **CistopicImputedFeatures** (imputed accessibility, gene activity)
    becomes its own AnnData, CSR or `--dense`.

  The file is read by an unpickler that resolves only the names such an
  object holds: numpy, scipy.sparse, pandas and a few builtins. Any other
  name is refused, so a pickle that names `os.system` is not run.
  pycisTopic pins pandas 1.5, and its pickles read with pandas 2 and 3, as
  CI checks.

  A public GEO pickle (GSE292438, 5,027 cells x 124,083 regions, 15
  topics; 76 MB gzipped, 328 MB unpacked) converted in 8.8 s at a 842 MB
  peak, the pickle being held whole. Every matrix, column and topic table
  equalled what `pickle.load` gives under pycisTopic's pandas 1.5.

### Fixed

- **Error messages keep their square brackets**
  ([#28](https://github.com/cellgeni/adata-cli/issues/28)). Errors and
  warnings were printed as Rich markup, so a bracketed word that read as a
  tag vanished: the install hint `'pyadata-cli[pickle]'` printed as
  `'pyadata-cli'`.

## 0.13.0

Adds the last of scanpy's core readers to `adata convert`: dense matrices
in delimited text, as `read_csv` and `read_text` read them, and one dataset
of a plain HDF5 file, as `read_hdf` does. Both are streamed, so memory is
set by a block of the file rather than by the matrix, and X is CSR by
default where anndata gives a dense float32 array.

### Added

- **`adata convert` reads text matrices** ([#25](https://github.com/cellgeni/adata-cli/issues/25)):
  `.csv`, `.tsv`, `.tab`, `.txt` and `.data`, gzipped or not.
  - The layout is found as anndata finds it: a header with a corner cell
    (pandas) or one field short (R's `write.table`), column names in the
    last `#` comment line, or no header; row names or none.
    `--delimiter` and `--first-column-names` / `--no-first-column-names`
    override the detection.
  - `--transpose` reads genes as rows, as count tables often store them.
  - Integers stay integers (int32 or int64, exact past 2**53); other
    values are float64.
  - A ragged line, or a value that is not a number, is refused by line and
    column before any output exists.

  A 400 MB CSV of 100,000 cells x 2,000 genes took 14 s at a 159 MB peak.
  `anndata.read_csv` plus `write_h5ad` took 12.6 s at 1.3 GB. A public GEO
  table (GSE75748, 1,018 cells x 19,097 genes, gzipped, genes as rows)
  matched `read_csv(...).T`. It took 7.5 s at 222 MB, against 2.9 s at
  315 MB: small tables are slower read twice.

- **`adata convert` reads one dataset of a plain HDF5 file** ([#25](https://github.com/cellgeni/adata-cli/issues/25)):
  - `--key` names it. Without it, the refusal lists the 2-D numeric
    datasets and the 1-D ones that could name their rows and columns.
  - Names come from `read_hdf`'s `row_names` and `col_names`, or from
    `--obs-names-key` and `--var-names-key`.
  - `--transpose` reads genes as rows.
  - The dataset is read a slab of whole chunks at a time, and keeps its
    dtype, complex included.
  - `.hdf5`, `.he5` and `.hdf` files are recognised by name. A plain `.h5`
    that is not a 10x matrix needs `--from hdf5`, and `convert` now says
    so.

## 0.12.0

Reads SCope looms: what pySCENIC, SCENIC+ and pycistopic write for the
SCope viewer. Their regulons, embeddings, clusterings and metadata, which
`anndata.read_loom` refuses and 0.11 left out, now convert with the rest
of the loom.

### Added

- **`adata convert` reads SCope looms** ([#27](https://github.com/cellgeni/adata-cli/issues/27)):
  - each regulon's AUC per cell and its target genes (or regions) as
    dataframes with the regulons' names as columns, `obsm["RegulonsAUC"]`
    and `varm["Regulons"]`; any other compound attribute, such as VSN's
    cluster markers, becomes a dataframe the same way;
  - the embeddings as `obsm["X_<name>"]` and the clusterings as categorical
    obs columns, named and labelled from the loom's `MetaData`;
  - `MetaData`, plain or compressed, decoded into `uns["scope"]`, with the
    regulon thresholds as a dataframe and each embedding's and clustering's
    SCope id and name.

  Region-based looms (SCENIC+, pycistopic) are named by region. On a public
  pySCENIC loom (Zenodo 10944066: 500 cells, 298 regulons, 5 embeddings)
  this took 0.5 s at a 143 MB peak and matched the pySCENIC tutorial's
  loompy recipe value for value, as CI checks on a fixture. The tables are
  read a block of cells at a time: a 2.8 GB loom of 600,000 cells and 600
  regulons took 6.4 s at a 294 MB peak. Writing a loom still writes a plain
  one; SCope export is [#61](https://github.com/cellgeni/adata-cli/issues/61).

### Fixed

- **R's factors and logicals in looms read as R meant them**
  ([#27](https://github.com/cellgeni/adata-cli/issues/27)). R's writers
  store a factor as an HDF5 enum, which came out as its codes: the
  `CellType` of SCENIC's example loom as 1 to 5. A factor is now a
  categorical of its levels, in R's order, and a logical (FALSE, TRUE, NA)
  a nullable boolean.

## 0.11.1

Reads the looms other tools write, as found converting 29 real looms from
Baysor, R's LoomExperiment and anndata/loompy: all now convert, and match
`anndata.read_loom` where it can read them.

### Fixed

- **A `.loom` holding the matrix alone is read**
  ([#59](https://github.com/cellgeni/adata-cli/pull/59)). Baysor writes some
  looms as `/matrix` with no attribute groups, which were not recognised.
  In a file named `.loom` that is now a loom, its cells and genes numbered;
  a 7,979,604 x 8,729 one converted in 10.5 minutes at a 211 MB peak.

- **Baysor's gene names are used** ([#59](https://github.com/cellgeni/adata-cli/pull/59)).
  Baysor names genes in a `Name` attribute; its looms came out with
  numbered genes.

- **R's LoomExperiment looms read cleanly** ([#59](https://github.com/cellgeni/adata-cli/pull/59)).
  The spec version, stored as a one-element array, showed as `['2.0.1']`,
  and graph vertices stored as floats were truncated without a check; they
  must now be whole numbers. loompy, and so `anndata.read_loom`, refuses
  these files.

## 0.11.0

Adds loom to `adata convert`, both ways. A velocyto, loompy or
kallisto|bustools loom converts a slab of cells at a time, with the
matrices sparse, their dtypes kept, and the graphs and global attributes
that `anndata.read_loom` drops; an AnnData exports to a loom that loompy
validates.

### Added

- **`adata convert` reads looms** ([#26](https://github.com/cellgeni/adata-cli/issues/26)),
  spec 2.0.1 and 3.0.0, as `anndata.read_loom` reads them:
  - X is the `spliced` layer when there is one, with the main matrix kept as
    `layers["matrix"]`; `--layer` picks another;
  - attributes become obs and var columns, or obsm and varm when they have
    several columns; graphs become obsp and varp, and global attributes
    `uns["loom"]`;
  - genes are named by id (`Accession`, or kallisto|bustools' `gene_id`),
    the symbols kept as a column, unless `--var-names gene_symbols`;
    `--obs-names-attr` and `--var-names-attr` name any other attribute;
  - X and the layers are CSR, or dense with `--dense`, and keep their dtype:
    velocyto's integer counts stay integers.

  On a public 10k-cell velocyto loom this took 26 s at a 159 MB peak, where
  `anndata.read_loom` took 29 s and 9.9 GB, and gave the same matrices,
  names and columns. SCope's compound attributes are left out with a
  warning, for #27.

- **`adata convert` writes looms** ([#26](https://github.com/cellgeni/adata-cli/issues/26)):
  spec 3.0.0 as loompy 3 writes it, which loompy's validator accepts. X (or
  `--layer`) is the main matrix, with every layer, obs and var as attributes
  (`CellID`, `Gene`, `Accession`), 2-D obsm and varm, obsp and varp as
  graphs, and uns scalars as global attributes. A loom's matrices are dense:
  the summary gives their size before compression and warns above 10 GB.

## 0.10.0

Adds Parse Biosciences split-pipe output to `adata convert`, read in two
streaming passes like the MTX folders of 0.8.0. Features are now named by
gene id by default, for every format: ids are unique where symbols repeat,
so a gene-level matrix no longer needs `--make-unique` (a probe-level Flex
matrix still does).

### Added

- **`adata convert` reads Parse Biosciences split-pipe output**
  ([#30](https://github.com/cellgeni/adata-cli/issues/30)). A run folder, a
  sample folder or a `DGE_filtered`/`DGE_unfiltered` folder is recognised
  whatever its name:
  - `count_matrix.mtx` (cells x genes) becomes X through the streaming MTX
    reader; integer counts stay integers;
  - `cell_metadata.csv` becomes obs, indexed by `bc_wells`, each column
    typed as pandas reads it and stored as anndata writes it;
  - `all_genes.csv` becomes var, with `gene_symbols` and `genome`;
  - `--sample NAME` reads one sample, `--all-samples` writes one output per
    sample, and `--unfiltered` reads every barcode;
  - `--genome G` keeps G's genes and the cells split-pipe called as G.

  Checked against Parse's own scanpy recipe in CI, and on a real split-pipe
  v1.6.1 run: X, obs and var all match, for 29,588 cells and for the
  431,061 barcodes of the unfiltered folder, at a peak of about 200 MB.

- **A streaming CSV table reader** ([#55](https://github.com/cellgeni/adata-cli/pull/55)),
  `formats/table.py`: two passes over the file, columns typed as
  `pd.read_csv` types them, the `obs` dataframe written a block of rows at
  a time. It reads Parse's cell metadata, and is there for #25's delimited
  tables.

### Changed

- **Features are named by gene id by default**
  ([#54](https://github.com/cellgeni/adata-cli/pull/54)). This changes what
  `adata convert` writes from 10x HDF5, MTX folders, CellBender and Parse:
  var names are the gene ids, and the symbols are in `var["gene_symbols"]`,
  the layout `scanpy.read_10x_mtx(var_names="gene_ids")` gives. Symbols
  repeat (Cell Ranger's GRCh38-2020-A reference repeats 10), so naming by
  them needed `--make-unique`. `--var-names gene_symbols` gives the old
  names, as scanpy and CellBender's loaders give them. A probe-level (Flex)
  matrix still repeats gene ids, and still warns.

### Fixed

- **Categories are ordered as anndata orders them**
  ([#55](https://github.com/cellgeni/adata-cli/pull/55)). anndata sorts the
  categories it makes from strings naturally (`A2` before `A10`); `convert`
  sorted them plainly, so a categorical var column holding numbers in its
  values came out in a different order from scanpy's.

## 0.9.0

Adds CellBender's `remove-background` output to `adata convert`. The
denoised counts become X, and CellBender's latent variables are kept under
the names its own loaders use, so notebooks written against those keep
working. `--raw` adds the Cell Ranger counts CellBender started from.

### Added

- **`adata convert` reads CellBender `remove-background` output**
  ([#29](https://github.com/cellgeni/adata-cli/issues/29)). An `output.h5`
  or `output_filtered.h5` from CellBender 0.2, 0.3 or 0.4 is recognised by
  its latents, whatever it is called:
  - per-droplet latents (`cell_probability`, `cell_size`,
    `background_fraction`, `droplet_efficiency`; `latent_*` in 0.2) become
    obs columns, NaN for droplets CellBender did not analyse, with
    `obs["cellbender_analyzed"]` marking the ones it did;
  - the embedding becomes `obsm["gene_expression_encoding"]`
    (`latent_gene_encoding` in 0.2), the ambient profile
    `var["ambient_expression"]`, and the analysed features
    `var["cellbender_analyzed"]`;
  - everything else (fitted constants, learning curve, settings) goes to
    `uns["cellbender"]`.

  The 10x HDF5 flags apply. `--from 10x-h5` still reads such a file as a
  plain 10x matrix. Checked against real CellBender 0.2.2 and 0.3.2 output,
  full and filtered, and CellBender's own `load_anndata_from_input_and_output`:
  X, both layers, every latent at its barcode, var and `uns` all match
  ([#52](https://github.com/cellgeni/adata-cli/pull/52)).

- **`--min-cell-probability P`** ([#29](https://github.com/cellgeni/adata-cli/issues/29))
  keeps the droplets CellBender gives a cell probability of at least `P`,
  in file order. At 0.5 it picks the same cells as CellBender's own
  `output_filtered.h5`.

- **`--raw raw_feature_bc_matrix.h5`** ([#29](https://github.com/cellgeni/adata-cli/issues/29))
  adds the Cell Ranger counts of the same barcodes as `layers["cellranger"]`,
  and the denoised counts as `layers["cellbender"]`, as CellBender's loader
  does. A filtered file lists cells by count rather than in the raw file's
  order, so its raw counts go through the on-disk bucket sort the MTX
  import uses, now shared in `core/csr_build.py`
  ([#51](https://github.com/cellgeni/adata-cli/pull/51)).

### Changed

- **An output extension picks a format that can be written**
  ([#51](https://github.com/cellgeni/adata-cli/pull/51)). `.h5` stays a 10x
  HDF5 matrix, and `--to cellbender` is refused, since CellBender files can
  be read but not written.

## 0.8.0

Adds 10x Genomics and STARsolo Matrix Market folders to `adata convert`, in
both directions, velocyto's included. They are read in two streaming passes,
so a folder's size is not a limit. `import sparse` and `export sparse` move
onto the same reader and writer, which changes what `import sparse` writes
and fixes a symmetric file being read as half its matrix.

### Added

- **`adata convert` reads 10x and STARsolo MTX folders**
  ([#24](https://github.com/cellgeni/adata-cli/issues/24)). A folder
  holding `matrix.mtx`, `barcodes.tsv` and `features.tsv` (or Cell Ranger
  2's `genes.tsv`), each plain or gzipped, is recognised whatever its name:
  - Cell Ranger 3 and later;
  - Cell Ranger 2, with one subfolder per genome, picked with `--genome`;
  - STARsolo `Gene/`, `GeneFull/` and the like;
  - STARsolo `Velocyto/raw/`, whose `spliced`, `unspliced` and `ambiguous`
    matrices become layers of those names, with X a copy of `spliced`;
  - any of these with every file under one GEO-style prefix
    (`GSM123_matrix.mtx.gz`).

  The result matches `scanpy.read_10x_mtx(gex_only=False)` column for
  column, checked against scanpy in CI, except that integer counts stay
  integers. The 10x HDF5 flags apply (`--genome`, `--var-names`,
  `--make-unique`, `--feature-type`, `--split-feature-types`), as does
  `--compression`. The first pass checks the whole folder before any output
  exists; the second writes X, straight through when entries are in barcode
  order, as Cell Ranger and STARsolo write them, and through on-disk buckets
  when they are not. On 50 million entries this took the time scanpy did, at
  under a fifth of its peak memory, and the peak did not grow with the input.

- **`adata convert --to 10x-mtx` writes a Cell Ranger 3 folder**
  ([#24](https://github.com/cellgeni/adata-cli/issues/24)): gzipped (or not,
  with `--compression none`), three feature columns, entries in barcode
  order, from X or the matrix `--layer` names. Feature ids and types come
  from var as for the 10x HDF5 export. A folder imported by `convert`
  exports with Cell Ranger's `%metadata_json` intact.

- **A benchmark case for MTX import** ([#49](https://github.com/cellgeni/adata-cli/pull/49)), against scanpy's
  `read_10x_mtx`.

### Changed

- **`import sparse` keeps integer values as integers**
  ([#24](https://github.com/cellgeni/adata-cli/issues/24)). An `integer`
  Matrix Market file becomes int32, or int64 when a value needs it; a `real`
  one becomes float64; a `pattern` one becomes int32 ones. All three were
  float32, which cannot hold counts above 2^24 exactly. `adata recast
  --dtype float32` narrows on request. Duplicate entries are now summed, as
  scipy does, and reported. The data is gzipped in an h5ad, and a parent
  group made on the way (`layers`) is a proper dict.

- **`import sparse` and `export sparse` stream** ([#48](https://github.com/cellgeni/adata-cli/pull/48)), and read or
  write gzipped files (`.mtx.gz`). Import makes two passes and checks the whole file
  before the store is touched; it used to build a Python object per entry.
  Export reads `indices` and `data` once per block, where it read both once
  per row, and writes floats with the digits to read back exactly and whole
  numbers as integers.

### Fixed

- **A `symmetric` Matrix Market file was read as half its matrix**
  ([#48](https://github.com/cellgeni/adata-cli/pull/48)). It
  stores one triangle and means both, and `import sparse` read it as
  `general`. `symmetric`, `skew-symmetric`, `hermitian`, `complex` and dense
  `array` files are now refused by name.

- **`import sparse` overflowed `indptr` past 2^31 entries** ([#48](https://github.com/cellgeni/adata-cli/pull/48)); it
  widens to int64 when needed.

- **`recast --layout` spent time in proportion to the matrix on every
  block** of a streaming transpose ([#47](https://github.com/cellgeni/adata-cli/pull/47)): pass 1 counted over every output row per
  block, and pass 2 compared each block against every bucket. Both now cost
  what the block costs.

## 0.7.1

Fixes ways that `subset`, `recast`, `split` and `concat` could lose or
corrupt data while reporting success. The worst was that `subset` emptied a
Zarr store given as both its input and its output. All four commands now
write beside their output and rename it into place only once it is complete,
as `convert` already did. Also: `concat` no longer upgrades Zarr v2 inputs to
v3, and two help examples that did not run now do.

### Fixed

- **`subset x.zarr -o x.zarr` emptied the store**
  ([#35](https://github.com/cellgeni/adata-cli/issues/35)). The output was
  opened for writing, which clears a Zarr store, before the selection was
  read from the source. An output that names the input is now refused; use
  `--inplace`. The selection is now resolved before the output exists, so a
  query that fails no longer leaves an empty store behind, or truncates an
  existing one.

- **`--inplace` deleted the source before moving the new store in**
  ([#35](https://github.com/cellgeni/adata-cli/issues/35)). This affected
  `subset` and `recast`: an interrupt between the two steps lost the data.
  The source is now renamed aside and deleted only once its replacement is
  in place.

- **A failure part way through could leave a partial store that read as a
  finished one** ([#35](https://github.com/cellgeni/adata-cli/issues/35)).
  This affected `subset`, `recast`, `split` and `concat`. Each now writes to
  a temporary path beside its output (`out.adata-tmp.zarr`) and moves it into
  place only on success. A failure leaves an existing output as it was.
  `split` moves all its outputs together, writes its manifest last, and
  refuses an output that would land on its input.

  The helper that does this, shared with `convert`, has three fixes of its
  own:
  - It no longer deletes a leftover `*.adata-old.*` backup, which after an
    interrupted run may be the only copy of the data; it now refuses to start.
  - It puts back the outputs already replaced if a rename fails part way.
  - It keeps a Zarr store named without the `.zarr` suffix as Zarr.

- **`concat` wrote Zarr v3 whatever its inputs' version**
  ([#36](https://github.com/cellgeni/adata-cli/issues/36)), contrary to the
  documentation. It now follows the version its Zarr inputs share, and writes
  v3 when they differ. h5ad inputs do not count. `--zarr-format` still
  overrides it.

- **`subset` and `concat` now check `--zarr-format` themselves**
  ([#37](https://github.com/cellgeni/adata-cli/issues/37)), as every other
  command does, and say that it must be 2 or 3. Contributed by
  @iamcipherdev.

- **Six `--help` examples still said `h5ad`, and two of them did not run**
  ([#38](https://github.com/cellgeni/adata-cli/issues/38)). A new test checks
  every help example against the real commands and options. Contributed by
  @iamcipherdev.

## 0.7.0

Adds format conversion. `adata convert` now changes a store's format --
between h5ad and Zarr v2/v3, and to and from 10x Genomics HDF5 matrices --
and the matrix command that used to be `convert` is `recast`; its old
spelling keeps working, with a warning, until 1.0.0. Also fixes four ways a
store could fail or bloat on its way between backends, one of which left
`recast --layout` outputs three times the size they should be.

### Added

- **`adata convert` changes a store's format**
  ([#20](https://github.com/cellgeni/adata-cli/issues/20),
  [#22](https://github.com/cellgeni/adata-cli/issues/22)). This release
  converts between h5ad and Zarr v2/v3 in any direction; the output's
  extension picks the format. Every element is copied as it is, streamed,
  keeping chunking and, where the target can express it, compression;
  `--compression` chooses instead. A Zarr output keeps the source's version
  unless `--zarr-format` says otherwise.

  Everything is checked before anything is written, and the output is
  written beside its destination and moved into place only once complete,
  so a failed conversion leaves nothing behind. An existing output is
  refused unless `--overwrite` is given. A flag that does not apply to the
  conversion asked for is refused rather than ignored.

  Formats are registered in `adata.conversion`, each with the converters
  that read and write it; MTX folders, loom and others follow as separate
  issues under #20.

- **`convert` reads and writes 10x Genomics HDF5 matrices**
  ([#23](https://github.com/cellgeni/adata-cli/issues/23)): Cell Ranger's
  `*_feature_bc_matrix.h5`, v2 and v3+, and the same layout from Space
  Ranger, Xenium and CellBender. Reading gives what
  `scanpy.read_10x_h5(gex_only=False)` gives, column for column -- checked
  against scanpy itself in CI -- except that `X` keeps the integer counts
  and the run's metadata is kept in `uns["cellranger"]`. `--feature-type`
  keeps one type, `--split-feature-types` writes one output per type from a
  single pass over the matrix, `--genome` picks a genome, and `--var-names`
  and `--make-unique` do what scanpy's options of the same name do.

  Writing produces the Cell Ranger 3+ layout from X, `raw/X` or any layer
  (`--layer`), for CellBender, SoupX, DropletUtils and Seurat: gzip and
  fixed-width strings, which PyTables -- and so CellBender -- needs. A file
  imported by `convert` exports with its metadata intact, and the round
  trip reproduces Cell Ranger's file dataset for dataset.

### Changed

- **The matrix command `convert` is now `recast`**
  ([#21](https://github.com/cellgeni/adata-cli/issues/21)). It changes a
  matrix's dtype, layout or density exactly as `convert` did in 0.6; the
  name `convert` now means format conversion. The hint `concat` prints for
  mismatched encodings names `recast`.

### Deprecated

- **`convert` with `recast`'s flags.** Given any of `--dtype`,
  `--indices-dtype`, `--layout`, `--all` or `--inplace`, `convert` still
  runs `recast` as before, with a warning. **This is removed in 1.0.0**,
  with the `h5ad` command name and the `info` alias.

### Fixed

- **`recast --layout` left its h5ad output up to three times too large.**
  The streaming transpose kept its intermediate buckets in the output and
  deleted them when done, but HDF5 never returns a deleted dataset's space
  to the file. On a 20,000 x 2,000 matrix at 5% the output was 55 MB where
  the in-memory transpose wrote 17 MB. The buckets now go in a separate
  file beside the output, removed afterwards; Zarr outputs were unaffected.

- **Writing an h5ad with a `None` in it to Zarr failed.** anndata stores
  `None` as a null, which on HDF5 is an empty dataspace with no value to
  read, and copying one raised `TypeError: float() argument must be a string
  or a real number, not 'Empty'`. Any `None` in `uns` was enough, so
  `subset in.h5ad -o out.zarr` failed on such stores, and so did `split`,
  `import` and `recast` writing Zarr from them. Nulls are now written in the
  target's own spelling: an empty dataspace on HDF5, a 0-d bool on Zarr, as
  anndata does. The 0-d bool that a Zarr store uses for a missing `raw` now
  becomes an empty dataspace in an h5ad too, rather than a stray `False`.

- **Zarr written to h5ad came out uncompressed.** Zarr codecs have no HDF5
  equivalent, so they were dropped and nothing replaced them; a compressed
  Zarr store became an h5ad several times its size. A compressed array now
  becomes a gzip-compressed dataset (level 4, h5py's default). Strings and
  arrays that were not compressed are left as they were.

- **An empty array chunked larger than itself could not be written to
  h5ad.** Zarr allows a chunk bigger than its array and h5py does not, so a
  zero-length array copied from Zarr failed, and so did `recast --dtype` or
  `--layout` on an all-zero sparse matrix from a compressed h5ad. A
  zero-length dataset now lets HDF5 choose its own chunks.

## 0.6.1

Makes the outputs of `subset`, `split` and `concat` the size anndata writes
(they were about twice that), and makes `split` read each matrix once rather
than once per group. In 0.6.0 a hand-written anndata loop split the ci-tier
benchmark 7x faster than `split` did; `split` is now the faster of the two.

### Fixed

- **`subset`, `split` and `concat` wrote sparse matrices about twice as large
  as they needed to be.** Every sparse output widened `indices` and `indptr`
  to int64, whatever the source used, and was written without the source's
  compression, so with an lzf input the `data` and `indices` came out
  uncompressed. Outputs now keep the source's index dtypes and storage
  settings. `concat` widens only when the combined matrix could not
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
  [docs/TESTING.md](reference/testing.md#performance). Every subcommand is covered:
  `ls`, `view`, `create`, all five `export` and all five `import` variants,
  `split` on both axes, and the `concat` options nothing else reached.
  Two claims are now enforced rather than described -- `view` and `ls` read
  **zero** data elements at any store size, and streaming stays far below the
  input curve at a fixed `--chunk`.

- **A comparative benchmark** (`benchmarks/`), run on every tag against
  anndata and against scanpy where scanpy has a real equivalent. Reports peak
  RSS, wall time and output size; publishes to
  [docs/BENCHMARKS.md](reference/benchmarks.md) and the release notes. Report-only
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

# Changelog

Notable changes to `adata-cli`. Versions are `MAJOR.MINOR.PATCH`; tags carry no
`v` prefix.

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

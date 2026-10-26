# Status

Working notes on what is in progress and what comes next. Updated as work
moves; the plan of record for the current item is in the section below.

## Released

| Version | What | Issues |
|---|---|---|
| 0.16.1 | `subset`, `split` and the MuData export narrow categoricals (and their colours) as anndata's subsetting does | #74 |
| 0.16.0 | MuData `.h5mu` read (each modality, global parts on its cells) and written (split by feature type); copies of anndata 0.7 dataframes readable | #31, #72 |
| 0.15.0 | `convert --scope` writes SCope looms; `concat` stacks obsm dataframes and merges varm; loom attribute-name clashes resolved before writing | #61, #69, #70 |
| 0.14.0 | pycisTopic pickles (optional `pickle` extra) | #28 |
| 0.13.0 | Delimited text tables and plain HDF5 matrices | #25 |
| 0.12.0 | SCope loom import | #27 |

The conversion epic's sub-issues (#21–#30) are all done and ticked in #20.

## In progress

Nothing. The documentation site is live (#76); the spatial work below can
resume.

## Done since 0.16.1

### Documentation site (#76)

The docs restyled as cellgeni's pipeline docs (nf-cellbender,
nf-reprocessing-public-10x): Just the Docs from `docs/` on `main`, a short
README.

Decisions:
- Pages moved with `git mv` in a commit of their own, so history follows them;
  each old URL (`/COMMANDS.html` and five more) has a `redirect_from`, and a
  custom redirect layout keeps the `#anchor`, sending the old command
  reference's format anchors to their new pages.
- `reference/commands.md` keeps one section per command; each `convert`
  format has its own page under `reference/formats/`.
- The changelog lives in `docs/changelog.md`; the root `CHANGELOG.md` points
  there.
- The docs test resolves links as Jekyll does under `permalink: pretty`: a
  relative link must name a page's `.md`, and its anchor must exist. It also
  checks the front matter the sidebar needs.

Steps:
- [x] Theme config, Gemfile, pages moved with front matter and redirects
- [x] COMMANDS split into `reference/commands.md` and nine format pages
- [x] Home, six guides, Troubleshooting, Developers
- [x] Docs tests, benchmark generator (`docs/reference/benchmarks.md`), pointers
- [x] README rewritten
- [x] Local `github-pages` build in Docker; html-proofer clean
- [x] PR, review, merge; check the live site and the old URLs' redirects

## Follow-ups noticed

- Fixed, for the next release's changelog: `export image` without `-o`
  crashed (`'NoneType' object has no attribute 'parent'`); it now writes PNG
  to stdout, as every other export writes there. Found by the docs review.

- Fixed in 0.16.1 (#74): `subset`, `split` and the MuData export kept
  every category of a categorical column, where anndata's subsetting keeps
  only those used (and narrows `uns["<col>_colors"]`). The ~50% "slack"
  noted here before was mostly a measuring error -- `get_storage_size()`
  leaves out the variable-length string heap -- but the unpruned categories
  were real: the pbmc3k Multiome `.h5mu` drops from 84.7 to 74.2 MB, and a
  subset's obs and var now equal anndata's own subset.

## Next candidates (#31)

- Spatial: Visium / Xenium (matrix via 10x h5, coordinates to
  `obsm["spatial"]`)
- alevin-fry USA mode, kallisto | bustools
- Seurat `.h5Seurat`

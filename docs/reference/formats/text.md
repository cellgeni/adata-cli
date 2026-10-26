---
title: Text tables
parent: Formats
grand_parent: Reference
nav_order: 6
---

# Text tables

A dense matrix in text, one line per row: GEO count tables, R's
`write.table` and `write.csv` output, a pandas `to_csv`.

```bash
adata convert counts.csv -o counts.h5ad
adata convert GSE123_counts.txt.gz -o counts.h5ad --transpose
adata convert table.txt -o table.h5ad --delimiter ";" --dense
```

| Text table | AnnData |
|---|---|
| each line (each column with `--transpose`) | a cell: X's row, named by the first column |
| the header, or the last `#` comment line | var names |
| no header; no names in the first column | var and obs numbered from 0, as anndata numbers them |
| the delimiter and layout found | `uns["text"]` |

| Flag | Meaning |
|---|---|
| `--delimiter` | A character, `tab` or `whitespace` (runs of spaces and tabs). Default from the extension: `,` for `.csv`, tab for `.tsv` and `.tab`, whitespace for `.txt` and `.data`, as `read_text` splits |
| `--transpose` | The file holds genes as rows and cells as columns, as count tables often do |
| `--first-column-names`, `--no-first-column-names` | Whether the first column names the rows, when detection gets it wrong |
| `--dense` | Keep X dense (default: CSR) |
| `--make-unique` | Suffix repeated var names, as `var_names_make_unique` does |

The layout is found as anndata's `read_text` finds it, checked against it
in CI. The first line is a header when its last field is not a number, and
the first column names the rows when its first value is not a number. A
header with a field more than the values has a corner cell over the row
names; one with as many (R's `write.table`) names only the columns. Lines
starting with `#` before the data are comments, the last of them naming
the columns when there is no header. Names have `"` stripped, and blank
lines are skipped.

Values keep integers: when every value is whole, X is int32 (int64 if a
value needs it), as for [MTX folders](10x-mtx.md); otherwise it is float64, which holds
what the text holds where anndata gives float32. A line with a different
number of values, or a value that is not a number (`NA`), is refused with
its line and column, before any output exists.

The file is read twice, a block of lines at a time, each block parsed by
numpy, so memory is set by the block rather than by the file: once to find
the layout, check every line and pick the dtype, and once to write X. A
transposed file's entries are sorted into cells through on-disk buckets.
Gzip is found by content, whatever the name.

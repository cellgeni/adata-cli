---
title: Query expressions
parent: Reference
nav_order: 3
---

# Query expressions

Used by `subset --obs-query` and `--var-query`.

| Operator | Example |
|---|---|
| `==`, `!=` | `cluster == Cortex_2` |
| `<`, `<=`, `>`, `>=` | `n_counts > 1000` |
| `in`, `not in` | `cluster in A,B,C` |
| `and`, `or`, `not` | `n_counts > 500 and not cluster == Fiber_tract` |
| parentheses | `(a == 1 or b == 2) and c > 3` |

## Values

A value is a bare word or a quoted string. A bare word ends at a space, a
parenthesis, a comma or one of `=`, `!`, `<` and `>`, so `n_counts>1000` and
`cluster==A` need no spaces. A value that holds any of those characters is
quoted, in single or double quotes, and a quoted value is taken as written,
commas included. A column name is quoted the same way.

An `in` list is either in parentheses, its values separated by commas or
spaces, or bare words separated by commas:

```bash
adata subset data.h5ad -o t.h5ad -q "cluster in A,B,C"
adata subset data.h5ad -o t.h5ad -q 'cell_type in ("CD4-positive, alpha-beta T cell", "B cell")'
```

`and`, `or`, `not` and `in` may be written in any case. Column names are
case-sensitive.

## How values compare

A column is compared at the type it is stored as:

- **Numbers** compare by value, at the column's dtype, as pandas compares
  them. `total_counts != 0` drops the zeros of a float column, `n == 1`
  matches `1.0`, and in a float32 column (scanpy's default for QC metrics)
  `pct_mt <= 0.1` keeps the values stored as 0.1. A value that is not a
  number is refused rather than matching nothing; Python's `1_000` is not a
  number here.
- **Booleans** compare with `true` or `false`, in any case, or `1` and `0`.
- **Text**, in a string column or in categories, compares by its string,
  exactly and case included. The ordering operators read text entries as
  numbers, and an entry that will not parse does not match.
- A **categorical** compares as its categories do, so integer categories
  compare as numbers.

A missing number or boolean (NaN, or masked in a nullable column) equals no
value: `==` and `in` never match it, while `!=` and `not in` keep it, as
pandas does with NaN. To select it, compare with `nan`: `score == nan`
keeps the rows with no score, and `score != nan` the rows that have one. A
missing text or category entry reads as the empty string, so
`cell_type == ""` selects it, along with any entry that really is empty.

Only the columns a query mentions are read, so filtering on one annotation does
not touch the rest of the frame.

This is deliberately not SQL. For anything more involved — joins, aggregates,
window functions — export to CSV and use [duckdb](https://duckdb.org):

```bash
adata export dataframe data.h5ad obs -o cells.csv
duckdb -noheader -list -c \
  "SELECT _index FROM 'cells.csv' WHERE cluster='Cortex_2' AND n_counts > 1000" \
  > barcodes.txt
adata subset data.h5ad -o cortex.h5ad --obs barcodes.txt
```

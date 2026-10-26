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

Values with spaces need quoting: `label == "cell type A"`. Equality and
membership compare the column's string form; the ordering operators parse it as
a number, and entries that will not parse simply do not match.

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

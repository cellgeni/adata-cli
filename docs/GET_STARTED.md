# Get Started

This short walkthrough shows the basic workflow: inspect a store, export metadata, and write a subset.

## 1 Install

Using uv (recommended):
```bash
git clone https://github.com/cellgeni/adata-cli.git
cd adata-cli
uv sync
```

With pip:
```bash
git clone https://github.com/cellgeni/adata-cli.git
cd adata-cli
pip install .
```

Most of this tutorial needs nothing else. Step 4 shows an optional
[duckdb](https://duckdb.org) workflow for filters too involved to express with
`--obs-query`; install it only if you want to follow that part:

```bash
# macOS
brew install duckdb

# or download a single static binary for any platform
# https://duckdb.org/docs/installation/
```

## 2 Inspect a store with `view`

Let's load an example `.h5ad` file:
```bash
wget -O visium.h5ad https://exampledata.scverse.org/squidpy/figshare/visium_hne_adata.h5ad
```

Now run `info` to see the file structure:
```bash
adata view visium.h5ad
```
```
An object with n_obs × n_var: 2688 × 18078
        obs:    array_col, array_row, cluster, in_tissue, leiden, log1p_n_genes_by_counts, log1p_total_counts, log1p_total_counts_mt, n_counts, n_genes_by_counts, pct_counts_in_top_100_genes, pct_counts_in_top_200_genes, pct_counts_in_top_500_genes, 
pct_counts_in_top_50_genes, pct_counts_mt, total_counts, total_counts_mt
        var:    feature_types, gene_ids, genome, highly_variable, highly_variable_rank, log1p_mean_counts, log1p_total_counts, mean_counts, means, mt, n_cells, n_cells_by_counts, pct_dropout_by_counts, total_counts, variances, variances_norm
        obsm:   X_pca, X_umap, spatial
        varm:   PCs
        obsp:   connectivities, distances
        uns:    cluster_colors, hvg, leiden, leiden_colors, neighbors, pca, rank_genes_groups, spatial, umap
        raw:    X, var
```

To inspect a specific entry:
```bash
adata view visium.h5ad obsm/X_pca
```
```
Path: obsm/X_pca
Type: dense-matrix
Shape: (2688, 50)
Dtype: float32
Details: Dense matrix 2688×50 (float32)
```

## 3 Export entries
View the first few lines of the `obs` dataframe:

```bash
adata export dataframe visium.h5ad obs --head 10
```
```csv
_index,array_col,array_row,cluster,in_tissue,leiden,log1p_n_genes_by_counts,log1p_total_counts,log1p_total_counts_mt,n_counts,n_genes_by_counts,pct_counts_in_top_100_genes,pct_counts_in_top_200_genes,pct_counts_in_top_500_genes,pct_counts_in_top_50_genes,pct_counts_mt,total_counts,total_counts_mt
AAACAAGTATCTCCCA-1,102,50,Cortex_2,1,Cortex_3,8.502891406705377,9.869983,8.257904,19340.0,4928,43.13340227507756,49.21406411582213,60.449844881075485,38.42812823164426,19.943123,19340.0,3857.0
AAACAATCTACTAGCA-1,43,3,Cortex_5,1,Pyramidal_layer_dentate_gyrus,8.145839612936841,9.528867,8.091933,13750.0,3448,55.14181818181818,60.95272727272727,70.57454545454546,50.516363636363636,23.76,13750.0,3267.0
AAACACCAATAACTGC-1,19,59,Thalamus_2,1,Hypothalamus_1,8.70334075304372,10.395467,8.499233,32710.0,6022,47.071232039131765,54.56435340874351,65.0871293182513,40.48303271170896,15.010699,32710.0,4910.0
AAACAGAGCGACTCCT-1,94,14,Cortex_5,1,Pyramidal_layer_dentate_gyrus,8.369157112588834,9.674704,8.092851,15909.0,4311,45.81054748884279,52.07744044251681,62.97693129675027,40.95794833113332,20.554403,15909.0,3270.0
AAACCGGGTAGGTACC-1,28,42,Thalamus_2,1,Hypothalamus_1,8.663542087751374,10.369013,8.808967,31856.0,5787,45.887744851833254,52.98216976393771,64.24849321948768,40.287543947764945,21.01017,31856.0,6693.0
AAACCGTTCGTCCAGG-1,42,52,Hypothalamus_2,1,Pyramidal_layer,8.682538124003075,10.337314,8.559678,30862.0,5898,43.79171797031949,51.18592443781998,62.65634113148856,37.80053139783553,16.901043,30862.0,5216.0
AAACCTCATGAAGTTG-1,19,37,Thalamus_2,1,Hypothalamus_1,9.027858802380862,11.007419,8.849371,60319.0,8331,34.28770370861586,42.45594257199224,55.48997828213332,27.803842901904872,11.553574,60319.0,6969.0
AAACGAAGAACATACC-1,64,6,Cortex_4,1,Hypothalamus_2,8.84246002419529,10.578089,8.855521,39264.0,6921,37.99663814180929,44.75346373268134,56.6320293398533,32.95639771801141,17.858597,39264.0,7012.0
AAACGAGACGGTTGAT-1,79,35,Fiber_tract,1,Cortex_5,8.80941494391005,10.458923,8.351847,34853.0,6696,39.947780678851174,47.52818982583996,58.838550483459095,33.7245000430379,12.156773,34853.0,4237.0
AAACGGTTGCGAACTG-1,59,67,Lateral_ventricle,1,Striatum,8.718663567048953,10.254004,8.416489,28395.0,6115,41.67635147032928,49.20232435287903,60.556435992252155,35.562599049128366,15.918295,28395.0,4520.0
```

Export cell metadata to a CSV file:
```bash
adata export dataframe visium.h5ad obs --output cells.csv
wc -l cells.csv # 2689 cells.csv
```

## 4 Subset by names

Let's get all cluster names from `cells.csv`:
```bash
awk -F ',' 'NR>1{print $4}' cells.csv | sort | uniq -c
```
```
284 Cortex_1
257 Cortex_2
244 Cortex_3
164 Cortex_4
129 Cortex_5
226 Fiber_tract
222 Hippocampus
208 Hypothalamus_1
133 Hypothalamus_2
105 Lateral_ventricle
42 Pyramidal_layer
68 Pyramidal_layer_dentate_gyrus
153 Striatum
261 Thalamus_1
192 Thalamus_2
```

### Filtering directly

For a filter this simple, `--obs-query` does the whole job without an
intermediate file:

```bash
adata subset visium.h5ad --output cortex2.h5ad --obs-query "cluster == Cortex_2"
```

The expression language covers `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`,
`not in`, `and`, `or`, `not` and parentheses, so most real filters fit:

```bash
adata subset visium.h5ad -o big_cortex.h5ad \
    -q "cluster in Cortex_1,Cortex_2 and total_counts > 20000"
```

Only the columns the query mentions are read, so this does not touch the rest
of the frame.

### Filtering with duckdb

When a filter needs more than that — a join against another table, an
aggregate, a window function — export `obs` and let
[duckdb](https://duckdb.org) do the query, then feed the resulting names back
in:

```bash
duckdb -noheader -list -c \
  "SELECT _index FROM 'cells.csv' WHERE cluster='Cortex_2'" > barcodes.txt
wc -l barcodes.txt  # 257 barcodes.txt
```

duckdb reads the CSV in place — no import step — and `-noheader -list` gives
one bare name per line, which is exactly the format `--obs` expects:

```bash
adata subset visium.h5ad --output cortex2.h5ad --obs barcodes.txt
```

It is also a quicker way to do the cluster tally above:

```bash
duckdb -c "SELECT cluster, count(*) FROM 'cells.csv' GROUP BY 1 ORDER BY 2 DESC"
```

Check the result:
```bash
adata view cortex2.h5ad
```
```
An object with n_obs × n_var: 257 × 18078
        obs:    array_col, array_row, cluster, in_tissue, leiden, log1p_n_genes_by_counts, log1p_total_counts, log1p_total_counts_mt, n_counts, n_genes_by_counts, 
pct_counts_in_top_100_genes, pct_counts_in_top_200_genes, pct_counts_in_top_500_genes, pct_counts_in_top_50_genes, pct_counts_mt, total_counts, total_counts_mt
        var:    feature_types, gene_ids, genome, highly_variable, highly_variable_rank, log1p_mean_counts, log1p_total_counts, mean_counts, means, mt, n_cells, n_cells_by_counts, 
pct_dropout_by_counts, total_counts, variances, variances_norm
        obsm:   X_pca, X_umap, spatial
        varm:   PCs
        obsp:   connectivities, distances
        uns:    cluster_colors, hvg, leiden, leiden_colors, neighbors, pca, rank_genes_groups, spatial, umap
```

## 5 Import or replace data
You can also import new data into an existing store. For example, let's replace the `obs` dataframe with a modified version. First, leave only first 5 columns in `cells.csv`:
```bash
cut -d ',' -f 1-5 cells.csv > cells1to5.csv
```

Now import it back into `cortex2.h5ad` with the `_index` column as index:
```bash
adata import dataframe visium.h5ad obs cells1to5.csv --index-column _index --output visium_obs1to5.h5ad
```

Check the updated `obs` structure:
```bash
adata view visium_obs1to5.h5ad
```
```
An object with n_obs × n_var: 2688 × 18078
        obs:    array_col, array_row, cluster, in_tissue
        var:    feature_types, gene_ids, genome, highly_variable, highly_variable_rank, log1p_mean_counts, log1p_total_counts, mean_counts, means, mt, n_cells, n_cells_by_counts, 
pct_dropout_by_counts, total_counts, variances, variances_norm
        obsm:   X_pca, X_umap, spatial
        varm:   PCs
        obsp:   connectivities, distances
        uns:    cluster_colors, hvg, leiden, leiden_colors, neighbors, pca, rank_genes_groups, spatial, umap
        raw:    X, var
```

You can also import the data into existing file:
```bash
adata import dataframe visium.h5ad obs cells1to5.csv --index-column _index --inplace
```

Check the updated `obs` structure:
```bash
adata view visium.h5ad
```
```
An object with n_obs × n_var: 2688 × 18078
        obs:    array_col, array_row, cluster, in_tissue
        var:    feature_types, gene_ids, genome, highly_variable, highly_variable_rank, log1p_mean_counts, log1p_total_counts, mean_counts, means, mt, n_cells, n_cells_by_counts, 
pct_dropout_by_counts, total_counts, variances, variances_norm
        obsm:   X_pca, X_umap, spatial
        varm:   PCs
        obsp:   connectivities, distances
        uns:    cluster_colors, hvg, leiden, leiden_colors, neighbors, pca, rank_genes_groups, spatial, umap
        raw:    X, var
```

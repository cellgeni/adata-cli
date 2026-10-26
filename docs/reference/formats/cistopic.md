---
title: pycisTopic pickles
parent: Formats
grand_parent: Reference
nav_order: 10
---

# pycisTopic pickles

pycisTopic, the scATAC half of SCENIC+, saves its objects only as pickles:
`cistopic_obj.pkl`, and the imputed accessibility and gene activity as
`CistopicImputedFeatures`. They convert without pycisTopic, which is not on
PyPI and pins an old pandas. pandas and scipy, which rebuild the pickle's
tables and matrices, come with an extra: `pip install 'pyadata-cli[pickle]'`.

```bash
adata convert cistopic_obj.pkl -o atac.h5ad
adata convert imputed_acc_obj.pkl -o imputed.h5ad --dense
```

| CistopicObject | AnnData |
|---|---|
| `fragment_matrix` (regions x cells) | `X`, CSR cells x regions, its dtype kept |
| `binary_matrix` | `layers["binary"]` |
| `cell_names`, `cell_data` | obs names, obs (in `cell_names` order) |
| `region_names`, `region_data` | var names, var |
| the selected model's `cell_topic` (`cell_topic_harmony`) | `obsm["X_topic"]` (`obsm["X_topic_harmony"]`), cells x topics |
| the model's `region_topic` | `varm["topic_region"]`, regions x topics |
| `projections["cell"]` and `["region"]`, such as UMAP | `obsm["X_<name>"]`, `varm["X_<name>"]` |
| `project`, `path_to_fragments`, the topic names, the model's metrics, coherence, parameters and other tables | `uns["pycistopic"]` |

A `CistopicImputedFeatures` becomes X, cells x features, CSR (`--dense`
keeps it dense), with `feature_names` and `cell_names` as var and obs names.

The pickle is read without running anything from it: an unpickler that
resolves only the names a pycisTopic object holds (numpy arrays, scipy
sparse matrices, pandas frames, a few builtins) and refuses any other by
name, so a pickle naming `os.system` is refused rather than run.
pycisTopic's own classes are stood in for by plain holders of their
attributes. A pickle written by pycisTopic's pandas 1.5 reads with pandas 2
or 3, as CI checks. Columns are written as anndata writes them: Categorical
as categorical, nullable integers and booleans as nullable, numbers held as
`object` as numbers, repeated text as categorical. Anything else on the
object is named in a warning and left out.

A pickle cannot be read in part, so it is held in memory whole, as it was
when pycisTopic saved it; the fragment matrix is turned into CSR on disk by
the streaming transpose rather than copied in memory.

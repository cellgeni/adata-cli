"""What gets compared, and the rules that keep the comparison honest.

The rules, because this is the part that decays fastest
-------------------------------------------------------

**Use the best idiom the baseline has, not the naive one.** Comparing
`adata export dataframe obs` against a full `ad.read_h5ad()` is a strawman:
anndata reads just `obs` cheaply via `read_elem` on the h5py group. The good
idiom is the primary baseline. A naive full load may appear only as a clearly
labelled second row, where the point is the gap between the two.

**Pin compression on both sides.** adata-cli forwards the source's settings to
its output; `write_h5ad` defaults to none. Left alone, adata-cli's output
looks smaller for reasons unrelated to the tool.

**Give the baseline everything it needs.** `concat_on_disk` imports `dask`
to handle a dense element and fails outright without it, so the baseline
environments install it. Measuring a library crippled by a missing optional
dependency would be measuring our own setup.

**`n/a` is a result.** `concat_on_disk` is CSR/CSC-oriented and its outer-join
support has varied by version; scanpy has no streaming concat at all. Print
what refused and why. An omitted row reads as an oversight; a stated `n/a`
reads as a finding.

**Startup is a floor, not noise.** The CLI takes 0.3-1 s to import typer, rich,
h5py and zarr, and `import scanpy` takes 3-8 s. On a small tier that is the
entire measurement, so every run includes a `--version` row to read the rest
against.

**Do not hide the rows where the baseline wins.** `_concat_csr` loops per row
in Python; scipy's C `vstack` will very likely be several times faster in wall
time at many times the memory. That trade *is* the argument for this tool.
The table is "peak RSS against wall time", not a leaderboard.

**Not every command has a baseline, and four are left out on purpose.**
`export image`, `export dict`, `import image` and `import dict` have no
library equivalent, so their rows would only ever read `n/a` and they would
add runtime to every tag for nothing. They are covered by the complexity
guards in `tests/test_performance.py` instead.

**scanpy's filter functions are not our subset.** `sc.pp.filter_cells`
*computes* `n_genes` by scanning X; `adata subset --obs-query` filters a
column that must already exist. Head to head, scanpy looks slow for doing
strictly more work. The like-for-like baseline is a plain boolean mask over a
precomputed column, and scanpy appears in a separately labelled row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Written into every generated fixture, and forced on every baseline write.
COMPRESSION = "lzf"


@dataclass(frozen=True)
class Contender:
    """One way of doing the case: a command template, or a script."""

    label: str
    #: Python source run by the baseline interpreter. `None` for the CLI.
    script: Optional[str] = None
    #: argv for the CLI, templated with {input0}, {input1}, {output}.
    argv: Optional[List[str]] = None
    #: Set when the operation has no equivalent, explaining what is missing.
    unsupported: Optional[str] = None
    #: Which prebuilt environment runs `script`.
    env: str = "anndata"


@dataclass(frozen=True)
class Case:
    """One operation, and every way of performing it."""

    name: str
    question: str
    contenders: List[Contender]
    #: How many inputs the case needs.
    inputs: int = 1
    #: Extension of the output the case writes ("" for read-only cases).
    output_suffix: str = ".h5ad"
    #: Use the var-heavy shape rather than the tier's main shape.
    var_heavy: bool = False
    #: Use a wide variant of the main input -- float64 values, int64 indices
    #: -- named for how it stores X: "uncompressed" or "lzf".
    wide: Optional[str] = None
    #: Report the output's size against the input's, as before -> after. For
    #: cases whose point is what they do to the file's size.
    compare_size: bool = False
    #: An auxiliary input the case needs: "csv", the CSV an import reads,
    #: "10x-mtx", an MTX folder of the first input's counts, or "dense-csv",
    #: a dense count table of its first cells. Built once from
    #: the first input and offered to CLI contenders as `{sidecar}` and to
    #: scripts as $BENCH_SIDECAR (`{sidecar_csv}` and $BENCH_SIDECAR_CSV too).
    sidecar: Optional[str] = None
    tags: List[str] = field(default_factory=list)


_PRELUDE = f"""
import sys, warnings
warnings.filterwarnings("ignore")
import anndata as ad, numpy as np, h5py
IN = sys.argv[1:-1]
OUT = sys.argv[-1]
COMPRESSION = {COMPRESSION!r}
"""


def _py(body: str) -> str:
    return _PRELUDE + body


CASES: List[Case] = [
    # -- startup floor ----------------------------------------------------
    Case(
        name="startup",
        question="What does each contender cost before it does any work?",
        output_suffix="",
        contenders=[
            Contender("adata-cli", argv=["adata", "--version"]),
            Contender("anndata", script="import anndata"),
            Contender("scanpy", script="import scanpy", env="scanpy"),
        ],
        tags=["floor"],
    ),
    # -- concat -----------------------------------------------------------
    Case(
        name="concat-inner",
        question="Concatenate two stores on the obs axis, inner join.",
        inputs=2,
        contenders=[
            Contender(
                "adata-cli",
                argv=["adata", "concat", "{input0}", "{input1}", "-o", "{output}"],
            ),
            Contender(
                "anndata (concat_on_disk)",
                script=_py(
                    "from anndata.experimental import concat_on_disk\n"
                    "concat_on_disk(IN, OUT, join='inner')\n"
                ),
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "parts = [ad.read_h5ad(p) for p in IN]\n"
                    "ad.concat(parts, join='inner')"
                    ".write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
            Contender(
                "scanpy",
                unsupported="scanpy has no concat of its own; it re-exports anndata's",
            ),
        ],
    ),
    Case(
        name="concat-merge-same",
        question=(
            "Concatenate carrying var columns forward -- the REQ-71798 case."
        ),
        inputs=2,
        var_heavy=True,
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "concat", "{input0}", "{input1}",
                    "-o", "{output}", "--merge", "same",
                ],
            ),
            Contender(
                "anndata (concat_on_disk)",
                script=_py(
                    "from anndata.experimental import concat_on_disk\n"
                    "concat_on_disk(IN, OUT, join='inner', merge='same')\n"
                ),
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "parts = [ad.read_h5ad(p) for p in IN]\n"
                    "ad.concat(parts, join='inner', merge='same')"
                    ".write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
        tags=["regression"],
    ),
    # -- subset -----------------------------------------------------------
    Case(
        name="subset-query",
        question="Keep the obs rows matching a predicate on an existing column.",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "subset", "{input0}", "-o", "{output}",
                    "-q", "quality < 50",
                ],
            ),
            Contender(
                "anndata (backed)",
                script=_py(
                    "obj = ad.read_h5ad(IN[0], backed='r')\n"
                    "keep = obj.obs['quality'] < 50\n"
                    "obj[keep].to_memory().write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "obj = ad.read_h5ad(IN[0])\n"
                    "obj[obj.obs['quality'] < 50]"
                    ".write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
            Contender(
                "scanpy (filter_cells)",
                env="scanpy",
                script=_py(
                    "import scanpy as sc\n"
                    "# NOT like-for-like: filter_cells computes n_genes by\n"
                    "# scanning X, which the rows above take as given. Kept\n"
                    "# for scale, labelled so nobody reads it as a race.\n"
                    "obj = ad.read_h5ad(IN[0])\n"
                    "sc.pp.filter_cells(obj, min_genes=1)\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    # -- conversion -------------------------------------------------------
    Case(
        name="h5ad-to-zarr",
        question="Convert a store to Zarr.",
        output_suffix=".zarr",
        contenders=[
            Contender(
                "adata-cli",
                argv=["adata", "convert", "{input0}", "-o", "{output}"],
            ),
            Contender(
                "anndata (in memory)",
                script=_py("ad.read_h5ad(IN[0]).write_zarr(OUT)\n"),
            ),
            Contender(
                "scanpy",
                unsupported="no streaming converter; scanpy defers to anndata",
            ),
        ],
    ),
    # -- metadata ---------------------------------------------------------
    Case(
        name="inspect",
        question="Report what is in the store, without reading the matrix.",
        output_suffix="",
        contenders=[
            Contender("adata-cli", argv=["adata", "view", "{input0}"]),
            Contender(
                "anndata (read_elem on obs)",
                script=_py(
                    "from anndata.io import read_elem\n"
                    "with h5py.File(IN[0], 'r') as f:\n"
                    "    obs = read_elem(f['obs']); var = read_elem(f['var'])\n"
                    "print(obs.shape, var.shape)\n"
                ),
            ),
            Contender(
                "anndata (full load)",
                script=_py("print(ad.read_h5ad(IN[0]))\n"),
            ),
        ],
        tags=["headline"],
    ),
    Case(
        name="export-obs",
        question="Write the obs table out as CSV.",
        output_suffix=".csv",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "export", "dataframe", "{input0}", "obs",
                    "-o", "{output}",
                ],
            ),
            Contender(
                "anndata (read_elem on obs)",
                script=_py(
                    "from anndata.io import read_elem\n"
                    "with h5py.File(IN[0], 'r') as f:\n"
                    "    read_elem(f['obs']).to_csv(OUT)\n"
                ),
            ),
            Contender(
                "anndata (full load)",
                script=_py("ad.read_h5ad(IN[0]).obs.to_csv(OUT)\n"),
            ),
        ],
    ),
    # -- split ------------------------------------------------------------
    Case(
        name="split-by-sample",
        question="Write one store per distinct value of an obs column.",
        output_suffix="",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "split", "{input0}", "--by", "sample",
                    "-o", "{outdir}",
                ],
            ),
            Contender(
                "anndata (hand-written loop)",
                script=_py(
                    "import os\n"
                    "obj = ad.read_h5ad(IN[0])\n"
                    "os.makedirs(OUT, exist_ok=True)\n"
                    "for key, idx in obj.obs.groupby('sample', observed=True)"
                    ".groups.items():\n"
                    "    obj[idx].write_h5ad(\n"
                    "        os.path.join(OUT, f'{key}.h5ad'), compression=COMPRESSION\n"
                    "    )\n"
                ),
            ),
        ],
    ),
    # -- structure and metadata -------------------------------------------
    Case(
        name="ls",
        question="List everything in the store.",
        output_suffix="",
        contenders=[
            Contender("adata-cli", argv=["adata", "ls", "{input0}", "--plain"]),
            Contender(
                "h5py (visit)",
                script=_py(
                    "names = []\n"
                    "with h5py.File(IN[0], 'r') as f:\n"
                    "    f.visit(names.append)\n"
                    "print(len(names))\n"
                ),
            ),
            # The obvious tool someone already has. If the CLI cannot beat a
            # 20-year-old C program at walking an HDF5 file, that is worth
            # knowing and worth printing.
            Contender("h5ls -r", argv=["h5ls", "-r", "{input0}"]),
        ],
        tags=["headline"],
    ),
    Case(
        name="create",
        question="Write an empty store with a given obs/var shape.",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "create", "{output}",
                    "--n-obs", "{n_obs}", "--n-var", "{n_var}",
                ],
            ),
            Contender(
                "anndata",
                script=_py(
                    "import os, pandas as pd\n"
                    "n_obs = int(os.environ['BENCH_N_OBS'])\n"
                    "n_var = int(os.environ['BENCH_N_VAR'])\n"
                    "obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_obs)])\n"
                    "var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_var)])\n"
                    "ad.AnnData(\n"
                    "    X=np.zeros((n_obs, n_var), dtype='float32'), obs=obs, var=var\n"
                    ").write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    # -- export -------------------------------------------------------------
    Case(
        name="export-array",
        question="Write a dense element out as .npy.",
        output_suffix=".npy",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "export", "array", "{input0}", "obsm/X_pca",
                    "-o", "{output}",
                ],
            ),
            Contender(
                "anndata (read_elem)",
                script=_py(
                    "from anndata.io import read_elem\n"
                    "with h5py.File(IN[0], 'r') as f:\n"
                    "    np.save(OUT, read_elem(f['obsm/X_pca']))\n"
                ),
            ),
            Contender(
                "anndata (full load)",
                script=_py("np.save(OUT, ad.read_h5ad(IN[0]).obsm['X_pca'])\n"),
            ),
        ],
    ),
    Case(
        name="export-sparse",
        question="Write X out as Matrix Market text.",
        output_suffix=".mtx",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "export", "sparse", "{input0}", "X", "-o", "{output}",
                ],
            ),
            Contender(
                "anndata (sparse_dataset)",
                script=_py(
                    "import scipy.io as sio\n"
                    "from anndata.abc import CSRDataset\n"
                    "from anndata.io import sparse_dataset\n"
                    "with h5py.File(IN[0], 'r') as f:\n"
                    "    sio.mmwrite(OUT, sparse_dataset(f['X'])[...])\n"
                ),
            ),
            Contender(
                "anndata (full load)",
                script=_py(
                    "import scipy.io as sio\n"
                    "sio.mmwrite(OUT, ad.read_h5ad(IN[0]).X)\n"
                ),
            ),
        ],
    ),
    # -- import -------------------------------------------------------------
    Case(
        name="import-dataframe",
        question="Replace obs from a CSV.",
        sidecar="csv",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "import", "dataframe", "{input0}", "obs",
                    "{sidecar_csv}", "-o", "{output}",
                ],
            ),
            Contender(
                "anndata",
                script=_py(
                    "import pandas as pd, os\n"
                    "csv = os.environ['BENCH_SIDECAR_CSV']\n"
                    "obj = ad.read_h5ad(IN[0])\n"
                    "obj.obs = pd.read_csv(csv, index_col=0)\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    Case(
        name="import-10x-mtx",
        question=(
            "Read a gzipped Cell Ranger 3 MTX folder of the same counts into "
            "an h5ad. adata-cli streams it in two passes and keeps integer "
            "counts; scanpy loads it and converts them to float32."
        ),
        sidecar="10x-mtx",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "convert", "{sidecar}", "-o", "{output}",
                    "--compression", COMPRESSION,
                ],
            ),
            Contender(
                "scanpy",
                env="scanpy",
                script=_py(
                    "import os, scanpy as sc\n"
                    "obj = sc.read_10x_mtx(os.environ['BENCH_SIDECAR'], "
                    "gex_only=False, make_unique=False)\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
        tags=["headline"],
    ),
    Case(
        name="import-xenium",
        question=(
            "Read a Xenium bundle of the same counts -- the cell matrix, a "
            "cells.csv.gz of centroids and per-cell counts, and analysis "
            "clusters -- into an h5ad with obsm['spatial']. adata-cli matches "
            "the cell table to the matrix as it scans it and streams both; "
            "10x's scanpy recipe loads the matrix and reads the tables with "
            "pandas. adata-cli keeps the matrix's own gzip, scanpy writes lzf."
        ),
        sidecar="xenium",
        contenders=[
            Contender(
                "adata-cli",
                argv=["adata", "convert", "{sidecar}", "-o", "{output}"],
            ),
            Contender(
                "scanpy",
                env="scanpy",
                script=_py(
                    "import os, pandas as pd, scanpy as sc\n"
                    "from pathlib import Path\n"
                    "bundle = Path(os.environ['BENCH_SIDECAR'])\n"
                    "obj = sc.read_10x_h5(bundle / 'cell_feature_matrix.h5', gex_only=False)\n"
                    "cells = pd.read_csv(bundle / 'cells.csv.gz', index_col='cell_id')\n"
                    "cells = cells.reindex(obj.obs_names)\n"
                    "obj.obsm['spatial'] = cells[['x_centroid', 'y_centroid']].to_numpy()\n"
                    "obj.obs = cells.drop(columns=['x_centroid', 'y_centroid'])\n"
                    "clusters = pd.read_csv(bundle / 'analysis/clustering/"
                    "gene_expression_graphclust/clusters.csv', index_col='Barcode')\n"
                    "obj.obs['graphclust'] = clusters['Cluster'].astype(str)"
                    ".reindex(obj.obs_names).astype('category')\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    Case(
        name="import-csv",
        question=(
            "Read a dense count table, a CSV of the first 10,000 cells, into an "
            "h5ad. adata-cli reads it twice, a block of lines at a time, and "
            "writes CSR int32; anndata's read_csv loads it as dense float32."
        ),
        sidecar="dense-csv",
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "convert", "{sidecar}", "-o", "{output}",
                    "--compression", COMPRESSION,
                ],
            ),
            Contender(
                "anndata",
                script=_py(
                    "import os\n"
                    "obj = ad.io.read_csv(os.environ['BENCH_SIDECAR'])\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    # -- concat, outer ------------------------------------------------------
    Case(
        name="concat-outer",
        question="Concatenate two stores keeping the union of variables.",
        inputs=2,
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "concat", "{input0}", "{input1}",
                    "-o", "{output}", "--join", "outer",
                ],
            ),
            Contender(
                "anndata (concat_on_disk)",
                script=_py(
                    "from anndata.experimental import concat_on_disk\n"
                    "concat_on_disk(IN, OUT, join='outer')\n"
                ),
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "parts = [ad.read_h5ad(p) for p in IN]\n"
                    "ad.concat(parts, join='outer')"
                    ".write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    # -- recast (named convert until 0.7.0; the case names keep the history) --
    #
    # `convert-dtype` ran until 0.7.0 on the main fixture, which is already
    # float32 with int32 indices: it measured a rewrite that changed nothing,
    # and its output was the size of its input. `recast-narrow` replaces it,
    # on an input worth narrowing, and reports the size it saves.
    Case(
        name="recast-narrow",
        question=(
            "Narrow X from float64 values with int64 indices to float32 and "
            "int32, as issue #13 asked. X is uncompressed, as `write_h5ad` "
            "leaves it by default, so the output keeps that and the baseline "
            "writes it the same way."
        ),
        wide="uncompressed",
        compare_size=True,
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "recast", "{input0}", "X", "-o", "{output}",
                    "--dtype", "float32", "--indices-dtype", "int32",
                ],
            ),
            Contender(
                "anndata (in memory)",
                # compression=None, not COMPRESSION: the input's X is
                # uncompressed and adata-cli keeps it that way, so lzf here
                # would credit anndata with a saving the cast did not make.
                script=_py(
                    "obj = ad.read_h5ad(IN[0])\n"
                    "X = obj.X.astype('float32')\n"
                    "X.indices = X.indices.astype('int32')\n"
                    "X.indptr = X.indptr.astype('int32')\n"
                    "obj.X = X\n"
                    "obj.write_h5ad(OUT, compression=None)\n"
                ),
            ),
            Contender(
                "scanpy",
                unsupported="no dtype rewrite; scanpy defers to anndata",
            ),
        ],
        tags=["headline"],
    ),
    Case(
        name="recast-narrow-lzf",
        question=(
            "The same narrowing on an lzf-compressed X. Integer counts in "
            "float64 are mostly zero bytes, which lzf already removes, so "
            "expect a small saving on disk; the halving is in memory, once "
            "the matrix is loaded."
        ),
        wide="lzf",
        compare_size=True,
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "recast", "{input0}", "X", "-o", "{output}",
                    "--dtype", "float32", "--indices-dtype", "int32",
                ],
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "obj = ad.read_h5ad(IN[0])\n"
                    "X = obj.X.astype('float32')\n"
                    "X.indices = X.indices.astype('int32')\n"
                    "X.indptr = X.indptr.astype('int32')\n"
                    "obj.X = X\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
    ),
    Case(
        name="convert-layout",
        question=(
            "Transpose X from CSR to CSC. The output should be the size of "
            "the input: 0.7.0 fixed a `--layout` that wrote it three times "
            "larger."
        ),
        compare_size=True,
        contenders=[
            Contender(
                "adata-cli (streaming)",
                argv=[
                    "adata", "recast", "{input0}", "X", "-o", "{output}",
                    "--layout", "csc",
                ],
            ),
            Contender(
                "adata-cli (--in-memory)",
                argv=[
                    "adata", "recast", "{input0}", "X", "-o", "{output}",
                    "--layout", "csc", "--in-memory",
                ],
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "obj = ad.read_h5ad(IN[0])\n"
                    "obj.X = obj.X.tocsc()\n"
                    "obj.write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
        tags=["headline"],
    ),
    # -- the claim itself -------------------------------------------------
    Case(
        name="rss-vs-size",
        question=(
            "Does peak memory track input size, at a fixed --chunk? "
            "This is the README's claim, stated as a measurement."
        ),
        inputs=2,
        contenders=[
            Contender(
                "adata-cli",
                argv=[
                    "adata", "concat", "{input0}", "{input1}",
                    "-o", "{output}", "--chunk", "1024",
                ],
            ),
            Contender(
                "anndata (in memory)",
                script=_py(
                    "parts = [ad.read_h5ad(p) for p in IN]\n"
                    "ad.concat(parts).write_h5ad(OUT, compression=COMPRESSION)\n"
                ),
            ),
        ],
        tags=["headline", "sweep"],
    ),
]


def by_name() -> Dict[str, Case]:
    return {case.name: case for case in CASES}

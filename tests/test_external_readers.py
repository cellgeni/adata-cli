"""The 10x conversions, checked by the tools whose files they read and write.

scanpy's `read_10x_h5` is the reference for import: our h5ad must hold what
scanpy reads from the same file. For export, the file we write must read in
scanpy, and in PyTables, which is how CellBender reads Cell Ranger output.
Each runs in its own uv environment, so neither becomes a dependency here.

Marked `integration`: building the environments needs `uv` and, the first
time, the network.

    pytest tests/test_external_readers.py -m integration
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import (
    write_10x_h5,
    write_cellbender_h5,
    write_hdf5_matrix,
    write_10x_mtx,
    write_10x_v2_h5,
    write_loom,
    write_parse_run,
    write_scope_loom,
    write_starsolo_velocyto,
    write_text_matrix,
)
from tests.reference_stores import must_build, offline, uv_available

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

runner = CliRunner()
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _run(packages: List[str], script: str, *args: str, python: str = PYTHON):
    """Run `script` with `packages` installed; return what it printed after RESULT."""
    if not uv_available() or offline():
        if must_build():
            pytest.fail("uv and the network are required for these tests in CI")
        pytest.skip("needs uv and the network")
    cmd = ["uv", "run", "--no-project", "--python", python]
    for package in packages:
        cmd += ["--with", package]
    cmd += ["python", "-c", script, *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    lines = [l for l in result.stdout.splitlines() if l.startswith("RESULT ")]
    if result.returncode != 0 or not lines:
        message = f"{' '.join(cmd[:8])} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        if must_build():
            pytest.fail(message)
        pytest.skip(message)
    return json.loads(lines[-1][len("RESULT "):])


def _convert(*args: str) -> None:
    result = runner.invoke(app, ["convert", *args])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")


def _counts(n_cells: int, n_features: int, seed: int = 0) -> np.ndarray:
    counts = np.random.default_rng(seed).poisson(0.6, (n_cells, n_features))
    counts[-1] = 0  # an empty barcode, as raw matrices are full of
    return counts


def _barcodes(n: int) -> List[str]:
    return [f"AAACCCAAG{i:07d}-1" for i in range(n)]


# Run inside the scanpy environment: read the 10x file with scanpy and our
# conversion with anndata, and list every difference.
_COMPARE_IMPORT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad

h5, h5ad, genome = sys.argv[1], sys.argv[2], sys.argv[3] or None
ref = sc.read_10x_h5(h5, gex_only=False, genome=genome)
got = ad.read_h5ad(h5ad)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(ref.X.toarray(), got.X.toarray()):
        problems.append("X differs")
    for axis in ("obs", "var"):
        a, b = getattr(ref, axis), getattr(got, axis)
        if list(a.index) != list(b.index):
            problems.append(f"{axis}_names differ")
        if list(a.columns) != list(b.columns):
            problems.append(f"{axis} columns {list(a.columns)} != {list(b.columns)}")
        for column in a.columns:
            if column not in b.columns:
                continue
            left, right = a[column], b[column]
            if left.dtype == bool or right.dtype == bool:
                same = list(left.astype(bool)) == list(right.astype(bool))
            else:
                same = list(left.astype(str)) == list(right.astype(str))
            if not same:
                problems.append(f"{axis}[{column!r}] differs")
print("RESULT " + json.dumps(problems))
"""


def _import_matches_scanpy(h5: Path, tmp_path: Path, *extra: str, genome: str = ""):
    out = tmp_path / f"{h5.stem}.h5ad"
    # scanpy names var by symbol; adata's default is ids.
    _convert(str(h5), "-o", str(out), "--var-names", "gene_symbols", *extra)
    problems = _run(["scanpy"], _COMPARE_IMPORT, str(h5), str(out), genome)
    assert problems == [], problems


def test_a_multimodal_v3_matrix_imports_as_scanpy_reads_it(tmp_path):
    counts = _counts(40, 9)
    h5 = write_10x_h5(
        tmp_path / "multi.h5",
        counts,
        barcodes=_barcodes(40),
        ids=[f"ENSG{i:05d}" for i in range(6)] + ["CD3", "CD4", "CD8"],
        names=["A", "B", "B", "C", "D", "E", "CD3_TotalSeqB", "CD4_TotalSeqB", "CD8"],
        types=["Gene Expression"] * 6 + ["Antibody Capture"] * 3,
        extra={
            "pattern": [""] * 6 + ["5PNNNNNNNNNN(BC)"] * 3,
            "read": [""] * 6 + ["R2"] * 3,
        },
    )
    _import_matches_scanpy(h5, tmp_path)


def test_a_probe_barcode_matrix_imports_as_scanpy_reads_it(tmp_path):
    h5 = write_10x_h5(
        tmp_path / "flex.h5",
        _counts(30, 6, seed=1),
        barcodes=_barcodes(30),
        ids=[f"ENSG{i}|PROBE{i}" for i in range(6)],
        names=[f"G{i}" for i in range(6)],
        probe_gene_ids=[f"ENSG{i // 2}" for i in range(6)],
        extra={"filtered_probes": [True, False, True, True, False, True]},
        filtered_barcodes=[i % 3 != 0 for i in range(30)],
    )
    _import_matches_scanpy(h5, tmp_path)


def test_a_v3_genome_filter_matches_scanpys(tmp_path):
    h5 = write_10x_h5(
        tmp_path / "barnyard.h5",
        _counts(25, 6, seed=2),
        barcodes=_barcodes(25),
        ids=[f"G{i}" for i in range(6)],
        names=[f"g{i}" for i in range(6)],
        genomes=["GRCh38"] * 3 + ["mm10"] * 3,
    )
    _import_matches_scanpy(h5, tmp_path, "--genome", "mm10", genome="mm10")


def test_a_cell_ranger_2_file_imports_as_scanpy_reads_it(tmp_path):
    h5 = write_10x_v2_h5(
        tmp_path / "v2.h5",
        {
            "hg19": (_counts(20, 5), _barcodes(20), [f"H{i}" for i in range(5)], list("abcde")),
            "mm10": (_counts(20, 4, seed=3), _barcodes(20), [f"M{i}" for i in range(4)], list("wxyz")),
        },
    )
    _import_matches_scanpy(h5, tmp_path, "--genome", "mm10", genome="mm10")


# ---------------------------------------------------------------------------
# MTX folders: scanpy's read_10x_mtx is the reference


_COMPARE_MTX = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad

folder, h5ad, prefix, compressed, make_unique = sys.argv[1:6]
ref = sc.read_10x_mtx(
    folder, var_names="gene_symbols", make_unique=make_unique == "1",
    gex_only=False, prefix=prefix or None, compressed=compressed == "1",
)
got = ad.read_h5ad(h5ad)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(ref.X.toarray(), got.X.toarray()):
        problems.append("X differs")
    for axis in ("obs", "var"):
        a, b = getattr(ref, axis), getattr(got, axis)
        if list(a.index) != list(b.index):
            problems.append(f"{axis}_names differ")
        if list(a.columns) != list(b.columns):
            problems.append(f"{axis} columns {list(a.columns)} != {list(b.columns)}")
        for column in a.columns:
            if column in b.columns and list(a[column].astype(str)) != list(b[column].astype(str)):
                problems.append(f"{axis}[{column!r}] differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize(
    "layout,make_unique",
    [("cr3", False), ("cr3", True), ("cr2", False), ("starsolo", False)],
)
def test_an_mtx_folder_imports_as_scanpy_reads_it(tmp_path, layout, make_unique):
    ids = [f"ENSG{i:05d}" for i in range(6)] + ["CD3", "CD4"]
    folder = write_10x_mtx(
        tmp_path / "folder",
        _counts(40, 8),
        barcodes=_barcodes(40),
        ids=ids,
        names=["A", "B", "B", "C", "D", "E", "CD3_TotalSeqB", "CD4_TotalSeqB"],
        types=["Gene Expression"] * 6 + ["Antibody Capture"] * 2,
        layout=layout,
        prefix="GSM1_" if layout == "cr3" else "",
    )
    out = tmp_path / "out.h5ad"
    _convert(
        str(folder), "-o", str(out), "--var-names", "gene_symbols",
        *(["--make-unique"] if make_unique else []),
    )
    problems = _run(
        ["scanpy"], _COMPARE_MTX, str(folder), str(out),
        "GSM1_" if layout == "cr3" else "", "1" if layout == "cr3" else "0",
        "1" if make_unique else "0",
    )
    assert problems == [], problems


_COMPARE_VELOCYTO = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad

folder, h5ad = sys.argv[1:3]
got = ad.read_h5ad(h5ad)
problems = []
for name in ("spliced", "unspliced", "ambiguous"):
    ref = sc.read_mtx(f"{folder}/{name}.mtx").T
    if not np.array_equal(ref.X.toarray(), got.layers[name].toarray()):
        problems.append(f"layers[{name!r}] differs")
if not np.array_equal(sc.read_mtx(f"{folder}/spliced.mtx").T.X.toarray(), got.X.toarray()):
    problems.append("X is not spliced")
print("RESULT " + json.dumps(problems))
"""


def test_velocyto_layers_match_scanpys_read_mtx(tmp_path):
    folder = write_starsolo_velocyto(
        tmp_path / "raw",
        {name: _counts(30, 5, seed=i) for i, name in enumerate(("spliced", "unspliced", "ambiguous"))},
        barcodes=_barcodes(30),
        ids=[f"G{i}" for i in range(5)],
        names=[f"g{i}" for i in range(5)],
    )
    out = tmp_path / "velo.h5ad"
    _convert(str(folder), "-o", str(out), "--var-names", "gene_symbols")
    problems = _run(["scanpy"], _COMPARE_VELOCYTO, str(folder), str(out))
    assert problems == [], problems


_READ_MTX_EXPORT_WITH_SCANPY = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad

ref, got = ad.read_h5ad(sys.argv[1]), sc.read_10x_mtx(sys.argv[2], gex_only=False, make_unique=False)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    X = ref.X.toarray() if hasattr(ref.X, "toarray") else np.asarray(ref.X)
    if not np.array_equal(X, got.X.toarray()):
        problems.append("X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append("obs_names differ")
    if list(ref.var_names) != list(got.var_names):
        problems.append("var_names differ")
    for column in ("gene_ids", "feature_types"):
        if list(ref.var[column].astype(str)) != list(got.var[column].astype(str)):
            problems.append(f"var[{column!r}] differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_scanpy_reads_the_mtx_folder_we_export(tmp_path, layout):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sparse = pytest.importorskip("scipy.sparse")

    counts = _counts(30, 7).astype(np.int32)
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](counts)
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=_barcodes(30)),
        var=pd.DataFrame(
            {
                "gene_ids": [f"ENSG{i}" for i in range(7)],
                "feature_types": ["Gene Expression"] * 7,
            },
            index=[f"G{i}" for i in range(7)],
        ),
    ).write_h5ad(source)

    out = tmp_path / "exported"
    _convert(str(source), "-o", str(out), "--to", "10x-mtx")
    problems = _run(["scanpy"], _READ_MTX_EXPORT_WITH_SCANPY, str(source), str(out))
    assert problems == [], problems


@pytest.mark.parametrize("filtered", [False, True], ids=["full", "filtered"])
def test_a_cellbender_file_imports_as_scanpy_reads_its_matrix(tmp_path, filtered):
    """scanpy reads CellBender output as a 10x matrix and ignores the latents;
    the matrix, barcodes and features must agree. The latents themselves are
    checked in test_cellbender.py."""
    truth = write_cellbender_h5(
        tmp_path / "cellbender.h5",
        _counts(40, 6),
        barcodes=_barcodes(40),
        ids=[f"ENSG{i}" for i in range(6)],
        names=[f"G{i}" for i in range(6)],
        filtered=filtered,
    )
    out = tmp_path / "cellbender.h5ad"
    _convert(str(truth.path), "-o", str(out), "--var-names", "gene_symbols")
    problems = _run(["scanpy"], _COMPARE_CELLBENDER, str(truth.path), str(out))
    assert problems == [], problems


_COMPARE_CELLBENDER = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad

ref, got = sc.read_10x_h5(sys.argv[1], gex_only=False), ad.read_h5ad(sys.argv[2])
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(ref.X.toarray(), got.X.toarray()):
        problems.append("X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append("obs_names differ")
    if list(ref.var_names) != list(got.var_names):
        problems.append("var_names differ")
    for column in ref.var.columns:
        if list(ref.var[column].astype(str)) != list(got.var[column].astype(str)):
            problems.append(f"var[{column!r}] differs")
print("RESULT " + json.dumps(problems))
"""


_READ_EXPORT_WITH_SCANPY = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, scanpy as sc, anndata as ad

ref, got = ad.read_h5ad(sys.argv[1]), sc.read_10x_h5(sys.argv[2], gex_only=False)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    X = ref.X if sys.argv[3] == "X" else ref.layers[sys.argv[3]]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if not np.array_equal(X, got.X.toarray()):
        problems.append("X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append("obs_names differ")
    if list(ref.var_names) != list(got.var_names):
        problems.append("var_names differ")
    for column in ("gene_ids", "feature_types"):
        if list(ref.var[column].astype(str)) != list(got.var[column].astype(str)):
            problems.append(f"var[{column!r}] differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_scanpy_reads_what_we_export(tmp_path, layout):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sparse = pytest.importorskip("scipy.sparse")

    counts = _counts(30, 7).astype(np.int32)
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](counts)
    source = tmp_path / "source.h5ad"
    ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=_barcodes(30)),
        var=pd.DataFrame(
            {
                "gene_ids": [f"ENSG{i}" for i in range(7)],
                "feature_types": ["Gene Expression"] * 7,
            },
            index=[f"G{i}" for i in range(7)],
        ),
    ).write_h5ad(source)

    out = tmp_path / "exported.h5"
    _convert(str(source), "-o", str(out))
    problems = _run(["scanpy"], _READ_EXPORT_WITH_SCANPY, str(source), str(out), "X")
    assert problems == [], problems


_READ_WITH_PYTABLES = r"""
import json, sys, tables
paths = ["/matrix/" + p for p in ("barcodes", "data", "indices", "indptr", "shape")]
paths += ["/matrix/features/" + p for p in ("id", "name", "feature_type", "genome", "_all_tag_keys")]
out = {}
with tables.open_file(sys.argv[1]) as f:
    for path in paths:
        values = f.get_node(path).read()
        out[path] = [v.decode() if isinstance(v, bytes) else int(v) for v in values.tolist()]
    filetype = f.get_node_attr("/", "filetype")
    out["filetype"] = filetype.decode() if isinstance(filetype, bytes) else str(filetype)
print("RESULT " + json.dumps(out))
"""


def test_pytables_reads_what_we_export(tmp_path):
    """CellBender reads Cell Ranger output with PyTables, which cannot read
    variable-length strings; Cell Ranger writes fixed-width ones, and so must
    we."""
    counts = _counts(12, 4)
    h5 = write_10x_h5(
        tmp_path / "in.h5",
        counts,
        barcodes=_barcodes(12),
        ids=[f"ENSG{i}" for i in range(4)],
        names=["A", "B", "C", "D"],
    )
    h5ad, back = tmp_path / "in.h5ad", tmp_path / "back.h5"
    _convert(str(h5), "-o", str(h5ad))
    _convert(str(h5ad), "-o", str(back))

    got = _run(["tables"], _READ_WITH_PYTABLES, str(back))
    rows, cols = np.nonzero(counts)
    assert got["/matrix/barcodes"] == _barcodes(12)
    assert got["/matrix/data"] == counts[rows, cols].tolist()
    assert got["/matrix/indices"] == cols.tolist()
    assert got["/matrix/shape"] == [4, 12]
    assert got["/matrix/features/name"] == ["A", "B", "C", "D"]
    assert got["/matrix/features/feature_type"] == ["Gene Expression"] * 4
    assert got["/matrix/features/_all_tag_keys"] == ["genome"]
    assert got["filetype"] == "matrix"


@pytest.mark.parametrize("genome", [None, "mm10"], ids=["all", "mm10"])
def test_a_parse_run_imports_as_parses_scanpy_recipe_reads_it(tmp_path, genome):
    """Parse's own recipe: read_mtx, then both tables with pandas, indexed by
    bc_wells and gene_name -- and, for one species of a barnyard run, the
    cells and genes of that species."""
    run = write_parse_run(tmp_path / "run", genomes=("hg38", "mm10"), n_cells=30, n_genes=8)
    out = tmp_path / "parse.h5ad"
    extra = ["--genome", genome] if genome else []
    _convert(str(run.root), "-o", str(out), "--var-names", "gene_symbols", *extra)
    problems = _run(
        ["scanpy"], _PARSE_RECIPE, str(run.dge().path), str(out), genome or "",
    )
    assert problems == [], problems


_PARSE_RECIPE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad

dge, ours, genome = sys.argv[1], sys.argv[2], sys.argv[3]
ref = sc.read_mtx(dge + "/count_matrix.mtx")
ref.var = pd.read_csv(dge + "/all_genes.csv").set_index("gene_name")
ref.obs = pd.read_csv(dge + "/cell_metadata.csv").set_index("bc_wells")
ref.var.index.name = ref.obs.index.name = None
if genome:
    ref = ref[ref.obs["species"] == genome, ref.var["genome"] == genome].copy()
ref.strings_to_categoricals()
got = ad.read_h5ad(ours)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(ref.X.toarray(), got.X.toarray()):
        problems.append("X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append("obs_names differ")
    if list(ref.var_names) != list(got.var_names):
        problems.append("var_names differ")
    if list(ref.var["gene_id"]) != list(got.var["gene_ids"]):
        problems.append("gene ids differ")
    for frame, mine in ((ref.obs, got.obs), (ref.var.drop(columns="gene_id"), got.var.drop(columns="gene_ids"))):
        for column in frame.columns:
            a, b = frame[column], mine[column]
            if str(a.dtype) != str(b.dtype) and not (
                isinstance(a.dtype, pd.CategoricalDtype) and isinstance(b.dtype, pd.CategoricalDtype)
            ):
                problems.append(f"{column!r} is {b.dtype}, the recipe's {a.dtype}")
            elif isinstance(a.dtype, pd.CategoricalDtype) and list(a.cat.categories) != list(b.cat.categories):
                problems.append(f"{column!r} categories differ")
            elif list(a.astype(str)) != list(b.astype(str)):
                problems.append(f"{column!r} differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("velocyto", [False, True], ids=["loompy", "velocyto"])
def test_a_loom_imports_as_anndata_reads_it(tmp_path, velocyto):
    """anndata.read_loom (with loompy) is the reference: X, layers, names and
    attributes. Ours keeps dtypes where anndata casts X to float32, so values
    are compared, not types."""
    truth = write_loom(tmp_path / "in.loom", velocyto=velocyto)
    out = tmp_path / "out.h5ad"
    _convert(str(truth.path), "-o", str(out), "--var-names", "gene_symbols")
    problems = _run(["anndata", "loompy", "scipy"], _COMPARE_LOOM, str(truth.path), str(out))
    assert problems == [], problems


_COMPARE_LOOM = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad

ref, got = ad.io.read_loom(sys.argv[1]), ad.read_h5ad(sys.argv[2])
problems = []
dense = lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m)
names = lambda a: sorted(k for k in a.layers.keys() if k)
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(dense(ref.X), dense(got.X).astype(np.float32)):
        problems.append("X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append("obs_names differ")
    if list(ref.var_names) != list(got.var_names):
        problems.append("var_names differ")
    if names(ref) != names(got):
        problems.append(f"layers {names(ref)} != {names(got)}")
    for key in names(ref):
        if key in got.layers and not np.array_equal(dense(ref.layers[key]), dense(got.layers[key])):
            problems.append(f"layers[{key!r}] differs")
    for frame, mine, axis in ((ref.obs, got.obs, "obs"), (ref.var, got.var, "var")):
        if sorted(frame.columns) != sorted(mine.columns):
            problems.append(f"{axis} columns {sorted(frame.columns)} != {sorted(mine.columns)}")
        for column in frame.columns:
            if column in mine and list(frame[column].astype(str)) != list(mine[column].astype(str)):
                problems.append(f"{axis}[{column!r}] differs")
    for key in ref.obsm:
        if key not in got.obsm or not np.allclose(ref.obsm[key], got.obsm[key]):
            problems.append(f"obsm[{key!r}] differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("layout", ["csr", "dense"])
def test_loompy_accepts_and_anndata_reads_what_we_export(tmp_path, layout):
    """loompy's own validator passes our loom, and anndata.read_loom reads
    back the matrix, layers, names and columns we wrote."""
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    counts = _counts(40, 7).astype(np.float32)
    X = sparse.csr_matrix(counts) if layout == "csr" else counts
    source = tmp_path / "in.h5ad"
    ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cluster": pd.Categorical(["a", "b"] * 20)}, index=_barcodes(40)),
        var=pd.DataFrame({"gene_symbols": [f"G{i}" for i in range(7)]}, index=[f"ENSG{i}" for i in range(7)]),
        layers={"spliced": sparse.csr_matrix(counts * 2)},
    ).write_h5ad(source)
    out = tmp_path / "out.loom"
    _convert(str(source), "-o", str(out))
    problems = _run(["anndata", "loompy", "scipy"], _CHECK_LOOM_EXPORT, str(source), str(out))
    assert problems == [], problems


_CHECK_LOOM_EXPORT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad
from loompy.loom_validator import LoomValidator

src = ad.read_h5ad(sys.argv[1])
validator = LoomValidator()
problems = [] if validator.validate(sys.argv[2]) else [f"validator: {validator.errors}"]
got = ad.io.read_loom(sys.argv[2], var_names="Accession")
dense = lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m)
if not np.array_equal(dense(got.X), dense(src.layers["spliced"]).astype(np.float32)):
    problems.append("X (spliced, as anndata picks it) differs")
if not np.array_equal(dense(got.layers["matrix"]), dense(src.X)):
    problems.append("the main matrix differs from X")
if list(got.obs_names) != list(src.obs_names):
    problems.append("obs_names differ")
if list(got.var_names) != list(src.var_names):
    problems.append("var_names differ")
if list(got.var["Gene"]) != list(src.var["gene_symbols"]):
    problems.append("Gene differs from gene_symbols")
if list(got.obs["cluster"].astype(str)) != list(src.obs["cluster"].astype(str)):
    problems.append("cluster differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("default_in_extra", [True, False], ids=["export2loom", "Embedding-only"])
def test_a_scope_loom_imports_as_the_pyscenic_recipe_reads_it(tmp_path, default_in_extra):
    """The pySCENIC tutorial reads a SCope loom with loompy: the AUC and
    regulons as DataFrames of the compound attributes, the embeddings paired
    by id (the default from Embedding), the clusterings labelled from
    MetaData. Ours must hold the same, with the names to find them."""
    truth = write_scope_loom(tmp_path / "scope.loom", default_in_extra=default_in_extra)
    out = tmp_path / "out.h5ad"
    _convert(str(truth.path), "-o", str(out))
    problems = _run(["anndata", "loompy", "pandas"], _COMPARE_SCOPE, str(truth.path), str(out))
    assert problems == [], problems


_COMPARE_SCOPE = r"""
import base64, json, sys, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, loompy as lp, anndata as ad

got = ad.read_h5ad(sys.argv[2])
problems = []
with lp.connect(sys.argv[1], mode="r", validate=False) as lf:
    meta = json.loads(zlib.decompress(base64.b64decode(lf.attrs.MetaData)))
    auc = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
    regulons = pd.DataFrame(lf.ra.Regulons, index=lf.ra.Gene)
    drx = pd.DataFrame(lf.ca.Embeddings_X, index=lf.ca.CellID)
    dry = pd.DataFrame(lf.ca.Embeddings_Y, index=lf.ca.CellID)
    default = pd.DataFrame(lf.ca.Embedding, index=lf.ca.CellID)
    clusterings = pd.DataFrame(lf.ca.Clusterings, index=lf.ca.CellID)

mine = got.obsm["RegulonsAUC"]
if list(mine.columns) != list(auc.columns) or list(mine.index) != list(auc.index):
    problems.append("RegulonsAUC is labelled differently")
elif not np.array_equal(mine.to_numpy(), auc.to_numpy()) or list(mine.dtypes) != list(auc.dtypes):
    problems.append("RegulonsAUC differs")
mine = got.varm["Regulons"]
if list(mine.columns) != list(regulons.columns):
    problems.append("Regulons is labelled differently")
elif not np.array_equal(mine.to_numpy(), regulons.to_numpy()):
    problems.append("Regulons differs")
else:
    # Each regulon's targets, as the tutorial lists them.
    for name in regulons.columns:
        if list(regulons.index[regulons[name] == 1]) != list(got.var["Gene"][mine[name] == 1]):
            problems.append(f"targets of {name} differ")

where = got.uns["scope"]["embeddings"]
for entry in meta["embeddings"]:
    id = str(entry["id"])
    if id in drx.columns:
        ref = np.column_stack([drx[id], dry[id]])
    else:
        ref = default[["_X", "_Y"]].to_numpy()
    if id not in where.index or where.loc[id, "name"] != entry["name"]:
        problems.append(f"embedding {id} is not recorded")
    elif not np.array_equal(got.obsm[where.loc[id, "obsm"]], ref):
        problems.append(f"embedding {id} differs")

where = got.uns["scope"]["clusterings"]
for entry in meta["clusterings"]:
    id = str(entry["id"])
    labels = {c["id"]: c.get("description", str(c["id"])) for c in entry["clusters"]}
    ref = [labels.get(v) for v in clusterings[id]]
    column = got.obs[where.loc[id, "obs"]]
    if [None if pd.isna(v) else v for v in column] != ref:
        problems.append(f"clustering {id} differs")
    if list(column.cat.categories) != [labels[k] for k in sorted(labels)]:
        problems.append(f"clustering {id}'s categories differ")

if json.loads(got.uns["scope"]["metadata"]) != meta:
    problems.append("MetaData differs")
thresholds = got.uns["scope"]["regulon_thresholds"]
ref = pd.DataFrame(meta["regulonThresholds"]).set_index("regulon")
for column in thresholds.columns:
    if list(thresholds.index) != list(ref.index) or list(thresholds[column]) != list(ref[column]):
        problems.append(f"regulon thresholds' {column} differs")
print("RESULT " + json.dumps(problems))
"""


def test_text_matrices_import_as_anndata_reads_them(tmp_path):
    """anndata's read_csv and read_text are the reference, for every layout
    they tell apart. They give float32 where ours keeps integers or float64,
    so values are compared, not types; a transposed file against `.T`."""
    cases = []
    for i, (kw, reader, delimiter, extra) in enumerate([
        ({"header": "corner"}, "read_csv", ",", []),
        ({"header": "short", "gz": True}, "read_csv", ",", []),
        ({"header": "comment", "delimiter": None}, "read_text", "", []),
        ({"header": "none", "delimiter": "\t", "row_names": False}, "read_text", "\t", []),
        ({"header": "corner", "transpose": True}, "read_csv", ",", ["--transpose"]),
    ]):
        suffix = ".txt" if reader == "read_text" else ".csv"
        truth = write_text_matrix(tmp_path / f"m{i}{suffix}{'.gz' if kw.get('gz') else ''}", **kw)
        out = tmp_path / f"m{i}.h5ad"
        _convert(str(truth.path), "-o", str(out), *extra)
        cases.append([str(truth.path), reader, delimiter, str(out), bool(extra)])
    problems = _run(["anndata", "scipy"], _COMPARE_TEXT, json.dumps(cases))
    assert problems == [], problems


_COMPARE_TEXT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad

problems = []
dense = lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m)
for path, reader, delimiter, out, transposed in json.loads(sys.argv[1]):
    ref = getattr(ad.io, reader)(path, delimiter=delimiter or None)
    if transposed:
        ref = ref.T
    got = ad.read_h5ad(out)
    if ref.shape != got.shape:
        problems.append(f"{path}: shape {ref.shape} != {got.shape}")
        continue
    if not np.array_equal(dense(ref.X), dense(got.X).astype(np.float32)):
        problems.append(f"{path}: X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append(f"{path}: obs_names differ")
    if list(ref.var_names) != list(got.var_names):
        problems.append(f"{path}: var_names differ")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("transpose", [False, True], ids=["cells-as-rows", "transposed"])
def test_an_hdf5_matrix_imports_as_anndata_reads_it(tmp_path, transpose):
    """anndata's read_hdf is the reference, its names from row_names and
    col_names; a transposed dataset against `.T`."""
    truth = write_hdf5_matrix(tmp_path / "m.hdf5", transpose=transpose, chunks=(4, 4))
    out = tmp_path / "out.h5ad"
    _convert(str(truth.path), "-o", str(out), "--key", truth.key,
             *(["--transpose"] if transpose else []))
    problems = _run(
        ["anndata", "scipy"], _COMPARE_HDF5, str(truth.path), truth.key, str(out), str(transpose)
    )
    assert problems == [], problems


_COMPARE_HDF5 = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad

ref = ad.io.read_hdf(sys.argv[1], sys.argv[2])
if sys.argv[4] == "True":
    ref = ref.T
got = ad.read_h5ad(sys.argv[3])
dense = lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(dense(ref.X), dense(got.X)):
        problems.append("X differs")
    if dense(ref.X).dtype != got.X.dtype:
        problems.append(f"dtype {dense(ref.X).dtype} != {got.X.dtype}")
    for axis in ("obs_names", "var_names"):
        mine = [str(v) for v in getattr(got, axis)]
        theirs = [v.decode() if isinstance(v, bytes) else str(v) for v in getattr(ref, axis)]
        if mine != theirs:
            problems.append(f"{axis} differ")
print("RESULT " + json.dumps(problems))
"""


# pycisTopic pins pandas 1.5, so its pickles hold pandas 1.5's internals:
# written there, read here with pandas 2 or 3, compared there again.
_PYCISTOPIC_ENV = ["pandas==1.5.3", "numpy<2", "scipy<1.14", "h5py<3.12", "anndata==0.10.9"]

_CISTOPIC_STAND_INS = r"""
import sys, types
package = types.ModuleType("pycisTopic"); package.__path__ = []; sys.modules["pycisTopic"] = package
for module, name in (("cistopic_class", "CistopicObject"), ("lda_models", "CistopicLDAModel")):
    mod = types.ModuleType(f"pycisTopic.{module}")
    setattr(mod, name, type(name, (), {"__module__": f"pycisTopic.{module}"}))
    sys.modules[f"pycisTopic.{module}"] = mod
    setattr(package, module, mod)
from pycisTopic.cistopic_class import CistopicObject
from pycisTopic.lda_models import CistopicLDAModel
"""

_WRITE_CISTOPIC = _CISTOPIC_STAND_INS + r"""
import json, pickle
import numpy as np, pandas as pd, scipy.sparse as sp
rng = np.random.default_rng(0)
cells = [f"AAAC{i:04d}-1___s{i % 2}" for i in range(40)]
regions = [f"chr1:{100 * i}-{100 * i + 50}" for i in range(25)]
counts = rng.poisson(0.7, (25, 40)).astype(np.int32)
obj = CistopicObject()
obj.fragment_matrix = sp.csr_matrix(counts)
obj.binary_matrix = sp.csr_matrix((counts > 0).astype(np.int32))
obj.cell_names, obj.region_names = cells, regions
obj.cell_data = pd.DataFrame({
    "sample_id": pd.Categorical([f"s{i % 2}" for i in range(40)]),
    "cisTopic_nr_frag": counts.sum(0).astype(object),
    "FRIP": rng.random(40),
    "leiden": [str(i % 3) for i in range(40)],
}, index=cells).iloc[::-1]
obj.region_data = pd.DataFrame({"Chromosome": ["chr1"] * 25, "Start": np.arange(25, dtype=np.int32) * 100}, index=regions)
obj.project, obj.path_to_fragments = "demo", {"s0": "/f0.tsv.gz"}
model = CistopicLDAModel()
model.cell_topic = pd.DataFrame(rng.random((5, 40)), index=[f"Topic{t + 1}" for t in range(5)], columns=cells)
model.topic_region = model.region_topic = pd.DataFrame(rng.random((25, 5)), index=regions, columns=model.cell_topic.index)
model.parameters = pd.DataFrame({"Parameter": ["lda", 5, True]}, index=["package", "n_topics", "alpha_by_topic"])
model.n_topic = 5
obj.selected_model = model
obj.projections = {"cell": {"UMAP": pd.DataFrame(rng.normal(size=(40, 2)), index=cells, columns=["UMAP_1", "UMAP_2"])}, "region": {}}
with open(sys.argv[1], "wb") as fh:
    pickle.dump(obj, fh)
print("RESULT " + json.dumps(pd.__version__))
"""

_COMPARE_CISTOPIC = _CISTOPIC_STAND_INS + r"""
import json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad
with open(sys.argv[1], "rb") as fh:
    obj = pickle.load(fh)
got = ad.read_h5ad(sys.argv[2])
problems = []
if (got.X != obj.fragment_matrix.T.tocsr()).nnz or got.X.dtype != np.int32:
    problems.append("X differs")
if (got.layers["binary"] != obj.binary_matrix.T.tocsr()).nnz:
    problems.append("binary differs")
if list(got.obs_names) != obj.cell_names or list(got.var_names) != obj.region_names:
    problems.append("names differ")
cd = obj.cell_data.loc[obj.cell_names]
if list(got.obs["sample_id"].astype(str)) != list(cd["sample_id"].astype(str)):
    problems.append("sample_id differs")
if list(got.obs["cisTopic_nr_frag"]) != list(cd["cisTopic_nr_frag"]):
    problems.append("cisTopic_nr_frag differs")
if not np.array_equal(got.obs["FRIP"], cd["FRIP"]) or list(got.obs["leiden"].astype(str)) != list(cd["leiden"]):
    problems.append("obs columns differ")
if list(got.var["Start"]) != list(obj.region_data["Start"]):
    problems.append("var differs")
model = obj.selected_model
if not np.array_equal(got.obsm["X_topic"], model.cell_topic[obj.cell_names].to_numpy().T):
    problems.append("X_topic differs")
if not np.array_equal(got.varm["topic_region"], model.region_topic.to_numpy()):
    problems.append("topic_region differs")
if not np.array_equal(got.obsm["X_UMAP"], obj.projections["cell"]["UMAP"].to_numpy()):
    problems.append("X_UMAP differs")
if got.uns["pycistopic"]["model"]["n_topic"] != 5 or got.uns["pycistopic"]["project"] != "demo":
    problems.append("uns differs")
print("RESULT " + json.dumps(problems))
"""


def test_a_pandas_15_cistopic_pickle_imports_as_pycistopic_holds_it(tmp_path):
    """pycisTopic pins pandas 1.5: a pickle written there is read here, with
    this environment's pandas, and compared there with the objects it holds."""
    path, out = tmp_path / "cistopic_obj.pkl", tmp_path / "out.h5ad"
    version = _run(_PYCISTOPIC_ENV, _WRITE_CISTOPIC, str(path), python="3.11")
    assert version.startswith("1.5")
    _convert(str(path), "-o", str(out))
    problems = _run(_PYCISTOPIC_ENV, _COMPARE_CISTOPIC, str(path), str(out), python="3.11")
    assert problems == [], problems

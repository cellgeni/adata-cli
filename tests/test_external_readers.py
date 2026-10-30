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
    write_visium_hd_outs,
    write_visium_outs,
    write_xenium_bundle,
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
from tests.reference_stores import must_build, offline, oracle, uv_available

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
        # Pinned: an oracle that moves on its own changes what is checked.
        cmd += ["--with", oracle(package)]
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


def test_a_scope_export_reads_with_the_pyscenic_recipe(tmp_path):
    """Our --scope loom, read the pySCENIC tutorial's way with loompy, holds
    what the AnnData holds: a SCope loom (fixture) to h5ad, to a SCope loom
    of ours, and back, compared through the recipe."""
    truth = write_scope_loom(tmp_path / "scope.loom", compressed=False)
    first, back, again = tmp_path / "a.h5ad", tmp_path / "back.loom", tmp_path / "b.h5ad"
    _convert(str(truth.path), "-o", str(first))
    _convert(str(first), "-o", str(back), "--scope")
    _convert(str(back), "-o", str(again))
    problems = _run(["anndata", "loompy", "pandas"], _COMPARE_SCOPE, str(back), str(again))
    assert problems == [], problems


_COMPARE_SCOPE = r"""
import base64, json, sys, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, loompy as lp, anndata as ad

got = ad.read_h5ad(sys.argv[2])
problems = []
with lp.connect(sys.argv[1], mode="r", validate=False) as lf:
    try:
        meta = json.loads(lf.attrs.MetaData)
    except ValueError:  # the tutorial's form: base64 of zlib
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


_CONVERT_HERE = r"""
import json, sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from adata.commands.convert import convert_store
convert_store(Path(sys.argv[1]), Path(sys.argv[2]), Console(quiet=True))
print("RESULT " + json.dumps(pd.__version__))
"""


def test_a_pandas_15_cistopic_pickle_converts_under_pandas_2_too(tmp_path):
    """The extra allows pandas 2 as well as 3; this environment has 3, so
    the conversion itself also runs under pandas 2.2, with this checkout."""
    path, out = tmp_path / "cistopic_obj.pkl", tmp_path / "out.h5ad"
    _run(_PYCISTOPIC_ENV, _WRITE_CISTOPIC, str(path), python="3.11")
    root = str(Path(__file__).resolve().parent.parent)
    version = _run([root, "pandas==2.2.3", "scipy"], _CONVERT_HERE, str(path), str(out))
    assert version.startswith("2.2")
    problems = _run(_PYCISTOPIC_ENV, _COMPARE_CISTOPIC, str(path), str(out), python="3.11")
    assert problems == [], problems


# ---------------------------------------------------------------------------
# MuData. mudata itself writes the file we read, and reads the file we write.

_MUDATA = ["mudata", "anndata", "scipy"]

# Write a MuData with mudata: two modalities sharing some cells, in another
# order, columns pulled into the global obs, a global embedding and graph.
_WRITE_MUDATA = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, mudata as md, scipy.sparse as sp
rng = np.random.default_rng(0)
cells = [f"c{i}" for i in range(50)]
rna = ad.AnnData(sp.random(50, 7, density=0.3, format="csr", random_state=0, dtype=np.float32),
                 obs=pd.DataFrame({"n_genes": rng.integers(1, 9, 50)}, index=cells),
                 var=pd.DataFrame(index=[f"g{j}" for j in range(7)]))
rna.obsm["X_pca"] = rng.normal(size=(50, 3))
atac_cells = cells[12:][::-1]
atac = ad.AnnData(sp.random(38, 5, density=0.3, format="csr", random_state=1, dtype=np.float32),
                  obs=pd.DataFrame({"tss": rng.random(38)}, index=atac_cells),
                  var=pd.DataFrame(index=[f"p{j}" for j in range(5)]))
m = md.MuData({"rna": rna, "atac": atac})
m.pull_obs()
m.obs["leiden"] = pd.Categorical(rng.choice(["a", "b"], m.n_obs))
m.obsm["X_wnn_umap"] = rng.normal(size=(m.n_obs, 2))
m.obsp["wnn"] = sp.random(m.n_obs, m.n_obs, density=0.1, format="csr", random_state=2)
m.write(sys.argv[1])
print("RESULT " + json.dumps({"mods": list(m.mod), "obs": list(m.obs.columns)}))
"""

# Compare our per-modality h5ads with what mudata reads, global parts taken
# onto each modality's cells by name.
_COMPARE_MUDATA_IMPORT = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad, mudata as md
m = md.read_h5mu(sys.argv[1])
problems = []
for mod in m.mod:
    ref, got = m[mod], ad.read_h5ad(sys.argv[2].replace("MOD", mod))
    rows = [m.obs_names.get_loc(n) for n in ref.obs_names]
    if list(got.obs_names) != list(ref.obs_names) or list(got.var_names) != list(ref.var_names):
        problems.append(f"{mod}: names differ")
    if (got.X != ref.X).nnz:
        problems.append(f"{mod}: X differs")
    for column in ref.obs.columns:
        if list(got.obs[column]) != list(ref.obs[column]):
            problems.append(f"{mod}: own obs {column} differs")
    if list(got.obs["leiden"]) != list(m.obs["leiden"].iloc[rows]):
        problems.append(f"{mod}: global leiden differs")
    if not np.allclose(got.obsm["X_wnn_umap"], m.obsm["X_wnn_umap"][rows]):
        problems.append(f"{mod}: global X_wnn_umap differs")
    if abs(got.obsp["wnn"] - m.obsp["wnn"][rows][:, rows]).max() != 0:
        problems.append(f"{mod}: global graph differs")
    pulled = [c for c in got.obs.columns if ":" in c]
    if pulled:
        problems.append(f"{mod}: pulled columns came along: {pulled}")
print("RESULT " + json.dumps(problems))
"""

# Read our .h5mu with mudata, warnings as errors (a file without the
# userblock warns), and compare each modality with the AnnData's subset.
_COMPARE_MUDATA_EXPORT = r"""
import json, sys, warnings
import numpy as np, anndata as ad, mudata as md
a = ad.read_h5ad(sys.argv[1])
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    m = md.read_h5mu(sys.argv[2])
problems = []
if list(m.mod) != ["rna", "atac"]:
    problems.append(f"modalities {list(m.mod)}")
for mod, kind in (("rna", "Gene Expression"), ("atac", "Peaks")):
    sub, got = a[:, a.var["feature_types"] == kind], m[mod]
    if list(got.var_names) != list(sub.var_names) or (got.X != sub.X).nnz:
        problems.append(f"{mod} differs")
if list(m.obs_names) != list(a.obs_names) or list(m.obs["leiden"]) != list(a.obs["leiden"]):
    problems.append("global obs differs")
if not np.allclose(m.obsm["X_umap"], a.obsm["X_umap"]):
    problems.append("global obsm differs")
print("RESULT " + json.dumps(problems))
"""


def test_a_mudata_imports_as_mudata_reads_it(tmp_path):
    source = tmp_path / "mu.h5mu"
    written = _run(_MUDATA, _WRITE_MUDATA, str(source))
    assert "rna:n_genes" in written["obs"], "mudata pulled the modalities' columns"
    _convert(str(source), "-o", str(tmp_path / "out.h5ad"))
    assert _run(_MUDATA, _COMPARE_MUDATA_IMPORT, str(source), str(tmp_path / "out.MOD.h5ad")) == []


def test_mudata_reads_what_we_export(tmp_path):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    types = ["Gene Expression"] * 6 + ["Peaks"] * 4
    obj = ad.AnnData(
        sp.random(25, 10, density=0.3, format="csr", random_state=0, dtype=np.float32),
        obs=pd.DataFrame({"leiden": pd.Categorical(rng.choice(["x", "y"], 25))},
                         index=[f"c{i}" for i in range(25)]),
        var=pd.DataFrame({"feature_types": pd.Categorical(types)}, index=[f"f{j}" for j in range(10)]),
    )
    obj.obsm["X_umap"] = rng.normal(size=(25, 2))
    obj.write_h5ad(tmp_path / "multi.h5ad")
    _convert(str(tmp_path / "multi.h5ad"), "-o", str(tmp_path / "multi.h5mu"))
    assert _run(_MUDATA, _COMPARE_MUDATA_EXPORT, str(tmp_path / "multi.h5ad"), str(tmp_path / "multi.h5mu")) == []


# ---------------------------------------------------------------------------
# spatial: Visium against squidpy's reader, Xenium against 10x's recipe

# Space Ranger's analysis/ beside spots or bins, as pandas reads it (neither
# squidpy nor scanpy does), under today's names or Space Ranger 1's.
_ANALYSIS_PROBLEMS = r"""
def analysis_problems(folder, got):
    from pathlib import Path
    import numpy as np, pandas as pd

    def table(*names):
        for name in names:
            path = Path(folder) / "analysis" / name
            if path.is_file():
                return pd.read_csv(path, index_col="Barcode", float_precision="round_trip")
        raise SystemExit(f"no {names[0]}")

    problems = []
    clusters = table("clustering/gene_expression_graphclust/clusters.csv", "clustering/graphclust/clusters.csv")
    expected = clusters["Cluster"].astype(str).reindex(got.obs_names)
    if expected.fillna("").tolist() != got.obs["graphclust"].astype(object).fillna("").astype(str).tolist():
        problems.append("graphclust differs")
    for key, names in (
        ("X_umap", ("umap/gene_expression_2_components/projection.csv", "umap/2_components/projection.csv")),
        ("X_pca", ("pca/gene_expression_10_components/projection.csv", "pca/10_components/projection.csv")),
    ):
        expected = table(*names).reindex(got.obs_names).to_numpy()
        if not np.array_equal(expected, got.obsm[key], equal_nan=True):
            problems.append(f"obsm[{key!r}] differs")
    return problems
"""

_COMPARE_VISIUM = _ANALYSIS_PROBLEMS + r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad, squidpy as sq

outs, h5ad = sys.argv[1], sys.argv[2]
ref = sq.read.visium(outs, counts_file="filtered_feature_bc_matrix.h5", load_images=True)
got = ad.read_h5ad(h5ad)
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
    for column in ("in_tissue", "array_row", "array_col"):
        if ref.obs[column].dtype != got.obs[column].dtype:
            problems.append(f"obs[{column!r}] is {got.obs[column].dtype}, not {ref.obs[column].dtype}")
        if not np.array_equal(ref.obs[column], got.obs[column]):
            problems.append(f"obs[{column!r}] differs")
    a, b = ref.obsm["spatial"], got.obsm["spatial"]
    if a.dtype != b.dtype:
        problems.append(f"obsm['spatial'] is {b.dtype}, not {a.dtype}")
    if not np.array_equal(a, b):
        problems.append("obsm['spatial'] differs")
    (lib,) = ref.uns["spatial"]
    if lib not in got.uns["spatial"]:
        problems.append(f"no library {lib!r}")
    else:
        r, g = ref.uns["spatial"][lib], got.uns["spatial"][lib]
        for name, image in r["images"].items():
            mine = g["images"].get(name)
            if mine is None or mine.dtype != image.dtype or not np.array_equal(mine, image):
                problems.append(f"image {name} differs")
        if {k: float(v) for k, v in r["scalefactors"].items()} != {k: float(v) for k, v in g["scalefactors"].items()}:
            problems.append("scalefactors differ")
        for key in ("chemistry_description", "software_version"):
            if str(r["metadata"].get(key)) != str(g["metadata"].get(key)):
                problems.append(f"metadata[{key!r}] differs")
    problems += analysis_problems(outs, got)
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("version", ["sr1", "sr2", "sr3_float"])
def test_visium_imports_as_squidpy_reads_it(tmp_path, version):
    truth = write_visium_outs(tmp_path / "run", version=version, analysis=True)
    out = tmp_path / "visium.h5ad"
    # squidpy keeps Gene Expression only and names var by symbol.
    _convert(
        str(truth.outs), "-o", str(out),
        "--feature-type", "Gene Expression", "--var-names", "gene_symbols",
    )
    assert _run(["squidpy"], _COMPARE_VISIUM, str(truth.outs), str(out)) == []


# 10x's documented recipe for Xenium in scanpy: the matrix, the cell table
# reindexed to it, the centroids as obsm["spatial"], and the analysis CSVs.
# spatialdata-io's reader needs the boundaries and morphology images too,
# which a fixture of this size does not carry.
_COMPARE_XENIUM = r"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, scanpy as sc

bundle, h5ad = Path(sys.argv[1]), sys.argv[2]
ref = sc.read_10x_h5(bundle / "cell_feature_matrix.h5", gex_only=False)
cells = pd.read_csv(bundle / "cells.csv.gz", index_col="cell_id")
cells.index = cells.index.astype(str)
cells = cells.reindex(ref.obs_names)
ref.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy()
ref.obs = cells.drop(columns=["x_centroid", "y_centroid"])
analysis = bundle / "analysis"
clusters = pd.read_csv(analysis / "clustering/gene_expression_graphclust/clusters.csv", index_col="Barcode")
clusters.index = clusters.index.astype(str)
ref.obs["graphclust"] = clusters["Cluster"].astype(str).reindex(ref.obs_names).astype("category")
umap = pd.read_csv(analysis / "umap/gene_expression_2_components/projection.csv", index_col="Barcode")
umap.index = umap.index.astype(str)
ref.obsm["X_umap"] = umap.reindex(ref.obs_names).to_numpy()
ref.strings_to_categoricals()

got = ad.read_h5ad(h5ad)
problems = []
if ref.shape != got.shape:
    problems.append(f"shape {ref.shape} != {got.shape}")
else:
    if not np.array_equal(ref.X.toarray(), got.X.toarray()):
        problems.append("X differs")
    if list(ref.obs_names) != list(got.obs_names):
        problems.append("obs_names differ")
    if list(ref.obs.columns) != list(got.obs.columns):
        problems.append(f"obs columns {list(ref.obs.columns)} != {list(got.obs.columns)}")
    for column in ref.obs.columns:
        if column not in got.obs:
            continue
        a, b = ref.obs[column], got.obs[column]
        if str(a.dtype) != str(b.dtype):
            problems.append(f"obs[{column!r}] is {b.dtype}, not {a.dtype}")
        elif a.astype(str).tolist() != b.astype(str).tolist():
            problems.append(f"obs[{column!r}] differs")
    for key in ("spatial", "X_umap"):
        if not np.allclose(ref.obsm[key], got.obsm[key], equal_nan=True):
            problems.append(f"obsm[{key!r}] differs")
print("RESULT " + json.dumps(problems))
"""


@pytest.mark.parametrize("ids,shuffled", [("int", False), ("string", True)])
def test_xenium_imports_as_10x_s_scanpy_recipe_reads_it(tmp_path, ids, shuffled):
    truth = write_xenium_bundle(tmp_path / "xenium", ids=ids, shuffled=shuffled, missing_in_analysis=2)
    out = tmp_path / "xenium.h5ad"
    _convert(str(truth.bundle), "-o", str(out), "--var-names", "gene_symbols")
    assert _run(["scanpy"], _COMPARE_XENIUM, str(truth.bundle), str(out)) == []


# Visium HD's bins as scanpy users read them, squidpy having no HD reader:
# the bin folder's matrix, the Parquet positions reindexed to it, and the
# pixel columns as obsm["spatial"], [col, row] as for standard Visium.
_COMPARE_VISIUM_HD = _ANALYSIS_PROBLEMS + r"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, scanpy as sc

folder, h5ad = Path(sys.argv[1]), sys.argv[2]
ref = sc.read_10x_h5(folder / "filtered_feature_bc_matrix.h5")
positions = pd.read_parquet(folder / "spatial" / "tissue_positions.parquet").set_index("barcode")
positions = positions.reindex(ref.obs_names)
got = ad.read_h5ad(h5ad)
problems = []
if list(ref.obs_names) != list(got.obs_names):
    problems.append("obs_names differ")
if not np.array_equal(ref.X.toarray(), got.X.toarray()):
    problems.append("X differs")
for column in ("in_tissue", "array_row", "array_col"):
    if not np.array_equal(positions[column].to_numpy().astype(np.int64), got.obs[column].to_numpy()):
        problems.append(f"obs[{column!r}] differs")
expected = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
if not np.array_equal(expected, got.obsm["spatial"]):
    problems.append("obsm['spatial'] differs")
scalefactors = json.loads((folder / "spatial" / "scalefactors_json.json").read_text())
if got.uns["spatial"]["tiny_mouse_brain"]["scalefactors"] != scalefactors:
    problems.append("scalefactors differ")
problems += analysis_problems(folder, got)
print("RESULT " + json.dumps(problems))
"""


def test_visium_hd_imports_as_scanpy_and_pandas_read_it(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", analysis=True)
    out = tmp_path / "hd.h5ad"
    _convert(str(hd.outs), "-o", str(out), "--var-names", "gene_symbols")
    assert _run(["scanpy", "pyarrow"], _COMPARE_VISIUM_HD, str(hd.bins[8].outs), str(out)) == []


# Space Ranger 4's segmented cells as 10x's Python tutorial reads them: the
# cell matrix, the polygons through geopandas, each cell's centroid as
# obsm["spatial"] and its area (here scaled to µm²), and the analysis CSVs.
_COMPARE_VISIUM_CELLS = r"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, geopandas as gpd

folder, h5ad = Path(sys.argv[1]), sys.argv[2]
ref = sc.read_10x_h5(folder / "filtered_feature_cell_matrix.h5")
mpp = json.loads((folder / "spatial" / "scalefactors_json.json").read_text())["microns_per_pixel"]

def polygons(name):
    frame = gpd.read_file(folder / name)
    frame.index = [f"cellid_{i:09d}-1" for i in frame["cell_id"]]
    return frame.reindex(ref.obs_names)

cells, nuclei = polygons("cell_segmentations.geojson"), polygons("nucleus_segmentations.geojson")
spatial = np.stack([cells.geometry.centroid.x, cells.geometry.centroid.y], axis=1)
clusters = pd.read_csv(folder / "analysis/clustering/gene_expression_graphclust/clusters.csv", index_col="Barcode")
umap = pd.read_csv(folder / "analysis/umap/gene_expression_2_components/projection.csv", index_col="Barcode")

got = ad.read_h5ad(h5ad)
problems = []
if list(ref.obs_names) != list(got.obs_names):
    problems.append("obs_names differ")
if not np.array_equal(ref.X.toarray(), got.X.toarray()):
    problems.append("X differs")
if not np.allclose(spatial, got.obsm["spatial"], rtol=0, atol=1e-9):
    problems.append("obsm['spatial'] differs")
if not np.allclose(cells.geometry.area * mpp ** 2, got.obs["cell_area"]):
    problems.append("cell_area differs")
if not np.allclose(nuclei.geometry.area * mpp ** 2, got.obs["nucleus_area"], equal_nan=True):
    problems.append("nucleus_area differs")
expected = clusters["Cluster"].astype(str).reindex(ref.obs_names)
if expected.fillna("").tolist() != got.obs["graphclust"].astype(object).fillna("").astype(str).tolist():
    problems.append("graphclust differs")
if not np.allclose(umap.reindex(ref.obs_names).to_numpy(), got.obsm["X_umap"], equal_nan=True):
    problems.append("obsm['X_umap'] differs")
print("RESULT " + json.dumps(problems))
"""


def test_visium_hd_cells_import_as_10x_s_geopandas_recipe_reads_them(tmp_path):
    hd = write_visium_hd_outs(tmp_path / "run", sizes=(16,), cells=True)
    out = tmp_path / "cells.h5ad"
    _convert(str(hd.outs), "-o", str(out), "--cells", "--var-names", "gene_symbols")
    assert _run(["scanpy", "geopandas"], _COMPARE_VISIUM_CELLS, str(hd.cells.folder), str(out)) == []


# ---------------------------------------------------------------------------
# scanpy's marker genes, written by scanpy, read back by scanpy after every
# copy: `rank_genes_groups` is a dict of record arrays, one field per group.

_WRITE_MARKERS = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
rng = np.random.default_rng(0)
a = ad.AnnData(rng.poisson(1.0, (40, 12)).astype("float32"))
a.var_names = [f"gene_{i}" for i in range(12)]
a.obs["group"] = pd.Categorical(["a", "bb"] * 20)
sc.pp.log1p(a)
sc.tl.rank_genes_groups(a, "group", method="wilcoxon")
a.write_h5ad(sys.argv[1] + ".h5ad")
a.write_zarr(sys.argv[1] + ".zarr")
print("RESULT", json.dumps(True))
"""

_COMPARE_MARKERS = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import scanpy as sc, anndata as ad
read = lambda p: ad.read_zarr(p) if p.endswith(".zarr") else ad.read_h5ad(p)
want = sc.get.rank_genes_groups_df(read(sys.argv[1]), group=None)
got = sc.get.rank_genes_groups_df(read(sys.argv[2]), group=None)
problems = []
if list(got.columns) != list(want.columns):
    problems.append(f"columns {list(got.columns)} != {list(want.columns)}")
else:
    for column in want.columns:
        if got[column].astype(str).tolist() != want[column].astype(str).tolist():
            problems.append(f"{column} differs")
print("RESULT", json.dumps(problems))
"""


@pytest.mark.parametrize(
    "source,target,extra",
    [
        ("h5ad", "zarr", []),
        ("h5ad", "zarr", ["--zarr-format", "2"]),
        ("zarr", "h5ad", []),
        ("h5ad", "h5ad", ["--compression", "gzip"]),
    ],
    ids=["h5ad-zarr3", "h5ad-zarr2", "zarr-h5ad", "h5ad-h5ad-gzip"],
)
def test_rank_genes_groups_survives_a_conversion_as_scanpy_reads_it(tmp_path, source, target, extra):
    """F017, F021: scanpy's marker genes crossed no backend (Zarr refused the
    HDF5 text fields, h5py the Zarr ones), and a copy with --compression
    crashed the process."""
    stem = tmp_path / "scanpy"
    _run(["scanpy"], _WRITE_MARKERS, str(stem))
    src = f"{stem}.{source}"
    out = tmp_path / f"out.{target}"
    _convert(src, "-o", str(out), *extra)
    assert _run(["scanpy"], _COMPARE_MARKERS, src, str(out)) == []

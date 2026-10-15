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
    write_10x_mtx,
    write_10x_v2_h5,
    write_starsolo_velocyto,
)
from tests.reference_stores import must_build, offline, uv_available

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

runner = CliRunner()
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _run(packages: List[str], script: str, *args: str):
    """Run `script` with `packages` installed; return what it printed after RESULT."""
    if not uv_available() or offline():
        if must_build():
            pytest.fail("uv and the network are required for these tests in CI")
        pytest.skip("needs uv and the network")
    cmd = ["uv", "run", "--no-project", "--python", PYTHON]
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
    _convert(str(h5), "-o", str(out), *extra)
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
    _convert(str(folder), "-o", str(out), *(["--make-unique"] if make_unique else []))
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
    _convert(str(folder), "-o", str(out))
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

"""Pytest configuration and fixtures for adata tests."""

import base64
import gzip
import json
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import tempfile
import pytest
import h5py
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_h5ad_file(temp_dir):
    """Create a sample h5ad file for testing."""
    file_path = temp_dir / "test.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs (observations/cells)
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Add some metadata
        obs.create_dataset(
            "cell_type",
            data=np.array(["TypeA", "TypeB", "TypeA", "TypeC", "TypeB"], dtype="S"),
        )
        obs.create_dataset(
            "n_counts", data=np.array([100, 200, 150, 300, 250], dtype=np.int32)
        )

        # Create var (variables/genes)
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2", "gene_3", "gene_4"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))
        var.create_dataset(
            "highly_variable", data=np.array([True, False, True, False], dtype=bool)
        )

        # Create dense X matrix (5 cells x 4 genes)
        X = np.array(
            [
                [1.0, 0.0, 2.5, 0.0],
                [0.0, 3.2, 0.0, 1.1],
                [2.1, 0.0, 1.8, 0.0],
                [0.0, 4.5, 0.0, 2.3],
                [1.5, 0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        f.create_dataset("X", data=X)

        # Create uns (unstructured annotations)
        uns = f.create_group("uns")
        uns.create_dataset("description", data=np.array(["Test dataset"], dtype="S"))

    return file_path


@pytest.fixture
def sample_sparse_csr_h5ad(temp_dir):
    """Create a sample h5ad file with CSR sparse matrix."""
    file_path = temp_dir / "test_csr.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2", "gene_3"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create CSR sparse matrix
        # Matrix (4x3):
        # [[1.0, 0.0, 2.0],
        #  [0.0, 0.0, 0.0],
        #  [3.0, 4.0, 0.0],
        #  [0.0, 5.0, 6.0]]
        X = f.create_group("X")
        X.attrs["encoding-type"] = "csr_matrix"
        X.attrs["shape"] = (4, 3)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        indices = np.array([0, 2, 0, 1, 1, 2], dtype=np.int32)
        indptr = np.array([0, 2, 2, 4, 6], dtype=np.int32)

        X.create_dataset("data", data=data)
        X.create_dataset("indices", data=indices)
        X.create_dataset("indptr", data=indptr)

    return file_path


@pytest.fixture
def sample_sparse_csc_h5ad(temp_dir):
    """Create a sample h5ad file with CSC sparse matrix."""
    file_path = temp_dir / "test_csc.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2", "gene_3", "gene_4"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create CSC sparse matrix (same logical matrix as CSR, transposed)
        # Matrix (3x4):
        # [[1.0, 0.0, 3.0, 0.0],
        #  [0.0, 0.0, 4.0, 5.0],
        #  [2.0, 0.0, 0.0, 6.0]]
        X = f.create_group("X")
        X.attrs["encoding-type"] = "csc_matrix"
        X.attrs["shape"] = (3, 4)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        indices = np.array([0, 2, 2, 2, 1, 1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 1, 3, 5, 7], dtype=np.int32)

        X.create_dataset("data", data=data)
        X.create_dataset("indices", data=indices)
        X.create_dataset("indptr", data=indptr)

    return file_path


@pytest.fixture
def sample_categorical_h5ad(temp_dir):
    """Create a sample h5ad file with categorical columns."""
    file_path = temp_dir / "test_categorical.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs with categorical column
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create categorical column
        cell_type_group = obs.create_group("cell_type")
        cell_type_group.attrs["encoding-type"] = "categorical"
        categories = np.array(["TypeA", "TypeB", "TypeC"], dtype="S")
        codes = np.array([0, 1, 0, 2], dtype=np.int8)
        cell_type_group.create_dataset("categories", data=categories)
        cell_type_group.create_dataset("codes", data=codes)

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create X matrix
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        f.create_dataset("X", data=X)

    return file_path


@pytest.fixture
def sample_legacy_v010_h5ad(temp_dir):
    """Create a sample h5ad file with legacy v0.1.0 categorical columns.

    In v0.1.0, categorical columns are stored as:
    - Integer code datasets with a 'categories' attribute (HDF5 object reference)
    - Categories stored in __categories/<colname> subgroup
    """
    file_path = temp_dir / "test_legacy_v010.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs with legacy categorical column
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.1.0"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create __categories subgroup (v0.1.0 convention)
        categories_group = obs.create_group("__categories")
        cell_type_cats = np.array(["TypeA", "TypeB", "TypeC"], dtype="S")
        cats_ds = categories_group.create_dataset("cell_type", data=cell_type_cats)

        # Create categorical column as integer codes with reference to categories
        codes = np.array([0, 1, 0, 2], dtype=np.int8)
        cell_type_ds = obs.create_dataset("cell_type", data=codes)
        # Store HDF5 object reference to categories
        cell_type_ds.attrs["categories"] = cats_ds.ref

        # Add a regular non-categorical column
        obs.create_dataset(
            "n_counts", data=np.array([100, 200, 150, 300], dtype=np.int32)
        )

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.1.0"
        var_names = ["gene_1", "gene_2"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create X matrix (no encoding-type for legacy)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        f.create_dataset("X", data=X)

    return file_path


# ---------------------------------------------------------------------------
# Shared builders
#
# The fixtures above are hand-rolled h5py, one per shape of store. These
# builders exist so a test can ask for a store containing a particular element
# on a particular backend without repeating that construction, and so the same
# test body can run against HDF5, Zarr v2 and Zarr v3.

from adata.elements import write as ew
from adata.storage import open_store


BACKENDS = ["h5ad", "zarr2", "zarr3"]


def backend_path(tmp_dir: Path, backend: str, stem: str = "store") -> Path:
    """Path with the extension that makes `detect_backend` pick `backend`."""
    return tmp_dir / (f"{stem}.h5ad" if backend == "h5ad" else f"{stem}.zarr")


def backend_zarr_format(backend: str) -> Optional[int]:
    return {"zarr2": 2, "zarr3": 3}.get(backend)


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    """Run a test once per storage backend, including both Zarr versions."""
    return request.param


@contextmanager
def open_new(path: Path, backend: str):
    """Open a fresh store for writing on the given backend."""
    with open_store(path, "w", zarr_format=backend_zarr_format(backend)) as store:
        yield store.root


def make_skeleton(
    root: Any,
    obs_names: Sequence[str] = ("c1", "c2", "c3"),
    var_names: Sequence[str] = ("g1", "g2"),
) -> None:
    """Write the obs/var frames and mapping groups every store needs."""
    ew.write_dataframe_header(root, "obs", list(obs_names), [])
    ew.write_dataframe_header(root, "var", list(var_names), [])
    ew.ensure_anndata_skeleton(root)


@pytest.fixture
def new_store(temp_dir, backend):
    """Factory returning ``(path, opener)`` for a store on the current backend.

    The opener is a context manager so a test can write, close, and reopen --
    which matters on Zarr, where the consolidated metadata index is only
    rewritten on close.
    """

    def _make(stem: str = "store"):
        path = backend_path(temp_dir, backend, stem)

        @contextmanager
        def _open(mode: str = "a"):
            with open_store(
                path, mode, zarr_format=backend_zarr_format(backend)
            ) as store:
                yield store.root

        with open_new(path, backend) as root:
            make_skeleton(root)
        return path, _open

    return _make


# ---------------------------------------------------------------------------
# 10x Genomics matrices
#
# Built the way Cell Ranger writes them: fixed-width byte strings, gzip,
# int32 counts with int64 indices, and the run's attributes at the root.


def _fixed(values: Sequence[str]) -> np.ndarray:
    width = max([1, *(len(v.encode()) for v in values)])
    return np.array([v.encode() for v in values], dtype=f"S{width}")


def _write_10x_arrays(group: Any, counts: np.ndarray, barcodes: Sequence[str]) -> None:
    """`counts` is cells x features; stored as features x cells CSC."""
    rows, cols = np.nonzero(counts)
    gz = {"compression": "gzip", "compression_opts": 4}
    group.create_dataset("data", data=counts[rows, cols].astype(np.int32), **gz)
    group.create_dataset("indices", data=cols.astype(np.int64), **gz)
    indptr = np.concatenate(([0], np.cumsum(np.count_nonzero(counts, axis=1))))
    group.create_dataset("indptr", data=indptr.astype(np.int64), **gz)
    group.create_dataset(
        "shape", data=np.array([counts.shape[1], counts.shape[0]], dtype=np.int32)
    )
    group.create_dataset("barcodes", data=_fixed(barcodes), **gz)


def write_10x_h5(
    path: Path,
    counts: np.ndarray,
    *,
    barcodes: Sequence[str],
    ids: Sequence[str],
    names: Sequence[str],
    types: Optional[Sequence[str]] = None,
    genomes: Optional[Sequence[str]] = None,
    extra: Optional[dict] = None,
    probe_gene_ids: Optional[Sequence[str]] = None,
    filtered_barcodes: Optional[Sequence[bool]] = None,
) -> Path:
    """A Cell Ranger 3+ `*_feature_bc_matrix.h5`."""
    n = counts.shape[1]
    with h5py.File(path, "w") as f:
        # As Cell Ranger writes them: a single string variable-length, an
        # array of them fixed-width.
        f.attrs["chemistry_description"] = "Single Cell 3' v3"
        f.attrs["filetype"] = "matrix"
        f.attrs["library_ids"] = np.array([b"lib1"])
        f.attrs["original_gem_groups"] = np.array([1], dtype=np.int64)
        f.attrs["software_version"] = "cellranger-8.0.1"
        f.attrs["version"] = np.int64(2)
        group = f.create_group("matrix")
        _write_10x_arrays(group, counts, barcodes)
        features = group.create_group("features")
        features.create_dataset("id", data=_fixed(ids))
        features.create_dataset("name", data=_fixed(names))
        features.create_dataset(
            "feature_type", data=_fixed(types or ["Gene Expression"] * n)
        )
        features.create_dataset("genome", data=_fixed(genomes or ["GRCh38"] * n))
        tag_keys = ["genome"]
        for key, values in (extra or {}).items():
            values = list(values)
            if isinstance(values[0], (bool, np.bool_)):
                features.create_dataset(key, data=np.array(values, dtype=bool))
            else:
                features.create_dataset(key, data=_fixed(values))
            tag_keys.append(key)
        if probe_gene_ids is not None:
            features.create_dataset("gene_id", data=_fixed(probe_gene_ids))
        features.create_dataset("_all_tag_keys", data=_fixed(tag_keys))
        if filtered_barcodes is not None:
            group.create_dataset(
                "filtered_barcodes", data=np.array(filtered_barcodes, dtype=bool)
            )
    return path


def write_10x_v2_h5(path: Path, genomes: dict) -> Path:
    """A Cell Ranger 2 file: `{genome: (counts, barcodes, ids, names)}`."""
    with h5py.File(path, "w") as f:
        # PyTables bookkeeping, which a v3 export must not carry over.
        f.attrs["PYTABLES_FORMAT_VERSION"] = np.bytes_(b"2.1")
        f.attrs["TITLE"] = np.bytes_(b"")
        for genome, (counts, barcodes, ids, names) in genomes.items():
            group = f.create_group(genome)
            _write_10x_arrays(group, counts, barcodes)
            group.create_dataset("genes", data=_fixed(ids))
            group.create_dataset("gene_names", data=_fixed(names))
    return path


# ---------------------------------------------------------------------------
# 10x / STARsolo Matrix Market folders
#
# Text written by hand, not by our writer, so a bug shared by the two cannot
# hide: genes x cells, 1-based, entries in barcode order unless asked
# otherwise.


def write_mtx_file(
    path: Path,
    counts: np.ndarray,
    *,
    gz: bool = False,
    order: str = "cell",
    field: str = "integer",
    duplicates: int = 0,
    comments: Sequence[str] = (),
    seed: int = 0,
) -> Path:
    """`counts` is cells x features; written as features x cells."""
    import gzip

    cells, features = np.nonzero(counts)
    values = counts[cells, features]
    entries = list(zip(features + 1, cells + 1, values))
    if order == "gene":
        entries.sort(key=lambda e: (e[0], e[1]))
    elif order == "shuffled":
        np.random.default_rng(seed).shuffle(entries)
    if duplicates:
        # Split values in two: the file sums back to `counts`.
        split = []
        for i, (g, c, v) in enumerate(entries):
            if i < duplicates and v > 1:
                split += [(g, c, v - 1), (g, c, 1)]
            else:
                split.append((g, c, v))
        entries = split
    fmt = (lambda v: f"{float(v)!r}") if field == "real" else (lambda v: str(int(v)))
    lines = [f"%%MatrixMarket matrix coordinate {field} general", *comments]
    lines.append(f"{counts.shape[1]} {counts.shape[0]} {len(entries)}")
    lines += [f"{g} {c} {fmt(v)}" for g, c, v in entries]
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(gzip.compress(data) if gz else data)
    return path


def _write_text(path: Path, lines: Sequence[str], gz: bool) -> Path:
    import gzip

    data = ("".join(f"{line}\n" for line in lines)).encode()
    path.write_bytes(gzip.compress(data) if gz else data)
    return path


def write_10x_mtx(
    folder: Path,
    counts: np.ndarray,
    *,
    barcodes: Sequence[str],
    ids: Sequence[str],
    names: Sequence[str],
    types: Optional[Sequence[str]] = None,
    layout: str = "cr3",
    prefix: str = "",
    **matrix: Any,
) -> Path:
    """A matrix folder in one of the layouts Cell Ranger and STARsolo write.

    `cr3`: gzipped, features.tsv with id, name, type. `cr2`: plain,
    genes.tsv with id and name. `starsolo`: plain, three columns.
    `starsolo2col`: plain, two columns, as older STARsolo wrote.
    """
    gz = layout == "cr3"
    ext = ".gz" if gz else ""
    folder.mkdir(parents=True, exist_ok=True)
    if layout in ("cr3", "starsolo"):
        kinds = types or ["Gene Expression"] * len(ids)
        rows = [f"{i}\t{n}\t{t}" for i, n, t in zip(ids, names, kinds)]
        _write_text(folder / f"{prefix}features.tsv{ext}", rows, gz)
    else:
        rows = [f"{i}\t{n}" for i, n in zip(ids, names)]
        name = "genes.tsv" if layout == "cr2" else "features.tsv"
        _write_text(folder / f"{prefix}{name}{ext}", rows, gz)
    _write_text(folder / f"{prefix}barcodes.tsv{ext}", barcodes, gz)
    write_mtx_file(folder / f"{prefix}matrix.mtx{ext}", counts, gz=gz, **matrix)
    return folder


def write_starsolo_velocyto(
    folder: Path,
    layers: Dict[str, np.ndarray],
    *,
    barcodes: Sequence[str],
    ids: Sequence[str],
    names: Sequence[str],
) -> Path:
    """STARsolo's Velocyto/raw: one .mtx per layer over shared barcodes and features."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = [f"{i}\t{n}\tGene Expression" for i, n in zip(ids, names)]
    _write_text(folder / "features.tsv", rows, False)
    _write_text(folder / "barcodes.tsv", barcodes, False)
    for name, counts in layers.items():
        write_mtx_file(folder / f"{name}.mtx", counts)
    return folder


# ---------------------------------------------------------------------------
# CellBender remove-background output
#
# As CellBender writes it (checked against its source at 0.2.2, 0.3.2 and
# 0.4.0): a 10x matrix of the denoised counts plus latent variables. 0.3 and
# later keep the latents in their own groups; 0.2 keeps them inside the
# matrix group. Analysed droplets are ordered by descending UMI count, so a
# filtered file's barcodes are not in the full file's order.


@dataclass
class CellbenderTruth:
    """What a fixture holds, for tests to check the import against."""

    path: Path
    rows: np.ndarray  # which of the full barcodes the file holds, in order
    analyzed: np.ndarray  # barcode indices of the analysed droplets
    latents: Dict[str, np.ndarray]  # per analysed droplet, by stored name
    encoding: np.ndarray
    ambient: np.ndarray


def write_cellbender_h5(
    path: Path,
    counts: np.ndarray,
    *,
    barcodes: Sequence[str],
    ids: Sequence[str],
    names: Sequence[str],
    types: Optional[Sequence[str]] = None,
    version: str = "0.3",
    filtered: bool = False,
    n_analyzed: Optional[int] = None,
    z_dim: int = 4,
    cellranger_v2: bool = False,
    seed: int = 0,
) -> CellbenderTruth:
    """`counts` is every droplet x features; `filtered` writes cells only."""
    rng = np.random.default_rng(seed)
    n_all, n_features = counts.shape
    n_analyzed = n_analyzed or max(2, n_all // 2)
    analyzed = np.argsort(counts.sum(axis=1), kind="stable")[::-1][:n_analyzed]
    p = rng.uniform(0, 1, n_analyzed)
    p[0], p[-1] = 0.99, 0.01  # at least one cell and one empty droplet
    latents = {
        "cell_probability": p,
        "cell_size": rng.uniform(100, 1000, n_analyzed),
        "droplet_efficiency": rng.uniform(0.5, 1.5, n_analyzed),
        "background_fraction": rng.uniform(0, 0.5, n_analyzed),
    }
    if version == "0.2":
        latents = {
            "latent_cell_probability": latents["cell_probability"],
            "latent_scale": latents["cell_size"],
            "latent_RT_efficiency": latents["droplet_efficiency"],
        }
    encoding = rng.normal(size=(n_analyzed, z_dim)).astype(np.float32)
    ambient = rng.dirichlet(np.ones(n_features))

    if filtered:
        keep = p > 0.5
        rows = analyzed[keep]
        latents = {k: v[keep] for k, v in latents.items()}
        encoding = encoding[keep]
    else:
        rows = np.arange(n_all)

    file_barcodes = [barcodes[i] for i in rows]
    file_counts = counts[rows]
    if cellranger_v2:
        with h5py.File(path, "w") as f:
            group = f.create_group("background_removed")
            _write_10x_arrays(group, file_counts, file_barcodes)
            group.create_dataset("genes", data=_fixed(ids))
            group.create_dataset("gene_names", data=_fixed(names))
            group.create_dataset("genome", data=_fixed(["NA"] * n_features))
    else:
        write_10x_h5(
            path, file_counts, barcodes=file_barcodes, ids=ids, names=names, types=types
        )

    with h5py.File(path, "a") as f:
        if version == "0.2":
            group = f["background_removed" if cellranger_v2 else "matrix"]
            if not filtered:
                group.create_dataset("barcode_indices_for_latents", data=analyzed)
            group.create_dataset("latent_gene_encoding", data=encoding)
            for key, values in latents.items():
                group.create_dataset(key, data=values)
            group.create_dataset("ambient_expression", data=ambient)
            group.create_dataset("target_false_positive_rate", data=0.01)
            group.create_dataset("lambda_multiplier", data=0.5)
            group.create_dataset("training_elbo_per_epoch", data=rng.normal(size=20))
            group.create_dataset("overdispersion_mean_and_scale", data=[0.1, 0.2])
        else:
            droplets = f.create_group("droplet_latents")
            # In 0.3+ every analysed droplet, even in the filtered file.
            droplets.create_dataset("barcode_indices_for_latents", data=analyzed)
            droplets.create_dataset("gene_expression_encoding", data=encoding)
            for key, values in latents.items():
                droplets.create_dataset(key, data=values)
            glob = f.create_group("global_latents")
            glob.create_dataset("ambient_expression", data=ambient)
            glob.create_dataset("empty_droplet_size_lognormal_loc", data=5.5)
            glob.create_dataset("swapping_fraction_dist_params", data=[1.0, 20.0])
            meta = f.create_group("metadata")
            meta.create_dataset("learning_curve_train_elbo", data=rng.normal(size=20))
            meta.create_dataset("barcodes_analyzed", data=_fixed([barcodes[i] for i in analyzed]))
            meta.create_dataset("barcodes_analyzed_inds", data=analyzed)
            meta.create_dataset("features_analyzed_inds", data=np.arange(n_features))
            meta.create_dataset("target_false_positive_rate", data=[0.01])
            meta.create_dataset("estimator", data=_fixed(["mckp"]))

    return CellbenderTruth(path, rows, analyzed, latents, encoding, ambient)


# ---------------------------------------------------------------------------
# Parse Biosciences split-pipe output
#
# As split-pipe v1.6.1 writes it (checked on a real run): per sample, and for
# the combined `all-sample`, DGE_filtered/ and DGE_unfiltered/ each hold a
# cells x genes count_matrix.mtx, all_genes.csv with quoted gene names, and
# cell_metadata.csv starting with bc_wells.

PARSE_METADATA = (
    "bc_wells", "sample", "species", "gene_count", "tscp_count", "mread_count",
    "bc1_wind", "bc2_wind", "bc3_wind", "bc1_well", "bc2_well", "bc3_well",
)


@dataclass
class ParseDge:
    """What one DGE folder holds."""

    path: Path
    counts: np.ndarray
    barcodes: List[str]
    samples: List[str]
    species: List[str]


@dataclass
class ParseRun:
    root: Path
    ids: List[str]
    names: List[str]
    genomes: List[str]
    dges: Dict[Tuple[str, str], ParseDge]

    def dge(self, sample: str = "all-sample", which: str = "DGE_filtered") -> ParseDge:
        return self.dges[(sample, which)]


def write_parse_dge(
    folder: Path,
    counts: np.ndarray,
    *,
    barcodes: Sequence[str],
    samples: Sequence[str],
    species: Sequence[str],
    ids: Sequence[str],
    names: Sequence[str],
    genomes: Sequence[str],
    gz: bool = False,
) -> Path:
    import gzip

    folder.mkdir(parents=True, exist_ok=True)
    cells, genes = np.nonzero(counts)  # row-major: in cell order, as split-pipe writes
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        f"%Rows=cells ({counts.shape[0]}), Cols=genes ({counts.shape[1]})",
        f"{counts.shape[0]} {counts.shape[1]} {cells.size}",
    ]
    lines += [f"{c + 1} {g + 1} {int(counts[c, g])}" for c, g in zip(cells, genes)]
    data = ("\n".join(lines) + "\n").encode()
    name = "count_matrix.mtx.gz" if gz else "count_matrix.mtx"
    (folder / name).write_bytes(gzip.compress(data) if gz else data)

    gene_lines = ["gene_id,gene_name,genome"]
    gene_lines += [f'{i},"{n}",{g}' for i, n, g in zip(ids, names, genomes)]
    (folder / "all_genes.csv").write_text("\n".join(gene_lines) + "\n")

    wells = [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]
    meta = [",".join(PARSE_METADATA)]
    for k, (barcode, sample, sp) in enumerate(zip(barcodes, samples, species)):
        winds = [int(part) for part in barcode.split("_")]
        row = counts[k]
        meta.append(",".join(map(str, [
            barcode, sample, sp, int((row > 0).sum()), int(row.sum()), int(row.sum()) + k,
            *winds, *(wells[(w - 1) % 96] for w in winds),
        ])))
    (folder / "cell_metadata.csv").write_text("\n".join(meta) + "\n")
    return folder


def write_parse_run(
    root: Path,
    *,
    n_cells: int = 12,
    n_genes: int = 6,
    samples: Sequence[str] = ("s1", "s2"),
    genomes: Sequence[str] = ("GRCh38",),
    combined: str = "all-sample",
    gz: bool = False,
    seed: int = 0,
) -> ParseRun:
    """A split-pipe run: the combined folder and one per sample, each with
    DGE_filtered/ and DGE_unfiltered/ (two more, near-empty barcodes).

    Genes and cells take the genomes in turn, so a barnyard run's cells
    each belong to one species.
    """
    rng = np.random.default_rng(seed)
    n_all = n_cells + 2
    counts = rng.poisson(2.0, (n_all, n_genes))
    counts[n_cells:] = 0
    counts[n_cells:, 0] = 1
    barcodes = [f"{1 + i // 4:02d}_{1 + i % 4:02d}_{1 + (i * 7) % 96:02d}" for i in range(n_all)]
    cell_samples = [samples[i % len(samples)] if samples else combined for i in range(n_all)]
    cell_species = [genomes[i % len(genomes)] for i in range(n_all)]
    ids = [f"{genomes[g % len(genomes)]}_ENSG{g:05d}" for g in range(n_genes)]
    names = [f"{genomes[g % len(genomes)]}_G{g % 4}" for g in range(n_genes)]  # some repeat
    gene_genomes = [genomes[g % len(genomes)] for g in range(n_genes)]

    dges = {}
    for sample in [combined, *samples]:
        for which, n in (("DGE_filtered", n_cells), ("DGE_unfiltered", n_all)):
            keep = [i for i in range(n) if sample == combined or cell_samples[i] == sample]
            folder = root / sample / which
            dge = ParseDge(
                folder, counts[keep], [barcodes[i] for i in keep],
                [cell_samples[i] for i in keep], [cell_species[i] for i in keep],
            )
            write_parse_dge(
                folder, dge.counts, barcodes=dge.barcodes, samples=dge.samples,
                species=dge.species, ids=ids, names=names, genomes=gene_genomes, gz=gz,
            )
            dges[(sample, which)] = dge
    return ParseRun(root, ids, names, gene_genomes, dges)


# ---------------------------------------------------------------------------
# Loom, as loompy 3 and velocyto write it (checked on velocyto 0.17 and
# kallisto|bustools looms): genes x cells, chunked 64 x 64 and gzipped;
# strings as variable-length ASCII with XML character references; booleans
# as uint8; spec 3.0.0 keeps global attributes as datasets under /attrs,
# spec 2.0.1 as root attributes.


@dataclass
class LoomTruth:
    path: Path
    counts: np.ndarray  # cells x genes, the main matrix
    layers: Dict[str, np.ndarray]
    cell_ids: List[str]
    gene_ids: List[str]
    gene_names: List[str]


def _loom_strings(values: Sequence[str]) -> np.ndarray:
    return np.array([str(v).encode("ascii", "xmlcharrefreplace") for v in values], dtype=object)


def write_loom(
    path: Path,
    *,
    n_cells: int = 30,
    n_genes: int = 10,
    spec: str = "3.0.0",
    velocyto: bool = False,
    graph: bool = True,
    seed: int = 0,
) -> LoomTruth:
    rng = np.random.default_rng(seed)
    ascii_str = h5py.string_dtype(encoding="ascii")
    counts = rng.poisson(0.7, (n_cells, n_genes)).astype(np.float32)
    layers: Dict[str, np.ndarray] = {}
    if velocyto:
        for name in ("spliced", "unspliced", "ambiguous"):
            layers[name] = rng.poisson(0.5, (n_cells, n_genes)).astype(np.uint32)
    else:
        layers["counts"] = counts.astype(np.int32) * 2
    cell_ids = [f"sample:CELL{i:05d}x" for i in range(n_cells)]
    gene_ids = [f"ENSMUSG{i:011d}" for i in range(n_genes)]
    gene_names = [f"Gène{i % (n_genes - 1)}" for i in range(n_genes)]  # one repeat, one accent

    def chunks(shape):
        return (min(64, shape[0]), min(64, shape[1]))

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "matrix", data=counts.T, chunks=chunks(counts.T.shape),
            compression="gzip", compression_opts=2,
        )
        group = f.create_group("layers")
        for name, values in layers.items():
            group.create_dataset(
                name, data=values.T, chunks=chunks(values.T.shape),
                compression="gzip", compression_opts=2,
            )
        col = f.create_group("col_attrs")
        col.create_dataset("CellID", data=_loom_strings(cell_ids), dtype=ascii_str)
        col.create_dataset("Clusters", data=np.arange(n_cells) % 3)
        col.create_dataset("Tissue", data=_loom_strings(["cortex", "hippocampus"] * (n_cells // 2) + ["cortex"] * (n_cells % 2)), dtype=ascii_str)
        col.create_dataset("Passed", data=(np.arange(n_cells) % 2).astype(np.uint8))
        col.create_dataset("_X", data=rng.normal(size=n_cells).astype(np.float32))
        col.create_dataset("embedding", data=rng.normal(size=(n_cells, 2)))
        row = f.create_group("row_attrs")
        row.create_dataset("Gene", data=_loom_strings(gene_names), dtype=ascii_str)
        row.create_dataset("Accession", data=_loom_strings(gene_ids), dtype=ascii_str)
        row.create_dataset("Start", data=np.arange(n_genes) * 100)
        if velocyto:
            row.create_dataset("Chromosome", data=_loom_strings(["chr1"] * n_genes), dtype=ascii_str)
        cg, rgraphs = f.create_group("col_graphs"), f.create_group("row_graphs")
        if graph:
            g = cg.create_group("knn")
            a = np.repeat(np.arange(n_cells), 2)
            b = (a + np.tile([1, 2], n_cells)) % n_cells
            order = rng.permutation(a.size)  # COO need not be sorted
            g.create_dataset("a", data=a[order])
            g.create_dataset("b", data=b[order])
            g.create_dataset("w", data=np.linspace(0.1, 1, a.size)[order].astype(np.float32))
        del rgraphs
        global_attrs = {"LOOM_SPEC_VERSION": spec, "title": "Démo"}
        if velocyto:
            global_attrs["velocyto.__version__"] = "0.17.17"
        if spec.startswith("3"):
            attrs = f.create_group("attrs")
            for key, value in global_attrs.items():
                attrs.create_dataset(key, data=str(value).encode("ascii", "xmlcharrefreplace"), dtype=ascii_str)
        else:
            for key, value in global_attrs.items():
                f.attrs[key] = str(value).encode("ascii", "xmlcharrefreplace")
    return LoomTruth(path, counts, layers, cell_ids, gene_ids, gene_names)


# ---------------------------------------------------------------------------
# SCope looms, as pySCENIC's export2loom writes them (checked against a public
# one, Zenodo 10944066): a loompy loom plus compound attributes with a field
# per regulon (RegulonsAUC per cell, float; Regulons per gene, 0/1), per
# embedding id (Embeddings_X and _Y; Embedding holds the default, id -1,
# again) and per clustering id (Clusterings), and a MetaData global
# attribute: JSON, compressed as base64 of zlib. SCENIC+ and pycistopic
# write the same, with regions instead of genes in region-based looms.


@dataclass
class ScopeTruth:
    loom: LoomTruth
    regulons: List[str]
    auc: np.ndarray  # cells x regulons
    membership: np.ndarray  # genes (or regions) x regulons
    embeddings: Dict[str, np.ndarray]  # id -> cells x 2
    clusterings: Dict[str, np.ndarray]  # id -> each cell's cluster id
    metadata: Dict[str, Any]
    var_names: List[str]

    @property
    def path(self) -> Path:
        return self.loom.path


SCOPE_REGULONS = ("ATF3_(+)", "Pax6_(-)", "SOX2 (12g)")


def _records(columns: Dict[str, np.ndarray]) -> np.ndarray:
    """A structured array, a field per column, as export2loom builds them."""
    first = next(iter(columns.values()))
    out = np.zeros(len(first), dtype=[(name, values.dtype) for name, values in columns.items()])
    for name, values in columns.items():
        out[name] = values
    return out


def write_scope_loom(
    path: Path,
    *,
    n_cells: int = 30,
    n_genes: int = 10,
    compressed: bool = True,
    region: bool = False,
    default_in_extra: bool = True,
    regulons: Sequence[str] = SCOPE_REGULONS,
    seed: int = 0,
) -> ScopeTruth:
    """A pySCENIC-style SCope loom on top of `write_loom`'s.

    Embeddings_X and _Y hold ids -1 (float64) and 0 (float32), as export2loom
    writes them; with `default_in_extra=False` they hold 0 alone, and the
    default embedding is only in `Embedding`, as in the Zenodo loom. Two
    clusterings, the second with a cluster that has no description.
    """
    loom = write_loom(path, n_cells=n_cells, n_genes=n_genes, graph=False, seed=seed)
    rng = np.random.default_rng(seed + 1)
    regulons = list(regulons)
    auc = rng.random((n_cells, len(regulons)))
    membership = (rng.random((n_genes, len(regulons))) < 0.4).astype(np.int64)
    embeddings = {
        "-1": rng.normal(size=(n_cells, 2)),
        "0": rng.normal(size=(n_cells, 2)).astype(np.float32),
    }
    clusterings = {"0": np.arange(n_cells) % 3, "1": (np.arange(n_cells) // 4) % 2}
    extra = embeddings if default_in_extra else {"0": embeddings["0"]}
    metadata = {
        "embeddings": [{"id": -1, "name": "tSNE (default)"}, {"id": 0, "name": "UMAP"}],
        "annotations": [{"name": "", "values": []}],
        "clusterings": [
            {
                "id": 0, "group": "celltype", "name": "Cell Type",
                "clusters": [
                    {"id": 0, "description": "B cell"},
                    {"id": 1, "description": "T cell"},
                    {"id": 2, "description": "NK"},
                ],
            },
            {
                "id": 1, "group": "Scanpy", "name": "Leiden resolution 0.4",
                "clusters": [{"id": 1}, {"id": 0, "description": "Leiden 0"}],
            },
        ],
        "metrics": [{"name": "nGene"}],
        "regulonThresholds": [
            {
                "regulon": name,
                "defaultThresholdValue": 0.1 * (i + 1),
                "defaultThresholdName": "gaussian_mixture_split",
                "allThresholds": {"gaussian_mixture_split": 0.1 * (i + 1)},
                "motifData": f"motif{i}.png",
            }
            for i, name in enumerate(regulons)
        ],
    }
    text = json.dumps(metadata)
    if compressed:
        text = base64.b64encode(zlib.compress(text.encode("ascii"))).decode("ascii")
    ascii_str = h5py.string_dtype(encoding="ascii")
    var_names = loom.gene_ids
    with h5py.File(path, "a") as f:
        col, row = f["col_attrs"], f["row_attrs"]
        col.create_dataset("RegulonsAUC", data=_records(dict(zip(regulons, auc.T))))
        col.create_dataset("Embedding", data=_records({"_X": embeddings["-1"][:, 0], "_Y": embeddings["-1"][:, 1]}))
        for axis, name in ((0, "Embeddings_X"), (1, "Embeddings_Y")):
            col.create_dataset(name, data=_records({id: e[:, axis] for id, e in extra.items()}))
        col.create_dataset("Clusterings", data=_records(clusterings))
        col.create_dataset("ClusterID", data=clusterings["0"])
        row.create_dataset("Regulons", data=_records(dict(zip(regulons, membership.T))))
        if region:
            var_names = [f"chr{1 + i % 2}:{1000 * i}-{1000 * i + 500}" for i in range(n_genes)]
            del row["Gene"], row["Accession"]
            row.create_dataset("Gene", data=_loom_strings(var_names), dtype=ascii_str)
            linked = [f"Gene{i // 2}" if i % 3 else "" for i in range(n_genes)]
            row.create_dataset("linkedGene", data=_loom_strings(linked), dtype=ascii_str)
        attrs = f["attrs"]
        for key, value in (
            ("MetaData", text), ("Genome", "hg38"), ("SCopeTreeL1", "Tissue"),
            ("SCopeTreeL2", ""), ("SCopeTreeL3", ""), ("CreationDate", "20240407T203321Z"),
        ):
            attrs.create_dataset(key, data=value.encode("ascii"), dtype=ascii_str)
    return ScopeTruth(
        loom, regulons, auc, membership, embeddings, clusterings, metadata, var_names
    )


# ---------------------------------------------------------------------------
# Text matrices, in the layouts anndata's read_csv / read_text tell apart:
# a header with a corner cell over the row names (pandas' to_csv), a header
# one field short (R's write.table), column names in the last comment line,
# or no header at all.


@dataclass
class TextTruth:
    path: Path
    counts: np.ndarray  # cells x genes
    cells: List[str]
    genes: List[str]


def write_text_matrix(
    path: Path,
    counts: Optional[np.ndarray] = None,
    *,
    n_cells: int = 12,
    n_genes: int = 5,
    delimiter: Optional[str] = ",",
    header: str = "corner",
    row_names: bool = True,
    transpose: bool = False,
    gz: bool = False,
    quote: bool = False,
    seed: int = 0,
) -> TextTruth:
    """`counts` (cells x genes) as text; genes as rows with `transpose`.

    `delimiter=None` separates fields with runs of spaces, as read_text
    reads them. `header` is "corner", "short", "comment" or "none".
    """
    import gzip

    if counts is None:
        counts = np.random.default_rng(seed).poisson(0.8, (n_cells, n_genes))
    cells = [f"cell{i:03d}" for i in range(counts.shape[0])]
    genes = [f"Gene{j}" for j in range(counts.shape[1])]
    table, rows, columns = (counts.T, genes, cells) if transpose else (counts, cells, genes)
    sep = "  " if delimiter is None else delimiter
    named = (lambda s: f'"{s}"') if quote else (lambda s: s)

    def value(v) -> str:
        return str(int(v)) if float(v).is_integer() else repr(float(v))

    lines = []
    if header == "comment":
        lines += ["# written by a test", "# " + " ".join(columns)]
    elif header == "corner":
        lines.append(sep.join([named("") if quote else "", *map(named, columns)]))
    elif header == "short":
        lines.append(sep.join(map(named, columns)))
    for name, row in zip(rows, table):
        fields = [value(v) for v in row]
        lines.append(sep.join(([named(name)] if row_names else []) + fields))
    data = ("\n".join(lines) + "\n").encode()
    if gz:
        data = gzip.compress(data)
    path.write_bytes(data)
    return TextTruth(path, np.asarray(counts), cells, genes)


# ---------------------------------------------------------------------------
# A plain HDF5 matrix, as anndata's read_hdf reads one: a 2-D dataset, with
# its rows' and columns' names in root datasets `row_names` and `col_names`.


@dataclass
class Hdf5Truth:
    path: Path
    key: str
    counts: np.ndarray  # cells x genes
    cells: List[str]
    genes: List[str]


def write_hdf5_matrix(
    path: Path,
    counts: Optional[np.ndarray] = None,
    *,
    key: str = "data/matrix",
    n_cells: int = 12,
    n_genes: int = 5,
    transpose: bool = False,
    chunks: Optional[Tuple[int, int]] = None,
    names: str = "root",
    seed: int = 0,
) -> Hdf5Truth:
    """`counts` (cells x genes) at `key`; genes as rows with `transpose`.

    `names` is "root" (read_hdf's row_names / col_names), "custom" (at
    `names/cells` and `names/genes`) or "none".
    """
    if counts is None:
        counts = np.random.default_rng(seed).poisson(0.8, (n_cells, n_genes)).astype(np.int32)
    cells = [f"cell{i:03d}" for i in range(counts.shape[0])]
    genes = [f"Gene{j}" for j in range(counts.shape[1])]
    stored = counts.T if transpose else counts
    rows, cols = (genes, cells) if transpose else (cells, genes)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=stored, chunks=chunks, compression="gzip" if chunks else None)
        if names == "root":
            f["row_names"] = np.array(rows, dtype="S")
            f["col_names"] = np.array(cols, dtype="S")
        elif names == "custom":
            f["names/cells"] = np.array(cells, dtype=object).astype(h5py.string_dtype())
            f["names/genes"] = np.array(genes, dtype=object).astype(h5py.string_dtype())
    return Hdf5Truth(path, key, np.asarray(counts), cells, genes)


# ---------------------------------------------------------------------------
# pycisTopic pickles, as pycisTopic writes them: `pickle.dump` of its own
# classes holding pandas frames and scipy matrices. pycisTopic is not
# installed here, so stand-ins are made under its module names, which is all
# a pickle records of a class.


@dataclass
class CistopicTruth:
    path: Path
    obj: Any


def _pycistopic_classes() -> Dict[str, type]:
    import sys
    import types

    classes = {}
    package = sys.modules.setdefault("pycisTopic", types.ModuleType("pycisTopic"))
    package.__path__ = []  # a package, so its submodules import
    for module, name in (
        ("pycisTopic.cistopic_class", "CistopicObject"),
        ("pycisTopic.lda_models", "CistopicLDAModel"),
        ("pycisTopic.diff_features", "CistopicImputedFeatures"),
    ):
        mod = sys.modules.setdefault(module, types.ModuleType(module))
        setattr(package, module.split(".")[1], mod)
        cls = getattr(mod, name, None)
        if cls is None:
            cls = type(name, (), {"__module__": module})
            setattr(mod, name, cls)
        classes[name] = cls
    return classes


def write_cistopic_pickle(
    path: Path,
    *,
    n_cells: int = 20,
    n_regions: int = 30,
    n_topics: int = 4,
    model: bool = True,
    projections: bool = True,
    harmony: bool = False,
    shuffled: bool = True,
    gz: bool = False,
    seed: int = 0,
) -> CistopicTruth:
    """A CistopicObject: fragment and binary matrices (regions x cells),
    cell_data with numeric, Categorical, nullable and object columns,
    region_data, a selected LDA model, and a UMAP. `shuffled` puts
    cell_data and cell_topic in another order than cell_names, as
    add_cell_data can leave them."""
    import gzip
    import pickle

    import pandas as pd
    import scipy.sparse as sp

    classes = _pycistopic_classes()
    rng = np.random.default_rng(seed)
    cells = [f"AAAC{i:04d}-1___sample{i % 2}" for i in range(n_cells)]
    regions = [f"chr{1 + i % 3}:{1000 * i}-{1000 * i + 500}" for i in range(n_regions)]
    counts = rng.poisson(0.6, (n_regions, n_cells)).astype(np.int32)
    obj = classes["CistopicObject"]()
    obj.fragment_matrix = sp.csr_matrix(counts)
    obj.binary_matrix = sp.csr_matrix((counts > 0).astype(np.int32))
    obj.cell_names, obj.region_names = list(cells), list(regions)
    order = rng.permutation(n_cells) if shuffled else np.arange(n_cells)
    cell_data = pd.DataFrame(
        {
            "cisTopic_nr_frag": np.asarray(counts.sum(0), dtype=object),  # numbers held as object
            "sample_id": pd.Categorical([f"sample{i % 2}" for i in range(n_cells)]),
            "TSS_enrichment": rng.random(n_cells),
            "doublet": pd.array([True, False, None] * (n_cells // 3) + [True] * (n_cells % 3), dtype="boolean"),
            "n_peaks": pd.array(list(range(n_cells - 1)) + [None], dtype="Int64"),
            "celltype": [f"type{i % 3}" for i in range(n_cells)],
            "note": [f"cell {i}" for i in range(n_cells)],
        },
        index=cells,
    ).iloc[order]
    obj.cell_data = cell_data
    obj.region_data = pd.DataFrame(
        {
            "Chromosome": [r.split(":")[0] for r in regions],
            "Start": np.array([1000 * i for i in range(n_regions)], dtype=np.int32),
            "End": np.array([1000 * i + 500 for i in range(n_regions)], dtype=np.int32),
        },
        index=regions,
    )
    obj.project = "demo"
    obj.path_to_fragments = {"sample0": "/data/s0/fragments.tsv.gz", "sample1": "/data/s1/fragments.tsv.gz"}
    obj.selected_model = []
    obj.projections = {"cell": {}, "region": {}}
    topics = [f"Topic{t + 1}" for t in range(n_topics)]
    cell_topic = rng.random((n_topics, n_cells))
    region_topic = rng.random((n_regions, n_topics))
    if model:
        m = classes["CistopicLDAModel"]()
        m.metrics = pd.DataFrame([[0.1, 0.2, -1.3, -4e6]], index=["Metric"],
                                 columns=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"])
        m.coherence = pd.DataFrame({"Topic": range(1, n_topics + 1), "Mimno_2011": rng.random(n_topics)})
        m.marg_topic = pd.DataFrame({"Topic": range(1, n_topics + 1), "Marg_Topic": rng.random(n_topics)})
        m.topic_ass = pd.DataFrame({"Topic": range(1, n_topics + 1), "Assignments": rng.integers(0, 99, n_topics)})
        m.cell_topic = pd.DataFrame(cell_topic, index=topics, columns=cells).iloc[:, order]
        m.cell_topic_harmony = (
            pd.DataFrame(cell_topic * 2, index=topics, columns=cells) if harmony else []
        )
        m.topic_region = pd.DataFrame(region_topic, index=regions, columns=topics)
        m.region_topic = m.topic_region
        m.parameters = pd.DataFrame(
            {"Parameter": ["lda", n_topics, 500, 555, 50, True, 0.1]},
            index=["package", "n_topics", "n_iter", "random_state", "alpha", "alpha_by_topic", "eta"],
        )
        m.n_cells, m.n_regions, m.n_topic = n_cells, n_regions, n_topics
        obj.selected_model = m
    if projections:
        obj.projections["cell"]["UMAP"] = pd.DataFrame(
            rng.normal(size=(n_cells, 2)), index=cells, columns=["UMAP_1", "UMAP_2"]
        ).iloc[order]
    data = pickle.dumps(obj, protocol=4)
    path.write_bytes(gzip.compress(data) if gz else data)
    return CistopicTruth(path, obj)


def write_imputed_pickle(path: Path, *, n_cells: int = 15, n_features: int = 8, seed: int = 0) -> CistopicTruth:
    """A CistopicImputedFeatures: a dense features x cells `mtx`."""
    import pickle

    classes = _pycistopic_classes()
    rng = np.random.default_rng(seed)
    obj = classes["CistopicImputedFeatures"]()
    obj.mtx = (rng.random((n_features, n_cells)) * (rng.random((n_features, n_cells)) > 0.5)).astype(np.float32)
    obj.feature_names = [f"Gene{j}" for j in range(n_features)]
    obj.cell_names = [f"AAAC{i:04d}-1___demo" for i in range(n_cells)]
    obj.project = "demo"
    path.write_bytes(pickle.dumps(obj, protocol=4))
    return CistopicTruth(path, obj)


# ---------------------------------------------------------------------------
# MuData, laid out as mudata's write_h5mu lays it out: each modality and each
# global element written with anndata's write_elem.


@dataclass
class MudataTruth:
    path: Path
    mods: Dict[str, Any]  # name -> AnnData
    obs: Any  # the global obs, a DataFrame
    obsm: Dict[str, np.ndarray]
    obsp: Dict[str, Any]
    uns: Dict[str, Any]

    def rows(self, mod: str) -> np.ndarray:
        """The global row of each of the modality's cells."""
        where = {name: i for i, name in enumerate(self.obs.index)}
        return np.array([where[n] for n in self.mods[mod].obs_names])


def write_h5mu(
    path: Path,
    *,
    zarr_format: Optional[int] = None,
    maps: bool = True,
    userblock: bool = True,
    root_attrs: bool = True,
    n_cells: int = 40,
    seed: int = 0,
) -> MudataTruth:
    """Two modalities sharing some cells: `rna` has all of them, `atac` the
    last three quarters, in reverse order, so its rows are out of the
    global order. Global obs columns (one clashing with rna's own, one
    pulled from atac as `atac:tss`), a global obsm, obsp and uns."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from anndata.io import write_elem

    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    rna = ad.AnnData(
        sp.random(n_cells, 6, density=0.4, format="csr", random_state=seed, dtype=np.float32),
        obs=pd.DataFrame({"n_genes": rng.integers(1, 9, n_cells)}, index=cells),
        var=pd.DataFrame({"gene_ids": [f"ENSG{j}" for j in range(6)]}, index=[f"G{j}" for j in range(6)]),
    )
    rna.obsm["X_pca"] = rng.normal(size=(n_cells, 3))
    rna.uns["rna_only"] = 1
    atac_cells = cells[n_cells // 4:][::-1]
    atac = ad.AnnData(
        sp.random(len(atac_cells), 5, density=0.4, format="csr", random_state=seed + 1, dtype=np.float32),
        obs=pd.DataFrame({"tss": rng.random(len(atac_cells))}, index=atac_cells),
        var=pd.DataFrame(index=[f"chr1:{100 * j}-{100 * j + 50}" for j in range(5)]),
    )
    mods = {"rna": rna, "atac": atac}
    obs = pd.DataFrame(index=cells)
    obs["leiden"] = pd.Categorical(rng.choice(["a", "b", "c"], n_cells))
    obs["n_genes"] = np.zeros(n_cells, dtype=np.int64)  # rna has its own
    obs["atac:tss"] = np.nan
    obs.loc[atac_cells, "atac:tss"] = atac.obs["tss"].to_numpy()
    obs["sample"] = pd.array([None if i % 7 == 0 else f"s{i % 2}" for i in range(n_cells)], dtype="string")
    obsm = {"X_wnn_umap": rng.normal(size=(n_cells, 2)),
            "X_pca": rng.normal(size=(n_cells, 3))}  # rna has its own
    # About five neighbours a cell, as a kNN graph has, whatever the count.
    obsp = {"wnn_connectivities": sp.random(n_cells, n_cells, density=min(0.2, 5 / n_cells),
                                            format="csr", random_state=seed + 2, dtype=np.float32)}
    uns = {"wnn": {"k": 5}, "rna_only": 2}
    var_names = list(rna.var_names) + list(atac.var_names)

    if zarr_format is None:
        handle = h5py.File(path, "w", userblock_size=512 if userblock else 0)
    else:
        import zarr

        handle = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    try:
        root = handle
        write_elem(root, "obs", obs)
        write_elem(root, "var", pd.DataFrame(index=var_names))
        masks = {m: np.isin(cells, list(a.obs_names)) for m, a in mods.items()}
        write_elem(root, "obsm", {**obsm, **masks})
        write_elem(root, "varm", {m: np.isin(var_names, list(a.var_names)) for m, a in mods.items()})
        write_elem(root, "obsp", obsp)
        write_elem(root, "varp", {})
        write_elem(root, "uns", uns)
        if maps:
            position = {m: {n: i for i, n in enumerate(a.obs_names)} for m, a in mods.items()}
            write_elem(root, "obsmap", {
                m: np.array([position[m].get(c, -1) + 1 for c in cells], dtype=np.uint32) for m in mods
            })
            vposition = {m: {n: i for i, n in enumerate(a.var_names)} for m, a in mods.items()}
            write_elem(root, "varmap", {
                m: np.array([vposition[m].get(v, -1) + 1 for v in var_names], dtype=np.uint32) for m in mods
            })
        group = root.create_group("mod")
        for name, adata in mods.items():
            write_elem(group, name, adata)
        group.attrs["mod-order"] = list(mods) if zarr_format else np.array(list(mods), dtype=h5py.string_dtype())
        if root_attrs:
            root.attrs.update({"encoding-type": "mudata", "encoding-version": "0.1.0",
                               "encoder": "mudata", "encoder-version": "0.3.2", "axis": 0})
    finally:
        if zarr_format is None:
            handle.close()
    if zarr_format is None and userblock:
        header = b"MuData (format-version=0.1.0;creator=muon;creator-version=0.3.2)"
        with open(path, "r+b") as raw:
            raw.write(header.ljust(512, b"\0"))
    return MudataTruth(path, mods, obs, obsm, obsp, uns)


# ---------------------------------------------------------------------------
# Space Ranger output for standard Visium
#
# As Space Ranger writes it (checked against 10x's SR 1.1 and SR 2.1 sample
# outputs): the matrix, and in spatial/ every spot's position -- in no
# particular order, and including spots the filtered matrix leaves out --
# the scale factors and two PNGs. SR 1 wrote the positions with no header.


@dataclass
class VisiumTruth:
    outs: Path
    barcodes: List[str]  # the filtered matrix's, in its order
    in_tissue: np.ndarray
    array_row: np.ndarray
    array_col: np.ndarray
    spatial: np.ndarray  # [pxl_col, pxl_row], per barcode
    images: Dict[str, np.ndarray]
    scalefactors: dict
    counts: np.ndarray


def write_visium_outs(
    folder: Path,
    *,
    version: str = "sr2",
    n_spots: int = 12,
    n_extra: int = 5,
    raw: bool = True,
    images: bool = True,
    seed: int = 0,
) -> VisiumTruth:
    """An `outs/` folder. `version` is sr1 (no header), sr2, or sr3_float
    (pixel positions with decimals, as Space Ranger 3 and 4 may write)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    outs = folder / "outs"
    spatial = outs / "spatial"
    spatial.mkdir(parents=True)
    n_features = 6
    barcodes = [f"AAAC{i:04d}-1" for i in range(n_spots)]
    extra = [f"TTTG{i:04d}-1" for i in range(n_extra)]
    counts = rng.integers(0, 5, size=(n_spots, n_features))
    ids = [f"ENSG{i:05d}" for i in range(n_features)]
    names = [f"G{i}" for i in range(n_features)]
    types = ["Gene Expression"] * (n_features - 1) + ["Antibody Capture"]

    def matrix(path: Path, rows: List[str], values: np.ndarray) -> None:
        write_10x_h5(path, values, barcodes=rows, ids=ids, names=names, types=types)
        with h5py.File(path, "a") as f:
            f.attrs["library_ids"] = np.array([b"V1_Test_Section"])
            f.attrs["chemistry_description"] = "Spatial 3' v1"
            f.attrs["software_version"] = "spaceranger-2.1.0"

    matrix(outs / "filtered_feature_bc_matrix.h5", barcodes, counts)
    if raw:
        matrix(
            outs / "raw_feature_bc_matrix.h5", barcodes + extra,
            np.vstack([counts, np.zeros((n_extra, n_features), dtype=counts.dtype)]),
        )

    every = barcodes + extra
    n = len(every)
    in_tissue = np.array([1] * n_spots + [0] * n_extra)
    array_row = rng.integers(0, 78, n)
    array_col = rng.integers(0, 128, n)
    pxl_row = rng.integers(-50, 2000, n).astype(float)
    pxl_col = rng.integers(0, 2000, n).astype(float)
    if version == "sr3_float":
        pxl_row += 0.25
        pxl_col += 0.5
    order = rng.permutation(n)
    lines = []
    if version != "sr1":
        lines.append(",".join(
            ("barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
        ))
    fmt = (lambda v: f"{v:g}") if version == "sr3_float" else (lambda v: str(int(v)))
    for i in order:
        lines.append(
            f"{every[i]},{in_tissue[i]},{array_row[i]},{array_col[i]},{fmt(pxl_row[i])},{fmt(pxl_col[i])}"
        )
    name = "tissue_positions_list.csv" if version == "sr1" else "tissue_positions.csv"
    (spatial / name).write_text("\n".join(lines) + "\n")

    scalefactors = {
        "spot_diameter_fullres": 89.4,
        "tissue_hires_scalef": 0.17,
        "fiducial_diameter_fullres": 144.5,
        "tissue_lowres_scalef": 0.05,
    }
    (spatial / "scalefactors_json.json").write_text(json.dumps(scalefactors))
    pictures: Dict[str, np.ndarray] = {}
    if images:
        for key, size in (("hires", (20, 16)), ("lowres", (6, 5))):
            pixels = rng.integers(0, 256, size=size + (3,), dtype=np.uint8)
            Image.fromarray(pixels).save(spatial / f"tissue_{key}_image.png")
            pictures[key] = pixels

    take = slice(0, n_spots)
    return VisiumTruth(
        outs=outs,
        barcodes=barcodes,
        in_tissue=in_tissue[take],
        array_row=array_row[take],
        array_col=array_col[take],
        spatial=np.stack([pxl_col, pxl_row], axis=1)[take],
        images=pictures,
        scalefactors=scalefactors,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Xenium output bundles
#
# As Xenium Onboard Analysis writes them (columns checked against 10x's XOA
# 1.0.1, 3.0 and 4.0 sample bundles): the cell matrix, cells.csv.gz in the
# matrix's order, experiment.xenium, and analysis/ CSVs keyed by Barcode.
# XOA 1 numbered cells; later versions name them like "aaafiiei-1".


@dataclass
class XeniumTruth:
    bundle: Path
    cell_ids: List[str]  # the matrix's, in its order
    columns: Dict[str, np.ndarray]  # cells.csv.gz's other columns, per cell
    spatial: np.ndarray
    clusters: Optional[np.ndarray]  # per cell, "" where analysis lacks it
    umap: Optional[np.ndarray]  # NaN where analysis lacks it
    counts: np.ndarray


def write_xenium_bundle(
    folder: Path,
    *,
    ids: str = "string",
    n_cells: int = 15,
    shuffled: bool = False,
    extra_cells: int = 0,
    analysis: bool = True,
    missing_in_analysis: int = 0,
    seed: int = 0,
) -> XeniumTruth:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    n_features = 5
    if ids == "int":
        cell_ids = [str(i + 1) for i in range(n_cells + extra_cells)]
    else:
        cell_ids = [
            "".join("abcdefghijklmnop"[d] for d in rng.integers(0, 16, 8)) + "-1"
            for _ in range(n_cells + extra_cells)
        ]
    counts = rng.integers(0, 6, size=(n_cells, n_features))
    write_10x_h5(
        folder / "cell_feature_matrix.h5", counts, barcodes=cell_ids[:n_cells],
        ids=[f"ENSG{i:05d}" for i in range(n_features - 1)] + ["NegControlProbe_00001"],
        names=[f"G{i}" for i in range(n_features - 1)] + ["NegControlProbe_00001"],
        types=["Gene Expression"] * (n_features - 1) + ["Negative Control Probe"],
    )
    (folder / "experiment.xenium").write_text(json.dumps({"pixel_size": 0.2125, "major_version": 5}))

    n = n_cells + extra_cells
    table = {
        "x_centroid": rng.uniform(0, 5000, n).round(4),
        "y_centroid": rng.uniform(0, 5000, n).round(4),
        "transcript_counts": rng.integers(0, 300, n),
        "control_probe_counts": rng.integers(0, 3, n),
        "cell_area": rng.uniform(10, 200, n).round(3),
        "nucleus_area": rng.uniform(5, 50, n).round(3),
        "segmentation_method": rng.choice(["Segmented by boundary stain", "Segmented by nucleus expansion"], n),
    }
    if extra_cells:
        # A method only the left-out cells use: its category must go.
        table["segmentation_method"][n_cells:] = "Imported cell segmentation"
    order = rng.permutation(n) if shuffled else np.arange(n)
    names = ["cell_id", *table]
    lines = [",".join(names)]
    for i in order:
        values = [cell_ids[i]] + [str(table[k][i]) for k in table]
        lines.append(",".join(f'"{v}"' if k in ("cell_id", "segmentation_method") else v for k, v in zip(names, values)))
    (folder / "cells.csv.gz").write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))

    clusters = umap = None
    if analysis:
        listed = rng.permutation(n_cells)[missing_in_analysis:]
        clusters = np.full(n_cells, "", dtype=object)
        clusters[listed] = [str(v) for v in rng.integers(1, 12, listed.size)]
        umap = np.full((n_cells, 2), np.nan)
        umap[listed] = rng.normal(size=(listed.size, 2)).round(5)
        path = folder / "analysis" / "clustering" / "gene_expression_graphclust"
        path.mkdir(parents=True)
        (path / "clusters.csv").write_text(
            "Barcode,Cluster\n" + "".join(f"{cell_ids[i]},{clusters[i]}\n" for i in listed)
        )
        path = folder / "analysis" / "umap" / "gene_expression_2_components"
        path.mkdir(parents=True)
        (path / "projection.csv").write_text(
            "Barcode,UMAP-1,UMAP-2\n" + "".join(f"{cell_ids[i]},{umap[i, 0]},{umap[i, 1]}\n" for i in listed)
        )

    keep = slice(0, n_cells)
    return XeniumTruth(
        bundle=folder,
        cell_ids=cell_ids[:n_cells],
        columns={k: v[keep] for k, v in table.items() if k not in ("x_centroid", "y_centroid")},
        spatial=np.stack([table["x_centroid"], table["y_centroid"]], axis=1)[keep],
        clusters=clusters,
        umap=umap,
        counts=counts,
    )

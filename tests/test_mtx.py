"""Tests for `adata convert` to and from 10x / STARsolo Matrix Market folders.

Fixtures are written as text by hand (see conftest), in each layout Cell
Ranger and STARsolo produce. What anndata reads back is checked against the
counts the fixture was built from; agreement with scanpy's `read_10x_mtx` is
checked in test_external_readers.py, which needs scanpy's environment.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import List

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_10x_mtx, write_mtx_file, write_starsolo_velocyto

ad = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
mmread = pytest.importorskip("scipy.io").mmread

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

N_CELLS = 30
IDS = [f"ENSG{i:05d}" for i in range(6)] + ["CD3", "CD4"]
NAMES = ["A", "B", "B", "C", "D", "E", "CD3_TotalSeqB", "CD4_TotalSeqB"]
TYPES = ["Gene Expression"] * 6 + ["Antibody Capture"] * 2


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _barcodes(n: int = N_CELLS) -> List[str]:
    return [f"AAACCCAAG{i:07d}-1" for i in range(n)]


def _counts(n_cells: int = N_CELLS, n_features: int = len(IDS), seed: int = 0):
    counts = np.random.default_rng(seed).poisson(1.2, (n_cells, n_features))
    counts[-1] = 0
    return counts


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _debris(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.rglob("*") if "adata-" in p.name)


@pytest.fixture
def counts():
    return _counts()


@pytest.fixture
def cr3(tmp_path, counts) -> Path:
    return write_10x_mtx(
        tmp_path / "filtered_feature_bc_matrix", counts,
        barcodes=_barcodes(), ids=IDS, names=NAMES, types=TYPES,
    )


# ---------------------------------------------------------------------------
# import


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("layout", ["cr3", "cr2", "starsolo", "starsolo2col"])
def test_each_layout_imports_as_scanpy_reads_it(tmp_path, counts, layout, suffix):
    types = TYPES if layout in ("cr3", "starsolo") else None
    folder = write_10x_mtx(
        tmp_path / "in", counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=types, layout=layout,
    )
    out = tmp_path / f"out{suffix}"
    _ok(folder, "-o", out, "--var-names", "gene_symbols")

    got = _read(out)
    assert got.X.dtype == np.int32, "integer counts stay integers"
    assert np.array_equal(got.X.toarray(), counts)
    assert got.X.has_sorted_indices
    assert list(got.obs_names) == _barcodes()
    assert list(got.var_names) == NAMES
    assert list(got.var["gene_ids"]) == IDS
    if types is None:
        assert list(got.var.columns) == ["gene_ids"]
    else:
        assert list(got.var.columns) == ["gene_ids", "feature_types"]
        assert got.var["feature_types"].dtype == "category"
        assert list(got.var["feature_types"]) == TYPES
    assert got.uns["cellranger"]["layout"] == "mtx"


@pytest.mark.parametrize("order", ["cell", "gene", "shuffled"])
def test_entry_order_does_not_change_the_store(tmp_path, counts, order, monkeypatch):
    import adata.core.csr_build as csr_build
    import adata.formats.mtx as mtx_module

    if order == "cell":
        # In barcode order, as Cell Ranger and STARsolo write it: no buckets.
        def refuse(*args, **kwargs):
            raise AssertionError("an ordered file must not go through the buckets")

        monkeypatch.setattr(csr_build, "BucketSorter", refuse)
    else:
        monkeypatch.setattr(mtx_module, "DEFAULT_CHUNK", 8)

    folder = write_10x_mtx(
        tmp_path / "in", counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=TYPES, order=order,
    )
    out = tmp_path / "out.h5ad"
    _ok(folder, "-o", out)
    assert np.array_equal(_read(out).X.toarray(), counts)
    assert _debris(tmp_path) == []


def test_duplicate_entries_are_summed_and_reported(tmp_path, counts):
    folder = write_10x_mtx(
        tmp_path / "in", counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=TYPES, duplicates=5,
    )
    out = tmp_path / "out.h5ad"
    result = _ok(folder, "-o", out)
    assert "duplicate entries" in _out(result)
    assert np.array_equal(_read(out).X.toarray(), counts)


def test_real_values_become_float64(tmp_path, counts):
    folder = write_10x_mtx(
        tmp_path / "in", counts * 0.5, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=TYPES, field="real",
    )
    out = tmp_path / "out.h5ad"
    _ok(folder, "-o", out)
    got = _read(out).X
    assert got.dtype == np.float64
    assert np.array_equal(got.toarray(), counts * 0.5)


def test_var_names_and_make_unique_follow_scanpy(cr3, tmp_path):
    out = tmp_path / "ids.h5ad"
    result = _ok(cr3, "-o", out)
    assert "repeat an earlier one" not in _out(result), "ids are the default names"
    got = _read(out)
    assert list(got.var_names) == IDS
    assert list(got.var["gene_symbols"]) == NAMES

    out = tmp_path / "unique.h5ad"
    result = _ok(cr3, "-o", out, "--var-names", "gene_symbols")
    assert "repeat an earlier one" in _out(result)
    out = tmp_path / "unique2.h5ad"
    _ok(cr3, "-o", out, "--var-names", "gene_symbols", "--make-unique")
    names = list(_read(out).var_names)
    assert len(set(names)) == len(names)


def test_feature_types_can_be_kept_or_split(cr3, tmp_path, counts):
    out = tmp_path / "gex.h5ad"
    _ok(cr3, "-o", out, "--feature-type", "Gene Expression")
    got = _read(out)
    assert list(got.var_names) == IDS[:6]
    assert np.array_equal(got.X.toarray(), counts[:, :6])

    out = tmp_path / "split.h5ad"
    _ok(cr3, "-o", out, "--split-feature-types")
    assert not out.exists()
    gex = _read(tmp_path / "split.Gene_Expression.h5ad")
    adt = _read(tmp_path / "split.Antibody_Capture.h5ad")
    assert np.array_equal(gex.X.toarray(), counts[:, :6])
    assert np.array_equal(adt.X.toarray(), counts[:, 6:])
    assert list(adt.var_names) == IDS[6:]


def test_velocyto_matrices_become_layers_with_x_as_spliced(tmp_path):
    layers = {
        "spliced": _counts(seed=1),
        "unspliced": _counts(seed=2),
        "ambiguous": _counts(seed=3),
    }
    folder = write_starsolo_velocyto(
        tmp_path / "Velocyto" / "raw", layers, barcodes=_barcodes(), ids=IDS, names=NAMES
    )
    out = tmp_path / "velo.h5ad"
    result = _ok(folder, "-o", out)
    assert "STARsolo velocyto" in _out(result)

    got = _read(out)
    for name, expected in layers.items():
        assert np.array_equal(got.layers[name].toarray(), expected), name
    assert np.array_equal(got.X.toarray(), layers["spliced"])


def test_a_velocyto_folder_without_ambiguous_is_fine(tmp_path):
    layers = {"spliced": _counts(seed=1), "unspliced": _counts(seed=2)}
    folder = write_starsolo_velocyto(
        tmp_path / "v", layers, barcodes=_barcodes(), ids=IDS, names=NAMES
    )
    out = tmp_path / "velo.h5ad"
    _ok(folder, "-o", out)
    # Recent anndata also lists X itself, under None.
    assert set(_read(out).layers) - {None} == {"spliced", "unspliced"}


def test_geo_prefixed_files_are_found(tmp_path, counts):
    folder = write_10x_mtx(
        tmp_path / "GSM123", counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=TYPES, prefix="GSM123_sample_",
    )
    out = tmp_path / "out.h5ad"
    _ok(folder, "-o", out)
    assert np.array_equal(_read(out).X.toarray(), counts)


def test_two_samples_in_one_folder_are_refused(tmp_path, counts):
    folder = tmp_path / "two"
    for prefix in ("A_", "B_"):
        write_10x_mtx(folder, counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
                      types=TYPES, prefix=prefix)
    result = _convert(folder, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "'A_', 'B_'" in _out(result)


def test_cell_ranger_2_genome_folders(tmp_path, counts):
    root = tmp_path / "filtered_gene_bc_matrices"
    write_10x_mtx(root / "hg19", counts, barcodes=_barcodes(), ids=IDS, names=NAMES, layout="cr2")
    out = tmp_path / "one.h5ad"
    _ok(root, "-o", out)
    assert _read(out).uns["cellranger"]["genome"] == "hg19"

    write_10x_mtx(root / "mm10", counts[:, :3], barcodes=_barcodes(), ids=IDS[:3],
                  names=NAMES[:3], layout="cr2")
    result = _convert(root, "-o", tmp_path / "two.h5ad")
    assert result.exit_code == 1
    assert "pick one with --genome" in _out(result)

    out = tmp_path / "mm10.h5ad"
    _ok(root, "-o", out, "--genome", "mm10")
    assert np.array_equal(_read(out).X.toarray(), counts[:, :3])


@pytest.mark.parametrize(
    "run,subfolders,layout",
    [
        ("Solo.out/Gene", ["filtered", "raw"], "starsolo"),
        ("outs", ["filtered_feature_bc_matrix", "raw_feature_bc_matrix"], "cr3"),
        ("count", ["sample_filtered_feature_bc_matrix"], "cr3"),
        # Cell Ranger 2's file names, as DropletUtils' write10xCounts writes them.
        ("droplets", ["filtered", "raw"], "cr2"),
        ("droplets", ["filtered"], "cr2"),
    ],
)
def test_a_runs_filtered_and_raw_folders_are_not_taken_for_genomes(tmp_path, counts, run, subfolders, layout):
    """F105: a folder holding a run's filtered/ and raw/ matrices, as
    STARsolo's Gene/ and Cell Ranger's outs/ do, was read as Cell Ranger 2's
    one folder per genome: refused with 'pick one with --genome'; and with
    one such folder, as Cell Ranger multi's count/ holds, converted with its
    name recorded as the genome. Only a Cell Ranger 2 folder (genes.tsv) is
    a genome."""
    root = tmp_path / run
    for name in subfolders:
        write_10x_mtx(root / name, counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
                      types=TYPES, layout=layout)
    out = tmp_path / "out.h5ad"
    result = _convert(root, "-o", out)
    assert result.exit_code == 1
    assert "not Cell Ranger 2's one folder per genome" in _out(result)
    assert f"such as {root.name}/{subfolders[0]}" in _out(result)
    assert not out.exists()

    _ok(root / subfolders[0], "-o", out)
    got = _read(out)
    assert np.array_equal(got.X.toarray(), counts)
    assert "genome" not in got.uns["cellranger"]


def test_cell_ranger_metadata_is_kept_and_written_back(tmp_path, counts):
    comment = '%metadata_json: {"software_version": "cellranger-7.1.0", "format_version": 2}'
    folder = write_10x_mtx(
        tmp_path / "in", counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=TYPES, comments=[comment],
    )
    out = tmp_path / "out.h5ad"
    _ok(folder, "-o", out)
    attrs = _read(out).uns["cellranger"]["attrs"]
    assert attrs["software_version"] == "cellranger-7.1.0"

    back = tmp_path / "back"
    _ok(out, "-o", back, "--to", "10x-mtx")
    with gzip.open(back / "matrix.mtx.gz", "rt") as fh:
        header = [next(fh) for _ in range(2)]
    assert header[1].startswith("%metadata_json:")
    assert '"software_version": "cellranger-7.1.0"' in header[1]


# ---------------------------------------------------------------------------
# refusals: nothing may be left behind


@pytest.mark.parametrize(
    "spoil,message",
    [
        ("barcodes", "lists 29 barcodes, but the matrix has 30"),
        ("features", "lists 7 features, but the matrix has 8"),
        ("symmetric", "'symmetric'"),
        ("range", "outside"),
        ("short", "has 3 entries"),
    ],
)
def test_a_broken_folder_writes_nothing(tmp_path, counts, spoil, message):
    folder = write_10x_mtx(
        tmp_path / "in", counts, barcodes=_barcodes(), ids=IDS, names=NAMES, types=TYPES,
        layout="starsolo",
    )
    if spoil == "barcodes":
        (folder / "barcodes.tsv").write_text("\n".join(_barcodes()[:-1]) + "\n")
    elif spoil == "features":
        lines = (folder / "features.tsv").read_text().splitlines()[:-1]
        (folder / "features.tsv").write_text("\n".join(lines) + "\n")
    elif spoil == "symmetric":
        text = (folder / "matrix.mtx").read_text().replace("general", "symmetric", 1)
        (folder / "matrix.mtx").write_text(text)
    elif spoil == "range":
        with open(folder / "matrix.mtx", "a") as fh:
            fh.write("99 1 1\n")
        lines = (folder / "matrix.mtx").read_text().splitlines()
        f, c, n = lines[1].split()
        lines[1] = f"{f} {c} {int(n) + 1}"
        (folder / "matrix.mtx").write_text("\n".join(lines) + "\n")
    elif spoil == "short":
        lines = (folder / "matrix.mtx").read_text().splitlines()[:5]
        f, c, _ = lines[1].split()
        lines[1] = f"{f} {c} 4"
        (folder / "matrix.mtx").write_text("\n".join(lines) + "\n")

    out = tmp_path / "out.h5ad"
    result = _convert(folder, "-o", out)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()
    assert _debris(tmp_path) == []


@pytest.mark.parametrize(
    "layout,args,message",
    [
        ("starsolo2col", ["--feature-type", "Gene Expression"], "records no feature types"),
        ("cr2", ["--split-feature-types"], "records no feature types"),
        ("cr3", ["--genome", "hg19"], "no genome per feature"),
        ("cr3", ["--var-names", "symbols"], "--var-names must be one of"),
        ("cr3", ["--compression", "zstd"], "not available for h5ad"),
        ("cr3", ["--layer", "counts"], "does not apply"),
    ],
)
def test_options_that_do_not_fit_are_refused(tmp_path, counts, layout, args, message):
    folder = write_10x_mtx(
        tmp_path / "in", counts, barcodes=_barcodes(), ids=IDS, names=NAMES,
        types=TYPES if layout == "cr3" else None, layout=layout,
    )
    result = _convert(folder, "-o", tmp_path / "out.h5ad", *args)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not (tmp_path / "out.h5ad").exists()


def test_a_single_mtx_file_points_to_its_folder(tmp_path, counts):
    path = write_mtx_file(tmp_path / "matrix.mtx", counts)
    result = _convert(path, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "import sparse" in _out(result)


def test_a_zarr_store_is_not_taken_for_a_matrix_folder(tmp_path, cr3):
    store = tmp_path / "s.zarr"
    _ok(cr3, "-o", store)
    out = tmp_path / "copy.h5ad"
    _ok(store, "-o", out)
    assert _read(out).shape == (N_CELLS, len(IDS))


# ---------------------------------------------------------------------------
# export


def _text(path: Path) -> str:
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as fh:
        return fh.read()


def test_import_then_export_gives_back_the_same_files(cr3, tmp_path):
    store = tmp_path / "s.h5ad"
    _ok(cr3, "-o", store)
    back = tmp_path / "back"
    _ok(store, "-o", back, "--to", "10x-mtx")

    for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
        assert _text(back / name) == _text(cr3 / name), name


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_export_from_any_layout_reads_back_in_scipy(tmp_path, counts, layout):
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](
        counts.astype(np.int32)
    )
    store = tmp_path / "s.h5ad"
    ad.AnnData(
        X=X,
        obs={"_": np.zeros(N_CELLS)},
        var={"gene_ids": IDS, "feature_types": TYPES},
    ).write_h5ad(store)
    folder = tmp_path / "out"
    _ok(store, "-o", folder, "--to", "10x-mtx")

    back = mmread(str(folder / "matrix.mtx.gz"))
    assert np.array_equal(back.toarray().T, counts)
    assert "integer" in _text(folder / "matrix.mtx.gz").splitlines()[0]


def test_export_uncompressed_with_a_layer(cr3, tmp_path, counts):
    store = tmp_path / "s.h5ad"
    _ok(cr3, "-o", store)
    obj = ad.read_h5ad(store)
    obj.layers["counts"] = obj.X.copy()
    obj.X = obj.X.astype(np.float32) / 2
    obj.write_h5ad(store)

    folder = tmp_path / "plain"
    _ok(store, "-o", folder, "--to", "10x-mtx", "--layer", "counts", "--compression", "none")
    assert sorted(p.name for p in folder.iterdir()) == [
        "barcodes.tsv", "features.tsv", "matrix.mtx",
    ]
    assert np.array_equal(mmread(str(folder / "matrix.mtx")).toarray().T, counts)

    result = _ok(store, "-o", tmp_path / "x", "--to", "10x-mtx")
    assert "holds float64 values" in _out(result)


def test_export_overwrites_an_existing_folder_only_when_asked(cr3, tmp_path):
    store = tmp_path / "s.h5ad"
    _ok(cr3, "-o", store)
    folder = tmp_path / "out"
    _ok(store, "-o", folder, "--to", "10x-mtx")

    result = _convert(store, "-o", folder, "--to", "10x-mtx")
    assert result.exit_code == 1
    assert "--overwrite" in _out(result)

    _ok(store, "-o", folder, "--to", "10x-mtx", "--overwrite")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["filtered_feature_bc_matrix", "s.h5ad", "out"]
    )


@pytest.mark.parametrize(
    "args,message",
    [
        (["--compression", "lzf"], "gzipped or not"),
        (["--genome", "hg19"], "does not apply"),
    ],
)
def test_export_refuses_options_that_do_not_fit(cr3, tmp_path, args, message):
    store = tmp_path / "s.h5ad"
    _ok(cr3, "-o", store)
    result = _convert(store, "-o", tmp_path / "out", "--to", "10x-mtx", *args)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not (tmp_path / "out").exists()


def test_a_barcode_with_a_tab_is_refused(tmp_path, counts):
    store = tmp_path / "s.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.int32)),
        obs={"_": np.zeros(N_CELLS)},
        var={"gene_ids": IDS, "feature_types": TYPES},
    ).write_h5ad(store)
    obj = ad.read_h5ad(store)
    obj.obs_names = ["a\tb"] + _barcodes()[1:]
    obj.write_h5ad(store)

    result = _convert(store, "-o", tmp_path / "out", "--to", "10x-mtx")
    assert result.exit_code == 1
    assert "tab or a line break" in _out(result)
    assert not (tmp_path / "out").exists()
    assert _debris(tmp_path) == []


def _store_named(tmp_path: Path, counts, *, barcodes=None, names=None, ids=None) -> Path:
    store = tmp_path / "s.h5ad"
    obj = ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.int32)),
        obs={"_": np.zeros(N_CELLS)},
        var={"gene_ids": ids or IDS, "gene_symbols": names or NAMES, "feature_types": TYPES},
    )
    obj.obs_names = barcodes or _barcodes()
    obj.var_names = IDS
    obj.write_h5ad(store)
    return store


@pytest.mark.parametrize(
    "where,value,message",
    [
        ("barcode", '"AAAC-1', "A barcode starts with a double quote"),
        ("name", '"quoted', "A feature name starts with a double quote"),
        ("barcode", "", "A barcode is empty"),
        ("barcode", "AAAC,1", "A barcode contains a comma"),
        ("id", "", "A feature id is empty"),
    ],
)
def test_names_pandas_and_r_would_misread_are_refused(tmp_path, counts, where, value, message):
    """F106: a name starting with a double quote was written as it was, and
    pandas (so scanpy) and R read it as a quoted field running on into the
    next lines; an empty barcode made a blank line, which they skip, moving
    every later barcode up a row. Found in review: scanpy reads barcodes.tsv
    as comma-separated, so a comma cut a barcode short, and pandas reads an
    empty id as missing."""
    barcodes, names, ids = _barcodes(), list(NAMES), list(IDS)
    if where == "barcode":
        barcodes[3] = value
    elif where == "id":
        ids[3] = value
    else:
        names[3] = value
    store = _store_named(tmp_path, counts, barcodes=barcodes, names=names, ids=ids)
    out = tmp_path / "out"
    result = _convert(store, "-o", out, "--to", "10x-mtx")
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()
    assert _debris(tmp_path) == []


def test_a_quote_inside_a_name_reads_back_in_pandas(tmp_path, counts):
    """Only a quote that starts a field is taken for quoting: one inside it
    is exported, and reads back as written. An empty name is exported too:
    pandas reads it as missing, as it reads a gap in the table it came from."""
    pd = pytest.importorskip("pandas")
    names = list(NAMES)
    names[3], names[4] = 'say "hi"', ""
    out = tmp_path / "out"
    _ok(_store_named(tmp_path, counts, names=names), "-o", out, "--to", "10x-mtx")
    # As scanpy's read_10x_mtx reads it.
    features = pd.read_csv(out / "features.tsv.gz", header=None, sep="\t")
    assert list(features[1].fillna("")) == names


def test_a_folder_output_needs_to(cr3, tmp_path):
    store = tmp_path / "s.h5ad"
    _ok(cr3, "-o", store)
    result = _convert(store, "-o", tmp_path / "out")
    assert result.exit_code == 1
    assert "--to 10x-mtx" in _out(result)


def test_export_writes_the_whole_uint64_range_exactly(tmp_path):
    """Found in review: values above INT64_MAX wrapped negative, and the
    int64 minimum hung the formatter."""
    values = np.array([np.iinfo(np.uint64).max, 2**63, 1], dtype=np.uint64)
    store = tmp_path / "s.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((values, [0, 1, 0], [0, 2, 3]), shape=(2, 2)),
        obs={"_": np.zeros(2)},
        var={"gene_ids": ["a", "b"], "feature_types": ["Gene Expression"] * 2},
    ).write_h5ad(store)
    folder = tmp_path / "out"
    _ok(store, "-o", folder, "--to", "10x-mtx")

    lines = _text(folder / "matrix.mtx.gz").splitlines()
    assert lines[0].endswith("integer general")
    assert lines[-3:] == [
        f"1 1 {np.iinfo(np.uint64).max}",
        f"2 1 {2**63}",
        "1 2 1",
    ]

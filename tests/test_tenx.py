"""Tests for `adata convert` to and from 10x Genomics HDF5 matrices.

The fixtures are built the way Cell Ranger writes them (see conftest). What
anndata reads back from our h5ad is checked here against the counts the
fixture was built from; agreement with scanpy itself is checked in
test_external_readers.py, which needs scanpy's environment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_10x_h5, write_10x_v2_h5

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

N_CELLS = 24
IDS = [f"ENSG{i:05d}" for i in range(6)] + ["CD3", "CD4"]
NAMES = ["A", "B", "B", "C", "D", "E", "CD3_TotalSeqB", "CD4_TotalSeqB"]
TYPES = ["Gene Expression"] * 6 + ["Antibody Capture"] * 2


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _barcodes(n: int = N_CELLS) -> List[str]:
    return [f"AAACCCAAG{i:07d}-1" for i in range(n)]


def _counts(n_cells: int = N_CELLS, n_features: int = len(IDS), seed: int = 0):
    counts = np.random.default_rng(seed).poisson(0.7, (n_cells, n_features))
    counts[-1] = 0
    return counts


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args: str):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args: str):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


@pytest.fixture
def counts():
    return _counts()


@pytest.fixture
def tenx(tmp_path, counts) -> Path:
    return write_10x_h5(
        tmp_path / "filtered_feature_bc_matrix.h5",
        counts,
        barcodes=_barcodes(),
        ids=IDS,
        names=NAMES,
        types=TYPES,
        extra={"pattern": [""] * 6 + ["5PNNNNNNNNNN(BC)"] * 2},
    )


# ---------------------------------------------------------------------------
# import


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_import_maps_the_matrix_as_scanpy_does(tenx, tmp_path, counts, suffix):
    out = tmp_path / f"out{suffix}"
    _ok(tenx, "-o", out, "--var-names", "gene_symbols")
    got = _read(out)

    assert got.X.dtype == np.dtype("int32"), "counts stay integers"
    assert np.array_equal(got.X.toarray(), counts)
    assert list(got.obs_names) == _barcodes()
    assert list(got.var_names) == NAMES
    assert list(got.var.columns) == ["gene_ids", "feature_types", "genome", "pattern"]
    assert list(got.var["gene_ids"]) == IDS
    assert list(got.var["feature_types"]) == TYPES
    assert set(got.var["genome"]) == {"GRCh38"}
    # Encoded as anndata's own writer would: repeated strings as categoricals.
    assert got.var["feature_types"].dtype == "category"
    assert got.var["gene_ids"].dtype != "category"


def test_import_keeps_the_run_metadata(tenx, tmp_path):
    out = tmp_path / "out.h5ad"
    _ok(tenx, "-o", out)
    meta = _read(out).uns["cellranger"]
    assert meta["layout"] == 3
    assert list(meta["feature_tag_keys"]) == ["genome", "pattern"]
    assert meta["attrs"]["software_version"] == "cellranger-8.0.1"
    assert list(meta["attrs"]["original_gem_groups"]) == [1]


def test_import_warns_about_repeated_names_and_can_fix_them(tenx, tmp_path):
    result = _ok(tenx, "-o", tmp_path / "dup.h5ad", "--var-names", "gene_symbols")
    assert "1 var names repeat an earlier one" in _out(result)

    out = tmp_path / "unique.h5ad"
    result = _ok(tenx, "-o", out, "--var-names", "gene_symbols", "--make-unique")
    assert "repeat an earlier one" not in _out(result)
    expected = ad.AnnData(np.zeros((1, len(NAMES))), var=pd.DataFrame(index=NAMES))
    expected.var_names_make_unique()
    assert list(_read(out).var_names) == list(expected.var_names)


def test_var_names_are_the_gene_ids_by_default(tenx, tmp_path):
    """Symbols repeat (the fixture repeats one, as real references do), so
    naming var by them needed --make-unique; ids do not."""
    out = tmp_path / "ids.h5ad"
    result = _ok(tenx, "-o", out)
    assert "repeat an earlier one" not in _out(result)
    got = _read(out)
    assert list(got.var_names) == IDS
    assert list(got.var.columns) == ["gene_symbols", "feature_types", "genome", "pattern"]
    assert list(got.var["gene_symbols"]) == NAMES


def test_a_feature_type_can_be_kept_on_its_own(tenx, tmp_path, counts):
    out = tmp_path / "rna.h5ad"
    result = _ok(tenx, "-o", out, "--feature-type", "Gene Expression")
    assert "keeping 6 features" in _out(result)
    got = _read(out)
    assert list(got.var_names) == IDS[:6]
    assert np.array_equal(got.X.toarray(), counts[:, :6])


def test_split_feature_types_writes_one_output_each(tenx, tmp_path, counts):
    out = tmp_path / "sample.h5ad"
    _ok(tenx, "-o", out, "--split-feature-types")

    rna = _read(tmp_path / "sample.Gene_Expression.h5ad")
    adt = _read(tmp_path / "sample.Antibody_Capture.h5ad")
    assert not out.exists()
    assert np.array_equal(rna.X.toarray(), counts[:, :6])
    assert np.array_equal(adt.X.toarray(), counts[:, 6:])
    assert list(adt.var_names) == IDS[6:]
    assert list(rna.obs_names) == list(adt.obs_names) == _barcodes()


def test_split_names_its_outputs_like_split_does(tmp_path):
    """Two types that sanitise to the same stem must not overwrite each other."""
    h5 = write_10x_h5(
        tmp_path / "m.h5", _counts(n_features=2), barcodes=_barcodes(),
        ids=["a", "b"], names=["a", "b"], types=["A/B", "A B"],
    )
    _ok(h5, "-o", tmp_path / "m.zarr", "--split-feature-types")
    assert sorted(p.name for p in tmp_path.glob("m.*.zarr")) == ["m.A_B.zarr", "m.A_B_1.zarr"]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--feature-type", "Peaks"], "No features of type 'Peaks'; this file has: Gene Expression, Antibody Capture"),
        (["--feature-type", "Gene Expression", "--split-feature-types"], "pass one or the other"),
        (["--genome", "mm10"], "No features of genome 'mm10'; this file has: GRCh38"),
        (["--var-names", "symbols"], "--var-names must be one of"),
        (["--layer", "counts"], "--layer does not apply when converting 10x-h5 to h5ad"),
        (["--compression", "gzip"], "--compression does not apply"),
    ],
)
def test_bad_import_options_are_refused_before_writing(tenx, tmp_path, extra, message):
    out = tmp_path / "out.h5ad"
    result = _convert(tenx, "-o", out, *extra)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


def test_a_genome_filter_keeps_that_genomes_features(tmp_path):
    counts = _counts(n_features=4)
    h5 = write_10x_h5(
        tmp_path / "barnyard.h5", counts, barcodes=_barcodes(),
        ids=list("abcd"), names=list("ABCD"), genomes=["GRCh38", "mm10", "GRCh38", "mm10"],
    )
    out = tmp_path / "mouse.h5ad"
    _ok(h5, "-o", out, "--genome", "mm10")
    got = _read(out)
    assert list(got.var_names) == ["b", "d"]
    assert np.array_equal(got.X.toarray(), counts[:, [1, 3]])


def test_a_probe_barcode_matrix_names_genes_and_probes(tmp_path):
    h5 = write_10x_h5(
        tmp_path / "flex.h5", _counts(n_features=4), barcodes=_barcodes(),
        ids=["p1", "p2", "p3", "p4"], names=["G1", "G1", "G2", "G3"],
        probe_gene_ids=["E1", "E1", "E2", "E3"],
        untagged={"filtered_probes": [True, False, True, True]},
        filtered_barcodes=[i % 2 == 0 for i in range(N_CELLS)],
    )
    out = tmp_path / "flex.h5ad"
    _ok(h5, "-o", out)
    got = _read(out)
    assert list(got.var_names) == ["E1", "E1", "E2", "E3"]
    assert list(got.var["gene_symbols"]) == ["G1", "G1", "G2", "G3"]
    assert list(got.var["probe_ids"]) == ["p1", "p2", "p3", "p4"]
    assert got.var["filtered_probes"].dtype == bool
    assert list(got.var["filtered_probes"]) == [True, False, True, True]
    assert got.obs["filtered_barcodes"].dtype == bool
    assert list(got.obs["filtered_barcodes"]) == [i % 2 == 0 for i in range(N_CELLS)]


def _v2(tmp_path: Path, genomes=("hg19", "mm10")) -> Path:
    return write_10x_v2_h5(
        tmp_path / "v2.h5",
        {
            genome: (_counts(n_features=3, seed=i), _barcodes(), [f"{genome}_{j}" for j in range(3)], list("xyz"))
            for i, genome in enumerate(genomes)
        },
    )


def test_a_cell_ranger_2_file_needs_a_genome_when_it_has_several(tmp_path):
    h5 = _v2(tmp_path)
    result = _convert(h5, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "holds 2 genomes (hg19, mm10); pick one with --genome" in _out(result)

    out = tmp_path / "mm10.h5ad"
    _ok(h5, "-o", out, "--genome", "mm10")
    got = _read(out)
    assert list(got.var_names) == ["mm10_0", "mm10_1", "mm10_2"]
    assert list(got.var.columns) == ["gene_symbols"]
    assert list(got.var["gene_symbols"]) == list("xyz")
    assert np.array_equal(got.X.toarray(), _counts(n_features=3, seed=1))
    assert got.uns["cellranger"]["genome"] == "mm10"
    assert got.uns["cellranger"]["layout"] == 2


def test_a_cell_ranger_2_file_with_one_genome_needs_no_flag(tmp_path):
    out = tmp_path / "out.h5ad"
    _ok(_v2(tmp_path, genomes=("hg19",)), "-o", out)
    assert _read(out).shape == (N_CELLS, 3)


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--genome", "dm6"], "No genome 'dm6' in this file; it has: hg19, mm10"),
        (["--genome", "hg19", "--feature-type", "Gene Expression"], "records no feature types"),
    ],
)
def test_bad_cell_ranger_2_options_are_refused(tmp_path, extra, message):
    result = _convert(_v2(tmp_path), "-o", tmp_path / "o.h5ad", *extra)
    assert result.exit_code == 1
    assert message in _out(result)


def test_an_inconsistent_matrix_is_refused_before_writing(tenx, tmp_path):
    with h5py.File(tenx, "a") as f:
        del f["matrix/barcodes"]
        f["matrix/barcodes"] = np.array([b"x"] * 3)
    out = tmp_path / "out.h5ad"
    result = _convert(tenx, "-o", out)
    assert result.exit_code == 1
    assert "Inconsistent 10x matrix" in _out(result) and "3 barcodes" in _out(result)
    assert not out.exists()


def _spoil(path: Path, how: str) -> None:
    with h5py.File(path, "a") as f:
        m = f["matrix"]
        if how == "feature_type":
            del m["features/feature_type"]
            m["features/feature_type"] = np.array([b"Gene Expression"] * 3)
        elif how == "a tag":
            del m["features/pattern"]
            m["features/pattern"] = np.array([b""] * 5)
        elif how == "a listed tag":
            del m["features/pattern"]
        elif how == "gene_id":
            m["features/gene_id"] = np.array([b"E1"] * 2)
        elif how == "filtered_barcodes":
            m["filtered_barcodes"] = np.ones(N_CELLS - 1, dtype=bool)
        elif how == "indptr end":
            indptr = m["indptr"][...]
            indptr[-1] -= 1
            m["indptr"][...] = indptr
        elif how == "indptr start":
            indptr = m["indptr"][...]
            indptr[0] = 1
            m["indptr"][...] = indptr
        elif how == "indptr order":
            indptr = m["indptr"][...]
            indptr[1], indptr[2] = indptr[2] + 1, indptr[1]
            m["indptr"][...] = indptr


@pytest.mark.parametrize(
    "how,message",
    [
        ("feature_type", "3 feature feature_type"),
        ("a tag", "5 feature pattern"),
        ("a listed tag", "no feature pattern, which _all_tag_keys lists"),
        ("gene_id", "2 feature gene_id"),
        ("filtered_barcodes", f"{N_CELLS - 1} filtered_barcodes"),
        ("indptr end", "an indptr that ends at"),
        ("indptr start", "an indptr that does not start at 0"),
        ("indptr order", "an indptr that goes back"),
    ],
)
def test_a_matrix_inconsistent_anywhere_is_refused_before_writing(tenx, tmp_path, how, message):
    """F109: only the ids, names, barcodes and indptr length were checked, so
    a file malformed anywhere else was copied into a store anndata and
    scanpy cannot read, or failed deep in the copy."""
    _spoil(tenx, how)
    out = tmp_path / "out.h5ad"
    result = _convert(tenx, "-o", out)
    assert result.exit_code == 1
    assert "Inconsistent 10x matrix" in _out(result) and message in _out(result)
    assert not out.exists()


@pytest.mark.parametrize("bad", [len(IDS), -1])
@pytest.mark.parametrize("extra", [[], ["--feature-type", "Gene Expression"]])
def test_an_entry_outside_the_features_is_refused(tenx, tmp_path, bad, extra):
    """F109: an index past the features was copied into X unchecked, which
    scipy then reads out of bounds; under a feature filter a negative one
    was quietly taken from the end of the lookup."""
    with h5py.File(tenx, "a") as f:
        indices = f["matrix/indices"][...]
        indices[3] = bad
        f["matrix/indices"][...] = indices
    out = tmp_path / "out.h5ad"
    result = _convert(tenx, "-o", out, *extra)
    assert result.exit_code == 1
    assert f"an entry has feature index {bad}, outside its {len(IDS)} features" in _out(result)
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [tenx.name]


def test_a_feature_type_absent_from_the_genome_is_refused(tmp_path):
    """F101: --genome G with --feature-type T, where T exists but not in G,
    wrote an output of no features at all."""
    h5 = write_10x_h5(
        tmp_path / "barnyard.h5", _counts(n_features=4), barcodes=_barcodes(),
        ids=list("abcd"), names=list("ABCD"),
        types=["Gene Expression", "Gene Expression", "Antibody Capture", "Gene Expression"],
        genomes=["GRCh38", "mm10", "mm10", "mm10"],
    )
    out = tmp_path / "out.h5ad"
    result = _convert(h5, "-o", out, "--genome", "GRCh38", "--feature-type", "Antibody Capture")
    assert result.exit_code == 1
    assert (
        "No features of type 'Antibody Capture' in genome 'GRCh38'; "
        "its types are: Gene Expression"
    ) in _out(result)
    assert not out.exists()

    _ok(h5, "-o", out, "--genome", "mm10", "--feature-type", "Antibody Capture")
    assert list(_read(out).var_names) == ["c"]


def test_text_cell_ranger_escaped_reads_as_the_characters(tmp_path):
    """F110: Cell Ranger writes text outside ASCII as XML character
    references; they are read back as the characters they stand for."""
    h5 = write_10x_h5(
        tmp_path / "escaped.h5", _counts(n_features=3), barcodes=_barcodes(),
        ids=["a", "b", "c"], names=["IFN&#947;", "Gr&#xE9;", "A&B"],
    )
    out = tmp_path / "out.h5ad"
    _ok(h5, "-o", out, "--var-names", "gene_symbols")
    assert list(_read(out).var_names) == ["IFNγ", "Gré", "A&B"]


def test_a_10x_file_is_recognised_by_its_content(tenx, tmp_path):
    renamed = tenx.rename(tmp_path / "counts.dat")
    _ok(renamed, "-o", tmp_path / "out.h5ad")


def test_an_h5ad_is_not_mistaken_for_a_10x_file(tenx, tmp_path):
    h5ad = tmp_path / "from10x.h5ad"
    _ok(tenx, "-o", h5ad)
    # An .h5 output is a 10x file, and so a 10x source is a 10x-to-10x request.
    result = _convert(tenx, "-o", tmp_path / "copy.h5")
    assert result.exit_code == 1
    assert "Cannot convert 10x-h5 to 10x-h5 directly" in _out(result)
    # Whereas an h5ad named .h5 is still an h5ad.
    renamed = h5ad.rename(tmp_path / "anndata.h5")
    _ok(renamed, "-o", tmp_path / "back.zarr")


# ---------------------------------------------------------------------------
# export


def _datasets(path: Path) -> dict:
    found = {}
    with h5py.File(path) as handle:
        handle.visititems(
            lambda name, obj: found.__setitem__(name, (obj[...], obj.dtype))
            if isinstance(obj, h5py.Dataset) else None
        )
        found["/attrs"] = {k: handle.attrs[k] for k in handle.attrs}
        found["/keys"] = list(handle.keys())
    return found


def test_a_round_trip_reproduces_the_cell_ranger_file(tenx, tmp_path):
    h5ad, back = tmp_path / "x.h5ad", tmp_path / "back.h5"
    _ok(tenx, "-o", h5ad)
    _ok(h5ad, "-o", back)

    before, after = _datasets(tenx), _datasets(back)
    assert after["/keys"] == ["matrix"], "Seurat reads every root member as a genome"
    assert sorted(before) == sorted(after)
    for key in before:
        if key.startswith("/"):
            continue
        assert np.array_equal(before[key][0], after[key][0]), key
        assert before[key][1] == after[key][1], key
    assert {k: np.asarray(v).tolist() for k, v in before["/attrs"].items()} == {
        k: np.asarray(v).tolist() for k, v in after["/attrs"].items()
    }


def _flex(path: Path) -> Path:
    """A probe-barcode matrix laid out as Cell Ranger's (see conftest)."""
    genes = ["ENSG1", "ENSG1", "ENSG2", "ENSG3"]
    symbols = ["G1", "G1", "G2", "G3"]
    return write_10x_h5(
        path, _counts(n_features=4), barcodes=_barcodes(),
        ids=[f"{g}|{n}|h{i}" for i, (g, n) in enumerate(zip(genes, symbols))],
        names=[f"{n}|h{i}" for i, n in enumerate(symbols)],
        probe_gene_ids=genes,
        extra={"gene_name": symbols, "probe_region": ["spliced", "unspliced", "spliced", "none"]},
        untagged={"filtered_probes": [True, False, True, True]},
        filtered_barcodes=[i % 2 == 0 for i in range(N_CELLS)],
    )


@pytest.mark.parametrize(
    "flags",
    [[], ["--var-names", "gene_symbols"], ["--make-unique"], ["--var-names", "gene_symbols", "--make-unique"]],
    ids=["ids", "symbols", "ids-unique", "symbols-unique"],
)
@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_probe_level_matrix_round_trips_to_the_same_file(tmp_path, flags, suffix):
    """F045, F048: an imported Flex matrix exported its gene ids -- or, after
    --make-unique, suffixed ones -- as features/id, dropped the probe ids and
    gene_id, and wrote filtered_probes as 'True'/'False' strings, which read
    back as always true."""
    flex, h5ad, back = _flex(tmp_path / "flex.h5"), tmp_path / f"x{suffix}", tmp_path / "back.h5"
    _ok(flex, "-o", h5ad, *flags)
    _ok(h5ad, "-o", back)

    before, after = _datasets(flex), _datasets(back)
    assert sorted(before) == sorted(after)
    for key in before:
        if key.startswith("/"):
            continue
        assert np.array_equal(before[key][0], after[key][0]), key
        assert before[key][1] == after[key][1], key


def test_make_unique_keeps_the_gene_ids_it_suffixes(tmp_path):
    """F045: naming a Flex matrix's features by gene id with --make-unique
    suffixed the ids, and nothing kept the real ones."""
    out = tmp_path / "x.h5ad"
    _ok(_flex(tmp_path / "flex.h5"), "-o", out, "--make-unique")
    var = _read(out).var
    assert list(var.index) == ["ENSG1", "ENSG1-1", "ENSG2", "ENSG3"]
    assert list(var["gene_ids"]) == ["ENSG1", "ENSG1", "ENSG2", "ENSG3"]
    assert list(var.columns[:2]) == ["gene_symbols", "gene_ids"]

    plain = tmp_path / "plain.h5ad"
    _ok(write_10x_h5(
        tmp_path / "genes.h5", _counts(), barcodes=_barcodes(), ids=IDS, names=NAMES, types=TYPES,
    ), "-o", plain, "--make-unique")
    assert "gene_ids" not in _read(plain).var, "unique ids need nothing kept"


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
@pytest.mark.parametrize("compression", [None, "lzf"])
def test_export_writes_any_matrix_layout_as_cell_ranger_does(tmp_path, layout, compression):
    counts = _counts().astype(np.int32)
    X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](counts)
    source = tmp_path / "in.h5ad"
    ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=_barcodes()),
        var=pd.DataFrame({"gene_ids": IDS, "feature_types": TYPES}, index=NAMES),
    ).write_h5ad(source, compression=compression)

    out = tmp_path / "out.h5"
    _ok(source, "-o", out)
    with h5py.File(out) as f:
        m = f["matrix"]
        for key in ("data", "indices", "indptr", "barcodes"):
            assert m[key].compression == "gzip", f"{key} must be readable outside h5py"
        assert dict(m.attrs) == {}, "no AnnData attributes on a Cell Ranger group"
        assert m["barcodes"].dtype.kind == "S"
        assert m["features/id"].dtype.kind == "S"
        assert list(m["shape"][...]) == [len(IDS), N_CELLS]
        assert list(m["features/_all_tag_keys"][...]) == [b"genome"]
        assert f.attrs["filetype"] == "matrix"
        assert not [k for k in f.keys() if k != "matrix"]
    assert not list(tmp_path.glob("*scratch*"))

    got = _read(_ok_import(out, tmp_path))
    assert np.array_equal(got.X.toarray(), counts)
    assert list(got.var_names) == IDS
    assert list(got.var["gene_symbols"]) == NAMES
    assert list(got.var["feature_types"]) == TYPES
    assert list(got.obs_names) == _barcodes()


def _ok_import(h5: Path, tmp_path: Path) -> Path:
    out = tmp_path / f"{h5.stem}.reread.h5ad"
    _ok(h5, "-o", out)
    return out


def _anndata_source(tmp_path: Path, **var_columns) -> Path:
    counts = _counts().astype(np.float32)
    obj = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=_barcodes()),
        var=pd.DataFrame(var_columns or None, index=NAMES),
    )
    obj.layers["counts"] = sparse.csr_matrix(counts.astype(np.int32))
    obj.raw = obj
    path = tmp_path / "source.h5ad"
    obj.write_h5ad(path)
    return path


def test_export_refuses_to_invent_ids_and_types_unless_forced(tmp_path):
    source = _anndata_source(tmp_path)
    out = tmp_path / "out.h5"
    result = _convert(source, "-o", out)
    assert result.exit_code == 1
    assert "no gene ids" in _out(result) and "no feature types" in _out(result)
    assert "--force" in _out(result)
    assert not out.exists()

    result = _ok(source, "-o", out, "--force")
    assert "Writing placeholders" in _out(result)
    with h5py.File(out) as f:
        assert [v.decode() for v in f["matrix/features/id"][...]] == NAMES
        assert set(f["matrix/features/feature_type"][...]) == {b"Gene Expression"}


def test_export_warns_when_the_matrix_is_not_counts(tmp_path):
    source = _anndata_source(tmp_path, gene_ids=IDS, feature_types=TYPES)
    result = _ok(source, "-o", tmp_path / "x.h5")
    assert "X holds float32 values" in _out(result)

    result = _ok(source, "-o", tmp_path / "counts.h5", "--layer", "counts")
    assert "holds" not in _out(result)
    with h5py.File(tmp_path / "counts.h5") as f:
        assert f["matrix/data"].dtype == np.dtype("int32")


@pytest.mark.parametrize("layer", ["counts", "layers/counts", "raw/X"])
def test_layer_picks_the_matrix_to_export(tmp_path, layer):
    source = _anndata_source(tmp_path, gene_ids=IDS, feature_types=TYPES)
    out = tmp_path / "out.h5"
    _ok(source, "-o", out, "--layer", layer)
    got = _read(_ok_import(out, tmp_path))
    assert np.array_equal(got.X.toarray(), _counts())
    assert list(got.var_names) == IDS


def test_var_names_that_are_ids_export_with_their_symbols(tmp_path):
    source = _anndata_source(tmp_path, gene_symbols=NAMES, feature_types=TYPES)
    # Make the index the ids, as --var-names gene_ids leaves it.
    obj = ad.read_h5ad(source)
    obj.var_names = IDS
    obj.write_h5ad(source)

    out = tmp_path / "out.h5"
    _ok(source, "-o", out)
    with h5py.File(out) as f:
        assert [v.decode() for v in f["matrix/features/id"][...]] == IDS
        assert [v.decode() for v in f["matrix/features/name"][...]] == NAMES


def _ids_named_source(tmp_path: Path) -> Path:
    """var named by gene id, as `convert` imports by default, with more columns."""
    source = _anndata_source(
        tmp_path, gene_symbols=NAMES, feature_types=TYPES,
        alias=[n.lower() for n in NAMES], ensembl=[f"v2.{i}" for i in IDS],
    )
    obj = ad.read_h5ad(source)
    obj.var_names = IDS
    obj.write_h5ad(source)
    return source


@pytest.mark.parametrize(
    "flags,ids,names",
    [
        ([], IDS, NAMES),
        (["--gene-names-column", "alias"], IDS, [n.lower() for n in NAMES]),
        (["--gene-ids-column", "ensembl"], [f"v2.{i}" for i in IDS], NAMES),
    ],
    ids=["default", "names-column", "ids-column"],
)
def test_naming_one_column_keeps_var_names_that_are_ids(tmp_path, flags, ids, names):
    """F046: --gene-names-column turned off reading the var names as ids, so
    an export of a store named by id was refused for having no gene ids;
    --gene-ids-column made the ids the feature names."""
    source = _ids_named_source(tmp_path)
    out = tmp_path / "out.h5"
    _ok(source, "-o", out, *flags)
    with h5py.File(out) as f:
        assert [v.decode() for v in f["matrix/features/id"][...]] == ids
        assert [v.decode() for v in f["matrix/features/name"][...]] == names


def test_float16_counts_export_as_float32(tmp_path):
    """F049: float16 data was written as it was, and imported back into a
    float16 CSR that scipy, and so anndata, cannot read."""
    counts = _counts()
    source = tmp_path / "half.h5ad"
    var = pd.DataFrame({"gene_ids": IDS, "feature_types": TYPES}, index=NAMES)
    ad.AnnData(X=counts.astype(np.float16), obs=pd.DataFrame(index=_barcodes()), var=var).write_h5ad(source)
    obj = ad.read_h5ad(source)
    obj.X = sparse.csr_matrix(counts.astype(np.float32))
    obj.layers["csc"] = sparse.csc_matrix(counts.astype(np.float32))
    obj.write_h5ad(tmp_path / "wide.h5ad")
    # Narrow the sparse matrices' data on disk: anndata cannot write float16 sparse.
    with h5py.File(tmp_path / "wide.h5ad", "a") as f:
        for path in ("X/data", "layers/csc/data"):
            data = f[path][...].astype(np.float16)
            del f[path]
            f[path] = data

    for src, layer in ((source, "X"), (tmp_path / "wide.h5ad", "X"), (tmp_path / "wide.h5ad", "csc")):
        out = tmp_path / f"{src.stem}.{layer}.h5"
        _ok(src, "-o", out, "--layer", layer)
        with h5py.File(out) as f:
            assert f["matrix/data"].dtype == np.float32
        got = _read(_ok_import(out, tmp_path))
        assert got.X.dtype == np.float32
        assert np.array_equal(got.X.toarray(), counts)


def test_text_outside_ascii_exports_as_cell_ranger_writes_it(tmp_path):
    """F110: names and barcodes were written as UTF-8 bytes in strings typed
    ASCII, which scanpy decodes as ASCII and fails on. Cell Ranger writes
    XML character references; written so, they also read back as written."""
    names = ["IFNγ", "Gré", "A&B", "C", "D", "E", "F", "G"]
    barcodes = [f"é{b}" for b in _barcodes()]
    source = tmp_path / "text.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix(_counts().astype(np.int32)),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame({"gene_ids": IDS, "feature_types": TYPES}, index=names),
    ).write_h5ad(source)
    out = tmp_path / "out.h5"
    _ok(source, "-o", out)
    with h5py.File(out) as f:
        for key in ("features/name", "barcodes"):
            assert h5py.check_string_dtype(f[f"matrix/{key}"].dtype).encoding == "ascii"
        stored = [v.decode("ascii") for v in f["matrix/features/name"][...]]
        assert stored == [n.encode("ascii", "xmlcharrefreplace").decode() for n in names]
        assert stored[0] == "IFN&#947;"

    got = _read(_ok_import(out, tmp_path))
    assert list(got.var["gene_symbols"]) == names
    assert list(got.obs_names) == barcodes


def test_columns_can_be_named_for_export(tmp_path):
    source = _anndata_source(
        tmp_path, ensembl=IDS, symbol=NAMES, modality=TYPES,
    )
    out = tmp_path / "out.h5"
    _ok(
        source, "-o", out, "--gene-ids-column", "ensembl",
        "--gene-names-column", "symbol", "--feature-types-column", "modality",
    )
    with h5py.File(out) as f:
        assert [v.decode() for v in f["matrix/features/id"][...]] == IDS
        assert [v.decode() for v in f["matrix/features/feature_type"][...]] == TYPES

    result = _convert(source, "-o", tmp_path / "bad.h5", "--gene-ids-column", "nope")
    assert result.exit_code == 1
    assert "--gene-ids-column nope: no such var column" in _out(result)


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--layer", "nope"], "'layers/nope' not found"),
        (["--genome", "mm10"], "--genome does not apply when converting h5ad to 10x-h5"),
        (["--zarr-format", "2"], "--zarr-format does not apply"),
    ],
)
def test_bad_export_options_are_refused_before_writing(tmp_path, extra, message):
    source = _anndata_source(tmp_path, gene_ids=IDS, feature_types=TYPES)
    out = tmp_path / "out.h5"
    result = _convert(source, "-o", out, *extra)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


def test_a_cell_ranger_2_import_exports_as_v3_without_pytables_attributes(tmp_path):
    """Carried over, PYTABLES_FORMAT_VERSION makes Seurat read the file as v2."""
    h5ad, back = tmp_path / "v2.h5ad", tmp_path / "v3.h5"
    _ok(_v2(tmp_path, genomes=("hg19",)), "-o", h5ad)
    obj = ad.read_h5ad(h5ad)
    obj.var["feature_types"] = "Gene Expression"
    obj.write_h5ad(h5ad)

    _ok(h5ad, "-o", back)
    with h5py.File(back) as f:
        assert "PYTABLES_FORMAT_VERSION" not in f.attrs
        assert f.attrs["filetype"] == "matrix" and f.attrs["version"] == 2
        assert "features" in f["matrix"]


def test_a_zarr_source_exports_too(tenx, tmp_path, counts):
    zarr_path, back = tmp_path / "x.zarr", tmp_path / "back.h5"
    _ok(tenx, "-o", zarr_path)
    _ok(zarr_path, "-o", back)
    got = _read(_ok_import(back, tmp_path))
    assert np.array_equal(got.X.toarray(), counts)
    assert got.uns["cellranger"]["attrs"]["software_version"] == "cellranger-8.0.1"


# ---------------------------------------------------------------------------
# names


@pytest.mark.parametrize(
    "names",
    [
        ["a", "a", "b"],
        ["a", "a", "a-1", "a"],
        ["x-1", "x", "x", "x"],
        ["g", "g-1", "g", "g-1", "g"],
        ["a", "b", "c"],
    ],
)
def test_make_index_unique_matches_anndata(names):
    from adata.util.names import make_index_unique

    expected = list(ad.utils.make_index_unique(pd.Index(names)))
    assert make_index_unique(names) == expected

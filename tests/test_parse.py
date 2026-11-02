"""Tests for `adata convert` of Parse Biosciences split-pipe output.

Fixtures are written as split-pipe v1.6.1 writes a run (see conftest): the
combined `all-sample` folder and one per sample, each with DGE_filtered/
and DGE_unfiltered/. What anndata reads back is checked against what each
fixture was built from, and against Parse's own scanpy recipe in
test_external_readers.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from tests.conftest import write_10x_mtx, write_parse_run

ad = pytest.importorskip("anndata")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _check(got, dge, run, keep=None) -> None:
    genes = np.arange(len(run.ids)) if keep is None else np.asarray(keep)
    assert got.X.dtype == np.int32, "counts stay integers"
    assert np.array_equal(got.X.toarray(), dge.counts[:, genes])
    assert list(got.obs_names) == dge.barcodes
    assert list(got.var_names) == [run.ids[g] for g in genes]
    assert list(got.var["gene_symbols"]) == [run.names[g] for g in genes]
    assert list(got.var["genome"]) == [run.genomes[g] for g in genes]
    assert list(got.obs["sample"]) == dge.samples
    assert list(got.obs["species"]) == dge.species
    assert got.obs["sample"].dtype == "category"
    assert got.obs["bc1_well"].dtype == "category"
    assert got.obs["tscp_count"].dtype == np.int64
    assert got.obs["tscp_count"].tolist() == dge.counts.sum(axis=1).tolist()


# ---------------------------------------------------------------------------
# finding the folders


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("given", ["run", "sample", "dge"])
def test_a_run_a_sample_folder_and_a_dge_folder_read_the_same(tmp_path, given, suffix):
    run = write_parse_run(tmp_path / "run")
    source = {
        "run": run.root,
        "sample": run.root / "all-sample",
        "dge": run.root / "all-sample" / "DGE_filtered",
    }[given]
    out = tmp_path / f"out{suffix}"
    result = _ok(source, "-o", out)
    assert "Parse split-pipe (all-sample, filtered)" in _out(result)
    _check(_read(out), run.dge(), run)


def test_unfiltered_reads_every_barcode(tmp_path):
    run = write_parse_run(tmp_path / "run")
    out = tmp_path / "all.h5ad"
    _ok(run.root, "-o", out, "--unfiltered")
    _check(_read(out), run.dge(which="DGE_unfiltered"), run)


def test_sample_picks_one_and_all_samples_writes_each(tmp_path):
    run = write_parse_run(tmp_path / "run")
    out = tmp_path / "s2.h5ad"
    _ok(run.root, "-o", out, "--sample", "s2")
    _check(_read(out), run.dge("s2"), run)

    out = tmp_path / "each.h5ad"
    result = _ok(run.root, "-o", out, "--all-samples", "--unfiltered")
    assert "2 samples, one output each (s1, s2)" in _out(result)
    assert not out.exists()
    for sample in ("s1", "s2"):
        _check(_read(tmp_path / f"each.{sample}.h5ad"), run.dge(sample, "DGE_unfiltered"), run)


def test_a_gzipped_matrix_is_read(tmp_path):
    run = write_parse_run(tmp_path / "run", gz=True)
    out = tmp_path / "out.h5ad"
    _ok(run.root, "-o", out)
    _check(_read(out), run.dge(), run)


def test_gzipped_tables_are_read(tmp_path):
    """F108: a gzipped count_matrix.mtx was found, but gzipped
    all_genes.csv and cell_metadata.csv were not, so the folder was not
    recognised."""
    import gzip

    pd = pytest.importorskip("pandas")
    run = write_parse_run(tmp_path / "run", gz=True)
    dge = run.dge()
    for name in ("all_genes.csv", "cell_metadata.csv"):
        plain = dge.path / name
        (dge.path / f"{name}.gz").write_bytes(gzip.compress(plain.read_bytes()))
        plain.unlink()
    out = tmp_path / "parse.h5ad"
    _ok(dge.path, "-o", out)

    got = _read(out)
    assert np.array_equal(got.X.toarray(), dge.counts)
    # As Parse's recipe reads the tables: pandas, which inflates them itself.
    cells = pd.read_csv(dge.path / "cell_metadata.csv.gz").set_index("bc_wells")
    genes = pd.read_csv(dge.path / "all_genes.csv.gz")
    assert list(got.obs_names) == list(cells.index)
    assert list(got.obs["tscp_count"]) == list(cells["tscp_count"])
    assert list(got.var_names) == list(genes["gene_id"])


def test_a_run_without_samples_has_only_the_combined_folder(tmp_path):
    run = write_parse_run(tmp_path / "run", samples=())
    out = tmp_path / "out.h5ad"
    _ok(run.root, "-o", out)
    _check(_read(out), run.dge(), run)
    result = _convert(run.root, "-o", tmp_path / "x.h5ad", "--all-samples")
    assert result.exit_code == 1
    assert "split-pipe was run without samples" in _out(result)


# ---------------------------------------------------------------------------
# --genome on a barnyard run


def test_genome_keeps_that_genomes_genes_and_cells(tmp_path):
    run = write_parse_run(tmp_path / "run", genomes=("hg38", "mm10"), n_genes=6)
    out = tmp_path / "mouse.h5ad"
    result = _ok(run.root, "-o", out, "--genome", "mm10")
    got = _read(out)
    dge = run.dge()
    cells = [i for i, sp in enumerate(dge.species) if sp == "mm10"]
    genes = [g for g, genome in enumerate(run.genomes) if genome == "mm10"]
    assert f"{len(cells):,} cells x {len(genes):,} genes of mm10" in _out(result)
    assert np.array_equal(got.X.toarray(), dge.counts[np.ix_(cells, genes)])
    assert list(got.obs_names) == [dge.barcodes[i] for i in cells]
    assert list(got.var_names) == [run.ids[g] for g in genes]
    # As anndata subsets: categories no kept cell uses are dropped.
    assert list(got.obs["species"].cat.categories) == ["mm10"]


def test_genome_drops_cells_from_a_matrix_out_of_order_too(tmp_path, monkeypatch):
    """The cell filter is a row map applied to each entry, so it must hold
    on the bucket path as well as the straight-through one."""
    import adata.formats.mtx as mtx_module

    monkeypatch.setattr(mtx_module, "DEFAULT_CHUNK", 8)
    run = write_parse_run(tmp_path / "run", genomes=("hg38", "mm10"), n_cells=20)
    path = run.dge().path / "count_matrix.mtx"
    lines = path.read_text().splitlines()
    head, body = lines[:3], lines[3:]
    np.random.default_rng(1).shuffle(body)
    path.write_text("\n".join(head + body) + "\n")

    out = tmp_path / "mouse.h5ad"
    _ok(run.root, "-o", out, "--genome", "mm10")
    got = _read(out)
    dge = run.dge()
    cells = [i for i, sp in enumerate(dge.species) if sp == "mm10"]
    genes = [g for g, genome in enumerate(run.genomes) if genome == "mm10"]
    assert np.array_equal(got.X.toarray(), dge.counts[np.ix_(cells, genes)])
    assert got.X.has_sorted_indices


def test_an_unknown_genome_is_refused_with_those_present(tmp_path):
    run = write_parse_run(tmp_path / "run", genomes=("hg38", "mm10"))
    out = tmp_path / "x.h5ad"
    result = _convert(run.root, "-o", out, "--genome", "dm6")
    assert result.exit_code == 1
    assert "No genes of genome 'dm6'; this run has: hg38, mm10" in _out(result)
    assert not out.exists()


# ---------------------------------------------------------------------------
# var names


def test_var_names_can_be_the_symbols_made_unique(tmp_path):
    run = write_parse_run(tmp_path / "run")
    out = tmp_path / "symbols.h5ad"
    result = _ok(run.root, "-o", out, "--var-names", "gene_symbols")
    assert "var names repeat an earlier one" in _out(result)
    got = _read(out)
    assert list(got.var_names) == run.names
    assert list(got.var["gene_ids"]) == run.ids

    out = tmp_path / "unique.h5ad"
    _ok(run.root, "-o", out, "--var-names", "gene_symbols", "--make-unique")
    names = list(_read(out).var_names)
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# refusals


@pytest.mark.parametrize(
    "spoil,message",
    [
        ("cells", "cell_metadata.csv lists 11 cells, but count_matrix.mtx has 12 rows"),
        ("genes", "all_genes.csv lists 5 genes, but count_matrix.mtx has 6 columns"),
        ("index", "cell_metadata.csv has no 'bc_wells' column"),
    ],
)
def test_files_that_disagree_are_refused_before_writing(tmp_path, spoil, message):
    run = write_parse_run(tmp_path / "run")
    folder = run.dge().path
    name = "all_genes.csv" if spoil == "genes" else "cell_metadata.csv"
    lines = (folder / name).read_text().splitlines()
    if spoil == "index":
        lines[0] = lines[0].replace("bc_wells", "barcode")
    else:
        lines = lines[:-1]
    (folder / name).write_text("\n".join(lines) + "\n")
    out = tmp_path / "out.h5ad"
    result = _convert(folder, "-o", out)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


@pytest.mark.parametrize(
    "where,extra,message",
    [
        ("run", ["--sample", "s9"], "No sample 's9'; this run has: all-sample, s1, s2"),
        ("run", ["--sample", "s1", "--all-samples"], "pass one or the other"),
        ("dge", ["--unfiltered"], "is a DGE folder already, so --unfiltered"),
        ("sample", ["--sample", "s1"], "a single sample's folder"),
        ("parent", [], "holds 1 split-pipe runs (run); convert one at a time"),
    ],
)
def test_choices_that_cannot_be_met_are_refused(tmp_path, where, extra, message):
    run = write_parse_run(tmp_path / "parent" / "run")
    source = {
        "run": run.root,
        "dge": run.dge().path,
        "sample": run.root / "s1",
        "parent": tmp_path / "parent",
    }[where]
    result = _convert(source, "-o", tmp_path / "out.h5ad", *extra)
    assert result.exit_code == 1
    assert message in _out(result)


def test_a_run_missing_the_folder_asked_for_says_which_it_has(tmp_path):
    run = write_parse_run(tmp_path / "run")
    import shutil

    for sample in ("all-sample", "s1", "s2"):
        shutil.rmtree(run.root / sample / "DGE_filtered")
    result = _convert(run.root, "-o", tmp_path / "out.h5ad")
    assert result.exit_code == 1
    assert "all-sample has no readable DGE_filtered/; it has DGE_unfiltered/ (pass --unfiltered)" in _out(result)


def test_parse_flags_are_refused_for_other_formats(tmp_path):
    folder = write_10x_mtx(
        tmp_path / "mtx", np.ones((3, 2), dtype=int), barcodes=["a", "b", "c"],
        ids=["g1", "g2"], names=["G1", "G2"],
    )
    result = _convert(folder, "-o", tmp_path / "o.h5ad", "--unfiltered")
    assert result.exit_code == 1
    assert "does not apply" in _out(result)


# ---------------------------------------------------------------------------
# sniffing


def test_parse_and_10x_folders_are_told_apart(tmp_path):
    run = write_parse_run(tmp_path / "run")
    folder = write_10x_mtx(
        tmp_path / "mtx", np.ones((3, 2), dtype=int), barcodes=["a", "b", "c"],
        ids=["g1", "g2"], names=["G1", "G2"],
    )
    assert "Parse split-pipe" in _out(_ok(run.root, "-o", tmp_path / "p.h5ad"))
    assert "10x mtx" in _out(_ok(folder, "-o", tmp_path / "m.h5ad"))


def test_parse_output_cannot_be_written(tmp_path):
    run = write_parse_run(tmp_path / "run")
    store = tmp_path / "s.h5ad"
    _ok(run.root, "-o", store)
    result = _convert(store, "-o", tmp_path / "back", "--to", "parse")
    assert result.exit_code == 1
    assert "can be read but not written" in _out(result)

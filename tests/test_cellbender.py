"""Tests for `adata convert` of CellBender `remove-background` output.

Fixtures are written as CellBender writes its files (see conftest), in both
the 0.2 and 0.3+ layouts, full and filtered. What anndata reads back is
checked against what each fixture was built from: the denoised counts, and
each latent at the barcode it belongs to.
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
from tests.conftest import write_10x_h5, write_cellbender_h5

ad = pytest.importorskip("anndata")

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

N_DROPLETS = 40
IDS = [f"ENSG{i:05d}" for i in range(6)] + ["CD3", "CD4"]
NAMES = ["A", "B", "C", "D", "E", "F", "CD3_TotalSeqB", "CD4_TotalSeqB"]
TYPES = ["Gene Expression"] * 6 + ["Antibody Capture"] * 2


def _out(result) -> str:
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _barcodes(n: int = N_DROPLETS) -> List[str]:
    return [f"AAACCCAAG{i:07d}-1" for i in range(n)]


def _read(path: Path):
    return ad.read_zarr(path) if path.suffix == ".zarr" else ad.read_h5ad(path)


def _convert(*args):
    return runner.invoke(app, ["convert", *map(str, args)])


def _ok(*args):
    result = _convert(*args)
    assert result.exit_code == 0, _out(result)
    return result


def _raw_counts(seed: int = 0) -> np.ndarray:
    counts = np.random.default_rng(seed).poisson(1.5, (N_DROPLETS, len(IDS)))
    counts[-3:] = 0  # empty droplets
    return counts


def _denoised(raw: np.ndarray) -> np.ndarray:
    return np.maximum(raw - 1, 0)


def _cellbender(tmp_path: Path, **kwargs):
    raw = _raw_counts()
    name = f"cb{kwargs.get('version', '0.3')}{'f' if kwargs.get('filtered') else ''}.h5"
    return write_cellbender_h5(
        tmp_path / name, _denoised(raw), barcodes=_barcodes(), ids=IDS,
        names=NAMES, types=TYPES, **kwargs,
    )


def _raw_file(tmp_path: Path, **kwargs) -> Path:
    return write_10x_h5(
        tmp_path / "raw_feature_bc_matrix.h5", _raw_counts(),
        barcodes=kwargs.pop("barcodes", _barcodes()), ids=kwargs.pop("ids", IDS),
        names=NAMES, types=TYPES,
    )


def _expected(truth, values: np.ndarray) -> np.ndarray:
    """Per-latent-row values, at the barcodes the file holds."""
    if len(values) == len(truth.rows):
        return values
    full = np.full(N_DROPLETS, np.nan)
    full[truth.analyzed] = values
    return full[truth.rows]


# ---------------------------------------------------------------------------
# the mapping


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("filtered", [False, True], ids=["full", "filtered"])
@pytest.mark.parametrize("version", ["0.3", "0.2"])
def test_each_latent_lands_on_its_own_barcode(tmp_path, version, filtered, suffix):
    truth = _cellbender(tmp_path, version=version, filtered=filtered)
    out = tmp_path / f"out{suffix}"
    result = _ok(truth.path, "-o", out)
    assert f"CellBender {version}" in _out(result)

    got = _read(out)
    assert np.array_equal(got.X.toarray(), _denoised(_raw_counts())[truth.rows])
    assert list(got.obs_names) == [_barcodes()[i] for i in truth.rows]
    assert list(got.var_names) == IDS
    assert list(got.var["gene_symbols"]) == NAMES

    for key, values in truth.latents.items():
        np.testing.assert_array_equal(got.obs[key].to_numpy(), _expected(truth, values), key)
    analyzed = np.isin(truth.rows, truth.analyzed)
    assert got.obs["cellbender_analyzed"].tolist() == analyzed.tolist()

    encoding_key = "latent_gene_encoding" if version == "0.2" else "gene_expression_encoding"
    encoding = got.obsm[encoding_key]
    if filtered:
        np.testing.assert_array_equal(encoding, truth.encoding)
    else:
        position = {b: i for i, b in enumerate(truth.analyzed)}
        for row, barcode in enumerate(truth.rows):
            if barcode in position:
                np.testing.assert_array_equal(encoding[row], truth.encoding[position[barcode]])
            else:
                assert np.isnan(encoding[row]).all()

    np.testing.assert_allclose(got.var["ambient_expression"], truth.ambient)
    uns = got.uns["cellbender"]
    assert uns["version"] == version
    assert np.ravel(uns["target_false_positive_rate"]).tolist() == [0.01]
    if version == "0.3":
        assert "features_analyzed_inds" in uns, "metadata stays in uns"
        assert "features_analyzed_inds" not in got.var
        assert got.var["cellbender_analyzed"].all(), "as CellBender's anndata_from_h5 adds it"


@pytest.mark.parametrize("filtered", [False, True], ids=["full", "filtered"])
def test_latents_land_right_when_read_in_many_blocks(tmp_path, filtered, monkeypatch):
    """The embedding is sorted into output blocks through scratch, and the
    barcodes picked out of contiguous reads; tiny blocks make every row of
    the fixture cross a block boundary."""
    from adata.conversion import cellbender

    monkeypatch.setattr(cellbender, "_ROW_STEP", 3)
    monkeypatch.setattr(cellbender, "STRING_STEP", 7)
    truth = _cellbender(tmp_path, filtered=filtered)
    raw = _raw_file(tmp_path)
    for extra in ([], ["--min-cell-probability", "0.5"]):
        out = tmp_path / f"out{len(extra)}.h5ad"
        _ok(truth.path, "-o", out, "--raw", raw, *extra)
        got = _read(out)
        position = {b: i for i, b in enumerate(truth.analyzed)}
        latent_row = (
            np.arange(len(truth.rows)) if filtered
            else np.array([position.get(b, -1) for b in truth.rows])
        )
        index = {barcode: i for i, barcode in enumerate(_barcodes()[r] for r in truth.rows)}
        for row, barcode in enumerate(got.obs_names):
            source = latent_row[index[barcode]]
            expected = truth.encoding[source] if source >= 0 else np.full(truth.encoding.shape[1], np.nan)
            np.testing.assert_array_equal(got.obsm["gene_expression_encoding"][row], expected)
        rows = [_barcodes().index(b) for b in got.obs_names]
        assert np.array_equal(got.layers["cellranger"].toarray(), _raw_counts()[rows])


@pytest.mark.parametrize("filtered", [False, True], ids=["full", "filtered"])
@pytest.mark.parametrize("version", ["0.3", "0.2"])
def test_uns_holds_what_cellbender_stored_as_its_loader_gives_it(tmp_path, version, filtered):
    """Checked against real CellBender 0.3.2 output, uns differed from
    CellBender's own loader twice: one-element settings were unwrapped to
    scalars, so `uns[...][0]` broke, and a filtered file's
    barcode_indices_for_latents, which places no rows there, was dropped."""
    truth = _cellbender(tmp_path, version=version, filtered=filtered)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    uns = _read(out).uns["cellbender"]
    # A scalar in 0.2, a one-element array since.
    assert np.shape(uns["target_false_positive_rate"]) == (() if version == "0.2" else (1,))
    # 0.2 drops the index from filtered files; 0.3+ keeps every analysed droplet's.
    kept = version != "0.2" and filtered
    assert ("barcode_indices_for_latents" in uns) == kept


def test_cellbender_0_2_on_cell_ranger_2_input(tmp_path):
    truth = _cellbender(tmp_path, version="0.2", cellranger_v2=True)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert "latent_cell_probability" in got.obs
    assert "genome" not in got.uns["cellranger"], "background_removed is not a genome"


# ---------------------------------------------------------------------------
# sniffing


def test_a_cellbender_file_is_recognised_and_plain_10x_still_is_10x(tmp_path):
    truth = _cellbender(tmp_path)
    plain = _raw_file(tmp_path)
    out = tmp_path / "plain.h5ad"
    result = _ok(plain, "-o", out)
    assert "10x h5 v3" in _out(result)
    assert "cellbender" not in _read(out).uns

    out = tmp_path / "as10x.h5ad"
    _ok(truth.path, "-o", out, "--from", "10x-h5")
    got = _read(out)
    assert "cell_probability" not in got.obs
    assert np.array_equal(got.X.toarray(), _denoised(_raw_counts()))


# ---------------------------------------------------------------------------
# --min-cell-probability


@pytest.mark.parametrize("version", ["0.3", "0.2"])
def test_min_cell_probability_keeps_likely_cells_in_file_order(tmp_path, version):
    truth = _cellbender(tmp_path, version=version)
    key = "latent_cell_probability" if version == "0.2" else "cell_probability"
    expected = sorted(truth.analyzed[truth.latents[key] >= 0.5])

    out = tmp_path / "cells.h5ad"
    _ok(truth.path, "-o", out, "--min-cell-probability", "0.5")
    got = _read(out)
    assert list(got.obs_names) == [_barcodes()[i] for i in expected]
    assert (got.obs[key] >= 0.5).all()
    assert np.array_equal(got.X.toarray(), _denoised(_raw_counts())[expected])


def test_min_cell_probability_is_checked(tmp_path):
    truth = _cellbender(tmp_path)
    result = _convert(truth.path, "-o", tmp_path / "o.h5ad", "--min-cell-probability", "2")
    assert result.exit_code == 1
    assert "between 0 and 1" in _out(result)

    with h5py.File(truth.path, "a") as handle:
        del handle["droplet_latents/cell_probability"]
    result = _convert(truth.path, "-o", tmp_path / "o.h5ad", "--min-cell-probability", "0.5")
    assert result.exit_code == 1
    assert "no cell probabilities" in _out(result)
    assert not (tmp_path / "o.h5ad").exists()


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_a_probability_cut_that_keeps_no_droplet_is_refused(tmp_path, suffix):
    """F094: keeping no droplet crashed an h5ad output part-way, creating
    obsm with a chunk larger than its zero rows, and wrote a Zarr store of
    no cells."""
    truth = _cellbender(tmp_path)
    out = tmp_path / f"o{suffix}"
    result = _convert(truth.path, "-o", out, "--min-cell-probability", "1")
    assert result.exit_code == 1
    assert "No droplet has a cell probability of at least 1; the highest is 0.99" in _out(result)
    assert sorted(p.name for p in tmp_path.iterdir()) == [truth.path.name]


def test_embeddings_of_no_rows_are_written_on_every_backend(new_store, tmp_path):
    """F094: an embedding of no rows was created with a chunk of one row,
    which HDF5 refuses."""
    from adata.conversion.cellbender import _Latents, _write_embeddings

    path, open_store = new_store()
    latents = _Latents("0.3", {}, {"z": "z"}, {}, {}, n_latent=3, rows=None)
    with h5py.File(tmp_path / "latents.h5", "w") as handle:
        handle["z"] = np.ones((3, 4), dtype=np.float32)
        with open_store("a") as root:
            _write_embeddings(root, handle, latents, 3, np.array([], dtype=np.int64))
    with open_store("r") as root:
        assert root["obsm"]["z"].shape == (0, 4)


def test_from_cellbender_refuses_a_plain_10x_file(tmp_path):
    """F098: --from cellbender on a 10x matrix with no latents wrote a fake
    'CellBender 0.2' result."""
    plain = _raw_file(tmp_path)
    out = tmp_path / "o.h5ad"
    result = _convert(plain, "-o", out, "--from", "cellbender")
    assert result.exit_code == 1
    assert "holds no CellBender latents, so it is not CellBender output" in _out(result)
    assert "--from 10x-h5" in _out(result)
    assert not out.exists()


def test_cellbender_0_2_keeps_global_arrays_out_of_var_whatever_their_length(tmp_path):
    """F097: a 0.2 file's loss per epoch, as long as the features by chance,
    was written into var as if it were a property of each feature."""
    n_features = 20  # the fixture trains for 20 epochs
    counts = np.random.default_rng(3).poisson(1.0, (N_DROPLETS, n_features))
    truth = write_cellbender_h5(
        tmp_path / "cb.h5", counts, barcodes=_barcodes(),
        ids=[f"E{i}" for i in range(n_features)], names=[f"G{i}" for i in range(n_features)],
        version="0.2",
    )
    out = tmp_path / "o.h5ad"
    _ok(truth.path, "-o", out)
    got = _read(out)
    assert "training_elbo_per_epoch" not in got.var
    with h5py.File(truth.path) as f:
        np.testing.assert_array_equal(
            got.uns["cellbender"]["training_elbo_per_epoch"], f["matrix/training_elbo_per_epoch"][...]
        )
    np.testing.assert_allclose(got.var["ambient_expression"], truth.ambient)


# ---------------------------------------------------------------------------
# --raw


@pytest.mark.parametrize("filtered", [False, True], ids=["full", "filtered"])
def test_raw_adds_the_cell_ranger_counts_of_the_same_barcodes(
    tmp_path, filtered, monkeypatch
):
    """The full output lists the raw barcodes in their own order, so the raw
    counts stream straight across; the filtered one orders cells by count,
    so they go through the buckets."""
    import adata.core.csr_build as csr_build

    sorters = []
    original = csr_build.BucketSorter

    def watching(*args, **kwargs):
        sorters.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(csr_build, "BucketSorter", watching)
    truth = _cellbender(tmp_path, filtered=filtered)
    raw = _raw_file(tmp_path)
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out, "--raw", raw)
    assert bool(sorters) == filtered

    got = _read(out)
    assert np.array_equal(got.layers["cellranger"].toarray(), _raw_counts()[truth.rows])
    assert np.array_equal(got.layers["cellbender"].toarray(), got.X.toarray())
    assert got.layers["cellranger"].has_sorted_indices


def test_raw_follows_feature_filters_and_splits(tmp_path):
    truth = _cellbender(tmp_path, filtered=True)
    raw = _raw_file(tmp_path)

    out = tmp_path / "gex.h5ad"
    _ok(truth.path, "-o", out, "--raw", raw, "--feature-type", "Gene Expression")
    got = _read(out)
    assert np.array_equal(got.layers["cellranger"].toarray(), _raw_counts()[truth.rows][:, :6])
    np.testing.assert_allclose(got.var["ambient_expression"], truth.ambient[:6])

    out = tmp_path / "split.h5ad"
    _ok(truth.path, "-o", out, "--raw", raw, "--split-feature-types")
    adt = _read(tmp_path / "split.Antibody_Capture.h5ad")
    assert np.array_equal(adt.layers["cellranger"].toarray(), _raw_counts()[truth.rows][:, 6:])
    assert np.array_equal(adt.X.toarray(), _denoised(_raw_counts())[truth.rows][:, 6:])


def test_raw_buckets_are_sized_by_exact_counts_under_a_feature_filter(tmp_path, monkeypatch):
    """Bucket sizes used to be estimated as each row's length scaled by the
    kept share of features, which rounds to zero for a narrow filter: every
    entry then landed in one bucket, held whole in memory."""
    from adata.conversion import cellbender

    seen = []
    original = cellbender.write_runs

    def watching(blocks, **kwargs):
        seen.append(kwargs["counts"].copy())
        return original(blocks, **kwargs)

    monkeypatch.setattr(cellbender, "write_runs", watching)
    truth = _cellbender(tmp_path, filtered=True)
    raw = _raw_file(tmp_path)
    _ok(truth.path, "-o", tmp_path / "adt.h5ad", "--raw", raw, "--feature-type", "Antibody Capture")
    expected = (_raw_counts()[truth.rows][:, 6:] != 0).sum(axis=1)
    assert np.array_equal(seen[-1], expected)


@pytest.mark.parametrize(
    "spoil,message",
    [
        ("features", "same features"),
        ("barcodes", "are not in --raw"),
    ],
)
def test_a_raw_file_that_does_not_match_is_refused(tmp_path, spoil, message):
    truth = _cellbender(tmp_path, filtered=True)
    if spoil == "features":
        raw = _raw_file(tmp_path, ids=[i + "x" for i in IDS])
    else:
        raw = _raw_file(tmp_path, barcodes=[b.replace("AAA", "TTT") for b in _barcodes()])
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out, "--raw", raw)
    assert result.exit_code == 1
    assert message in _out(result)
    assert not out.exists()


@pytest.mark.parametrize("given", ["text", "h5ad"])
def test_raw_that_is_not_a_10x_matrix_is_refused_clearly(tmp_path, given):
    """F099: a --raw that was not HDF5 at all failed with h5py's 'file
    signature not found'."""
    truth = _cellbender(tmp_path, filtered=True)
    raw = tmp_path / "raw_feature_bc_matrix.h5"
    if given == "text":
        raw.write_text("barcode,count\n")
    else:
        _ok(truth.path, "-o", tmp_path / "cb.h5ad")
        (tmp_path / "cb.h5ad").rename(raw)
    out = tmp_path / "out.h5ad"
    result = _convert(truth.path, "-o", out, "--raw", raw)
    assert result.exit_code == 1
    assert "--raw raw_feature_bc_matrix.h5 is not a 10x HDF5 matrix" in _out(result)
    assert not out.exists()


def _layout(store: Path, path: str) -> dict:
    """dtype, chunks and compression of each array of a sparse matrix."""
    found = {}
    if store.suffix == ".zarr":
        import zarr

        group = zarr.open_group(store, mode="r")[path]
        for key in ("data", "indices", "indptr"):
            array = group[key]
            found[key] = (array.dtype, array.chunks, repr(array.compressors))
    else:
        with h5py.File(store) as f:
            for key in ("data", "indices", "indptr"):
                array = f[path][key]
                found[key] = (array.dtype, array.chunks, array.compression, array.compression_opts)
    return found


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
@pytest.mark.parametrize("filtered", [False, True], ids=["full", "filtered"])
def test_the_raw_layer_is_laid_out_as_x(tmp_path, suffix, filtered):
    """F131: layers["cellranger"] was written with int32 indices and default
    chunks and compression, where X and layers["cellbender"] keep the
    CellBender file's int64 indices and its layout."""
    truth = _cellbender(tmp_path, filtered=filtered)
    out = tmp_path / f"out{suffix}"
    _ok(truth.path, "-o", out, "--raw", _raw_file(tmp_path))
    x = _layout(out, "X")
    assert x["indices"][0] == np.int64
    assert _layout(out, "layers/cellranger") == x
    assert _layout(out, "layers/cellbender") == x
    got = _read(out)
    assert np.array_equal(got.layers["cellranger"].toarray(), _raw_counts()[truth.rows])


def test_float16_counts_import_as_float32_in_x_and_both_layers(tmp_path):
    """F049: float16 data, in the CellBender file or the raw one, was
    written into a float16 CSR, which anndata cannot read."""
    truth = _cellbender(tmp_path, filtered=True)
    raw = _raw_file(tmp_path)
    for path in (truth.path, raw):
        with h5py.File(path, "a") as f:
            data = f["matrix/data"][...].astype(np.float16)
            del f["matrix/data"]
            f["matrix/data"] = data
    out = tmp_path / "out.h5ad"
    _ok(truth.path, "-o", out, "--raw", raw)
    got = _read(out)
    for matrix in (got.X, got.layers["cellbender"], got.layers["cellranger"]):
        assert matrix.dtype == np.float32
    assert np.array_equal(got.X.toarray(), _denoised(_raw_counts())[truth.rows])
    assert np.array_equal(got.layers["cellranger"].toarray(), _raw_counts()[truth.rows])


def test_cellbender_flags_are_refused_for_other_formats(tmp_path):
    plain = _raw_file(tmp_path)
    result = _convert(plain, "-o", tmp_path / "o.h5ad", "--min-cell-probability", "0.5")
    assert result.exit_code == 1
    assert "does not apply" in _out(result)


def test_latents_of_mismatched_lengths_are_refused(tmp_path):
    truth = _cellbender(tmp_path)
    with h5py.File(truth.path, "a") as handle:
        del handle["droplet_latents/cell_size"]
        handle["droplet_latents"].create_dataset("cell_size", data=np.ones(3))
    result = _convert(truth.path, "-o", tmp_path / "o.h5ad")
    assert result.exit_code == 1
    assert "differ in length" in _out(result)


def test_latents_that_fit_no_barcode_are_refused(tmp_path):
    truth = _cellbender(tmp_path, version="0.2")
    with h5py.File(truth.path, "a") as handle:
        del handle["matrix/barcode_indices_for_latents"]
    result = _convert(truth.path, "-o", tmp_path / "o.h5ad")
    assert result.exit_code == 1
    assert "Cannot tell which barcode" in _out(result)


def test_cellbender_output_cannot_be_written(tmp_path):
    truth = _cellbender(tmp_path)
    store = tmp_path / "s.h5ad"
    _ok(truth.path, "-o", store)
    result = _convert(store, "-o", tmp_path / "x.h5", "--to", "cellbender")
    assert result.exit_code == 1
    assert "can be read but not written" in _out(result)
    # An .h5 output is still a 10x matrix.
    _ok(store, "-o", tmp_path / "x.h5")

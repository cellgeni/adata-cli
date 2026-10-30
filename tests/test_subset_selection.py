"""subset keeps exactly the rows named, and narrows every element kind.

Each test checks the output against anndata's own subsetting, or against
pandas where anndata cannot read the layout, on HDF5, Zarr v2 and Zarr v3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adata.elements.read import read_str_all
from adata.storage import open_store
from tests.conftest import backend_path
from tests.test_subset_oracle import _out, invoke, ok, read_anndata, write_anndata

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")


def _names(path: Path, names) -> Path:
    path.write_text("".join(f"{n}\n" for n in names))
    return path


# ---------------------------------------------------------------------------
# F014: a name shared by several rows


def _repeated_names() -> "ad.AnnData":
    obs_names = ["AAAC-1", "AAAG-1", "AAAC-1", "AAAT-1", "AAAG-1", "AAAC-1"]
    var_names = ["G1", "G2", "G1", "G3"]
    rng = np.random.default_rng(0)
    obj = ad.AnnData(
        sparse.csr_matrix(rng.poisson(1.0, (6, 4)).astype(np.float32)),
        obs=pd.DataFrame({"sample": pd.Categorical(list("aabbcc"))}, index=obs_names),
        var=pd.DataFrame(index=var_names),
    )
    obj.obsm["X_pca"] = rng.normal(size=(6, 2))
    raw_names = ["G4", "G1", "G2", "G1", "G3"]
    obj.raw = ad.AnnData(
        rng.random((6, 5)), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=raw_names)
    )
    return obj


def test_a_name_shared_by_several_rows_selects_every_one_of_them(tmp_path, backend):
    """F014: the name list kept only the first row with each name and stopped
    once every name had been seen, so after a concat without --index-unique
    (`AAAC-1` once per sample) `--obs barcodes.txt` kept only the first
    sample's cells, with exit 0. It now keeps what anndata's
    `obs_names.isin(names)` keeps, on both axes and in raw."""
    with pytest.warns(UserWarning):
        source = _repeated_names()
    src = write_anndata(source, tmp_path, backend)
    cells = _names(tmp_path / "cells.txt", ["AAAC-1", "AAAT-1", "absent"])
    genes = _names(tmp_path / "genes.txt", ["G1", "G3"])
    out = backend_path(tmp_path, backend, "out")

    result = ok("subset", src, "-o", out, "--obs", cells, "--var", genes)

    obs = source.obs_names.isin(["AAAC-1", "AAAT-1"])
    var = source.var_names.isin(["G1", "G3"])
    want = source[obs, var]
    got = read_anndata(out)
    assert list(got.obs_names) == list(want.obs_names)
    assert list(got.var_names) == list(want.var_names)
    np.testing.assert_array_equal(got.X.toarray(), want.X.toarray())
    np.testing.assert_array_equal(got.obsm["X_pca"], want.obsm["X_pca"])
    raw = source.raw.to_adata()[obs]
    raw = raw[:, raw.var_names.isin(["G1", "G3"])]
    assert list(got.raw.var_names) == list(raw.var_names)
    np.testing.assert_array_equal(got.raw.X, raw.X)

    text = _out(result)
    assert "Selected 4 obs (of 6)" in text
    assert "1 of the names match more than one obs row" in text
    assert "1 of the names match more than one var row" in text
    assert "1 obs names not found" in text


# ---------------------------------------------------------------------------
# F059: obsm and varm arrays of any rank


def _cubes() -> "ad.AnnData":
    rng = np.random.default_rng(1)
    obj = ad.AnnData(
        rng.random((6, 4)).astype(np.float32),
        obs=pd.DataFrame({"c": list("aabbcc")}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame({"k": list("xyxy")}, index=[f"g{j}" for j in range(4)]),
    )
    obj.obsm["three_d"] = rng.normal(size=(6, 2, 3))
    obj.obsm["four_d"] = rng.integers(0, 9, (6, 2, 1, 2)).astype(np.int16)
    obj.varm["three_d"] = rng.normal(size=(4, 3, 2)).astype(np.float32)
    return obj


@pytest.mark.parametrize(
    "selection,obs_rule,var_rule",
    [
        (("-q", "c == a"), lambda o: o.c == "a", None),
        (("--var-query", "k == y"), None, lambda v: v.k == "y"),
        (("-q", "c != b", "--var-query", "k == x"), lambda o: o.c != "b", lambda v: v.k == "x"),
    ],
)
def test_obsm_and_varm_arrays_of_any_rank_are_subset_on_their_first_axis(
    tmp_path, backend, selection, obs_rule, var_rule
):
    """F059: an obsm or varm array that was not 2-D was copied whole, so a
    (6, 2, 3) obsm stayed at 6 rows beside an X of 2, exit 0, and anndata
    refused to read the output ('Value passed for key three_d is of
    incorrect shape'); with --inplace the source became unreadable."""
    source = _cubes()
    src = write_anndata(source, tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")

    ok("subset", src, "-o", out, *selection)

    obs = obs_rule(source.obs).to_numpy() if obs_rule else slice(None)
    var = var_rule(source.var).to_numpy() if var_rule else slice(None)
    want = source[obs, var]
    got = read_anndata(out)
    for mapping, key in [("obsm", "three_d"), ("obsm", "four_d"), ("varm", "three_d")]:
        expected = getattr(want, mapping)[key]
        actual = getattr(got, mapping)[key]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_an_element_subset_cannot_narrow_is_refused_before_anything_is_written(
    tmp_path, backend
):
    """F059: an awkward array in obsm failed part way through the write with
    'Unsupported obsm:... encoding type'. It is refused, naming it, before
    the output exists; an existing output is left as it was."""
    src = write_anndata(_cubes(), tmp_path, backend)
    with open_store(src, "a") as store:
        group = store.root["obsm"].create_group("ragged")
        group.attrs["encoding-type"] = "awkward-array"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["length"] = 6
    out = backend_path(tmp_path, backend, "out")

    result = invoke("subset", src, "-o", out, "-q", "c == a")
    assert result.exit_code == 1
    assert "obsm/ragged" in _out(result)
    assert "awkward-array" in _out(result)
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [src.name]

    result = invoke("split", src, "--by", "c", "-o", tmp_path / "parts")
    assert result.exit_code == 1
    assert "obsm/ragged" in _out(result)
    assert not (tmp_path / "parts").exists()


# ---------------------------------------------------------------------------
# F123: a categorical group with no encoding attributes


def test_an_untagged_categorical_group_is_narrowed_with_the_rest_of_the_frame(
    tmp_path, backend
):
    """F123: a `codes` + `categories` group without `encoding-type` (which
    the reader accepts) was meant to be narrowed, but the test for it
    compared the missing attribute with None while it read as '', so the
    group was copied whole: the output frame was ragged, and exporting it
    gave the wrong label for every row past the first few."""
    labels = ["a", "a", "b", "b", "c", "c"]
    source = ad.AnnData(
        np.ones((6, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"c": pd.Categorical(labels), "n_counts": np.arange(6) * 500},
            index=[f"c{i}" for i in range(6)],
        ),
    )
    src = write_anndata(source, tmp_path, backend)
    with open_store(src, "a") as store:
        for key in ("encoding-type", "encoding-version", "ordered"):
            del store.root["obs/c"].attrs[key]
    out = backend_path(tmp_path, backend, "out")

    ok("subset", src, "-o", out, "-q", "n_counts >= 2000")

    rows = np.flatnonzero(source.obs["n_counts"].to_numpy() >= 2000)
    expected = pd.Categorical(labels)[rows].remove_unused_categories()
    with open_store(out, "r") as store:
        column = store.root["obs/c"]
        assert column["codes"].shape == (len(rows),)
        assert read_str_all(column) == list(expected.astype(str))
        assert read_str_all(column["categories"]) == list(expected.categories)


# ---------------------------------------------------------------------------
# F124: categorical columns of obsm and varm dataframes


def _frames() -> "ad.AnnData":
    obs_names = [f"c{i}" for i in range(4)]
    var_names = ["x", "y", "z"]
    obj = ad.AnnData(
        np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame({"keep": [False, True, False, False]}, index=obs_names),
        var=pd.DataFrame({"keep": [True, True, False]}, index=var_names),
    )
    obj.obsm["df"] = pd.DataFrame(
        {"c": pd.Categorical(list("pqrs"), categories=list("pqrst")), "v": np.arange(4.0)},
        index=obs_names,
    )
    obj.varm["df"] = pd.DataFrame(
        {"c": pd.Categorical(list("uvw"), categories=list("uvwz"))}, index=var_names
    )
    return obj


@pytest.mark.parametrize(
    "obs_query,var_query",
    [("keep == true", None), (None, "keep == true"), ("keep == true", "keep == true")],
)
def test_obsm_and_varm_dataframes_drop_unused_categories_as_anndata_does(
    tmp_path, backend, obs_query, var_query
):
    """F124: subset kept every category of an obsm or varm dataframe's
    categorical columns, unused ones included, where anndata's subsetting
    drops them, on the axis kept whole too."""
    source = _frames()
    src = write_anndata(source, tmp_path, backend)
    out = backend_path(tmp_path, backend, "out")

    selection = []
    if obs_query:
        selection += ["-q", obs_query]
    if var_query:
        selection += ["--var-query", var_query]
    ok("subset", src, "-o", out, *selection)

    obs = source.obs["keep"].to_numpy() if obs_query else slice(None)
    var = source.var["keep"].to_numpy() if var_query else slice(None)
    want = source[obs, var].copy()
    got = read_anndata(out)
    pd.testing.assert_frame_equal(got.obsm["df"], want.obsm["df"])
    pd.testing.assert_frame_equal(got.varm["df"], want.varm["df"])


def test_split_outputs_drop_unused_obsm_categories_as_anndata_does(tmp_path, backend):
    """F124: split, through the same writer, kept every category of obsm and
    varm dataframes in every output."""
    source = _frames()
    source.obs["group"] = pd.Categorical(["one", "two", "one", "two"])
    src = write_anndata(source, tmp_path, backend)
    ok("split", src, "--by", "group", "-o", tmp_path / "parts")
    for label in ("one", "two"):
        want = source[source.obs["group"] == label].copy()
        got = read_anndata(tmp_path / "parts" / f"{label}{src.suffix}")
        pd.testing.assert_frame_equal(got.obsm["df"], want.obsm["df"])
        pd.testing.assert_frame_equal(got.varm["df"], want.varm["df"])

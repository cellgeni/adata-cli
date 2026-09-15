"""Invariants that must hold regardless of the data, plus concat internals.

These assert relationships rather than specific values -- a subset of
everything is the original, splitting partitions the rows exactly, concat
undoes split -- which catches classes of bug that example-based tests walk
past.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app
from adata.core.info import format_type_info, get_entry_type
from adata.elements import spec
from adata.elements import write as ew
from adata.storage import open_store

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()


def _out(result) -> str:
    return result.stdout + (result.stderr or "")


def _build(path, n_obs=8, n_var=5, seed=0, layout="csr"):
    rng = np.random.default_rng(seed)
    X = sparse.random(
        n_obs, n_var, density=0.4, format="csr", dtype="float32", random_state=seed
    )
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(rng.choice(["a", "b", "c"], n_obs)),
            "score": rng.normal(size=n_obs).astype("float32"),
            "nullable": pd.array(
                [None if i % 3 == 0 else i for i in range(n_obs)], dtype="Int32"
            ),
            "text": [f"t{i}" for i in range(n_obs)],
        },
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"kind": pd.Categorical(["x"] * n_var)},
        index=[f"g{i}" for i in range(n_var)],
    )
    obj = ad.AnnData(X=X if layout == "csr" else X.tocsc(), obs=obs, var=var)
    obj.layers["counts"] = obj.X.copy()
    obj.obsm["X_umap"] = rng.normal(size=(n_obs, 2)).astype("float32")
    obj.write_h5ad(path)
    return obj


@pytest.fixture
def source(temp_dir):
    path = temp_dir / "src.h5ad"
    return path, _build(path)


# ---------------------------------------------------------------------------
# subset invariants


def test_subsetting_everything_is_the_identity(source, temp_dir):
    path, original = source
    names = temp_dir / "all.txt"
    names.write_text("\n".join(original.obs_names))
    out = temp_dir / "all.h5ad"

    assert runner.invoke(
        app, ["subset", str(path), "-o", str(out), "--obs", str(names)]
    ).exit_code == 0

    got = ad.read_h5ad(out)
    assert list(got.obs_names) == list(original.obs_names)
    assert list(got.var_names) == list(original.var_names)
    assert abs(got.X - original.X).nnz == 0
    assert got.obs["nullable"].tolist() == original.obs["nullable"].tolist()
    assert np.allclose(got.obsm["X_umap"], original.obsm["X_umap"])


def test_subsetting_is_idempotent(source, temp_dir):
    path, original = source
    keep = list(original.obs_names)[:4]
    names = temp_dir / "keep.txt"
    names.write_text("\n".join(keep))

    first = temp_dir / "one.h5ad"
    second = temp_dir / "two.h5ad"
    assert runner.invoke(
        app, ["subset", str(path), "-o", str(first), "--obs", str(names)]
    ).exit_code == 0
    assert runner.invoke(
        app, ["subset", str(first), "-o", str(second), "--obs", str(names)]
    ).exit_code == 0

    a, b = ad.read_h5ad(first), ad.read_h5ad(second)
    assert list(a.obs_names) == list(b.obs_names)
    assert abs(a.X - b.X).nnz == 0


def test_subset_preserves_source_row_order_not_the_name_files(source, temp_dir):
    """Selection is by membership; the store's own order is what survives."""
    path, original = source
    names = temp_dir / "keep.txt"
    names.write_text("c5\nc1\nc3\n")
    out = temp_dir / "o.h5ad"

    assert runner.invoke(
        app, ["subset", str(path), "-o", str(out), "--obs", str(names)]
    ).exit_code == 0
    assert list(ad.read_h5ad(out).obs_names) == ["c1", "c3", "c5"]


def test_a_query_and_the_equivalent_name_list_agree(source, temp_dir):
    path, original = source
    expected = [n for n, g in zip(original.obs_names, original.obs["group"]) if g == "a"]
    if not expected:
        pytest.skip("fixture has no rows in group 'a'")

    names = temp_dir / "keep.txt"
    names.write_text("\n".join(expected))
    by_name = temp_dir / "name.h5ad"
    by_query = temp_dir / "query.h5ad"

    assert runner.invoke(
        app, ["subset", str(path), "-o", str(by_name), "--obs", str(names)]
    ).exit_code == 0
    assert runner.invoke(
        app, ["subset", str(path), "-o", str(by_query), "-q", "group == a"]
    ).exit_code == 0

    a, b = ad.read_h5ad(by_name), ad.read_h5ad(by_query)
    assert list(a.obs_names) == list(b.obs_names)
    assert abs(a.X - b.X).nnz == 0


def test_unknown_names_are_reported_and_ignored(source, temp_dir):
    path, original = source
    names = temp_dir / "keep.txt"
    names.write_text("c0\nnot-a-cell\nc1\n")
    out = temp_dir / "o.h5ad"

    result = runner.invoke(
        app, ["subset", str(path), "-o", str(out), "--obs", str(names)]
    )
    assert result.exit_code == 0
    assert "not found" in _out(result)
    assert list(ad.read_h5ad(out).obs_names) == ["c0", "c1"]


# ---------------------------------------------------------------------------
# split invariants


def test_split_partitions_the_rows_exactly(source, temp_dir):
    """Every row lands in exactly one output, and nothing is invented."""
    path, original = source
    out_dir = temp_dir / "parts"
    assert runner.invoke(
        app, ["split", str(path), "--by", "group", "-o", str(out_dir)]
    ).exit_code == 0

    seen: list = []
    for part in sorted(out_dir.glob("*.h5ad")):
        piece = ad.read_h5ad(part)
        assert len(set(piece.obs["group"])) == 1, "a split must be homogeneous"
        seen.extend(piece.obs_names)

    assert sorted(seen) == sorted(original.obs_names)
    assert len(seen) == len(set(seen)), "no row may appear twice"


def test_split_then_concat_recovers_every_value(source, temp_dir):
    path, original = source
    out_dir = temp_dir / "parts"
    assert runner.invoke(
        app, ["split", str(path), "--by", "group", "-o", str(out_dir)]
    ).exit_code == 0

    parts = sorted(str(p) for p in out_dir.glob("*.h5ad"))
    merged = temp_dir / "merged.h5ad"
    assert runner.invoke(app, ["concat", *parts, "-o", str(merged)]).exit_code == 0

    got = ad.read_h5ad(merged)
    ref = original[list(got.obs_names)]
    assert got.shape == original.shape
    assert abs(got.X - ref.X).nnz == 0
    assert got.obs["nullable"].tolist() == ref.obs["nullable"].tolist()
    assert list(got.obs["group"]) == list(ref.obs["group"])
    assert np.allclose(got.obsm["X_umap"], ref.obsm["X_umap"])


# ---------------------------------------------------------------------------
# concat internals


def _two_stores(temp_dir, **kwargs):
    a = temp_dir / "a.h5ad"
    b = temp_dir / "b.h5ad"
    return a, b


def test_concat_pads_a_column_missing_from_one_input(temp_dir):
    """Outer join keeps the union of columns, padding where absent."""
    for name, extra in (("a", True), ("b", False)):
        obs = pd.DataFrame(
            {"shared": np.arange(2, dtype="int32")},
            index=[f"{name}{i}" for i in range(2)],
        )
        if extra:
            obs["only_a"] = np.arange(2, dtype="int32")
            obs["cat_only_a"] = pd.Categorical(["p", "q"])
            obs["text_only_a"] = ["u", "v"]
        ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=obs,
            var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"),
         "-o", str(out), "--join", "outer"],
    )
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(out)
    assert "only_a" in got.obs
    # Absent rows become missing, not zero.
    assert got.obs["only_a"].isna().tolist() == [False, False, True, True]
    assert got.obs["cat_only_a"].isna().tolist() == [False, False, True, True]
    assert got.obs["text_only_a"].isna().tolist()[2:] == [True, True]


def test_concat_inner_join_keeps_only_shared_columns(temp_dir):
    for name, extra in (("a", True), ("b", False)):
        obs = pd.DataFrame(
            {"shared": np.arange(2, dtype="int32")},
            index=[f"{name}{i}" for i in range(2)],
        )
        if extra:
            obs["only_a"] = np.arange(2, dtype="int32")
        ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=obs,
            var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    assert runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"), "-o", str(out)],
    ).exit_code == 0

    got = ad.read_h5ad(out)
    assert list(got.obs.columns) == ["shared"]


def test_concat_handles_dense_matrices(temp_dir):
    for name in ("a", "b"):
        ad.AnnData(
            X=np.arange(4, dtype="float32").reshape(2, 2),
            obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]),
            var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    assert runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"), "-o", str(out)],
    ).exit_code == 0

    got = ad.read_h5ad(out)
    assert got.X.shape == (4, 2)
    assert not sparse.issparse(got.X)


def test_concat_outer_join_fills_dense_gaps(temp_dir):
    ad.AnnData(
        X=np.ones((2, 2), dtype="float32"),
        obs=pd.DataFrame(index=["a0", "a1"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(temp_dir / "a.h5ad")
    ad.AnnData(
        X=np.full((2, 2), 5.0, dtype="float32"),
        obs=pd.DataFrame(index=["b0", "b1"]),
        var=pd.DataFrame(index=["g2", "g3"]),
    ).write_h5ad(temp_dir / "b.h5ad")

    out = temp_dir / "m.h5ad"
    assert runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"),
         "-o", str(out), "--join", "outer", "--fill-value", "-1"],
    ).exit_code == 0

    got = ad.read_h5ad(out)
    assert list(got.var_names) == ["g1", "g2", "g3"]
    assert got.X[0, 2] == -1, "a cell with no value gets the fill value"
    assert got.X[2, 0] == -1


def test_concat_skips_obsm_absent_from_an_input(temp_dir):
    for name, with_obsm in (("a", True), ("b", False)):
        obj = ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]),
            var=pd.DataFrame(index=["g1", "g2"]),
        )
        if with_obsm:
            obj.obsm["X_umap"] = np.zeros((2, 2), dtype="float32")
        obj.write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"), "-o", str(out)],
    )
    assert result.exit_code == 0, _out(result)
    assert "X_umap" in _out(result) and "Skipping" in _out(result)
    assert "X_umap" not in ad.read_h5ad(out).obsm


def test_concat_skips_obsm_whose_width_disagrees(temp_dir):
    for name, width in (("a", 2), ("b", 3)):
        obj = ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]),
            var=pd.DataFrame(index=["g1", "g2"]),
        )
        obj.obsm["X_umap"] = np.zeros((2, width), dtype="float32")
        obj.write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"), "-o", str(out)],
    )
    assert result.exit_code == 0, _out(result)
    assert "disagree on shape" in _out(result)


def test_concat_skips_a_layer_absent_from_an_input(temp_dir):
    for name, with_layer in (("a", True), ("b", False)):
        obj = ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]),
            var=pd.DataFrame(index=["g1", "g2"]),
        )
        if with_layer:
            obj.layers["counts"] = np.ones((2, 2), dtype="float32")
        obj.write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"), "-o", str(out)],
    )
    assert result.exit_code == 0, _out(result)
    assert "not present in every input" in _out(result)


def test_concat_refuses_to_overwrite(temp_dir):
    for name in ("a", "b"):
        ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]),
            var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(temp_dir / f"{name}.h5ad")

    out = temp_dir / "m.h5ad"
    out.write_text("in the way")
    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"), "-o", str(out)],
    )
    assert result.exit_code == 1
    assert "already exists" in _out(result)


def test_concat_rejects_an_unknown_join(temp_dir):
    for name in ("a", "b"):
        ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=pd.DataFrame(index=[f"{name}{i}" for i in range(2)]),
            var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(temp_dir / f"{name}.h5ad")

    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"),
         "-o", str(temp_dir / "m.h5ad"), "--join", "sideways"],
    )
    assert result.exit_code == 1
    assert "inner" in _out(result)


def test_concat_label_cannot_shadow_an_existing_column(temp_dir):
    for name in ("a", "b"):
        ad.AnnData(
            X=np.ones((2, 2), dtype="float32"),
            obs=pd.DataFrame(
                {"batch": pd.Categorical([name] * 2)},
                index=[f"{name}{i}" for i in range(2)],
            ),
            var=pd.DataFrame(index=["g1", "g2"]),
        ).write_h5ad(temp_dir / f"{name}.h5ad")

    result = runner.invoke(
        app,
        ["concat", str(temp_dir / "a.h5ad"), str(temp_dir / "b.h5ad"),
         "-o", str(temp_dir / "m.h5ad"), "--label", "batch"],
    )
    assert result.exit_code == 1
    assert "collides" in _out(result)


# ---------------------------------------------------------------------------
# type detection


def test_entry_types_are_reported_for_every_encoding(new_store):
    path, opener = new_store()
    with opener("a") as root:
        uns = root["uns"]
        ew.write_dense(uns, "arr", np.arange(4))
        ew.write_dense(uns, "matrix", np.zeros((2, 2)))
        ew.write_string_array(uns, "text", ["a", "b"])
        ew.write_categorical(uns, "cat", [0, 1], ["x", "y"])
        ew.write_scalar(uns, "num", 1)
        ew.write_scalar(uns, "str", "s")
        ew.write_null(uns, "nul")
        ew.write_sparse(uns, "sp", [1.0], [0], [0, 1], (1, 2))
        ew.write_masked(uns, "msk", [1], [False], spec.NULLABLE_INTEGER)

    expected = {
        "arr": "array",
        "matrix": "array",
        "text": "string-array",
        "cat": "categorical",
        "num": "scalar",
        "str": "scalar",
        "nul": "null",
        "sp": "sparse-matrix",
        "msk": "nullable-array",
    }
    with opener("r") as root:
        for key, kind in expected.items():
            info = get_entry_type(root["uns"][key])
            assert info["type"] == kind, f"{key} reported as {info['type']}"
            assert format_type_info(info).startswith("[")


def test_a_dict_containing_obs_names_is_not_a_dataframe(new_store):
    """Structural inference must not override a declared encoding."""
    path, opener = new_store()
    with opener("a") as root:
        group = ew.write_mapping(root["uns"], "trap")
        ew.write_string_array(group, "obs_names", ["a", "b"])
    with opener("r") as root:
        assert get_entry_type(root["uns"]["trap"])["type"] == "dict"


def test_the_root_is_reported_as_an_anndata_object(new_store):
    path, opener = new_store()
    with opener("r") as root:
        assert get_entry_type(root)["type"] == "anndata"


def test_categorical_detail_mentions_ordering(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_categorical(root["uns"], "c", [0], ["x"], ordered=True)
    with opener("r") as root:
        assert "ordered" in get_entry_type(root["uns"]["c"])["details"]


def test_masked_detail_mentions_the_na_value(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_masked(
            root["uns"], "m", ["a"], [True], spec.NULLABLE_STRING_ARRAY,
            na_value="NaN",
        )
    with opener("r") as root:
        assert "na-value=NaN" in get_entry_type(root["uns"]["m"])["details"]

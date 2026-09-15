"""Tests for create, split, query-based subset and concat.

These lean on anndata to build inputs and check outputs, for the same reason
as tests/test_anndata_roundtrip.py: the point is interoperability, not
self-consistency.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adata.cli import app

ad = pytest.importorskip("anndata", reason="anndata is required for these tests")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

runner = CliRunner()


_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _out(result) -> str:
    """Merged stdout+stderr with Rich markup and its line wrapping removed.

    Rich wraps long paths across lines, so a message can be split mid-sentence;
    collapsing whitespace lets assertions match the message as written.
    """
    text = result.stdout + (result.stderr or "")
    return " ".join(_ANSI.sub("", text).split())


def _make(
    path: Path,
    cells: list,
    genes: list,
    batch: str = "b",
    extra_col: str | None = None,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(1.0, (len(cells), len(genes))).astype("float32"))
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical([batch] * len(cells)),
            "score": np.arange(len(cells), dtype="float32"),
        },
        index=cells,
    )
    if extra_col:
        obs[extra_col] = np.arange(len(cells), dtype="int32")
    var = pd.DataFrame({"gene_type": ["protein"] * len(genes)}, index=genes)
    obj = ad.AnnData(X=X, obs=obs, var=var)
    obj.obsm["X_umap"] = rng.normal(size=(len(cells), 2)).astype("float32")
    obj.uns["shared"] = "same-everywhere"
    obj.uns["only_here"] = batch
    obj.write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# create


@pytest.mark.parametrize("fmt", ["h5ad", "zarr"])
def test_create_makes_a_valid_empty_store(tmp_path, fmt):
    out = tmp_path / f"new.{fmt}"
    result = runner.invoke(
        app, ["create", str(out), "--n-obs", "5", "--n-var", "3"]
    )
    assert result.exit_code == 0, _out(result)

    obj = ad.read_zarr(out) if fmt == "zarr" else ad.read_h5ad(out)
    assert obj.shape == (5, 3)
    assert list(obj.obs_names) == [f"cell_{i}" for i in range(5)]


def test_create_takes_names_from_files(tmp_path):
    (tmp_path / "cells.txt").write_text("c1\nc2\n")
    (tmp_path / "genes.txt").write_text("g1\ng2\ng3\n")
    out = tmp_path / "named.h5ad"

    result = runner.invoke(
        app,
        ["create", str(out),
         "--obs-names", str(tmp_path / "cells.txt"),
         "--var-names", str(tmp_path / "genes.txt")],
    )
    assert result.exit_code == 0, _out(result)

    obj = ad.read_h5ad(out)
    assert list(obj.obs_names) == ["c1", "c2"]
    assert list(obj.var_names) == ["g1", "g2", "g3"]


def test_create_refuses_to_clobber_without_force(tmp_path):
    out = tmp_path / "exists.h5ad"
    assert runner.invoke(app, ["create", str(out), "--n-obs", "2", "--n-var", "2"]).exit_code == 0
    result = runner.invoke(app, ["create", str(out), "--n-obs", "3", "--n-var", "2"])
    assert result.exit_code == 1
    assert "--force" in _out(result)


def test_create_rejects_conflicting_axis_size(tmp_path):
    (tmp_path / "cells.txt").write_text("c1\nc2\n")
    result = runner.invoke(
        app,
        ["create", str(tmp_path / "x.h5ad"), "--n-obs", "5",
         "--obs-names", str(tmp_path / "cells.txt"), "--n-var", "2"],
    )
    assert result.exit_code == 1
    assert "has 2 names" in _out(result)


def test_create_then_import_builds_a_usable_object(tmp_path):
    """The workflow the create command exists to enable."""
    out = tmp_path / "built.h5ad"
    np.save(tmp_path / "X.npy", np.arange(6, dtype="float32").reshape(3, 2))
    (tmp_path / "meta.csv").write_text("_index,sample\nc1,S1\nc2,S2\nc3,S1\n")
    (tmp_path / "cells.txt").write_text("c1\nc2\nc3\n")
    (tmp_path / "genes.txt").write_text("g1\ng2\n")

    steps = [
        ["create", str(out),
         "--obs-names", str(tmp_path / "cells.txt"),
         "--var-names", str(tmp_path / "genes.txt")],
        ["import", "array", str(out), "X", str(tmp_path / "X.npy"), "--inplace"],
        ["import", "dataframe", str(out), "obs", str(tmp_path / "meta.csv"),
         "--inplace", "-i", "_index"],
    ]
    for step in steps:
        result = runner.invoke(app, step)
        assert result.exit_code == 0, _out(result)

    obj = ad.read_h5ad(out)
    assert obj.shape == (3, 2)
    assert list(obj.obs["sample"]) == ["S1", "S2", "S1"]
    assert np.array_equal(obj.X, np.arange(6, dtype="float32").reshape(3, 2))


# ---------------------------------------------------------------------------
# subset --query


@pytest.fixture
def sample(tmp_path) -> Path:
    return _make(tmp_path / "sample.h5ad", ["c1", "c2", "c3", "c4"], ["g1", "g2"])


def test_subset_by_query(tmp_path, sample):
    out = tmp_path / "q.h5ad"
    result = runner.invoke(
        app, ["subset", str(sample), "-o", str(out), "-q", "score > 1"]
    )
    assert result.exit_code == 0, _out(result)
    assert list(ad.read_h5ad(out).obs_names) == ["c3", "c4"]


def test_subset_query_combines_conditions(tmp_path, sample):
    out = tmp_path / "q2.h5ad"
    result = runner.invoke(
        app,
        ["subset", str(sample), "-o", str(out),
         "-q", "score >= 1 and score < 3"],
    )
    assert result.exit_code == 0, _out(result)
    assert list(ad.read_h5ad(out).obs_names) == ["c2", "c3"]


def test_subset_var_query(tmp_path, sample):
    out = tmp_path / "qv.h5ad"
    result = runner.invoke(
        app,
        ["subset", str(sample), "-o", str(out), "--var-query", "gene_type == protein"],
    )
    assert result.exit_code == 0, _out(result)
    assert list(ad.read_h5ad(out).var_names) == ["g1", "g2"]


def test_subset_query_matching_nothing_is_an_error(tmp_path, sample):
    result = runner.invoke(
        app,
        ["subset", str(sample), "-o", str(tmp_path / "e.h5ad"), "-q", "score > 999"],
    )
    assert result.exit_code == 1
    assert "matched no rows" in _out(result)


def test_subset_query_on_unknown_column_is_reported(tmp_path, sample):
    result = runner.invoke(
        app,
        ["subset", str(sample), "-o", str(tmp_path / "e.h5ad"), "-q", "nope == 1"],
    )
    assert result.exit_code == 1
    assert "not found" in _out(result)


def test_subset_requires_some_selection(tmp_path, sample):
    result = runner.invoke(app, ["subset", str(sample), "-o", str(tmp_path / "e.h5ad")])
    assert result.exit_code == 1
    assert "--obs-query" in _out(result)


# ---------------------------------------------------------------------------
# split


def test_split_writes_one_store_per_value(tmp_path):
    src = tmp_path / "many.h5ad"
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {"group": pd.Categorical(["x", "y", "x", "z", "y", "x"])},
        index=[f"c{i}" for i in range(6)],
    )
    obj = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(1.0, (6, 3)).astype("float32")),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    obj.write_h5ad(src)

    out_dir = tmp_path / "split"
    result = runner.invoke(
        app, ["split", str(src), "--by", "group", "-o", str(out_dir)]
    )
    assert result.exit_code == 0, _out(result)

    assert {p.name for p in out_dir.glob("*.h5ad")} == {"x.h5ad", "y.h5ad", "z.h5ad"}
    assert list(ad.read_h5ad(out_dir / "x.h5ad").obs_names) == ["c0", "c2", "c5"]
    assert ad.read_h5ad(out_dir / "z.h5ad").shape == (1, 3)

    manifest = (out_dir / "many_manifest.csv").read_text().splitlines()
    assert manifest[0] == "id,column,value,anndatas,n"
    assert len(manifest) == 4


def test_split_dry_run_writes_nothing(tmp_path, sample):
    out_dir = tmp_path / "dry"
    result = runner.invoke(
        app, ["split", str(sample), "--by", "batch", "-o", str(out_dir), "--dry-run"]
    )
    assert result.exit_code == 0, _out(result)
    assert not out_dir.exists()


def test_split_min_size_skips_small_groups(tmp_path):
    src = tmp_path / "skew.h5ad"
    obs = pd.DataFrame(
        {"group": pd.Categorical(["big", "big", "big", "tiny"])},
        index=[f"c{i}" for i in range(4)],
    )
    ad.AnnData(
        X=np.ones((4, 2), dtype="float32"),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(src)

    out_dir = tmp_path / "split"
    result = runner.invoke(
        app,
        ["split", str(src), "--by", "group", "-o", str(out_dir), "--min-size", "2"],
    )
    assert result.exit_code == 0, _out(result)
    assert {p.name for p in out_dir.glob("*.h5ad")} == {"big.h5ad"}
    assert "Skipping" in _out(result)


def test_split_sanitises_labels_for_filenames(tmp_path):
    src = tmp_path / "messy.h5ad"
    obs = pd.DataFrame(
        {"group": ["a/b", "c d"]}, index=["c1", "c2"]
    )
    ad.AnnData(
        X=np.ones((2, 2), dtype="float32"),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(src)

    out_dir = tmp_path / "split"
    assert runner.invoke(
        app, ["split", str(src), "--by", "group", "-o", str(out_dir)]
    ).exit_code == 0
    assert {p.name for p in out_dir.glob("*.h5ad")} == {"a_b.h5ad", "c_d.h5ad"}


def test_split_unknown_column_is_reported(tmp_path, sample):
    result = runner.invoke(
        app, ["split", str(sample), "--by", "nope", "-o", str(tmp_path / "o")]
    )
    assert result.exit_code == 1
    assert "not found" in _out(result)


# ---------------------------------------------------------------------------
# concat


def test_split_then_concat_reconstructs_the_original(tmp_path):
    """The strongest end-to-end check available: the two must be inverses."""
    rng = np.random.default_rng(3)
    cells = [f"c{i}" for i in range(6)]
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(["x", "y", "x", "z", "y", "x"]),
            "score": np.arange(6, dtype="float32"),
            "nullable": pd.array([1, 2, None, 4, 5, None], dtype="Int32"),
        },
        index=cells,
    )
    original = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(1.0, (6, 3)).astype("float32")),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    original.obsm["X_umap"] = rng.normal(size=(6, 2)).astype("float32")
    src = tmp_path / "src.h5ad"
    original.write_h5ad(src)

    out_dir = tmp_path / "parts"
    assert runner.invoke(
        app, ["split", str(src), "--by", "group", "-o", str(out_dir)]
    ).exit_code == 0

    parts = sorted(str(p) for p in out_dir.glob("*.h5ad"))
    assert len(parts) == 3

    merged = tmp_path / "merged.h5ad"
    result = runner.invoke(app, ["concat", *parts, "-o", str(merged)])
    assert result.exit_code == 0, _out(result)

    got = ad.read_h5ad(merged)
    assert got.shape == original.shape
    assert sorted(got.obs_names) == sorted(original.obs_names)

    # Reorder the original to the concatenated order before comparing.
    ref = original[list(got.obs_names)]
    assert abs(got.X - ref.X).nnz == 0
    assert np.allclose(got.obsm["X_umap"], ref.obsm["X_umap"])
    assert list(got.obs["group"]) == list(ref.obs["group"])
    assert str(got.obs["group"].dtype) == "category"
    assert got.obs["nullable"].tolist() == ref.obs["nullable"].tolist()


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_matches_anndata(tmp_path, join):
    """Our output must agree with anndata.concat on the same inputs."""
    a = _make(tmp_path / "a.h5ad", ["c1", "c2"], ["g1", "g2", "g3"], batch="A", seed=1)
    b = _make(tmp_path / "b.h5ad", ["c1", "c3"], ["g2", "g3", "g4"], batch="B", seed=2)
    out = tmp_path / f"m_{join}.h5ad"

    result = runner.invoke(
        app,
        ["concat", str(a), str(b), "-o", str(out), "--join", join,
         "--label", "batch_id", "--keys", "A,B", "--index-unique", "-"],
    )
    assert result.exit_code == 0, _out(result)

    ours = ad.read_h5ad(out)
    theirs = ad.concat(
        [ad.read_h5ad(a), ad.read_h5ad(b)],
        join=join, label="batch_id", keys=["A", "B"], index_unique="-",
    )

    assert list(ours.var_names) == list(theirs.var_names)
    assert list(ours.obs_names) == list(theirs.obs_names)
    assert np.array_equal(
        np.asarray(ours.X.todense()), np.asarray(theirs.X.todense())
    )
    assert list(ours.obs["batch_id"]) == list(theirs.obs["batch_id"])


def test_concat_preserves_column_dtypes(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1", "c2"], ["g1", "g2"], batch="A", seed=1)
    b = _make(tmp_path / "b.h5ad", ["c3", "c4"], ["g1", "g2"], batch="B", seed=2)
    out = tmp_path / "m.h5ad"

    assert runner.invoke(app, ["concat", str(a), str(b), "-o", str(out)]).exit_code == 0

    obj = ad.read_h5ad(out)
    assert str(obj.obs["batch"].dtype) == "category"
    assert sorted(obj.obs["batch"].cat.categories) == ["A", "B"]
    assert str(obj.obs["score"].dtype) == "float32"
    assert "X_umap" in obj.obsm


def test_concat_uns_merge_keeps_only_agreeing_values(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1"], ["g1", "g2"], batch="A", seed=1)
    b = _make(tmp_path / "b.h5ad", ["c2"], ["g1", "g2"], batch="B", seed=2)
    out = tmp_path / "m.h5ad"

    result = runner.invoke(
        app, ["concat", str(a), str(b), "-o", str(out), "--uns-merge", "same"]
    )
    assert result.exit_code == 0, _out(result)

    uns = ad.read_h5ad(out).uns
    assert uns["shared"] == "same-everywhere"
    assert "only_here" not in uns, "values that differ must not survive 'same'"


def test_concat_drops_uns_by_default(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1"], ["g1"], batch="A")
    b = _make(tmp_path / "b.h5ad", ["c2"], ["g1"], batch="B")
    out = tmp_path / "m.h5ad"

    assert runner.invoke(app, ["concat", str(a), str(b), "-o", str(out)]).exit_code == 0
    assert dict(ad.read_h5ad(out).uns) == {}


def test_concat_warns_about_duplicate_obs_names(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1"], ["g1"], batch="A")
    b = _make(tmp_path / "b.h5ad", ["c1"], ["g1"], batch="B")
    result = runner.invoke(
        app, ["concat", str(a), str(b), "-o", str(tmp_path / "m.h5ad")]
    )
    assert result.exit_code == 0, _out(result)
    assert "not unique" in _out(result)


def test_concat_rejects_disjoint_vars_on_inner_join(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1"], ["g1", "g2"], batch="A")
    b = _make(tmp_path / "b.h5ad", ["c2"], ["g3", "g4"], batch="B")
    result = runner.invoke(
        app, ["concat", str(a), str(b), "-o", str(tmp_path / "m.h5ad")]
    )
    assert result.exit_code == 1
    assert "--join outer" in _out(result)


def test_concat_needs_two_inputs(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1"], ["g1"])
    result = runner.invoke(app, ["concat", str(a), "-o", str(tmp_path / "m.h5ad")])
    assert result.exit_code == 1
    assert "at least two" in _out(result)


def test_concat_rejects_an_unknown_merge_strategy(tmp_path):
    a = _make(tmp_path / "a.h5ad", ["c1"], ["g1"])
    b = _make(tmp_path / "b.h5ad", ["c2"], ["g1"])
    result = runner.invoke(
        app,
        ["concat", str(a), str(b), "-o", str(tmp_path / "m.h5ad"),
         "--uns-merge", "bogus"],
    )
    assert result.exit_code == 1
    assert "must be one of" in _out(result)

"""Output names from labels: never two for one file, and missing kept apart.

The stems come from `util.names.unique_names`, which split, convert's
`--split-feature-types` and Parse's `--all-samples` share.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from adata.util.names import unique_names
from tests.conftest import write_10x_h5
from tests.test_subset_oracle import _out, invoke, ok, read_anndata, write_anndata

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")


@pytest.mark.parametrize(
    "labels",
    [
        ["CD4+", "CD4", "CD4_1"],
        ["a ", "a_1", "a"],
        ["T", "t", "T cell", "t_cell", "t cell"],
        ["a/b", "a b", "a_b", "A_B_1"],
        ["NA", "", None, "+++"],
    ],
)
def test_unique_names_never_gives_two_labels_one_file(labels):
    """F044: a suffixed stem was not checked against the stems of later
    labels (`CD4+`, `CD4`, `CD4_1` gave `CD4`, `CD4_1`, `CD4_1`), and stems
    differing only in case (`T`, `t`) are one file on macOS and Windows.
    Both made two outputs share one path."""
    stems = unique_names(labels)
    assert set(stems) == set(labels)
    folded = [stem.casefold() for stem in stems.values()]
    assert len(set(folded)) == len(folded), stems
    assert all(stem and stem not in (".", "..") for stem in stems.values())


def test_a_label_keeps_its_own_name_when_nothing_else_claims_it():
    assert unique_names(["B cell", "T/NK", "x"]) == {"B cell": "B_cell", "T/NK": "T_NK", "x": "x"}


def test_feature_types_that_name_one_file_get_their_own_outputs(tmp_path):
    """F044 through convert: feature types `A+`, `A` and `A_1` were given
    `m.A`, `m.A_1` and `m.A_1`, and the conversion failed with a false
    'another adata run is writing' message about its own lock."""
    counts = np.arange(12, dtype=np.int32).reshape(4, 3) % 3
    h5 = write_10x_h5(
        tmp_path / "m.h5", counts, barcodes=[f"b{i}-1" for i in range(4)],
        ids=["f0", "f1", "f2"], names=["f0", "f1", "f2"], types=["A+", "A", "A_1"],
    )
    ok("convert", h5, "-o", tmp_path / "m.h5ad", "--split-feature-types")
    names = {
        p.name: list(read_anndata(p).var_names) for p in tmp_path.glob("m.*.h5ad")
    }
    assert names == {"m.A.h5ad": ["f0"], "m.A_1.h5ad": ["f1"], "m.A_1_1.h5ad": ["f2"]}


def test_split_labels_that_sanitise_alike_or_differ_in_case_get_their_own_stores(
    tmp_path, backend
):
    labels = ["CD4+", "CD4", "CD4_1", "T", "t", "CD4+"]
    obs = pd.DataFrame({"grp": pd.Categorical(labels)}, index=[f"c{i}" for i in range(6)])
    source = ad.AnnData(np.arange(12, dtype=np.float32).reshape(6, 2), obs=obs)
    src = write_anndata(source, tmp_path, backend)
    parts = tmp_path / "parts"

    dry = invoke("split", src, "--by", "grp", "-o", parts, "--dry-run")
    assert dry.exit_code == 0, _out(dry)
    ok("split", src, "--by", "grp", "-o", parts)

    stems = {"CD4+": "CD4", "CD4": "CD4_1", "CD4_1": "CD4_1_1", "T": "T", "t": "t_1"}
    listed = "".join(_out(dry).split())  # Rich wraps the long paths
    for label, stem in stems.items():
        path = parts / f"{stem}{src.suffix}"
        assert f"parts/{path.name}(" in listed
        assert list(read_anndata(path).obs_names) == list(obs.index[obs.grp == label])


def test_split_writes_missing_labels_and_empty_labels_to_different_stores(
    tmp_path, backend
):
    """F122: missing values read as '', so split wrote the rows with no label
    and the rows labelled '' into one NA store, listed once in the manifest.
    Missing values now get their own store, NA, as groupby(dropna=False)
    gives them their own group."""
    labels = [None, "", None, "x", "", "x"]
    obs = pd.DataFrame(
        {"grp": pd.Categorical(labels), "txt": pd.array(labels, dtype="string")},
        index=[f"c{i}" for i in range(6)],
    )
    source = ad.AnnData(np.ones((6, 2), dtype=np.float32), obs=obs)
    src = write_anndata(source, tmp_path, backend)

    for column in ("grp", "txt"):
        parts = tmp_path / f"parts-{column}"
        result = ok("split", src, "--by", column, "-o", parts)
        assert "missing" in _out(result)
        names = {"NA": obs[column].isna(), "NA_1": obs[column] == "", "x": obs[column] == "x"}
        for stem, rows in names.items():
            got = read_anndata(parts / f"{stem}{src.suffix}")
            assert list(got.obs_names) == list(obs.index[rows.fillna(False).to_numpy(bool)])
        with open(parts / f"{src.stem}_manifest.csv", newline="") as fh:
            manifest = list(csv.DictReader(fh))
        assert [(row["value"], row["n"]) for row in manifest] == [("", "2"), ("", "2"), ("x", "2")]
        assert [row["anndatas"].rsplit("/", 1)[-1] for row in manifest] == [
            f"NA{src.suffix}", f"NA_1{src.suffix}", f"x{src.suffix}"
        ]

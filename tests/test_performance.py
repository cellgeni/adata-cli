"""Complexity guards: what an operation *costs*, never how long it takes.

Why this file exists
--------------------
`concat --merge` hung for hours on a 36,601-var input (REQ-71798). One line in
`_write_var` re-read a whole var column from disk once per target variable.
The output would have been correct; only the cost was wrong. 1,029 tests
missed it, and they were never going to catch it: every fixture in the suite
is at most a few hundred elements, and nothing measured cost at all.

So this file closes a defect class, not a bug. The instruments are in
`tests/perf_counters.py`; the rule they enforce is that cost must grow no
faster than linearly in each axis, measured in operations rather than seconds.

Why not wall clock
------------------
A test that can fail because a CI runner was busy does not belong in a merge
gate. Every number here is deterministic: the same input produces the same
count on every machine. That is what lets these run in the ordinary test job
on both interpreters and block a merge when they fail.

Reading a failure
-----------------
A failure here says cost grew super-linearly on the named axis. Usually the
code is wrong. Occasionally the expectation is -- an operation legitimately
gains work -- and then the new number needs a comment saying why, in the same
style as the exact counts in `test_commands_phase2.py`.

One axis at a time. Scaling two at once makes legitimate work look quadratic:
an outer-join concat really does produce n_obs x n_var_union cells.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
from rich.console import Console

from adata.commands.create import create_store
from adata.commands.export import (
    export_json,
    export_mtx,
    export_npy,
    export_table,
)
from adata.commands.import_data import import_object
from adata.commands.info import show_info
from adata.commands.ls import list_store
from adata.core.concat import concat_on_disk
from adata.core.subset import subset_h5ad

from tests.perf_counters import (
    GROWTH_LIMIT,
    SIZES,
    assert_grows_linearly,
    assert_grows_slower_than_input,
    assert_independent_of,
    count_allocations,
    count_scanned_elements,
    count_io,
    count_lines,
)

ad = pytest.importorskip("anndata", reason="anndata builds the fixtures")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

QUIET = Console(quiet=True)

#: Only Python executed inside the package counts towards the CPU proxy.
SOURCE_PREFIX = str(pathlib.Path(__file__).resolve().parent.parent / "src" / "adata")


# ---------------------------------------------------------------------------
# fixtures built to scale exactly one axis


def _store(
    path: Path,
    *,
    name: str,
    n_obs: int = 4,
    n_var: int = 8,
    n_var_columns: int = 2,
    n_obs_columns: int = 0,
    n_categories: int = 0,
    obs_kind: Optional[str] = None,
    shared_var: bool = True,
    frames: bool = False,
) -> Path:
    """One input store. Every argument is an axis a guard can scale."""
    obs = pd.DataFrame(index=[f"{name}c{i}" for i in range(n_obs)])
    if n_categories:
        obs["ct"] = pd.Categorical(
            [f"{name}-t{i % n_categories}" for i in range(n_obs)]
        )
    if obs_kind == "numeric":
        obs["col"] = np.arange(n_obs, dtype="int32")
    elif obs_kind == "masked":
        obs["col"] = pd.array(
            [i if i % 2 else None for i in range(n_obs)], dtype="Int32"
        )
    elif obs_kind == "string":
        obs["col"] = [f"{name}-{i}" for i in range(n_obs)]
    if n_obs_columns:
        extra = pd.DataFrame(
            {f"m{c}": np.arange(n_obs, dtype="int32") for c in range(n_obs_columns)},
            index=obs.index,
        )
        obs = pd.concat([obs, extra], axis=1)

    prefix = "" if shared_var else name
    var = pd.DataFrame(
        {f"v{c}": [f"{c}-{i}" for i in range(n_var)] for c in range(n_var_columns)},
        index=[f"{prefix}g{i}" for i in range(n_var)],
    )

    obj = ad.AnnData(
        X=sparse.csr_matrix(np.ones((n_obs, n_var), dtype="float32")),
        obs=obs,
        var=var,
    )
    obj.obsm["X_pca"] = np.zeros((n_obs, 3), dtype="float32")
    if frames:
        # What a SCope import leaves: dataframes in obsm and varm, and an array in varm.
        obj.obsm["RegulonsAUC"] = pd.DataFrame(
            {"r0": np.arange(n_obs, dtype="float64"), "r1": np.zeros(n_obs)}, index=obs.index
        )
        obj.varm["Regulons"] = pd.DataFrame(
            {"r0": np.arange(n_var) % 2, "r1": np.ones(n_var, dtype="int64")}, index=var.index
        )
        obj.varm["topic_region"] = np.ones((n_var, 3))
    obj.write_h5ad(path)
    return path


def _inputs(tmp_path: Path, tag: str, count: int = 2, **kwargs) -> List[Path]:
    directory = tmp_path / tag
    directory.mkdir(parents=True, exist_ok=True)
    return [
        _store(directory / f"{chr(ord('a') + i)}.h5ad", name=chr(ord("a") + i), **kwargs)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# canary
#
# Every growth guard below is a comparison between measurements. If a refactor
# moves a read onto a path the counters cannot see -- `read_direct`,
# `np.asarray(dataset)`, `asstr()[...]`; see perf_counters' docstring -- the
# comparisons would all pass on zeros. The vacuity check inside
# `assert_grows_linearly` catches most of that; this catches the rest, and
# says plainly what broke.


def test_the_counters_see_a_known_operation(tmp_path):
    """A concat of two 4 x 64 stores must register substantial work.

    If this fails, the hooks have stopped matching the libraries and every
    other test in this file has become meaningless. Fix the hooks first.
    """
    files = _inputs(tmp_path, "canary", n_obs=4, n_var=64)
    with count_io() as io:
        concat_on_disk(files, tmp_path / "out.h5ad", QUIET, merge="same")

    assert io.h5_calls > 0, "h5py.Dataset.__getitem__ hook is not firing"
    assert io.h5_elements >= 64, f"suspiciously few elements read: {io}"

    with count_allocations() as peak:
        concat_on_disk(files, tmp_path / "out2.h5ad", QUIET, merge="same")
    assert peak[0] > 0, "tracemalloc recorded no allocation"

    with count_lines(SOURCE_PREFIX) as lines:
        concat_on_disk(files, tmp_path / "out3.h5ad", QUIET, merge="same")
    assert lines[0] > 100, f"line tracer saw only {lines[0]} events in src/adata"


def test_the_zarr_store_hooks_fire(tmp_path):
    """Metadata and chunk traffic must be visible, not just array reads.

    An attribute-write storm never touches `Array.__getitem__` -- it is all
    `LocalStore.set`. That shape of bug has bitten this repo before, so the
    store hooks get their own canary.
    """
    files = _inputs(tmp_path, "zcanary", n_obs=4, n_var=32)
    with count_io() as io:
        concat_on_disk(files, tmp_path / "out.zarr", QUIET, merge="same")

    assert io.store_set > 0, "LocalStore.set hook is not firing"
    assert io.store_get > 0, "LocalStore.get hook is not firing"


# ---------------------------------------------------------------------------
# concat: reads must not grow faster than linearly on any axis
#
# The n_var case is the direct regression guard for REQ-71798. The others
# exist because concat's cost surface is n_obs x n_var x n_columns x n_inputs
# and a defect can hide in any of them; scaling only the axis that broke last
# time would be fighting the last war.


@pytest.mark.parametrize("merge", [None, "same", "unique", "first", "only"])
def test_concat_reads_grow_linearly_in_var_count(tmp_path, merge):
    """The REQ-71798 axis. Every merge strategy, not just the one reported."""

    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"var{merge}{n}", n_var=n)
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"o{merge}{n}.h5ad", QUIET, merge=merge)
        return io.reads

    assert_grows_linearly(measure, what=f"concat --merge {merge}", axis="n_var")


@pytest.mark.parametrize("merge", ["same", "unique", "first", "only"])
def test_concat_varm_reads_grow_linearly_in_var_count(tmp_path, merge):
    """varm is merged as var's columns are: read once per input, not per gene."""

    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"varm{merge}{n}", n_var=n, frames=True)
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"om{merge}{n}.h5ad", QUIET, merge=merge)
        return io.reads

    assert_grows_linearly(measure, what=f"concat varm --merge {merge}", axis="n_var")


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_obsm_frames_grow_linearly_in_obs_count(tmp_path, join):
    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"obsmf{join}{n}", n_obs=n, n_var=4, frames=True)
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"of{join}{n}.h5ad", QUIET, join=join)
        return io.reads

    assert_grows_linearly(measure, what=f"concat obsm dataframes ({join})", axis="n_obs")


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_reads_grow_linearly_in_obs_count(tmp_path, join):
    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"obs{join}{n}", n_obs=n, n_var=4)
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"o{join}{n}.h5ad", QUIET, join=join)
        return io.reads

    assert_grows_linearly(measure, what=f"concat --join {join}", axis="n_obs")


def test_concat_reads_grow_linearly_in_input_count(tmp_path):
    """Pairwise work across inputs would show up here and nowhere else."""

    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"in{n}", count=n, n_obs=2, n_var=4)
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"oin{n}.h5ad", QUIET, merge="same")
        return io.reads

    # Sizes are scaled down: n inputs means n files on disk.
    assert_grows_linearly(
        measure, what="concat", axis="n_inputs", sizes=(4, 16, 64)
    )


def test_concat_reads_grow_linearly_in_obs_column_count(tmp_path):
    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"oc{n}", n_obs=8, n_var=4, n_obs_columns=n)
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"ooc{n}.h5ad", QUIET)
        return io.reads

    assert_grows_linearly(measure, what="concat", axis="n_obs_columns")


def test_concat_reads_grow_linearly_in_var_column_count(tmp_path):
    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"vc{n}", n_obs=4, n_var=8, n_var_columns=n)
        with count_io() as io:
            concat_on_disk(tmp_path / f"vc{n}" and files, tmp_path / f"ovc{n}.h5ad",
                           QUIET, merge="same")
        return io.reads

    assert_grows_linearly(measure, what="concat --merge same", axis="n_var_columns")


# ---------------------------------------------------------------------------
# concat: the category union
#
# This one needs its own instrument. Merging categories reads each category
# list exactly once however the union is computed, so a read counter sees
# nothing; and `x not in some_list` is a single bytecode, so the line tracer
# sees nothing either -- the quadratic lives inside C-level list membership.
# Counting string comparisons is the only thing that makes it visible.


class _CountingStr(str):
    """A string that tallies its own comparisons.

    `list.__contains__` rich-compares each element, and a subclass's `__eq__`
    takes priority, so the tally reflects real membership-probe cost.
    """

    count = 0

    def __eq__(self, other: object) -> bool:
        _CountingStr.count += 1
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return str.__hash__(self)


def _count_category_comparisons(monkeypatch, tmp_path, k: int) -> int:
    import adata.core.concat as concat_module

    original = concat_module._categories
    monkeypatch.setattr(
        concat_module,
        "_categories",
        lambda col, frame=None: np.asarray(
            [_CountingStr(c) for c in original(col, frame)], dtype=object
        ),
    )

    files = _inputs(tmp_path, f"cat{k}", n_obs=k, n_var=2, n_categories=k)
    _CountingStr.count = 0
    concat_on_disk(files, tmp_path / f"ocat{k}.h5ad", QUIET)
    return _CountingStr.count


def test_concat_unions_categories_without_quadratic_comparisons(
    tmp_path, monkeypatch
):
    """Category merging must use hash lookup, not list membership.

    `if category not in categories` on a list was O(k) per probe and so
    O(k^2) over the union: 2,096,128 string comparisons at k=1024, and around
    5e9 for an obs column with 100k categories -- the same silent hang as
    REQ-71798, in a different function.

    A dict makes it O(k) with comparisons only on hash collision, so the
    budget is generous and still three orders of magnitude below the old cost.
    """
    k = 1024
    comparisons = _count_category_comparisons(monkeypatch, tmp_path, k)
    assert comparisons <= 10 * k, (
        f"{comparisons} string comparisons to union {k} categories across two "
        f"inputs. Linear would be near zero (hash collisions only); the "
        f"list-membership implementation cost 2,096,128."
    )


# ---------------------------------------------------------------------------
# concat: per-row cost of an obs column
#
# Growth ratios cannot see these: building a Python object per row is linear,
# just with a fat constant. Comparing a column kind against a plain numeric
# column in the same run calibrates that constant away, so the assertion holds
# across interpreters and platforms without a hand-tuned byte budget.


def _obs_column_cost(tmp_path, kind: str, n_obs: int) -> tuple:
    files = _inputs(tmp_path, f"k{kind}{n_obs}", n_obs=n_obs, n_var=2, obs_kind=kind)
    with count_lines(SOURCE_PREFIX) as lines, count_allocations() as peak:
        concat_on_disk(files, tmp_path / f"ok{kind}{n_obs}.h5ad", QUIET)
    return lines[0], peak[0]


@pytest.mark.perf
@pytest.mark.parametrize("kind", ["masked", "string"])
def test_obs_columns_cost_no_more_python_work_per_row_than_a_numeric_one(
    tmp_path, kind
):
    """No Python-level loop over rows when concatenating an obs column.

    `_concat_masked` used to fill a `List[Any]` of length n_obs one element at
    a time and then walk it twice more. That is invisible to a read counter
    and to a growth ratio, but it shows up immediately as executed line events
    per row relative to the numeric path, which has always been vectorised.

    Measured after the fix: masked is 1.00x numeric, string 1.35x (the string
    path still materialises Python `str` objects via `read_str_all`, which is
    inherent to how strings are read, not a per-row loop). Before the fix
    masked was above 1.2x.

    Then CSR concatenation stopped looping over rows in Python, which took
    about eight line events a row out of the numeric baseline (114k to 50k
    here) while the string column's own cost stayed at 40k: five events a
    row, all in decoding and encoding text. The same work is 1.79x the
    smaller baseline, so the string limit moved from 1.45 to 1.9, which
    allows fewer extra events per row than 1.45 did before.
    """
    n_obs = 4000
    numeric_lines, _ = _obs_column_cost(tmp_path, "numeric", n_obs)
    kind_lines, _ = _obs_column_cost(tmp_path, kind, n_obs)

    limit = 1.10 if kind == "masked" else 1.9
    ratio = kind_lines / numeric_lines
    assert ratio <= limit, (
        f"a {kind} obs column executes {ratio:.2f}x the Python lines of a "
        f"numeric one ({kind_lines} vs {numeric_lines} at n_obs={n_obs}); "
        f"limit {limit}. Look for a per-element loop over rows."
    )


@pytest.mark.parametrize("kind", ["numeric", "masked", "string"])
def test_obs_column_allocation_grows_linearly(tmp_path, kind):
    """Peak allocation must not grow faster than the data itself.

    Note what this does *not* claim: obs columns are read whole, so peak
    allocation is O(n_obs) and not O(chunk). The streaming guarantee that the
    README makes holds for X, not for obs annotation. `benchmarks/` reports
    the real curve; this only stops it getting worse than linear.
    """

    def measure(n: int) -> int:
        _, peak = _obs_column_cost(tmp_path, kind, n)
        return peak

    assert_grows_linearly(
        measure,
        what=f"concat allocation, {kind} obs column",
        axis="n_obs",
        sizes=(256, 1024, 4096),
    )


# ---------------------------------------------------------------------------
# subset, split and conversion
#
# concat is where the defect was found, but nothing about the defect was
# specific to concat: every command that aligns one index onto another has the
# same opportunity to re-read a column per element.


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_subset_by_name_reads_grow_linearly(tmp_path, axis):
    def measure(n: int) -> int:
        source = _store(
            tmp_path / f"s{axis}{n}.h5ad",
            name="s",
            n_obs=n if axis == "obs" else 4,
            n_var=n if axis == "var" else 4,
        )
        # A name file, not indices: resolving names onto the stored index is
        # exactly the kind of alignment that went quadratic in concat.
        names = [f"sc{i}" for i in range(0, n, 2)] if axis == "obs" else [
            f"g{i}" for i in range(0, n, 2)
        ]
        name_file = tmp_path / f"n{axis}{n}.txt"
        name_file.write_text("\n".join(names) + "\n")
        with count_io() as io:
            subset_h5ad(
                source,
                tmp_path / f"os{axis}{n}.h5ad",
                name_file if axis == "obs" else None,
                name_file if axis == "var" else None,
                console=QUIET,
            )
        return io.reads

    assert_grows_linearly(
        measure, what=f"subset --{axis}", axis=f"n_{axis}", sizes=(64, 256, 1024)
    )


def test_subset_by_query_reads_grow_linearly(tmp_path):
    def measure(n: int) -> int:
        source = _store(
            tmp_path / f"q{n}.h5ad", name="q", n_obs=n, n_var=4, obs_kind="numeric"
        )
        with count_io() as io:
            subset_h5ad(
                source,
                tmp_path / f"oq{n}.h5ad",
                None,
                None,
                console=QUIET,
                obs_query="col >= 0",
            )
        return io.reads

    assert_grows_linearly(measure, what="subset --obs-query", axis="n_obs")


def test_h5ad_to_zarr_conversion_reads_grow_linearly(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"c{n}.h5ad", name="c", n_obs=n, n_var=4)
        # subset is also the conversion path, but it insists on a selection;
        # selecting every row is the identity and still rewrites the store.
        keep = tmp_path / f"ck{n}.txt"
        keep.write_text("\n".join(f"cc{i}" for i in range(n)) + "\n")
        with count_io() as io:
            subset_h5ad(
                source, tmp_path / f"oc{n}.zarr", keep, None, console=QUIET
            )
        return io.reads

    assert_grows_linearly(measure, what="h5ad to zarr", axis="n_obs")


@pytest.mark.parametrize("src,dst", [(".h5ad", ".zarr"), (".zarr", ".h5ad")])
def test_convert_reads_grow_linearly(tmp_path, src, dst):
    """The copy behind `convert` streams each array in bounded reads."""
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        source = tmp_path / f"cv{n}{src}"
        first = _store(tmp_path / f"cv{n}.h5ad", name="c", n_obs=n, n_var=4)
        if src == ".zarr":
            convert_store(first, source, QUIET)
        with count_io() as io:
            convert_store(source, tmp_path / f"ocv{n}{dst}", QUIET)
        return io.reads

    assert_grows_linearly(measure, what=f"convert {src} to {dst}", axis="n_obs")


def _tenx(path: Path, n: int, n_features: int = 8) -> Path:
    from tests.conftest import write_10x_h5

    counts = np.random.default_rng(0).poisson(0.5, (n, n_features))
    return write_10x_h5(
        path, counts,
        barcodes=[f"BC{i:08d}-1" for i in range(n)],
        ids=[f"ENSG{i}" for i in range(n_features)],
        names=[f"G{i}" for i in range(n_features)],
        types=["Gene Expression"] * (n_features - 2) + ["Antibody Capture"] * 2,
    )


@pytest.mark.parametrize("extra", [{}, {"split_feature_types": True}])
def test_10x_import_reads_grow_linearly(tmp_path, extra):
    """Barcodes are streamed, and a split reads the matrix once for all types."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options

    def measure(n: int) -> int:
        source = _tenx(tmp_path / f"tx{n}{len(extra)}.h5", n)
        with count_io() as io:
            convert_store(
                source, tmp_path / f"otx{n}{len(extra)}.h5ad", QUIET,
                options=Options(**extra),
            )
        return io.reads

    assert_grows_linearly(measure, what="convert 10x h5 to h5ad", axis="n_barcodes")


def test_10x_export_reads_grow_linearly(tmp_path):
    """Two passes over the barcodes -- width, then data -- and one over X."""
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        h5ad = tmp_path / f"ex{n}.h5ad"
        convert_store(_tenx(tmp_path / f"ex{n}.h5", n), h5ad, QUIET)
        with count_io() as io:
            convert_store(h5ad, tmp_path / f"oex{n}.h5", QUIET)
        return io.reads

    assert_grows_linearly(measure, what="convert h5ad to 10x h5", axis="n_barcodes")


def _mtx_folder(path: Path, n: int, n_features: int = 8, order: str = "cell") -> Path:
    from tests.conftest import write_10x_mtx

    counts = np.random.default_rng(0).poisson(0.5, (n, n_features))
    return write_10x_mtx(
        path, counts,
        barcodes=[f"BC{i:08d}-1" for i in range(n)],
        ids=[f"ENSG{i}" for i in range(n_features)],
        names=[f"G{i}" for i in range(n_features)],
        types=["Gene Expression"] * (n_features - 2) + ["Antibody Capture"] * 2,
        order=order,
    )


@pytest.mark.parametrize(
    "order,extra",
    [("cell", {}), ("shuffled", {}), ("cell", {"split_feature_types": True})],
)
def test_10x_mtx_import_work_grows_linearly(tmp_path, monkeypatch, order, extra):
    """Sorted, shuffled through buckets, or split among outputs: linear in nnz."""
    import adata.formats.mtx as mtx_module
    from adata.commands.convert import convert_store
    from adata.conversion import Options

    monkeypatch.setattr(mtx_module, "DEFAULT_CHUNK", 64)
    tag = f"{order}{len(extra)}"

    def measure(n: int) -> int:
        source = _mtx_folder(tmp_path / f"mx{tag}{n}", n, order=order)
        with count_io() as io:
            convert_store(
                source, tmp_path / f"omx{tag}{n}.h5ad", QUIET, options=Options(**extra)
            )
        return io.work

    assert_grows_linearly(
        measure, what=f"convert 10x mtx ({order}, {extra}) to h5ad", axis="n_barcodes"
    )


def test_10x_mtx_import_writes_in_blocks_not_per_barcode(tmp_path):
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        source = _mtx_folder(tmp_path / f"mw{n}", n)
        with count_io() as io:
            convert_store(source, tmp_path / f"omw{n}.h5ad", QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what="convert 10x mtx write calls", axis="n_barcodes")


def test_10x_mtx_export_reads_each_block_once(tmp_path):
    """One read of indices and of data per block of cells, not per cell."""
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        h5ad = tmp_path / f"me{n}.h5ad"
        convert_store(_mtx_folder(tmp_path / f"me{n}", n), h5ad, QUIET)
        with count_io() as io:
            convert_store(h5ad, tmp_path / f"ome{n}", QUIET, dst_format="10x-mtx")
        return io.calls

    assert_independent_of(measure, what="convert h5ad to 10x mtx read calls", axis="n_obs")


# A budget of one chunk's width of cells (64 cells of 10 genes, 4 bytes)
# reads slabs of cells; one of a byte, tiles sorted through buckets.
@pytest.mark.parametrize("slab_bytes", [64 * 10 * 4, 1], ids=["slabs", "tiles"])
@pytest.mark.parametrize("extra", [{}, {"dense": True}])
def test_loom_import_work_grows_linearly(tmp_path, monkeypatch, extra, slab_bytes):
    """Slabs of cells, or tiles of chunks: each chunk read a fixed number of
    times, whatever the cell count."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options, loom
    from tests.conftest import write_loom

    monkeypatch.setattr(loom, "SLAB_BYTES", slab_bytes)
    tag = "".join(extra) + str(slab_bytes)

    def measure(n: int) -> int:
        source = write_loom(tmp_path / f"lm{tag}{n}.loom", n_cells=n, velocyto=True).path
        with count_io() as io:
            convert_store(source, tmp_path / f"olm{tag}{n}.h5ad", QUIET, options=Options(**extra))
        return io.work

    assert_grows_linearly(measure, what=f"convert loom ({extra}) to h5ad", axis="n_cells")


def test_loom_import_writes_in_blocks(tmp_path):
    from adata.commands.convert import convert_store
    from tests.conftest import write_loom

    def measure(n: int) -> int:
        source = write_loom(tmp_path / f"lw{n}.loom", n_cells=n, velocyto=True).path
        with count_io() as io:
            convert_store(source, tmp_path / f"olw{n}.h5ad", QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what="convert loom write calls", axis="n_cells")


def test_scope_loom_import_work_grows_linearly(tmp_path, monkeypatch):
    """Regulons, embeddings and clusterings are read a block of cells at a
    time, each compound attribute once per block for all its fields."""
    from adata.commands.convert import convert_store
    from adata.conversion import loom
    from tests.conftest import write_scope_loom

    monkeypatch.setattr(loom, "STRING_STEP", 16)

    def measure(n: int) -> int:
        source = write_scope_loom(tmp_path / f"sc{n}.loom", n_cells=n).path
        with count_io() as io:
            convert_store(source, tmp_path / f"osc{n}.h5ad", QUIET)
        return io.work

    assert_grows_linearly(measure, what="convert SCope loom to h5ad", axis="n_cells")


def test_scope_loom_export_work_grows_linearly(tmp_path, monkeypatch):
    """Compound attributes are written a block of rows at a time, each field
    read once per block."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options, loom
    from tests.conftest import write_scope_loom

    monkeypatch.setattr(loom, "STRING_STEP", 16)

    def measure(n: int) -> int:
        h5ad = tmp_path / f"se{n}.h5ad"
        convert_store(write_scope_loom(tmp_path / f"se{n}.loom", n_cells=n).path, h5ad, QUIET)
        with count_io() as io:
            convert_store(h5ad, tmp_path / f"ose{n}.loom", QUIET, options=Options(scope=True))
        return io.work

    assert_grows_linearly(measure, what="convert h5ad to a SCope loom", axis="n_cells")


def test_scope_loom_import_writes_tables_in_blocks(tmp_path):
    """A block of rows is written per field, not a row: the write calls do
    not depend on the cell count."""
    from adata.commands.convert import convert_store
    from tests.conftest import write_scope_loom

    def measure(n: int) -> int:
        source = write_scope_loom(tmp_path / f"scw{n}.loom", n_cells=n).path
        with count_io() as io:
            convert_store(source, tmp_path / f"oscw{n}.h5ad", QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what="convert SCope loom write calls", axis="n_cells")


def test_mudata_import_work_grows_linearly(tmp_path, monkeypatch):
    """Global rows are taken onto a modality's a block at a time, out of
    order too (atac's cells are reversed), and its graph through buckets."""
    from adata.commands.convert import convert_store
    from adata.conversion import mudata
    from tests.conftest import write_h5mu

    monkeypatch.setattr(mudata, "ROW_BLOCK", 16)

    def measure(n: int) -> int:
        source = write_h5mu(tmp_path / f"mu{n}.h5mu", n_cells=n).path
        with count_io() as io:
            convert_store(source, tmp_path / f"mu{n}.h5ad", QUIET)
        return io.work

    assert_grows_linearly(measure, what="convert MuData to h5ads", axis="n_cells")


def test_mudata_export_work_grows_linearly(tmp_path, monkeypatch):
    """Every modality is written in one pass over the matrices."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options, mudata
    from tests.conftest import write_h5mu

    monkeypatch.setattr(mudata, "ROW_BLOCK", 16)

    def measure(n: int) -> int:
        h5ad = tmp_path / f"me{n}.h5ad"
        source = write_h5mu(tmp_path / f"me{n}.h5mu", n_cells=n).path
        convert_store(source, h5ad, QUIET, options=Options(mod="rna", given=frozenset({"--mod"})))
        with count_io() as io:
            convert_store(h5ad, tmp_path / f"out{n}.h5mu", QUIET)
        return io.work

    assert_grows_linearly(measure, what="convert h5ad to MuData", axis="n_cells")


@pytest.mark.parametrize("transpose", [False, True], ids=["cells-as-rows", "transposed"])
def test_text_import_work_grows_linearly(tmp_path, monkeypatch, transpose):
    """Blocks of lines, each parsed once; a transposed file's entries go
    through buckets, whatever the cell count."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options, text
    from tests.conftest import write_text_matrix

    monkeypatch.setattr(text, "BLOCK_BYTES", 256)
    monkeypatch.setattr(text, "BUCKET_ENTRIES", 64)
    tag = "t" if transpose else "c"

    def measure(n: int) -> int:
        source = write_text_matrix(
            tmp_path / f"tx{tag}{n}.csv", n_cells=n, n_genes=8, transpose=transpose
        ).path
        with count_io() as io:
            convert_store(
                source, tmp_path / f"otx{tag}{n}.h5ad", QUIET,
                options=Options(transpose=transpose),
            )
        return io.work

    assert_grows_linearly(measure, what=f"convert text ({tag}) to h5ad", axis="n_cells")


@pytest.mark.perf
@pytest.mark.parametrize("transpose", [False, True], ids=["cells-as-rows", "transposed"])
def test_text_parsing_python_work_grows_linearly(tmp_path, monkeypatch, transpose):
    """The parse is numpy's but for a split per line, to cut off its name:
    Python work per line, not per value and not per block seen before."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options, text
    from tests.conftest import write_text_matrix

    monkeypatch.setattr(text, "BLOCK_BYTES", 256)
    tag = "t" if transpose else "c"

    def measure(n: int) -> int:
        source = write_text_matrix(
            tmp_path / f"tl{tag}{n}.csv", n_cells=n, n_genes=8, transpose=transpose
        ).path
        with count_lines(SOURCE_PREFIX) as lines:
            convert_store(
                source, tmp_path / f"otl{tag}{n}.h5ad", QUIET,
                options=Options(transpose=transpose),
            )
        return lines[0]

    assert_grows_linearly(measure, what=f"text ({tag}) parsing, Python lines", axis="n_cells")


@pytest.mark.parametrize("transpose", [False, True], ids=["cells-as-rows", "transposed"])
def test_text_import_writes_in_blocks(tmp_path, transpose):
    from adata.commands.convert import convert_store
    from adata.conversion import Options
    from tests.conftest import write_text_matrix

    tag = "t" if transpose else "c"

    def measure(n: int) -> int:
        source = write_text_matrix(
            tmp_path / f"tw{tag}{n}.csv", n_cells=n, n_genes=8, transpose=transpose
        ).path
        with count_io() as io:
            convert_store(
                source, tmp_path / f"otw{tag}{n}.h5ad", QUIET,
                options=Options(transpose=transpose),
            )
        return io.h5_write_calls

    assert_independent_of(measure, what=f"convert text ({tag}) write calls", axis="n_cells")


# One chunk's width of cells (16 cells of 8 genes, 4 bytes) reads slabs of
# whole chunks; a byte, tiles sorted through buckets.
@pytest.mark.parametrize("slab_bytes", [16 * 8 * 4, 1], ids=["slabs", "tiles"])
@pytest.mark.parametrize("transpose", [False, True], ids=["cells-as-rows", "transposed"])
def test_hdf5_import_work_grows_linearly(tmp_path, monkeypatch, transpose, slab_bytes):
    """Slabs of whole chunks, or tiles of them: each chunk read a fixed
    number of times, whatever the cell count."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options, hdf5
    from tests.conftest import write_hdf5_matrix

    monkeypatch.setattr(hdf5, "SLAB_BYTES", slab_bytes)
    tag = ("t" if transpose else "c") + str(slab_bytes)
    chunks = (8, 16) if transpose else (16, 8)

    def measure(n: int) -> int:
        truth = write_hdf5_matrix(
            tmp_path / f"h{tag}{n}.hdf5", n_cells=n, n_genes=8, transpose=transpose, chunks=chunks
        )
        with count_io() as io:
            convert_store(
                truth.path, tmp_path / f"oh{tag}{n}.h5ad", QUIET,
                options=Options(key=truth.key, transpose=transpose),
            )
        return io.work

    assert_grows_linearly(measure, what=f"convert hdf5 ({tag}) to h5ad", axis="n_cells")


def test_hdf5_import_writes_in_blocks(tmp_path):
    from adata.commands.convert import convert_store
    from adata.conversion import Options
    from tests.conftest import write_hdf5_matrix

    def measure(n: int) -> int:
        truth = write_hdf5_matrix(tmp_path / f"hw{n}.hdf5", n_cells=n, n_genes=8)
        with count_io() as io:
            convert_store(
                truth.path, tmp_path / f"ohw{n}.h5ad", QUIET, options=Options(key=truth.key)
            )
        return io.h5_write_calls

    assert_independent_of(measure, what="convert hdf5 write calls", axis="n_cells")


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_loom_export_work_grows_linearly(tmp_path, monkeypatch, layout):
    from adata.commands.convert import convert_store
    from adata.conversion import loom

    monkeypatch.setattr(loom, "SLAB_BYTES", 1)

    def measure(n: int) -> int:
        from scipy import sparse

        counts = np.random.default_rng(0).poisson(0.5, (n, 8)).astype(np.float32)
        X = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "dense": np.asarray}[layout](counts)
        source = tmp_path / f"le{layout}{n}.h5ad"
        ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)])).write_h5ad(source)
        with count_io() as io:
            convert_store(source, tmp_path / f"ole{layout}{n}.loom", QUIET)
        return io.work

    assert_grows_linearly(measure, what=f"convert {layout} h5ad to loom", axis="n_obs")


def _parse_run(path: Path, n: int) -> Path:
    from tests.conftest import write_parse_run

    return write_parse_run(path, n_cells=n, n_genes=8, genomes=("hg38", "mm10")).root


@pytest.mark.parametrize("extra", [{}, {"genome": "mm10"}, {"all_samples": True}])
def test_parse_import_work_grows_linearly(tmp_path, extra):
    """Plain, with --genome's row map and gene filter, and one output per sample."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options

    tag = "".join(extra)

    def measure(n: int) -> int:
        source = _parse_run(tmp_path / f"pa{tag}{n}", n)
        with count_io() as io:
            convert_store(source, tmp_path / f"opa{tag}{n}.h5ad", QUIET, options=Options(**extra))
        return io.work

    assert_grows_linearly(measure, what=f"convert Parse ({extra}) to h5ad", axis="n_cells")


def test_parse_import_writes_obs_and_x_in_blocks(tmp_path):
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        source = _parse_run(tmp_path / f"pw{n}", n)
        with count_io() as io:
            convert_store(source, tmp_path / f"opw{n}.h5ad", QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what="convert Parse write calls", axis="n_cells")


def _cellbender_pair(path: Path, n: int, filtered: bool):
    from tests.conftest import write_10x_h5, write_cellbender_h5

    counts = np.random.default_rng(0).poisson(0.5, (n, 8))
    kwargs = dict(
        barcodes=[f"BC{i:08d}-1" for i in range(n)],
        ids=[f"ENSG{i}" for i in range(8)],
        names=[f"G{i}" for i in range(8)],
    )
    truth = write_cellbender_h5(path / "cb.h5", counts, filtered=filtered, **kwargs)
    raw = write_10x_h5(path / "raw.h5", counts, **kwargs)
    return truth.path, raw


@pytest.mark.parametrize(
    "filtered,extra",
    [
        (False, {}),
        (False, {"min_cell_probability": 0.5}),
        (False, {"raw": True}),
        (True, {"raw": True}),
    ],
)
def test_cellbender_import_work_grows_linearly(tmp_path, filtered, extra):
    """Latents, obs and the embedding stream with X; --raw reads the raw
    matrix once, through the buckets when the rows come in another order."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options

    tag = f"{filtered}{sorted(extra)}".replace(" ", "")

    def measure(n: int) -> int:
        where = tmp_path / f"cb{tag}{n}"
        where.mkdir()
        source, raw = _cellbender_pair(where, n, filtered)
        options = dict(extra)
        if options.pop("raw", False):
            options["raw"] = raw
        with count_io() as io:
            convert_store(source, where / "out.h5ad", QUIET, options=Options(**options))
        return io.work

    assert_grows_linearly(
        measure, what=f"convert CellBender ({'filtered' if filtered else 'full'}, {extra})",
        axis="n_barcodes",
    )


def test_cellbender_embedding_is_written_in_blocks(tmp_path):
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        where = tmp_path / f"ce{n}"
        where.mkdir()
        source, _ = _cellbender_pair(where, n, False)
        with count_io() as io:
            convert_store(source, where / "out.h5ad", QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what="convert CellBender write calls", axis="n_barcodes")


def _spatial_bundle(path: Path, kind: str, n: int, shuffled: bool) -> Path:
    from tests.conftest import write_segmented_outputs, write_visium_hd_outs, write_visium_outs, write_xenium_bundle

    if kind == "visium":
        # With Space Ranger's analysis/, its rows in another order.
        return write_visium_outs(path, n_spots=n, n_extra=n // 4, analysis=True).outs
    if kind == "visium-hd":
        # 16 µm bins on a grid of side ~sqrt(n): every bin on the slide in the
        # positions, about 60% of them in the matrix, and analysis/.
        return write_visium_hd_outs(path, sizes=(16,), grid=max(2, int(n ** 0.5)), analysis=True).outs
    if kind == "visium-cells":
        # Polygons for a fifth more cells than the matrix holds, the cells'
        # in file order or not.
        return write_segmented_outputs(path / "outs", n_cells=n, n_extra=n // 4, shuffled=shuffled).folder
    return write_xenium_bundle(path, n_cells=n, shuffled=shuffled, extra_cells=n // 8 if shuffled else 0).bundle


@pytest.mark.parametrize(
    "kind,shuffled",
    [("visium", True), ("visium-hd", True), ("visium-cells", False), ("visium-cells", True),
     ("xenium", False), ("xenium", True)],
)
def test_spatial_import_work_grows_linearly(tmp_path, kind, shuffled):
    """The positions or cell table are matched to the matrix while scanned,
    and written straight through when in its order (Xenium's own) or through
    the scatter buckets when not: linear either way, never a lookup per row."""
    from adata.commands.convert import convert_store
    from adata.conversion import Options

    def measure(n: int) -> int:
        source = _spatial_bundle(tmp_path / f"sp{kind}{shuffled}{n}", kind, n, shuffled)
        options = Options(bin_size=16) if kind == "visium-hd" else Options()
        with count_io() as io:
            convert_store(source, tmp_path / f"osp{kind}{shuffled}{n}.h5ad", QUIET, options=options)
        return io.work

    assert_grows_linearly(measure, what=f"convert {kind} (shuffled={shuffled})", axis="n_cells")


@pytest.mark.parametrize("kind", ["visium", "visium-hd"])
def test_spot_and_bin_analysis_is_read_once_a_pass_and_x_once(tmp_path, monkeypatch, kind):
    """Each analysis table is read twice whatever its size -- scanned and
    matched in the plan, written in the write -- and adding them reads X
    no more than once."""
    import h5py

    from adata.commands.convert import convert_store
    from adata.conversion import Options
    from adata.formats import table

    opened: dict = {}
    real_open = table.open_binary

    def counting_open(path, *args, **kwargs):
        key = f"{Path(path).parent.name}/{Path(path).name}"
        opened[key] = opened.get(key, 0) + 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(table, "open_binary", counting_open)
    source = _spatial_bundle(tmp_path / "sa", kind, 400, True)
    with count_io() as io:
        convert_store(
            source, tmp_path / "out.h5ad", QUIET,
            options=Options(bin_size=16) if kind == "visium-hd" else Options(),
        )
    with h5py.File(next(Path(source).rglob("filtered_feature_bc_matrix.h5")), "r") as f:
        nnz = f["matrix/data"].shape[0]

    assert {k: v for k, v in opened.items() if k.endswith(("clusters.csv", "projection.csv"))} == {
        "gene_expression_graphclust/clusters.csv": 2,
        "gene_expression_2_components/projection.csv": 2,
        "gene_expression_10_components/projection.csv": 2,
    }
    assert io.by_name["/matrix/data"] == nnz > 0


@pytest.mark.parametrize("kind", ["xenium", "visium-cells"])
@pytest.mark.parametrize("shuffled", [False, True])
def test_cells_are_written_in_blocks_not_per_cell(tmp_path, kind, shuffled):
    from adata.commands.convert import convert_store

    def measure(n: int) -> int:
        source = _spatial_bundle(tmp_path / f"xw{kind}{shuffled}{n}", kind, n, shuffled)
        with count_io() as io:
            convert_store(source, tmp_path / f"oxw{kind}{shuffled}{n}.h5ad", QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what=f"convert {kind} write calls (shuffled={shuffled})", axis="n_cells")


def test_split_reads_grow_linearly_in_group_count(tmp_path, monkeypatch):
    """Splitting into k groups must stay linear in k at fixed store size.

    n_obs is held constant deliberately. Scaling rows alongside groups would
    make the total legitimately quadratic -- k passes over 4k rows -- and the
    guard would be measuring the fixture, not the implementation.

    The output batch is raised above the largest k: batching adds one pass
    per batch, a step the linearity check would misread as curvature.
    `test_split_in_batches_matches_split_in_one` covers the batches.
    """
    from adata.commands.split import split_store
    from adata.core import subset

    monkeypatch.setattr(subset, "MAX_OPEN_OUTPUTS", 1024)

    n_obs = 512

    def measure(n: int) -> int:
        source = _store(
            tmp_path / f"sp{n}.h5ad",
            name="sp",
            n_obs=n_obs,
            n_var=4,
            n_categories=n,
        )
        out = tmp_path / f"osp{n}"
        with count_io() as io:
            split_store(source, "ct", out, QUIET, axis="obs", manifest=False)
        return io.reads

    assert_grows_linearly(
        measure, what="split --by", axis="n_groups", sizes=(8, 32, 128)
    )


def test_split_reads_the_matrix_once_whatever_the_group_count(tmp_path, monkeypatch):
    """X is read once for all groups, not once per group.

    Linear growth in k, which the guard above accepts, is exactly what split
    used to do: it subset once per group, and with interleaved groups each
    subset read the span of its own rows -- nearly all of X. On the release
    benchmark that was 56.7 s against 7.8 s for a naive anndata loop. The
    other elements are still written per group, so only X is counted here.
    """
    from adata.commands.split import split_store
    from adata.core import subset

    monkeypatch.setattr(subset, "MAX_OPEN_OUTPUTS", 1024)
    n_obs = 256

    def measure(n: int) -> int:
        source = _store(
            tmp_path / f"x{n}.h5ad", name="x", n_obs=n_obs, n_var=64, n_categories=n
        )
        with count_io() as io:
            split_store(source, "ct", tmp_path / f"ox{n}", QUIET, manifest=False)
        read = sum(
            v for k, v in io.by_name.items() if k.endswith(("X/data", "X/indices"))
        )
        # Every nonzero at least once: the counters can see these reads.
        assert read >= 2 * n_obs * 64, io.by_name
        return read

    assert_independent_of(
        measure, what="split X reads", axis="n_groups", sizes=(2, 128)
    )


# ---------------------------------------------------------------------------
# the constants themselves


def test_growth_limit_separates_linear_from_super_linear():
    """Documenting why K is 6, as an executable statement rather than a note.

    At the 4x spacing in SIZES the increment ratio is 4.0 for linear work,
    about 4.4 for n log n, 8 for n**1.5 and 16 for quadratic.
    """
    small, mid, large = SIZES
    assert mid == small * 4 and large == mid * 4, (
        "GROWTH_LIMIT is calibrated for 4x spacing; changing SIZES without "
        "recomputing it invalidates every guard in this file."
    )

    def ratio(exponent: float) -> float:
        cost = [float(n) ** exponent for n in SIZES]
        return (cost[2] - cost[1]) / (cost[1] - cost[0])

    assert ratio(1.0) < GROWTH_LIMIT, "linear work must pass"
    assert ratio(1.0) == pytest.approx(4.0, rel=0.01)
    assert ratio(2.0) > GROWTH_LIMIT, "quadratic work must fail"
    assert ratio(1.5) > GROWTH_LIMIT, "n**1.5 must fail"


# ---------------------------------------------------------------------------
# inspection is free
#
# `view` and `ls` reach only `.shape`, `.dtype` and `.attrs`: `axis_len` goes
# through `element_len`, which reads a shape, and `_array_details` and
# `_infer_untagged` never touch a value. That is the whole reason they return
# instantly on a store far too large to open, and it is the one claim in the
# README that can be stated as an exact number rather than a ratio.


def _reads_for_inspection(path: Path, run) -> "object":
    with count_io() as io:
        run(path)
    return io


@pytest.mark.parametrize("n_obs", [64, 4096])
@pytest.mark.parametrize(
    "label,run",
    [
        ("view", lambda p: show_info(p, QUIET, out_console=QUIET)),
        ("view --types", lambda p: show_info(p, QUIET, show_types=True,
                                             out_console=QUIET)),
        ("ls", lambda p: list_store(p, QUIET)),
        ("ls --long", lambda p: list_store(p, QUIET, long=True)),
        ("ls --plain", lambda p: list_store(p, QUIET, plain=True)),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_inspection_reads_no_data_at_all(tmp_path, n_obs, label, run):
    """Not "grows slowly" -- zero. An exact count, so it needs no tolerance.

    A ratio would be the wrong instrument here: 0 to 0 is not a meaningful
    ratio, and the moment inspection reads *one* column the answer stops
    being zero regardless of how the store scales.
    """
    source = _store(tmp_path / f"i{label}{n_obs}.h5ad", name="i", n_obs=n_obs,
                    n_var=16, obs_kind="numeric", n_categories=4)
    io = _reads_for_inspection(source, run)

    assert io.elements == 0, (
        f"`{label}` read {io.elements} data elements from a {n_obs}-row store; "
        f"inspection is supposed to touch only shapes and attributes. "
        f"Largest reader: {max(io.by_name.items(), key=lambda kv: kv[1], default=('-', 0))}"
    )


def test_the_inspection_fixture_really_does_hold_readable_data(tmp_path):
    """Keeps the test above from passing because there was nothing to read.

    Same store, read by a command that is supposed to read: if this registers
    nothing either, the fixture or the hooks are broken, not the claim.
    """
    source = _store(tmp_path / "control.h5ad", name="i", n_obs=4096, n_var=16,
                    obs_kind="numeric", n_categories=4)
    with count_io() as io:
        export_table(source, "obs", None, tmp_path / "control.csv", 10_000, None, QUIET)

    assert io.elements >= 4096, (
        f"the control read only {io.elements} elements from a 4096-row store, "
        "so the zero above proves nothing"
    )


# ---------------------------------------------------------------------------
# grouping work does not depend on the number of groups
#
# `split --by` groups rows through `core.select.group_indices`. Its cost is
# invisible to every other instrument in this file: the chunk is already in
# memory so no read counter moves, nothing lasting is allocated, and the
# Python line count per label is constant. Only counting what is handed to
# numpy shows it.


def test_grouping_does_not_rescan_the_column_once_per_group(tmp_path):
    """One pass over the column, however many distinct values it holds.

    Scanning per label is O(n_rows * n_groups): measured at n_obs=4096 it was
    16,384 elements for 4 groups and 1,048,576 for 256 -- exactly n_rows per
    group. On a million cells split by a thousand samples that is 10^9
    comparisons, and `split` would appear to hang for the same reason
    `concat --merge` did.

    n_obs is fixed. The claim is about work per row, not total work.
    """
    from adata.core.select import group_indices
    from adata.storage import open_store

    n_obs = 4096

    def measure(n_groups: int) -> int:
        source = _store(
            tmp_path / f"g{n_groups}.h5ad",
            name="g",
            n_obs=n_obs,
            n_var=2,
            n_categories=n_groups,
        )
        with open_store(source, "r") as store, count_scanned_elements() as scanned:
            groups, order = group_indices(store.root, "obs", "ct")
        assert len(order) == n_groups, "fixture did not produce the groups asked for"
        # A floor, not a measurement -- see count_scanned_elements. If the
        # implementation stops calling numpy by name the count collapses to
        # zero and the ratio below would pass for the wrong reason.
        assert scanned[0] >= n_obs, (
            f"only {scanned[0]} elements scanned for {n_obs} rows; the counter "
            "is no longer seeing this code path"
        )
        return scanned[0]

    assert_independent_of(
        measure, what="group_indices", axis="n_groups", sizes=(4, 256)
    )


# ---------------------------------------------------------------------------
# export
#
# Each export reads the thing it is asked for and nothing else. The axis
# differs per subcommand -- rows, elements, nonzeros, keys -- so each says
# which one it scales and holds the rest fixed.


def test_export_dataframe_reads_grow_linearly_in_rows(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"ed{n}.h5ad", name="e", n_obs=n, n_var=4,
                        obs_kind="numeric")
        with count_io() as io:
            export_table(source, "obs", None, tmp_path / f"ed{n}.csv",
                         10_000, None, QUIET)
        return io.reads

    assert_grows_linearly(measure, what="export dataframe", axis="n_obs")


def test_export_dataframe_reads_grow_linearly_in_column_count(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"ec{n}.h5ad", name="e", n_obs=16, n_var=4,
                        n_obs_columns=n)
        with count_io() as io:
            export_table(source, "obs", None, tmp_path / f"ec{n}.csv",
                         10_000, None, QUIET)
        return io.reads

    assert_grows_linearly(measure, what="export dataframe", axis="n_obs_columns")


def test_export_array_reads_grow_linearly_in_element_count(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"ea{n}.h5ad", name="e", n_obs=n, n_var=4)
        with count_io() as io:
            export_npy(source, "obsm/X_pca", tmp_path / f"ea{n}.npy",
                       100_000, QUIET)
        return io.reads

    assert_grows_linearly(measure, what="export array", axis="n_obs")


@pytest.mark.parametrize("in_memory", [False, True])
def test_export_sparse_reads_grow_linearly_in_nonzeros(tmp_path, in_memory):
    """Both paths: the streamed one and the one that loads the matrix."""

    def measure(n: int) -> int:
        source = _store(tmp_path / f"es{in_memory}{n}.h5ad", name="e",
                        n_obs=n, n_var=4)
        with count_io() as io:
            export_mtx(source, "X", tmp_path / f"es{in_memory}{n}.mtx",
                       None, 1_000, in_memory, QUIET)
        return io.reads

    assert_grows_linearly(
        measure, what=f"export sparse (in_memory={in_memory})", axis="nnz"
    )


def test_export_sparse_reads_each_block_once_not_each_row(tmp_path):
    """One read of `indices` and of `data` per block, however many rows.

    The streaming export used to read both once per row: two small reads per
    cell, each decompressing a chunk. Counting elements cannot see that --
    the same elements are read either way -- so this counts calls.
    """

    def measure(n: int) -> int:
        source = _store(tmp_path / f"ec{n}.h5ad", name="c", n_obs=n, n_var=4)
        with count_io() as io:
            export_mtx(source, "X", tmp_path / f"ec{n}.mtx", None, 1_000_000,
                       False, QUIET)
        return io.calls

    assert_independent_of(measure, what="export sparse read calls", axis="n_obs")


def test_export_dict_reads_grow_linearly_in_key_count(tmp_path):
    def measure(n: int) -> int:
        source = _store_with_uns_keys(tmp_path / f"ej{n}.h5ad", n)
        with count_io() as io:
            export_json(source, "uns", tmp_path / f"ej{n}.json",
                        100_000, False, QUIET)
        return io.reads

    assert_grows_linearly(
        measure, what="export dict", axis="n_keys", sizes=(64, 256, 1024)
    )


def test_export_image_reads_grow_linearly_in_pixels(tmp_path):
    from adata.commands.export import export_image

    def measure(n: int) -> int:
        source = _store_with_image(tmp_path / f"ei{n}.h5ad", n)
        with count_io() as io:
            export_image(source, "uns/picture", tmp_path / f"ei{n}.png", QUIET)
        return io.reads

    # n is the side of a square image, so pixels grow as n**2 -- the axis
    # being scaled is the pixel count, and the sizes below keep it at 4x.
    assert_grows_linearly(
        measure, what="export image", axis="pixels", sizes=(16, 32, 64)
    )


# ---------------------------------------------------------------------------
# import


def test_import_dataframe_reads_grow_linearly_in_rows(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"id{n}.h5ad", name="m", n_obs=n, n_var=4)
        csv = tmp_path / f"id{n}.csv"
        pd.DataFrame(
            {"score": np.arange(n, dtype="int32")},
            index=[f"mc{i}" for i in range(n)],
        ).to_csv(csv)
        with count_io() as io:
            import_object(source, "obs", csv, tmp_path / f"od{n}.h5ad",
                          False, None, QUIET)
        # `work`, not `reads`: import is a write path, and measuring only
        # reads made this vacuous.
        return io.work

    assert_grows_linearly(measure, what="import dataframe", axis="n_obs")


def test_import_array_reads_grow_linearly_in_element_count(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"ia{n}.h5ad", name="m", n_obs=n, n_var=4)
        npy = tmp_path / f"ia{n}.npy"
        np.save(npy, np.zeros((n, 3), dtype="float32"))
        with count_io() as io:
            import_object(source, "obsm/imported", npy,
                          tmp_path / f"oa{n}.h5ad", False, None, QUIET)
        return io.work

    assert_grows_linearly(measure, what="import array", axis="n_obs")


def test_import_sparse_reads_grow_linearly_in_nonzeros(tmp_path):
    def measure(n: int) -> int:
        source = _store(tmp_path / f"is{n}.h5ad", name="m", n_obs=n, n_var=4)
        mtx = tmp_path / f"is{n}.mtx"
        sparse_io = pytest.importorskip("scipy.io")
        sparse_io.mmwrite(
            str(mtx), sparse.csr_matrix(np.ones((n, 4), dtype="float32"))
        )
        with count_io() as io:
            import_object(source, "layers/imported", mtx,
                          tmp_path / f"os{n}.h5ad", False, None, QUIET)
        return io.work

    assert_grows_linearly(measure, what="import sparse", axis="nnz")


def test_import_sparse_writes_in_blocks_not_per_row(tmp_path):
    """Entries are appended a block at a time, whatever the number of rows."""
    sparse_io = pytest.importorskip("scipy.io")

    def measure(n: int) -> int:
        source = _store(tmp_path / f"iw{n}.h5ad", name="m", n_obs=n, n_var=4)
        mtx = tmp_path / f"iw{n}.mtx"
        sparse_io.mmwrite(str(mtx), sparse.csr_matrix(np.ones((n, 4), dtype="int32")))
        with count_io() as io:
            import_object(source, "layers/imported", mtx, None, True, None, QUIET)
        return io.h5_write_calls

    assert_independent_of(measure, what="import sparse write calls", axis="n_obs")


def test_import_dict_reads_grow_linearly_in_key_count(tmp_path):
    import json

    def measure(n: int) -> int:
        source = _store(tmp_path / f"ij{n}.h5ad", name="m", n_obs=8, n_var=4)
        blob = tmp_path / f"ij{n}.json"
        blob.write_text(json.dumps({f"k{i}": i for i in range(n)}))
        with count_io() as io:
            import_object(source, "uns/imported", blob,
                          tmp_path / f"oj{n}.h5ad", False, None, QUIET)
        return io.work

    assert_grows_linearly(
        measure, what="import dict", axis="n_keys", sizes=(64, 256, 1024)
    )


def test_import_image_reads_grow_linearly_in_pixels(tmp_path):
    """Images take their own entry point.

    `import_object` dispatches on extension and `.png` is deliberately not in
    `EXTENSION_FORMAT`; the CLI's `import image` calls `_import_image`
    directly, and it edits in place rather than writing a copy.
    """
    Image = pytest.importorskip("PIL.Image")
    from adata.commands.import_data import _import_image

    def measure(n: int) -> int:
        source = _store(tmp_path / f"ii{n}.h5ad", name="m", n_obs=8, n_var=4)
        png = tmp_path / f"ii{n}.png"
        Image.fromarray(np.zeros((n, n, 3), dtype="uint8")).save(png)
        with count_io() as io:
            _import_image(source, "uns/picture", png, QUIET)
        return io.work

    assert_grows_linearly(
        measure, what="import image", axis="pixels", sizes=(16, 32, 64)
    )


# ---------------------------------------------------------------------------
# create, and the concat options nothing else covers


@pytest.mark.parametrize("from_file", [False, True])
def test_create_writes_grow_linearly_in_obs_count(tmp_path, from_file):
    """Both the generated-names path and the name-file path."""

    def measure(n: int) -> int:
        names = None
        if from_file:
            names = tmp_path / f"cn{n}.txt"
            names.write_text("\n".join(f"c{i}" for i in range(n)) + "\n")
        output = tmp_path / f"cr{from_file}{n}.h5ad"
        with count_io() as io:
            create_store(
                output,
                QUIET,
                n_obs=None if from_file else n,
                n_var=4,
                obs_names=names,
            )
        return io.work

    assert_grows_linearly(
        measure, what=f"create (from_file={from_file})", axis="n_obs"
    )


@pytest.mark.parametrize(
    "option", ["label", "index_unique"], ids=["--label", "--index-unique"]
)
def test_concat_option_reads_grow_linearly_in_obs_count(tmp_path, option):
    """Both build a per-row value, so both are worth a guard of their own."""

    def measure(n: int) -> int:
        files = _inputs(tmp_path, f"co{option}{n}", n_obs=n, n_var=4)
        kwargs = {"label": "batch"} if option == "label" else {"index_unique": "-"}
        with count_io() as io:
            concat_on_disk(files, tmp_path / f"oco{option}{n}.h5ad", QUIET, **kwargs)
        return io.reads

    assert_grows_linearly(measure, what=f"concat --{option}", axis="n_obs")


def test_split_by_var_reads_grow_linearly_in_group_count(tmp_path):
    """The var axis takes a different path through `split` than obs does."""
    from adata.commands.split import split_store

    n_var = 512

    def measure(n: int) -> int:
        source = _store_with_var_groups(tmp_path / f"sv{n}.h5ad", n_var, n)
        with count_io() as io:
            split_store(source, "kind", tmp_path / f"osv{n}", QUIET,
                        axis="var", manifest=False)
        return io.reads

    assert_grows_linearly(
        measure, what="split --axis var", axis="n_groups", sizes=(8, 32, 128)
    )


# ---------------------------------------------------------------------------
# fixture variants the guards above need


def _store_with_uns_keys(path: Path, n_keys: int) -> Path:
    obj = ad.AnnData(
        X=np.ones((4, 2), dtype="float32"),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    obj.uns.update({f"k{i}": i for i in range(n_keys)})
    obj.write_h5ad(path)
    return path


def _store_with_image(path: Path, side: int) -> Path:
    obj = ad.AnnData(
        X=np.ones((4, 2), dtype="float32"),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    obj.uns["picture"] = np.zeros((side, side, 3), dtype="uint8")
    obj.write_h5ad(path)
    return path


def _store_with_var_groups(path: Path, n_var: int, n_groups: int) -> Path:
    ad.AnnData(
        X=np.ones((4, n_var), dtype="float32"),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(
            {"kind": pd.Categorical([f"k{i % n_groups}" for i in range(n_var)])},
            index=[f"g{i}" for i in range(n_var)],
        ),
    ).write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# streaming is bounded by the chunk, not by the input
#
# The README's actual claim, and the reason this tool exists. It is only
# wholly true in one place today, so these guards say what is true rather
# than what would be nice, and record the measured numbers so that drift is
# visible. Everything below holds the chunk size fixed and grows the input by
# 256x; a command that loaded everything would grow 256x with it.

#: 256x span. Wide enough that "flat" and "linear" cannot be confused, and
#: the reason these carry the `slow` marker: building a 65,536-row store is
#: most of the ~45 s this section costs. They still gate merges; the marker
#: is so a local run can say `-m "not slow"`.
STREAM_SIZES = (256, 65536)


def _peak_for(tmp_path: Path, tag: str, n: int, run) -> int:
    source = _store(tmp_path / f"{tag}{n}.h5ad", name="s", n_obs=n, n_var=8,
                    obs_kind="numeric")
    with count_allocations() as peak:
        run(source, n)
    return peak[0]


@pytest.mark.slow
def test_export_array_peak_memory_is_set_by_the_chunk_not_the_input(tmp_path):
    """The closest the tool comes to the claim outright.

    Measured at a fixed 1,000-element chunk: 21.5 KiB at 256 rows and
    34.3 KiB at 65,536 -- 1.6x for a 256x input. Not flat, so this does not
    assert flat; the residue is fixed-size bookkeeping that grows with the
    length of a formatted shape rather than with the data. Stating it as "at
    least 64x better than the input" is both true and strong: loading
    everything would be 256x.
    """

    def measure(n: int) -> int:
        return _peak_for(
            tmp_path, "sa", n,
            lambda p, k: export_npy(p, "obsm/X_pca", tmp_path / f"sa{k}.npy",
                                    1_000, QUIET),
        )

    assert_grows_slower_than_input(
        measure,
        what="export array peak allocation at --chunk 1000",
        axis="n_obs",
        sizes=STREAM_SIZES,
        at_least=64.0,
    )


@pytest.mark.parametrize(
    "label,factor,run",
    [
        # Measured over the 256x span: export sparse 6.5x, export dataframe
        # 2.2x, subset 46x. The factors below sit roughly midway between the
        # measurement and linear, so ordinary variation passes and a real
        # drift towards loading everything fails.
        (
            "export sparse",
            8.0,
            lambda p, k, t: export_mtx(p, "X", t / f"ss{k}.mtx", None, 1_000,
                                       False, QUIET),
        ),
        (
            "export dataframe",
            8.0,
            lambda p, k, t: export_table(p, "obs", None, t / f"st{k}.csv",
                                         1_000, None, QUIET),
        ),
        (
            "subset",
            2.0,
            lambda p, k, t: subset_h5ad(p, t / f"su{k}.h5ad", None, None,
                                        console=QUIET, obs_query="col >= 0",
                                        chunk_rows=256),
        ),
    ],
    ids=["export-sparse", "export-dataframe", "subset"],
)
@pytest.mark.slow
def test_streamed_peak_memory_grows_far_slower_than_the_input(
    tmp_path, label, factor, run
):
    """Not flat, and the guards should not pretend otherwise.

    These paths read an index or an indptr whole, so peak allocation does
    track n_obs -- just far below it. `subset` is the weakest of the three
    because obs columns are materialised per column; that is a known gap,
    reported by `benchmarks/` rather than hidden here.
    """

    def measure(n: int) -> int:
        return _peak_for(tmp_path, f"st{label[-4:]}", n,
                         lambda p, k: run(p, k, tmp_path))

    assert_grows_slower_than_input(
        measure,
        what=f"{label} peak allocation at a fixed chunk",
        axis="n_obs",
        sizes=STREAM_SIZES,
        at_least=factor,
    )


@pytest.mark.slow
def test_streaming_export_sparse_costs_far_less_than_loading_the_matrix(
    tmp_path,
):
    """`--in-memory` exists as the fast path; the default must earn its place.

    Self-calibrating: both are measured in the same run, so the assertion
    holds regardless of platform. Measured at 65,536 rows: 777 KiB streamed
    against 31,763 KiB in memory, a factor of 41.
    """
    n = 65_536
    source = _store(tmp_path / "cmp.h5ad", name="s", n_obs=n, n_var=8)

    with count_allocations() as streamed:
        export_mtx(source, "X", tmp_path / "streamed.mtx", None, 1_000, False, QUIET)
    with count_allocations() as loaded:
        export_mtx(source, "X", tmp_path / "loaded.mtx", None, 1_000, True, QUIET)

    assert streamed[0] <= loaded[0] / 4, (
        f"streaming export used {streamed[0] // 1024} KiB against "
        f"{loaded[0] // 1024} KiB for --in-memory, a factor of only "
        f"{loaded[0] / max(streamed[0], 1):.1f}. The default path is supposed "
        "to be the one you reach for when the matrix does not fit."
    )


# ---------------------------------------------------------------------------
# copying a store must stay inside its read budget
#
# `copy_dataset` sizes each read to TARGET_READ_BYTES. For variable-length
# strings it cannot ask the dtype how wide an element is -- h5py reports the
# itemsize of a pointer -- so it estimated. An estimate is not a bound: at
# 4 KiB elements the 32 MiB budget became an 827 MB peak, and at 200,000 rows
# the computed step exceeded the dataset, so the whole array was read at once.
#
# This is the one path the rest of this file did not reach, and it is the
# path every `copy:` task in `subset` and every uns entry goes through.


def _vlen_source(path: Path, width: int, n_rows: int):
    """An HDF5 variable-length string dataset of `n_rows` x `width` bytes."""
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "t",
            data=np.array([("x" * width).encode()] * n_rows, dtype=object),
            dtype=h5py.string_dtype(),
            chunks=(1024,),
        )
    return path


@pytest.mark.parametrize("width", [16, 256, 4096])
def test_a_read_of_variable_length_strings_stays_inside_the_byte_budget(
    tmp_path, width
):
    """The budget is bytes per read, so a wider element means fewer elements.

    This is the exact invariant, and it costs nothing to check: the step is
    computed, not measured. Before the fix the element width was assumed to
    be 64 bytes, so the step was the same 524,288 elements whatever the data
    actually held -- 2 GiB per read at 4 KiB elements, against a stated
    32 MiB budget.
    """
    import h5py

    from adata.storage import (
        TARGET_READ_BYTES,
        _chunk_step,
        _sample_element_bytes,
    )

    n_rows = 50_000
    source = _vlen_source(tmp_path / f"s{width}.h5", width, n_rows)
    with h5py.File(source, "r") as handle:
        dataset = handle["t"]
        sampled = _sample_element_bytes(dataset, n_rows)
        step = _chunk_step(dataset.shape, dataset.chunks, dataset.dtype, sampled)

    assert sampled >= min(width, 64), (
        f"sampled {sampled} bytes for {width}-byte elements; the width is "
        "being guessed rather than measured"
    )
    # A read may round up to a whole source chunk, hence the slack.
    assert step * width <= TARGET_READ_BYTES * 2, (
        f"one read would take {step * width / 1e6:.0f} MB of {width}-byte "
        f"strings ({step} elements), against a {TARGET_READ_BYTES / 1e6:.0f} "
        "MB budget"
    )


@pytest.mark.slow
def test_copying_wide_strings_does_not_read_the_whole_array(tmp_path):
    """And the budget holds in practice, not just in the arithmetic.

    Sized so the array is several times the budget: before the fix the
    computed step exceeded the row count, so the whole thing was read at once
    -- 164 MB here, and 827 MB in the 200,000-row case that prompted this.
    """
    import h5py

    from adata.storage import TARGET_READ_BYTES, copy_dataset

    width, n_rows = 4096, 40_000  # ~164 MB, about 5x the budget
    source = _vlen_source(tmp_path / "wide.h5", width, n_rows)

    with h5py.File(source, "r") as src, h5py.File(tmp_path / "out.h5", "w") as dst:
        with count_allocations() as peak:
            copy_dataset(src["t"], dst, "t")

    assert peak[0] <= TARGET_READ_BYTES * 3, (
        f"copying a {width * n_rows / 1e6:.0f} MB array of {width}-byte "
        f"strings peaked at {peak[0] / 1e6:.0f} MB, against a "
        f"{TARGET_READ_BYTES / 1e6:.0f} MB read budget. A step larger than "
        "the array means the whole array is read in one go."
    )


# ---------------------------------------------------------------------------
# recast
#
# The streaming transpose exists because the in-memory one does not scale.
# That is a claim about peak memory, so it is the peak that is asserted --
# a correctness test cannot tell the two implementations apart, which is
# exactly why they are both allowed to exist.


def _sparse_store(path: Path, n_obs: int, n_var: int = 32, density: float = 0.2):
    rng = np.random.default_rng(0)
    matrix = sparse.random(
        n_obs, n_var, density=density, format="csr", dtype="float32",
        random_state=rng,
    )
    ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_var)]),
    ).write_h5ad(path)
    return path


def _recast(source: Path, out: Path, **kwargs):
    from adata.commands.recast import recast_store

    recast_store(source, ["X"], out, QUIET, **kwargs)


def test_recast_dtype_reads_grow_linearly_in_nonzeros(tmp_path):
    def measure(n: int) -> int:
        source = _sparse_store(tmp_path / f"cd{n}.h5ad", n)
        with count_io() as io:
            _recast(source, tmp_path / f"od{n}.h5ad", dtype="float64")
        return io.work

    assert_grows_linearly(measure, what="recast --dtype", axis="nnz")


@pytest.mark.parametrize("layout", ["csc", "dense"])
def test_recast_layout_reads_grow_linearly(tmp_path, layout):
    def measure(n: int) -> int:
        source = _sparse_store(tmp_path / f"cl{layout}{n}.h5ad", n)
        with count_io() as io:
            _recast(
                source, tmp_path / f"ol{layout}{n}.h5ad", layout=layout, force=True
            )
        return io.work

    assert_grows_linearly(measure, what=f"recast --layout {layout}", axis="n_obs")


@pytest.mark.slow
def test_the_streaming_transpose_does_not_hold_the_matrix(tmp_path):
    """Peak allocation must not track nnz. This is the whole point of it.

    Measured against the in-memory path in the same run, which does hold the
    matrix and therefore does grow -- so the comparison shows the difference
    is real rather than an artefact of how the fixture is built.
    """

    def peak(n: int, in_memory: bool) -> int:
        source = _sparse_store(tmp_path / f"tp{in_memory}{n}.h5ad", n, n_var=64)
        with count_allocations() as measured:
            _recast(
                source,
                tmp_path / f"otp{in_memory}{n}.h5ad",
                layout="csc",
                in_memory=in_memory,
                chunk=4096,
            )
        return measured[0]

    small, large = 256, 8192
    streaming = peak(large, False) / max(1, peak(small, False))
    loaded = peak(large, True) / max(1, peak(small, True))

    assert streaming < loaded, (
        f"the streaming transpose grew {streaming:.1f}x over a 32x larger "
        f"matrix and the in-memory one grew {loaded:.1f}x -- if streaming is "
        "not the cheaper of the two it has no reason to exist"
    )
    assert streaming <= 8.0, (
        f"streaming transpose peak grew {streaming:.1f}x for 32x the "
        "nonzeros; it is supposed to be bounded by the chunk"
    )


def test_the_transpose_streams_unless_asked_not_to(tmp_path, monkeypatch):
    """The safe path is the default; --in-memory is opt-in.

    Asserted by watching which function runs, because the two produce
    identical output and no result can distinguish them.
    """
    import adata.core.recast as recast_module

    called: List[str] = []
    for name in ("transpose_sparse_streaming", "transpose_sparse_in_memory"):
        original = getattr(recast_module, name)

        def record(*args, _name=name, _original=original, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(recast_module, name, record)

    source = _sparse_store(tmp_path / "d.h5ad", 64)
    _recast(source, tmp_path / "default.h5ad", layout="csc")
    assert called == ["transpose_sparse_streaming"], called

    called.clear()
    _recast(source, tmp_path / "asked.h5ad", layout="csc", in_memory=True)
    assert called == ["transpose_sparse_in_memory"], called

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

from adata.core.concat import concat_on_disk
from adata.core.subset import subset_h5ad

from tests.perf_counters import (
    GROWTH_LIMIT,
    SIZES,
    assert_grows_linearly,
    count_allocations,
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

    ad.AnnData(
        X=sparse.csr_matrix(np.ones((n_obs, n_var), dtype="float32")),
        obs=obs,
        var=var,
    ).write_h5ad(path)
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

    original = concat_module.read_categories
    monkeypatch.setattr(
        concat_module,
        "read_categories",
        lambda col: [_CountingStr(c) for c in original(col)],
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
    """
    n_obs = 4000
    numeric_lines, _ = _obs_column_cost(tmp_path, "numeric", n_obs)
    kind_lines, _ = _obs_column_cost(tmp_path, kind, n_obs)

    limit = 1.10 if kind == "masked" else 1.45
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


def test_split_reads_grow_linearly_in_group_count(tmp_path):
    """Splitting into k groups must stay linear in k at fixed store size.

    n_obs is held constant deliberately. Scaling rows alongside groups would
    make the total legitimately quadratic -- k passes over 4k rows -- and the
    guard would be measuring the fixture, not the implementation.
    """
    from adata.commands.split import split_store

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

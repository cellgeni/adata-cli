"""Deterministic cost counters for the complexity guards.

These measure *how much work* an operation does, never how long it takes. A
test that can fail because a CI runner was busy does not belong in a merge
gate, so nothing here touches the clock.

Why patch the libraries rather than `adata`
-------------------------------------------
The hooks wrap h5py and zarr themselves. A seam inside `src/adata/` would only
measure the call sites that remembered to use it, which is precisely the wrong
property for a guard whose job is to catch the read you did not think of. The
matrix paths in particular slice the backend objects directly -- see
`core/subset.py` `_copy_dense_rows` and `core/concat.py` `_concat_csr` -- and
no in-repo helper sees them at all.

What is counted
---------------
Elements and bytes, not just calls. The `--merge` hang was n calls each reading
n elements; counting calls alone would also miss a vectorised-but-quadratic
variant that reads n elements n times in one call.

Known bypasses
--------------
These reach the file without passing through any hook below:

* ``h5py.Dataset.read_direct``
* ``np.asarray(dataset)`` and ``dataset[...]`` via ``__array__`` (verified:
  ``np.asarray`` adds nothing to the element count)
* ``dataset.asstr()[...]``, which goes through ``AsStrWrapper.__getitem__``
* ``dataset.fields(...)``

None are used in `src/adata/` today. If you add one, the counters will silently
report less work than actually happened -- which is why every guard also
asserts a *lower* bound, and why `test_performance.py` keeps a canary test with
a known absolute count.
"""

from __future__ import annotations

import sys
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List

import numpy as np


@dataclass
class IOCounts:
    """Live tally of backend reads and writes.

    The object is yielded before the work runs and mutated in place, so read
    the fields after the `with` block closes.
    """

    h5_calls: int = 0
    h5_elements: int = 0
    h5_write_calls: int = 0
    h5_written_elements: int = 0
    zarr_calls: int = 0
    zarr_elements: int = 0
    store_get: int = 0
    store_get_bytes: int = 0
    store_set: int = 0
    store_delete: int = 0

    #: Per-dataset element counts, for diagnosing which column blew up.
    by_name: Dict[str, int] = field(default_factory=dict)

    @property
    def elements(self) -> int:
        """Elements read through either backend's array interface."""
        return self.h5_elements + self.zarr_elements

    @property
    def calls(self) -> int:
        return self.h5_calls + self.zarr_calls

    @property
    def writes(self) -> int:
        """Elements written through either backend.

        Zarr is counted in store keys rather than elements -- a chunk write
        is one `set` -- so the two are not the same unit. For a guard that
        only compares a command against itself at two sizes, that is fine.
        """
        return self.h5_written_elements + self.store_set

    @property
    def work(self) -> int:
        """Reads plus writes, for commands whose job is mostly writing.

        `import` and `create` read almost nothing; measuring only reads made
        their guards vacuous, which the lower bound in `assert_grows_linearly`
        caught.
        """
        return self.reads + self.writes

    @property
    def reads(self) -> int:
        """The headline number: array elements plus store-level fetches.

        Covers both the "read the same column n times" shape and the
        "re-fetch the same chunk n times" shape.
        """
        return self.elements + self.store_get

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"read={self.elements} (h5={self.h5_elements} "
            f"zarr={self.zarr_elements}) wrote={self.h5_written_elements} "
            f"calls={self.calls} store: get={self.store_get} "
            f"set={self.store_set} delete={self.store_delete}"
        )


def _size(value: Any) -> int:
    try:
        return int(np.asarray(value).size)
    except Exception:  # pragma: no cover - exotic dtypes
        return 1


@contextmanager
def count_io() -> Iterator[IOCounts]:
    """Count backend reads and writes for the duration of the block.

    Patches four seams, all verified against h5py 3.15.1 and zarr 3.1.5:

    ======================================  =================================
    ``h5py.Dataset.__getitem__``            HDF5 reads
    ``h5py.Dataset.__setitem__``            HDF5 writes
    ``h5py.Group.create_dataset``           HDF5 writes made at creation
    ``zarr.Array.__getitem__``              Zarr reads, v2 and v3 alike
    ``zarr.storage.LocalStore.get``         chunk and metadata fetches
    ``LocalStore.set`` / ``.delete``        write storms
    ======================================  =================================

    Both write seams are needed: `create_dataset(name, data=...)` writes its
    payload during creation and never touches `__setitem__`, so hooking only
    the latter leaves `import` and `create` measuring zero.

    The store hooks are the ones that see a metadata write storm: attribute
    writes never touch `Array.__getitem__`, so an O(k^2) attribute rewrite is
    invisible at the array level and obvious at the store level.

    The `LocalStore` methods are coroutines and are wrapped as coroutines.
    """
    import h5py
    import zarr
    from zarr.storage import LocalStore

    counts = IOCounts()

    h5_get = h5py.Dataset.__getitem__
    h5_set = h5py.Dataset.__setitem__
    h5_create = h5py.Group.create_dataset
    zarr_get = zarr.Array.__getitem__
    store_get = LocalStore.get
    store_set = LocalStore.set
    store_delete = LocalStore.delete

    def _record(name: str, n: int) -> None:
        counts.by_name[name] = counts.by_name.get(name, 0) + n

    def h5_wrapper(self: Any, key: Any) -> Any:
        result = h5_get(self, key)
        n = _size(result)
        counts.h5_calls += 1
        counts.h5_elements += n
        _record(getattr(self, "name", "?"), n)
        return result

    def zarr_wrapper(self: Any, key: Any) -> Any:
        result = zarr_get(self, key)
        n = _size(result)
        counts.zarr_calls += 1
        counts.zarr_elements += n
        _record(getattr(self, "path", "?") or "/", n)
        return result

    def h5_set_wrapper(self: Any, key: Any, value: Any) -> Any:
        counts.h5_write_calls += 1
        counts.h5_written_elements += _size(value)
        return h5_set(self, key, value)

    def h5_create_wrapper(self: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
        counts.h5_write_calls += 1
        data = kwargs.get("data")
        if data is None and args:
            # positional signature is (name, shape, dtype, data, ...)
            data = args[2] if len(args) > 2 else None
        if data is not None:
            counts.h5_written_elements += _size(data)
        return h5_create(self, name, *args, **kwargs)

    async def get_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await store_get(self, *args, **kwargs)
        counts.store_get += 1
        if result is not None:
            try:
                counts.store_get_bytes += len(result)
            except TypeError:  # pragma: no cover - prototype buffers
                pass
        return result

    async def set_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts.store_set += 1
        return await store_set(self, *args, **kwargs)

    async def delete_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts.store_delete += 1
        return await store_delete(self, *args, **kwargs)

    h5py.Dataset.__getitem__ = h5_wrapper
    h5py.Dataset.__setitem__ = h5_set_wrapper
    h5py.Group.create_dataset = h5_create_wrapper
    zarr.Array.__getitem__ = zarr_wrapper
    LocalStore.get = get_wrapper
    LocalStore.set = set_wrapper
    LocalStore.delete = delete_wrapper
    try:
        yield counts
    finally:
        h5py.Dataset.__getitem__ = h5_get
        h5py.Dataset.__setitem__ = h5_set
        h5py.Group.create_dataset = h5_create
        zarr.Array.__getitem__ = zarr_get
        LocalStore.get = store_get
        LocalStore.set = store_set
        LocalStore.delete = store_delete


@contextmanager
def count_allocations() -> Iterator[List[int]]:
    """Peak Python allocation during the block, in bytes, as `[peak]`.

    `tracemalloc` sees Python-level allocation, which includes the numpy array
    *objects* and every Python list and string built along the way, but not
    memory a C extension takes outside the allocator. That is the right scope
    here: the defects this catches are whole-column Python lists, and h5py's
    own buffers are not what we are policing.

    Deterministic for deterministic input, which is why it can gate a merge.
    """
    peak = [0]
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    else:  # pragma: no cover - nested use
        tracemalloc.reset_peak()
    try:
        yield peak
    finally:
        peak[0] = tracemalloc.get_traced_memory()[1]
        if started:
            tracemalloc.stop()


@contextmanager
def count_lines(prefix: str) -> Iterator[List[int]]:
    """Count executed Python line events in files under `prefix`, as `[n]`.

    A deterministic stand-in for CPU time. It is what catches a quadratic that
    performs no extra I/O and allocates nothing extra -- `if x not in a_list`
    inside a loop, say. Costs a 10-50x slowdown, so keep it to the few paths
    with known Python-level inner loops and to small inputs.

    Blind to work done inside numpy's C code, so it complements the allocation
    and I/O counters rather than replacing them.
    """
    total = [0]

    def local_trace(frame: Any, event: str, arg: Any) -> Any:
        if event == "line":
            total[0] += 1
        return local_trace

    def trace(frame: Any, event: str, arg: Any) -> Any:
        if event == "call" and frame.f_code.co_filename.startswith(prefix):
            return local_trace
        return None

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        yield total
    finally:
        sys.settrace(previous)


#: numpy entry points that scan their operand, and the argument that is
#: scanned. Patched by name, so only calls written as `np.f(...)` are seen.
_SCANNING = {
    "nonzero": 0,
    "flatnonzero": 0,
    "unique": 0,
    "argsort": 0,
    "sort": 0,
    "searchsorted": 0,
    "bincount": 0,
    "equal": 0,
    "isin": 0,
}


@contextmanager
def count_scanned_elements() -> Iterator[List[int]]:
    """Elements handed to numpy's scanning primitives, as `[n]`.

    The third instrument, and the one the other two cannot replace. A loop of
    the shape::

        for label in distinct_labels:
            np.nonzero(values == label)

    reads nothing extra (the chunk is already in memory), allocates nothing
    that lasts, and executes a constant number of Python lines per label. It
    is invisible to `count_io`, to `count_allocations` and to `count_lines`
    alike, and it is O(len(values) * len(distinct_labels)).

    Counting what is passed into numpy makes it visible, in the same spirit
    as the `_CountingStr` guard in `test_performance.py`: when the cost lives
    inside C, count what is handed to C.

    What this does and does not see
    -------------------------------
    Only calls that go through the `numpy` module namespace by name. An
    operator -- `values == label` -- dispatches to `ndarray.__eq__` and then
    to the ufunc in C, so it never passes the patched `np.equal`; a scan
    written purely as `(a == b).sum()` is invisible. Method form,
    `arr.argsort()`, is invisible for the same reason.

    So this is a floor on the real work, not a measurement of it, which is
    exactly what a guard needs: it can only under-report, and the guards
    using it assert a lower bound so that under-reporting to nothing fails
    rather than passes.
    """
    total = [0]
    originals = {name: getattr(np, name) for name in _SCANNING}

    def wrap(name: str, position: int):
        original = originals[name]

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) > position:
                try:
                    total[0] += int(np.size(args[position]))
                except Exception:  # pragma: no cover - exotic operands
                    pass
            return original(*args, **kwargs)

        return wrapper

    for name, position in _SCANNING.items():
        setattr(np, name, wrap(name, position))
    try:
        yield total
    finally:
        for name, original in originals.items():
            setattr(np, name, original)


# ---------------------------------------------------------------------------
# the growth assertion


#: Ratio of successive increments that counts as super-linear.
#:
#: Measured at 4x spacing, the increment ratio is 4.0 for linear work, about
#: 4.4 for n log n, 8 for n**1.5 and 16 for quadratic. 6 sits in the gap with
#: room on both sides, so it needs no per-test tuning.
GROWTH_LIMIT = 6.0

#: Sizes every growth guard runs at. 4x spacing is what makes GROWTH_LIMIT
#: mean what it says; changing one without the other invalidates the constant.
SIZES = (64, 256, 1024)


def assert_grows_linearly(
    measure: Callable[[int], int],
    *,
    what: str,
    axis: str,
    sizes: tuple = SIZES,
    limit: float = GROWTH_LIMIT,
) -> None:
    """Assert `measure` grows no faster than linearly in one axis.

    Compares successive *increments* rather than raw counts::

        d1 = c(4n)  - c(n)
        d2 = c(16n) - c(4n)
        assert d2 <= limit * d1

    The increment form is what removes the arbitrary additive constant. Any
    fixed setup cost -- opening the store, writing the skeleton, reading the
    index once -- appears in both differences and cancels exactly, so there is
    no slack parameter to pick and no floor for a small-coefficient quadratic
    to hide under.

    `d1 >= sizes[1]` is a vacuity guard. If a future refactor moves a read onto
    a path the counters do not see (see the module docstring), every ratio
    would pass trivially; this fails loudly instead.

    Scale exactly one axis per call and hold the others fixed. Scaling two at
    once makes legitimate work look quadratic -- an outer-join concat really
    does produce n_obs x n_var_union cells.
    """
    small, mid, large = sizes
    c_small, c_mid, c_large = (measure(n) for n in (small, mid, large))

    d1 = c_mid - c_small
    d2 = c_large - c_mid
    series = (
        f"{what}, scaling {axis}: "
        f"{small}->{c_small}, {mid}->{c_mid}, {large}->{c_large} "
        f"(increments {d1} then {d2}"
        + (f", ratio {d2 / d1:.1f}" if d1 > 0 else "")
        + ")"
    )

    # Half an operation per element added. Work that is exactly one
    # operation per element -- `export dict` reads each key once -- gives
    # d1 = mid - small, which is 0.75 * mid at 4x spacing, so a bound of
    # `mid` would reject correct code. Anything the counters cannot see at
    # all gives 0 and still fails.
    floor = (mid - small) / 2
    assert d1 >= floor, (
        f"Too little measured work for the growth ratio to mean anything -- "
        f"the counters are probably not seeing this operation. Expected at "
        f"least {floor:.0f} more operations between {small} and {mid}. "
        f"{series}"
    )
    assert d2 <= limit * d1, (
        f"Cost grows faster than linearly in {axis}. Expect an increment "
        f"ratio near 4.0 for linear work; {limit} is the limit and quadratic "
        f"would be about 16. {series}"
    )


#: How much a "flat" cost may drift across the whole size span.
#:
#: Not 1.0: opening a store, resolving an index and printing a tree all cost
#: a little more when there is more to describe -- a longer index name, a
#: wider shape to format. 1.5 across a 64x span leaves room for that and none
#: at all for reading the data, which would be 64x.
FLAT_TOLERANCE = 1.5

#: Span for the flat guards. Wider than SIZES, because the claim is stronger
#: and a wide span is what makes it convincing.
FLAT_SIZES = (64, 4096)


def assert_independent_of(
    measure: Callable[[int], int],
    *,
    what: str,
    axis: str,
    sizes: tuple = FLAT_SIZES,
    tolerance: float = FLAT_TOLERANCE,
) -> None:
    """Assert `measure` does not grow with `axis` at all.

    A stronger claim than `assert_grows_linearly`, and the right one wherever
    the tool promises work proportional to something other than input size:

    * **Inspection.** `view` and `ls` read shapes, dtypes and attributes and
      never the values behind them, which is the whole reason they return
      instantly on a store too large to open. Linear growth here would mean
      the promise had quietly stopped holding.
    * **Grouping.** Work per row must not depend on how many groups there are.
    * **Streaming.** At a fixed chunk size, peak memory must not track the
      size of the input.

    Each of those reads as obvious prose and none of them is checked by a
    linear-growth guard, which would happily accept a 64x increase.
    """
    small, large = sizes[0], sizes[-1]
    c_small, c_large = measure(small), measure(large)

    span = large / small
    series = (
        f"{what}, scaling {axis}: {small}->{c_small}, {large}->{c_large} "
        f"over a {span:.0f}x span"
    )
    ratio = (c_large / c_small) if c_small else float("inf") if c_large else 1.0

    assert ratio <= tolerance, (
        f"Cost tracks {axis}, and it is supposed to be independent of it. "
        f"Growing in step with the input would be about {span:.0f}x; "
        f"{tolerance}x is the limit. {series} (ratio {ratio:.2f})"
    )


def assert_grows_slower_than_input(
    measure: Callable[[int], int],
    *,
    what: str,
    axis: str,
    sizes: tuple,
    at_least: float,
) -> None:
    """Assert cost grows at least `at_least` times slower than the input.

    The middle ground between `assert_grows_linearly` and
    `assert_independent_of`, and the honest shape of most of this tool's
    streaming: peak memory is not flat -- an index or an indptr is read whole
    -- but it is far below the input curve, and that margin is the feature.

    Stating the margin as a factor rather than an absolute ceiling keeps the
    guard meaningful across platforms and interpreters, and makes a drift
    back towards linear fail long before it becomes a bug report.
    """
    small, large = sizes[0], sizes[-1]
    c_small, c_large = measure(small), measure(large)

    span = large / small
    growth = (c_large / c_small) if c_small else float("inf")
    budget = span / at_least

    assert growth <= budget, (
        f"{what} cost is tracking {axis} too closely. Input grew {span:.0f}x "
        f"and cost grew {growth:.1f}x; the requirement is at least "
        f"{at_least:.0f}x better than the input, i.e. no more than "
        f"{budget:.0f}x. Series: {small}->{c_small}, {large}->{c_large}"
    )


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
    def reads(self) -> int:
        """The headline number: array elements plus store-level fetches.

        Covers both the "read the same column n times" shape and the
        "re-fetch the same chunk n times" shape.
        """
        return self.elements + self.store_get

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"elements={self.elements} (h5={self.h5_elements} "
            f"zarr={self.zarr_elements}) calls={self.calls} "
            f"store: get={self.store_get} set={self.store_set} "
            f"delete={self.store_delete}"
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

    ===================================  ====================================
    ``h5py.Dataset.__getitem__``         HDF5 reads
    ``zarr.Array.__getitem__``           Zarr reads, v2 and v3 alike
    ``zarr.storage.LocalStore.get``      chunk and metadata fetches
    ``LocalStore.set`` / ``.delete``     write storms
    ===================================  ====================================

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
    zarr.Array.__getitem__ = zarr_wrapper
    LocalStore.get = get_wrapper
    LocalStore.set = set_wrapper
    LocalStore.delete = delete_wrapper
    try:
        yield counts
    finally:
        h5py.Dataset.__getitem__ = h5_get
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

    assert d1 >= mid, (
        f"Too little measured work for the growth ratio to mean anything -- "
        f"the counters are probably not seeing this operation. {series}"
    )
    assert d2 <= limit * d1, (
        f"Cost grows faster than linearly in {axis}. Expect an increment "
        f"ratio near 4.0 for linear work; {limit} is the limit and quadratic "
        f"would be about 16. {series}"
    )

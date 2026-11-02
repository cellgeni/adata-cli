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

Bytes come in three kinds, because each catches a different defect:

* **returned** (`read_bytes`, `largest_read`): what the caller holds. The
  largest single read is the block a streaming loop keeps in memory, so a
  guard on it guards memory without needing an allocator hook at all.
* **decoded** (`decoded_bytes`): every chunk a read touches, whole. The
  backends decompress whole chunks, so taking one row from each of a thousand
  tall chunks costs a thousand chunks, however little the read returned.
* **encoded** (`encoded_bytes`): every chunk a write touches, whole, and every
  shard for a sharded Zarr array, since writing part of a shard rewrites all
  of it. A loop writing blocks smaller than that unit re-encodes each unit
  once per block.

Decoded and encoded bytes are a model worked out from the selection, the
chunk grid and the codecs, not a measurement of the library, which is what
keeps them deterministic. It follows HDF5's per-dataset chunk cache; see
`_ChunkCache`.

Memory has two instruments. `count_allocations` (tracemalloc) is exact and
in-process, and sees numpy's buffers but not what HDF5, blosc or zarr's codecs
allocate in C. `peak_rss` runs a command in a fresh interpreter and reads the
operating system's high-water mark, which sees everything, with a few
megabytes of noise: guard it with generous ratios.

Known bypasses
--------------
These reach the file without passing through any hook below:

* ``h5py.Dataset.read_direct``
* ``np.asarray(dataset)`` and ``dataset[...]`` via ``__array__`` (verified:
  ``np.asarray`` adds nothing to the element count)
* ``dataset.asstr()[...]``, which goes through ``AsStrWrapper.__getitem__``
* ``dataset.fields(...)``
* ``zarr.Array.vindex[...]`` and the ``get_*_selection`` methods

``zarr.Array.oindex[...]`` is counted: `core/subset.py` uses it.

None of the others are used in `src/adata/` today. If you add one, the
counters will silently report less work than actually happened -- which is why every guard also
asserts a *lower* bound, and why `test_performance.py` keeps a canary test with
a known absolute count.
"""

from __future__ import annotations

import itertools
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
    store_set_bytes: int = 0
    store_delete: int = 0

    #: Bytes the reads handed back, through either backend.
    read_bytes: int = 0
    #: The most bytes one read handed back: the block a loop holds.
    largest_read: int = 0
    #: Bytes of the whole chunks those reads touched (see the module notes).
    decoded_bytes: int = 0
    #: Bytes handed to writes, including data given at creation.
    written_bytes: int = 0
    #: The most bytes one write was handed.
    largest_write: int = 0
    #: Bytes of the whole chunks, or shards, those writes touched.
    encoded_bytes: int = 0

    #: Per-dataset element counts, for diagnosing which column blew up.
    by_name: Dict[str, int] = field(default_factory=dict)
    #: The largest read per dataset, in bytes and in elements.
    largest_by_name: Dict[str, int] = field(default_factory=dict)
    largest_elements_by_name: Dict[str, int] = field(default_factory=dict)

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

    @property
    def read_amplification(self) -> float:
        """Bytes decoded per byte returned: 1.0 when reads follow the chunks."""
        return self.decoded_bytes / self.read_bytes if self.read_bytes else 0.0

    @property
    def write_amplification(self) -> float:
        """Bytes re-encoded per byte written: 1.0 when writes fill whole units."""
        return self.encoded_bytes / self.written_bytes if self.written_bytes else 0.0

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"read={self.elements} (h5={self.h5_elements} "
            f"zarr={self.zarr_elements}) wrote={self.h5_written_elements} "
            f"calls={self.calls} store: get={self.store_get} "
            f"set={self.store_set} delete={self.store_delete} "
            f"bytes: read={self.read_bytes} (largest {self.largest_read}) "
            f"decoded={self.decoded_bytes} written={self.written_bytes} "
            f"(largest {self.largest_write}) encoded={self.encoded_bytes}"
        )


def _size(value: Any) -> int:
    try:
        return int(np.asarray(value).size)
    except Exception:  # pragma: no cover - exotic dtypes
        return 1


def _nbytes(value: Any) -> int:
    try:
        return int(np.asarray(value).nbytes)
    except Exception:  # pragma: no cover - exotic dtypes
        return 0


def _axis_chunks(key: Any, extent: int, size: int) -> int:
    """How many chunks of `size` along an axis of `extent` a key touches."""
    if extent == 0 or size <= 0:
        return 0
    if isinstance(key, (int, np.integer)):
        return 1
    if isinstance(key, slice):
        start, stop, step = key.indices(extent)
        if step == 1:
            return (stop - 1) // size - start // size + 1 if stop > start else 0
        picked = np.arange(start, stop, step)
    else:
        picked = np.asarray(key)
        if picked.dtype == bool:
            picked = np.flatnonzero(picked)
        picked = picked.astype(np.int64, copy=False).ravel()
        picked = np.where(picked < 0, picked + extent, picked)
    return int(np.unique(picked // size).size) if picked.size else 0


def _axis_keys(key: Any, ndim: int) -> tuple:
    """One key per axis: Ellipsis expanded, missing trailing axes whole."""
    keys = key if isinstance(key, tuple) else (key,)
    keys = tuple(k for k in keys if k is not None and not isinstance(k, str))
    if any(k is Ellipsis for k in keys):
        at = next(i for i, k in enumerate(keys) if k is Ellipsis)
        keys = keys[:at] + (slice(None),) * (ndim - len(keys) + 1) + keys[at + 1 :]
    return keys + (slice(None),) * (ndim - len(keys))


def chunks_touched(key: Any, shape: Any, chunks: Any) -> int:
    """How many chunks of a grid a selection touches, axis by axis.

    Orthogonal selections only, which is what h5py and `oindex` take: the
    count is the product of the chunks touched along each axis.
    """
    shape = tuple(int(n) for n in shape)
    total = 1
    for k, extent, size in zip(_axis_keys(key, len(shape)), shape, chunks):
        total *= _axis_chunks(k, extent, int(size))
    return total


def _unit_bytes(array: Any, unit: Any) -> int:
    count = 1
    for c in unit:
        count *= int(c)
    return count * int(np.dtype(array.dtype).itemsize)


def _chunk_grid(array: Any, *, write: bool) -> Any:
    """The unit a read decodes, or a write re-encodes: None if contiguous."""
    if write:
        shards = getattr(array, "shards", None)
        if shards:
            return shards
    return getattr(array, "chunks", None)


def _axis_indices(key: Any, extent: int, size: int) -> np.ndarray:
    """The chunk indices along one axis a key touches, sorted."""
    if isinstance(key, (int, np.integer)):
        return np.array([(int(key) % extent) // size]) if extent else np.array([], int)
    if isinstance(key, slice):
        start, stop, step = key.indices(extent)
        if step == 1:
            if stop <= start:
                return np.array([], dtype=np.int64)
            return np.arange(start // size, (stop - 1) // size + 1)
        picked = np.arange(start, stop, step)
    else:
        picked = np.asarray(key)
        if picked.dtype == bool:
            picked = np.flatnonzero(picked)
        picked = picked.astype(np.int64, copy=False).ravel()
        picked = np.where(picked < 0, picked + extent, picked)
    return np.unique(picked // size)


#: Past this many chunks in one selection, the cache is not modelled: a read
#: that large evicts any cache a backend has anyway.
_MODELLED_CHUNKS = 1 << 16


class _ChunkCache:
    """Which chunks a backend still holds decoded, dataset by dataset.

    HDF5 keeps a chunk cache per dataset (1 MiB in the stores `open_store`
    opens, 8 MiB by default since HDF5 2.0, none for a chunk bigger than the
    cache) and serves a read or a write from it without touching the file:
    appending a block at a time to a growing dataset encodes each chunk
    once, when it leaves the cache, not once a block. Modelled as least
    recently used, at the dataset's own cache size. A chunk is read into the
    cache whole, filtered or not; only an unfiltered chunk too big for the
    cache is read and written in place, just the part asked for. Zarr caches
    nothing, so every chunk a selection touches is decoded and every shard
    it writes is rewritten.
    """

    def __init__(self) -> None:
        self.held: Dict[Any, Dict[Any, None]] = {}

    @staticmethod
    def _unfiltered_hdf5(array: Any) -> bool:
        """Whether an HDF5 dataset's chunks have no filter."""
        plist = getattr(getattr(array, "id", None), "get_create_plist", None)
        if plist is None:
            return False  # Zarr
        try:
            return plist().get_nfilters() == 0
        except Exception:  # pragma: no cover - an HDF5 without the call
            return False

    @staticmethod
    def _capacity(array: Any, unit_bytes: int) -> int:
        plist = getattr(getattr(array, "id", None), "get_access_plist", None)
        if plist is None or unit_bytes <= 0:
            return 0  # Zarr
        try:
            cache_bytes = int(plist().get_chunk_cache()[1])
        except Exception:  # pragma: no cover - an HDF5 without the call
            return 0
        return cache_bytes // unit_bytes

    def cost(self, array: Any, key: Any, returned: int, *, write: bool) -> int:
        """Bytes of the whole chunks (or shards) `key` makes the backend code."""
        try:
            grid = _chunk_grid(array, write=write)
            shape = tuple(int(n) for n in array.shape)
        except Exception:  # pragma: no cover - an object without a layout
            return returned
        if not grid or not shape:
            return returned  # contiguous, or a scalar: what was asked is what moved
        unit = _unit_bytes(array, grid)
        capacity = self._capacity(array, unit)
        if capacity == 0 and self._unfiltered_hdf5(array):
            return returned  # bypasses the cache, in place
        touched = chunks_touched(key, shape, grid)
        if capacity == 0 or touched > _MODELLED_CHUNKS:
            return touched * unit

        axes = [
            _axis_indices(k, extent, int(size))
            for k, extent, size in zip(_axis_keys(key, len(shape)), shape, grid)
        ]
        held = self.held.setdefault((str(array.file.filename), array.name), {})
        missed = 0
        for coordinate in itertools.product(*(a.tolist() for a in axes)):
            if coordinate in held:
                del held[coordinate]  # re-inserted below, as most recent
            else:
                missed += 1
            held[coordinate] = None
            if len(held) > capacity:
                del held[next(iter(held))]
        return missed * unit


@contextmanager
def count_io() -> Iterator[IOCounts]:
    """Count backend reads and writes for the duration of the block.

    Patches these seams, verified against h5py 3.16.0 and zarr 3.1.5:

    ======================================  =================================
    ``h5py.Dataset.__getitem__``            HDF5 reads
    ``h5py.Dataset.__setitem__``            HDF5 writes
    ``h5py.Group.create_dataset``           HDF5 writes made at creation
    ``zarr.Array.__getitem__``              Zarr reads, v2 and v3 alike
    ``zarr.Array.oindex[...]``              Zarr orthogonal reads
    ``zarr.Array.__setitem__``              Zarr writes (bytes only)
    ``zarr.Group.create_array``             Zarr writes made at creation
    ``zarr.storage.LocalStore.get``         chunk and metadata fetches
    ``LocalStore.set`` / ``.delete``        write storms
    ======================================  =================================

    Both HDF5 write seams are needed: `create_dataset(name, data=...)` writes
    its payload during creation and never touches `__setitem__`, so hooking
    only the latter leaves `import` and `create` measuring zero. Zarr's array
    writes add to the byte counts but not to `writes`, which counts them in
    store keys (see `IOCounts.writes`).

    The store hooks are the ones that see a metadata write storm: attribute
    writes never touch `Array.__getitem__`, so an O(k^2) attribute rewrite is
    invisible at the array level and obvious at the store level.

    The `LocalStore` methods are coroutines and are wrapped as coroutines.
    """
    import h5py
    import zarr
    from zarr.core.indexing import OIndex
    from zarr.storage import LocalStore

    counts = IOCounts()
    cache = _ChunkCache()

    h5_get = h5py.Dataset.__getitem__
    h5_set = h5py.Dataset.__setitem__
    h5_create = h5py.Group.create_dataset
    zarr_get = zarr.Array.__getitem__
    zarr_set = zarr.Array.__setitem__
    zarr_create = zarr.Group.create_array
    oindex_get = OIndex.__getitem__
    store_get = LocalStore.get
    store_set = LocalStore.set
    store_delete = LocalStore.delete

    def _record(name: str, n: int) -> None:
        counts.by_name[name] = counts.by_name.get(name, 0) + n

    def _read(array: Any, key: Any, result: Any, name: str) -> int:
        n = _size(result)
        returned = _nbytes(result)
        counts.read_bytes += returned
        counts.largest_read = max(counts.largest_read, returned)
        counts.largest_by_name[name] = max(counts.largest_by_name.get(name, 0), returned)
        counts.largest_elements_by_name[name] = max(
            counts.largest_elements_by_name.get(name, 0), n
        )
        counts.decoded_bytes += cache.cost(array, key, returned, write=False)
        _record(name, n)
        return n

    def _write(array: Any, key: Any, value: Any) -> None:
        given = _nbytes(value)
        counts.written_bytes += given
        counts.largest_write = max(counts.largest_write, given)
        counts.encoded_bytes += cache.cost(array, key, given, write=True)

    def h5_wrapper(self: Any, key: Any) -> Any:
        result = h5_get(self, key)
        counts.h5_calls += 1
        counts.h5_elements += _read(self, key, result, getattr(self, "name", "?"))
        return result

    def zarr_wrapper(self: Any, key: Any) -> Any:
        result = zarr_get(self, key)
        counts.zarr_calls += 1
        counts.zarr_elements += _read(self, key, result, getattr(self, "path", "?") or "/")
        return result

    def oindex_wrapper(self: Any, key: Any) -> Any:
        result = oindex_get(self, key)
        array = self.array
        counts.zarr_calls += 1
        counts.zarr_elements += _read(array, key, result, getattr(array, "path", "?") or "/")
        return result

    def h5_set_wrapper(self: Any, key: Any, value: Any) -> Any:
        counts.h5_write_calls += 1
        counts.h5_written_elements += _size(value)
        _write(self, key, value)
        return h5_set(self, key, value)

    def zarr_set_wrapper(self: Any, key: Any, value: Any) -> Any:
        _write(self, key, value)
        return zarr_set(self, key, value)

    def h5_create_wrapper(self: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
        counts.h5_write_calls += 1
        data = kwargs.get("data")
        if data is None and args:
            # positional signature is (name, shape, dtype, data, ...)
            data = args[2] if len(args) > 2 else None
        if data is not None:
            counts.h5_written_elements += _size(data)
        created = h5_create(self, name, *args, **kwargs)
        if data is not None:
            _write(created, Ellipsis, data)
        return created

    def zarr_create_wrapper(self: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
        created = zarr_create(self, name, *args, **kwargs)
        if kwargs.get("data") is not None:
            _write(created, Ellipsis, kwargs["data"])
        return created

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
        value = args[1] if len(args) > 1 else kwargs.get("value")
        try:
            counts.store_set_bytes += len(value)
        except TypeError:  # pragma: no cover - prototype buffers
            pass
        return await store_set(self, *args, **kwargs)

    async def delete_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts.store_delete += 1
        return await store_delete(self, *args, **kwargs)

    patched = [
        (h5py.Dataset, "__getitem__", h5_get, h5_wrapper),
        (h5py.Dataset, "__setitem__", h5_set, h5_set_wrapper),
        (h5py.Group, "create_dataset", h5_create, h5_create_wrapper),
        (zarr.Array, "__getitem__", zarr_get, zarr_wrapper),
        (zarr.Array, "__setitem__", zarr_set, zarr_set_wrapper),
        (zarr.Group, "create_array", zarr_create, zarr_create_wrapper),
        (OIndex, "__getitem__", oindex_get, oindex_wrapper),
        (LocalStore, "get", store_get, get_wrapper),
        (LocalStore, "set", store_set, set_wrapper),
        (LocalStore, "delete", store_delete, delete_wrapper),
    ]
    for owner, name, _, wrapper in patched:
        setattr(owner, name, wrapper)
    try:
        yield counts
    finally:
        for owner, name, original, _ in patched:
            setattr(owner, name, original)


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


#: Run by `peak_rss` in a fresh interpreter. The high-water mark is read once
#: the CLI and everything it imports is loaded, and again after the command,
#: so the difference is what the command itself added.
_RSS_RUNNER = r"""
import json, resource, sys

def high_water():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024  # Linux: KiB

argv, report = json.loads(sys.argv[1]), sys.argv[2]
from adata.cli import app

before = high_water()
try:
    code = app(args=argv, prog_name="adata", standalone_mode=False) or 0
except SystemExit as stop:
    code = stop.code or 0
with open(report, "w") as handle:
    json.dump({"before": before, "after": high_water(), "code": code}, handle)
"""


@dataclass
class PeakRSS:
    """Resident memory of one command, in bytes."""

    #: The high-water mark once the CLI was imported, before the command ran.
    before: int
    #: The high-water mark when the command had finished.
    after: int
    #: What the command's stderr said, for a failure message.
    stderr: str = ""

    @property
    def added(self) -> int:
        """How far the command raised the high-water mark."""
        return max(0, self.after - self.before)


def peak_rss(argv: List[str], *, cwd: Any = None) -> PeakRSS:
    """Peak resident memory of `adata <argv>`, run in a fresh interpreter.

    The instrument that sees everything: HDF5's chunk caches, the buffers
    blosc and zarr's codecs decode into, and anything else allocated in C,
    none of which tracemalloc can see. A subprocess because the high-water
    mark only rises: in this process it would hold whatever the test suite
    reached before. Not exact -- the allocator keeps a few megabytes it has
    freed -- so guard it with ratios that leave room, against inputs large
    enough (tens of megabytes) that holding one would be unmistakable.
    """
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "rss.json"
        done = subprocess.run(
            [sys.executable, "-c", _RSS_RUNNER, json.dumps([str(a) for a in argv]),
             str(report)],
            cwd=cwd, capture_output=True, text=True, check=False,
        )
        if done.returncode != 0 or not report.exists():
            raise RuntimeError(
                f"adata {' '.join(map(str, argv))} could not be measured "
                f"(exit {done.returncode}):\n{done.stderr[-2000:]}"
            )
        measured = json.loads(report.read_text())
    if measured["code"]:
        raise RuntimeError(
            f"adata {' '.join(map(str, argv))} failed (exit {measured['code']}):\n"
            f"{done.stdout[-2000:]}{done.stderr[-2000:]}"
        )
    return PeakRSS(measured["before"], measured["after"], done.stderr)


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



#: The most memory a streaming path may add per byte of input added. A path
#: holding a block of `--chunk` rows of a matrix many blocks tall adds a few
#: hundredths; one holding the matrix adds a whole byte, and more for every
#: copy it makes. A quarter leaves room for the per-row arrays a path keeps
#: by design (an indptr, a row's count) on a narrow fixture.
INPUT_SHARE = 0.25


def assert_memory_is_a_fraction_of_the_input(
    measure: Callable[[int], int],
    input_bytes: Callable[[int], int],
    *,
    what: str,
    axis: str,
    sizes: tuple,
    at_most: float = INPUT_SHARE,
) -> None:
    """Assert peak memory grows by at most `at_most` of what the input grows by.

    `measure(n)` is peak memory at size `n` (allocations or RSS), and
    `input_bytes(n)` the size of the input in memory at `n`. Increments, as
    in `assert_grows_linearly`, so fixed costs cancel.

    This holds along either axis, which a growth ratio does not: widen a
    matrix and a block of `--chunk` rows widens with it, so its memory grows
    exactly as fast as the input does, and a ratio cannot tell it from the
    whole matrix. As a share of the input it is still `--chunk` / n_obs,
    against 1 or more for a path that holds the matrix.
    """
    small, large = sizes[0], sizes[-1]
    m_small, m_large = measure(small), measure(large)
    i_small, i_large = input_bytes(small), input_bytes(large)
    added = i_large - i_small
    assert added > 0, f"{what}: the input did not grow between {small} and {large}"
    share = (m_large - m_small) / added
    assert share <= at_most, (
        f"{what} holds memory in proportion to the input, scaling {axis}: the "
        f"input grew by {added / 1e6:.2f} MB ({small}->{large}) and peak memory by "
        f"{(m_large - m_small) / 1e6:.2f} MB, {share:.2f} of it, against at most "
        f"{at_most}. Peak {m_small / 1e6:.2f} MB -> {m_large / 1e6:.2f} MB."
    )

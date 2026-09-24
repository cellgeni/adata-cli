"""Subset operations for .h5ad and .zarr stores."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional, Set, Tuple, List, Dict, Any

import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from adata.elements import spec
from adata.elements.write import set_shape_attr, write_mapping
from adata.core.select import read_names, select_indices
from adata.elements.read import (
    decode_str_array,
    element_len,
    read_str_chunk,
    resolve_index,
)
from adata.storage import (
    create_dataset,
    copy_attrs,
    copy_tree,
    dataset_create_kwargs,
    detect_backend,
    is_dataset,
    is_group,
    is_zarr_group,
    is_zarr_array,
    open_store,
)


def _target_backend(dst_group: Any) -> str:
    return "zarr" if is_zarr_group(dst_group) else "hdf5"


def _ensure_group(parent: Any, name: str) -> Any:
    """Get or create an AnnData mapping group, tagged `encoding-type: dict`.

    Every container here (layers, obsm, obsp, varm, varp) is a mapping in the
    spec; leaving one untagged makes anndata fall back to its legacy reader
    and emit an OldFormatWarning.
    """
    return write_mapping(parent, name)


def _group_get(parent: Any, key: str) -> Any | None:
    return parent[key] if key in parent else None


def _ensure_optional_anndata_groups(dst: Any) -> None:
    """Create the optional mapping groups an AnnData store is expected to have."""
    for key in ("layers", "obsm", "obsp", "varm", "varp"):
        _ensure_group(dst, key)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_name_file(path: Path) -> Set[str]:
    names: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line)
    return names


def indices_from_name_set(
    names_ds: Any,
    keep: Set[str],
    *,
    chunk_size: int = 200_000,
) -> Tuple[np.ndarray, Set[str]]:
    """Resolve a set of names to sorted row indices, streaming the index.

    `names_ds` may be a dataset or -- as anndata >= 0.11 writes indices -- a
    `nullable-string-array` group, so it is read through
    :func:`adata.elements.read.read_str_chunk` rather than sliced directly.
    Returns the indices found and the names that were not.
    """
    flat_len = element_len(names_ds)

    remaining = set(keep)
    found_indices: List[int] = []
    cache: Dict[str, np.ndarray] = {}

    for start in range(0, flat_len, chunk_size):
        end = min(start + chunk_size, flat_len)
        for i, name in enumerate(read_str_chunk(names_ds, start, end, cache)):
            if name in remaining:
                found_indices.append(start + i)
                remaining.remove(name)

        if not remaining:
            break

    return np.asarray(found_indices, dtype=np.int64), remaining


#: Rows `_take_rows` reads from HDF5 in one contiguous slice.
TAKE_BLOCK_ROWS = 1 << 20


def _take_rows(obj: Any, indices: Optional[np.ndarray]) -> Any:
    """Read the selected rows of a dataset, handling both backends' indexing.

    On HDF5 the rows are read as contiguous blocks and picked out in memory,
    not by passing `indices` to h5py. h5py turns a fancy index into one
    hyperslab per row, which made reading 6,250 rows of a 50,000-row column
    take 0.38 s against 5 ms for the whole column. `split` does that for every
    column of every group, so it came to most of split's run time.
    """
    if indices is None:
        return obj[...]
    if is_zarr_array(obj):
        if obj.ndim == 1:
            return obj.oindex[indices]
        return obj.oindex[(indices,) + (slice(None),) * (obj.ndim - 1)]
    _check_sorted(indices, "row")
    if indices.size == 0:
        return obj[0:0, ...]

    parts = []
    first, last = int(indices[0]), int(indices[-1]) + 1
    for start in range(first, last, TAKE_BLOCK_ROWS):
        end = min(start + TAKE_BLOCK_ROWS, last)
        lo, hi = np.searchsorted(indices, [start, end], side="left")
        if hi > lo:
            parts.append(np.asarray(obj[start:end, ...])[indices[lo:hi] - start])
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _copy_rows(
    src_ds: Any,
    dst_parent: Any,
    name: str,
    indices: Optional[np.ndarray],
) -> Any:
    """Write the selected rows of `src_ds` into `dst_parent` as `name`."""
    if indices is None:
        return copy_tree(src_ds, dst_parent, name)

    target_backend = _target_backend(dst_parent)
    kw = dataset_create_kwargs(
        src_ds, target_backend=target_backend, dst_parent=dst_parent
    )
    kw = _clamp_chunks(kw, len(indices))
    ds = create_dataset(
        dst_parent,
        name,
        data=_take_rows(src_ds, indices),
        **kw,
    )
    copy_attrs(src_ds.attrs, ds.attrs, target_backend=target_backend)
    return ds


def _clamp_chunks(kw: dict, *out_shape: int) -> dict:
    """Shrink a forwarded chunk shape to fit the subset's dimensions.

    h5py rejects a chunk larger than the dataset, so a chunked source subset
    below its own chunk size would otherwise fail outright.

    Zarr additionally requires a shard to be a whole number of chunks, so a
    clamped chunk invalidates the source's shard geometry. Sharding is a
    storage-layout choice rather than data, so it is dropped and left to the
    backend rather than recomputed into something arbitrary.
    """
    chunks = kw.get("chunks")
    if not isinstance(chunks, (tuple, list)) or not chunks:
        return kw

    original = tuple(int(c) for c in chunks)
    clamped = tuple(
        min(c, out_shape[i]) if i < len(out_shape) and out_shape[i] > 0 else c
        for i, c in enumerate(original)
    )
    if clamped == original:
        return kw

    kw = dict(kw)
    kw["chunks"] = clamped
    kw.pop("shards", None)
    return kw


def subset_axis_group(
    src: Any,
    dst: Any,
    indices: Optional[np.ndarray],
) -> None:
    """Copy a dataframe group, taking only `indices` along its rows.

    Every column layout the spec allows has to be narrowed here, not just plain
    datasets: `categorical` keeps its categories and subsets only `codes`,
    while the masked layouts (`nullable-*`, which anndata >= 0.11 uses for the
    index and every string column) subset both `values` and `mask`. Copying a
    masked column whole would leave it longer than the rest of the frame.
    """
    target_backend = _target_backend(dst)
    copy_attrs(src.attrs, dst.attrs, target_backend=target_backend)

    for key in src.keys():
        obj = src[key]

        if is_dataset(obj):
            _copy_rows(obj, dst, key, indices)
            continue

        if not is_group(obj):
            continue

        enc = _decode_attr(obj.attrs.get("encoding-type", b""))

        if enc == spec.CATEGORICAL or (enc is None and "codes" in obj):
            gdst = dst.create_group(key)
            copy_attrs(obj.attrs, gdst.attrs, target_backend=target_backend)
            copy_tree(obj["categories"], gdst, "categories")
            _copy_rows(obj["codes"], gdst, "codes", indices)
            continue

        if enc in spec.MASKED_TYPES or ("values" in obj and "mask" in obj):
            gdst = dst.create_group(key)
            copy_attrs(obj.attrs, gdst.attrs, target_backend=target_backend)
            _copy_rows(obj["values"], gdst, "values", indices)
            _copy_rows(obj["mask"], gdst, "mask", indices)
            continue

        # Not row-aligned (e.g. __categories) -- copy verbatim.
        copy_tree(obj, dst, key)


def _minor_remap(keep: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
    """Build a lookup from old minor index to new, with -1 for dropped entries.

    A dense lookup table costs one int64 per column of the source, which is
    negligible beside the matrix itself and turns the remap into a single
    vectorised gather rather than a per-entry dict lookup.
    """
    if keep is None:
        return None
    remap = np.full(size, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep), dtype=np.int64)
    return remap


#: One output of a matrix write: the parent to create it in, and the obs and
#: var indices it keeps (None for all). Several of these share one read of
#: the source, which is what lets `split` make a single pass.
MatrixTarget = Tuple[Any, Optional[np.ndarray], Optional[np.ndarray]]


def _check_sorted(indices: Optional[np.ndarray], what: str) -> None:
    """Refuse a selection the streaming writers cannot honour.

    The writers walk the source once, in order, so the rows they emit come
    out in source order. A selection that is not strictly increasing would
    be silently reordered or deduplicated, so it is refused instead.
    h5py's fancy indexing has the same requirement, so every selection the
    commands produce already meets it.
    """
    if indices is not None and indices.size > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError(f"{what} indices must be sorted and unique.")


class _Cursor:
    """Hand out, block by block, the rows of one target's major selection.

    The blocks arrive in increasing order and the selection is sorted, so
    each block's rows are the next slice of it: finding them is one
    `searchsorted`, not a scan.
    """

    def __init__(self, selection: Optional[np.ndarray]) -> None:
        self.selection = selection
        self.pos = 0

    def rows(self, lo: int, hi: int) -> np.ndarray:
        if self.selection is None:
            return np.arange(lo, hi, dtype=np.int64)
        end = int(np.searchsorted(self.selection, hi, side="left"))
        rows = self.selection[self.pos : end]
        self.pos = end
        return rows


def subset_dense_matrix(
    src: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int = 1024,
) -> None:
    fan_out_dense_matrix(src, name, [(dst_parent, obs_idx, var_idx)], chunk_rows=chunk_rows)


def fan_out_dense_matrix(
    src: Any,
    name: str,
    targets: List[MatrixTarget],
    *,
    chunk_rows: int = 1024,
) -> None:
    """Write one row/column subset of a dense matrix per target, in one pass.

    The source is read a block of `chunk_rows` rows at a time, each block
    once whatever the number of targets, and only the span between the
    first and last row any target keeps in it.
    """
    if src.ndim != 2:
        for dst_parent, _, _ in targets:
            copy_tree(src, dst_parent, name)
        return

    n_obs, n_var = src.shape
    outputs = []
    for dst_parent, obs_idx, var_idx in targets:
        _check_sorted(obs_idx, "obs")
        out_obs = len(obs_idx) if obs_idx is not None else n_obs
        out_var = len(var_idx) if var_idx is not None else n_var

        target_backend = _target_backend(dst_parent)
        kw = dataset_create_kwargs(
            src, target_backend=target_backend, dst_parent=dst_parent
        )
        kw = _clamp_chunks(kw, out_obs, out_var)
        dst = create_dataset(
            dst_parent, name, shape=(out_obs, out_var), dtype=src.dtype, **kw
        )
        copy_attrs(src.attrs, dst.attrs, target_backend=target_backend)
        outputs.append([dst, _Cursor(obs_idx), var_idx, 0])

    for start in range(0, n_obs, chunk_rows):
        end = min(start + chunk_rows, n_obs)
        wanted = [out[1].rows(start, end) for out in outputs]
        nonempty = [rows for rows in wanted if rows.size]
        if not nonempty:
            continue
        first = min(int(rows[0]) for rows in nonempty)
        last = max(int(rows[-1]) for rows in nonempty)
        block = np.asarray(src[first : last + 1, :])

        for out, rows in zip(outputs, wanted):
            if not rows.size:
                continue
            dst, _, var_idx, written = out
            part = block[rows - first]
            if var_idx is not None:
                part = part[:, var_idx]
            dst[written : written + len(rows), :] = part
            out[3] = written + len(rows)


class _SparseOutput:
    """One target's share of a streamed sparse subset."""

    def __init__(
        self,
        src: Any,
        dst_parent: Any,
        name: str,
        enc: str,
        out_shape: Tuple[int, int],
        major_idx: Optional[np.ndarray],
        remap: Optional[np.ndarray],
    ) -> None:
        self.group = dst_parent.create_group(name)
        copy_attrs(src.attrs, self.group.attrs, target_backend=_target_backend(dst_parent))
        spec.set_encoding(self.group, enc)
        set_shape_attr(self.group, out_shape)

        # The source's own dtypes, not int64: a subset has no more nonzeros
        # than its source and no coordinate beyond the source's dimensions,
        # so whatever held the source holds the subset. Widening them made
        # every output about half as large again as it needed to be.
        self.indptr_template = src["indptr"]
        self.data = _growable_like(self.group, "data", src["data"].dtype, src["data"])
        self.indices = _growable_like(
            self.group, "indices", src["indices"].dtype, src["indices"]
        )
        self.cursor = _Cursor(major_idx)
        self.remap = remap
        self.counts: List[np.ndarray] = []

    def take(
        self,
        rows: np.ndarray,
        indptr: np.ndarray,
        lo: int,
        block_indices: np.ndarray,
        block_data: np.ndarray,
    ) -> None:
        """Append the entries of `rows`, whose data starts at offset `lo`."""
        starts = indptr[rows] - lo
        lengths = indptr[rows + 1] - indptr[rows]
        total = int(lengths.sum())

        if total and self.cursor.selection is None:
            # Every row of the block: its entries are one contiguous run.
            begin = int(starts[0])
            positions = slice(begin, begin + total)
        else:
            # Each row's run, laid end to end: the offset of an entry within
            # its row plus the start of that row in the block.
            ends = np.cumsum(lengths)
            positions = np.repeat(starts - (ends - lengths), lengths) + np.arange(
                total, dtype=np.int64
            )

        minor = block_indices[positions]
        values = block_data[positions]
        counts = lengths
        if self.remap is not None:
            mapped = self.remap[minor]
            keep = mapped >= 0
            owner = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
            counts = np.bincount(owner[keep], minlength=len(rows))
            minor, values = mapped[keep], values[keep]

        self.counts.append(counts)
        _append(self.indices, minor.astype(self.indices.dtype, copy=False))
        _append(self.data, values)

    def finish(self) -> None:
        counts = (
            np.concatenate(self.counts) if self.counts else np.empty(0, dtype=np.int64)
        )
        indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        dtype = self.indptr_template.dtype
        dataset = _sparse_dataset(
            self.group, "indptr", dtype, indptr.size, self.indptr_template
        )
        dataset[:] = indptr.astype(dtype, copy=False)


def subset_sparse_matrix_group(
    src: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_major: int = 4096,
) -> None:
    """Subset a CSR/CSC matrix, streaming it a block of major axis at a time."""
    fan_out_sparse_matrix(
        src, name, [(dst_parent, obs_idx, var_idx)], chunk_major=chunk_major
    )


def fan_out_sparse_matrix(
    src: Any,
    name: str,
    targets: List[MatrixTarget],
    *,
    chunk_major: int = 4096,
) -> None:
    """Write one subset of a CSR/CSC matrix per target, reading it once.

    The source's major axis is walked in contiguous blocks of `chunk_major`,
    and each block's `data`/`indices` are read once -- only the span between
    the first and last row any target keeps -- then shared out among the
    targets. Peak memory is set by the block, not the matrix.

    This is what makes `split` one pass. An earlier version subset once per
    target, reading the span of each target's own selected rows; when the
    groups were interleaved that span was nearly the whole matrix, so a split
    into k groups read and decompressed X k times.
    """
    enc = _decode_attr(src.attrs.get("encoding-type", b""))
    if enc not in spec.SPARSE_TYPES:
        raise ValueError(f"Unsupported sparse encoding type: {enc}")

    shape = src.attrs.get("shape", None)
    if shape is None:
        raise ValueError("Sparse matrix group missing 'shape' attribute.")
    n_rows, n_cols = int(shape[0]), int(shape[1])

    data_ds, indices_ds = src["data"], src["indices"]
    indptr = np.asarray(src["indptr"][...], dtype=np.int64)
    csr = enc == spec.CSR_MATRIX
    n_major, n_minor = (n_rows, n_cols) if csr else (n_cols, n_rows)

    outputs: List[_SparseOutput] = []
    for dst_parent, obs_idx, var_idx in targets:
        major_idx, minor_keep = (obs_idx, var_idx) if csr else (var_idx, obs_idx)
        _check_sorted(major_idx, "obs" if csr else "var")
        out_shape = (
            len(obs_idx) if obs_idx is not None else n_rows,
            len(var_idx) if var_idx is not None else n_cols,
        )
        outputs.append(
            _SparseOutput(
                src,
                dst_parent,
                name,
                enc,
                out_shape,
                major_idx,
                _minor_remap(minor_keep, n_minor),
            )
        )

    for start in range(0, n_major, chunk_major):
        end = min(start + chunk_major, n_major)
        wanted = [out.cursor.rows(start, end) for out in outputs]
        nonempty = [rows for rows in wanted if rows.size]
        if not nonempty:
            continue
        lo = int(indptr[min(int(rows[0]) for rows in nonempty)])
        hi = int(indptr[max(int(rows[-1]) for rows in nonempty) + 1])
        if hi > lo:
            block_indices = np.asarray(indices_ds[lo:hi])
            block_data = np.asarray(data_ds[lo:hi])
        else:
            block_indices = np.empty(0, dtype=indices_ds.dtype)
            block_data = np.empty(0, dtype=data_ds.dtype)

        for out, rows in zip(outputs, wanted):
            if rows.size:
                out.take(rows, indptr, lo, block_indices, block_data)

    for out in outputs:
        out.finish()


def _growable(group: Any, name: str, dtype: Any) -> Any:
    """Create an empty 1-D scratch dataset that can be extended as blocks arrive.

    For intermediate buffers only: it forwards no storage settings. Output
    arrays go through `_growable_like`, which lays them out like the source.
    """
    if is_zarr_group(group):
        return group.create_array(name, shape=(0,), dtype=dtype, chunks=(65536,))
    return group.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(65536,)
    )


def _sparse_dataset(
    group: Any, name: str, dtype: np.dtype, n: int, template: Any
) -> Any:
    """A 1-D dataset of `n` elements laid out like `template`.

    Forwarding compression and chunking matters more here than anywhere
    else: the point of a dtype change is usually to make the file smaller,
    and creating the output with a fixed 65,536-element chunk made a
    2,400-nonzero matrix allocate 786 KB of mostly empty chunk -- eight
    times the source, from a conversion asked for to halve it.
    """
    backend = _target_backend(group)
    kw = dataset_create_kwargs(template, target_backend=backend, dst_parent=group)
    kw = _clamp_chunks(kw, max(1, n))
    if "chunks" not in kw and n:
        kw["chunks"] = (min(n, 1 << 16),)
    return create_dataset(group, name, shape=(n,), dtype=dtype, **kw)


def _growable_like(group: Any, name: str, dtype: np.dtype, template: Any) -> Any:
    """Like `_sparse_dataset`, but extensible for a size not yet known.

    The sparse writers of subset and concat once used a bare growable
    dataset here, which dropped the source's compression: their `data` and
    `indices` were written uncompressed whatever the input.
    """
    backend = _target_backend(group)
    kw = dataset_create_kwargs(template, target_backend=backend, dst_parent=group)
    kw.pop("shards", None)
    chunks = kw.pop("chunks", None)
    step = int(chunks[0]) if chunks else 1 << 16
    if is_zarr_group(group):
        return group.create_array(name, shape=(0,), dtype=dtype, chunks=(step,), **kw)
    return group.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(step,), **kw
    )


def _append(ds: Any, values: np.ndarray) -> None:
    """Append a block to a growable 1-D dataset."""
    if values.size == 0:
        return
    start = ds.shape[0]
    ds.resize((start + values.size,))
    ds[start:] = values


def subset_matrix_entry(
    obj: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int,
    entry_label: str,
) -> None:
    fan_out_matrix_entry(
        obj,
        name,
        [(dst_parent, obs_idx, var_idx)],
        chunk_rows=chunk_rows,
        entry_label=entry_label,
    )


def fan_out_matrix_entry(
    obj: Any,
    name: str,
    targets: List[MatrixTarget],
    *,
    chunk_rows: int,
    entry_label: str,
) -> None:
    if is_dataset(obj):
        fan_out_dense_matrix(obj, name, targets, chunk_rows=chunk_rows)
        return

    if is_group(obj):
        enc = _decode_attr(obj.attrs.get("encoding-type", b""))
        if enc in spec.SPARSE_TYPES:
            fan_out_sparse_matrix(obj, name, targets)
            return
        if enc == spec.DATAFRAME:
            # obsm/varm may hold a dataframe; it is row-aligned like obs/var.
            for dst_parent, row_idx, _ in targets:
                subset_axis_group(obj, dst_parent.create_group(name), row_idx)
            return
        raise ValueError(f"Unsupported {entry_label} encoding type: {enc}")

    raise ValueError(f"Unsupported {entry_label} object type")


HANDLED_KEYS = frozenset(
    {"obs", "var", "X", "layers", "obsm", "varm", "obsp", "varp", "uns", "raw"}
)


@dataclass
class Target:
    """One output store of a subset, with the rows it keeps on each axis.

    `var_keep` is the names behind `var_idx`, for matching `raw/var`, which
    has its own var axis.
    """

    root: Any
    obs_idx: Optional[np.ndarray]
    var_idx: Optional[np.ndarray]
    var_keep: Optional[Set[str]] = None


def subset_raw_group(
    src_raw: Any,
    dst: Any,
    obs_idx: Optional[np.ndarray],
    var_keep: Optional[Set[str]],
    *,
    chunk_rows: int,
    console: Console,
) -> None:
    _fan_out_raw(
        src_raw,
        [Target(dst, obs_idx, None, var_keep)],
        chunk_rows=chunk_rows,
        console=console,
    )


def _fan_out_raw(
    src_raw: Any,
    targets: List[Target],
    *,
    chunk_rows: int,
    console: Console,
) -> None:
    """Subset a `raw/` group into every target, reading its matrices once.

    `raw` typically holds more genes than the main object, so its var names are
    matched independently rather than reusing the outer var indices -- using
    those would select the wrong columns entirely.
    """
    raw_dsts: List[Any] = []
    raw_var_idxs: List[Optional[np.ndarray]] = []
    for target in targets:
        raw_dst = target.root.create_group("raw")
        copy_attrs(
            src_raw.attrs, raw_dst.attrs, target_backend=_target_backend(target.root)
        )
        spec.set_encoding(raw_dst, spec.RAW)
        raw_dsts.append(raw_dst)

        raw_var_idx: Optional[np.ndarray] = None
        if target.var_keep is not None and "var" in src_raw:
            raw_var_names, _ = resolve_index(src_raw["var"], "var")
            raw_var_idx, missing = indices_from_name_set(
                raw_var_names, target.var_keep
            )
            console.print(
                f"[green]Selected {len(raw_var_idx)} raw/var "
                f"(of {element_len(raw_var_names)})[/]"
            )
            if missing:
                console.print(
                    f"[yellow]Warning: {len(missing)} var names not found in raw/var[/]"
                )
        raw_var_idxs.append(raw_var_idx)

    if "var" in src_raw:
        for raw_dst, raw_var_idx in zip(raw_dsts, raw_var_idxs):
            subset_axis_group(src_raw["var"], raw_dst.create_group("var"), raw_var_idx)

    if "X" in src_raw:
        fan_out_matrix_entry(
            src_raw["X"],
            "X",
            [
                (raw_dst, target.obs_idx, raw_var_idx)
                for raw_dst, target, raw_var_idx in zip(raw_dsts, targets, raw_var_idxs)
            ],
            chunk_rows=chunk_rows,
            entry_label="raw/X",
        )

    if "varm" in src_raw:
        varm_dsts = [_ensure_group(raw_dst, "varm") for raw_dst in raw_dsts]
        for key in src_raw["varm"].keys():
            fan_out_matrix_entry(
                src_raw["varm"][key],
                key,
                [
                    (varm_dst, raw_var_idx, None)
                    for varm_dst, raw_var_idx in zip(varm_dsts, raw_var_idxs)
                ],
                chunk_rows=chunk_rows,
                entry_label=f"raw/varm:{key}",
            )

    for key in src_raw.keys():
        if key not in ("X", "var", "varm"):
            for raw_dst in raw_dsts:
                copy_tree(src_raw[key], raw_dst, key)


#: Mapping groups whose entries are matrices, and the axes each entry is
#: aligned to, as the (rows, columns) a target keeps.
_MATRIX_MAPPINGS = {
    "layers": ("obs", "var"),
    "obsm": ("obs", None),
    "varm": ("var", None),
    "obsp": ("obs", "obs"),
    "varp": ("var", "var"),
}

_TASK_LABELS = {"layers": "layer"}


def _axis_idx(target: Target, axis: Optional[str]) -> Optional[np.ndarray]:
    if axis is None:
        return None
    return target.obs_idx if axis == "obs" else target.var_idx


def _write_targets(
    src: Any,
    targets: List[Target],
    *,
    chunk_rows: int,
    console: Console,
) -> None:
    """Write every element of `src`, narrowed, into each target's store.

    Matrices are read once and shared out among the targets; the rest -- the
    dataframes, `uns`, anything unrecognised -- is small or row-proportional,
    and is written target by target with the single-output helpers.
    """
    tasks: List[str] = []
    if "obs" in src:
        tasks.append("obs")
    if "var" in src:
        tasks.append("var")
    if "X" in src:
        tasks.append("X")
    for mapping in _MATRIX_MAPPINGS:
        if mapping in src:
            label = _TASK_LABELS.get(mapping, mapping)
            tasks.extend(f"{label}:{k}" for k in src[mapping].keys())
    if "uns" in src:
        tasks.append("uns")
    # anndata writes a placeholder `raw` even when there is none, and
    # on Zarr that placeholder is an array rather than a group.
    if "raw" in src and is_group(src["raw"]):
        tasks.append("raw")
    elif "raw" in src:
        tasks.append("copy:raw")

    passthrough = [k for k in src.keys() if k not in HANDLED_KEYS]
    if passthrough:
        console.print(
            "[yellow]Copying unrecognised top-level "
            f"{'keys' if len(passthrough) > 1 else 'key'} verbatim: "
            f"{', '.join(sorted(passthrough))}[/]"
        )
        tasks.extend(f"copy:{k}" for k in passthrough)

    labels = {v: k for k, v in _TASK_LABELS.items()}

    with Progress(
        SpinnerColumn(finished_text="[green]✓[/]"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        for task in tasks:
            task_id = progress.add_task(f"[cyan]Subsetting {task}...[/]", total=None)
            if task in ("obs", "var"):
                for target in targets:
                    subset_axis_group(
                        src[task],
                        target.root.create_group(task),
                        _axis_idx(target, task),
                    )
            elif task == "X":
                X = src["X"]
                pairs = [(t.root, t.obs_idx, t.var_idx) for t in targets]
                if is_dataset(X):
                    fan_out_dense_matrix(X, "X", pairs, chunk_rows=chunk_rows)
                elif is_group(X):
                    fan_out_sparse_matrix(X, "X", pairs)
                else:
                    for target in targets:
                        copy_tree(X, target.root, "X")
            elif task == "uns":
                for target in targets:
                    copy_tree(src["uns"], target.root, "uns")
            elif task == "raw":
                _fan_out_raw(
                    src["raw"], targets, chunk_rows=chunk_rows, console=console
                )
            elif task.startswith("copy:"):
                key = task.split(":", 1)[1]
                for target in targets:
                    copy_tree(src[key], target.root, key)
            else:
                label, key = task.split(":", 1)
                mapping = labels.get(label, label)
                rows, cols = _MATRIX_MAPPINGS[mapping]
                fan_out_matrix_entry(
                    src[mapping][key],
                    key,
                    [
                        (
                            _ensure_group(t.root, mapping),
                            _axis_idx(t, rows),
                            _axis_idx(t, cols),
                        )
                        for t in targets
                    ],
                    chunk_rows=chunk_rows,
                    entry_label=task,
                )
            progress.update(
                task_id,
                description=f"[green]Subsetting {task}[/]",
                completed=1,
                total=1,
            )

    for target in targets:
        _ensure_optional_anndata_groups(target.root)


def _select_axis(
    src: Any,
    axis: str,
    name_file: Optional[Path],
    query: Optional[str],
    console: Console,
) -> Optional[np.ndarray]:
    """Resolve an axis selection from a name list or a query, or None for all."""
    if name_file is None and query is None:
        return None

    if name_file is not None and query is not None:
        raise ValueError(f"Give either --{axis} or --{axis}-query, not both.")

    if query is not None:
        console.print(f"[cyan]Evaluating {axis} query...[/]")
        indices = select_indices(src, axis, query)
        console.print(f"[green]Selected {len(indices)} {axis}[/]")
        if len(indices) == 0:
            raise ValueError(f"The {axis} query matched no rows.")
        return indices

    keep = _read_name_file(name_file)
    console.print(f"[cyan]Found {len(keep)} {axis} names to keep[/]")
    names_ds, _ = resolve_index(src[axis], axis)
    indices, missing = indices_from_name_set(names_ds, keep)
    if missing:
        console.print(
            f"[yellow]Warning: {len(missing)} {axis} names not found in file[/]"
        )
    console.print(
        f"[green]Selected {len(indices)} {axis} (of {element_len(names_ds)})[/]"
    )
    return indices


def subset_h5ad(
    file: Path,
    output: Optional[Path],
    obs_file: Optional[Path],
    var_file: Optional[Path],
    *,
    chunk_rows: int = 1024,
    console: Console,
    inplace: bool = False,
    obs_query: Optional[str] = None,
    var_query: Optional[str] = None,
    obs_indices: Optional[np.ndarray] = None,
    var_indices: Optional[np.ndarray] = None,
    zarr_format: Optional[int] = None,
    quiet: bool = False,
) -> None:
    """Write a copy of `file` narrowed to the selected obs and/or var.

    Selection comes from a name file, a query, or indices computed by a caller
    such as `split`. Exactly one source per axis.
    """
    has_selection = any(
        x is not None
        for x in (obs_file, var_file, obs_query, var_query, obs_indices, var_indices)
    )
    if not has_selection:
        raise ValueError("At least one of --obs or --var must be provided.")

    if not inplace and output is None:
        raise ValueError("Output file is required unless --inplace is specified.")

    if inplace:
        src_backend = detect_backend(file)
        if src_backend == "zarr":
            base_name = file.stem if file.suffix else file.name
            tmp_path = file.with_name(f"{base_name}.subset-tmp.zarr")
        else:
            tmp_path = file.with_name(f"{file.name}.subset-tmp")
        if tmp_path.exists():
            raise FileExistsError(f"Temporary path already exists: {tmp_path}")
        dst_path = tmp_path
    else:
        dst_path = output

    if zarr_format is None and detect_backend(file) == "zarr":
        with open_store(file, "r") as probe:
            zarr_format = probe.zarr_format

    # Rich allows only one live display per console, and the progress bar
    # below is a second one. The spinner is therefore stopped explicitly once
    # the selection is known, rather than left running around it.
    status = console.status("[magenta]Opening files...[/]")
    status.start()
    try:
        with open_store(file, "r") as src_store, open_store(
            dst_path, "w", zarr_format=zarr_format
        ) as dst_store:
            src = src_store.root
            dst = dst_store.root

            obs_idx = (
                obs_indices
                if obs_indices is not None
                else _select_axis(src, "obs", obs_file, obs_query, console)
            )
            var_idx = (
                var_indices
                if var_indices is not None
                else _select_axis(src, "var", var_file, var_query, console)
            )

            # raw/ has its own var axis, so it is matched by name rather than
            # by reusing these indices.
            var_keep: Optional[Set[str]] = None
            if var_idx is not None and "var" in src:
                var_keep = set(read_names(src, "var", var_idx))

            status.stop()

            _write_targets(
                src,
                [Target(dst, obs_idx, var_idx, var_keep)],
                chunk_rows=chunk_rows,
                console=console,
            )
    finally:
        status.stop()

    if inplace:
        if file.exists():
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()
        if dst_path.is_dir():
            shutil.move(str(dst_path), str(file))
        else:
            dst_path.replace(file)


def _open_output_limit() -> int:
    """How many output stores `split_h5ad` may hold open at once.

    Each open HDF5 file costs a descriptor, and the soft limit is 256 on
    macOS and 1024 on most Linux systems by default. A quarter of it leaves
    room for everything else the process has open.
    """
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):  # pragma: no cover - Windows
        return 64
    if soft == resource.RLIM_INFINITY:
        return 1024
    return max(8, min(1024, soft // 4))


#: A split into more groups than this makes one pass per batch of this many,
#: rather than running out of file descriptors.
MAX_OPEN_OUTPUTS = _open_output_limit()


def split_h5ad(
    file: Path,
    outputs: List[Tuple[Path, Optional[np.ndarray], Optional[np.ndarray]]],
    *,
    chunk_rows: int = 1024,
    console: Console,
    zarr_format: Optional[int] = None,
) -> None:
    """Write one subset of `file` per ``(path, obs_idx, var_idx)`` in `outputs`.

    Equivalent to calling :func:`subset_h5ad` once per output, but every
    matrix is read once for all of them (per batch of `MAX_OPEN_OUTPUTS`)
    rather than once per output.
    """
    if zarr_format is None and detect_backend(file) == "zarr":
        with open_store(file, "r") as probe:
            zarr_format = probe.zarr_format

    for start in range(0, len(outputs), MAX_OPEN_OUTPUTS):
        batch = outputs[start : start + MAX_OPEN_OUTPUTS]
        with ExitStack() as stack:
            src = stack.enter_context(open_store(file, "r")).root
            targets = []
            for path, obs_idx, var_idx in batch:
                dst = stack.enter_context(
                    open_store(path, "w", zarr_format=zarr_format)
                ).root
                var_keep = (
                    set(read_names(src, "var", var_idx))
                    if var_idx is not None and "var" in src
                    else None
                )
                targets.append(Target(dst, obs_idx, var_idx, var_keep))
            _write_targets(src, targets, chunk_rows=chunk_rows, console=console)

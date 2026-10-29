"""Subset operations for .h5ad and .zarr stores."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
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
from adata.elements.strings import is_string_dtype
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
    fit_chunks,
    is_dataset,
    is_group,
    is_legacy_categorical,
    is_legacy_dataframe,
    is_zarr_group,
    is_zarr_array,
    open_store,
    with_compression,
    write_legacy_categorical,
    zarr_format_of,
)
from adata.util.path import output_target
from adata.util.staging import check_free_space, disk_size, staged_outputs


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


def aligned_rows(obj: Any, rows: np.ndarray) -> np.ndarray:
    """`obj[rows]`, for rows in any order: one block of an output's rows.

    Rows in order are read as the one slab they span; rows out of order, or
    scattered, by index, which both backends take only sorted, then put back
    in the output's order. The caller bounds memory by the block it asks for.
    """
    order = np.argsort(rows, kind="stable")
    wanted = rows[order]
    lo, hi = int(wanted[0]), int(wanted[-1]) + 1
    if hi - lo <= 2 * len(rows):
        block = np.asarray(obj[lo:hi, ...])[wanted - lo]
    elif is_zarr_array(obj):
        block = np.asarray(obj.oindex[(wanted,) + (slice(None),) * (obj.ndim - 1)])
    else:
        block = np.asarray(obj[wanted, ...])
    out = np.empty_like(block)
    out[order] = block
    # h5py writes text only from str objects, not numpy's `<U`.
    return decode_str_array(out).astype(object) if is_string_dtype(out.dtype) else out


def _copy_rows(
    src_ds: Any,
    dst_parent: Any,
    name: str,
    indices: Optional[np.ndarray],
) -> Any:
    """Write the selected rows of `src_ds` into `dst_parent` as `name`."""
    if indices is None:
        return copy_tree(src_ds, dst_parent, name)
    return _write_like(src_ds, dst_parent, name, _take_rows(src_ds, indices))


def _write_like(src_ds: Any, dst_parent: Any, name: str, values: Any) -> Any:
    """`values` as `dst_parent[name]`, laid out and attributed as `src_ds`."""
    target_backend = _target_backend(dst_parent)
    kw = dataset_create_kwargs(
        src_ds, target_backend=target_backend, dst_parent=dst_parent
    )
    kw = fit_chunks(kw, np.shape(values), backend=target_backend)
    ds = create_dataset(dst_parent, name, data=values, **kw)
    copy_attrs(src_ds.attrs, ds.attrs, target_backend=target_backend)
    return ds


#: Per categorical column written: the categories kept (positions in the
#: source's; None for all of them), and how many the source had.
Pruned = Dict[str, Tuple[Optional[np.ndarray], int]]


def prune_categories(codes: np.ndarray, n_categories: int) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """The categories `codes` use, and the codes renumbered onto them, as
    anndata's subsetting does (`remove_unused_categories`); None and the
    codes themselves when every category is used."""
    used = np.zeros(n_categories, dtype=bool)
    present = codes[codes >= 0]
    used[present] = True
    if used.all():
        return None, codes
    kept = np.flatnonzero(used)
    remap = np.full(n_categories, -1, dtype=np.int64)
    remap[kept] = np.arange(kept.size)
    renumbered = np.where(codes >= 0, remap[np.maximum(codes, 0)], codes)
    return kept, renumbered.astype(codes.dtype, copy=False)


def prune_colors(uns: Any, pruned: Pruned) -> None:
    """`uns["<col>_colors"]` for the categories kept, by anndata's rule
    (`_remove_unused_categories`), applied to every categorical written:

    * a single colour (0-d) becomes a list of one;
    * a list of a length other than the source's categories is dropped,
      since it matches none of them;
    * otherwise the colours of the categories kept, along the first axis
      (RGBA rows too).

    Called once per axis, obs then var, on the output's `uns` as the
    previous call left it, as anndata applies them.
    """
    for column, (kept, n) in pruned.items():
        key = f"{column}_colors"
        if key not in uns or not is_dataset(uns[key]):
            continue
        current = uns[key]
        if current.ndim == 0:
            values = np.asarray(current[()])[None]
        elif current.shape[0] != n:
            del uns[key]
            continue
        elif kept is None:
            continue
        else:
            values = np.asarray(current[...])[kept]
        attrs = dict(current.attrs)
        backend = _target_backend(uns)
        kw = dataset_create_kwargs(current, target_backend=backend, dst_parent=uns)
        del uns[key]
        ds = create_dataset(uns, key, data=values, **fit_chunks(kw, values.shape, backend=backend))
        copy_attrs(attrs, ds.attrs, target_backend=backend)


def subset_axis_group(
    src: Any,
    dst: Any,
    indices: Optional[np.ndarray],
    *,
    prune: bool = False,
) -> Pruned:
    """Copy a dataframe group, taking only `indices` along its rows.

    Every column layout the spec allows has to be narrowed here, not just plain
    datasets: `categorical` keeps its categories and subsets only `codes`,
    while the masked layouts (`nullable-*`, which anndata >= 0.11 uses for the
    index and every string column) subset both `values` and `mask`. Copying a
    masked column whole would leave it longer than the rest of the frame.

    With `prune`, as for obs and var, a categorical keeps only the
    categories its rows use, as anndata's subsetting keeps them (on an axis
    kept whole too); returns which, per column, for `prune_colors`.
    """
    pruned: Pruned = {}
    target_backend = _target_backend(dst)
    copy_attrs(src.attrs, dst.attrs, target_backend=target_backend)
    # anndata 0.7's categoricals point into `__categories` by object
    # reference, which cannot leave the file: they are written as current
    # `categorical` groups, and the frame as the current version.
    legacy = is_legacy_dataframe(src)
    if legacy:
        dst.attrs["encoding-version"] = "0.2.0"

    for key in src.keys():
        obj = src[key]

        if legacy and key == "__categories":
            continue
        if legacy and is_legacy_categorical(obj):
            codes = kept = None
            if indices is not None or prune:
                codes = np.asarray(obj[...] if indices is None else _take_rows(obj, indices))
            if prune:
                from adata.elements.read import categories_dataset

                n = element_len(categories_dataset(obj, src))
                kept, codes = prune_categories(codes, n)
                pruned[key] = (kept, n)
            write_legacy_categorical(src, key, obj, dst, codes, kept=kept)
            continue

        if is_dataset(obj):
            _copy_rows(obj, dst, key, indices)
            continue

        if not is_group(obj):
            continue

        enc = _decode_attr(obj.attrs.get("encoding-type", b""))

        if enc == spec.CATEGORICAL or (enc is None and "codes" in obj):
            gdst = dst.create_group(key)
            copy_attrs(obj.attrs, gdst.attrs, target_backend=target_backend)
            if not prune:
                copy_tree(obj["categories"], gdst, "categories")
                _copy_rows(obj["codes"], gdst, "codes", indices)
                continue
            # Even an axis kept whole: anndata drops a category no row uses
            # whenever it subsets, on either axis.
            n = element_len(obj["categories"])
            codes = np.asarray(obj["codes"][...] if indices is None else _take_rows(obj["codes"], indices))
            kept, codes = prune_categories(codes, n)
            pruned[key] = (kept, n)
            if kept is None:
                copy_tree(obj["categories"], gdst, "categories")
            else:
                _write_like(obj["categories"], gdst, "categories", np.asarray(obj["categories"][...])[kept])
            # The codes are in memory already, read to find the kept ones.
            _write_like(obj["codes"], gdst, "codes", codes)
            continue

        if enc in spec.MASKED_TYPES or ("values" in obj and "mask" in obj):
            gdst = dst.create_group(key)
            copy_attrs(obj.attrs, gdst.attrs, target_backend=target_backend)
            _copy_rows(obj["values"], gdst, "values", indices)
            _copy_rows(obj["mask"], gdst, "mask", indices)
            continue

        # Not row-aligned (e.g. __categories) -- copy verbatim.
        copy_tree(obj, dst, key)
    return pruned


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
        kw = fit_chunks(kw, (out_obs, out_var), backend=target_backend)
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
    for target in targets:
        # A fourth element names this target's matrix, where it differs
        # from `name`: one read can then fill X and a layer alike.
        dst_parent, obs_idx, var_idx = target[:3]
        out_name = target[3] if len(target) > 3 else name
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
                out_name,
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
    group: Any,
    name: str,
    dtype: np.dtype,
    n: int,
    template: Any,
    *,
    compression: Optional[str] = None,
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
    kw = fit_chunks(kw, (n,), backend=backend)
    if compression is not None:
        kw = with_compression(
            kw, compression, shape=(n,), dtype=dtype, backend=backend,
            zarr_format=zarr_format_of(group),
        )
    if "chunks" not in kw and n:
        kw["chunks"] = (min(n, 1 << 16),)
    return create_dataset(group, name, shape=(n,), dtype=dtype, **kw)


def _growable_like(
    group: Any,
    name: str,
    dtype: np.dtype,
    template: Any,
    *,
    compression: Optional[str] = None,
) -> Any:
    """Like `_sparse_dataset`, but extensible for a size not yet known.

    The sparse writers of subset and concat once used a bare growable
    dataset here, which dropped the source's compression: their `data` and
    `indices` were written uncompressed whatever the input.
    """
    backend = _target_backend(group)
    kw = dataset_create_kwargs(template, target_backend=backend, dst_parent=group)
    if compression is not None:
        kw = with_compression(
            kw, compression, shape=(0,), dtype=dtype, backend=backend,
            zarr_format=zarr_format_of(group),
        )
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
    skip: Set[str] = frozenset(),
    verb: str = "Subsetting",
    prune_whole_axes: bool = True,
) -> None:
    """Write every element of `src`, narrowed, into each target's store.

    Matrices are read once and shared out among the targets; the rest -- the
    dataframes, `uns`, anything unrecognised -- is small or row-proportional,
    and is written target by target with the single-output helpers.
    `skip` names top-level elements the caller writes itself (a MuData
    export keeps obs, obsm, obsp and uns once, for all modalities).

    obs and var drop unused categories as anndata's subsetting does, on an
    axis kept whole too; `prune_whole_axes=False` leaves such an axis as
    stored, for a conversion that subsets nothing there.
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
    pruned: List[Dict[str, Pruned]] = [{} for _ in targets]
    tasks = [t for t in tasks if labels.get(t.split(":", 1)[0], t.split(":", 1)[0]) not in skip
             and not (t.startswith("copy:") and t.split(":", 1)[1] in skip)]

    with Progress(
        SpinnerColumn(finished_text="[green]✓[/]"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        for task in tasks:
            task_id = progress.add_task(f"[cyan]{verb} {task}...[/]", total=None)
            if task in ("obs", "var"):
                for target, narrowed in zip(targets, pruned):
                    indices = _axis_idx(target, task)
                    narrowed[task] = subset_axis_group(
                        src[task],
                        target.root.create_group(task),
                        indices,
                        prune=prune_whole_axes or indices is not None,
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
                for target, narrowed in zip(targets, pruned):
                    copy_tree(src["uns"], target.root, "uns")
                    for axis in ("obs", "var"):  # in anndata's order
                        prune_colors(target.root["uns"], narrowed.get(axis, {}))
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
                description=f"[green]{verb} {task}[/]",
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
    space_check: bool = True,
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
    if chunk_rows < 1:
        raise ValueError("--chunk must be at least 1.")

    # Refuses -o naming the input (opening it "w" would empty a Zarr source
    # before a single element had been read from it), or a folder.
    dst_path = output_target(file, output, inplace)

    if zarr_format is None and detect_backend(file) == "zarr":
        with open_store(file, "r") as probe:
            zarr_format = probe.zarr_format

    # Rich allows only one live display per console, and the progress bar
    # below is a second one. The spinner is therefore stopped explicitly once
    # the selection is known, rather than left running around it.
    status = console.status("[magenta]Opening files...[/]")
    status.start()
    try:
        # The output is written beside its destination and renamed into place
        # only once complete, so a failure leaves any existing output (or,
        # with --inplace, the source) as it was. The source is opened inside
        # the staging block, so it is closed before it is renamed aside.
        with staged_outputs([dst_path]) as (temp,), open_store(
            file, "r"
        ) as src_store:
            src = src_store.root

            # The selection is resolved before the output exists: a query
            # that fails must not leave an empty store behind.
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
            for axis, idx, name_file in (
                ("obs", obs_idx, obs_file),
                ("var", var_idx, var_file),
            ):
                if idx is not None and len(idx) == 0:
                    # An empty store is never what was meant -- typically
                    # barcodes without the "-1" the index carries -- and
                    # with --inplace it replaced the user's data.
                    source = f"the names in '{name_file}'" if name_file else "the selection"
                    raise ValueError(
                        f"No {axis} selected: none of {source} is in the "
                        f"store's {axis} index. Nothing was written."
                    )

            check_free_space(
                dst_path,
                lambda: disk_size(file)
                * _kept(src, "obs", obs_idx)
                * _kept(src, "var", var_idx),
                check=space_check,
            )

            # raw/ has its own var axis, so it is matched by name rather than
            # by reusing these indices.
            var_keep: Optional[Set[str]] = None
            if var_idx is not None and "var" in src:
                var_keep = set(read_names(src, "var", var_idx))

            status.stop()

            with open_store(temp, "w", zarr_format=zarr_format) as dst_store:
                _write_targets(
                    src,
                    [Target(dst_store.root, obs_idx, var_idx, var_keep)],
                    chunk_rows=chunk_rows,
                    console=console,
                )
    finally:
        status.stop()


def _kept(src: Any, axis: str, indices: Optional[np.ndarray]) -> float:
    """The fraction of `axis` a selection keeps, for sizing the output."""
    if indices is None or axis not in src:
        return 1.0
    try:
        total = element_len(resolve_index(src[axis], axis)[0])
    except Exception:  # pragma: no cover - an axis without an index
        return 1.0
    return min(1.0, len(indices) / total) if total else 1.0


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

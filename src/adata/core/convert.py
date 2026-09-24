"""Rewriting a matrix's dtype, layout or density, on disk.

Three conversions, one command. Counts stored as float64 cost twice the disk
and twice the read for no information (issue #13); a CSR store handed to a
tool that wants CSC has to be transposed somewhere; and `concat` refuses
inputs whose encodings disagree, which until now left nothing to do about it.

Everything here streams. The matrix these conversions matter for is the one
too large to load, so a converter that loads it would only work on the files
that did not need converting.

Two safety rules, both of which fail before anything is written -- the same
principle `check_matrix_encodings` states for concat, because a half-written
store is worse than a refusal:

* a cast that does not round-trip is refused unless forced, so `int32`
  indices that would overflow, or a float downcast that would lose real
  precision, are reported rather than silently written;
* densifying is refused when it would inflate the store beyond
  `MAX_GROWTH_FACTOR`, because a 5% dense matrix becomes ten times its size
  and that should not be a surprise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.elements import spec
from adata.elements.write import set_shape_attr
from adata.storage import (
    copy_attrs,
    create_dataset,
    dataset_create_kwargs,
    is_dataset,
    is_group,
    is_zarr_group,
)

#: Layouts the user can ask for. "sparse" means "whichever sparse encoding
#: keeps the major axis it already has", so densify/sparsify round-trips.
LAYOUTS = ("csr", "csc", "dense", "sparse")

#: What `--dtype` accepts. An allowlist rather than `np.dtype(text)`: that
#: would cheerfully accept "S10" or "datetime64[ns]" and produce a store no
#: reader expects from a matrix.
DTYPES = (
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
)

#: Indices and indptr must be integers, and signed: anndata and scipy both
#: expect a signed index type.
INDEX_DTYPES = ("int32", "int64")

#: How much larger a densified store may get before it needs --force.
MAX_GROWTH_FACTOR = 4.0

#: Values per read while streaming `data`/`indices`.
DEFAULT_CHUNK = 1 << 20

#: Smallest bucket worth making during a streaming transpose. Below this
#: the per-bucket overhead dominates and the extra passes buy nothing.
MIN_BUCKET_ENTRIES = 1 << 12


def parse_dtype(text: str, *, allowed: Tuple[str, ...] = DTYPES) -> np.dtype:
    """Resolve a user-supplied dtype name, or say what is accepted."""
    name = text.strip().lower()
    if name not in allowed:
        raise ValueError(
            f"Unknown dtype {text!r}. Choose from: {', '.join(allowed)}."
        )
    return np.dtype(name)


# ---------------------------------------------------------------------------
# safety


@dataclass
class CastReport:
    """What a cast would do to the values, measured rather than assumed."""

    total: int = 0
    changed: int = 0
    worst_absolute: float = 0.0
    overflowed: bool = False

    @property
    def lossless(self) -> bool:
        return self.changed == 0 and not self.overflowed

    def describe(self, source: Any, target: Any) -> str:
        if self.lossless:
            return f"{source} -> {target}: every value round-trips"
        share = self.changed / self.total if self.total else 0.0
        detail = (
            "values exceed its range"
            if self.overflowed
            else f"largest change {self.worst_absolute:g}"
        )
        return (
            f"{source} -> {target} is lossy: {self.changed:,} of "
            f"{self.total:,} values ({share:.2%}) do not round-trip, {detail}"
        )


def check_cast(dataset: Any, target: np.dtype, *, chunk: int = DEFAULT_CHUNK) -> CastReport:
    """Would casting `dataset` to `target` lose anything?

    Streams the values, casts each block and casts it back. An exact
    round-trip is the only honest test: whether float64 counts survive
    float32 depends on the counts, not on the dtypes, and that is precisely
    the question issue #13 is asking.
    """
    report = CastReport()
    source = np.dtype(getattr(dataset, "dtype", "float64"))
    n = int(dataset.shape[0]) if getattr(dataset, "shape", None) else 0

    info = np.finfo(target) if target.kind == "f" else (
        np.iinfo(target) if target.kind in "iu" else None
    )

    for start in range(0, n, chunk):
        block = np.asarray(dataset[start : min(start + chunk, n)])
        report.total += block.size
        if block.size == 0:
            continue

        with np.errstate(invalid="ignore", over="ignore"):
            cast = block.astype(target)
            back = cast.astype(source)

        if info is not None:
            finite = block[np.isfinite(block)] if source.kind == "f" else block
            if finite.size and (
                float(finite.max()) > float(info.max)
                or float(finite.min()) < float(info.min)
            ):
                report.overflowed = True

        # NaN never equals itself, so compare those separately rather than
        # counting every missing value as a loss.
        differs = back != block
        if source.kind == "f":
            both_nan = np.isnan(block) & np.isnan(back)
            differs &= ~both_nan
        count = int(differs.sum())
        if count:
            report.changed += count
            delta = np.abs(
                block[differs].astype("float64") - back[differs].astype("float64")
            )
            finite_delta = delta[np.isfinite(delta)]
            if finite_delta.size:
                report.worst_absolute = max(
                    report.worst_absolute, float(finite_delta.max())
                )

    return report


def check_index_dtype(
    group: Any,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype,
    shape: Tuple[int, int],
) -> None:
    """Refuse index dtypes that cannot address this matrix.

    Cheaper than `check_cast`: the largest value each array must hold is
    bounded by the dimensions and the nonzero count, so no pass over the
    data is needed. `indices` holds coordinates, bounded by the larger
    dimension; `indptr` holds offsets, bounded by nnz. They are checked
    separately because they can legitimately need different widths.
    """
    nnz = int(group["indices"].shape[0])
    for name, dtype, largest, what in (
        ("indices", index_dtype, max(int(shape[0]), int(shape[1])), "coordinates"),
        ("indptr", pointer_dtype, nnz, "offsets"),
    ):
        limit = int(np.iinfo(dtype).max)
        if largest > limit:
            raise ValueError(
                f"{dtype} cannot hold this matrix's {what}: {name} must reach "
                f"{largest:,} for a {shape[0]:,} x {shape[1]:,} matrix with "
                f"{nnz:,} nonzeros, and {dtype} tops out at {limit:,}. "
                "Use int64."
            )


def _stored_bytes(obj: Any) -> int:
    """Bytes this element occupies, as best the backend will say."""
    total = 0
    targets = [obj] if is_dataset(obj) else [
        obj[k] for k in ("data", "indices", "indptr") if k in obj
    ]
    for item in targets:
        try:
            total += int(item.nbytes)
        except Exception:  # pragma: no cover - backend without nbytes
            shape = getattr(item, "shape", ()) or ()
            size = int(np.prod(shape)) if shape else 0
            total += size * int(getattr(item.dtype, "itemsize", 8) or 8)
    return total


def check_growth(
    current: int, projected: int, *, force: bool, what: str
) -> None:
    """Refuse a conversion that inflates the store, unless asked twice."""
    if force or current <= 0 or projected <= current * MAX_GROWTH_FACTOR:
        return
    raise ValueError(
        f"{what} would grow from {current / 1e6:,.0f} MB to "
        f"{projected / 1e6:,.0f} MB ({projected / current:.1f}x). That is "
        f"above the {MAX_GROWTH_FACTOR:g}x limit; pass --force if it is what "
        "you want."
    )


# ---------------------------------------------------------------------------
# reading a source


@dataclass
class Matrix:
    """A matrix on disk, described enough to convert it."""

    obj: Any
    kind: str  # "csr_matrix" | "csc_matrix" | "dense"
    shape: Tuple[int, int]
    dtype: np.dtype

    @property
    def sparse(self) -> bool:
        return self.kind in spec.SPARSE_TYPES

    @property
    def nnz(self) -> int:
        return int(self.obj["indices"].shape[0]) if self.sparse else 0


def describe(obj: Any) -> Matrix:
    """Classify a matrix element, or say why it is not one."""
    enc = spec.encoding_type(obj)
    if is_group(obj) and enc in spec.SPARSE_TYPES:
        shape = obj.attrs.get("shape", None)
        if shape is None:
            raise ValueError("Sparse matrix group is missing its 'shape' attribute.")
        return Matrix(obj, enc, (int(shape[0]), int(shape[1])), obj["data"].dtype)

    if is_dataset(obj):
        if getattr(obj, "ndim", 0) != 2:
            raise ValueError(
                f"Only 2-D matrices can be converted; this one has "
                f"{getattr(obj, 'ndim', '?')} dimension(s)."
            )
        return Matrix(obj, "dense", (int(obj.shape[0]), int(obj.shape[1])), obj.dtype)

    raise ValueError(
        f"Not a matrix: encoding {enc!r}. `convert` handles X, layers, raw/X "
        "and any 2-D dense array."
    )


def resolve_layout(source: Matrix, requested: Optional[str]) -> str:
    """The concrete target encoding for a possibly-vague request."""
    if requested is None:
        return source.kind
    if requested == "dense":
        return "dense"
    if requested == "sparse":
        # Keep the major axis it already had, so sparse -> dense -> sparse is
        # the identity rather than a silent transpose.
        return source.kind if source.sparse else spec.CSR_MATRIX
    return spec.CSR_MATRIX if requested == "csr" else spec.CSC_MATRIX


# ---------------------------------------------------------------------------
# writers


def _new_sparse_group(
    src: Matrix, dst_parent: Any, name: str, enc: str, shape: Tuple[int, int]
) -> Any:
    group = dst_parent.create_group(name)
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    copy_attrs(src.obj.attrs, group.attrs, target_backend=backend)
    spec.set_encoding(group, enc)
    set_shape_attr(group, shape)
    return group


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
    from adata.core.subset import _clamp_chunks

    backend = "zarr" if is_zarr_group(group) else "hdf5"
    kw = dataset_create_kwargs(template, target_backend=backend, dst_parent=group)
    kw = _clamp_chunks(kw, max(1, n))
    if "chunks" not in kw and n:
        kw["chunks"] = (min(n, 1 << 16),)
    return create_dataset(group, name, shape=(n,), dtype=dtype, **kw)


def _growable_like(group: Any, name: str, dtype: np.dtype, template: Any) -> Any:
    """Like `_sparse_dataset`, but extensible for a size not yet known."""
    backend = "zarr" if is_zarr_group(group) else "hdf5"
    kw = dataset_create_kwargs(template, target_backend=backend, dst_parent=group)
    kw.pop("shards", None)
    chunks = kw.pop("chunks", None)
    step = int(chunks[0]) if chunks else 1 << 16
    if is_zarr_group(group):
        return group.create_array(name, shape=(0,), dtype=dtype, chunks=(step,), **kw)
    return group.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(step,), **kw
    )


def _write_sparse_arrays(
    group: Any,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype,
    source: Any,
) -> None:
    for name, values, dtype, template in (
        ("data", data, data_dtype, source["data"]),
        ("indices", indices, index_dtype, source["indices"]),
        ("indptr", indptr, pointer_dtype, source["indptr"]),
    ):
        cast = values.astype(dtype, copy=False)
        dataset = _sparse_dataset(group, name, dtype, cast.size, template)
        if cast.size:
            dataset[:] = cast


def cast_sparse(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype,
    chunk: int = DEFAULT_CHUNK,
) -> None:
    """Rewrite a sparse matrix with new dtypes, keeping its layout.

    The cheap case, and the one issue #13 asks for: the structure is
    untouched, so this is a straight streamed copy of `data` and `indices`
    into differently typed datasets. Sized up front, because nnz is known.
    """
    group = _new_sparse_group(src, dst_parent, name, src.kind, src.shape)
    source_data, source_indices = src.obj["data"], src.obj["indices"]
    nnz = int(source_data.shape[0])

    out_data = _sparse_dataset(group, "data", data_dtype, nnz, source_data)
    out_indices = _sparse_dataset(group, "indices", index_dtype, nnz, source_indices)

    for start in range(0, nnz, chunk):
        end = min(start + chunk, nnz)
        out_data[start:end] = np.asarray(source_data[start:end]).astype(
            data_dtype, copy=False
        )
        out_indices[start:end] = np.asarray(source_indices[start:end]).astype(
            index_dtype, copy=False
        )

    indptr = np.asarray(src.obj["indptr"][...]).astype(pointer_dtype, copy=False)
    out_indptr = _sparse_dataset(
        group, "indptr", pointer_dtype, indptr.size, src.obj["indptr"]
    )
    out_indptr[:] = indptr


def transpose_sparse_in_memory(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype = np.dtype("int64"),
) -> None:
    """Swap CSR<->CSC by loading the matrix and sorting it once.

    Faster than streaming whenever the matrix fits, and the whole matrix is
    what it needs -- so it is opt-in, never the default.
    """
    target = (
        spec.CSC_MATRIX if src.kind == spec.CSR_MATRIX else spec.CSR_MATRIX
    )
    n_major_in = src.shape[0] if src.kind == spec.CSR_MATRIX else src.shape[1]
    n_major_out = src.shape[1] if src.kind == spec.CSR_MATRIX else src.shape[0]

    indptr = np.asarray(src.obj["indptr"][...], dtype=np.int64)
    minor = np.asarray(src.obj["indices"][...], dtype=np.int64)
    values = np.asarray(src.obj["data"][...])
    major = np.repeat(np.arange(n_major_in, dtype=np.int64), np.diff(indptr))

    # Sort by the new major axis, then the new minor, which is what both
    # encodings require of `indices` within a row.
    order = np.lexsort((major, minor))
    out_indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(minor, minlength=n_major_out)))
    )

    group = _new_sparse_group(src, dst_parent, name, target, src.shape)
    _write_sparse_arrays(
        group,
        values[order],
        major[order],
        out_indptr,
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        pointer_dtype=pointer_dtype,
        source=src.obj,
    )


def transpose_sparse_streaming(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype = np.dtype("int64"),
    chunk: int = DEFAULT_CHUNK,
    bucket_entries: Optional[int] = None,
    console: Optional[Console] = None,
) -> None:
    """Swap CSR<->CSC without loading the matrix.

    A transpose cannot be done in one pass: the first entry of the output
    may come from the last row of the input. Three passes instead, with
    memory set by `bucket_entries` rather than by nnz:

    1. count nonzeros per output major, by streaming `indices` alone. That
       gives the output `indptr` by cumulative sum.
    2. stream the input again, splitting each block's entries into buckets by
       which slice of the output they land in, and append each bucket to
       scratch datasets.
    3. read one bucket at a time, sort it, and append to the output in order.

    Costs about two extra passes over nnz, which is the price of not holding
    the matrix. `--in-memory` is there for when you would rather pay in RAM.
    """
    from adata.core.subset import _append, _growable

    target = (
        spec.CSC_MATRIX if src.kind == spec.CSR_MATRIX else spec.CSR_MATRIX
    )
    n_major_in = src.shape[0] if src.kind == spec.CSR_MATRIX else src.shape[1]
    n_major_out = src.shape[1] if src.kind == spec.CSR_MATRIX else src.shape[0]

    indptr = np.asarray(src.obj["indptr"][...], dtype=np.int64)
    source_indices, source_data = src.obj["indices"], src.obj["data"]
    nnz = int(source_indices.shape[0])

    # Pass 1: how many entries land in each output major.
    counts = np.zeros(n_major_out, dtype=np.int64)
    for start in range(0, nnz, chunk):
        block = np.asarray(source_indices[start : min(start + chunk, nnz)], dtype=np.int64)
        counts += np.bincount(block, minlength=n_major_out)
    out_indptr = np.concatenate(([0], np.cumsum(counts)))

    # One bucket per slice of the output major axis, holding about as many
    # nonzeros as one read. Tying this to `chunk` rather than a constant is
    # what makes the peak follow the setting the caller chose: with a fixed
    # bucket size, anything below it went into a single bucket and the
    # "streaming" path quietly held the whole matrix.
    # An explicit request is honoured as given; the floor applies only to
    # the value derived from `chunk`, where it stops a tiny chunk producing
    # thousands of buckets whose overhead outweighs the saving.
    per_bucket = (
        max(1, int(bucket_entries))
        if bucket_entries
        else max(MIN_BUCKET_ENTRIES, int(chunk))
    )
    n_buckets = max(1, int(np.ceil(nnz / per_bucket))) if nnz else 1
    n_buckets = min(n_buckets, n_major_out) or 1

    # Split by nonzero count, not by coordinate. Equal-width bounds put
    # nearly everything in one bucket whenever the matrix is skewed -- and
    # single-cell matrices are: a handful of genes carry most of the
    # counts. Measured on one such matrix, the largest of three equal-width
    # buckets held 88% of the entries, so the bucket, not the chunk, set
    # the peak. `counts` is already to hand from pass 1.
    cumulative = out_indptr
    targets = np.linspace(0, nnz, n_buckets + 1)[1:-1]
    bounds = np.concatenate((
        [0],
        np.searchsorted(cumulative, targets, side="left").astype(np.int64),
        [n_major_out],
    ))
    bounds = np.unique(bounds)
    n_buckets = len(bounds) - 1
    if console is not None and n_buckets > 1:
        console.print(
            f"[dim]Transposing {nnz:,} nonzeros through {n_buckets} buckets[/]"
        )

    # nnz is invariant under a transpose, so the output can be sized now and
    # written by slice rather than grown block by block.
    group = _new_sparse_group(src, dst_parent, name, target, src.shape)
    out_data = _sparse_dataset(group, "data", data_dtype, nnz, source_data)
    out_indices = _sparse_dataset(group, "indices", index_dtype, nnz, source_indices)
    written = 0

    scratch_name = f"__{name}_transpose_scratch__"
    scratch = dst_parent.create_group(scratch_name)
    try:
        buckets = [
            (
                _growable(scratch, f"major{b}", np.int64),
                _growable(scratch, f"minor{b}", np.int64),
                _growable(scratch, f"value{b}", src.dtype),
            )
            for b in range(n_buckets)
        ]

        # Pass 2: scatter the input into buckets, one input block at a time.
        # Step over the major axis in blocks holding about `chunk` nonzeros,
        # so the read size is set by the data rather than by how many rows
        # happen to be empty.
        average = max(1, nnz // max(1, n_major_in))
        major_step = max(1, chunk // average)
        for lo in range(0, n_major_in, major_step):
            hi = min(lo + major_step, n_major_in)
            start, end = int(indptr[lo]), int(indptr[hi])
            if end <= start:
                continue
            minor = np.asarray(source_indices[start:end], dtype=np.int64)
            values = np.asarray(source_data[start:end])
            major = np.repeat(
                np.arange(lo, hi, dtype=np.int64), np.diff(indptr[lo : hi + 1])
            )
            which = np.clip(np.searchsorted(bounds, minor, side="right") - 1, 0, n_buckets - 1)
            for b in range(n_buckets):
                pick = which == b
                if not pick.any():
                    continue
                _append(buckets[b][0], minor[pick])
                _append(buckets[b][1], major[pick])
                _append(buckets[b][2], values[pick])

        # Pass 3: each bucket in turn, sorted into output order.
        for b in range(n_buckets):
            new_major, new_minor, value = buckets[b]
            if new_major.shape[0] == 0:
                continue
            majors = np.asarray(new_major[...], dtype=np.int64)
            minors = np.asarray(new_minor[...], dtype=np.int64)
            values = np.asarray(value[...])
            order = np.lexsort((minors, majors))
            count = order.size
            out_indices[written : written + count] = minors[order].astype(
                index_dtype, copy=False
            )
            out_data[written : written + count] = values[order].astype(
                data_dtype, copy=False
            )
            written += count
    finally:
        del dst_parent[scratch_name]

    cast_indptr = out_indptr.astype(pointer_dtype, copy=False)
    dataset = _sparse_dataset(
        group, "indptr", pointer_dtype, cast_indptr.size, src.obj["indptr"]
    )
    dataset[:] = cast_indptr


def densify(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    chunk_rows: int = 1024,
) -> None:
    """Write a sparse matrix out as a dense array, a block of rows at a time.

    The zeros are the point: nothing here materialises the whole grid, so a
    matrix too large to densify in memory still converts -- it just produces
    a file that is honestly much larger, which `check_growth` warns about
    before any of it is written.
    """
    n_rows, n_cols = src.shape
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    dst = create_dataset(
        dst_parent, name, shape=(n_rows, n_cols), dtype=data_dtype
    )
    copy_attrs(src.obj.attrs, dst.attrs, target_backend=backend)
    spec.set_encoding(dst, spec.ARRAY)
    # `shape` belongs to the sparse encoding and would contradict the array's
    # own shape if it were carried over.
    if "shape" in dst.attrs:
        del dst.attrs["shape"]

    indptr = np.asarray(src.obj["indptr"][...], dtype=np.int64)
    indices, values = src.obj["indices"], src.obj["data"]
    csr = src.kind == spec.CSR_MATRIX
    n_major = n_rows if csr else n_cols

    for lo in range(0, n_major, chunk_rows):
        hi = min(lo + chunk_rows, n_major)
        start, end = int(indptr[lo]), int(indptr[hi])
        block = np.zeros(
            (hi - lo, n_cols) if csr else (n_rows, hi - lo), dtype=data_dtype
        )
        if end > start:
            minor = np.asarray(indices[start:end], dtype=np.int64)
            data = np.asarray(values[start:end])
            major = np.repeat(
                np.arange(hi - lo, dtype=np.int64), np.diff(indptr[lo : hi + 1])
            )
            # Repeated coordinates are legal in a CSR/CSC store and mean
            # their sum, which is what scipy's own `toarray` produces.
            # Plain assignment keeps whichever came last, so a
            # non-canonical input silently changed value on densifying.
            if csr:
                np.add.at(block, (major, minor), data)
            else:
                np.add.at(block, (minor, major), data)
        if csr:
            dst[lo:hi, :] = block
        else:
            dst[:, lo:hi] = block


def sparsify(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    enc: str,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype = np.dtype("int64"),
    chunk_rows: int = 1024,
    console: Optional[Console] = None,
) -> Tuple[int, float]:
    """Write a dense array out as CSR or CSC, a block at a time.

    Returns `(nnz, density)` so the caller can say whether it was worth it --
    a matrix that is half nonzero gets bigger, not smaller, and the user
    should hear that from us rather than from `du`.
    """
    from adata.core.subset import _append, _growable

    n_rows, n_cols = src.shape
    csr = enc == spec.CSR_MATRIX
    n_major = n_rows if csr else n_cols

    group = dst_parent.create_group(name)
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    copy_attrs(src.obj.attrs, group.attrs, target_backend=backend)
    spec.set_encoding(group, enc)
    set_shape_attr(group, src.shape)

    out_data = _growable_like(group, "data", data_dtype, src.obj)
    out_indices = _growable_like(group, "indices", index_dtype, src.obj)
    counts: List[int] = []

    for lo in range(0, n_major, chunk_rows):
        hi = min(lo + chunk_rows, n_major)
        block = np.asarray(src.obj[lo:hi, :] if csr else src.obj[:, lo:hi])
        if not csr:
            block = block.T  # iterate majors as rows either way
        nonzero_major, nonzero_minor = np.nonzero(block)
        counts.extend(np.bincount(nonzero_major, minlength=hi - lo).tolist())
        _append(out_indices, nonzero_minor.astype(index_dtype, copy=False))
        _append(out_data, block[nonzero_major, nonzero_minor].astype(
            data_dtype, copy=False
        ))

    indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    create_dataset(group, "indptr", data=indptr.astype(pointer_dtype, copy=False))


    nnz = int(indptr[-1])
    density = nnz / max(1, n_rows * n_cols)
    if console is not None and density > 0.5:
        console.print(
            f"[yellow]{name} is {density:.0%} nonzero; the sparse form is "
            f"larger than the dense one below about 33%.[/]"
        )
    return nnz, density


def cast_dense(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    chunk_rows: int = 1024,
) -> None:
    """Rewrite a dense matrix with a new dtype, a block of rows at a time."""
    n_rows, n_cols = src.shape
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    kw = dataset_create_kwargs(
        src.obj, target_backend=backend, dst_parent=dst_parent
    )
    from adata.core.subset import _clamp_chunks

    dst = create_dataset(
        dst_parent,
        name,
        shape=(n_rows, n_cols),
        dtype=data_dtype,
        **_clamp_chunks(kw, n_rows, n_cols),
    )
    copy_attrs(src.obj.attrs, dst.attrs, target_backend=backend)

    for lo in range(0, n_rows, chunk_rows):
        hi = min(lo + chunk_rows, n_rows)
        dst[lo:hi, :] = np.asarray(src.obj[lo:hi, :]).astype(
            data_dtype, copy=False
        )


# ---------------------------------------------------------------------------
# dispatch


@dataclass
class Plan:
    """A checked conversion, ready to run.

    Separating the decision from the writing is what makes "fails before
    anything is written" true rather than aspirational: the caller resolves
    every plan first, and only then creates the destination store. An
    earlier version checked inside the writer, so a refusal still left an
    output file holding a copy of obs and var.
    """

    source: Matrix
    layout: str
    data_dtype: np.dtype
    index_dtype: np.dtype
    #: `indptr` is tracked apart from `indices` because the two can
    #: legitimately differ: a narrow matrix with more than 2^31 nonzeros
    #: needs int64 offsets over int32 column indices. Inferring one from
    #: the other silently overflowed the offsets and corrupted the matrix.
    pointer_dtype: np.dtype = np.dtype("int64")
    report: Optional[CastReport] = None


def plan_conversion(
    obj: Any,
    name: str,
    *,
    dtype: Optional[np.dtype] = None,
    index_dtype: Optional[np.dtype] = None,
    layout: Optional[str] = None,
    chunk: int = DEFAULT_CHUNK,
    force: bool = False,
    console: Optional[Console] = None,
) -> Plan:
    """Decide what to do, and refuse here if it should not be done."""
    src = describe(obj)
    target_layout = resolve_layout(src, layout)
    data_dtype = np.dtype(dtype) if dtype is not None else src.dtype

    if index_dtype is not None:
        idx_dtype = ptr_dtype = np.dtype(index_dtype)
    elif src.sparse:
        # Keep what the source used, each independently. Defaulting to
        # int64 doubled the index arrays of every int32 store; inferring
        # indptr from indices narrowed the offsets of every store that
        # needed them wider.
        idx_dtype = np.dtype(src.obj["indices"].dtype)
        ptr_dtype = np.dtype(src.obj["indptr"].dtype)
    else:
        idx_dtype = ptr_dtype = np.dtype("int64")

    report: Optional[CastReport] = None
    if dtype is not None and data_dtype != src.dtype:
        values = src.obj["data"] if src.sparse else src.obj
        report = check_cast(values, data_dtype, chunk=chunk)
        message = report.describe(src.dtype, data_dtype)
        if not report.lossless and not force:
            raise ValueError(f"{name}: {message}. Pass --force to convert anyway.")
        if console is not None:
            colour = "dim" if report.lossless else "yellow"
            console.print(f"[{colour}]{name}: {message}[/]")

    # Always, not only when asked: an inferred dtype can be too narrow too,
    # and a silently overflowed offset is indistinguishable from corruption.
    if src.sparse and not force:
        check_index_dtype(src.obj, idx_dtype, ptr_dtype, src.shape)

    if target_layout == "dense" and src.sparse:
        projected = src.shape[0] * src.shape[1] * data_dtype.itemsize
        check_growth(
            _stored_bytes(src.obj), projected, force=force, what=f"{name} as dense"
        )

    return Plan(src, target_layout, data_dtype, idx_dtype, ptr_dtype, report)


def convert_matrix(
    obj: Any,
    dst_parent: Any,
    name: str,
    *,
    plan: Optional[Plan] = None,
    dtype: Optional[np.dtype] = None,
    index_dtype: Optional[np.dtype] = None,
    layout: Optional[str] = None,
    chunk: int = DEFAULT_CHUNK,
    chunk_rows: int = 1024,
    in_memory: bool = False,
    force: bool = False,
    console: Optional[Console] = None,
) -> None:
    """Write `obj` into `dst_parent` under `name`, converted.

    Pass a `plan` from `plan_conversion` to have the checks already done;
    without one they run here, which is convenient for a direct caller but
    means the destination already exists by the time a refusal is raised.
    """
    if plan is None:
        plan = plan_conversion(
            obj, name, dtype=dtype, index_dtype=index_dtype, layout=layout,
            chunk=chunk, force=force, console=console,
        )
    src = plan.source
    target_layout = plan.layout
    data_dtype = plan.data_dtype
    idx_dtype = plan.index_dtype
    ptr_dtype = plan.pointer_dtype

    # --- then write -------------------------------------------------------
    if target_layout == "dense":
        if src.sparse:
            densify(src, dst_parent, name, data_dtype=data_dtype, chunk_rows=chunk_rows)
        else:
            cast_dense(src, dst_parent, name, data_dtype=data_dtype, chunk_rows=chunk_rows)
        return

    if not src.sparse:
        sparsify(
            src,
            dst_parent,
            name,
            enc=target_layout,
            data_dtype=data_dtype,
            index_dtype=idx_dtype,
            pointer_dtype=ptr_dtype,
            chunk_rows=chunk_rows,
            console=console,
        )
        return

    if target_layout == src.kind:
        cast_sparse(
            src,
            dst_parent,
            name,
            data_dtype=data_dtype,
            index_dtype=idx_dtype,
            pointer_dtype=ptr_dtype,
            chunk=chunk,
        )
        return

    transpose = (
        transpose_sparse_in_memory if in_memory else transpose_sparse_streaming
    )
    extra: Dict[str, Any] = (
        {} if in_memory else {"chunk": chunk, "console": console}
    )
    transpose(
        src,
        dst_parent,
        name,
        data_dtype=data_dtype,
        index_dtype=idx_dtype,
        pointer_dtype=ptr_dtype,
        **extra,
    )

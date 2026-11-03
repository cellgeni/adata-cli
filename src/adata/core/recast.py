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
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
from rich.console import Console

from adata.elements import spec
from adata.elements.write import set_shape_attr
from adata.core.csr_build import (
    MIN_BUCKET_ENTRIES,
    BucketSorter,
    bucket_bounds,
    count_keys,
    scratch_group,
)
from adata.core.subset import _sparse_dataset
from adata.storage import (
    RowWriter,
    backend_chunks,
    row_bands,
    copy_attrs,
    create_dataset,
    dataset_create_kwargs,
    fit_chunks,
    is_dataset,
    is_group,
    is_zarr_group,
    with_compression,
    zarr_format_of,
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

#: Elements per chunk of a `data` or `indices` built from a dense matrix,
#: whose own chunks are a grid of rows and columns, not of elements.
ELEMENT_CHUNK = 1 << 16


def parse_dtype(text: str, *, allowed: Tuple[str, ...] = DTYPES) -> np.dtype:
    """Resolve a user-supplied dtype name, or say what is accepted."""
    name = text.strip().lower()
    if name not in allowed:
        raise ValueError(
            f"Unknown dtype {text!r}. Choose from: {', '.join(allowed)}."
        )
    return np.dtype(name)


def parse_index_dtype(text: str) -> np.dtype:
    """Resolve `--indices-dtype`, against `INDEX_DTYPES` as it stands."""
    return parse_dtype(text, allowed=INDEX_DTYPES)


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


def check_cast(
    dataset: Any,
    target: np.dtype,
    *,
    chunk: int = DEFAULT_CHUNK,
    rows: Optional[int] = None,
) -> CastReport:
    """Would casting `dataset` to `target` lose anything?

    Streams the values, casts each block and casts it back. An exact
    round-trip is the only honest test: whether float64 counts survive
    float32 depends on the counts, not on the dtypes, and that is precisely
    the question issue #13 is asking.

    A block is `chunk` elements of a 1-D array (a sparse matrix's `data`),
    or `rows` rows of a 2-D one -- `--chunk`'s rows, the block the cast then
    writes. It was `chunk` rows of a dense matrix too, which is every row of
    any real one: the whole matrix read at once, and two more copies of it
    made, whatever `--chunk` said (F029). Without `rows`, as many rows as
    hold `chunk` elements.
    """
    report = CastReport()
    source = np.dtype(getattr(dataset, "dtype", "float64"))
    shape = tuple(getattr(dataset, "shape", None) or ())
    n = int(shape[0]) if shape else 0
    step = int(chunk)
    if len(shape) > 1:
        width = int(np.prod(shape[1:], dtype=np.int64))
        step = int(rows) if rows is not None else chunk // max(1, width)
    step = max(1, step)

    info = np.finfo(target) if target.kind == "f" else (
        np.iinfo(target) if target.kind in "iu" else None
    )

    # In whole chunks of it: the check reads every value, as the cast does.
    for start, end in row_bands(dataset, step) if shape else ():
        block = np.asarray(dataset[start:end])
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
    *,
    minor: Optional[int] = None,
) -> None:
    """Refuse index dtypes that cannot address this matrix.

    Cheaper than `check_cast`: the largest value each array must hold is
    bounded by the dimensions and the nonzero count, so no pass over the
    data is needed. `indices` holds coordinates, bounded by the larger
    dimension; `indptr` holds offsets, bounded by nnz. They are checked
    separately because they can legitimately need different widths.

    A dense source has no nnz to read: pass `group=None` and only the
    coordinates are checked here, and `sparsify` checks the offsets as it
    counts them. `minor`, the extent of the output's minor axis, is what the
    coordinates reach (columns for CSR); without it, the larger dimension.
    """
    nnz = None if group is None else int(group["indices"].shape[0])
    reach = int(minor) if minor is not None else max(int(shape[0]), int(shape[1]))
    check_index_range("indices", index_dtype, reach, shape, nnz)
    if nnz is not None:
        check_index_range("indptr", pointer_dtype, nnz, shape, nnz)


def check_index_range(
    name: str, dtype: np.dtype, largest: int, shape: Tuple[int, int], nnz: Optional[int]
) -> None:
    """Refuse `dtype` for `indices` or `indptr` if `largest` is past its range."""
    limit = int(np.iinfo(dtype).max)
    if largest <= limit:
        return
    what = "offsets" if name == "indptr" else "coordinates"
    nonzeros = (
        f" with {nnz:,} nonzeros" if nnz is not None
        else f" with at least {largest:,} nonzeros" if name == "indptr" else ""
    )
    raise ValueError(
        f"{dtype} cannot hold this matrix's {what}: {name} must reach "
        f"{largest:,} for a {shape[0]:,} x {shape[1]:,} matrix{nonzeros}, "
        f"and {dtype} tops out at {limit:,}. Use int64."
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
        f"Not a matrix: encoding {enc!r}. `recast` handles X, layers, raw/X "
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
    codecs_from: Any = None,
) -> None:
    for name, values, dtype, template in (
        ("data", data, data_dtype, source["data"]),
        ("indices", indices, index_dtype, source["indices"]),
        ("indptr", indptr, pointer_dtype, source["indptr"]),
    ):
        cast = values.astype(dtype, copy=False)
        dataset = _output_array(group, name, dtype, cast.size, template, codecs_from)
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

    # Written whole shards at a time: a block of `chunk` entries into the
    # source's shard, a whole array under anndata 0.13, rewrote it once a
    # block (F036).
    out_data = RowWriter(_sparse_dataset(group, "data", data_dtype, nnz, source_data))
    out_indices = RowWriter(
        _sparse_dataset(group, "indices", index_dtype, nnz, source_indices)
    )

    for start in range(0, nnz, chunk):
        end = min(start + chunk, nnz)
        out_data.write(np.asarray(source_data[start:end]).astype(data_dtype, copy=False))
        out_indices.write(
            np.asarray(source_indices[start:end]).astype(index_dtype, copy=False)
        )
    out_data.flush()
    out_indices.flush()

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
    codecs_from: Any = None,
) -> None:
    """Swap CSR<->CSC by loading the matrix and sorting it once.

    Faster than streaming whenever the matrix fits, and the whole matrix is
    what it needs -- so it is opt-in, never the default. `codecs_from` as for
    `transpose_sparse_streaming`.
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
        codecs_from=codecs_from,
    )


#: Kept under its old name: the scratch group moved to `core.csr_build`.
_scratch_group = scratch_group


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
    compression: Optional[str] = None,
    keep_layout: bool = True,
    codecs_from: Any = None,
) -> None:
    """Swap CSR<->CSC without loading the matrix.

    A transpose cannot be done in one pass: the first entry of the output
    may come from the last row of the input. Three passes instead, with
    memory set by `chunk` and `bucket_entries` rather than by nnz:

    1. count nonzeros per output major, by streaming `indices` alone. That
       gives the output `indptr` by cumulative sum.
    2. stream the input again, `chunk` entries at a time, splitting each
       block's entries into buckets by which slice of the output they land
       in, and append each bucket to scratch datasets.
    3. read one bucket at a time, sort it, and append to the output in order.

    Costs about two extra passes over nnz, which is the price of not holding
    the matrix. `--in-memory` is there for when you would rather pay in RAM.
    `keep_layout=False` writes plain arrays rather than the source's chunks
    and codecs, for a transpose into scratch. `codecs_from`, a dense matrix,
    gives the output that matrix's codecs and element-sized chunks instead,
    for a transpose out of a scratch built from it.
    """
    target = (
        spec.CSC_MATRIX if src.kind == spec.CSR_MATRIX else spec.CSR_MATRIX
    )
    n_major_out = src.shape[1] if src.kind == spec.CSR_MATRIX else src.shape[0]

    indptr = np.asarray(src.obj["indptr"][...], dtype=np.int64)
    source_indices, source_data = src.obj["indices"], src.obj["data"]
    nnz = int(source_indices.shape[0])

    # Pass 1: how many entries land in each output major.
    counts = np.zeros(n_major_out, dtype=np.int64)
    for start in range(0, nnz, chunk):
        block = np.asarray(source_indices[start : min(start + chunk, nnz)], dtype=np.int64)
        count_keys(counts, block)
    out_indptr = np.concatenate(([0], np.cumsum(counts)))

    bounds = _transpose_bounds(out_indptr, chunk=chunk, bucket_entries=bucket_entries)
    n_buckets = len(bounds) - 1
    if console is not None and n_buckets > 1:
        console.print(
            f"[dim]Transposing {nnz:,} nonzeros through {n_buckets} buckets[/]"
        )

    # nnz is invariant under a transpose, so the output can be sized now and
    # written by slice rather than grown block by block.
    group = _new_sparse_group(src, dst_parent, name, target, src.shape)
    templates = (source_data, source_indices, src.obj["indptr"]) if keep_layout else (None,) * 3
    out_data = RowWriter(_output_array(
        group, "data", data_dtype, nnz, templates[0], codecs_from, compression=compression
    ))
    out_indices = RowWriter(_output_array(
        group, "indices", index_dtype, nnz, templates[1], codecs_from,
        compression=compression,
    ))

    with scratch_group(dst_parent, f"__{name}_transpose_scratch__") as scratch:
        sorter = BucketSorter(scratch, bounds, src.dtype)

        # Pass 2: scatter the input into buckets, `chunk` entries at a time.
        # It stepped by majors of average size, so on a skewed matrix a step
        # covered several majors far above the average -- adjacent genes
        # expressed in every cell -- and read many times the chunk at once,
        # more as n_obs grew (F077). A major holding more than the chunk is
        # split across blocks.
        for start in range(0, nnz, chunk):
            end = min(start + chunk, nnz)
            minor = np.asarray(source_indices[start:end], dtype=np.int64)
            values = np.asarray(source_data[start:end])
            # The input's minor axis is the output's major one.
            sorter.add(minor, _majors_of(indptr, start, end), values)

        # Pass 3: each bucket in turn, in output order, written a whole
        # shard at a time: a bucket at a time rewrote the shard (F036).
        for new_minor, values in _bucket_runs(sorter, chunk):
            out_indices.write(new_minor.astype(index_dtype, copy=False))
            out_data.write(values.astype(data_dtype, copy=False))
        out_indices.flush()
        out_data.flush()

    cast_indptr = out_indptr.astype(pointer_dtype, copy=False)
    dataset = _output_array(
        group, "indptr", pointer_dtype, cast_indptr.size, templates[2], codecs_from,
        compression=compression,
    )
    dataset[:] = cast_indptr


def _majors_of(indptr: np.ndarray, start: int, end: int) -> np.ndarray:
    """The major each entry in `[start, end)` belongs to, by `indptr`."""
    first = int(np.searchsorted(indptr, start, side="right")) - 1
    last = int(np.searchsorted(indptr, end - 1, side="right")) - 1
    spans = np.diff(np.clip(indptr[first : last + 2], start, end))
    return np.repeat(np.arange(first, last + 1, dtype=np.int64), spans)


def _transpose_bounds(
    out_indptr: np.ndarray, *, chunk: int, bucket_entries: Optional[int]
) -> np.ndarray:
    """Buckets of about one read each, and a bucket of its own for any
    output major holding more than that, so `_bucket_runs` can stream it.

    `bucket_bounds` balances by entries, but a bound can only fall between
    majors, so a gene in every cell shared a bucket with its neighbours and
    the bucket held n_obs entries or more. Concat's CSC buckets do the same
    (`core/concat.py`).
    """
    per_bucket = (
        max(1, int(bucket_entries)) if bucket_entries
        else max(MIN_BUCKET_ENTRIES, int(chunk))
    )
    bounds = bucket_bounds(out_indptr, chunk=chunk, bucket_entries=bucket_entries)
    big = np.flatnonzero(np.diff(out_indptr) > per_bucket)
    return np.unique(np.concatenate((bounds, big, big + 1)))


def _bucket_runs(sorter: BucketSorter, chunk: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Every bucket's `(new_minor, values)`, in output order.

    A bucket of several majors holds about a read's worth: it is read whole
    and sorted. A bucket of one major can hold far more -- n_obs entries for
    a gene in every cell -- and is streamed a chunk at a time instead. It
    needs no sort: pass 2 adds entries in input order, so one output major's
    entries arrive in the order of the input majors they came from, which is
    their new minor order.
    """
    for b, (majors, minors, values) in enumerate(sorter.buckets):
        n = int(majors.shape[0])
        if n == 0:
            continue
        if int(sorter.bounds[b + 1] - sorter.bounds[b]) == 1 and n > chunk:
            for lo in range(0, n, chunk):
                yield (
                    np.asarray(minors[lo : lo + chunk], dtype=np.int64),
                    np.asarray(values[lo : lo + chunk]),
                )
            continue
        major = np.asarray(majors[...], dtype=np.int64)
        minor = np.asarray(minors[...], dtype=np.int64)
        order = np.lexsort((minor, major))
        yield minor[order], np.asarray(values[...])[order]


def _transpose(src: Matrix, dst_parent: Any, name: str, *, in_memory: bool,
               chunk: int, **kwargs: Any) -> None:
    """`transpose_sparse_in_memory` or `_streaming`, as asked."""
    if in_memory:
        kwargs.pop("keep_layout", None)
        kwargs.pop("compression", None)
        transpose_sparse_in_memory(src, dst_parent, name, **kwargs)
    else:
        transpose_sparse_streaming(src, dst_parent, name, chunk=chunk, **kwargs)


def densify(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    chunk_rows: int = 1024,
    chunk: int = DEFAULT_CHUNK,
    in_memory: bool = False,
) -> None:
    """Write a sparse matrix out as a dense array, a block of rows at a time.

    The zeros are the point: nothing here materialises the whole grid, so a
    matrix too large to densify in memory still converts -- it just produces
    a file that is honestly much larger, which `check_growth` warns about
    before any of it is written.

    A block is `chunk_rows` rows whatever the layout. A CSC matrix is first
    transposed to CSR in scratch: its blocks were `chunk_rows` columns of
    every row, n_obs x --chunk values, so memory followed n_obs and not
    --chunk (F030). Writing narrow column slabs instead would rewrite every
    chunk of the dense output once per slab.
    """
    if src.kind != spec.CSC_MATRIX:
        _densify_rows(src, dst_parent, name, src.obj.attrs,
                      data_dtype=data_dtype, chunk_rows=chunk_rows)
        return
    with scratch_group(dst_parent, f"__{name}_densify_scratch__") as scratch:
        _transpose(
            src, scratch, "csr", in_memory=in_memory, chunk=chunk,
            data_dtype=src.dtype, index_dtype=np.dtype("int64"),
            pointer_dtype=np.dtype("int64"), keep_layout=False,
        )
        _densify_rows(describe(scratch["csr"]), dst_parent, name, src.obj.attrs,
                      data_dtype=data_dtype, chunk_rows=chunk_rows)


def _densify_rows(
    src: Matrix, dst_parent: Any, name: str, attrs: Any, *,
    data_dtype: np.dtype, chunk_rows: int,
) -> None:
    """Densify a CSR matrix `chunk_rows` rows at a time, with `attrs`."""
    n_rows, n_cols = src.shape
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    # zarr's own chunks, cut to what `RowWriter` holds, so each is written
    # once: --chunk rows into chunks 12,500 tall encoded 13 times the matrix.
    kw = fit_chunks(
        backend_chunks(dst_parent, {}, (n_rows, n_cols), data_dtype), (n_rows, n_cols),
        backend=backend, itemsize=np.dtype(data_dtype).itemsize,
    )
    dst = create_dataset(
        dst_parent, name, shape=(n_rows, n_cols), dtype=data_dtype, **kw
    )
    copy_attrs(attrs, dst.attrs, target_backend=backend)
    spec.set_encoding(dst, spec.ARRAY)
    # `shape` belongs to the sparse encoding and would contradict the array's
    # own shape if it were carried over.
    if "shape" in dst.attrs:
        del dst.attrs["shape"]

    indptr = np.asarray(src.obj["indptr"][...], dtype=np.int64)
    indices, values = src.obj["indices"], src.obj["data"]

    writer = RowWriter(dst)
    for lo in range(0, n_rows, chunk_rows):
        hi = min(lo + chunk_rows, n_rows)
        start, end = int(indptr[lo]), int(indptr[hi])
        block = np.zeros((hi - lo, n_cols), dtype=data_dtype)
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
            np.add.at(block, (major, minor), data)
        writer.write(block)
    writer.flush()


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
    chunk: int = DEFAULT_CHUNK,
    in_memory: bool = False,
    console: Optional[Console] = None,
    compression: Optional[str] = None,
    room: Optional[Callable[[int], None]] = None,
) -> Tuple[int, float]:
    """Write a dense array out as CSR or CSC, `chunk_rows` rows at a time.

    Returns `(nnz, density)` so the caller can say whether it was worth it --
    a matrix that is half nonzero gets bigger, not smaller, and the user
    should hear that from us rather than from `du`.

    CSC goes through a CSR in scratch and the streaming transpose. It read
    `chunk_rows` columns of every row at a time, n_obs x --chunk values
    (F030), and each of those reads decoded every chunk of a row-chunked
    source, once per block of columns. `room(nbytes)` is called before the
    transpose with what it will write, nnz being known only then, so that a
    free-space check can refuse first.
    """
    n_rows, n_cols = src.shape
    if enc == spec.CSC_MATRIX:
        with scratch_group(dst_parent, f"__{name}_sparsify_scratch__") as scratch:
            # Plain arrays: scratch has no chunk cache, so a compressed one
            # decoded and re-encoded a chunk on every append, 30 to 100
            # times slower. The output takes the source's codecs instead.
            nnz = _sparsify_rows(
                src, scratch, "csr", data_dtype=data_dtype,
                index_dtype=np.dtype("int64"), pointer_dtype=np.dtype("int64"),
                chunk_rows=chunk_rows, compression=None, template=None,
                offsets_within=pointer_dtype,
            )
            if room is not None:
                # nnz is known only now: the buckets and the output next.
                room(nnz * (16 + 2 * np.dtype(data_dtype).itemsize
                            + np.dtype(index_dtype).itemsize))
            _transpose(
                describe(scratch["csr"]), dst_parent, name, in_memory=in_memory,
                chunk=chunk, data_dtype=data_dtype, index_dtype=index_dtype,
                pointer_dtype=pointer_dtype, compression=compression,
                codecs_from=src.obj,
            )
    else:
        nnz = _sparsify_rows(
            src, dst_parent, name, data_dtype=data_dtype, index_dtype=index_dtype,
            pointer_dtype=pointer_dtype, chunk_rows=chunk_rows,
            compression=compression, template=src.obj,
        )

    density = nnz / max(1, n_rows * n_cols)
    if console is not None and density > 0.5:
        console.print(
            f"[yellow]{name} is {density:.0%} nonzero; the sparse form is "
            f"larger than the dense one below about 33%.[/]"
        )
    return nnz, density


def _sparsify_rows(
    src: Matrix,
    dst_parent: Any,
    name: str,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    pointer_dtype: np.dtype,
    chunk_rows: int,
    compression: Optional[str],
    template: Any,
    offsets_within: Optional[np.dtype] = None,
) -> int:
    """Write a dense matrix out as CSR, `chunk_rows` rows at a time; its nnz.

    `template` gives the arrays its codecs (None for plain ones). nnz is
    refused as soon as it passes `offsets_within` (by default
    `pointer_dtype`), so a CSR written to scratch on its way to narrower
    offsets is not finished first.
    """
    from adata.core.subset import _append

    n_rows, n_cols = src.shape
    group = dst_parent.create_group(name)
    backend = "zarr" if is_zarr_group(dst_parent) else "hdf5"
    copy_attrs(src.obj.attrs, group.attrs, target_backend=backend)
    spec.set_encoding(group, spec.CSR_MATRIX)
    set_shape_attr(group, src.shape)

    # Appended to whole chunks at a time: Zarr keeps no cache, so a block
    # appended to part of a chunk decoded and encoded it again.
    out_data = RowWriter(_growable_from(
        group, "data", data_dtype, template, compression=compression
    ), grow=True)
    out_indices = RowWriter(_growable_from(
        group, "indices", index_dtype, template, compression=compression
    ), grow=True)
    # Per-row counts in one int64 array: a Python list of int objects took
    # 36 to 40 bytes a row against 8.
    counts = np.zeros(n_rows, dtype=np.int64)
    offsets_within = np.dtype(offsets_within or pointer_dtype)
    pointer_limit = int(np.iinfo(offsets_within).max)
    nnz = 0

    for lo, hi in row_bands(src.obj, chunk_rows):
        # Cast before looking for nonzeros: a lossy cast turns 0.3 into 0,
        # and finding the nonzeros first stored those as explicit zeros
        # (F121), 616 entries where scipy has 194.
        with np.errstate(invalid="ignore", over="ignore"):
            block = np.asarray(src.obj[lo:hi, :]).astype(data_dtype, copy=False)
        rows, cols = np.nonzero(block)
        nnz += int(rows.size)
        if nnz > pointer_limit:
            # Offsets cast with a plain astype wrapped negative (F013).
            check_index_range("indptr", offsets_within, nnz, src.shape, None)
        counts[lo:hi] = np.bincount(rows, minlength=hi - lo)
        _append(out_indices, cols.astype(index_dtype, copy=False))
        _append(out_data, block[rows, cols])

    out_indices.flush()
    out_data.flush()
    indptr = np.concatenate(([0], np.cumsum(counts)))
    kw = {}
    if compression is not None:
        kw = with_compression(
            kw, compression, shape=indptr.shape, dtype=pointer_dtype,
            backend=backend, zarr_format=zarr_format_of(dst_parent),
        )
    create_dataset(
        group, "indptr", data=indptr.astype(pointer_dtype, copy=False), **kw
    )
    return nnz


def _growable_from(
    group: Any,
    name: str,
    dtype: np.dtype,
    template: Any,
    *,
    compression: Optional[str] = None,
) -> Any:
    """A growable 1-D array with `template`'s codecs and never its chunks.

    `_growable_like` takes a template's first chunk dimension as the element
    chunk, which for a dense matrix is a row height: a source chunked 16
    rows tall gave `data` 16-element chunks, 281,188 of them for 4.5M
    nonzeros and a file three times the size (F078). With no template, a
    plain array.
    """
    kw = _codecs(group, dtype, template, compression)
    if is_zarr_group(group):
        return group.create_array(name, shape=(0,), dtype=dtype, chunks=(ELEMENT_CHUNK,), **kw)
    return group.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(ELEMENT_CHUNK,), **kw
    )


def _output_array(
    group: Any,
    name: str,
    dtype: np.dtype,
    n: int,
    template: Any,
    codecs_from: Any,
    *,
    compression: Optional[str] = None,
) -> Any:
    """A 1-D array of `n` elements: laid out like `template`, or, given
    `codecs_from` (a dense matrix), with its codecs and element chunks."""
    if codecs_from is None:
        return _sparse_dataset(group, name, dtype, n, template, compression=compression)
    backend = "zarr" if is_zarr_group(group) else "hdf5"
    kw = _codecs(group, dtype, codecs_from, compression)
    kw = fit_chunks({**kw, "chunks": (ELEMENT_CHUNK,)}, (n,), backend=backend)
    return create_dataset(group, name, shape=(n,), dtype=dtype, **kw)


def _codecs(group: Any, dtype: np.dtype, template: Any, compression: Optional[str]) -> dict:
    """Creation kwargs for a 1-D array: `template`'s codecs, not its layout.

    Its chunks and shards are a grid of the template's own shape, and its
    fill value and scale-offset filter belong to the template's dtype.
    """
    backend = "zarr" if is_zarr_group(group) else "hdf5"
    kw = (
        dataset_create_kwargs(template, target_backend=backend, dst_parent=group)
        if template is not None else {}
    )
    for key in ("chunks", "shards", "fillvalue", "scaleoffset"):
        kw.pop(key, None)
    if compression is not None:
        kw = with_compression(
            kw, compression, shape=(0,), dtype=dtype, backend=backend,
            zarr_format=zarr_format_of(group),
        )
    return kw


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
    dst = create_dataset(
        dst_parent,
        name,
        shape=(n_rows, n_cols),
        dtype=data_dtype,
        **fit_chunks(kw, (n_rows, n_cols), backend=backend,
                     itemsize=np.dtype(data_dtype).itemsize),
    )
    copy_attrs(src.obj.attrs, dst.attrs, target_backend=backend)

    # A whole shard at a time: a block of --chunk rows rewrote the source's
    # shard once a block, 537 MB encoded for 4.4 MB written (F036).
    writer = RowWriter(dst)
    for lo, hi in row_bands(src.obj, chunk_rows):
        writer.write(np.asarray(src.obj[lo:hi, :]).astype(data_dtype, copy=False))
    writer.flush()


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
    chunk_rows: int = 1024,
    force: bool = False,
    console: Optional[Console] = None,
) -> Plan:
    """Decide what to do, and refuse here if it should not be done.

    `chunk_rows` is the block of a dense matrix the cast check reads, as the
    conversion will.
    """
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

    # The refusals that need no pass over the values come first, so that a
    # range or growth refusal does not wait for the whole matrix to be read.
    if target_layout in spec.SPARSE_TYPES and data_dtype == np.dtype("float16"):
        # scipy has no float16 sparse matrix, so anndata, scanpy and this
        # tool's own readers fail on one. It was written, with a message
        # that every value round-tripped (F058).
        raise ValueError(
            f"{name}: a sparse matrix cannot hold float16 -- scipy, and so "
            "anndata and scanpy, cannot read one. Use --dtype float32, or "
            "--layout dense."
        )

    # Always, not only when asked: an inferred dtype can be too narrow too,
    # and a silently overflowed offset is indistinguishable from corruption.
    # Nor does --force waive it: it accepts lossy values and a large store,
    # and it let indptr wrap negative (F013). A dense source's coordinates
    # are checked here and its offsets as `sparsify` counts them.
    if target_layout in spec.SPARSE_TYPES:
        check_index_dtype(
            src.obj if src.sparse else None, idx_dtype, ptr_dtype, src.shape,
            minor=src.shape[1] if target_layout == spec.CSR_MATRIX else src.shape[0],
        )

    if target_layout == "dense" and src.sparse:
        projected = src.shape[0] * src.shape[1] * data_dtype.itemsize
        check_growth(
            _stored_bytes(src.obj), projected, force=force, what=f"{name} as dense"
        )

    report: Optional[CastReport] = None
    if dtype is not None and data_dtype != src.dtype:
        values = src.obj["data"] if src.sparse else src.obj
        report = check_cast(values, data_dtype, chunk=chunk, rows=chunk_rows)
        message = report.describe(src.dtype, data_dtype)
        if not report.lossless and not force:
            raise ValueError(f"{name}: {message}. Pass --force to convert anyway.")
        if console is not None:
            colour = "dim" if report.lossless else "yellow"
            console.print(f"[{colour}]{name}: {message}[/]")

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
    room: Optional[Callable[[int], None]] = None,
) -> None:
    """Write `obj` into `dst_parent` under `name`, converted.

    Pass a `plan` from `plan_conversion` to have the checks already done;
    without one they run here, which is convenient for a direct caller but
    means the destination already exists by the time a refusal is raised.
    """
    if plan is None:
        plan = plan_conversion(
            obj, name, dtype=dtype, index_dtype=index_dtype, layout=layout,
            chunk=chunk, chunk_rows=chunk_rows, force=force, console=console,
        )
    src = plan.source
    target_layout = plan.layout
    data_dtype = plan.data_dtype
    idx_dtype = plan.index_dtype
    ptr_dtype = plan.pointer_dtype

    # --- then write -------------------------------------------------------
    if target_layout == "dense":
        if src.sparse:
            densify(
                src, dst_parent, name, data_dtype=data_dtype,
                chunk_rows=chunk_rows, chunk=chunk, in_memory=in_memory,
            )
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
            chunk=chunk,
            in_memory=in_memory,
            console=console,
            room=room,
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

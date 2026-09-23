from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import shutil
import warnings

import h5py

try:
    import zarr
except Exception:  # pragma: no cover - optional dependency
    zarr = None

import numpy as np


ROOT_ENCODING_TYPE = "anndata"
ROOT_ENCODING_VERSION = "0.1.0"


@dataclass
class Store:
    backend: str
    root: Any
    path: Path
    zarr_format: Optional[int] = None
    #: Whether to rewrite the Zarr consolidated metadata index on close.
    consolidate: bool = False

    def close(self) -> None:
        if self.backend == "hdf5":
            try:
                self.root.close()
            except Exception:
                return
            return

        if self.consolidate and zarr is not None:
            # anndata writes a consolidated metadata index at the root. New
            # members land on disk regardless, but every reader that honours
            # the index -- anndata included -- keeps using the stale copy and
            # cannot see them, so the write looks like a silent no-op.
            try:
                with warnings.catch_warnings():
                    # zarr notes that consolidated metadata is not part of the
                    # v3 spec. We write it deliberately, because anndata does
                    # and because without it our writes are invisible to
                    # readers that trust the index.
                    warnings.simplefilter("ignore")
                    zarr.consolidate_metadata(self.root.store, path=self.root.path)
            except Exception:
                return

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _require_zarr() -> None:
    if zarr is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "zarr is required for .zarr support. Install with: uv sync --extra zarr"
        )


def is_hdf5_group(obj: Any) -> bool:
    return isinstance(obj, (h5py.File, h5py.Group))


def is_hdf5_dataset(obj: Any) -> bool:
    return isinstance(obj, h5py.Dataset)


def is_zarr_group(obj: Any) -> bool:
    return zarr is not None and isinstance(obj, zarr.Group)


def is_zarr_array(obj: Any) -> bool:
    return zarr is not None and isinstance(obj, zarr.Array)


def is_group(obj: Any) -> bool:
    return is_hdf5_group(obj) or is_zarr_group(obj)


def is_dataset(obj: Any) -> bool:
    return is_hdf5_dataset(obj) or is_zarr_array(obj)


def is_zarr_path(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "zarr.json").exists():
        return True
    if (path / ".zgroup").exists() or (path / ".zattrs").exists():
        return True
    return False


def detect_backend(path: Path) -> str:
    if path.exists():
        if path.is_dir():
            if is_zarr_path(path):
                return "zarr"
            raise ValueError(
                f"Path '{path}' is a directory but does not look like a Zarr store."
            )
        return "hdf5"
    if path.suffix == ".zarr":
        return "zarr"
    return "hdf5"


def open_store(
    path: Path,
    mode: str,
    zarr_format: Optional[int] = None,
    require_anndata: bool = True,
) -> Store:
    """Open a store, auto-detecting the backend from `path`.

    `zarr_format` selects the Zarr spec version for a store being created;
    without it zarr-python uses its own default, which is v3. Callers writing a
    derived store should pass the source store's format so a v2 input is not
    silently upgraded.

    Set `require_anndata=False` for format-agnostic commands such as `ls`,
    which are expected to open plain HDF5 and Zarr stores and should not warn
    about a missing AnnData root.
    """
    path = Path(path)
    backend = detect_backend(path)
    if backend == "zarr":
        _require_zarr()
        kwargs: dict = {}
        if zarr_format is not None:
            kwargs["zarr_format"] = zarr_format

        writable = _is_writable_mode(mode)
        if writable:
            # Work against the real hierarchy: a consolidated index is a
            # snapshot, so members added through it are invisible even to the
            # handle that created them.
            kwargs["use_consolidated"] = False

        root = zarr.open_group(str(path), mode=mode, **kwargs)
        if writable:
            ensure_anndata_root_attrs(root)
        elif require_anndata:
            warn_if_missing_anndata_root_attrs(root, path=path)
        return Store(
            backend="zarr",
            root=root,
            path=path,
            zarr_format=zarr_format_of(root),
            consolidate=writable,
        )
    root = h5py.File(path, mode)
    if _is_writable_mode(mode):
        ensure_anndata_root_attrs(root)
    elif require_anndata:
        warn_if_missing_anndata_root_attrs(root, path=path)
    return Store(backend="hdf5", root=root, path=path)


def zarr_format_of(obj: Any) -> Optional[int]:
    """The Zarr spec version (2 or 3) backing `obj`, or None for HDF5."""
    return getattr(getattr(obj, "metadata", None), "zarr_format", None)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _is_writable_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "+", "x"))


def has_valid_anndata_root_attrs(root: Any) -> bool:
    enc_type = _decode_attr(root.attrs.get("encoding-type", None))
    enc_ver = _decode_attr(root.attrs.get("encoding-version", None))
    return enc_type == ROOT_ENCODING_TYPE and enc_ver == ROOT_ENCODING_VERSION


def ensure_anndata_root_attrs(root: Any) -> None:
    root.attrs["encoding-type"] = ROOT_ENCODING_TYPE
    root.attrs["encoding-version"] = ROOT_ENCODING_VERSION


def warn_if_missing_anndata_root_attrs(root: Any, *, path: Path) -> None:
    if has_valid_anndata_root_attrs(root):
        return

    enc_type = _decode_attr(root.attrs.get("encoding-type", None))
    enc_ver = _decode_attr(root.attrs.get("encoding-version", None))
    warnings.warn(
        (
            f"Store '{path}' root has missing or invalid AnnData attrs "
            f"(encoding-type={ROOT_ENCODING_TYPE!r}, encoding-version={ROOT_ENCODING_VERSION!r}). "
            f"Found encoding-type={enc_type!r}, encoding-version={enc_ver!r}."
        ),
        UserWarning,
        stacklevel=2,
    )


def _normalize_attr_value(value: Any, target_backend: str) -> Any:
    if target_backend == "zarr":
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (list, tuple)):
            return [v.decode("utf-8") if isinstance(v, bytes) else v for v in value]
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ("S", "O"):
                return [
                    v.decode("utf-8") if isinstance(v, bytes) else v
                    for v in value.tolist()
                ]
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value


def copy_attrs(src_attrs: Any, dst_attrs: Any, *, target_backend: str) -> None:
    """Copy attributes across, normalising values for the target backend.

    Written in one go on Zarr. Each `attrs[k] = v` there persists the whole
    metadata document through zarr's sync-over-async bridge, so setting them
    one at a time means a separate round trip per attribute -- slow, and the
    place a `split` over many groups was observed to wedge in CI.
    """
    normalized = {
        str(k): _normalize_attr_value(v, target_backend)
        for k, v in src_attrs.items()
    }
    if not normalized:
        return

    put = getattr(dst_attrs, "put", None)
    if target_backend == "zarr" and callable(put):
        put({**dict(dst_attrs), **normalized})
        return

    for k, v in normalized.items():
        dst_attrs[k] = v


def dataset_create_kwargs(
    src: Any,
    *,
    target_backend: str,
    zarr_format: Optional[int] = None,
    dst_parent: Any = None,
) -> dict:
    """Derive creation kwargs that carry a source's layout onto a new dataset.

    Chunking, compression and sharding are preserved where the target backend
    can express them; codecs that do not survive the crossing are dropped
    rather than forwarded into an error.
    """
    # Prefer the destination's own version over anything the caller guessed,
    # so a call site that forgets to pass one still behaves correctly.
    kw_target = zarr_format
    if dst_parent is not None:
        kw_target = zarr_format_of(dst_parent)
    kw: dict = {}
    chunks = getattr(src, "chunks", None)
    if chunks is not None:
        kw["chunks"] = chunks
    if target_backend == "hdf5" and is_hdf5_dataset(src):
        if src.compression is not None:
            kw["compression"] = src.compression
            kw["compression_opts"] = src.compression_opts
        kw["shuffle"] = bool(src.shuffle)
        kw["fletcher32"] = bool(src.fletcher32)
        if src.scaleoffset is not None:
            kw["scaleoffset"] = src.scaleoffset
        if src.fillvalue is not None:
            kw["fillvalue"] = src.fillvalue
    if target_backend == "zarr" and is_zarr_array(src):
        src_zarr_format = getattr(getattr(src, "metadata", None), "zarr_format", None)
        target_format = _target_zarr_format(kw_target)
        # Only when both versions are known and equal can codecs travel.
        same_version = (
            target_format is not None and src_zarr_format == target_format
        )

        # Codecs only travel between stores of the same Zarr version: v2 holds
        # numcodecs objects, v3 holds its own codec classes, and neither
        # accepts the other's. Across versions the target's default is used
        # rather than a translation that fails at creation time.
        if same_version:
            if src_zarr_format == 3:
                try:
                    compressors = getattr(src, "compressors", None)
                except Exception:
                    compressors = None
                if compressors is not None:
                    kw["compressors"] = compressors
            else:
                try:
                    compressor = getattr(src, "compressor", None)
                except Exception:
                    compressor = None
                if compressor is not None:
                    kw["compressor"] = compressor

            try:
                filters = getattr(src, "filters", None)
            except Exception:
                filters = None
            # A v2 string array carries VLenUTF8 in `filters`; the v3 string
            # dtype encodes variable length itself and rejects it.
            if filters and not _is_string_src(src):
                kw["filters"] = filters

        try:
            shards = getattr(src, "shards", None)
        except Exception:
            shards = None
        if shards is not None and target_format == 3:
            kw["shards"] = shards

        try:
            fill_value = getattr(src, "fill_value", None)
        except Exception:
            fill_value = None
        if fill_value is not None and not _is_string_src(src):
            kw["fill_value"] = fill_value
    return kw


def _create_string_dataset(
    parent: Any,
    name: str,
    data: Any,
    **kwargs: Any,
) -> Any:
    """Create a spec-compliant variable-length UTF-8 array from `data`.

    Text needs a different spelling on each backend and neither accepts the
    other's: Zarr rejects the `object` dtype an h5py vlen dataset reports, and
    h5py rejects Zarr's `<U` dtype. Both are routed to variable-length UTF-8
    here, which is what `string-array` requires.
    """
    from adata.elements.strings import as_str_array, string_dtype_for

    values = as_str_array(data)

    if is_zarr_group(parent):
        kwargs.pop("compressor", None)
        kwargs.pop("filters", None)
        arr = parent.create_array(
            name,
            shape=values.shape,
            dtype=string_dtype_for("zarr"),
            **kwargs,
        )
        if values.shape == ():
            arr[()] = values.item()
        else:
            arr[...] = values
        return arr

    return parent.create_dataset(
        name, data=values, dtype=string_dtype_for("hdf5"), **kwargs
    )


def create_dataset(
    parent: Any,
    name: str,
    *,
    data: Any = None,
    shape: Optional[Sequence[int]] = None,
    dtype: Any = None,
    **kwargs: Any,
) -> Any:
    """Create an array under `parent`, backend-agnostically.

    String data is always written as variable-length UTF-8 regardless of how
    it arrived, so callers can hand over bytes, `<U`, `object` or StringDType
    without knowing what the destination backend needs.
    """
    from adata.elements.strings import is_string_dtype

    if data is not None and dtype is None:
        probe = data if hasattr(data, "dtype") else np.asarray(data)
        if is_string_dtype(getattr(probe, "dtype", None)):
            return _create_string_dataset(parent, name, probe, **kwargs)
    elif data is None and dtype is not None and is_string_dtype(dtype):
        from adata.elements.strings import string_dtype_for

        backend = "zarr" if is_zarr_group(parent) else "hdf5"
        dtype = string_dtype_for(backend)
        if is_zarr_group(parent):
            kwargs.pop("compressor", None)
            kwargs.pop("filters", None)

    if is_zarr_group(parent):
        zarr_format = getattr(getattr(parent, "metadata", None), "zarr_format", None)
        if zarr_format == 3:
            kwargs = dict(kwargs)
            kwargs.pop("compressor", None)
        elif (
            zarr_format == 2 and "compressors" in kwargs and "compressor" not in kwargs
        ):
            kwargs = dict(kwargs)
            compressors = kwargs.pop("compressors")
            if isinstance(compressors, (list, tuple)) and len(compressors) == 1:
                kwargs["compressor"] = compressors[0]
        if data is not None:
            return parent.create_array(name, data=data, **kwargs)
        return parent.create_array(name, shape=shape, dtype=dtype, **kwargs)
    if data is not None:
        return parent.create_dataset(name, data=data, **kwargs)
    return parent.create_dataset(name, shape=shape, dtype=dtype, **kwargs)


def _target_zarr_format(zarr_format: Optional[int]) -> Optional[int]:
    """The destination's Zarr version, or None when the caller did not say.

    Deliberately not defaulting to 3: guessing meant v3-only options such as
    sharding were forwarded into v2 arrays, which reject them outright.
    Unknown means "carry nothing version-specific".
    """
    return zarr_format


def _is_string_src(src: Any) -> bool:
    from adata.elements.strings import is_string_dtype

    return is_string_dtype(getattr(src, "dtype", None))


#: Byte budget for a single read when streaming a dataset.
#:
#: Using a source's chunk height as the read size means inheriting whatever
#: the writer chose, and a store chunked `(1, n_cols)` is then copied one row
#: per read. On a local disk that is merely wasteful; on a network filesystem
#: (Lustre, NFS) every read is a round-trip costing milliseconds, so a
#: million-row copy spends nearly all of its time waiting. 32 MiB sits
#: comfortably above a typical 1 MiB Lustre stripe while bounding peak memory.
TARGET_READ_BYTES = 32 * 1024 * 1024

#: Fallback width of a variable-length string element, used only when the
#: real width cannot be sampled. h5py reports `itemsize` 8 for vlen strings
#: because the value is a pointer, which would overestimate the row count by
#: an order of magnitude. Real cell and gene names sit well under this.
VLEN_ELEMENT_BYTES = 64

#: Elements read when sampling a variable-length array's real element width.
#: One small read, against a copy that is about to stream the whole array.
VLEN_SAMPLE_ROWS = 256


def _is_vlen(dtype: Any) -> bool:
    """Does this dtype hide its real element width behind a pointer?

    'O' is how h5py spells vlen str, 'T' is numpy StringDType as reported by
    zarr-python 3. Neither `itemsize` reflects the bytes actually stored.
    """
    itemsize = int(getattr(dtype, "itemsize", 0) or 0)
    return itemsize <= 0 or getattr(dtype, "kind", None) in ("O", "T")


def _sample_element_bytes(src: Any, n_rows: int) -> int:
    """Mean stored width of a variable-length element, by reading a few.

    An assumed width is not a bound. Estimating 64 bytes and reading
    `TARGET_READ_BYTES // 64` elements means the read is 32 MiB only if the
    guess holds: at 4 KiB elements it is 2 GiB, and for an array shorter than
    the computed step the whole thing is read at once. Measured before this:
    copying 200,000 strings of 4 KiB peaked at 827 MB against a stated 32 MiB
    budget.

    One small read fixes that, and it is negligible beside the copy it is
    about to size -- `uns` can hold arbitrary text, so the width is not
    something this layer can assume.
    """
    try:
        sample = np.asarray(src[: min(VLEN_SAMPLE_ROWS, max(1, n_rows))])
    except Exception:  # pragma: no cover - unreadable source
        return VLEN_ELEMENT_BYTES

    total = 0
    count = 0
    for value in sample.reshape(-1)[:VLEN_SAMPLE_ROWS]:
        try:
            total += len(value)
        except TypeError:  # pragma: no cover - non-sized element
            total += VLEN_ELEMENT_BYTES
        count += 1

    if not count:
        return VLEN_ELEMENT_BYTES
    # The floor keeps a column of empty strings from producing an unbounded
    # step; the object header dominates at that size anyway.
    return max(VLEN_ELEMENT_BYTES, total // count)


def _row_bytes(
    shape: Sequence[int], dtype: Any, element_bytes: int = VLEN_ELEMENT_BYTES
) -> int:
    """In-memory size of one row along the first axis.

    `element_bytes` is the measured width for a variable-length dtype; the
    default is the fallback for callers that have no sample to offer.
    """
    width = 1
    for dim in shape[1:]:
        width *= max(1, int(dim))

    itemsize = int(getattr(dtype, "itemsize", 0) or 0)
    if _is_vlen(dtype):
        itemsize = element_bytes

    return max(1, width * itemsize)


def _chunk_step(
    shape: Sequence[int],
    chunks: Optional[Sequence[int]],
    dtype: Any,
    element_bytes: int = VLEN_ELEMENT_BYTES,
) -> int:
    """Rows to copy per read, sized for the filesystem rather than the source.

    The step is grown to `TARGET_READ_BYTES`, then rounded down to a whole
    number of source chunks: reading part of a chunk still costs decompressing
    all of it, so a step that splits one wastes the remainder.
    """
    if not shape:
        return 1

    n_rows = int(shape[0])
    if n_rows <= 0:
        return 1

    chunk_rows = 0
    if chunks is not None and len(chunks) > 0 and chunks[0]:
        chunk_rows = max(1, int(chunks[0]))

    step = max(1, TARGET_READ_BYTES // _row_bytes(shape, dtype, element_bytes))
    if chunk_rows:
        # Never go below a single chunk: a partial read still decompresses the
        # whole thing, so a smaller step costs the same I/O for less data. That
        # makes TARGET_READ_BYTES a target rather than a cap -- a source whose
        # own chunk already exceeds the budget (say 1000 x 1e6 float32, chunked
        # whole) reads that chunk regardless. This matches the previous
        # behaviour, which used the chunk height verbatim.
        step = max(chunk_rows, (step // chunk_rows) * chunk_rows)

    return min(step, n_rows)


def copy_dataset(src: Any, dst_group: Any, name: str) -> Any:
    """Copy a dataset into `dst_group`, streaming it in chunks.

    The destination dtype is resolved for the target backend rather than
    reused: an h5py variable-length string dataset reports `object`, which Zarr
    refuses to create, so copying one verbatim fails on every real store.
    """
    from adata.elements.strings import target_dtype

    shape = tuple(src.shape) if getattr(src, "shape", None) is not None else ()
    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    zformat = zarr_format_of(dst_group)
    ds = create_dataset(
        dst_group,
        name,
        shape=shape,
        dtype=target_dtype(src.dtype, target_backend, zformat),
        **dataset_create_kwargs(
            src, target_backend=target_backend, zarr_format=zformat
        ),
    )
    copy_attrs(src.attrs, ds.attrs, target_backend=target_backend)

    if shape == ():
        ds[()] = src[()]
        return ds

    element_bytes = (
        _sample_element_bytes(src, shape[0]) if _is_vlen(src.dtype)
        else VLEN_ELEMENT_BYTES
    )
    step = _chunk_step(shape, getattr(src, "chunks", None), src.dtype, element_bytes)
    for start in range(0, shape[0], step):
        end = min(start + step, shape[0])
        if len(shape) == 1:
            ds[start:end] = src[start:end]
        else:
            ds[start:end, ...] = src[start:end, ...]
    return ds


def copy_tree(
    src_obj: Any, dst_group: Any, name: str, *, exclude: Iterable[str] = ()
) -> Any:
    if is_hdf5_group(dst_group) and (
        is_hdf5_group(src_obj) or is_hdf5_dataset(src_obj)
    ):
        if not exclude:
            dst_group.copy(src_obj, dst_group, name)
            return dst_group[name]
    if is_dataset(src_obj):
        return copy_dataset(src_obj, dst_group, name)
    if not is_group(src_obj):
        raise TypeError(f"Unsupported object type for copy: {type(src_obj)}")

    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    grp = dst_group.create_group(name)
    copy_attrs(src_obj.attrs, grp.attrs, target_backend=target_backend)
    for key in src_obj.keys():
        if key in exclude:
            continue
        child = src_obj[key]
        copy_tree(child, grp, key, exclude=exclude)
    return grp


def copy_store_contents(src_root: Any, dst_root: Any) -> None:
    target_backend = "zarr" if is_zarr_group(dst_root) else "hdf5"
    copy_attrs(src_root.attrs, dst_root.attrs, target_backend=target_backend)
    ensure_anndata_root_attrs(dst_root)
    for key in src_root.keys():
        copy_tree(src_root[key], dst_root, key)


def copy_path(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if is_zarr_path(src):
        if dst.exists():
            raise FileExistsError(f"Destination '{dst}' already exists.")
        shutil.copytree(src, dst)
        return
    shutil.copy2(src, dst)

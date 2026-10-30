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

from adata.elements import spec
from adata.util.path import is_zarr_path  # noqa: F401 - re-exported


ROOT_ENCODING_TYPE = spec.ANNDATA
ROOT_ENCODING_VERSION = spec.CURRENT_VERSION[spec.ANNDATA]


@dataclass
class Store:
    backend: str
    root: Any
    path: Path
    zarr_format: Optional[int] = None
    #: Whether to rewrite the Zarr consolidated metadata index on close.
    consolidate: bool = False

    def close(self, suppress: bool = False) -> None:
        """Finish the store: HDF5 flushes and closes, Zarr rewrites its index.

        Either can fail -- a full disk, a quota, a lost mount -- and then the
        store on disk is incomplete. That is raised, so a staged write is
        thrown away rather than renamed into place as if it were whole.
        `suppress` is for a close made while another exception is already
        propagating: that one says what went wrong first, and is kept.
        """
        writing = self.consolidate
        try:
            if self.backend == "hdf5":
                writing = bool(self.root) and self.root.mode != "r"
                self.root.close()
            elif self.consolidate and zarr is not None:
                # anndata writes a consolidated metadata index at the root.
                # New members land on disk regardless, but every reader that
                # honours the index -- anndata included -- keeps using the
                # stale copy and cannot see them, so the write looks like a
                # silent no-op.
                with warnings.catch_warnings():
                    # zarr notes that consolidated metadata is not part of
                    # the v3 spec. We write it deliberately, because anndata
                    # does and because without it our writes are invisible
                    # to readers that trust the index.
                    warnings.simplefilter("ignore")
                    zarr.consolidate_metadata(self.root.store, path=self.root.path)
                self.consolidate = False
        except Exception as error:
            if suppress:
                return
            what = "finish writing" if writing else "close"
            raise OSError(f"Could not {what} '{self.path}': {error}") from error

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(suppress=exc_type is not None)


def _require_zarr() -> None:
    if zarr is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "zarr is required for .zarr support, and is a dependency of "
            "pyadata-cli; reinstall it, or install zarr: pip install 'zarr>=3'"
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
    if mode not in ("w", "w-", "x"):
        # Anything opened for its contents, read or appended to (--inplace).
        try:
            check_hdf5_contents(root, path)
        except BaseException:
            root.close()
            raise
    if _is_writable_mode(mode):
        ensure_anndata_root_attrs(root)
    elif require_anndata:
        warn_if_missing_anndata_root_attrs(root, path=path)
    return Store(backend="hdf5", root=root, path=path)


class ExternalDataError(ValueError):
    """An HDF5 file holds a dataset or link whose data lives in another file."""


def _outside_data(obj: Any) -> Optional[str]:
    """Why reading `obj` would read another file, or None if it would not.

    A dataset with external storage reads its bytes from files named in its
    creation properties, and a virtual dataset maps datasets of other files.
    h5py follows both on any read, silently, so a copy carries that file's
    contents into the output.
    """
    if not is_hdf5_dataset(obj):
        return None
    return _dataset_id_outside(obj.id)


def refuse_outside_data(obj: Any) -> None:
    reason = _outside_data(obj)
    if reason:
        raise ExternalDataError(
            f"{obj.name} {reason}. adata reads only data held in the file itself."
        )


def _dataset_id_outside(dsid: Any) -> Optional[str]:
    plist = dsid.get_create_plist()
    if plist.get_layout() == h5py.h5d.VIRTUAL:
        return "is a virtual dataset, which maps data held in other files"
    if plist.get_external_count() > 0:
        return "keeps its data in external files"
    return None


def _references(oid: Any) -> list:
    """The object references `oid` holds: in its attributes, and as its
    values when it is a dataset of them. Other values are not read."""
    refs: list = []
    for index in range(h5py.h5a.get_num_attrs(oid)):
        attr = h5py.h5a.open(oid, index=index)
        if h5py.check_ref_dtype(attr.dtype) is None or attr.shape is None:
            continue
        values = np.empty(attr.shape, dtype=attr.dtype)
        attr.read(values)
        refs.extend(values.reshape(-1))
    if isinstance(oid, h5py.h5d.DatasetID) and h5py.check_ref_dtype(oid.dtype) is not None:
        refs.extend(np.asarray(h5py.Dataset(oid)[()]).reshape(-1))
    return refs


def _object_problem(oid: Any, within: Optional[str]) -> Optional[str]:
    """Why `oid` cannot be read, or copied as part of `within`, or None.

    A reference reaches an object no link does -- one created without a
    name, which a walk of the links never meets -- so its target is
    refused unless it has a name of its own (and then the walk checks it).
    Copied as part of `within`, a reference must stay inside it: HDF5 takes
    what it points to along, so a target elsewhere would be duplicated into
    the output as a stray copy, or, if it read another file, carry that.
    """
    if isinstance(oid, h5py.h5d.DatasetID):
        reason = _dataset_id_outside(oid)
        if reason:
            return reason
    for ref in _references(oid):
        if not ref:
            continue
        target = h5py.h5r.dereference(ref, oid)
        name = h5py.h5i.get_name(target) if target is not None else None
        if name is None:
            return "refers to an object with no name in the file"
        name = name.decode("utf-8", "replace")
        if within is not None and not (
            name == within or name.startswith(within.rstrip("/") + "/")
        ):
            return (
                f"refers to {name}, outside {within}, which a copy of "
                f"{within} cannot take along"
            )
    return None


def check_hdf5_contents(group: Any, path: Any = None, *, within: Optional[str] = None) -> None:
    """Refuse an HDF5 file, or a group of one, that reads data from elsewhere.

    Every link below `group` is looked at once, without reading any data:
    an external link opens another file, and a dataset stored externally or
    virtually reads one. None of these is something anndata writes, and
    any of them would quietly copy another file's contents into an output,
    so the input is refused before anything is read from it. The object
    references each object holds are followed too (`_object_problem`);
    `within` names the element being copied, which they must stay inside.
    """
    found: list = []
    root_problem = _object_problem(group.id, within)
    if root_problem:
        found.append(("", root_problem))

    def visit(name: bytes, info: Any) -> None:
        if found:
            return
        where = name.decode("utf-8", "replace")
        if info.type == h5py.h5l.TYPE_EXTERNAL:
            found.append((where, "is an external link into another file"))
        elif info.type == h5py.h5l.TYPE_HARD:
            reason = _object_problem(h5py.h5o.open(group.id, name), within)
            if reason:
                found.append((where, reason))
        elif info.type != h5py.h5l.TYPE_SOFT:
            found.append((where, "is a user-defined link"))

    if is_hdf5_group(group):
        group.id.links.visit(visit, info=True)
    if found:
        where, reason = found[0]
        base = getattr(group, "name", "/") or "/"
        element = f"{base.rstrip('/')}/{where}".rstrip("/") or "/"
        source = f"'{path}'" if path is not None else "The input"
        raise ExternalDataError(
            f"{source} is refused: {element} {reason}. adata reads only "
            "data held in the file itself."
        )


def check_hdf5_file(path: Path) -> None:
    """`check_hdf5_contents` for a file on disk that is not open yet."""
    with h5py.File(path, "r") as handle:
        check_hdf5_contents(handle, path)


#: Suffixes of the HDF5 files a folder source can hold (Space Ranger's
#: matrices, Xenium's), checked before a converter reads them.
HDF5_SUFFIXES = (".h5", ".hdf5", ".h5ad", ".h5mu", ".loom")


def check_hdf5_inputs(path: Path) -> None:
    """`check_hdf5_file` for a source that is not opened through
    `open_store`: an HDF5 file, or the HDF5 files in a folder of outputs."""
    import os

    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir() and not is_zarr_path(path):
        # Into linked folders too, as the converters' own globs go; each
        # real folder once, so a link loop ends.
        files, seen = [], set()
        for folder, subfolders, names in os.walk(path, followlinks=True):
            real = os.path.realpath(folder)
            if real in seen:
                subfolders[:] = []
                continue
            seen.add(real)
            files.extend(
                Path(folder, n) for n in sorted(names)
                if Path(n).suffix in HDF5_SUFFIXES and Path(folder, n).is_file()
            )
    else:
        return
    for file in files:
        if h5py.is_hdf5(file):
            check_hdf5_file(file)


def zarr_format_of(obj: Any) -> Optional[int]:
    """The Zarr spec version (2 or 3) backing `obj`, or None for HDF5."""
    return getattr(getattr(obj, "metadata", None), "zarr_format", None)


def _is_writable_mode(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "+", "x"))


def has_valid_anndata_root_attrs(root: Any) -> bool:
    enc_type = spec.decode_attr(root.attrs.get("encoding-type", None))
    enc_ver = spec.decode_attr(root.attrs.get("encoding-version", None))
    return enc_type == ROOT_ENCODING_TYPE and enc_ver == ROOT_ENCODING_VERSION


def ensure_anndata_root_attrs(root: Any) -> None:
    root.attrs["encoding-type"] = ROOT_ENCODING_TYPE
    root.attrs["encoding-version"] = ROOT_ENCODING_VERSION


def is_legacy_root(root: Any) -> bool:
    """A root as anndata 0.7 wrote it: no attributes of its own, and `obs`
    and `var` tagged as that release's dataframes."""
    if spec.encoding_type(root) is not None:
        return False
    for key in ("obs", "var"):
        frame = root[key] if key in root else None
        if not is_group(frame) or spec.encoding_of(frame) != (
            spec.DATAFRAME, spec.LEGACY_DATAFRAME_VERSION
        ):
            return False
    return True


def warn_if_missing_anndata_root_attrs(root: Any, *, path: Path) -> None:
    # anndata 0.7 never stamped the root, so its stores are valid without.
    if has_valid_anndata_root_attrs(root) or is_legacy_root(root):
        return

    enc_type = spec.decode_attr(root.attrs.get("encoding-type", None))
    enc_ver = spec.decode_attr(root.attrs.get("encoding-version", None))
    warnings.warn(
        (
            f"Store '{path}' root has missing or invalid AnnData attrs "
            f"(encoding-type={ROOT_ENCODING_TYPE!r}, encoding-version={ROOT_ENCODING_VERSION!r}). "
            f"Found encoding-type={enc_type!r}, encoding-version={enc_ver!r}."
        ),
        UserWarning,
        stacklevel=2,
    )


def _decoded(value: Any) -> Any:
    """Bytes as text and numpy values as Python's, at any depth of nesting."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_decoded(v) for v in value]
    if isinstance(value, np.ndarray):
        return _decoded(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_reference(value: Any) -> bool:
    if isinstance(value, (h5py.Reference, h5py.RegionReference)):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and h5py.check_ref_dtype(dtype) is not None


def _attrs_owner(attrs: Any) -> str:
    """The name of the element `attrs` belong to, for a message."""
    oid = getattr(attrs, "_id", None)
    try:
        return h5py.h5i.get_name(oid).decode("utf-8", "replace")
    except Exception:  # not an h5py AttributeManager
        return "an element"


def _normalize_attr_value(value: Any, target_backend: str) -> Any:
    if isinstance(value, h5py.Empty):
        # An attribute with a null dataspace: present, but holding nothing.
        return None if target_backend == "zarr" else value
    if target_backend == "zarr":
        return _decoded(value)
    if value is None:
        # JSON's null, from a Zarr attribute; HDF5 spells it as nothing.
        return h5py.Empty("f4")
    return value


def copy_attrs(src_attrs: Any, dst_attrs: Any, *, target_backend: str) -> None:
    """Copy attributes across, normalising values for the target backend.

    Written in one go on Zarr. Each `attrs[k] = v` there persists the whole
    metadata document through zarr's sync-over-async bridge, so setting them
    one at a time means a separate round trip per attribute -- slow, and the
    place a `split` over many groups was observed to wedge in CI.
    """
    normalized = {}
    for k, v in src_attrs.items():
        if _is_reference(v):
            # It names an object by its address in the source file, which
            # means nothing in another: h5py would write a dangling pointer,
            # and Zarr has no way to hold one.
            raise ValueError(
                f"Attribute {k!r} of {_attrs_owner(src_attrs)} is an HDF5 "
                "object reference, which cannot be copied out of its file."
            )
        normalized[str(k)] = _normalize_attr_value(v, target_backend)
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
        fill = _carried_fill(src)
        if fill is not None:
            kw["fillvalue"] = fill
    if (
        target_backend == "hdf5"
        and is_zarr_array(src)
        and chunks is not None
        and tuple(getattr(src, "shape", ())) != ()
        and not _is_string_src(src)
        and _zarr_is_compressed(src)
    ):
        # Zarr codecs have no HDF5 equivalent, but compression itself does.
        # Without this a compressed Zarr store came out of every command
        # uncompressed, several times its size.
        kw["compression"] = "gzip"
        kw["compression_opts"] = GZIP_LEVEL
    if target_backend == "zarr" and is_zarr_array(src):
        src_zarr_format = getattr(getattr(src, "metadata", None), "zarr_format", None)
        # Unknown means "carry nothing version-specific": guessing 3 once
        # forwarded sharding into v2 arrays, which reject it outright.
        target_format = kw_target
        # Only when both versions are known and equal can codecs travel.
        same_version = (
            target_format is not None and src_zarr_format == target_format
        )

        # Codecs only travel between stores of the same Zarr version: v2 holds
        # numcodecs objects, v3 holds its own codec classes, and neither
        # accepts the other's. Across versions the target's default is used
        # rather than a translation that fails at creation time.
        if same_version:
            # `compressors` on both versions: v2's own `compressor` is
            # deprecated, and an error reading it once meant zarr's default
            # Blosc. None, or an empty tuple, is "uncompressed", and is
            # passed on as such -- left out, it too would mean Blosc.
            compressors = tuple(src.compressors or ())
            if src_zarr_format == 3:
                kw["compressors"] = compressors
            else:
                kw["compressors"] = compressors[0] if compressors else None

            filters = src.filters
            # A v2 string array carries VLenUTF8 in `filters`; the v3 string
            # dtype encodes variable length itself and rejects it.
            if filters and not _is_string_src(src):
                kw["filters"] = filters

        shards = src.shards
        if shards is not None and target_format == 3:
            kw["shards"] = shards

        fill_value = src.fill_value
        # A record's fill is a value of its fields, which change type when
        # text is respelt for the target.
        if (
            fill_value is not None
            and not _is_string_src(src)
            and src.dtype.names is None
        ):
            kw["fill_value"] = fill_value
    return kw


def _carried_fill(src: Any) -> Any:
    """An HDF5 dataset's fill value, when a copy should keep it.

    Only a number other than zero, HDF5's own default, is worth carrying:
    every copy writes every value, so the fill is only what the dataset
    reports. A record's fill holds a value for each field, and when a field
    is text h5py crashes the process creating a dataset with it.
    """
    dtype = src.dtype
    if dtype.names is not None or dtype.kind not in "biufc":
        return None
    fill = src.fillvalue
    if fill is None or not np.any(np.asarray(fill) != 0):
        return None
    return fill


def _fill_as(fill: Any, dtype: Any) -> Any:
    """`fill` converted to `dtype`, or None when it does not survive that.

    A fill carried over from a source of another dtype -- recast to bool of
    a float matrix -- is refused by h5py unless converted, and one that the
    conversion would change (NaN for an integer) is better left to HDF5's
    default than replaced by an arbitrary value.
    """
    if dtype is None:
        return fill
    try:
        target = np.dtype(dtype)
    except TypeError:
        return None
    value = np.asarray(fill)
    if value.dtype == target:
        return fill
    if target.names is not None or target.kind not in "biufc":
        return None
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = value.astype(target)
        if not np.array_equal(converted.astype(value.dtype), value, equal_nan=True):
            return None
    return converted[()]


def fit_fill(kw: dict, dtype: Any) -> dict:
    """Creation kwargs `kw` with the fill value made fit for `dtype`
    (`_fill_as`), for a caller creating the HDF5 dataset itself."""
    if "fillvalue" not in kw:
        return kw
    kw = dict(kw)
    fill = _fill_as(kw.pop("fillvalue"), dtype)
    if fill is not None:
        kw["fillvalue"] = fill
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

    if "fillvalue" in kwargs and not is_zarr_group(parent):
        target = dtype if dtype is not None else getattr(data, "dtype", None)
        if target is None and data is not None:
            target = np.asarray(data).dtype
        kwargs = fit_fill(kwargs, target)

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
            elif not isinstance(compressors, (list, tuple)):
                # A single codec, or None for no compression at all. Dropping
                # None, as a list of several is, would bring back the default.
                kwargs["compressors"] = compressors
        if data is not None:
            return parent.create_array(name, data=data, **kwargs)
        return parent.create_array(name, shape=shape, dtype=dtype, **kwargs)
    if data is not None:
        return parent.create_dataset(name, data=data, **kwargs)
    return parent.create_dataset(name, shape=shape, dtype=dtype, **kwargs)


#: gzip level for an HDF5 copy of a compressed Zarr array: h5py's default,
#: and what Cell Ranger and most other HDF5 writers use.
GZIP_LEVEL = 4


def _zarr_is_compressed(src: Any) -> bool:
    """Does this Zarr array carry a compressor? (`compressors` covers v2 too.)"""
    return bool(src.compressors)


def fit_chunks(kw: dict, shape: Sequence[int], *, backend: str) -> dict:
    """Make a forwarded chunk shape legal for a dataset of `shape`.

    h5py rejects a chunk larger than the dataset, so a chunked source copied
    or subset below its own chunk size would otherwise fail outright. A
    zero-length dimension cannot be fitted at all; HDF5 is then left to pick
    its own chunking, which it does whenever compression asks for one. Zarr
    accepts a chunk larger than its array, so there it only has to shrink.

    Zarr additionally requires a shard to be a whole number of chunks, so a
    clamped chunk invalidates the source's shard geometry. Sharding is a
    storage-layout choice rather than data, so it is dropped and left to the
    backend rather than recomputed into something arbitrary.
    """
    chunks = kw.get("chunks")
    if not isinstance(chunks, (tuple, list)) or not chunks:
        return kw

    original = tuple(int(c) for c in chunks)
    dims = [int(n) for n in shape]
    if backend == "hdf5" and any(n == 0 for n in dims[: len(original)]):
        kw = dict(kw)
        kw.pop("chunks")
        kw.pop("shards", None)
        return kw

    clamped = tuple(
        min(c, dims[i]) if i < len(dims) and dims[i] > 0 else c
        for i, c in enumerate(original)
    )
    if clamped == original:
        return kw

    kw = dict(kw)
    kw["chunks"] = clamped
    kw.pop("shards", None)
    return kw


#: What `--compression` accepts, per target backend. lzf is HDF5-only (and
#: unreadable outside h5py); zstd and blosc have no HDF5 filter in h5py.
COMPRESSIONS = {
    "hdf5": ("none", "gzip", "lzf"),
    "zarr": ("none", "gzip", "zstd", "blosc"),
}


def _zarr_compressors(choice: str, zarr_format: Optional[int]) -> Any:
    """The `compressors` argument that gives a Zarr array `choice`."""
    if choice == "none":
        return None
    if zarr_format == 2:
        import numcodecs

        return {
            "gzip": lambda: numcodecs.GZip(level=GZIP_LEVEL),
            "zstd": lambda: numcodecs.Zstd(level=3),
            "blosc": lambda: numcodecs.Blosc(cname="lz4", clevel=5),
        }[choice]()
    from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

    return [
        {
            "gzip": lambda: GzipCodec(level=GZIP_LEVEL),
            "zstd": lambda: ZstdCodec(level=3),
            "blosc": lambda: BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),
        }[choice]()
    ]


def with_compression(
    kw: dict,
    choice: str,
    *,
    shape: Sequence[int],
    dtype: Any,
    backend: str,
    zarr_format: Optional[int] = None,
) -> dict:
    """Replace the compression in creation kwargs `kw` with `choice`.

    HDF5 strings and scalars are left uncompressed whatever is asked: a
    variable-length string compresses only its heap pointers, and a scalar
    cannot be chunked at all.
    """
    if choice not in COMPRESSIONS[backend]:
        raise ValueError(
            f"Compression {choice!r} is not available for "
            f"{'h5ad' if backend == 'hdf5' else 'zarr'}; use one of: "
            f"{', '.join(COMPRESSIONS[backend])}."
        )
    kw = dict(kw)
    for key in ("compression", "compression_opts", "compressor", "compressors"):
        kw.pop(key, None)
    if backend == "zarr":
        kw["compressors"] = _zarr_compressors(choice, zarr_format)
        return kw

    from adata.elements.strings import is_string_dtype

    if choice == "none" or tuple(shape) == () or is_string_dtype(dtype):
        return kw
    kw["compression"] = choice
    if choice == "gzip":
        kw["compression_opts"] = GZIP_LEVEL
    return kw


def _is_null(src: Any) -> bool:
    """Is `src` anndata's null: tagged `null`, an empty dataspace on HDF5 and
    a 0-d bool on Zarr? An empty dataspace without the tag is not one."""
    return spec.encoding_type(src) == spec.NULL


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


def copy_dataset(
    src: Any, dst_group: Any, name: str, *, compression: Optional[str] = None,
    skip_attrs: Sequence[str] = (),
) -> Any:
    """Copy a dataset into `dst_group`, streaming it in chunks.

    The destination dtype is resolved for the target backend rather than
    reused: an h5py variable-length string dataset reports `object`, which Zarr
    refuses to create, so copying one verbatim fails on every real store.

    `compression` replaces whatever compression the copy would otherwise
    carry (one of `COMPRESSIONS` for the target backend).
    """
    from adata.elements.strings import target_dtype

    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    refuse_outside_data(src)
    if _is_null(src):
        # anndata writes an absent element, `raw` most often, as a null: an
        # empty dataspace on HDF5 and a 0-d bool on Zarr. Neither can be read
        # and written back as a value -- `h5py.Empty` has none -- so it is
        # written afresh in the target's own spelling.
        from adata.elements.write import write_null

        ds = write_null(dst_group, name)
        copy_attrs(src.attrs, ds.attrs, target_backend=target_backend)
        return ds
    if is_hdf5_dataset(src) and src.shape is None:
        return _copy_empty(src, dst_group, name, target_backend)
    if is_hdf5_dataset(src):
        _refuse_unportable(src, target_backend)
    if src.dtype.names is not None:
        return _copy_records(
            src, dst_group, name, compression=compression, skip_attrs=skip_attrs
        )

    shape = tuple(src.shape)
    zformat = zarr_format_of(dst_group)
    kw = dataset_create_kwargs(src, target_backend=target_backend, zarr_format=zformat)
    if compression is not None:
        kw = with_compression(
            kw, compression, shape=shape, dtype=src.dtype,
            backend=target_backend, zarr_format=zformat,
        )
    ds = create_dataset(
        dst_group,
        name,
        shape=shape,
        dtype=target_dtype(src.dtype, target_backend, zformat),
        **fit_chunks(kw, shape, backend=target_backend),
    )
    attrs = src.attrs if not skip_attrs else {
        k: v for k, v in src.attrs.items() if k not in skip_attrs
    }
    copy_attrs(attrs, ds.attrs, target_backend=target_backend)

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


def _copy_empty(src: Any, dst_group: Any, name: str, target_backend: str) -> Any:
    """An HDF5 empty dataspace that is not tagged `null`.

    anndata never writes one; other HDF5 writers leave them. Tagging it
    `null` on the way, as once happened, made a store anndata 0.8 to 0.11
    read into one they refuse. HDF5 copies it as it is; Zarr has no empty
    dataspace, and any value written in its place would be invented, so it
    is left out, and said so.
    """
    if target_backend == "hdf5":
        ds = dst_group.create_dataset(name, data=h5py.Empty(src.dtype))
        copy_attrs(src.attrs, ds.attrs, target_backend=target_backend)
        return ds
    warnings.warn(
        f"Not copied: {src.name}, an HDF5 empty dataspace with no encoding, "
        "which Zarr cannot hold.",
        UserWarning,
        stacklevel=3,
    )
    return None


def _refuse_unportable(src: Any, target_backend: str) -> None:
    """Refuse an HDF5 dataset a copy element by element cannot take, before
    anything is written.

    Its dtype is an `object` to numpy, which this layer once took for text:
    variable-length numbers came out as the strings '[1 2 3]'. References
    name objects by their address in this file, which means nothing in
    another; HDF5's own copy of a whole element (`copy_tree`) repoints those
    that stay inside it, and that is the only way one is copied. Zarr holds
    neither; a record's fields are checked one by one.
    """
    dtype = src.dtype
    fields = [dtype.fields[f][0] for f in dtype.names] if dtype.names else [dtype]
    for sub in fields:
        if h5py.check_ref_dtype(sub) is not None:
            raise ValueError(
                f"{src.name} holds HDF5 object references, which cannot be "
                "copied out of its file on their own."
            )
        base = h5py.check_vlen_dtype(sub)
        if target_backend == "zarr" and base is not None and base not in (str, bytes):
            raise ValueError(
                f"{src.name} holds variable-length sequences of {np.dtype(base)}, "
                "which Zarr cannot store."
            )


def _copy_records(
    src: Any, dst_group: Any, name: str, *, compression: Optional[str] = None,
    skip_attrs: Sequence[str] = (),
) -> Any:
    """A record array (`rec-array`, scanpy's `rank_genes_groups`), field by
    field.

    Each backend spells a text field its own way and refuses the other's:
    h5py variable-length str (an `object` to numpy, which Zarr calls
    ambiguous), Zarr fixed-width `<U` (which h5py has no conversion for).
    Text fields are respelt as anndata writes them, and the rest kept.
    """
    from adata.elements.strings import as_records, record_dtype

    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    zformat = zarr_format_of(dst_group)
    shape = tuple(src.shape)
    # Rows per read: a text field's real width hides behind a pointer.
    row = sum(
        VLEN_ELEMENT_BYTES if sub.hasobject else sub.itemsize
        for sub, *_ in src.dtype.fields.values()
    )
    step = _chunk_step(shape, getattr(src, "chunks", None), np.dtype(f"V{max(1, row)}"))
    blocks = [(0, 0)] if shape == () else [
        (start, min(start + step, shape[0])) for start in range(0, shape[0], step)
    ]

    def read(block: tuple) -> np.ndarray:
        return np.asarray(src[()] if shape == () else src[block[0]:block[1]])

    widths: dict = {}
    if target_backend == "zarr":
        # A fixed width has to be known before the array exists: one pass
        # to measure the variable-length fields, a second to copy.
        from adata.elements.strings import text_field_widths

        for block in blocks:
            for field, width in text_field_widths(read(block)).items():
                widths[field] = max(width, widths.get(field, 1))
    dtype = record_dtype(src.dtype, target_backend, widths)

    kw = dataset_create_kwargs(src, target_backend=target_backend, zarr_format=zformat)
    if compression is not None:
        kw = with_compression(
            kw, compression, shape=shape, dtype=dtype,
            backend=target_backend, zarr_format=zformat,
        )
    with warnings.catch_warnings():
        # zarr notes, on every write, that Zarr v3 has no specification for a
        # record dtype yet. anndata writes rec-arrays to v3 just the same.
        warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification")
        ds = create_dataset(
            dst_group, name, shape=shape, dtype=dtype,
            **fit_chunks(kw, shape, backend=target_backend),
        )
        attrs = src.attrs if not skip_attrs else {
            k: v for k, v in src.attrs.items() if k not in skip_attrs
        }
        copy_attrs(attrs, ds.attrs, target_backend=target_backend)
        for block in blocks:
            values = as_records(read(block), dtype)
            if shape == ():
                ds[()] = values
            else:
                ds[block[0]:block[1]] = values
    return ds


def is_legacy_dataframe(obj: Any) -> bool:
    """A dataframe as anndata 0.7 wrote it: each categorical a dataset of
    codes whose `categories` attribute is an object reference into
    `__categories`."""
    return is_group(obj) and spec.encoding_of(obj) == (
        spec.DATAFRAME, spec.LEGACY_DATAFRAME_VERSION
    )


def is_legacy_categorical(obj: Any) -> bool:
    return is_dataset(obj) and "categories" in obj.attrs


def write_legacy_categorical(
    frame: Any, name: str, codes_src: Any, dst_group: Any, codes: Optional[np.ndarray] = None,
    *, compression: Optional[str] = None, kept: Optional[np.ndarray] = None,
) -> Any:
    """A legacy categorical column of `frame` as a current `categorical`
    group: its codes (all of them, or `codes`, already taken) and the labels
    its reference points to, copied beside them (only those at `kept`,
    when given, `codes` being renumbered onto them).

    An object reference cannot leave its file -- h5py's copy into another
    file nulls it, and Zarr cannot hold one -- so the column is rewritten in
    the layout anndata has written since 0.8, which needs none.
    """
    from adata.elements.read import categories_dataset, is_ordered

    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    group = dst_group.create_group(name)
    copy_attrs({
        "encoding-type": spec.CATEGORICAL,
        "encoding-version": spec.CURRENT_VERSION[spec.CATEGORICAL],
        "ordered": is_ordered(codes_src, frame),
    }, group.attrs, target_backend=target_backend)
    categories = categories_dataset(codes_src, frame)
    if kept is None:
        copy_dataset(categories, group, "categories", compression=compression)
    else:
        values = np.asarray(categories[...])[kept]
        kw = dataset_create_kwargs(categories, target_backend=target_backend, dst_parent=group)
        create_dataset(group, "categories", data=values, **fit_chunks(kw, values.shape, backend=target_backend))
    if codes is None:
        copy_dataset(codes_src, group, "codes", compression=compression, skip_attrs=("categories", "ordered"))
    else:
        codes = np.asarray(codes)
        kw = dataset_create_kwargs(codes_src, target_backend=target_backend, dst_parent=group)
        if compression is not None:
            kw = with_compression(kw, compression, shape=codes.shape, dtype=codes.dtype,
                                  backend=target_backend, zarr_format=zarr_format_of(group))
        create_dataset(group, "codes", data=codes, **fit_chunks(kw, codes.shape, backend=target_backend))
    return group


def copy_legacy_dataframe(
    src: Any, dst_group: Any, name: str, *, compression: Optional[str] = None
) -> Any:
    """A 0.1.0 dataframe copied as a current (0.2.0) one: its categoricals
    rewritten as `categorical` groups (`write_legacy_categorical`), the
    `__categories` they pointed into dropped, every other column as it is."""
    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    grp = dst_group.create_group(name)
    copy_attrs(src.attrs, grp.attrs, target_backend=target_backend)
    grp.attrs["encoding-version"] = spec.CURRENT_VERSION[spec.DATAFRAME]
    for key in src.keys():
        child = src[key]
        if key == "__categories":
            continue
        if is_legacy_categorical(child):
            write_legacy_categorical(src, key, child, grp, compression=compression)
        else:
            copy_tree(child, grp, key, compression=compression)
    return grp


def copy_tree(
    src_obj: Any,
    dst_group: Any,
    name: str,
    *,
    exclude: Iterable[str] = (),
    compression: Optional[str] = None,
) -> Any:
    if is_legacy_dataframe(src_obj) and not exclude:
        return copy_legacy_dataframe(src_obj, dst_group, name, compression=compression)
    if is_hdf5_group(dst_group) and (
        is_hdf5_group(src_obj) or is_hdf5_dataset(src_obj)
    ):
        # HDF5's own copy keeps every dataset exactly as it was, which is
        # what is wanted unless the caller asked for something else.
        if not exclude and compression is None:
            check_hdf5_contents(src_obj, within=src_obj.name)
            # References below it -- anndata 0.7's categoricals, in a frame
            # nested in raw, obsm or uns -- would otherwise come out null:
            # with expand_refs HDF5 repoints them at their copies. A target
            # copied before its place in the tree is reached also gets a
            # second link, `~obj_pointed_by_<n>` at the root; every target is
            # inside what was copied, so that link is only a duplicate name.
            root = dst_group.file
            before = set(root.keys())
            dst_group.copy(src_obj, dst_group, name, expand_refs=True)
            for key in set(root.keys()) - before:
                if key.startswith("~obj_pointed_by_") and h5py.h5o.get_info(root[key].id).rc > 1:
                    del root[key]
            return dst_group[name]
    if is_dataset(src_obj):
        return copy_dataset(src_obj, dst_group, name, compression=compression)
    if not is_group(src_obj):
        raise TypeError(f"Unsupported object type for copy: {type(src_obj)}")

    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    grp = dst_group.create_group(name)
    copy_attrs(src_obj.attrs, grp.attrs, target_backend=target_backend)
    for key in src_obj.keys():
        if key in exclude:
            continue
        child = src_obj[key]
        copy_tree(child, grp, key, exclude=exclude, compression=compression)
    return grp


def copy_store_contents(
    src_root: Any, dst_root: Any, *, compression: Optional[str] = None
) -> None:
    target_backend = "zarr" if is_zarr_group(dst_root) else "hdf5"
    copy_attrs(src_root.attrs, dst_root.attrs, target_backend=target_backend)
    ensure_anndata_root_attrs(dst_root)
    for key in src_root.keys():
        copy_tree(src_root[key], dst_root, key, compression=compression)


def copy_path(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if is_zarr_path(src):
        if dst.exists():
            raise FileExistsError(f"Destination '{dst}' already exists.")
        shutil.copytree(src, dst)
        return
    shutil.copy2(src, dst)

"""Subset operations for .h5ad and .zarr stores."""

from __future__ import annotations

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


def _take_rows(obj: Any, indices: Optional[np.ndarray]) -> Any:
    """Read the selected rows of a dataset, handling both backends' indexing."""
    if indices is None:
        return obj[...]
    if is_zarr_array(obj):
        if obj.ndim == 1:
            return obj.oindex[indices]
        return obj.oindex[(indices,) + (slice(None),) * (obj.ndim - 1)]
    return obj[indices, ...]


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


def subset_dense_matrix(
    src: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int = 1024,
) -> None:
    if src.ndim != 2:
        copy_tree(src, dst_parent, name)
        return

    n_obs, n_var = src.shape
    out_obs = len(obs_idx) if obs_idx is not None else n_obs
    out_var = len(var_idx) if var_idx is not None else n_var

    target_backend = _target_backend(dst_parent)
    kw = dataset_create_kwargs(
        src, target_backend=target_backend, dst_parent=dst_parent
    )
    kw = _clamp_chunks(kw, out_obs, out_var)

    dst = create_dataset(
        dst_parent,
        name,
        shape=(out_obs, out_var),
        dtype=src.dtype,
        **kw,
    )
    copy_attrs(src.attrs, dst.attrs, target_backend=_target_backend(dst_parent))

    for out_start in range(0, out_obs, chunk_rows):
        out_end = min(out_start + chunk_rows, out_obs)

        if obs_idx is None:
            block = src[out_start:out_end, :]
        else:
            rows = obs_idx[out_start:out_end]
            block = src[rows, :]

        if var_idx is not None:
            block = block[:, var_idx]

        dst[out_start:out_end, :] = block


def _minor_remap(keep: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
    """Build a lookup from old minor index to new, with -1 for dropped entries.

    A dense lookup table costs one int32 per column of the source, which is
    negligible beside the matrix itself and turns the remap into a single
    vectorised gather rather than a per-entry dict lookup.
    """
    if keep is None:
        return None
    remap = np.full(size, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep), dtype=np.int64)
    return remap


def subset_sparse_matrix_group(
    src: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_major: int = 4096,
) -> None:
    """Subset a CSR/CSC matrix, streaming it a block of major axis at a time.

    Only the slice of `data`/`indices` spanned by the current block is read, so
    peak memory is set by `chunk_major` rather than by the matrix. The output
    datasets are grown as each block is appended, since the final nnz is not
    known until the pass completes.
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

    if enc == spec.CSR_MATRIX:
        major_idx, minor_keep, n_minor = obs_idx, var_idx, n_cols
        out_rows = len(obs_idx) if obs_idx is not None else n_rows
        out_cols = len(var_idx) if var_idx is not None else n_cols
    else:
        major_idx, minor_keep, n_minor = var_idx, obs_idx, n_rows
        out_rows = len(obs_idx) if obs_idx is not None else n_rows
        out_cols = len(var_idx) if var_idx is not None else n_cols

    n_major = n_rows if enc == spec.CSR_MATRIX else n_cols
    majors = major_idx if major_idx is not None else np.arange(n_major, dtype=np.int64)
    remap = _minor_remap(minor_keep, n_minor)

    group = dst_parent.create_group(name)
    copy_attrs(src.attrs, group.attrs, target_backend=_target_backend(dst_parent))
    spec.set_encoding(group, enc)
    set_shape_attr(group, (out_rows, out_cols))

    out_data = _growable(group, "data", data_ds.dtype)
    out_indices = _growable(group, "indices", np.int64)
    out_indptr = [0]
    nnz = 0

    for block_start in range(0, len(majors), chunk_major):
        block = majors[block_start : block_start + chunk_major]
        # One contiguous read covers the whole block's entries.
        lo, hi = int(indptr[block].min()), int(indptr[block + 1].max())
        if hi > lo:
            block_indices = np.asarray(indices_ds[lo:hi], dtype=np.int64)
            block_data = np.asarray(data_ds[lo:hi])
        else:
            block_indices = np.empty(0, dtype=np.int64)
            block_data = np.empty(0, dtype=data_ds.dtype)

        kept_indices: List[np.ndarray] = []
        kept_data: List[np.ndarray] = []
        for m in block:
            sl = slice(int(indptr[m]) - lo, int(indptr[m + 1]) - lo)
            minor = block_indices[sl]
            values = block_data[sl]
            if remap is not None:
                mapped = remap[minor]
                keep = mapped >= 0
                minor, values = mapped[keep], values[keep]
            kept_indices.append(minor)
            kept_data.append(values)
            nnz += len(minor)
            out_indptr.append(nnz)

        if kept_indices:
            _append(out_indices, np.concatenate(kept_indices))
            _append(out_data, np.concatenate(kept_data))

    create_dataset(
        group, "indptr", data=np.asarray(out_indptr, dtype=np.int64)
    )


def _growable(group: Any, name: str, dtype: Any) -> Any:
    """Create an empty 1-D dataset that can be extended as blocks arrive."""
    if is_zarr_group(group):
        return group.create_array(name, shape=(0,), dtype=dtype, chunks=(65536,))
    return group.create_dataset(
        name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(65536,)
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
    if is_dataset(obj):
        subset_dense_matrix(
            obj, dst_parent, name, obs_idx, var_idx, chunk_rows=chunk_rows
        )
        return

    if is_group(obj):
        enc = _decode_attr(obj.attrs.get("encoding-type", b""))
        if enc in spec.SPARSE_TYPES:
            subset_sparse_matrix_group(obj, dst_parent, name, obs_idx, var_idx)
            return
        if enc == spec.DATAFRAME:
            # obsm/varm may hold a dataframe; it is row-aligned like obs/var.
            subset_axis_group(obj, dst_parent.create_group(name), obs_idx)
            return
        raise ValueError(f"Unsupported {entry_label} encoding type: {enc}")

    raise ValueError(f"Unsupported {entry_label} object type")


HANDLED_KEYS = frozenset(
    {"obs", "var", "X", "layers", "obsm", "varm", "obsp", "varp", "uns", "raw"}
)


def subset_raw_group(
    src_raw: Any,
    dst: Any,
    obs_idx: Optional[np.ndarray],
    var_keep: Optional[Set[str]],
    *,
    chunk_rows: int,
    console: Console,
) -> None:
    """Subset a `raw/` group, which carries its own var axis.

    `raw` typically holds more genes than the main object, so its var names are
    matched independently rather than reusing the outer var indices -- using
    those would select the wrong columns entirely.
    """
    raw_dst = dst.create_group("raw")
    copy_attrs(src_raw.attrs, raw_dst.attrs, target_backend=_target_backend(dst))
    spec.set_encoding(raw_dst, spec.RAW)

    raw_var_idx: Optional[np.ndarray] = None
    if var_keep is not None and "var" in src_raw:
        raw_var_names, _ = resolve_index(src_raw["var"], "var")
        raw_var_idx, missing = indices_from_name_set(raw_var_names, var_keep)
        console.print(
            f"[green]Selected {len(raw_var_idx)} raw/var "
            f"(of {element_len(raw_var_names)})[/]"
        )
        if missing:
            console.print(
                f"[yellow]Warning: {len(missing)} var names not found in raw/var[/]"
            )

    if "var" in src_raw:
        subset_axis_group(src_raw["var"], raw_dst.create_group("var"), raw_var_idx)

    if "X" in src_raw:
        subset_matrix_entry(
            src_raw["X"],
            raw_dst,
            "X",
            obs_idx,
            raw_var_idx,
            chunk_rows=chunk_rows,
            entry_label="raw/X",
        )

    if "varm" in src_raw:
        varm_dst = _ensure_group(raw_dst, "varm")
        for key in src_raw["varm"].keys():
            subset_matrix_entry(
                src_raw["varm"][key],
                varm_dst,
                key,
                raw_var_idx,
                None,
                chunk_rows=chunk_rows,
                entry_label=f"raw/varm:{key}",
            )

    for key in src_raw.keys():
        if key not in ("X", "var", "varm"):
            copy_tree(src_raw[key], raw_dst, key)


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

    with console.status("[magenta]Opening files...[/]"):
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

            tasks: List[str] = []
            if "obs" in src:
                tasks.append("obs")
            if "var" in src:
                tasks.append("var")
            if "X" in src:
                tasks.append("X")
            if "layers" in src:
                tasks.extend([f"layer:{k}" for k in src["layers"].keys()])
            if "obsm" in src:
                tasks.extend([f"obsm:{k}" for k in src["obsm"].keys()])
            if "varm" in src:
                tasks.extend([f"varm:{k}" for k in src["varm"].keys()])
            if "obsp" in src:
                tasks.extend([f"obsp:{k}" for k in src["obsp"].keys()])
            if "varp" in src:
                tasks.extend([f"varp:{k}" for k in src["varp"].keys()])
            if "uns" in src:
                tasks.append("uns")
            # anndata writes a placeholder `raw` even when there is none, and
            # on Zarr that placeholder is an array rather than a group.
            if "raw" in src and is_group(src["raw"]):
                tasks.append("raw")
            elif "raw" in src:
                tasks.append("copy:raw")

            passthrough = [
                k for k in src.keys() if k not in HANDLED_KEYS
            ]
            if passthrough:
                console.print(
                    "[yellow]Copying unrecognised top-level "
                    f"{'keys' if len(passthrough) > 1 else 'key'} verbatim: "
                    f"{', '.join(sorted(passthrough))}[/]"
                )
                tasks.extend(f"copy:{k}" for k in passthrough)

            with Progress(
                SpinnerColumn(finished_text="[green]✓[/]"),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False,
            ) as progress:
                for task in tasks:
                    task_id = progress.add_task(
                        f"[cyan]Subsetting {task}...[/]", total=None
                    )
                    if task == "obs":
                        obs_dst = dst.create_group("obs")
                        subset_axis_group(src["obs"], obs_dst, obs_idx)
                    elif task == "var":
                        var_dst = dst.create_group("var")
                        subset_axis_group(src["var"], var_dst, var_idx)
                    elif task == "X":
                        X = src["X"]
                        if is_dataset(X):
                            subset_dense_matrix(
                                X, dst, "X", obs_idx, var_idx, chunk_rows=chunk_rows
                            )
                        elif is_group(X):
                            subset_sparse_matrix_group(X, dst, "X", obs_idx, var_idx)
                        else:
                            copy_tree(X, dst, "X")
                    elif task.startswith("layer:"):
                        key = task.split(":", 1)[1]
                        layer_src = src["layers"][key]
                        layers_dst = _ensure_group(dst, "layers")
                        subset_matrix_entry(
                            layer_src,
                            layers_dst,
                            key,
                            obs_idx,
                            var_idx,
                            chunk_rows=chunk_rows,
                            entry_label=f"layer:{key}",
                        )
                    elif task.startswith("obsm:"):
                        key = task.split(":", 1)[1]
                        obsm_dst = _ensure_group(dst, "obsm")
                        obsm_obj = src["obsm"][key]
                        subset_matrix_entry(
                            obsm_obj,
                            obsm_dst,
                            key,
                            obs_idx,
                            None,
                            chunk_rows=chunk_rows,
                            entry_label=f"obsm:{key}",
                        )
                    elif task.startswith("varm:"):
                        key = task.split(":", 1)[1]
                        varm_dst = _ensure_group(dst, "varm")
                        varm_obj = src["varm"][key]
                        subset_matrix_entry(
                            varm_obj,
                            varm_dst,
                            key,
                            var_idx,
                            None,
                            chunk_rows=chunk_rows,
                            entry_label=f"varm:{key}",
                        )
                    elif task.startswith("obsp:"):
                        key = task.split(":", 1)[1]
                        obsp_dst = _ensure_group(dst, "obsp")
                        obsp_obj = src["obsp"][key]
                        subset_matrix_entry(
                            obsp_obj,
                            obsp_dst,
                            key,
                            obs_idx,
                            obs_idx,
                            chunk_rows=chunk_rows,
                            entry_label=f"obsp:{key}",
                        )
                    elif task.startswith("varp:"):
                        key = task.split(":", 1)[1]
                        varp_dst = _ensure_group(dst, "varp")
                        varp_obj = src["varp"][key]
                        subset_matrix_entry(
                            varp_obj,
                            varp_dst,
                            key,
                            var_idx,
                            var_idx,
                            chunk_rows=chunk_rows,
                            entry_label=f"varp:{key}",
                        )
                    elif task == "uns":
                        copy_tree(src["uns"], dst, "uns")
                    elif task == "raw":
                        subset_raw_group(
                            src["raw"],
                            dst,
                            obs_idx,
                            var_keep,
                            chunk_rows=chunk_rows,
                            console=console,
                        )
                    elif task.startswith("copy:"):
                        key = task.split(":", 1)[1]
                        copy_tree(src[key], dst, key)
                    progress.update(
                        task_id,
                        description=f"[green]Subsetting {task}[/]",
                        completed=1,
                        total=1,
                    )

            _ensure_optional_anndata_groups(dst)

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

"""MuData (`.h5mu`, and MuData Zarr stores), read and written.

A MuData holds one AnnData per modality under `mod/<name>`, each a complete
AnnData group in the element spec, beside global elements that span them
all: `obs`, `var`, `obsm`, `varm`, `obsp`, `varp` and `uns`, written with
anndata's own element writers. `obsmap/<mod>` and `varmap/<mod>` place the
modality's rows among the global ones: one uint32 per global row, the
modality's row plus one, 0 where the modality lacks it. `obsm/<mod>` and
`varm/<mod>` are the same as masks. The root says `encoding-type: mudata`;
an `.h5mu` also opens with a 512-byte HDF5 userblock that starts `MuData`.

Reading writes one AnnData per modality (or the one `--mod` names): the
modality is copied as it is, and the global parts go onto its rows through
the maps -- a joint clustering in global obs, a WNN embedding in obsm, the
joint graph in obsp. Where a global name is one the modality already has,
the modality's own is kept. Global columns pulled from a modality are named
`<mod>:col`; this modality's are added under `col` when it lacks one, the
other modalities' are left out.

Writing splits var by its feature types, as muon reads a 10x Multiome
matrix: Gene Expression becomes `rna`, Peaks `atac`, Antibody Capture
`prot`. Each modality keeps X, the layers, var, varm, varp and raw for its
features; obs, obsm, obsp and uns are written once, globally, since they
describe the cells. The matrices are read once for every modality, as
`split` reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import type_counts, write_index_frame
from adata.core.csr_build import CsrAppender, count_keys, write_runs
from adata.core.select import read_names
from adata.core.subset import Target, _take_rows, _write_targets, aligned_rows
from adata.elements import spec
from adata.elements.read import dataframe_columns, element_len, read_str_all, resolve_index
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    set_shape_attr,
    write_dense,
    write_mapping,
)
from adata.storage import (
    COMPRESSIONS,
    copy_attrs,
    copy_tree,
    create_dataset,
    dataset_create_kwargs,
    ensure_anndata_root_attrs,
    fit_chunks,
    is_dataset,
    is_group,
    is_legacy_categorical,
    is_legacy_dataframe,
    is_zarr_group,
    is_zarr_path,
    open_store,
    with_compression,
    write_legacy_categorical,
    zarr_format_of,
)
from adata.util.names import claim, sanitize_label, storable
from adata.util.path import same_path

#: The root's encoding, as mudata writes it (and reads it, in any case).
MUDATA = "mudata"
ENCODING_VERSION = "0.1.0"

#: The HDF5 userblock mudata reserves and the magic it starts with.
USERBLOCK = 512
MAGIC = b"MuData"

#: muon's names for 10x feature types; any other type is named after itself.
MODALITY_NAMES = {"Gene Expression": "rna", "Peaks": "atac", "Antibody Capture": "prot"}

#: Rows read at a time when global rows are taken onto a modality's.
ROW_BLOCK = 1 << 14

#: Bytes read at a time from a dense graph, whose rows span the global axis.
SLAB_BYTES = 1 << 26

_AXES = (("obs", "obsm", "obsp"), ("var", "varm", "varp"))


def _attr(obj: Any, name: str) -> Any:
    return spec.decode_attr(obj.attrs.get(name, None))


def is_mudata_root(root: Any) -> bool:
    """A group laid out as a MuData: modalities under `mod`, and the root's
    encoding saying so."""
    return (
        "mod" in root and is_group(root["mod"])
        and str(_attr(root, "encoding-type") or "").lower() == MUDATA
    )


def sniff(path: Path) -> bool:
    path = Path(path)
    if path.is_dir():
        if not is_zarr_path(path):
            return False
        try:
            with open_store(path, "r", require_anndata=False) as store:
                return is_mudata_root(store.root)
        except Exception:  # not a store this can open: not this format
            return False
    if not path.is_file() or not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as handle:
        if is_mudata_root(handle):
            return True
        # An older writer's root attributes may be missing; muon's userblock
        # and its modalities are not.
        if "mod" in handle and isinstance(handle["mod"], h5py.Group) and handle.userblock_size:
            with open(path, "rb") as raw:
                return raw.read(len(MAGIC)) == MAGIC
    return False


def modality_names(root: Any) -> List[str]:
    """The modalities in their recorded order (`mod-order`), else as stored."""
    mods = root["mod"]
    present = [k for k in mods.keys() if is_group(mods[k])]
    order = _attr(mods, "mod-order")
    if order is not None:
        listed = [str(m) for m in np.atleast_1d(order)]
        if sorted(listed) == sorted(present):
            return listed
    return present


# ---------------------------------------------------------------------------
# Reading


@dataclass
class _Axis:
    """What comes from one global axis onto a modality's."""

    name: str  # "obs" or "var"
    n_global: int
    #: The global row behind each of the modality's rows.
    rows: np.ndarray
    #: (global name, name in the modality) per element, for the frame and
    #: its two mappings.
    columns: List[Tuple[str, str]] = field(default_factory=list)
    arrays: List[Tuple[str, str]] = field(default_factory=list)
    graphs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class _Modality:
    name: str
    output: Path
    axes: List[_Axis]
    uns: List[str]


@dataclass
class _Import:
    mods: List[_Modality]
    zarr_format: Optional[int]
    compression: Optional[str]


def _global_rows(root: Any, mod: str, axis: str, warnings: List[str]) -> Tuple[np.ndarray, int]:
    """The global row of each of the modality's rows, and the global length.

    From `obsmap/<mod>` (`varmap`) when there is one; by name otherwise, as
    a file whose writer left the maps out still has both indexes.
    """
    n_mod = element_len(resolve_index(root["mod"][mod][axis], axis)[0])
    n_global = element_len(resolve_index(root[axis], axis)[0]) if axis in root else 0
    maps = _child(root, f"{axis}map")
    if maps is not None and mod in maps:
        where = np.asarray(maps[mod][...]).astype(np.int64).reshape(-1)
        if where.size != n_global:
            raise ValueError(
                f"{axis}map/{mod} has {where.size:,} entries for {n_global:,} global {axis}."
            )
        present = np.flatnonzero(where)
        positions = where[present] - 1
        if present.size != n_mod or positions.max(initial=-1) >= n_mod or (
            np.bincount(positions, minlength=n_mod) != 1
        ).any():
            raise ValueError(
                f"{axis}map/{mod} does not place each of the modality's {n_mod:,} "
                f"{axis} once among the global ones."
            )
        rows = np.empty(n_mod, dtype=np.int64)
        rows[positions] = present
        return rows, n_global
    if not n_global:
        return np.arange(n_mod, dtype=np.int64), n_mod
    names = read_names(root, axis)
    lookup: Dict[str, int] = {}
    for i, name in enumerate(names):
        lookup.setdefault(name, i)
    if len(lookup) != len(names):
        raise ValueError(
            f"There is no {axis}map/{mod}, and the global {axis} names repeat, "
            f"so the modality's {axis} cannot be matched to them."
        )
    own = read_names(root["mod"][mod], axis)
    if len(set(own)) != len(own):
        raise ValueError(
            f"There is no {axis}map/{mod}, and {mod}'s {axis} names repeat, so its "
            f"{axis} cannot be told apart among the global ones."
        )
    missing = [n for n in own if n not in lookup]
    if missing:
        raise ValueError(
            f"There is no {axis}map/{mod}, and {len(missing):,} of its {axis} "
            f"names are not global ones (first: {missing[0]!r})."
        )
    warnings.append(f"No {axis}map/{mod}: its {axis} are matched to the global ones by name.")
    return np.fromiter((lookup[n] for n in own), dtype=np.int64, count=len(own)), n_global


def _carried(
    global_names: Sequence[str], own: Sequence[str], mod: str, mods: Sequence[str],
    where: str, warnings: List[str], skipped: Dict[str, int], *, warn: bool = True,
) -> List[Tuple[str, str]]:
    """Which global names come onto the modality, and under which name.

    `warn=False` for the columns of an axis the modalities do not share:
    there a global column the modality also has is one muon or mudata
    copied from the modalities (`join_common`), not a value of its own.
    """
    taken = set(own)
    out: List[Tuple[str, str]] = []
    for name in global_names:
        prefix, _, rest = name.partition(":")
        target = name
        if rest and prefix in mods:
            if prefix != mod:
                skipped[where] = skipped.get(where, 0) + 1
                continue
            target = rest  # this modality's own, pulled into the global frame
            if target in taken:
                continue
        elif name in taken:
            if not warn:
                continue
            warnings.append(
                f"Not written: global {where}/{name} -- {mod} has its own {where}/{name}, "
                "which is kept."
            )
            continue
        taken.add(target)
        out.append((name, target))
    return out


def _keys(group: Any) -> List[str]:
    return [str(k) for k in group.keys()] if group is not None else []


def _child(root: Any, name: str) -> Any:
    return root[name] if name in root else None


def _shared(root: Any) -> Tuple[str, ...]:
    """The axes the modalities share, from the root's `axis`: 0 the cells,
    1 the features, -1 both."""
    axis = _attr(root, "axis")
    return {1: ("var",), -1: ("obs", "var")}.get(int(axis) if axis is not None else 0, ("obs",))


def _plan_axis(
    root: Any, mod: str, mods: Sequence[str], axis: str, mapping: str, graphs: str,
    warnings: List[str], skipped: Dict[str, int],
) -> _Axis:
    rows, n_global = _global_rows(root, mod, axis, warnings)
    found = _Axis(axis, n_global, rows)
    if axis not in root or not n_global:
        return found
    group = root["mod"][mod]
    frame = root[axis]
    own_frame = group[axis]
    found.columns = _carried(
        dataframe_columns(frame, resolve_index(frame, axis)[1]),
        dataframe_columns(own_frame, resolve_index(own_frame, axis)[1]),
        mod, mods, axis, warnings, skipped, warn=axis in _shared(root),
    )
    # The masks, one per modality, say which global rows each has: they
    # describe the MuData, not the cells.
    arrays = [k for k in _keys(_child(root, mapping)) if k not in mods]
    found.arrays = _carried(arrays, _keys(_child(group, mapping)), mod, mods, mapping, warnings, skipped)
    found.graphs = _carried(
        _keys(_child(root, graphs)), _keys(_child(group, graphs)), mod, mods, graphs, warnings, skipped,
    )
    for key, _ in found.graphs:
        obj = root[graphs][key]
        shape = _sparse_shape(obj) if spec.encoding_type(obj) in spec.SPARSE_TYPES else tuple(obj.shape)
        if tuple(shape) != (n_global, n_global):
            raise ValueError(f"Global {graphs}/{key} is {shape}, not {n_global:,} x {n_global:,}.")
    for key, _ in found.arrays:
        obj = root[mapping][key]
        length = element_len(resolve_index(obj)[0]) if spec.encoding_type(obj) == spec.DATAFRAME else (
            _sparse_shape(obj)[0] if spec.encoding_type(obj) in spec.SPARSE_TYPES else element_len(obj)
        )
        if length != n_global:
            raise ValueError(f"Global {mapping}/{key} has {length:,} rows for {n_global:,} {axis}.")
    return found


def _sparse_shape(group: Any) -> Tuple[int, int]:
    return tuple(int(v) for v in np.asarray(spec.decode_attr(group.attrs["shape"])))  # type: ignore[return-value]


class MudataImport(Converter):
    options = frozenset({"--mod", "--compression"})

    def plan(
        self, source: Path, output: Path, src_format: str, dst_format: str, options: Options,
    ) -> ConversionPlan:
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )
        warnings: List[str] = []
        with open_store(source, "r", require_anndata=False) as store:
            root = store.root
            if not is_mudata_root(root) and "mod" not in root:
                raise ValueError(f"'{source.name}' holds no modalities (no mod group).")
            mods = modality_names(root)
            if not mods:
                raise ValueError(f"'{source.name}' holds no modalities.")
            if options.mod is not None and options.mod not in mods:
                raise ValueError(f"No modality {options.mod!r}; this file has: {', '.join(mods)}.")
            wanted = [options.mod] if options.mod is not None else mods
            # Claimed in turn against every stem so far: sanitising can make
            # `a b`, `a_b` and `a_b_1` meet, and each needs its own file.
            taken_stems: set = set()
            stems = {mod: claim(sanitize_label(mod), taken_stems) for mod in wanted}
            skipped: Dict[str, int] = {}
            planned: List[_Modality] = []
            for mod in wanted:
                path = output if len(wanted) == 1 else output.with_name(
                    f"{output.stem}.{stems[mod]}{output.suffix}"
                )
                if same_path(source, path):
                    raise ValueError(f"The output {path} is the input.")
                axes = [_plan_axis(root, mod, mods, *names, warnings, skipped) for names in _AXES]
                own_uns = _keys(_child(root["mod"][mod], "uns"))
                uns = []
                for key in _keys(_child(root, "uns")):
                    if key in own_uns:
                        warnings.append(f"Not written: global uns/{key} -- {mod} has its own, which is kept.")
                    else:
                        uns.append(key)
                planned.append(_Modality(mod, path, axes, uns))
            source_zarr_format = store.zarr_format
            axis = _attr(root, "axis")
        if skipped:
            warnings.append(
                "Not written: "
                + ", ".join(f"{n} global {where} {'entry' if n == 1 else 'entries'}" for where, n in skipped.items())
                + " pulled from other modalities (named <mod>:...)."
            )
        zarr_format = (options.zarr_format or source_zarr_format or 3) if dst_format == "zarr" else None
        shared = {0: "shared cells", 1: "shared features", -1: "shared cells and features"}.get(
            int(axis) if axis is not None else 0, "shared cells"
        )
        described = []
        for m in planned:
            obs, var = m.axes
            extra = len(obs.columns) + len(obs.arrays) + len(obs.graphs)
            described.append(
                f"{m.name} ({len(var.rows):,} var x {len(obs.rows):,} obs"
                + (f", {extra} global obs parts" if extra else "") + ")"
            )
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[m.output for m in planned],
            summary=f"MuData ({shared}) → {dst_format}: " + "; ".join(described),
            warnings=warnings,
            detail=_Import(planned, zarr_format, options.compression),
        )

    def write(self, plan: ConversionPlan, staged: Sequence[Path], console: Console) -> None:
        detail: _Import = plan.detail
        for mod, path in zip(detail.mods, staged):
            with open_store(plan.source, "r", require_anndata=False) as src, open_store(
                path, "w", zarr_format=detail.zarr_format
            ) as dst:
                group = src.root["mod"][mod.name]
                backend = "zarr" if is_zarr_group(dst.root) else "hdf5"
                copy_attrs(group.attrs, dst.root.attrs, target_backend=backend)
                ensure_anndata_root_attrs(dst.root)
                for key in group.keys():
                    with console.status(f"[cyan]Copying {mod.name}/{key}...[/]"):
                        copy_tree(group[key], dst.root, key, compression=detail.compression)
                ensure_anndata_skeleton(dst.root)
                with console.status(f"[cyan]Adding the global parts to {mod.name}...[/]"):
                    for axis, (_, mapping, graphs) in zip(mod.axes, _AXES):
                        _write_axis(src.root, dst.root, axis, mapping, graphs, detail.compression)
                    for key in mod.uns:
                        copy_tree(src.root["uns"][key], dst.root["uns"], key, compression=detail.compression)


def _write_axis(
    src: Any, dst: Any, axis: _Axis, mapping: str, graphs: str, compression: Optional[str],
) -> None:
    if axis.columns:
        frame = dst[axis.name]
        own = dataframe_columns(frame, resolve_index(frame, axis.name)[1])
        for key, name in axis.columns:
            _take_member(src[axis.name][key], frame, name, axis.rows, src[axis.name], compression)
        set_column_order(frame, own + [name for _, name in axis.columns])
    position = None
    for key, name in axis.arrays:
        obj = src[mapping][key]
        if spec.encoding_type(obj) in spec.SPARSE_TYPES:
            position = _positions(axis) if position is None else position
            _take_sparse(obj, dst[mapping], name, position, None, axis, compression, dst)
        elif spec.encoding_type(obj) == spec.DATAFRAME:
            _take_frame(obj, dst[mapping], name, axis.rows, compression)
        else:
            _take_array(obj, dst[mapping], name, axis.rows, compression)
    for key, name in axis.graphs:
        obj = src[graphs][key]
        position = _positions(axis) if position is None else position
        if spec.encoding_type(obj) in spec.SPARSE_TYPES:
            _take_sparse(obj, dst[graphs], name, position, position, axis, compression, dst)
        else:
            _take_square(obj, dst[graphs], name, axis.rows, compression)


def _positions(axis: _Axis) -> np.ndarray:
    """Each global row's row in the modality, -1 where it has none."""
    position = np.full(axis.n_global, -1, dtype=np.int64)
    position[axis.rows] = np.arange(len(axis.rows), dtype=np.int64)
    return position


def take_rows(obj: Any, rows: np.ndarray) -> np.ndarray:
    """A 1-D element's values at `rows`, in any order: sorted for the read."""
    if rows.size < 2 or bool((np.diff(rows) > 0).all()):
        return np.asarray(_take_rows(obj, rows))
    order = np.argsort(rows, kind="stable")
    values = np.asarray(_take_rows(obj, rows[order]))
    out = np.empty_like(values)
    out[order] = values
    return out


def _layout(src: Any, parent: Any, shape: Tuple[int, ...], dtype: Any, compression: Optional[str]) -> dict:
    """Creation kwargs for a dataset taken from `src`: its chunks and codecs,
    or `--compression` when given, fitted to `shape`."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    kw = dataset_create_kwargs(src, target_backend=backend, dst_parent=parent)
    if compression is not None:
        kw = with_compression(kw, compression, shape=shape, dtype=dtype, backend=backend,
                              zarr_format=zarr_format_of(parent))
    return fit_chunks(kw, shape, backend=backend)


def _put(src: Any, parent: Any, name: str, values: np.ndarray, compression: Optional[str]) -> Any:
    """`values` as `parent[name]`, laid out as `src` is (chunks, codecs)."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    ds = create_dataset(parent, name, data=values, **_layout(src, parent, values.shape, values.dtype, compression))
    copy_attrs(src.attrs, ds.attrs, target_backend=backend)
    return ds


def _take_member(
    obj: Any, parent: Any, name: str, rows: np.ndarray, frame: Any, compression: Optional[str] = None,
) -> None:
    """One column of `frame` (any layout the spec allows) at `rows`; an
    anndata 0.7 categorical as a current one, as copies write it."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    if is_legacy_dataframe(frame) and is_legacy_categorical(obj):
        write_legacy_categorical(frame, name, obj, parent, take_rows(obj, rows), compression=compression)
        return
    if is_dataset(obj):
        _put(obj, parent, name, take_rows(obj, rows), compression)
        return
    group = parent.create_group(name)
    copy_attrs(obj.attrs, group.attrs, target_backend=backend)
    if "codes" in obj:
        copy_tree(obj["categories"], group, "categories", compression=compression)
        _put(obj["codes"], group, "codes", take_rows(obj["codes"], rows), compression)
    elif "values" in obj and "mask" in obj:
        _put(obj["values"], group, "values", take_rows(obj["values"], rows), compression)
        _put(obj["mask"], group, "mask", take_rows(obj["mask"], rows), compression)
    else:
        for key in obj.keys():  # not row-aligned
            copy_tree(obj[key], group, key)


def _take_frame(obj: Any, parent: Any, name: str, rows: np.ndarray, compression: Optional[str]) -> None:
    """A dataframe in obsm or varm, its rows taken, its index among them."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    frame = parent.create_group(name)
    copy_attrs(obj.attrs, frame.attrs, target_backend=backend)
    legacy = is_legacy_dataframe(obj)
    if legacy:
        frame.attrs["encoding-version"] = "0.2.0"
    for key in obj.keys():
        if key == "__categories":
            if not legacy:
                copy_tree(obj[key], frame, key)
        else:
            _take_member(obj[key], frame, key, rows, obj, compression)


def _take_array(obj: Any, parent: Any, name: str, rows: np.ndarray, compression: Optional[str]) -> None:
    """A dense array's rows, a block at a time, its layout kept."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    shape = (len(rows),) + tuple(obj.shape[1:])
    out = create_dataset(parent, name, shape=shape, dtype=obj.dtype,
                         **_layout(obj, parent, shape, obj.dtype, compression))
    copy_attrs(obj.attrs, out.attrs, target_backend=backend)
    for lo in range(0, len(rows), ROW_BLOCK):
        out[lo:lo + ROW_BLOCK, ...] = aligned_rows(obj, rows[lo:lo + ROW_BLOCK])


def _take_square(obj: Any, parent: Any, name: str, rows: np.ndarray, compression: Optional[str]) -> None:
    """A dense graph: rows a block at a time, then the same columns. A
    block is bounded in bytes, since a row is as long as the global axis."""
    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    shape = (len(rows), len(rows))
    out = create_dataset(parent, name, shape=shape, dtype=obj.dtype,
                         **_layout(obj, parent, shape, obj.dtype, compression))
    copy_attrs(obj.attrs, out.attrs, target_backend=backend)
    step = max(1, min(ROW_BLOCK, SLAB_BYTES // max(1, obj.shape[1] * obj.dtype.itemsize)))
    for lo in range(0, len(rows), step):
        out[lo:lo + step, :] = aligned_rows(obj, rows[lo:lo + step])[:, rows]


def _take_sparse(
    obj: Any, parent: Any, name: str, row_pos: np.ndarray, col_pos: Optional[np.ndarray],
    axis: _Axis, compression: Optional[str], scratch: Any,
) -> None:
    """A sparse matrix's rows (and, for a graph, columns) onto the
    modality's, in any order: entries are remapped as they stream and built
    into the output through on-disk buckets, the transpose's way.

    CSC stays CSC: it is built as the CSR of its transpose. The buckets go
    under `scratch`, the output's root, where no entry of the file's can
    have their name."""
    enc = spec.encoding_type(obj)
    n_rows, n_cols = _sparse_shape(obj)
    out_rows = len(axis.rows)
    out_cols = n_cols if col_pos is None else out_rows
    csc = enc == spec.CSC_MATRIX
    # The major axis of the stored layout, and its map.
    major_pos, minor_pos = (col_pos, row_pos) if csc else (row_pos, col_pos)
    n_major = n_cols if csc else n_rows
    out_major, out_minor = (out_cols, out_rows) if csc else (out_rows, out_cols)
    indptr = np.asarray(obj["indptr"][...], dtype=np.int64)
    data, indices = obj["data"], obj["indices"]

    def blocks() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for lo in range(0, n_major, ROW_BLOCK):
            hi = min(lo + ROW_BLOCK, n_major)
            a, b = int(indptr[lo]), int(indptr[hi])
            minor = np.asarray(indices[a:b], dtype=np.int64)
            values = np.asarray(data[a:b])
            major = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(indptr[lo:hi + 1]))
            if major_pos is not None:
                major = major_pos[major]
            if minor_pos is not None:
                minor = minor_pos[minor]
            keep = (major >= 0) & (minor >= 0)
            yield major[keep], minor[keep], values[keep]

    counts = np.zeros(out_major, dtype=np.int64)
    grouped, last = True, -1
    for major, _, _ in blocks():
        count_keys(counts, major)
        if major.size:
            grouped = grouped and major[0] >= last and bool((np.diff(major) >= 0).all())
            last = int(major[-1])
    appender = CsrAppender(
        parent, name, shape=(out_major, out_minor), data_dtype=data.dtype,
        index_dtype=indices.dtype, compression=compression,
        templates=(data, indices, obj["indptr"]),
    )
    write_runs(
        blocks(), n_rows=out_major, n_targets=1, destinations=[[appender]], scratch_parent=scratch,
        grouped=grouped, canonical=False, counts=counts, value_dtype=data.dtype, chunk=1 << 20,
    )
    appender.finish()
    if csc:
        spec.set_encoding(appender.group, spec.CSC_MATRIX)
        set_shape_attr(appender.group, (out_rows, out_cols))


# ---------------------------------------------------------------------------
# Writing


@dataclass
class _Export:
    #: (modality name, its var rows), in mod order.
    mods: List[Tuple[str, np.ndarray]]
    n_obs: int
    n_var: int
    #: Per modality, the raw/var names it keeps; None for all of raw.
    raw_keep: List[Optional[set]] = field(default_factory=list)


def modality_name(label: str) -> str:
    return MODALITY_NAMES.get(label) or storable(label.strip().lower().replace(" ", "_")) or "mod"


class MudataExport(Converter):
    options = frozenset({"--mod", "--feature-types-column"})

    def plan(
        self, source: Path, output: Path, src_format: str, dst_format: str, options: Options,
    ) -> ConversionPlan:
        warnings: List[str] = []
        if options.mod is not None and options.feature_types_column is not None:
            raise ValueError(
                "--mod writes the whole AnnData as one modality and "
                "--feature-types-column splits it by type: pass one."
            )
        if options.mod is not None and storable(options.mod) != options.mod:
            # "", "." and ".." name a group itself or its parent; "/" a path.
            raise ValueError(
                f"--mod {options.mod!r}: a modality name cannot be empty, '.', '..' or hold '/'."
            )
        with open_store(source, "r", require_anndata=False) as store:
            root = store.root
            n_obs = element_len(resolve_index(root["obs"], "obs")[0])
            var = root["var"]
            n_var = element_len(resolve_index(var, "var")[0])
            columns = dataframe_columns(var, resolve_index(var, "var")[1])
            types_col = options.feature_types_column
            if types_col is not None and types_col not in columns:
                raise ValueError(f"--feature-types-column {types_col}: no such var column.")
            if types_col is None and options.mod is None and "feature_types" in columns:
                types_col = "feature_types"
            mods: List[Tuple[str, np.ndarray]] = []
            if types_col is not None:
                types = np.asarray(read_str_all(var[types_col], parent_group=var), dtype=object)
                counted = type_counts(types)
                # By position: two types can name one modality (`Foo Bar`,
                # `foo_bar`), and each needs a group of its own.
                taken: set = set()
                for label, _ in counted:
                    mods.append((claim(modality_name(label), taken), np.flatnonzero(types == label)))
            if not mods:  # no features, or nothing to split by: one modality
                mods.append((options.mod or "rna", np.arange(n_var, dtype=np.int64)))
            raw_keep = _plan_raw(root, mods, n_var, types_col, warnings)
            for key in _keys(_child(root, "obsm")):
                if key in {m for m, _ in mods}:
                    raise ValueError(
                        f"obsm/{key} has the name of a modality, which a MuData keeps "
                        "for that modality's mask. Rename it first."
                    )
        if same_path(source, output):
            raise ValueError(f"The output {output} is the input.")
        described = ", ".join(f"{name} ({len(rows):,} var)" for name, rows in mods)
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=f"{src_format} → MuData (.h5mu), {n_obs:,} obs; {described}",
            warnings=warnings,
            detail=_Export(mods, n_obs, n_var, raw_keep),
        )

    def write(self, plan: ConversionPlan, staged: Sequence[Path], console: Console) -> None:
        from adata import __version__

        detail: _Export = plan.detail
        with open_store(plan.source, "r", require_anndata=False) as store, h5py.File(
            staged[0], "w", userblock_size=USERBLOCK
        ) as out:
            src = store.root
            mod_group = out.create_group("mod")
            targets = []
            obs_names = read_names(src, "obs")
            for (name, rows), keep in zip(detail.mods, detail.raw_keep):
                group = mod_group.create_group(name)
                ensure_anndata_root_attrs(group)
                group.attrs["encoder"] = "adata"
                group.attrs["encoder-version"] = __version__
                targets.append(Target(group, None, None if len(rows) == detail.n_var else rows, keep))
            _write_targets(
                src, targets, chunk_rows=ROW_BLOCK, console=console,
                skip={"obs", "obsm", "obsp", "uns"}, verb="Writing",
            )
            for target in targets:
                set_column_order(write_index_frame(target.root, "obs", detail.n_obs, [obs_names]), [])
                ensure_anndata_skeleton(target.root)
            with console.status("[cyan]Writing the global parts...[/]"):
                _write_globals(src, out, detail)
            out.attrs["encoding-type"] = MUDATA
            out.attrs["encoding-version"] = ENCODING_VERSION
            out.attrs["encoder"] = "adata"
            out.attrs["encoder-version"] = __version__
            out.attrs["axis"] = 0
            mod_group.attrs["mod-order"] = np.array(
                [name for name, _ in detail.mods], dtype=h5py.string_dtype()
            )
        header = f"MuData (format-version={ENCODING_VERSION};creator=adata;creator-version={__version__})"
        with open(staged[0], "r+b") as raw:
            raw.write(header.encode("ascii")[:USERBLOCK].ljust(USERBLOCK, b"\0"))


def _plan_raw(
    root: Any, mods: List[Tuple[str, np.ndarray]], n_var: int, types_col: Optional[str],
    warnings: List[str],
) -> List[Optional[set]]:
    """Which of raw's features each modality keeps.

    The whole AnnData as one modality keeps raw whole: raw often holds genes
    var was filtered of. Split, each modality takes raw's features of its
    type, by raw/var's own column where it has one; otherwise those named in
    its var, and raw's others are left out, with a warning.
    """
    raw = _child(root, "raw")
    if raw is None or not is_group(raw) or "var" not in raw:
        return [None] * len(mods)
    if types_col is None or not n_var:  # --mod, or nothing to split by: the whole AnnData
        return [None] * len(mods)
    raw_var = raw["var"]
    raw_names = read_names(raw, "var")
    if len(set(raw_names)) != len(raw_names):
        # raw's features are chosen by name, which a repeated one cannot
        # place: it would land in one modality, once.
        raise ValueError(
            "raw's var names repeat, so its features cannot be shared out among "
            "the modalities by name. Make them unique first, or pass --mod to "
            "write the whole AnnData, raw with it, as one modality."
        )
    raw_columns = dataframe_columns(raw_var, resolve_index(raw_var, "var")[1])
    if types_col is not None and types_col in raw_columns:
        raw_types = np.asarray(read_str_all(raw_var[types_col], parent_group=raw_var), dtype=object)
        var = root["var"]
        types = np.asarray(read_str_all(var[types_col], parent_group=var), dtype=object)
        labels = [types[rows[0]] for _, rows in mods]
        orphans = sorted({str(t) for t in raw_types.tolist()} - {str(t) for t in labels})
        if orphans:
            count = int(np.isin(raw_types, orphans).sum())
            warnings.append(
                f"Not written: {count:,} raw features of type {', '.join(map(repr, orphans))}, "
                "which var no longer has, so no modality is written for them."
            )
        names = np.asarray(raw_names, dtype=object)
        return [set(names[raw_types == label].tolist()) for label in labels]
    var_names = read_names(root, "var")
    keeps = [set(var_names[i] for i in rows) for _, rows in mods]
    shared = set().union(*(a & b for i, a in enumerate(keeps) for b in keeps[i + 1:]))
    if shared:
        raise ValueError(
            f"{len(shared):,} var names (first: {sorted(shared)[0]!r}) are in more than one "
            "modality, so raw's feature of that name cannot be given to one. Give raw/var a "
            "feature-types column, or pass --mod to write the whole AnnData as one modality."
        )
    left = len(set(raw_names) - set().union(*keeps))
    if left:
        warnings.append(
            f"Not written: {left:,} raw features that var lacks -- raw/var has no "
            f"{types_col or 'feature types'} column to say which modality they belong to."
        )
    return keeps


def _write_globals(src: Any, out: h5py.File, detail: _Export) -> None:
    """obs, obsm, obsp and uns once for every modality; var as the
    modalities' names stacked; the maps and masks that place each
    modality's rows among the global ones."""
    copy_tree(src["obs"], out, "obs")
    for key in ("obsm", "obsp", "uns"):
        if key in src:
            copy_tree(src[key], out, key)
        write_mapping(out, key)
    var_names = read_names(src, "var")
    stacked = [[var_names[i] for i in rows] for _, rows in detail.mods]
    total = sum(len(rows) for _, rows in detail.mods)
    set_column_order(write_index_frame(out, "var", total, stacked), [])
    for key in ("varm", "varp"):
        write_mapping(out, key)
    obsmap, varmap = write_mapping(out, "obsmap"), write_mapping(out, "varmap")
    offset = 0
    for name, rows in detail.mods:
        write_dense(obsmap, name, np.arange(1, detail.n_obs + 1, dtype=np.uint32))
        where = np.zeros(total, dtype=np.uint32)
        where[offset:offset + len(rows)] = np.arange(1, len(rows) + 1, dtype=np.uint32)
        write_dense(varmap, name, where)
        write_dense(out["obsm"], name, np.ones(detail.n_obs, dtype=bool))
        write_dense(out["varm"], name, where > 0)
        offset += len(rows)

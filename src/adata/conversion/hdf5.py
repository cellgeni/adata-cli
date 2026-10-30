"""One 2-D dataset of a plain HDF5 file as X, as anndata's `read_hdf` reads it.

`--key` names the dataset. Without it nothing is written: the refusal lists
the file's 2-D numeric datasets, and the 1-D ones that could name their rows
and columns, as `read_hdf` lists the file's keys.

The dataset's rows are cells (`--transpose`: genes). Its names are the root
datasets `row_names` and `col_names`, as `read_hdf` takes them, naming the
dataset's rows and columns; `--obs-names-key` and `--var-names-key` name
others. Values keep their dtype. The dataset is read as a loom's matrix is
(`dense.py`): a slab of cells at a time, memory set by the slab whatever its
chunks.

A plain `.h5` is claimed by 10x, so it is read with `--from hdf5`; only
`.hdf5`, `.he5` and `.hdf` are taken for this format by their extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.dense import write_dense_matrix
from adata.conversion.features import STRING_STEP, Target, write_obs_index, write_var
from adata.elements.write import ensure_anndata_skeleton, set_column_order, write_mapping, write_scalar
from adata.storage import COMPRESSIONS, open_store
from adata.util.names import count_duplicates, make_index_unique

EXTENSIONS = (".hdf5", ".he5", ".hdf")

#: Bytes of the dataset read at once.
SLAB_BYTES = 64 << 20

#: Where `read_hdf` looks for the names of the dataset's rows and columns.
ROW_NAMES, COL_NAMES = "row_names", "col_names"

#: Candidates listed in a refusal before the rest are counted instead.
_LISTED = 20


def _numeric_2d(obj: Any) -> bool:
    # Complex too: anndata stores complex arrays, and read_hdf reads them.
    return isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.dtype.kind in "biufc"


def _names_like(obj: Any) -> bool:
    return isinstance(obj, h5py.Dataset) and (
        obj.ndim == 1 or (obj.ndim == 2 and obj.shape[1] == 1)
    ) and (obj.dtype.kind in "biuSO")


def _listing(handle: h5py.File) -> str:
    def shown(found: List[Tuple[str, Any]]) -> str:
        parts = [
            f"{path} ({' x '.join(f'{d:,}' for d in obj.shape)}, {obj.dtype})"
            for path, obj in found[:_LISTED]
        ]
        if len(found) > _LISTED:
            parts.append(f"and {len(found) - _LISTED:,} more")
        return ", ".join(parts) or "none"

    members: List[Tuple[str, Any]] = []
    handle.visititems(lambda path, obj: members.append((path, obj)))
    matrices = [(p, o) for p, o in members if _numeric_2d(o)]
    names = [(p, o) for p, o in members if _names_like(o)]
    return (
        f"The 2-D numeric datasets: {shown(matrices)}. The 1-D ones, which "
        f"could hold names (--obs-names-key, --var-names-key): {shown(names)}."
    )


def _names(ds: h5py.Dataset, lo: int, hi: int) -> List[str]:
    values = np.asarray(ds[lo:hi]).reshape(-1)
    return [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in values]


@dataclass
class _Import:
    key: str
    transpose: bool
    n_obs: int
    n_var: int
    #: Paths of the datasets naming obs and var; None to number them.
    obs_names: Optional[str]
    var_names: Optional[str]
    var_index: List[str]
    dense: bool
    compression: Optional[str]
    zarr_format: Optional[int]


class Hdf5Import(Converter):
    options = frozenset({
        "--key", "--obs-names-key", "--var-names-key", "--transpose", "--dense",
        "--compression", "--make-unique",
    })

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )
        name = Path(source).name
        warnings: List[str] = []
        with h5py.File(source, "r") as handle:
            key = options.key
            if key is None:
                raise ValueError(
                    f"{name}: name the dataset that becomes X with --key. {_listing(handle)}"
                )
            obj = handle.get(key)
            if not _numeric_2d(obj):
                what = "nothing" if obj is None else (
                    "a group" if isinstance(obj, h5py.Group)
                    else f"a {obj.ndim}-D {obj.dtype} dataset"
                )
                raise ValueError(
                    f"--key {key}: {name} holds {what} there, not a 2-D numeric "
                    f"dataset. {_listing(handle)}"
                )
            rows, cols = obj.shape
            n_obs, n_var = (cols, rows) if options.transpose else (rows, cols)
            # read_hdf's names follow the dataset's rows and columns.
            row_axis, col_axis = ("var", "obs") if options.transpose else ("obs", "var")
            defaults = {row_axis: (ROW_NAMES, rows), col_axis: (COL_NAMES, cols)}
            found = {}
            for axis, flag, given, n in (
                ("obs", "--obs-names-key", options.obs_names_key, n_obs),
                ("var", "--var-names-key", options.var_names_key, n_var),
            ):
                if given is not None:
                    ds = handle.get(given)
                    if not _names_like(ds) or ds.shape[0] != n:
                        raise ValueError(
                            f"{flag} {given}: not a dataset of {n:,} names "
                            f"({axis} has {n:,} entries). {_listing(handle)}"
                        )
                    found[axis] = given
                    continue
                default, length = defaults[axis]
                ds = handle.get(default)
                if _names_like(ds) and ds.shape[0] == length:
                    found[axis] = default
                    continue
                if _names_like(ds):
                    warnings.append(
                        f"{default} holds {ds.shape[0]:,} names, but {axis} has "
                        f"{n:,} entries; {axis} is numbered instead."
                    )
                found[axis] = None
            var_index = (
                _names(handle[found["var"]], 0, n_var) if found["var"] else
                [str(i) for i in range(n_var)]
            )
            matrix_dtype = obj.dtype
        if options.make_unique:
            var_index = make_index_unique(var_index)
        elif count_duplicates(var_index):
            warnings.append(
                f"{count_duplicates(var_index):,} var names repeat an earlier one. "
                "Pass --make-unique to suffix them as anndata's "
                "var_names_make_unique does."
            )
        for axis in ("obs", "var"):
            if found[axis] is None:
                warnings.append(
                    f"Nothing names {axis}; it is numbered 0..{(n_obs if axis == 'obs' else n_var) - 1}."
                )
        detail = _Import(
            key=key,
            transpose=options.transpose,
            n_obs=n_obs,
            n_var=n_var,
            obs_names=found["obs"],
            var_names=found["var"],
            var_index=var_index,
            dense=options.dense,
            compression=options.compression or ("gzip" if backend == "hdf5" else None),
            zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
        )
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"hdf5 {key}{' transposed' if options.transpose else ''} → {dst_format}, "
                f"{n_obs:,} cells x {n_var:,} genes, {matrix_dtype}"
                f"{', dense' if options.dense else ''}"
            ),
            warnings=warnings,
            detail=detail,
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Import = plan.detail
        with h5py.File(plan.source, "r") as handle, open_store(
            staged[0], "w", zarr_format=detail.zarr_format
        ) as store:
            root = store.root

            def obs_names() -> Iterator[List[str]]:
                for lo in range(0, detail.n_obs, STRING_STEP):
                    hi = min(lo + STRING_STEP, detail.n_obs)
                    if detail.obs_names is None:
                        yield [str(i) for i in range(lo, hi)]
                    else:
                        yield _names(handle[detail.obs_names], lo, hi)

            obs = write_obs_index(root, detail.n_obs, obs_names())
            set_column_order(obs, [])
            write_var(root, Target(label=None, keep=None, var_names=detail.var_index, columns={}))
            ensure_anndata_skeleton(root)
            meta = write_mapping(root["uns"], "hdf5")
            write_scalar(meta, "key", detail.key)
            write_scalar(meta, "transposed", bool(detail.transpose))
            for axis, path in (("obs_names", detail.obs_names), ("var_names", detail.var_names)):
                if path is not None:
                    write_scalar(meta, axis, path)
            with console.status(f"[cyan]Reading {detail.key}...[/]"):
                write_dense_matrix(
                    handle[detail.key], [(root, "X")], cells_axis=1 if detail.transpose else 0,
                    dense=detail.dense, compression=detail.compression,
                    zarr_format=detail.zarr_format, slab_bytes=SLAB_BYTES,
                )

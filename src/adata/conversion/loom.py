"""Loom files: loompy's, velocyto's, kallisto|bustools', and SCope's.

A loom is one HDF5 file:

* `/matrix`, genes x cells, dense, usually chunked 64 x 64 and gzipped, and
  `/layers/<name>` of the same shape;
* `/col_attrs/<name>`, one value (or row) per cell, and `/row_attrs/<name>`
  per gene; SCope's are also compound, one record of named fields per cell
  or gene;
* `/col_graphs/<name>` and `/row_graphs/<name>`, COO graphs (`a`, `b`, `w`);
* global attributes: datasets under `/attrs` in spec 3.0.0, root HDF5
  attributes before it.

Import follows `anndata.read_loom`: X is the `spliced` layer when there is
one (with the main matrix kept as `layers["matrix"]`), else the main matrix;
attributes with one value per cell are obs columns and those with several
are obsm. Four things differ, on purpose:

* values keep their dtype, where anndata casts X to float32;
* graphs become obsp/varp, and global attributes `uns["loom"]`, where
  anndata drops them;
* var is named by gene id when the file records one, as `convert` names
  every other format;
* compound attributes become dataframes in obsm and varm, a column per
  field, and SCope's embeddings, clusterings and metadata are read as
  scope.py describes, where anndata refuses the file: loompy's validator
  rejects compound attributes.

Strings are read as loompy reads them -- XML character references such as
`&#233;` unescaped -- and written as loompy writes them, ASCII with those
references, so loompy reads our files back unchanged.

The matrix is read and written a slab of cells at a time, aligned to the
file's chunks, so each chunk is decompressed or compressed once and memory
is set by the slab, not the file.
"""

from __future__ import annotations

import html
import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.conversion import scope
from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.dense import slab_size, write_dense_matrix
from adata.conversion.features import (
    STRING_STEP,
    Target,
    check_import_options,
    resolve,
    write_index_frame,
    write_obs_index,
    write_var,
)
from adata.core.csr_build import CsrAppender, count_keys, scratch_group, write_runs
from adata.core.recast import Matrix, describe, transpose_sparse_streaming
from adata.elements import spec
from adata.elements.read import (
    dataframe_columns,
    element_len,
    read_str_chunk,
    resolve_index,
)
from adata.elements.strings import string_dtype_for
from adata.elements.write import (
    create_categorical,
    ensure_anndata_skeleton,
    set_column_order,
    write_dense,
    write_mapping,
    write_scalar,
    write_string_array,
)
from adata.formats.table import write_text_column
from adata.storage import (
    COMPRESSIONS,
    create_dataset,
    is_dataset,
    is_group,
    is_zarr_group,
    open_store,
)
from adata.util.names import claim, count_duplicates, make_index_unique, storable

#: Attributes that name the cells, the gene ids and the gene symbols, in the
#: order they are looked for: loompy and velocyto's names first, then
#: kallisto|bustools'.
OBS_NAMES = ("CellID", "cell_id", "obs_names", "barcode", "barcodes")
GENE_IDS = ("Accession", "gene_id", "gene_ids")
GENE_SYMBOLS = ("Gene", "gene_name", "gene_names", "gene_symbols", "Name")

#: Bytes of matrix held at once: one slab of cells across every gene.
SLAB_BYTES = 64 << 20

#: Above this many bytes, dense, an export says how large the matrix is.
LARGE_DENSE = 10 << 30

_MATRIX_TYPES = frozenset(
    "float16 float32 float64 int8 int16 int32 int64 uint8 uint16 uint32 uint64".split()
)

#: How loompy writes the matrix and layers.
_CHUNK = 64
_GZIP_LEVEL = 2


# ---------------------------------------------------------------------------
# reading


def sniff(path: Path) -> bool:
    """A 2-D `/matrix` dataset with `row_attrs` and `col_attrs` -- or, in a
    file named `.loom`, the matrix alone, as some tools (Baysor) write it."""
    path = Path(path)
    if path.is_dir() or not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as handle:
        matrix = handle.get("matrix")
        if not isinstance(matrix, h5py.Dataset) or matrix.ndim != 2:
            return False
        return path.suffix.lower() == ".loom" or (
            isinstance(handle.get("row_attrs"), h5py.Group)
            and isinstance(handle.get("col_attrs"), h5py.Group)
        )


def decode(values: Any) -> Any:
    """Text as loompy reads it: bytes decoded, XML references unescaped."""
    if isinstance(values, (str, bytes, np.str_, np.bytes_)):
        text = values.decode("utf-8", "replace") if isinstance(values, bytes) else str(values)
        return html.unescape(text)
    arr = np.asarray(values)
    if arr.dtype.kind not in "SOU":
        return values
    flat = [
        html.unescape(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
        for v in arr.reshape(-1)
    ]
    out = np.array(flat, dtype=object).reshape(arr.shape)
    return out if arr.ndim else out[()]


def _version(handle: h5py.File) -> str:
    value = None
    if isinstance(handle.get("attrs"), h5py.Group) and "LOOM_SPEC_VERSION" in handle["attrs"]:
        value = handle["attrs"]["LOOM_SPEC_VERSION"][()]
    elif "LOOM_SPEC_VERSION" in handle.attrs:
        value = handle.attrs["LOOM_SPEC_VERSION"]
    if value is None:
        return "pre-2"
    # R's LoomExperiment writes it as a one-element array.
    return str(np.asarray(decode(value), dtype=object).reshape(-1)[0])


@dataclass
class _Axis:
    """One axis's attributes, sorted by where they go."""

    n: int
    names: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    matrices: List[str] = field(default_factory=list)
    #: Compound attributes -> the column each field becomes, in field order.
    tables: Dict[str, List[str]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)


def _field_readable(dtype: np.dtype) -> bool:
    """A compound field a dataframe column can hold: a number or text."""
    return dtype.names is None and dtype.shape == () and (
        dtype.kind in "biuf" or h5py.check_string_dtype(dtype) is not None
    )


def _axis(group: Any, n: int, what: str) -> _Axis:
    axis = _Axis(n)
    for name, ds in group.items():
        if not isinstance(ds, h5py.Dataset) or ds.ndim == 0:
            axis.skipped.append(name)
            continue
        compound = ds.dtype.names is not None
        if (compound and (ds.ndim != 1 or not all(
            _field_readable(ds.dtype[f]) for f in ds.dtype.names
        ))) or (not compound and ds.dtype.kind == "V"):
            axis.skipped.append(name)
            continue
        if ds.shape[0] != n:
            raise ValueError(
                f"{what} attribute {name!r} has {ds.shape[0]:,} entries, but the "
                f"matrix has {n:,} {'cells' if what == 'Column' else 'genes'}."
            )
        if compound:
            axis.tables[name] = list(ds.dtype.names)
        elif ds.ndim == 1 or (ds.ndim == 2 and ds.shape[1] == 1):
            axis.columns.append(name)
        elif ds.ndim == 2 and ds.dtype.kind in "biuf":
            axis.matrices.append(name)
        else:
            axis.skipped.append(name)
    return axis


def _table_columns(axis: _Axis, what: str, warnings: List[str]) -> None:
    """Each compound field's column name: its own, unless HDF5 or Zarr cannot
    hold it (`/` separates their paths) or a dataframe reserves it."""
    for name, fields in axis.tables.items():
        taken = set(spec.RESERVED_DATAFRAME_KEYS)
        columns = [claim(storable(f), taken) for f in fields]
        renamed = [f"{f!r} as {c!r}" for f, c in zip(fields, columns) if f != c]
        if renamed:
            warnings.append(f"Renamed in {what} {name}: {', '.join(renamed)}.")
        axis.tables[name] = columns


def _pick(axis: _Axis, explicit: Optional[str], candidates: Sequence[str], flag: str) -> Optional[str]:
    if explicit is not None:
        if explicit not in axis.columns:
            raise ValueError(
                f"{flag} {explicit!r}: no such attribute; there are "
                f"{', '.join(axis.columns) or 'none'}."
            )
        return explicit
    return next((name for name in candidates if name in axis.columns), None)


@dataclass
class _Import:
    version: str
    velocyto: bool
    n_cells: int
    n_genes: int
    #: Loom matrix ("" for the main one) -> the AnnData places it fills.
    sources: Dict[str, List[str]]
    obs: _Axis
    var: _Axis
    var_names: List[str]
    graphs: Dict[str, List[str]]
    dense: bool
    compression: Optional[str]
    zarr_format: Optional[int]
    #: What a SCope loom holds beyond a loom's; None for other looms.
    scope: Optional[scope.Scope] = None


class LoomImport(Converter):
    options = frozenset({
        "--layer", "--var-names", "--make-unique", "--compression", "--dense",
        "--obs-names-attr", "--var-names-attr",
    })

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        check_import_options(options)
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )
        warnings: List[str] = []
        with h5py.File(source, "r") as handle:
            matrix = handle["matrix"]
            n_genes, n_cells = matrix.shape
            layers = handle["layers"] if isinstance(handle.get("layers"), h5py.Group) else {}
            for name, layer in layers.items():
                if not isinstance(layer, h5py.Dataset) or layer.shape != matrix.shape:
                    raise ValueError(
                        f"Layer {name!r} is not a {n_genes:,} x {n_cells:,} matrix "
                        "like the main one."
                    )
            for name, ds in (("matrix", matrix), *layers.items()):
                if ds.dtype.kind not in "biuf":
                    raise ValueError(f"{name} holds {ds.dtype} values, which a loom may not.")

            x = options.layer
            if x is None:
                x = "spliced" if "spliced" in layers else ""
            elif x == "matrix":
                x = ""
            elif x not in layers:
                raise ValueError(
                    f"--layer {x!r}: no such layer; there are matrix (the main one)"
                    + "".join(f", {name}" for name in layers) + "."
                )
            sources: Dict[str, List[str]] = {"": [], **{name: [] for name in layers}}
            sources[x].append("X")
            for name in layers:
                sources[name].append(f"layers/{name}")
            if x != "" and "matrix" not in layers:
                # anndata's place for the main matrix, unless a layer of
                # that name -- from an exported AnnData -- takes it.
                sources[""].append("layers/matrix")

            empty: Dict[str, Any] = {}
            obs = _axis(handle.get("col_attrs", empty), n_cells, "Column")
            var = _axis(handle.get("row_attrs", empty), n_genes, "Row")
            obs.names = _pick(obs, options.obs_names_attr, OBS_NAMES, "--obs-names-attr")
            if obs.names is None:
                warnings.append(
                    "No attribute names the cells (looked for "
                    f"{', '.join(OBS_NAMES)}); they are numbered 0..{n_cells - 1}, "
                    "as anndata numbers them. Name one with --obs-names-attr."
                )
            first, second = (
                (GENE_IDS, GENE_SYMBOLS) if options.var_names == "gene_ids"
                else (GENE_SYMBOLS, GENE_IDS)
            )
            var.names = _pick(var, options.var_names_attr, first, "--var-names-attr")
            if var.names is None and options.var_names_attr is None:
                var.names = _pick(var, None, second, "--var-names-attr")
                if var.names is not None:
                    warnings.append(
                        f"No gene {'ids' if first is GENE_IDS else 'symbols'} recorded; "
                        f"genes are named by {var.names!r}."
                    )
            if var.names is None:
                warnings.append(
                    f"No attribute names the genes; they are numbered 0..{n_genes - 1}. "
                    "Name one with --var-names-attr."
                )
            for axis in (obs, var):
                if axis.names in axis.columns:
                    axis.columns.remove(axis.names)
            attrs = _global_attrs(handle)
            found = None
            if scope.is_scope(attrs, handle.get("col_attrs", empty).keys()):
                metadata = attrs.get(scope.METADATA)
                if metadata is not None:
                    # One value; R's writers may store it as a one-element array.
                    values = np.asarray(decode(metadata), dtype=object).reshape(-1)
                    metadata = str(values[0]) if values.size else None
                found = scope.plan(handle.get("col_attrs", empty), metadata, obs, var, warnings)
            for axis, what in ((obs, "col_attrs"), (var, "row_attrs")):
                _table_columns(axis, what, warnings)
                if axis.skipped:
                    warnings.append(
                        f"Not read: {what} {', '.join(axis.skipped)} -- a 0-d, "
                        "higher-dimensional or nested value, which no column holds."
                    )

            var_names = (
                [str(v) for v in decode(handle.get("row_attrs", {})[var.names][...]).reshape(-1)]
                if var.names is not None else [str(i) for i in range(n_genes)]
            )
            if options.make_unique:
                var_names = make_index_unique(var_names)
            elif count_duplicates(var_names):
                warnings.append(
                    f"{count_duplicates(var_names):,} var names repeat an earlier one. "
                    "Pass --make-unique to suffix them as anndata's "
                    "var_names_make_unique does."
                )

            graphs = {
                kind: [
                    name for name, graph in handle[kind].items()
                    if isinstance(graph, h5py.Group) and {"a", "b", "w"} <= set(graph)
                ]
                for kind in ("col_graphs", "row_graphs")
                if isinstance(handle.get(kind), h5py.Group)
            }
            version = _version(handle)
            velocyto = any(
                key.startswith("velocyto") for key in attrs
            ) or {"spliced", "unspliced"} <= set(layers)

        detail = _Import(
            version=version,
            velocyto=velocyto,
            n_cells=n_cells,
            n_genes=n_genes,
            sources=sources,
            obs=obs,
            var=var,
            var_names=var_names,
            graphs=graphs,
            dense=options.dense,
            compression=options.compression or ("gzip" if backend == "hdf5" else None),
            zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
            scope=found,
        )
        kind = "velocyto" if velocyto else "SCope" if found is not None else f"spec {version}"
        named = f"X = {x or 'matrix'}"
        others = [p.split("/", 1)[1] for places in sources.values() for p in places if p != "X"]
        if others:
            named += f", layers: {', '.join(others)}"
        if found is not None and found.describe():
            named += f", {found.describe()}"
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"loom ({kind}) → {dst_format}, {n_cells:,} cells x {n_genes:,} "
                f"genes, {named}{', dense' if options.dense else ''}"
            ),
            warnings=warnings,
            detail=detail,
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Import = plan.detail
        with h5py.File(plan.source, "r") as handle, ExitStack() as stack:
            root = stack.enter_context(
                open_store(staged[0], "w", zarr_format=detail.zarr_format)
            ).root
            cols, rows = handle.get("col_attrs", {}), handle.get("row_attrs", {})
            found = detail.scope
            with console.status("[cyan]Writing obs, var and the attributes...[/]"):
                _write_obs(root, cols, detail.obs, found)
                _write_var(root, rows, detail.var, detail.var_names)
                ensure_anndata_skeleton(root)

                def var_names(step: int) -> Iterator[List[str]]:
                    for lo in range(0, detail.n_genes, step):
                        yield detail.var_names[lo : lo + step]

                for axis, group, key, names in (
                    (detail.obs, cols, "obsm", lambda step: _obs_names(cols, detail.obs, step)),
                    (detail.var, rows, "varm", var_names),
                ):
                    for name in axis.matrices:
                        _copy_rows(group[name], root[key], name)
                    for name, columns in axis.tables.items():
                        _write_table(group[name], root[key], name, columns, names)
                if found is not None:
                    sources = {name for e in found.embeddings for name in (e.x[0], e.y[0])}
                    scope.write_embeddings(
                        root["obsm"], cols, found, detail.n_cells,
                        _block_rows(*(cols[name] for name in sources)),
                    )
                for kind, key in (("col_graphs", "obsp"), ("row_graphs", "varp")):
                    n = detail.n_cells if kind == "col_graphs" else detail.n_genes
                    for name in detail.graphs.get(kind, []):
                        _write_graph(handle[kind][name], root, key, name, n)
                _write_uns(root, handle, detail)
                if found is not None:
                    scope.write_uns(root["uns"], found)

            for loom_name, places in detail.sources.items():
                if not places:
                    continue
                ds = handle["matrix"] if loom_name == "" else handle["layers"][loom_name]
                with console.status(f"[cyan]Reading {loom_name or 'matrix'}...[/]"):
                    _write_matrix(ds, root, places, detail)


def _global_attrs(handle: h5py.File) -> Dict[str, Any]:
    """Global attributes, spec 3's datasets and older root attributes alike."""
    found: Dict[str, Any] = {key: value for key, value in handle.attrs.items()}
    attrs = handle.get("attrs")
    if isinstance(attrs, h5py.Group):
        for key, ds in attrs.items():
            if isinstance(ds, h5py.Dataset):
                found[key] = ds[()]
    return found


def _obs_names(group: h5py.Group, axis: _Axis, step: int) -> Iterator[List[str]]:
    """The cells' names, `step` at a time: the names attribute's, or numbers."""
    n = axis.n
    for lo in range(0, n, step):
        if axis.names is None:
            yield [str(i) for i in range(lo, min(lo + step, n))]
        else:
            yield [str(v) for v in decode(group[axis.names][lo : lo + step]).reshape(-1)]


def _write_obs(root: Any, group: h5py.Group, axis: _Axis, found: Optional[scope.Scope]) -> None:
    n = axis.n
    obs = write_obs_index(root, n, _obs_names(group, axis, STRING_STEP))
    for name in axis.columns:
        _write_column(obs, name, group[name], n)
    clusterings = []
    if found is not None and found.clusterings:
        step = _block_rows(group[scope.CLUSTERINGS])
        clusterings = scope.write_clusterings(obs, group, found, n, step)
    set_column_order(obs, axis.columns + clusterings)


def _write_column(parent: Any, name: str, ds: h5py.Dataset, n: int) -> None:
    """One attribute as a dataframe column, a block of rows at a time."""
    if ds.dtype.kind in "SOU":
        def blocks() -> Iterator[List[str]]:
            for lo in range(0, n, STRING_STEP):
                yield [str(v) for v in decode(ds[lo : lo + STRING_STEP]).reshape(-1)]

        write_text_column(parent, name, n, blocks)
        return
    write = _values_writer(parent, name, ds.dtype, n)
    for lo in range(0, n, STRING_STEP):
        write(lo, np.asarray(ds[lo : lo + STRING_STEP]).reshape(-1))


def _write_var(root: Any, group: h5py.Group, axis: _Axis, var_names: List[str]) -> None:
    columns, enums = {}, []
    for name in axis.columns:
        if h5py.check_enum_dtype(group[name].dtype):
            enums.append(name)
            continue
        values = group[name][...]
        values = values.reshape(-1) if values.ndim == 2 else values
        columns[name] = decode(values) if values.dtype.kind in "SOU" else values
    write_var(root, Target(label=None, keep=None, var_names=var_names, columns=columns))
    if enums:
        var = root["var"]
        for name in enums:
            _values_writer(var, name, group[name].dtype, axis.n)(0, group[name][...].reshape(-1))
        set_column_order(var, axis.columns)


def _block_rows(*datasets: h5py.Dataset) -> int:
    """Rows of compound attributes read at once: as many as `SLAB_BYTES` holds
    across all their fields, and at most `STRING_STEP`."""
    width = sum(ds.dtype.itemsize for ds in datasets)
    return max(1, min(STRING_STEP, SLAB_BYTES // max(1, width)))


def _write_table(
    ds: h5py.Dataset,
    parent: Any,
    name: str,
    columns: List[str],
    names: Callable[[int], Iterable[List[str]]],
) -> None:
    """A compound attribute as a dataframe, a column per field in field order,
    indexed by `names(step)`: read a block of rows at a time across every
    field, so memory is one block, never the table."""
    n = ds.shape[0]
    step = _block_rows(ds)
    frame = write_index_frame(parent, name, n, names(step))
    writers = [
        _field_writer(frame, column, ds.dtype[f], n)
        for f, column in zip(ds.dtype.names, columns)
    ]
    set_column_order(frame, columns)
    for lo in range(0, n, step):
        block = ds[lo : lo + step]
        for f, write in zip(ds.dtype.names, writers):
            write(lo, block[f])
        del block  # before the next is read: one block held, not two


def _field_writer(
    frame: Any, name: str, dtype: np.dtype, n: int
) -> Callable[[int, np.ndarray], None]:
    """Create the column for one compound field; return what writes a block of it.

    Text is decoded as loompy reads it, and kept as text: anndata makes
    categoricals of repeated text in obs and var only.
    """
    if h5py.check_string_dtype(dtype) is None:
        return _values_writer(frame, name, dtype, n)
    backend = "zarr" if is_zarr_group(frame) else "hdf5"
    target = create_dataset(frame, name, shape=(n,), dtype=string_dtype_for(backend))
    spec.set_encoding(target, spec.STRING_ARRAY)

    def text(lo: int, values: np.ndarray) -> None:
        target[lo : lo + values.size] = np.asarray(
            [str(v) for v in decode(values).reshape(-1)], dtype=object
        )

    return text


def _values_writer(
    parent: Any, name: str, dtype: np.dtype, n: int
) -> Callable[[int, np.ndarray], None]:
    """Create a column of numbers; return what writes a block of it.

    R's writers store a factor as an HDF5 enum, which becomes a categorical
    of its labels, in R's level order; and a logical as an enum of FALSE,
    TRUE and NA, which becomes a nullable boolean. (h5py itself reads an
    enum of FALSE and TRUE alone as booleans.) Other numbers keep their
    dtype.
    """
    labels = h5py.check_enum_dtype(dtype) or {}
    if labels and set(labels) <= {"FALSE", "TRUE", "NA"}:
        true, missing = labels.get("TRUE"), labels.get("NA")
        group = parent.create_group(name)
        spec.set_encoding(group, spec.NULLABLE_BOOLEAN)
        values = create_dataset(group, "values", shape=(n,), dtype=bool)
        mask = create_dataset(group, "mask", shape=(n,), dtype=bool)
        for part in (values, mask):
            spec.set_encoding(part, spec.ARRAY)

        def logical(lo: int, codes: np.ndarray) -> None:
            values[lo : lo + codes.size] = codes == true
            mask[lo : lo + codes.size] = codes == missing

        return logical
    if labels:
        levels = sorted(labels.items(), key=lambda item: item[1])
        known = np.array([code for _, code in levels])
        codes_target = create_categorical(parent, name, [label for label, _ in levels], n)

        def factor(lo: int, codes: np.ndarray) -> None:
            at = np.minimum(np.searchsorted(known, codes), known.size - 1)
            codes_target[lo : lo + codes.size] = np.where(known[at] == codes, at, -1)

        return factor
    plain = np.dtype(dtype.str)  # without h5py's metadata
    target = create_dataset(parent, name, shape=(n,), dtype=plain)
    spec.set_encoding(target, spec.ARRAY)

    def number(lo: int, values: np.ndarray) -> None:
        target[lo : lo + values.size] = values.astype(plain, copy=False)

    return number


def _copy_rows(ds: h5py.Dataset, parent: Any, name: str) -> None:
    """A 2-D attribute as obsm/varm, a block of rows at a time."""
    n = ds.shape[0]
    target = create_dataset(parent, name, shape=ds.shape, dtype=ds.dtype)
    spec.set_encoding(target, spec.ARRAY)
    for lo in range(0, n, STRING_STEP):
        target[lo : lo + STRING_STEP] = ds[lo : lo + STRING_STEP]


def _write_graph(graph: h5py.Group, root: Any, key: str, name: str, n: int) -> None:
    """A COO graph as an n x n CSR matrix, its edges sorted through buckets."""
    a, b, w = graph["a"], graph["b"], graph["w"]
    m = a.shape[0]
    if b.shape[0] != m or w.shape[0] != m:
        raise ValueError(f"Graph {name!r} has a, b and w of different lengths.")
    step = 1 << 20

    def coordinates(ds: h5py.Dataset, lo: int) -> np.ndarray:
        values = np.asarray(ds[lo : lo + step])
        if values.dtype.kind == "f" and not np.array_equal(values, np.trunc(values)):
            raise ValueError(f"Graph {name!r} has a vertex that is not a whole number.")
        return values.astype(np.int64)

    def blocks() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for lo in range(0, m, step):
            # R's LoomExperiment stores the vertices as floats.
            rows, cols = coordinates(a, lo), coordinates(b, lo)
            if rows.size and (rows.min() < 0 or rows.max() >= n or cols.min() < 0 or cols.max() >= n):
                raise ValueError(f"Graph {name!r} has an edge outside 0..{n - 1}.")
            yield rows, cols, np.asarray(w[lo : lo + step])

    counts = np.zeros(n, dtype=np.int64)
    grouped, last = True, -1
    for rows, _, _ in blocks():
        count_keys(counts, rows)
        if rows.size:
            grouped = grouped and rows[0] >= last and bool((np.diff(rows) >= 0).all())
            last = int(rows[-1])
    appender = CsrAppender(root[key], name, shape=(n, n), data_dtype=w.dtype)
    write_runs(
        blocks(), n_rows=n, n_targets=1, destinations=[[appender]], scratch_parent=root,
        grouped=grouped, canonical=False, counts=counts, value_dtype=w.dtype, chunk=step,
    )
    appender.finish()


def _write_uns(root: Any, handle: h5py.File, detail: _Import) -> None:
    meta = write_mapping(root["uns"], "loom")
    write_scalar(meta, "spec", detail.version)
    for key, name in (("obs_names", detail.obs.names), ("var_names", detail.var.names)):
        if name is not None:
            write_scalar(meta, key, name)
    attrs = write_mapping(meta, "attrs")
    for key, value in _global_attrs(handle).items():
        if key == scope.METADATA and detail.scope is not None:
            continue  # uns["scope"] holds it, decoded
        value = decode(value)
        arr = np.asarray(value)
        if arr.ndim == 0:
            item = arr.item() if arr.dtype.kind != "O" else value
            if isinstance(item, (str, int, float, np.integer, np.floating)):
                write_scalar(attrs, key, item)
        elif arr.dtype.kind == "O":
            write_string_array(attrs, key, [str(v) for v in arr.reshape(-1)])
        elif arr.dtype.kind in "biuf":
            write_dense(attrs, key, arr)


def slab_cells(n_genes: int, itemsize: int, chunk_width: int) -> int:
    """Cells per slab: as many whole chunk columns as `SLAB_BYTES` holds, one at least."""
    return slab_size(n_genes, itemsize, chunk_width, SLAB_BYTES)


def _write_matrix(ds: h5py.Dataset, root: Any, places: List[str], detail: _Import) -> None:
    """One loom matrix, read once, written to every place that holds it."""
    parents = [
        (root, "X") if place == "X" else (root["layers"], place.split("/", 1)[1])
        for place in places
    ]
    write_dense_matrix(
        ds, parents, cells_axis=1, dense=detail.dense, compression=detail.compression,
        zarr_format=detail.zarr_format, slab_bytes=SLAB_BYTES,
    )


# ---------------------------------------------------------------------------
# export


def _ascii(values: Sequence[str]) -> np.ndarray:
    """Text as loompy writes it: ASCII, anything else as XML references."""
    return np.array([str(v).encode("ascii", "xmlcharrefreplace") for v in values], dtype=object)


_ASCII = h5py.string_dtype(encoding="ascii")


@dataclass
class _Export:
    matrix: str
    layers: List[str]
    obs_columns: List[str]
    var_columns: List[str]
    obsm: List[str]
    varm: List[str]
    obsp: List[str]
    varp: List[str]
    #: Where Gene and Accession come from: a var column, "" for var_names,
    #: None (Accession only) for nowhere.
    gene: str
    accession: Optional[str]
    uns: List[str]
    #: var, or raw/var when the matrix is raw/X.
    var: str = "var"
    varm_key: str = "varm"
    #: SCope's parts, with --scope.
    scope: Optional[scope.ScopeExport] = None


class LoomExport(Converter):
    options = frozenset({"--layer", "--scope", "--clusterings", "--default-embedding"})

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        warnings: List[str] = []
        for flag in ("--clusterings", "--default-embedding"):
            if flag in options.given and not options.scope:
                raise ValueError(f"{flag} shapes a SCope loom: pass --scope too.")
        path = options.layer or "X"
        if path not in ("X", "raw/X") and not path.startswith("layers/"):
            path = f"layers/{path}"
        with open_store(source, "r", require_anndata=False) as store:
            root = store.root
            raw = path.startswith("raw/")
            var_path = "raw/var" if raw else "var"
            var_group = resolve(root, var_path)
            obs_index, obs_name = resolve_index(root["obs"], "obs")
            var_index, var_name = resolve_index(var_group, "var")
            n_obs, n_var = element_len(obs_index), element_len(var_index)
            main = _matrix(root, path, (n_obs, n_var))
            layers = []
            if raw:
                # raw/X has raw/var's genes; X and the layers have var's.
                if "layers" in root and len(root["layers"]) or path != "X":
                    warnings.append(
                        "Not written: X and the layers -- they follow var, and "
                        "raw/X follows raw/var."
                    )
            elif "layers" in root:
                for name in root["layers"]:
                    if f"layers/{name}" == path:
                        continue
                    try:
                        _matrix(root, f"layers/{name}", (n_obs, n_var))
                        layers.append(name)
                    except ValueError as exc:
                        warnings.append(f"Not written: layers/{name}: {exc}")
            if path != "X" and not raw:
                layers.insert(0, "X")
            obs_columns = _columns(root["obs"], obs_name, warnings, "obs")
            var_columns = _columns(var_group, var_name, warnings, var_path)
            gene, accession = _gene_attrs(var_columns)
            varm_key = "raw/varm" if raw else "varm"
            obsm = _arrays(root, "obsm", n_obs, warnings, frames=options.scope)
            varm = _arrays(root, varm_key, n_var, warnings, frames=options.scope)
            obsp = _graphs(root, "obsp", n_obs, warnings)
            varp = [] if raw else _graphs(root, "varp", n_var, warnings)
            uns = _uns_scalars(root, warnings)
            # A clustering goes out inside Clusterings alone (a plain copy too
            # would come back from the loom as a second column), so it claims
            # no attribute name of its own.
            clustered = set(scope.clustering_columns(root, options.clusterings)) if options.scope else set()
            obs_columns, var_columns, owners = _column_owners(
                [c for c in obs_columns if c not in clustered], var_columns, gene, accession,
                options.scope, warnings,
            )
            found = None
            if options.scope:
                found = scope.plan_export(
                    root, obs_columns=[c for c in obs_columns if c != "CellID"],
                    varm_key=varm_key, clusterings=options.clusterings,
                    default_embedding=options.default_embedding, warnings=warnings,
                    owners=owners,
                )
                if found.cluster_id and "ClusterID" in obs_columns:
                    # A stale copy would disagree with Clusterings once the
                    # first clustering is edited.
                    obs_columns = [c for c in obs_columns if c != "ClusterID"]
                    warnings.append("obs column 'ClusterID' is written afresh from the "
                                    f"first clustering, {found.clusterings[0].column!r}.")
                # Embeddings go out as SCope's, not again as plain attributes.
                embedded = {e.key for e in found.embeddings}
                obsm = [name for name in obsm if name not in embedded]
            elif "uns" in root and "scope" in root["uns"]:
                warnings.append(
                    "This came from a SCope loom: pass --scope to write its regulons, "
                    "embeddings and clusterings as SCope reads them; a plain loom "
                    "leaves them out."
                )

            # Arrays last: every name they could want is settled by now.
            obsm = [k for k in obsm if scope.claim_attr(owners, "col_attrs", k, f"obsm/{k}", warnings)]
            varm = [k for k in varm
                    if scope.claim_attr(owners, "row_attrs", k, f"{varm_key}/{k}", warnings)]

        dense = n_obs * n_var * max(1, main.dtype.itemsize) * (1 + len(layers))
        summary = (
            f"{src_format} → loom 3.0.0{' (SCope)' if found is not None else ''}, "
            f"{n_var:,} genes x {n_obs:,} cells from "
            f"{path}{', layers: ' + ', '.join(layers) if layers else ''}"
            f"{', ' + found.describe() if found is not None and found.describe() else ''}; "
            f"dense {_size(dense)} before compression"
        )
        if dense > LARGE_DENSE:
            warnings.append(
                f"A loom holds its matrices dense: {_size(dense)} to compress and "
                "write. It is written, but may take a while; h5ad or zarr keep "
                "them sparse."
            )
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=summary,
            warnings=warnings,
            detail=_Export(
                matrix=path, layers=layers, obs_columns=obs_columns,
                var_columns=var_columns, obsm=obsm, varm=varm, obsp=obsp, varp=varp,
                gene=gene, accession=accession, uns=uns, var=var_path,
                varm_key=varm_key, scope=found,
            ),
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Export = plan.detail
        with open_store(plan.source, "r", require_anndata=False) as store, h5py.File(
            staged[0], "w"
        ) as out:
            root = store.root
            with console.status(f"[cyan]Writing {detail.matrix} dense...[/]"):
                _export_matrix(resolve(root, detail.matrix), out, "matrix")
            layers = out.create_group("layers")
            for name in detail.layers:
                with console.status(f"[cyan]Writing layer {name} dense...[/]"):
                    obj = root["X"] if name == "X" else root["layers"][name]
                    _export_matrix(obj, layers, name)

            with console.status("[cyan]Writing the attributes...[/]"):
                cols = out.create_group("col_attrs")
                rows = out.create_group("row_attrs")
                obs, var = root["obs"], resolve(root, detail.var)
                obs_index, _ = resolve_index(obs, "obs")
                var_index, _ = resolve_index(var, "var")
                _export_strings(cols, "CellID", obs_index, obs)
                for name in detail.obs_columns:
                    if name != "CellID":
                        _export_column(cols, name, obs[name], obs)
                for attr, where in (("Gene", detail.gene), ("Accession", detail.accession)):
                    if where == "":
                        _export_strings(rows, attr, var_index, var)
                    elif where is not None:
                        _export_column(rows, attr, var[where], var)
                skip = {detail.gene, detail.accession, "Gene", "Accession"}
                for name in detail.var_columns:
                    if name not in skip:
                        _export_column(rows, name, var[name], var)
                for key, group, where in (
                    ("obsm", cols, "obsm"), ("varm", rows, detail.varm_key)
                ):
                    for name in getattr(detail, key):
                        _export_array(resolve(root, where)[name], group, name)
                for key, kind in (("obsp", "col_graphs"), ("varp", "row_graphs")):
                    graphs = out.create_group(kind)
                    for name in getattr(detail, key):
                        _export_graph(describe(root[key][name]), graphs.create_group(name))
                attrs = out.create_group("attrs")
                _attr(attrs, "LOOM_SPEC_VERSION", "3.0.0")
                _attr(attrs, "CreationDate", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
                if detail.scope is not None:
                    _export_scope(root, cols, rows, detail.scope, detail.varm_key)
                    # Before uns's own attributes, which a stale MetaData among
                    # them must not replace.
                    _attr(attrs, scope.METADATA, json.dumps(scope.json_safe(detail.scope.metadata)))
                for name in detail.uns:
                    value = _uns_value(root, name)
                    if name not in ("LOOM_SPEC_VERSION", "CreationDate"):
                        _attr(attrs, name, value)


def _export_scope(
    root: Any, cols: h5py.Group, rows: h5py.Group, found: scope.ScopeExport, varm_key: str
) -> None:
    """SCope's compound attributes, a block of rows at a time across their
    fields: the dataframes, the embeddings, the clusterings."""
    n_obs = element_len(resolve_index(root["obs"], "obs")[0])
    for table in found.tables:
        source = resolve(root, table.source)
        target = cols if table.axis == "col" else rows
        n = element_len(resolve_index(source)[0])
        _write_records(target, table.key, n, table.dtype, [
            (field, (lambda lo, hi, c=column: np.asarray(source[c][lo:hi])))
            for field, column in zip(table.fields, table.columns)
        ])
    obsm = root["obsm"] if "obsm" in root else {}
    if found.embeddings:
        default = found.embeddings[0]
        for axis, attr in ((0, "Embeddings_X"), (1, "Embeddings_Y")):
            dtype = np.dtype([(str(e.id), np.dtype(obsm[e.key].dtype.str)) for e in found.embeddings])
            _write_records(cols, attr, n_obs, dtype, [
                (str(e.id), (lambda lo, hi, k=e.key, a=axis: np.asarray(obsm[k][lo:hi, a])))
                for e in found.embeddings
            ])
        if default.id == -1:
            dtype = np.dtype([(f, np.dtype(obsm[default.key].dtype.str)) for f in ("_X", "_Y")])
            _write_records(cols, "Embedding", n_obs, dtype, [
                (f, (lambda lo, hi, a=axis: np.asarray(obsm[default.key][lo:hi, a])))
                for axis, f in enumerate(("_X", "_Y"))
            ])
    if found.clusterings:
        obs = root["obs"]
        dtype = np.dtype([(str(c.id), np.int64) for c in found.clusterings])
        codes = [
            (str(c.id), (lambda lo, hi, col=c.column: np.asarray(scope.categorical_codes(obs[col])[lo:hi], dtype=np.int64)))
            for c in found.clusterings
        ]
        _write_records(cols, "Clusterings", n_obs, dtype, codes)
        if found.cluster_id:
            first = codes[0][1]
            target = cols.create_dataset("ClusterID", shape=(n_obs,), dtype=np.int64)
            for lo in range(0, n_obs, STRING_STEP):
                target[lo : lo + STRING_STEP] = first(lo, lo + STRING_STEP)


def _write_records(
    group: h5py.Group, name: str, n: int, dtype: np.dtype,
    fields: Sequence[Tuple[str, Callable[[int, int], np.ndarray]]],
) -> None:
    """A compound attribute of `n` records, each field read a block at a time."""
    target = group.create_dataset(name, shape=(n,), dtype=dtype)
    step = max(1, min(STRING_STEP, SLAB_BYTES // max(1, dtype.itemsize)))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        block = np.empty(hi - lo, dtype=dtype)
        for field, read in fields:
            block[field] = read(lo, hi)
        target[lo:hi] = block


def _size(n: int) -> str:
    for unit, scale in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20)):
        if n >= scale:
            return f"{n / scale:,.1f} {unit}"
    return f"{n:,} bytes"


def _matrix(root: Any, path: str, shape: Tuple[int, int]) -> Matrix:
    obj = resolve(root, path)
    matrix = describe(obj)
    if matrix.shape != shape:
        raise ValueError(
            f"{path} is {matrix.shape[0]:,} x {matrix.shape[1]:,}, not "
            f"{shape[0]:,} x {shape[1]:,}."
        )
    if matrix.dtype.kind not in "biuf":
        raise ValueError(f"{path} holds {matrix.dtype} values, which a loom may not.")
    return matrix


def _column_kind(col: Any) -> Optional[str]:
    """'number', 'text', or None for what a loom attribute cannot hold."""
    if is_dataset(col):
        if col.ndim != 1:
            return None
        kind = col.dtype.kind
        if kind in "biuf":
            return "number"
        if kind in "SOUT":
            return "text"
        return None
    if is_group(col):
        enc = spec.encoding_type(col)
        if enc in (spec.CATEGORICAL, spec.NULLABLE_STRING_ARRAY):
            return "text"
        if enc in (spec.NULLABLE_INTEGER, spec.NULLABLE_BOOLEAN):
            return "masked"
    return None


def _columns(group: Any, index_name: str, warnings: List[str], axis: str) -> List[str]:
    kept, dropped = [], []
    for name in dataframe_columns(group, index_name):
        (kept if _column_kind(group[name]) else dropped).append(name)
    if dropped:
        warnings.append(
            f"Not written: {axis} {', '.join(dropped)} -- a loom attribute holds "
            "numbers or text only."
        )
    return kept


def _column_owners(
    obs_columns: List[str], var_columns: List[str], gene: str, accession: Optional[str],
    scope_export: bool, warnings: List[str],
) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
    """A loom's col_attrs and row_attrs are flat, so an obs column, an obsm
    entry and one of SCope's parts can ask for one name, and the second
    write failed with the output half written. Names go to the first that
    asks: loom's and SCope's own attributes, then columns here; SCope's
    tables (`scope.plan_export`) and arrays claim theirs after. The others
    are left out, each with a warning."""
    owners: Dict[str, Dict[str, str]] = {"col_attrs": {"CellID": "the cell names"},
                                         "row_attrs": {"Gene": "the gene names"}}
    if accession is not None:
        owners["row_attrs"]["Accession"] = "the gene ids"
    if scope_export:
        owners["col_attrs"].update({
            scope.EMBEDDING: "SCope's default embedding", scope.EMBEDDINGS_X: "SCope's embeddings",
            scope.EMBEDDINGS_Y: "SCope's embeddings", scope.CLUSTERINGS: "SCope's clusterings",
        })
    # ClusterID is SCope's only once there are clusterings; the plan decides.
    obs_columns = [c for c in obs_columns if c in ("CellID", "ClusterID")
                   or scope.claim_attr(owners, "col_attrs", c, f"obs column {c!r}", warnings)]
    skip = {gene, accession, "Gene", "Accession"}
    var_columns = [c for c in var_columns if c in skip
                   or scope.claim_attr(owners, "row_attrs", c, f"var column {c!r}", warnings)]
    if scope_export:
        owners["col_attrs"].setdefault("ClusterID", "SCope's first clustering")
    return obs_columns, var_columns, owners


def _gene_attrs(columns: List[str]) -> Tuple[str, Optional[str]]:
    """Where loom's Gene (symbol) and Accession (id) come from.

    Each is chosen on its own: an explicit column first (`gene_symbols`,
    `Gene`, ... and `gene_ids`, `Accession`), and var_names ("") only for the
    one no column holds -- ids when there is a symbols column (as `convert`
    imports by default), symbols otherwise. Accession is None when nothing
    can be taken for ids.
    """
    ids = next((n for n in ("gene_ids", "Accession") if n in columns), None)
    symbols = next(
        (n for n in ("gene_symbols", "Gene", "gene_name", "gene_names") if n in columns), None
    )
    if symbols is None:
        return "", ids
    return symbols, "" if ids is None else ids


def _arrays(root: Any, key: str, n: int, warnings: List[str], *, frames: bool = False) -> List[str]:
    """The numeric 2-D arrays a loom attribute can hold; with `frames`, the
    dataframes are not reported, since a SCope loom writes them."""
    kept, dropped = [], []
    for name in (root[key] if key in root else []):
        obj = root[key][name]
        if is_dataset(obj) and obj.ndim == 2 and obj.dtype.kind in "biuf" and obj.shape[0] == n:
            kept.append(name)
        elif not (frames and spec.encoding_type(obj) == spec.DATAFRAME):
            dropped.append(name)
    if dropped:
        warnings.append(f"Not written: {key} {', '.join(dropped)} -- only numeric 2-D arrays fit a loom.")
    return kept


def _graphs(root: Any, key: str, n: int, warnings: List[str]) -> List[str]:
    kept, dropped = [], []
    for name in (root[key] if key in root else []):
        try:
            matrix = describe(root[key][name])
            if not matrix.sparse or matrix.shape != (n, n):
                raise ValueError
            kept.append(name)
        except (ValueError, KeyError):
            dropped.append(name)
    if dropped:
        warnings.append(f"Not written: {key} {', '.join(dropped)} -- only sparse graphs fit a loom.")
    return kept


def _uns_scalars(root: Any, warnings: List[str]) -> List[str]:
    """uns entries a global attribute can hold: scalars and 1-D arrays, top
    level or, from a loom import, under uns["loom"]["attrs"]."""
    found = []
    uns = root["uns"] if "uns" in root else {}
    for name in uns:
        if name == "loom":
            attrs = uns["loom"].get("attrs") if is_group(uns["loom"]) else None
            found += [f"loom/attrs/{k}" for k in (attrs or []) if is_dataset(attrs[k]) and attrs[k].ndim <= 1]
        elif is_dataset(uns[name]) and uns[name].ndim <= 1:
            found.append(name)
    return found


def _uns_value(root: Any, name: str) -> Any:
    value = root["uns"][name][()]
    return decode(value) if np.asarray(value).dtype.kind in "SOUT" else value


def _attr(group: h5py.Group, name: str, value: Any) -> None:
    key = name.rsplit("/", 1)[-1]
    if key in group:
        return
    arr = np.asarray(value)
    if arr.dtype.kind in "OSUT":
        if arr.ndim == 0:
            group.create_dataset(key, data=str(value).encode("ascii", "xmlcharrefreplace"), dtype=_ASCII)
        else:
            group.create_dataset(key, data=_ascii(arr.reshape(-1)), dtype=_ASCII)
    elif arr.dtype.kind == "b":
        group.create_dataset(key, data=arr.astype(np.uint8))
    else:
        group.create_dataset(key, data=arr)


def _export_strings(group: h5py.Group, name: str, column: Any, parent: Any) -> None:
    n = element_len(column)
    target = group.create_dataset(name, shape=(n,), dtype=_ASCII)
    cache: Dict[str, np.ndarray] = {}
    for lo in range(0, n, STRING_STEP):
        hi = min(lo + STRING_STEP, n)
        target[lo:hi] = _ascii(read_str_chunk(column, lo, hi, cache, parent_group=parent))


def _export_column(group: h5py.Group, name: str, column: Any, parent: Any) -> None:
    kind = _column_kind(column)
    if kind == "text":
        _export_strings(group, name, column, parent)
        return
    n = element_len(column)
    if kind == "masked":
        # A loom number cannot be missing but as NaN, so these become floats.
        target = group.create_dataset(name, shape=(n,), dtype=np.float64)
        for lo in range(0, n, STRING_STEP):
            values = np.asarray(column["values"][lo : lo + STRING_STEP], dtype=np.float64)
            values[np.asarray(column["mask"][lo : lo + STRING_STEP], dtype=bool)] = np.nan
            target[lo : lo + values.size] = values
        return
    dtype = np.dtype(np.uint8) if column.dtype.kind == "b" else column.dtype
    target = group.create_dataset(name, shape=(n,), dtype=dtype)
    for lo in range(0, n, STRING_STEP):
        target[lo : lo + STRING_STEP] = np.asarray(column[lo : lo + STRING_STEP]).astype(dtype, copy=False)


def _export_array(source: Any, group: h5py.Group, name: str) -> None:
    target = group.create_dataset(name, shape=source.shape, dtype=source.dtype)
    for lo in range(0, source.shape[0], STRING_STEP):
        target[lo : lo + STRING_STEP] = source[lo : lo + STRING_STEP]


def _export_graph(matrix: Matrix, group: h5py.Group) -> None:
    """A sparse graph as loom's COO vectors: a (from), b (to), w (weight)."""
    weight = matrix.dtype if matrix.dtype.kind == "f" else np.dtype(np.float32)
    nnz = matrix.nnz
    a = group.create_dataset("a", shape=(nnz,), dtype=np.int64)
    b = group.create_dataset("b", shape=(nnz,), dtype=np.int64)
    w = group.create_dataset("w", shape=(nnz,), dtype=weight)
    indptr = np.asarray(matrix.obj["indptr"][...], dtype=np.int64)
    step = 1 << 20
    for lo in range(0, nnz, step):
        hi = min(lo + step, nnz)
        minor = np.asarray(matrix.obj["indices"][lo:hi], dtype=np.int64)
        # The major index of each entry, for this block only.
        major = np.searchsorted(indptr, np.arange(lo, hi, dtype=np.int64), side="right") - 1
        rows, cols = (major, minor) if matrix.kind == spec.CSR_MATRIX else (minor, major)
        a[lo:hi], b[lo:hi] = rows, cols
        w[lo:hi] = np.asarray(matrix.obj["data"][lo:hi]).astype(weight, copy=False)


def _loom_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype(np.uint8) if dtype.kind == "b" else dtype


def _export_matrix(obj: Any, parent: h5py.Group, name: str) -> None:
    """An AnnData matrix (cells x genes, any layout) as a dense genes x cells
    dataset, a slab of cells at a time, each slab a whole number of chunks."""
    matrix = describe(obj)
    n_cells, n_genes = matrix.shape
    dtype = _loom_dtype(matrix.dtype)
    kw: Dict[str, Any] = {}
    if n_cells and n_genes:
        kw = {
            "chunks": (min(_CHUNK, n_genes), min(_CHUNK, n_cells)),
            "compression": "gzip", "compression_opts": _GZIP_LEVEL,
        }
    target = parent.create_dataset(name, shape=(n_genes, n_cells), dtype=dtype, fillvalue=0, **kw)
    slab = slab_cells(n_genes, dtype.itemsize, _CHUNK)

    if not matrix.sparse:
        for lo in range(0, n_cells, slab):
            target[:, lo : lo + slab] = np.asarray(matrix.obj[lo : lo + slab]).astype(dtype, copy=False).T
        return
    if matrix.kind == spec.CSC_MATRIX:
        # Cells are the minor axis: sort them to the major one on disk first,
        # rather than hold every cell of a slab of genes.
        with scratch_group(parent, "__loom_scratch__") as scratch:
            transpose_sparse_streaming(
                matrix, scratch, "csr", data_dtype=matrix.dtype, index_dtype=np.dtype(np.int64)
            )
            _dense_from_csr(describe(scratch["csr"]), target, slab, dtype)
        return
    _dense_from_csr(matrix, target, slab, dtype)


def _dense_from_csr(matrix: Matrix, target: h5py.Dataset, slab: int, dtype: np.dtype) -> None:
    n_cells, n_genes = matrix.shape
    indptr = np.asarray(matrix.obj["indptr"][...], dtype=np.int64)
    indices, data = matrix.obj["indices"], matrix.obj["data"]
    for lo in range(0, n_cells, slab):
        hi = min(lo + slab, n_cells)
        block = np.zeros((n_genes, hi - lo), dtype=dtype)
        start, end = int(indptr[lo]), int(indptr[hi])
        if end > start:
            genes = np.asarray(indices[start:end], dtype=np.int64)
            cells = np.repeat(np.arange(hi - lo, dtype=np.int64), np.diff(indptr[lo : hi + 1]))
            # Repeated coordinates mean their sum, as scipy's toarray gives.
            np.add.at(block, (genes, cells), np.asarray(data[start:end]).astype(dtype, copy=False))
        target[:, lo:hi] = block

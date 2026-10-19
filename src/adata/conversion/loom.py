"""Loom files: loompy's, velocyto's, kallisto|bustools'.

A loom is one HDF5 file:

* `/matrix`, genes x cells, dense, usually chunked 64 x 64 and gzipped, and
  `/layers/<name>` of the same shape;
* `/col_attrs/<name>`, one value (or row) per cell, and `/row_attrs/<name>`
  per gene;
* `/col_graphs/<name>` and `/row_graphs/<name>`, COO graphs (`a`, `b`, `w`);
* global attributes: datasets under `/attrs` in spec 3.0.0, root HDF5
  attributes before it.

Import follows `anndata.read_loom`: X is the `spliced` layer when there is
one (with the main matrix kept as `layers["matrix"]`), else the main matrix;
attributes with one value per cell are obs columns and those with several
are obsm. Three things differ, on purpose:

* values keep their dtype, where anndata casts X to float32;
* graphs become obsp/varp, and global attributes `uns["loom"]`, where
  anndata drops them;
* var is named by gene id when the file records one, as `convert` names
  every other format.

Strings are read as loompy reads them -- XML character references such as
`&#233;` unescaped -- and written as loompy writes them, ASCII with those
references, so loompy reads our files back unchanged.

The matrix is read and written a slab of cells at a time, aligned to the
file's chunks, so each chunk is decompressed or compressed once and memory
is set by the slab, not the file.
"""

from __future__ import annotations

import html
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import (
    STRING_STEP,
    Target,
    check_import_options,
    resolve,
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
from adata.elements.write import (
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
    with_compression,
)
from adata.util.names import count_duplicates, make_index_unique

#: Attributes that name the cells, the gene ids and the gene symbols, in the
#: order they are looked for: loompy and velocyto's names first, then
#: kallisto|bustools'.
OBS_NAMES = ("CellID", "cell_id", "obs_names", "barcode", "barcodes")
GENE_IDS = ("Accession", "gene_id", "gene_ids")
GENE_SYMBOLS = ("Gene", "gene_name", "gene_names", "gene_symbols")

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
    path = Path(path)
    if path.is_dir() or not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as handle:
        matrix = handle.get("matrix")
        return (
            isinstance(matrix, h5py.Dataset)
            and matrix.ndim == 2
            and isinstance(handle.get("row_attrs"), h5py.Group)
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
    if isinstance(handle.get("attrs"), h5py.Group) and "LOOM_SPEC_VERSION" in handle["attrs"]:
        return str(decode(handle["attrs"]["LOOM_SPEC_VERSION"][()]))
    if "LOOM_SPEC_VERSION" in handle.attrs:
        return str(decode(handle.attrs["LOOM_SPEC_VERSION"]))
    return "pre-2"


@dataclass
class _Axis:
    """One axis's attributes, sorted by where they go."""

    n: int
    names: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    matrices: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def _axis(group: h5py.Group, n: int, what: str) -> _Axis:
    axis = _Axis(n)
    for name, ds in group.items():
        if not isinstance(ds, h5py.Dataset):
            axis.skipped.append(name)
            continue
        if ds.dtype.names is not None or ds.dtype.kind == "V" or ds.ndim == 0:
            axis.skipped.append(name)
            continue
        if ds.shape[0] != n:
            raise ValueError(
                f"{what} attribute {name!r} has {ds.shape[0]:,} entries, but the "
                f"matrix has {n:,} {'cells' if what == 'Column' else 'genes'}."
            )
        if ds.ndim == 1 or (ds.ndim == 2 and ds.shape[1] == 1):
            axis.columns.append(name)
        elif ds.ndim == 2 and ds.dtype.kind in "biuf":
            axis.matrices.append(name)
        else:
            axis.skipped.append(name)
    return axis


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

            obs = _axis(handle["col_attrs"], n_cells, "Column")
            var = _axis(handle["row_attrs"], n_genes, "Row")
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
            for axis, what in ((obs, "col_attrs"), (var, "row_attrs")):
                if axis.names in axis.columns:
                    axis.columns.remove(axis.names)
                if axis.skipped:
                    warnings.append(
                        f"Not read: {what} {', '.join(axis.skipped)} -- compound or "
                        "higher-dimensional values, such as SCope's regulons (#27)."
                    )

            var_names = (
                [str(v) for v in decode(handle["row_attrs"][var.names][...]).reshape(-1)]
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
                key.startswith("velocyto") for key in _global_attrs(handle)
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
        )
        kind = "velocyto" if velocyto else f"spec {version}"
        named = f"X = {x or 'matrix'}"
        others = [p.split("/", 1)[1] for places in sources.values() for p in places if p != "X"]
        if others:
            named += f", layers: {', '.join(others)}"
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
            with console.status("[cyan]Writing obs, var and the attributes...[/]"):
                _write_obs(root, handle["col_attrs"], detail.obs)
                _write_var(root, handle["row_attrs"], detail.var, detail.var_names)
                ensure_anndata_skeleton(root)
                for axis, group, key in (
                    (detail.obs, handle["col_attrs"], "obsm"),
                    (detail.var, handle["row_attrs"], "varm"),
                ):
                    for name in axis.matrices:
                        _copy_rows(group[name], root[key], name)
                for kind, key in (("col_graphs", "obsp"), ("row_graphs", "varp")):
                    n = detail.n_cells if kind == "col_graphs" else detail.n_genes
                    for name in detail.graphs.get(kind, []):
                        _write_graph(handle[kind][name], root, key, name, n)
                _write_uns(root, handle, detail)

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


def _write_obs(root: Any, group: h5py.Group, axis: _Axis) -> None:
    n = axis.n
    if axis.names is not None:
        names = group[axis.names]
        blocks = (
            [str(v) for v in decode(names[lo : lo + STRING_STEP]).reshape(-1)]
            for lo in range(0, n, STRING_STEP)
        )
    else:
        blocks = (
            [str(i) for i in range(lo, min(lo + STRING_STEP, n))]
            for lo in range(0, n, STRING_STEP)
        )
    obs = write_obs_index(root, n, blocks)
    for name in axis.columns:
        _write_column(obs, name, group[name], n)
    set_column_order(obs, axis.columns)


def _write_column(parent: Any, name: str, ds: h5py.Dataset, n: int) -> None:
    """One attribute as a dataframe column, a block of rows at a time."""
    if ds.dtype.kind in "SOU":
        def blocks() -> Iterator[List[str]]:
            for lo in range(0, n, STRING_STEP):
                yield [str(v) for v in decode(ds[lo : lo + STRING_STEP]).reshape(-1)]

        write_text_column(parent, name, n, blocks)
        return
    target = create_dataset(parent, name, shape=(n,), dtype=ds.dtype)
    spec.set_encoding(target, spec.ARRAY)
    for lo in range(0, n, STRING_STEP):
        target[lo : lo + STRING_STEP] = np.asarray(ds[lo : lo + STRING_STEP]).reshape(-1)


def _write_var(root: Any, group: h5py.Group, axis: _Axis, var_names: List[str]) -> None:
    columns = {}
    for name in axis.columns:
        values = group[name][...]
        values = values.reshape(-1) if values.ndim == 2 else values
        columns[name] = decode(values) if values.dtype.kind in "SOU" else values
    write_var(root, Target(label=None, keep=None, var_names=var_names, columns=columns))


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

    def blocks() -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for lo in range(0, m, step):
            rows = np.asarray(a[lo : lo + step], dtype=np.int64)
            cols = np.asarray(b[lo : lo + step], dtype=np.int64)
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
    per_cell = max(1, n_genes * itemsize)
    fit = max(1, SLAB_BYTES // per_cell)
    return max(chunk_width, fit // chunk_width * chunk_width)


def _write_matrix(ds: h5py.Dataset, root: Any, places: List[str], detail: _Import) -> None:
    """One loom matrix, read once, written to every place that holds it."""
    n_genes, n_cells = ds.shape
    width = ds.chunks[1] if ds.chunks else _CHUNK
    slab = slab_cells(n_genes, ds.dtype.itemsize, width)
    parents = [
        (root, "X") if place == "X" else (root["layers"], place.split("/", 1)[1])
        for place in places
    ]
    backend = "zarr" if is_zarr_group(root) else "hdf5"
    if detail.dense:
        targets = []
        for parent, name in parents:
            kw: Dict[str, Any] = {"chunks": (min(max(n_cells, 1), slab), max(n_genes, 1))}
            if detail.compression is not None:
                kw = with_compression(
                    kw, detail.compression, shape=(n_cells, n_genes), dtype=ds.dtype,
                    backend=backend, zarr_format=detail.zarr_format,
                )
            target = create_dataset(parent, name, shape=(n_cells, n_genes), dtype=ds.dtype, **kw)
            spec.set_encoding(target, spec.ARRAY)
            targets.append(target)
        for lo in range(0, n_cells, slab):
            block = np.asarray(ds[:, lo : lo + slab]).T
            for target in targets:
                target[lo : lo + block.shape[0]] = block
        return

    appenders = [
        CsrAppender(
            parent, name, shape=(n_cells, n_genes), data_dtype=ds.dtype,
            compression=detail.compression,
        )
        for parent, name in parents
    ]
    for lo in range(0, n_cells, slab):
        block = np.asarray(ds[:, lo : lo + slab]).T
        rows, cols = np.nonzero(block)
        values = block[rows, cols]
        for appender in appenders:
            appender.append(rows + lo, cols, values)
    for appender in appenders:
        appender.finish()


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
    gene: Optional[str]
    accession: Optional[str]
    uns: List[str]


class LoomExport(Converter):
    options = frozenset({"--layer"})

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        warnings: List[str] = []
        path = options.layer or "X"
        if path not in ("X", "raw/X") and not path.startswith("layers/"):
            path = f"layers/{path}"
        with open_store(source, "r", require_anndata=False) as store:
            root = store.root
            obs_index, obs_name = resolve_index(root["obs"], "obs")
            var_index, var_name = resolve_index(root["var"], "var")
            n_obs, n_var = element_len(obs_index), element_len(var_index)
            main = _matrix(root, path, (n_obs, n_var))
            layers = []
            if "layers" in root:
                for name in root["layers"]:
                    if f"layers/{name}" == path:
                        continue
                    try:
                        _matrix(root, f"layers/{name}", (n_obs, n_var))
                        layers.append(name)
                    except ValueError as exc:
                        warnings.append(f"Not written: layers/{name}: {exc}")
            if path != "X":
                layers.insert(0, "X")
            obs_columns = _columns(root["obs"], obs_name, warnings, "obs")
            var_columns = _columns(root["var"], var_name, warnings, "var")
            gene, accession = _gene_attrs(root["var"], var_columns)
            obsm = _arrays(root, "obsm", n_obs, warnings)
            varm = _arrays(root, "varm", n_var, warnings)
            obsp = _graphs(root, "obsp", n_obs, warnings)
            varp = _graphs(root, "varp", n_var, warnings)
            uns = _uns_scalars(root, warnings)

        dense = n_obs * n_var * max(1, main.dtype.itemsize) * (1 + len(layers))
        summary = (
            f"{src_format} → loom 3.0.0, {n_var:,} genes x {n_obs:,} cells from "
            f"{path}{', layers: ' + ', '.join(layers) if layers else ''}; "
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
                gene=gene, accession=accession, uns=uns,
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
                obs, var = root["obs"], root["var"]
                obs_index, _ = resolve_index(obs, "obs")
                var_index, _ = resolve_index(var, "var")
                _export_strings(cols, "CellID", obs_index, obs)
                for name in detail.obs_columns:
                    if name != "CellID":
                        _export_column(cols, name, obs[name], obs)
                if detail.accession == "":
                    # var_names are the ids; the symbols are a column.
                    _export_strings(rows, "Accession", var_index, var)
                    _export_column(rows, "Gene", var[detail.gene], var)
                else:
                    _export_strings(rows, "Gene", var_index, var)
                    if detail.accession:
                        _export_column(rows, "Accession", var[detail.accession], var)
                skip = {detail.gene, detail.accession, "Gene"}
                if detail.accession is not None:
                    skip.add("Accession")
                for name in detail.var_columns:
                    if name not in skip:
                        _export_column(rows, name, var[name], var)
                for key, group in (("obsm", cols), ("varm", rows)):
                    for name in getattr(detail, key):
                        _export_array(root[key][name], group, name)
                for key, kind in (("obsp", "col_graphs"), ("varp", "row_graphs")):
                    graphs = out.create_group(kind)
                    for name in getattr(detail, key):
                        _export_graph(describe(root[key][name]), graphs.create_group(name))
                attrs = out.create_group("attrs")
                _attr(attrs, "LOOM_SPEC_VERSION", "3.0.0")
                _attr(attrs, "CreationDate", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
                for name in detail.uns:
                    value = _uns_value(root, name)
                    if name not in ("LOOM_SPEC_VERSION", "CreationDate"):
                        _attr(attrs, name, value)


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


def _gene_attrs(var: Any, columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Which var columns become Gene and Accession, loom's symbol and id.

    Returns (symbols column, ids column): the ids column is "" when var_names
    are the ids themselves, None when nothing holds them.
    """
    if "gene_symbols" in columns:
        return "gene_symbols", ""
    for name in ("Gene", "gene_name", "gene_names"):
        if name in columns and "gene_ids" not in columns and "Accession" not in columns:
            return name, ""
    return None, next((n for n in ("gene_ids", "Accession") if n in columns), None)


def _arrays(root: Any, key: str, n: int, warnings: List[str]) -> List[str]:
    kept, dropped = [], []
    for name in (root[key] if key in root else []):
        obj = root[key][name]
        if is_dataset(obj) and obj.ndim == 2 and obj.dtype.kind in "biuf" and obj.shape[0] == n:
            kept.append(name)
        else:
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
    majors = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr))
    step = 1 << 20
    for lo in range(0, nnz, step):
        hi = min(lo + step, nnz)
        minor = np.asarray(matrix.obj["indices"][lo:hi], dtype=np.int64)
        major = majors[lo:hi]
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

"""pycisTopic's pickles: a `CistopicObject`, or a `CistopicImputedFeatures`.

pycisTopic, the scATAC half of SCENIC+, saves its objects as pickles and
nothing else. They are read here without pycisTopic, through the
allowlisted unpickler in `formats/pickled.py`, which runs nothing from the
file; pandas and scipy, which rebuild their frames and matrices, come with
the `pickle` extra.

A `CistopicObject` becomes:

* X: `fragment_matrix`, regions x cells, as CSR cells x regions, its dtype
  kept; `layers["binary"]`: `binary_matrix`;
* obs and var: `cell_data` and `region_data`, in `cell_names` and
  `region_names` order;
* from the selected topic model: `obsm["X_topic"]` (`cell_topic`, cells x
  topics; `X_topic_harmony` too, when there is one) and `varm["topic_region"]`;
* `obsm["X_<name>"]` and `varm["X_<name>"]`: its cell and region projections;
* `uns["pycistopic"]`: the project, the fragment files, the topic names, and
  the model's tables and parameters.

A `CistopicImputedFeatures` (imputed accessibility, gene activity) becomes X,
cells x features, CSR, or dense with `--dense`.

A pickle cannot be read in part, so the object is held in memory, as it was
when pycisTopic saved it. What is written from it is not copied again: the
fragment matrix, regions x cells CSR, is the same three arrays as a CSC cells
x regions, and is turned into CSR on disk by the streaming transpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import STRING_STEP, Target, write_obs_index, write_var
from adata.core.csr_build import CsrAppender, scratch_group
from adata.core.recast import describe, transpose_sparse_streaming
from adata.elements import spec
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_dense,
    write_mapping,
    write_scalar,
    write_sparse,
    write_string_array,
)
from adata.formats import pickled
from adata.storage import COMPRESSIONS, create_dataset, is_zarr_group, open_store, with_compression
from adata.util.names import claim, count_duplicates, make_index_unique, sanitize_label

EXTENSIONS = (".pkl", ".pickle", ".pkl.gz", ".pickle.gz")

#: Attributes of a CistopicObject and of its model that the mapping reads.
OBJECT_ATTRS = frozenset({
    "fragment_matrix", "binary_matrix", "cell_names", "region_names", "cell_data",
    "region_data", "project", "path_to_fragments", "selected_model", "projections",
})
MODEL_TABLES = ("metrics", "coherence", "marg_topic", "topic_ass", "parameters", "topic_qc_metrics")
MODEL_SCALARS = ("n_topic", "n_cells", "n_regions")
MODEL_ATTRS = frozenset({
    *MODEL_TABLES, *MODEL_SCALARS, "cell_topic", "cell_topic_harmony", "region_topic",
    "topic_region",
})

#: Cells of a dense imputed matrix written at once.
DENSE_CELLS = 4096


def require_extra() -> None:
    try:
        import pandas  # noqa: F401
        import scipy.sparse  # noqa: F401
    except ImportError:
        raise ValueError(
            "Reading a pycisTopic pickle needs pandas and scipy, which rebuild its "
            "tables and matrices: pip install 'pyadata-cli[pickle]'."
        ) from None


def _is_frame(value: Any) -> bool:
    import pandas as pd

    return isinstance(value, pd.DataFrame)


@dataclass
class _Import:
    kind: str
    obj: Any
    n_obs: int
    n_var: int
    obs_names: List[str]
    var_names: List[str]
    cell_data: Any = None
    region_data: Any = None
    #: (obsm or varm, key, values): topics and projections.
    arrays: List[Tuple[str, str, np.ndarray]] = field(default_factory=list)
    topics: Optional[List[str]] = None
    projections: Dict[str, str] = field(default_factory=dict)
    dense: bool = False
    compression: Optional[str] = None
    zarr_format: Optional[int] = None


class CistopicImport(Converter):
    options = frozenset({"--compression", "--make-unique", "--dense"})

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        require_extra()
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )
        name = Path(source).name
        try:
            obj = pickled.load(source)
        except pickled.RefusedGlobal as exc:
            raise ValueError(f"{name}: {exc}") from None
        kind = getattr(obj, "kind", type(obj).__name__)
        warnings: List[str] = []
        if kind == "CistopicObject":
            if options.dense:
                raise ValueError(
                    "--dense applies to an imputed-features pickle; a CistopicObject's "
                    "fragment matrix is sparse."
                )
            detail = _plan_object(obj, name, warnings)
        elif kind == "CistopicImputedFeatures":
            detail = _plan_imputed(obj, name)
        else:
            raise ValueError(
                f"{name} holds a {kind}; convert reads pickles of a CistopicObject "
                "or a CistopicImputedFeatures."
            )
        if options.make_unique:
            detail.var_names = make_index_unique(detail.var_names)
        elif count_duplicates(detail.var_names):
            warnings.append(
                f"{count_duplicates(detail.var_names):,} var names repeat an earlier one. "
                "Pass --make-unique to suffix them as anndata's var_names_make_unique does."
            )
        detail.dense = options.dense
        detail.compression = options.compression or ("gzip" if backend == "hdf5" else None)
        detail.zarr_format = (options.zarr_format or 3) if dst_format == "zarr" else None

        features = "regions" if kind == "CistopicObject" else "features"
        extra = []
        if detail.topics:
            extra.append(f"{len(detail.topics)} topics")
        if detail.projections:
            extra.append(", ".join(detail.projections))
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"pycisTopic {kind} → {dst_format}, {detail.n_obs:,} cells x "
                f"{detail.n_var:,} {features}" + "".join(f", {e}" for e in extra)
                + (", dense" if options.dense else "")
            ),
            warnings=warnings,
            detail=detail,
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Import = plan.detail
        warnings: List[str] = []
        with open_store(staged[0], "w", zarr_format=detail.zarr_format) as store:
            root = store.root
            with console.status("[cyan]Writing obs and var...[/]"):
                _write_axis(root, "obs", detail.cell_data, detail.obs_names, warnings)
                _write_axis(root, "var", detail.region_data, detail.var_names, warnings)
                ensure_anndata_skeleton(root)
            with console.status("[cyan]Writing the matrices...[/]"):
                if detail.kind == "CistopicObject":
                    _write_fragments(root, "X", detail.obj.fragment_matrix, detail)
                    binary = getattr(detail.obj, "binary_matrix", None)
                    if _is_sparse(binary) and binary.shape == detail.obj.fragment_matrix.shape:
                        _write_fragments(root["layers"], "binary", binary, detail)
                else:
                    _write_imputed(root, detail)
            for key, label, values in detail.arrays:
                write_dense(root[key], label, values)
            _write_uns(root["uns"], detail, warnings)
        for message in warnings:
            console.print(f"[yellow]Warning:[/] {message}")


def _is_sparse(value: Any) -> bool:
    import scipy.sparse as sp

    return sp.issparse(value)


def _names(values: Any) -> List[str]:
    return [str(v) for v in list(values)]


def _ordered(frame: Any, names: List[str], axis: str, what: str, warnings: List[str]) -> Any:
    """`frame` with its rows (or, `axis="columns"`, columns) in `names` order."""
    labels = frame.index if axis == "index" else frame.columns
    if [str(v) for v in labels] == names:
        return frame
    frame = frame.copy()
    if axis == "index":
        frame.index = [str(v) for v in frame.index]
    else:
        frame.columns = [str(v) for v in frame.columns]
    missing = len(set(names) - set(frame.index if axis == "index" else frame.columns))
    if missing:
        warnings.append(f"{what} lacks {missing:,} of the names; they are missing values.")
    return frame.reindex(index=names) if axis == "index" else frame.reindex(columns=names)


def _plan_object(obj: Any, name: str, warnings: List[str]) -> _Import:
    matrix = getattr(obj, "fragment_matrix", None)
    if not _is_sparse(matrix):
        raise ValueError(f"{name}: its CistopicObject holds no sparse fragment_matrix.")
    cells, regions = _names(obj.cell_names), _names(obj.region_names)
    if matrix.shape != (len(regions), len(cells)):
        raise ValueError(
            f"{name}: fragment_matrix is {matrix.shape[0]:,} x {matrix.shape[1]:,}, but "
            f"the object names {len(regions):,} regions and {len(cells):,} cells."
        )
    detail = _Import("CistopicObject", obj, len(cells), len(regions), cells, regions)
    if _is_frame(getattr(obj, "cell_data", None)):
        detail.cell_data = _ordered(obj.cell_data, cells, "index", "cell_data", warnings)
    if _is_frame(getattr(obj, "region_data", None)):
        detail.region_data = _ordered(obj.region_data, regions, "index", "region_data", warnings)

    taken = set()
    model = getattr(obj, "selected_model", None)
    if isinstance(model, pickled.PycistopicObject):
        cell_topic = getattr(model, "cell_topic", None)
        if _is_frame(cell_topic):
            detail.topics = _names(cell_topic.index)
            frame = _ordered(cell_topic, cells, "columns", "cell_topic", warnings)
            detail.arrays.append(("obsm", claim("X_topic", taken), frame.to_numpy().T))
            harmony = getattr(model, "cell_topic_harmony", None)
            if _is_frame(harmony):
                frame = _ordered(harmony, cells, "columns", "cell_topic_harmony", warnings)
                detail.arrays.append(("obsm", claim("X_topic_harmony", taken), frame.to_numpy().T))
        region_topic = getattr(model, "region_topic", None)
        if not _is_frame(region_topic):
            region_topic = getattr(model, "topic_region", None)
        if _is_frame(region_topic):
            frame = _ordered(region_topic, regions, "index", "topic_region", warnings)
            detail.arrays.append(("varm", "topic_region", frame.to_numpy()))
        extra = sorted(set(vars(model)) - MODEL_ATTRS)
        if extra:
            warnings.append(f"Not read: the model's {', '.join(extra)}.")

    projections = getattr(obj, "projections", None)
    for target, key, labels in (("cell", "obsm", cells), ("region", "varm", regions)):
        found = projections.get(target, {}) if isinstance(projections, dict) else {}
        for label, frame in found.items():
            if not _is_frame(frame):
                continue
            frame = _ordered(frame, labels, "index", f"projection {label}", warnings)
            slot = claim(f"X_{sanitize_label(str(label))}", taken)
            detail.arrays.append((key, slot, frame.to_numpy(dtype=np.float64)))
            detail.projections[slot] = str(label)
    extra = sorted(set(vars(obj)) - OBJECT_ATTRS)
    if extra:
        warnings.append(f"Not read: the CistopicObject's {', '.join(extra)}.")
    return detail


def _plan_imputed(obj: Any, name: str) -> _Import:
    matrix = getattr(obj, "mtx", None)
    features, cells = _names(obj.feature_names), _names(obj.cell_names)
    if matrix is None or tuple(matrix.shape) != (len(features), len(cells)):
        shape = "nothing" if matrix is None else f"{matrix.shape[0]:,} x {matrix.shape[1]:,}"
        raise ValueError(
            f"{name}: its matrix is {shape}, but it names {len(features):,} features "
            f"and {len(cells):,} cells."
        )
    return _Import("CistopicImputedFeatures", obj, len(cells), len(features), cells, features)


def _write_axis(root: Any, key: str, frame: Any, names: List[str], warnings: List[str]) -> None:
    if frame is not None:
        from adata.formats.pandas_frame import write_frame

        write_frame(root, key, frame, index=names, categorize=True, warnings=warnings, what=key)
        return
    if key == "obs":
        group = write_obs_index(
            root, len(names), (names[lo : lo + STRING_STEP] for lo in range(0, len(names), STRING_STEP))
        )
        set_column_order(group, [])
    else:
        write_var(root, Target(label=None, keep=None, var_names=names, columns={}))


def _write_fragments(parent: Any, name: str, matrix: Any, detail: _Import) -> None:
    """A regions x cells sparse matrix as CSR cells x regions."""
    shape = (detail.n_obs, detail.n_var)
    if matrix.format == "csc":
        # regions x cells CSC is cells x regions CSR already.
        write_sparse(parent, name, matrix.data, matrix.indices, matrix.indptr, shape)
        return
    if matrix.format != "csr":
        matrix = matrix.tocsr()
    index_dtype = np.dtype(np.int32 if detail.n_var < 2**31 else np.int64)
    with scratch_group(parent, "__cistopic_scratch__") as scratch:
        # The same three arrays, read as a CSC of the transpose: nothing copied.
        write_sparse(scratch, "csc", matrix.data, matrix.indices, matrix.indptr, shape,
                     enc_type=spec.CSC_MATRIX)
        transpose_sparse_streaming(
            describe(scratch["csc"]), parent, name, data_dtype=matrix.dtype,
            index_dtype=index_dtype, compression=detail.compression,
        )


def _write_imputed(root: Any, detail: _Import) -> None:
    """The features x cells matrix as cells x features, a block of cells at a time."""
    matrix = detail.obj.mtx
    if _is_sparse(matrix):
        _write_fragments(root, "X", matrix, detail)
        return
    matrix = np.asarray(matrix)
    shape = (detail.n_obs, detail.n_var)
    dtype = np.dtype(matrix.dtype.str)
    if detail.dense:
        kw: Dict[str, Any] = {"chunks": (min(max(shape[0], 1), DENSE_CELLS), max(shape[1], 1))}
        if detail.compression is not None:
            kw = with_compression(
                kw, detail.compression, shape=shape, dtype=dtype,
                backend="zarr" if is_zarr_group(root) else "hdf5", zarr_format=detail.zarr_format,
            )
        target = create_dataset(root, "X", shape=shape, dtype=dtype, **kw)
        spec.set_encoding(target, spec.ARRAY)
        for lo in range(0, shape[0], DENSE_CELLS):
            target[lo : lo + DENSE_CELLS] = matrix[:, lo : lo + DENSE_CELLS].T
        return
    appender = CsrAppender(root, "X", shape=shape, data_dtype=dtype, compression=detail.compression)
    for lo in range(0, shape[0], DENSE_CELLS):
        block = matrix[:, lo : lo + DENSE_CELLS].T
        rows, cols = np.nonzero(block)
        appender.append(rows + lo, cols, block[rows, cols])
    appender.finish()


def _write_uns(uns: Any, detail: _Import, warnings: List[str]) -> None:
    from adata.formats.pandas_frame import write_frame

    obj = detail.obj
    meta = write_mapping(uns, "pycistopic")
    write_scalar(meta, "kind", detail.kind)
    project = getattr(obj, "project", None)
    if isinstance(project, str):
        write_scalar(meta, "project", project)
    fragments = getattr(obj, "path_to_fragments", None)
    if isinstance(fragments, dict) and fragments:
        group = write_mapping(meta, "path_to_fragments")
        for sample, path in fragments.items():
            write_scalar(group, str(sample), str(path))
    if detail.topics:
        write_string_array(meta, "topics", detail.topics)
    if detail.projections:
        group = write_mapping(meta, "projections")
        for slot, label in detail.projections.items():
            write_scalar(group, slot, label)
    model = getattr(obj, "selected_model", None)
    if isinstance(model, pickled.PycistopicObject):
        group = write_mapping(meta, "model")
        for key in MODEL_TABLES:
            value = getattr(model, key, None)
            if _is_frame(value):
                write_frame(group, key, value, warnings=warnings, what=f"uns/pycistopic/model/{key}")
        for key in MODEL_SCALARS:
            value = getattr(model, key, None)
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                write_scalar(group, key, value)

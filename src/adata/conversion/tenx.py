"""10x Genomics feature-barcode matrices in HDF5.

Cell Ranger's `filtered_feature_bc_matrix.h5` and `raw_feature_bc_matrix.h5`,
and the same layout from Space Ranger, Xenium and CellBender. Two layouts:

* v3 (Cell Ranger 3 onwards): `/matrix/{data,indices,indptr,shape,barcodes}`
  and `/matrix/features/{id,name,feature_type,genome,...}`.
* v2 (Cell Ranger 2): one group per genome,
  `/<genome>/{data,indices,indptr,shape,barcodes,genes,gene_names}`.

The matrix is stored genes x cells in CSC, which is byte for byte cells x
genes in CSR -- AnnData's layout for X. So the arrays are streamed straight
across in both directions and nothing is transposed, unless the AnnData side
holds CSC or dense.

The AnnData produced matches `scanpy.read_10x_h5(gex_only=False)` column for
column, so code written against scanpy's output works on ours. Two
differences: X keeps Cell Ranger's integer counts, where scanpy converts
them to float32; and the run's metadata, which scanpy drops, is kept in
`uns["cellranger"]` so an export can write it back.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.core.recast import describe, sparsify, transpose_sparse_streaming
from adata.core.subset import fan_out_sparse_matrix
from adata.elements import spec
from adata.elements.read import (
    dataframe_columns,
    decode_str_array,
    element_len,
    read_str_all,
    read_str_chunk,
    resolve_index,
)
from adata.elements.strings import string_dtype_for
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_categorical,
    write_dataframe_header,
    write_dense,
    write_mapping,
    write_scalar,
    write_string_array,
)
from adata.storage import (
    GZIP_LEVEL,
    copy_dataset,
    create_dataset,
    has_valid_anndata_root_attrs,
    is_dataset,
    is_group,
    is_zarr_group,
    open_store,
)
from adata.util.names import count_duplicates, make_index_unique, unique_names

#: What a group must hold to be a matrix, in each layout.
_V3_KEYS = frozenset({"barcodes", "data", "indices", "indptr", "shape", "features"})
_V2_KEYS = frozenset(
    {"barcodes", "data", "indices", "indptr", "shape", "genes", "gene_names"}
)

#: Feature datasets that do not become a var column of the same name.
_FEATURE_CORE = frozenset({"name", "feature_type", "id", "gene_id", "_all_tag_keys"})

#: Root attributes describing the run. A v2 file's other root attributes are
#: PyTables bookkeeping; carried into a v3 file they would make Seurat read
#: it as v2.
_RUN_ATTRS = (
    "chemistry_description",
    "filetype",
    "library_ids",
    "original_gem_groups",
    "software_version",
    "version",
)

#: Rows of a string column handled per read when streaming one.
_STRING_STEP = 1 << 18

VAR_NAMES = ("gene_symbols", "gene_ids")


# ---------------------------------------------------------------------------
# reading the layout


def _is_v3(handle: h5py.File) -> bool:
    group = handle.get("matrix")
    return (
        isinstance(group, h5py.Group)
        and _V3_KEYS <= set(group.keys())
        and isinstance(group["features"], h5py.Group)
    )


def _v2_genomes(handle: h5py.File) -> List[str]:
    return [
        name
        for name, obj in handle.items()
        if isinstance(obj, h5py.Group) and _V2_KEYS <= set(obj.keys())
    ]


def sniff(path: Path) -> bool:
    path = Path(path)
    if path.is_dir() or not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as handle:
        if has_valid_anndata_root_attrs(handle):
            return False
        return _is_v3(handle) or bool(_v2_genomes(handle))


def _strings(dataset: Any) -> np.ndarray:
    return decode_str_array(np.asarray(dataset[...]))


@dataclass
class _Matrix:
    """A 10x matrix: where it is and what describes its features."""

    version: int
    group: str
    n_barcodes: int
    n_features: int
    names: np.ndarray
    ids: np.ndarray
    #: var columns other than the names, in scanpy's order.
    columns: Dict[str, np.ndarray]
    feature_types: Optional[np.ndarray]
    tag_keys: List[str]
    attrs: Dict[str, Any]


def _read_matrix(handle: h5py.File, genome: Optional[str]) -> _Matrix:
    """Everything about the matrix but its entries, checked for consistency."""
    if _is_v3(handle):
        version, name = 3, "matrix"
    else:
        genomes = _v2_genomes(handle)
        if not genomes:
            raise ValueError("Not a 10x matrix: no /matrix group or genome groups.")
        if genome is not None:
            if genome not in genomes:
                raise ValueError(
                    f"No genome {genome!r} in this file; it has: {', '.join(genomes)}."
                )
            name = genome
        elif len(genomes) > 1:
            raise ValueError(
                f"This Cell Ranger 2 file holds {len(genomes)} genomes "
                f"({', '.join(genomes)}); pick one with --genome."
            )
        else:
            name = genomes[0]
        version = 2

    group = handle[name]
    n_features, n_barcodes = (int(v) for v in np.asarray(group["shape"][...])[:2])
    problems = []
    if group["barcodes"].shape[0] != n_barcodes:
        problems.append(f"{group['barcodes'].shape[0]} barcodes")
    if group["indptr"].shape[0] != n_barcodes + 1:
        problems.append(f"an indptr of {group['indptr'].shape[0]}")
    if group["indices"].shape[0] != group["data"].shape[0]:
        problems.append("indices and data of different lengths")
    if problems:
        raise ValueError(
            f"Inconsistent 10x matrix: shape says {n_features} features x "
            f"{n_barcodes} barcodes, but it has {' and '.join(problems)}."
        )

    columns: Dict[str, np.ndarray] = {}
    feature_types = None
    tag_keys: List[str] = []
    if version == 3:
        features = group["features"]
        ids, names = _strings(features["id"]), _strings(features["name"])
        if "gene_id" in features:
            # A probe-barcode matrix (Flex): `id` names probes, not genes.
            columns["gene_ids"] = _strings(features["gene_id"])
            columns["probe_ids"] = ids
        else:
            columns["gene_ids"] = ids
        feature_types = _strings(features["feature_type"])
        columns["feature_types"] = feature_types
        for key, item in features.items():
            if (
                key in _FEATURE_CORE
                or not isinstance(item, h5py.Dataset)
                or item.ndim != 1
                or item.shape[0] != n_features
            ):
                continue
            values = np.asarray(item[...])
            columns[key] = values.astype(bool) if values.dtype.kind == "b" else decode_str_array(values)
        if "_all_tag_keys" in features:
            tag_keys = list(_strings(features["_all_tag_keys"]))
    else:
        ids, names = _strings(group["genes"]), _strings(group["gene_names"])
        columns["gene_ids"] = ids

    for label, values in (("ids", ids), ("names", names)):
        if values.shape[0] != n_features:
            raise ValueError(
                f"Inconsistent 10x matrix: {values.shape[0]} feature {label} "
                f"for {n_features} features."
            )

    attrs = {
        key: handle.attrs[key] for key in _RUN_ATTRS if key in handle.attrs
    }
    return _Matrix(
        version=version,
        group=name,
        n_barcodes=n_barcodes,
        n_features=n_features,
        names=names,
        ids=ids,
        columns=columns,
        feature_types=feature_types,
        tag_keys=tag_keys,
        attrs=attrs,
    )


class _AsCsr:
    """A 10x matrix group, seen as the AnnData CSR matrix it already is.

    Lets the streaming subset writer read it unchanged: that writer wants
    the encoding and shape as attributes, where 10x keeps neither.
    """

    def __init__(self, group: h5py.Group, shape: Tuple[int, int]) -> None:
        self._group = group
        self.attrs = {
            "encoding-type": spec.CSR_MATRIX,
            "encoding-version": spec.CURRENT_VERSION[spec.CSR_MATRIX],
            "shape": tuple(shape),
        }

    def __getitem__(self, key: str) -> Any:
        return self._group[key]


# ---------------------------------------------------------------------------
# import


@dataclass
class _Output:
    """One AnnData to write: which features it keeps, and their annotation."""

    label: Optional[str]
    keep: Optional[np.ndarray]
    var_names: List[str]
    columns: Dict[str, np.ndarray]


def _type_counts(types: np.ndarray) -> List[Tuple[str, int]]:
    """Feature types in order of first appearance, with how many of each."""
    labels, first, counts = np.unique(types, return_index=True, return_counts=True)
    order = np.argsort(first)
    return [(str(labels[i]), int(counts[i])) for i in order]


class TenxImport(Converter):
    options = frozenset(
        {"--genome", "--var-names", "--make-unique", "--feature-type",
         "--split-feature-types"}
    )

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        if options.var_names not in VAR_NAMES:
            raise ValueError(f"--var-names must be one of: {', '.join(VAR_NAMES)}.")
        if options.feature_type is not None and options.split_feature_types:
            raise ValueError(
                "--feature-type keeps one type and --split-feature-types writes "
                "every type separately; pass one or the other."
            )

        with h5py.File(source, "r") as handle:
            matrix = _read_matrix(handle, options.genome)
            has_filtered = "filtered_barcodes" in handle[matrix.group]

        if matrix.version == 2 and (
            options.feature_type is not None or options.split_feature_types
        ):
            raise ValueError(
                "This is a Cell Ranger 2 file, which records no feature types; "
                "--feature-type and --split-feature-types need a v3 file."
            )

        keep = np.ones(matrix.n_features, dtype=bool)
        if options.genome is not None and matrix.version == 3:
            genomes = matrix.columns.get("genome")
            if genomes is None:
                raise ValueError("This file records no genome per feature to filter on.")
            if options.genome not in set(genomes):
                raise ValueError(
                    f"No features of genome {options.genome!r}; this file has: "
                    f"{', '.join(sorted(set(genomes)))}."
                )
            keep &= genomes == options.genome

        types = matrix.feature_types
        groups: List[Tuple[Optional[str], np.ndarray]] = []
        if options.feature_type is not None:
            available = [t for t, _ in _type_counts(types)]
            if options.feature_type not in available:
                raise ValueError(
                    f"No features of type {options.feature_type!r}; this file "
                    f"has: {', '.join(available)}."
                )
            groups.append((options.feature_type, keep & (types == options.feature_type)))
        elif options.split_feature_types:
            for label, _ in _type_counts(types[keep]):
                groups.append((label, keep & (types == label)))
        else:
            groups.append((None, keep))

        stems = unique_names([label for label, _ in groups if label is not None])
        outputs: List[Path] = []
        targets: List[_Output] = []
        warnings: List[str] = []
        for label, mask in groups:
            path = output
            if options.split_feature_types:
                path = output.with_name(f"{output.stem}.{stems[label]}{output.suffix}")
            selection = None if mask.all() else np.flatnonzero(mask)
            target = self._annotate(matrix, selection, options, label)
            duplicated = count_duplicates(target.var_names)
            if duplicated:
                where = f" in {path.name}" if options.split_feature_types else ""
                warnings.append(
                    f"{duplicated:,} var names{where} repeat an earlier one. "
                    "Pass --make-unique to suffix them as anndata's "
                    "var_names_make_unique does."
                )
            outputs.append(path)
            targets.append(target)

        described = f"{matrix.n_barcodes:,} barcodes x {matrix.n_features:,} features"
        if types is not None:
            described += " (" + ", ".join(f"{t} {n:,}" for t, n in _type_counts(types)) + ")"
        if len(outputs) > 1:
            described += f", into {len(outputs)} outputs by feature type"
        elif targets[0].keep is not None:
            described += f", keeping {len(targets[0].keep):,} features"

        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=f"10x h5 v{matrix.version} → {dst_format}, {described}",
            warnings=warnings,
            detail={
                "matrix": matrix,
                "targets": targets,
                "filtered_barcodes": has_filtered,
                "zarr_format": (options.zarr_format or 3) if dst_format == "zarr" else None,
            },
        )

    @staticmethod
    def _annotate(
        matrix: _Matrix,
        keep: Optional[np.ndarray],
        options: Options,
        label: Optional[str],
    ) -> _Output:
        def pick(values: np.ndarray) -> np.ndarray:
            return values if keep is None else values[keep]

        columns = {key: pick(values) for key, values in matrix.columns.items()}
        if options.var_names == "gene_ids":
            # As scanpy's read_10x_mtx does it: ids become the names, and the
            # symbols move into a column where the ids were.
            index = list(columns["gene_ids"])
            columns = {
                ("gene_symbols" if key == "gene_ids" else key): (
                    pick(matrix.names) if key == "gene_ids" else values
                )
                for key, values in columns.items()
            }
        else:
            index = list(pick(matrix.names))
        if options.make_unique:
            index = make_index_unique(index)
        return _Output(label=label, keep=keep, var_names=index, columns=columns)

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail = plan.detail
        matrix: _Matrix = detail["matrix"]
        targets: List[_Output] = detail["targets"]

        with h5py.File(plan.source, "r") as handle, ExitStack() as stack:
            group = handle[matrix.group]
            roots = [
                stack.enter_context(
                    open_store(path, "w", zarr_format=detail["zarr_format"])
                ).root
                for path in staged
            ]
            for root, target in zip(roots, targets):
                with console.status("[cyan]Writing obs and var...[/]"):
                    _write_obs(group, root, detail["filtered_barcodes"])
                    _write_var(root, target)
                ensure_anndata_skeleton(root)
                _write_uns(root, matrix)

            # One pass over the matrix, shared out among every output.
            with console.status("[cyan]Copying X...[/]"):
                fan_out_sparse_matrix(
                    _AsCsr(group, (matrix.n_barcodes, matrix.n_features)),
                    "X",
                    [(root, None, target.keep) for root, target in zip(roots, targets)],
                )


def _write_obs(group: h5py.Group, root: Any, filtered_barcodes: bool) -> None:
    """obs: the barcodes, streamed, and Space Ranger's `filtered_barcodes`."""
    obs = root.create_group("obs")
    spec.set_encoding(obs, spec.DATAFRAME)
    obs.attrs["_index"] = "_index"

    barcodes = group["barcodes"]
    n = barcodes.shape[0]
    backend = "zarr" if is_zarr_group(root) else "hdf5"
    index = create_dataset(obs, "_index", shape=(n,), dtype=string_dtype_for(backend))
    for lo in range(0, n, _STRING_STEP):
        hi = min(lo + _STRING_STEP, n)
        index[lo:hi] = np.asarray(_strings_slice(barcodes, lo, hi), dtype=object)
    spec.set_encoding(index, spec.STRING_ARRAY)

    columns = []
    if filtered_barcodes:
        write_dense(obs, "filtered_barcodes", np.asarray(group["filtered_barcodes"][...]).astype(bool))
        columns.append("filtered_barcodes")
    set_column_order(obs, columns)


def _strings_slice(dataset: Any, lo: int, hi: int) -> List[str]:
    return [str(v) for v in decode_str_array(np.asarray(dataset[lo:hi]))]


def _write_var(root: Any, target: _Output) -> None:
    """var, with string columns encoded as anndata's writer would encode them.

    anndata turns any string column with a repeated value into a categorical
    when it writes (`strings_to_categoricals`), so an h5ad scanpy wrote from
    the same file holds `feature_types` and `genome` as categoricals. Doing
    the same keeps the two interchangeable, and those columns a few bytes
    instead of a string per feature.
    """
    var = write_dataframe_header(root, "var", target.var_names, list(target.columns))
    for key, values in target.columns.items():
        if values.dtype.kind == "b":
            write_dense(var, key, values)
            continue
        categories, codes = np.unique(values, return_inverse=True)
        if len(categories) < len(values):
            write_categorical(var, key, codes, list(categories))
        else:
            write_string_array(var, key, values)


def _write_uns(root: Any, matrix: _Matrix) -> None:
    """The run's metadata, which scanpy drops and an export puts back."""
    meta = write_mapping(root["uns"], "cellranger")
    write_scalar(meta, "layout", matrix.version)
    if matrix.version == 2:
        write_scalar(meta, "genome", matrix.group)
    if matrix.tag_keys:
        write_string_array(meta, "feature_tag_keys", matrix.tag_keys)
    if matrix.attrs:
        attrs = write_mapping(meta, "attrs")
        for key, value in matrix.attrs.items():
            _write_value(attrs, key, value)


def _write_value(parent: Any, key: str, value: Any) -> None:
    """An HDF5 attribute value, as the AnnData element that holds it."""
    if isinstance(value, (bytes, np.bytes_, str)):
        text = value.decode("utf-8", "replace") if isinstance(value, (bytes, np.bytes_)) else value
        write_scalar(parent, key, text)
        return
    array = np.asarray(value)
    if array.ndim == 0:
        write_scalar(parent, key, array.item())
    elif array.dtype.kind in ("S", "O", "U"):
        write_string_array(parent, key, decode_str_array(array))
    else:
        write_dense(parent, key, array)


# ---------------------------------------------------------------------------
# export


def _matrix_path(layer: Optional[str]) -> str:
    """`--layer` as a path: X, raw/X, a layer's name, or layers/NAME."""
    if layer is None or layer in ("X", "raw/X") or layer.startswith("layers/"):
        return layer or "X"
    return f"layers/{layer}"


def _resolve(root: Any, path: str) -> Any:
    node = root
    for part in path.split("/"):
        if not is_group(node) or part not in node:
            raise KeyError(f"{path!r} not found in the store.")
        node = node[part]
    return node


def _read_value(obj: Any) -> Any:
    """An AnnData element read back as the attribute value it came from."""
    enc = spec.encoding_type(obj)
    if enc == spec.STRING_ARRAY:
        return np.array([s.encode("utf-8") for s in read_str_all(obj)])
    value = obj[()] if getattr(obj, "shape", None) == () else np.asarray(obj[...])
    # Cell Ranger writes a single string as variable-length and an array of
    # them as fixed-width; h5py does the same with a str and an `S` array.
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return str(value)
    return value


@dataclass
class _Features:
    ids: List[str]
    names: List[str]
    types: List[str]
    genomes: List[str]
    tags: Dict[str, List[str]]


class TenxExport(Converter):
    options = frozenset(
        {"--layer", "--gene-ids-column", "--gene-names-column", "--feature-types-column"}
    )

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        path = _matrix_path(options.layer)
        warnings: List[str] = []
        with open_store(source, "r", require_anndata=False) as store:
            root = store.root
            matrix = describe(_resolve(root, path))
            var_path = "raw/var" if path.startswith("raw/") else "var"
            var, obs = _resolve(root, var_path), _resolve(root, "obs")
            obs_index, _ = resolve_index(obs, "obs")
            var_index, index_name = resolve_index(var, "var")
            n_obs, n_var = element_len(obs_index), element_len(var_index)
            if matrix.shape != (n_obs, n_var):
                raise ValueError(
                    f"{path} is {matrix.shape[0]} x {matrix.shape[1]}, but obs has "
                    f"{n_obs} and {var_path} has {n_var} entries."
                )

            features = _features(root, var, var_index, options, warnings)
            attrs = _run_attrs(root)

        if matrix.dtype.kind == "f":
            warnings.append(
                f"{path} holds {matrix.dtype} values. Cell Ranger writes integer "
                "counts, and tools that read its output may expect them: export "
                "a counts layer with --layer, or recast to int32 first."
            )

        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"{src_format} → 10x h5, {n_obs:,} barcodes x {n_var:,} features "
                f"from {path}"
            ),
            warnings=warnings,
            detail={"path": path, "features": features, "attrs": attrs},
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail = plan.detail
        features: _Features = detail["features"]
        with open_store(plan.source, "r", require_anndata=False) as store, h5py.File(
            staged[0], "w"
        ) as out:
            root = store.root
            matrix = describe(_resolve(root, detail["path"]))
            with console.status(f"[cyan]Copying {detail['path']}...[/]"):
                group = _write_matrix(matrix, out)
            n_obs, n_var = matrix.shape
            group.create_dataset("shape", data=np.array([n_var, n_obs], dtype=np.int32))

            obs = root["obs"]
            obs_index, _ = resolve_index(obs, "obs")
            _write_fixed_strings_streamed(group, "barcodes", obs_index, obs)

            feats = group.create_group("features")
            for key, values in (
                ("id", features.ids),
                ("name", features.names),
                ("feature_type", features.types),
                ("genome", features.genomes),
                *features.tags.items(),
            ):
                _write_fixed_strings(feats, key, values)
            _write_fixed_strings(feats, "_all_tag_keys", ["genome", *features.tags])

            for key, value in detail["attrs"].items():
                out.attrs[key] = value


def _features(
    root: Any,
    var: Any,
    var_index: Any,
    options: Options,
    warnings: List[str],
) -> _Features:
    """Work out each feature's id, name, type and genome from var."""
    columns = dataframe_columns(var)
    n = element_len(var_index)

    def column(name: str) -> List[str]:
        return read_str_all(var[name], parent_group=var)

    def requested(flag: str, name: Optional[str]) -> Optional[str]:
        if name is not None and name not in columns:
            raise ValueError(
                f"{flag} {name}: no such var column. It has: "
                f"{', '.join(columns) or 'none'}."
            )
        return name

    ids_col = requested("--gene-ids-column", options.gene_ids_column)
    names_col = requested("--gene-names-column", options.gene_names_column)
    types_col = requested("--feature-types-column", options.feature_types_column)
    index = read_str_all(var_index)

    missing = []
    if ids_col is None and names_col is None and "gene_ids" not in columns and "gene_symbols" in columns:
        # var_names are ids already, as `--var-names gene_ids` leaves them.
        ids, names = index, column("gene_symbols")
    else:
        names = column(names_col) if names_col else index
        if ids_col is None and "gene_ids" in columns:
            ids_col = "gene_ids"
        if ids_col is not None:
            ids = column(ids_col)
        else:
            missing.append("gene ids (a gene_ids column, or --gene-ids-column)")
            ids = names
    if types_col is None and "feature_types" in columns:
        types_col = "feature_types"
    if types_col is not None:
        types = column(types_col)
    else:
        missing.append("feature types (a feature_types column, or --feature-types-column)")
        types = ["Gene Expression"] * n

    if missing:
        if not options.force:
            raise ValueError(
                f"var has no {' and no '.join(missing)}, which a 10x matrix "
                "records for every feature. Pass --force to write placeholders: "
                "the names as ids, and 'Gene Expression' as the type."
            )
        warnings.append(f"Writing placeholders for {' and '.join(missing)}.")

    genomes = column("genome") if "genome" in columns else [""] * n
    tags: Dict[str, List[str]] = {}
    for key in _recorded_tag_keys(root):
        if key != "genome" and key in columns:
            tags[key] = column(key)
    return _Features(ids=ids, names=names, types=types, genomes=genomes, tags=tags)


def _cellranger_meta(root: Any) -> Optional[Any]:
    """`uns/cellranger`, as an import recorded it, if there is one."""
    if "uns" not in root or "cellranger" not in root["uns"]:
        return None
    meta = root["uns"]["cellranger"]
    return meta if is_group(meta) else None


def _recorded_tag_keys(root: Any) -> List[str]:
    meta = _cellranger_meta(root)
    if meta is None or "feature_tag_keys" not in meta:
        return []
    return read_str_all(meta["feature_tag_keys"])


def _run_attrs(root: Any) -> Dict[str, Any]:
    """Root attributes: the recorded run's, with Cell Ranger's format markers."""
    attrs: Dict[str, Any] = {}
    meta = _cellranger_meta(root)
    if meta is not None and "attrs" in meta:
        for key in _RUN_ATTRS:
            if key in meta["attrs"] and is_dataset(meta["attrs"][key]):
                attrs[key] = _read_value(meta["attrs"][key])
    # Always the current format, whatever the source recorded: a v2 source's
    # attributes describe a layout this file does not have.
    attrs["filetype"] = "matrix"
    attrs["version"] = np.int64(2)
    return attrs


def _write_matrix(matrix: Any, out: h5py.File) -> h5py.Group:
    """`/matrix/{data,indices,indptr}`, gzip-compressed whatever the source used.

    lzf and Blosc are not readable by PyTables or R's hdf5r, which is how
    CellBender and Seurat read these files.
    """
    index_dtype = np.dtype("int64")
    if matrix.kind == spec.CSR_MATRIX:
        group = out.create_group("matrix")
        for key in ("data", "indices", "indptr"):
            copy_dataset(matrix.obj[key], group, key, compression="gzip")
            for attr in list(group[key].attrs):
                del group[key].attrs[attr]
        return group

    if matrix.kind == spec.CSC_MATRIX:
        transpose_sparse_streaming(
            matrix, out, "matrix", data_dtype=matrix.dtype,
            index_dtype=index_dtype, compression="gzip",
        )
    else:
        sparsify(
            matrix, out, "matrix", enc=spec.CSR_MATRIX, data_dtype=matrix.dtype,
            index_dtype=index_dtype, compression="gzip",
        )
    group = out["matrix"]
    for attr in list(group.attrs):
        del group.attrs[attr]
    return group


def _fixed_width(values: Sequence[str]) -> int:
    return max([1, *(len(v.encode("utf-8")) for v in values)])


def _create_fixed(parent: h5py.Group, name: str, n: int, width: int) -> h5py.Dataset:
    kw: Dict[str, Any] = {}
    if n:
        kw = {"chunks": (min(n, 1 << 16),), "compression": "gzip", "compression_opts": GZIP_LEVEL}
    return parent.create_dataset(name, shape=(n,), dtype=f"S{width}", **kw)


def _write_fixed_strings(parent: h5py.Group, name: str, values: Sequence[str]) -> None:
    """Fixed-width bytes, as Cell Ranger writes them.

    Not variable-length strings: PyTables, which CellBender reads these
    files with, does not support them.
    """
    data = np.array([v.encode("utf-8") for v in values], dtype=f"S{_fixed_width(values)}")
    _create_fixed(parent, name, len(values), data.dtype.itemsize)[...] = data


def _write_fixed_strings_streamed(
    parent: h5py.Group, name: str, column: Any, frame: Any
) -> None:
    """Like `_write_fixed_strings`, for a column too long to hold: one pass
    for the width, one for the data."""
    n = element_len(column)
    cache: Dict[str, np.ndarray] = {}
    width = 1
    for lo in range(0, n, _STRING_STEP):
        block = read_str_chunk(column, lo, min(lo + _STRING_STEP, n), cache, frame)
        width = max(width, _fixed_width(block))
    dataset = _create_fixed(parent, name, n, width)
    for lo in range(0, n, _STRING_STEP):
        hi = min(lo + _STRING_STEP, n)
        block = read_str_chunk(column, lo, hi, cache, frame)
        dataset[lo:hi] = np.array([v.encode("utf-8") for v in block], dtype=f"S{width}")

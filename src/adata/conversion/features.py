"""The feature table shared by every 10x-style format.

A 10x HDF5 matrix and a 10x MTX folder record the same things about each
feature -- an id, a name, a type -- and import and export them the same way.
What is common lives here, so `tenx` and `mtx` differ only in how they read
and write their files:

* import: which features each output keeps (`plan_targets`), how they are
  named (`annotate`), and how obs and var are written (`write_obs_index`,
  `write_var`);
* export: which matrix to write (`plan_matrix_source`) and each feature's id,
  name and type, worked out from var (`export_features`).

With `--var-names gene_symbols`, the AnnData produced has the names and
columns scanpy's `read_10x_h5` and `read_10x_mtx` give with `gex_only=False`;
the default names features by gene id instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from adata.conversion.base import Options
from adata.core.recast import describe
from adata.elements import spec
from adata.elements.read import (
    dataframe_columns,
    element_len,
    read_str_all,
    resolve_index,
)
from adata.elements.strings import string_dtype_for
from adata.elements.write import (
    write_categorical,
    write_dataframe_header,
    write_dense,
    write_string_array,
)
from adata.storage import create_dataset, is_dataset, is_group, is_zarr_group, open_store
from adata.util.names import (
    count_duplicates,
    make_index_unique,
    natural_sorted,
    unique_names,
)

VAR_NAMES = ("gene_symbols", "gene_ids")

#: Rows of a string column handled per read when streaming one.
STRING_STEP = 1 << 18


# ---------------------------------------------------------------------------
# import


@dataclass
class Target:
    """One AnnData to write: which features it keeps, and their annotation."""

    label: Optional[str]
    keep: Optional[np.ndarray]
    var_names: List[str]
    columns: Dict[str, np.ndarray]


def type_counts(types: np.ndarray) -> List[Tuple[str, int]]:
    """Feature types in order of first appearance, with how many of each."""
    labels, first, counts = np.unique(types, return_index=True, return_counts=True)
    order = np.argsort(first)
    return [(str(labels[i]), int(counts[i])) for i in order]


def check_import_options(options: Options) -> None:
    if options.var_names not in VAR_NAMES:
        raise ValueError(f"--var-names must be one of: {', '.join(VAR_NAMES)}.")
    if options.feature_type is not None and options.split_feature_types:
        raise ValueError(
            "--feature-type keeps one type and --split-feature-types writes "
            "every type separately; pass one or the other."
        )


def annotate(
    names: np.ndarray,
    columns: Dict[str, np.ndarray],
    keep: Optional[np.ndarray],
    options: Options,
    label: Optional[str],
) -> Target:
    """var for the features in `keep`: their names, and their columns."""

    def pick(values: np.ndarray) -> np.ndarray:
        return values if keep is None else values[keep]

    picked = {key: pick(values) for key, values in columns.items()}
    ids = picked.get("gene_ids")
    if options.var_names == "gene_ids":
        # As scanpy's read_10x_mtx does it: ids become the names, and the
        # symbols move into a column where the ids were.
        index = list(ids)
        picked = {
            ("gene_symbols" if key == "gene_ids" else key): (
                pick(names) if key == "gene_ids" else values
            )
            for key, values in picked.items()
        }
    else:
        index = list(pick(names))
    if options.make_unique:
        unique = make_index_unique(index)
        if options.var_names == "gene_ids" and unique != index:
            # Ids that repeat -- a probe-level matrix's, several probes to a
            # gene -- are not ids once suffixed. The originals are kept
            # beside the symbols, so an export writes them rather than the
            # suffixed names.
            kept: Dict[str, np.ndarray] = {}
            for key, values in picked.items():
                kept[key] = values
                if key == "gene_symbols":
                    kept["gene_ids"] = ids
            picked = kept
        index = unique
    return Target(label=label, keep=keep, var_names=index, columns=picked)


def plan_targets(
    output: Path,
    names: np.ndarray,
    columns: Dict[str, np.ndarray],
    types: Optional[np.ndarray],
    keep: np.ndarray,
    options: Options,
    *,
    noun: str,
) -> Tuple[List[Path], List[Target], List[str]]:
    """The outputs to write, the features each keeps, and any warnings.

    One output, unless `--split-feature-types` asks for one per type. `keep`
    is what survives filters the caller applied first, such as `--genome`;
    `noun` names the input in messages ("file", "folder").
    """
    groups: List[Tuple[Optional[str], np.ndarray]] = []
    if options.feature_type is not None:
        available = [t for t, _ in type_counts(types)]
        if options.feature_type not in available:
            raise ValueError(
                f"No features of type {options.feature_type!r}; this {noun} "
                f"has: {', '.join(available)}."
            )
        mask = keep & (types == options.feature_type)
        if not mask.any():
            # The type exists, but not among the features an earlier filter
            # kept: writing would give an output of no features at all.
            kept = [t for t, _ in type_counts(types[keep])]
            among = (
                f"in genome {options.genome!r}" if options.genome is not None
                else "among the features kept"
            )
            raise ValueError(
                f"No features of type {options.feature_type!r} {among}; "
                f"{'its' if options.genome is not None else 'their'} types are: "
                f"{', '.join(kept)}."
            )
        groups.append((options.feature_type, mask))
    elif options.split_feature_types:
        for label, _ in type_counts(types[keep]):
            groups.append((label, keep & (types == label)))
    else:
        groups.append((None, keep))

    stems = unique_names([label for label, _ in groups if label is not None])
    outputs: List[Path] = []
    targets: List[Target] = []
    warnings: List[str] = []
    for label, mask in groups:
        path = output
        if options.split_feature_types:
            path = output.with_name(f"{output.stem}.{stems[label]}{output.suffix}")
        selection = None if mask.all() else np.flatnonzero(mask)
        target = annotate(names, columns, selection, options, label)
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
    return outputs, targets, warnings


def describe_import(
    n_barcodes: int,
    n_features: int,
    types: Optional[np.ndarray],
    targets: Sequence[Target],
) -> str:
    """`1,222 barcodes x 33,538 features (Gene Expression 33,538)`, and so on."""
    described = f"{n_barcodes:,} barcodes x {n_features:,} features"
    if types is not None:
        described += " (" + ", ".join(f"{t} {n:,}" for t, n in type_counts(types)) + ")"
    if len(targets) > 1:
        described += f", into {len(targets)} outputs by feature type"
    elif targets[0].keep is not None:
        described += f", keeping {len(targets[0].keep):,} features"
    return described


def write_obs_index(root: Any, n: int, blocks: Iterable[Sequence[str]]) -> Any:
    """obs, holding only its index: the barcodes, written a block at a time.

    Returns the group, for the caller to add columns to. A source that gives
    more or fewer than `n` barcodes is refused.
    """
    return write_index_frame(root, "obs", n, blocks, noun="barcodes")


def write_index_frame(
    parent: Any, name: str, n: int, blocks: Iterable[Sequence[str]], *, noun: str = "names"
) -> Any:
    """A `dataframe` holding only its index, written a block at a time.

    Returns the group, for the caller to add columns to; `noun` names the
    index in the refusal when the blocks give more or fewer than `n`.
    """
    frame = parent.create_group(name)
    spec.set_encoding(frame, spec.DATAFRAME)
    frame.attrs["_index"] = "_index"

    backend = "zarr" if is_zarr_group(parent) else "hdf5"
    index = create_dataset(frame, "_index", shape=(n,), dtype=string_dtype_for(backend))
    written = 0
    for block in blocks:
        if written + len(block) > n:
            raise ValueError(f"There are more than {n:,} {noun}.")
        index[written : written + len(block)] = np.asarray(block, dtype=object)
        written += len(block)
    if written != n:
        raise ValueError(f"There are {written:,} {noun}, not {n:,}.")
    spec.set_encoding(index, spec.STRING_ARRAY)
    return frame


def write_var(root: Any, target: Target) -> None:
    """var, with string columns encoded as anndata's writer would encode them.

    anndata turns any string column with a repeated value into a categorical
    when it writes (`strings_to_categoricals`), so an h5ad scanpy wrote from
    the same file holds `feature_types` and `genome` as categoricals. Doing
    the same keeps the two interchangeable, and those columns a few bytes
    instead of a string per feature.
    """
    var = write_dataframe_header(root, "var", target.var_names, list(target.columns))
    for key, values in target.columns.items():
        if values.dtype.kind in "biuf":
            write_dense(var, key, values)
            continue
        categories = natural_sorted(set(values.tolist()))
        if len(categories) < len(values):
            lookup = {c: i for i, c in enumerate(categories)}
            write_categorical(var, key, [lookup[v] for v in values.tolist()], categories)
        else:
            write_string_array(var, key, values)


# ---------------------------------------------------------------------------
# export


def matrix_path(layer: Optional[str]) -> str:
    """`--layer` as a path: X, raw/X, a layer's name, or layers/NAME."""
    if layer is None or layer in ("X", "raw/X") or layer.startswith("layers/"):
        return layer or "X"
    return f"layers/{layer}"


def resolve(root: Any, path: str) -> Any:
    node = root
    for part in path.split("/"):
        if not is_group(node) or part not in node:
            raise KeyError(f"{path!r} not found in the store.")
        node = node[part]
    return node


@dataclass
class ExportFeatures:
    """Each feature's id, name and type, and every other field to write.

    `fields` hold the other per-feature datasets in the order they are
    written: `genome` always, a probe-level matrix's `gene_id`, then those an
    import recorded. Booleans stay arrays of bool, as Cell Ranger stores
    them; the rest are strings. `tag_keys` is `_all_tag_keys`: which of them
    Cell Ranger counts as tags.
    """

    ids: List[str]
    names: List[str]
    types: List[str]
    fields: Dict[str, Any]
    tag_keys: List[str]


@dataclass
class MatrixSource:
    """The matrix an export writes, and what is known about it before writing."""

    path: str
    n_obs: int
    n_var: int
    dtype: np.dtype
    features: ExportFeatures


def plan_matrix_source(
    source: Path, options: Options, warnings: List[str]
) -> MatrixSource:
    """Check the matrix `--layer` names against obs and var; read var's features.

    Warns when the matrix holds floats, which 10x formats are not expected to.
    """
    path = matrix_path(options.layer)
    with open_store(source, "r", require_anndata=False) as store:
        root = store.root
        matrix = describe(resolve(root, path))
        var_path = "raw/var" if path.startswith("raw/") else "var"
        var, obs = resolve(root, var_path), resolve(root, "obs")
        obs_index, _ = resolve_index(obs, "obs")
        var_index, _ = resolve_index(var, "var")
        n_obs, n_var = element_len(obs_index), element_len(var_index)
        if matrix.shape != (n_obs, n_var):
            raise ValueError(
                f"{path} is {matrix.shape[0]} x {matrix.shape[1]}, but obs has "
                f"{n_obs} and {var_path} has {n_var} entries."
            )
        features = export_features(root, var, var_index, options, warnings)

    if matrix.dtype.kind == "f":
        warnings.append(
            f"{path} holds {matrix.dtype} values. Cell Ranger writes integer "
            "counts, and tools that read its output may expect them: export "
            "a counts layer with --layer, or recast to int32 first."
        )
    return MatrixSource(
        path=path, n_obs=n_obs, n_var=n_var, dtype=matrix.dtype, features=features
    )


def export_features(
    root: Any,
    var: Any,
    var_index: Any,
    options: Options,
    warnings: List[str],
) -> ExportFeatures:
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
    if ids_col is None and "gene_ids" in columns:
        ids_col = "gene_ids"
    if names_col is None and "gene_symbols" in columns:
        names_col = "gene_symbols"
    if types_col is None and "feature_types" in columns:
        types_col = "feature_types"
    index = read_str_all(var_index)

    # Ids and names each come from their column, and the var names stand in
    # for whichever has none: ids when the names have a column, as
    # `--var-names gene_ids` leaves var, and names otherwise.
    missing = []
    names = column(names_col) if names_col else index
    if ids_col is not None:
        ids = column(ids_col)
    elif names_col is not None:
        ids = index
    else:
        missing.append("gene ids (a gene_ids column, or --gene-ids-column)")
        ids = names
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

    fields: Dict[str, Any] = {"genome": column("genome") if "genome" in columns else [""] * n}
    if "probe_ids" in columns:
        # A probe-level (Flex) matrix: Cell Ranger's `id` names the probe,
        # and `gene_id` the gene it measures.
        fields["gene_id"] = ids
        ids = column("probe_ids")
    tags = recorded_tag_keys(root)
    recorded = recorded_feature_keys(root)
    for key in tags if recorded is None else recorded:
        if key not in fields and key not in _EXPORT_CORE and key in columns:
            fields[key] = _field(var, key)
    tag_keys = [key for key in tags if key in fields] if tags else list(fields)
    if "genome" not in tag_keys:
        tag_keys.insert(0, "genome")
    return ExportFeatures(ids=ids, names=names, types=types, fields=fields, tag_keys=tag_keys)


#: var columns that are Cell Ranger's `id`, `name` or `feature_type`, not a
#: field of their own.
_EXPORT_CORE = frozenset({"gene_ids", "gene_symbols", "probe_ids", "feature_types"})


def _field(var: Any, key: str) -> Any:
    """A var column as a feature dataset holds it: bool, or strings."""
    obj = var[key]
    if is_dataset(obj) and "categories" not in obj.attrs and obj.dtype.kind == "b":
        return np.asarray(obj[...], dtype=bool)
    return read_str_all(obj, parent_group=var)


def cellranger_meta(root: Any) -> Optional[Any]:
    """`uns/cellranger`, as an import recorded it, if there is one."""
    if "uns" not in root or "cellranger" not in root["uns"]:
        return None
    meta = root["uns"]["cellranger"]
    return meta if is_group(meta) else None


def recorded_tag_keys(root: Any) -> List[str]:
    meta = cellranger_meta(root)
    if meta is None or "feature_tag_keys" not in meta:
        return []
    return read_str_all(meta["feature_tag_keys"])


def recorded_feature_keys(root: Any) -> Optional[List[str]]:
    """The feature datasets an import read into var, tags or not; None
    from an import that did not record them, whose tags are all it kept."""
    meta = cellranger_meta(root)
    if meta is None or "feature_keys" not in meta:
        return None
    return read_str_all(meta["feature_keys"])


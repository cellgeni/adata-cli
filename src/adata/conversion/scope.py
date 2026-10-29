"""SCope looms: the results pySCENIC, SCENIC+ and pycistopic write for the
SCope viewer, as VSN and R's SCopeLoomR write them too.

A SCope loom is a loom with more in it:

* compound attributes with a field per regulon: each cell's AUC
  (`col_attrs/RegulonsAUC`) and each gene's or region's membership
  (`row_attrs/Regulons`). These, and any other compound attribute (VSN's
  `ClusterMarkers_<i>`, SCopeLoomR's `MotifRegulonsAUC`), are read by
  loom.py as dataframes in obsm and varm, under their own names;
* `col_attrs/Embeddings_X` and `Embeddings_Y`, a field per embedding id,
  `-1` being the default embedding, which `Embedding` holds again as `_X`
  and `_Y`. Read here, as `obsm["X_<name>"]`;
* `col_attrs/Clusterings`, a field of cluster ids per clustering. Read
  here, as categorical obs columns labelled with the clusters' descriptions;
* a global `MetaData` attribute: JSON, or base64 of it compressed (zlib
  from pySCENIC, gzip from SCopeLoomR's spec 2 looms), naming the
  embeddings, the clusterings and their clusters, and holding the regulon
  thresholds. It is decoded into `uns["scope"]`, inflated only up to
  `MAX_METADATA_BYTES`.

Embeddings and clusterings are read a block of cells at a time, each
attribute once for all of them.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np

from adata.elements import spec
from adata.elements.write import (
    create_categorical,
    write_dataframe_header,
    write_dense,
    write_mapping,
    write_scalar,
    write_string_array,
)
from adata.storage import create_dataset, is_dataset, is_group
from adata.util.names import claim, make_index_unique, sanitize_label, storable

METADATA = "MetaData"
EMBEDDING = "Embedding"
EMBEDDINGS_X, EMBEDDINGS_Y = "Embeddings_X", "Embeddings_Y"
CLUSTERINGS = "Clusterings"

#: The regulon thresholds' fields kept as columns. `allThresholds`, a
#: mapping per regulon, stays in the metadata text.
THRESHOLD_COLUMNS = ("defaultThresholdValue", "defaultThresholdName", "motifData")

#: The most bytes compressed `MetaData` may inflate to. pySCENIC's run to
#: a few thousand regulons stays under a megabyte; past this the attribute
#: is malformed, and kept as the text it came as.
MAX_METADATA_BYTES = 64 << 20


def is_scope(global_attrs: Iterable[str], col_attrs: Iterable[str]) -> bool:
    """A `MetaData` global attribute, or a column attribute only SCope writes."""
    return METADATA in set(global_attrs) or any(
        name in (EMBEDDINGS_X, CLUSTERINGS) or name.endswith("RegulonsAUC")
        for name in col_attrs
    )


def read_metadata(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """`MetaData` as (its JSON, the JSON's text).

    The attribute is JSON, or base64 of compressed JSON: zlib from pySCENIC,
    gzip from SCopeLoomR. When it is neither, or inflates to more than
    `MAX_METADATA_BYTES`, the JSON is None and the text is returned as it
    came.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        try:
            decoded = _inflate(base64.b64decode(text)).decode("utf-8")
            parsed = json.loads(decoded)
        except (ValueError, zlib.error):
            return None, text
        text = decoded
    return (parsed if isinstance(parsed, dict) else None), text


def _inflate(data: bytes) -> bytes:
    """`data` decompressed, as `zlib.decompress(data, 47)` would, but never
    more than `MAX_METADATA_BYTES` of it (47: a zlib or a gzip header,
    whichever is there)."""
    inflater = zlib.decompressobj(47)
    out = inflater.decompress(data, MAX_METADATA_BYTES + 1)
    if len(out) > MAX_METADATA_BYTES:
        raise ValueError(f"MetaData inflates to more than {MAX_METADATA_BYTES:,} bytes.")
    if not inflater.eof:
        raise zlib.error("MetaData's compressed stream is cut short.")
    return out


@dataclass
class Embedding:
    id: str
    name: Optional[str]
    #: Where it goes in obsm.
    key: str
    #: The (attribute, field) holding each coordinate.
    x: Tuple[str, str]
    y: Tuple[str, str]


@dataclass
class Clustering:
    id: str
    name: Optional[str]
    #: Where it goes in obs.
    column: str
    #: The cluster ids, in order, and their labels; empty when MetaData
    #: describes no clusters, which are then the ids found in the file.
    ids: List[float]
    labels: List[str]


@dataclass
class Scope:
    #: MetaData's JSON text, None when there is no MetaData.
    metadata: Optional[str]
    parsed: Dict[str, Any]
    embeddings: List[Embedding] = field(default_factory=list)
    clusterings: List[Clustering] = field(default_factory=list)
    regulons: int = 0

    def describe(self) -> str:
        """`298 regulons, 5 embeddings, 1 clustering`, leaving out what is not there."""
        counts = (
            (self.regulons, "regulon"),
            (len(self.embeddings), "embedding"),
            (len(self.clusterings), "clustering"),
        )
        return ", ".join(f"{n:,} {noun}{'' if n == 1 else 's'}" for n, noun in counts if n)


def plan(
    cols: Any, metadata: Optional[str], obs: Any, var: Any, warnings: List[str]
) -> Scope:
    """Where a SCope loom's embeddings and clusterings go, and its MetaData.

    `obs` and `var` are loom.py's axes. The attributes read here are taken
    out of `obs.tables`, so they are not read again as dataframes, and the
    names they get are chosen around obs's columns and obsm's other keys.
    Attributes these rules do not fit are left as dataframes.
    """
    parsed: Dict[str, Any] = {}
    if metadata is not None:
        decoded, metadata = read_metadata(metadata)
        if decoded is None:
            warnings.append(
                "MetaData is neither JSON nor compressed JSON (of at most "
                f"{MAX_METADATA_BYTES:,} bytes): it is kept as text in "
                'uns["scope"]["metadata"], and embeddings and clusterings are named by id.'
            )
        parsed = decoded or {}
    found = Scope(metadata=metadata, parsed=parsed)
    found.clusterings = _clusterings(cols, obs, parsed, warnings)
    found.embeddings = _embeddings(cols, obs, parsed)
    found.regulons = sum(
        len(columns) for name, columns in obs.tables.items() if name.endswith("RegulonsAUC")
    ) or sum(len(columns) for name, columns in var.tables.items() if name.endswith("Regulons"))
    return found


def _numeric(dtype: np.dtype) -> bool:
    return dtype.names is None and dtype.shape == () and dtype.kind in "biuf"


def _fields(cols: Any, obs: Any, name: str) -> Optional[Sequence[str]]:
    """The fields of compound attribute `name`, if every one holds a number."""
    if name not in obs.tables:
        return None
    dtype = cols[name].dtype
    return dtype.names if all(_numeric(dtype[f]) for f in dtype.names) else None


def _by_id(parsed: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    """MetaData's `embeddings` or `clusterings`, by id as text, as SCope
    names the attributes' fields (SCopeLoomR records ids as text too)."""
    entries = parsed.get(key)
    if not isinstance(entries, list):
        return {}
    return {str(e["id"]): e for e in entries if isinstance(e, dict) and "id" in e}


def _name(entry: Dict[str, Any]) -> Optional[str]:
    name = entry.get("name")
    return None if name is None or str(name) == "" else str(name)


def _embeddings(cols: Any, obs: Any, parsed: Dict[str, Any]) -> List[Embedding]:
    """Embeddings_X and _Y paired by id; `Embedding` as id -1 when they lack it."""
    pairs: List[Tuple[str, Tuple[str, str], Tuple[str, str]]] = []
    xs, ys = _fields(cols, obs, EMBEDDINGS_X), _fields(cols, obs, EMBEDDINGS_Y)
    if xs is not None and ys is not None and set(xs) == set(ys):
        pairs = [(id, (EMBEDDINGS_X, id), (EMBEDDINGS_Y, id)) for id in xs]
        del obs.tables[EMBEDDINGS_X], obs.tables[EMBEDDINGS_Y]
    default = _fields(cols, obs, EMBEDDING)
    if default is not None and set(default) == {"_X", "_Y"}:
        if all(id != "-1" for id, _, _ in pairs):
            pairs.insert(0, ("-1", (EMBEDDING, "_X"), (EMBEDDING, "_Y")))
        # Otherwise it repeats Embeddings_X and _Y's -1.
        del obs.tables[EMBEDDING]

    entries = _by_id(parsed, "embeddings")
    taken = set(obs.matrices) | set(obs.tables)
    embeddings = []
    for id, x, y in pairs:
        name = _name(entries.get(id, {}))
        key = claim(f"X_{sanitize_label(name)}" if name else f"X_embedding_{id}", taken)
        embeddings.append(Embedding(id=id, name=name, key=key, x=x, y=y))
    return embeddings


def _clusterings(
    cols: Any, obs: Any, parsed: Dict[str, Any], warnings: List[str]
) -> List[Clustering]:
    ids = _fields(cols, obs, CLUSTERINGS)
    if ids is None:
        return []
    del obs.tables[CLUSTERINGS]
    entries = _by_id(parsed, "clusterings")
    taken = set(obs.columns) | set(spec.RESERVED_DATAFRAME_KEYS)
    clusterings, renamed = [], []
    for id in ids:
        entry = entries.get(id, {})
        name = _name(entry)
        wanted = storable(name or f"clustering_{id}")
        column = claim(wanted, taken)
        if column != (name or wanted):
            renamed.append(f"{name or wanted!r} as {column!r}")
        clusters = _clusters(entry.get("clusters"))
        clusterings.append(
            Clustering(
                id=id, name=name, column=column, ids=[i for i, _ in clusters],
                labels=make_index_unique([label for _, label in clusters]),
            )
        )
    if renamed:
        warnings.append(
            f"Clusterings written under other obs names: {', '.join(renamed)} -- "
            "the name is taken, or HDF5 and Zarr cannot hold it."
        )
    return clusterings


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


def _clusters(entries: Any) -> List[Tuple[float, str]]:
    """A clustering's clusters as (id, label), in id order: the label is the
    cluster's description, or its id when it has none."""
    found: Dict[float, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        number = _number(entry.get("id")) if isinstance(entry, dict) else None
        if number is None or number in found:
            continue
        description = entry.get("description")
        found[number] = _label(number) if description in (None, "") else str(description)
    return sorted(found.items())


def _match(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each value's position in `keys`, which are sorted, or -1 where it is not one."""
    if not keys.size:
        return np.full(values.shape, -1, dtype=np.int64)
    at = np.minimum(np.searchsorted(keys, values), keys.size - 1)
    return np.where(keys[at] == values, at, -1)


def _cluster_ids(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """A clustering's values as cluster ids, as floats so a gap can be NaN.

    Old SCopeLoomR versions wrote a clustering as R's factor: an enum of
    "0", "1", ... over codes 1, 2, .... The labels are the ids.
    """
    mapping = h5py.check_enum_dtype(dtype)
    if not mapping:
        return np.asarray(values, dtype=np.float64)
    by_code = sorted((code, _number(label)) for label, code in mapping.items())
    codes = np.array([code for code, _ in by_code], dtype=np.float64)
    ids = np.array([np.nan if n is None else n for _, n in by_code], dtype=np.float64)
    at = _match(codes, np.asarray(values, dtype=np.float64))
    return np.where(at >= 0, ids[at], np.nan)


def write_clusterings(obs: Any, cols: Any, found: Scope, n: int, step: int) -> List[str]:
    """Each clustering as a categorical obs column, `step` cells at a time;
    returns the columns written.

    Clusterings is read once, or twice when a clustering's clusters are not
    described and are found in the file first. Values with no cluster are
    missing (code -1).
    """
    if not found.clusterings:
        return []
    ds = cols[CLUSTERINGS]

    def blocks() -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        for lo in range(0, n, step):
            block = ds[lo : lo + step]
            yield lo, {c.id: _cluster_ids(block[c.id], ds.dtype[c.id]) for c in found.clusterings}

    keys = {c.id: (np.asarray(c.ids, dtype=np.float64), c.labels) for c in found.clusterings}
    undescribed = [c.id for c in found.clusterings if not c.ids]
    if undescribed:
        seen: Dict[str, Set[float]] = {id: set() for id in undescribed}
        for _, values in blocks():
            for id in undescribed:
                block = values[id]
                seen[id].update(np.unique(block[~np.isnan(block)]).tolist())
        for id in undescribed:
            ordered = sorted(seen[id])
            keys[id] = (np.asarray(ordered, dtype=np.float64), [_label(i) for i in ordered])

    codes = {c.id: create_categorical(obs, c.column, keys[c.id][1], n) for c in found.clusterings}
    for lo, values in blocks():
        for id, target in codes.items():
            target[lo : lo + values[id].size] = _match(keys[id][0], values[id])
    return [c.column for c in found.clusterings]


def _plain(dtype: np.dtype) -> np.dtype:
    """The dtype without h5py's metadata (an enum's labels)."""
    return np.dtype(dtype.str)


def write_embeddings(obsm: Any, cols: Any, found: Scope, n: int, step: int) -> None:
    """Each embedding as an n x 2 array in obsm, `step` cells at a time, each
    attribute read once for all of them."""
    if not found.embeddings:
        return
    sources = {name: cols[name] for e in found.embeddings for name in (e.x[0], e.y[0])}
    targets = []
    for e in found.embeddings:
        dtype = np.result_type(
            _plain(sources[e.x[0]].dtype[e.x[1]]), _plain(sources[e.y[0]].dtype[e.y[1]])
        )
        target = create_dataset(obsm, e.key, shape=(n, 2), dtype=dtype)
        spec.set_encoding(target, spec.ARRAY)
        targets.append((target, dtype))
    for lo in range(0, n, step):
        blocks = {name: ds[lo : lo + step] for name, ds in sources.items()}
        for e, (target, dtype) in zip(found.embeddings, targets):
            x, y = blocks[e.x[0]][e.x[1]], blocks[e.y[0]][e.y[1]]
            target[lo : lo + x.size] = np.column_stack([x, y]).astype(dtype, copy=False)
        del blocks  # before the next are read


def write_uns(uns: Any, found: Scope) -> None:
    """`uns["scope"]`: MetaData's text, its regulon thresholds as a table, and
    each embedding's and clustering's id and name, with where it went."""
    group = write_mapping(uns, "scope")
    if found.metadata is not None:
        write_scalar(group, "metadata", found.metadata)
    _write_thresholds(group, found.parsed.get("regulonThresholds"))
    if found.embeddings:
        _write_frame(group, "embeddings", [e.id for e in found.embeddings], {
            "name": [e.name or "" for e in found.embeddings],
            "obsm": [e.key for e in found.embeddings],
        })
    if found.clusterings:
        _write_frame(group, "clusterings", [c.id for c in found.clusterings], {
            "name": [c.name or "" for c in found.clusterings],
            "obs": [c.column for c in found.clusterings],
        })


def _write_frame(parent: Any, name: str, index: List[str], columns: Dict[str, Any]) -> None:
    frame = write_dataframe_header(parent, name, index, list(columns))
    for key, values in columns.items():
        if isinstance(values, np.ndarray):
            write_dense(frame, key, values)
        else:
            write_string_array(frame, key, np.asarray(values, dtype=object))


def _float(value: Any) -> float:
    number = _number(value)
    return np.nan if number is None else number


def _write_thresholds(parent: Any, entries: Any) -> None:
    """The regulon thresholds as a dataframe, indexed by regulon."""
    rows = [e for e in entries if isinstance(e, dict) and "regulon" in e] if isinstance(entries, list) else []
    if not rows:
        return
    columns: Dict[str, Any] = {}
    for key in THRESHOLD_COLUMNS:
        if not any(key in row for row in rows):
            continue
        if key == "defaultThresholdValue":
            columns[key] = np.array([_float(row.get(key)) for row in rows], dtype=np.float64)
        else:
            columns[key] = ["" if row.get(key) is None else str(row[key]) for row in rows]
    _write_frame(parent, "regulon_thresholds", [str(row["regulon"]) for row in rows], columns)


# ---------------------------------------------------------------------------
# export: an AnnData as a SCope loom (`adata convert ... -o x.loom --scope`)
#
# What SCope's server reads, and must find consistent, decides the rules:
# compound attributes with numeric fields; embedding ids as field names, -1
# the default; clusterings as int codes with every cluster described in
# MetaData, clustering and cluster ids 0..n-1 in list order; regulon names
# without spaces, alike in the AUC and the Regulons attributes; markers only
# with their metrics described. An AnnData that came from a SCope loom keeps
# its ids and names, recorded in uns["scope"] on import.

#: Categories an obs column may have to be offered as a SCope annotation, the
#: length of SCope's colour list.
MAX_ANNOTATION_VALUES = 245

_MARKERS = re.compile(r"ClusterMarkers_(\d+)(?:_(.+))?$")


@dataclass
class ExportTable:
    #: "col" or "row": which attribute group it goes to.
    axis: str
    key: str
    #: The obsm or varm entry it comes from, and its numeric columns.
    source: str
    columns: List[str]
    fields: List[str]
    dtype: np.dtype


@dataclass
class ExportEmbedding:
    id: int
    name: str
    key: str


@dataclass
class ExportClustering:
    id: int
    column: str
    entry: Dict[str, Any]


@dataclass
class ScopeExport:
    tables: List[ExportTable]
    embeddings: List[ExportEmbedding]
    clusterings: List[ExportClustering]
    metadata: Dict[str, Any]
    cluster_id: bool
    regulons: int

    def describe(self) -> str:
        counts = ((self.regulons, "regulon"), (len(self.embeddings), "embedding"),
                  (len(self.clusterings), "clustering"))
        return ", ".join(f"{n:,} {noun}{'' if n == 1 else 's'}" for n, noun in counts if n)


def compound_fits(dtype: np.dtype) -> bool:
    """Whether HDF5 can hold `dtype`: a compound type must fit one object
    header message, 64 KiB, so a few hundred to a thousand fields."""
    import io

    with h5py.File(io.BytesIO(), "w") as handle:
        try:
            handle.create_dataset("probe", shape=(1,), dtype=dtype)
        except ValueError:
            return False
    return True


def _frame_rows(group: Any) -> Dict[str, Dict[str, str]]:
    """A small text dataframe (uns["scope"]["embeddings"]) as {index: {column: value}}."""
    from adata.elements.read import dataframe_columns, read_str_all, resolve_index

    index, name = resolve_index(group)
    labels = read_str_all(index)
    rows: Dict[str, Dict[str, str]] = {label: {} for label in labels}
    for column in dataframe_columns(group, name):
        for label, value in zip(labels, read_str_all(group[column], parent_group=group)):
            rows[label][column] = value
    return rows


def _recorded(root: Any) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, str]], Dict[str, Tuple[int, str]]]:
    """What a SCope import recorded: MetaData's JSON, and the obsm keys and obs
    columns of its embeddings and clusterings, each with its id and name."""
    uns = root["uns"] if "uns" in root else {}
    group = uns["scope"] if "scope" in uns else None
    parsed: Dict[str, Any] = {}
    embeddings: Dict[str, Tuple[int, str]] = {}
    clusterings: Dict[str, Tuple[int, str]] = {}
    if group is None:
        return parsed, embeddings, clusterings
    if "metadata" in group:
        decoded, _ = read_metadata(_scalar_text(group["metadata"]))
        parsed = decoded or {}
    for key, target, slot in (("embeddings", embeddings, "obsm"), ("clusterings", clusterings, "obs")):
        if key in group:
            for id, row in _frame_rows(group[key]).items():
                number = _number(id)
                if number is not None and row.get(slot):
                    target[row[slot]] = (int(number), row.get("name", ""))
    return parsed, embeddings, clusterings


def _scalar_text(ds: Any) -> str:
    value = ds[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _check_fits(fields: Sequence[Tuple[str, Any]], what: str, advice: str) -> None:
    """Refuse, in the plan, a compound attribute HDF5 could not create."""
    if fields and not compound_fits(np.dtype(list(fields))):
        raise ValueError(
            f"{what} are more than HDF5 can hold as one compound attribute "
            f"(its type must fit 64 KiB). {advice}"
        )


def clustering_columns(root: Any, clusterings: Optional[str]) -> List[str]:
    """The obs columns to write as clusterings: those a SCope import
    recorded, then those `--clusterings` names (checked by the plan)."""
    obs = root["obs"]
    wanted = [c for c in _recorded(root)[2] if c in obs]
    for column in (clusterings or "").split(","):
        column = column.strip()
        if column and column not in wanted:
            wanted.append(column)
    return wanted


def plan_export(
    root: Any,
    *,
    obs_columns: Sequence[str],
    varm_key: str,
    clusterings: Optional[str],
    default_embedding: Optional[str],
    warnings: List[str],
    owners: Optional[Dict[str, Dict[str, str]]] = None,
) -> ScopeExport:
    """Everything a SCope loom needs, worked out and checked before writing.

    `owners` holds the attribute names already given away (`claim_attr`): a
    table asking for one is left out before anything is derived from it, so
    the metadata only ever describes tables that are written.
    """
    from adata.elements.read import categories_dataset, dataframe_columns, read_categories, resolve_index

    parsed, recorded_embeddings, recorded_clusterings = _recorded(root)
    owners = owners if owners is not None else {"col_attrs": {}, "row_attrs": {}}

    # Compound tables: every dataframe in obsm and varm, numeric columns only.
    frames = []
    for axis, where in (("col", "obsm"), ("row", varm_key)):
        group = _resolve(root, where)
        for key in (group if group is not None else []):
            obj = group[key]
            if spec.encoding_type(obj) != spec.DATAFRAME:
                continue
            numeric, other = [], []
            for column in dataframe_columns(obj, resolve_index(obj)[1]):
                member = obj[column]
                ok = (spec.encoding_type(member) in (None, spec.ARRAY) and getattr(member, "ndim", 0) == 1
                      and member.dtype.kind in "biuf")
                (numeric if ok else other).append(column)
            if other:
                warnings.append(f"Not written in {where}/{key}: {', '.join(other)} -- a SCope "
                                "table holds numbers only.")
            # Marker tables claim their names once renumbered, below.
            if numeric and (axis == "row" and _MARKERS.match(key) or claim_attr(
                owners, "col_attrs" if axis == "col" else "row_attrs", key, f"{where}/{key}", warnings
            )):
                frames.append((axis, where, key, obj, numeric))

    # One field name per regulon across AUC and Regulons, so that when two
    # names meet once spaces go (`A B`, `A_B`) a regulon's scores and its
    # targets still share a field.
    is_regulons = lambda key: key.endswith("Regulons") or key.endswith("RegulonsAUC")  # noqa: E731
    regulon_fields: Dict[str, str] = {}
    taken_regulons: Set[str] = set()
    for _, _, key, _, numeric in frames:
        if is_regulons(key):
            for c in numeric:
                if c not in regulon_fields:
                    regulon_fields[c] = claim(c.replace(" ", "_"), taken_regulons)

    tables: List[ExportTable] = []
    for axis, where, key, obj, numeric in frames:
        regulons = is_regulons(key)
        if regulons:
            fields = [regulon_fields[c] for c in numeric]
        else:
            taken: Set[str] = set()
            fields = [claim(c, taken) for c in numeric]
        renamed = [f"{c!r} as {f!r}" for c, f in zip(numeric, fields) if c != f]
        if renamed:
            warnings.append(f"Renamed in {key}: {', '.join(renamed)} -- SCope's regulon names hold no spaces.")
        dtype = np.dtype([(f, np.dtype(obj[c].dtype.str)) for f, c in zip(fields, numeric)])
        if not compound_fits(dtype):
            raise ValueError(
                f"{where}/{key} has {len(fields):,} columns, more than HDF5 can hold "
                "as one compound attribute (its type must fit 64 KiB); pySCENIC's own "
                "export stops at the same limit. Keep fewer columns first."
            )
        tables.append(ExportTable(axis, key, f"{where}/{key}", numeric, fields, dtype))

    # Embeddings: obsm arrays of two numeric columns.
    obsm = root["obsm"] if "obsm" in root else {}
    candidates = [
        key for key in obsm
        if spec.encoding_type(obsm[key]) in (None, spec.ARRAY) and getattr(obsm[key], "ndim", 0) == 2
        and obsm[key].shape[1] == 2 and obsm[key].dtype.kind in "biuf"
    ]
    if default_embedding is not None and default_embedding not in candidates:
        raise ValueError(
            f"--default-embedding {default_embedding}: not an obsm array of two columns; "
            f"there are {', '.join(candidates) or 'none'}."
        )
    default = default_embedding or next(
        (k for k in candidates if recorded_embeddings.get(k, (None,))[0] == -1), None
    ) or next((k for k in ("X_umap", "X_UMAP", "X_tsne", "X_tSNE") if k in candidates), None) or (
        candidates[0] if candidates else None
    )
    used = {-1}
    ids: Dict[str, int] = {}
    for key in candidates:
        recorded = recorded_embeddings.get(key, (None,))[0]
        if key != default and recorded is not None and recorded not in used:
            ids[key] = recorded
            used.add(recorded)
    next_id = 0
    embeddings: List[ExportEmbedding] = []
    for key in candidates:
        if key == default:
            id = -1
        elif key in ids:
            id = ids[key]
        else:
            while next_id in used:
                next_id += 1
            id = next_id
            used.add(id)
        name = recorded_embeddings.get(key, (None, ""))[1] or (key[2:] if key.startswith("X_") else key)
        embeddings.append(ExportEmbedding(id, name, key))
    embeddings.sort(key=lambda e: e.id)
    _check_fits([(str(e.id), np.dtype(obsm[e.key].dtype.str)) for e in embeddings],
                f"{len(embeddings):,} embeddings (two-column obsm arrays)", "Keep fewer of them first.")

    # Clusterings: the recorded ones, and those --clusterings names.
    obs = root["obs"]
    wanted = clustering_columns(root, clusterings)
    entries = {str(e.get("id")): e for e in parsed.get("clusterings", []) if isinstance(e, dict)}
    found: List[ExportClustering] = []
    renumbered: Dict[int, int] = {}
    #: Per clustering written, its recorded cluster ids (as marker fields
    #: name them) to the new ones, its categories' positions.
    cluster_fields: Dict[int, Dict[str, str]] = {}
    #: Per clustering written, the marker metrics its import recorded.
    recorded_metrics: Dict[int, List[Dict[str, Any]]] = {}
    for position, column in enumerate(wanted):
        if column not in obs:
            raise ValueError(f"--clusterings {column}: no such obs column.")
        if not is_categorical(obs[column]):
            raise ValueError(
                f"--clusterings {column}: not a categorical obs column, which SCope's "
                "clusterings need (cluster ids with descriptions)."
            )
        old, name = recorded_clusterings.get(column, (None, column))
        recorded = entries.get(str(old), {}) if old is not None else {}
        if old is not None:
            renumbered[old] = position
        categories = [str(c) for c in read_categories(obs[column], parent_group=obs)]
        if old is not None:
            # The import labelled the categories by description, in id order.
            clusters = _clusters(recorded.get("clusters"))
            labels = make_index_unique([label for _, label in clusters])
            position_of = {c: i for i, c in enumerate(categories)}
            cluster_fields[position] = {
                _label(number): str(position_of[label])
                for (number, _), label in zip(clusters, labels) if label in position_of
            } if clusters else {
                # Undescribed clusters: the import labelled them by their ids.
                label: str(i) for label, i in position_of.items()
            }
        entry = {
            "id": position,
            "group": recorded.get("group", ""),
            "name": name or column,
            "clusters": [{"id": i, "description": c} for i, c in enumerate(categories)],
        }
        recorded_metrics[position] = [m for m in recorded.get("clusterMarkerMetrics") or []
                                      if isinstance(m, dict) and m.get("accessor")]
        found.append(ExportClustering(position, column, entry))
    _check_fits([(str(c.id), np.int64) for c in found],
                f"{len(found):,} clusterings", "Name fewer with --clusterings.")
    # A clustering no import recorded owns the marker tables of its new id.
    for clustering in found:
        if clustering.id not in cluster_fields:
            renumbered.setdefault(clustering.id, clustering.id)

    # Cluster markers: only for a clustering written, and with its metrics.
    markers = [t for t in tables if t.axis == "row" and _MARKERS.match(t.key)]
    keep: List[ExportTable] = [t for t in tables if t not in markers]
    by_old: Dict[int, List[ExportTable]] = {}
    for table in markers:
        by_old.setdefault(int(_MARKERS.match(table.key).group(1)), []).append(table)
    for old, group_tables in sorted(by_old.items()):
        new = renumbered.get(old)
        if new is None:
            warnings.append(f"Not written: ClusterMarkers_{old} -- no clustering written has id {old}.")
            continue
        entry = found[new].entry
        for table in group_tables:
            accessor = _MARKERS.match(table.key).group(2)
            table.key = f"ClusterMarkers_{new}" + (f"_{accessor}" if accessor else "")
        named = [t for t in group_tables if claim_attr(owners, "row_attrs", t.key, t.source, warnings)]
        base = f"ClusterMarkers_{new}"
        if len(named) < len(group_tables) and any(t.key == base for t in group_tables) \
                and not any(t.key == base for t in named):
            warnings.append(f"Not written: the rest of {base} -- its table of markers is left out.")
            continue
        # Fields are cluster ids, renumbered with the clusters; a table left
        # with none is not written, so the metrics are worked out after.
        fields_map = cluster_fields.get(new)
        group_tables = []
        for table in named:
            if fields_map is not None:
                gone = [f for f in table.fields if f not in fields_map]
                if gone:
                    warnings.append(f"Not written in {table.source}: {', '.join(gone)} -- no "
                                    f"cluster of {found[new].column!r} has that id.")
                kept = [(fields_map[f], c, table.dtype[f])
                        for f, c in zip(table.fields, table.columns) if f in fields_map]
                table.fields = [f for f, _, _ in kept]
                table.columns = [c for _, c, _ in kept]
                table.dtype = np.dtype([(f, t) for f, _, t in kept])
            if table.fields:
                group_tables.append(table)
        accessors = [m.group(2) for t in group_tables if (m := _MARKERS.match(t.key)).group(2)]
        # The metrics of the tables written: those recorded, named as they
        # were, then any table added since, named after its accessor.
        metrics = [m for m in recorded_metrics.get(new, []) if m["accessor"] in accessors]
        listed = {m["accessor"] for m in metrics}
        metrics += [{"accessor": a, "name": a, "description": ""} for a in accessors if a not in listed]
        if not metrics or not any(t.key == base for t in group_tables):
            warnings.append(
                f"Not written: ClusterMarkers_{old} -- without metrics (_avg_logFC, "
                "_pval, ...) or markers, SCope cannot show them."
            )
            continue
        entry["clusterMarkerMetrics"] = metrics
        keep.extend(group_tables)
    tables = keep

    # Metrics and annotations: the obs columns written; clusterings are SCope's own.
    clustered = {c.column for c in found}
    metrics, annotations = [], []
    for column in obs_columns:
        if found and column == "ClusterID" or column in clustered:
            continue  # written afresh from the first clustering; a clustering
        member = obs[column]
        enc = spec.encoding_type(member)
        if is_categorical(member):
            # .shape, not len(): a zarr-python 3 Array has no len().
            if categories_dataset(member, obs).shape[0] > MAX_ANNOTATION_VALUES:
                continue  # too many to list, and not read to find that out
            values = [str(c) for c in read_categories(member, parent_group=obs)]
        elif enc in (None, spec.ARRAY, spec.STRING_ARRAY) and getattr(member, "ndim", 0) == 1 \
                and member.dtype.kind in "biuf":
            metrics.append({"name": column})
            continue
        elif enc in (spec.STRING_ARRAY, spec.NULLABLE_STRING_ARRAY) or (
            getattr(member, "dtype", None) is not None and member.dtype.kind in "OSUT"
        ):
            values = _distinct_text(member, obs, MAX_ANNOTATION_VALUES)
            if values is None:
                continue
        elif enc in (spec.NULLABLE_INTEGER, spec.NULLABLE_BOOLEAN):
            metrics.append({"name": column})
            continue
        else:
            continue
        if len(values) <= MAX_ANNOTATION_VALUES:
            annotations.append({"name": column, "values": values})

    # Regulon thresholds, for the regulons written.
    auc_fields = {f for t in tables if t.axis == "col" and t.key.endswith("RegulonsAUC") for f in t.fields}
    field_of = lambda regulon: regulon_fields.get(regulon, regulon.replace(" ", "_"))  # noqa: E731
    recorded_thresholds: Dict[str, Dict[str, Any]] = {}
    for entry in parsed.get("regulonThresholds", []) or []:
        if isinstance(entry, dict) and field_of(str(entry.get("regulon", ""))) in auc_fields:
            recorded_thresholds[field_of(str(entry["regulon"]))] = entry
    table = _resolve(root, "uns/scope/regulon_thresholds")
    if table is not None:
        # The table is what may have been edited since the import: its rows
        # and values win, the metadata adding what the table does not hold
        # (allThresholds, and fields a regulon never had).
        thresholds = []
        for regulon, row in _frame_rows(table).items():
            name = field_of(regulon)
            if name not in auc_fields:
                continue
            entry = dict(recorded_thresholds.get(name, {}), regulon=name)
            for key, value in row.items():
                if key == "defaultThresholdValue":
                    number = _float(value)
                    if np.isfinite(number):
                        entry[key] = number
                    else:
                        entry.pop(key, None)  # missing in the table
                elif value != "" or key in entry:
                    entry[key] = value
            thresholds.append(entry)
    else:
        thresholds = [dict(e, regulon=name) for name, e in recorded_thresholds.items()]

    metadata = dict(parsed)
    metadata.update(
        embeddings=[{"id": e.id, "name": e.name} for e in embeddings],
        clusterings=[c.entry for c in found],
        annotations=annotations,
        metrics=metrics,
        regulonThresholds=thresholds,
    )
    regulons = sum(len(t.fields) for t in tables if t.axis == "col" and t.key.endswith("RegulonsAUC")) or sum(
        len(t.fields) for t in tables if t.axis == "row" and t.key.endswith("Regulons")
    )
    return ScopeExport(
        tables, embeddings, found, metadata,
        cluster_id=bool(found), regulons=regulons,
    )


def claim_attr(owners: Dict[str, Dict[str, str]], group: str, name: str, what: str,
               warnings: List[str]) -> bool:
    """Give `name` in a loom's flat `group` (col_attrs, row_attrs) to `what`,
    unless something already has it: then `what` is left out, with a warning."""
    owner = owners[group].get(name)
    if owner is None:
        owners[group][name] = what
        return True
    warnings.append(f"Not written: {what} -- {group}/{name} is {owner}.")
    return False


def json_safe(value: Any) -> Any:
    """`value` with NaN and infinities as null: SCope's client parses MetaData
    as strict JSON, which has no token for them."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def is_categorical(member: Any) -> bool:
    """A categorical column, in the current layout or anndata 0.7's: codes
    whose `categories` attribute points into `__categories`."""
    if is_group(member):
        return spec.encoding_type(member) == spec.CATEGORICAL or "codes" in member
    return is_dataset(member) and "categories" in member.attrs


def categorical_codes(member: Any) -> Any:
    return member["codes"] if is_group(member) else member


def _distinct_text(member: Any, obs: Any, limit: int) -> Optional[List[str]]:
    """A text column's distinct values, read a block at a time; None as soon
    as there are more than `limit`, without reading the rest."""
    from adata.conversion.features import STRING_STEP
    from adata.elements.read import element_len, read_str_chunk
    from adata.util.names import natural_sorted

    seen: Set[str] = set()
    cache: Dict[str, np.ndarray] = {}
    n = element_len(member)
    for lo in range(0, n, STRING_STEP):
        seen.update(read_str_chunk(member, lo, min(lo + STRING_STEP, n), cache, parent_group=obs))
        if len(seen) > limit:
            return None
    return natural_sorted(seen)


def _resolve(root: Any, path: str) -> Any:
    node = root
    for part in path.split("/"):
        if part not in node:
            return None
        node = node[part]
    return node

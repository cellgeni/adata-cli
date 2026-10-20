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
  thresholds. It is decoded into `uns["scope"]`.

Embeddings and clusterings are read a block of cells at a time, each
attribute once for all of them.
"""

from __future__ import annotations

import base64
import json
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
from adata.storage import create_dataset
from adata.util.names import claim, make_index_unique, sanitize_label

METADATA = "MetaData"
EMBEDDING = "Embedding"
EMBEDDINGS_X, EMBEDDINGS_Y = "Embeddings_X", "Embeddings_Y"
CLUSTERINGS = "Clusterings"

#: The regulon thresholds' fields kept as columns. `allThresholds`, a
#: mapping per regulon, stays in the metadata text.
THRESHOLD_COLUMNS = ("defaultThresholdValue", "defaultThresholdName", "motifData")


def is_scope(global_attrs: Iterable[str], col_attrs: Iterable[str]) -> bool:
    """A `MetaData` global attribute, or a column attribute only SCope writes."""
    return METADATA in set(global_attrs) or any(
        name in (EMBEDDINGS_X, CLUSTERINGS) or name.endswith("RegulonsAUC")
        for name in col_attrs
    )


def read_metadata(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """`MetaData` as (its JSON, the JSON's text).

    The attribute is JSON, or base64 of compressed JSON: zlib from pySCENIC,
    gzip from SCopeLoomR. When it is neither, the JSON is None and the text
    is returned as it came.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        try:
            # 47: a zlib or a gzip header, whichever is there.
            decoded = zlib.decompress(base64.b64decode(text), 47).decode("utf-8")
            parsed = json.loads(decoded)
        except (ValueError, zlib.error):
            return None, text
        text = decoded
    return (parsed if isinstance(parsed, dict) else None), text


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
                "MetaData is neither JSON nor compressed JSON: it is kept as text in "
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
        wanted = (name or f"clustering_{id}").replace("/", "_")
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
            "the name is taken, or holds '/'."
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

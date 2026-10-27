"""GeoJSON polygons, read one feature at a time.

Space Ranger 4 writes each segmented cell as a polygon in a GeoJSON
FeatureCollection: some hundreds of thousands of them on a full slide, all
on one line, a few hundred MB. `json.load` would hold every vertex as a
Python object at once, so the collection is walked here key by key, and its
features decoded one at a time from a text buffer refilled a block at a time.

What is kept of each polygon is what an AnnData holds of a cell: its
centroid and its area.
"""

from __future__ import annotations

import codecs
import json
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import numpy as np

from adata.formats.textio import open_binary

#: Bytes read from the file at a time.
READ_BLOCK = 1 << 20

#: Polygons measured together, and handed out per block.
BATCH_FEATURES = 1 << 14

#: The most text one feature may take. A cell's outline is a few hundred
#: bytes; past this the file is broken, not a feature still to come.
MAX_FEATURE = 1 << 26

#: How far from the end of the text a decoding error may lie and still be
#: a value cut off by the block's end ("tr" of "true", "\u00" of "\u00e9").
_TAIL = 16

_WHITESPACE = " \t\n\r"
#: What may follow a number: a number cut off by the block's end (the "12."
#: of "12.5") decodes, but is followed by more of itself.
_AFTER_NUMBER = _WHITESPACE + ",]}"


class _Text:
    """A file's text, held a block at a time, decoded a JSON value at a time."""

    def __init__(self, fh: Any, path: Path, block: int) -> None:
        self._fh = fh
        self._path = path
        self._block = block
        self._decode = codecs.getincrementaldecoder("utf-8-sig")().decode
        self._json = json.JSONDecoder()
        self.text = ""
        self.pos = 0
        self.eof = False

    def _fill(self, size: int = 0) -> None:
        raw = self._fh.read(size or self._block)
        self.text = self.text[self.pos :] + self._decode(raw, final=not raw)
        self.pos = 0
        self.eof = not raw

    def peek(self) -> str:
        """The next character that is not whitespace, '' at the end."""
        while True:
            while self.pos < len(self.text) and self.text[self.pos] in _WHITESPACE:
                self.pos += 1
            if self.pos < len(self.text) or self.eof:
                return self.text[self.pos : self.pos + 1]
            self._fill()

    def expect(self, char: str, where: str) -> None:
        found = self.peek()
        if found != char:
            raise ValueError(
                f"{self._path.name} is not a GeoJSON FeatureCollection: expected "
                f"{char!r} {where}, found {found or 'the end of the file'!r}."
            )
        self.pos += 1

    def value(self, where: str) -> Any:
        """The next JSON value, reading on while it may be cut off.

        An error well before the end of the text is the file's; one near it
        may be a value the next block completes, except past MAX_FEATURE,
        so that a broken file is refused before it is held whole. Each read
        doubles, so a long value is not copied once per block.
        """
        self.peek()
        size = self._block
        while True:
            try:
                value, end = self._json.raw_decode(self.text, self.pos)
            except json.JSONDecodeError as error:
                cut = (
                    error.pos >= len(self.text) - _TAIL
                    or error.msg.startswith("Unterminated string")
                )
                if self.eof or not cut or len(self.text) - self.pos > MAX_FEATURE:
                    raise ValueError(f"{self._path.name}: {where} is not valid JSON ({error.msg}).") from None
            else:
                number = isinstance(value, (int, float)) and not isinstance(value, bool)
                if self.eof or (end < len(self.text) and (not number or self.text[end] in _AFTER_NUMBER)):
                    self.pos = end
                    return value
            self._fill(size)
            size *= 2


def iter_features(path: Path, block: int = READ_BLOCK) -> Iterator[Any]:
    """Each feature of a GeoJSON FeatureCollection, in file order."""
    path = Path(path)
    with open_binary(path) as fh:
        text = _Text(fh, path, block)
        text.expect("{", "at the start")
        if text.peek() == "}":
            raise ValueError(f"{path.name} is an empty object, not a FeatureCollection.")
        found = False
        while True:
            key = text.value("a key")
            text.expect(":", f"after {key!r}")
            if key == "features":
                found = True
                yield from _array(text)
            else:
                value = text.value(repr(key))
                if key == "type" and value != "FeatureCollection":
                    raise ValueError(f"{path.name} is a GeoJSON {value}, not a FeatureCollection.")
            if text.peek() == ",":
                text.pos += 1
                continue
            text.expect("}", "at the end of the collection")
            break
    if not found:
        raise ValueError(f"{path.name} has no features.")


def _array(text: _Text) -> Iterator[Any]:
    text.expect("[", "after 'features'")
    if text.peek() == "]":
        text.pos += 1
        return
    n = 0
    while True:
        yield text.value(f"feature {n}")
        n += 1
        if text.peek() == ",":
            text.pos += 1
            continue
        text.expect("]", f"after feature {n - 1}")
        return


def polygon_stats(
    path: Path, id_key: str = "cell_id", batch: int = BATCH_FEATURES, block: int = READ_BLOCK
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(ids, x, y, area) of each Polygon or MultiPolygon, a batch at a time.

    The centroid is weighted by area, as shapely's is, and the area counts
    holes out: a feature's integer `id_key` property names it.
    """
    path = Path(path)
    ids: List[int] = []
    rings: List[Tuple[int, int, np.ndarray]] = []  # (feature, +1 shell / -1 hole, vertices)
    for n, feature in enumerate(iter_features(path, block)):
        ids.append(_feature_id(feature, id_key, path, n))
        for role, ring in _rings(feature, path, n):
            rings.append((len(ids) - 1, role, ring))
        if len(ids) == batch:
            yield _measure(ids, rings)
            ids, rings = [], []
    if ids:
        yield _measure(ids, rings)


def _feature_id(feature: Any, key: str, path: Path, n: int) -> int:
    properties = feature.get("properties") if isinstance(feature, dict) else None
    value = properties.get(key) if isinstance(properties, dict) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path.name}: feature {n} has no integer {key} property.")
    if not -(1 << 63) <= value < 1 << 63:
        raise ValueError(f"{path.name}: feature {n} has a {key} too large to be an id ({value}).")
    return value


def _rings(feature: Any, path: Path, n: int) -> List[Tuple[int, np.ndarray]]:
    geometry = feature.get("geometry")
    kind = geometry.get("type") if isinstance(geometry, dict) else None
    if kind == "Polygon":
        polygons = [geometry.get("coordinates")]
    elif kind == "MultiPolygon":
        polygons = geometry.get("coordinates")
    else:
        raise ValueError(
            f"{path.name}: feature {n} is a {kind or 'feature with no geometry'}, "
            "not a Polygon or MultiPolygon."
        )
    found = []
    try:
        for polygon in polygons:
            for i, ring in enumerate(polygon):
                # Numbers only: numpy would take "2", true and null too.
                vertices = np.asarray(ring)
                if (
                    vertices.ndim != 2 or vertices.shape[1] < 2 or vertices.shape[0] == 0
                    or vertices.dtype.kind not in "iuf"
                    or any(type(v) is bool for point in ring for v in point)
                ):
                    raise ValueError
                vertices = vertices[:, :2].astype(np.float64)
                if not np.isfinite(vertices).all():
                    raise ValueError
                found.append(((1 if i == 0 else -1), vertices))
    except (TypeError, ValueError):
        found = []
    if not found:
        raise ValueError(f"{path.name}: feature {n} has coordinates that are not rings of points.")
    return found


def _measure(ids: List[int], rings: List[Tuple[int, int, np.ndarray]]):
    """Shoelace area and first moments of every ring at once.

    Vertices are taken relative to their feature's first vertex, as GEOS
    does: pixel coordinates in the thousands make the products large and the
    area of a small cell a difference of them.
    """
    n = len(ids)
    feature = np.array([r[0] for r in rings], dtype=np.int64)
    sizes = np.array([len(r[2]) for r in rings], dtype=np.int64)
    points = np.concatenate([r[2] for r in rings])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    # Rings come in feature order, each feature with one at least.
    first = starts[np.unique(feature, return_index=True)[1]]
    owner = np.repeat(feature, sizes)
    origin = points[first[owner]]
    x, y = (points - origin).T
    # Each vertex's successor in its ring, wrapping to the ring's start.
    following = np.arange(1, len(points) + 1)
    following[starts + sizes - 1] = starts
    cross = x * y[following] - x[following] * y
    ring_of = np.repeat(np.arange(len(rings)), sizes)
    area = np.bincount(ring_of, cross, len(rings)) / 2
    moment_x = np.bincount(ring_of, (x + x[following]) * cross, len(rings)) / 6
    moment_y = np.bincount(ring_of, (y + y[following]) * cross, len(rings)) / 6
    # A shell counts its area whichever way it winds, a hole against it.
    sign = np.array([r[1] for r in rings], dtype=np.float64) * np.sign(area)
    total = np.bincount(feature, area * sign, n)
    sum_x = np.bincount(feature, moment_x * sign, n)
    sum_y = np.bincount(feature, moment_y * sign, n)
    # A polygon with no area has no centroid by weight; as GEOS does, that
    # of its outline stands in, each edge weighted by its length, and for a
    # single point, the point.
    flat = total == 0
    length = np.hypot(x[following] - x, y[following] - y)
    perimeter = np.bincount(owner, length, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        line_x = np.bincount(owner, length * (x + x[following]) / 2, n) / perimeter
        line_y = np.bincount(owner, length * (y + y[following]) / 2, n) / perimeter
        cx = np.where(flat, np.where(perimeter == 0, 0.0, line_x), sum_x / total)
        cy = np.where(flat, np.where(perimeter == 0, 0.0, line_y), sum_y / total)
    base = points[first]
    return np.array(ids, dtype=np.int64), cx + base[:, 0], cy + base[:, 1], total

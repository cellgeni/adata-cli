"""Tests for reading GeoJSON polygons a feature at a time (`formats/geojson.py`)."""

from __future__ import annotations

import contextlib
import gzip
import json

import numpy as np
import pytest
import shapely

from adata.formats import geojson
from adata.formats.geojson import iter_features, polygon_stats


def _square(x, y, side, clockwise=False):
    ring = [[x, y], [x + side, y], [x + side, y + side], [x, y + side], [x, y]]
    return ring[::-1] if clockwise else ring


def _features():
    """Polygons of every shape Space Ranger might write, and some it does not."""
    rng = np.random.default_rng(3)
    features = []
    for cell_id in range(1, 40):
        x, y = rng.uniform(1000, 6000, 2)
        ring = [[x + 9 * np.cos(t) * rng.uniform(0.6, 1), y + 7 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 17)[:-1]]
        ring.append(ring[0])
        features.append({"type": "Polygon", "coordinates": [ring]})
    features += [
        # A shell wound clockwise, with a hole wound either way.
        {"type": "Polygon", "coordinates": [_square(5000, 5000, 20, clockwise=True), _square(5005, 5005, 5)]},
        {"type": "Polygon", "coordinates": [_square(5100, 5000, 20), _square(5101, 5101, 4)[::-1]]},
        {"type": "MultiPolygon", "coordinates": [[_square(10, 10, 4)], [_square(30, 10, 2)]]},
        # An open ring: GeoJSON closes them, but the area must not depend on it.
        {"type": "Polygon", "coordinates": [_square(300, 300, 10)[:-1]]},
        # Accented, and 3-D points.
        {"type": "Polygon", "coordinates": [[[p[0], p[1], 7.0] for p in _square(40.25, 50.5, 3.5)]]},
    ]
    return [
        {"type": "Feature", "geometry": geometry, "properties": {"cell_id": i + 1, "név": "é"}}
        for i, geometry in enumerate(features)
    ]


def _write(path, features, **dump):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, ensure_ascii=False, **dump)
    return path


def _stats(path, **kwargs):
    blocks = list(polygon_stats(path, **kwargs))
    return tuple(np.concatenate([b[i] for b in blocks]) for i in range(4))


@pytest.mark.parametrize("block", [1, 2, 7, 64, 1 << 20])
def test_features_split_across_any_block_size_read_as_json_load_reads_them(tmp_path, block):
    features = _features()
    path = _write(tmp_path / "cells.geojson", features)
    assert list(iter_features(path, block=block)) == features


@pytest.mark.parametrize("batch", [1, 5, 1 << 14])
def test_centroids_and_areas_equal_shapely(tmp_path, batch):
    features = _features()
    path = _write(tmp_path / "cells.geojson", features)
    ids, x, y, area = _stats(path, batch=batch, block=97)

    shapes = [shapely.geometry.shape(f["geometry"]) for f in features]
    assert ids.tolist() == [f["properties"]["cell_id"] for f in features]
    np.testing.assert_allclose(x, [s.centroid.x for s in shapes], rtol=0, atol=1e-9)
    np.testing.assert_allclose(y, [s.centroid.y for s in shapes], rtol=0, atol=1e-9)
    np.testing.assert_allclose(area, [s.area for s in shapes], rtol=1e-12)


def test_pretty_printed_reordered_and_gzipped_collections_read_the_same(tmp_path):
    features = _features()
    plain = _write(tmp_path / "plain.geojson", features)
    pretty = _write(tmp_path / "pretty.geojson", features, indent=2)
    reordered = tmp_path / "reordered.geojson"
    reordered.write_text(json.dumps({"features": features, "bbox": [0, 0, 1, 1], "type": "FeatureCollection"}))
    zipped = tmp_path / "cells.geojson.gz"
    with gzip.open(zipped, "wt", encoding="utf-8") as fh:
        fh.write("﻿" + plain.read_text(encoding="utf-8"))

    expected = _stats(plain)
    for path in (pretty, reordered, zipped):
        for got, want in zip(_stats(path, block=13), expected):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "ring",
    [
        [[0, 0], [4, 0], [8, 0], [0, 0]],
        [[1000, 2000], [1003, 2004], [1009, 2012], [1000, 2000]],
        [[5, 7], [5, 7], [5, 7], [5, 7]],
    ],
)
def test_a_polygon_with_no_area_is_placed_where_shapely_places_it(tmp_path, ring):
    """Its centroid used to be the mean of its vertices, closing one
    included: (3, 0) for the first, where shapely gives (4, 0)."""
    geometry = {"type": "Polygon", "coordinates": [ring]}
    path = _write(tmp_path / "c.geojson", [{"type": "Feature", "geometry": geometry, "properties": {"cell_id": 3}}])
    ids, x, y, area = _stats(path)
    expected = shapely.geometry.shape(geometry).centroid
    assert (ids[0], area[0]) == (3, 0.0)
    np.testing.assert_allclose([x[0], y[0]], [expected.x, expected.y], rtol=0, atol=1e-9)


@pytest.mark.parametrize("block", [1, 3, 4, 5])
def test_a_number_cut_by_a_block_end_is_read_whole(tmp_path, block):
    """The "12." of "12.5" decoded as 12, and the collection was then refused
    for the ".5" that followed."""
    path = tmp_path / "c.geojson"
    features = _features()[:2]
    path.write_text(json.dumps({"version": 12.5, "scale": 1e30, "type": "FeatureCollection", "features": features}))
    assert list(iter_features(path, block=block)) == features


def test_a_file_broken_early_is_refused_without_reading_the_rest(tmp_path, monkeypatch):
    """A broken feature used to make the reader read on to the end of the
    file, holding it all as text, before refusing it: 3 GB for 100 MB."""
    path = tmp_path / "c.geojson"
    good = json.dumps(_features()[0])
    path.write_text('{"type": "FeatureCollection", "features": [' + good.replace("[", "[x", 3) + ", " + ", ".join([good] * 2000) + "]}")
    read = []
    opener = geojson.open_binary

    @contextlib.contextmanager
    def counting(p):
        with opener(p) as fh:
            real = fh.read
            fh_read = lambda n=-1: read.append(n) or real(n)  # noqa: E731
            yield type("Counted", (), {"read": staticmethod(fh_read)})()

    monkeypatch.setattr(geojson, "open_binary", counting)
    with pytest.raises(ValueError, match="feature 0 is not valid JSON"):
        list(polygon_stats(path, block=256))
    assert sum(read) < path.stat().st_size // 20


def test_a_feature_past_the_size_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(geojson, "MAX_FEATURE", 1000)
    path = tmp_path / "c.geojson"
    path.write_text('{"type": "FeatureCollection", "features": [{"properties": {"cell_id": 1, "note": "' + "a" * 5000 + '"}}]}')
    with pytest.raises(ValueError, match="feature 0 is not valid JSON"):
        list(polygon_stats(path, block=64))


def test_an_empty_collection_has_no_polygons(tmp_path):
    path = _write(tmp_path / "c.geojson", [])
    assert list(polygon_stats(path)) == []


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Poly', "feature 0 is not valid JSON"),
        ('{"type": "FeatureCollection", "features": [{"type": "Feature"', "feature 0 is not valid JSON"),
        ('{"type": "FeatureCollection", "features": []', "expected '}' at the end of the collection"),
        ('{"type": "FeatureCollection"}', "has no features"),
        ('{"type": "Feature", "features": []}', "is a GeoJSON Feature, not a FeatureCollection"),
        ("[]", "expected '{' at the start"),
        ("", "expected '{' at the start"),
        ("{}", "empty object"),
        ('{"features": [{"geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {"cell_id": 1}}]}', "feature 0 is a Point"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": []}, "properties": {"cell_id": 1}}]}', "feature 0 has coordinates that are not rings"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {"cell_id": 1}}]}', "feature 0 has coordinates that are not rings"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 1]]]}, "properties": {"cell_id": "a"}}]}', "feature 0 has no integer cell_id"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 1]]]}, "properties": {"cell_id": true}}]}', "feature 0 has no integer cell_id"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 1]]]}}]}', "feature 0 has no integer cell_id"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 1]]]}, "properties": {"cell_id": 99999999999999999999}}]}', "feature 0 has a cell_id too large"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, null], [1, 1], [0, 1]]]}, "properties": {"cell_id": 1}}]}', "feature 0 has coordinates that are not rings"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, "2"], [1, 1], [0, 1]]]}, "properties": {"cell_id": 1}}]}', "feature 0 has coordinates that are not rings"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, true], [1, 1], [0, 1]]]}, "properties": {"cell_id": 1}}]}', "feature 0 has coordinates that are not rings"),
        ('{"features": [{"geometry": {"type": "Polygon", "coordinates": [[[0, NaN], [1, 1], [0, 1]]]}, "properties": {"cell_id": 1}}]}', "feature 0 has coordinates that are not rings"),
    ],
)
def test_a_file_that_is_not_a_collection_of_polygons_is_refused(tmp_path, content, message):
    path = tmp_path / "c.geojson"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        list(polygon_stats(path, block=5))

"""Tests for the 1km validation grid construction logic."""

import numpy as np
from rasterio.transform import from_origin

from src.data_processing.build_validation_grid import build_1km_grid_transform


def test_grid_snapping_alignment():
    """Grid cells snap to clean degree multiples, so the same physical
    location always falls in the same grid cell regardless of which
    ALS tile defines the extent."""
    res = 1.0 / 111.0  # ~0.009009°

    # Two overlapping bounding boxes (simulating adjacent tiles)
    bounds_a = (-47.25, -7.55, -47.15, -7.45)
    bounds_b = (-47.20, -7.52, -47.10, -7.42)

    tf_a, w_a, h_a = build_1km_grid_transform(bounds_a, res)
    tf_b, w_b, h_b = build_1km_grid_transform(bounds_b, res)

    # Both grids should snap to the same origin alignment
    # (origins differ, but are exact multiples of res)
    assert abs(tf_a.c % res) < 1e-10 or abs(tf_a.c % res - res) < 1e-10
    assert abs(tf_b.c % res) < 1e-10 or abs(tf_b.c % res - res) < 1e-10

    # A point in the overlap should have the same grid_id in both grids
    test_lon, test_lat = -47.18, -7.48
    col_a = int((test_lon - tf_a.c) / tf_a.a)
    row_a = int((test_lat - tf_a.f) / tf_a.e)
    col_b = int((test_lon - tf_b.c) / tf_b.a)
    row_b = int((test_lat - tf_b.f) / tf_b.e)

    # Centroids should be identical
    centroid_lon_a = tf_a.c + (col_a + 0.5) * tf_a.a
    centroid_lat_a = tf_a.f + (row_a + 0.5) * tf_a.e
    centroid_lon_b = tf_b.c + (col_b + 0.5) * tf_b.a
    centroid_lat_b = tf_b.f + (row_b + 0.5) * tf_b.e

    assert abs(centroid_lon_a - centroid_lon_b) < 1e-10
    assert abs(centroid_lat_a - centroid_lat_b) < 1e-10


def test_grid_dimensions_positive():
    """Grid should always produce positive width and height."""
    res = 1.0 / 111.0
    bounds = (-47.25, -7.55, -47.15, -7.45)
    tf, w, h = build_1km_grid_transform(bounds, res)

    assert w > 0
    assert h > 0
    assert tf.a > 0   # positive x resolution (east)
    assert tf.e < 0   # negative y resolution (south)


def test_grid_covers_bounds():
    """The grid extent should fully contain the input bounds."""
    res = 1.0 / 111.0
    bounds = (-47.225, -7.534, -47.171, -7.456)
    tf, w, h = build_1km_grid_transform(bounds, res)

    grid_west = tf.c
    grid_north = tf.f
    grid_east = tf.c + w * tf.a
    grid_south = tf.f + h * tf.e

    assert grid_west <= bounds[0]
    assert grid_south <= bounds[1]
    assert grid_east >= bounds[2]
    assert grid_north >= bounds[3]

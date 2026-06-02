"""Tests for polygon_utils.py"""
import numpy as np
import pytest
import cv2

from polygon_utils import mask_to_polygons, polygons_to_bbox, polygons_to_area


def test_mask_to_polygons_simple_square():
    """A filled square should produce one polygon with >= 4 points."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    poly = polygons[0]
    assert len(poly) >= 8          # at least 4 (x,y) pairs
    assert len(poly) % 2 == 0     # always even (x,y pairs)


def test_mask_to_polygons_empty_mask():
    """Empty mask returns no polygons."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert mask_to_polygons(mask) == []


def test_mask_to_polygons_two_disconnected_regions():
    """Two separate blobs produce two polygons."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:20, 5:20] = 1
    mask[60:80, 60:80] = 1
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 2


def test_mask_to_polygons_simplify_reduces_points():
    """Simplification with epsilon > 0 should not increase point count."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 1, -1)
    full = mask_to_polygons(mask, simplify_epsilon=0.0)
    simplified = mask_to_polygons(mask, simplify_epsilon=5.0)
    assert len(full) == 1
    assert len(simplified) == 1
    assert len(simplified[0]) <= len(full[0])


def test_mask_to_polygons_tiny_contour_skipped():
    """A single-pixel region (< 3 contour points) is skipped."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 1  # single pixel
    # single pixel may produce a contour of 1 point — must be skipped
    polygons = mask_to_polygons(mask)
    for poly in polygons:
        assert len(poly) >= 6  # at least 3 (x,y) pairs


def test_polygons_to_bbox_known_square():
    """Bbox from a known 20x20 square polygon."""
    poly = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    bbox = polygons_to_bbox([poly])
    assert bbox == pytest.approx([10.0, 10.0, 20.0, 20.0])


def test_polygons_to_bbox_none_on_empty():
    """No polygons returns None."""
    assert polygons_to_bbox([]) is None


def test_polygons_to_area_known_square():
    """Area of a 10x10 square polygon is 100."""
    poly = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    area = polygons_to_area([poly])
    assert abs(area - 100.0) < 1.0


def test_polygons_to_area_two_polygons():
    """Area sums correctly across multiple polygons."""
    poly = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    area = polygons_to_area([poly, poly])
    assert abs(area - 200.0) < 2.0

"""Tests for coco_visualizer.py — no GPU, no model needed."""
import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from coco_visualizer import _flat_to_xy_pairs, _category_color, visualize_coco_output


# ---------------------------------------------------------------------------
# _flat_to_xy_pairs
# ---------------------------------------------------------------------------

def test_flat_to_xy_pairs_basic():
    flat = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    result = _flat_to_xy_pairs(flat)
    assert result == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


def test_flat_to_xy_pairs_four_points():
    flat = [0, 0, 10, 0, 10, 10, 0, 10]
    result = _flat_to_xy_pairs(flat)
    assert result == [(0, 0), (10, 0), (10, 10), (0, 10)]


# ---------------------------------------------------------------------------
# _category_color
# ---------------------------------------------------------------------------

def test_category_color_returns_rgb_tuple():
    color = _category_color(1)
    assert isinstance(color, tuple)
    assert len(color) == 3
    assert all(0 <= c <= 255 for c in color)


def test_category_color_consistent():
    assert _category_color(1) == _category_color(1)
    assert _category_color(2) == _category_color(2)


def test_category_color_different_ids_differ():
    assert _category_color(1) != _category_color(2)


def test_category_color_cycles():
    # IDs that are 20 apart should give same color (palette has 20 entries)
    assert _category_color(1) == _category_color(21)

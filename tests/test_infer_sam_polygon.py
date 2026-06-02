"""Tests for infer_sam_polygon.py — model is mocked, no GPU needed."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from infer_sam_polygon import draw_polygons


@pytest.fixture
def small_image():
    return PILImage.new("RGB", (100, 80), color=(200, 200, 200))


@pytest.fixture
def fake_results(small_image):
    """Mimic the dict returned by SAM3LoRAInference.predict()."""
    mask = np.zeros((80, 100), dtype=bool)
    mask[10:40, 10:50] = True  # one rectangular region
    return {
        0: {
            "prompt": "crack",
            "boxes": np.array([[10.0, 10.0, 50.0, 40.0]]),
            "scores": np.array([0.85]),
            "masks": mask[np.newaxis],  # [1, H, W]
            "num_detections": 1,
        },
        "_image": small_image,
    }


def test_draw_polygons_returns_image(fake_results, small_image):
    """draw_polygons returns a PIL Image of the same size."""
    out = draw_polygons(fake_results, simplify_epsilon=0.0, show_boxes=False)
    assert isinstance(out, PILImage.Image)
    assert out.size == small_image.size


def test_draw_polygons_no_detections(small_image):
    """draw_polygons handles zero-detection results without error."""
    results = {
        0: {
            "prompt": "crack",
            "boxes": None,
            "scores": None,
            "masks": None,
            "num_detections": 0,
        },
        "_image": small_image,
    }
    out = draw_polygons(results, simplify_epsilon=0.0, show_boxes=False)
    assert isinstance(out, PILImage.Image)


def test_draw_polygons_with_simplify(fake_results):
    """draw_polygons with simplify_epsilon > 0 runs without error."""
    out = draw_polygons(fake_results, simplify_epsilon=3.0, show_boxes=False)
    assert isinstance(out, PILImage.Image)


def test_draw_polygons_with_boxes(fake_results):
    """draw_polygons with show_boxes=True runs without error."""
    out = draw_polygons(fake_results, simplify_epsilon=0.0, show_boxes=True)
    assert isinstance(out, PILImage.Image)

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_image(tmp_path):
    """Create a 100x80 red JPEG and a 60x60 blue PNG in tmp_path."""
    img_jpg = PILImage.new("RGB", (100, 80), color=(200, 50, 50))
    img_jpg.save(tmp_path / "img_a.jpg")
    img_png = PILImage.new("RGB", (60, 60), color=(50, 50, 200))
    img_png.save(tmp_path / "img_b.png")
    return tmp_path


@pytest.fixture
def coco_with_annotations(tmp_path, small_image):
    """COCO JSON: img_a has 1 annotation, img_b has none."""
    coco = {
        "categories": [{"id": 1, "name": "crack", "supercategory": "defect"}],
        "images": [
            {"id": 1, "file_name": "img_a.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "img_b.png", "width": 60,  "height": 60},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 40.0, 10.0, 40.0]],
                "bbox": [10.0, 10.0, 40.0, 30.0],
                "area": 1200.0,
                "score": 0.92,
                "iscrowd": 0,
            }
        ],
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(coco))
    return p


# ---------------------------------------------------------------------------
# visualize_coco_output
# ---------------------------------------------------------------------------

def test_output_files_created(coco_with_annotations, small_image, tmp_path):
    viz_dir = tmp_path / "viz"
    visualize_coco_output(str(coco_with_annotations), str(small_image), str(viz_dir))
    assert (viz_dir / "img_a.jpg").exists()
    assert (viz_dir / "img_b.png").exists()


def test_unannotated_image_saved(coco_with_annotations, small_image, tmp_path):
    viz_dir = tmp_path / "viz"
    visualize_coco_output(str(coco_with_annotations), str(small_image), str(viz_dir))
    out = PILImage.open(viz_dir / "img_b.png")
    assert out.size == (60, 60)


def test_annotated_image_differs_from_source(coco_with_annotations, small_image, tmp_path):
    import numpy as np
    viz_dir = tmp_path / "viz"
    visualize_coco_output(str(coco_with_annotations), str(small_image), str(viz_dir))
    src = np.array(PILImage.open(small_image / "img_a.jpg").convert("RGB"))
    out = np.array(PILImage.open(viz_dir / "img_a.jpg").convert("RGB"))
    assert not np.array_equal(src, out), "Annotated image should differ from source"


def test_preserves_jpg_extension(coco_with_annotations, small_image, tmp_path):
    viz_dir = tmp_path / "viz"
    visualize_coco_output(str(coco_with_annotations), str(small_image), str(viz_dir))
    assert (viz_dir / "img_a.jpg").exists()
    assert not (viz_dir / "img_a.png").exists()


def test_preserves_png_extension(coco_with_annotations, small_image, tmp_path):
    viz_dir = tmp_path / "viz"
    visualize_coco_output(str(coco_with_annotations), str(small_image), str(viz_dir))
    assert (viz_dir / "img_b.png").exists()
    assert not (viz_dir / "img_b.jpg").exists()


def test_missing_image_skipped_no_crash(coco_with_annotations, tmp_path):
    empty_dir = tmp_path / "no_images"
    empty_dir.mkdir()
    viz_dir = tmp_path / "viz"
    # Should not raise even though images are missing
    visualize_coco_output(str(coco_with_annotations), str(empty_dir), str(viz_dir))
    assert not (viz_dir / "img_a.jpg").exists()


def test_score_optional(small_image, tmp_path):
    """Annotation without score field must not raise."""
    coco = {
        "categories": [{"id": 1, "name": "crack", "supercategory": "defect"}],
        "images": [{"id": 1, "file_name": "img_a.jpg", "width": 100, "height": 80}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 40.0, 10.0, 40.0]],
                "bbox": [10.0, 10.0, 40.0, 30.0],
                "area": 1200.0,
                "iscrowd": 0,
                # note: no "score" key
            }
        ],
    }
    p = tmp_path / "no_score.json"
    p.write_text(json.dumps(coco))
    viz_dir = tmp_path / "viz"
    visualize_coco_output(str(p), str(small_image), str(viz_dir))
    assert (viz_dir / "img_a.jpg").exists()

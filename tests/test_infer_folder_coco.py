"""Tests for infer_folder_coco.py — model is mocked, no GPU needed."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import numpy as np
import pytest
from PIL import Image as PILImage

from infer_folder_coco import (
    load_categories,
    results_to_coco_annotations,
    run_folder_inference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def categories_coco_file(tmp_path):
    """Write a minimal COCO JSON with categories and return its path."""
    data = {
        "categories": [
            {"id": 1, "name": "crack", "supercategory": "defect"},
            {"id": 2, "name": "rust", "supercategory": "defect"},
        ],
        "images": [],
        "annotations": [],
    }
    p = tmp_path / "_annotations.coco.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def fake_results():
    """Mimic SAM3LoRAInference.predict() output with two detections."""
    h, w = 80, 100
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[10:40, 10:50] = True
    mask2 = np.zeros((h, w), dtype=bool)
    mask2[50:70, 60:90] = True
    return {
        0: {
            "prompt": "crack",
            "boxes": np.array([[10.0, 10.0, 50.0, 40.0], [60.0, 50.0, 90.0, 70.0]]),
            "scores": np.array([0.9, 0.75]),
            "masks": np.stack([mask1, mask2]),  # [2, H, W]
            "num_detections": 2,
        },
        1: {
            "prompt": "rust",
            "boxes": None,
            "scores": None,
            "masks": None,
            "num_detections": 0,
        },
        "_image": PILImage.new("RGB", (w, h)),
    }


# ---------------------------------------------------------------------------
# load_categories
# ---------------------------------------------------------------------------

def test_load_categories_returns_list(categories_coco_file):
    cats = load_categories(categories_coco_file)
    assert isinstance(cats, list)
    assert len(cats) == 2
    assert cats[0]["name"] == "crack"


def test_load_categories_missing_file():
    with pytest.raises(FileNotFoundError):
        load_categories("/nonexistent/file.json")


# ---------------------------------------------------------------------------
# results_to_coco_annotations
# ---------------------------------------------------------------------------

def test_annotations_count(fake_results):
    """Each valid contour becomes one annotation; zero-detection prompts skipped."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=1,
    )
    # 2 detections from "crack" prompt, each mask has 1 contour → 2 annotations
    assert len(anns) == 2


def test_annotations_category_id(fake_results):
    """category_id is mapped correctly from prompt."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=1,
    )
    for ann in anns:
        assert ann["category_id"] == 1  # both belong to "crack"


def test_annotations_iscrowd_zero(fake_results):
    """All annotations must have iscrowd=0 for CVAT compatibility."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=1,
    )
    for ann in anns:
        assert ann["iscrowd"] == 0


def test_annotations_segmentation_format(fake_results):
    """segmentation must be a list of lists of floats."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=1,
    )
    for ann in anns:
        assert isinstance(ann["segmentation"], list)
        assert len(ann["segmentation"]) == 1        # one polygon per annotation
        assert isinstance(ann["segmentation"][0], list)
        assert len(ann["segmentation"][0]) >= 6     # at least 3 (x,y) pairs
        assert len(ann["segmentation"][0]) % 2 == 0


def test_annotations_bbox_format(fake_results):
    """bbox must be [x, y, w, h] (4 floats)."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=1,
    )
    for ann in anns:
        assert len(ann["bbox"]) == 4
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0


def test_annotations_sequential_ids(fake_results):
    """Annotation IDs start at start_ann_id and increment."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=5,
    )
    ids = [a["id"] for a in anns]
    assert ids == list(range(5, 5 + len(anns)))


def test_annotations_image_id_set(fake_results):
    """All annotations reference the given image_id."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=42, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=0.0, start_ann_id=1,
    )
    for ann in anns:
        assert ann["image_id"] == 42


def test_annotations_with_simplify(fake_results):
    """simplify_epsilon > 0 still produces valid annotations."""
    prompt_to_cat = {"crack": 1, "rust": 2}
    anns = results_to_coco_annotations(
        fake_results, image_id=1, prompt_to_cat_id=prompt_to_cat,
        simplify_epsilon=3.0, start_ann_id=1,
    )
    assert len(anns) >= 1


# ---------------------------------------------------------------------------
# Integration: full COCO JSON output via mocked model
# ---------------------------------------------------------------------------

def test_full_coco_json_structure(categories_coco_file, tmp_path):
    """End-to-end: mocked model → valid COCO JSON written to disk."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = PILImage.new("RGB", (100, 80), color=128)
    img.save(img_dir / "img_001.jpg")

    output_path = tmp_path / "output.json"

    h, w = 80, 100
    mask = np.zeros((h, w), dtype=bool)
    mask[10:40, 10:50] = True
    mock_results = {
        0: {
            "prompt": "crack",
            "boxes": np.array([[10.0, 10.0, 50.0, 40.0]]),
            "scores": np.array([0.9]),
            "masks": mask[np.newaxis],
            "num_detections": 1,
        },
        "_image": img,
    }

    # Mock infer_sam module to avoid torch import at patch time
    mock_infer_sam = MagicMock()
    MockModel = MagicMock()
    instance = MockModel.return_value
    instance.predict.return_value = mock_results
    mock_infer_sam.SAM3LoRAInference = MockModel
    
    with patch.dict(sys.modules, {"infer_sam": mock_infer_sam}):
        run_folder_inference(
            input_dir=str(img_dir),
            output_path=str(output_path),
            prompts=["crack"],
            category_ids=[1],
            categories_file=categories_coco_file,
            config_path="dummy.yaml",
            weights_path=None,
            threshold=0.5,
            resolution=1008,
            nms_iou=0.5,
            simplify_epsilon=0.0,
            image_exts=["jpg"],
            device="cpu",
        )

    with open(output_path) as f:
        coco = json.load(f)

    assert "info" in coco
    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) >= 1
    assert coco["images"][0]["file_name"] == "img_001.jpg"
    ann = coco["annotations"][0]
    assert ann["iscrowd"] == 0
    assert ann["category_id"] == 1
    assert isinstance(ann["segmentation"], list)


def test_prompt_category_id_mismatch_raises(categories_coco_file, tmp_path):
    """run_folder_inference raises ValueError when prompt and category_id counts differ."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Mock infer_sam to avoid torch import error
    mock_infer_sam = MagicMock()
    with patch.dict(sys.modules, {"infer_sam": mock_infer_sam}):
        with pytest.raises(ValueError, match="same number"):
            run_folder_inference(
                input_dir=str(img_dir),
                output_path=str(tmp_path / "out.json"),
                prompts=["crack", "rust"],
                category_ids=[1],           # mismatch: 2 prompts, 1 ID
                categories_file=categories_coco_file,
                config_path="dummy.yaml",
                weights_path=None,
                threshold=0.5,
                resolution=1008,
                nms_iou=0.5,
                simplify_epsilon=0.0,
                image_exts=["jpg"],
                device="cpu",
            )


def test_duplicate_prompts_raises(categories_coco_file, tmp_path):
    """run_folder_inference raises ValueError when duplicate prompts are provided."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    mock_infer_sam = MagicMock()
    with patch.dict(sys.modules, {"infer_sam": mock_infer_sam}):
        with pytest.raises(ValueError, match="Duplicate prompts"):
            run_folder_inference(
                input_dir=str(img_dir),
                output_path=str(tmp_path / "out.json"),
                prompts=["crack", "crack"],
                category_ids=[1, 2],
                categories_file=categories_coco_file,
                config_path="dummy.yaml",
                weights_path=None,
                threshold=0.5,
                resolution=1008,
                nms_iou=0.5,
                simplify_epsilon=0.0,
                image_exts=["jpg"],
                device="cpu",
            )


def test_inference_failure_skips_image(categories_coco_file, tmp_path):
    """When model.predict() raises, the image is skipped but output is still written."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = PILImage.new("RGB", (100, 80), color=128)
    img.save(img_dir / "img_001.jpg")
    output_path = tmp_path / "output.json"

    # Mock infer_sam module to avoid torch import at patch time
    mock_infer_sam = MagicMock()
    MockModel = MagicMock()
    instance = MockModel.return_value
    instance.predict.side_effect = RuntimeError("model exploded")
    mock_infer_sam.SAM3LoRAInference = MockModel

    with patch.dict(sys.modules, {"infer_sam": mock_infer_sam}):
        run_folder_inference(
            input_dir=str(img_dir),
            output_path=str(output_path),
            prompts=["crack"],
            category_ids=[1],
            categories_file=categories_coco_file,
            config_path="dummy.yaml",
            weights_path=None,
            threshold=0.5,
            resolution=1008,
            nms_iou=0.5,
            simplify_epsilon=0.0,
            image_exts=["jpg"],
            device="cpu",
        )

    with open(output_path) as f:
        coco = json.load(f)

    # Output written despite failure; annotations empty (inference was skipped)
    assert "images" in coco
    assert "annotations" in coco
    assert len(coco["annotations"]) == 0

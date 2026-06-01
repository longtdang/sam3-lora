"""Tests for split_dataset.py"""
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_mock_coco(num_images: int, with_segmentation: bool = False) -> dict:
    """Build a minimal valid COCO dict with num_images images."""
    images = [
        {
            "id": i,
            "width": 100,
            "height": 80,
            "file_name": f"img_{i:02d}.png",
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
        for i in range(1, num_images + 1)
    ]
    annotations = []
    ann_id = 1
    for i in range(1, num_images + 1):
        seg = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]] if with_segmentation else []
        annotations.append(
            {
                "id": ann_id,
                "image_id": i,
                "category_id": 1 if i % 2 == 0 else 2,
                "segmentation": seg,
                "area": 400.0,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "iscrowd": 0,
                "attributes": {"occluded": False},
            }
        )
        ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "crack", "supercategory": ""},
            {"id": 2, "name": "spall", "supercategory": ""},
        ],
    }


@pytest.fixture
def tmp_dataset(tmp_path):
    """Return a factory that creates a versioned dataset directory."""
    def _make(version: str = "v1", num_images: int = 10, with_segmentation: bool = False):
        dataset_dir = tmp_path / "dataset" / version
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)
        coco = make_mock_coco(num_images, with_segmentation)
        # Write COCO JSON
        coco_json = dataset_dir / "_annotations.coco.json"
        coco_json.write_text(json.dumps(coco))
        # Write small 1×1 PNG images
        for img_info in coco["images"]:
            img = Image.new("RGB", (100, 80), color=(128, 64, 32))
            img.save(images_dir / img_info["file_name"])
        return dataset_dir, coco

    return _make


# ---------------------------------------------------------------------------
# validate_ratios
# ---------------------------------------------------------------------------

def test_validate_ratios_valid():
    from split_dataset import validate_ratios
    # Should not raise
    validate_ratios(0.7, 0.2, 0.1)


def test_validate_ratios_invalid_sum():
    from split_dataset import validate_ratios
    with pytest.raises(SystemExit):
        validate_ratios(0.6, 0.2, 0.1)


def test_validate_ratios_negative():
    from split_dataset import validate_ratios
    with pytest.raises(SystemExit):
        validate_ratios(-0.1, 0.8, 0.3)


def test_validate_ratios_zero_train():
    from split_dataset import validate_ratios
    with pytest.raises(SystemExit):
        validate_ratios(0.0, 0.7, 0.3)


# ---------------------------------------------------------------------------
# find_coco_json
# ---------------------------------------------------------------------------

def test_find_coco_json_default_name(tmp_path):
    from split_dataset import find_coco_json
    coco = tmp_path / "_annotations.coco.json"
    coco.write_text("{}")
    assert find_coco_json(tmp_path) == coco


def test_find_coco_json_fallback_single(tmp_path):
    from split_dataset import find_coco_json
    coco = tmp_path / "custom_name.json"
    coco.write_text("{}")
    assert find_coco_json(tmp_path) == coco


def test_find_coco_json_explicit_override(tmp_path):
    from split_dataset import find_coco_json
    coco = tmp_path / "my.json"
    coco.write_text("{}")
    assert find_coco_json(tmp_path, explicit=coco) == coco


def test_find_coco_json_multiple_json_exits(tmp_path):
    from split_dataset import find_coco_json
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    with pytest.raises(SystemExit):
        find_coco_json(tmp_path)


def test_find_coco_json_none_exits(tmp_path):
    from split_dataset import find_coco_json
    with pytest.raises(SystemExit):
        find_coco_json(tmp_path)


# ---------------------------------------------------------------------------
# build_image_index
# ---------------------------------------------------------------------------

def test_build_image_index_groups_annotations():
    from split_dataset import build_image_index
    coco = make_mock_coco(3)
    index = build_image_index(coco)
    assert len(index) == 3
    # Each entry has image info + annotations list
    for entry in index:
        assert "id" in entry
        assert "file_name" in entry
        assert "annotations" in entry
        assert len(entry["annotations"]) == 1  # one ann per image in mock


def test_build_image_index_image_no_annotations_is_included():
    from split_dataset import build_image_index
    coco = make_mock_coco(2)
    # Add an image with no annotation
    coco["images"].append({"id": 99, "width": 10, "height": 10, "file_name": "orphan.png",
                           "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0})
    index = build_image_index(coco)
    assert len(index) == 3
    orphan = next(e for e in index if e["id"] == 99)
    assert orphan["annotations"] == []


# ---------------------------------------------------------------------------
# split_images
# ---------------------------------------------------------------------------

def test_split_images_alphabetical_order():
    from split_dataset import split_images
    images = [{"file_name": f"img_{i:02d}.png"} for i in range(10)]
    random.shuffle(images)  # deliberately unsorted input
    train, valid, test = split_images(images, 0.7, 0.2, 0.1, random_order=False)
    all_files = [e["file_name"] for e in train + valid + test]
    # Must be sorted alphabetically
    assert all_files == sorted(all_files)


def test_split_images_sizes_70_20_10():
    from split_dataset import split_images
    images = [{"file_name": f"img_{i:02d}.png"} for i in range(10)]
    train, valid, test = split_images(images, 0.7, 0.2, 0.1)
    assert len(train) == 7
    assert len(valid) == 2
    assert len(test) == 1


def test_split_images_remainder_goes_to_train():
    from split_dataset import split_images
    # 11 images at 70/20/10: floor(11*0.2)=2, floor(11*0.1)=1, train=11-2-1=8
    images = [{"file_name": f"img_{i:02d}.png"} for i in range(11)]
    train, valid, test = split_images(images, 0.7, 0.2, 0.1)
    assert len(train) == 8
    assert len(valid) == 2
    assert len(test) == 1


def test_split_images_random_reproducible():
    from split_dataset import split_images
    images = [{"file_name": f"img_{i:02d}.png"} for i in range(20)]
    train_a, _, _ = split_images(images, 0.7, 0.2, 0.1, random_order=True, seed=42)
    train_b, _, _ = split_images(images, 0.7, 0.2, 0.1, random_order=True, seed=42)
    assert [e["file_name"] for e in train_a] == [e["file_name"] for e in train_b]


def test_split_images_random_differs_from_alphabetical():
    from split_dataset import split_images
    images = [{"file_name": f"img_{i:02d}.png"} for i in range(20)]
    train_alpha, _, _ = split_images(images, 0.7, 0.2, 0.1, random_order=False)
    train_rand, _, _ = split_images(images, 0.7, 0.2, 0.1, random_order=True, seed=99)
    # Very unlikely to be identical
    assert [e["file_name"] for e in train_alpha] != [e["file_name"] for e in train_rand]


def test_split_images_no_overlap():
    from split_dataset import split_images
    images = [{"file_name": f"img_{i:02d}.png"} for i in range(10)]
    train, valid, test = split_images(images, 0.7, 0.2, 0.1)
    all_files = set(e["file_name"] for e in train + valid + test)
    assert len(all_files) == 10  # no duplicates

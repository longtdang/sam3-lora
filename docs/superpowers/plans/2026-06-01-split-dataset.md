# split_dataset.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `split_dataset.py` — a standalone CLI script that converts a CVAT COCO-format dataset (`dataset/vX/`) into SAM3 per-image JSON format and splits it into `data/train/`, `data/valid/`, and `data/test/` directories.

**Architecture:** A single module at the repo root exposing six pure functions (`find_coco_json`, `validate_ratios`, `build_image_index`, `split_images`, `convert_annotations`, `copy_split`) plus a `main()` entry point. Each function is independently testable. The script reads from an input versioned dataset folder and writes to `data/` with `exist_ok=True` for safe multi-version merging.

**Tech Stack:** Python 3 standard library (`argparse`, `json`, `shutil`, `pathlib`, `random`, `math`), `pycocotools>=2.0.6` (optional, for mask rasterization), `Pillow>=10.0.0` (already in `requirements.txt`).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `split_dataset.py` | **Create** | CLI entry point + all split/convert logic |
| `tests/__init__.py` | **Create** | Make tests a package |
| `tests/test_split_dataset.py` | **Create** | Unit + integration tests |

---

## Task 1: Create test fixtures and infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_split_dataset.py`

- [ ] **Step 1: Create the tests package and fixture helpers**

Create `tests/__init__.py` (empty):
```python
```

Create `tests/test_split_dataset.py` with fixtures only (no test functions yet):

```python
"""Tests for split_dataset.py"""
import json
import math
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
```

- [ ] **Step 2: Verify the fixture file is importable**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -c "import tests.test_split_dataset; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit the test skeleton**

```bash
git add tests/__init__.py tests/test_split_dataset.py
git commit -m "test: add split_dataset test infrastructure and fixtures"
```

---

## Task 2: Argument parsing and ratio validation (TDD)

**Files:**
- Create: `split_dataset.py` (initial scaffold + `validate_ratios` + `parse_args`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests for ratio validation**

Append to `tests/test_split_dataset.py`:

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py::test_validate_ratios_valid tests/test_split_dataset.py::test_validate_ratios_invalid_sum -v 2>&1 | tail -15
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'split_dataset'`

- [ ] **Step 3: Create `split_dataset.py` with argument parsing and `validate_ratios`**

Create `split_dataset.py`:

```python
"""
split_dataset.py — Convert a CVAT COCO dataset and split into SAM3 train/valid/test.

Usage:
    python split_dataset.py --dataset-dir dataset/v1 [--random] [--seed 42]
                            [--train 0.7] [--val 0.2] [--test 0.1]
                            [--coco-json path/to/annotations.json]
"""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a CVAT COCO dataset and split into SAM3 train/valid/test."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Path to the versioned dataset folder (e.g. dataset/v1)",
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=None,
        help="Explicit path to the COCO JSON file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle images randomly before splitting (default: alphabetical order)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --random is set (default: 42)",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Training fraction (default: 0.7)")
    parser.add_argument("--val", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default: 0.1)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Ratio validation
# ---------------------------------------------------------------------------

def validate_ratios(train: float, val: float, test: float) -> None:
    """Exit with an error message if the ratios are invalid."""
    for name, value in [("--train", train), ("--val", val), ("--test", test)]:
        if value < 0:
            sys.exit(f"Error: {name} must be >= 0, got {value}")
    if train <= 0:
        sys.exit("Error: --train must be > 0")
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        sys.exit(
            f"Error: --train + --val + --test must sum to 1.0, got {total:.6f}"
        )
```

- [ ] **Step 4: Run tests — they should pass now**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py::test_validate_ratios_valid tests/test_split_dataset.py::test_validate_ratios_invalid_sum tests/test_split_dataset.py::test_validate_ratios_negative tests/test_split_dataset.py::test_validate_ratios_zero_train -v 2>&1 | tail -15
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: add split_dataset arg parsing and ratio validation"
```

---

## Task 3: COCO JSON auto-detection (TDD)

**Files:**
- Modify: `split_dataset.py` (add `find_coco_json`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_dataset.py`:

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "find_coco_json" -v 2>&1 | tail -15
```

Expected: `ERROR` — `ImportError: cannot import name 'find_coco_json'`

- [ ] **Step 3: Implement `find_coco_json` in `split_dataset.py`**

Append to `split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# COCO JSON discovery
# ---------------------------------------------------------------------------

def find_coco_json(dataset_dir: Path, explicit: Path = None) -> Path:
    """
    Locate the COCO JSON file inside dataset_dir.

    Priority:
      1. explicit path (from --coco-json)
      2. _annotations.coco.json in dataset_dir
      3. any single .json file at the root of dataset_dir
      4. error
    """
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"Error: specified --coco-json '{explicit}' does not exist")
        return explicit

    default = dataset_dir / "_annotations.coco.json"
    if default.exists():
        return default

    candidates = [p for p in dataset_dir.glob("*.json") if p.is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        sys.exit(
            f"Error: no COCO JSON file found in '{dataset_dir}'. "
            "Use --coco-json to specify it explicitly."
        )
    sys.exit(
        f"Error: multiple JSON files found in '{dataset_dir}': "
        f"{[p.name for p in candidates]}. "
        "Use --coco-json to specify which one to use."
    )
```

- [ ] **Step 4: Run tests — they should pass**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "find_coco_json" -v 2>&1 | tail -15
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: add COCO JSON auto-detection"
```

---

## Task 4: Build image index from COCO data (TDD)

**Files:**
- Modify: `split_dataset.py` (add `build_image_index`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_dataset.py`:

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "build_image_index" -v 2>&1 | tail -10
```

Expected: `ERROR` — `ImportError: cannot import name 'build_image_index'`

- [ ] **Step 3: Implement `build_image_index` in `split_dataset.py`**

Append to `split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# COCO index construction
# ---------------------------------------------------------------------------

def build_image_index(coco_data: dict) -> list:
    """
    Return a list of image dicts, each augmented with an 'annotations' key
    containing all COCO annotation dicts for that image.
    """
    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    ann_by_image: dict = {}
    for ann in coco_data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    result = []
    for img in coco_data["images"]:
        entry = dict(img)
        entry["annotations"] = ann_by_image.get(img["id"], [])
        entry["_category_map"] = category_map  # carried along for conversion step
        result.append(entry)
    return result
```

- [ ] **Step 4: Run tests — they should pass**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "build_image_index" -v 2>&1 | tail -10
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: add COCO image index builder"
```

---

## Task 5: Split logic — alphabetical and random (TDD)

**Files:**
- Modify: `split_dataset.py` (add `split_images`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_dataset.py`:

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "split_images" -v 2>&1 | tail -15
```

Expected: `ERROR` — `ImportError: cannot import name 'split_images'`

- [ ] **Step 3: Implement `split_images` in `split_dataset.py`**

Append to `split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def split_images(
    images: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_order: bool = False,
    seed: int = 42,
) -> tuple:
    """
    Sort or shuffle images and split into (train, valid, test) lists.

    Remainder (from floor rounding) is added to the train set.
    """
    ordered = sorted(images, key=lambda img: img["file_name"])
    if random_order:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    n = len(ordered)
    n_val = math.floor(n * val_ratio)
    n_test = math.floor(n * test_ratio)
    n_train = n - n_val - n_test

    train = ordered[:n_train]
    valid = ordered[n_train: n_train + n_val]
    test = ordered[n_train + n_val:]
    return train, valid, test
```

- [ ] **Step 4: Run tests — they should pass**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "split_images" -v 2>&1 | tail -15
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: add split_images with alphabetical and random modes"
```

---

## Task 6: COCO-to-SAM3 annotation conversion (TDD)

**Files:**
- Modify: `split_dataset.py` (add `convert_annotations`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# convert_annotations
# ---------------------------------------------------------------------------

def test_convert_annotations_bbox_format():
    from split_dataset import convert_annotations
    anns = [{"category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "segmentation": []}]
    category_map = {1: "crack"}
    result = convert_annotations(anns, category_map, width=100, height=80)
    assert result["bboxes"] == [[10, 20, 40, 60]]  # [x, y, x+w, y+h]


def test_convert_annotations_text_prompt_deduplicated():
    from split_dataset import convert_annotations
    anns = [
        {"category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": []},
        {"category_id": 1, "bbox": [5, 5, 10, 10], "segmentation": []},
        {"category_id": 2, "bbox": [20, 20, 10, 10], "segmentation": []},
    ]
    category_map = {1: "crack", 2: "spall"}
    result = convert_annotations(anns, category_map, width=100, height=80)
    # Sorted, deduplicated
    assert result["text_prompt"] == "crack, spall"


def test_convert_annotations_empty_segmentation_gives_empty_masks():
    from split_dataset import convert_annotations
    anns = [{"category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": []}]
    category_map = {1: "crack"}
    result = convert_annotations(anns, category_map, width=100, height=80)
    assert result["masks"] == []


def test_convert_annotations_no_annotations_gives_empty():
    from split_dataset import convert_annotations
    result = convert_annotations([], {}, width=100, height=80)
    assert result["bboxes"] == []
    assert result["masks"] == []
    assert result["text_prompt"] == ""


def test_convert_annotations_with_segmentation_gives_mask(tmp_dataset):
    from split_dataset import convert_annotations
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        pytest.skip("pycocotools not installed")
    anns = [
        {
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "segmentation": [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
        }
    ]
    category_map = {1: "crack"}
    result = convert_annotations(anns, category_map, width=100, height=80)
    assert len(result["masks"]) == 1
    mask = result["masks"][0]
    assert len(mask) == 80        # height rows
    assert len(mask[0]) == 100    # width cols
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "convert_annotations" -v 2>&1 | tail -15
```

Expected: `ERROR` — `ImportError: cannot import name 'convert_annotations'`

- [ ] **Step 3: Implement `convert_annotations` in `split_dataset.py`**

Append to `split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# Annotation conversion
# ---------------------------------------------------------------------------

def _try_rasterize_segmentation(segmentation: list, width: int, height: int):
    """
    Convert a COCO polygon segmentation to a binary mask (list of lists).
    Returns None if pycocotools is unavailable or segmentation is empty.
    """
    if not segmentation:
        return None
    try:
        from pycocotools import mask as mask_util
        import numpy as np
        rles = mask_util.frPyObjects(segmentation, height, width)
        rle = mask_util.merge(rles)
        binary = mask_util.decode(rle)  # numpy array H x W uint8
        return binary.tolist()
    except ImportError:
        return None


def convert_annotations(
    annotations: list,
    category_map: dict,
    width: int,
    height: int,
) -> dict:
    """
    Convert COCO annotation dicts for a single image to SAM3 format.

    Returns:
        {
            "text_prompt": "crack, spall",
            "bboxes": [[x1, y1, x2, y2], ...],
            "masks": [[[0, 1, ...], ...], ...]  # or [] if not available
        }
    """
    if not annotations:
        return {"text_prompt": "", "bboxes": [], "masks": []}

    bboxes = []
    masks = []
    category_names = []

    for ann in annotations:
        # COCO bbox: [x, y, width, height] → SAM3: [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])

        cat_name = category_map.get(ann["category_id"], f"class_{ann['category_id']}")
        category_names.append(cat_name)

        mask = _try_rasterize_segmentation(ann.get("segmentation", []), width, height)
        if mask is not None:
            masks.append(mask)

    text_prompt = ", ".join(sorted(set(category_names)))
    return {"text_prompt": text_prompt, "bboxes": bboxes, "masks": masks}
```

- [ ] **Step 4: Run tests — they should pass**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "convert_annotations" -v 2>&1 | tail -15
```

Expected: `5 passed` (or `4 passed, 1 skipped` if pycocotools is absent)

- [ ] **Step 5: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: add COCO-to-SAM3 annotation converter with optional mask rasterization"
```

---

## Task 7: File output — copy images and write SAM3 JSON (TDD)

**Files:**
- Modify: `split_dataset.py` (add `copy_split`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# copy_split
# ---------------------------------------------------------------------------

def test_copy_split_creates_directories(tmp_dataset, tmp_path):
    from split_dataset import build_image_index, copy_split
    dataset_dir, coco = tmp_dataset("v1", num_images=3)
    index = build_image_index(coco)
    output_root = tmp_path / "data"
    copy_split("train", index, dataset_dir / "images", output_root)
    assert (output_root / "train" / "images").is_dir()
    assert (output_root / "train" / "annotations").is_dir()


def test_copy_split_copies_images(tmp_dataset, tmp_path):
    from split_dataset import build_image_index, copy_split
    dataset_dir, coco = tmp_dataset("v1", num_images=3)
    index = build_image_index(coco)
    output_root = tmp_path / "data"
    copy_split("train", index, dataset_dir / "images", output_root)
    copied = list((output_root / "train" / "images").iterdir())
    assert len(copied) == 3


def test_copy_split_writes_sam3_json(tmp_dataset, tmp_path):
    from split_dataset import build_image_index, copy_split
    dataset_dir, coco = tmp_dataset("v1", num_images=3)
    index = build_image_index(coco)
    output_root = tmp_path / "data"
    copy_split("train", index, dataset_dir / "images", output_root)
    ann_files = list((output_root / "train" / "annotations").iterdir())
    assert len(ann_files) == 3
    for ann_file in ann_files:
        ann = json.loads(ann_file.read_text())
        assert "text_prompt" in ann
        assert "bboxes" in ann
        assert "masks" in ann


def test_copy_split_skips_missing_image(tmp_dataset, tmp_path, capsys):
    from split_dataset import build_image_index, copy_split
    dataset_dir, coco = tmp_dataset("v1", num_images=3)
    # Remove one image from disk
    (dataset_dir / "images" / "img_02.png").unlink()
    index = build_image_index(coco)
    output_root = tmp_path / "data"
    copy_split("train", index, dataset_dir / "images", output_root)
    out = capsys.readouterr().out
    assert "Warning" in out
    assert len(list((output_root / "train" / "images").iterdir())) == 2


def test_copy_split_safe_merge(tmp_dataset, tmp_path):
    from split_dataset import build_image_index, copy_split
    dataset_dir, coco = tmp_dataset("v1", num_images=2)
    index = build_image_index(coco)
    output_root = tmp_path / "data"
    # Run twice — second call must not raise
    copy_split("train", index, dataset_dir / "images", output_root)
    copy_split("train", index, dataset_dir / "images", output_root)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "copy_split" -v 2>&1 | tail -15
```

Expected: `ERROR` — `ImportError: cannot import name 'copy_split'`

- [ ] **Step 3: Implement `copy_split` in `split_dataset.py`**

Append to `split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def copy_split(
    split_name: str,
    image_entries: list,
    images_src_dir: Path,
    output_root: Path,
) -> dict:
    """
    Copy images and write SAM3 JSON annotations for one split.

    Returns a summary dict: {"copied": int, "skipped": int}
    """
    images_dst = output_root / split_name / "images"
    annotations_dst = output_root / split_name / "annotations"
    images_dst.mkdir(parents=True, exist_ok=True)
    annotations_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for entry in image_entries:
        file_name = entry["file_name"]
        src_image = images_src_dir / file_name

        if not src_image.exists():
            print(f"  Warning: image not found, skipping — {src_image}")
            skipped += 1
            continue

        # Copy image
        shutil.copy2(src_image, images_dst / file_name)

        # Convert and write SAM3 annotation
        sam3_ann = convert_annotations(
            entry["annotations"],
            entry["_category_map"],
            width=entry["width"],
            height=entry["height"],
        )
        stem = Path(file_name).stem
        ann_path = annotations_dst / f"{stem}.json"
        ann_path.write_text(json.dumps(sam3_ann, indent=2))

        copied += 1

    return {"copied": copied, "skipped": skipped}
```

- [ ] **Step 4: Run tests — they should pass**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "copy_split" -v 2>&1 | tail -15
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: add copy_split — image copy and SAM3 JSON output"
```

---

## Task 8: Wire up `main()` and end-to-end integration test

**Files:**
- Modify: `split_dataset.py` (add `main()`)
- Modify: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# Integration: main()
# ---------------------------------------------------------------------------

def test_main_end_to_end_alphabetical(tmp_dataset, tmp_path, monkeypatch):
    from split_dataset import main
    dataset_dir, coco = tmp_dataset("v1", num_images=10)
    output_root = tmp_path / "data"
    monkeypatch.chdir(tmp_path)

    # Patch sys.argv
    import sys
    monkeypatch.setattr(
        sys, "argv",
        [
            "split_dataset.py",
            "--dataset-dir", str(dataset_dir),
            "--train", "0.7",
            "--val", "0.2",
            "--test", "0.1",
        ],
    )
    main(output_root=output_root)

    train_imgs = list((output_root / "train" / "images").iterdir())
    valid_imgs = list((output_root / "valid" / "images").iterdir())
    test_imgs  = list((output_root / "test"  / "images").iterdir())

    assert len(train_imgs) == 7
    assert len(valid_imgs) == 2
    assert len(test_imgs)  == 1

    # Check alphabetical order — train should hold first 7 filenames sorted
    all_sorted = sorted(img["file_name"] for img in coco["images"])
    train_names = sorted(p.name for p in train_imgs)
    assert train_names == all_sorted[:7]


def test_main_end_to_end_random(tmp_dataset, tmp_path, monkeypatch):
    from split_dataset import main
    dataset_dir, _ = tmp_dataset("v1", num_images=10)
    output_root = tmp_path / "data"
    monkeypatch.chdir(tmp_path)

    import sys
    monkeypatch.setattr(
        sys, "argv",
        [
            "split_dataset.py",
            "--dataset-dir", str(dataset_dir),
            "--random",
            "--seed", "7",
        ],
    )
    main(output_root=output_root)

    total = (
        len(list((output_root / "train" / "images").iterdir()))
        + len(list((output_root / "valid" / "images").iterdir()))
        + len(list((output_root / "test"  / "images").iterdir()))
    )
    assert total == 10
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -k "test_main" -v 2>&1 | tail -15
```

Expected: `ERROR` — `ImportError: cannot import name 'main'` or `TypeError` (main has no `output_root` param yet)

- [ ] **Step 3: Implement `main()` in `split_dataset.py`**

Append to `split_dataset.py`:

```python
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(output_root: Path = None, argv=None) -> None:
    args = parse_args(argv)
    validate_ratios(args.train, args.val, args.test)

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        sys.exit(f"Error: --dataset-dir '{dataset_dir}' does not exist")

    coco_json_path = find_coco_json(dataset_dir, explicit=args.coco_json)
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    image_index = build_image_index(coco_data)
    total = len(image_index)
    print(f"Found {total} images in COCO JSON")

    train_imgs, valid_imgs, test_imgs = split_images(
        image_index,
        args.train,
        args.val,
        args.test,
        random_order=args.random,
        seed=args.seed,
    )

    images_src = dataset_dir / "images"
    root = output_root if output_root is not None else Path("data")

    print(f"\nSplitting: {len(train_imgs)} train / {len(valid_imgs)} valid / {len(test_imgs)} test")
    print(f"Output root: {root.resolve()}\n")

    for split_name, entries in [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]:
        result = copy_split(split_name, entries, images_src, root)
        print(
            f"  [{split_name:6s}]  copied: {result['copied']}  skipped: {result['skipped']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -v 2>&1 | tail -30
```

Expected: all tests pass (one possible skip for pycocotools mask test if not installed)

- [ ] **Step 5: Smoke-test the script directly**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python split_dataset.py --dataset-dir data/train --train 0.7 --val 0.2 --test 0.1 2>&1
```

Expected output (approximate):
```
Loading COCO annotations from: .../data/train/_annotations.coco.json
Found 16 images in COCO JSON

Splitting: 12 train / 3 valid / 1 test
Output root: .../data

  [train ]  copied: 12  skipped: 0
  [valid ]  copied: 3  skipped: 0
  [test  ]  copied: 1  skipped: 0

Done.
```

- [ ] **Step 6: Commit**

```bash
git add split_dataset.py tests/test_split_dataset.py
git commit -m "feat: wire up main() entry point and add end-to-end integration tests"
```

---

## Task 9: Final run — all tests green

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python -m pytest tests/test_split_dataset.py -v 2>&1
```

Expected: all tests pass (1 possible skip for pycocotools mask test)

- [ ] **Step 2: Verify the CLI help text is clean**

```bash
cd /home/longtdang/KMS/SAM3_LoRA && python split_dataset.py --help
```

Expected: all arguments listed with descriptions, no errors.

- [ ] **Step 3: Final commit (if any cleanup needed)**

```bash
git add -u
git commit -m "chore: finalize split_dataset.py"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** `find_coco_json` ✓, `validate_ratios` ✓, alphabetical/random split ✓, 70/20/10 default ✓, `--coco-json` override ✓, `--seed` ✓, COCO→SAM3 bbox conversion ✓, text_prompt ✓, mask rasterization ✓, merge-safe output ✓, warn on missing image ✓, warn on multiple JSON ✓
- [x] **No placeholders:** all steps have concrete code and expected output
- [x] **Type consistency:** `copy_split` calls `convert_annotations(entry["annotations"], entry["_category_map"], width=..., height=...)` — matches the signature defined in Task 6; `build_image_index` attaches `_category_map` to each entry in Task 4
- [x] **`main()` `output_root` param:** used in tests to redirect output to `tmp_path`; production usage (no arg) defaults to `Path("data")`

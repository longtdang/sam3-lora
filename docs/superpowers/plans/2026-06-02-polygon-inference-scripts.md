# Polygon Inference Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two reusable inference scripts — one that outputs a polygon-overlay PNG for a single image, and one that runs batch inference over a folder and writes a CVAT-importable COCO JSON file.

**Architecture:** A shared `polygon_utils.py` module holds all mask→polygon conversion logic (contour extraction, RDP simplification, bbox/area helpers). `infer_sam_polygon.py` imports both `SAM3LoRAInference` from `infer_sam.py` and `polygon_utils` to render contours. `infer_folder_coco.py` does the same for batch COCO output.

**Tech Stack:** Python 3, `opencv-python` (cv2), `numpy`, `Pillow`, `argparse`, `pytest`, existing `SAM3LoRAInference` class from `infer_sam.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `polygon_utils.py` | Create | Mask→polygon conversion, simplification, bbox/area helpers |
| `infer_sam_polygon.py` | Create | Single-image inference → polygon overlay PNG |
| `infer_folder_coco.py` | Create | Folder inference → COCO JSON |
| `tests/test_polygon_utils.py` | Create | Unit tests for polygon_utils |
| `tests/test_infer_sam_polygon.py` | Create | Tests for infer_sam_polygon (mocked model) |
| `tests/test_infer_folder_coco.py` | Create | Tests for infer_folder_coco (mocked model) |
| `infer_sam.py` | No change | Existing — SAM3LoRAInference class stays untouched |

---

## Task 1: Shared polygon utilities (`polygon_utils.py`)

**Files:**
- Create: `polygon_utils.py`
- Test: `tests/test_polygon_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_polygon_utils.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/test_polygon_utils.py -v
```

Expected: `ImportError: No module named 'polygon_utils'`

- [ ] **Step 3: Implement `polygon_utils.py`**

Create `polygon_utils.py` at the repo root:

```python
"""
Shared polygon utilities for SAM3 inference scripts.

Provides mask-to-polygon conversion, RDP simplification, and COCO bbox/area helpers.
"""

from typing import List, Optional

import cv2
import numpy as np


def mask_to_polygons(
    mask: np.ndarray,
    simplify_epsilon: float = 0.0,
) -> List[List[float]]:
    """
    Convert a binary mask to a list of COCO-format polygon coordinate lists.

    Each external contour with >= 3 points becomes one flat list [x1, y1, x2, y2, ...].
    Contours with fewer than 3 points are silently skipped.

    Args:
        mask: Binary array [H, W] (bool or uint8).
        simplify_epsilon: Ramer-Douglas-Peucker epsilon in pixels. 0 = no simplification.

    Returns:
        List of flat float coordinate lists, one per valid contour.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[List[float]] = []
    for contour in contours:
        if simplify_epsilon > 0:
            contour = cv2.approxPolyDP(contour, simplify_epsilon, closed=True)
        if len(contour) < 3:
            continue
        polygons.append(contour.reshape(-1).astype(float).tolist())

    return polygons


def polygons_to_bbox(polygons: List[List[float]]) -> Optional[List[float]]:
    """
    Compute COCO bbox [x, y, w, h] encompassing all given polygons.

    Returns None if the input list is empty.
    """
    if not polygons:
        return None
    all_x: List[float] = []
    all_y: List[float] = []
    for poly in polygons:
        all_x.extend(poly[0::2])
        all_y.extend(poly[1::2])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def polygons_to_area(polygons: List[List[float]]) -> float:
    """
    Compute total area across all polygons using cv2.contourArea (Shoelace formula).
    """
    total = 0.0
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        total += float(cv2.contourArea(pts))
    return total
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_polygon_utils.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add polygon_utils.py tests/test_polygon_utils.py
git commit -m "feat: add polygon_utils with mask-to-polygon conversion and COCO helpers"
```

---

## Task 2: Single-image polygon inference script (`infer_sam_polygon.py`)

**Files:**
- Create: `infer_sam_polygon.py`
- Test: `tests/test_infer_sam_polygon.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_infer_sam_polygon.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_infer_sam_polygon.py -v
```

Expected: `ImportError: No module named 'infer_sam_polygon'`

- [ ] **Step 3: Implement `infer_sam_polygon.py`**

Create `infer_sam_polygon.py`:

```python
#!/usr/bin/env python3
"""
SAM3 + LoRA Inference — Polygon Output

Same as infer_sam.py but renders polygon contours instead of filled masks.
Optionally simplifies polygons using Ramer-Douglas-Peucker (--simplify epsilon).

Usage:
    python infer_sam_polygon.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt crack defect \
        --output output_polygon.png \
        --simplify 2.0
"""

import argparse

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from infer_sam import SAM3LoRAInference
from polygon_utils import mask_to_polygons


# Per-prompt BGR colors for cv2 drawing
_COLORS_BGR = [
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
]


def draw_polygons(
    results: dict,
    simplify_epsilon: float = 0.0,
    show_boxes: bool = False,
) -> PILImage.Image:
    """
    Draw polygon contours (and optionally bounding boxes) on the original image.

    Args:
        results: Output dict from SAM3LoRAInference.predict().
        simplify_epsilon: RDP epsilon in pixels. 0 = no simplification.
        show_boxes: If True, also draw bounding boxes.

    Returns:
        PIL Image with contours drawn.
    """
    pil_image = results["_image"]
    canvas = np.array(pil_image.convert("RGB"))
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    prompt_indices = sorted(k for k in results if k != "_image")

    for idx in prompt_indices:
        result = results[idx]
        color = _COLORS_BGR[idx % len(_COLORS_BGR)]

        if result["num_detections"] == 0 or result["masks"] is None:
            continue

        masks = result["masks"]       # [N, H, W] bool
        scores = result["scores"]     # [N]
        boxes = result["boxes"]       # [N, 4] xyxy
        prompt = result["prompt"]

        for i in range(result["num_detections"]):
            mask = masks[i]
            score = float(scores[i]) if scores is not None else 0.0
            polygons = mask_to_polygons(mask, simplify_epsilon)

            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas_bgr, [pts], isClosed=True, color=color, thickness=2)

            if polygons:
                # Label at the top-left of the first polygon's bbox
                first_pts = np.array(polygons[0], dtype=np.int32).reshape(-1, 2)
                label_x = int(first_pts[:, 0].min())
                label_y = max(0, int(first_pts[:, 1].min()) - 5)
                label = f"{prompt}: {score:.2f}"
                cv2.putText(
                    canvas_bgr, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            if show_boxes and boxes is not None:
                x1, y1, x2, y2 = (int(v) for v in boxes[i])
                cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), color, 2)

    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(canvas_rgb)


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Polygon Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--weights", type=str, default=None, help="LoRA weights path (auto-detected if omitted)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, nargs="+", default=["object"], help="Text prompt(s)")
    parser.add_argument("--output", type=str, default="output_polygon.png", help="Output PNG path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--resolution", type=int, default=1008, help="Input resolution")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="RDP simplification epsilon in pixels (0 = off)")
    parser.add_argument("--boundingbox", action="store_true", help="Also draw bounding boxes")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        device=args.device,
    )

    results = inferencer.predict(args.image, args.prompt)
    out_image = draw_polygons(results, simplify_epsilon=args.simplify, show_boxes=args.boundingbox)
    out_image.save(args.output)

    print(f"\n✅ Saved polygon output to {args.output}")
    for idx in sorted(k for k in results if k != "_image"):
        r = results[idx]
        print(f"   Prompt '{r['prompt']}': {r['num_detections']} detections")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_infer_sam_polygon.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add infer_sam_polygon.py tests/test_infer_sam_polygon.py
git commit -m "feat: add infer_sam_polygon.py for polygon contour visualization"
```

---

## Task 3: Folder inference → COCO JSON (`infer_folder_coco.py`)

**Files:**
- Create: `infer_folder_coco.py`
- Test: `tests/test_infer_folder_coco.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_infer_folder_coco.py`:

```python
"""Tests for infer_folder_coco.py — model is mocked, no GPU needed."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    with patch("infer_folder_coco.SAM3LoRAInference") as MockModel:
        instance = MockModel.return_value
        instance.predict.return_value = mock_results

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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_infer_folder_coco.py -v
```

Expected: `ImportError: No module named 'infer_folder_coco'`

- [ ] **Step 3: Implement `infer_folder_coco.py`**

Create `infer_folder_coco.py`:

```python
#!/usr/bin/env python3
"""
SAM3 + LoRA Folder Inference → COCO JSON

Runs inference on all images in a folder and writes a CVAT-importable COCO JSON file.
Supports multiple text prompts each mapped to a COCO category ID.

Usage:
    python infer_folder_coco.py \
        --config configs/full_lora_config.yaml \
        --input-dir images/ \
        --output results.json \
        --categories dataset/_annotations.coco.json \
        --prompt "surface crack" defect rust \
        --category_id 1 2 3 \
        --simplify 2.0
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image as PILImage

from infer_sam import SAM3LoRAInference
from polygon_utils import mask_to_polygons, polygons_to_bbox, polygons_to_area


def load_categories(categories_file: str) -> list:
    """
    Read the 'categories' array from an existing COCO JSON file.

    Args:
        categories_file: Path to a COCO-format JSON file.

    Returns:
        List of category dicts with at least 'id', 'name', 'supercategory'.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(categories_file)
    if not path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_file}")
    with open(path) as f:
        data = json.load(f)
    return data["categories"]


def results_to_coco_annotations(
    results: dict,
    image_id: int,
    prompt_to_cat_id: Dict[str, int],
    simplify_epsilon: float,
    start_ann_id: int,
) -> List[dict]:
    """
    Convert SAM3LoRAInference.predict() results for one image into COCO annotation dicts.

    Each valid contour from each detection mask becomes one annotation entry.
    Contours with fewer than 3 points are silently skipped.

    Args:
        results: Output from SAM3LoRAInference.predict().
        image_id: COCO image ID for this image.
        prompt_to_cat_id: Mapping from prompt string to category ID.
        simplify_epsilon: RDP epsilon in pixels. 0 = no simplification.
        start_ann_id: First annotation ID to assign.

    Returns:
        List of COCO annotation dicts.
    """
    annotations = []
    ann_id = start_ann_id

    for idx in sorted(k for k in results if k != "_image"):
        result = results[idx]
        if result["num_detections"] == 0 or result["masks"] is None:
            continue

        prompt = result["prompt"]
        category_id = prompt_to_cat_id[prompt]
        masks = result["masks"]    # [N, H, W] bool
        scores = result["scores"]  # [N]

        for i in range(result["num_detections"]):
            mask = masks[i]
            score = float(scores[i]) if scores is not None else 0.0
            polygons = mask_to_polygons(mask, simplify_epsilon)

            for poly in polygons:
                bbox = polygons_to_bbox([poly])
                area = polygons_to_area([poly])
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [poly],
                    "bbox": bbox,
                    "area": area,
                    "score": round(score, 5),
                    "iscrowd": 0,
                })
                ann_id += 1

    return annotations


def run_folder_inference(
    input_dir: str,
    output_path: str,
    prompts: List[str],
    category_ids: List[int],
    categories_file: str,
    config_path: str,
    weights_path: Optional[str],
    threshold: float,
    resolution: int,
    nms_iou: float,
    simplify_epsilon: float,
    image_exts: List[str],
    device: str,
) -> None:
    """
    Run batch inference and write a COCO JSON file.

    This function is the testable core — main() is just argument parsing + a call here.
    """
    if len(prompts) != len(category_ids):
        raise ValueError(
            f"--prompt and --category_id must have the same number of values "
            f"(got {len(prompts)} prompts, {len(category_ids)} IDs)"
        )
    categories = load_categories(categories_file)
    prompt_to_cat_id = dict(zip(prompts, category_ids))

    # Collect images
    input_path = Path(input_dir)
    image_files = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lstrip(".").lower() in image_exts
    )
    if not image_files:
        print(f"⚠️  No images found in {input_dir} with extensions: {image_exts}")

    # Load model once
    model = SAM3LoRAInference(
        config_path=config_path,
        weights_path=weights_path,
        resolution=resolution,
        detection_threshold=threshold,
        nms_iou_threshold=nms_iou,
        device=device,
    )

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for image_id, image_file in enumerate(image_files, start=1):
        print(f"\n[{image_id}/{len(image_files)}] {image_file.name}")
        try:
            pil_img = PILImage.open(image_file).convert("RGB")
            w, h = pil_img.size
        except Exception as e:
            print(f"   ⚠️  Could not read {image_file.name}: {e} — skipping")
            continue

        coco_images.append({
            "id": image_id,
            "file_name": image_file.name,
            "width": w,
            "height": h,
        })

        try:
            results = model.predict(str(image_file), prompts)
        except Exception as e:
            print(f"   ⚠️  Inference failed for {image_file.name}: {e} — skipping annotations")
            continue

        new_anns = results_to_coco_annotations(
            results,
            image_id=image_id,
            prompt_to_cat_id=prompt_to_cat_id,
            simplify_epsilon=simplify_epsilon,
            start_ann_id=ann_id,
        )
        coco_annotations.extend(new_anns)
        ann_id += len(new_anns)
        print(f"   → {len(new_anns)} annotations")

    coco_output = {
        "info": {
            "description": "SAM3 LoRA inference output",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"\n✅ COCO JSON written to {output_path}")
    print(f"   Images: {len(coco_images)} | Annotations: {len(coco_annotations)}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Folder Inference → COCO JSON")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--weights", type=str, default=None, help="LoRA weights path (auto-detected if omitted)")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder of images to infer")
    parser.add_argument("--output", type=str, default="output_coco.json", help="Output COCO JSON path")
    parser.add_argument("--prompt", type=str, nargs="+", required=True, help="Text prompts")
    parser.add_argument("--category_id", type=int, nargs="+", required=True,
                        help="Category IDs, one per prompt in the same order")
    parser.add_argument("--categories", type=str, required=True,
                        help="COCO JSON file to read categories array from")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--resolution", type=int, default=1008, help="Input resolution")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="RDP simplification epsilon in pixels (0 = off)")
    parser.add_argument("--image-exts", type=str, nargs="+", default=["jpg", "jpeg", "png", "bmp"],
                        help="Image file extensions to scan (without dot)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    if len(args.prompt) != len(args.category_id):
        parser.error(
            f"--prompt and --category_id must have the same number of values "
            f"(got {len(args.prompt)} prompts, {len(args.category_id)} IDs)"
        )

    run_folder_inference(        input_dir=args.input_dir,
        output_path=args.output,
        prompts=args.prompt,
        category_ids=args.category_id,
        categories_file=args.categories,
        config_path=args.config,
        weights_path=args.weights,
        threshold=args.threshold,
        resolution=args.resolution,
        nms_iou=args.nms_iou,
        simplify_epsilon=args.simplify,
        image_exts=[e.lstrip(".").lower() for e in args.image_exts],
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests — verify they pass**

```bash
pytest tests/test_polygon_utils.py tests/test_infer_sam_polygon.py tests/test_infer_folder_coco.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: All tests PASS (including pre-existing `test_split_dataset.py`). Total new tests: 9 (polygon_utils) + 4 (infer_sam_polygon) + 12 (infer_folder_coco) = 25.

- [ ] **Step 6: Commit**

```bash
git add infer_folder_coco.py tests/test_infer_folder_coco.py
git commit -m "feat: add infer_folder_coco.py for batch inference to COCO JSON"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All flags documented in the spec are present in the implementation. `--device` added to both scripts. Prompt/category_id count validation in `main()`. `--categories` loads from COCO JSON. `--simplify` wired through all layers.
- [x] **No placeholders:** All code blocks are complete and runnable.
- [x] **Type consistency:** `mask_to_polygons` used identically in both `draw_polygons` and `results_to_coco_annotations`. `polygons_to_bbox`/`polygons_to_area` used only in `infer_folder_coco.py` as designed.
- [x] **CVAT compatibility:** `iscrowd: 0`, `segmentation: [[flat floats]]`, `bbox: [x,y,w,h]`, categories array with id/name/supercategory — all present.
- [x] **Error handling:** Missing categories file → `FileNotFoundError` at startup. Unreadable image → logged warning, skip. Mismatched prompt/category_id → argparse error before model load.

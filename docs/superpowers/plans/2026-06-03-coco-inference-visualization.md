# COCO Inference Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--visualize` flag to `infer_folder_coco.py` that renders each output image with filled mask overlays, polygon outlines, and category/score labels into a sibling `_viz/` folder.

**Architecture:** A new `coco_visualizer.py` module at the project root exposes a single `visualize_coco_output()` function. It reads the COCO JSON, builds lookup maps, draws all annotation fills onto a single RGBA overlay per image (alpha-composited), then draws outlines and labels on top. `infer_folder_coco.py`'s `main()` calls it after inference when `--visualize` is set — no changes to `run_folder_inference`'s signature.

**Tech Stack:** Python 3, PIL/Pillow (`Image`, `ImageDraw`), `json`, `pathlib`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `coco_visualizer.py` | **Create** | `visualize_coco_output()` — reads COCO JSON, renders annotated images |
| `tests/test_coco_visualizer.py` | **Create** | Unit tests for `coco_visualizer.py` |
| `infer_folder_coco.py` | **Modify** | Add `--visualize` CLI flag; call visualizer from `main()` |
| `tests/test_infer_folder_coco.py` | **Modify** | Add test: `--visualize` absent → no `_viz` folder created |

---

## Task 1: Create `coco_visualizer.py` with helper functions

**Files:**
- Create: `coco_visualizer.py`
- Test: `tests/test_coco_visualizer.py`

- [ ] **Step 1: Write the failing tests for `_flat_to_xy_pairs` and `_category_color`**

Create `tests/test_coco_visualizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/longtdang/KMS/SAM3_LoRA
python -m pytest tests/test_coco_visualizer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'coco_visualizer'`

- [ ] **Step 3: Create `coco_visualizer.py` with helpers only**

```python
"""COCO annotation visualizer.

Reads a COCO JSON file and renders each image with filled mask overlays,
polygon outlines, and category + score labels into an output directory.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


# 20 visually distinct RGB colors, cycling for category IDs beyond 20
_PALETTE: List[Tuple[int, int, int]] = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
]


def _category_color(category_id: int) -> Tuple[int, int, int]:
    """Return a consistent RGB color for a category_id (cycles every 20)."""
    return _PALETTE[(category_id - 1) % len(_PALETTE)]


def _flat_to_xy_pairs(flat: List[float]) -> List[Tuple[float, float]]:
    """Convert flat COCO polygon [x1,y1,x2,y2,...] to [(x1,y1),(x2,y2),...]."""
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]


def visualize_coco_output(
    coco_json_path: str,
    image_dir: str,
    viz_dir: str,
) -> None:
    pass  # implemented in Task 2
```

- [ ] **Step 4: Run helper tests to verify they pass**

```bash
python -m pytest tests/test_coco_visualizer.py::test_flat_to_xy_pairs_basic \
  tests/test_coco_visualizer.py::test_flat_to_xy_pairs_four_points \
  tests/test_coco_visualizer.py::test_category_color_returns_rgb_tuple \
  tests/test_coco_visualizer.py::test_category_color_consistent \
  tests/test_coco_visualizer.py::test_category_color_different_ids_differ \
  tests/test_coco_visualizer.py::test_category_color_cycles -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add coco_visualizer.py tests/test_coco_visualizer.py
git commit -m "feat: scaffold coco_visualizer module with helper functions and tests"
```

---

## Task 2: Implement `visualize_coco_output()`

**Files:**
- Modify: `coco_visualizer.py`
- Modify: `tests/test_coco_visualizer.py`

- [ ] **Step 1: Write the failing tests for `visualize_coco_output`**

Append to `tests/test_coco_visualizer.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
python -m pytest tests/test_coco_visualizer.py -v -k "not flat_to and not category_color" 2>&1 | tail -15
```

Expected: all new tests FAIL with `AssertionError` (function returns `None`)

- [ ] **Step 3: Implement `visualize_coco_output` in `coco_visualizer.py`**

Replace the `pass` stub:

```python
def visualize_coco_output(
    coco_json_path: str,
    image_dir: str,
    viz_dir: str,
) -> None:
    """
    Render each image listed in a COCO JSON with annotation overlays.

    For each image:
      - Fills each annotated polygon at 40% alpha in the category color.
      - Draws polygon outlines at full opacity.
      - Draws a "{category_name} {score:.2f}" label (score omitted if absent).
    Images with no annotations are copied unchanged.
    Missing source images are skipped with a warning.

    Args:
        coco_json_path: Path to COCO-format JSON.
        image_dir: Directory containing source images.
        viz_dir: Output directory (created if absent).
    """
    coco_json_path = Path(coco_json_path)
    image_dir = Path(image_dir)
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        coco = json.load(f)

    cat_name: Dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}

    ann_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_entry in coco.get("images", []):
        image_id = img_entry["id"]
        file_name = img_entry["file_name"]
        src_path = image_dir / file_name

        if not src_path.exists():
            print(f"   ⚠️  Image not found, skipping: {src_path}")
            continue

        img = Image.open(src_path).convert("RGBA")
        annotations = ann_by_image.get(image_id, [])

        if annotations:
            # Pass 1: draw all filled polygons onto a single overlay, then composite
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            for ann in annotations:
                r, g, b = _category_color(ann["category_id"])
                for seg in ann.get("segmentation", []):
                    if len(seg) < 6:
                        continue
                    xy = _flat_to_xy_pairs(seg)
                    draw_overlay.polygon(xy, fill=(r, g, b, 102))  # 40% alpha

            img = Image.alpha_composite(img, overlay)

            # Pass 2: draw outlines and labels on the composited image
            draw_img = ImageDraw.Draw(img)
            for ann in annotations:
                r, g, b = _category_color(ann["category_id"])
                for seg in ann.get("segmentation", []):
                    if len(seg) < 6:
                        continue
                    xy = _flat_to_xy_pairs(seg)
                    draw_img.line(xy + [xy[0]], fill=(r, g, b), width=2)

                cat = cat_name.get(ann["category_id"], str(ann["category_id"]))
                score = ann.get("score")
                label = f"{cat} {score:.2f}" if score is not None else cat
                bx, by = ann["bbox"][0], ann["bbox"][1]
                draw_img.text((bx, max(0, by - 2)), label, fill=(255, 255, 255))

        out_path = viz_dir / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            img.convert("RGB").save(out_path)
        else:
            img.save(out_path)
```

- [ ] **Step 4: Run all `coco_visualizer` tests**

```bash
python -m pytest tests/test_coco_visualizer.py -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add coco_visualizer.py tests/test_coco_visualizer.py
git commit -m "feat: implement visualize_coco_output with mask/outline/label rendering"
```

---

## Task 3: Add `--visualize` flag to `infer_folder_coco.py`

**Files:**
- Modify: `infer_folder_coco.py`
- Modify: `tests/test_infer_folder_coco.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_infer_folder_coco.py`:

```python
def test_no_viz_dir_without_flag(categories_coco_file, tmp_path):
    """Running without --visualize must not create a _viz directory."""
    img = PILImage.new("RGB", (50, 50), color=(100, 100, 100))
    img_path = tmp_path / "dummy.jpg"
    img.save(img_path)
    output_json = str(tmp_path / "out.json")

    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "_image": None,
        0: {"prompt": "crack", "num_detections": 0, "masks": None, "scores": None},
    }

    with patch("infer_folder_coco.SAM3LoRAInference", return_value=mock_model):
        run_folder_inference(
            input_dir=str(tmp_path),
            output_path=output_json,
            prompts=["crack"],
            category_ids=[1],
            categories_file=categories_coco_file,
            config_path="dummy_config.yaml",
            weights_path=None,
            threshold=0.5,
            resolution=1008,
            nms_iou=0.5,
            simplify_epsilon=0.0,
            image_exts=["jpg"],
            device="cpu",
        )

    viz_dir = tmp_path / "out_viz"
    assert not viz_dir.exists(), "_viz folder must not be created when --visualize is not passed"
```

- [ ] **Step 2: Run test to verify it passes already** (no viz dir is created today — confirms baseline)

```bash
python -m pytest tests/test_infer_folder_coco.py::test_no_viz_dir_without_flag -v
```

Expected: PASS (baseline already correct)

- [ ] **Step 3: Add `--visualize` flag and call in `main()`**

In `infer_folder_coco.py`, find the `main()` function. Add the `--visualize` argument to the parser:

```python
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Render annotated images to a sibling _viz/ folder after inference",
    )
```

Then, at the end of `main()`, after the `run_folder_inference(...)` call, add:

```python
    if args.visualize:
        from coco_visualizer import visualize_coco_output
        viz_dir = Path(args.output).parent / (Path(args.output).stem + "_viz")
        visualize_coco_output(args.output, args.input_dir, str(viz_dir))
        print(f"🖼️  Visualizations saved to {viz_dir}/")
```

Also add `from pathlib import Path` at the top of `main()` if not already imported at module level. (It already is — `from pathlib import Path` is at the top of the file.)

- [ ] **Step 4: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still pass, new test passes

- [ ] **Step 5: Commit**

```bash
git add infer_folder_coco.py tests/test_infer_folder_coco.py
git commit -m "feat: add --visualize flag to infer_folder_coco.py"
```

---

## Task 4: End-to-end smoke test (manual)

> This task requires no model — use the existing `tests/` fixtures to produce a real COCO JSON and then run the visualizer on it to confirm the output looks right.

- [ ] **Step 1: Generate a synthetic COCO JSON and images**

```bash
python - <<'EOF'
import json
from pathlib import Path
from PIL import Image

out_dir = Path("/tmp/viz_smoke")
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

# Create 2 test images
Image.new("RGB", (200, 150), (180, 100, 80)).save(img_dir / "frame_01.jpg")
Image.new("RGB", (200, 150), (80, 120, 180)).save(img_dir / "frame_02.jpg")

# COCO JSON with annotations on frame_01, none on frame_02
coco = {
    "categories": [
        {"id": 1, "name": "crack", "supercategory": "defect"},
        {"id": 2, "name": "rust",  "supercategory": "defect"},
    ],
    "images": [
        {"id": 1, "file_name": "frame_01.jpg", "width": 200, "height": 150},
        {"id": 2, "file_name": "frame_02.jpg", "width": 200, "height": 150},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1,
         "segmentation": [[20.0,20.0, 100.0,20.0, 100.0,80.0, 20.0,80.0]],
         "bbox": [20.0, 20.0, 80.0, 60.0], "area": 4800.0, "score": 0.91, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2,
         "segmentation": [[120.0,50.0, 180.0,50.0, 180.0,130.0, 120.0,130.0]],
         "bbox": [120.0, 50.0, 60.0, 80.0], "area": 4800.0, "score": 0.75, "iscrowd": 0},
    ],
}
(out_dir / "results.json").write_text(json.dumps(coco, indent=2))
print("Done — images and COCO JSON written to /tmp/viz_smoke/")
EOF
```

- [ ] **Step 2: Run the visualizer**

```bash
cd /home/longtdang/KMS/SAM3_LoRA
python -c "
from coco_visualizer import visualize_coco_output
visualize_coco_output('/tmp/viz_smoke/results.json', '/tmp/viz_smoke/images', '/tmp/viz_smoke/viz')
"
```

Expected output:
```
🖼️  Visualizations saved to /tmp/viz_smoke/viz/
```

- [ ] **Step 3: Verify outputs exist and open them**

```bash
ls /tmp/viz_smoke/viz/
# Expected: frame_01.jpg  frame_02.jpg
```

Open `frame_01.jpg` — should show the original image with two colored polygon overlays, outlines, and labels `"crack 0.91"` and `"rust 0.75"`.  
Open `frame_02.jpg` — should look identical to the source (no annotations).

- [ ] **Step 4: Clean up**

```bash
rm -rf /tmp/viz_smoke
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `coco_visualizer.py` new module with `visualize_coco_output()`
- ✅ Lookup maps: `image_id → file_name`, `category_id → name`, `image_id → [annotations]`
- ✅ Images with zero annotations still saved
- ✅ Filled polygon overlay at 40% alpha
- ✅ Polygon outline (2px)
- ✅ Category label + score; score optional
- ✅ Single overlay composited once (not per-annotation)
- ✅ Colors consistent per category_id
- ✅ Missing image → warning + skip, no abort
- ✅ `--visualize` flag in `infer_folder_coco.py` (store_true)
- ✅ `viz_dir` derived as `{output_stem}_viz/` next to JSON
- ✅ Called from `main()`, not `run_folder_inference()`
- ✅ Preserves original file extension

**Tests:**
- ✅ Helper functions tested (Task 1)
- ✅ Output files created (Task 2)
- ✅ Unannotated image still saved (Task 2)
- ✅ Annotated image differs from source (Task 2)
- ✅ Extension preservation — jpg and png (Task 2)
- ✅ Missing image skipped without crash (Task 2)
- ✅ Score field optional (Task 2)
- ✅ No viz dir without `--visualize` (Task 3)

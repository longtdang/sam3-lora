# Design: Polygon Inference Scripts

**Date:** 2026-06-02  
**Status:** Approved  

## Overview

Two new reusable Python inference scripts for the SAM3 + LoRA project:

1. **`infer_sam_polygon.py`** — Single-image inference that outputs a polygon-overlay image instead of a filled-mask image, with optional polygon simplification.
2. **`infer_folder_coco.py`** — Folder-level inference that runs all images through the model and writes a CVAT-importable COCO JSON file with polygon segmentations.

Both scripts reuse the existing `SAM3LoRAInference` class from `infer_sam.py` with no changes to that file.

---

## Architecture

```
infer_sam.py                  # existing — SAM3LoRAInference class (importable as-is)
infer_sam_polygon.py          # NEW: single image → polygon overlay PNG
infer_folder_coco.py          # NEW: image folder → COCO JSON
```

**Shared dependency:** Both new scripts import `SAM3LoRAInference` directly:

```python
from infer_sam import SAM3LoRAInference
```

No changes to `infer_sam.py` are required.

---

## Script 1 — `infer_sam_polygon.py`

### Purpose

Same function as `infer_sam.py` but renders polygon contours on the output image instead of filled semi-transparent masks. Useful for annotation review and visual QA.

### CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | required | Path to training config YAML |
| `--weights` | str | None | LoRA weights path (auto-detected if omitted) |
| `--image` | str | required | Path to input image |
| `--prompt` | str+ | `["object"]` | One or more text prompts |
| `--output` | str | `output_polygon.png` | Output PNG path |
| `--threshold` | float | `0.5` | Detection confidence threshold |
| `--resolution` | int | `1008` | Input resolution |
| `--nms-iou` | float | `0.5` | NMS IoU threshold |
| `--simplify` | float | `0.0` | RDP epsilon in pixels; `0` = no simplification |
| `--boundingbox` | bool | `False` | Also draw bounding boxes |
| `--device` | str | `cuda` | Device to run on (`cuda` or `cpu`) |

### Mask → Polygon Pipeline

1. Call `SAM3LoRAInference.predict(image, prompts)` → binary masks `numpy bool [H, W]`
2. `cv2.findContours(mask.astype(uint8), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` → list of contours
3. If `--simplify > 0`: `cv2.approxPolyDP(contour, epsilon=args.simplify, closed=True)` per contour
4. Draw contours on a copy of the original image using per-prompt colors
5. Annotate each contour with `"<prompt>: <score>"` text label
6. Save output PNG

### Output

A single PNG with polygon outlines drawn over the original image. No mask fill — clean contour lines only.

---

## Script 2 — `infer_folder_coco.py`

### Purpose

Run batch inference over a folder of images and produce a single COCO JSON file suitable for import into CVAT. Supports multiple text prompts, each mapped to a specific COCO category ID.

### CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | required | Path to training config YAML |
| `--weights` | str | None | LoRA weights path (auto-detected if omitted) |
| `--input-dir` | str | required | Folder containing images to infer |
| `--output` | str | `output_coco.json` | Output COCO JSON path |
| `--prompt` | str+ | required | One or more text prompts |
| `--category_id` | int+ | required | Category IDs, one per prompt (must match count) |
| `--categories` | str | required | Path to a COCO JSON file to read the `categories` array from |
| `--threshold` | float | `0.5` | Detection confidence threshold |
| `--resolution` | int | `1008` | Input resolution |
| `--nms-iou` | float | `0.5` | NMS IoU threshold |
| `--simplify` | float | `0.0` | RDP epsilon in pixels for polygon simplification |
| `--image-exts` | str+ | `jpg jpeg png bmp` | Image file extensions to scan |
| `--device` | str | `cuda` | Device to run on (`cuda` or `cpu`) |

### Prompt → Category ID Mapping

Prompts and category IDs are paired positionally. Example:

```bash
python infer_folder_coco.py \
  --prompt "surface crack" defect rust \
  --category_id 1 2 3
```

- `"surface crack"` → `category_id: 1`
- `"defect"` → `category_id: 2`
- `"rust"` → `category_id: 3`

Multi-word prompts are supported via shell quoting. The script validates that `len(--prompt) == len(--category_id)` at startup and exits with a clear error if they mismatch.

### Categories Source

The `--categories` flag points to an existing COCO JSON file (e.g. `_annotations.coco.json`). The script reads its `categories` array verbatim and writes it into the output JSON. This ensures category `id`, `name`, and `supercategory` fields match the project's annotation standard.

### Annotation Pipeline (per image)

1. Load image, record `width`, `height`, assign sequential `image_id`
2. Call `SAM3LoRAInference.predict(image_path, prompts)` → per-prompt detections
3. For each detection (prompt index → mask, score, box):
   a. Convert binary mask → contours via `cv2.findContours`
   b. If `--simplify > 0`: apply `cv2.approxPolyDP` per contour
   c. Skip any contour with fewer than 3 points (invalid polygon)
   d. Flatten each valid contour to `[x1, y1, x2, y2, ...]` float list
   e. Compute `bbox` as `[x, y, w, h]` from contour bounding rect
   f. Compute `area` from `cv2.contourArea`
   g. Look up `category_id` from the positional mapping
   h. **Each contour becomes its own annotation entry** (disconnected regions of the same mask get separate annotation IDs)
4. Accumulate annotation dicts; assign sequential global `annotation_id`

### COCO JSON Output Structure

```json
{
  "info": {
    "description": "SAM3 LoRA inference output",
    "date_created": "<ISO timestamp>"
  },
  "licenses": [],
  "categories": [ ...from --categories file... ],
  "images": [
    {"id": 1, "file_name": "image01.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 2,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, w, h],
      "area": 1234.5,
      "score": 0.87,
      "iscrowd": 0
    }
  ]
}
```

### CVAT Compatibility

- `iscrowd: 0` on all annotations (required by CVAT)
- `segmentation` as list-of-lists of flat floats `[[x, y, x, y, ...]]` (COCO polygon format)
- `bbox` in `[x, y, w, h]` format (not `xyxy`)
- Categories array contains `id`, `name`, `supercategory` fields
- Detections with no valid contour (empty mask or all contours < 3 points) are silently skipped
- `score` is a non-standard extra field included for traceability; CVAT ignores unknown fields

---

## Error Handling

- Mismatched `--prompt` / `--category_id` counts → clear error at startup, no model load
- Image file not readable → log warning, skip image, continue batch
- No detections for an image → image still written to `images` list, zero annotations
- Missing `--categories` file → error at startup

---

## Dependencies

- `opencv-python` (`cv2`) — contour extraction and polygon simplification
- `infer_sam.SAM3LoRAInference` — model loading and inference
- Standard library: `json`, `pathlib`, `datetime`, `argparse`

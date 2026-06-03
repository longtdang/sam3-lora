# COCO Inference Visualization — Design Spec

**Date:** 2026-06-03  
**Status:** Approved

## Problem

After running `infer_folder_coco.py`, users receive a COCO JSON file but have no quick way to visually verify the inference results before importing into CVAT. A bad import wastes time and may corrupt annotations.

## Goal

When `--visualize` is passed to `infer_folder_coco.py`, produce a folder of annotated images (masks + outlines + labels) alongside the COCO JSON so the user can visually review all predictions first.

---

## Architecture

### New module: `coco_visualizer.py` (project root)

Single public function:

```python
def visualize_coco_output(
    coco_json_path: str,
    image_dir: str,
    viz_dir: str,
) -> None
```

- Reads the COCO JSON produced by inference.
- Builds lookup maps: `image_id → file_name`, `category_id → name`, `image_id → [annotations]`.
- Iterates over every image in the COCO JSON and renders it to `viz_dir/`.
- Images with zero annotations are still saved (unmodified copy) so `viz_dir/` mirrors the full input set.

### Changes to `infer_folder_coco.py`

1. New CLI flag: `--visualize` (store_true, default False).
2. In `main()`, after `run_folder_inference()` returns, if `--visualize` is set:
   - Derive `viz_dir = Path(args.output).parent / (Path(args.output).stem + "_viz")`.
   - Call `visualize_coco_output(args.output, args.input_dir, viz_dir)`.
   - Print: `🖼️  Visualizations saved to {viz_dir}/`

No changes to `run_folder_inference`'s signature or logic. Visualization is a pure post-processing step driven from `main()`.

---

## Rendering Details

Segmentation format: `infer_folder_coco.py` always writes polygons as flat coordinate lists `[x1, y1, x2, y2, ...]` — this is the only format `coco_visualizer.py` needs to handle.

For each image:

1. Open source image with PIL, convert to RGBA.
2. Assign each `category_id` a distinct color from a fixed palette (cycling if needed).
3. For each annotation on this image:
   - Parse `segmentation` polygons (flat `[x1, y1, x2, y2, ...]` lists).
   - Draw filled polygon on a separate RGBA overlay at 40% alpha.
   - Draw polygon outline (fully opaque, 2px).
   - Render `"{category_name} {score:.2f}"` text at the top-left corner of the bounding box using `PIL.ImageDraw` with the default font. If the `score` field is absent (non-standard COCO files), show only `"{category_name}"`.
4. Composite overlay onto the original image.
5. Save to `viz_dir/{original_filename}`, **preserving the original file extension** (e.g., `.jpg` stays `.jpg`, `.png` stays `.png`).

Colors are consistent per category across all images (same `category_id` always gets the same color).

**Error handling:** If an image referenced in the COCO JSON is not found in `image_dir`, print a warning and skip that image — do not abort the entire run.

---

## File Changes Summary

| File | Change |
|------|--------|
| `coco_visualizer.py` | New file — rendering module |
| `infer_folder_coco.py` | Add `--visualize` flag; call visualizer after JSON write |

---

## Usage Example

```bash
python infer_folder_coco.py \
    --config configs/full_lora_config.yaml \
    --input-dir images/ \
    --output results/output_coco.json \
    --categories dataset/_annotations.coco.json \
    --prompt "surface crack" defect rust \
    --category_id 1 2 3 \
    --visualize
```

Output:
```
✅ COCO JSON written to results/output_coco.json
   Images: 42 | Annotations: 137
🖼️  Visualizations saved to results/output_coco_viz/
```

---

## Testing

- Unit test for `visualize_coco_output` using a synthetic COCO JSON + small test images.
- Verify output folder contains the expected number of files.
- Verify that images with no annotations are still written.
- Verify that running without `--visualize` does not create a viz folder.

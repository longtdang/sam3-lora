# Design: `split_dataset.py` — COCO-to-SAM3 Dataset Splitter

**Date:** 2026-06-01  
**Status:** Approved

---

## Overview

A standalone script that takes a versioned dataset folder (`dataset/vX/`) containing images and a COCO JSON annotation file (exported from CVAT), converts the COCO annotations into SAM3 per-image JSON format, and splits everything into `data/train/`, `data/val/`, and `data/test/` directories — following the exact dataset structure expected by the codebase's `SAM3Dataset` class.

---

## Input

```
dataset/
  v1/
    images/                  ← raw images (jpg, jpeg, png)
    _annotations.coco.json   ← COCO format from CVAT (any .json name works)
  v2/
    ...
```

The script auto-detects the COCO JSON by looking for `_annotations.coco.json` first, then falling back to any single `.json` file at the root of `--dataset-dir`. A `--coco-json` argument allows explicitly specifying the file path when there are multiple JSON files.  
Multiple versions can be run sequentially; outputs merge safely into `data/`.

---

## Output

```
data/
  train/
    images/        ← copied images
    annotations/   ← per-image SAM3 JSON
  valid/
    images/
    annotations/
  test/
    images/
    annotations/
```

> **Note:** The split output directory for validation is `valid/` (not `val/`), matching the existing `data/valid/` directory on disk and the majority of configs (`base_config.yaml`, `crack_detection_config.yaml`). The outlier `minimal_lora_config.yaml` uses `data/val` and would need updating if used with this script.

SAM3 per-image annotation format (as expected by `SAM3Dataset.__getitem__`):

```json
{
  "text_prompt": "crack, spall",
  "bboxes": [[x1, y1, x2, y2], ...],
  "masks": []
}
```

- `bboxes` are converted from COCO `[x, y, w, h]` → `[x1, y1, x2, y2]`
- `text_prompt` is the deduplicated union of category names for that image
- `masks` is populated with rasterized binary masks if segmentation polygons are present in the COCO file (via `pycocotools`), otherwise left empty

---

## CLI Interface

```
python split_dataset.py --dataset-dir dataset/v1 [options]

Arguments:
  --dataset-dir   Path to the versioned dataset folder (required)
  --coco-json     Path to the COCO JSON file (optional; auto-detected if omitted)
  --random        Shuffle images randomly before splitting (default: alphabetical)
  --seed          Random seed for reproducibility when --random is used (default: 42)
  --train         Fraction for training set (default: 0.7)
  --val           Fraction for validation set (default: 0.2)
  --test          Fraction for test set (default: 0.1)
```

Ratios must sum to 1.0 (validated at startup).

---

## Split Logic

- **Default (no `--random`):** Images are sorted alphabetically by `file_name` before splitting. This is deterministic and reproducible.
- **With `--random`:** Images are shuffled using `random.shuffle` seeded by `--seed`.
- Split boundaries are calculated as `floor(N * ratio)` with any remainder going to train.

---

## Error Handling

- Image file not found in `images/` → warn and skip
- Image has no annotations in COCO file → warn and skip
- Ratio sum ≠ 1.0 → exit with error message
- COCO JSON not found and cannot be auto-detected → exit with clear error message
- Multiple `.json` files found and `--coco-json` not specified → exit asking user to use `--coco-json`
- Output dirs created with `exist_ok=True` (safe merge with existing data)

---

## File

`split_dataset.py` — placed at the repository root, consistent with `prepare_data.py` and `prepare_data_split.py`.

---

## Dependencies

- Standard library: `os`, `json`, `shutil`, `argparse`, `random`, `pathlib`
- `pycocotools` (optional, only for mask rasterization from segmentation polygons)
- `Pillow` (for image size lookup when rasterizing masks)
- Both are already present in the project's `requirements.txt`

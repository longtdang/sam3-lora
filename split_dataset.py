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
    except Exception:
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
    candidate_masks = []
    category_names = []

    for ann in annotations:
        # COCO bbox: [x, y, width, height] → SAM3: [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])

        cat_name = category_map.get(ann["category_id"], f"class_{ann['category_id']}")
        category_names.append(cat_name)

        candidate_masks.append(
            _try_rasterize_segmentation(ann.get("segmentation", []), width, height)
        )

    # All-or-nothing: only use masks if every annotation produced one
    masks = candidate_masks if all(m is not None for m in candidate_masks) else []

    text_prompt = ", ".join(sorted(set(category_names)))
    return {"text_prompt": text_prompt, "bboxes": bboxes, "masks": masks}


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

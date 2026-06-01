"""
split_dataset.py — Split a CVAT COCO dataset into train/valid/test for SAM3 LoRA training.

Produces the directory layout expected by COCOSegmentDataset in train_sam3_lora_native.py:

    data/
        train/
            _annotations.coco.json   # filtered COCO JSON for this split
            image1.jpg               # images placed directly in split dir
            image2.jpg
        valid/
            _annotations.coco.json
            ...
        test/
            _annotations.coco.json
            ...

Usage:
    python split_dataset.py --dataset-dir dataset/v1 [--random] [--seed 42]
                            [--train 0.7] [--val 0.2] [--test 0.1]
                            [--coco-json path/to/annotations.json]
                            [--symlink-images]
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
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink images into output dirs instead of copying them (saves disk space)",
    )
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
# File output
# ---------------------------------------------------------------------------

def copy_split(
    split_name: str,
    image_entries: list,
    images_src_dir: Path,
    output_root: Path,
    coco_meta: dict,
    symlink: bool = False,
) -> dict:
    """
    Copy (or symlink) images into output/{split_name}/ and write a filtered
    _annotations.coco.json for this split.

    ``coco_meta`` must contain at minimum a ``"categories"`` key; ``"info"``
    and ``"licenses"`` are optional and forwarded as-is.

    Returns a summary dict: {"copied": int, "skipped": int}
    """
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    kept_images = []
    kept_annotations = []

    for entry in image_entries:
        file_name = entry["file_name"]
        src_image = images_src_dir / file_name

        if not src_image.exists():
            print(f"  Warning: image not found, skipping — {src_image}")
            skipped += 1
            continue

        dst_image = split_dir / file_name
        if symlink:
            if dst_image.exists() or dst_image.is_symlink():
                dst_image.unlink()
            dst_image.symlink_to(src_image.resolve())
        else:
            shutil.copy2(src_image, dst_image)

        kept_images.append({
            "id": entry["id"],
            "file_name": file_name,
            "width": entry["width"],
            "height": entry["height"],
        })
        kept_annotations.extend(entry["annotations"])
        copied += 1

    coco_out = {
        "info": coco_meta.get("info", {}),
        "licenses": coco_meta.get("licenses", []),
        "categories": coco_meta.get("categories", []),
        "images": kept_images,
        "annotations": kept_annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco_out, indent=2))

    return {"copied": copied, "skipped": skipped}


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

    coco_meta = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
    }

    print(f"\nSplitting: {len(train_imgs)} train / {len(valid_imgs)} valid / {len(test_imgs)} test")
    print(f"Output root: {root.resolve()}\n")

    for split_name, entries in [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]:
        result = copy_split(split_name, entries, images_src, root, coco_meta, symlink=args.symlink_images)
        action = "symlinked" if args.symlink_images else "copied"
        print(
            f"  [{split_name:6s}]  {action}: {result['copied']}  skipped: {result['skipped']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

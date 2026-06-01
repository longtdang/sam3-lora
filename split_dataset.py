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

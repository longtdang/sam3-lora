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

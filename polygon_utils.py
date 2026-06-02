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

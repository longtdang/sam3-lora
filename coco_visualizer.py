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

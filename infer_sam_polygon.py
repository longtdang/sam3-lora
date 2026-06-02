#!/usr/bin/env python3
"""
SAM3 + LoRA Inference — Polygon Output

Same as infer_sam.py but renders polygon contours instead of filled masks.
Optionally simplifies polygons using Ramer-Douglas-Peucker (--simplify epsilon).

Usage:
    python infer_sam_polygon.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt crack defect \
        --output output_polygon.png \
        --simplify 2.0
"""

import argparse

import cv2
import numpy as np
from PIL import Image as PILImage

from polygon_utils import mask_to_polygons


# Per-prompt BGR colors for cv2 drawing
_COLORS_BGR = [
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
]


def draw_polygons(
    results: dict,
    simplify_epsilon: float = 0.0,
    show_boxes: bool = False,
) -> PILImage.Image:
    """
    Draw polygon contours (and optionally bounding boxes) on the original image.

    Args:
        results: Output dict from SAM3LoRAInference.predict().
        simplify_epsilon: RDP epsilon in pixels. 0 = no simplification.
        show_boxes: If True, also draw bounding boxes.

    Returns:
        PIL Image with contours drawn.
    """
    pil_image = results["_image"]
    canvas = np.array(pil_image.convert("RGB"))
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    prompt_indices = sorted(k for k in results if k != "_image")

    for idx in prompt_indices:
        result = results[idx]
        color = _COLORS_BGR[idx % len(_COLORS_BGR)]

        if result["num_detections"] == 0 or result["masks"] is None:
            continue

        masks = result["masks"]       # [N, H, W] bool
        scores = result["scores"]     # [N]
        boxes = result["boxes"]       # [N, 4] xyxy
        prompt = result["prompt"]

        for i in range(result["num_detections"]):
            mask = masks[i]
            score = float(scores[i]) if scores is not None else 0.0
            polygons = mask_to_polygons(mask, simplify_epsilon)

            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas_bgr, [pts], isClosed=True, color=color, thickness=2)

            if polygons:
                # Label at the top-left of the first polygon's bbox
                first_pts = np.array(polygons[0], dtype=np.int32).reshape(-1, 2)
                label_x = int(first_pts[:, 0].min())
                label_y = max(0, int(first_pts[:, 1].min()) - 5)
                label = f"{prompt}: {score:.2f}"
                cv2.putText(
                    canvas_bgr, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            if show_boxes and boxes is not None:
                x1, y1, x2, y2 = (int(v) for v in boxes[i])
                cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), color, 2)

    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(canvas_rgb)


def main():
    import torch
    from infer_sam import SAM3LoRAInference

    parser = argparse.ArgumentParser(description="SAM3 + LoRA Polygon Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--weights", type=str, default=None, help="LoRA weights path (auto-detected if omitted)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, nargs="+", default=["object"], help="Text prompt(s)")
    parser.add_argument("--output", type=str, default="output_polygon.png", help="Output PNG path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--resolution", type=int, default=1008, help="Input resolution")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="RDP simplification epsilon in pixels (0 = off)")
    parser.add_argument("--boundingbox", action="store_true", help="Also draw bounding boxes")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        device=args.device,
    )

    results = inferencer.predict(args.image, args.prompt)
    out_image = draw_polygons(results, simplify_epsilon=args.simplify, show_boxes=args.boundingbox)
    out_image.save(args.output)

    print(f"\n✅ Saved polygon output to {args.output}")
    for idx in sorted(k for k in results if k != "_image"):
        r = results[idx]
        print(f"   Prompt '{r['prompt']}': {r['num_detections']} detections")


if __name__ == "__main__":
    main()

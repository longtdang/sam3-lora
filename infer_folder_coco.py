#!/usr/bin/env python3
"""
SAM3 + LoRA Folder Inference → COCO JSON

Runs inference on all images in a folder and writes a CVAT-importable COCO JSON file.
Supports multiple text prompts each mapped to a COCO category ID.

Usage:
    python infer_folder_coco.py \
        --config configs/full_lora_config.yaml \
        --input-dir images/ \
        --output results.json \
        --categories dataset/_annotations.coco.json \
        --prompt "surface crack" defect rust \
        --category_id 1 2 3 \
        --simplify 2.0
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image as PILImage

from infer_sam import SAM3LoRAInference
from polygon_utils import mask_to_polygons, polygons_to_bbox, polygons_to_area


def load_categories(categories_file: str) -> list:
    """
    Read the 'categories' array from an existing COCO JSON file.

    Args:
        categories_file: Path to a COCO-format JSON file.

    Returns:
        List of category dicts with at least 'id', 'name', 'supercategory'.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(categories_file)
    if not path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_file}")
    with open(path) as f:
        data = json.load(f)
    return data["categories"]


def results_to_coco_annotations(
    results: dict,
    image_id: int,
    prompt_to_cat_id: Dict[str, int],
    simplify_epsilon: float,
    start_ann_id: int,
) -> List[dict]:
    """
    Convert SAM3LoRAInference.predict() results for one image into COCO annotation dicts.

    Each valid contour from each detection mask becomes one annotation entry.
    Contours with fewer than 3 points are silently skipped.

    Args:
        results: Output from SAM3LoRAInference.predict().
        image_id: COCO image ID for this image.
        prompt_to_cat_id: Mapping from prompt string to category ID.
        simplify_epsilon: RDP epsilon in pixels. 0 = no simplification.
        start_ann_id: First annotation ID to assign.

    Returns:
        List of COCO annotation dicts.
    """
    annotations = []
    ann_id = start_ann_id

    for idx in sorted(k for k in results if k != "_image"):
        result = results[idx]
        if result["num_detections"] == 0 or result["masks"] is None:
            continue

        prompt = result["prompt"]
        category_id = prompt_to_cat_id[prompt]
        masks = result["masks"]    # [N, H, W] bool
        scores = result["scores"]  # [N]

        for i in range(result["num_detections"]):
            mask = masks[i]
            score = float(scores[i]) if scores is not None else 0.0
            polygons = mask_to_polygons(mask, simplify_epsilon)

            for poly in polygons:
                bbox = polygons_to_bbox([poly])
                area = polygons_to_area([poly])
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [poly],
                    "bbox": bbox,
                    "area": area,
                    "score": round(score, 5),
                    "iscrowd": 0,
                })
                ann_id += 1

    return annotations


def run_folder_inference(
    input_dir: str,
    output_path: str,
    prompts: List[str],
    category_ids: List[int],
    categories_file: str,
    config_path: str,
    weights_path: Optional[str],
    threshold: float,
    resolution: int,
    nms_iou: float,
    simplify_epsilon: float,
    image_exts: List[str],
    device: str,
) -> None:
    """
    Run batch inference and write a COCO JSON file.

    This function is the testable core — main() is just argument parsing + a call here.
    """
    if len(prompts) != len(category_ids):
        raise ValueError(
            f"--prompt and --category_id must have the same number of values "
            f"(got {len(prompts)} prompts, {len(category_ids)} IDs)"
        )
    categories = load_categories(categories_file)
    prompt_to_cat_id = dict(zip(prompts, category_ids))

    # Collect images
    input_path = Path(input_dir)
    image_files = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lstrip(".").lower() in image_exts
    )
    if not image_files:
        print(f"⚠️  No images found in {input_dir} with extensions: {image_exts}")

    # Load model once
    model = SAM3LoRAInference(
        config_path=config_path,
        weights_path=weights_path,
        resolution=resolution,
        detection_threshold=threshold,
        nms_iou_threshold=nms_iou,
        device=device,
    )

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for image_id, image_file in enumerate(image_files, start=1):
        print(f"\n[{image_id}/{len(image_files)}] {image_file.name}")
        try:
            pil_img = PILImage.open(image_file).convert("RGB")
            w, h = pil_img.size
        except Exception as e:
            print(f"   ⚠️  Could not read {image_file.name}: {e} — skipping")
            continue

        coco_images.append({
            "id": image_id,
            "file_name": image_file.name,
            "width": w,
            "height": h,
        })

        try:
            results = model.predict(str(image_file), prompts)
        except Exception as e:
            print(f"   ⚠️  Inference failed for {image_file.name}: {e} — skipping annotations")
            continue

        new_anns = results_to_coco_annotations(
            results,
            image_id=image_id,
            prompt_to_cat_id=prompt_to_cat_id,
            simplify_epsilon=simplify_epsilon,
            start_ann_id=ann_id,
        )
        coco_annotations.extend(new_anns)
        ann_id += len(new_anns)
        print(f"   → {len(new_anns)} annotations")

    coco_output = {
        "info": {
            "description": "SAM3 LoRA inference output",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco_output, f, indent=2)

    print(f"\n✅ COCO JSON written to {output_path}")
    print(f"   Images: {len(coco_images)} | Annotations: {len(coco_annotations)}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Folder Inference → COCO JSON")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--weights", type=str, default=None, help="LoRA weights path (auto-detected if omitted)")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder of images to infer")
    parser.add_argument("--output", type=str, default="output_coco.json", help="Output COCO JSON path")
    parser.add_argument("--prompt", type=str, nargs="+", required=True, help="Text prompts")
    parser.add_argument("--category_id", type=int, nargs="+", required=True,
                        help="Category IDs, one per prompt in the same order")
    parser.add_argument("--categories", type=str, required=True,
                        help="COCO JSON file to read categories array from")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--resolution", type=int, default=1008, help="Input resolution")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="RDP simplification epsilon in pixels (0 = off)")
    parser.add_argument("--image-exts", type=str, nargs="+", default=["jpg", "jpeg", "png", "bmp"],
                        help="Image file extensions to scan (without dot)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    if len(args.prompt) != len(args.category_id):
        parser.error(
            f"--prompt and --category_id must have the same number of values "
            f"(got {len(args.prompt)} prompts, {len(args.category_id)} IDs)"
        )

    run_folder_inference(
        input_dir=args.input_dir,
        output_path=args.output,
        prompts=args.prompt,
        category_ids=args.category_id,
        categories_file=args.categories,
        config_path=args.config,
        weights_path=args.weights,
        threshold=args.threshold,
        resolution=args.resolution,
        nms_iou=args.nms_iou,
        simplify_epsilon=args.simplify,
        image_exts=[e.lstrip(".").lower() for e in args.image_exts],
        device=args.device,
    )


if __name__ == "__main__":
    main()

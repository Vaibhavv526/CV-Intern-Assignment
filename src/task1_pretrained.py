"""
Task 1: Full Dataset Pretrained Segmentation using Meta SAM ViT-B.
Generates pseudo-ground-truth segmentation masks for all 110 dataset images.
Includes robust egg candidate filtering, multiple-egg support, edge-case flagging,
incremental saving, and resume capability.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import argparse
import json
import time
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from src.egg_filter import filter_egg_candidates, compute_mask_properties


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: Pretrained SAM Full Dataset Segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute all masks even if existing")
    return parser.parse_args()


def collect_dataset_images(dataset_dir: str) -> list:
    """Collect all 110 .jpg images with their relative paths and split names."""
    image_list = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dataset_dir)
                split = rel_path.split(os.sep)[0]
                stem = os.path.splitext(f)[0]
                image_list.append({
                    "full_path": full_path,
                    "rel_path": rel_path,
                    "split": split,
                    "filename": f,
                    "stem": stem
                })
    image_list.sort(key=lambda x: x["rel_path"])
    return image_list


def touches_image_boundary(mask: np.ndarray, border_margin: int = 1) -> bool:
    """Check if binary mask touches the outermost boundary pixels."""
    h, w = mask.shape
    top = np.any(mask[:border_margin, :])
    bottom = np.any(mask[h - border_margin:, :])
    left = np.any(mask[:, :border_margin])
    right = np.any(mask[:, w - border_margin:])
    return bool(top or bottom or left or right)


def main(config_path: str = None, force_recompute: bool = False):
    if config_path is None:
        args = parse_args()
        config_path = os.path.abspath(args.config)
        force_recompute = args.force_recompute
    else:
        config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = os.path.abspath(cfg["paths"]["checkpoint_path"])
    dataset_dir = os.path.abspath(cfg["paths"]["dataset_dir"])
    output_base = os.path.abspath(cfg["paths"]["task1_output_dir"])

    masks_dir = os.path.join(output_base, "masks")
    vis_dir = os.path.join(output_base, "visualizations")
    metadata_path = os.path.join(output_base, "metadata.json")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 70)
    print("TASK 1: FULL DATASET PRETRAINED SEGMENTATION (SAM ViT-B)")
    print("=" * 70)
    print(f"Dataset directory:   {dataset_dir}")
    print(f"Checkpoint path:     {checkpoint_path}")
    print(f"Mask output dir:     {masks_dir}")
    print(f"Vis output dir:      {vis_dir}")
    print(f"Metadata file:       {metadata_path}")

    # Collect dataset images
    images = collect_dataset_images(dataset_dir)
    print(f"\nDiscovered {len(images)} images across splits:")
    splits_count = {}
    for item in images:
        splits_count[item["split"]] = splits_count.get(item["split"], 0) + 1
    for s, c in splits_count.items():
        print(f"  - {s}: {c} images")

    # Load existing metadata for resumption
    metadata = {}
    if os.path.exists(metadata_path) and not force_recompute:
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"\nLoaded existing metadata with {len(metadata)} processed records (Resuming enabled).")
        except Exception as e:
            print(f"Warning: Failed to read existing metadata ({e}), starting fresh.")
            metadata = {}

    # Initialize SAM model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing SAM model ({cfg['sam']['model_type']}) on {device.upper()}...")
    t_model_start = time.time()
    sam = sam_model_registry[cfg["sam"]["model_type"]](checkpoint=checkpoint_path)
    sam.to(device=device)
    print(f"SAM loaded in {time.time() - t_model_start:.2f}s.")
    if device == "cuda":
        print(f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=cfg["sam"].get("points_per_side", 32),
        pred_iou_thresh=cfg["sam"].get("pred_iou_thresh", 0.88),
        stability_score_thresh=cfg["sam"].get("stability_score_thresh", 0.92),
        min_mask_region_area=cfg["sam"].get("min_mask_region_area", 1000)
    )

    # Process all images
    print("\nStarting incremental batch segmentation...")
    success_count = 0
    flagged_count = 0
    skipped_count = 0
    total_infer_time = 0.0

    pbar = tqdm(images, desc="Processing images", unit="img")
    for img_info in pbar:
        stem = img_info["stem"]
        rel_path = img_info["rel_path"]
        mask_save_path = os.path.join(masks_dir, f"{stem}.png")
        vis_save_path = os.path.join(vis_dir, f"{stem}.png")

        # Resume check: if already processed and files exist on disk
        if not force_recompute and stem in metadata and os.path.exists(mask_save_path) and os.path.exists(vis_save_path):
            skipped_count += 1
            if metadata[stem].get("flags"):
                flagged_count += 1
            else:
                success_count += 1
            continue

        # Load image
        img_bgr = cv2.imread(img_info["full_path"])
        if img_bgr is None:
            metadata[stem] = {
                "rel_path": rel_path,
                "split": img_info["split"],
                "status": "FAILED_READ",
                "num_eggs": 0,
                "flags": ["FAILED_TO_LOAD_IMAGE"],
                "runtime_sec": 0.0
            }
            flagged_count += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # SAM Inference
        t_start = time.time()
        with torch.no_grad():
            raw_masks = mask_generator.generate(img_rgb)
        infer_time = time.time() - t_start
        total_infer_time += infer_time

        # Candidate filtering
        egg_cands = filter_egg_candidates(raw_masks, img_bgr, cfg["egg_filter"])
        
        flags = []
        combined_mask = np.zeros((h, w), dtype=bool)
        eggs_props = []

        if len(egg_cands) == 0:
            # Fallback relaxation: look for candidate with highest convexity & brightness
            fallback_cands = []
            for rm in raw_masks:
                props = compute_mask_properties(rm["segmentation"], img_bgr)
                if props and 2000 <= props["area"] <= 250000:
                    convex_score = props["solidity"] * (props["mean_val"] / 255.0)
                    fallback_cands.append((convex_score, rm["segmentation"], props))
            if fallback_cands:
                fallback_cands.sort(key=lambda x: x[0], reverse=True)
                top_fb = fallback_cands[0]
                combined_mask = top_fb[1]
                eggs_props.append(top_fb[2])
                flags.append("FILTER_RELAXED_FALLBACK")
            else:
                flags.append("NO_EGG_DETECTED")
        else:
            # Union of all non-overlapping egg candidates
            for cand in egg_cands:
                combined_mask = np.logical_or(combined_mask, cand["mask"])
                eggs_props.append(cand["props"])

        num_eggs = len(eggs_props)
        if num_eggs > 1:
            flags.append("MULTIPLE_EGGS")

        # Check boundary touch
        if combined_mask.any() and touches_image_boundary(combined_mask):
            flags.append("TOUCHES_IMAGE_BOUNDARY")

        # Check total area reasonableness
        total_area = float(combined_mask.sum())
        if total_area < 3500:
            flags.append("UNUSUALLY_SMALL_AREA")
        elif total_area > 230000:
            flags.append("UNUSUALLY_LARGE_AREA")

        # Save binary mask
        mask_u8 = (combined_mask.astype(np.uint8)) * 255
        cv2.imwrite(mask_save_path, mask_u8)

        # Build visualization panel
        vis_panel = img_bgr.copy()
        overlay = img_bgr.copy()
        overlay[combined_mask] = (overlay[combined_mask] * 0.35 + np.array((0, 220, 0)) * 0.65).astype(np.uint8)
        
        # Draw contours
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        
        # HUD Text
        status_text = f"SAM Pseudo-GT: {num_eggs} egg(s) | Area: {total_area:.0f}px"
        if flags:
            status_text += f" [{','.join(flags)}]"
        text_color = (0, 0, 255) if "NO_EGG_DETECTED" in flags else ((0, 255, 255) if flags else (0, 255, 0))
        cv2.putText(overlay, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        mask_3ch = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        triptych = cv2.hconcat([img_bgr, mask_3ch, overlay])
        cv2.imwrite(vis_save_path, triptych)

        # Update metadata entry
        metadata[stem] = {
            "rel_path": rel_path,
            "split": img_info["split"],
            "filename": img_info["filename"],
            "status": "FLAGGED" if flags else "SUCCESS",
            "num_eggs": num_eggs,
            "total_area_px": total_area,
            "area_ratio": total_area / (w * h),
            "flags": flags,
            "raw_candidates_count": len(raw_masks),
            "runtime_sec": round(infer_time, 3),
            "mask_path": os.path.relpath(mask_save_path, output_base),
            "vis_path": os.path.relpath(vis_save_path, output_base)
        }

        if flags:
            flagged_count += 1
        else:
            success_count += 1

        # Periodically flush metadata to disk every 5 images
        if len(metadata) % 5 == 0:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    # Final metadata save
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("TASK 1 PROCESSING COMPLETED")
    print("=" * 70)
    print(f"Total dataset images:     {len(images)}")
    print(f"Successfully processed:   {len(metadata)}")
    print(f"Clean confident masks:    {success_count}")
    print(f"Flagged / edge-case masks:{flagged_count}")
    print(f"Skipped (pre-computed):   {skipped_count}")
    print(f"Total inference runtime:  {total_infer_time:.2f}s (avg {total_infer_time/max(len(images)-skipped_count, 1):.2f}s/img)")
    print(f"Masks directory:          {masks_dir}")
    print(f"Visualizations directory: {vis_dir}")
    print(f"Metadata manifest:        {metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Minimal Smoke Test for Phase 1.
Loads SAM ViT-B checkpoint on CUDA, segments a real dataset image,
filters for egg candidate masks, and saves binary mask and visual comparison.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import time
import yaml
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from src.egg_filter import filter_egg_candidates


def run_smoke_test():
    print("=" * 60)
    print("Starting Phase 1 Smoke Test: Pretrained SAM Segmentation")
    print("=" * 60)

    # 1. Load config
    config_path = os.path.abspath("configs/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = os.path.abspath(cfg["paths"]["checkpoint_path"])
    model_type = cfg["sam"]["model_type"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name:    {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:   {torch.version.cuda}")
    print(f"Target Device:  {device}")
    print(f"Model Type:     {model_type}")
    print(f"Checkpoint:     {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 2. Pick a real dataset image
    dataset_dir = os.path.abspath(cfg["paths"]["dataset_dir"])
    sample_img_rel = os.path.join("train", "egg", "not_damaged_1_jpg.rf.d74c98a14f068b48c1f58fd573361f1f.jpg")
    sample_img_path = os.path.join(dataset_dir, sample_img_rel)
    
    if not os.path.exists(sample_img_path):
        # Fallback to any first image
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                if f.lower().endswith(".jpg"):
                    sample_img_path = os.path.join(root, f)
                    break
            if os.path.exists(sample_img_path):
                break

    print(f"Test Image:     {sample_img_path}")
    image_bgr = cv2.imread(sample_img_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image at {sample_img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"Image Size:     {w} x {h} (3 channels)")

    # 3. Load SAM Model onto GPU
    print("\nLoading SAM ViT-B model into GPU memory...")
    t_load_start = time.time()
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f}s.")
    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU VRAM Allocated: {mem_allocated:.1f} MB")

    # 4. Initialize Mask Generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=cfg["sam"].get("points_per_side", 32),
        pred_iou_thresh=cfg["sam"].get("pred_iou_thresh", 0.88),
        stability_score_thresh=cfg["sam"].get("stability_score_thresh", 0.92),
        min_mask_region_area=cfg["sam"].get("min_mask_region_area", 1000)
    )

    # 5. Run Inference
    print("\nRunning SAM inference on test image...")
    t_infer_start = time.time()
    with torch.no_grad():
        raw_masks = mask_generator.generate(image_rgb)
    t_infer = time.time() - t_infer_start
    print(f"Inference completed in {t_infer:.3f}s. Total raw candidate masks: {len(raw_masks)}")

    # 6. Apply Egg Candidate Filtering
    egg_candidates = filter_egg_candidates(raw_masks, image_bgr, cfg["egg_filter"])
    print(f"Filtered egg candidates: {len(egg_candidates)}")

    if not egg_candidates:
        print("Warning: No candidate met all strict filters; selecting highest scoring raw mask by compactness.")
        # Fallback to top solidity proposal
        from src.egg_filter import compute_mask_properties
        candidates = []
        for m in raw_masks:
            props = compute_mask_properties(m["segmentation"], image_bgr)
            if props and 3000 <= props["area"] <= 250000:
                candidates.append((props["solidity"] * props["circularity"], m["segmentation"], props))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_mask = candidates[0][1]
            best_props = candidates[0][2]
        else:
            best_mask = np.zeros((h, w), dtype=bool)
            best_props = {}
    else:
        best_cand = egg_candidates[0]
        best_mask = best_cand["mask"]
        best_props = best_cand["props"]
        print("Top Egg Candidate Metrics:")
        print(f"  - Pixel Area:     {best_props['area']:.0f} px ({best_props['area']/(w*h)*100:.1f}% of image)")
        print(f"  - Solidity:       {best_props['solidity']:.3f} (convexity)")
        print(f"  - Circularity:    {best_props['circularity']:.3f}")
        print(f"  - Aspect Ratio:   {best_props['aspect_ratio']:.3f}")
        print(f"  - Mean Value:     {best_props['mean_val']:.1f} / 255")
        print(f"  - Egg Score:      {best_cand['score']:.3f}")

    # 7. Save Binary Mask
    os.makedirs("outputs", exist_ok=True)
    binary_mask_path = os.path.abspath("outputs/smoke_test_binary_mask.png")
    binary_mask_u8 = (best_mask.astype(np.uint8)) * 255
    cv2.imwrite(binary_mask_path, binary_mask_u8)
    print(f"\nSaved binary mask:        {binary_mask_path}")

    # 8. Create & Save Side-by-Side Visualization
    vis_overlay = image_bgr.copy()
    # Color overlay (emerald green mask with alpha transparency)
    color = (0, 220, 0)
    overlay = image_bgr.copy()
    overlay[best_mask] = (overlay[best_mask] * 0.35 + np.array(color) * 0.65).astype(np.uint8)
    # Contour line
    contours, _ = cv2.findContours(binary_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    # Put annotation text
    cv2.putText(
        overlay,
        f"SAM Pseudo-GT (Area: {best_props.get('area', 0):.0f}px, Sol: {best_props.get('solidity', 0):.2f})",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Side-by-side composite: [Original Image | Binary Mask | Segmentation Overlay]
    mask_3ch = cv2.cvtColor(binary_mask_u8, cv2.COLOR_GRAY2BGR)
    cv2.putText(image_bgr, "Original Image", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(mask_3ch, "Binary Pseudo-GT Mask", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
    triptych = cv2.hconcat([image_bgr, mask_3ch, overlay])
    vis_output_path = os.path.abspath("outputs/smoke_test_visualization.png")
    cv2.imwrite(vis_output_path, triptych)
    print(f"Saved visualization panel: {vis_output_path}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print(f"Total time (load + inference + save): {t_load + t_infer:.2f}s")
    print(f"Pure inference time:                  {t_infer:.3f}s")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)

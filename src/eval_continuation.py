"""
Evaluation and Comparison Script:
Base SAM vs 5-Epoch Pilot SAM vs 15-Epoch Continued SAM.

Evaluates all three models on the exact 22 held-out evaluation images.
Computes IoU, Dice, Precision, Recall, and latency.
Tracks per-image deltas and flags regressions between Epoch 5 and Epoch 15.
Generates a 6-column qualitative comparison figure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def compute_binary_metrics(
    pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6
) -> Dict[str, float]:
    pred_bin = (pred_mask > 0).astype(bool)
    gt_bin = (gt_mask > 0).astype(bool)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def predict_sam_mask(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    box_tensor: torch.Tensor,
    input_size: Tuple[int, int],
    orig_size: Tuple[int, int],
    device: str,
) -> Tuple[np.ndarray, float]:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        image_embeddings = model.image_encoder(image_tensor)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None,
        )
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        pred_logits = model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=orig_size,
        )

    if device == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    mask_prob = torch.sigmoid(pred_logits[0, 0]).cpu().numpy()
    binary_mask = (mask_prob > 0.5).astype(np.uint8)

    return binary_mask, latency_ms


def create_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h, w = pred.shape
    err_map = np.zeros((h, w, 3), dtype=np.uint8)

    tp = np.logical_and(pred > 0, gt > 0)
    fp = np.logical_and(pred > 0, gt == 0)
    fn = np.logical_and(pred == 0, gt > 0)

    err_map[tp] = [40, 200, 60]    # Green
    err_map[fp] = [220, 50, 50]    # Red
    err_map[fn] = [50, 100, 240]   # Blue
    return err_map


def evaluate_continuation(
    base_checkpoint: str | Path = "models/checkpoints/sam_vit_b_01ec64.pth",
    pilot_checkpoint: str | Path = "models/checkpoints/finetuned/checkpoint_final.pt",
    cont_checkpoint: str | Path = "models/checkpoints/finetuned/continuation_10_epochs/checkpoint_final.pt",
    splits_file: str | Path = "data/splits.json",
    dataset_dir: str | Path = "egg.v1i.folder",
    mask_dir: str | Path = "data/task1_sam_masks/masks",
    output_dir: str | Path = "outputs",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    eval_stems = splits["eval"]
    records = splits.get("records", {})
    assert len(eval_stems) == 22, f"Expected 22 eval images, found {len(eval_stems)}"
    print(f"Loaded {len(eval_stems)} held-out evaluation images from {splits_file}")

    # Build Base SAM
    print(f"Loading Base SAM ViT-B from: {base_checkpoint}")
    base_sam = sam_model_registry["vit_b"](checkpoint=str(base_checkpoint)).to(device)
    base_sam.eval()

    # Build Pilot SAM (Epoch 5)
    print(f"Loading Pilot SAM (Epoch 5) from: {pilot_checkpoint}")
    pilot_sam = sam_model_registry["vit_b"](checkpoint=str(base_checkpoint)).to(device)
    pilot_ckpt = torch.load(str(pilot_checkpoint), map_location=device)
    pilot_sam.mask_decoder.load_state_dict(pilot_ckpt["mask_decoder_state_dict"])
    pilot_sam.eval()

    # Build Continued SAM (Epoch 15)
    print(f"Loading Continued SAM (Epoch 15) from: {cont_checkpoint}")
    cont_sam = sam_model_registry["vit_b"](checkpoint=str(base_checkpoint)).to(device)
    cont_ckpt = torch.load(str(cont_checkpoint), map_location=device)
    cont_sam.mask_decoder.load_state_dict(cont_ckpt["mask_decoder_state_dict"])
    cont_sam.eval()

    transform = ResizeLongestSide(base_sam.image_encoder.img_size)
    pixel_mean = base_sam.pixel_mean
    pixel_std = base_sam.pixel_std

    per_image_results: List[Dict[str, Any]] = []
    qual_samples: List[Dict[str, Any]] = []

    target_prefixes = [
        "not_damaged_95_",   # ash_charcoal_bowl
        "not_damaged_164_",  # patterned_cloth_fabric
        "not_damaged_121_",  # water_metallic_bowl
        "not_damaged_147_",  # kitchen_countertop_multiple
    ]

    print("\nEvaluating Base vs Pilot (Ep 5) vs Continued (Ep 15) across 22 evaluation samples...")

    for stem in eval_stems:
        record = records.get(stem, {})
        rel_path = record.get("rel_path")
        if rel_path:
            img_path = Path(dataset_dir) / rel_path
        else:
            matches = list(Path(dataset_dir).glob(f"**/{stem}.jpg"))
            img_path = matches[0]

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        mask_path = Path(mask_dir) / f"{stem}.png"
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask > 127).astype(np.uint8)

        # Derive identical bounding box prompt
        fg_coords = np.argwhere(gt_binary > 0)
        if len(fg_coords) > 0:
            y_min, x_min = fg_coords.min(axis=0)
            y_max, x_max = fg_coords.max(axis=0)
            if x_max <= x_min:
                x_max = min(orig_w - 1, x_min + 1)
            if y_max <= y_min:
                y_max = min(orig_h - 1, y_min + 1)
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        else:
            box = np.array([0, 0, orig_w - 1, orig_h - 1], dtype=np.float32)

        # Preprocessing
        img_resized = transform.apply_image(img_rgb)
        resized_h, resized_w = img_resized.shape[:2]
        img_tensor = torch.as_tensor(img_resized, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor - pixel_mean) / pixel_std

        pad_h = base_sam.image_encoder.img_size - resized_h
        pad_w = base_sam.image_encoder.img_size - resized_w
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))

        box_torch = torch.as_tensor(box, dtype=torch.float32, device=device).unsqueeze(0)
        transformed_box = transform.apply_boxes_torch(box_torch, (orig_h, orig_w)).unsqueeze(0)

        # Inference: Base
        base_pred, base_latency = predict_sam_mask(
            base_sam, img_tensor, transformed_box, (resized_h, resized_w), (orig_h, orig_w), device
        )
        base_m = compute_binary_metrics(base_pred, gt_binary)

        # Inference: Pilot (Epoch 5)
        pilot_pred, pilot_latency = predict_sam_mask(
            pilot_sam, img_tensor, transformed_box, (resized_h, resized_w), (orig_h, orig_w), device
        )
        pilot_m = compute_binary_metrics(pilot_pred, gt_binary)

        # Inference: Continued (Epoch 15)
        cont_pred, cont_latency = predict_sam_mask(
            cont_sam, img_tensor, transformed_box, (resized_h, resized_w), (orig_h, orig_w), device
        )
        cont_m = compute_binary_metrics(cont_pred, gt_binary)

        delta_cont_pilot_iou = cont_m["iou"] - pilot_m["iou"]
        delta_cont_pilot_dice = cont_m["dice"] - pilot_m["dice"]

        sample_record = {
            "stem": stem,
            "scene": record.get("scene", "unknown"),
            "base_iou": round(base_m["iou"], 4),
            "base_dice": round(base_m["dice"], 4),
            "pilot_ep5_iou": round(pilot_m["iou"], 4),
            "pilot_ep5_dice": round(pilot_m["dice"], 4),
            "pilot_ep5_precision": round(pilot_m["precision"], 4),
            "pilot_ep5_recall": round(pilot_m["recall"], 4),
            "cont_ep15_iou": round(cont_m["iou"], 4),
            "cont_ep15_dice": round(cont_m["dice"], 4),
            "cont_ep15_precision": round(cont_m["precision"], 4),
            "cont_ep15_recall": round(cont_m["recall"], 4),
            "delta_iou_ep15_vs_ep5": round(delta_cont_pilot_iou, 4),
            "delta_dice_ep15_vs_ep5": round(delta_cont_pilot_dice, 4),
            "is_regression": bool(delta_cont_pilot_iou < -0.01),
        }
        per_image_results.append(sample_record)

        if any(stem.startswith(p) for p in target_prefixes):
            qual_samples.append({
                "stem": stem,
                "scene": record.get("scene", "unknown"),
                "img_rgb": img_rgb,
                "box": box,
                "gt_mask": gt_binary,
                "base_pred": base_pred,
                "pilot_pred": pilot_pred,
                "cont_pred": cont_pred,
                "base_iou": base_m["iou"],
                "pilot_iou": pilot_m["iou"],
                "cont_iou": cont_m["iou"],
            })

    def agg_stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    regressions = [r for r in per_image_results if r["is_regression"]]
    improvements = [r for r in per_image_results if r["delta_iou_ep15_vs_ep5"] > 0.01]
    neutral = [r for r in per_image_results if abs(r["delta_iou_ep15_vs_ep5"]) <= 0.01]

    summary = {
        "num_eval_images": len(eval_stems),
        "target_reference": "SAM pseudo-labels (Task 1 reference), NOT human annotations",
        "base_sam": {
            "iou": agg_stats([r["base_iou"] for r in per_image_results]),
            "dice": agg_stats([r["base_dice"] for r in per_image_results]),
        },
        "pilot_epoch5": {
            "iou": agg_stats([r["pilot_ep5_iou"] for r in per_image_results]),
            "dice": agg_stats([r["pilot_ep5_dice"] for r in per_image_results]),
            "precision": agg_stats([r["pilot_ep5_precision"] for r in per_image_results]),
            "recall": agg_stats([r["pilot_ep5_recall"] for r in per_image_results]),
        },
        "continuation_epoch15": {
            "iou": agg_stats([r["cont_ep15_iou"] for r in per_image_results]),
            "dice": agg_stats([r["cont_ep15_dice"] for r in per_image_results]),
            "precision": agg_stats([r["cont_ep15_precision"] for r in per_image_results]),
            "recall": agg_stats([r["cont_ep15_recall"] for r in per_image_results]),
        },
        "delta_ep15_vs_ep5": {
            "delta_mean_iou": round(float(np.mean([r["delta_iou_ep15_vs_ep5"] for r in per_image_results])), 4),
            "delta_mean_dice": round(float(np.mean([r["delta_dice_ep15_vs_ep5"] for r in per_image_results])), 4),
            "num_improved": len(improvements),
            "num_regressed": len(regressions),
            "num_neutral": len(neutral),
        },
    }

    # Save JSON
    eval_json_path = output_dir / "continuation_vs_pilot_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2)
    print(f"Saved evaluation JSON to: {eval_json_path}")

    # Qualitative Figure: 6 columns
    if qual_samples:
        fig, axes = plt.subplots(len(qual_samples), 6, figsize=(24, 4 * len(qual_samples)))
        if len(qual_samples) == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx, sample in enumerate(qual_samples):
            img_rgb = sample["img_rgb"].copy()
            box = sample["box"]
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 215, 0), 3)

            gt_mask = sample["gt_mask"]
            base_pred = sample["base_pred"]
            pilot_pred = sample["pilot_pred"]
            cont_pred = sample["cont_pred"]
            err_map = create_error_map(cont_pred, gt_mask)

            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f"Sample {row_idx+1}: {sample['scene']}\nInput + Box", fontsize=10)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(gt_mask, cmap="gray")
            axes[row_idx, 1].set_title("Pseudo-GT Target\n(Task 1 Reference)", fontsize=10)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(base_pred, cmap="Blues")
            axes[row_idx, 2].set_title(f"Base SAM\nIoU: {sample['base_iou']:.4f}", fontsize=10)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(pilot_pred, cmap="Greens")
            axes[row_idx, 3].set_title(f"Pilot (Ep 5)\nIoU: {sample['pilot_iou']:.4f}", fontsize=10)
            axes[row_idx, 3].axis("off")

            axes[row_idx, 4].imshow(cont_pred, cmap="Oranges")
            axes[row_idx, 4].set_title(f"Continued (Ep 15)\nIoU: {sample['cont_iou']:.4f}", fontsize=10)
            axes[row_idx, 4].axis("off")

            axes[row_idx, 5].imshow(err_map)
            axes[row_idx, 5].set_title("Ep 15 Error Map\nTP=Grn, FP=Red, FN=Blu", fontsize=10)
            axes[row_idx, 5].axis("off")

        plt.tight_layout()
        qual_fig_path = figures_dir / "continuation_vs_pilot_qualitative.png"
        plt.savefig(qual_fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved qualitative figure to: {qual_fig_path}")

    # Markdown Report
    md_report_path = output_dir / "continuation_evaluation_analysis.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Controlled Continuation Analysis: Pilot (Epoch 5) vs Continued (Epoch 15)\n\n")
        f.write("## 1. Methodology & Split Integrity\n\n")
        f.write("> **METHODOLOGICAL NOTE**: All evaluations were performed on the exact **22 held-out evaluation images** from `data/splits.json`. ")
        f.write("These images were strictly excluded from both the pilot run and the 10-epoch continuation. ")
        f.write("All metrics reflect agreement with **SAM-generated pseudo-labels**, not human annotations.\n\n")
        f.write("## 2. Quantitative Comparison Across Checkpoints\n\n")
        f.write("| Model / Checkpoint | Mean IoU | Median IoU | Mean Dice | Median Dice | Mean Precision | Mean Recall |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Base Pretrained SAM** | {summary['base_sam']['iou']['mean']:.4f} | {summary['base_sam']['iou']['median']:.4f} | {summary['base_sam']['dice']['mean']:.4f} | {summary['base_sam']['dice']['median']:.4f} | - | - |\n")
        f.write(f"| **Pilot (Epoch 5)** | {summary['pilot_epoch5']['iou']['mean']:.4f} | {summary['pilot_epoch5']['iou']['median']:.4f} | {summary['pilot_epoch5']['dice']['mean']:.4f} | {summary['pilot_epoch5']['dice']['median']:.4f} | {summary['pilot_epoch5']['precision']['mean']:.4f} | {summary['pilot_epoch5']['recall']['mean']:.4f} |\n")
        f.write(f"| **Continued (Epoch 15)** | {summary['continuation_epoch15']['iou']['mean']:.4f} | {summary['continuation_epoch15']['iou']['median']:.4f} | {summary['continuation_epoch15']['dice']['mean']:.4f} | {summary['continuation_epoch15']['dice']['median']:.4f} | {summary['continuation_epoch15']['precision']['mean']:.4f} | {summary['continuation_epoch15']['recall']['mean']:.4f} |\n")
        f.write(f"| **Delta (Ep 15 - Ep 5)** | **{summary['delta_ep15_vs_ep5']['delta_mean_iou']:+.4f}** | - | **{summary['delta_ep15_vs_ep5']['delta_mean_dice']:+.4f}** | - | - | - |\n\n")
        f.write(f"**Outcome Summary**: {summary['delta_ep15_vs_ep5']['num_improved']} images improved ($\\Delta > +0.01$), ")
        f.write(f"{summary['delta_ep15_vs_ep5']['num_neutral']} remained neutral ($|\\Delta| \\le 0.01$), ")
        f.write(f"and {summary['delta_ep15_vs_ep5']['num_regressed']} regressed ($\\Delta < -0.01$).\n\n")
        f.write("## 3. Per-Image Breakdown & Regression Analysis\n\n")
        f.write("| Image Stem | Scene | Base IoU | Pilot Ep5 IoU | Cont Ep15 IoU | Delta (15 - 5) | Regression Flag |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in per_image_results:
            flag_str = "⚠️ REGRESSED" if r["is_regression"] else ("✅ IMPROVED" if r["delta_iou_ep15_vs_ep5"] > 0.01 else "—")
            f.write(f"| `{r['stem'][:25]}...` | {r['scene']} | {r['base_iou']:.4f} | {r['pilot_ep5_iou']:.4f} | {r['cont_ep15_iou']:.4f} | {r['delta_iou_ep15_vs_ep5']:+.4f} | {flag_str} |\n")

    print(f"Saved evaluation markdown report to: {md_report_path}")
    return {"summary": summary, "per_image": per_image_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Continuation Checkpoints")
    parser.add_argument("--base-checkpoint", type=str, default="models/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--pilot-checkpoint", type=str, default="models/checkpoints/finetuned/checkpoint_final.pt")
    parser.add_argument("--cont-checkpoint", type=str, default="models/checkpoints/finetuned/continuation_10_epochs/checkpoint_final.pt")
    parser.add_argument("--splits-file", type=str, default="data/splits.json")
    parser.add_argument("--dataset-dir", type=str, default="egg.v1i.folder")
    parser.add_argument("--mask-dir", type=str, default="data/task1_sam_masks/masks")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_continuation(
        base_checkpoint=args.base_checkpoint,
        pilot_checkpoint=args.pilot_checkpoint,
        cont_checkpoint=args.cont_checkpoint,
        splits_file=args.splits_file,
        dataset_dir=args.dataset_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

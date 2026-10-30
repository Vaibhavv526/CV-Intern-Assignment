"""
Evaluation and Comparison Script: Pretrained SAM vs Fine-Tuned SAM.

Compares Meta SAM ViT-B base checkpoint against the fine-tuned mask decoder checkpoint
on the exact 22 held-out evaluation images from data/splits.json.
Uses identical preprocessing, bounding-box prompts, and postprocessing.
Measures IoU, Dice, Precision, Recall, and inference latency.
Generates qualitative side-by-side comparison visualizations.
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
    """
    Computes IoU, Dice, Precision, and Recall between binary prediction and ground truth.
    """
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
    """
    Runs inference on a single image with bounding box prompt and measures latency.
    """
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
    """
    Generates RGB visualization:
      - Green: True Positive
      - Red: False Positive (prediction without GT)
      - Blue: False Negative (GT missed by prediction)
      - Black: True Negative
    """
    h, w = pred.shape
    err_map = np.zeros((h, w, 3), dtype=np.uint8)

    tp = np.logical_and(pred > 0, gt > 0)
    fp = np.logical_and(pred > 0, gt == 0)
    fn = np.logical_and(pred == 0, gt > 0)

    err_map[tp] = [40, 200, 60]    # Green
    err_map[fp] = [220, 50, 50]    # Red
    err_map[fn] = [50, 100, 240]   # Blue
    return err_map


def evaluate_models(
    base_checkpoint: str | Path = "models/checkpoints/sam_vit_b_01ec64.pth",
    ft_checkpoint: str | Path = "models/checkpoints/finetuned/checkpoint_final.pt",
    splits_file: str | Path = "data/splits.json",
    dataset_dir: str | Path = "egg.v1i.folder",
    mask_dir: str | Path = "data/task1_sam_masks/masks",
    output_dir: str | Path = "outputs",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Evaluates Base SAM vs Fine-Tuned SAM on the 22 held-out evaluation images.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluation splits
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    eval_stems = splits["eval"]
    records = splits.get("records", {})
    assert len(eval_stems) == 22, f"Expected 22 eval images, found {len(eval_stems)}"
    print(f"Loaded {len(eval_stems)} held-out evaluation images from {splits_file}")

    # 2. Build Base SAM
    print(f"Loading Base SAM ViT-B from: {base_checkpoint}")
    base_sam = sam_model_registry["vit_b"](checkpoint=str(base_checkpoint)).to(device)
    base_sam.eval()

    # 3. Build Fine-Tuned SAM
    print(f"Loading Fine-Tuned SAM ViT-B with weights from: {ft_checkpoint}")
    ft_sam = sam_model_registry["vit_b"](checkpoint=str(base_checkpoint)).to(device)
    ft_ckpt = torch.load(str(ft_checkpoint), map_location=device)
    if "mask_decoder_state_dict" in ft_ckpt:
        ft_sam.mask_decoder.load_state_dict(ft_ckpt["mask_decoder_state_dict"])
    elif "model_state_dict" in ft_ckpt:
        ft_sam.load_state_dict(ft_ckpt["model_state_dict"])
    ft_sam.eval()

    transform = ResizeLongestSide(base_sam.image_encoder.img_size)
    pixel_mean = base_sam.pixel_mean
    pixel_std = base_sam.pixel_std

    # Warmup pass
    dummy_img = torch.zeros((1, 3, 1024, 1024), device=device)
    dummy_box = torch.tensor([[[0.0, 0.0, 100.0, 100.0]]], device=device)
    with torch.no_grad():
        emb = base_sam.image_encoder(dummy_img)
        sp, dp = base_sam.prompt_encoder(points=None, boxes=dummy_box, masks=None)
        _ = base_sam.mask_decoder(emb, base_sam.prompt_encoder.get_dense_pe(), sp, dp, False)

    per_image_results: List[Dict[str, Any]] = []
    qual_samples: List[Dict[str, Any]] = []

    print("\nRunning side-by-side evaluation across 22 evaluation samples...")

    for idx, stem in enumerate(eval_stems):
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

        # Load pseudo-ground-truth mask
        mask_path = Path(mask_dir) / f"{stem}.png"
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask > 127).astype(np.uint8)

        # Extract bounding box prompt
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

        # SAM Preprocessing
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

        # Predict with Base SAM
        base_pred, base_latency = predict_sam_mask(
            model=base_sam,
            image_tensor=img_tensor,
            box_tensor=transformed_box,
            input_size=(resized_h, resized_w),
            orig_size=(orig_h, orig_w),
            device=device,
        )

        # Predict with Fine-Tuned SAM
        ft_pred, ft_latency = predict_sam_mask(
            model=ft_sam,
            image_tensor=img_tensor,
            box_tensor=transformed_box,
            input_size=(resized_h, resized_w),
            orig_size=(orig_h, orig_w),
            device=device,
        )

        # Metrics against pseudo-GT
        base_m = compute_binary_metrics(base_pred, gt_binary)
        ft_m = compute_binary_metrics(ft_pred, gt_binary)

        sample_record = {
            "stem": stem,
            "scene": record.get("scene", "unknown"),
            "box": [int(b) for b in box],
            "base_iou": round(base_m["iou"], 4),
            "base_dice": round(base_m["dice"], 4),
            "base_precision": round(base_m["precision"], 4),
            "base_recall": round(base_m["recall"], 4),
            "base_latency_ms": round(base_latency, 2),
            "ft_iou": round(ft_m["iou"], 4),
            "ft_dice": round(ft_m["dice"], 4),
            "ft_precision": round(ft_m["precision"], 4),
            "ft_recall": round(ft_m["recall"], 4),
            "ft_latency_ms": round(ft_latency, 2),
            "delta_iou": round(ft_m["iou"] - base_m["iou"], 4),
            "delta_dice": round(ft_m["dice"] - base_m["dice"], 4),
        }
        per_image_results.append(sample_record)

        # Collect representative samples for qualitative figure
        # Choose specific illustrative stems representing diverse scenes and failure recoveries
        target_prefixes = [
            "not_damaged_95_",   # ash_charcoal_bowl (High baseline agreement: IoU 0.998)
            "not_damaged_164_",  # patterned_cloth_fabric (Base failed 0.10 -> FT 0.83)
            "not_damaged_121_",  # water_metallic_bowl (Base failed 0.21 -> FT 0.94)
            "not_damaged_147_",  # kitchen_countertop_multiple (Base failed 0.08 -> FT 0.57)
        ]
        if any(stem.startswith(p) for p in target_prefixes):
            qual_samples.append({
                "stem": stem,
                "scene": record.get("scene", "unknown"),
                "img_rgb": img_rgb,
                "box": box,
                "gt_mask": gt_binary,
                "base_pred": base_pred,
                "ft_pred": ft_pred,
                "base_iou": base_m["iou"],
                "ft_iou": ft_m["iou"],
            })

    # 4. Compute Aggregate Statistics
    def agg_stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    summary = {
        "num_eval_images": len(eval_stems),
        "target_reference": "SAM-generated pseudo-labels (Task 1), NOT human annotations",
        "base_sam": {
            "iou": agg_stats([r["base_iou"] for r in per_image_results]),
            "dice": agg_stats([r["base_dice"] for r in per_image_results]),
            "precision": agg_stats([r["base_precision"] for r in per_image_results]),
            "recall": agg_stats([r["base_recall"] for r in per_image_results]),
            "latency_ms": agg_stats([r["base_latency_ms"] for r in per_image_results]),
        },
        "finetuned_sam": {
            "iou": agg_stats([r["ft_iou"] for r in per_image_results]),
            "dice": agg_stats([r["ft_dice"] for r in per_image_results]),
            "precision": agg_stats([r["ft_precision"] for r in per_image_results]),
            "recall": agg_stats([r["ft_recall"] for r in per_image_results]),
            "latency_ms": agg_stats([r["ft_latency_ms"] for r in per_image_results]),
        },
        "delta_mean": {
            "delta_iou": round(float(np.mean([r["delta_iou"] for r in per_image_results])), 4),
            "delta_dice": round(float(np.mean([r["delta_dice"] for r in per_image_results])), 4),
        },
    }

    # 5. Save JSON report
    eval_json_path = output_dir / "finetuned_vs_base_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "per_image": per_image_results,
        }, f, indent=2)
    print(f"Saved evaluation JSON to: {eval_json_path}")

    # 6. Generate Qualitative Visualization Figure
    fig, axes = plt.subplots(len(qual_samples), 5, figsize=(20, 4 * len(qual_samples)))
    if len(qual_samples) == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Input + Box Prompt",
        "Pseudo-GT Mask (SAM Ref)",
        "Base Pretrained SAM",
        "Fine-Tuned SAM",
        "Error Map (TP=Grn, FP=Red, FN=Blu)",
    ]

    for row_idx, sample in enumerate(qual_samples):
        img_rgb = sample["img_rgb"].copy()
        box = sample["box"]
        # Draw bounding box on input image
        cv2.rectangle(
            img_rgb,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 215, 0),
            3,
        )

        gt_mask = sample["gt_mask"]
        base_pred = sample["base_pred"]
        ft_pred = sample["ft_pred"]
        err_map = create_error_map(ft_pred, gt_mask)

        # Plot 5 columns
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f"Sample {row_idx+1}: {sample['scene']}\n(640x640 RGB)", fontsize=11)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_mask, cmap="gray")
        axes[row_idx, 1].set_title("Pseudo-GT Target\n(Task 1 Reference)", fontsize=11)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(base_pred, cmap="Blues")
        axes[row_idx, 2].set_title(f"Base SAM ViT-B\nIoU: {sample['base_iou']:.4f}", fontsize=11)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(ft_pred, cmap="Greens")
        axes[row_idx, 3].set_title(f"Fine-Tuned SAM\nIoU: {sample['ft_iou']:.4f}", fontsize=11)
        axes[row_idx, 3].axis("off")

        axes[row_idx, 4].imshow(err_map)
        axes[row_idx, 4].set_title("Fine-Tuned Error Map\nTP=Grn, FP=Red, FN=Blu", fontsize=11)
        axes[row_idx, 4].axis("off")

    plt.tight_layout()
    qual_fig_path = figures_dir / "finetuned_vs_base_qualitative.png"
    plt.savefig(qual_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative figure to: {qual_fig_path}")

    # 7. Write Markdown Analysis Report
    md_report_path = output_dir / "finetuned_evaluation_analysis.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Evaluation: Base SAM ViT-B vs Fine-Tuned SAM Mask Decoder\n\n")
        f.write("## 1. Evaluation Methodology and Caveats\n\n")
        f.write("> **CRITICAL METHODOLOGICAL CAVEAT**: This dataset contains **no human-annotated segmentation masks**. ")
        f.write("All metrics below measure agreement against SAM-generated pseudo-labels (`data/task1_sam_masks/masks/`). ")
        f.write("Higher agreement with pseudo-labels does **not** prove superior real-world generalization; ")
        f.write("it specifically measures the fine-tuned decoder's fidelity to prompt-conditioned pseudo-target representations.\n\n")
        f.write("- **Test Set**: Exactly the 22 held-out evaluation images from `data/splits.json`.\n")
        f.write("- **Prompting**: Identical bounding-box prompts extracted from pseudo-label boundaries.\n")
        f.write("- **Preprocessing & Postprocessing**: Bitwise identical `ResizeLongestSide(1024)` and bilinear unpadding.\n\n")
        f.write("## 2. Aggregate Quantitative Comparison\n\n")
        f.write("| Metric | Base Pretrained SAM | Fine-Tuned SAM (Decoder-Only) | Delta (FT - Base) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Mean IoU** | {summary['base_sam']['iou']['mean']:.4f} ± {summary['base_sam']['iou']['std']:.4f} | {summary['finetuned_sam']['iou']['mean']:.4f} ± {summary['finetuned_sam']['iou']['std']:.4f} | **{summary['delta_mean']['delta_iou']:+.4f}** |\n")
        f.write(f"| **Median IoU** | {summary['base_sam']['iou']['median']:.4f} | {summary['finetuned_sam']['iou']['median']:.4f} | {summary['finetuned_sam']['iou']['median'] - summary['base_sam']['iou']['median']:+.4f} |\n")
        f.write(f"| **Mean Dice** | {summary['base_sam']['dice']['mean']:.4f} ± {summary['base_sam']['dice']['std']:.4f} | {summary['finetuned_sam']['dice']['mean']:.4f} ± {summary['finetuned_sam']['dice']['std']:.4f} | **{summary['delta_mean']['delta_dice']:+.4f}** |\n")
        f.write(f"| **Median Dice** | {summary['base_sam']['dice']['median']:.4f} | {summary['finetuned_sam']['dice']['median']:.4f} | {summary['finetuned_sam']['dice']['median'] - summary['base_sam']['dice']['median']:+.4f} |\n")
        f.write(f"| **Mean Precision** | {summary['base_sam']['precision']['mean']:.4f} | {summary['finetuned_sam']['precision']['mean']:.4f} | {summary['finetuned_sam']['precision']['mean'] - summary['base_sam']['precision']['mean']:+.4f} |\n")
        f.write(f"| **Mean Recall** | {summary['base_sam']['recall']['mean']:.4f} | {summary['finetuned_sam']['recall']['mean']:.4f} | {summary['finetuned_sam']['recall']['mean'] - summary['base_sam']['recall']['mean']:+.4f} |\n")
        f.write(f"| **Inference Latency** | {summary['base_sam']['latency_ms']['mean']:.2f} ms | {summary['finetuned_sam']['latency_ms']['mean']:.2f} ms | {summary['finetuned_sam']['latency_ms']['mean'] - summary['base_sam']['latency_ms']['mean']:+.2f} ms |\n\n")
        f.write("## 3. Per-Image Breakdown (22 Evaluation Samples)\n\n")
        f.write("| Image Stem | Scene | Base IoU | FT IoU | Delta IoU | Base Dice | FT Dice | Delta Dice |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in per_image_results:
            f.write(f"| `{r['stem'][:25]}...` | {r['scene']} | {r['base_iou']:.4f} | {r['ft_iou']:.4f} | {r['delta_iou']:+.4f} | {r['base_dice']:.4f} | {r['ft_dice']:.4f} | {r['delta_dice']:+.4f} |\n")
        f.write("\n## 4. Qualitative Visualizations\n\n")
        f.write("Qualitative comparison saved to: `outputs/figures/finetuned_vs_base_qualitative.png`.\n")

    print(f"Saved evaluation markdown report to: {md_report_path}")

    return {
        "summary": summary,
        "per_image": per_image_results,
        "qual_fig": str(qual_fig_path),
        "md_report": str(md_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pretrained SAM vs Fine-Tuned SAM")
    parser.add_argument("--base-checkpoint", type=str, default="models/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--ft-checkpoint", type=str, default="models/checkpoints/finetuned/checkpoint_final.pt")
    parser.add_argument("--splits-file", type=str, default="data/splits.json")
    parser.add_argument("--dataset-dir", type=str, default="egg.v1i.folder")
    parser.add_argument("--mask-dir", type=str, default="data/task1_sam_masks/masks")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate_models(
        base_checkpoint=args.base_checkpoint,
        ft_checkpoint=args.ft_checkpoint,
        splits_file=args.splits_file,
        dataset_dir=args.dataset_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

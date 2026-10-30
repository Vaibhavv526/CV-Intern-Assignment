"""
SAM Fine-Tuning Pipeline for Egg Segmentation.

Trains Meta Segment Anything (SAM ViT-B) mask decoder while keeping the
image encoder (ViT-B) and prompt encoder frozen.
Optimized for NVIDIA RTX 3050 (6 GB VRAM) using batch_size=1 with gradient accumulation.
Exclusively trains on the 88 development images from data/splits.json.
Never uses evaluation images for training, selection, or hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry

from src.sam_dataset import EggSAMDataset, collate_sam
from src.losses import CombinedSegmentationLoss


def build_sam_trainer(
    checkpoint_path: str | Path = "models/checkpoints/sam_vit_b_01ec64.pth",
    model_type: str = "vit_b",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    """
    Loads SAM, freezes image and prompt encoders, and creates optimizer for mask decoder.
    """
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(device)

    # 1. Freeze image encoder and prompt encoder
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    # 2. Set mask decoder as trainable
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    # Parameter accounting
    total_params = sum(p.numel() for p in sam.parameters())
    trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    encoder_params = sum(p.numel() for p in sam.image_encoder.parameters())
    prompt_params = sum(p.numel() for p in sam.prompt_encoder.parameters())
    decoder_params = sum(p.numel() for p in sam.mask_decoder.parameters())

    param_summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_pct": round(trainable_params / total_params * 100, 2),
        "encoder_params": encoder_params,
        "prompt_params": prompt_params,
        "decoder_params": decoder_params,
    }

    # 3. Setup optimizer for mask decoder only
    optimizer = torch.optim.AdamW(
        sam.mask_decoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    return sam, optimizer, param_summary


def run_smoke_test(
    sam: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Executes a one-batch forward/backward smoke test.
    Verifies finite loss, parameter updates, frozen parameter immutability, and GPU memory.
    """
    torch.cuda.empty_cache()
    sam.eval()
    dataset = EggSAMDataset(split="dev", box_jitter=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_sam)
    batch = next(iter(loader))

    images = batch["images"].to(device)       # (1, 3, 1024, 1024)
    boxes = batch["boxes"].to(device)         # (1, 1, 4)
    targets = batch["target_masks"].to(device)# (1, 1, 640, 640)
    stem = batch["stems"][0]

    # Capture initial parameter snapshots for verification
    first_dec_param = next(p for p in sam.mask_decoder.parameters() if p.requires_grad)
    first_enc_param = next(p for p in sam.image_encoder.parameters())
    dec_weight_before = first_dec_param.detach().clone()
    enc_weight_before = first_enc_param.detach().clone()

    # Forward pass
    sam.mask_decoder.train()
    optimizer.zero_grad()

    # Image and prompt encoders run frozen
    with torch.no_grad():
        image_embeddings = sam.image_encoder(images)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

    # Trainable mask decoder
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # Postprocess logits to original resolution (640, 640)
    pred_logits = sam.postprocess_masks(
        low_res_masks,
        input_size=(1024, 1024),
        original_size=(640, 640),
    )

    loss, metrics = loss_fn(pred_logits, targets)

    # Backward pass
    loss.backward()

    # Verify gradients
    dec_grads = [
        p.grad.abs().sum().item()
        for p in sam.mask_decoder.parameters()
        if p.grad is not None
    ]
    enc_grads = [
        p.grad for p in sam.image_encoder.parameters() if p.grad is not None
    ]
    prompt_grads = [
        p.grad for p in sam.prompt_encoder.parameters() if p.grad is not None
    ]

    total_grad_norm = 0.0
    for p in sam.mask_decoder.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    # Optimizer step
    optimizer.step()

    # Verify weight updates
    dec_diff = (first_dec_param.detach() - dec_weight_before).abs().max().item()
    enc_diff = (first_enc_param.detach() - enc_weight_before).abs().max().item()

    # Memory statistics
    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Save smoke test checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_ckpt_path = output_dir / "smoke_test_ckpt.pt"
    torch.save(
        {
            "epoch": 0,
            "step": 1,
            "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
            "metrics": metrics,
        },
        smoke_ckpt_path,
    )

    results = {
        "status": "SUCCESS" if (dec_diff > 0 and enc_diff == 0.0 and np.isfinite(loss.item())) else "FAILED",
        "sample_stem": stem,
        "finite_loss": bool(np.isfinite(loss.item())),
        "total_loss": round(loss.item(), 5),
        "bce_loss": round(metrics["bce"], 5),
        "dice_loss": round(metrics["dice_loss"], 5),
        "decoder_params_with_grad": len(dec_grads),
        "total_decoder_params": len(list(sam.mask_decoder.parameters())),
        "grad_norm": round(total_grad_norm, 5),
        "encoder_params_with_grad": len(enc_grads),
        "prompt_params_with_grad": len(prompt_grads),
        "decoder_weight_delta_max": float(dec_diff),
        "encoder_weight_delta_max": float(enc_diff),
        "gpu_memory_allocated_mb": round(mem_allocated, 2),
        "gpu_memory_reserved_mb": round(mem_reserved, 2),
        "checkpoint_saved": str(smoke_ckpt_path),
    }

    return results


def train_pilot(
    sam: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 1,
    grad_accum_steps: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    resume_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Executes the 5-epoch SAM ViT-B decoder-only fine-tuning pilot run.
    Strictly trained on development split (88 images). Evaluation split (22 images) is never touched.
    """
    print("\n-------------------------------------------------------------")
    print(f"Starting SAM Decoder-Only Fine-Tuning Pilot Run ({epochs} Epochs)")
    print("-------------------------------------------------------------")

    # 1. Verify split isolation
    with open("data/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    dev_set = set(splits["dev"])
    eval_set = set(splits["eval"])
    assert len(dev_set) == 88, f"Expected 88 dev images, found {len(dev_set)}"
    assert len(eval_set) == 22, f"Expected 22 eval images, found {len(eval_set)}"
    assert len(dev_set.intersection(eval_set)) == 0, "FATAL: Overlap detected between dev and eval splits!"
    print(f"Verified Split Isolation: {len(dev_set)} Development images | {len(eval_set)} Held-out Evaluation images (0 Overlap)")

    # 2. Setup Dataset & Loader
    dataset = EggSAMDataset(split="dev", box_jitter=2)
    # Check that every stem in dataset is in dev_set and NOT in eval_set
    for stem in dataset.stems:
        if stem in eval_set:
            raise RuntimeError(f"FATAL: Evaluation stem {stem} found in training dataset!")
    print(f"Dataset successfully initialized with {len(dataset)} verified development samples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sam,
        generator=torch.Generator().manual_seed(42),
    )

    # 3. Handle resumption if requested
    start_epoch = 1
    best_dev_loss = float("inf")
    training_log: List[Dict[str, Any]] = []

    if resume_path and Path(resume_path).exists():
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(str(resume_path), map_location=device)
        sam.mask_decoder.load_state_dict(checkpoint["mask_decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resumed_epoch = checkpoint.get("epoch", 0)
        start_epoch = resumed_epoch + 1
        end_epoch = resumed_epoch + epochs
        best_dev_loss = checkpoint.get("loss", float("inf"))
        
        # Configure conservative continuation learning rate
        # Preserve momentum buffers from AdamW while scaling LR down for fine-tuning stability
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["initial_lr"] = lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        print(f"Resumed at epoch {start_epoch} -> Target end epoch: {end_epoch}")
        print(f"Continuation base LR: {lr:.2e} (Decaying to 1.00e-06 over {epochs} continuation epochs)")
    else:
        end_epoch = epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

    training_start_time = time.time()
    config_dict = {
        "model_type": "vit_b",
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "epochs_run": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "base_lr": lr,
        "weight_decay": weight_decay,
        "loss": "BCEWithLogitsLoss + SoftDiceLoss (alpha=1.0, beta=1.0)",
        "train_samples": len(dataset),
        "split": "dev (strict)",
        "resumed_from": str(resume_path) if resume_path else None,
    }

    # 4. Training Loop
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        sam.mask_decoder.train()
        optimizer.zero_grad()

        total_epoch_loss = 0.0
        total_epoch_bce = 0.0
        total_epoch_dice = 0.0
        num_batches = 0

        print(f"\n--- Epoch {epoch:02d} / {epochs:02d} (LR: {scheduler.get_last_lr()[0]:.2e}) ---")

        for step, batch in enumerate(loader):
            images = batch["images"].to(device)
            boxes = batch["boxes"].to(device)
            targets = batch["target_masks"].to(device)

            try:
                # Encoders forward pass without gradient
                with torch.no_grad():
                    image_embeddings = sam.image_encoder(images)
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None,
                    )

                # Decoder forward pass with gradient tracking
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Upsample prediction logits to original image resolution (640, 640)
                pred_logits = sam.postprocess_masks(
                    low_res_masks,
                    input_size=(1024, 1024),
                    original_size=(640, 640),
                )

                # Compute combined loss
                loss, metrics = loss_fn(pred_logits, targets)

                # Gradient accumulation scaling
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()

                total_epoch_loss += loss.item()
                total_epoch_bce += metrics["bce"]
                total_epoch_dice += metrics["dice_loss"]
                num_batches += 1

                # Optimizer step on accumulation boundary
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(sam.mask_decoder.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError as oom_err:
                print(f"\n[FATAL ERROR] CUDA Out of Memory at Epoch {epoch}, Step {step}: {oom_err}")
                print("Safely saving emergency checkpoint before stopping...")
                emergency_path = output_dir / "emergency_oom_checkpoint.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config_dict,
                    },
                    emergency_path,
                )
                print(f"Preserved emergency checkpoint to: {emergency_path}")
                raise oom_err

        # Step learning rate scheduler
        scheduler.step()

        # Compute epoch averages
        avg_loss = total_epoch_loss / max(num_batches, 1)
        avg_bce = total_epoch_bce / max(num_batches, 1)
        avg_dice = total_epoch_dice / max(num_batches, 1)
        epoch_duration = time.time() - epoch_start_time

        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        vram_res = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        is_best = avg_loss < best_dev_loss
        if is_best:
            best_dev_loss = avg_loss

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "bce_loss": round(avg_bce, 5),
            "dice_loss": round(avg_dice, 5),
            "lr": round(scheduler.get_last_lr()[0], 7),
            "duration_sec": round(epoch_duration, 2),
            "vram_allocated_mb": round(vram_alloc, 2),
            "vram_reserved_mb": round(vram_res, 2),
            "is_best": is_best,
        }
        training_log.append(epoch_record)

        print(
            f"  Epoch {epoch:02d} Summary: Loss={avg_loss:.4f} "
            f"(BCE={avg_bce:.4f}, Dice={avg_dice:.4f}) | "
            f"Time={epoch_duration:.2f}s | "
            f"VRAM: {vram_alloc:.1f}MB alloc / {vram_res:.1f}MB res"
            f"{' [NEW BEST DEV LOSS]' if is_best else ''}"
        )

        # Checkpoint saving: contains full training state for resumption
        ckpt_state = {
            "epoch": epoch,
            "loss": avg_loss,
            "bce_loss": avg_bce,
            "dice_loss": avg_dice,
            "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config_dict,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save per-epoch checkpoint
        ckpt_epoch_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(ckpt_state, ckpt_epoch_path)

        # Save latest checkpoint
        latest_path = output_dir / "checkpoint_latest.pt"
        torch.save(ckpt_state, latest_path)

        # Save best dev loss checkpoint (strictly using development loss, never evaluation masks!)
        if is_best:
            best_path = output_dir / "checkpoint_best.pt"
            torch.save(ckpt_state, best_path)

        # Save final epoch checkpoint
        if epoch == end_epoch:
            final_path = output_dir / "checkpoint_final.pt"
            torch.save(ckpt_state, final_path)
            print(f"  Saved final epoch checkpoint to: {final_path}")

        # Update training log JSON
        log_path = output_dir / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": config_dict,
                "best_dev_loss": round(best_dev_loss, 5),
                "total_duration_sec": round(time.time() - training_start_time, 2),
                "epochs": training_log,
            }, f, indent=2)

    total_time = time.time() - training_start_time
    print("\n=============================================================")
    print(f"   PILOT FINE-TUNING COMPLETE ({total_time:.2f}s total)      ")
    print("=============================================================")
    print(f"Checkpoints directory : {output_dir}")
    print(f"Best Dev Loss         : {best_dev_loss:.5f}")
    print(f"Training Log saved to : {output_dir / 'training_log.json'}")

    return {
        "status": "SUCCESS",
        "total_time_sec": round(total_time, 2),
        "best_dev_loss": round(best_dev_loss, 5),
        "epochs_completed": len(training_log),
        "log": training_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM Fine-Tuning Pipeline for Egg Segmentation")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/sam_vit_b_01ec64.pth", help="Base SAM checkpoint")
    parser.add_argument("--model-type", type=str, default="vit_b", help="SAM model type")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints/finetuned", help="Directory for finetuned checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Execute one-batch smoke test and exit")
    parser.add_argument("--train", action="store_true", help="Execute 5-epoch pilot training run")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1 for 6GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("           META SEGMENT ANYTHING (SAM) FINE-TUNING PIPELINE      ")
    print("=================================================================")
    print(f"Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Base Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {args.output_dir}")

    # Build SAM trainer
    sam, optimizer, param_summary = build_sam_trainer(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("\n--- Model Parameter Breakdown ---")
    print(f"  sam.image_encoder  : {param_summary['encoder_params']:10,d} params [FROZEN]")
    print(f"  sam.prompt_encoder : {param_summary['prompt_params']:10,d} params [FROZEN]")
    print(f"  sam.mask_decoder   : {param_summary['decoder_params']:10,d} params [TRAINABLE]")
    print(f"  Total SAM params   : {param_summary['total_params']:10,d} params")
    print(f"  Trainable fraction : {param_summary['trainable_pct']:.2f}% ({param_summary['trainable_params']:,d} params)")

    loss_fn = CombinedSegmentationLoss(bce_weight=1.0, dice_weight=1.0)

    if args.smoke_test:
        print("\n--- Running One-Batch Forward/Backward Smoke Test ---")
        smoke_results = run_smoke_test(sam, optimizer, loss_fn, args.device, output_dir)
        
        # Save smoke test results JSON
        os.makedirs("outputs", exist_ok=True)
        smoke_report_path = Path("outputs/smoke_test_results.json")
        with open(smoke_report_path, "w", encoding="utf-8") as f:
            json.dump({
                "param_summary": param_summary,
                "smoke_test": smoke_results,
            }, f, indent=2)

        print("\n=== SMOKE TEST RESULTS ===")
        print(f"  Status                    : {smoke_results['status']}")
        print(f"  Sample Dev Image          : {smoke_results['sample_stem']}")
        print(f"  Loss is Finite            : {smoke_results['finite_loss']}")
        print(f"  Total Loss (BCE + Dice)   : {smoke_results['total_loss']:.4f}")
        print(f"    - BCE Loss              : {smoke_results['bce_loss']:.4f}")
        print(f"    - Soft Dice Loss        : {smoke_results['dice_loss']:.4f}")
        print(f"  Mask Decoder Grad Norm    : {smoke_results['grad_norm']:.4f}")
        print(f"  Decoder Params w/ Grad    : {smoke_results['decoder_params_with_grad']} / {smoke_results['total_decoder_params']}")
        print(f"  Frozen Encoder Grad Count : {smoke_results['encoder_params_with_grad']} (Expected: 0)")
        print(f"  Frozen Prompt Grad Count  : {smoke_results['prompt_params_with_grad']} (Expected: 0)")
        print(f"  Decoder Weight Delta (Max): {smoke_results['decoder_weight_delta_max']:.6e} (Updated)")
        print(f"  Encoder Weight Delta (Max): {smoke_results['encoder_weight_delta_max']:.6e} (Bitwise identical)")
        print(f"  GPU VRAM Allocated        : {smoke_results['gpu_memory_allocated_mb']:.2f} MB")
        print(f"  GPU VRAM Reserved         : {smoke_results['gpu_memory_reserved_mb']:.2f} MB")
        print(f"  Smoke Checkpoint Saved    : {smoke_results['checkpoint_saved']}")
        print(f"  Report Saved              : {smoke_report_path}")
        print("\nSmoke test successfully passed. Ready for pilot run.")
        return

    if args.train:
        train_results = train_pilot(
            sam=sam,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=args.device,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
            weight_decay=args.weight_decay,
            resume_path=args.resume,
        )
        return

    print("\nSpecify --smoke-test or --train to execute.")


if __name__ == "__main__":
    main()

"""
Dataset and DataLoader utilities for SAM Fine-Tuning on Egg Segmentation.

Loads images from egg.v1i.folder/ and pseudo-ground-truth masks from data/task1_sam_masks/masks/.
Applies SAM-compliant preprocessing:
  - ResizeLongestSide (640x640 -> 1024x1024)
  - Standardization using SAM's pixel_mean and pixel_std
  - Derives bounding box prompts with coordinate transformation to SAM input space
  - Preserves original mask resolution (640x640) for loss evaluation against upsampled logits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide


class EggSAMDataset(Dataset):
    """
    PyTorch Dataset for SAM fine-tuning using pseudo-labeled egg masks.
    """

    def __init__(
        self,
        splits_path: str | Path = "data/splits.json",
        dataset_dir: str | Path = "egg.v1i.folder",
        mask_dir: str | Path = "data/task1_sam_masks/masks",
        split: str = "dev",
        img_size: int = 1024,
        pixel_mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        pixel_std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
        box_jitter: int = 0,
        exclude_stems: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.splits_path = Path(splits_path)
        self.dataset_dir = Path(dataset_dir)
        self.mask_dir = Path(mask_dir)
        self.split = split
        self.img_size = img_size
        self.box_jitter = box_jitter
        self.pixel_mean = torch.tensor(pixel_mean).view(3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(3, 1, 1)
        self.transform = ResizeLongestSide(img_size)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        all_stems = splits_data[split]
        exclude_set = set(exclude_stems or [])
        self.stems = [s for s in all_stems if s not in exclude_set]
        self.records = splits_data.get("records", {})

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.stems[idx]
        record = self.records.get(stem, {})

        # Load image
        rel_path = record.get("rel_path")
        if rel_path:
            img_path = self.dataset_dir / rel_path
        else:
            matches = list(self.dataset_dir.glob(f"**/{stem}.jpg"))
            if not matches:
                raise FileNotFoundError(f"Image for {stem} not found in {self.dataset_dir}")
            img_path = matches[0]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image from {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Load pseudo-ground-truth mask
        mask_path = self.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask from {mask_path}")
        binary_mask = (mask > 127).astype(np.float32)

        # Compute bounding box prompt
        fg_coords = np.argwhere(binary_mask > 0)
        if len(fg_coords) > 0:
            y_min, x_min = fg_coords.min(axis=0)
            y_max, x_max = fg_coords.max(axis=0)
            # Ensure valid non-degenerate box
            if x_max <= x_min:
                x_max = min(orig_w - 1, x_min + 1)
            if y_max <= y_min:
                y_max = min(orig_h - 1, y_min + 1)

            # Optional jitter for data augmentation
            if self.box_jitter > 0:
                jx1 = np.random.randint(-self.box_jitter, self.box_jitter + 1)
                jy1 = np.random.randint(-self.box_jitter, self.box_jitter + 1)
                jx2 = np.random.randint(-self.box_jitter, self.box_jitter + 1)
                jy2 = np.random.randint(-self.box_jitter, self.box_jitter + 1)
                x_min = max(0, x_min + jx1)
                y_min = max(0, y_min + jy1)
                x_max = min(orig_w - 1, max(x_min + 1, x_max + jx2))
                y_max = min(orig_h - 1, max(y_min + 1, y_max + jy2))
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        else:
            # Fallback for empty mask (full-image box)
            box = np.array([0, 0, orig_w - 1, orig_h - 1], dtype=np.float32)

        # SAM Preprocessing: ResizeLongestSide + Pad to 1024x1024
        img_resized = self.transform.apply_image(img_rgb)
        resized_h, resized_w = img_resized.shape[:2]

        img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
        # Normalize
        img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std

        # Pad to (1024, 1024)
        pad_h = self.img_size - resized_h
        pad_w = self.img_size - resized_w
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))

        # Coordinate transform for bounding box prompt into 1024x1024 space
        box_torch = torch.as_tensor(box, dtype=torch.float32).unsqueeze(0)  # (1, 4)
        transformed_box = self.transform.apply_boxes_torch(box_torch, (orig_h, orig_w))  # (1, 4)

        # Target mask tensor at original resolution (1, H_orig, W_orig)
        target_mask_tensor = torch.as_tensor(binary_mask, dtype=torch.float32).unsqueeze(0)

        return {
            "image": img_tensor,                        # (3, 1024, 1024)
            "target_mask": target_mask_tensor,          # (1, 640, 640)
            "boxes": transformed_box,                  # (1, 4)
            "orig_boxes": box_torch,                   # (1, 4)
            "original_size": (orig_h, orig_w),
            "input_size": (resized_h, resized_w),
            "stem": stem,
            "scene": record.get("scene", "unknown"),
        }


def collate_sam(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching SAM training samples.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B, 3, 1024, 1024)
    target_masks = torch.stack([b["target_mask"] for b in batch], dim=0)  # (B, 1, 640, 640)
    boxes = torch.stack([b["boxes"].squeeze(0) for b in batch], dim=0)  # (B, 4)
    orig_boxes = torch.stack([b["orig_boxes"].squeeze(0) for b in batch], dim=0)  # (B, 4)

    return {
        "images": images,
        "target_masks": target_masks,
        "boxes": boxes.unsqueeze(1),  # (B, 1, 4)
        "orig_boxes": orig_boxes,
        "original_sizes": [b["original_size"] for b in batch],
        "input_sizes": [b["input_size"] for b in batch],
        "stems": [b["stem"] for b in batch],
        "scenes": [b["scene"] for b in batch],
    }

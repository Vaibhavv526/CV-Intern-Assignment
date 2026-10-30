"""
Loss functions for SAM Fine-Tuning.

Combines Binary Cross-Entropy with Logits and Soft Dice Loss:
  L_total = alpha * L_BCE + beta * L_Dice
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Operates directly on unnormalized logits (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) unnormalized prediction logits.
            targets: (B, 1, H, W) binary ground truth {0.0, 1.0}.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and Soft Dice Loss.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: combined scalar tensor for backward().
            metrics: dict with detached float values for logging.
        """
        bce_val = self.bce(logits, targets)
        dice_val = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_val + self.dice_weight * dice_val

        metrics = {
            "loss": total_loss.item(),
            "bce": bce_val.item(),
            "dice_loss": dice_val.item(),
        }
        return total_loss, metrics

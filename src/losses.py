from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.constants import IGNORE_INDEX, NUM_CLASSES


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    valid_mask = targets != IGNORE_INDEX
    safe_targets = targets.clone()
    safe_targets[~valid_mask] = 0
    one_hot = F.one_hot(safe_targets, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    valid_mask = valid_mask.unsqueeze(1)
    probabilities = probabilities * valid_mask
    one_hot = one_hot * valid_mask
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dim=dims)
    cardinality = torch.sum(probabilities + one_hot, dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class SegmentationLoss:
    def __init__(self, config: dict, class_weights: Optional[torch.Tensor] = None) -> None:
        self.name = config["loss"]["name"]
        self.ce_weight = float(config["loss"].get("ce_weight", 1.0))
        self.dice_weight = float(config["loss"].get("dice_weight", 1.0))
        self.label_smoothing = float(config["loss"].get("label_smoothing", 0.0))
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
        )
        dice = multiclass_dice_loss(logits, targets)
        if self.name == "ce":
            return ce
        if self.name == "dice":
            return dice
        if self.name == "ce_dice":
            return self.ce_weight * ce + self.dice_weight * dice
        raise ValueError(f"Unknown loss name: {self.name}")

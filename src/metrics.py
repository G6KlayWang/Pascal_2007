from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.constants import IGNORE_INDEX, NUM_CLASSES, VOC_CLASSES


def _surface_distances(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return np.array([], dtype=np.float32)
    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)
    pred_dt = distance_transform_edt(~pred_border)
    target_dt = distance_transform_edt(~target_border)
    distances = np.concatenate([pred_dt[target_border], target_dt[pred_border]])
    return distances.astype(np.float32)


def hd95_for_class(pred: np.ndarray, target: np.ndarray) -> float:
    distances = _surface_distances(pred, target)
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


@dataclass
class RunningMetrics:
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    hd95_values: dict[int, list[float]] = field(default_factory=lambda: {i: [] for i in range(NUM_CLASSES)})

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1)
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        valid = targets_np != IGNORE_INDEX
        filtered_true = targets_np[valid].astype(np.int64)
        filtered_pred = preds_np[valid].astype(np.int64)
        flat = NUM_CLASSES * filtered_true + filtered_pred
        self.confusion += np.bincount(flat, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        for batch_idx in range(preds_np.shape[0]):
            for class_id in range(NUM_CLASSES):
                pred_mask = preds_np[batch_idx] == class_id
                true_mask = targets_np[batch_idx] == class_id
                if pred_mask.any() and true_mask.any():
                    self.hd95_values[class_id].append(hd95_for_class(pred_mask, true_mask))

    def summary(self) -> dict[str, Any]:
        tp = np.diag(self.confusion).astype(np.float64)
        gt = self.confusion.sum(axis=1).astype(np.float64)
        pred = self.confusion.sum(axis=0).astype(np.float64)
        union = gt + pred - tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        acc = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
        dice = np.divide(2 * tp, gt + pred, out=np.zeros_like(tp), where=(gt + pred) > 0)
        pixel_acc = float(tp.sum() / max(1.0, self.confusion.sum()))
        hd95 = []
        for class_id in range(NUM_CLASSES):
            values = [value for value in self.hd95_values[class_id] if not np.isnan(value)]
            hd95.append(float(np.mean(values)) if values else float("nan"))

        per_class = []
        for class_id, class_name in enumerate(VOC_CLASSES):
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": float(iou[class_id]),
                    "accuracy": float(acc[class_id]),
                    "dice": float(dice[class_id]),
                    "hd95": hd95[class_id],
                }
            )
        valid_hd95 = [value for value in hd95 if not np.isnan(value)]
        return {
            "mean_iou": float(np.mean(iou)),
            "mean_dice": float(np.mean(dice)),
            "pixel_accuracy": pixel_acc,
            "mean_hd95": float(np.mean(valid_hd95)) if valid_hd95 else float("nan"),
            "per_class": per_class,
            "confusion_matrix": self.confusion.tolist(),
        }

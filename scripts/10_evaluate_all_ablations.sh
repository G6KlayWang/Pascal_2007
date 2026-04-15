#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

# Runs src.evaluate on every ablation checkpoint, which emits:
#   metrics.json (mean_iou, mean_dice, pixel_accuracy, mean_hd95, per_class, confusion_matrix)
#   per_class_metrics.csv
#   confusion_matrix.png
#   prediction_mosaic.png              <- qualitative mosaic
#   best_worst_person.png              <- top-3 best / top-3 worst on 'person' class
# then aggregates per-ablation summary.csv + outputs/ablations/all_summary.csv.
python -m src.evaluate_ablations \
    --ablations \
        configs/unet.yaml:model_size \
        configs/sam2.yaml:augmentation \
        configs/sam2.yaml:loss \
    --output outputs/ablations/all_summary.csv

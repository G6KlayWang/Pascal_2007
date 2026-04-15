#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

# 1) Model size (U-Net): ResNet-18 vs ResNet-50
python -m src.ablations --config configs/unet.yaml --type model_size

# 2) Data augmentation (SAM2): off vs on
python -m src.ablations --config configs/sam2.yaml --type augmentation

# 3) Loss function (SAM2): CE vs Dice
python -m src.ablations --config configs/sam2.yaml --type loss

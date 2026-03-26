#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

python -m src.evaluate --config configs/unet.yaml --checkpoint outputs/train_logs/unet/unet-default/checkpoints/best.pt
python -m src.evaluate --config configs/deeplabv3.yaml --checkpoint outputs/train_logs/deeplabv3/deeplabv3-default/checkpoints/best.pt
python -m src.evaluate --config configs/sam2.yaml --checkpoint outputs/train_logs/sam2/sam2-default/checkpoints/best.pt
python -m src.aggregate_results --configs configs/unet.yaml configs/deeplabv3.yaml configs/sam2.yaml

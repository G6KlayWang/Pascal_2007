#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/01_download_prepare.sh
bash scripts/02_train_unet.sh
bash scripts/03_train_deeplabv3.sh
bash scripts/04_train_sam.sh
bash scripts/05_evaluate_all.sh

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate
export MPLCONFIGDIR="$ROOT_DIR/.codex_tmp/matplotlib"
export TORCH_HOME="$ROOT_DIR/.codex_tmp/torch"
export XDG_CACHE_HOME="$ROOT_DIR/.codex_tmp/cache"
python -m src.ablations --config configs/unet.yaml --type augmentation "$@"

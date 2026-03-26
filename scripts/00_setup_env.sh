#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p .codex_tmp .codex_tmp/matplotlib .codex_tmp/torch .codex_tmp/cache
export MPLCONFIGDIR="$ROOT_DIR/.codex_tmp/matplotlib"
export TORCH_HOME="$ROOT_DIR/.codex_tmp/torch"
export XDG_CACHE_HOME="$ROOT_DIR/.codex_tmp/cache"
echo "Environment ready. Activate with: source .venv/bin/activate"

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import apply_overrides, load_config
from src.data import build_dataloaders
from src.engine import fit
from src.losses import SegmentationLoss
from src.models import build_model
from src.utils import count_parameters, device_from_config, ensure_dir, save_json, seed_everything, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(config["experiment"]["seed"])
    device = device_from_config(config["experiment"].get("device"))
    loaders, _, _ = build_dataloaders(config)
    model = build_model(config, device)
    criterion = SegmentationLoss(config)

    run_name = config["experiment"].get("run_name") or f"{config['model']['type']}-{timestamp()}"
    run_dir = ensure_dir(Path(config["experiment"]["output_root"]) / "train_logs" / config["model"]["type"] / run_name)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "model_info.json", {"trainable_parameters": count_parameters(model), "device": str(device)})

    best_ckpt, history, summary = fit(model, loaders, config, criterion, device, run_dir)
    print(json.dumps({"best_checkpoint": str(best_ckpt), "summary": summary, "epochs": len(history)}, indent=2))


if __name__ == "__main__":
    main()

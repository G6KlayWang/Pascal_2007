from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config import apply_overrides, load_config
from src.data import build_dataloaders, compute_class_weights
from src.dist import cleanup, is_main, setup_distributed
from src.engine import fit
from src.losses import SegmentationLoss
from src.models import build_model
from src.utils import count_parameters, ensure_dir, save_json, seed_everything, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    distributed, rank, world_size, local_rank, device = setup_distributed()
    seed_everything(config["experiment"]["seed"] + rank)

    loaders, voc_root, split_manifest = build_dataloaders(config)
    model = build_model(config, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    class_weights = None
    if config["loss"].get("use_class_weights", False):
        class_weights = compute_class_weights(
            voc_root=voc_root,
            sample_ids=split_manifest["train"],
            power=float(config["loss"].get("class_weight_power", 0.5)),
            clip_max=float(config["loss"].get("class_weight_clip_max", 10.0)),
        ).to(device)
    criterion = SegmentationLoss(config, class_weights=class_weights)

    run_name = config["experiment"].get("run_name") or f"{config['model']['type']}-{timestamp()}"
    run_dir = ensure_dir(Path(config["experiment"]["output_root"]) / "train_logs" / config["model"]["type"] / run_name)
    if is_main():
        save_json(run_dir / "resolved_config.json", config)
        save_json(
            run_dir / "model_info.json",
            {
                "trainable_parameters": count_parameters(model),
                "device": str(device),
                "world_size": world_size,
            },
        )

    best_ckpt, history, summary = fit(model, loaders, config, criterion, device, run_dir)
    if is_main():
        print(json.dumps({"best_checkpoint": str(best_ckpt), "summary": summary, "epochs": len(history)}, indent=2))
    cleanup()


if __name__ == "__main__":
    main()

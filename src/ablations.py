from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.utils import ensure_dir, save_csv


ABLATIONS = {
    "augmentation": [
        ("no_aug", ["augmentation.enabled=false", "experiment.run_name=unet-no-aug"]),
        ("default_aug", ["augmentation.enabled=true", "experiment.run_name=unet-default-aug"]),
    ],
    "loss": [
        ("ce", ["loss.name=ce", "experiment.run_name=unet-loss-ce"]),
        ("dice", ["loss.name=dice", "experiment.run_name=unet-loss-dice"]),
    ],
    "model_size": [
        ("resnet18", ["model.encoder=resnet18", "experiment.run_name=unet-resnet18"]),
        ("resnet50", ["model.encoder=resnet50", "experiment.run_name=unet-resnet50"]),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--type", choices=sorted(ABLATIONS.keys()), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ablation_dir = ensure_dir(Path(config["experiment"]["output_root"]) / "ablations" / args.type)
    root = Path(__file__).resolve().parent.parent
    rows = []
    for _, overrides in ABLATIONS[args.type]:
        cmd = [sys.executable, "-m", "src.train", "--config", args.config, "--set", *overrides]
        subprocess.run(cmd, cwd=root, check=True)
        run_name = next(value.split("=", 1)[1] for value in overrides if value.startswith("experiment.run_name="))
        checkpoint_path = root / "outputs" / "train_logs" / "unet" / run_name / "checkpoints" / "best.pt"
        eval_cmd = [sys.executable, "-m", "src.evaluate", "--config", args.config, "--checkpoint", str(checkpoint_path), "--set", *overrides]
        subprocess.run(eval_cmd, cwd=root, check=True)
        metrics_path = root / "outputs" / "eval" / "unet" / run_name / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "run_name": run_name,
                "mean_iou": metrics["mean_iou"],
                "mean_dice": metrics["mean_dice"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "mean_hd95": metrics["mean_hd95"],
            }
        )
    save_csv(ablation_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()

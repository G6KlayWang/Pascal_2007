from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.utils import ensure_dir, save_json


def build_ablations(model_type: str) -> dict[str, list[tuple[str, list[str]]]]:
    prefix = model_type
    return {
        "model_size": [
            ("resnet18", [f"model.encoder=resnet18", f"experiment.run_name={prefix}-resnet18"]),
            ("resnet50", [f"model.encoder=resnet50", f"experiment.run_name={prefix}-resnet50"]),
        ],
        "augmentation": [
            ("no_aug",   [f"augmentation.enabled=false", f"experiment.run_name={prefix}-no-aug"]),
            ("with_aug", [f"augmentation.enabled=true",  f"experiment.run_name={prefix}-with-aug"]),
        ],
        "loss": [
            ("ce",   [f"loss.name=ce",   f"experiment.run_name={prefix}-loss-ce"]),
            ("dice", [f"loss.name=dice", f"experiment.run_name={prefix}-loss-dice"]),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ablation variants (training only; evaluation handled separately).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--type", required=True, choices=["model_size", "augmentation", "loss"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_type = config["model"]["type"]
    variants = build_ablations(model_type)[args.type]
    output_root = Path(config["experiment"]["output_root"])
    ablation_dir = ensure_dir(output_root / "ablations" / model_type / args.type)
    project_root = Path(__file__).resolve().parent.parent

    runs = []
    for label, overrides in variants:
        run_name = next(value.split("=", 1)[1] for value in overrides if value.startswith("experiment.run_name="))
        print(f"\n>>> [ablation:{args.type}] training {label} -> {run_name}")
        cmd = [sys.executable, "-m", "src.train", "--config", args.config, "--set", *overrides]
        subprocess.run(cmd, cwd=project_root, check=True)
        runs.append({
            "label": label,
            "run_name": run_name,
            "model_type": model_type,
            "config": args.config,
            "overrides": overrides,
            "checkpoint": str(output_root / "train_logs" / model_type / run_name / "checkpoints" / "best.pt"),
        })
    save_json(ablation_dir / "runs.json", {"type": args.type, "model_type": model_type, "runs": runs})
    print(f"\n[ablation:{args.type}] manifest -> {ablation_dir / 'runs.json'}")


if __name__ == "__main__":
    main()

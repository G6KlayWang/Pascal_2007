from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.utils import ensure_dir, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate training and evaluation results across models.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/comparisons")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    rows = []
    for config_path in args.configs:
        config = load_config(config_path)
        model_type = config["model"]["type"]
        run_name = config["experiment"]["run_name"]
        output_root = Path(config["experiment"]["output_root"])
        train_summary_path = output_root / "train_logs" / model_type / run_name / "logs" / "summary.json"
        eval_metrics_path = output_root / "eval" / model_type / run_name / "metrics.json"
        model_info_path = output_root / "train_logs" / model_type / run_name / "model_info.json"

        if not train_summary_path.exists() or not eval_metrics_path.exists():
            continue

        train_summary = load_config(train_summary_path)
        eval_metrics = load_config(eval_metrics_path)
        model_info = load_config(model_info_path) if model_info_path.exists() else {}
        rows.append(
            {
                "model": model_type,
                "run_name": run_name,
                "trainable_parameters": model_info.get("trainable_parameters"),
                "best_epoch": train_summary.get("best_epoch"),
                "best_val_mean_iou": train_summary.get("best_val_mean_iou"),
                "history_length": train_summary.get("history_length"),
                "total_train_seconds": train_summary.get("total_train_seconds"),
                "avg_epoch_seconds": train_summary.get("avg_epoch_seconds"),
                "test_mean_iou": eval_metrics.get("mean_iou"),
                "test_mean_dice": eval_metrics.get("mean_dice"),
                "test_pixel_accuracy": eval_metrics.get("pixel_accuracy"),
                "test_mean_hd95": eval_metrics.get("mean_hd95"),
            }
        )

    save_csv(Path(output_dir) / "model_comparison.csv", rows)
    save_json(Path(output_dir) / "model_comparison.json", {"rows": rows})


if __name__ == "__main__":
    main()

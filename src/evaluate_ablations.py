from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.utils import ensure_dir, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all ablation runs and aggregate PDF-required metrics. "
            "Reads manifests produced by src.ablations (runs.json) and invokes src.evaluate "
            "for each run, which regenerates metrics.json, per_class_metrics.csv, "
            "confusion_matrix.png, prediction_mosaic.png, and best_worst_person.png."
        )
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        required=True,
        help="One or more <config>:<type> pairs, e.g. configs/unet.yaml:model_size configs/sam2.yaml:loss",
    )
    parser.add_argument("--output", default="outputs/ablations/all_summary.csv")
    return parser.parse_args()


def run_evaluate(project_root: Path, config: str, checkpoint: str, overrides: list[str]) -> None:
    cmd = [sys.executable, "-m", "src.evaluate",
           "--config", config,
           "--checkpoint", checkpoint,
           "--set", *overrides]
    subprocess.run(cmd, cwd=project_root, check=True)


def metrics_row(run: dict, metrics: dict, eval_dir: Path, ablation_type: str) -> dict:
    per_class = metrics.get("per_class", [])
    person = next((pc for pc in per_class if pc["class_name"].strip() == "person"), None)
    return {
        "ablation_type": ablation_type,
        "label": run["label"],
        "run_name": run["run_name"],
        "model_type": run["model_type"],
        "test_mean_iou": metrics.get("mean_iou"),
        "test_mean_dice": metrics.get("mean_dice"),
        "test_pixel_accuracy": metrics.get("pixel_accuracy"),
        "test_mean_hd95": metrics.get("mean_hd95"),
        "person_iou": person["iou"] if person else None,
        "person_dice": person["dice"] if person else None,
        "person_hd95": person["hd95"] if person else None,
        "eval_dir": str(eval_dir),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    combined_rows: list[dict] = []

    for spec in args.ablations:
        if ":" not in spec:
            raise ValueError(f"Expected <config>:<type>, got: {spec}")
        config_path, abl_type = spec.split(":", 1)
        config = load_config(config_path)
        model_type = config["model"]["type"]
        output_root = Path(config["experiment"]["output_root"])
        manifest_path = output_root / "ablations" / model_type / abl_type / "runs.json"
        if not manifest_path.exists():
            print(f"[warn] manifest missing: {manifest_path} — skipping {spec}")
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        runs = manifest.get("runs", [])
        rows: list[dict] = []
        for run in runs:
            ckpt = Path(run["checkpoint"])
            if not ckpt.exists():
                print(f"[warn] missing checkpoint for {run['run_name']}: {ckpt} — skipping")
                continue
            print(f"\n>>> [eval:{abl_type}] {run['run_name']}")
            run_evaluate(project_root, run["config"], str(ckpt), run["overrides"])
            eval_dir = output_root / "eval" / run["model_type"] / run["run_name"]
            metrics_path = eval_dir / "metrics.json"
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            rows.append(metrics_row(run, metrics, eval_dir, abl_type))

        abl_dir = ensure_dir(output_root / "ablations" / model_type / abl_type)
        save_csv(abl_dir / "summary.csv", rows)
        save_json(abl_dir / "summary.json", {"type": abl_type, "model_type": model_type, "rows": rows})
        print(f"[eval:{abl_type}] summary -> {abl_dir / 'summary.csv'}")
        combined_rows.extend(rows)

    combined_path = Path(args.output)
    ensure_dir(combined_path.parent)
    save_csv(combined_path, combined_rows)
    print(f"\n[eval] combined summary -> {combined_path}")


if __name__ == "__main__":
    main()

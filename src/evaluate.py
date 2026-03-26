from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import apply_overrides, load_config
from src.constants import VOC_CLASSES
from src.data import build_dataloaders, compute_class_weights
from src.engine import evaluate_loader
from src.losses import SegmentationLoss
from src.models import build_model
from src.utils import device_from_config, ensure_dir, save_csv, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def save_confusion_matrix(confusion: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.array(confusion), cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_qualitative_examples(model: torch.nn.Module, loader, device: torch.device, output_dir: Path, class_id: int) -> None:
    per_sample = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for idx in range(images.shape[0]):
                pred_mask = preds[idx] == class_id
                true_mask = masks[idx] == class_id
                intersection = torch.logical_and(pred_mask, true_mask).sum().item()
                union = torch.logical_or(pred_mask, true_mask).sum().item()
                score = intersection / max(1, union)
                per_sample.append(
                    {
                        "id": batch["id"][idx],
                        "score": score,
                        "image": images[idx].cpu(),
                        "pred": preds[idx].cpu(),
                        "mask": masks[idx].cpu(),
                    }
                )
    per_sample.sort(key=lambda item: item["score"])
    chosen = per_sample[:3] + per_sample[-3:]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])
    for row, sample in enumerate(chosen):
        image = sample["image"].permute(1, 2, 0).numpy()
        image = np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(sample["id"])
        axes[row, 1].imshow(sample["mask"].numpy(), cmap="tab20", vmin=0, vmax=len(VOC_CLASSES) - 1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(sample["pred"].numpy(), cmap="tab20", vmin=0, vmax=len(VOC_CLASSES) - 1)
        axes[row, 2].set_title(f"Prediction IoU={sample['score']:.3f}")
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "best_worst_person.png", dpi=200)
    plt.close(fig)


def save_random_mosaic(model: torch.nn.Module, loader, device: torch.device, output_dir: Path, max_samples: int = 6) -> None:
    samples = []
    sample_cap = max_samples * max(1, loader.batch_size or 1)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for idx in range(images.shape[0]):
                samples.append(
                    {
                        "id": batch["id"][idx],
                        "image": images[idx].cpu(),
                        "pred": preds[idx].cpu(),
                    }
                )
                if len(samples) >= sample_cap:
                    break
            if len(samples) >= sample_cap:
                break

    if len(samples) > max_samples:
        order = torch.randperm(len(samples))[:max_samples].tolist()
        samples = [samples[idx] for idx in order]

    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 3 * len(samples)))
    if len(samples) == 1:
        axes = np.array([axes])
    for row, sample in enumerate(samples):
        image = sample["image"].permute(1, 2, 0).numpy()
        image = np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(sample["id"])
        axes[row, 1].imshow(sample["pred"].numpy(), cmap="tab20", vmin=0, vmax=len(VOC_CLASSES) - 1)
        axes[row, 1].set_title("Prediction")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_mosaic.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(config["experiment"]["seed"])
    device = device_from_config(config["experiment"].get("device"))
    loaders, voc_root, split_manifest = build_dataloaders(config)
    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    class_weights = None
    if config["loss"].get("use_class_weights", False):
        class_weights = compute_class_weights(
            voc_root=voc_root,
            sample_ids=split_manifest["train"],
            power=float(config["loss"].get("class_weight_power", 0.5)),
            clip_max=float(config["loss"].get("class_weight_clip_max", 10.0)),
        ).to(device)
    criterion = SegmentationLoss(config, class_weights=class_weights)

    run_name = checkpoint["config"]["experiment"].get("run_name", Path(args.checkpoint).parent.parent.name)
    output_dir = ensure_dir(Path(config["experiment"]["output_root"]) / "eval" / config["model"]["type"] / run_name)
    summary = evaluate_loader(model, loaders["test"], criterion, device)
    save_json(output_dir / "metrics.json", summary)
    save_csv(output_dir / "per_class_metrics.csv", summary["per_class"])
    save_confusion_matrix(summary["confusion_matrix"], output_dir / "confusion_matrix.png")
    person_class_id = VOC_CLASSES.index(config["eval"].get("best_worst_class", "person"))
    save_random_mosaic(model, loaders["test"], device, output_dir)
    save_qualitative_examples(model, loaders["test"], device, output_dir, class_id=person_class_id)
    print(json.dumps({"evaluation_dir": str(output_dir), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()

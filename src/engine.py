from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dist import all_reduce_mean, all_reduce_sum, is_distributed, is_main
from src.losses import SegmentationLoss
from src.metrics import RunningMetrics
from src.utils import ensure_dir, save_csv, save_json


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        base = _unwrap(model)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in base.state_dict().items()
            if torch.is_floating_point(tensor)
        }

    def update(self, model: torch.nn.Module) -> None:
        base = _unwrap(model)
        with torch.no_grad():
            for name, tensor in base.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        state = _unwrap(model).state_dict()
        for name, tensor in self.shadow.items():
            state[name].copy_(tensor)


def create_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    decoder_lr = train_cfg.get("decoder_lr", train_cfg.get("lr", 1e-3))
    backbone_lr = train_cfg.get("backbone_lr", decoder_lr)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": decoder_lr})

    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=decoder_lr, weight_decay=train_cfg["weight_decay"])
    if optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, lr=decoder_lr, weight_decay=train_cfg["weight_decay"], momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    epochs = config["train"]["epochs"]
    warmup_epochs = config["train"].get("warmup_epochs", 0)

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / max(1, warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: SegmentationLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    accum_steps: int,
    grad_clip: float,
    ema: EMA | None = None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    metrics = RunningMetrics()
    progress = tqdm(loader, leave=False, disable=not is_main())
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast_context(device, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks) / accum_steps

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                if step % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
            else:
                loss.backward()
                if step % accum_steps == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)

        running_loss += loss.item() * accum_steps
        metrics.update(logits.detach(), masks.detach())
        progress.set_description(f"{'train' if training else 'eval'} loss={running_loss / step:.4f}")

    if training and len(loader) % accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    if is_distributed():
        conf = torch.from_numpy(metrics.confusion).to(device)
        all_reduce_sum(conf)
        metrics.confusion = conf.cpu().numpy()
    summary = metrics.summary()
    loss_tensor = torch.tensor(running_loss / max(1, len(loader)), device=device)
    all_reduce_mean(loss_tensor)
    summary["loss"] = float(loss_tensor.item())
    return summary["loss"], summary


def save_history_plots(history: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_mean_iou"] for row in history], label="mIoU")
    axes[1].plot(epochs, [row["val_mean_dice"] for row in history], label="Dice")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curves.png", dpi=200)
    plt.close(fig)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
) -> dict[str, Any]:
    _, summary = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        scaler=None,
        accum_steps=1,
        grad_clip=0.0,
    )
    return summary


def fit(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    criterion: SegmentationLoss,
    device: torch.device,
    run_dir: str | Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    run_dir = ensure_dir(run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    log_dir = ensure_dir(run_dir / "logs")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    train_cfg = config["train"]
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    accum_steps = int(train_cfg.get("accum_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    min_epochs_before_stopping = int(train_cfg.get("min_epochs_before_stopping", train_cfg["epochs"]))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    use_ema = bool(train_cfg.get("use_ema", False))
    ema = EMA(model, train_cfg.get("ema_decay", 0.999)) if use_ema else None

    history = []
    best_metric = -float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    total_start = time.perf_counter()

    epoch_bar = tqdm(range(1, train_cfg["epochs"] + 1), desc="epochs", unit="ep", disable=not is_main())
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        train_sampler = loaders["train"].sampler
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_loss, train_summary = run_epoch(model, loaders["train"], criterion, optimizer, device, scaler, accum_steps, grad_clip, ema=ema)
        if ema is not None:
            eval_model = deepcopy(_unwrap(model))
            ema.copy_to(eval_model)
            eval_model.to(device)
        else:
            eval_model = model
        val_summary = evaluate_loader(eval_model, loaders["val_internal"], criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_loss": train_loss,
            "train_mean_iou": train_summary["mean_iou"],
            "train_mean_dice": train_summary["mean_dice"],
            "val_loss": val_summary["loss"],
            "val_mean_iou": val_summary["mean_iou"],
            "val_mean_dice": val_summary["mean_dice"],
            "val_pixel_accuracy": val_summary["pixel_accuracy"],
            "lr": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)
        current_metric = val_summary["mean_iou"]
        previous_best_metric = best_metric
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            if is_main():
                best_state = deepcopy(_unwrap(eval_model).state_dict())
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": best_state,
                        "config": config,
                        "val_summary": val_summary,
                    },
                    ckpt_dir / "best.pt",
                )
        if current_metric > previous_best_metric + min_delta:
            bad_epochs = 0
        else:
            bad_epochs += 1

        if is_main():
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": _unwrap(model).state_dict(),
                    "config": config,
                    "val_summary": val_summary,
                },
                ckpt_dir / "last.pt",
            )
        if ema is not None:
            del eval_model
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            val_loss=f"{val_summary['loss']:.3f}",
            val_miou=f"{current_metric:.3f}",
            best_miou=f"{best_metric:.3f}",
            best_ep=best_epoch,
        )
        if epoch >= min_epochs_before_stopping and bad_epochs >= patience:
            break
    epoch_bar.close()

    summary = {
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_metric,
        "history_length": len(history),
        "total_train_seconds": time.perf_counter() - total_start,
        "avg_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])) if history else 0.0,
    }
    if is_main():
        save_csv(log_dir / "history.csv", history)
        save_history_plots(history, log_dir)
        save_json(log_dir / "summary.json", summary)
    return ckpt_dir / "best.pt", history, summary

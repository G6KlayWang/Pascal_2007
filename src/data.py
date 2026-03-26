from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F
from tqdm import tqdm

from src.constants import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, VOC_CLASSES
from src.utils import ensure_dir, save_csv, save_json


def maybe_download_voc(root: str | Path, download: bool) -> Path:
    root = Path(root)
    voc_root = root / "VOCdevkit" / "VOC2007"
    if voc_root.exists():
        return voc_root
    if not download:
        raise FileNotFoundError(
            f"Pascal VOC 2007 not found under {voc_root}. Set dataset.download=true or place the dataset there."
        )
    VOCSegmentation(root=str(root), year="2007", image_set="train", download=True)
    VOCSegmentation(root=str(root), year="2007", image_set="val", download=True)
    return voc_root


def get_split_ids(voc_root: str | Path, split: str) -> list[str]:
    split_file = Path(voc_root) / "ImageSets" / "Segmentation" / f"{split}.txt"
    with split_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_internal_split(
    voc_root: str | Path,
    output_dir: str | Path,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    voc_root = Path(voc_root)
    output_dir = ensure_dir(output_dir)
    manifest_path = output_dir / "splits.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    train_ids = get_split_ids(voc_root, "train")
    test_ids = get_split_ids(voc_root, "val")
    rng = random.Random(seed)
    shuffled = train_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    internal_val_ids = sorted(shuffled[:val_count])
    internal_train_ids = sorted(shuffled[val_count:])
    manifest = {
        "train": internal_train_ids,
        "val_internal": internal_val_ids,
        "test": test_ids,
    }
    save_json(manifest_path, manifest)
    return manifest


class SegmentationTransform:
    def __init__(self, image_size: int, augment: bool) -> None:
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        image = F.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=Image.NEAREST)
        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._resize(image, mask)
        if self.augment:
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < 0.8:
                i, j, h, w = torch.randint(0, max(1, self.image_size // 8), (4,)).tolist()
                crop_h = max(self.image_size - h, int(self.image_size * 0.8))
                crop_w = max(self.image_size - w, int(self.image_size * 0.8))
                i = min(i, self.image_size - crop_h)
                j = min(j, self.image_size - crop_w)
                image = F.resized_crop(image, i, j, crop_h, crop_w, [self.image_size, self.image_size], antialias=True)
                mask = F.resized_crop(
                    mask,
                    i,
                    j,
                    crop_h,
                    crop_w,
                    [self.image_size, self.image_size],
                    interpolation=Image.NEAREST,
                )
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
                mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST, fill=IGNORE_INDEX)
            if random.random() < 0.8:
                image = self.color_jitter(image)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor


class PascalVOCDataset(Dataset):
    def __init__(
        self,
        voc_root: str | Path,
        ids: list[str],
        image_size: int,
        augment: bool,
    ) -> None:
        self.voc_root = Path(voc_root)
        self.ids = ids
        self.transform = SegmentationTransform(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.ids[index]
        image = Image.open(self.voc_root / "JPEGImages" / f"{sample_id}.jpg").convert("RGB")
        mask = Image.open(self.voc_root / "SegmentationClass" / f"{sample_id}.png")
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": sample_id,
        }


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], Path, dict[str, list[str]]]:
    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    voc_root = maybe_download_voc(dataset_cfg["root"], dataset_cfg.get("download", False))
    split_manifest = build_internal_split(
        voc_root=voc_root,
        output_dir=Path(config["experiment"]["output_root"]) / "data_overview",
        val_ratio=dataset_cfg["internal_val_ratio"],
        seed=dataset_cfg["split_seed"],
    )

    batch_size = train_cfg["batch_size"]
    eval_batch_size = config["eval"].get("batch_size", batch_size)
    num_workers = dataset_cfg.get("num_workers", 4)
    image_size = dataset_cfg["image_size"]

    train_ds = PascalVOCDataset(voc_root, split_manifest["train"], image_size=image_size, augment=config["augmentation"]["enabled"])
    val_ds = PascalVOCDataset(voc_root, split_manifest["val_internal"], image_size=image_size, augment=False)
    test_ds = PascalVOCDataset(voc_root, split_manifest["test"], image_size=image_size, augment=False)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val_internal": DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, voc_root, split_manifest


def build_data_overview(voc_root: str | Path, split_manifest: dict[str, list[str]], output_dir: str | Path, sample_count: int = 8) -> None:
    output_dir = ensure_dir(output_dir)
    rows = []
    class_pixels = Counter()
    train_ids = split_manifest["train"]
    fig, axes = plt.subplots(min(sample_count, len(train_ids)), 2, figsize=(8, max(4, 3 * min(sample_count, len(train_ids)))))
    if min(sample_count, len(train_ids)) == 1:
        axes = np.array([axes])

    for idx, sample_id in enumerate(train_ids[:sample_count]):
        image = Image.open(Path(voc_root) / "JPEGImages" / f"{sample_id}.jpg").convert("RGB")
        mask = Image.open(Path(voc_root) / "SegmentationClass" / f"{sample_id}.png")
        mask_np = np.array(mask, dtype=np.int64)
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"{sample_id} image")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(mask_np, cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1)
        axes[idx, 1].set_title(f"{sample_id} mask")
        axes[idx, 1].axis("off")

    for sample_id in tqdm(train_ids, desc="Computing class frequencies", leave=False):
        mask = Image.open(Path(voc_root) / "SegmentationClass" / f"{sample_id}.png")
        mask_np = np.array(mask, dtype=np.int64)
        valid = mask_np[mask_np != IGNORE_INDEX]
        class_pixels.update(valid.tolist())

    for split_name, ids in split_manifest.items():
        rows.append({"split": split_name, "count": len(ids)})

    plt.tight_layout()
    plt.savefig(output_dir / "sample_mosaic.png", dpi=200)
    plt.close(fig)

    pixel_rows = [{"class_id": idx, "class_name": VOC_CLASSES[idx], "pixels": int(class_pixels.get(idx, 0))} for idx in range(NUM_CLASSES)]
    save_csv(output_dir / "split_counts.csv", rows)
    save_csv(output_dir / "class_pixel_counts.csv", pixel_rows)
    save_json(
        output_dir / "dataset_summary.json",
        {
            "voc_root": str(voc_root),
            "classes": VOC_CLASSES,
            "split_sizes": {key: len(value) for key, value in split_manifest.items()},
        },
    )
    with (output_dir / "overview.md").open("w", encoding="utf-8") as handle:
        handle.write("# Pascal VOC 2007 Overview\n\n")
        handle.write(f"- Dataset root: `{voc_root}`\n")
        for split_name, ids in split_manifest.items():
            handle.write(f"- {split_name}: {len(ids)} samples\n")
        handle.write("\n")
        handle.write("The internal validation split is derived from the official training split with a fixed random seed.\n")

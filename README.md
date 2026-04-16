# Semantic Segmentation on Pascal VOC 2007

Three architectures. One benchmark. A clear winner.

<p align="center">
  <img src="outputs/eval/sam2/sam2-default/prediction_mosaic.png" width="480"/>
  <br>
  <em>SAM2 with LoRA — 0.63 mIoU, 5.1M trainable parameters</em>
</p>

## Results

| Model | mIoU | Dice | Pixel Acc | HD95 | Trainable Params |
|:------|-----:|-----:|----------:|-----:|-----------------:|
| U-Net (ResNet50) | 0.121 | 0.167 | 0.801 | 190.9 | 40.8M |
| DeepLabV3 (ResNet50) | 0.483 | 0.620 | 0.884 | 94.4 | 39.6M |
| **SAM2 (LoRA + FPN)** | **0.630** | **0.742** | **0.927** | **63.2** | **5.1M** |

SAM2 achieves the best performance on every metric with 8x fewer parameters and the fastest training time. The frozen Hiera-Large backbone, adapted with rank-16 LoRA, provides pretrained representations that make conventional architectures obsolete on small datasets.

## Architecture

**U-Net** — ResNet50 encoder, 4-stage decoder with skip connections, bilinear upsampling. Fully trained end-to-end.

**DeepLabV3** — ResNet50 backbone with atrous convolutions and ASPP for multi-scale context. Torchvision implementation.

**SAM2** — Frozen Hiera-Large encoder from SAM2, LoRA injected into all linear projections (qkv, proj, fc1, fc2), lightweight FPN-UNet decoder with GroupNorm, dropout, and DropPath. Only the decoder and LoRA weights are trained.

## Ablations

| Experiment | Variable | mIoU | Finding |
|:-----------|:---------|-----:|:--------|
| Encoder depth | ResNet18 vs 50 (U-Net) | 0.147 vs 0.131 | Deeper hurts in low-data regimes |
| Augmentation | Off vs on (SAM2) | 0.650 vs 0.652 | Foundation models already encode invariance |
| Loss function | CE vs Dice (SAM2) | 0.634 vs 0.505 | Pure Dice degrades rare-class performance |

## Quick Start

```bash
# Setup
bash scripts/00_setup_env.sh
bash scripts/01_download_prepare.sh

# Train all three models
bash scripts/02_train_unet.sh
bash scripts/03_train_deeplabv3.sh
bash scripts/04_train_sam.sh

# Evaluate
bash scripts/05_evaluate_all.sh

# Run ablations
bash scripts/11_ablation_pipeline.sh
```

## Project Structure

```
configs/          Model configs (unet.yaml, deeplabv3.yaml, sam2.yaml)
src/              Training engine, models, losses, metrics, data loading
scripts/          Numbered pipeline scripts — run in order
outputs/
  train_logs/     Training curves, checkpoints, configs per run
  eval/           Metrics, confusion matrices, mosaics, best/worst cases
  ablations/      Ablation summaries
  comparisons/    Cross-model comparison tables
report/           LaTeX report and compiled PDF
```

## Key Takeaway

Foundation-model adaptation (SAM2 + LoRA) outperforms fully trained CNNs while updating 87% fewer parameters. On small medical-scale datasets, pretrained representations matter more than architecture or training tricks.

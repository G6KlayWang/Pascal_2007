from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

from src.constants import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str, pretrained: bool) -> None:
        super().__init__()
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = resnet50(weights=weights)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported U-Net encoder: {backbone}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class ResNetUNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)
        ch = self.encoder.channels
        self.center = ConvBlock(ch[4], 512)
        self.up3 = UpBlock(512, ch[3], 256)
        self.up2 = UpBlock(256, ch[2], 128)
        self.up1 = UpBlock(128, ch[1], 64)
        self.up0 = UpBlock(64, ch[0], 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.center(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        x = self.head(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class DeepLabWrapper(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights, weights_backbone=weights_backbone, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


class SemanticDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.decoder(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class SAM2SemanticSeg(nn.Module):
    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. Install dependencies from requirements.txt and download the SAM2 checkpoint."
            ) from exc

        self.backbone = build_sam2(sam2_config, sam2_checkpoint, device=device)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        dummy = torch.zeros(1, 3, 256, 256, device=device)
        with torch.no_grad():
            features = self._extract_features(self.backbone.forward_image(dummy))
        self.decoder = SemanticDecoder(features.shape[1], num_classes=num_classes)

    def _extract_features(self, backbone_output: Any) -> torch.Tensor:
        if isinstance(backbone_output, torch.Tensor):
            if backbone_output.ndim == 4:
                return backbone_output
            raise ValueError("SAM2 backbone output tensor must be 4D.")
        if isinstance(backbone_output, dict):
            candidates = [self._extract_features(value) for value in backbone_output.values() if value is not None]
            candidates = [item for item in candidates if isinstance(item, torch.Tensor)]
            return max(candidates, key=lambda item: item.shape[1] * item.shape[2] * item.shape[3])
        if isinstance(backbone_output, (list, tuple)):
            candidates = [self._extract_features(value) for value in backbone_output if value is not None]
            candidates = [item for item in candidates if isinstance(item, torch.Tensor)]
            return max(candidates, key=lambda item: item.shape[1] * item.shape[2] * item.shape[3])
        raise TypeError(f"Unsupported SAM2 feature container: {type(backbone_output)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(self.backbone.forward_image(x))
        return self.decoder(features, output_size=x.shape[-2:])


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    if model_type == "unet":
        return ResNetUNet(backbone=model_cfg["encoder"], pretrained=model_cfg.get("pretrained", True)).to(device)
    if model_type == "deeplabv3":
        return DeepLabWrapper(pretrained=model_cfg.get("pretrained", True)).to(device)
    if model_type == "sam2":
        return SAM2SemanticSeg(
            sam2_config=model_cfg["sam2_config"],
            sam2_checkpoint=model_cfg["sam2_checkpoint"],
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            device=str(device),
        ).to(device)
    raise ValueError(f"Unknown model type: {model_type}")

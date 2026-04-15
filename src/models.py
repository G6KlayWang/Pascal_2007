from __future__ import annotations

from pathlib import Path
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


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling


def apply_lora_to_module(
    module: nn.Module,
    target_suffixes: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> int:
    replaced = 0
    for parent_name, parent in module.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name in target_suffixes:
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced += 1
    return replaced


class FPNUNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        num_classes: int,
        fpn_channels: int = 256,
        dropout_p: float = 0.1,
        drop_path_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False) for ch in in_channels_list]
        )
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, fpn_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ]
        )
        self.drop_path = DropPath(drop_path_p)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        laterals = [lat(feat) for lat, feat in zip(self.lateral, features)]
        x = self.drop_path(self.fuse[-1](laterals[-1]))
        x = self.dropout(x)
        for idx in range(len(laterals) - 2, -1, -1):
            skip = laterals[idx]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
            x = self.drop_path(self.fuse[idx](x))
            x = self.dropout(x)
        x = self.head(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class SAM2SemanticSeg(nn.Module):
    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        unfreeze_patterns: list[str] | None = None,
        device: str = "cpu",
        lora: dict[str, Any] | None = None,
        decoder_dropout: float = 0.1,
        decoder_drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. Install dependencies from requirements.txt and download the SAM2 checkpoint."
            ) from exc

        ckpt_path = Path(sam2_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path.resolve()}")

        config_path = Path(sam2_config)
        if config_path.exists():
            GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=str(config_path.resolve().parent)):
                self.backbone = build_sam2(config_path.name, sam2_checkpoint, device=device)
            GlobalHydra.instance().clear()
        else:
            self.backbone = build_sam2(sam2_config, sam2_checkpoint, device=device)

        self._verify_checkpoint_loaded(ckpt_path)
        self._configure_backbone_training(
            freeze_backbone=freeze_backbone,
            unfreeze_patterns=unfreeze_patterns or [],
        )

        lora_cfg = lora or {}
        if lora_cfg.get("enabled", False):
            replaced = apply_lora_to_module(
                self.backbone,
                target_suffixes=list(lora_cfg.get("target_suffixes", ["qkv", "proj"])),
                rank=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16)),
                dropout=float(lora_cfg.get("dropout", 0.0)),
            )
            print(f"[SAM2] applied LoRA to {replaced} linear layers "
                  f"(rank={lora_cfg.get('rank', 8)}, alpha={lora_cfg.get('alpha', 16)})")

        dummy = torch.zeros(1, 3, 1024, 1024, device=device)
        with torch.no_grad():
            pyramid = self._extract_pyramid(self.backbone.forward_image(dummy))
        for idx, feat in enumerate(pyramid):
            print(f"[SAM2] pyramid level {idx} shape: {tuple(feat.shape)}")
        in_channels_list = [feat.shape[1] for feat in pyramid]
        self.decoder = FPNUNetDecoder(
            in_channels_list,
            num_classes=num_classes,
            dropout_p=decoder_dropout,
            drop_path_p=decoder_drop_path,
        )

    def _configure_backbone_training(self, freeze_backbone: bool, unfreeze_patterns: list[str]) -> None:
        if not freeze_backbone:
            return

        for param in self.backbone.parameters():
            param.requires_grad = False

        if not unfreeze_patterns:
            return

        matched = set()
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(pattern) for pattern in unfreeze_patterns):
                param.requires_grad = True
                matched.add(name.split(".", 1)[0])

        if matched:
            print(f"[SAM2] partially unfroze backbone parameters matching: {unfreeze_patterns}")
        else:
            print(f"[SAM2] no backbone parameters matched unfreeze_patterns={unfreeze_patterns}")

    def _verify_checkpoint_loaded(self, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ckpt_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model_state = self.backbone.state_dict()
        matched, mismatched, missing = 0, [], []
        for key, ckpt_tensor in ckpt_state.items():
            if key not in model_state:
                missing.append(key)
                continue
            model_tensor = model_state[key]
            if model_tensor.shape != ckpt_tensor.shape:
                mismatched.append((key, tuple(model_tensor.shape), tuple(ckpt_tensor.shape)))
                continue
            if torch.allclose(model_tensor.float().cpu(), ckpt_tensor.float().cpu(), atol=1e-6):
                matched += 1
            else:
                mismatched.append((key, "value mismatch", None))
        total_ckpt = len(ckpt_state)
        print(
            f"[SAM2] checkpoint load check: matched {matched}/{total_ckpt} tensors "
            f"(missing={len(missing)}, mismatched={len(mismatched)})"
        )
        if mismatched[:3]:
            print(f"[SAM2] first mismatches: {mismatched[:3]}")
        if missing[:3]:
            print(f"[SAM2] first missing (ckpt→model): {missing[:3]}")

    def _extract_pyramid(self, backbone_output: Any) -> list[torch.Tensor]:
        if isinstance(backbone_output, dict):
            for preferred_key in ("backbone_fpn", "vision_features"):
                if preferred_key in backbone_output and backbone_output[preferred_key] is not None:
                    value = backbone_output[preferred_key]
                    if isinstance(value, (list, tuple)):
                        tensors = [item for item in value if isinstance(item, torch.Tensor) and item.ndim == 4]
                        if tensors:
                            return sorted(tensors, key=lambda t: -t.shape[2] * t.shape[3])
                    if isinstance(value, torch.Tensor) and value.ndim == 4:
                        return [value]
        feat = self._extract_features(backbone_output)
        return [feat]

    def _extract_features(self, backbone_output: Any) -> torch.Tensor:
        if isinstance(backbone_output, torch.Tensor):
            if backbone_output.ndim == 4:
                return backbone_output
            raise ValueError("SAM2 backbone output tensor must be 4D.")
        if isinstance(backbone_output, dict):
            for preferred_key in ("backbone_fpn", "vision_features"):
                if preferred_key in backbone_output and backbone_output[preferred_key] is not None:
                    return self._extract_features(backbone_output[preferred_key])
            candidates = [
                self._extract_features(value)
                for key, value in backbone_output.items()
                if value is not None and "pos" not in key.lower()
            ]
            candidates = [item for item in candidates if isinstance(item, torch.Tensor)]
            return min(candidates, key=lambda item: item.shape[2] * item.shape[3])
        if isinstance(backbone_output, (list, tuple)):
            candidates = [self._extract_features(value) for value in backbone_output if value is not None]
            candidates = [item for item in candidates if isinstance(item, torch.Tensor)]
            return min(candidates, key=lambda item: item.shape[2] * item.shape[3])
        raise TypeError(f"Unsupported SAM2 feature container: {type(backbone_output)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = self._extract_pyramid(self.backbone.forward_image(x))
        return self.decoder(pyramid, output_size=x.shape[-2:])


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
            unfreeze_patterns=model_cfg.get("unfreeze_patterns", []),
            device=str(device),
            lora=model_cfg.get("lora"),
            decoder_dropout=float(model_cfg.get("decoder_dropout", 0.1)),
            decoder_drop_path=float(model_cfg.get("decoder_drop_path", 0.0)),
        ).to(device)
    raise ValueError(f"Unknown model type: {model_type}")

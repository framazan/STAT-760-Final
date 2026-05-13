"""
models.py — All model architectures for wildfire detection.

Contains:
  • SimpleCNN      – original baseline (kept for reference)
  • FireNet        – custom CNN with SE-attention and residual connections
  • build_model()  – factory that also handles timm transfer-learning models

Design rationale is documented in the companion LaTeX report.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Stochastic depth per-sample (aka DropPath).

    Scales the residual branch by a binary mask with keep probability (1-drop_prob).
    Implementation follows the common per-sample DropPath used in ResNet/ViT variants.
    """
    if drop_prob <= 0.0 or not x.requires_grad:
        return x
    keep_prob = 1.0 - drop_prob
    # shape = (batch, 1, 1, 1) to broadcast across spatial dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_tensor = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_tensor

# ═════════════════════════════════════════════════════════════════════════
#  SimpleCNN  (original baseline — kept intact for comparison)
# ═════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Original student-authored baseline CNN (~3 M params)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=4, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 128, kernel_size=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ═════════════════════════════════════════════════════════════════════════
#  SE and Residual Blocks + FireNet
# ═════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.act = nn.GELU()
        self.drop_prob = float(drop_prob)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.drop_prob > 0.0 and self.training:
            out = drop_path(out, self.drop_prob)
        return self.act(out + identity)


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1, drop_probs=None):
    """Create a sequence of ResidualSEBlocks. Optionally accepts a list of drop_probs
    (one per block) to apply stochastic depth progressively.
    """
    if drop_probs is None:
        drop_probs = [0.0] * n_blocks
    layers = [ResidualSEBlock(in_ch, out_ch, stride=stride, drop_prob=drop_probs[0])]
    for i in range(1, n_blocks):
        layers.append(ResidualSEBlock(out_ch, out_ch, stride=1, drop_prob=drop_probs[i]))
    return nn.Sequential(*layers)


class FireNet(nn.Module):
    """Custom CNN for wildfire detection (~8–10 M params)."""

    def __init__(self, num_classes: int = C.NUM_CLASSES, dropout: float = C.DROPOUT):
        super().__init__()
        # Patchify stem — ConvNeXt-style: single 4×4 stride-4 conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Slightly deeper stage configuration (more blocks) and per-block
        # stochastic depth schedule controlled by C.STOCHASTIC_DEPTH
        blocks_per_stage = [2, 3, 5, 3]
        total_blocks = sum(blocks_per_stage)
        max_drop = float(getattr(C, "STOCHASTIC_DEPTH", 0.0))
        # linear schedule from 0 -> max_drop across blocks
        drop_probs_all = [max_drop * i / max(1, total_blocks - 1) for i in range(total_blocks)]

        idx = 0
        s1 = blocks_per_stage[0]
        s2 = blocks_per_stage[1]
        s3 = blocks_per_stage[2]
        s4 = blocks_per_stage[3]

        self.stage1 = _make_stage(64, 64, n_blocks=s1, stride=1,
                                  drop_probs=drop_probs_all[idx: idx + s1])
        idx += s1
        self.stage2 = _make_stage(64, 128, n_blocks=s2, stride=2,
                                  drop_probs=drop_probs_all[idx: idx + s2])
        idx += s2
        self.stage3 = _make_stage(128, 256, n_blocks=s3, stride=2,
                                  drop_probs=drop_probs_all[idx: idx + s3])
        idx += s3
        self.stage4 = _make_stage(256, 512, n_blocks=s4, stride=2,
                                  drop_probs=drop_probs_all[idx: idx + s4])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════
#  Model factory
# ═════════════════════════════════════════════════════════════════════════


def build_model(name: str, num_classes: int = C.NUM_CLASSES, pretrained: bool = True, dropout: float = C.DROPOUT) -> nn.Module:
    """Instantiate a model by name.

    Supported names:
        simple      – SimpleCNN (baseline)
        firenet     – custom CNN with SE + residuals
        convnext    – ConvNeXt-Tiny (timm, pretrained)
        vit         – ViT-Base/16 (timm, pretrained)
    """
    name = name.lower().strip()

    if name == "simple":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    if name == "firenet":
        return FireNet(num_classes=num_classes, dropout=dropout)

    # ── timm transfer-learning models ────────────────────────────────
    import timm

    if name == "convnext":
        return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)

    if name == "vit":
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)

    raise ValueError(f"Unknown model name: {name!r}")

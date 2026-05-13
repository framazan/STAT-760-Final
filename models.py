import config as C
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0.0 or not x.requires_grad:
        return x
    keep_prob = 1.0 - drop_prob
    # shape = (batch, 1, 1, 1) to broadcast across spatial dimensions
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_tensor = torch.floor(random_tensor)
    return x.div(keep_prob) * binary_tensor

class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
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
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

class SEBlock(nn.Module):
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

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.drop_prob > 0.0 and self.training:
            out = drop_path(out, self.drop_prob)
        return self.act(out + identity)


def create_block(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1, drop_probs=None):
    # Create a sequence of ResidualSEBlocks, with drop path probabilities
    if drop_probs is None:
        drop_probs = [0.0] * n_blocks
    layers = [ResidualSEBlock(in_ch, out_ch, stride=stride, drop_prob=drop_probs[0])]
    for i in range(1, n_blocks):
        layers.append(ResidualSEBlock(out_ch, out_ch, stride=1, drop_prob=drop_probs[i]))
    return nn.Sequential(*layers)


class AdvancedCNN(nn.Module):
    def __init__(self, dropout: float = C.DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        blocks_per_stage = [2, 3, 5, 3]
        total_blocks = sum(blocks_per_stage)
        max_drop = float(getattr(C, "STOCHASTIC_DEPTH", 0.0))
        # linear from 0 to max_drop across blocks
        drop_probs_all = [max_drop * i / max(1, total_blocks - 1) for i in range(total_blocks)]

        idx = 0
        s1 = blocks_per_stage[0]
        s2 = blocks_per_stage[1]
        s3 = blocks_per_stage[2]
        s4 = blocks_per_stage[3]

        self.block1 = create_block(64, 64, n_blocks=s1, stride=1, drop_probs=drop_probs_all[idx: idx + s1])
        idx += s1
        self.block2 = create_block(64, 128, n_blocks=s2, stride=2, drop_probs=drop_probs_all[idx: idx + s2])
        idx += s2
        self.block3 = create_block(128, 256, n_blocks=s3, stride=2, drop_probs=drop_probs_all[idx: idx + s3])
        idx += s3
        self.block4 = create_block(256, 512, n_blocks=s4, stride=2, drop_probs=drop_probs_all[idx: idx + s4])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.classifier(x)

def build_model(name: str, pretrained: bool = True, dropout: float = C.DROPOUT) -> nn.Module:
    # just instantiate the model, simple, advancedcnn, convnext, or vit
    name = name.lower().strip()
    if name == "simple":
        return SimpleCNN(dropout=dropout)
    if name == "advancedcnn":
        return AdvancedCNN(dropout=dropout)
    import timm
    if name == "convnext":
        return timm.create_model("convnext_tiny", pretrained=pretrained)
    if name == "vit":
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained)

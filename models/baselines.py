"""
IEEE Benchmark Baseline Models for 12-Lead ECG Multi-Label MI Localization.
All models share identical input/output signature: (B, 12, 180) -> (B, 5) logits.

Included Baselines for Pareto & Publication Comparison (Params / MACs / AUC / F1):
1. MLPTinyBaseline:   Compact Tokenizer + Standard 2-Layer ReLU MLP (~22K params) — contrasts MLP vs Chebyshev KAN.
2. ResNet1DBaseline:  1D Residual Network inspired by Strodthoff et al. (IEEE JBHI / PTB-XL benchmark, ~490K params).
3. InceptionTime1D:   Multi-scale Inception-1D Network inspired by Fawaz et al. (DAMI 2020, ~385K params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Conv1DTokenizer


class MLPTinyBaseline(nn.Module):
    """
    Same 1D-CNN Tokenizer (6,872 params) paired with standard linear + ReLU MLP head
    (64 -> 128 -> 64 -> 5) instead of Chebyshev KAN, isolating the representational
    advantage of orthogonal degree-4 polynomials under extreme parameter budgets.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 5):
        super().__init__()
        self.tokenizer = Conv1DTokenizer(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tokenizer(x)
        return self.classifier(z)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet1DBaseline(nn.Module):
    """
    Standard 4-Stage 1D ResNet benchmark (~490K params) with Global Average + Max Pooling.
    Too large for <12K param / <6KB activation SRAM MCU constraints; establishes cloud/heavy accuracy reference.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 5, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=False),
        )
        self.stage1 = nn.Sequential(ResBlock1D(base_ch, base_ch, stride=1), ResBlock1D(base_ch, base_ch, stride=2))
        self.stage2 = nn.Sequential(ResBlock1D(base_ch, base_ch * 2, stride=2), ResBlock1D(base_ch * 2, base_ch * 2, stride=1))
        self.stage3 = nn.Sequential(ResBlock1D(base_ch * 2, base_ch * 4, stride=2), ResBlock1D(base_ch * 4, base_ch * 4, stride=1))
        self.head = nn.Linear(base_ch * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        avg_p = h.mean(dim=-1)
        max_p = h.amax(dim=-1)
        return self.head(torch.cat([avg_p, max_p], dim=-1))

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, nb_filters: int = 32, bottleneck_ch: int = 32):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
            if in_ch > 1
            else nn.Identity()
        )
        b_out = bottleneck_ch if in_ch > 1 else in_ch
        self.conv_k9 = nn.Conv1d(b_out, nb_filters, kernel_size=9, padding=4, bias=False)
        self.conv_k19 = nn.Conv1d(b_out, nb_filters, kernel_size=19, padding=9, bias=False)
        self.conv_k39 = nn.Conv1d(b_out, nb_filters, kernel_size=39, padding=19, bias=False)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, nb_filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self.bottleneck(x)
        out = torch.cat([
            self.conv_k9(xb),
            self.conv_k19(xb),
            self.conv_k39(xb),
            self.maxpool_conv(x),
        ], dim=1)
        return F.relu(self.bn(out))


class InceptionTime1D(nn.Module):
    """
    Multi-Scale InceptionTime-1D architecture (6 Inception blocks with 2 residual skips)
    providing state-of-the-art large-CNN benchmark on PTB-XL.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 5, nb_filters: int = 32):
        super().__init__()
        out_ch = nb_filters * 4
        self.b1 = InceptionBlock1D(in_channels, nb_filters=nb_filters)
        self.b2 = InceptionBlock1D(out_ch, nb_filters=nb_filters)
        self.b3 = InceptionBlock1D(out_ch, nb_filters=nb_filters)
        self.skip1 = nn.Sequential(nn.Conv1d(in_channels, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))

        self.b4 = InceptionBlock1D(out_ch, nb_filters=nb_filters)
        self.b5 = InceptionBlock1D(out_ch, nb_filters=nb_filters)
        self.b6 = InceptionBlock1D(out_ch, nb_filters=nb_filters)
        self.skip2 = nn.Identity()

        self.head = nn.Linear(out_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = self.skip1(x)
        h = self.b3(self.b2(self.b1(x)))
        h = F.relu(h + res1)

        res2 = self.skip2(h)
        h = self.b6(self.b5(self.b4(h)))
        h = F.relu(h + res2)

        return self.head(h.mean(dim=-1))

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


BASELINE_REGISTRY = {
    "mlp_tiny": MLPTinyBaseline,
    "resnet1d": ResNet1DBaseline,
    "inception_time": InceptionTime1D,
}


def build_baseline(name: str, in_channels: int = 12, num_classes: int = 5) -> nn.Module:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](in_channels=in_channels, num_classes=num_classes)

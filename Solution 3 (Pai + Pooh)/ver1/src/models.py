"""
Neural Network Architectures for Thai Character Classification.

Implements:
1. CustomGlyphCNN (4-stage Conv-BN-Mish-SE designed natively for 32x32 / 64x64 glyphs).
2. AdaptedResNet18 (Stem-adapted ResNet-18 with ImageNet transfer learning).
3. AdaptedMobileNetV3 (Lightweight inverted residual architecture with SE attention).
4. Model factory function `build_model`.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention Block.
    Recalibrates channel-wise feature responses to emphasize distinctive stroke features.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_dim = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_dim, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ConvStage(nn.Module):
    """
    Standard convolutional stage: [Conv -> BN -> Mish] x N + SEBlock + MaxPool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        pool: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        layers = []
        curr_in = in_channels
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(curr_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Mish(inplace=True),
            ])
            curr_in = out_channels

        if use_se:
            layers.append(SEBlock(out_channels))

        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomGlyphCNN(nn.Module):
    """
    Custom 4-stage CNN optimized specifically for low-resolution character glyphs.
    Avoids aggressive early downsampling to preserve fine loops, clefts, and tone markers.
    """

    def __init__(self, num_classes: int = 72, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.stage1 = ConvStage(in_channels, 32, num_convs=2, pool=True, use_se=True)   # -> 16x16 (for 32x32 input)
        self.stage2 = ConvStage(32, 64, num_convs=2, pool=True, use_se=True)            # -> 8x8
        self.stage3 = ConvStage(64, 128, num_convs=2, pool=True, use_se=True)           # -> 4x4
        self.stage4 = ConvStage(128, 256, num_convs=1, pool=False, use_se=True)          # -> 4x4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


class AdaptedResNet18(nn.Module):
    """
    ResNet-18 adapted for low-resolution character classification.
    Replaces 7x7 stride-2 stem with 3x3 stride-1 convolution to retain stroke details.
    """

    def __init__(
        self,
        num_classes: int = 72,
        pretrained: bool = True,
        adapt_stem: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        if adapt_stem:
            # Replace 7x7 stride 2 conv with 3x3 stride 1 conv
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Replace initial maxpool with Identity to preserve 32x32 feature map
            base.maxpool = nn.Identity()

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )

        in_features = base.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class AdaptedMobileNetV3(nn.Module):
    """
    MobileNetV3-Small adapted for Thai character classification.
    Highly parameter-efficient with inverted residual bottlenecks and SE attention.
    """

    def __init__(
        self,
        num_classes: int = 72,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        # Replace classification head
        in_features = base.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


def build_model(
    model_name: str = "custom_cnn",
    num_classes: int = 72,
    pretrained: bool = True,
    adapt_stem: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Factory function to instantiate models by name.
    
    Supported names:
    - 'custom_cnn' / 'glyph_cnn'
    - 'resnet18' / 'adapted_resnet18'
    - 'mobilenet_v3' / 'mobilenet'
    """
    name = model_name.lower().strip()
    if name in ("custom_cnn", "glyph_cnn", "custom"):
        return CustomGlyphCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif name in ("resnet18", "adapted_resnet18", "resnet"):
        return AdaptedResNet18(
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_stem=adapt_stem,
            dropout_rate=dropout_rate,
        )
    elif name in ("mobilenet_v3", "mobilenet", "mobilenet_v3_small"):
        return AdaptedMobileNetV3(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: custom_cnn, resnet18, mobilenet_v3")

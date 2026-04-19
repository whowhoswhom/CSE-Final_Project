"""
models.py - All model definitions for the CSE-547 final project.

Contains:
  - 4 CNN architectures (Part 1) with verified 10x param growth
  - VGG16 transfer learning builder (Part 3)
  - Convolutional AutoEncoder (Part 4)
  - Dense classifier head
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from collections import OrderedDict


NUM_CLASSES = 8

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: CNN Architectures
# ═══════════════════════════════════════════════════════════════════════════

def _conv_block(in_ch, out_ch, use_bn=False):
    """Conv2D(3x3, same padding) + optional BN + ReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class CNNArchA(nn.Module):
    """Micro: ~880 params. 2x Conv(8), GAP, Linear(8)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 8),
            nn.MaxPool2d(2),
            _conv_block(8, 8),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class CNNArchB(nn.Module):
    """Small: ~14K params. Conv(16)->Conv(32)->Conv(32), GAP, Linear(8)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 16),
            nn.MaxPool2d(2),
            _conv_block(16, 32),
            nn.MaxPool2d(2),
            _conv_block(32, 32),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class CNNArchC(nn.Module):
    """Medium: ~148K params. 2xConv(32), 2xConv(64), Conv(128), GAP, Linear(64)->Linear(8)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 32),
            nn.MaxPool2d(2),
            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),
            _conv_block(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class CNNArchD(nn.Module):
    """Large: ~2.5M params. Deep with BN: Conv(64)x2, Conv(128)x2, Conv(256)x2, Conv(512), GAP, Linear(256)->Linear(8)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            _conv_block(64, 128),
            _conv_block(128, 128),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            _conv_block(256, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# Factory with optional regularization for Part 2
def build_cnn(arch: str, num_classes=NUM_CLASSES, dropout=0.0, weight_decay=0.0):
    """
    Build a CNN architecture by name. dropout and weight_decay are applied
    externally (dropout via wrapper, weight_decay via optimizer).
    Returns (model, param_count).
    """
    constructors = {"A": CNNArchA, "B": CNNArchB, "C": CNNArchC, "D": CNNArchD}
    if arch not in constructors:
        raise ValueError(f"Unknown arch: {arch}. Choose from {list(constructors.keys())}")

    model = constructors[arch](num_classes)

    # Wrap with dropout if requested
    if dropout > 0:
        model = _add_dropout(model, dropout)

    param_count = sum(p.numel() for p in model.parameters())
    return model, param_count


def _add_dropout(model, rate):
    """Insert Dropout layers after each MaxPool2d in the feature extractor
    and before the classifier."""
    new_features = []
    for layer in model.features:
        new_features.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            new_features.append(nn.Dropout2d(rate))
    model.features = nn.Sequential(*new_features)

    # Add dropout before classifier
    if isinstance(model.classifier, nn.Sequential):
        new_clf = []
        for layer in model.classifier:
            if isinstance(layer, nn.Linear) and layer.out_features != NUM_CLASSES:
                new_clf.append(layer)
                new_clf.append(nn.Dropout(rate))
            else:
                new_clf.append(layer)
        model.classifier = nn.Sequential(*new_clf)

    return model


def verify_10x_rule():
    """Verify that each architecture has >= 10x the params of its predecessor."""
    archs = ["A", "B", "C", "D"]
    counts = []
    for arch in archs:
        _, count = build_cnn(arch)
        counts.append(count)
        print(f"  Arch {arch}: {count:>10,} params")

    for i in range(1, len(counts)):
        ratio = counts[i] / counts[i - 1]
        ok = "PASS" if ratio >= 10 else "FAIL"
        print(f"  {archs[i]}/{archs[i-1]} ratio: {ratio:.1f}x [{ok}]")

    return dict(zip(archs, counts))


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: VGG16 Transfer Learning
# ═══════════════════════════════════════════════════════════════════════════

# VGG16 block boundaries (semantic, not brittle indices)
VGG16_BLOCKS = {
    "block1": (0, 4),    # features[0:4]   - Conv, ReLU, Conv, ReLU
    "block2": (5, 9),    # features[5:9]
    "block3": (10, 16),  # features[10:16]
    "block4": (17, 23),  # features[17:23]
    "block5": (24, 30),  # features[24:30]
}


class VGG16Transfer(nn.Module):
    """VGG16 conv_base + 2 Dense layers for classification."""

    def __init__(self, num_classes=NUM_CLASSES, freeze_level=1):
        """
        freeze_level:
            1 = all features frozen (only classifier trains)
            2 = block5 unfrozen
            3 = blocks 4+5 unfrozen
        """
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        # Apply freezing by block boundaries
        self._apply_freeze(freeze_level)

        # Classifier: Flatten + Dense(256) + ReLU + Dense(num_classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def _apply_freeze(self, level):
        # First freeze everything
        for param in self.features.parameters():
            param.requires_grad = False

        if level >= 2:
            # Unfreeze block5
            start, end = VGG16_BLOCKS["block5"]
            for i in range(start, end + 1):
                if i < len(self.features):
                    for param in self.features[i].parameters():
                        param.requires_grad = True

        if level >= 3:
            # Also unfreeze block4
            start, end = VGG16_BLOCKS["block4"]
            for i in range(start, end + 1):
                if i < len(self.features):
                    for param in self.features[i].parameters():
                        param.requires_grad = True

    def get_param_groups(self, lr_pretrained=1e-5, lr_head=1e-3):
        """Return parameter groups with differential learning rates."""
        pretrained_params = [p for p in self.features.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())
        return [
            {"params": pretrained_params, "lr": lr_pretrained},
            {"params": head_params, "lr": lr_head},
        ]

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Convolutional AutoEncoder
# ═══════════════════════════════════════════════════════════════════════════

class ConvEncoder(nn.Module):
    """Convolutional encoder: input (3, 64, 64) -> bottleneck vector."""

    def __init__(self, filters, bottleneck_dim):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in filters:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flattened size: 64 / (2^n_layers) squared * last_filter
        spatial = 64 // (2 ** len(filters))
        self.flat_size = in_ch * spatial * spatial
        self.fc = nn.Linear(self.flat_size, bottleneck_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    """Convolutional decoder: bottleneck vector -> reconstructed (3, 64, 64)."""

    def __init__(self, filters, bottleneck_dim):
        super().__init__()
        # Reverse filters for decoder
        rev_filters = list(reversed(filters))
        spatial = 64 // (2 ** len(filters))
        self.spatial = spatial
        self.first_ch = rev_filters[0]
        self.flat_size = self.first_ch * spatial * spatial

        self.fc = nn.Linear(bottleneck_dim, self.flat_size)

        layers = []
        in_ch = rev_filters[0]
        for out_ch in rev_filters[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        # Final upsample to 3 channels
        layers.extend([
            nn.ConvTranspose2d(in_ch, 3, 2, stride=2),
            nn.Sigmoid(),
        ])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.first_ch, self.spatial, self.spatial)
        return self.deconv(x)


class ConvAutoEncoder(nn.Module):
    """Full convolutional autoencoder."""

    def __init__(self, filters, bottleneck_dim):
        super().__init__()
        self.encoder = ConvEncoder(filters, bottleneck_dim)
        self.decoder = ConvDecoder(filters, bottleneck_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


# AE configurations from the plan
AE_CONFIGS = {
    "AE1": {"filters": [16, 32], "bottleneck_dim": 64},
    "AE2": {"filters": [32, 64], "bottleneck_dim": 128},
    "AE3": {"filters": [16, 32, 64], "bottleneck_dim": 64},
    "AE4": {"filters": [32, 64, 128], "bottleneck_dim": 128},
    "AE5": {"filters": [16, 32, 64, 128], "bottleneck_dim": 256},
    "AE6": {"filters": [32, 64, 128, 256], "bottleneck_dim": 512},
}


def build_autoencoder(config_name):
    """Build a ConvAutoEncoder by config name (AE1-AE6)."""
    cfg = AE_CONFIGS[config_name]
    return ConvAutoEncoder(**cfg)


class DenseClassifier(nn.Module):
    """2-layer dense classifier on frozen encoder features."""

    def __init__(self, input_dim, num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, 128), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("Verifying CNN 10x param growth rule:")
    verify_10x_rule()

    print("\nVGG16 Transfer (freeze_level=1):")
    vgg = VGG16Transfer(freeze_level=1)
    trainable = sum(p.numel() for p in vgg.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vgg.parameters())
    print(f"  Total: {total:,}, Trainable: {trainable:,}")

    print("\nAutoEncoder configs:")
    for name, cfg in AE_CONFIGS.items():
        ae = build_autoencoder(name)
        n = sum(p.numel() for p in ae.parameters())
        enc_n = sum(p.numel() for p in ae.encoder.parameters())
        print(f"  {name}: {n:>10,} total, {enc_n:>10,} encoder, bottleneck={cfg['bottleneck_dim']}")

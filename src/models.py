"""Two hand-built CNNs for CIFAR-10.

BaselineCNN is roughly what you'd write after reading your first CNN tutorial.
CifarNet is the "properly built" version: stacked 3x3 convs, batch norm,
dropout, and global average pooling instead of a huge dense layer.
"""
import torch
from torch import nn


class BaselineCNN(nn.Module):
    """3 conv layers -> 2 dense layers. No batch norm, no dropout."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_bn(c_in, c_out, use_bn=True):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU(inplace=True))
    return layers


class CifarNet(nn.Module):
    """VGG-style: three blocks of two 3x3 convs, each block halving the resolution.

    32x32 -> 16x16 -> 8x8 -> 4x4, with 48 -> 96 -> 192 channels, then global
    average pooling. `use_bn` and `dropout` are switches so the ablation study
    can turn each idea off and measure what it was worth.
    """

    def __init__(self, num_classes=10, width=48, use_bn=True, dropout=0.3):
        super().__init__()
        w = width
        blocks = []
        for i, (c_in, c_out) in enumerate([(3, w), (w, 2 * w), (2 * w, 4 * w)]):
            blocks += conv_bn(c_in, c_out, use_bn) + conv_bn(c_out, c_out, use_bn)
            blocks.append(nn.MaxPool2d(2))
            if dropout:
                blocks.append(nn.Dropout2d(dropout * (i + 1) / 3))  # more dropout deeper in
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(4 * w, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @property
    def last_conv(self) -> nn.Conv2d:
        """The final conv layer: Grad-CAM hooks in here."""
        return [m for m in self.features if isinstance(m, nn.Conv2d)][-1]


# Named experiment configs used by train.py. The ablation is "leave one out":
# start from the full model and remove one idea at a time.
EXPERIMENTS = {
    "baseline":       dict(arch="baseline", use_bn=False, dropout=0.0, augment=False),
    "full":           dict(arch="cifarnet", use_bn=True, dropout=0.3, augment=True),
    "no-batchnorm":   dict(arch="cifarnet", use_bn=False, dropout=0.3, augment=True),
    "no-augment":     dict(arch="cifarnet", use_bn=True, dropout=0.3, augment=False),
    "no-dropout":     dict(arch="cifarnet", use_bn=True, dropout=0.0, augment=True),
}


def build_model(arch="cifarnet", use_bn=True, dropout=0.3, **_) -> nn.Module:
    if arch == "baseline":
        return BaselineCNN()
    return CifarNet(use_bn=use_bn, dropout=dropout)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_trained(path, device="cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()

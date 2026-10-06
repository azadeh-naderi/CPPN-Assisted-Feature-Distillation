"""MobileNetV2 (Sandler et al., 2018) in the CIFAR configuration used as the
cross-architecture student in the standard CIFAR-100 KD benchmark (Tian et
al., ICLR 2020, "MobileNetV2" = expansion 6, width multiplier 0.5).

CIFAR adaptation, per that benchmark: the first conv keeps stride 2, but the
second inverted-residual stage (c=24) uses stride 1 instead of ImageNet's
2, so a 32x32 input ends at 2x2 before global pooling. The final 1x1 conv
stays at 1280 channels for width multipliers <= 1.

Least certain of the four architectures to match the benchmark exactly
(activation and t=1 block details aren't pinned down by the spec this was
written from) -- the Phase 0 sanity check against the published student
accuracy (64.60%) is what confirms it.
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int, stride: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(_conv_bn_relu(in_ch, hidden, 1, 1))
        layers += [
            _conv_bn_relu(hidden, hidden, 3, stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


class CifarMobileNetV2(nn.Module):
    def __init__(self, num_classes: int, expansion: int = 6, width_mult: float = 0.5):
        super().__init__()
        setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expansion, 24, 2, 1],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]
        in_ch = int(32 * width_mult)
        layers: list[nn.Module] = [_conv_bn_relu(3, in_ch, 3, 2)]
        for t, c, n, s in setting:
            out_ch = int(c * width_mult)
            for i in range(n):
                layers.append(InvertedResidual(in_ch, out_ch, s if i == 0 else 1, t))
                in_ch = out_ch
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        layers.append(_conv_bn_relu(in_ch, self.last_channel, 1, 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.last_channel, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        features = torch.flatten(self.pool(self.features(x)), 1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

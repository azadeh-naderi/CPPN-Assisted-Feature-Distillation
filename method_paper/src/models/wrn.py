"""Wide ResNets (Zagoruyko & Komodakis, 2016) for CIFAR: pre-activation
basic blocks, depth = 6n + 4, group widths [16, 16k, 32k, 64k], no dropout
by default -- the wrn_40_2 / wrn_16_2 configurations used by the standard
CIFAR-100 KD benchmark (Tian et al., ICLR 2020)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WRNBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.equal_in_out = in_planes == out_planes
        self.shortcut = None if self.equal_in_out else nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.conv1(pre)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        # Identity shortcut takes the raw input; the projection shortcut
        # takes the pre-activated input, as in the original WRN.
        residual = x if self.equal_in_out else self.shortcut(pre)
        return out + residual


class WideResNet(nn.Module):
    def __init__(self, depth: int, widen_factor: int, num_classes: int, drop_rate: float = 0.0):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"depth must be 6n+4, got {depth}")
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        in_planes = widths[0]
        groups = []
        for out_planes, stride in zip(widths[1:], (1, 2, 2)):
            blocks = []
            for i in range(n):
                blocks.append(WRNBlock(in_planes, out_planes, stride if i == 0 else 1, drop_rate))
                in_planes = out_planes
            groups.append(nn.Sequential(*blocks))
        self.block1, self.block2, self.block3 = groups

        self.bn = nn.BatchNorm2d(widths[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        out = self.conv1(x)
        out = self.block3(self.block2(self.block1(out)))
        out = F.relu(self.bn(out))
        features = torch.flatten(self.pool(out), 1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


WRN_CONFIGS = {
    "wrn_16_1": (16, 1),
    "wrn_16_2": (16, 2),
    "wrn_40_1": (40, 1),
    "wrn_40_2": (40, 2),
}

"""CIFAR-style ResNets (He et al., 2016, Sec. 4.2), in the configurations
used by the standard CIFAR-100 KD benchmark (Tian et al., ICLR 2020 /
RepDistiller): resnet{8,14,20,32,44,56,110} with filters [16, 16, 32, 64],
and the wider resnet8x4/resnet32x4 with filters [32, 64, 128, 256].

Unlike src/models/resnet.py (torchvision's ImageNet ResNet18 with a 7x7
stride-2 stem and a maxpool, which shrinks a 32x32 input to 8x8 before the
first residual stage), the stem here is a single 3x3 stride-1 conv, so the
three stages run at 32x32, 16x16 and 8x8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # 1x1 projection shortcut (option B), matching the benchmark
            # models rather than He et al.'s parameter-free option A.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, depth: int, filters: list[int], num_classes: int):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(3, filters[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])

        in_planes = filters[0]
        stages = []
        for planes, stride in zip(filters[1:], (1, 2, 2)):
            blocks = []
            for i in range(n):
                blocks.append(BasicBlock(in_planes, planes, stride if i == 0 else 1))
                in_planes = planes
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = stages

        self.pool = nn.AdaptiveAvgPool2d(1)  # == 8x8 avg pool at 32x32 input
        self.fc = nn.Linear(filters[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        features = torch.flatten(self.pool(out), 1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


_SMALL = [16, 16, 32, 64]
_WIDE = [32, 64, 128, 256]

RESNET_CONFIGS = {
    "resnet8": (8, _SMALL),
    "resnet14": (14, _SMALL),
    "resnet20": (20, _SMALL),
    "resnet32": (32, _SMALL),
    "resnet44": (44, _SMALL),
    "resnet56": (56, _SMALL),
    "resnet110": (110, _SMALL),
    "resnet8x4": (8, _WIDE),
    "resnet32x4": (32, _WIDE),
}

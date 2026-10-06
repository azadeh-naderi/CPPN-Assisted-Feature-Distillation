"""CIFAR VGGs with batch norm, in the vgg8/vgg11/vgg13 configurations used
by the standard CIFAR-100 KD benchmark (Tian et al., ICLR 2020): five conv
groups, 2x2 max pooling after the first four (32 -> 16 -> 8 -> 4 -> 2),
global average pooling after the fifth, and a single 512 -> num_classes
linear layer.

Separate from src/models/vgg.py, which adapts torchvision's ImageNet vgg16
for the main pipeline's VGG ablation."""

import torch
import torch.nn as nn

VGG_CONFIGS = {
    "vgg8": [[64], [128], [256], [512], [512]],
    "vgg11": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "vgg13": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
}


class CifarVGG(nn.Module):
    def __init__(self, cfg: list[list[int]], num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for i, group in enumerate(cfg):
            for out_ch in group:
                layers += [nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                in_ch = out_ch
            if i < len(cfg) - 1:
                layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
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

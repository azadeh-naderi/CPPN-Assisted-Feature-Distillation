import torch.nn as nn

from src.models.lenet import LeNet
from src.models.resnet import ResNet


def build_model(name: str, input_channels: int, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name == "lenet":
        return LeNet(input_channels=input_channels, num_classes=num_classes)
    if name == "resnet18":
        return ResNet(input_channels=input_channels, pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}. Choose one of ['lenet', 'resnet18'].")

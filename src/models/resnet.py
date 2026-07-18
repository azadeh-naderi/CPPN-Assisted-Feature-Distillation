import torch.nn as nn
import torchvision.models as torch_models
from torchvision.models import ResNet18_Weights


class ResNet(nn.Module):
    """Ported from legacy/test_cppn_resnet.py. `torchvision.models.resnet18`'s
    `pretrained=` bool kwarg is deprecated as of torchvision>=0.13 in favor of
    `weights=`; switched here to avoid a warning/break on newer environments."""

    def __init__(self, input_channels: int, pretrained: bool, num_classes: int):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base_model = torch_models.resnet18(weights=weights)

        self.base_model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

        self._features: object = None
        self.base_model.avgpool.register_forward_hook(self._capture_features)

    def _capture_features(self, module, inputs, output):
        self._features = output.flatten(1)

    def forward(self, x, return_features: bool = False):
        logits = self.base_model(x)
        if return_features:
            return logits, self._features
        return logits

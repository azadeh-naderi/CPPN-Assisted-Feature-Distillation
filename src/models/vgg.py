import torch.nn as nn
import torchvision.models as torch_models
from torchvision.models import VGG16_Weights


class VGG(nn.Module):
    """torchvision's vgg16, adapted for CIFAR-sized (32x32) inputs. Chosen as
    the "different architecture" ablation specifically because it has no
    skip connections at all -- maximally different in design from ResNet18,
    unlike e.g. a wider/deeper ResNet variant.

    vgg16's own classifier head (3 large FC layers, `AdaptiveAvgPool2d((7,7))`)
    is sized for 224x224 ImageNet input, where the conv stack still has
    7x7 spatial resolution left after 5 stride-2 maxpools. A 32x32 CIFAR
    input is already reduced to 1x1 by the same 5 maxpools (32/2^5=1), so
    reusing the original head would just waste capacity upsampling 1x1 back
    to 7x7 for no benefit -- replaced with `AdaptiveAvgPool2d((1,1))` (a
    no-op at this resolution, kept only for input-size robustness) and a
    single `Linear(512, num_classes)`, following standard CIFAR-VGG
    practice. The conv `features` stack itself (where pretrained ImageNet
    weights actually live) is untouched.
    """

    def __init__(self, input_channels: int, pretrained: bool, num_classes: int):
        super().__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        base_model = torch_models.vgg16(weights=weights)
        self.features = base_model.features

        if input_channels != 3:
            first_conv = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features: bool = False):
        x = self.features(x)
        x = self.avgpool(x)
        features = x.flatten(1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

import torch.nn as nn

from method_paper.src.models.cifar_resnet import RESNET_CONFIGS, CifarResNet
from method_paper.src.models.mobilenetv2 import CifarMobileNetV2
from method_paper.src.models.vgg import VGG_CONFIGS, CifarVGG
from method_paper.src.models.wrn import WRN_CONFIGS, WideResNet

# Architectures trained with lr=0.01 instead of 0.05 in the benchmark
# protocol (Tian et al., ICLR 2020), applied to both teacher and student runs.
SMALL_LR_MODELS = {"mobilenetv2"}


def available_models() -> list[str]:
    return sorted([*RESNET_CONFIGS, *WRN_CONFIGS, *VGG_CONFIGS, "mobilenetv2"])


def build_model(name: str, num_classes: int) -> nn.Module:
    """Every model returns logits, or (logits, penultimate_features) with
    return_features=True -- the same contract src/cppn/ fitness code relies
    on for the teacher."""
    if name in RESNET_CONFIGS:
        depth, filters = RESNET_CONFIGS[name]
        return CifarResNet(depth, filters, num_classes)
    if name in WRN_CONFIGS:
        depth, widen = WRN_CONFIGS[name]
        return WideResNet(depth, widen, num_classes)
    if name in VGG_CONFIGS:
        return CifarVGG(VGG_CONFIGS[name], num_classes)
    if name == "mobilenetv2":
        return CifarMobileNetV2(num_classes)
    raise ValueError(f"Unknown model {name!r}. Choose one of {available_models()}.")

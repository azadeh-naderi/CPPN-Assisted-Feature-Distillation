import pytest
import torch

from method_paper.src.models.registry import available_models, build_model

# Parameter counts (millions, CIFAR-100 heads) published for the standard
# KD benchmark (Tian et al., ICLR 2020). Matching these within 2% is the
# main evidence our from-spec implementations equal the benchmark models.
PUBLISHED_PARAMS_M = {
    "resnet20": 0.28,
    "resnet56": 0.86,
    "resnet110": 1.73,
    "resnet8x4": 1.23,
    "resnet32x4": 7.43,
    "wrn_16_2": 0.70,
    "wrn_40_2": 2.26,
    "vgg8": 3.96,
    "vgg13": 9.46,
    "mobilenetv2": 0.81,
}


@pytest.mark.parametrize("name", sorted(PUBLISHED_PARAMS_M))
def test_param_count_matches_published(name):
    model = build_model(name, num_classes=100)
    millions = sum(p.numel() for p in model.parameters()) / 1e6
    assert millions == pytest.approx(PUBLISHED_PARAMS_M[name], rel=0.02)


@pytest.mark.parametrize("name", available_models())
def test_forward_shapes_and_features(name):
    model = build_model(name, num_classes=100).eval()
    x = torch.rand(2, 3, 32, 32)
    logits = model(x)
    logits_f, features = model(x, return_features=True)
    assert logits.shape == (2, 100)
    assert torch.allclose(logits, logits_f)
    assert features.ndim == 2 and features.shape[0] == 2


def test_resnet_stem_keeps_full_resolution():
    # The whole point of these models vs src/models/resnet.py: no stride-2
    # stem or maxpool, so the first stage sees the full 32x32 input.
    model = build_model("resnet20", num_classes=100)
    assert model.conv1.stride == (1, 1) and model.conv1.kernel_size == (3, 3)
    stem_out = model.bn1(model.conv1(torch.rand(1, 3, 32, 32)))
    assert stem_out.shape[-2:] == (32, 32)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("resnet18_imagenet", num_classes=100)

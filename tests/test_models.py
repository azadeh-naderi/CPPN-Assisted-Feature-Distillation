import pytest
import torch

from src.models.registry import build_model


@pytest.mark.parametrize(
    "name,input_channels,image_size,feature_dim",
    [
        ("lenet", 1, 28, 84),
        ("resnet18", 3, 32, 512),
        ("vgg16", 3, 32, 512),
    ],
)
def test_build_model_forward_and_features(name, input_channels, image_size, feature_dim):
    model = build_model(name, input_channels=input_channels, num_classes=10, pretrained=False)
    x = torch.rand(2, input_channels, image_size, image_size)
    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 10)
    assert features.shape == (2, feature_dim)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("not_a_real_model", input_channels=3, num_classes=10)

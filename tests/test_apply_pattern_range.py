import pytest
import torch

from src.cppn.apply import apply_pattern


def _random_batch(B=4, C=3, H=8, W=8):
    return torch.rand(B, C, H, W)


def _random_pattern(H=8, W=8, C=3):
    return torch.rand(H, W, C)


@pytest.mark.parametrize("mode", ["multiplicative", "additive"])
def test_view_stays_in_unit_range(mode):
    images = _random_batch()
    pattern = _random_pattern()
    view = apply_pattern(images, pattern, mode=mode, scale=0.5)
    assert view.min() >= 0.0 - 1e-6
    assert view.max() <= 1.0 + 1e-6
    assert view.shape == images.shape


def test_multiplicative_identity_pattern_is_noop():
    images = _random_batch()
    pattern = torch.ones(8, 8, 3)
    view = apply_pattern(images, pattern, mode="multiplicative")
    assert torch.allclose(view, images)


def test_multiplicative_blank_pattern_zeros_image():
    images = _random_batch()
    pattern = torch.zeros(8, 8, 3)
    view = apply_pattern(images, pattern, mode="multiplicative")
    assert torch.allclose(view, torch.zeros_like(images))


def test_warp_raises_not_implemented():
    images = _random_batch()
    pattern = _random_pattern()
    with pytest.raises(NotImplementedError):
        apply_pattern(images, pattern, mode="warp")


def test_unknown_mode_raises_value_error():
    images = _random_batch()
    pattern = _random_pattern()
    with pytest.raises(ValueError):
        apply_pattern(images, pattern, mode="bogus")

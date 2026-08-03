import torch

from src.distill.losses import combined_loss


def test_no_cppn_losses_is_plain_kd():
    loss_hard = torch.tensor(2.0)
    loss_soft = torch.tensor(1.0)
    result = combined_loss(loss_hard, loss_soft, [], alpha=0.9)
    assert torch.allclose(result, 0.1 * loss_hard + 0.9 * loss_soft)


def test_default_averages_soft_and_cppn_under_one_alpha():
    loss_hard = torch.tensor(2.0)
    loss_soft = torch.tensor(1.0)
    cppn_losses = [torch.tensor(3.0)]
    result = combined_loss(loss_hard, loss_soft, cppn_losses, alpha=0.9)
    expected = 0.1 * loss_hard + 0.9 * ((loss_soft + 3.0) / 2)
    assert torch.allclose(result, expected)


def test_cppn_weight_draws_from_the_same_fixed_budget_as_alpha():
    loss_hard = torch.tensor(2.0)
    loss_soft = torch.tensor(1.0)
    cppn_losses = [torch.tensor(3.0), torch.tensor(5.0)]
    result = combined_loss(loss_hard, loss_soft, cppn_losses, alpha=0.6, cppn_weight=0.3)
    # hard_weight = 1 - 0.6 - 0.3 = 0.1, so all three terms sum to 1
    expected = 0.1 * loss_hard + 0.6 * loss_soft + 0.3 * 4.0  # mean(3.0, 5.0) == 4.0
    assert torch.allclose(result, expected)


def test_cppn_weight_ignored_when_no_cppn_losses():
    loss_hard = torch.tensor(2.0)
    loss_soft = torch.tensor(1.0)
    result = combined_loss(loss_hard, loss_soft, [], alpha=0.6, cppn_weight=0.3)
    assert torch.allclose(result, 0.4 * loss_hard + 0.6 * loss_soft)

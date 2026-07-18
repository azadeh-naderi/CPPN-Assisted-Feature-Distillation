"""Vectorized torch mirrors of neat-python's `neat/activations.py` functions.

Each must replicate the original's internal scale/clamp constants exactly
(e.g. sigmoid pre-scales its input by 5.0 and clamps to [-60,60] before the
logistic function) — these are NOT the same as `torch.sigmoid`/`torch.tanh`/
`torch.sin`. `compile_genome` uses these to evaluate a genome's evolved
per-node activation over a whole coordinate grid at once; if any of these
formulas drift from neat-python's, the compiled network computes a different
function than the genome actually encodes, silently invalidating fitness.
Cross-checked against installed neat-python 2.0.0 source and covered by
`tests/test_compile_matches_neat.py`.
"""

import torch


def sigmoid_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(5.0 * z, -60.0, 60.0)
    return 1.0 / (1.0 + torch.exp(-z))


def tanh_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(2.5 * z, -60.0, 60.0)
    return torch.tanh(z)


def sin_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(5.0 * z, -60.0, 60.0)
    return torch.sin(z)


def gauss_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(z, -3.4, 3.4)
    return torch.exp(-5.0 * z**2)


def relu_activation(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z > 0.0, z, torch.zeros_like(z))


def elu_activation(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z > 0.0, z, torch.exp(z) - 1)


def lelu_activation(z: torch.Tensor) -> torch.Tensor:
    leaky = 0.005
    return torch.where(z > 0.0, z, leaky * z)


def selu_activation(z: torch.Tensor) -> torch.Tensor:
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return torch.where(z > 0.0, lam * z, lam * alpha * (torch.exp(z) - 1))


def softplus_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(5.0 * z, -60.0, 60.0)
    return 0.2 * torch.log(1 + torch.exp(z))


def identity_activation(z: torch.Tensor) -> torch.Tensor:
    return z


def clamped_activation(z: torch.Tensor) -> torch.Tensor:
    return torch.clamp(z, -1.0, 1.0)


def inv_activation(z: torch.Tensor) -> torch.Tensor:
    safe = torch.where(z == 0, torch.ones_like(z), z)
    result = 1.0 / safe
    return torch.where(z == 0, torch.zeros_like(z), result)


def log_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(z, min=1e-7)
    return torch.log(z)


def exp_activation(z: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(z, -60.0, 60.0)
    return torch.exp(z)


def abs_activation(z: torch.Tensor) -> torch.Tensor:
    return torch.abs(z)


def hat_activation(z: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1 - torch.abs(z), min=0.0)


def square_activation(z: torch.Tensor) -> torch.Tensor:
    return z**2


def cube_activation(z: torch.Tensor) -> torch.Tensor:
    return z**3


TORCH_ACTIVATIONS = {
    "sigmoid": sigmoid_activation,
    "tanh": tanh_activation,
    "sin": sin_activation,
    "gauss": gauss_activation,
    "relu": relu_activation,
    "elu": elu_activation,
    "lelu": lelu_activation,
    "selu": selu_activation,
    "softplus": softplus_activation,
    "identity": identity_activation,
    "clamped": clamped_activation,
    "inv": inv_activation,
    "log": log_activation,
    "exp": exp_activation,
    "abs": abs_activation,
    "hat": hat_activation,
    "square": square_activation,
    "cube": cube_activation,
}

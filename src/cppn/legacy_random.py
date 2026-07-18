"""Literal reproduction of legacy/test_cppn_lenet.py's `SimpleCPPN` +
`image_to_cppn_coords` — a fixed-architecture, never-trained-or-evolved MLP
that takes (flat pixel index, -log(pixel intensity)) as input. This is NOT a
coordinate-only CPPN (it depends on pixel content directly) and is kept only
as the `kd_random_cppn --random-cppn-variant legacy` baseline, to measure
against the exact thing the original prototype scripts did before this
rewrite, in addition to the fairer `coord` variant in src/cppn/evolve.py's
`create_random_genome`.
"""

import torch
import torch.nn as nn


class LegacyCPPN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(coords))
        x = torch.tanh(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def legacy_image_to_cppn_coords(images: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    B, C, H, W = images.shape
    pixels = images.permute(0, 2, 3, 1).reshape(B, H * W * C)
    y = -torch.log(pixels.clamp(min=0.0) + epsilon)
    spatial_index = torch.arange(H * W, dtype=torch.float32, device=images.device).repeat(C)
    x = spatial_index.unsqueeze(0).expand(B, -1)
    return torch.stack([x, y], dim=-1)


def make_legacy_cppn(seed: int, device: torch.device) -> LegacyCPPN:
    """A single randomly-initialized, never-trained LegacyCPPN instance,
    matching the legacy scripts' `cppn = SimpleCPPN().to(device)` created
    once outside the training loop and reused (unchanged) for every batch."""
    torch.manual_seed(seed)
    cppn = LegacyCPPN().to(device)
    cppn.eval()
    return cppn


def legacy_view(cppn: LegacyCPPN, images_raw01: torch.Tensor) -> torch.Tensor:
    """Applies a (fixed, random-weight) LegacyCPPN instance to a batch of
    raw-[0,1] images. Unlike the coordinate-only CPPNs, the output depends on
    each image's own pixel values, not just spatial position, so it cannot be
    precomputed once as a reusable pattern — recomputed every batch."""
    B, C, H, W = images_raw01.shape
    coords = legacy_image_to_cppn_coords(images_raw01)
    with torch.no_grad():
        out = cppn(coords).squeeze(-1)  # [B, H*W*C]
    return out.view(B, C, H, W)

import torch

# Fixed NEAT input order: (x, y, r, channel). Channel is always present (even
# for grayscale, where it's a constant 0) so one NEAT config architecture
# (num_inputs=4) works for both grayscale (LeNet/MNIST-family) and RGB
# (ResNet18/CIFAR-family) datasets.
NUM_COORD_FEATURES = 4


def make_coord_grid(H: int, W: int, C: int, device: torch.device | None = None) -> torch.Tensor:
    """Returns coords of shape [H*W*C, 4] = (x, y, r, channel), each in
    [-1, 1]. Channel-major layout: all H*W points for channel 0, then channel
    1, etc. — pairs with `reshape_pattern` below."""
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
    r = torch.sqrt(grid_x**2 + grid_y**2)

    per_channel = []
    for c in range(C):
        ch_value = 0.0 if C == 1 else (c / (C - 1)) * 2 - 1
        ch = torch.full_like(grid_x, ch_value)
        coords = torch.stack([grid_x, grid_y, r, ch], dim=-1).reshape(-1, 4)  # [H*W, 4]
        per_channel.append(coords)
    return torch.cat(per_channel, dim=0)  # [H*W*C, 4]


def reshape_pattern(pattern_flat: torch.Tensor, H: int, W: int, C: int) -> torch.Tensor:
    """[H*W*C, 1] (or [H*W*C]) -> [H, W, C], undoing the channel-major layout
    from `make_coord_grid`."""
    return pattern_flat.reshape(C, H, W).permute(1, 2, 0).contiguous()

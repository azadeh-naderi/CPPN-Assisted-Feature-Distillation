import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cppn.apply import apply_pattern
from src.cppn.coords import make_coord_grid, reshape_pattern
from src.data.datasets import get_probe_batch
from src.utils.logging import get_logger

log = get_logger(__name__)


class TrainableCPPN(nn.Module):
    """Gradient-trainable analogue of the evolved CPPN: same coordinate-only
    input (x, y, r, channel) and output convention (sigmoid-squashed pattern
    in [0,1]) as a compiled NEAT genome, but a fixed architecture trained by
    backprop instead of NEAT's topology/weight evolution. Used for the
    `kd_trained_cppn` baseline that isolates "does evolution help" from "does
    gradient-learning a coordinate CPPN at all help"."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = torch.sin(self.fc1(coords))
        x = torch.tanh(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def train_trainable_cppn(
    teacher: nn.Module,
    probe_dataset,
    image_size: int,
    channels: int,
    view_op: str,
    view_scale: float,
    num_steps: int,
    probe_batch_size: int,
    lr: float,
    temperature: float,
    diversity_weight: float,
    kl_weight: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Gradient-trains a TrainableCPPN with a differentiable proxy of the
    evolved-CPPN fitness objective: minimize teacher-logit KL divergence
    between raw and view (a smooth analogue of the non-differentiable
    top-1 agreement gate) while maximizing feature diversity. This is the
    closest smooth analogue of the evolution fitness, not an identical
    objective under gradient descent (documented in the plan) — the
    non-differentiable top-1 agreement gate has no exact gradient-based
    equivalent.

    Returns a fixed, detached pattern [H, W, C], used identically to an
    evolved genome's compiled pattern by the distillation trainer.
    """
    cppn = TrainableCPPN().to(device)
    optimizer = torch.optim.Adam(cppn.parameters(), lr=lr)
    coord_grid = make_coord_grid(image_size, image_size, channels, device=device)

    was_training = teacher.training
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    for step in range(num_steps):
        images_raw01, _labels = get_probe_batch(
            probe_dataset, probe_batch_size, seed=seed + step, device=device
        )

        with torch.no_grad():
            logits_raw, features_raw = teacher(images_raw01, return_features=True)

        pattern_flat = cppn(coord_grid)  # [N, 1], grad-connected
        pattern = reshape_pattern(pattern_flat, image_size, image_size, channels)
        view = apply_pattern(images_raw01, pattern, mode=view_op, scale=view_scale)

        logits_view, features_view = teacher(view, return_features=True)

        diversity_soft = 1.0 - F.cosine_similarity(features_raw, features_view, dim=-1).mean()
        consistency_kl = F.kl_div(
            F.log_softmax(logits_view / temperature, dim=-1),
            F.softmax(logits_raw / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)

        loss = kl_weight * consistency_kl - diversity_weight * diversity_soft

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % max(1, num_steps // 5) == 0:
            log.info(
                "trainable_cppn step %d/%d: loss=%.4f diversity=%.4f kl=%.4f",
                step,
                num_steps,
                loss.item(),
                diversity_soft.item(),
                consistency_kl.item(),
            )

    for p in teacher.parameters():
        p.requires_grad_(True)
    teacher.train(was_training)

    with torch.no_grad():
        final_pattern = reshape_pattern(cppn(coord_grid), image_size, image_size, channels).detach()
    return final_pattern

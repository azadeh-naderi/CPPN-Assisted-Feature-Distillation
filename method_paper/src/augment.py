"""Batch-level augmentation for the KD + strong-augmentation baselines.
(RandAugment is per-image and lives in data.train_transform.)"""

import math

import torch


def cutmix(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """CutMix (Yun et al., ICCV 2019): pastes a random box from a shuffled
    copy of the batch into each image. The box covers a fraction 1 - lam of
    the image, lam ~ Beta(alpha, alpha), and lam is recomputed from the box
    actually used after clipping it to the image border.

    Returns (mixed_images, shuffled_labels, lam). The CE target is
    lam * CE(labels) + (1 - lam) * CE(shuffled_labels); the KD target is the
    teacher's output on the same mixed images."""
    batch, _channels, height, width = images.shape
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(batch, device=images.device)

    cut = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut), int(width * cut)
    cy = torch.randint(height, (1,)).item()
    cx = torch.randint(width, (1,)).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (height * width)
    return mixed, labels[perm], lam

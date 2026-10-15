"""Post-hoc metrics on trained checkpoints (Phase 3): calibration, agreement
with the teacher, and robustness on CIFAR-100-C (Hendrycks & Dietterich,
ICLR 2019).

CIFAR-100-C layout, as distributed: <root>/<corruption>.npy, each uint8
[50000, 32, 32, 3] holding severities 1-5 in consecutive blocks of 10000
test images, plus <root>/labels.npy.
"""

import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from method_paper.src.data import normalize

# The 15 corruptions used for the standard corruption-robustness average;
# the dataset's 4 extra ones are reported separately.
CIFAR_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
CIFAR_C_EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
SEVERITIES = (1, 2, 3, 4, 5)
IMAGES_PER_SEVERITY = 10000


@torch.no_grad()
def predict(model: nn.Module, loader, dataset: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and labels over a loader yielding raw [0, 1] images (CPU tensors)."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(normalize(images.to(device, non_blocking=True), dataset)).float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins (Guo et al., ICML 2017), in percent:
    sum over bins of |accuracy - mean confidence| weighted by the bin's share."""
    confidences, predictions = F.softmax(logits, dim=1).max(dim=1)
    correct = (predictions == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * (correct[in_bin].mean() - confidences[in_bin].mean()).abs()
    return 100.0 * ece.item()


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return {
        "top1": 100.0 * (logits.argmax(1) == labels).float().mean().item(),
        "nll": F.cross_entropy(logits, labels).item(),
        "ece": expected_calibration_error(logits, labels),
    }


def agreement_metrics(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> dict[str, float]:
    """Top-1 agreement with the teacher (%) and KL(teacher || student) at T=1."""
    kl = F.kl_div(F.log_softmax(student_logits, 1), F.softmax(teacher_logits, 1), reduction="batchmean")
    return {
        "teacher_agreement": 100.0 * (student_logits.argmax(1) == teacher_logits.argmax(1)).float().mean().item(),
        "teacher_kl": kl.item(),
    }


@functools.lru_cache(maxsize=None)
def _load_corruption(root: str, corruption: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole corruption file as uint8 [N, 32, 32, 3] plus labels, read once
    and kept in memory (~150 MB each, ~2.9 GB for all 19), so evaluating
    many checkpoints doesn't re-read the files from shared storage."""
    images = torch.from_numpy(np.load(Path(root) / f"{corruption}.npy"))
    labels = torch.from_numpy(np.load(Path(root) / "labels.npy")).long()
    return images, labels


def cifar_c_loader(root: str | Path, corruption: str, severity: int, batch_size: int = 500):
    """Yields (raw [0, 1] images [B, 3, 32, 32], labels) for one severity."""
    images, labels = _load_corruption(str(root), corruption)
    lo, hi = (severity - 1) * IMAGES_PER_SEVERITY, severity * IMAGES_PER_SEVERITY
    for start in range(lo, hi, batch_size):
        stop = min(start + batch_size, hi)
        yield images[start:stop].permute(0, 3, 1, 2).float() / 255.0, labels[start:stop]


def available_corruptions(root: str | Path) -> list[str]:
    root = Path(root)
    return [c for c in (*CIFAR_C_CORRUPTIONS, *CIFAR_C_EXTRA) if (root / f"{c}.npy").exists()]


def cifar_c_metrics(model: nn.Module, root: str | Path, dataset: str, device: torch.device,
                    batch_size: int = 500) -> dict:
    """Accuracy and ECE per corruption and severity, plus `mean_top1` /
    `mean_ece` over the standard 15 corruptions x 5 severities (only if all
    15 files are present)."""
    per = {}
    for corruption in available_corruptions(root):
        per[corruption] = {}
        for severity in SEVERITIES:
            logits, labels = predict(model, cifar_c_loader(root, corruption, severity, batch_size), dataset, device)
            m = classification_metrics(logits, labels)
            per[corruption][str(severity)] = {"top1": m["top1"], "ece": m["ece"]}
    result = {"per_corruption": per}
    if all(c in per for c in CIFAR_C_CORRUPTIONS):
        cells = [per[c][str(s)] for c in CIFAR_C_CORRUPTIONS for s in SEVERITIES]
        result["mean_top1"] = float(np.mean([x["top1"] for x in cells]))
        result["mean_ece"] = float(np.mean([x["ece"] for x in cells]))
    return result

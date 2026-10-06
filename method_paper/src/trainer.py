"""Training loop for the benchmark protocol (Tian et al., ICLR 2020):
SGD + momentum + weight decay, multi-step LR decay, and the standard KD loss
`gamma * CE + alpha * KL(teacher/T || student/T) * T^2`.

Two things the main pipeline's DistillTrainer doesn't do that this does:
weight decay (the main pipeline's SGD has none, which costs real accuracy
on CIFAR) and evaluation on the official test split every epoch.

Modes:
- "ce": plain cross-entropy (teachers and student_only)
- "kd": Hinton KD against a frozen teacher
- "kd_view": KD plus a KD term on CPPN views of the batch (Phase 1's
  kd_random_cppn / kd_trained_cppn / kd_evolved_cppn / kd_random_search):
      gamma * CE + alpha * ((1 - w) * KD(raw) + w * mean_v KD(view_v))
  with w = view_weight. w = 0.5 is the main pipeline's combined_loss form.
  view_sampling="all" uses every pattern on every batch, as the main
  pipeline did; "one" samples a single pattern per batch.
- "kd_cutmix": KD on CutMix-ed batches (applied with probability
  cutmix_prob); CE uses the area-weighted label mix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from method_paper.src.augment import cutmix
from method_paper.src.data import normalize
from src.cppn.apply import apply_pattern
from src.distill.losses import kd_loss

MODES = ("ce", "kd", "kd_view", "kd_cutmix")
VIEW_SAMPLING = ("all", "one")


@torch.no_grad()
def evaluate(model: nn.Module, loader, dataset: str, device: torch.device) -> dict[str, float]:
    model.eval()
    correct1 = correct5 = total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(normalize(images, dataset))
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    return {"top1": 100.0 * correct1 / total, "top5": 100.0 * correct5 / total}


def train_one_epoch(
    student: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    dataset: str,
    device: torch.device,
    mode: str,
    teacher: nn.Module | None = None,
    temperature: float = 4.0,
    alpha: float = 0.9,
    gamma: float = 0.1,
    max_batches: int | None = None,
    patterns: list[torch.Tensor] | None = None,
    view_op: str = "multiplicative",
    view_scale: float = 0.5,
    view_weight: float = 0.5,
    view_sampling: str = "all",
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 0.5,
) -> dict[str, float]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {MODES}.")
    if mode != "ce" and teacher is None:
        raise ValueError(f"mode={mode!r} requires a teacher")
    if mode == "kd_view":
        if not patterns:
            raise ValueError("mode='kd_view' requires at least one pattern")
        if view_sampling not in VIEW_SAMPLING:
            raise ValueError(f"Unknown view_sampling {view_sampling!r}. Choose one of {VIEW_SAMPLING}.")

    student.train()
    loss_sum = correct = total = 0.0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        mixed_labels, lam = None, 1.0
        if mode == "kd_cutmix" and torch.rand(1).item() < cutmix_prob:
            images, mixed_labels, lam = cutmix(images, labels, cutmix_alpha)
        images_norm = normalize(images, dataset)
        logits = student(images_norm)
        loss_ce = F.cross_entropy(logits, labels)
        if mixed_labels is not None:
            loss_ce = lam * loss_ce + (1.0 - lam) * F.cross_entropy(logits, mixed_labels)
        if mode == "ce":
            loss = loss_ce
        else:
            with torch.no_grad():
                teacher_logits = teacher(images_norm)
            loss_kd = kd_loss(logits, teacher_logits, temperature)
            if mode == "kd_view":
                loss_kd = (1.0 - view_weight) * loss_kd + view_weight * _view_kd_loss(
                    student, teacher, images, patterns, dataset, temperature, view_op, view_scale, view_sampling
                )
            loss = gamma * loss_ce + alpha * loss_kd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        loss_sum += loss.item() * batch
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch
    return {"loss": loss_sum / max(total, 1), "acc": 100.0 * correct / max(total, 1)}


def _view_kd_loss(
    student: nn.Module,
    teacher: nn.Module,
    images_raw01: torch.Tensor,
    patterns: list[torch.Tensor],
    dataset: str,
    temperature: float,
    view_op: str,
    view_scale: float,
    view_sampling: str,
) -> torch.Tensor:
    """Mean KD loss over CPPN views of the batch. The student sees each view
    in its own forward pass (so BatchNorm statistics never mix raw images
    and views, as in the main pipeline); the frozen teacher, whose BatchNorm
    uses running statistics, sees all views in one batched pass."""
    if view_sampling == "one":
        patterns = [patterns[torch.randint(len(patterns), (1,)).item()]]
    views = [normalize(apply_pattern(images_raw01, p, view_op, view_scale), dataset) for p in patterns]
    with torch.no_grad():
        teacher_view_logits = teacher(torch.cat(views)).chunk(len(views))
    losses = [kd_loss(student(v), t, temperature) for v, t in zip(views, teacher_view_logits)]
    return torch.stack(losses).mean()


def fit(
    student: nn.Module,
    train_loader,
    test_loader,
    dataset: str,
    device: torch.device,
    epochs: int,
    lr: float,
    lr_milestones: list[int],
    lr_decay: float,
    momentum: float,
    weight_decay: float,
    mode: str,
    teacher: nn.Module | None = None,
    temperature: float = 4.0,
    alpha: float = 0.9,
    gamma: float = 0.1,
    run_logger=None,
    max_batches: int | None = None,
    **mode_kwargs,
) -> dict[str, float]:
    """Trains `student` and returns final- and best-epoch test accuracy.
    `mode_kwargs` (patterns, view_*, cutmix_*) go to train_one_epoch.

    Report `final_test_top1` as the primary number. `best_test_top1` is the
    max over epochs of test accuracy -- i.e. selected on the test set -- and
    is logged only for comparison with papers that report it."""
    student = student.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)

    best_top1, best_epoch = -1.0, -1
    test = {"top1": float("nan"), "top5": float("nan")}
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(
            student, train_loader, optimizer, dataset, device, mode,
            teacher=teacher, temperature=temperature, alpha=alpha, gamma=gamma, max_batches=max_batches,
            **mode_kwargs,
        )
        scheduler.step()
        test = evaluate(student, test_loader, dataset, device)
        if test["top1"] > best_top1:
            best_top1, best_epoch = test["top1"], epoch
        if run_logger is not None:
            run_logger.log_epoch(
                epoch, lr=current_lr, train_loss=train["loss"], train_acc=train["acc"],
                test_top1=test["top1"], test_top5=test["top5"],
            )

    return {
        "final_test_top1": test["top1"],
        "final_test_top5": test["top5"],
        "best_test_top1": best_top1,
        "best_epoch": best_epoch,
    }

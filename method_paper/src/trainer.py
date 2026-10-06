"""Training loop for the benchmark protocol (Tian et al., ICLR 2020):
SGD + momentum + weight decay, multi-step LR decay, and the standard KD loss
`gamma * CE + alpha * KL(teacher/T || student/T) * T^2`.

Two things the main pipeline's DistillTrainer doesn't do that this does:
weight decay (the main pipeline's SGD has none, which costs real accuracy
on CIFAR) and evaluation on the official test split every epoch.

Modes:
- "ce": plain cross-entropy (teachers and student_only)
- "kd": Hinton KD against a frozen teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from method_paper.src.data import normalize
from src.distill.losses import kd_loss

MODES = ("ce", "kd")


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
) -> dict[str, float]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {MODES}.")
    if mode == "kd" and teacher is None:
        raise ValueError("mode='kd' requires a teacher")

    student.train()
    loss_sum = correct = total = 0.0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        images_norm = normalize(images, dataset)
        logits = student(images_norm)
        loss_ce = F.cross_entropy(logits, labels)
        if mode == "ce":
            loss = loss_ce
        else:
            with torch.no_grad():
                teacher_logits = teacher(images_norm)
            loss = gamma * loss_ce + alpha * kd_loss(logits, teacher_logits, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        loss_sum += loss.item() * batch
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch
    return {"loss": loss_sum / max(total, 1), "acc": 100.0 * correct / max(total, 1)}


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
) -> dict[str, float]:
    """Trains `student` and returns final- and best-epoch test accuracy.

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

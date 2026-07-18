import torch
import torch.nn as nn
import torch.optim as optim

from src.cppn.apply import apply_pattern
from src.cppn.legacy_random import LegacyCPPN, legacy_view, make_legacy_cppn
from src.data.datasets import normalize_batch
from src.distill.losses import combined_loss, kd_loss
from src.distill.modes import MODES, RANDOM_CPPN_VARIANTS
from src.utils.logging import get_logger

log = get_logger(__name__)


class DistillTrainer:
    """One class backing every baseline/ablation via `mode`:
    - student_only: pure CE, no teacher.
    - kd: standard soft-target KD (reused legacy `simple_distillation` logic).
    - kd_random_cppn: KD + a view from an untrained CPPN. `random_cppn_variant`
      picks 'legacy' (literal reproduction of the old repo's 2-input
      pixel-intensity CPPN) or 'coord' (untrained coordinate-only CPPN from
      the same family as the evolved ones — the fairer ablation).
    - kd_trained_cppn: KD + a view from a gradient-trained coordinate CPPN.
    - kd_evolved_cppn: KD + view(s) from evolved genome pattern(s); pass
      multiple `patterns` for the ensemble ablation.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module | None,
        mode: str,
        dataset_name: str,
        device: torch.device,
        patterns: list[torch.Tensor] | None = None,
        random_cppn_variant: str = "coord",
        random_cppn_seed: int = 0,
        neat_config=None,
        image_size: int | None = None,
        channels: int | None = None,
        view_op: str = "multiplicative",
        view_scale: float = 0.5,
        temperature: float = 4.0,
        alpha: float = 0.9,
        lr: float = 0.05,
        momentum: float = 0.9,
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode!r}. Choose one of {MODES}.")
        if mode != "student_only" and teacher is None:
            raise ValueError(f"mode={mode!r} requires a teacher.")

        self.mode = mode
        self.dataset_name = dataset_name
        self.device = device
        self.view_op = view_op
        self.view_scale = view_scale
        self.temperature = temperature
        self.alpha = alpha

        self.student = student.to(device)
        self.teacher = teacher.to(device) if teacher is not None else None
        if self.teacher is not None:
            self.teacher.eval()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.student.parameters(), lr=lr, momentum=momentum)

        self._legacy_cppn: LegacyCPPN | None = None
        self.patterns = patterns

        if mode == "kd_random_cppn":
            if random_cppn_variant not in RANDOM_CPPN_VARIANTS:
                raise ValueError(f"random_cppn_variant must be one of {RANDOM_CPPN_VARIANTS}")
            if random_cppn_variant == "legacy":
                self._legacy_cppn = make_legacy_cppn(random_cppn_seed, device)
            elif self.patterns is None:
                if neat_config is None or image_size is None or channels is None:
                    raise ValueError(
                        "random_cppn_variant='coord' needs neat_config/image_size/channels "
                        "to build a random coordinate-CPPN pattern (or pass `patterns` directly)."
                    )
                from src.cppn.compile import genome_to_pattern
                from src.cppn.evolve import create_random_genome

                genome = create_random_genome(neat_config, random_cppn_seed)
                self.patterns = [
                    genome_to_pattern(genome, neat_config.genome_config, image_size, channels, device)
                ]

        if mode in ("kd_trained_cppn", "kd_evolved_cppn") and self.patterns is None:
            raise ValueError(f"mode={mode!r} requires precomputed `patterns`.")

    def _get_views(self, images_raw01: torch.Tensor) -> list[torch.Tensor]:
        if self.mode in ("student_only", "kd"):
            return []
        if self._legacy_cppn is not None:
            return [legacy_view(self._legacy_cppn, images_raw01)]
        return [apply_pattern(images_raw01, p, self.view_op, self.view_scale) for p in self.patterns]

    def _step(self, images_raw01: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images_norm = normalize_batch(images_raw01, self.dataset_name)
        student_logits = self.student(images_norm)
        loss_hard = self.criterion(student_logits, labels)

        if self.mode == "student_only":
            return loss_hard

        with torch.no_grad():
            teacher_logits = self.teacher(images_norm)
        loss_soft = kd_loss(student_logits, teacher_logits, self.temperature)

        cppn_losses = []
        for view_raw01 in self._get_views(images_raw01):
            view_norm = normalize_batch(view_raw01, self.dataset_name)
            student_view_logits = self.student(view_norm)
            with torch.no_grad():
                teacher_view_logits = self.teacher(view_norm)
            cppn_losses.append(kd_loss(student_view_logits, teacher_view_logits, self.temperature))

        return combined_loss(loss_hard, loss_soft, cppn_losses, self.alpha)

    def fit(self, train_loader, val_loader, num_epochs: int, run_logger=None) -> nn.Module:
        for epoch in range(num_epochs):
            self.student.train()
            for images_raw01, labels in train_loader:
                images_raw01, labels = images_raw01.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self._step(images_raw01, labels)
                loss.backward()
                self.optimizer.step()

            val_acc = self.evaluate(val_loader)
            if run_logger is not None:
                run_logger.log_epoch(epoch, val_accuracy=val_acc)
            else:
                log.info("epoch %d/%d: val_accuracy=%.2f", epoch + 1, num_epochs, val_acc)

        return self.student

    def evaluate(self, loader) -> float:
        self.student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images_raw01, labels in loader:
                images_raw01, labels = images_raw01.to(self.device), labels.to(self.device)
                images_norm = normalize_batch(images_raw01, self.dataset_name)
                logits = self.student(images_norm)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        self.student.train()
        return 100.0 * correct / total


def train_teacher(
    model: nn.Module,
    dataset_name: str,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
    run_logger=None,
) -> nn.Module:
    """Plain CE training loop for the teacher, reused from legacy `train()`."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(num_epochs):
        model.train()
        for images_raw01, labels in train_loader:
            images_raw01, labels = images_raw01.to(device), labels.to(device)
            images_norm = normalize_batch(images_raw01, dataset_name)
            optimizer.zero_grad()
            loss = criterion(model(images_norm), labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_model(model, dataset_name, val_loader, device)
        if run_logger is not None:
            run_logger.log_epoch(epoch, val_accuracy=val_acc)
        else:
            log.info("teacher epoch %d/%d: val_accuracy=%.2f", epoch + 1, num_epochs, val_acc)

    return model


def evaluate_model(model: nn.Module, dataset_name: str, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images_raw01, labels in loader:
            images_raw01, labels = images_raw01.to(device), labels.to(device)
            images_norm = normalize_batch(images_raw01, dataset_name)
            pred = model(images_norm).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

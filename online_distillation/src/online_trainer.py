import torch
import torch.nn as nn
import torch.optim as optim

from src.cppn.apply import apply_pattern
from src.data.datasets import normalize_batch
from src.distill.losses import kd_loss
from src.utils.logging import get_logger

log = get_logger(__name__)

ONLINE_MODES = ["hard_label_augmentation", "self_consistency_random_cppn"]
"""Teacher-free variants of the CPPN-view idea -- no pretrained/fine-tuned
teacher model anywhere in this trainer's loop, unlike
src/distill/trainer.py's DistillTrainer.

- hard_label_augmentation: the CPPN view is treated as a pure data
  augmentation -- (1-alpha)*CE(raw, labels) + alpha*CE(view, labels), both
  terms compared against the true label. No consistency/soft-target loss,
  no self-reference at all.
- self_consistency_random_cppn: the student's own raw-image prediction
  (detached, no gradient) stands in for a teacher, reusing
  src.distill.losses.kd_loss exactly as DistillTrainer does elsewhere in
  this repo -- just with the "teacher" being the student's own detached
  raw-image logits instead of a separate pretrained model's raw-image
  output.

Pattern source for both modes is always an untrained, randomly-initialized
coordinate CPPN (same construction as kd_random_cppn's 'coord' variant,
src.cppn.evolve.create_random_genome) -- genuinely teacher-free at every
stage. Evolving genomes (kd_evolved_cppn) or gradient-training a CPPN
(kd_trained_cppn) both currently need a frozen teacher to score fitness /
compute their own training loss, which is a separate, harder problem this
first round doesn't attempt -- see online_distillation/README.md.
"""


class OnlineDistillTrainer:
    def __init__(
        self,
        student: nn.Module,
        mode: str,
        dataset_name: str,
        device: torch.device,
        pattern: torch.Tensor,
        view_op: str = "multiplicative",
        view_scale: float = 0.5,
        temperature: float = 4.0,
        alpha: float = 0.5,
        lr: float = 0.1,
        momentum: float = 0.9,
        scheduler: bool = False,
        step_size: int = 30,
        gamma: float = 0.1,
    ):
        if mode not in ONLINE_MODES:
            raise ValueError(f"Unknown online mode: {mode!r}. Choose one of {ONLINE_MODES}.")

        self.mode = mode
        self.dataset_name = dataset_name
        self.device = device
        self.pattern = pattern.to(device)
        self.view_op = view_op
        self.view_scale = view_scale
        self.temperature = temperature
        self.alpha = alpha

        self.student = student.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.student.parameters(), lr=lr, momentum=momentum)
        self.lr_scheduler = (
            optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma) if scheduler else None
        )

    def _step(self, images_raw01: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images_norm = normalize_batch(images_raw01, self.dataset_name)
        raw_logits = self.student(images_norm)
        loss_hard = self.criterion(raw_logits, labels)

        view_raw01 = apply_pattern(images_raw01, self.pattern, mode=self.view_op, scale=self.view_scale)
        view_norm = normalize_batch(view_raw01, self.dataset_name)
        view_logits = self.student(view_norm)

        if self.mode == "hard_label_augmentation":
            loss_view = self.criterion(view_logits, labels)
        else:  # self_consistency_random_cppn
            # raw_logits.detach(): the consistency term should pull the
            # view prediction toward the raw prediction, not the reverse --
            # same asymmetry as kd_loss(student_view, teacher_view) in
            # DistillTrainer, just with a detached copy of the student's own
            # output standing in for the teacher's.
            loss_view = kd_loss(view_logits, raw_logits.detach(), self.temperature)

        return (1 - self.alpha) * loss_hard + self.alpha * loss_view

    def fit(self, train_loader, val_loader, num_epochs: int, run_logger=None) -> nn.Module:
        for epoch in range(num_epochs):
            self.student.train()
            for images_raw01, labels in train_loader:
                images_raw01, labels = images_raw01.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self._step(images_raw01, labels)
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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

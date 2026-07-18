import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard soft-target KD loss (reused from legacy simple_distillation)."""
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )


# Same KL form, applied to a CPPN-transformed view instead of the raw image.
cppn_consistency_loss = kd_loss


def combined_loss(
    loss_hard: torch.Tensor,
    loss_soft: torch.Tensor,
    cppn_losses: list[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """(1-alpha)*CE + alpha*KD, generalized from legacy's
    `(1-alpha)*hard + alpha*((soft+cppn)/2)` to average over zero or more
    CPPN-view consistency losses (supports the ensemble-of-evolved-genomes
    ablation, where multiple views each contribute a term)."""
    if not cppn_losses:
        return (1 - alpha) * loss_hard + alpha * loss_soft
    cppn_mean = torch.stack(cppn_losses).mean()
    return (1 - alpha) * loss_hard + alpha * ((loss_soft + cppn_mean) / 2)

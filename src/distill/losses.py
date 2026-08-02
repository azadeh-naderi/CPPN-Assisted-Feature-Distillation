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
    cppn_weight: float | None = None,
) -> torch.Tensor:
    """(1-alpha)*CE + alpha*KD, generalized from legacy's
    `(1-alpha)*hard + alpha*((soft+cppn)/2)` to average over zero or more
    CPPN-view consistency losses (supports the ensemble-of-evolved-genomes
    ablation, where multiple views each contribute a term).

    If `cppn_weight` is given, the CPPN-view term gets its own independent
    weight instead of being averaged 50/50 into the alpha-weighted soft-label
    term: `(1-alpha)*hard + alpha*soft + cppn_weight*cppn_mean`. Motivated by
    real-run evidence (see experiments/EXPERIMENT_LOG.md) that a genome
    selected for teacher-feature diversity is a harder, noisier training
    signal than plain soft labels — random_cppn > trained_cppn > evolved_cppn
    in student accuracy cost tracks directly with how hard each was optimized
    to be different. Averaging both under one alpha both dilutes standard
    KD's contribution (vs. plain `kd` mode, which gets full alpha on
    loss_soft) and gives the costlier evolved term equal footing with the
    safe one. Splitting them restores full-strength standard KD and lets the
    CPPN term contribute a smaller, independently-tunable amount.
    """
    if not cppn_losses:
        return (1 - alpha) * loss_hard + alpha * loss_soft
    cppn_mean = torch.stack(cppn_losses).mean()
    if cppn_weight is not None:
        return (1 - alpha) * loss_hard + alpha * loss_soft + cppn_weight * cppn_mean
    return (1 - alpha) * loss_hard + alpha * ((loss_soft + cppn_mean) / 2)

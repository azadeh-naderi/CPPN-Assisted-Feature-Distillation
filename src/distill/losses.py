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
    loss_soft: torch.Tensor | None,
    cppn_losses: list[torch.Tensor],
    alpha: float,
    cppn_weight: float | None = None,
    use_soft_kd: bool = True,
) -> torch.Tensor:
    """(1-alpha)*CE + alpha*KD, generalized from legacy's
    `(1-alpha)*hard + alpha*((soft+cppn)/2)` to average over zero or more
    CPPN-view consistency losses (supports the ensemble-of-evolved-genomes
    ablation, where multiple views each contribute a term).

    If `cppn_weight` is given, the CPPN-view term gets its own independent
    weight, drawn from the same fixed budget as `alpha` (a proper convex
    combination, all three terms summing to 1) rather than the previous
    50/50-averaged form:
        (1 - alpha - cppn_weight) * loss_hard + alpha * loss_soft + cppn_weight * cppn_mean
    Motivated by real-run evidence (see experiments/EXPERIMENT_LOG.md) that a
    genome selected for teacher-feature diversity is a harder, noisier
    training signal than plain soft labels — random_cppn > trained_cppn >
    evolved_cppn in student accuracy cost tracks directly with how hard each
    was optimized to be different. An earlier version of this made the CPPN
    term purely *additive* on top of full-strength alpha (total
    distillation-related weight = alpha + cppn_weight > alpha), which a real
    run showed was actively harmful (attempt 6, EXPERIMENT_LOG.md) — pushing
    total non-hard-label weight past 1 apparently destabilizes optimization
    on its own, independent of which genome evolution picks. This version
    keeps the same fixed budget as the original averaged form while still
    letting soft-label KD and the CPPN term be weighted independently within
    it (e.g. alpha=0.6, cppn_weight=0.3 gives soft more weight than the old
    45/45 split while giving the costlier CPPN term less).

    `use_soft_kd=False` (default True) drops the raw-image soft-label KD
    term entirely for CPPN-view modes -- loss becomes a strict two-term
    convex combination `(1-alpha)*loss_hard + alpha*mean(cppn_losses)`, with
    no teacher-vs-raw-image comparison contributing at all (`loss_soft` is
    ignored even if provided). Isolates whether the CPPN-view consistency
    signal alone -- not blended with or diluted by standard soft-label KD --
    is what actually drives kd_random_cppn/kd_trained_cppn/kd_evolved_cppn's
    behavior, rather than the two signals interacting. Requires at least one
    CPPN loss (not meaningful for modes without a CPPN view, e.g. plain
    'kd', which should always leave this at the default `True`).
    """
    if not use_soft_kd:
        if not cppn_losses:
            raise ValueError("use_soft_kd=False requires at least one cppn loss")
        cppn_mean = torch.stack(cppn_losses).mean()
        return (1 - alpha) * loss_hard + alpha * cppn_mean

    if not cppn_losses:
        return (1 - alpha) * loss_hard + alpha * loss_soft
    cppn_mean = torch.stack(cppn_losses).mean()
    if cppn_weight is not None:
        hard_weight = 1 - alpha - cppn_weight
        return hard_weight * loss_hard + alpha * loss_soft + cppn_weight * cppn_mean
    return (1 - alpha) * loss_hard + alpha * ((loss_soft + cppn_mean) / 2)

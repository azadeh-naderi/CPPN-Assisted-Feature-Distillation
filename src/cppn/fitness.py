import torch
import torch.nn.functional as F


def diversity_term(features_raw: torch.Tensor, features_view: torch.Tensor) -> float:
    """1 - mean cosine similarity between frozen-teacher penultimate features
    on the raw image vs. the CPPN-transformed view, over a probe batch.
    Higher = the view exposes a more different "angle" on the features."""
    cos_sim = F.cosine_similarity(features_raw, features_view, dim=-1)
    return (1.0 - cos_sim.mean()).item()


def agreement_term(logits_raw: torch.Tensor, logits_view: torch.Tensor) -> float:
    """Teacher top-1 prediction agreement rate between raw and view, over a
    probe batch. Guards against genomes that destroy all class-relevant
    signal (agreement -> 0)."""
    pred_raw = logits_raw.argmax(dim=-1)
    pred_view = logits_view.argmax(dim=-1)
    return (pred_raw == pred_view).float().mean().item()


def gate(agreement: float, tau_low: float, tau_high: float) -> float:
    """Smooth ramp from 0 (agreement <= tau_low) to 1 (agreement >= tau_high).
    A ramp rather than a hard step keeps a ranking signal among genomes that
    are all "not good enough yet", which a single threshold would flatten."""
    if tau_high <= tau_low:
        raise ValueError("tau_high must be > tau_low")
    return max(0.0, min(1.0, (agreement - tau_low) / (tau_high - tau_low)))


def fitness_from_terms(
    diversity: float,
    agreement: float,
    tau_low: float = 0.3,
    tau_high: float = 0.7,
    gamma: float = 0.0,
    num_connections: int = 0,
    pattern_std: float = 0.0,
    contrast_penalty: float = 0.0,
    contrast_std_threshold: float = 0.0,
) -> float:
    """Gated combination, not a plain weighted sum: a sum lets a
    class-destroying genome (diversity high, agreement ~0) outscore a
    meaningful perturbation whenever diversity dominates. Multiplying
    diversity by gate(agreement) collapses both failure modes to ~0 fitness
    structurally:
      - identity-like genomes:  agreement~1, diversity~0 -> fitness~0
      - adversarial-noise genomes: agreement~0 -> gate~0 -> fitness~0
        regardless of how high diversity is.
    `gamma` is an optional parsimony penalty (default off; a config knob for
    later ablations on genome bloat, not part of the MVP default).

    `contrast_penalty` (default off) penalizes pattern std *above*
    `contrast_std_threshold` -- i.e. high-contrast, near-binary spatial
    masks -- rather than minimizing std monotonically. Real CIFAR-10 runs
    (experiments/EXPERIMENT_LOG.md, attempts 3-4) found the agreement gate
    alone let evolution repeatedly select genomes with std~0.4+ that amount
    to a static occlusion mask (identical across every image and every
    training epoch, unlike per-batch-random augmentation), occasionally
    causing catastrophic training collapse even when ensembled over the
    top-5 genomes (attempts 5-7). A first attempt (attempt 8) penalized std
    directly with no threshold (`contrast_std_threshold=0.0`, the default,
    reproduces that exact prior behavior) -- it fixed the instability, but
    the minimum of that penalty is std=0 exactly, and evolution took the
    cheapest path: every winning genome collapsed to a single-bias-node,
    zero-connection constant pattern (a uniform brightness scalar, not a
    spatial view at all). Gating the penalty to only kick in above a
    threshold (e.g. 0.2) removes that shortcut -- genomes with genuine but
    moderate spatial variation face zero penalty, only genomes that exceed
    the threshold (the actual diagnosed failure mode) are discouraged.
    """
    g = gate(agreement, tau_low, tau_high)
    contrast_excess = max(0.0, pattern_std - contrast_std_threshold)
    return diversity * g - gamma * num_connections - contrast_penalty * contrast_excess

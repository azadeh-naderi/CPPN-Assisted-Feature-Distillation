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
    signal (agreement -> 0). Kept for logging/comparison — see
    `soft_agreement_term` for what fitness actually uses as of attempt 10."""
    pred_raw = logits_raw.argmax(dim=-1)
    pred_view = logits_view.argmax(dim=-1)
    return (pred_raw == pred_view).float().mean().item()


def soft_agreement_term(logits_raw: torch.Tensor, logits_view: torch.Tensor) -> float:
    """Smooth agreement measure: cosine similarity between the teacher's full
    softmax probability distributions (raw vs. view), not just whether the
    top-1 class matches. Bounded in [0,1] like `agreement_term` (softmax
    outputs are non-negative, so cosine similarity of two such vectors is
    non-negative), so it's a drop-in replacement for the same gate()
    mechanism and tau_low/tau_high thresholds.

    Motivation (experiments/EXPERIMENT_LOG.md, attempts 8-9): top-1 agreement
    is a coarse, effectively binary constraint from NEAT's search
    perspective — a genome only needs to keep the *argmax* class the same,
    and is free to scramble the rest of the distribution arbitrarily while
    still "passing." Evolution repeatedly found views that did exactly that,
    landing on a real, repeatable accuracy cost even after fixing the
    contrast/occlusion-mask failure mode directly (attempt 9). Notably,
    `kd_trained_cppn` — which optimizes a smooth KL-divergence consistency
    penalty via gradient descent instead of a hard top-1 threshold — has
    never shown this cost in any attempt. This gives evolution the same kind
    of continuously-graded pressure back toward agreement across the whole
    distribution, not just the argmax, instead of a threshold a strong
    discrete optimizer can cheaply satisfy while diverging everywhere else.
    """
    probs_raw = F.softmax(logits_raw, dim=-1)
    probs_view = F.softmax(logits_view, dim=-1)
    cos_sim = F.cosine_similarity(probs_raw, probs_view, dim=-1)
    return cos_sim.mean().item()


def channel_divergence_term(pattern: torch.Tensor) -> float:
    """Mean per-pixel std across the channel dimension of a compiled pattern
    [H, W, C]. High values mean the pattern treats color channels very
    differently at the *same* spatial location -- e.g. suppressing green
    while letting red/blue through in one region, producing a colored
    stripe/tint -- rather than a uniform (per-pixel-shared) brightness or
    contrast adjustment. Visual inspection of a real evolved CIFAR-10 genome
    (experiments/EXPERIMENT_LOG.md, attempt 10 visualization) showed exactly
    this: fixed magenta/green vertical stripes, identical across every image
    regardless of content, localized enough to slip past both the top-1 and
    soft agreement measures while still registering as "diverse" in feature
    space. Returns 0.0 for single-channel (grayscale) patterns, where the
    concept doesn't apply.
    """
    if pattern.shape[-1] <= 1:
        return 0.0
    return pattern.std(dim=-1).mean().item()


DISQUALIFIED_FITNESS = -1e6
"""Sentinel fitness for genomes below `min_connections` (see
`fitness_from_terms`). Finite (not -inf) so it stays safe through pandas/CSV
logging and torch tensor ops, but far below any value a legitimate genome
(diversity/agreement/penalty terms all O(1)) could ever reach -- guarantees
disqualified genomes never win a fitness comparison or tie-break."""


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
    channel_divergence: float = 0.0,
    channel_divergence_penalty: float = 0.0,
    min_connections: int = 0,
    min_pattern_std: float = 0.0,
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

    `channel_divergence_penalty` (default off, attempt 11) penalizes
    `channel_divergence_term` directly and monotonically -- driving this
    toward 0 was intended to be a harmless outcome (a pattern with zero
    channel divergence can still vary richly across space, contributing to
    `pattern_std`/diversity normally; it just can't treat R/G/B differently
    at the same pixel), unlike `contrast_penalty` where std=0 was a known
    degenerate shortcut. In practice (real CIFAR-10 run, attempt 11) this
    assumption was only half right: it did eliminate the specific colored-
    stripe artifact it targeted (every top genome across all 3 seeds reached
    exactly `channel_divergence=0`), but it also narrows the achievable
    diversity ceiling for genuinely spatial genomes that would otherwise use
    channel-dependent connections, which made the pre-existing
    `contrast_std_threshold`-gated plateau (genomes with std <= threshold,
    zero contrast penalty either way) even flatter. One seed's population
    drifted into that flattened plateau's zero-connection corner -- the
    exact degenerate single-bias-node genome attempt 9 was supposed to have
    made non-competitive -- and this time landed at an extreme constant
    (~0.0125, near-total blackout) instead of attempt 8's benign ~0.7,
    causing a catastrophic training collapse (12.82% test accuracy) that
    ensembling didn't save because 3 of the top-5 genomes were the same
    degenerate genome. See `min_connections` below for the fix (attempt 12).

    `min_connections` (default 0, attempt 12) disqualifies any genome with
    fewer enabled connections than this by returning `DISQUALIFIED_FITNESS`
    immediately, before any of the terms above are computed -- rather than
    trying to out-design yet another penalty term that hopes to make the
    zero-connection genome merely non-optimal (which attempts 8, 9, and 11
    each did in a different way, and attempt 11 still lost to it in one
    seed), this removes it as a selectable option entirely. A genome with no
    connections isn't really a spatial "view" in any sense the method's
    premise depends on -- it's a constant, and NEAT's search has now
    reached that exact degenerate corner from two different fitness
    formulations, which is a strong enough signal to just exclude it rather
    than continue tuning penalties around it.

    `min_pattern_std` (default 0, attempt 13) disqualifies on the compiled
    pattern's own std directly, same sentinel mechanism as
    `min_connections` -- added after a real 10-seed CIFAR-10 run of
    attempt 12 showed `min_connections` is gameable: one seed's winning
    genome had 2 *enabled* connections (satisfying `min_connections=1`)
    that formed a subgraph entirely disconnected from the output node,
    leaving the output as a pure function of its own bias -- a literal
    constant (`pattern_std=0.0` exactly) despite `num_connections>=1`.
    `min_connections` counts enabled connections anywhere in the genome,
    not whether any of them actually reach the output, so a genome can
    satisfy the floor while still being functionally constant. Measuring
    `pattern_std` instead checks the thing we actually care about --
    whether the compiled output varies at all -- directly and robustly,
    independent of which specific graph-structure mechanism (zero
    connections, disconnected dead branches, or anything else not yet
    seen) produced the constancy. Kept `min_connections` too (still a
    cheap, valid, if weaker, first-pass filter) rather than removing it.
    """
    if num_connections < min_connections:
        return DISQUALIFIED_FITNESS
    if pattern_std < min_pattern_std:
        return DISQUALIFIED_FITNESS
    g = gate(agreement, tau_low, tau_high)
    contrast_excess = max(0.0, pattern_std - contrast_std_threshold)
    return (
        diversity * g
        - gamma * num_connections
        - contrast_penalty * contrast_excess
        - channel_divergence_penalty * channel_divergence
    )

import pytest
import torch

from src.cppn.fitness import (
    DISQUALIFIED_FITNESS,
    agreement_term,
    channel_divergence_term,
    fitness_from_terms,
    gate,
    soft_agreement_term,
)


def test_gate_ramps_between_thresholds():
    assert gate(0.0, tau_low=0.3, tau_high=0.7) == 0.0
    assert gate(0.3, tau_low=0.3, tau_high=0.7) == 0.0
    assert gate(0.7, tau_low=0.3, tau_high=0.7) == 1.0
    assert gate(1.0, tau_low=0.3, tau_high=0.7) == 1.0
    assert gate(0.5, tau_low=0.3, tau_high=0.7) == pytest.approx(0.5)


def test_gate_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        gate(0.5, tau_low=0.7, tau_high=0.3)


def test_identity_like_genome_scores_near_zero():
    # agreement ~1 (near-identity view), diversity ~0 (no new information)
    fitness = fitness_from_terms(diversity=0.02, agreement=0.98, tau_low=0.3, tau_high=0.7)
    assert fitness < 0.05


def test_adversarial_noise_genome_scores_near_zero():
    # diversity high, but agreement collapsed -> gate suppresses fitness
    # regardless of how high diversity is.
    fitness = fitness_from_terms(diversity=0.95, agreement=0.05, tau_low=0.3, tau_high=0.7)
    assert fitness < 0.05


def test_meaningful_perturbation_scores_higher_than_either_extreme():
    balanced = fitness_from_terms(diversity=0.45, agreement=0.65, tau_low=0.3, tau_high=0.7)
    identity_like = fitness_from_terms(diversity=0.02, agreement=0.98, tau_low=0.3, tau_high=0.7)
    adversarial = fitness_from_terms(diversity=0.95, agreement=0.05, tau_low=0.3, tau_high=0.7)
    assert balanced > identity_like
    assert balanced > adversarial


def test_parsimony_penalty_reduces_fitness():
    base = fitness_from_terms(diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, gamma=0.0, num_connections=50)
    penalized = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, gamma=0.01, num_connections=50
    )
    assert penalized < base


def test_contrast_penalty_reduces_fitness_for_high_std_patterns():
    base = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, pattern_std=0.42, contrast_penalty=0.0
    )
    penalized = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, pattern_std=0.42, contrast_penalty=0.3
    )
    assert penalized < base


def test_contrast_penalty_prefers_milder_pattern_at_equal_diversity_and_agreement():
    # Same diversity/agreement, but one genome's pattern is a high-contrast
    # near-binary mask (std~0.42, the failure mode diagnosed in
    # experiments/EXPERIMENT_LOG.md) and the other is milder (std~0.16, like
    # the range-capped patterns from before the OUTER_SIGMOID_SCALE fix).
    mild = fitness_from_terms(
        diversity=0.5, agreement=0.6, tau_low=0.3, tau_high=0.7, pattern_std=0.16, contrast_penalty=0.5
    )
    high_contrast = fitness_from_terms(
        diversity=0.5, agreement=0.6, tau_low=0.3, tau_high=0.7, pattern_std=0.42, contrast_penalty=0.5
    )
    assert mild > high_contrast


def test_contrast_penalty_zero_by_default():
    with_default = fitness_from_terms(diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, pattern_std=0.42)
    without_std = fitness_from_terms(diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7)
    assert with_default == without_std


def test_contrast_std_threshold_gives_zero_penalty_below_threshold():
    # Attempt 8's un-gated penalty (contrast_std_threshold=0.0, the default)
    # is minimized exactly at pattern_std=0, so evolution collapsed to
    # degenerate constant-output genomes (real run: all three winning
    # genomes had std=0.000, num_connections=0). Gating the penalty so it
    # only kicks in above a threshold removes that shortcut: genomes with
    # moderate genuine spatial variation should face no penalty at all.
    no_variation = fitness_from_terms(
        diversity=0.5,
        agreement=0.6,
        tau_low=0.3,
        tau_high=0.7,
        pattern_std=0.0,
        contrast_penalty=0.5,
        contrast_std_threshold=0.2,
    )
    moderate_variation = fitness_from_terms(
        diversity=0.5,
        agreement=0.6,
        tau_low=0.3,
        tau_high=0.7,
        pattern_std=0.15,
        contrast_penalty=0.5,
        contrast_std_threshold=0.2,
    )
    # Both below the 0.2 threshold -> identical fitness, no incentive to
    # collapse toward std=0 specifically.
    assert no_variation == moderate_variation


def test_contrast_std_threshold_still_penalizes_above_threshold():
    at_threshold = fitness_from_terms(
        diversity=0.5,
        agreement=0.6,
        tau_low=0.3,
        tau_high=0.7,
        pattern_std=0.2,
        contrast_penalty=0.5,
        contrast_std_threshold=0.2,
    )
    above_threshold = fitness_from_terms(
        diversity=0.5,
        agreement=0.6,
        tau_low=0.3,
        tau_high=0.7,
        pattern_std=0.42,
        contrast_penalty=0.5,
        contrast_std_threshold=0.2,
    )
    assert above_threshold < at_threshold


def test_soft_agreement_is_one_for_identical_distributions():
    logits = torch.tensor([[3.0, 1.0, 0.2, 0.1], [0.5, 2.0, 1.0, 0.3]])
    assert soft_agreement_term(logits, logits) == pytest.approx(1.0, abs=1e-5)


def test_soft_agreement_is_bounded_in_unit_interval():
    torch.manual_seed(0)
    logits_raw = torch.randn(32, 10) * 5
    logits_view = torch.randn(32, 10) * 5
    score = soft_agreement_term(logits_raw, logits_view)
    assert 0.0 <= score <= 1.0


def test_soft_agreement_penalizes_distribution_scrambling_that_top1_agreement_misses():
    # Same argmax class in both (class 0 wins both times), but the view's
    # distribution over the *other* classes is scrambled -- exactly the kind
    # of "technically still agrees" case that motivated switching from
    # top1_agreement to soft_agreement for fitness (see soft_agreement_term
    # docstring / experiments/EXPERIMENT_LOG.md attempt 10).
    logits_raw = torch.tensor([[5.0, 1.0, 0.5, 0.2]])
    logits_view_mild = torch.tensor([[5.0, 1.1, 0.4, 0.3]])  # barely changed
    logits_view_scrambled = torch.tensor([[5.0, 4.9, 4.8, 4.7]])  # top-1 unchanged, rest flattened

    assert agreement_term(logits_raw, logits_view_mild) == 1.0
    assert agreement_term(logits_raw, logits_view_scrambled) == 1.0  # top-1 blind to this

    mild_soft = soft_agreement_term(logits_raw, logits_view_mild)
    scrambled_soft = soft_agreement_term(logits_raw, logits_view_scrambled)
    assert scrambled_soft < mild_soft


def test_channel_divergence_zero_for_grayscale_pattern():
    pattern = torch.rand(8, 8, 1)
    assert channel_divergence_term(pattern) == 0.0


def test_channel_divergence_zero_when_channels_move_together():
    # Same value repeated across channels at every pixel -- a plain spatial
    # luminance/contrast pattern shared across R/G/B, not a colored artifact.
    base = torch.rand(8, 8, 1)
    pattern = base.expand(8, 8, 3)
    assert channel_divergence_term(pattern) == pytest.approx(0.0, abs=1e-6)


def test_channel_divergence_high_for_striped_color_artifact():
    # One channel high, others low, identical across every pixel -- the
    # magenta/green stripe artifact diagnosed in experiments/EXPERIMENT_LOG.md
    # attempt 10's visualization.
    pattern = torch.zeros(8, 8, 3)
    pattern[..., 0] = 1.0  # red channel on, green/blue off everywhere
    assert channel_divergence_term(pattern) > 0.4


def test_channel_divergence_penalty_reduces_fitness_for_striped_patterns():
    base = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, channel_divergence=0.5, channel_divergence_penalty=0.0
    )
    penalized = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, channel_divergence=0.5, channel_divergence_penalty=0.3
    )
    assert penalized < base


def test_channel_divergence_penalty_zero_by_default():
    with_default = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, channel_divergence=0.5
    )
    without = fitness_from_terms(diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7)
    assert with_default == without


def test_channel_divergence_penalty_prefers_channel_uniform_pattern_at_equal_diversity():
    # Same diversity/agreement, but one genome's pattern varies R/G/B
    # independently at each pixel (colored-artifact failure mode) and the
    # other varies spatially while keeping channels in lockstep (a legitimate
    # luminance-style pattern) -- the penalty should favor the latter.
    channel_uniform = fitness_from_terms(
        diversity=0.5, agreement=0.6, tau_low=0.3, tau_high=0.7, channel_divergence=0.0, channel_divergence_penalty=0.3
    )
    channel_striped = fitness_from_terms(
        diversity=0.5, agreement=0.6, tau_low=0.3, tau_high=0.7, channel_divergence=0.5, channel_divergence_penalty=0.3
    )
    assert channel_uniform > channel_striped


def test_min_connections_disqualifies_zero_connection_genome():
    # A zero-connection, single-bias-node genome (the degenerate collapse
    # from attempts 8 and 11) can otherwise look great on every other term --
    # this must still lose to a genuinely spatial genome once min_connections
    # is set, regardless of how favorable diversity/agreement look.
    disqualified = fitness_from_terms(
        diversity=0.99, agreement=0.99, tau_low=0.3, tau_high=0.7, num_connections=0, min_connections=1
    )
    real_genome = fitness_from_terms(
        diversity=0.1, agreement=0.6, tau_low=0.3, tau_high=0.7, num_connections=3, min_connections=1
    )
    assert disqualified == DISQUALIFIED_FITNESS
    assert real_genome > disqualified


def test_min_connections_zero_by_default_preserves_old_behavior():
    with_default = fitness_from_terms(diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, num_connections=0)
    without = fitness_from_terms(
        diversity=0.6, agreement=0.6, tau_low=0.3, tau_high=0.7, num_connections=0, min_connections=0
    )
    assert with_default == without
    assert with_default != DISQUALIFIED_FITNESS


def test_min_connections_does_not_disqualify_genomes_at_or_above_the_floor():
    at_floor = fitness_from_terms(
        diversity=0.5, agreement=0.6, tau_low=0.3, tau_high=0.7, num_connections=1, min_connections=1
    )
    above_floor = fitness_from_terms(
        diversity=0.5, agreement=0.6, tau_low=0.3, tau_high=0.7, num_connections=5, min_connections=1
    )
    assert at_floor != DISQUALIFIED_FITNESS
    assert above_floor != DISQUALIFIED_FITNESS

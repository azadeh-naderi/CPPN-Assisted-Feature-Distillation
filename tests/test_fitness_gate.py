import pytest

from src.cppn.fitness import fitness_from_terms, gate


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

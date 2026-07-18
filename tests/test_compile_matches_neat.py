"""Cross-checks compile_genome()'s vectorized torch evaluation against
neat-python's own (slow, per-point) FeedForwardNetwork.activate(). This is
the correctness gate for the whole custom-compile approach used to make
population-wide fitness evaluation fast: if these two ever diverge, the
evolution loop is silently optimizing against a different function than the
genome actually encodes."""

import neat
import torch

from src.cppn.compile import compile_genome
from src.cppn.coords import make_coord_grid
from src.cppn.evolve import create_random_genome, load_neat_config

NEAT_CONFIG_PATH = "configs/neat/cppn_neat_smoke.cfg"


def test_compile_matches_neat_activate():
    config = load_neat_config(NEAT_CONFIG_PATH)
    coords = make_coord_grid(4, 4, 1)  # 16 points x 4 features, small on purpose

    for seed in range(8):
        genome = create_random_genome(config, seed)
        for _ in range(3):
            genome.mutate(config.genome_config)

        compiled = compile_genome(genome, config.genome_config, coords)  # [N, 1], sigmoid-squashed

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        expected_raw = torch.tensor([net.activate(point.tolist()) for point in coords])  # [N, 1]
        expected = torch.sigmoid(expected_raw)

        assert torch.allclose(compiled, expected, atol=1e-5), (
            f"compile_genome diverged from neat's own activate() for seed {seed}: "
            f"max abs diff = {(compiled - expected).abs().max().item()}"
        )


def test_compile_handles_orphan_output_node():
    """A freshly-created genome under initial_connection=full always has
    incoming connections to its output node, but structural mutation can
    disable/delete all of them, leaving an output with no inputs (falls back
    to activation(bias), matching neat's own sum([])==0.0 convention)."""
    config = load_neat_config(NEAT_CONFIG_PATH)
    genome = create_random_genome(config, seed=0)
    for cg in genome.connections.values():
        cg.enabled = False

    coords = make_coord_grid(3, 3, 1)
    compiled = compile_genome(genome, config.genome_config, coords)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    expected = torch.sigmoid(torch.tensor([net.activate(point.tolist()) for point in coords]))

    assert torch.allclose(compiled, expected, atol=1e-5)

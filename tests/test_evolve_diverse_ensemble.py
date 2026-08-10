import torch

from src.cppn.compile import compile_genome
from src.cppn.coords import make_coord_grid, reshape_pattern
from src.cppn.evolve import create_random_genome, load_neat_config, select_diverse_ensemble

NEAT_CONFIG = load_neat_config("configs/neat/cppn_neat_smoke.cfg")
IMAGE_SIZE = 8
CHANNELS = 3
DEVICE = torch.device("cpu")
COORD_GRID = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, CHANNELS, device=DEVICE)


def _pattern(genome):
    flat = compile_genome(genome, NEAT_CONFIG.genome_config, COORD_GRID)
    return reshape_pattern(flat, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)


def test_default_min_distance_preserves_plain_top_k_by_fitness():
    # Reproduces attempt 12's exact scenario: the same genome appears at
    # multiple ranks in the pool. With min_pattern_distance=0.0 (the
    # default), selection must be unaffected -- a plain top-k slice,
    # duplicates included, matching every prior attempt's behavior exactly.
    genome_a = create_random_genome(NEAT_CONFIG, seed=0)
    genome_b = create_random_genome(NEAT_CONFIG, seed=1)
    pool = [(0.9, genome_a), (0.8, genome_a), (0.7, genome_a), (0.6, genome_b)]
    selected = select_diverse_ensemble(
        pool, top_k=3, genome_config=NEAT_CONFIG.genome_config, coord_grid=COORD_GRID,
        image_size=IMAGE_SIZE, channels=CHANNELS, min_pattern_distance=0.0,
    )
    assert [f for f, _g in selected] == [0.9, 0.8, 0.7]


def test_nonzero_min_distance_filters_out_duplicate_genomes():
    genome_a = create_random_genome(NEAT_CONFIG, seed=0)
    genome_b = create_random_genome(NEAT_CONFIG, seed=1)
    real_distance = (_pattern(genome_a) - _pattern(genome_b)).abs().mean().item()
    assert real_distance > 1e-4  # sanity: seeds 0/1 produce meaningfully different genomes

    # genome_a occupies the top 3 fitness ranks (the attempt-12 bug); genome_b
    # is 4th. A diversity-aware selection of 2 should skip the duplicate
    # genome_a entries in favor of genome_b, not return [genome_a, genome_a].
    pool = [(0.9, genome_a), (0.8, genome_a), (0.7, genome_a), (0.6, genome_b)]
    selected = select_diverse_ensemble(
        pool, top_k=2, genome_config=NEAT_CONFIG.genome_config, coord_grid=COORD_GRID,
        image_size=IMAGE_SIZE, channels=CHANNELS, min_pattern_distance=real_distance / 2,
    )
    selected_genomes = [g for _f, g in selected]
    assert selected_genomes[0] is genome_a
    assert selected_genomes[1] is genome_b


def test_falls_back_to_next_best_when_no_candidate_meets_threshold():
    genome = create_random_genome(NEAT_CONFIG, seed=0)
    pool = [(0.9, genome), (0.8, genome), (0.7, genome)]
    selected = select_diverse_ensemble(
        pool, top_k=3, genome_config=NEAT_CONFIG.genome_config, coord_grid=COORD_GRID,
        image_size=IMAGE_SIZE, channels=CHANNELS, min_pattern_distance=999.0,
    )
    # every candidate is identical, so no distance threshold can ever be
    # satisfied -- must still fill all 3 slots via fallback, not crash or
    # silently return fewer than top_k.
    assert len(selected) == 3


def test_empty_pool_returns_empty_list():
    selected = select_diverse_ensemble(
        [], top_k=5, genome_config=NEAT_CONFIG.genome_config, coord_grid=COORD_GRID,
        image_size=IMAGE_SIZE, channels=CHANNELS,
    )
    assert selected == []


def test_candidate_pool_size_limits_the_search_window():
    # A genuinely diverse but low-fitness genome sitting outside the
    # candidate_pool_size window must never be selected, even though it
    # would satisfy the distance threshold -- confirms the window is
    # actually applied before greedy selection, not just a hint.
    diverse_low_fitness_genome = create_random_genome(NEAT_CONFIG, seed=2)
    high_fitness_dupes = [(1.0 - i * 0.001, create_random_genome(NEAT_CONFIG, seed=0)) for i in range(10)]
    pool = high_fitness_dupes + [(0.0, diverse_low_fitness_genome)]
    selected = select_diverse_ensemble(
        pool, top_k=3, genome_config=NEAT_CONFIG.genome_config, coord_grid=COORD_GRID,
        image_size=IMAGE_SIZE, channels=CHANNELS, min_pattern_distance=0.5, candidate_pool_size=5,
    )
    selected_genomes = [g for _f, g in selected]
    assert diverse_low_fitness_genome not in selected_genomes
    assert len(selected) == 3

import copy
from pathlib import Path

import neat
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.cppn.apply import apply_pattern
from src.cppn.compile import compile_genome
from src.cppn.coords import make_coord_grid, reshape_pattern
from src.cppn.fitness import agreement_term, diversity_term, fitness_from_terms
from src.data.datasets import get_probe_batch
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_evolution(
    teacher: nn.Module,
    probe_dataset: Dataset,
    neat_config_path: str,
    image_size: int,
    channels: int,
    view_op: str,
    view_scale: float,
    tau_low: float,
    tau_high: float,
    gamma: float,
    probe_batch_size: int,
    num_generations: int,
    seed: int,
    device: torch.device,
    log_dir: str | Path,
    top_k: int = 5,
):
    """Evolves a population of CPPN genomes whose compiled patterns, applied
    to a probe batch of real images, maximize the gated diversity/agreement
    fitness in src/cppn/fitness.py. Uses a frozen teacher's forward pass only
    — no per-genome training loop (see plan: cheap-proxy fitness).

    Returns (best_genome, neat_config, evolution_log_df, generation_summary_df,
    top_k_genomes) where top_k_genomes is a list of (fitness, genome) pairs
    for the highest-scoring genomes seen across the whole run (for the
    ensemble ablation), since neat-python's Population only tracks a single
    best-ever genome.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_path),
    )

    coord_grid = make_coord_grid(image_size, image_size, channels, device=device)

    was_training = teacher.training
    teacher.eval()

    evolution_rows: list[dict] = []
    generation_summary_rows: list[dict] = []
    top_k_pool: list[tuple[float, object]] = []  # (fitness, deep-copied genome)
    generation_counter = {"n": 0}

    def eval_genomes(genomes: list[tuple[int, object]], cfg: neat.Config) -> None:
        gen = generation_counter["n"]
        images_raw01, _labels = get_probe_batch(
            probe_dataset, probe_batch_size, seed=seed + gen, device=device
        )

        with torch.no_grad():
            logits_raw, features_raw = teacher(images_raw01, return_features=True)

        fitnesses = []
        for genome_id, genome in genomes:
            pattern_flat = compile_genome(genome, cfg.genome_config, coord_grid)  # [N, 1]
            pattern = reshape_pattern(pattern_flat, image_size, image_size, channels)
            view = apply_pattern(images_raw01, pattern, mode=view_op, scale=view_scale)

            with torch.no_grad():
                logits_view, features_view = teacher(view, return_features=True)

            diversity = diversity_term(features_raw, features_view)
            agreement = agreement_term(logits_raw, logits_view)
            num_connections = sum(1 for cg in genome.connections.values() if cg.enabled)
            fitness = fitness_from_terms(diversity, agreement, tau_low, tau_high, gamma, num_connections)

            genome.fitness = fitness
            fitnesses.append(fitness)

            evolution_rows.append(
                {
                    "generation": gen,
                    "genome_id": genome_id,
                    "fitness": fitness,
                    "diversity": diversity,
                    "agreement": agreement,
                    "num_nodes": len(genome.nodes),
                    "num_connections": num_connections,
                }
            )
            top_k_pool.append((fitness, copy.deepcopy(genome)))

        fitnesses_t = torch.tensor(fitnesses)
        generation_summary_rows.append(
            {
                "generation": gen,
                "best_fitness": fitnesses_t.max().item(),
                "mean_fitness": fitnesses_t.mean().item(),
                "std_fitness": fitnesses_t.std(unbiased=False).item(),
            }
        )
        log.info(
            "generation %d: best=%.4f mean=%.4f",
            gen,
            fitnesses_t.max().item(),
            fitnesses_t.mean().item(),
        )
        generation_counter["n"] += 1

    population = neat.Population(config, seed=seed)
    best_genome = population.run(eval_genomes, num_generations)

    teacher.train(was_training)

    top_k_pool.sort(key=lambda pair: pair[0], reverse=True)
    top_k_genomes = top_k_pool[:top_k]

    evolution_log_df = pd.DataFrame(evolution_rows)
    generation_summary_df = pd.DataFrame(generation_summary_rows)
    evolution_log_df.to_csv(log_dir / "evolution_log.csv", index=False)
    generation_summary_df.to_csv(log_dir / "generation_summary.csv", index=False)

    return best_genome, config, evolution_log_df, generation_summary_df, top_k_genomes


def load_neat_config(neat_config_path: str) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_path),
    )


def create_random_genome(config: neat.Config, seed: int):
    """A single freshly-initialized genome from the same architecture family
    (coordinate inputs, per-node activation drawn from the same
    activation_options, same initial_connection scheme) as the evolved ones —
    but never scored or selected. Used for the `kd_random_cppn
    --random-cppn-variant coord` baseline that isolates "does evolution help"
    from "does having any coordinate-CPPN view help".

    neat-python 2.0.0 requires an innovation tracker on `genome_config`
    before any genome can be built via `configure_new` — normally installed
    as a side effect of `DefaultReproduction.__init__`, which only runs when
    a `Population` is constructed. There's no public standalone genome
    factory, so we construct a throwaway `Population` (which builds a full
    pop_size batch of freshly-initialized genomes as part of its own
    __init__) and just take one of its genomes, rather than reimplementing
    reproduction's genome-construction internals ourselves.
    """
    population = neat.Population(config, seed=seed)
    genome_id = next(iter(population.population))
    return population.population[genome_id]

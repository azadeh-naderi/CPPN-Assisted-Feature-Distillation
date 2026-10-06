"""Offline CPPN view search for the Phase 1 baselines. Each source returns a
list of fixed patterns [H, W, C] in [0, 1], used by trainer mode "kd_view":

- random:        one freshly initialized, never-scored genome (the main
                 pipeline's kd_random_cppn, 'coord' variant)
- trained:       one gradient-trained coordinate CPPN (kd_trained_cppn)
- evolved:       NEAT with the attempt-14 fitness and diverse top-K
                 ensemble selection (kd_evolved_cppn)
- random_search: the same fitness and ensemble selection over randomly
                 mutated genomes, at the same number of genome evaluations
                 as evolution (pop_size x num_generations). Separates "does
                 the evolutionary search matter" from "does fitness-based
                 selection matter".

All sources reuse src/cppn. One protocol fix relative to the main pipeline:
the teacher is wrapped in `NormalizedModel`, so view search scores views
on normalized inputs, as the teacher saw during training. The main
pipeline's run_evolution and train_trainable_cppn pass raw [0, 1] images
straight to a teacher trained on normalized ones.
"""

import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pattern PNGs are only written to disk; no GUI backend

import pandas as pd
import torch
import torch.nn as nn

from method_paper.src.data import normalize
from src.cppn.coords import make_coord_grid
from src.cppn.compile import genome_to_pattern
from src.cppn.evolve import (
    create_random_genome,
    load_neat_config,
    run_evolution,
    score_genome,
    select_diverse_ensemble,
)
from src.cppn.serialize import save_pattern
from src.cppn.trainable import train_trainable_cppn
from src.data.datasets import get_probe_batch
from src.utils.logging import get_logger

log = get_logger(__name__)

VIEW_SOURCES = ("random", "trained", "evolved", "random_search")
IMAGE_SIZE, CHANNELS = 32, 3

_FITNESS_KEYS = (
    "tau_low", "tau_high", "gamma", "contrast_penalty", "contrast_std_threshold",
    "channel_divergence_penalty", "min_connections", "min_pattern_std",
)


class NormalizedModel(nn.Module):
    """Takes raw [0, 1] images, normalizes them with the dataset's statistics,
    and forwards to `model` (including return_features)."""

    def __init__(self, model: nn.Module, dataset: str):
        super().__init__()
        self.model = model
        self.dataset = dataset

    def forward(self, images_raw01: torch.Tensor, return_features: bool = False):
        return self.model(normalize(images_raw01, self.dataset), return_features=return_features)


def random_search(
    teacher_raw01: nn.Module,
    probe_dataset,
    neat_config,
    view_op: str,
    view_scale: float,
    fitness_kwargs: dict,
    probe_batch_size: int,
    num_batches: int,
    batch_evals: int,
    max_mutations: int,
    seed: int,
    device: torch.device,
) -> tuple[list[tuple[float, object]], pd.DataFrame]:
    """Scores num_batches x batch_evals random genomes with the evolution
    fitness. Each genome is a fresh genome from the evolution config with
    m ~ Uniform{0..max_mutations} random mutations applied, so the sampled
    topologies span roughly the complexity evolution reaches, rather than
    only the minimal initial topology. A fresh probe batch is drawn every
    batch_evals genomes, matching evolution's per-generation resampling.

    Returns ([(fitness, genome), ...], per-genome log)."""
    # Population's constructor seeds python's `random` (which NEAT's
    # mutations draw from) and installs the innovation tracker that
    # configure_new needs; the population itself is discarded.
    create_random_genome(neat_config, seed)
    genome_config = neat_config.genome_config
    coord_grid = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, CHANNELS, device=device)

    was_training = teacher_raw01.training
    teacher_raw01.eval()
    pool, rows = [], []
    key = 0
    for batch_idx in range(num_batches):
        images_raw01, _ = get_probe_batch(probe_dataset, probe_batch_size, seed=seed + batch_idx, device=device)
        with torch.no_grad():
            logits_raw, features_raw = teacher_raw01(images_raw01, return_features=True)
        for _ in range(batch_evals):
            genome = neat_config.genome_type(key)
            genome.configure_new(genome_config)
            num_mutations = random.randint(0, max_mutations)
            for _ in range(num_mutations):
                genome.mutate(genome_config)
            terms = score_genome(
                genome, genome_config, coord_grid, IMAGE_SIZE, CHANNELS, teacher_raw01,
                images_raw01, logits_raw, features_raw, view_op, view_scale, fitness_kwargs,
            )
            genome.fitness = terms["fitness"]
            pool.append((terms["fitness"], copy.deepcopy(genome)))
            rows.append({"batch": batch_idx, "genome_id": key, "num_mutations": num_mutations, **terms})
            key += 1
        log.info("random search batch %d/%d: best so far %.4f", batch_idx + 1, num_batches,
                 max(f for f, _ in pool))
    teacher_raw01.train(was_training)
    return pool, pd.DataFrame(rows)


def build_views(
    source: str,
    teacher: nn.Module,
    dataset: str,
    probe_dataset,
    vcfg: dict,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> list[torch.Tensor]:
    """Runs view search for `source` against `teacher` (a model taking
    normalized inputs) and returns the patterns on `device`. Logs and
    patterns (.pt + .png) are written to out_dir."""
    if source not in VIEW_SOURCES:
        raise ValueError(f"Unknown view source {source!r}. Choose one of {VIEW_SOURCES}.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_raw01 = NormalizedModel(teacher, dataset).to(device).eval()
    neat_config = load_neat_config(vcfg["neat_config"])
    ecfg = vcfg["evolution"]
    fitness_kwargs = {k: ecfg[k] for k in _FITNESS_KEYS}

    if source == "random":
        genome = create_random_genome(neat_config, seed)
        patterns = [genome_to_pattern(genome, neat_config.genome_config, IMAGE_SIZE, CHANNELS, device)]
    elif source == "trained":
        tcfg = vcfg["trained"]
        patterns = [train_trainable_cppn(
            teacher_raw01, probe_dataset, IMAGE_SIZE, CHANNELS, vcfg["view_op"], vcfg["view_scale"],
            num_steps=tcfg["num_steps"], probe_batch_size=vcfg["probe_batch_size"], lr=tcfg["lr"],
            temperature=tcfg["temperature"], diversity_weight=tcfg["diversity_weight"],
            kl_weight=tcfg["kl_weight"], seed=seed, device=device,
        )]
    else:
        if source == "evolved":
            _best, _cfg, _log, _summary, ensemble = run_evolution(
                teacher_raw01, probe_dataset, vcfg["neat_config"], IMAGE_SIZE, CHANNELS,
                vcfg["view_op"], vcfg["view_scale"], probe_batch_size=vcfg["probe_batch_size"],
                num_generations=ecfg["num_generations"], seed=seed, device=device, log_dir=out_dir,
                top_k=ecfg["top_k"], min_pattern_distance=ecfg["min_pattern_distance"], **fitness_kwargs,
            )
        else:
            pool, search_log = random_search(
                teacher_raw01, probe_dataset, neat_config, vcfg["view_op"], vcfg["view_scale"],
                fitness_kwargs, probe_batch_size=vcfg["probe_batch_size"],
                num_batches=ecfg["num_generations"], batch_evals=neat_config.pop_size,
                max_mutations=vcfg["random_search"]["max_mutations"], seed=seed, device=device,
            )
            search_log.to_csv(out_dir / "random_search_log.csv", index=False)
            coord_grid = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, CHANNELS, device=device)
            ensemble = select_diverse_ensemble(
                pool, ecfg["top_k"], neat_config.genome_config, coord_grid, IMAGE_SIZE, CHANNELS,
                min_pattern_distance=ecfg["min_pattern_distance"],
            )
        patterns = [
            genome_to_pattern(g, neat_config.genome_config, IMAGE_SIZE, CHANNELS, device) for _f, g in ensemble
        ]
        pd.DataFrame(
            [{"rank": i, "fitness": f, "genome_id": g.key} for i, (f, g) in enumerate(ensemble)]
        ).to_csv(out_dir / "selected_genomes.csv", index=False)

    for i, pattern in enumerate(patterns):
        save_pattern(pattern, out_dir / f"pattern_{i}.pt", out_dir / f"pattern_{i}.png")
    log.info("%s view search: %d pattern(s), std %s", source, len(patterns),
             [round(p.std().item(), 3) for p in patterns])
    return [p.to(device) for p in patterns]

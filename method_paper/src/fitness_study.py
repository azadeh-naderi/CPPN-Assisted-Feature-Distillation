"""Fitness-vs-accuracy study (analysis paper): does the teacher-only view
fitness predict how much a view helps or hurts the student?

1. `build_bank`: run NEAT and an equal-budget random search with the
   Phase 1 fitness (attempt-14 settings) against the frozen teacher, pick
   genomes spread over the fitness range from both pools, and rescore every
   pick on one fixed probe set so all terms are comparable (during search
   each generation used a different probe batch).
2. Short student runs, one per bank view (train.py mode `kd_bank_view`,
   launch.py stage `fitness_study`).
3. `scripts/fitness_study.py analyze`: correlate fitness and each of its
   terms with the student's accuracy change over short-schedule `kd`.
"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only written to disk

import matplotlib.pyplot as plt
import neat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from method_paper.src.views import (
    _FITNESS_KEYS,
    CHANNELS,
    IMAGE_SIZE,
    NormalizedModel,
    random_search,
)
from src.cppn.apply import apply_pattern
from src.cppn.compile import genome_to_pattern
from src.cppn.coords import make_coord_grid
from src.cppn.evolve import load_neat_config, score_genome
from src.cppn.fitness import DISQUALIFIED_FITNESS
from src.data.datasets import get_probe_batch
from src.utils.logging import get_logger

log = get_logger(__name__)

RESCORE_SEED = 12345  # fixed probe set shared by every bank genome


def bank_dir(results_root: Path, dataset: str) -> Path:
    return Path(results_root) / dataset / "fitness_study" / "bank"


def load_bank_patterns(results_root: Path, dataset: str) -> torch.Tensor:
    """[N, H, W, C] patterns, in bank index order."""
    return torch.load(bank_dir(results_root, dataset) / "patterns.pt")


def _neat_pool(teacher_raw01, probe_dataset, neat_config, vcfg, fitness_kwargs, generations, seed, device):
    """Every genome NEAT scores, as [(fitness, genome)], with the Phase 1 fitness."""
    coord_grid = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, CHANNELS, device=device)
    pool, counter = [], {"gen": 0}

    def eval_genomes(genomes, cfg):
        images, _ = get_probe_batch(probe_dataset, vcfg["probe_batch_size"], seed=seed + counter["gen"],
                                    device=device)
        with torch.no_grad():
            logits_raw, features_raw = teacher_raw01(images, return_features=True)
        for _gid, genome in genomes:
            terms = score_genome(genome, cfg.genome_config, coord_grid, IMAGE_SIZE, CHANNELS, teacher_raw01, images,
                                 logits_raw, features_raw, vcfg["view_op"], vcfg["view_scale"], fitness_kwargs)
            genome.fitness = terms["fitness"]
            pool.append((terms["fitness"], copy.deepcopy(genome)))
        counter["gen"] += 1

    neat.Population(neat_config, seed=seed).run(eval_genomes, generations)
    return pool


def stratified_pick(pool: list[tuple[float, object]], k: int) -> list[tuple[float, object]]:
    """k genomes at evenly spaced fitness quantiles of `pool`, skipping
    disqualified genomes. A genome NEAT re-evaluates in several generations
    (elites keep their key) counts once, at its latest fitness."""
    latest = {}
    for fitness, genome in pool:
        if fitness > DISQUALIFIED_FITNESS / 2:
            latest[genome.key] = (fitness, genome)
    ranked = sorted(latest.values(), key=lambda fg: fg[0])
    if len(ranked) <= k:
        return ranked
    positions = np.unique(np.round(np.linspace(0, len(ranked) - 1, k)).astype(int))
    return [ranked[i] for i in positions]


@torch.no_grad()
def rescore(genome, neat_config, teacher_raw01, images, labels, logits_raw, features_raw, vcfg, fitness_kwargs,
            coord_grid, device) -> tuple[dict, torch.Tensor]:
    """Fitness terms on the shared probe set, plus pattern and teacher-accuracy
    descriptors. Returns (terms, pattern)."""
    terms = score_genome(genome, neat_config.genome_config, coord_grid, IMAGE_SIZE, CHANNELS, teacher_raw01, images,
                         logits_raw, features_raw, vcfg["view_op"], vcfg["view_scale"], fitness_kwargs)
    pattern = genome_to_pattern(genome, neat_config.genome_config, IMAGE_SIZE, CHANNELS, device)
    view = apply_pattern(images, pattern, vcfg["view_op"], vcfg["view_scale"])
    teacher_raw_acc = (logits_raw.argmax(1) == labels).float().mean().item()
    teacher_view_acc = (teacher_raw01(view).argmax(1) == labels).float().mean().item()
    terms.update(
        pattern_mean=pattern.mean().item(),
        teacher_raw_acc=teacher_raw_acc,
        teacher_view_acc=teacher_view_acc,
        rel_teacher_acc=teacher_view_acc / max(teacher_raw_acc, 1e-6),
    )
    return terms, pattern


def build_bank(
    teacher: nn.Module,
    dataset: str,
    probe_dataset,
    vcfg: dict,
    num_genomes: int,
    generations: int,
    rescore_probe_size: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    """Writes patterns.pt, bank.csv and bank.png to out_dir; returns the table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_raw01 = NormalizedModel(teacher, dataset).to(device).eval()
    neat_config = load_neat_config(vcfg["neat_config"])
    ecfg = vcfg["evolution"]
    fitness_kwargs = {k: ecfg[k] for k in _FITNESS_KEYS}

    log.info("bank: NEAT, %d generations x %d", generations, neat_config.pop_size)
    evolved = _neat_pool(teacher_raw01, probe_dataset, neat_config, vcfg, fitness_kwargs, generations, seed, device)
    log.info("bank: random search, same budget")
    randoms, _log = random_search(teacher_raw01, probe_dataset, neat_config, vcfg["view_op"], vcfg["view_scale"],
                                  fitness_kwargs, vcfg["probe_batch_size"], generations, neat_config.pop_size,
                                  vcfg["random_search"]["max_mutations"], seed + 1, device)
    picks = [("evolved", fg) for fg in stratified_pick(evolved, num_genomes // 2)]
    picks += [("random", fg) for fg in stratified_pick(randoms, num_genomes - num_genomes // 2)]

    images, labels = get_probe_batch(probe_dataset, rescore_probe_size, seed=RESCORE_SEED, device=device)
    with torch.no_grad():
        logits_raw, features_raw = teacher_raw01(images, return_features=True)
    coord_grid = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, CHANNELS, device=device)

    rows, patterns = [], []
    for index, (source, (search_fitness, genome)) in enumerate(picks):
        terms, pattern = rescore(genome, neat_config, teacher_raw01, images, labels, logits_raw, features_raw, vcfg,
                                 fitness_kwargs, coord_grid, device)
        rows.append({"index": index, "source": source, "search_fitness": search_fitness, **terms})
        patterns.append(pattern.cpu())

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "bank.csv", index=False)
    torch.save(torch.stack(patterns), out_dir / "patterns.pt")
    _plot_bank(torch.stack(patterns), table, out_dir / "bank.png")
    log.info("bank: %d genomes (%d evolved, %d random), rescored fitness %.3f .. %.3f", len(table),
             (table.source == "evolved").sum(), (table.source == "random").sum(), table.fitness.min(),
             table.fitness.max())
    return table


def _plot_bank(patterns: torch.Tensor, table: pd.DataFrame, path: Path) -> None:
    order = table.sort_values("fitness").index.tolist()
    cols = 10
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), order):
        ax.imshow(patterns[i].numpy(), vmin=0, vmax=1)
        ax.set_title(f"#{i} {table.source[i][0]} {table.fitness[i]:.2f}", fontsize=6)
    fig.suptitle("bank views sorted by rescored fitness (e = evolved, r = random)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

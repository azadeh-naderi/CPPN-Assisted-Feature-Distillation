"""Phase 2: student-aware online view evolution.

Instead of evolving views once against the frozen teacher before training
(Phase 1's kd_evolved_cppn), one NEAT population persists for the whole run
and is re-evolved for a few generations every `evolve_every` epochs, scored
against the *current* student. The top-K diverse genomes of each round
become the views for the following epochs; the trainer samples one per
batch with a random shift/flip (trainer.py, view_sampling="one").

Fitness options (the main Phase 2 ablation axis):
- "disagreement": KL(teacher(view) || student(view)) * T^2, averaged over
  the probe images the teacher still classifies correctly on the view --
  views the student hasn't learned yet but the teacher gets right.
- "gradalign": cosine between the student's last-layer gradient of the KD
  loss on the view and of the CE loss on a separate batch of training
  images -- does a step on this view also reduce CE elsewhere? Mapped to
  [0, 1] as (cos + 1) / 2 so the gate below keeps its meaning.
- "teacher": the old teacher-only fitness (feature diversity gated on soft
  agreement), re-evolved online -- isolates "student-aware" from "online".

Every fitness goes through src.cppn.fitness.fitness_from_terms with the
existing guardrails (min_connections / min_pattern_std disqualification,
contrast and channel-divergence penalties). For the two student-aware
fitnesses the gate input is the teacher's accuracy on the view relative to
its accuracy on the raw probe images, so views that break the teacher score
~0 however much the student disagrees.

Probe and CE batches are unaugmented training images, never the test split.

Phase 3 controls (config `online.search` / `online.family`):
- search="neat" (default): the method as above.
- search="random": each round scores generations x pop_size freshly sampled
  random genomes with the same fitness and keeps the diverse top-K -- the
  online "does evolution matter" control at equal evaluation budget.
- search="none": each round takes top_k fresh random genomes that pass the
  disqualification floors, unscored -- "does selection matter at all".
- family="noise": each round draws top_k low-frequency noise masks (uniform
  noise on an 8x8 grid per channel, bilinearly upsampled) instead of CPPN
  patterns, with no search -- "does the CPPN family matter".
- family="noise" + noise_selection="random": noise masks scored and
  selected like search="random" (same fitness, gate and budget), separating
  "noise is a bad family" from "unselected views are bad".
All controls keep the same per-batch sampling, jitter and loss.
"""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pattern PNGs are only written to disk; no GUI backend

import neat
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from method_paper.src.views import CHANNELS, IMAGE_SIZE, NormalizedModel, sample_random_genome
from src.cppn.apply import apply_pattern
from src.cppn.compile import compile_genome, genome_to_pattern
from src.cppn.coords import make_coord_grid, reshape_pattern
from src.cppn.evolve import load_neat_config, select_diverse_ensemble
from src.cppn.fitness import (
    channel_divergence_term,
    diversity_term,
    fitness_from_terms,
    soft_agreement_term,
)
from src.cppn.serialize import save_pattern
from src.data.datasets import get_probe_batch
from src.utils.logging import get_logger

log = get_logger(__name__)

FITNESSES = ("disagreement", "gradalign", "teacher")
SEARCHES = ("neat", "random", "none")
FAMILIES = ("cppn", "noise")
NOISE_SELECTIONS = ("none", "random")
NOISE_GRID = 8

_PENALTY_KEYS = (
    "gamma", "contrast_penalty", "contrast_std_threshold",
    "channel_divergence_penalty", "min_connections", "min_pattern_std",
)


def per_sample_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(softmax(t/T) || softmax(s/T)) * T^2 per sample -- the per-sample
    version of src.distill.losses.kd_loss."""
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1)
    return (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_p_s)).sum(dim=1) * temperature**2


def last_layer_grad(dlogits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Gradient of a batch-mean loss w.r.t. a final Linear layer's
    [weight, bias], flattened, given dL/dlogits per sample (already divided
    by the batch size) and the layer's input features."""
    return torch.cat([(dlogits.T @ features).flatten(), dlogits.sum(dim=0)])


def kd_dlogits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """d kd_loss / d student_logits for the batch-mean KD loss:
    T * (softmax(s/T) - softmax(t/T)) / B."""
    batch = student_logits.shape[0]
    diff = F.softmax(student_logits / temperature, dim=1) - F.softmax(teacher_logits / temperature, dim=1)
    return temperature * diff / batch


def ce_dlogits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """d cross_entropy / d logits for the batch-mean CE loss."""
    return (F.softmax(logits, dim=1) - F.one_hot(labels, logits.shape[1]).float()) / logits.shape[0]


def select_diverse_patterns(pool: list[tuple[float, torch.Tensor]], top_k: int, min_pattern_distance: float = 0.0,
                            candidate_pool_size: int = 50) -> list[tuple[float, torch.Tensor]]:
    """select_diverse_ensemble for raw patterns (noise masks have no genome):
    greedily take the fittest candidate at least `min_pattern_distance` (mean
    absolute pixel difference) from every pick, else the next fittest."""
    candidates = sorted(pool, key=lambda fp: fp[0], reverse=True)[:candidate_pool_size]
    picked: list[int] = []
    for _ in range(min(top_k, len(candidates))):
        remaining = [i for i in range(len(candidates)) if i not in picked]
        far = [i for i in remaining
               if all((candidates[i][1] - candidates[j][1]).abs().mean().item() >= min_pattern_distance for j in picked)]
        picked.append((far or remaining)[0])
    return [candidates[i] for i in picked]


class OnlineViewEvolver:
    """One NEAT population that persists across rounds (warm start). Each
    `evolve(student, generations)` call runs that many more generations
    scored against the given student and returns the round's top-K diverse
    patterns. Genomes are compared only within a round: the student changes
    between rounds, so fitness from earlier rounds is stale."""

    def __init__(
        self,
        teacher: nn.Module,
        dataset: str,
        probe_dataset,
        neat_config_path: str,
        fitness: str,
        ocfg: dict,
        ecfg: dict,
        view_op: str,
        view_scale: float,
        probe_batch_size: int,
        temperature: float,
        seed: int,
        device: torch.device,
        out_dir: Path,
        max_mutations: int = 40,
    ):
        if fitness not in FITNESSES:
            raise ValueError(f"Unknown fitness {fitness!r}. Choose one of {FITNESSES}.")
        self.search = ocfg.get("search", "neat")
        self.family = ocfg.get("family", "cppn")
        if self.search not in SEARCHES:
            raise ValueError(f"Unknown online.search {self.search!r}. Choose one of {SEARCHES}.")
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown online.family {self.family!r}. Choose one of {FAMILIES}.")
        self.noise_selection = ocfg.get("noise_selection", "none")
        if self.noise_selection not in NOISE_SELECTIONS:
            raise ValueError(f"Unknown online.noise_selection {self.noise_selection!r}. "
                             f"Choose one of {NOISE_SELECTIONS}.")
        self.max_mutations = max_mutations
        self.min_connections = ecfg["min_connections"]
        self.min_pattern_std = ecfg["min_pattern_std"]
        self.teacher = NormalizedModel(teacher, dataset).to(device).eval()
        self.dataset = dataset
        self.probe_dataset = probe_dataset
        self.fitness = fitness
        self.top_k = ocfg["top_k"]
        self.min_pattern_distance = ecfg["min_pattern_distance"]
        # Gate thresholds: on soft agreement for the teacher-only fitness (as
        # in Phase 1), on relative teacher accuracy for the student-aware ones.
        gate = ecfg if fitness == "teacher" else ocfg["gate"]
        self.fitness_kwargs = {
            "tau_low": gate["tau_low"], "tau_high": gate["tau_high"], **{k: ecfg[k] for k in _PENALTY_KEYS}
        }
        self.view_op, self.view_scale = view_op, view_scale
        self.probe_batch_size = probe_batch_size
        self.temperature = temperature
        self.seed = seed
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_neat_config(neat_config_path)
        self.population = neat.Population(self.config, seed=seed)
        self.coord_grid = make_coord_grid(IMAGE_SIZE, IMAGE_SIZE, CHANNELS, device=device)
        self.rows: list[dict] = []
        self.round = 0
        self._generation = 0  # scored generations so far; equals population.generation under NEAT
        self._next_key = 0

    @torch.no_grad()
    def _score_generation(self, genomes, student_raw01: nn.Module, pool: list, epoch: int) -> None:
        """Scores a generation of genomes; appends (fitness, genome copy) to pool."""
        items = []
        for genome_id, genome in genomes:
            pattern = reshape_pattern(compile_genome(genome, self.config.genome_config, self.coord_grid),
                                      IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
            num_connections = sum(1 for cg in genome.connections.values() if cg.enabled)
            items.append((genome_id, pattern, num_connections, genome))
        self._score_items(items, student_raw01, pool, epoch)

    @torch.no_grad()
    def _score_items(self, items, student_raw01: nn.Module, pool: list, epoch: int) -> None:
        """Scores (id, pattern, num_connections, genome-or-None) items on one
        fresh probe batch. Genomes get their fitness set and a copy appended
        to pool; items without a genome (noise masks) append the pattern."""
        gen = self._generation
        self._generation += 1
        images, labels = get_probe_batch(self.probe_dataset, self.probe_batch_size, seed=self.seed + 7919 * gen,
                                          device=self.device)
        t_logits_raw, t_feat_raw = self.teacher(images, return_features=True)
        teacher_acc_raw = (t_logits_raw.argmax(1) == labels).float().mean().clamp_min(1e-6)
        if self.fitness == "gradalign":
            ce_images, ce_labels = get_probe_batch(self.probe_dataset, self.probe_batch_size,
                                                   seed=self.seed + 7919 * gen + 1, device=self.device)
            s_logits_ce, s_feat_ce = student_raw01(ce_images, return_features=True)
            g_ce = last_layer_grad(ce_dlogits(s_logits_ce, ce_labels), s_feat_ce)

        for item_id, pattern, num_connections, genome in items:
            view = apply_pattern(images, pattern, self.view_op, self.view_scale)
            t_logits, t_feat = self.teacher(view, return_features=True)
            correct = t_logits.argmax(1) == labels
            rel_teacher_acc = (correct.float().mean() / teacher_acc_raw).clamp(max=1.0).item()

            if self.fitness == "teacher":
                score = diversity_term(t_feat_raw, t_feat)
                agreement = soft_agreement_term(t_logits_raw, t_logits)
            else:
                s_logits, s_feat = student_raw01(view, return_features=True)
                agreement = rel_teacher_acc
                if self.fitness == "disagreement":
                    kl = per_sample_kd(s_logits, t_logits, self.temperature)
                    score = (kl * correct).sum().item() / max(correct.sum().item(), 1)
                else:
                    g_view = last_layer_grad(kd_dlogits(s_logits, t_logits, self.temperature), s_feat)
                    score = (F.cosine_similarity(g_view, g_ce, dim=0).item() + 1.0) / 2.0

            pattern_std = pattern.std().item()
            fitness = fitness_from_terms(
                score, agreement, num_connections=num_connections, pattern_std=pattern_std,
                channel_divergence=channel_divergence_term(pattern), **self.fitness_kwargs,
            )
            if genome is not None:
                genome.fitness = fitness
                pool.append((fitness, copy.deepcopy(genome)))
            else:
                pool.append((fitness, pattern))
            self.rows.append({
                "round": self.round, "epoch": epoch, "generation": gen, "genome_id": item_id,
                "fitness": fitness, "score": score, "gate_input": agreement,
                "rel_teacher_acc": rel_teacher_acc, "num_connections": num_connections, "pattern_std": pattern_std,
            })

    def _random_genomes(self, n: int) -> list[tuple[int, object]]:
        genomes = []
        for _ in range(n):
            genome, _m = sample_random_genome(self.config, self._next_key, self.max_mutations)
            genomes.append((self._next_key, genome))
            self._next_key += 1
        return genomes

    def _unscored_random_patterns(self, max_tries: int = 1000) -> list[torch.Tensor]:
        """top_k fresh random genomes that pass the disqualification floors
        (min_connections, min_pattern_std), with no fitness scoring."""
        patterns = []
        for _ in range(max_tries):
            (_key, genome), = self._random_genomes(1)
            num_connections = sum(1 for cg in genome.connections.values() if cg.enabled)
            pattern = genome_to_pattern(genome, self.config.genome_config, IMAGE_SIZE, CHANNELS, self.device)
            if num_connections >= self.min_connections and pattern.std().item() >= self.min_pattern_std:
                patterns.append(pattern)
                if len(patterns) == self.top_k:
                    break
        if not patterns:
            raise RuntimeError(f"no random genome passed the floors in {max_tries} tries")
        return patterns

    def _noise_masks(self, n: int, seed: int) -> list[torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        low = torch.rand(n, CHANNELS, NOISE_GRID, NOISE_GRID, generator=generator)
        masks = F.interpolate(low, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        return [m.permute(1, 2, 0).to(self.device) for m in masks]  # [H, W, C] like a compiled CPPN

    def _noise_patterns(self) -> list[torch.Tensor]:
        return self._noise_masks(self.top_k, self.seed * 1000 + self.round)

    def _filtered_noise_patterns(self, student_raw01: nn.Module, generations: int, epoch: int):
        """Noise-mask counterpart of search="random": generations x pop_size
        noise masks scored with the same fitness, diverse top-K kept. Masks
        have no connections, so they're given exactly min_connections to stay
        clear of that (genome-only) floor; min_pattern_std still applies."""
        pool: list = []
        for _ in range(generations):
            masks = self._noise_masks(self.config.pop_size, self.seed * 1_000_003 + self._generation)
            items = [(self._next_key + i, m, self.min_connections, None) for i, m in enumerate(masks)]
            self._next_key += len(masks)
            self._score_items(items, student_raw01, pool, epoch)
        ensemble = select_diverse_patterns(pool, self.top_k, self.min_pattern_distance)
        return [p for _f, p in ensemble], [round(f, 4) for f, _p in ensemble]

    def evolve(self, student: nn.Module, generations: int, epoch: int) -> list[torch.Tensor]:
        """Runs `generations` more generations scored against `student` (a
        model taking normalized inputs) and returns the top-K diverse
        patterns. The student is evaluated in eval mode and restored. The
        search="none" and family="noise" controls skip scoring entirely."""
        fitnesses: list[float] = []
        if self.family == "noise" and self.noise_selection == "random":
            student_raw01 = NormalizedModel(student, self.dataset)
            was_training = student.training
            student.eval()
            patterns, fitnesses = self._filtered_noise_patterns(student_raw01, generations, epoch)
            student.train(was_training)
            source = f"{generations} generations of scored noise masks"
        elif self.family == "noise":
            patterns, source = self._noise_patterns(), "noise masks"
        elif self.search == "none":
            patterns, source = self._unscored_random_patterns(), "unscored random genomes"
        else:
            student_raw01 = NormalizedModel(student, self.dataset)
            was_training = student.training
            student.eval()
            pool: list = []
            if self.search == "neat":
                self.population.run(
                    lambda genomes, _cfg: self._score_generation(genomes, student_raw01, pool, epoch), generations
                )
            else:  # random search, same number of evaluations as NEAT's generations x pop_size
                for _ in range(generations):
                    genomes = self._random_genomes(self.config.pop_size)
                    for _key, genome in genomes:
                        genome.fitness = None
                    self._score_generation(genomes, student_raw01, pool, epoch)
            student.train(was_training)

            ensemble = select_diverse_ensemble(pool, self.top_k, self.config.genome_config, self.coord_grid,
                                               IMAGE_SIZE, CHANNELS, min_pattern_distance=self.min_pattern_distance)
            patterns = [genome_to_pattern(g, self.config.genome_config, IMAGE_SIZE, CHANNELS, self.device)
                        for _f, g in ensemble]
            fitnesses = [round(f, 4) for f, _g in ensemble]
            source = f"{generations} generations ({self.search})"

        round_dir = self.out_dir / f"round{self.round:02d}_epoch{epoch:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for i, pattern in enumerate(patterns):
            save_pattern(pattern, round_dir / f"pattern_{i}.pt", round_dir / f"pattern_{i}.png")
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.out_dir / "online_evolution_log.csv", index=False)
        log.info("view round %d (epoch %d, %s): %s, selected fitness %s", self.round, epoch, self.fitness,
                 source, fitnesses or "n/a")
        self.round += 1
        return patterns

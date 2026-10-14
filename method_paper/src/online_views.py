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

from method_paper.src.views import CHANNELS, IMAGE_SIZE, NormalizedModel
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
    ):
        if fitness not in FITNESSES:
            raise ValueError(f"Unknown fitness {fitness!r}. Choose one of {FITNESSES}.")
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

    @torch.no_grad()
    def _score_generation(self, genomes, student_raw01: nn.Module, pool: list, epoch: int) -> None:
        gen = self.population.generation
        images, labels = get_probe_batch(self.probe_dataset, self.probe_batch_size, seed=self.seed + 7919 * gen,
                                          device=self.device)
        t_logits_raw, t_feat_raw = self.teacher(images, return_features=True)
        teacher_acc_raw = (t_logits_raw.argmax(1) == labels).float().mean().clamp_min(1e-6)
        if self.fitness == "gradalign":
            ce_images, ce_labels = get_probe_batch(self.probe_dataset, self.probe_batch_size,
                                                   seed=self.seed + 7919 * gen + 1, device=self.device)
            s_logits_ce, s_feat_ce = student_raw01(ce_images, return_features=True)
            g_ce = last_layer_grad(ce_dlogits(s_logits_ce, ce_labels), s_feat_ce)

        for genome_id, genome in genomes:
            pattern = reshape_pattern(compile_genome(genome, self.config.genome_config, self.coord_grid),
                                      IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
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

            num_connections = sum(1 for cg in genome.connections.values() if cg.enabled)
            pattern_std = pattern.std().item()
            fitness = fitness_from_terms(
                score, agreement, num_connections=num_connections, pattern_std=pattern_std,
                channel_divergence=channel_divergence_term(pattern), **self.fitness_kwargs,
            )
            genome.fitness = fitness
            pool.append((fitness, copy.deepcopy(genome)))
            self.rows.append({
                "round": self.round, "epoch": epoch, "generation": gen, "genome_id": genome_id,
                "fitness": fitness, "score": score, "gate_input": agreement,
                "rel_teacher_acc": rel_teacher_acc, "num_connections": num_connections, "pattern_std": pattern_std,
            })

    def evolve(self, student: nn.Module, generations: int, epoch: int) -> list[torch.Tensor]:
        """Runs `generations` more generations scored against `student` (a
        model taking normalized inputs) and returns the top-K diverse
        patterns. The student is evaluated in eval mode and restored."""
        student_raw01 = NormalizedModel(student, self.dataset)
        was_training = student.training
        student.eval()
        pool: list = []
        self.population.run(lambda genomes, _cfg: self._score_generation(genomes, student_raw01, pool, epoch),
                            generations)
        student.train(was_training)

        ensemble = select_diverse_ensemble(pool, self.top_k, self.config.genome_config, self.coord_grid,
                                           IMAGE_SIZE, CHANNELS, min_pattern_distance=self.min_pattern_distance)
        patterns = [genome_to_pattern(g, self.config.genome_config, IMAGE_SIZE, CHANNELS, self.device)
                    for _f, g in ensemble]

        round_dir = self.out_dir / f"round{self.round:02d}_epoch{epoch:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for i, pattern in enumerate(patterns):
            save_pattern(pattern, round_dir / f"pattern_{i}.pt", round_dir / f"pattern_{i}.png")
        pd.DataFrame(self.rows).to_csv(self.out_dir / "online_evolution_log.csv", index=False)
        fitnesses = [f for f, _g in ensemble]
        log.info("view round %d (epoch %d, %s): %d generations, selected fitness %s", self.round, epoch,
                 self.fitness, generations, [round(f, 4) for f in fitnesses])
        self.round += 1
        return patterns

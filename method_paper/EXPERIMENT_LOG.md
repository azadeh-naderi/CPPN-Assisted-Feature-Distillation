# Method paper — Experiment Log

Running record for the positive-method paper: a student-aware evolved
augmentation curriculum for KD. See [`README.md`](README.md) for layout and
why this uses a separate protocol from the main pipeline
(`../experiments/EXPERIMENT_LOG.md`).

## Plan and decision gates

1. **Phase 0 — protocol.** Reproduce published `student_only` and `kd`
   numbers within ~0.5 points on the development pairs. If that fails, fix
   it before anything else — every later comparison depends on it.
2. **Phase 1 — corrected baselines.** `kd_random_cppn`, `kd_trained_cppn`,
   `kd_evolved_cppn` (with attempt 14's diverse ensemble), KD + RandAugment
   / CutMix, and random search over genomes at the same evaluation budget
   as NEAT. *Gate 1:* note whether the existing `kd_evolved_cppn` already
   beats `kd` once the protocol is fixed.
3. **Phase 2 — new method.** Re-evolve views every N epochs from a
   warm-started population; sample one view per batch from the current
   top-K with random shift/flip; fitness either (a) teacher–student
   disagreement on the view, gated on the teacher still being correct, or
   (b) cosine alignment between the KD-loss gradient on the view and the
   CE gradient on a held-out batch. *Gate 2:* on ResNet56→ResNet20,
   CIFAR-100, 3 seeds, must beat `kd` by ≥ ~1 point and beat KD +
   augmentation and random search. Otherwise fall back to the analysis
   paper framing.
4. **Phase 3 — full results.** 4–6 pairs, ablations (N, sampling vs
   averaging, fitness a/b/old, evolution vs random search, CPPN vs noise
   masks), CIFAR-100-C robustness, ECE.

## Reference numbers (published)

CIFAR-100 top-1 (%), from the standard benchmark (Tian et al., "Contrastive
Representation Distillation", ICLR 2020), as listed in the RepDistiller
README. Protocol: 240 epochs, SGD lr 0.05 (0.01 for MobileNetV2/ShuffleNet),
decay ×0.1 at 150/180/210, weight decay 5e-4, batch 64, KD with T=4,
CE weight 0.1, KD weight 0.9.

| teacher → student | teacher | student | KD |
|---|---|---|---|
| resnet56 → resnet20 | 72.34 | 69.06 | 70.66 |
| wrn_40_2 → wrn_16_2 | 75.61 | 73.26 | 74.92 |
| resnet32x4 → resnet8x4 | 79.42 | 72.50 | 73.33 |
| vgg13 → MobileNetV2 | 74.64 | 64.60 | 67.37 |
| resnet110 → resnet20 | 74.31 | 69.06 | 70.67 |
| resnet110 → resnet32 | 74.31 | 71.14 | 73.08 |
| vgg13 → vgg8 | 74.64 | 70.36 | 72.98 |

The first four are the Phase 0 development pairs. Our runs log both
final-epoch test accuracy (primary — no test-set selection) and best-epoch
test accuracy; compare the latter against the table if final-epoch numbers
come in slightly lower, since published numbers may be best-epoch.

---

## Phase 3 ablations, new pairs, calibration/robustness (2026-10-15)

**New benchmark pairs reproduce** (final / best vs published KD):
resnet110 → resnet32 `kd` 72.79 / 73.12 vs 73.08 (student_only 71.38);
vgg13 → vgg8 `kd` 72.62 / 73.03 vs 72.98 (student_only 70.61). Teacher
resnet110 73.95 / 74.08 vs 74.31.

**Ablations of `kd_online_disagree`** (70.87 ± 0.28), resnet56 →
resnet20, 3 seeds, Welch tests:

| ablation | final | vs method | p |
|---|---|---|---|
| `every40` (re-evolve every 40 epochs) | 71.04 ± 0.08 | +0.17 | 0.41 |
| `nojitter` | 70.83 ± 0.12 | −0.04 | 0.84 |
| `rsearch` (random search, same fitness/budget) | 70.68 ± 0.28 | −0.19 | 0.45 |
| `every10` | 70.61 ± 0.23 | −0.26 | 0.28 |
| `average` (all 5 views every batch) | 69.65 ± 0.50 | **−1.22** | **0.03** |
| `randviews` (unscored random genomes) | 68.39 ± 3.20 | −2.48 | 0.31 |
| `noise` (noise masks, no search) | 67.79 ± 0.41 | **−3.08** | **0.001** |

Reading:
- **Averaging all selected views on every batch hurts** (−1.22 vs
  sampling one; −0.94 vs `kd`, p = 0.06) even with the student-aware
  fitness. That's the configuration Phase 1's `kd_evolved_cppn` /
  `kd_random_search` used, so part of their harm is how the views were
  applied, not only which were picked.
- Re-evolution frequency, jitter and evolution-vs-random-search make no
  measurable difference.
- **Unscored random views are unstable** (std 3.20: one seed collapsed;
  best-epoch 69.96 ± 0.80) — the fitness gate works as a safety filter
  against catastrophic views even though it doesn't add accuracy.
- **Noise masks are much worse than CPPN views** (−2.80 vs `kd`, p =
  0.001). Confound: noise masks get no gate/selection, so compare with
  `randviews` (also unselected, 68.39) — both unselected families do worse
  than gated CPPNs, noise more consistently.

**Calibration / robustness** (resnet56 → resnet20 rows; ablations and new
pairs pending a second `evaluate_extra` pass):
- KD students are markedly *worse calibrated* than student_only (ECE ~12%
  vs 6.5%); every view/augmentation variant lowers ECE somewhat (evolved
  10.5, CutMix 10.5, online_teacher 10.5, disagree 11.3) at no NLL gain.
- CIFAR-100-C mean accuracy: `kd` 43.30 ± 0.89; RandAugment 44.50 (+1.20,
  p = 0.13); `kd_online_disagree` 43.82 (+0.52, p = 0.42); **evolved 40.81
  (−2.49, p = 0.06) and random search 41.56 (−1.74, p = 0.06)**. The
  hypothesis that evolved views trade clean accuracy for robustness is not
  supported — they're worse on both.
- Teacher agreement drops for the harmful configurations (evolved 71.95,
  random search 72.54 vs `kd` 73.97).

## Decision: analysis paper (2026-10-19)

Gate 2 failed and the ablations and fitness study sharpened the negative
story, so the paper is the analysis paper. Remaining controls
(`slurm/analysis.sbatch`, 21 jobs):
- **Generality:** `kd_random_cppn` / `kd_evolved_cppn` /
  `kd_online_teacher` on wrn_40_2 → wrn_16_2 (KD headroom +2.0) and
  vgg13 → MobileNetV2 (cross-architecture), 3 seeds each.
- **Filtered noise:** noise masks scored and selected with the method's
  disagreement fitness and gate (`online.noise_selection=random`, same
  budget as `rsearch`), to separate the view family from the absence of
  selection in the `noise` ablation.
- The fitness-study analysis now adds partial Spearman correlations
  controlling for pattern mean, and the evolved-vs-random pool means.

## Fitness-vs-accuracy study — results (2026-10-19)

All 83 runs finished. Short-schedule (60-epoch) `kd`: 67.83 ± 0.34. Of 40
single bank views, **37 hurt the student, 3 help; mean change −2.89 pt.**

Spearman ρ between each property and the change vs `kd` (95% bootstrap CI):

| predictor | all 40 | evolved 20 | random 20 |
|---|---|---|---|
| **fitness** | +0.45 [0.10, 0.71] | **+0.01 [−0.55, 0.55]** | +0.75 [0.38, 0.93] |
| diversity (fitness's objective term) | −0.75 | **−0.81 [−0.95, −0.53]** | −0.74 |
| soft agreement (fitness's gate input) | +0.74 | +0.78 | +0.74 |
| teacher accuracy on view (relative) | +0.74 | +0.79 | +0.75 |
| pattern mean (image kept) | **+0.78** | +0.67 | +0.79 |
| pattern std (contrast) | −0.41 | −0.07 | −0.62 |
| channel divergence | −0.27 (n.s.) | −0.23 (n.s.) | −0.32 (n.s.) |
| connections | −0.07 (n.s.) | −0.01 (n.s.) | −0.09 (n.s.) |

**Reading — the core result for the analysis paper:**
1. **Among the genomes evolution actually produces, the fitness has no
   predictive power at all (ρ = 0.01).** The overall +0.45 comes entirely
   from the random pool, where the agreement gate separates destructive
   genomes (gate ≈ 0) from acceptable ones (ρ = 0.75 there).
2. **The fitness's objective term anti-predicts student benefit**
   (diversity ρ = −0.81 within evolved), while its constraint term predicts
   it (agreement +0.78). The fitness rewards diversity × gate(agreement),
   so once the gate is satisfied, evolution trades agreement for diversity
   — exactly the direction that hurts the student. This is the mechanism
   behind Gate 1's result and the 14 legacy attempts.
3. The fitness works as a **coarse filter, not a ranker** — consistent with
   the Phase 3 finding that selection matters only as a safety filter
   against catastrophic views.
4. The simplest descriptor, how much of the image the mask keeps (pattern
   mean), is the best single predictor overall (ρ = 0.78). Diversity,
   agreement, teacher accuracy and pattern mean are strongly intercorrelated,
   so their individual contributions need partial correlations before any
   causal reading.

**Caveats:** (a) single views on every batch hurt much more on the short
schedule (−2.89 mean) than an unscored random view did on the full one
(+0.07), so absolute effects don't transfer — the study is about ranking;
(b) the Gate 1 sanity check (evolved views worse than random-search views
on average) still has to be read off `per_genome.csv`.

## Fitness-vs-accuracy study (design)

The analysis paper's central question: does the teacher-only fitness
predict how useful a view is? Gate 1 showed fitness-*selected* views hurt
as a group; this measures it view by view.

**Design** (`src/fitness_study.py`, `scripts/fitness_study.py`):
- Bank: 40 genomes against the resnet56 teacher — 20 from a 40-generation
  NEAT run and 20 from an equal-budget random search (6000 evaluations
  each), at evenly spaced fitness quantiles of each pool (re-evaluated
  NEAT elites counted once, disqualified genomes excluded), so the bank
  covers both the low range random genomes reach and the high range only
  evolution reaches.
- Rescoring: every bank genome on one shared probe set of 1024 training
  images — during search each generation saw a different 256-image batch,
  so raw search fitnesses aren't comparable. Recorded: fitness, diversity,
  soft and top-1 agreement, teacher accuracy on the view (absolute and
  relative to raw), pattern mean and std, channel divergence, connections.
- Students: resnet20, KD + that one view on every batch (Phase 1 loss),
  60 epochs with milestones 37/45/52 (the 240-epoch schedule's
  proportions), seeds 0–1; `kd` on the same schedule, seeds 0–2, as the
  reference. 83 runs, ~30 min each.
- Analysis: Spearman ρ (bootstrap 95% CI) between each predictor and the
  change vs short `kd`, overall and within each pool (pool is a confound:
  it determines the fitness range).

**Caveat to check when results come in:** the short schedule is a proxy
for the full one. The Gate 1 ranking (evolved < random search < random
view) should reappear in miniature; if it doesn't, the short runs don't
reflect full-length behavior.

---

## Gate 2 answer (2026-10-15): not passed

resnet56 → resnet20, final-epoch top-1, 3 seeds; Welch t-tests (n = 3
each, so only large effects are detectable):

| mode | final | vs `kd` | t | p |
|---|---|---|---|---|
| `kd` | 70.59 ± 0.32 | — | | |
| `kd_online_disagree` (a) | 70.87 ± 0.28 | +0.28 | 1.14 | 0.32 |
| `kd_online_gradalign` (b) | 70.72 ± 0.43 | +0.13 | 0.42 | 0.70 |
| `kd_online_teacher` | 70.41 ± 0.29 | −0.18 | | |
| `kd_trained_cppn` (offline, 1 view) | 70.87 ± 0.31 | +0.28 | 1.09 | 0.34 |
| `kd_random_cppn` (offline, 1 unscored view) | 70.66 ± 0.39 | +0.07 | | |
| `kd_random_search` (offline top-5) | 69.07 ± 0.63 | −1.52 | −3.73 | 0.03 |
| `kd_evolved_cppn` (offline top-5) | 68.65 ± 0.28 | −1.94 | −7.90 | 0.001 |

**Gate 2 fails.** The seed-0 pilot's +0.60 for disagreement shrank to
+0.28 over three seeds (seeds 1–2 averaged ~70.7), not significant and
far from the ≥ ~1 pt target. It ties `kd_trained_cppn` exactly (+0.00)
and doesn't separate from one unscored random view (+0.21, p = 0.50), so
student-aware selection adds nothing measurable once the views stop doing
harm. Gradient alignment is flat.

**What the data do support** (the analysis-paper story):
1. *Selecting views with the teacher-only fitness hurts the student.*
   Random genomes chosen by the fitness (−1.52, p = 0.03) and evolved
   genomes (−1.94, p = 0.001) are both clearly below `kd`, while an
   unscored random view is not (+0.07); fitness-selected vs unscored is
   −1.59 (p = 0.03).
2. *Correction to the earlier "stronger search hurts more" reading:*
   evolved vs random search is −0.42 with p = 0.38 — the right direction
   but not established with 3 seeds. The robust claim is "fitness-based
   selection hurts", not "more optimization hurts more".
3. *How views are used matters more than which views:* the same
   teacher-only fitness, moved online with one view per batch, random
   shift/flip and periodic re-evolution, recovers +1.76 (p = 0.002) to
   −0.18 vs `kd`.
4. Strong augmentation (RandAugment −0.28, CutMix −0.06) doesn't help this
   student either.

**Validation reference:** `kd` with 5k held out scores 73.45 ± 1.15 on
validation vs 70.58 ± 0.43 on test. Validation runs ~3 pt above test
(plausibly near-duplicate images inside the CIFAR-100 training set) and
its seed spread is large because each seed holds out different images,
so tuning comparisons must be paired by seed.

## Phase 3 — full results (implemented; scope depends on the Gate 2 decision)

**Main table** (`phase3_main`, 36 jobs): all 6 benchmark pairs — the four
Phase 0 pairs plus resnet110 → resnet32 and vgg13 → vgg8 (published KD
73.08 / 72.98; needs a resnet110 teacher and the new student_only / kd
runs from the teachers / students stages) — × {`kd_cutmix`,
`phase3.method`} × 3 seeds. CutMix stands in for the augmentation
baselines (RandAugment was no better on the development pair).
`phase3.method` defaults to `kd_online_disagree`; if tuning changes its
settings, `method_set` applies them under the `method_tag` run tag.

**Ablations** (`phase3_ablations`, 21 jobs, resnet56 → resnet20): N = 10
/ 40, averaging all views, no jitter, online random search (same fitness,
same evaluations), unscored random views, and noise masks (8×8 uniform
noise per channel, bilinearly upsampled; no search). New code:
`online.search` ∈ {neat, random, none} and `online.family` ∈ {cppn, noise}
in `OnlineViewEvolver`; the default path is unchanged (same probe seeds).

**Secondary metrics** (`scripts/evaluate_extra.py`, one GPU job): ECE
(15 bins), NLL, top-1 agreement and KL with the teacher on the test split;
accuracy and ECE on CIFAR-100-C averaged over the 15 standard corruptions
× 5 severities. Independent of Gate 2 — can already run on every Phase 0–2
checkpoint, which also serves the analysis-paper framing (e.g. whether the
evolved views trade clean accuracy for robustness).

---

## Gate 1 answer and Phase 2 pilot (2026-10-14)

resnet56 → resnet20, CIFAR-100, final-epoch top-1 (3 seeds unless noted):

| mode | final | vs `kd` |
|---|---|---|
| `student_only` | 69.76 ± 0.50 | −0.83 |
| `kd` | 70.59 ± 0.32 | — |
| `kd_random_cppn` (1 unscored view) | 70.66 ± 0.39 | +0.07 |
| `kd_trained_cppn` | 70.87 ± 0.31 | +0.28 |
| `kd_random_search` (top-5 by fitness, random genomes; n = 2) | 69.43 ± 0.11 | −1.16 |
| `kd_evolved_cppn` (top-5 by fitness, NEAT) | 68.65 ± 0.28 | −1.94 |
| `kd_online_teacher` (pilot, seed 0) | 70.19 | −0.40 |
| `kd_online_gradalign` (pilot, seed 0) | 70.73 | +0.14 |
| `kd_online_disagree` (pilot, seed 0) | 71.19 | +0.60 |

**Gate 1: no.** Under the fixed protocol (including normalized inputs to
the teacher during view search), the existing `kd_evolved_cppn` is ~1.9 pt
*below* `kd` and ~1.1 pt below training the student alone. The protocol
fixes did not rescue it.

**The ordering is the main finding for the paper:** the harder views are
optimized for the teacher-only fitness, the more they hurt the student —
an unscored random view (+0.07) > random genomes selected by the fitness
(−1.16) > NEAT-evolved genomes (−1.94). Same fitness, same top-5 diverse
selection, same loss for the last two; only the optimizer's strength
differs. This is direct evidence that the teacher-only proxy is gamed by
stronger search, and it motivates making the fitness student-aware.
(Caveat: `kd_random_cppn` uses one view, the other two five, all on every
batch.)

**Phase 2 pilot (one seed each):** moving the *same* teacher-only fitness
online (one view per batch, shift/flip, re-evolved every 20 epochs) lifts
it from −1.94 to −0.40, so per-batch sampling + jitter + re-evolution
removes most of the damage. The student-aware disagreement fitness is the
only configuration so far above every baseline (71.19, +0.60 over the `kd`
mean, ~1.9 seed-std above it), but it's a single seed and still short of
the Gate 2 target (~71.6). Gradient alignment is flat (+0.14).

Pilot cost: ~1.85 h per run (0.45 min/epoch including evolution; the
epoch-0 round of 10 generations took ~1 min).

**Next:** seeds 1–2 for all three online modes. Any tuning of Phase 2
hyperparameters (gate thresholds, `view_weight`, `evolve_every`) must use
a validation split, not the test set these numbers come from.

**Validation reporting added** (see README "Tuning"): `val_size > 0` holds
out a seeded validation split, logs its accuracy each epoch, excludes it
from the view-search probe pool, and writes `__val<N>[__<tag>]` run
directories. `slurm/tune.sbatch` runs the `kd` reference (indices 0–2) and
the method (3–5) on that split and refuses to start without `val_size`.
Protocol: choose by `val_final` against `kd`'s, then rerun the winner once
on the full training set for the reported test number.

---

## Phase 2 — student-aware online view evolution (implemented, piloted)

**Modes** (`configs/cifar100.yaml` → `phase2`, `online`):
`kd_online_disagree` (fitness a), `kd_online_gradalign` (fitness b),
`kd_online_teacher` (old teacher-only fitness, re-evolved online — the
"student-aware vs just online" ablation), on resnet56 → resnet20 × 3 seeds
= 9 jobs (`slurm/phase2_online.sbatch`, array 0–8, 12 h).

**Design** (`src/online_views.py`):
- One NEAT population per run, warm-started across rounds: 10 generations
  at epoch 0, then 5 every 20 epochs (65 generations × 150 genomes per
  run, vs 40 × 150 for Phase 1's offline evolution). Genomes are ranked
  only within a round, since the student changes between rounds.
- Each batch: one view from the round's diverse top-5, randomly shifted
  (circular, ±4 px) and flipped (p = 0.5); loss as in Phase 1,
  `gamma·CE + alpha·(½·KD(raw) + ½·KD(view))`. Costs ~2× a `kd` epoch
  instead of ~6×.
- **(a) disagreement:** mean of KL(teacher‖student)·T² on the view, over
  probe images the teacher still classifies correctly on the view.
- **(b) gradalign:** cosine between the student's last-layer gradient of
  KD on the view and of CE on a second batch of training images, mapped to
  [0, 1] as (cos + 1)/2. Computed in closed form from the penultimate
  features (every model here ends in one Linear layer), so it needs no
  backward pass; a test checks it against autograd. The "held-out" batch is
  other *training* images, not a validation split, so the student trains
  on all 50k images like every baseline — a limitation to state.
- (a) and (b) are gated on teacher accuracy on the view divided by teacher
  accuracy on the raw probe images (`online.gate`: 0 at ≤ 0.6, full at
  ≥ 0.9; untuned). All three keep the Phase 1 guardrails
  (`min_connections`, `min_pattern_std`, contrast and channel-divergence
  penalties, diverse top-K selection).
- The student is scored in eval mode (BatchNorm running statistics) on
  unaugmented training images, never the test split.

**Gate 2 target:** final-epoch mean ≥ ~71.6 on resnet56 → resnet20
(`kd` 70.59 + 1), and above `kd_randaugment`, `kd_cutmix` and
`kd_random_search`.

**Not yet covered:** a budget-matched online random-search control and the
CPPN-vs-noise-mask ablation belong to Phase 3 if Gate 2 passes.

---

## Phase 1 — corrected baselines (running)

**Submitted 2026-10-14** as job `1406759` (array 0–17). `dept_dms`'s
`MaxCpuPerAccount` limit runs 5 jobs at a time; measured 18–19 s/epoch for
`kd_randaugment` and `kd_cutmix`.

**Partial results (12 of 18 runs, resnet56 → resnet20, final epoch, 3 seeds):**

| mode | final | best | vs `kd` (final) |
|---|---|---|---|
| `kd` (Phase 0) | 70.59 ± 0.32 | 70.85 ± 0.39 | — |
| `kd_randaugment` | 70.31 ± 0.31 | 70.60 ± 0.15 | −0.28 |
| `kd_cutmix` | 70.53 ± 0.14 | 70.62 ± 0.08 | −0.06 |
| `kd_random_cppn` | 70.66 ± 0.39 | 71.06 ± 0.17 | +0.07 |
| `kd_trained_cppn` | 70.87 ± 0.31 | 71.10 ± 0.11 | +0.28 |

Every difference is within about one seed standard deviation (~0.3), so
none is distinguishable from `kd` with 3 seeds. Neither strong augmentation
helps this small student. The evolved and random-search runs are about 1.07
min/epoch (~4.3 h per run), well under the 16 h limit.

**Modes** (`configs/cifar100.yaml` → `phase1`): `kd_randaugment`,
`kd_cutmix`, `kd_random_cppn`, `kd_trained_cppn`, `kd_evolved_cppn`,
`kd_random_search`, on resnet56→resnet20 × 3 seeds = 18 jobs
(`slurm/phase1_baselines.sbatch`, array 0–17). `student_only` and `kd`
come from Phase 0.

**Design decisions:**
- CPPN-view modes keep the main pipeline's method unchanged — attempt-14
  fitness and settings (`tau_low` 0.5, contrast/channel penalties 0.3,
  `min_connections` 1, `min_pattern_std` 0.01, diverse top-5 with
  `min_pattern_distance` 0.02, pop 150 × 40 generations), every view used
  on every batch, loss `gamma·CE + alpha·(½·KD(raw) + ½·mean KD(view))` —
  so Gate 1 tests the existing method under the fixed protocol, nothing
  else.
- **One fix carried in:** the main pipeline's `run_evolution` and
  `train_trainable_cppn` feed raw [0,1] images to a teacher trained on
  normalized inputs, so every legacy fitness score was computed on
  off-distribution inputs. Here the teacher is wrapped in
  `NormalizedModel`. This may change which genomes win, and is worth a
  sentence in the analysis/motivation section.
- **Random search** samples fresh genomes from the evolution config with
  Uniform{0..40} random mutation rounds (40 = the deepest lineage 40
  generations can produce), scores them with the identical fitness
  (`src.cppn.evolve.score_genome`, now shared with `run_evolution`),
  resamples the probe batch every 150 genomes as evolution does per
  generation, and uses the same diverse top-5 selection. Budget = 150 × 40
  = 6000 evaluations.
- View search runs inside each student job, seeded by the student seed, on
  unaugmented training images (never the test split); patterns, PNGs and
  logs go to `<run_dir>/views/`.
- KD + augmentation baselines apply the augmentation to the input both
  teacher and student see (standard practice). RandAugment: torchvision
  defaults (2 ops, magnitude 9). CutMix: Beta(1,1), probability 0.5, CE on
  the area-weighted label mix.

**Cost note:** `view_sampling: all` with 5 views means 6 student
forward/backward passes per batch for `kd_evolved_cppn` and
`kd_random_search` — roughly 6× a `kd` epoch. Phase 0 measured 0.25
min/epoch for resnet56 → resnet20 `kd`, so ~6 h per run; the sbatch asks
for 16 h.

---

## Phase 0 — results (2026-10-14): gate passed

All 28 jobs completed (teachers `1385644`, students `1385645`). CIFAR-100
top-1, 3 student seeds (mean ± std), final epoch / best epoch vs published:

| teacher | ours final / best | published |
|---|---|---|
| resnet56 | 72.26 / 72.96 | 72.34 |
| wrn_40_2 | 76.40 / 76.70 | 75.61 |
| resnet32x4 | 78.78 / 78.99 | 79.42 |
| vgg13 | 74.59 / 74.83 | 74.64 |

| pair | student_only final | kd final | kd best | published kd | kd gain (final) |
|---|---|---|---|---|---|
| resnet56 → resnet20 | 69.76 ± 0.50 | 70.59 ± 0.32 | 70.85 ± 0.39 | 70.66 | +0.83 |
| wrn_40_2 → wrn_16_2 | 73.02 ± 0.15 | 75.01 ± 0.06 | 75.34 ± 0.16 | 74.92 | +1.99 |
| resnet32x4 → resnet8x4 | 72.46 ± 0.10 | 72.92 ± 0.21 | 73.14 ± 0.12 | 73.33 | +0.46 |
| vgg13 → mobilenetv2 | 64.50 ± 0.44 | 68.24 ± 0.36 | 68.51 ± 0.24 | 67.37 | +3.74 |

**Gate:** `kd` best-epoch is within 0.5 pt of published on three pairs and
1.1 pt *above* it on vgg13 → MobileNetV2 (student_only there is also +0.8
above), so the protocol reproduces the benchmark. MobileNetV2 was the
least certain architecture match; it errs on the strong side, which only
makes our baselines harder to beat. **Report final-epoch numbers** in the
paper (no test-set selection); best-epoch is ~0.2–0.4 higher throughout.

**Implication for Gate 2:** on the development pair resnet56 → resnet20,
KD adds only +0.83 over student_only, and the gate asks the new method for
≥ ~1 pt over `kd` (target ≈ 71.6 final). Worth also tracking wrn_40_2 →
wrn_16_2, where KD's headroom is larger.

**Wall time** on an a100_20g MIG slice: teachers 0.7–2.3 h, students
0.16–0.30 min/epoch (40–75 min per run). Phase 1's 5-view modes at ~6× a
`kd` epoch come to ~1.5 min/epoch ≈ 6 h, plus view search.

## Phase 0 — benchmark protocol (setup)

**Submitted 2026-10-06:** teachers job `1385644` (array 0–3), students job
`1385645` (array 0–23, `afterok` on the teachers). Env:
`/project/ikoutis/conda_env/dml-torch` (python 3.11, torch 2.13.0+cu130,
torchvision 0.28.0) with `neat-python` 2.0.0, `pyyaml`, `tqdm` added to
`~/.local`.

**What was built** (`method_paper/`):
- **Models:** CIFAR ResNets (resnet8–110, resnet8x4/32x4), WRN-16/40-1/2,
  CIFAR VGG8/11/13, MobileNetV2 (CIFAR, width 0.5), all written from the
  published architecture specs. Parameter counts match the published
  benchmark within 0.6% for every model checked (resnet20 0.28M, resnet56
  0.86M, resnet110 1.74M, resnet8x4 1.23M, resnet32x4 7.43M, wrn_16_2
  0.70M, wrn_40_2 2.26M, vgg8 3.97M, vgg13 9.46M, mobilenetv2 0.81M).
  MobileNetV2 is the least certain match on details the parameter count
  can't reveal (activation choice, t=1 block) — the student-accuracy
  sanity check will confirm or refute it.
- **Data:** official 10k test split, full 50k training set, CIFAR-100
  normalization statistics, standard crop+flip augmentation, raw-[0,1]
  loaders with on-device normalization (keeps CPPN views possible later).
- **Training:** SGD + momentum 0.9 + weight decay 5e-4, MultiStepLR,
  per-epoch test evaluation; KD loss = 0.1·CE + 0.9·KL·T² with T=4
  (reusing `src/distill/losses.kd_loss`).
- **Runs:** one teacher per architecture (seed 0) shared by all student
  seeds; deterministic run directories so students locate their teacher by
  name and resubmission skips finished runs.
- **Tests:** 46 new tests (`tests/test_mp_*.py`) covering parameter counts,
  stem resolution, official-split usage, validation hold-out, LR
  milestones, weight decay, teacher freezing, job enumeration, and an
  end-to-end teacher → KD-student run on fake data. Full repo suite: 113
  passing.

**Jobs:** 4 teachers (`phase0_teachers.sbatch`, array 0–3), then 24 student
runs (`phase0_students.sbatch`, array 0–23: 4 student_only archs × 3 seeds
+ 4 KD pairs × 3 seeds), submitted with an `afterok` dependency on the
teacher job.

**Setup note.** Access to the old `/project/ahoover` project space was
lost after a Wulver update; work moves to the user's `ikoutis` allocation,
which needs a fresh clone and Python env first. All SLURM scripts in the
repo were made path-independent at the same time (they `cd` to
`$SLURM_SUBMIT_DIR` and source a gitignored per-machine `cluster_env.sh`,
template in `cluster_env.sh.example`), so a future move only means editing
that one file. The template now defaults to the DML project's existing
conda env on the same allocation (`/project/ikoutis/conda_env/dml-torch`,
python 3.11, PyPI CUDA 12 torch wheels) plus `neat-python`, rather than building a
new one.

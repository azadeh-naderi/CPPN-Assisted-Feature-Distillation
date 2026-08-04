# Experiment Log

Running record of experiments, results, bugs found, and fixes applied while
evaluating the NEAT-evolved-CPPN-assisted distillation method. Raw per-run
artifacts (config snapshots, per-epoch training logs, evolution logs, genome
pickles, pattern images) are archived under [`../results_archive/`](../results_archive/);
this document is the human-readable narrative and summary tables.

Method recap: a population of CPPNs (coordinate-only inputs, evolved topology
+ per-node activation function via NEAT) is evolved against a frozen teacher
to find an image "view" — a spatial pattern combined with the raw image —
that's maximally different from the raw image in the teacher's feature space
while still preserving enough prediction agreement to not be pure noise. The
evolved view is then used as an extra consistency-loss term during
teacher→student knowledge distillation (`kd_evolved_cppn` mode). Compared
against `student_only` (no teacher), `kd` (standard distillation),
`kd_random_cppn` (untrained CPPN view), and `kd_trained_cppn`
(gradient-trained CPPN view).

---

## Experiment 1 — FashionMNIST / LeNet

**Setup:** LeNet teacher + student, 30 epochs each, 3 seeds, real GPU run via
SLURM (`slurm/run_fashionmnist_gpu.sbatch`, `configs/datasets/fashionmnist_lenet.yaml`).

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 88.97 | 88.45 | 88.90 | 88.77 |
| kd | 88.93 | 88.60 | 88.90 | 88.81 |
| kd_random_cppn | 89.17 | 88.57 | 88.75 | 88.83 |
| kd_trained_cppn | 88.77 | 88.57 | 88.93 | 88.76 |
| kd_evolved_cppn | 88.78 | 88.47 | 88.73 | 88.66 |

**Read:** no meaningful separation between any mode — all five land in an
~88.5–89.2% band, seed-to-seed variance within a mode is about as large as
the spread across modes. Not a bug: LeNet on FashionMNIST is near-saturated
even without distillation, leaving little headroom for KD of any kind to
show an effect. Useful as a "pipeline produces sane numbers" sanity check,
not informative about the method itself. CIFAR-10/ResNet18 is where a real
gap has room to show up.

---

## Experiment 2 — CIFAR-10 / ResNet18

Five successive attempts, each surfacing a real problem that the previous
one's results were confounded by. Summarized in order; see "Bugs found and
fixed" below for the underlying root-cause writeups.

**Common setup:** ResNet18 (ImageNet-pretrained teacher, from-scratch
student), 100 epochs each, 3 seeds run as parallel SLURM array tasks
(`slurm/run_cifar10_resnet18_gpu.sbatch`, `configs/datasets/cifar10_resnet18.yaml`).

### Attempt 1 — broken teacher (invalid, kept for the record)

Teacher test accuracy: 78.10 / 78.92 / 81.50 (mean **79.51**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 81.52 | 83.02 | 83.68 | **82.74** |
| kd | 79.36 | 80.00 | 82.08 | **80.48** |
| kd_random_cppn | 79.22 | 80.12 | 82.02 | **80.45** |
| kd_trained_cppn | 79.96 | 79.92 | 82.26 | **80.71** |
| kd_evolved_cppn | 70.98 | 79.42 | 70.92 | **73.77** |

**Problem:** the teacher (79.51% mean) is *worse* than the from-scratch
student baseline (82.74%) — same architecture, teacher should have won
easily. Root cause: `teacher.lr=0.1` was a from-scratch-appropriate learning
rate applied to *fine-tuning* a pretrained backbone, with no decay over 100
epochs — degraded the pretrained ImageNet features instead of adapting them.
With a teacher weaker than the student, every KD variant dragging accuracy
down relative to `student_only` was the expected consequence of a broken
teacher, not a finding about any method. **Every number in this attempt is
invalid for drawing conclusions.**

### Attempt 2 — teacher fixed (`lr=0.01` + step decay)

Teacher test accuracy: 82.72 / 82.86 / 83.18 (mean **82.92**) — now properly
above the student baseline.

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| teacher | 82.72 | 82.86 | 83.18 | **82.92** |
| student_only | 82.80 | 83.48 | 82.96 | **83.08** |
| kd | 82.10 | 82.74 | 83.26 | **82.70** |
| kd_random_cppn | 82.90 | 81.48 | 82.64 | **82.34** |
| kd_trained_cppn | 82.76 | 82.24 | 83.76 | **82.92** |
| kd_evolved_cppn | 76.32 | 80.18 | 79.70 | **78.73** |

**Read:** `kd`/`kd_random_cppn`/`kd_trained_cppn` now cluster tightly around
82.3–82.9%, statistically indistinguishable from `student_only` and each
other (a legitimate null result — no evidence any of these three help or
hurt on this setup). `kd_evolved_cppn` remains the clear outlier at 78.73%,
now more cleanly isolated as specifically-evolved-CPPN-related rather than
"broken teacher makes everything worse."

**Bug found while diagnosing:** `pattern.pt` stats for the winning genomes
showed `min≈0.269, max≈0.731` almost exactly regardless of genome structure
— `sigmoid(±1)`. `compile_genome()`'s outer sigmoid had no pre-scale, and
small genomes leaning on naturally-bounded activations (`sin`/`tanh`/`clamped`,
all cap near ±1) produce ~[-1,1] raw outputs that plain sigmoid squashes into
a narrow band around 0.5 — a hard architectural ceiling preventing the
evolved pattern from ever reaching near-identity (~1.0) or near-blank (~0.0),
regardless of what the fitness function would have preferred.

### Attempt 3 — pattern range widened (`OUTER_SIGMOID_SCALE=5.0`)

Teacher test accuracy: 82.04 / 83.02 / 83.54 (mean **82.87**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 83.92 | 83.88 | 83.94 | **83.91** |
| kd | 81.78 | 83.02 | 83.72 | **82.84** |
| kd_random_cppn | 81.48 | 82.74 | 83.16 | **82.46** |
| kd_trained_cppn | 81.84 | 81.84 | 82.74 | **82.14** |
| kd_evolved_cppn | 79.12 | 73.66 | 61.80 | **71.53** |

**Read:** widening the achievable pattern range made `kd_evolved_cppn`
*worse* (78.73% → 71.53%), not better as hoped, with much larger seed
variance (79.12 → 61.80, a 17-point spread within one mode).

**Bug/mechanism found:** the seed-2 genome's pattern had `std=0.425,
min=0.007, max=0.993` — a high-contrast, near-binary spatial mask. Applied
multiplicatively, this amounts to a **static occlusion mask, identical
across every image and every one of 100 training epochs** — structurally
different from normal per-batch-random augmentation (Cutout/RandomErasing),
which varies every step. Hypothesis: the fitness gate's `tau_low=0.3`
(agreement floor on a single 256-image probe batch, frozen teacher) wasn't a
strict enough bar to rule these out.

### Attempt 4 — fitness gate tightened (`tau_low` 0.3 → 0.5)

Teacher test accuracy: 83.30 / 82.80 / 83.08 (mean **83.06**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 82.60 | 83.24 | 84.10 | **83.31** |
| kd | 81.74 | 81.38 | 83.76 | **82.29** |
| kd_random_cppn | 83.06 | 82.52 | 83.42 | **83.00** |
| kd_trained_cppn | 82.44 | 82.40 | 81.70 | **82.18** |
| kd_evolved_cppn | 73.88 | **30.84** | 83.56 | **62.76** |

**Read:** did not fix it — if anything, worse in the worst case. Seed 1
crashed to 30.84%, barely above CIFAR-10's ~10% random-guessing floor, on a
pattern that still passed the raised gate with nearly identical stats to
the earlier bad one (`min=0.007/max=0.993/std=0.425`). The training curve
(`results/students/cifar_10_resnet18_kd_evolved_cppn_1_.../training_log.csv`)
showed a smooth climb from ~15% to a **permanent ~29-30% plateau** for the
remaining ~90 epochs — genuine learning, just of something that doesn't
generalize to real test images. Consistent with `alpha=0.9` putting 90% of
the gradient signal on a badly-degraded static view for the entire run.

**Conclusion:** a one-shot fitness snapshot on a frozen teacher, evaluated
against 256 probe images, is not a reliable predictor of "safe to repeat
identically for 100 epochs." Further threshold-tuning is unlikely to fix
this — it's a proxy-vs-actual-training mismatch, not a threshold-calibration
problem.

### Attempt 5 — ensemble mode

Pivoted from threshold-tuning to a structural fix: `run_baseline_sweep.py
--use-ensemble` now averages the `kd_evolved_cppn` consistency loss over the
top-5 evolved genomes (`top_k_genomes.pkl`) instead of betting the whole run
on a single one — directly dilutes the damage any one problematic pattern
can do, rather than trying to predict in advance which patterns are safe.
Costs ~3x more compute for that mode (one view forward/backward pass per
ensemble member).

Teacher test accuracy: 83.88 / 82.38 / 82.42 (mean **82.89**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 83.20 | 82.96 | 83.68 | **83.28** |
| kd | 82.82 | 81.90 | 82.22 | **82.31** |
| kd_random_cppn | 82.88 | 83.00 | 83.32 | **83.07** |
| kd_trained_cppn | 83.86 | 82.78 | 80.86 | **82.50** |
| kd_evolved_cppn | 80.90 | 80.30 | 80.38 | **80.53** |

**Read:** ensembling fixed the instability completely — `kd_evolved_cppn`
went from a 17–52-point seed spread (including the 30.84% near-random-guessing
collapse in attempt 4) down to a **0.6-point spread** (80.30–80.90). No
catastrophic outliers. This confirms the diagnosis from attempts 3–4: betting
an entire 100-epoch run on a single static evolved pattern was the actual
problem, not the evolution/fitness process producing fundamentally unusable
genomes.

Not a clean win, though: `kd_evolved_cppn` (80.53 mean) is now **consistently
~2.5–3 points below every other mode**, including `student_only` — small and
stable rather than occasionally catastrophic, but still a real, repeatable
accuracy cost that `kd_random_cppn`/`kd_trained_cppn` don't show. Evolution
appears to reliably find views that are somewhat harder for the student to
learn from on average, not just occasionally-harmful ones. This is a
legitimate, reportable result as-is (stable-but-costly), and also a natural
opening for the next round of investigation.

### Attempt 6 — independent CPPN-view loss weight (additive, regressed)

Diagnosis from attempt 5: the cost ordering `kd_random_cppn` (83.07) >
`kd_trained_cppn` (82.50) > `kd_evolved_cppn` (80.53) tracks monotonically
with how hard each CPPN was optimized to be different from the raw image —
not noise, a real gradient. `combined_loss()` previously averaged `loss_soft`
and the CPPN-view term 50/50 under one `alpha=0.9`, which (a) dilutes
standard KD's own contribution relative to plain `kd` mode (which gets full
`alpha` on `loss_soft` alone), and (b) gives the costliest term (the
diversity-selected evolved view) equal footing with the safest one (plain
soft labels).

Added an independent `cppn_weight` (config: `student.cppn_weight`, code:
`src/distill/losses.py`'s `combined_loss()`): when set, loss becomes
`(1-alpha)*hard + alpha*soft + cppn_weight*cppn_mean` instead of averaging
`soft` and `cppn_mean` together — restores full-strength standard KD and
lets the CPPN term contribute independently, at a smaller weight
(`cppn_weight=0.3` in the CIFAR configs, a starting point, not tuned).
`None` (unset) preserves the exact previous averaged behavior — FashionMNIST/
MNIST configs untouched, only the CIFAR configs opt in.

**Caveat, decided to accept for this run rather than fix first:** the new
formula is additive, not a convex combination — total distillation-related
loss weight for CPPN-view modes is now `alpha + cppn_weight` = 0.9 + 0.3 =
**1.2**, vs. plain `kd`'s `alpha` = 0.9. Under the old averaged formula, a
CPPN view redistributed a fixed 0.9 budget; now it adds on top of it. This
means if `kd_evolved_cppn` improves in this run, it won't cleanly separate
"the reweighting fixed the diversity-cost imbalance" from "the larger total
loss magnitude/effective gradient step helped on its own." A cleaner
ablation later would use `(1-alpha-cppn_weight)*hard + alpha*soft +
cppn_weight*cppn_mean` (all three terms drawn from one fixed budget summing
to 1) instead. Kept the simpler additive version for this run rather than
delay it further.

Teacher test accuracy: 83.28 / 82.68 / 82.98 (mean **82.98**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 83.50 | 83.90 | 84.06 | **83.82** |
| kd | 82.42 | 81.86 | 83.58 | **82.62** |
| kd_random_cppn | 81.74 | 81.00 | 82.58 | **81.77** |
| kd_trained_cppn | 80.52 | 82.60 | 80.42 | **81.18** |
| kd_evolved_cppn | 66.40 | 59.64 | 79.62 | **68.55** |

**Read: regressed, confirming the caveat above was a real problem, not just
theoretical.** `kd_evolved_cppn` got worse on both counts — mean dropped
further (80.53% → 68.55%) and the seed instability ensembling had fixed came
back (0.6-point spread → 20-point spread, 59.64–79.62). `kd_random_cppn` and
`kd_trained_cppn` also both dipped slightly below their attempt-5 levels.
Pushing total non-hard-label loss weight to 1.2 (0.9 alpha + 0.3 cppn_weight)
— only 10% weight left on actual ground-truth labels while two different
soft-target sources compete for the rest — appears to destabilize
optimization on its own, independent of which genome evolution picks.
**Conclusion: the additive form is actively harmful; use the fixed-budget
form instead (attempt 7).**

### Attempt 7 — fixed-budget CPPN-view weight (still regressed)

Same idea as attempt 6, but fixing the confound: `cppn_weight` now draws
from the same fixed budget as `alpha` instead of being additive on top of
it. `combined_loss()` changed to `(1-alpha-cppn_weight)*hard + alpha*soft +
cppn_weight*cppn_mean` — all three terms sum to 1. Since `alpha` must
mathematically come down for `hard_weight = 1-alpha-cppn_weight` to stay
non-negative, set `alpha=0.6, cppn_weight=0.3` (hard stays at `0.1`).

**Caveat:** `alpha` is a single shared config value, and plain `kd` mode
also reads it directly (no `cppn_weight` involved for that mode) — so this
run's `kd` baseline uses `alpha=0.6`, not the `0.9` used in every attempt
2–6. Not comparable to earlier attempts' `kd` numbers, only valid as a
same-run reference point against this attempt's own CPPN-view modes.

Teacher test accuracy: 83.14 / 83.54 / 82.90 (mean **83.19**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 81.82 | 83.02 | 83.76 | **82.87** |
| kd | 82.60 | 82.90 | 83.48 | **82.99** |
| kd_random_cppn | 83.64 | 82.76 | 81.08 | **82.49** |
| kd_trained_cppn | 82.10 | 82.30 | 83.10 | **82.50** |
| kd_evolved_cppn | 77.96 | 80.76 | **34.84** | **64.52** |

**Read: worse than attempt 6, not better.** Seed 2 crashed to 34.84% — another
near-random-guessing collapse, despite the theoretically cleaner fixed-budget
weighting. Seeds 0 and 1 actually looked reasonable (77.96, 80.76, close to
attempt 5's range), but one catastrophic outlier drags the mean down again.

**Conclusion: two separate loss-weighting attempts (6 and 7) both failed to
beat attempt 5, and both reintroduced the catastrophic single-seed collapse
that ensembling alone had fixed.** This is real evidence against the
loss-weighting hypothesis — the instability isn't really about *how much*
the CPPN term is weighted, it's about *which specific genomes* end up in a
given seed's top-5 ensemble. Reducing the weight doesn't reliably neutralize
a bad ensemble; it just sometimes does (seeds 0/1 here) and sometimes
doesn't (seed 2). Reverted `alpha`/`cppn_weight` in the CIFAR configs back to
attempt 5's settings (`alpha=0.9`, no `cppn_weight`) rather than continue
tuning this lever.

### Attempt 8 — direct contrast penalty in the fitness function (best result so far)

Pivoted from loss-weighting (exhausted, attempts 6–7) to fixing genome
*selection* directly. Added `contrast_penalty` to `fitness_from_terms()`
(`src/cppn/fitness.py`): `fitness = diversity*gate(agreement) -
gamma*num_connections - contrast_penalty*pattern_std`. Targets the actual
diagnosed mechanism from attempts 3–4 — bad genomes consistently had
`pattern_std~0.4+` (a near-binary static occlusion mask), while range-capped
patterns from before the `OUTER_SIGMOID_SCALE` fix (`std~0.16`) were
comparatively harmless. `contrast_penalty=0.3` set in the CIFAR configs
(same order of magnitude as observed `best_fitness` values, ~0.03–0.22, so
it meaningfully affects ranking). `pattern_std` now also logged per-genome in
`evolution_log.csv` for post-hoc inspection. Loss-weighting reverted to
attempt 5's settings (`alpha=0.9`, no `cppn_weight`) so this attempt isolates
the new variable cleanly rather than stacking it on an already-harmful
change.

Teacher test accuracy: 83.56 / 82.58 / 84.08 (mean **83.41**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 83.24 | 83.64 | 83.50 | **83.46** |
| kd | 81.76 | 81.92 | 83.16 | **82.28** |
| kd_random_cppn | 84.28 | 82.04 | 83.92 | **83.41** |
| kd_trained_cppn | 82.52 | 81.94 | 81.66 | **82.04** |
| kd_evolved_cppn | 82.98 | 81.78 | 83.50 | **82.75** |

**Read: this worked, on both counts we'd been chasing.**

*Stability:* 82.98/81.78/83.50, a 1.72-point spread — no catastrophic
outliers, in the same regime as attempt 5's 0.6-point spread and nowhere
near attempts 4/6/7's 20–52 point collapses.

*Performance:* mean jumped from attempt 5's 80.53% to **82.75%**, and
`kd_evolved_cppn` is now *better* than plain `kd` (82.28%) and
`kd_trained_cppn` (82.04%) — only slightly behind `kd_random_cppn` (83.41%)
and `student_only` (83.46%), under a point rather than the 2.5–3 point gap
seen in every previous attempt.

**Conclusion: this confirms the diagnosis from attempts 3–4.** The problem
was genome *selection* (evolution converging on high-contrast static
occlusion masks that passed the agreement gate but were harmful as a
100-epoch training signal), not loss weighting — two rounds of
loss-reweighting (attempts 6–7) made things worse, while penalizing pattern
contrast directly in the fitness function fixed both the instability and
most of the performance gap in one attempt. This is the best result so far
for `kd_evolved_cppn` and a legitimate candidate for the paper's headline
number, pending more seeds for statistical confidence (see status section).

**Caveat discovered after reporting the above:** inspecting the actual
winning genomes revealed all three seeds converged to `pattern_std=0.000`
exactly — degenerate single-bias-node, zero-connection genomes producing a
spatially *constant* pattern (a uniform brightness scalar of ~0.67–0.73, not
a spatial view at all). `contrast_penalty*pattern_std` is minimized exactly
at `std=0`, so evolution took the cheapest path rather than genuinely
exploring spatial variation. The stability/accuracy improvement is real, but
attempt 8 as configured isn't actually demonstrating "evolution finds
spatially-informative views" — a fixed random brightness scalar would
plausibly achieve the same effect without any evolution. See attempt 9.

### Attempt 9 — gated contrast penalty (fixed collapse, but cost returned)

Fix for attempt 8's collapse: `contrast_std_threshold` added to
`fitness_from_terms()` — penalty is now `contrast_penalty *
max(0, pattern_std - contrast_std_threshold)` instead of
`contrast_penalty * pattern_std`. Default `0.0` reproduces attempt 8's exact
prior behavior; set to `0.2` in the CIFAR configs (between the "harmless"
~0.16 std of pre-range-fix patterns and the "harmful" ~0.4+ std of the
diagnosed occlusion masks), so genomes with moderate genuine spatial
variation face zero penalty and only the actual failure mode is
discouraged. Verified via a tiny synthetic `run_evolution()` call before
running for real: winning genomes cluster around `std~0.20` (the threshold
boundary, the "free" edge of maximum allowed contrast) instead of collapsing
to `std=0`.

Teacher test accuracy: 83.76 / 83.22 / 83.62 (mean **83.53**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 83.12 | 83.60 | 83.32 | **83.35** |
| kd | 82.88 | 82.54 | 82.00 | **82.47** |
| kd_random_cppn | 83.06 | 82.74 | 83.02 | **82.94** |
| kd_trained_cppn | 83.52 | 83.16 | 82.68 | **83.12** |
| kd_evolved_cppn | 77.36 | 82.12 | 81.16 | **80.21** |

Winning genomes confirmed genuinely spatial this time — no more collapse:

| seed | std | nodes | connections | activations |
|---|---|---|---|---|
| 0 | 0.196 | 4 | 7 | gauss, identity, relu |
| 1 | 0.092 | 3 | 3 | abs, gauss, sin |
| 2 | 0.108 | 3 | 5 | clamped, gauss |

**Read: the collapse is fixed, but the accuracy went with it.** `kd_evolved_cppn`
dropped back to 80.21% — essentially attempt 5's 80.53%, not attempt 8's
82.75%. This is itself an important, clean finding: **it strongly suggests
attempt 8's improvement came specifically from the degenerate collapse, not
from generally-better genome selection.** A near-constant brightness scalar
is a genuinely mild, safe transform; real spatial variation — even
carefully bounded, non-catastrophic amounts (std ≤ 0.2 here) — carries a
repeatable ~2.5–3 point cost that has now shown up consistently across two
structurally different fitness designs (attempt 5's ensembling-only and
attempt 9's gated-contrast-penalty).

**Hypothesis for why:** the fitness function's agreement term (top-1
prediction match) is a coarse, effectively binary constraint from NEAT's
search perspective — a genome only needs to keep the *argmax* class the
same, and is free to scramble the rest of the predicted distribution
arbitrarily while still "passing" the gate. A powerful discrete optimizer
like NEAT's evolutionary search is well-suited to finding exactly this kind
of proxy-gaming solution (a Goodhart's-law pattern: optimizing hard against
a proxy stops it measuring what you actually wanted). Supporting evidence:
`kd_trained_cppn` — which optimizes a *smooth* KL-divergence consistency
penalty via gradient descent instead of a hard top-1 threshold — has never
shown this cost in any attempt (82–83% every time). See attempt 10.

### Attempt 10 — smooth (logit-distribution) agreement instead of top-1 (hypothesis disconfirmed)

Direct test of the hypothesis above: replaced the top-1 argmax agreement
measure used for fitness with `soft_agreement_term()` — cosine similarity
between the teacher's full softmax probability distributions (raw vs. view),
not just whether the top-1 class matches. Bounded `[0,1]` like the old
measure, so it's a drop-in replacement for the same `gate()`/`tau_low`/
`tau_high` mechanism (`src/cppn/fitness.py`). `run_evolution()` now computes
both — `top1_agreement` logged for comparison, `soft_agreement` used for
actual fitness (`src/cppn/evolve.py`).

**Calibration caveat:** `tau_low=0.5`/`tau_high=0.7` were tuned for the old
top-1 scale. A toy synthetic sanity check (untrained random-weight teacher)
showed the new soft measure staying in a much narrower, higher range
(0.97–1.0) than top-1 agreement did in the same test (0.06–1.0) — expected
for an under-confident toy model with non-peaked outputs, but worth
confirming the thresholds still discriminate meaningfully against a real,
well-trained ResNet18 teacher's more peaked predictions. Check the actual
`agreement` column range in `evolution_log.csv` once this runs for real,
before trusting the result.

Teacher test accuracy: 84.16 / 83.46 / 84.18 (mean **83.93**)

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 82.90 | 83.42 | 83.12 | **83.15** |
| kd | 83.38 | 83.62 | 82.72 | **83.24** |
| kd_random_cppn | 82.76 | 82.58 | 84.10 | **83.15** |
| kd_trained_cppn | 81.30 | 82.86 | 82.84 | **82.33** |
| kd_evolved_cppn | 80.84 | 80.18 | 79.18 | **80.07** |

**Calibration confirmed fine, not the issue:** soft agreement ranged widely
against the real teacher (0.11–1.0, `evolution_log.csv`), the gate is
genuinely discriminating, not saturating near 1.0 like the toy sanity check
worried it might. Winning genomes remain genuinely spatial (`std` 0.14–0.19,
real multi-node topologies with 2–7 connections) — no collapse.

**Read: this disconfirms the hypothesis.** `kd_evolved_cppn` landed at
80.07% — essentially identical to attempt 5 (80.53%) and attempt 9 (80.21%).
Replacing the coarse top-1 gate with a smooth, full-distribution agreement
measure changed almost nothing.

**Conclusion: three structurally different fitness designs — ensembling
alone (attempt 5), a gated contrast penalty (attempt 9), and smooth
full-distribution agreement (attempt 10) — all converge to the same
~80–80.5% result for genuinely-spatial evolved views.** This is no longer
explainable as an artifact of any single fitness-function design choice; it
looks like a genuine, robust property of the method as currently
formulated: evolved CPPN views that succeed at creating real spatial
feature diversity carry a consistent ~2.5–3 point accuracy cost relative to
`student_only`/`kd`/`kd_random_cppn`, and it does not appear to be fixable
by further fitness-function reformulation. **Recommendation: stop tuning
fitness-function variants here — further redesigns are unlikely to change
the outcome given three independent approaches already triangulated the
same answer.** Treat ~80.2–80.5% (attempt 9 or attempt 10, both legitimate
and genuinely spatial) as the real, reportable result for
`kd_evolved_cppn`, and prioritize more seeds for statistical confidence
over further fitness redesign attempts.

---

## Bugs found and fixed (chronological)

| # | Commit | What broke | Fix |
|---|---|---|---|
| 1 | `ca0d08f` | `.gitignore`'s `data/`/`results/` patterns (no leading `/`) matched `src/data/` by basename anywhere in the tree, silently excluding a real source module from every commit | Anchored patterns to repo root (`/data/`, `/results/`) |
| 2 | `35f9007` | `run_baseline_sweep.py`'s `sweep_id = sweep_{int(time.time())}` had 1-second resolution; parallel SLURM array tasks launched within the same second collided, each overwriting the previous task's `summary.csv` | Added a `uuid.uuid4().hex[:8]` suffix |
| 3 | `153a096` | Teacher fine-tuning used a from-scratch-appropriate LR (`0.1`) with no decay, degrading pretrained ImageNet features — teacher ended up *worse* than the from-scratch student | `teacher.lr=0.01` + `StepLR` decay added to `train_teacher()` and `DistillTrainer` |
| 4 | `38bdbb6` | `compile_genome()`'s outer sigmoid had no pre-scale; genomes with naturally-bounded raw outputs (`sin`/`tanh`/`clamped` activations) got squashed into a narrow `[0.27, 0.73]` band, an architectural ceiling preventing near-identity or near-blank patterns | Added `OUTER_SIGMOID_SCALE=5.0` (matches neat-python's own internal sigmoid scale) before the squash |
| 5 | `5721236` | (hypothesis, not confirmed) `tau_low=0.3` fitness gate too permissive for the newly-widened pattern range | Raised to `0.5` — **did not resolve the underlying issue**, kept as a mild additional safeguard |
| 6 | `c39b125` | Single evolved genome applied as a static, unchanging transform for all 100 epochs — occasionally a genome that looks fine on a small fitness-evaluation probe batch is actually harmful as a repeated training-time signal | `--use-ensemble`: average consistency loss over top-5 evolved genomes instead of one — **confirmed fixed**, seed spread dropped from 17-52 points to 0.6 points in attempt 5 |
| 7 | `c662cd1` (regressed), `b791c75` (still regressed) | Two attempts to reweight the CPPN-view loss term (additive, then fixed-budget) — both made `kd_evolved_cppn` worse, not better, reintroducing catastrophic single-seed collapse | Reverted to attempt 5's loss settings (`alpha=0.9`, no `cppn_weight`) — this lever doesn't fix the actual problem |
| 8 | `3424162` | Agreement gate alone wasn't enough to rule out high-contrast, near-binary genomes (`pattern_std~0.4+`) that amount to a static occlusion mask | Added `contrast_penalty` to `fitness_from_terms()`, directly penalizing `pattern_std` — **confirmed fixed**, mean rose to 82.75% (vs. attempt 5's 80.53%) with stability intact (1.72-point seed spread) |
| 9 | `6a745e2` | Attempt 8's un-gated penalty is minimized exactly at `pattern_std=0`, so evolution collapsed to degenerate constant-pattern genomes (all 3 seeds, `std=0.000`, `num_connections=0`) — a uniform brightness scalar, not a spatial view | Added `contrast_std_threshold`: penalty only applies above a threshold (`0.2`), removing the incentive to collapse toward zero while still discouraging the diagnosed failure mode — **confirmed fixed** (genuinely spatial genomes again), but accuracy cost returned (80.21% mean), suggesting attempt 8's gain was specifically from the collapse |
| 10 | `525fcb5` | Top-1 argmax agreement is a coarse, effectively binary constraint — NEAT's search can satisfy it while scrambling the rest of the predicted distribution arbitrarily, plausibly explaining the repeatable ~2.5-3 point cost seen in attempts 5 and 9 | Added `soft_agreement_term()`: cosine similarity of full softmax distributions instead of top-1 match, used for fitness (top-1 kept for logged comparison) — **hypothesis disconfirmed**, landed at 80.07%, essentially unchanged from attempts 5/9 |

---

## Current status / open questions

- **FashionMNIST/LeNet:** done, sane null result, not the paper's headline experiment.
- **CIFAR-10/ResNet18 — fitness-tuning phase is closed.** Three structurally different fitness designs (attempt 5: ensembling alone; attempt 9: gated contrast penalty; attempt 10: smooth full-distribution agreement) all converged to the same ~80–80.5% result for `kd_evolved_cppn` once genuinely-spatial (non-degenerate) genomes are required. Attempt 8's higher number (82.75%) is now understood to have come from a degenerate genome collapse, not real genome selection — not a valid basis for the reported result. The ~2.5–3 point accuracy cost relative to `student_only`/`kd`/`kd_random_cppn` looks like a genuine, robust property of evolved-and-genuinely-diverse CPPN views on this setup, not an artifact of any one fitness formulation. **Recommendation: stop iterating on fitness-function redesigns; report attempt 9 or attempt 10 (statistically indistinguishable, both legitimate) as the result.**
- **Next step: more seeds (5–10)** on the reported config (attempt 9 or 10) for statistical confidence — 3 seeds is enough to catch instability/collapse but not enough for a confident published claim on a ~2.5 point gap.
- **CIFAR-100/ResNet18:** not yet run — same fixes already wired into `slurm/run_cifar100_resnet18_gpu.sbatch`/its config (the soft-agreement change from attempt 10 isn't config-gated, so it applies there automatically too), ready to launch once CIFAR-10 is settled.
- **Open question for the paper's narrative:** given three fitness redesigns landed on the same cost, the more interesting framing may not be "how do we close this gap" but "what does a stable ~2.5 point cost for genuinely-diverse evolved views tell us about the diversity/accuracy tradeoff in CPPN-assisted distillation" — worth deciding whether that's the actual story to tell rather than continuing to treat it as a bug to fix.

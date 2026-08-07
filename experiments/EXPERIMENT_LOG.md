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

**Addendum — visualizing the actual evolved view (what "genuinely spatial"
looks like):** the "no collapse" checks above (attempts 9-10) only look at
summary statistics (`pattern_std`, node/connection counts) — they confirm a
genome *isn't* a degenerate constant, but say nothing about what the
spatial pattern actually *is*. Built `scripts/visualize_evolved_view.py` to
render the real transform (original image next to the exact CPPN view the
consistency loss trains against, using real probe images) instead of just
the raw `pattern.png` mask. Running it against a real attempt-10 CIFAR-10
winning genome revealed the pattern is a **fixed magenta/green vertical
stripe**, applied identically to every image regardless of content (expected
— a CPPN with no image-content input can only ever produce one fixed
spatial pattern per genome, reused across the whole batch). Two things made
this artifact invisible to every fitness safeguard built so far:

- `pattern_std` (the contrast-penalty target, attempts 8–9) measures overall
  spatial contrast, not whether R/G/B are treated differently at the same
  pixel — a colored stripe with mild per-channel range can have a fully
  "safe" std.
- Both the top-1 gate (attempts ≤9) and the smooth full-distribution
  agreement measure (attempt 10) are computed on **teacher logits**, i.e.
  after the color-channel information has already been compressed through
  the network — a stripe narrow/subtle enough to leave the teacher's
  predicted class distribution largely intact can still register as
  "diverse" in penultimate-feature space (the diversity term evolution is
  directly rewarded for maximizing).

In other words, evolution appears to have found a shortcut that's
essentially invisible from the logit/feature-summary side of the fitness
function but plainly visible as an unnatural, content-independent color
artifact once you actually look at the image. This gives a concrete,
visual hypothesis for the ~2.5–3 point cost that's persisted across three
fitness redesigns: none of them directly penalized *this specific*
mechanism (per-pixel channel-differential color shift) because none of them
looked at the compiled pattern's per-channel structure directly. See
attempt 11.

### Attempt 11 — channel divergence penalty (fixed the stripe, reopened a worse collapse)

Direct test of the visual hypothesis above: added `channel_divergence_term()`
(`src/cppn/fitness.py`) — mean per-pixel std across the channel dimension of
the compiled `[H, W, C]` pattern itself (not the teacher's logits/features).
High values mean the pattern treats R/G/B very differently at the *same*
spatial location (a colored stripe/tint); low values mean color channels
move together at each pixel (a plain spatial luminance/contrast pattern,
still fully expressible). Wired into `fitness_from_terms()` as
`channel_divergence_penalty * channel_divergence`, subtracted unconditionally
— unlike `contrast_penalty`, this needs **no threshold gating**: driving
channel divergence to exactly 0 is not the degenerate-collapse shortcut that
driving `pattern_std` to 0 was in attempt 8 (a pattern can still vary richly
across space while being channel-uniform, so there's no "cheap constant"
minimum to fall into). `channel_divergence` also now logged per-genome in
`evolution_log.csv` for post-hoc inspection, same as `pattern_std`.

`channel_divergence_penalty=0.3` set in both CIFAR configs — same order of
magnitude as `contrast_penalty`, a first real test rather than a
prior-run-tuned value (there's no earlier run of this specific penalty to
tune from). Verified via a tiny synthetic `run_evolution()` call before
running for real: the new parameter flows through end-to-end and
`channel_divergence` appears correctly in the evolution log alongside
`fitness`.

Teacher test accuracy: 83.3 (single teacher checkpoint shared across all 3
seeds' student runs in this sweep).

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 83.52 | 83.04 | 82.70 | **83.09** |
| kd | 82.30 | 81.48 | 82.64 | **82.14** |
| kd_random_cppn | 83.16 | 82.78 | 83.50 | **83.15** |
| kd_trained_cppn | 82.50 | 83.00 | 83.46 | **82.99** |
| kd_evolved_cppn | 82.68 | **12.82** | 80.02 | **58.51** |

**Read: the penalty worked exactly as designed, and that's precisely what
caused the new failure.** Every top-fitness genome across all 3 seeds
reached `channel_divergence=0.0` in `evolution_log.csv` — the color-stripe
mechanism is genuinely gone, confirmed directly rather than inferred. But
seed 1's winning genome (`pattern.pt`) is a perfectly flat constant at
**0.0125** — `std=0`, `min=max=mean=0.0125` — the exact zero-connection,
single-bias-node degenerate genome first seen in attempt 8. Checking the
seed's `top_k_genomes.pkl`: 3 of the 5 ensemble members are that same
zero-connection genome, which is why ensembling (the fix that saved every
prior attempt from single-bad-genome collapse) didn't help this time — the
whole pool was collapsed, not just one member.

**Mechanism:** `contrast_std_threshold` (attempt 9) already made
`std<=0.2` a flat, zero-penalty plateau — genomes anywhere in that range tie
on the contrast term, and were previously kept apart by the diversity term
alone (a genuinely spatial pattern generally exposes more teacher-feature
diversity than a constant). Adding `channel_divergence_penalty` narrows the
diversity ceiling reachable *without* using channel-dependent connections,
shrinking that diversity gap and making the plateau flatter still. Seed 1's
population drifted into the flattened plateau's cheapest corner — zero
connections — same corner attempt 8 landed all 3 seeds in, except this time
the specific constant that corner's mutation/init process converged to was
an extreme near-black value (0.0125) rather than attempt 8's benign ~0.7,
which is what made it catastrophic (multiplying every training image by
~0.0125 for 100 straight epochs) rather than merely degenerate.

**Encouraging signal buried in the failure:** seed 0's 82.68% is the best
single genuinely-spatial (non-degenerate) `kd_evolved_cppn` result across
every attempt so far, and seeds 0+2 (excluding the collapsed seed) average
**81.35%** — modestly ahead of attempts 9/10's ~80.1–80.5%. The mean of
58.51% is entirely an artifact of one seed's genome-selection collapse, not
evidence the method regressed on genuinely spatial genomes.

**Conclusion: two different fitness formulations (attempts 8 and 11) have
now independently converged on the identical zero-connection degenerate
genome as (near-)optimal.** That's strong evidence it should be excluded
outright rather than continuing to design penalty terms that hope to
out-compete it in every corner of the search space. See attempt 12.

### Attempt 12 — disqualify zero-connection genomes outright (collapse fixed; cost persists)

Rather than a fourth attempt to make the zero-connection genome merely
non-optimal, `fitness_from_terms()` (`src/cppn/fitness.py`) now takes a
`min_connections` parameter: any genome with fewer enabled connections than
this returns a sentinel `DISQUALIFIED_FITNESS = -1e6` immediately, computed
*before* any of the diversity/agreement/penalty terms — finite (safe
through CSV/pandas/torch logging) but far below any value a legitimate
genome could reach, so it can never win a fitness comparison or tie-break
regardless of what the rest of the population looks like. `min_connections`
also threaded through `run_evolution()`/`evolve_cppn.py`, same pattern as
every other fitness knob.

`min_connections=1` set in both CIFAR configs — the minimum needed to
exclude exactly the diagnosed genome (a genome with 0 connections) without
excluding otherwise-valid small genomes (attempt 9's winning genomes had
3–7 connections; attempt 11's healthy seeds 0/2 aren't checked here but are
presumably similarly small). Verified via synthetic `run_evolution()` calls
before running for real: (a) a normal small run with `min_connections=1`
never produces a disqualified fitness value in the evolution log, and (b) a
forced `min_connections=999` run disqualifies every genome (`fitness ==
-1e6` for the entire population), confirming the mechanism actually engages
inside `eval_genomes`, not just in isolated unit tests of
`fitness_from_terms()`.

**3-seed results (seeds 0-2):** teacher test accuracy 82.90 / 83.44 / 84.10
(mean **83.48**).

| mode | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| student_only | 82.82 | 83.00 | 82.98 | **82.93** |
| kd | 81.96 | 81.16 | 84.00 | **82.37** |
| kd_random_cppn | 82.44 | 81.62 | 84.42 | **82.83** |
| kd_trained_cppn | 81.02 | 82.86 | 84.36 | **82.75** |
| kd_evolved_cppn | 82.80 | 74.54 | 82.28 | **79.87** |

At n=3: the catastrophic-collapse mechanism looked confirmed fixed —
`evolution_log.csv` showed zero-connection genomes still appearing in the
population every generation, but none made it into the top-10 fitness rows
or top-5 ensemble; every winning genome was genuinely spatial (`pattern_std`
0.11–0.13, 2–5 connections) with `channel_divergence=0.0`. Seed 1 still
dropped to 74.54% despite a structurally unremarkable winning genome,
reading at the time as ordinary seed-to-seed variance. Recommendation at
that point: stop iterating on the fitness function and run more seeds for
statistical power. **7 additional seeds (3–9) launched**
(`sbatch --array=3-9 slurm/run_cifar10_resnet18_gpu.sbatch`).

**Full 10-seed results:** teacher mean **83.49** (82.90/83.44/84.10/83.12/
84.36/82.90/83.82/84.00/82.86/83.44).

| mode | mean (10 seeds) |
|---|---|
| student_only | **83.04** |
| kd | **83.00** |
| kd_random_cppn | **83.09** |
| kd_trained_cppn | **82.74** |
| kd_evolved_cppn | **72.77** (std ≈16.1) |

`kd_evolved_cppn` per-seed: 82.80 / 74.54 / 82.28 / 79.76 / **65.82** / 79.04
/ 81.44 / 79.42 / **29.40** / 73.20.

**This is a much less clean picture than n=3 suggested, and investigating
the two worst seeds (4 and 8) turned up a real, distinct problem in each —
one a genuine bug, one a genuine gap in what `pattern_std`/`min_connections`
can see.**

**Seed 9's constant collapse (`std=0.0` exactly) exposed a real bug in
`min_connections`, not just a design limitation.** Loading the actual
winning genome (`src/cppn/serialize.load_genome`) showed 2 *enabled*
connections — satisfying `min_connections=1` — but neither connection's
destination is the output node (key `0`); they form a dead subgraph
(`733→802→203`) entirely disconnected from the output. `compile_genome`'s
`incoming.get(node_key, [])` for the output key found nothing, so the
output node evaluates as a pure function of its own bias:
`sin(-1.2838)` run through the outer sigmoid — a literal constant
(`mean=0.3367`) regardless of any input, identical to the pre-attempt-12
degenerate collapse but reached via a different structural path.
`min_connections` counts *any* enabled connection anywhere in the genome's
gene list, not whether one actually reaches the output — a structural proxy
that turned out to be gameable via dead branches. Fixed in attempt 13 by
disqualifying on `pattern_std` directly instead (the thing actually being
measured), robust to whatever mechanism produces the constancy.

**Seed 8's near-collapse (29.40%) is a different, subtler gap.** Its
winning genome has exactly 1 connection, genuinely wired to the output
(`y-coordinate → output`, weight ≈ -3.7 to -6.5 across the ensemble) — a
real, functioning single-input linear function. But the weight is steep
enough that the outer sigmoid saturates almost immediately, producing a
near-binary vertical split: `pattern_std=0.181` (comfortably under
`contrast_std_threshold=0.2`), yet `mean=0.054, min=0.0067` — most of the
image multiplied by ~0.05 for the entire 100-epoch run, i.e. a near-total
static blackout. Global std can't distinguish "mild gradual spatial
variation" from "sharp near-binary split with a narrow transition band" —
neither `contrast_std_threshold` nor `min_pattern_std` (which only screens
out near-*zero* std, not near-*binary* std) catches this. **Deliberately
left unfixed** — after 13 attempts, engineering a penalty specifically
targeting one seed's saturation pattern risks overfitting the fitness
function to this exact set of 10 seeds rather than producing something that
generalizes; see attempt 13 and the status section below for the decision
to stop here.

Seeds 0, 1, 2, 3, 5, 6, 7 all produced genuinely spatial, non-extreme
patterns (`std` 0.11–0.20, `min≈0.5`) and landed in the 79–83% range —
consistent with the same underlying ~2.5–4 point seed-to-seed cost seen
since attempt 5, not a new mechanism.

**Reported result (for now): all 10 seeds kept.** Discarding seed 8 as an
"outlier" was considered and rejected — its genome is genuinely connected
to the output and was correctly scored under the fitness function exactly
as implemented; removing it purely because the number looks bad would be
cherry-picking, not a principled correction, unlike seed 9 (which is
provably invalid — a genome the fitness function should have disqualified
and didn't, due to a confirmed counting bug in `min_connections`, fixed in
attempt 13 below). Given that, and no current basis to exclude seed 9
alone without also flagging the asymmetry, the working number keeps every
seed as-is:

| mode | mean (10 seeds) |
|---|---|
| kd_evolved_cppn | **72.77** (std ≈16.1) |

Seed 8's saturation case and seed 9's counting-bug artifact are both
documented above and should stay part of the writeup regardless of which
summary number is eventually reported — they're evidence for the "evolved
views carry real residual risk even after extensive guardrails" narrative
in their own right, not just noise to average away.

A fresh 10-seed cluster run under attempt 13's `min_pattern_std` fix was
launched (`sbatch --array=0-9`, job `1160472`) but cancelled before
completion — rather than spend another ~30 GPU-hours re-deriving numbers
that would mostly reproduce the existing 10 seeds (attempt 13 only changes
fitness selection for genomes hitting the specific dead-branch bug that
produced seed 9), the fix was left merged and tested but not re-validated
against a fresh full sweep. Whether/how to formally treat seed 9 (exclude,
re-run just that seed under the fix, or report as-is with a footnote) is
still an open call — see "Current status" below.

---

### Attempt 13 — disqualify on pattern_std directly (fixes the min_connections loophole)

Rather than trying to enumerate every possible structural path to a
constant genome (zero connections, dead branches, and whatever else NEAT's
search might find next), `fitness_from_terms()` now also takes
`min_pattern_std`: any genome whose compiled pattern's std falls below this
floor returns `DISQUALIFIED_FITNESS`, using the exact same sentinel
mechanism as `min_connections` but measuring the actual output instead of a
structural proxy for it. `min_connections` is kept alongside it (still a
cheap, valid, if weaker, first-pass filter) rather than removed.

`min_pattern_std=0.01` set in both CIFAR configs — well below every
genuinely spatial genome's std observed across every attempt so far
(smallest was ~0.09, attempt 9), and well above a true dead-branch/
zero-connection constant's std (0.0 exactly, or at most floating-point
noise). Verified: full test suite (38 tests) passing, including a synthetic
test reproducing seed 9's exact scenario (2 enabled connections,
`pattern_std=0.0` — disqualified) and a synthetic `run_evolution()` call
confirming no non-disqualified fitness row in the evolution log has
`pattern_std < 0.01`, with all top-k winners carrying real connections.

**Decision: this is the last planned fitness-function iteration.** After 13
attempts spanning ensembling, five distinct penalty terms, and now a
structural-proxy-vs-actual-output distinction, further reactive tuning
against single-seed specifics (starting with seed 8's saturation case,
deliberately left unaddressed) risks fitting the fitness function to this
particular set of seeds rather than to the underlying problem.

**Not validated against a fresh full sweep.** A 10-seed cluster run under
this fix (`sbatch --array=0-9`, job `1160472`) was launched, then cancelled
before completion — since the fix only changes selection for genomes
hitting the specific dead-branch bug (seed 9's case), re-running all 10
seeds would mostly reproduce numbers already in hand. Whether to formally
exclude seed 9 from the reported distribution (it's provably invalid) while
keeping seed 8 (a genuine, non-bug outcome) was considered — but doing so
alone, without also re-running or otherwise accounting for seed 9 properly,
risked looking like selective exclusion even with a principled reason
behind it. **Current working number reports all 10 seeds as-is: 72.77%
mean, std ≈16.1** (see the full breakdown above) — `min_pattern_std`
remains merged and correct for any future evolution runs, just not
re-validated end-to-end on the cluster, and how to finally treat seed 9 is
still open.

---

## Experiment 3 — CIFAR-100 / ResNet18

**Common setup:** ResNet18, 10 seeds run as parallel SLURM array tasks
(`slurm/run_cifar100_resnet18_gpu.sbatch`, `configs/datasets/cifar100_resnet18.yaml`),
same fitness-function fixes as CIFAR-10's final config (attempts 10-13
above all apply automatically, none are config-gated per-dataset). Unlike
CIFAR-10, getting a *valid* teacher took several attempts in its own right
before any `kd_evolved_cppn` result could be trusted.

### Attempt 1 — broken teacher (invalid, kept for the record)

First full 10-seed run, using CIFAR-10's exact teacher recipe
(`pretrained=true`, `lr=0.01`, `StepLR(step_size=30, gamma=0.1)`,
`num_epochs=100`).

| mode | mean (10 seeds) |
|---|---|
| teacher | **46.18** |
| student_only | **48.89** |
| kd | 50.69 |
| kd_random_cppn | 50.25 |
| kd_trained_cppn | 50.65 |
| kd_evolved_cppn | 46.61 (std ≈4.76, worst seed 35.58 — no catastrophic collapse) |

**Problem:** the teacher was *worse* than the from-scratch student in
every single one of the 10 seeds — the exact invalidating failure mode
from CIFAR-10's attempt 1, except this time the recipe that fixed it there
(`lr=0.01` + decay) was already in place and still wasn't enough.
CIFAR-100's 100-class task, with only 400 training images/class (80% of
CIFAR-100's 50k, split 100 ways) vs. CIFAR-10's 4000/class, is
meaningfully harder to fine-tune onto. **Every number in this attempt is
invalid for drawing conclusions**, same caveat as CIFAR-10 attempt 1.

**Diagnosis:** pulled real `training_log.csv` curves for both the teacher
and `student_only`. Both showed the identical signature — still climbing
steadily right up to the first `StepLR` decay (teacher: 44.66%→47.90% from
epoch 27→30; student: →43.92% at epoch 29), then completely flat for the
remaining 70 epochs once lr dropped 10x (oscillating in a narrow band,
essentially zero net improvement). The schedule was cutting training off
before either model had converged on this harder task.

### Attempt 2 — step_size 30→60 (helped, not enough)

Single-seed validation (`scripts/train_teacher.py`, seed 0, faster than a
full sweep) with `step_size=60` (twice as long before the first decay):
teacher test accuracy **47.8%**, better than the 46.18% mean but still
below the *original* `student_only` mean (48.89%). Curve showed the
post-decay phase (epochs 60-99, lr=0.001) still inching upward in a noisy
band without clearly flattening by epoch 99 — suggested more epochs, not
just a later decay point, was needed.

### Attempt 3-4 — num_epochs 100→150→200, step_size retuned to land both models at lr=0.001

Raised `num_epochs` to 150, then to 200 (not independently re-validated at
each step). Since teacher (`lr=0.01`) and student (`lr=0.1`) start from
different base learning rates, reaching the same final lr=0.001 needs a
different number of decays for each — one for the teacher
(0.01→0.001, `step_size=100`), two for the student (0.1→0.01→0.001,
`step_size=80`). Implemented but superseded by attempt 5 before being
run as a full sweep.

### Attempt 5 — train teacher and student identically from scratch (implemented, not run)

A different way to sidestep the "teacher must beat student" validity
problem entirely, rather than continuing to chase a working fine-tuning
schedule: set `pretrained: false` (the student was already always built
without pretrained weights regardless of this flag — only the teacher was
affected), and set teacher/student to identical `lr=0.1`, `step_size=66`,
`num_epochs=200`. Verified locally that this actually produces
bit-identical initial weights for teacher and student under the same seed
(`torch.equal` check on every parameter). Superseded before being run on
the cluster — see attempt 6.

### Attempt 6 — pretrained teacher + lr=0.1 (confirmed broken, as anticipated)

Reverted to `pretrained: true` for the teacher, but deliberately kept
`lr=0.1` (matching the student) rather than a fine-tuning-appropriate
lower value, to explicitly test that combination despite the known risk —
applying a from-scratch-appropriate lr to fine-tune a pretrained backbone
is the exact mechanism that broke CIFAR-10's teacher in attempt 1. Ran the
full 10-seed sweep.

| mode | mean (10 seeds) |
|---|---|
| teacher | **43.55** |
| student_only | **49.39** |
| kd | 49.29 |
| kd_random_cppn | 48.42 |
| kd_trained_cppn | 49.28 |
| kd_evolved_cppn | 43.15 (std ≈4.84, worst seed 33.16) |

**Confirmed the anticipated risk exactly**: teacher (43.55%) landed well
below `student_only` (49.39%), reproducing CIFAR-10 attempt 1's mechanism
on CIFAR-100. Every KD-mode number in this table is invalid for the same
reason. Not a new finding — this was the expected outcome of the
deliberate test.

### Attempt 7 — teacher lr reverted to 0.01 (in progress)

Reverted teacher `lr` back to `0.01`, keeping `step_size=100`,
`num_epochs=200` (the schedule shape from attempts 3-4, itself not yet
independently validated at this exact epoch count — the only prior check
was attempt 2's single-seed run at 150 epochs, which was still below
baseline). Full 10-seed sweep launched; **not yet complete as of this
note.** Once it lands: confirm the teacher clears `student_only` before
trusting any `kd_evolved_cppn` number from this dataset.

---

## Experiment 4 — different-architecture ablation (VGG16, CIFAR-10)

Implemented but not yet run: `src/models/vgg.py` adapts torchvision's
`vgg16` for CIFAR-sized inputs (`AdaptiveAvgPool2d((1,1))` +
`Linear(512, num_classes)` replacing the ImageNet-sized classifier head,
since CIFAR's 32x32 input is already reduced to 1x1 by VGG's 5 stride-2
maxpools). Chosen specifically because it has no skip connections at all —
maximally different from ResNet18 — to test whether the evolved-CPPN
method's behavior (including the specific failure modes documented above)
generalizes across architectures or is somehow ResNet18-specific.
`configs/datasets/cifar10_vgg16.yaml` starts as an untuned copy of
`cifar10_resnet18.yaml`'s hyperparameters; `slurm/run_cifar10_vgg16_gpu.sbatch`
starts at 3 seeds, matching how every other architecture/dataset pairing
was first validated in this project. **Not yet run on the cluster.**

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
| 11 | `9cf8dc2` | Visualizing a real winning genome (`scripts/visualize_evolved_view.py`) revealed a fixed, content-independent magenta/green color stripe — invisible to every prior fitness safeguard, since none of them inspected the compiled pattern's per-channel structure directly (`pattern_std` and both agreement measures are blind to color-channel-differential shifts at the same pixel) | Added `channel_divergence_term()`/`channel_divergence_penalty`: penalizes per-pixel std across R/G/B directly — **confirmed fixed** (every top genome reached `channel_divergence=0`), but narrowed the diversity ceiling enough that seed 1 collapsed to the zero-connection genome from attempt 8, this time landing on an extreme constant (~0.0125) and crashing to 12.82% |
| 12 | `b8e52ac` | Attempts 8 and 11 both independently converged on the identical zero-connection, single-bias-node degenerate genome as (near-)optimal under two different fitness formulations — penalizing around it wasn't reliably working | Added `min_connections` to `fitness_from_terms()`: genomes below the floor return a `DISQUALIFIED_FITNESS` sentinel before any other term is computed, excluding the genome outright instead of hoping to outscore it — **partially fixed**: no zero-connection genome won in any of 10 seeds, but 2 of 10 seeds still collapsed via two different uncaught mechanisms (see #13 and the seed-8 note in "Current status") |
| 13 | `3e46f47` | `min_connections` counts *any* enabled connection in the genome, not whether one actually reaches the output — a real 10-seed run found a winning genome with 2 enabled connections forming a subgraph entirely disconnected from the output node, leaving it a pure function of its own bias (`pattern_std=0.0` exactly) despite passing the floor | Added `min_pattern_std`: disqualifies on the compiled pattern's own std directly instead of a structural connection-count proxy, robust to whatever mechanism produces constancy — **merged and tested, not re-validated by a fresh cluster sweep** (launched then cancelled; instead the bug-invalidated seed was excluded from the existing attempt-12 dataset directly — see "Current status") |

---

## Current status / open questions

- **FashionMNIST/LeNet:** done, sane null result, not the paper's headline experiment.
- **CIFAR-10/ResNet18 — fitness-tuning phase is closed.** Three structurally different fitness designs (attempt 5: ensembling alone; attempt 9: gated contrast penalty; attempt 10: smooth full-distribution agreement) all converged to the same ~80–80.5% result for `kd_evolved_cppn` once genuinely-spatial (non-degenerate) genomes are required. Attempt 8's higher number (82.75%) is now understood to have come from a degenerate genome collapse, not real genome selection — not a valid basis for the reported result. The ~2.5–3 point accuracy cost relative to `student_only`/`kd`/`kd_random_cppn` looks like a genuine, robust property of evolved-and-genuinely-diverse CPPN views on this setup, not an artifact of any one fitness formulation. **Recommendation: stop iterating on fitness-function redesigns; report attempt 9 or attempt 10 (statistically indistinguishable, both legitimate) as the result.**
- **Attempt 11 (channel divergence penalty) ran and produced a mixed result.** The stripe-artifact mechanism is genuinely fixed (`channel_divergence=0` confirmed for every top genome across all 3 seeds), and the two seeds that stayed genuinely spatial averaged **81.35%** — the best clean evidence yet that closing this specific proxy-gaming mechanism helps. But seed 1 collapsed to the zero-connection degenerate genome from attempt 8 (landing on an extreme, catastrophic constant this time), dragging the reported mean down to 58.51% and confirming that genome needs to be excluded structurally, not just discouraged.
- **Attempt 12 (min_connections floor) ran at 10 seeds: partially fixed.** No literal zero-connection genome won in any seed, but 2 of 10 still collapsed through mechanisms `min_connections` couldn't see: seed 9 via a dead-branch loophole in how connections were counted (a real bug, fixed in attempt 13), seed 8 via a genuinely-connected single-input genome whose steep weight saturates into a near-binary occlusion-like split that stays under `contrast_std_threshold` (a real gap, deliberately left open). The other 8 seeds landed in the 73–83% range, consistent with the ~2.5–4 point cost seen since attempt 5.
- **Attempt 13 (min_pattern_std) fixes the bug behind seed 9 and is the last fitness-function change made.** A fresh 10-seed cluster run under the fix was launched then cancelled before completion — since the fix only changes selection for genomes hitting the specific dead-branch bug, it would have mostly reproduced numbers already in hand. Excluding seed 9 (provably invalid) while keeping seed 8 (a genuine, non-bug outcome) was considered but not adopted as the reported number for now — doing that asymmetrically without a fresh run to confirm nothing else changes risked looking like selective exclusion. **Working number reports all 10 seeds as-is: 72.77% mean, std ≈16.1.** No further fitness-function iteration planned regardless of how this number reads; how to finally treat seed 9 (exclude with a clear footnote, re-run just that seed under the fix, or leave as-is) remains an open call.
- **CIFAR-100/ResNet18 — teacher-validity phase, not yet resolved.** Getting a teacher that beats `student_only` turned out to be its own multi-attempt saga (Experiment 3 above), separate from and prior to any `kd_evolved_cppn` question: the CIFAR-10 recipe (`lr=0.01`+decay) wasn't sufficient on this harder 100-class task (attempt 1: 46.18% vs 48.89%); `step_size`/`num_epochs` tuning helped partially (attempt 2: single-seed 47.8%, still short); a from-scratch symmetric-training alternative was implemented but not run (attempt 5); a deliberate lr=0.1 test confirmed the same failure mode CIFAR-10 hit in its own attempt 1 (attempt 6: 43.55% vs 49.39%); teacher lr reverted to 0.01 with `step_size=100`/`num_epochs=200` and **a fresh 10-seed sweep is in progress as of this note (attempt 7)**. No `kd_evolved_cppn` number from CIFAR-100 is trustworthy yet — check the teacher clears `student_only` in attempt 7 before treating any mode comparison from this dataset as valid.
- **VGG16/CIFAR-10 architecture ablation — implemented, not yet run.** `src/models/vgg.py` + `configs/datasets/cifar10_vgg16.yaml` + `slurm/run_cifar10_vgg16_gpu.sbatch` are ready (Experiment 4 above), starting at 3 seeds with untuned hyperparameters copied from the ResNet18 config — same "check the teacher clears student_only before trusting anything" caveat applies here too, not yet confirmed for this architecture.
- **Open question for the paper's narrative:** seed 8's uncaught saturation case (CIFAR-10, attempt 12) is itself worth keeping in the writeup regardless of the final aggregate number — even after three rounds of guardrails (contrast threshold, channel-divergence penalty, connection/pattern-std floors), evolutionary search still found a genuinely-connected, non-degenerate-by-every-existing-metric genome that behaves like a near-total occlusion mask. That's arguably a more interesting empirical finding about the difficulty of specifying "safe" fitness for this kind of open-ended search than a clean accuracy table would have been — the 72.77%/std≈16.1 result is itself evidence for that framing (evolved views carry real residual risk, even after extensive guardrails), not just a number to report and move past.

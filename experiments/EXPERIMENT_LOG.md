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

### Attempt 5 — ensemble mode (in progress)

Pivoted from threshold-tuning to a structural fix: `run_baseline_sweep.py
--use-ensemble` now averages the `kd_evolved_cppn` consistency loss over the
top-5 evolved genomes (`top_k_genomes.pkl`) instead of betting the whole run
on a single one — directly dilutes the damage any one problematic pattern
can do, rather than trying to predict in advance which patterns are safe.
Costs ~3x more compute for that mode (one view forward/backward pass per
ensemble member).

**Status: submitted, not yet complete as of this writeup.** This is the
result to watch — if it stabilizes `kd_evolved_cppn` closer to the other
modes, the single-genome selection was the core problem; if it's still
unstable, the issue likely runs deeper (candidates: `alpha=0.9` too
aggressive specifically for CPPN-view modes, no direct contrast/patchiness
penalty in the fitness function, or static single-pattern application being
fundamentally too risky regardless of which genome(s) get used).

---

## Bugs found and fixed (chronological)

| # | Commit | What broke | Fix |
|---|---|---|---|
| 1 | `ca0d08f` | `.gitignore`'s `data/`/`results/` patterns (no leading `/`) matched `src/data/` by basename anywhere in the tree, silently excluding a real source module from every commit | Anchored patterns to repo root (`/data/`, `/results/`) |
| 2 | `35f9007` | `run_baseline_sweep.py`'s `sweep_id = sweep_{int(time.time())}` had 1-second resolution; parallel SLURM array tasks launched within the same second collided, each overwriting the previous task's `summary.csv` | Added a `uuid.uuid4().hex[:8]` suffix |
| 3 | `153a096` | Teacher fine-tuning used a from-scratch-appropriate LR (`0.1`) with no decay, degrading pretrained ImageNet features — teacher ended up *worse* than the from-scratch student | `teacher.lr=0.01` + `StepLR` decay added to `train_teacher()` and `DistillTrainer` |
| 4 | `38bdbb6` | `compile_genome()`'s outer sigmoid had no pre-scale; genomes with naturally-bounded raw outputs (`sin`/`tanh`/`clamped` activations) got squashed into a narrow `[0.27, 0.73]` band, an architectural ceiling preventing near-identity or near-blank patterns | Added `OUTER_SIGMOID_SCALE=5.0` (matches neat-python's own internal sigmoid scale) before the squash |
| 5 | `5721236` | (hypothesis, not confirmed) `tau_low=0.3` fitness gate too permissive for the newly-widened pattern range | Raised to `0.5` — **did not resolve the underlying issue**, kept as a mild additional safeguard |
| 6 | `c39b125` | Single evolved genome applied as a static, unchanging transform for all 100 epochs — occasionally a genome that looks fine on a small fitness-evaluation probe batch is actually harmful as a repeated training-time signal | `--use-ensemble`: average consistency loss over top-5 evolved genomes instead of one |

---

## Current status / open questions

- **FashionMNIST/LeNet:** done, sane null result, not the paper's headline experiment.
- **CIFAR-10/ResNet18:** teacher training and the pattern-range architecture are now on solid footing; `kd`/`kd_random_cppn`/`kd_trained_cppn` behave sensibly and consistently across attempts 2–4. `kd_evolved_cppn` has not yet produced a stable, trustworthy result — attempt 5 (ensemble) is the open question.
- **CIFAR-100/ResNet18:** not yet run — waiting on CIFAR-10 to stabilize first, since the same fixes (and open questions) apply.
- If ensembling doesn't stabilize `kd_evolved_cppn`, next candidates to try (not yet attempted): reduce `alpha` specifically for CPPN-view consistency loss, add a direct contrast/std penalty to the fitness function, or try `view_op: additive` instead of `multiplicative` (structurally can't fully zero out a region the way multiplicative can).
- Once `kd_evolved_cppn` is stable, more seeds (5–10 rather than 3) would be needed for a statistically confident claim either way — real seed-to-seed variance has been substantial throughout this investigation.

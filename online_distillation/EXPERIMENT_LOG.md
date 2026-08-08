# Online Distillation — Experiment Log

Running record for the teacher-free line of experiments in this folder,
separate from the main pipeline's [`../experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md)
(which always requires a pretrained/fine-tuned teacher). See
[`README.md`](README.md) for the full method rationale.

Method recap: no teacher model anywhere in the loop. An untrained,
randomly-initialized coordinate CPPN pattern (same construction as the main
pipeline's `kd_random_cppn --random-cppn-variant coord`, no evolution, no
teacher-based fitness scoring) produces a fixed "view" of every training
image. Two modes combine that view with plain cross-entropy training:

- **`hard_label_augmentation`** (Option B) — `(1-alpha)*CE(raw, labels) +
  alpha*CE(view, labels)`. The view is pure data augmentation; no
  consistency/soft-target loss, no self-reference.
- **`self_consistency_random_cppn`** (Option A, self-distillation) —
  `(1-alpha)*CE(raw, labels) + alpha*KL(student(raw).detach() ||
  student(view))`. The student's own detached raw-image prediction stands
  in for a teacher.

Compared against the main pipeline's `student_only` baseline (plain CE, no
CPPN view at all — not reimplemented here, use the existing number from
`../experiments/EXPERIMENT_LOG.md`).

---

## Attempt 1 — first 3-seed sweep, CIFAR-10 / ResNet18

**Setup:** ResNet18 from scratch (no pretrained weights, no teacher),
`alpha=0.5`, 100 epochs, `lr=0.1`/`step_size=30`/`gamma=0.1`, 3 seeds
(`online_distillation/slurm/run_cifar10_online_gpu.sbatch`,
`online_distillation/configs/cifar10_resnet18.yaml`). Untuned first values
for `alpha`/schedule — a direct copy of the main pipeline's student
hyperparameters, not independently tuned for this teacher-free setting.

Job `1167370`, array `0-2`. All 3 seeds completed cleanly (exit 0, ~51-53
min each — dramatically cheaper than the main pipeline's 3-7 hour sweeps,
since there's no teacher training and no CPPN evolution).

| mode | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| hard_label_augmentation | 82.76 | 82.26 | 83.42 | **82.81** | ≈0.58 |
| self_consistency_random_cppn | 81.06 | 83.06 | 82.70 | **82.27** | ≈1.07 |

For reference, the main pipeline's `student_only` (10-seed, teacher-based
setup config, `../experiments/EXPERIMENT_LOG.md`) landed at **83.04%**.

**Read:** both teacher-free modes land close behind `student_only` — within
0.2-0.8 points — and both are remarkably stable (std well under 1.1, no
outliers), a sharp contrast to the teacher-based `kd_evolved_cppn` results
throughout the main pipeline, which repeatedly showed large variance
(std often 4-16) and occasional catastrophic collapses across 13 attempts
of fitness-function iteration. A random CPPN view, used with no teacher at
all, doesn't meaningfully hurt training whether treated as pure
augmentation or as a self-consistency target — but at this sample size (3
seeds) neither mode shows a clear improvement over `student_only` either,
just a similar or very slightly lower mean.

**Not yet answered:** whether either mode provides a genuine, above-noise
lift over `student_only`, or whether the CPPN view is essentially inert
here (neither helping nor hurting) — 3 seeds isn't enough to distinguish
"slightly worse" from "statistically indistinguishable," and `alpha`/the
LR schedule are both untuned first guesses copied from the teacher-based
config, not validated for this setting.

---

## Current status / open questions

- **Attempt 1 complete.** Both teacher-free modes are stable and land near
  (slightly below) `student_only`, unlike the teacher-based
  `kd_evolved_cppn`'s persistent ~2.5-4+ point cost and instability. Open
  call: run more seeds to check if the small gap to `student_only` is real
  or noise, try tuning `alpha`/schedule for this setting specifically, or
  treat "teacher-free CPPN views are roughly neutral, not harmful" as
  itself an interesting enough finding to report as-is.
- `evolve_cppn`-without-a-teacher (README's "Not yet attempted" section) is
  still unstarted; not a near-term priority until it's clearer whether
  teacher-free CPPN views are worth pursuing further at all.
- A genuine peer-network "online distillation" mode (two co-trained
  students, Deep-Mutual-Learning style) was discussed but not implemented
  — `self_consistency_random_cppn` is single-model self-distillation, not
  peer-based online distillation, despite the folder's name.

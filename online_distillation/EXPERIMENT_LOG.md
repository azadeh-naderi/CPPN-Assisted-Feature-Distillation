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

**Status: launched, not yet complete.** Job `1167370`, array `0-2`. Results
to follow once it finishes.

---

## Current status / open questions

- First sweep in progress (attempt 1) — no results yet.
- `evolve_cppn`-without-a-teacher (README's "Not yet attempted" section) is
  still unstarted; not a near-term priority until attempt 1's simpler
  modes show whether teacher-free CPPN views are worth pursuing at all.
- A genuine peer-network "online distillation" mode (two co-trained
  students, Deep-Mutual-Learning style) was discussed but not implemented
  — `self_consistency_random_cppn` is single-model self-distillation, not
  peer-based online distillation, despite the folder's name.

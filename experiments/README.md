# Experiment Results Summary

Quick-reference tables for the final, reported 10-seed results across every
dataset/mode covered in this project. These are summary numbers only — see
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (main pipeline, teacher-based
modes) and [`../online_distillation/EXPERIMENT_LOG.md`](../online_distillation/EXPERIMENT_LOG.md)
(teacher-free modes) for the full narrative: every attempt, bug, diagnosis,
and the reasoning behind each config choice. Learning curves for the
CIFAR-10 row below: [`figures/cifar10_resnet18_learning_curves.png`](figures/cifar10_resnet18_learning_curves.png).

> **Protocol caveat:** every number on this page uses the main pipeline's
> legacy setup — torchvision's ImageNet ResNet18 stem on 32x32 inputs, a
> 10% slice of the *training* set as the test set, no weight decay, and a
> same-architecture teacher/student. That's why ResNet18 sits around 83% /
> 50% on CIFAR-10 / CIFAR-100 instead of the usual ~95% / ~77%, and these
> numbers aren't comparable to published KD results. A corrected benchmark
> protocol is being built in [`../method_paper/`](../method_paper/).

## CIFAR-10 / ResNet18 (10 seeds)

| mode | mean test accuracy | notes |
|---|---|---|
| teacher | 83.48% | pretrained + fine-tuned, clears `student_only` comfortably |
| student_only | 83.04% | |
| kd | 83.00% | |
| kd_random_cppn | 83.09% | |
| kd_trained_cppn | 82.74% | |
| kd_evolved_cppn | 72.77% | std≈16.1 — includes seed 9, whose collapse was later found to be caused by a confirmed `min_connections` counting bug (fixed in attempt 13); kept in this number per an explicit decision to report all 10 seeds as-is rather than exclude asymmetrically (`EXPERIMENT_LOG.md`) |
| hard_label_augmentation *(teacher-free)* | 82.77% | std≈1.50, one dip at seed 3 (79.02%) |
| self_consistency_random_cppn *(teacher-free)* | 82.71% | std≈0.79 — the most stable result of any mode, teacher-based or not, in this project |

## CIFAR-100 / ResNet18 (10 seeds)

| mode | mean test accuracy | notes |
|---|---|---|
| teacher | 48.57% | still ~1pt below `student_only` on average (a per-seed coin flip — wins 5/10, loses 4, ties 1 — not a clean pass); took 7 attempts of schedule tuning just to get this close, see `EXPERIMENT_LOG.md` Experiment 3 |
| student_only | 49.51% | |
| kd | 53.26% | |
| kd_random_cppn | 52.24% | |
| kd_trained_cppn | 53.39% | |
| kd_evolved_cppn | 50.35% | std≈1.95, no outliers — the most stable `kd_evolved_cppn` result of any dataset in this project |
| hard_label_augmentation *(teacher-free)* | 50.02% | std≈0.81 — beats both `student_only` and the teacher here, unlike on CIFAR-10 |
| self_consistency_random_cppn *(teacher-free)* | 48.22% | std≈0.96 — slightly below `student_only`, roughly level with the teacher |

## Reading these numbers together

- **CIFAR-10:** every mode except `kd_evolved_cppn` lands in a tight
  82.7–83.1% band — teacher-based and teacher-free approaches are
  statistically indistinguishable from `student_only` and from each other.
  `kd_evolved_cppn` is the clear outlier, both in mean (10 points lower)
  and variance (std an order of magnitude larger) — see the "evolved views
  carry real residual risk" framing in `EXPERIMENT_LOG.md`'s open questions
  for why that gap was never fully closed despite 13 fitness-function
  iterations.
- **CIFAR-100:** `kd`/`kd_random_cppn`/`kd_trained_cppn` all show a real
  3-4 point lift over `student_only` (unlike CIFAR-10, where distillation
  barely moves the needle) — but the teacher itself never cleanly beat
  `student_only` here, so treat any CIFAR-100 comparison with that caveat
  in mind. `kd_evolved_cppn` sits between `student_only` and the other KD
  modes, and is notably *more* stable here than on CIFAR-10.
- **Teacher-free modes** (`hard_label_augmentation`,
  `self_consistency_random_cppn`) land close behind `student_only` on
  CIFAR-10 (attempts 2-3 tried closing that gap via pattern resampling and
  didn't beat the simple fixed-pattern baseline, `online_distillation/EXPERIMENT_LOG.md`),
  but the picture flips on CIFAR-100: `hard_label_augmentation` actually
  *beats* both `student_only` and the teacher there. Both teacher-free
  modes remain consistently more stable than `kd_evolved_cppn` on every
  dataset tried, and consistently behind the real teacher-based KD modes
  where a teacher is available — whether the CIFAR-100 crossover is a
  genuine dataset-dependent effect or something more specific to that run
  is still open.

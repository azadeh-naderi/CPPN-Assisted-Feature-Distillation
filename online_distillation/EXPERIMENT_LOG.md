# Online Distillation — Experiment Log

Running record for the teacher-free line of experiments in this folder,
separate from the main pipeline's [`../experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md)
(which always requires a pretrained/fine-tuned teacher). See
[`README.md`](README.md) for the full method rationale.

Method recap: no teacher model anywhere in the loop. An untrained,
randomly-initialized coordinate CPPN pattern (same construction as the main
pipeline's `kd_random_cppn --random-cppn-variant coord`, no evolution, no
teacher-based fitness scoring) produces a fixed "view" of every training
image. Both modes share one `OnlineDistillTrainer._step()`
(`online_distillation/src/online_trainer.py`) that always computes:

```python
raw_logits  = student(normalize(images_raw))          # gradient flows
loss_hard   = CrossEntropy(raw_logits, labels)

view_raw    = apply_pattern(images_raw, cppn_pattern)  # CPPN-warped images
view_logits = student(normalize(view_raw))             # gradient flows
```

and then branches on `mode` for the one remaining term:

- **`hard_label_augmentation`** (Option B) — the view is treated as pure
  data augmentation, scored against the *true label* like any other
  augmented image:
  ```python
  loss_view = CrossEntropy(view_logits, labels)
  loss = (1 - alpha) * loss_hard + alpha * loss_view
  ```
  No soft-target/consistency term, no self-reference — mechanically no
  different from adding a second augmented copy of the batch with a
  CPPN-specific transform instead of e.g. random crop.

- **`self_consistency_random_cppn`** (Option A, self-distillation) — the
  view is instead pushed toward matching the model's *own* prediction on
  the unmodified image, reusing `src.distill.losses.kd_loss` (the exact
  same KD formula the main, teacher-based pipeline uses elsewhere):
  ```python
  loss_view = kd_loss(view_logits, raw_logits.detach(), temperature)
  #         = KL( softmax(raw_logits.detach()/T) || softmax(view_logits/T) ) * T^2
  loss = (1 - alpha) * loss_hard + alpha * loss_view
  ```
  `raw_logits.detach()` is critical: it stops gradient from flowing back
  into the raw-image branch through this term, so the consistency loss
  only pulls the *view* prediction toward the raw prediction, never the
  reverse (the same asymmetry `kd_loss(student_view, teacher_view)` has in
  the main pipeline — here the "teacher" role is just played by the
  model's own detached raw-image output instead of a separate pretrained
  model). Confirmed via test (`online_distillation/tests/test_online_trainer.py`)
  that `raw_logits.detach()` genuinely has no `grad_fn`, i.e. this isn't
  merely `.detach()` being ignored/tracked incorrectly.

`kd_loss`'s full definition (`src/distill/losses.py`), for reference:
```python
def kd_loss(student_logits, teacher_logits, temperature):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature ** 2
```
`reduction="batchmean"` sums the per-class KL terms then divides by batch
size only (the mathematically correct batch-averaged KL, not PyTorch's
plain `"mean"`, which would also divide by the number of classes).
Multiplying by `temperature**2` rescales the gradient magnitude back up,
since the `/temperature` inside both softmaxes shrinks it by ~`1/T²` —
standard practice from the original Hinton et al. distillation paper, so
`alpha` behaves consistently regardless of which `T` is chosen.

Both formulas reduce to the same `(1-alpha)*loss_hard + alpha*loss_view`
shape as the main pipeline's `combined_loss(..., use_soft_kd=False)`
(`configs/datasets/cifar10_resnet18_cppn_only.yaml`) — the only difference
between that ablation and this folder is *where* `loss_view`'s target
comes from: a real frozen teacher's prediction on the CPPN view there, vs.
either the true label (Option B) or the model's own detached raw-image
prediction (Option A) here.

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

**Extended to 10 seeds** (`sbatch --array=3-9 online_distillation/slurm/run_cifar10_online_gpu.sbatch`,
job `1167740`, seeds 3-9, all completed cleanly at ~51-52 min each):

| mode | mean (10 seeds) | std |
|---|---|---|
| hard_label_augmentation | **82.77** | ≈1.50 (worst seed: 79.02, seed 3) |
| self_consistency_random_cppn | **82.71** | ≈0.79 (no real outliers) |

Full per-seed values: `hard_label_augmentation` — 82.76 / 82.26 / 83.42 /
79.02 / 83.10 / 82.12 / 84.20 / 83.94 / 82.86 / 84.04. `self_consistency_random_cppn`
— 81.06 / 83.06 / 82.70 / 83.12 / 81.64 / 82.86 / 83.74 / 83.34 / 82.70 /
82.92.

**Read: the 3-seed read holds up.** Both modes stayed essentially flat
(82.81%→82.77%, 82.27%→82.71%) rather than converging toward or away from
`student_only` (83.04%) as more seeds came in — a 0.27-0.33 point gap that
looks like a small, real, stable effect rather than noise that would
average out, though still small enough that it's not a dramatic finding
either way. `self_consistency_random_cppn`'s std (≈0.79) remains the
tightest of any mode — teacher-based or not — seen anywhere in this
project. `hard_label_augmentation` picked up one real dip at 10 seeds
(seed 3, 79.02%, a ~3.7 point drop from its own mean) that wasn't visible
at n=3, widening its std to ≈1.50 — worth a quick look at that seed's
training curve if pursued further, but nowhere near the near-random-guessing
collapses `kd_evolved_cppn` showed under the teacher-based fitness search.

---

## Attempt 2 — per-epoch CPPN pattern resampling (regressed badly, unguarded)

Attempt 1 used one fixed random CPPN pattern for the entire 100-epoch run
per seed. Hypothesis: this trains the model to be invariant to one
specific, arbitrary transform — a narrower task than the genuine
augmentation-style regularization real per-batch augmentations (crop/flip)
provide by varying every step, and a plausible explanation for the small,
consistent ~0.3-point cost relative to `student_only` seen in attempt 1.

Added `resample_pattern` to `OnlineDistillTrainer`
(`online_distillation/src/online_trainer.py`): when enabled, draws a
fresh random genome/pattern at the start of every epoch instead of once
at the start of training. Per-batch resampling was considered and
rejected — each draw constructs a full throwaway `neat.Population`
(`create_random_genome`'s only available construction path) purely to
discard all but one genome; fine once per epoch (~3ms, measured locally,
negligible next to a full epoch of gradient descent), wasteful thousands
of times per epoch. Verified end-to-end with a real ResNet18: the pattern
demonstrably changes across epochs when enabled and stays exactly fixed
when disabled (the default, preserving attempt 1's exact behavior for any
future re-run). Applies to both modes when run via `--modes all`, not just
`self_consistency_random_cppn`.

New config `online_distillation/configs/cifar10_resnet18_resample.yaml`
(`cppn.resample_pattern: true`, otherwise identical to attempt 1's
config) and `slurm/run_cifar10_online_resample_gpu.sbatch`, 3 seeds.

**Ran, regressed badly.** Job `1169655`, all 3 seeds completed cleanly
(exit 0, ~51-53 min each — same cost as attempt 1, resampling adds no
meaningful overhead).

| mode | mean (3 seeds) | attempt 1 (3-seed) | change |
|---|---|---|---|
| hard_label_augmentation | **72.87%** | 82.81% | **−9.9** |
| self_consistency_random_cppn | **68.43%** | 82.27% | **−13.8** |

**Diagnosis:** pulled a real `training_log.csv` — the curve is a noisy
sawtooth for the entire 100 epochs, never converging cleanly (repeated
sharp crashes: epoch 2 52→43%, epoch 11 62→41%, epochs 16-18 down to
33-44%, and similar drops recurring roughly every 10-20 epochs throughout,
each followed by a partial recovery before the next one). Root cause:
`apply_pattern`'s multiplicative view is `image * pattern`
(`src/cppn/apply.py`), so a low-mean pattern is a near-blackout mask — the
exact mechanism behind the main pipeline's worst evolved-genome collapses
(e.g. attempt 11's 0.0125-mean constant, `../experiments/EXPERIMENT_LOG.md`).
But evolved genomes there get guardrails specifically built to prevent
this (`contrast_std_threshold`, `min_pattern_std`); raw, unconstrained
random genomes here get none. A 200-draw local sample of unconstrained
random genomes found **47% have `std > 0.2`** (the main pipeline's own
validated "harmful contrast" threshold) and **35% have `mean` outside
`[0.2, 0.8]`**. Attempt 1's single fixed draw either happened to land in
the "safe" ~50-65% majority or didn't; resampling every epoch means
repeated exposure to that risk across the whole run instead of one
one-time roll of the dice — consistent with the observed recurring crash
pattern. **Hypothesis disconfirmed: per-epoch resampling, done naively,
actively hurts.** See attempt 3.

---

## Attempt 3 — guardrail on resampled patterns (not yet run)

Direct fix for attempt 2's diagnosed mechanism: added
`max_pattern_std`/`min_pattern_mean` to `OnlineDistillTrainer._resample()`
— reject-and-redraw any pattern outside bounds (up to
`max_resample_attempts=50`, falling back to the last draw with a logged
warning rather than hanging or crashing if no draw ever passes). Same idea
as the main pipeline's fitness-time guardrails, just applied at draw time
since there's no fitness function to gate on here.

`max_pattern_std=0.2`, `min_pattern_mean=0.3` — the former matches the
main pipeline's own validated threshold directly; the latter is a
conservative cutoff against near-blackout patterns, informed by the main
pipeline's catastrophic 0.0125-mean collapse case. Verified with a real
ResNet18 over 10 resampled epochs that every single draw satisfies both
bounds. New config
`online_distillation/configs/cifar10_resnet18_resample_guarded.yaml` and
`slurm/run_cifar10_online_resample_guarded_gpu.sbatch`, 3 seeds. **Not yet
run.**

---

## Current status / open questions

- **Attempt 1 complete at 10 seeds.** Both teacher-free modes are stable
  and land consistently ~0.3 points below `student_only` (82.77% and
  82.71% vs. 83.04%) — a small, real-looking gap, not the ~2.5-4+ point
  cost and instability the teacher-based `kd_evolved_cppn` showed across
  13 fitness-tuning attempts.
- **Attempt 2 (unguarded per-epoch resampling) complete at 3 seeds:
  regressed badly** (72.87%/68.43% vs. attempt 1's 82.81%/82.27%) —
  unconstrained random genomes have no safety net against near-blackout/
  high-contrast draws, and resampling every epoch repeatedly exposed
  training to that risk. Hypothesis disconfirmed as tested.
- **Attempt 3 (guardrailed resampling) implemented, not yet run.** Tests
  whether resampling *with* a guardrail against extreme draws recovers
  attempt 1's stability while still getting pattern diversity — if it
  lands back near attempt 1's ~82.5-82.8%, that would suggest the
  diversity idea itself wasn't wrong, just unguarded exposure to extreme
  patterns was. If it's still worse than attempt 1, the fixed-pattern
  design may just be better for this setting, full stop.
- `evolve_cppn`-without-a-teacher (README's "Not yet attempted" section) is
  still unstarted; not a near-term priority until it's clearer whether
  teacher-free CPPN views are worth pursuing further at all.
- A genuine peer-network "online distillation" mode (two co-trained
  students, Deep-Mutual-Learning style) was discussed but not implemented
  — `self_consistency_random_cppn` is single-model self-distillation, not
  peer-based online distillation, despite the folder's name.

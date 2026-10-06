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

## Phase 0 — benchmark protocol (implemented, not yet run)

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

**Not yet run** — blocked on cluster access (`/project/ahoover/an57` gives
`Permission denied` from login node `app01` since the recent Wulver
update; works from `app02`).

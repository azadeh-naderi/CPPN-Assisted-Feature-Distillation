# Online distillation (teacher-free CPPN experiments)

Separate line of experiments from the main pipeline (`src/`, `configs/`,
`scripts/`, `slurm/`), which always requires a pretrained/fine-tuned
teacher. This folder tests whether the CPPN-view idea still provides value
with **no teacher model anywhere** — no teacher training, no CPPN-fitness
scoring against a frozen teacher, no teacher-vs-student soft-label loss.

Motivated by how much of this project's effort (`experiments/EXPERIMENT_LOG.md`)
went into just getting a valid teacher on CIFAR-100 — this asks whether the
core mechanism (comparing a model's behavior on a CPPN-transformed view
against some reference) needs a *separate, stronger* model at all, or
whether a model can usefully learn from its own view-consistency.

## Modes

Both use an untrained, randomly-initialized coordinate CPPN pattern (same
construction as the main pipeline's `kd_random_cppn --random-cppn-variant
coord`, drawn once per seed via `src.cppn.evolve.create_random_genome`) —
genuinely teacher-free at every stage, unlike `kd_evolved_cppn`/
`kd_trained_cppn`, which both need a frozen teacher to score fitness or
train the CPPN's own weights (a harder, separate problem not attempted
here — see "Not yet attempted" below).

- **`hard_label_augmentation`** — the CPPN view is treated as a pure data
  augmentation: `(1-alpha)*CE(raw, labels) + alpha*CE(view, labels)`, both
  terms against the true label. No consistency/soft-target loss, no
  self-reference.
- **`self_consistency_random_cppn`** — the student's own raw-image
  prediction (detached, no gradient) stands in for a teacher, reusing
  `src.distill.losses.kd_loss` exactly as the main pipeline's
  `DistillTrainer` does elsewhere, just comparing the model against itself:
  `(1-alpha)*CE(raw, labels) + alpha*KL(student(raw).detach() || student(view))`.

Compare both against the main pipeline's `student_only` baseline (plain CE,
no CPPN view at all) — not reimplemented here, use the existing numbers
from `experiments/EXPERIMENT_LOG.md`.

## Running

```bash
python online_distillation/scripts/train_student_online.py \
    --config online_distillation/configs/cifar10_resnet18.yaml \
    --mode self_consistency_random_cppn --seed 0

# or the full sweep (both modes, one seed):
python online_distillation/scripts/run_online_sweep.py \
    --config online_distillation/configs/cifar10_resnet18.yaml --modes all --seeds 0
```

Results land in `results/online_distillation/` (gitignored, same convention
as the main pipeline's `results/`). SLURM: `slurm/run_cifar10_online_gpu.sbatch`,
starting at 3 seeds.

## Not yet attempted

**Evolving (or gradient-training) a CPPN without any teacher.** The main
pipeline's `evolve_cppn.py` fitness function (`diversity_term`,
`agreement_term`) is entirely defined relative to a frozen teacher's
features/predictions — removing the teacher there means finding a
different reference to evolve against (e.g. the student's own features,
made self-referential and evolving alongside training rather than frozen;
or a fixed feature extractor unrelated to the classification task). That's
a materially harder redesign than either mode above and hasn't been
started.

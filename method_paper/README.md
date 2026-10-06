# Method paper: student-aware evolved views for KD

Everything for the positive-method paper lives here: benchmark-protocol
models, data, training, configs, SLURM scripts and tests. It reuses
low-level pieces from the main pipeline (`src/cppn/`, `src/distill/losses.py`,
`src/utils/`) but none of its models, data loading or trainers — those use a
protocol that isn't comparable to published KD numbers (see "Why a separate
protocol" below).

Plan and running results: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Phases

| phase | goal | status |
|---|---|---|
| 0 | Benchmark protocol + reproduce published `student_only` / `kd` numbers | implemented, not yet run |
| 1 | Corrected baselines: `kd_random_cppn`, `kd_trained_cppn`, `kd_evolved_cppn`, KD + RandAugment/CutMix, random search at equal budget | not started |
| 2 | New method: student-aware online view evolution (disagreement / gradient-alignment fitness) | not started |
| 3 | Full results: 4–6 pairs, ablations, CIFAR-100-C robustness, calibration | not started |

## Why a separate protocol

The main pipeline's numbers (`../experiments/`) were produced under a setup
that reviewers would reject and that likely depressed every result:

| issue | main pipeline | here |
|---|---|---|
| ResNet stem | torchvision ImageNet stem (7x7 stride 2 + maxpool) — 32x32 input is 8x8 before stage 1 | CIFAR stem (3x3 stride 1, no maxpool) |
| test set | 10% slice of the *training* set (`train=True` only) | official 10k test split |
| training data | 80% of training set | full 50k |
| weight decay | none | 5e-4 |
| schedule | 100 epochs, step 30 | 240 epochs, decay at 150/180/210 |
| teacher/student | same architecture (ResNet18→ResNet18) | standard capacity-gap pairs |
| CIFAR-100 normalization | CIFAR-10 stats | CIFAR-100 stats |

## Layout

```
method_paper/
├── configs/cifar100.yaml      # protocol + Phase 0 pairs
├── src/
│   ├── models/                # cifar_resnet, wrn, vgg, mobilenetv2, registry
│   ├── data.py                # official splits, raw-[0,1] loaders + normalize()
│   └── trainer.py             # SGD+WD, MultiStepLR, CE / KD, per-epoch test eval
├── scripts/
│   ├── train.py               # one teacher or student run
│   └── launch.py              # SLURM array index -> job
├── slurm/                     # phase0_teachers.sbatch, phase0_students.sbatch
└── tests/                     # test_mp_*.py
```

Models implement the benchmark architectures from their published specs
(Tian et al., ICLR 2020; reference code RepDistiller, BSD-2-Clause) —
written from spec, not copied. Every architecture matches the published
parameter count within 0.6% (`tests/test_mp_models.py` enforces 2%).

## Running Phase 0

```bash
T=$(sbatch --parsable method_paper/slurm/phase0_teachers.sbatch)
sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase0_students.sbatch
```

Results go to `results/method_paper/cifar100/{teachers,students}/` with
deterministic names (`resnet56_seed0`, `resnet56__resnet20__kd__seed0`, …);
finished runs are skipped on resubmission unless `--overwrite` is passed.

Wiring check without a GPU or dataset download:

```bash
python method_paper/scripts/train.py --config method_paper/configs/cifar100.yaml \
    --role student --arch resnet20 --mode ce --fake-data --max-batches 2 --epochs 1 \
    --results-root /tmp/mp_smoke
```

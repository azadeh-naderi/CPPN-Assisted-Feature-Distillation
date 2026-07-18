# CPPN-Evolution-Assisted Knowledge Distillation

Evolves CPPNs (Compositional Pattern Producing Networks, in the genuine
NEAT/evolutionary-computation sense — evolved topology **and** heterogeneous
per-node activation functions, à la Stanley's original CPPN work) to discover
auxiliary "views" of training images that improve teacher→student knowledge
distillation.

The original prototype (see [`legacy/`](legacy/)) called a fixed-architecture,
randomly-initialized MLP a "CPPN" and used it to generate a second image view
for an auxiliary distillation loss — but that network was never trained *or*
evolved. This repo replaces it with an actual NEAT evolutionary search over
CPPN genomes, selected by a fitness function computed from a frozen teacher.

## How it works

1. **Teacher training** — standard supervised training (`scripts/train_teacher.py`).
2. **CPPN evolution** (`scripts/evolve_cppn.py`) — a population of CPPN
   genomes (inputs: `x, y, r, channel` — coordinates only, no pixel data) is
   evolved with `neat-python`. Each genome is compiled once into a spatially
   coherent pattern over the image's coordinate grid (via a custom vectorized
   torch evaluator in [`src/cppn/compile.py`](src/cppn/compile.py) — NEAT's own
   `activate()` is far too slow to evaluate a full grid across a population
   every generation). That pattern is combined with every image in a probe
   batch via a multiplicative mask (`view = image ⊙ pattern`), and fitness is
   a **cheap, frozen-teacher-only proxy**: feature diversity between the raw
   image and the view, gated by prediction agreement so genomes that either
   do nothing (identity) or destroy all class signal (adversarial noise) both
   score near zero. See [`src/cppn/fitness.py`](src/cppn/fitness.py) and the
   plan doc for the full derivation.
3. **Student distillation** (`scripts/train_student.py`) — trains a student
   under one of several modes (`--mode`), optionally consuming the evolved
   CPPN's view as an extra consistency-loss term alongside standard KD.

## Baselines / ablations

All via one flag: `scripts/train_student.py --mode {...}`

| mode | description |
|---|---|
| `student_only` | plain CE, no teacher |
| `kd` | standard soft-target KD |
| `kd_random_cppn --random-cppn-variant legacy` | literal reproduction of the old prototype's untrained pixel-intensity CPPN |
| `kd_random_cppn --random-cppn-variant coord` | untrained coordinate-only CPPN (same family as evolved ones, just unselected) — isolates "does evolution help" |
| `kd_trained_cppn` | gradient-trained coordinate CPPN (differentiable proxy objective) — isolates "does evolution help" from "does gradient-learning help" |
| `kd_evolved_cppn` | the method: evolved genome's view. Add `--ensemble-genomes top_k_genomes.pkl` for the multi-genome ensemble ablation |

Further ablations (fitness-gate thresholds, population size/generations,
`view_op` multiplicative-vs-additive) are config/CLI overlays on
`kd_evolved_cppn`, no new scripts needed.

`scripts/run_baseline_sweep.py --config <cfg> --modes all --seeds 0,1,2` runs
the full matrix against one shared teacher checkpoint and writes
`results/sweeps/<sweep_id>/summary.csv` for paper tables.

## Setup

```bash
pip install -r requirements.txt
```

## Smoke test (CPU, no cluster needed)

Verifies the whole pipeline end-to-end on tiny settings (population=6, 2
generations, 384-image subset, 2 epochs) in a few minutes — **not** meant to
produce meaningful accuracy, just to catch plumbing bugs before scaling up:

```bash
python scripts/smoke_test.py
```

Also run the unit tests, especially `test_compile_matches_neat.py`, which
cross-checks the vectorized genome compiler against neat-python's own
evaluator — the correctness gate for the whole custom-compile approach:

```bash
python -m pytest tests/ -v
```

## Real runs

Point at a full-size config and drop `--smoke`, e.g.:

```bash
python scripts/train_teacher.py --config configs/datasets/cifar10_resnet18.yaml
python scripts/evolve_cppn.py --config configs/datasets/cifar10_resnet18.yaml --teacher-ckpt <path>
python scripts/train_student.py --config configs/datasets/cifar10_resnet18.yaml --mode kd_evolved_cppn --teacher-ckpt <path> --genome-path <path>
```

or run everything via `scripts/run_baseline_sweep.py`. `configs/neat/cppn_neat.cfg`
has the full-scale NEAT settings (`pop_size=150`); tune population/generations
there for your compute budget.

## Layout

- `src/data/` — dataset loading/splitting
- `src/models/` — LeNet / ResNet18
- `src/cppn/` — coordinate grid, genome compiler, evolution loop, fitness,
  gradient-trainable CPPN baseline, legacy-CPPN baseline
- `src/distill/` — losses, training loop for all modes
- `configs/` — per-dataset YAML + NEAT `.cfg` files
- `scripts/` — CLI entry points
- `results/` — gitignored; per-run checkpoints, logs, and genome artifacts
- `legacy/` — original prototype scripts, kept for provenance

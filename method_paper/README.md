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
| 0 | Benchmark protocol + reproduce published `student_only` / `kd` numbers | done — reproduces published numbers |
| 1 | Corrected baselines: `kd_random_cppn`, `kd_trained_cppn`, `kd_evolved_cppn`, KD + RandAugment/CutMix, random search at equal budget | done — Gate 1 negative |
| 2 | New method: student-aware online view evolution (disagreement / gradient-alignment fitness) | running (3 seeds) |
| 3 | Full results: 6 pairs, ablations, CIFAR-100-C robustness, calibration | implemented, waiting on Gate 2 |

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
| view search inputs | evolution and the trained CPPN feed raw [0,1] images to a teacher trained on normalized ones | teacher wrapped in `NormalizedModel` |

## Layout

```
method_paper/
├── configs/cifar100.yaml      # protocol, Phase 0 pairs, Phase 1 modes/views/augment
├── src/
│   ├── models/                # cifar_resnet, wrn, vgg, mobilenetv2, registry
│   ├── data.py                # official splits, raw-[0,1] loaders + normalize(), RandAugment
│   ├── augment.py             # CutMix
│   ├── views.py               # CPPN view search: random / trained / evolved / random search
│   ├── online_views.py        # Phase 2: online view evolution, student-aware fitnesses; Phase 3 controls
│   ├── metrics.py             # ECE, NLL, teacher agreement, CIFAR-100-C
│   ├── fitness_study.py       # genome bank for the fitness-vs-accuracy study
│   └── trainer.py             # SGD+WD, MultiStepLR, CE / KD / KD+views / KD+CutMix
├── scripts/
│   ├── train.py               # one teacher or student run (any mode)
│   ├── launch.py              # SLURM array index -> job (one stage per phase, plus tune)
│   ├── evaluate_extra.py      # calibration / agreement / robustness for finished runs
│   ├── fitness_study.py       # bank building + correlation analysis
│   └── summarize.py           # results tables vs published numbers
├── slurm/                     # one sbatch per stage: phase0_*, phase1_baselines, phase2_online,
│                              #   tune, phase3_main, phase3_ablations, evaluate_extra,
│                              #   fitness_study_bank, fitness_study_train
└── tests/                     # test_mp_*.py
```

Models implement the benchmark architectures from their published specs
(Tian et al., ICLR 2020; reference code RepDistiller, BSD-2-Clause) —
written from spec, not copied. Every architecture matches the published
parameter count within 0.6% (`tests/test_mp_models.py` enforces 2%).

## Running Phase 0

From the repo root, once per clone: create `cluster_env.sh` from
`cluster_env.sh.example` and stage CIFAR-100 on a login node, since Wulver
compute nodes have no internet and the jobs would otherwise fail at
`download=True`:

```bash
cp cluster_env.sh.example cluster_env.sh && source cluster_env.sh
mkdir -p results/logs
python -c "from torchvision import datasets; [datasets.CIFAR100('data', train=t, download=True) for t in (True, False)]"
```

Then submit:

```bash
T=$(sbatch --parsable method_paper/slurm/phase0_teachers.sbatch)
sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase0_students.sbatch
```

Results go to `results/method_paper/cifar100/{teachers,students}/` with
deterministic names (`resnet56_seed0`, `resnet56__resnet20__kd__seed0`, …);
finished runs are skipped on resubmission unless `--overwrite` is passed.

Compare finished runs (any phase) against the published numbers, with mean
± std over seeds and wall time per epoch:

```bash
python method_paper/scripts/summarize.py
```

## Running Phase 1

| mode | what the student trains on |
|---|---|
| `kd_randaugment` | KD, RandAugment (2 ops, magnitude 9) on top of crop+flip |
| `kd_cutmix` | KD on CutMix-ed batches (prob 0.5, Beta(1,1)) |
| `kd_random_cppn` | KD + one unscored random coordinate-CPPN view |
| `kd_trained_cppn` | KD + one gradient-trained coordinate-CPPN view |
| `kd_evolved_cppn` | KD + top-5 diverse evolved views (attempt-14 fitness) |
| `kd_random_search` | KD + top-5 diverse views from random genomes, same fitness, same number of evaluations as NEAT |

View modes use `gamma·CE + alpha·(½·KD(raw) + ½·mean KD(view))`, the main
pipeline's loss, and search their views at the start of each run (seeded by
the student seed), saving patterns and search logs to `<run_dir>/views/`.
Needs the Phase 0 teacher checkpoints:

```bash
python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase1 --list
sbatch method_paper/slurm/phase1_baselines.sbatch
```

If the teachers are still training, add `--dependency=afterok:<teacher job id>`.

## Running Phase 2

One NEAT population per run, re-evolved for a few generations every 20
epochs against the current student (warm start: 10 generations at epoch 0,
5 per round after). Each batch uses one view sampled from the round's top-5,
randomly shifted (±4 px) and flipped, with the same loss as Phase 1's view
modes.

| mode | fitness |
|---|---|
| `kd_online_disagree` | (a) KL(teacher ‖ student) on the view, over images the teacher still gets right |
| `kd_online_gradalign` | (b) cosine between the student's last-layer gradients of KD on the view and CE on other training images |
| `kd_online_teacher` | old teacher-only fitness, re-evolved online (isolates "student-aware" from "online") |

For (a) and (b) the fitness is gated on the teacher's accuracy on the view
relative to the raw images (`online.gate`), and every mode keeps the
existing guardrails. Each round's patterns and the full per-genome log go
to `<run_dir>/views/round<NN>_epoch<EEE>/` and `views/online_evolution_log.csv`.

```bash
sbatch method_paper/slurm/phase2_online.sbatch   # 3 modes x 3 seeds on resnet56 -> resnet20
```

## Running Phase 3

Set `phase3.method` (and `phase3.method_set`, if tuning changed anything)
in `configs/cifar100.yaml` to Gate 2's winner, then:

```bash
# 1. teachers / student_only / kd for the two new pairs (resnet110 -> resnet32, vgg13 -> vgg8);
#    finished runs exit immediately, so only the new ones train
T=$(sbatch --parsable method_paper/slurm/phase0_teachers.sbatch)
sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase0_students.sbatch
# 2. main table: 6 pairs x (CutMix + method) x 3 seeds
sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase3_main.sbatch
# 3. ablations on resnet56 -> resnet20 (21 tagged runs)
sbatch method_paper/slurm/phase3_ablations.sbatch
```

| ablation tag | what changes |
|---|---|
| `every10`, `every40` | re-evolution frequency N (default 20) |
| `average` | all top-5 views every batch instead of one sampled |
| `nojitter` | no random shift/flip of the pattern |
| `rsearch` | random genomes scored with the same fitness, same evaluations per round (evolution vs random search) |
| `randviews` | random genomes, no scoring (does selection matter at all) |
| `noise` | low-frequency noise masks instead of CPPN patterns (does the CPPN family matter) |

Fitness (a) vs (b) vs old comes from the Phase 2 modes.

**Calibration and robustness** (`scripts/evaluate_extra.py`): ECE, NLL and
top-1 agreement with the teacher on the test split, plus accuracy and ECE
on CIFAR-100-C, for every finished student; results land in
`extra_metrics.json` and in an extra `summarize.py` table. Download
CIFAR-100-C (Hendrycks & Dietterich, ICLR 2019; ~2.9 GB) on a login node
first:

```bash
wget -P data https://zenodo.org/records/3555552/files/CIFAR-100-C.tar && tar -xf data/CIFAR-100-C.tar -C data
sbatch method_paper/slurm/evaluate_extra.sbatch
```

`data/CIFAR-100-C/` should then hold 19 corruption `.npy` files plus
`labels.npy`; the reported average uses the standard 15.

## Fitness-vs-accuracy study (analysis paper)

Does the teacher-only fitness predict how much a view helps the student?
A genome bank of 40 views — 20 from a NEAT run, 20 from an equal-budget
random search, at evenly spaced fitness quantiles of each pool — is
rescored on one shared probe set (fitness, every term, teacher accuracy on
the view, pattern brightness/contrast). Each view then trains a resnet20
student for 60 epochs (same schedule shape as the 240-epoch one, 2 seeds),
against a `kd` reference on the same short schedule (3 seeds).

```bash
B=$(sbatch --parsable method_paper/slurm/fitness_study_bank.sbatch)          # ~30 min
sbatch --dependency=afterok:$B method_paper/slurm/fitness_study_train.sbatch  # 83 runs, ~30 min each
python method_paper/scripts/fitness_study.py analyze --config method_paper/configs/cifar100.yaml
```

`analyze` prints Spearman correlations (with bootstrap 95% CIs) between the
accuracy change and each predictor — overall and within the evolved and
random pools, since the two pools cover different fitness ranges — and
writes `per_genome.csv`, `correlations.csv` and `fitness_study.png` under
`results/method_paper/cifar100/fitness_study/`. The bank's views are in
`fitness_study/bank/bank.png`.

## Analysis-paper controls

Two remaining controls (`analysis` section of the config, 21 jobs):

| jobs | what | question |
|---|---|---|
| 0–17 | `kd_random_cppn`, `kd_evolved_cppn`, `kd_online_teacher` on wrn_40_2 → wrn_16_2 and vgg13 → MobileNetV2 | does "fitness-selected views hurt, unscored don't, online use recovers" generalize beyond resnet56 → resnet20? |
| 18–20 | noise masks selected by the method's fitness (`noise_filtered`) | is noise a bad view family, or were the `noise` ablation's masks bad only because nothing filtered them? |

```bash
sbatch method_paper/slurm/analysis.sbatch
```

`fitness_study.py analyze` also reports partial correlations (controlling
for how much of the image a view keeps) and the evolved-vs-random pool
means — the short-schedule version of Gate 1.

## Archiving results

`results/` is gitignored and lives only on the cluster. After each batch of
runs, copy everything except model checkpoints into the tracked
`results_archive/method_paper/` (same layout) and commit it:

```bash
bash method_paper/scripts/archive_results.sh
git add results_archive/method_paper && git commit -m "Archive method_paper results" && git push
```

Every analysis script reads the archive like the live results, e.g.
`python3 method_paper/scripts/summarize.py --results-root results_archive/method_paper`.
Checkpoints (needed only for `evaluate_extra.py`) stay on the cluster; back
them up separately if the project space is at risk.

## Tuning (validation split only)

Every number above is test accuracy, so it can't be used to choose
hyperparameters. Tuning runs hold out 5,000 training images
(`--set val_size=5000`, seeded per student seed), report their accuracy
every epoch (`val_top1` in `training_log.csv`, `final_val_top1` in
`summary.json`), and keep them out of view search. They distill from the
same full-data teacher, write to `…__val5000[__<tag>]` run directories, and
show up as separate rows in `summarize.py` with a `val_final` column. Any
`--set` other than `val_size` requires a `--tag`, so a variant can never
overwrite or be confused with a benchmark run.

```bash
# once: kd reference on the validation split
LAUNCH_ARGS="--set val_size=5000" sbatch --array=0-2 method_paper/slurm/tune.sbatch
# one tagged submission per variant of the method
LAUNCH_ARGS="--set val_size=5000 --set online.gate.tau_low=0.5 --tag gate05" \
    sbatch --array=3-5 method_paper/slurm/tune.sbatch
```

Pick the variant with the best `val_final` over `kd`'s, then run it once on
the full training set (`phase2_online.sbatch` with the winning `--set`s,
without `val_size`) for the test number that goes in the paper.

Wiring check without a GPU or dataset download:

```bash
python method_paper/scripts/train.py --config method_paper/configs/cifar100.yaml \
    --role student --arch resnet20 --mode ce --fake-data --max-batches 2 --epochs 1 \
    --results-root /tmp/mp_smoke
```

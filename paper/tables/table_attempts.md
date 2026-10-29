# Table: 14 revisions of the teacher-only fitness (Section 3)

Compiled by hand from `experiments/EXPERIMENT_LOG.md` ("Experiment 2 —
CIFAR-10 / ResNet18"). **Legacy protocol** — ImageNet-stem ResNet18 teacher
and student on 32×32 inputs, test accuracy on a 10% slice of the training
set, no weight decay, 100 epochs, 3 seeds unless noted — so the numbers are
not comparable to the benchmark results in the rest of the paper. What the
table documents is the sequence: each fix closes the exploit just found and
the search finds another. `kd_evolved_cppn` and `kd` are mean test top-1
(%); student_only was 82.9–83.9 throughout.

| # | change to fitness / pipeline | evolved | KD | what evolution found (exploit or failure) |
|---|---|---|---|---|
| 1 | baseline (teacher misconfigured; run invalid) | 73.77 | 80.48 | — |
| 2 | teacher fixed | 78.73 | 82.70 | patterns confined to a mid-gray band by an output squashing bug |
| 3 | output range widened | 71.53 | 82.84 | near-binary static occlusion masks (std 0.43), seeds 61.8–79.1 |
| 4 | agreement gate raised (tau_low 0.3 → 0.5) | 62.76 | 82.29 | occlusion masks still pass; one seed at 30.8 |
| 5 | average over top-5 genomes | 80.53 | 82.31 | stable but consistently 2–3 pt below every baseline |
| 6 | separate view-loss weight (additive) | 68.55 | 82.62 | loss imbalance destabilizes training (not a fitness change) |
| 7 | separate view-loss weight (fixed budget) | 64.52 | 82.99 | one seed at 34.8: a bad genome in the ensemble |
| 8 | contrast penalty on pattern std | 82.75 | 82.28 | **zero-connection genomes: constant brightness, no spatial view** |
| 9 | contrast penalty above a threshold only | 80.21 | 82.47 | spatial views return, and so does the 2–3 pt cost |
| 10 | soft (full-distribution) agreement | 80.07 | 83.24 | **content-independent colored stripes** pass logit-based agreement |
| 11 | channel-divergence penalty | 58.51 | 82.14 | **constant near-black genome (0.0125)**, 3 of 5 ensemble slots duplicates |
| 12 | disqualify zero-connection genomes (10 seeds) | 72.77 | 83.00 | **disconnected subgraph → constant output**; saturated near-binary blackout (seed at 29.4) |
| 13 | disqualify on pattern std directly | — | — | fix for #12's constant; not re-run |
| 14 | diversity-aware ensemble selection | — | — | carried into the corrected protocol (Phase 1) |

Bold: degenerate solutions that satisfied every constraint in force at the
time. Only the student's accuracy revealed them — none violated the fitness.

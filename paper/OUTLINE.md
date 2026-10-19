# Analysis paper — outline

Working title: **"When the Proxy Points the Wrong Way: Evolved Auxiliary Views
for Knowledge Distillation as a Case Study in Fitness-Proxy Gaming"**
(alternatives: "Evolving Views a Student Can't Use"; "Teacher-Only Fitness
Filters but Does Not Rank").

Target: GECCO (Evolutionary Machine Learning track) or an EML/NeurIPS
workshop. Check the current call for page limits and deadlines.

All numbers: CIFAR-100, final-epoch test top-1, mean ± std over 3 seeds,
unless noted. Figures: `python paper/make_figures.py` (reads
`results_archive/method_paper/`). Source of every number:
`method_paper/EXPERIMENT_LOG.md`.

---

## Abstract (draft, ~150 words)

Neuroevolution of compositional pattern-producing networks (CPPNs) is an
appealing way to generate auxiliary training views, and a frozen teacher
offers a cheap fitness signal: reward views that change the teacher's
features while preserving its predictions. This study shows that such a
teacher-only fitness is gamed. Under a standard CIFAR-100 distillation
benchmark, views selected by the fitness lower student accuracy by 1.9
points relative to plain distillation, while an unscored random view does
not. In a controlled study of 40 views spanning the fitness range, the
fitness separates catastrophic views from benign ones, but among the
evolved views that pass its own agreement gate it is negatively correlated
with student benefit (ρ = −0.61), driven by its diversity term (ρ = −0.70,
robust to controls). How views are applied matters more than which are
selected: sampling one view per batch removes most of the damage, and
student-aware fitness functions add nothing measurable. [Generality and
robustness results to be added.]

---

## 1. Introduction

- Hook: evolutionary search optimizes exactly what it is told to; cheap
  proxy fitnesses are the norm in evolutionary ML, and their failure modes
  are under-documented.
- Setting: NEAT-evolved CPPN masks as auxiliary views for knowledge
  distillation (KD); fitness = teacher-feature diversity × gate(teacher
  agreement).
- Contributions:
  1. A documented case of proxy gaming: 14 fitness revisions, each closing
     one exploit and opening another (Section 3).
  2. Under a corrected benchmark protocol, fitness-selected views hurt the
     student while unscored views don't (Section 5, Fig. 1).
  3. A view-level study showing the fitness filters but does not rank, and
     is anti-aligned with student benefit where evolution operates
     (Section 6, Fig. 3).
  4. Ablations isolating what actually matters: how views are applied, not
     which fitness selects them (Section 7, Fig. 2).
- Takeaway for practitioners: validate a proxy against the downstream
  objective *in the region the optimizer reaches*, not across random
  samples.

## 2. Background and related work

- CPPNs and NEAT (Stanley 2007; Stanley & Miikkulainen 2002); CPPNs as
  image generators/augmenters.
- Knowledge distillation (Hinton et al. 2015); benchmark protocol (Tian et
  al., CRD, ICLR 2020).
- Augmentation and auxiliary views for KD; adversarial / data-free KD;
  learned augmentation (AutoAugment, RandAugment, CutMix).
- Proxy objectives and reward hacking / specification gaming; Goodhart's
  law in evolutionary computation; surrogate fitness.
- TODO: literature pass.

## 3. The teacher-only fitness and how evolution gamed it

- Fitness: diversity (1 − cosine of teacher penultimate features, raw vs
  view) × gate(soft agreement of teacher softmax), plus penalties and
  disqualification floors (`src/cppn/fitness.py`).
- The 14 revisions as a table: exploit found → fix → next exploit
  (static occlusion masks; constant single-node patterns; colored stripes
  passing top-1 agreement; disconnected subgraphs gaming
  `min_connections`; duplicate genomes in the ensemble). Source:
  `experiments/EXPERIMENT_LOG.md`, attempts 1–14.
- Point: each fix was local; none checked the student.
- Figure candidate: gallery of exploit patterns (from `results_archive/`
  legacy genomes) — TODO, not in `make_figures.py` yet.

## 4. Experimental protocol

- Standard benchmark (Tian et al. 2020): CIFAR ResNet/WRN/VGG/MobileNetV2,
  official test split, 240 epochs, SGD 0.05, decay at 150/180/210,
  WD 5e-4, KD 0.1·CE + 0.9·KL·T², T = 4. Reproduction: KD within 0.5 pt of
  published on 5 of 6 pairs, 1.1 pt above on vgg13 → MobileNetV2 (Table 1).
- Why the protocol matters: the legacy setup (ImageNet stem on 32×32,
  test = slice of train, no WD, same-arch pair, unnormalized fitness
  inputs) produced numbers not comparable to the literature — one
  paragraph, appendix table.
- Views: multiplicative mask, loss γ·CE + α·(½·KD(raw) + ½·mean KD(view)).
- Statistics: Welch t-tests (n = 3), Spearman ρ with bootstrap CIs.

**Table 1** — teachers, student_only and KD vs published, 6 pairs.

## 5. Fitness-selected views hurt (resnet56 → resnet20)

**Fig. 1** (`fig1_main_results`).

| | top-1 | vs KD | p |
|---|---|---|---|
| KD | 70.59 ± 0.32 | — | |
| one unscored random CPPN view | 70.66 ± 0.39 | +0.07 | n.s. |
| random search, top-5 by fitness | 69.07 ± 0.63 | −1.52 | 0.03 |
| NEAT, top-5 by fitness | 68.65 ± 0.28 | −1.94 | 0.001 |

- Fitness-selected vs unscored: −1.59 (p = 0.03). NEAT vs random search:
  −0.42 (p = 0.38, not established).
- Strong augmentation doesn't help this student either (RandAugment
  −0.28, CutMix −0.06): the harm isn't generic to perturbing inputs.
- **Generality (PENDING `analysis.sbatch`, Fig. 6):** same three
  conditions on wrn_40_2 → wrn_16_2 and vgg13 → MobileNetV2.

## 6. The fitness filters but does not rank (view-level study)

**Fig. 3** (`fig3_fitness_study`), **Fig. 4** (`fig4_view_gallery`).

- Design: 40 views (20 NEAT, 20 equal-budget random search) at fitness
  quantiles, rescored on a shared probe set; one 60-epoch student per view
  × 2 seeds vs short-schedule KD (67.83 ± 0.34).
- Two regimes: 11 views close the gate (teacher keeps ≤ 39% of its
  accuracy) and are catastrophic (−5 to −19 pt); 29 pass and sit within
  −1.6 .. +0.4 pt. Whole-bank ρ(fitness) = +0.45 just reflects the gate
  separating these regimes.
- Within the gate-open regime, evolved views (n = 17): ρ(fitness) =
  −0.61 (p = 0.009); ρ(diversity) = −0.70 (p = 0.002), −0.52 (p = 0.04)
  controlling for pattern brightness; ρ(agreement) = +0.65.
- Robustness: diversity holds at every cutoff; the fitness correlation is
  sensitive to one borderline view (−0.36, p = 0.15 with it) — state this.
- Gallery: the highest-fitness views are near-identical gray masks with a
  thin bright stripe (diversity from dimming plus a structured edge); the
  most harmful are saturated colour fields and high-frequency rings.
- Caveat: absolute effects on the short schedule are larger than on the
  full one; the study is about ranking.

## 7. What actually matters: ablations

**Fig. 2** (`fig2_ablations`), base = online view evolution with the
student-aware disagreement fitness (70.87 ± 0.28).

- No effect: re-evolution frequency (10 / 40 epochs), random shift/flip,
  NEAT vs random search (all |Δ| ≤ 0.26, p ≥ 0.28).
- **Applying all 5 views every batch: −1.22 (p = 0.03)** — the offline
  configuration of Section 5, so part of its harm is how views are applied.
- **Moving the teacher-only fitness online with one view per batch
  recovers +1.76 (p = 0.002)** (68.65 → 70.41).
- Student-aware fitnesses (disagreement 70.87, gradient alignment 70.72)
  don't beat KD significantly (+0.28, p = 0.32) or an unscored trained CPPN
  view (70.87) — selection adds nothing once views stop doing harm.
- **Selection as a safety filter:** unscored random views online are
  unstable (68.39 ± 3.20; one seed collapsed to 64.71 after a late view
  refresh at epoch 200, low LR) — Fig. candidate: that seed's curve.
- **CPPN vs noise:** unscored noise masks −2.80 vs KD (p = 0.001);
  fitness-selected noise masks PENDING (separates family from selection).

## 8. Secondary effects: calibration and robustness

**Fig. 5** (`fig5_robustness`).

- KD students are worse calibrated than student_only (ECE ≈ 12% vs 6.5%);
  view variants lower ECE slightly.
- CIFAR-100-C (15 corruptions × 5 severities): fitness-selected views are
  *worse* (NEAT 40.81, random search 41.56 vs KD 43.30; p ≈ 0.06) — no
  hidden robustness benefit. RandAugment best (44.50, n.s.).
- PENDING: ablation and new-pair rows after the second evaluation pass.

## 9. Discussion

- Why the proxy fails: a multiplicative diversity × gate objective with a
  ramped gate lets search trade agreement for diversity once the gate
  opens; the student pays for exactly that trade.
- General lesson: validate proxies in the optimizer's operating region;
  a proxy that correlates across random samples can anti-correlate where
  search ends up (Fig. 3a vs 3b).
- Limitations: one dataset (CIFAR-100), small-student KD, 3 seeds, a
  short-schedule proxy for the view-level study, one mask operator
  (multiplicative), threshold sensitivity of the fitness correlation.

## 10. Conclusion

---

## Remaining work

- [ ] `analysis.sbatch` → Fig. 6 (generality) and the filtered-noise row of
      Fig. 2; re-run `archive_results.sh`, then `make_figures.py`.
- [ ] Second `evaluate_extra` pass archived → complete Fig. 5 / Section 8.
- [ ] Table 1 generator (teachers / student_only / KD vs published).
- [ ] Legacy exploit gallery + 14-attempt table (Section 3).
- [ ] Collapsed-seed training curve (Section 7).
- [ ] Literature review.

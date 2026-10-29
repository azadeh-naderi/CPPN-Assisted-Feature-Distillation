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
student-aware fitness functions add nothing measurable. The pattern holds
across three teacher–student pairs, including a cross-architecture pair
where evolved views cost 4.5 points, and fitness-selected views are also
less robust to common corruptions.

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
- **Table 2** (`tables/table_attempts.md/.tex`, compiled by hand from
  `experiments/EXPERIMENT_LOG.md`): the 14 revisions, exploit found → fix →
  next exploit (gray-band squashing, static occlusion masks, constant
  zero-connection genomes, colored stripes passing logit-based agreement,
  a near-black constant, disconnected subgraphs gaming `min_connections`,
  duplicate ensemble members). Legacy-protocol numbers, flagged as such.
- Point: each fix was local; four degenerate solutions satisfied every
  constraint in force, and only the student's accuracy exposed them.
- No legacy exploit gallery: the archive only holds genomes from attempts
  1–3, and the genome compiler's output scaling changed after them, so
  recompiling them wouldn't reproduce the patterns that were trained on.
  Fig. 4 shows the same failure families (blackouts, saturated colour
  fields) from the corrected protocol instead.

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

**Table 1** (`tables/table1_benchmark.md/.tex`) — teachers, student_only
and KD vs published, 6 pairs, final-epoch (ours) vs published (which may
be best-epoch; ours is ~0.2–0.4 lower at final epoch than at best).

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
- **Generality (Fig. 6, `fig6_generality`):** change vs KD for one random
  view / NEAT top-5 / online teacher fitness — wrn_40_2 → wrn_16_2: +0.14 /
  **−1.57** (p = 0.049) / −0.22; vgg13 → MobileNetV2: −0.66 / **−4.51**
  (p = 0.06, seeds −2.4 to −6.5) / −0.89. Evolved views hurt on every pair,
  most on the weakest student; online use recovers most of the loss
  (+1.35, p = 0.047 on WRN).

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
- **Selection as a safety filter (Fig. 7, `fig7_refresh_instability`):**
  with unscored random views, test accuracy swings at every late view
  refresh in all three seeds — e.g. 70.7 → 59.6 (seed 1) and 69.1 → 56.4
  (seed 2) right after the epoch-200 refresh, single epochs down to ~31% —
  and recovers when the next refresh replaces the bad views. Final numbers
  (69.94 / 70.53 / 64.71) depend on which views happen to be active at the
  end: seed 2's last round includes a near-black mask (mean 0.14). The
  fitness-filtered method's curves stay flat. The instability is general to
  unscored views, not one unlucky seed.
- **CPPN vs noise:** noise masks hurt whether unscored (67.79, −2.80 vs
  KD, p = 0.001) or selected by the same fitness (67.93; vs unscored
  p = 0.63), while CPPN masks selected that way score 70.87 (p < 0.001).
  The CPPN family's smoothness is what makes views benign; selection
  doesn't rescue a bad family.

## 8. Secondary effects: calibration and robustness

**Fig. 5** (`fig5_robustness`).

- KD students are worse calibrated than student_only (ECE ≈ 12% vs 6.5%);
  view variants lower ECE slightly.
- CIFAR-100-C (15 corruptions × 5 severities): fitness-selected views are
  *worse* (NEAT 40.81, random search 41.56 vs KD 43.30; p ≈ 0.06) — no
  hidden robustness benefit. RandAugment best (44.50, n.s.).
- Fig. 5, all three pairs (CIFAR-100-C top-1, change vs KD):
  wrn_40_2 → wrn_16_2: NEAT −0.89 (44.62 vs 45.51), random view −0.02,
  online teacher fitness +0.30; vgg13 → MobileNetV2: NEAT **−5.79**
  (40.68 vs 46.47), random view −1.36, online −1.79. Robustness tracks
  clean accuracy: evolved views are worst on every pair.
- Calibration: on MobileNetV2, KD is badly overconfident (ECE 18.3%) and
  any view lowers it a lot (online 10.3, evolved 10.5) — calibration gains
  don't separate good views from bad ones, so they aren't evidence for the
  method. Noise masks are also least robust (unscored 38.65, filtered
  39.38 vs KD 43.30 on resnet56 → resnet20).

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

- [x] `analysis.sbatch` → Fig. 6 (generality) and the filtered-noise row of
      Fig. 2.
- [x] Second `evaluate_extra` pass archived.
- [x] Third `evaluate_extra` pass (all 191 runs evaluated) → Fig. 5 has
      all three pairs.
- [x] Table 1 generator (`make_figures.py` → `tables/table1_benchmark.*`).
- [x] 14-attempt table (`tables/table_attempts.*`); legacy gallery dropped
      (see Section 3).
- [x] Refresh-instability figure (Fig. 7).
- [x] LaTeX draft (`paper/main.tex`, acmart sigconf, anonymous review mode;
      compile in Overleaf by uploading the whole `paper/` folder). Full
      draft of every section; orange \todo markers for related work and
      acknowledgments.
- [ ] Literature review → Section 2 and the intro's framing; verify
      `references.bib` details against publisher versions.
- [ ] Trim to the GECCO page limit once the call is out (draft is long;
      candidates: shorten Section 3 prose, move Table 1 or Fig. 7 to the
      supplement).

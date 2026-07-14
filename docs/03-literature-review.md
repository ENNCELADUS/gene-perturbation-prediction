# Literature Review: Results

**Status:** complete. All six stages closed; the novelty statement is frozen (§7).
**Graded against:** [`docs/02-acceptance-criteria.md`](02-acceptance-criteria.md), frozen before evidence collection. **Contract:** [`docs/01-blueprint.md`](01-blueprint.md). **What happens next:** [`docs/04-roadmap.md`](04-roadmap.md).
**Full evidence tables and the UNVERIFIED registers** stay in the source memos under [`ideaspark_run/cell-fate-outcome-dynamics/`](../ideaspark_run/cell-fate-outcome-dynamics/). This document carries the verdicts and the decision-relevant evidence.

## Verdict summary

| Stage | Verdict | Source memo |
|---|---|---|
| L0 — Ontology | Complete. No selection between Candidate A and Candidate B (by design). | [`L0/L0_ontology_memo.md`](../ideaspark_run/cell-fate-outcome-dynamics/L0/L0_ontology_memo.md) |
| Gate 1 — Measurement validity | A: `proceed` (narrowed to the A2 ceiling). B: `proceed`. Neither triggers a stop. | [`gate1/gate1_measurement_map.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate1/gate1_measurement_map.md) |
| Gate 2A — Lineage prevalence | **Insufficient evidence -> bounded validation pilot** (not a stop). | [`gate2a/gate2a_lineage_phenomenon.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate2a/gate2a_lineage_phenomenon.md) |
| Gate 2B — Population prevalence | **Insufficient evidence -> bounded validation pilot** (not a stop). | [`gate2b/gate2b_population_phenomenon.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate2b/gate2b_population_phenomenon.md) |
| Gate 3 — Prior art / novelty | **Partially-scooped.** Generic phenomenon demonstrated; the exact target conjunction is not. | [`gate3/gate3_prior_art_matrix.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate3/gate3_prior_art_matrix.md), [`gate3/gate3_rerun_expanded.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate3/gate3_rerun_expanded.md) |
| Gate 4 — Prospective designs | Both candidates route to **finding 3: a genuine methodological opportunity.** No design supports Analysis P as specified. | [`gate4/gate4_prospective_designs.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate4/gate4_prospective_designs.md) |

---

## 1. L0 — Ontology and Latent-to-Observable Maps

**Verdict:** complete. Per the blueprint §8, this stage does **not** select between Candidate A
(lineage/clone) and Candidate B (population) — both latent-to-observable maps are
presented with equal weight, and that discipline is preserved here.

### Decision-relevant evidence

| Question | Finding | Citation |
|---|---|---|
| Is there a citable, off-the-shelf "point of no return" for division commitment? | **No.** The classical restriction-point view (a fixed point in G1) is contradicted by live-cell evidence: sister cells bifurcate at mitotic exit into immediate-recommit vs. uncommitted populations whose eventual commitment depends on mitogen signaling integrated over a **restriction window spanning the previous cell cycle** — a probabilistic window, not a point. | Spencer et al. 2013, *Cell*, DOI `10.1016/j.cell.2013.08.062`; reframed as a probabilistic transition by Pennycook & Barr 2020 review |
| Is death's point of no return well-characterized? | Better-characterized than division's (MOMP is a directly measurable, reporter-verifiable commitment step) but **the field's own nomenclature committee calls the general boundary "hitherto poorly defined."** | Galluzzi et al. 2018 NCCD recommendations, full text fetched, DOI `10.1038/s41418-017-0012-4` |
| Does high mitochondrial-read fraction reliably flag dying / dissociation-attrited cells? | **UNRESOLVED — preserved as an open disagreement, not adjudicated.** Standard QC treats high-mito as dissociation-induced stress/attrition. A cancer-specific reanalysis of 9 datasets (~441,000 cells, 134 patients) found this association weak-or-absent in *malignant* cells specifically, and that mito-based filtering **discards viable, metabolically distinct cells** rather than removing dying ones. | Standard QC view: van den Brink et al. 2017 (*Nat Methods*), Hong et al. 2022 (*Nat Commun*, full text fetched). Contra: Yates, Kraft & Boeva 2025 (*Genome Biol*, full text fetched, DOI `10.1186/s13059-025-03559-w`) |
| Can quiescent and senescent cells be told apart from a single-timepoint snapshot? | **No.** Canonical senescence biomarkers are graded, track duration of cell-cycle withdrawal, and are "nearly molecularly indistinguishable from each other at a snapshot in time." | Ashraf, Fernandez & Spencer 2023, *Nat Commun*, DOI `10.1038/s41467-023-40132-0` |

### Observation / Implication / Decision

- **Observation.** Every term this program needs to operationalize — fate commitment,
  death's point of no return, biological loss vs. assay attrition, quiescence vs.
  senescence — has at least one genuinely unresolved definitional or measurement
  disagreement in the current literature, evidenced on both sides.
- **Implication.** Any "early" cutoff this program adopts (blueprint §4.4's `t <= 0.25*T` +
  one full cell cycle fallback) is an **operational choice made in the absence of a
  ready-made biological cutoff**, not a citation of settled science — and the mito-QC
  disagreement means a standard QC pipeline cannot be trusted uncritically in K562
  without first checking which side of the disagreement applies there.
- **Decision.** Carry both the restriction-window and MOMP-based commitment concepts
  forward as *candidate* operationalizations, not resolved facts; treat any K562 mito-QC
  filtering decision as an explicit, checkable assumption, not a default.

---

## 2. Gate 1 — Measurement and Observation Validity

**Verdict:** **Candidate A: `proceed`**, narrowed to the A2 ceiling already fixed at L0.
**Candidate B: `proceed`.** Neither candidate triggers a stop.

### Decision-relevant evidence

| Question | Finding | Citation |
|---|---|---|
| Which readout combinations jointly identify division rate *and* death rate as separate quantities? | Only two: **(a) direct single-cell event-calling imaging** (FUCCI + caspase/Annexin, when per-cell tracking is maintained), and **(b) formal stochastic/phylodynamic inference** from population-count variance or lineage trees. In both inference cases, **death-rate recovery is markedly weaker and more bias-prone than division-rate recovery** — an asymmetry that recurs across every count-based or tree-based method found. | Chen et al. 2024, *PLOS Comput Biol*, DOI `10.1371/journal.pcbi.1011888`; Pilarski, Stadler & Seidel 2026, *PLOS Comput Biol*, DOI `10.1371/journal.pcbi.1014370` |
| Does any existing genetic-perturbation dataset supply a genuinely early, lineage-linked, cancer-relevant state? | **No.** Components exist only separately: RENGE (CRISPR KO + 4 timepoints, non-cancer hiPSC, no lineage barcode); an HT-29 CRISPRi preprint (cancer + 2 timepoints, no lineage, abstract-only-verified); LARRY (lineage barcode + state->fate, no genetic perturbation); Watermelon (lineage + cancer lines, drug not CRISPR). Stated as written: **an absence-of-a-combined-dataset finding, not evidence that such a design is infeasible.** | Gate 1 §4 dataset table |
| Do quantitative bounds exist on state-dependent, pre-capture (MNAR) cell loss? | **No general bound was found.** Stated exactly as the evidence permits: *"no bound was found," not "no bound exists."* | Denisenko et al. 2020 (*Genome Biol*); Hong et al. 2022 (*Nat Commun*, full text); Yates, Kraft & Boeva 2025 (*Genome Biol*, full text) |
| Can imputation recover a cell that was never captured? | **No.** Gene-level dropout imputation (a captured cell's undetected transcript) is a distinct, separately studied problem from whole-cell MNAR loss; no imputation method addresses the latter. | Gate 1 §3 |

### Observation / Implication / Decision

- **Observation.** The identifiability problem this program is built around is real and
  already named in the field's own foundational papers (Chronos, GR/DIP metrics), but it
  is **tractable**, not a wall: purpose-built readout combinations exist that separate
  division from death, at the cost of weaker death-rate recovery specifically.
- **Implication.** Neither candidate is blocked by a fundamental measurement
  impossibility. The open item for Candidate A is a **data-generation/resourcing gap**
  (no lab has assembled CRISPR(i) + early/multi-timepoint + lineage barcoding + a
  cancer-dependency line in one dataset), not an identifiability failure. The MNAR/mito-QC
  literature constrains *which* readouts are trustworthy (ruling out mito-fraction-based
  death inference) without preventing the target estimand from being defined through
  readouts that avoid that confounded proxy.
- **Decision.** Both candidates proceed to Gate 2 phenomenon-prevalence testing.
  Candidate A's evidence ceiling is fixed at A2 (sibling/clone proxy) pending a
  purpose-built dataset; per-cell prospective (A1) claims are not licensed by anything
  found at Gate 1.

---

## 3. Gate 2A — Phenomenon Prevalence, Lineage Level

**Verdict:** **INSUFFICIENT EVIDENCE -> bounded validation pilot (NOT a stop).**

### Decision-relevant evidence

| Criterion | Finding | Citation |
|---|---|---|
| COMMON — qualitative existence | **Existence-positive at R2.** Sister-cell fate concordance repeatedly demonstrated: TRAIL sister death-time correlation R²=0.93 (born <7 h before treatment) vs. R²=0.04 for random pairs; HCT116 cisplatin sisters ~80% concordant in fate vs. ~53% expected by chance. | Spencer et al. 2009, *Nature*, DOI `10.1038/nature08012`; "Hidden heterogeneity..." 2018, *Nat Commun*, DOI `10.1038/s41467-018-07788-5` |
| COMMON — the one genuine CRISPR existence point | **The one genuine CRISPR-perturbation existence data point found in this entire review.** Sub-clones carrying the *identical* sgRNA diverge in abundance trajectory at Wilcoxon p<10⁻¹². Silent on whether early state predicts the divergence — no `S_early` variable was measured. | "Tracing cellular heterogeneity in pooled genetic screens via multi-level barcoding," *BMC Genomics* 2019, DOI `10.1186/s12864-019-5480-0` |
| COMMON — the frozen quantitative threshold | **NOT EVALUABLE.** The literal `>= 20%` denominator (net-fitness-matched *perturbation pairs* with an R2-divergence arm) does not exist in the literature — only a non-random set of ~9 published contexts a lab specifically chose to look at. Publication bias makes any literature-derived percentage uninterpretable in either direction. | Gate 2A per-criterion verdict |
| LARGE | **Insufficient.** No paper reports the frozen quantities (founder-referenced fraction lost, or division-rate difference, each with a defined noise bar). The methodologically correct design for founder-referenced accounting — continuous single-cell imaging to a directly observed outcome — exists (Spencer 2009; HCT116-cisplatin 2018) but none of those papers computes the frozen LARGE quantities in that form. | Gate 2A per-criterion verdict |
| CONSEQUENTIAL | **Insufficient.** The "resistance emergence" row is populated (Emert/Rewind, Watermelon, ClonMapper, Fennell/SPLINTR all show early lineage state predicting later resistant/persister outcome, R2 across >=3 systems) but **every populating study is drug/chemotherapy, not CRISPR**, and none benchmarks the improvement against an `F_net`-only baseline as the criterion requires. | Gate 2A CONSEQUENTIAL verdict |

### Observation / Implication / Decision

- **Observation.** The qualitative phenomenon (within-perturbation lineage divergence,
  tied to an inherited early state that decays over roughly 2-4 generations) is about as
  strongly R2-reproduced as literature-only evidence can make it — but almost entirely
  from **drug** perturbation, never assembled at the frozen quantitative thresholds, and
  the one genuine CRISPR data point cannot speak to the state-association half of the
  question at all.
- **Implication.** This is not a null result and must not be reported as one: the gap is
  that the necessary studies (genetic perturbation, matched-net-fitness pairs, a
  founder-referenced denominator) do not exist yet, which is explicitly *not*
  falsification (blueprint §4.3).
- **Decision.** Route to a bounded validation pilot: K562 continuous pedigree imaging or
  a sister-split barcode design, 5-10 CRISPR knockouts matched in pairs at `tau`,
  `0.5*tau`, `2*tau`, power-calculated for >=100 founder lineages per gene. Full pilot
  design and stop rule: [`gate2a/gate2a_lineage_phenomenon.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate2a/gate2a_lineage_phenomenon.md#bounded-validation-pilot-sketch).

---

## 4. Gate 2B — Phenomenon Prevalence, Population Level

**Verdict:** **Insufficient evidence -> bounded validation pilot, NOT a stop.**

### Decision-relevant evidence

| Criterion | Finding | Citation |
|---|---|---|
| The wedge, directly demonstrated | **The existence proof of the wedge.** "These drugs yield similar final numbers through distinct impacts on the cell cycle": lapatinib (G1 arrest, minimal death) and gemcitabine (S-G2 extension, substantial death) both reach ~0.5x control final cell number via opposite decompositions; an endpoint-only Bliss-additivity model **mispredicts** their combination. **Drug, not genetic.** | Gross et al. 2023, *Nat Commun*, DOI `10.1038/s41467-023-39122-z` |
| COMMON — wrong-denominator trap | Mitocheck's ~4% of ~51,766 constructs showing reproducibly-timed, qualitatively distinct dynamics classes is **the wrong denominator for COMMON** (fraction of *all* reagents, not of net-fitness-matched pairs) — flagged explicitly as **precisely the trap this review was built to catch**, and one that `DECISION_MEMO` v1 then committed anyway, one level up, in its own framing. | Neumann et al. 2010 / Pau et al. 2013 (Mitocheck), DOI `10.1038/nature08869` / `10.1186/1471-2105-14-308` |
| CONSEQUENTIAL — closest on-point evidence | **The closest on-point CONSEQUENTIAL evidence found in the whole review.** DFFB-KO, matched on short-term viability/persister-formation, is "severely hindered" in long-term (5-11 week) colony regrowth — reproduced across 3 independent cell-line/drug systems. Adjacent, not on-target: matching variable is short-term viability, not a tau-defined DepMap/Chronos GeneEffect; "state" is a stable genotype, not a measured early single-cell distribution. | Williams et al. 2025, *Nat Cell Biol*, DOI `10.1038/s41556-025-01810-x` |
| Boundary condition, not counter-evidence | At a **matched** intermediate caspase-activating stimulus dose, prospective fate information from caspase kinetics is **near-zero at baseline** (Tjur R² 0.00-0.16, combined score p=0.75); rises to **R²=0.70 only under added phototoxic/proteotoxic stress**. In 5 of 33 tracked sister pairs, the *surviving* sister had *higher* caspase than its dying sibling. This shows the phenomenon is real but **context-gated**, not absent. | Nano, Mondo, Harwood, Balasanyan & Montell 2023, *PNAS*, DOI `10.1073/pnas.2216531120` |

### Observation / Implication / Decision

- **Observation.** The measurement-level premise is already in print, in the field's own
  foundational papers: net population-count metrics (DIP rate, GR metrics) **cannot**, on
  their own, distinguish "no division / no death" from "matched high division and high
  death." When an independent, mechanistic decomposition is applied to a real system, the
  wedge is directly observed and can be large (Gross et al. 2023) — but only for drugs.
- **Implication.** No genome-scale CRISPR/CRISPRi screen was found that independently
  measures population dynamics per gene, matches genes on net fitness, and reports
  prevalence/effect-size/consequence at that matched fitness. This is a genuine design
  gap, not a null result — and the Mitocheck denominator trap is a concrete warning that
  this review's own earlier framing (`DECISION_MEMO` v1) fell into it once already.
- **Decision.** Route to a bounded validation pilot: K562 arrayed CRISPRi knockdown of
  ~10-20 net-fitness-matched pairs, live-cell imaging + death dye or a FUCCI reporter
  (reusing the Gross et al. 2023 phase-resolved decomposition design), plus a
  post-withdrawal regrowth arm borrowing the Williams et al. 2025 logic. Full pilot
  design and stop rule: [`gate2b/gate2b_population_phenomenon.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate2b/gate2b_population_phenomenon.md#bounded-validation-pilot-sketch).

---

## 5. Gate 3 — Nearest Prior Art and Exact Novelty

**Verdict:** **partially-scooped.** The generic phenomenon is already demonstrated; the
**specific conjunction** — transcriptome-wide early state + CRISPR perturbation + a
cancer-dependency line + an independent pre-existing anchor + a full multi-state
trajectory — was found in **no single paper**.

### Decision-relevant evidence

| Question | Finding | Citation |
|---|---|---|
| Nearest work on modality/rigor | Weinreb et al. 2020 (LARRY): transcriptome-wide early state, real evidence-tier A2 clonal design, prospective (T2) task, with a documented quantitative accuracy ceiling of **~50-60%**. Done in unperturbed normal hematopoiesis — no genetic perturbation, no net-fitness anchor at all. | Weinreb et al. 2020, *Science*, DOI `10.1126/science.aaw3381` |
| Nearest work on fitness-independence logic | Iyer et al. 2025: A1-tier, same-lineage, live-imaged, human cancer lines, under a lethal perturbation, explicitly tests and reports pre-existing state predicting survival/death with no accompanying fitness-rate difference. Drug (cisplatin), not genetic; internal matched (Analysis-R) comparison, not an independent pre-existing anchor (Analysis-P). | Iyer, Alva, Granada & Chakrabarti 2025, *PLOS Comput Biol*, DOI `10.1371/journal.pcbi.1013446` |
| T1-presented-as-T2, textbook instances | CellRank's "terminal states" are computed from present-snapshot transition-graph structure, not lineage ground truth, in most applications — a **textbook instance of the T1-vs-T2 conflation the governing spec exists to catch**. Waddington-OT's own growth-rate prior is itself derived from a proliferation/apoptosis marker-**signature score** — a T1-style state readout feeding the method that then claims to reconstruct fate trajectories: a **circularity risk**, not resolved by the paper. | Lange et al. 2022 (CellRank), DOI `10.1038/s41592-021-01346-6`; Schiebinger et al. 2019 (Waddington-OT), DOI `10.1016/j.cell.2019.01.006` |
| The comparison sitting undone | Norman 2019 and Replogle 2022 (K562 Perturb-seq) run the transcriptome measurement and the growth-phenotype measurement in the same experiment, side by side, and **never model one from the other** — growth is used only as a side annotation. Stated as written: *"exactly the comparison this program wants to run, left undone."* | Norman et al. 2019, *Science*, DOI `10.1126/science.aax4438`; Replogle et al. 2022, *Cell*, DOI `10.1016/j.cell.2022.05.013` |
| Forward-model ceiling | GEARS, CPA, and Arc's STATE model all predict the post-perturbation transcriptome and stop there — none predicts a phenotype, fitness, or dependency outcome from it. | Roohani et al. 2024 (GEARS); Lotfollahi et al. 2023 (CPA); Adduri et al. 2025 (State, bioRxiv preprint) |
| Is response burden the whole story? (Gate 3 rerun, Jost 2020 deep read) | **No — burden is real, but state is not reducible to it.** Response magnitude tracks growth within a gene's own titration series (support for a burden axis), but at a **fixed** knockdown level individual cells split into discrete subpopulations: *"The frequency of ISR activation increased with lower ATP5E mRNA levels, but even at the lowest levels some cells did not exhibit ISR activation."* Analysis R only (no independent anchor); `γ` is a net rate, silent on division/death decomposition. | Jost et al. 2020, *Nat Biotechnol*, DOI `10.1038/s41587-019-0387-5`, full text |

### Observation / Implication / Decision

- **Observation.** The generic claim — pre-existing state predicts divergent fate under a
  fitness-reducing perturbation, independent of an aggregate fitness readout — is
  positively and repeatedly demonstrated at A1/A2 tier across bacteria and human cancer
  lines **under drug perturbation**. That phenomenon class should not be presented as an
  open unknown. The rerun additionally closes several narrower "never co-occur" claims
  (Jost 2020 combines transcriptome + CRISPR(i) + K562 + growth phenotype; Replogle 2022
  gives a genome-wide burden-vs-growth correlation of Spearman ρ = -0.51 with 771/9,608
  perturbations dissociating the two entirely; Dixit 2016 shows two same-direction-fitness
  genetic perturbations reaching that fitness through materially different transcriptional
  routes).
- **Implication.** None of the newly closed claims touch the program's actual target
  claim: every fitness/growth readout available anywhere in this literature — Chronos,
  Jost's `γ`, Dixit's fold-change, Replogle's `γ` — is a **single net rate**, and no
  dataset found is more than one or two snapshots deep. A reanalysis, however
  sophisticated, cannot manufacture a division/death/recovery trajectory label the
  underlying assay never captured.
- **Decision.** The defensible novelty surface is narrower and more specific than an
  "ingredients never co-occurred" framing: it is not that the components have never been
  combined (Jost comes close), but that no one has used any near-miss system to ask
  whether transcriptomic state predicts **how** a given net fitness was reached — division
  suppressed, cells lost, or both partially and recovered. Cheap reanalysis extensions
  (residualization audits on Jost's/Dixit's/Replogle's own public data) should run before
  any wet-lab commitment, per the Gate 3 rerun's expected-information-value analysis:
  [`gate3/gate3_rerun_expanded.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate3/gate3_rerun_expanded.md#5-expected-information-value-comparison-retrospective-reanalysis-vs-new-study-0).

---

## 6. Gate 4 — Prospective Incremental Information

**Verdict:** both candidates route to **finding 3 of the required trichotomy: a genuine
methodological opportunity** — not technically infeasible, and not merely a resourcing
problem.

### Decision-relevant evidence

| Finding | Detail | Citation |
|---|---|---|
| (a) No A2-tier design uses a CRISPR-knockout library | **All four** validated sibling/clone-split scaffolds found (ReSisTrace, SIS-seq, CellTag-multi, STRACK/LARRY) are otherwise complete and demonstrated — they differ from what Candidate A needs only in the nature of the perturbing agent (drug, unperturbed differentiation, reprogramming-factor overexpression, or a Cre-activated point mutation — never a CRISPR knockout). Stated as written: **"the most actionable near-term methodological opportunity in this entire review."** | ReSisTrace (Nat Commun 2024, DOI `10.1038/s41467-024-45478-7`); SIS-seq/LARRY; CellTag-multi (Nat Biotechnol 2024, DOI `10.1038/s41587-023-01931-4`); STRACK (Cell Stem Cell 2025, DOI `10.1016/j.stem.2025.01.012`) |
| (b) Candidate B's rank-1 anchor | **"None found despite extensive search."** No design measures an early single-cell state distribution for a genetic perturbation, independently measures population dynamics over `[t0,T]` for the same perturbation, **and** anchors to an independent, pre-existing screen (Analysis P). Stated as written: **"the single most consequential negative finding in this memo."** | Gate 4 §2.5 |
| Net effect on claim ceiling | **No design found in this review can currently support Analysis P as specified.** Every usable existing scaffold lacks the independent pre-existing anchor, lacks molecular-state richness, or lacks the prospective pairing altogether. | Gate 4 §5 |
| Consequence | Absent a purpose-built anchor, **all near-term claims default to Analysis R** (retrospective conditional decomposition), which **must never be reported as prospective prediction.** | Gate 4 §5 |
| Confirmed T1-presented-as-T2 instance | Cells are time-lapse imaged, a deep-learning model **predicts** eventual fate from the imaging trajectory, and the profiled cell is then **destructively sequenced at an early timepoint** — DEGs are computed between transcriptomes of cells whose fate was *predicted* by the classifier, not independently observed. Because the sequenced cell is destroyed, its true future is never observed; the paper is transparent about this in its methods, but the substitution ("DEGs between predicted fates" read as "DEGs that predict fate") is exactly the conflation the governing spec's T1-vs-T2 rule exists to catch. | Okaniwa, Kryukov & Shiroguchi 2025, *Biophysics and Physicobiology*, DOI `10.2142/biophysico.bppb-v22.0022`, full text |

### Observation / Implication / Decision

- **Observation.** Neither negative finding traces to technical infeasibility: Live-seq
  (the A1 existence proof) works at 85-89% post-biopsy viability with only ~12
  differentially expressed genes at 1-4h; GR/DIP-style time-resolved imaging and an
  independent DepMap anchor are each independently mature. Nobody has assembled them
  together for a genetic perturbation.
- **Implication.** This is, per the spec's own framing, close to the best possible
  negative outcome: the reason no incremental-information claim can yet be made is that
  the measurement has not been built, not that the biology forbids it. But it also means
  **no currently available dataset or design licenses a prospective (T2) claim** for
  either candidate today.
- **Decision.** Any near-term empirical work must either (i) build the missing anchor
  (an A2-tier sibling-split design instantiated on a CRISPR-knockout library for
  Candidate A; a condition-level paired prospective anchor combining an early
  transcriptomic snapshot, time-resolved population dynamics, and an independent
  pre-existing `F_net` for Candidate B), or (ii) explicitly label all output as Analysis R
  and scope claims accordingly. Full design table and per-candidate finding:
  [`gate4/gate4_prospective_designs.md`](../ideaspark_run/cell-fate-outcome-dynamics/gate4/gate4_prospective_designs.md#4-the-three-way-finding-per-candidate).

---

## 7. The Surviving Novelty Statement

### Closed — five claims that no longer hold

Each was protecting a weaker version of the program. All five are now answered by the
literature or by our own prior work:

| # | Claim | Closed by |
|---|---|---|
| 1 | "No paper combines graded genetic perturbation strength + Perturb-seq + growth phenotype in K562." | **Jost 2020 does exactly this.** Attribute-stacking is dead as a novelty claim. |
| 2 | "No quantitative burden-vs-fitness correlation exists for K562 genetic perturbation." | **Replogle 2022**: Spearman ρ = −0.51 genome-wide, with **771 of 9,608** perturbations showing a significant transcriptional response but negligible growth effect. |
| 3 | "Distinct transcriptional profiles at matched fitness have never been shown for genetic perturbation." | **Dixit 2016** (CABP7 vs. CIT — both increase fitness via materially different cell-cycle programs). |
| 4 | "Population state *distribution* (variance, not mean) has never been linked to fitness in a genetic system." | **Nadal-Ribelles 2025** — yeast, **325 mutants**. |
| 5 | "Whether the transcriptome beats a burden scalar against an independent fitness anchor is untested." | **exp02 answers it affirmatively** — see [`docs/results/prior-internal-evidence.md`](results/prior-internal-evidence.md). |

### What survives

> Every fitness readout in this entire literature — Jost's `γ`, Dixit's sgRNA fold-change,
> Replogle's `γ`, Nadal-Ribelles's competition fitness, DepMap's Chronos GeneEffect — **is a
> single net rate.** Every dataset is one or two snapshots of a **survivor** population.
>
> **No one has asked whether transcriptomic state predicts *how* a given net fitness was
> reached** — division suppressed, cells lost, or both partially and recovered — **because
> no existing dataset, public or internal, captures that decomposition at all.**

### Where the program's centre of gravity now sits

The **specificity** half (does state beat a burden scalar?) is **substantially answered —
affirmatively, modestly** — by four independent sources: exp02, Jost 2020, Dixit 2016, and
Replogle 2022. It is no longer the open question.

The **trajectory-decomposition** half is untouched by every paper and every reanalysis
found. That is where the program lives.

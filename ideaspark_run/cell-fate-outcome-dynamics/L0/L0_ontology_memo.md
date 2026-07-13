# L0 — Ontology and Latent-to-Observable Maps

Stage: L0 (Ontology, Latent-to-Observable Maps), Outcome-Dynamics-Behind-Net-Fitness
literature-review funnel. Governing spec: `docs/superpowers/specs/2026-07-13-cell-fate-
prediction-research-direction-design.md` §3, §4, §8. Shared brief: `ideaspark_run/cell-
fate-outcome-dynamics/SHARED_BRIEF.md`.

**Scope discipline.** Per spec §8, this memo does **not** select between Candidate A
(lineage/clone) and Candidate B (population). It defines terms with citations and builds
parallel latent-to-observable maps. All citations below were retrieved via WebSearch /
WebFetch / Crossref / Europe PMC tool calls in this run; each carries a DOI/PMID/PMCID and
the URL actually fetched. Reference numbers `[R#]` point to the consolidated list at the
end. Where I read only an abstract or a secondary summary (not the full text), I mark the
record `abstract-only`.

---

## 1. Cited Definitional Ontology

### 1.1 State

**Operative definition (used in the primary literature):** a cell's molecular
configuration at a given point in time — operationally, a position (or distribution over
positions) in a high-dimensional expression/feature space, typically identified by
unsupervised clustering or embedding rather than by a privileged causal variable
(Trapnell 2015, *Genome Res* [R1], `abstract+summary`). Weinreb et al. 2018 formalize
state as a point in a continuous state space with an associated cell **density** and
**flux** (rates of entry/exit); the density and the local vector field jointly, not the
density alone, determine dynamics (Weinreb et al. 2018, *PNAS* [R2], `abstract+summary`).

State is descriptive, not predictive by definition — nothing in the term "state" implies
it forecasts fate. That gap is exactly T1 vs T2 (SHARED_BRIEF §3.5) and is the reason this
review exists.

### 1.2 Fate

**Operative definition:** the terminal or future identity/outcome that a cell (or its
lineage) actually realizes, established retrospectively by pairing an earlier
observation with a later one — via lineage barcode, imaging, or direct non-destructive
re-measurement (Wagner & Klein 2020, *Nat Rev Genet* [R3], `abstract+summary`; Weinreb et
al. 2020, *Science*, LARRY [R4], `abstract+summary`). Fate is a property of a realized
trajectory, not of a snapshot. Wagner & Klein 2020 [R3] explicitly warn that
transcriptomically inferred "branching" need not coincide with actual cell-division
events or form a strict tree — i.e., inferring fate topology from state similarity alone
is itself an assumption, not a direct observation.

### 1.3 Fate commitment ("point of no return") — genuinely contested, and central to "early"

This is the load-bearing term for the spec's requirement that "early" be defined relative
to commitment, not to sample collection (§3.2, §4.4).

**Two distinct literatures give two different answers, and they disagree with each
other even within cell-cycle biology:**

- **Classical restriction-point view.** The restriction point (R) is the point in G1
  beyond which a cell no longer needs mitogen signaling to complete division — historically
  treated as *the* point of no return for the proliferation decision (Pennycook & Barr
  2020, *FEBS Lett*, review [R5], `abstract+summary`).
- **Single-cell imaging view directly contradicts a single fixed point.** Using a live
  CDK2-activity sensor across six cell types (non-cancer MCF10A, RPE-hTERT; cancer MCF7,
  U2OS, HCT116; primary lung fibroblasts), Spencer et al. 2013 showed that at mitotic
  exit, genetically identical sister cells **bifurcate**: a fraction immediately
  re-commit to the next cycle (rising CDK2 activity, hyperphosphorylated Rb) while the
  remainder enter an "uncommitted," temporarily quiescent state whose eventual commitment
  depends on mitogen signaling integrated over a **restriction window spanning the
  previous cell cycle**, not a single moment (Spencer et al. 2013, *Cell* [R6],
  `abstract+summary`, DOI `10.1016/j.cell.2013.08.062`, PMID 24075009). Pennycook & Barr's
  2020 review explicitly reframes the restriction point as a **probabilistic, heterogeneous
  transition**, not a fixed point in time relative to any external clock, including sample
  collection [R5].

  **Practical consequence for this program:** commitment to the next division can begin
  *within minutes of the prior mitosis* for some cells and only resolve *hours later,
  contingent on continued signaling*, for others in the same clonal population. Any
  dataset with a single late timepoint (the live constraint flagged in spec §4.4 for
  Replogle Perturb-seq) cannot distinguish "already committed before capture" from
  "not yet committed" without an explicit model of this window — which is a candidate
  falsifiable operationalization, not yet an existing off-the-shelf one for CRISPR
  perturbation contexts.

- **Death has a better-characterized, but still explicitly unsettled, point of no
  return.** For the intrinsic apoptosis pathway specifically, mitochondrial outer membrane
  permeabilization (MOMP) is treated as *the* commitment step: "numerous pro-apoptotic
  signal-transducing molecules and pathological stimuli converge on mitochondria to induce
  MOMP... MOMP is lethal" (Green & Kroemer 2004, *Science* [R7], `abstract+summary`, DOI
  `10.1126/science.1099320`, PMID 15286356). This is directly measurable by single-cell
  reporters: Spencer et al. 2009 used live imaging of caspase-activation and MOMP reporters
  to show that **naturally occurring cell-to-cell differences in protein levels — not
  genetics — set the timing and probability of TRAIL-induced death**, i.e., commitment is a
  stochastic, quantifiable threshold-crossing event at single-cell resolution (Spencer et
  al. 2009, *Nature* [R8], `abstract+summary`, DOI `10.1038/nature08012`, PMID 19363473).

  However, the field's own nomenclature body does **not** treat this as fully settled in
  general: the 2018 NCCD recommendations state that blocking post-mitochondrial caspase
  activation "generally delays (but does not prevent) intrinsic apoptosis" once "a
  **hitherto poorly defined point-of-no-return** has been trespassed" (Galluzzi et al.
  2018, *Cell Death Differ* [R9], **full text fetched**, quote verified, DOI
  `10.1038/s41418-017-0012-4`, PMID 29362479). That phrase — "hitherto poorly defined" —
  is the NCCD's own admission that the operational boundary of commitment, even for the
  best-characterized death subroutine, is not a crisp, universally agreed molecular event.

**Bottom line for L0:** a citable, measurable point-of-no-return notion exists and is
operationalizable for the death subroutine (MOMP, reporter-measurable) but is explicitly
described by its own nomenclature committee as poorly defined at the general level; for
the proliferation/quiescence decision, single-cell evidence has replaced a fixed
"restriction point" with a **probabilistic commitment window** that can start well before
or after any single sampled timepoint. Neither literature offers a ready-made,
off-the-shelf "committed by hour X" cutoff for a genetic-perturbation, multi-day Perturb-
seq context — this is squarely a Gate 1 "not yet known" finding, not an assumption this
memo may make.

### 1.4 Death — event vs process; regulated cell death subroutines

The NCCD 2018 recommendations (full text fetched) explicitly separate **death as
process** from **death as event**:

> "Irreversible degeneration of vital cellular functions (notably ATP production and
> preservation of redox homeostasis) culminating in the loss of cellular integrity
> (permanent plasma membrane permeabilization or cellular fragmentation)." (Galluzzi et
> al. 2018 [R9])

Death is thus modeled as a process (degeneration) with a terminal event (permeabilization
/ fragmentation). The NCCD further separates:

- **Accidental cell death (ACD):** "virtually instantaneous and uncontrollable... physical
  disassembly of the plasma membrane caused by extreme physical, chemical, or mechanical
  cues" [R9].
- **Regulated cell death (RCD):** "results from the activation of one or more signal
  transduction modules, and hence can be pharmacologically or genetically modulated" [R9].

**"Regulated cell death subroutines"** is the NCCD's own term for the molecularly defined
RCD pathways it classifies — "signal transduction modules involved in the initiation,
execution, and propagation of cell death," organized into "an updated classification of
cell death modalities" [R9]: intrinsic and extrinsic apoptosis, MPT-driven necrosis,
necroptosis, ferroptosis, pyroptosis, parthanatos, entotic, NETotic, lysosome-dependent,
autophagy-dependent, and immunogenic cell death, plus mitotic catastrophe (secondary
summary confirms this list; per spec §7, mechanism-level death typing is explicitly
**out of scope** for this program's current stage — recorded here only to fix the term).

The **2023 NCCD apoptosis-specific update** (Vitale, Pietrocola, Guilbaud, et al. 2023,
*Cell Death Differ* 30(5):1097-1154 [R10], `abstract+summary`, DOI
`10.1038/s41418-023-01153-w`) is the current apoptosis-specific consensus refinement; I
was not able to fetch its full text (Nature paywall redirect loop) so I cite it only for
its existence and scope (apoptosis-in-disease mechanisms), not for any specific
definitional sentence beyond what is in the 2018 general nomenclature document.

Critically, per the NCCD: **"Cellular senescence does not constitute a form of RCD"**
[R9] — senescence and death are explicitly disjoint categories in the field's own
nomenclature, which matters directly for not conflating "arrested" with "dying" (spec
§3.1's retirement of that mixed category, see §4 below).

### 1.5 Biological loss / lineage extinction vs. assay attrition — never conflate

**Biological loss / lineage extinction:** the lineage genuinely ends — no descendants
remain alive. In lineage-tracing practice this is typically *inferred*, not directly
observed, from the absence of any read carrying a given barcode at the endpoint (Yang et
al. 2022, *Cell*, CRISPR lineage tracing in a KP lung-adenocarcinoma mouse model [R11],
`abstract+summary`, DOI `10.1016/j.cell.2022.04.015`). This inference is confounded with
technical failure to detect a rare surviving clade — the same identifiability problem
appears explicitly in birth-death-based statistical treatments of CRISPR lineage-tracing
data (see §3 below).

**Assay attrition:** the unit is alive-or-was-alive but is not present in the analyzed
dataset — died before collection but was never a lineage-extinction event at the time of
measurement, was lost during dissociation, failed library capture, or was removed by QC.
This is a **measurement-process** phenomenon with its own, separately documented
literature:

- Warm collagenase/protease dissociation induces a conserved heat-shock/immediate-early
  stress-gene signature and **differentially depletes fragile cell types** — "podocytes
  being the extreme example of a cell type practically lost in warm-dissociated
  libraries," with "a subpopulation representing transcriptomically dying cells" showing
  elevated MHC-class-I expression (van den Brink et al. 2017, *Nat Methods* [R12],
  `abstract+summary`, DOI `10.1038/nmeth.4437`, PMID 28960196).
- Systematic comparison of dissociation and storage protocols shows cryopreservation
  causes "a major loss of epithelial cell types," while methanol fixation avoids that loss
  but introduces ambient-RNA leakage (Denisenko et al. 2020, *Genome Biol* [R13],
  `abstract+summary`, DOI `10.1186/s13059-020-02048-6`).
- The standard QC pipeline used across the field explicitly treats percent-mitochondrial
  reads as a marker of dissociation-induced stress: "cells that become stressed during
  tissue dissociation may express abnormally large proportions of mitochondrial genes in
  their transcriptome" (Hong et al. 2022, SCTK-QC, *Nat Commun* [R14], **full text
  fetched**, quote verified, DOI `10.1038/s41467-022-29212-9`). Note: the introductory
  framing in this paper repeats the standard heuristic but the paper itself is a QC-tooling
  paper, not a study establishing the causal claim; it does not discuss context-dependence
  of thresholds.
- **This heuristic does not generalize safely to cancer cell lines**, which is directly
  relevant to a K562 context: across nine cancer scRNA-seq datasets (441,445 cells, 134
  patients), malignant cells had significantly higher mitochondrial-read percentage than
  non-malignant cells in 72% of samples, and — critically — dissociation-stress scores
  showed "no significant difference" or even lower stress in high-mito malignant cells in
  most studies tested (max point-biserial correlation < 0.3), while high-mito malignant
  cells showed genuine metabolic dysregulation (xenobiotic metabolism, drug-resistance
  associations) rather than dying-cell signatures (Yates, Kraft & Boeva 2025, *Genome
  Biol* [R15], **full text fetched**, quote verified, DOI `10.1186/s13059-025-03559-w`,
  PMID 40205439). This is direct evidence supporting the spec §6 claim boundary — high
  mito/low UMI is not a definitional dying-cell signature — and additionally shows the
  *opposite* failure mode is live: filtering by a fixed mito threshold in a cancer line can
  **discard genuinely viable, biologically distinct cells**, which is its own form of
  assay attrition, self-inflicted by QC.

**Disagreement to preserve, not resolve:** the standard QC literature ([R12], [R13],
[R14]) treats high mitochondrial content as evidence *for* dissociation-induced
attrition/stress; the cancer-focused reanalysis ([R15]) shows this association is weak
or absent in malignant cells specifically and argues the standard heuristic causes
unwarranted exclusion of viable cells. Both are evidenced positions in the current
literature; which applies to K562 specifically is an open, dataset-specific empirical
question, not resolved by either paper alone.

### 1.6 Quiescence

**Operative definition:** reversible cell-cycle arrest (G0), distinguished from
senescence by reversibility. Coller, Sang & Roberts 2006 showed that three independent
quiescence-inducing signals (mitogen withdrawal, contact inhibition, loss of adhesion)
each activate a distinct transcriptional program in fibroblasts, arguing quiescence is
not one state but a family of molecularly distinct, signal-dependent reversible states —
"A New Description of Cellular Quiescence" (Coller, Sang & Roberts 2006, *PLoS Biol*
[R16], `abstract+summary`, DOI `10.1371/journal.pbio.0040083`, PMID 16509772).

### 1.7 Senescence — genuinely contested boundary with quiescence

**Classical definition:** "irreversible loss of proliferative potential associated with
specific morphological and biochemical features, including the senescence-associated
secretory phenotype (SASP)" — and explicitly **not** a form of regulated cell death
(Galluzzi et al. 2018 [R9], full text fetched, quote verified). Van Deursen 2014 frames
senescence similarly as historically "an irreversible cell-cycle arrest mechanism" that
protects against cancer, while noting newer work extends it to a "dynamic series of
cellular states" in development, repair, and aging (van Deursen 2014, *Nature* [R17],
`abstract+summary`, DOI `10.1038/nature13193`).

**But the field's own leaders flag the definition as operationally broken.** Sharpless &
Sherr 2015 state directly that senescence lacks "a uniform definition," and that
biomarker application to identify/enumerate senescent cells in vivo is "inconsistent"
(Sharpless & Sherr 2015, *Nat Rev Cancer* [R18], `abstract+summary`, DOI
`10.1038/nrc3960`, PMID 26105537).

**Recent single-cell evidence goes further and directly undercuts snapshot-based
separability from quiescence:** Ashraf, Fernandez & Spencer 2023 show that the canonical
senescence biomarkers (SA-β-gal, LAMP1, IL8, 53BP1, p21, Lamin B1, cell size) are
**graded, not binary**, and track the **duration of cell-cycle withdrawal** rather than a
qualitatively distinct senescent identity — "quiescent and apparent senescent cells are
nearly molecularly indistinguishable from each other at a snapshot in time" (Ashraf,
Fernandez & Spencer 2023, *Nat Commun* [R19], `abstract+summary`, DOI
`10.1038/s41467-023-40132-0`). A 2025 follow-up from the same lab, using scRNA-seq after
chemotherapy, finds a **quiescence-senescence continuum** with distinct "senotypes"
reached via a gradual (mitosis-to-G0) path or a direct mitotic-slippage path, and reports
that "senescent phenotypes begin to manifest early and gradually... even in shallow
quiescent cells" (Fernandez, Passanisi, Ashraf & Spencer 2025, *Nat Commun* [R20],
`abstract+summary`, DOI `10.1038/s41467-025-66836-z`).

**This is a genuine, unresolved disagreement, stated explicitly, not adjudicated here:**
the classical view treats quiescence and senescence as distinct states separable by
markers and reversibility; the most recent single-cell evidence treats them as points on
a **continuum indexed by arrest duration**, with markers reflecting time-since-withdrawal
rather than a discrete cell-fate category — which is a direct threat to any pipeline that
labels cells "arrested" vs. "senescent" from a single-timepoint snapshot.

### 1.8 Recovery

**Operative definition:** a transition from a reversible non-dividing or drug-tolerant
state back to active division/drug-sensitive proliferation upon removal of the
inhibiting stimulus. Operationalized in two literatures:

- Cell-cycle: the "uncommitted" post-mitotic population identified by Spencer et al. 2013
  [R6] recovers by later crossing the restriction point once mitogen signaling resumes/
  accumulates — recovery here is a resumed-division transition, not a separate cell type.
- Drug-tolerant persistence: "removal of drug allows regrowth of cells which become
  resensitized to drug treatment" is the defining reversibility criterion distinguishing a
  persister population from a genetically resistant one (Sharma et al. 2010 [R21],
  `abstract+summary`, DOI `10.1016/j.cell.2010.02.027`, PMID 20371346). The 2024 review
  states this most explicitly as a fork in the road: persister cells "can either produce
  drug-sensitive progeny [recovery] or evolve towards irreversible, acquired resistance
  mediated by acquired genomic alterations [no recovery]" (Russo et al. 2024, *Nat Rev
  Cancer* [R22], `abstract+summary`, DOI `10.1038/s41568-024-00737-z`).

### 1.9 Persister

**Operative definition, transplanted from bacteria to cancer with the same operational
criteria:** a small, non-genetically-distinct subpopulation that survives a lethal/
growth-inhibitory challenge via a reversible, non-mutational mechanism, and (unlike
resistant mutants) reverts to full sensitivity once regrown without the challenge —
originally defined for *E. coli* persistence to antibiotics via a phenotypic growth-rate
switch (Balaban et al. 2004, *Science* [R23], `abstract+summary`, DOI
`10.1126/science.1099390`, PMID 15308767), then identified in cancer as a rare (<5%),
reversibly drug-tolerant, chromatin-state-altered subpopulation (Sharma et al. 2010 [R21])
that is selectively dependent on GPX4 to survive oxidative/ferroptotic stress (Hangauer et
al. 2017, *Nature* [R24], `abstract+summary`, DOI `10.1038/nature24297`). The 2024 review
[R22] frames persistence as a **plastic, transitional cellular state** rather than a fixed
subpopulation identity — cells move between DTP phenotypes within a tumor — which is a
further complication for any static "persister" label.

### 1.10 Time horizon T and denominator

Defined structurally by spec §3.2, not by a single literature term; the relevant
supporting literature (what "denominator" must actually mean and how to estimate it
correctly) is developed in full in §3 below, because this is where the field's technical
machinery — not a single citable definition — does the work.

---

## 2. Parallel Latent-to-Observable Maps

Per spec §8/L0, both maps are presented with equal weight. **No ranking or selection is
implied by presentation order.**

### 2.1 Candidate A — Lineage / clone unit

**Latent estimand (spec §3.3):** the lineage's own stochastic trajectory over
`[t0, t0+T]` — division history, alive-but-non-dividing-through-`T`, the recovery
transition (arrest → resumed division), lineage extinction, and descendant abundance at
`T`. This is a marked, multi-type branching/point process per founder lineage.

| Tier | What is physically observable | The map latent → observable, and its assumption | Where it breaks |
|---|---|---|---|
| **A1** same-cell prospective | Live-seq: a cell's own baseline transcriptome, extracted via fluidic-force-microscopy cytoplasmic biopsy without killing the cell, is "preregistered," then the *same* cell is time-lapse imaged for its own downstream phenotype — demonstrated on RAW264.7 macrophage LPS response and ASPC/IBA adipogenic differentiation (Chen et al. 2022, *Nature* 608:733-740 [R25], **full text fetched**, DOI `10.1038/s41586-022-05046-9`, PMID 35978187, PMC9402441) | Assumes the biopsy itself does not materially perturb the cell's future trajectory: post-extraction viability was 85-89% and only 12 genes were significantly differentially expressed relative to unbiopsied cells, in the demonstrated systems [R25]. Throughput is ~4-5 extractions/hour due to per-cell downstream processing and imaging follow-up [R25] | Never demonstrated on CRISPR/genetic perturbation in a cancer line [R25]; whether the biopsy perturbation and viability/throughput figures transfer to a multi-day division-tracking assay in a proliferating cancer line (as opposed to a largely non-dividing macrophage/adipocyte assay) is unverified; throughput as stated is far below what a systematic perturbation panel would need |
| **A2** sibling/clone proxy | One clone member is destructively sequenced early; barcode-linked siblings' later division/extinction fate is observed — exemplified by LARRY barcoding coupling early transcriptional state to later clone-fate distribution in hematopoietic differentiation (Weinreb et al. 2020, *Science* [R4], `abstract+summary`, DOI `10.1126/science.aaw3381`, PMID 31974159); the statistical logic that licenses inferring transition dynamics from siblings without ever observing the same cell twice is "kin correlation analysis" (Hormoz et al. 2016, *Cell Syst* [R26], `abstract+summary`, DOI `10.1016/j.cels.2016.10.015`, PMID 27883889) | Assumes clonal coherence: recently divided siblings are correlated enough in state/behavior to stand in for "the same" early condition — Hormoz et al. explicitly model this correlation and its decay with lineage distance rather than assuming perfect identity [R26] | Coherence decays as siblings diverge in state after division (differentiation, stochastic gene expression); destructive sequencing of the "proxy" member means the exact individual whose future is later observed was never itself measured — ceiling is clone-level association, not per-cell fate (per spec §4.1) |
| **A3** clone-average | An evolving CRISPR lineage-tracing barcode with scRNA-seq readout, reconstructing a phylogeny of surviving descendants and their terminal transcriptomic states, applied in a KP-mutant mouse lung-adenocarcinoma model (Yang et al. 2022, *Cell* 185:1905-1923 [R11], `abstract+summary`, DOI `10.1016/j.cell.2022.04.015`) | Assumes phylogenetic reconstruction from the barcode is accurate and that surviving clade transcriptomic composition at the single endpoint reflects the clade's history | Barcode character loss/homoplasy biases inferred tree topology and rates (general finding in CRISPR-lineage-recording inference-accuracy literature, e.g., bioRxiv assessment of inference from CRISPR lineage recordings, `abstract-only`, not independently confirmed with DOI in this pass — see UNVERIFIED); **extinguished lineages leave zero reads and are structurally invisible** — this is exactly the denominator problem (§3) |

**Where the whole Candidate-A map is threatened regardless of tier:** destructive
sequencing plus lineage barcoding, even at its best (A2), does not give same-founder
longitudinal observation (spec §4.1) — only Live-seq-class methods do, and those are
unproven at CRISPR-perturbation, cancer-line, multi-day scale.

### 2.2 Candidate B — Population unit

**Latent estimand (spec §3.3):** under matched, independently measured net fitness, the
population-level multi-state transition process (division/quiescence/recovery/death
rates and their time-evolution), not a categorical mixture.

| Approach | What is physically observable | The map latent → observable, and its assumption | Where it breaks |
|---|---|---|---|
| Chronos (pooled CRISPR fitness model) | Time-series sgRNA read-count proportions from a pooled screen (Dempster et al. 2021, *Genome Biol* 22:343 [R27], **full text fetched**, DOI `10.1186/s13059-021-02540-7`, PMID 34930405, PMC8686573) | Fits an exponential-growth mixture (knockout cells at rate `R*_cg`, unperturbed cells at rate `R_c`), assumes binary knockout efficacy and negative-binomial read-count noise; outputs relative growth-rate effect `r_cg` | **Explicitly, by the authors' own statement, cannot decompose reduced division from increased loss**, and its performance is stated to degrade "in the event that most cells are dying" [R27] — i.e., it is a *net-fitness* readout, not a dynamics decomposition, which is exactly the premise this whole program is testing |
| GR/DIP metrics | Time-course cell counts plus a known/estimated initial count under a drug or perturbation (Hafner et al. 2016, *Nat Methods* [R28], **full text fetched**, DOI `10.1038/nmeth.3853`, PMID 27135972, PMC4887336) | Assumes exponential growth in treated and untreated arms; `GR=0` signals complete cytostasis, `GR<0` signals net cytotoxicity [R28] | Decomposes only the **population-average net rate** into cytostatic-vs-cytotoxic categories; cannot attribute the loss to any single-cell early state, and if some cells stop dividing while a distinct subpopulation dies, GR/DIP sees only the pooled net effect, not the mixture |
| Waddington-OT (unbalanced optimal transport across serial snapshots) | Serial scRNA-seq snapshots plus a per-cell growth-rate prior derived from proliferation/apoptosis marker-gene signature scores (Schiebinger et al. 2019, *Cell* 176:928-943 [R29], `abstract+summary`, DOI `10.1016/j.cell.2019.01.006`, PMID 30712874) | Assumes the growth-rate prior is accurate and that entropy-regularized transport plans approximate true ancestor-descendant relationships absent lineage-tracing ground truth | The growth-rate prior is itself inferred from a marker-gene classifier — a T1-style state readout — so using WOT output to test "does early state predict future loss" risks **circularity** if the same or correlated markers feed both the prior and the claimed discovery; OT-inferred ancestry is not independently validated against ground-truth lineage in general |
| Condition-level paired prospective anchor (Rank 1 per spec §8.1 for Candidate B) | Would require an early single-cell state-distribution measurement at `t0` plus an **independently measured** future population-dynamics readout over `[t0, T]` (e.g., real-time division/death imaging, or repeated pooled counts, for the *same* condition) | Not located as an existing, already-executed design for a genetic-perturbation cancer-line context in this search pass | This is precisely the Gate 1/Gate 4 open question the spec flags (§4.4, §8) — **absence of a located example is reported here as insufficient evidence, not falsification** (spec §4.3) |

**Where the whole Candidate-B map is threatened regardless of method:** every population
method above either (a) explicitly cannot decompose division from loss (Chronos), (b)
decomposes only the pooled average, not the underlying mixture (GR/DIP), or (c) needs an
externally supplied growth/death prior that is itself a state-based inference and
therefore threatens circularity (Waddington-OT). None of the three located methods gives
a **loss-free, independent** anchor for "future population dynamics" at the resolution
Candidate B's own evidence hierarchy (spec §8.1, rank 1) requires.

---

## 3. The Denominator Problem, Made Concrete

Spec §3.2: "Observed units at `T` are survivors of a branching-plus-loss process;
fraction-of-observed is not fraction-of-founders." This section documents which existing
statistical treatments handle this correctly, and what each assumes.

**3.1 The generic identifiability result.** Weinreb, Wolock, Tusi, Socolovsky & Klein 2018
prove that for a population evolving on a continuous state space, "symmetries and
inhomogeneities of the population balance law set fundamental limits on dynamic
inference": for any single observed cross-sectional distribution of cell states, **there
exist multiple distinct underlying dynamics** (including different entry/exit — i.e.,
birth/loss — rates across the space) that produce the identical snapshot. Accurate
inference additionally requires "the density of cells in high-dimensional state space, as
well as the rates of cell entry and exit across the density" (Weinreb et al. 2018, *PNAS*
115:E2467-E2476 [R2], `abstract+summary`, DOI `10.1073/pnas.1714723115`, PMID 29463712).
**This is the formal statement of why "fraction observed at T" cannot, by construction,
be read off a single endpoint snapshot as "fraction of founders" — the loss/entry rates
that would let you correct for it are exactly the unobserved quantities.**

**3.2 The competing-risks statistical machinery that handles it correctly.** The general
statistical framework for outcomes where a unit can be lost to one of several mutually
exclusive events, with some units never reaching any event before the observation window
ends (censoring), is **competing risks and multi-state modeling**: cumulative incidence
functions and cause-specific hazards, rather than naive fraction-of-observed proportions
or a Kaplan-Meier estimator that wrongly assumes the competing events are independent
(Putter, Fiocco & Geskus 2007, *Stat Med* [R30], `abstract+summary`, DOI
`10.1002/sim.2712`, PMID 17031868).

Applied directly to single-cell pedigree data, Cornwell, Hallett, Auf der Mauer, et al.
2016 build exactly this machinery for time-lapse cell-lineage tracking: they explicitly
note that "censored lifetimes are often discarded, although [they] do contain
information on whether a cell's fate was realised before it was censored," and that
naive Kaplan-Meier analysis "overestimates" outcome probabilities because it "makes the
erroneous assumption that... division and death are independent." Their competing-risks
regression (CRR) approach "accurately estimates the cumulative incidence of the observed
competing fates... with no dependence on the correlation coefficient [between division and
death]" — i.e., it is the one located method in this pass that **does not assume**
independence between the competing fates it is trying to disentangle (Cornwell et al.
2016, *Sci Rep* 6:27100 [R31], **full text fetched**, quote verified, DOI
`10.1038/srep27100`, PMID 27250534, PMC4890426). This is applied in that paper to
p53-genotype-dependent breast-cancer-cell fate after chemotherapy and to hematopoietic
progenitor pedigrees — i.e., it is a validated general estimator, not a bespoke one-off.

**3.3 Lineage-tree-based inference without repeated same-cell observation.** Hormoz,
Singer, Linton, Antebi, Shraiman & Elowitz 2016 formalize "kin correlation analysis":
because sister cells begin from the identical state at division, the *statistics* of
their later divergence across many independent lineage trees can be used to infer
transition rates between cell states **without ever observing the same cell twice**
(Hormoz et al. 2016, *Cell Syst* [R26], `abstract+summary`, DOI
`10.1016/j.cels.2016.10.015`, PMID 27883889). This is the direct statistical justification
for the Candidate-A2 tier, and its assumption — decaying but nonzero correlation between
kin as a function of lineage distance, estimated from the tree structure itself — is
exactly the assumption that breaks down if death/loss is state-dependent in a way not
shared by surviving kin.

**3.4 Applying birth-death models directly to genomic lineage-tracing barcodes.**
CRISPR/Cas9-based lineage-recording systems are now analyzed with explicit birth-death
process models that jointly infer time-scaled phylogenies, division (birth) rates, and
sampling intensity — treating the *sampling/observation* process as a parameter to be
estimated, not ignored (general finding surfaced via bioRxiv preprints on Bayesian
phylodynamic inference for CRISPR/Cas9 lineage-tracing barcode data and on assessing
inference of single-cell phylogenies and population dynamics from CRISPR lineage
recordings; `abstract-only`, titles and general claims located via search but full
text/DOI not independently confirmed in this pass — listed in UNVERIFIED). The
recurring, load-bearing point surfaced across this literature is that **filtering/sampling
intensity systematically biases inferred division rates** (stated in search-summary form
as "slight systematic overestimation") — i.e., even birth-death-aware methods must model
the observation process explicitly or their rate estimates are biased, which is the same
warning as Weinreb et al.'s formal result (§3.1) applied to genomic barcodes instead of
transcriptomic snapshots.

**3.5 The applied tumor-evolution case.** Yang et al. 2022 combine an evolving lineage-
tracing barcode with scRNA-seq readout in a KP-mutant lung-adenocarcinoma mouse model to
track single transformed cells through metastasis, explicitly modeling transitions in
transcriptional state and plasticity along reconstructed lineages (Yang et al. 2022,
*Cell* [R11], `abstract+summary`, DOI `10.1016/j.cell.2022.04.015`). This is a concrete
demonstration that clade-level survivorship (not founder-level) is what a phylogeny built
from surviving sequence reads can show — extinguished branches are, by construction,
absent from the tree, so the same denominator problem recurs at the clade level even in a
technically sophisticated lineage-tracing system.

**Summary of correct treatments and their assumptions:**

| Estimator | What it assumes | What it buys you |
|---|---|---|
| Population balance / dynamic-inference limits (Weinreb 2018 [R2]) | None — this is an impossibility result | Tells you what information (density + entry/exit rates) is *necessary*, ruling out naive snapshot-only inference |
| Competing-risks CIF / cause-specific hazards (Putter 2007 [R30]; Cornwell 2016 [R31]) | Right-censoring is at random given covariates (not necessarily independence between competing fates) | A denominator that correctly separates division, death, and censoring without assuming they're independent, at true single-cell/lineage resolution with time-lapse data |
| Kin correlation analysis (Hormoz 2016 [R26]) | Sibling-state correlation decays with lineage distance in a modelable way | Transition-rate inference from lineage trees + one endpoint measurement, without needing to reobserve the same cell |
| Birth-death phylodynamic models on CRISPR barcodes (surfaced, `abstract-only`, see UNVERIFIED) | Sampling/observation intensity must be jointly estimated, not assumed complete | A denominator-aware division-rate estimate from barcode data, at the cost of rate-estimate bias if the observation model is mis-specified |
| Chronos / GR-DIP (population-only, not denominator-correcting) [R27], [R28] | Exponential growth; binary knockout efficacy or known initial counts | A **net** growth-rate or cytostasis/cytotoxicity flag — explicitly *not* a corrected denominator, included here as the contrast case |

**Where naive single-cell analysis silently goes wrong (the failure mode this section
exists to name):** treating "fraction of cells observed in cluster X at endpoint T" as an
estimate of "fraction of founders whose lineage ended up in fate X" silently assumes (a)
no differential loss before capture correlated with state, and (b) no extinguished
lineages that would have contributed differently. Neither assumption is licensed by a
single endpoint snapshot; both require either an explicit birth-death/competing-risks
model with an estimated observation process, or a genuinely longitudinal/lineage-linked
design.

---

## 4. Retired-Vocabulary Audit

Spec §3.1 retires `proliferating / arrested / dying / mechanism / timing` as a category
set and retires "outcome composition." Per candidate term, what the literature actually
uses instead, and the citation for it:

| Retired term | Why it was invalid as a category | What the literature uses instead |
|---|---|---|
| **proliferating** | A rate, not a category — cells are not binary proliferating/not | Explicit growth-rate estimation: Chronos's `r_cg` (Dempster et al. 2021 [R27]) and GR/DIP's `GR(c)` (Hafner et al. 2016 [R28]) both replace the label with a continuous rate; at single-cell resolution, division becomes one of several competing hazards in a competing-risks model (Cornwell et al. 2016 [R31]) |
| **arrested** | A window-defined, possibly reversible state, not a fixed label — duration and reversibility are part of its meaning | Continuous, duration-indexed models: Ashraf, Fernandez & Spencer 2023 show canonical arrest/senescence markers scale with **duration of cell-cycle withdrawal** rather than marking a discrete category [R19]; Spencer et al. 2013's restriction-window model treats "arrested" as a probabilistic, time-dependent commitment state, not a snapshot label [R6] |
| **dying** (process) vs. **death** (event) | Conflating an ongoing process with its terminal event erases exactly the process/event distinction the field itself maintains | NCCD's own explicit process (degeneration) → event (permeabilization/fragmentation) structure (Galluzzi et al. 2018 [R9]); statistically, a cause-specific hazard / cumulative incidence function models death as a time-to-event outcome with its own hazard, not a static state (Putter et al. 2007 [R30]; Cornwell et al. 2016 [R31]) |
| **mechanism** (death mechanism) | Mechanisms overlap and are not mutually exclusive labels for an outcome category | NCCD's classified regulated-cell-death subroutines (apoptosis, necroptosis, ferroptosis, pyroptosis, etc. — Galluzzi et al. 2018 [R9]) are a separate, molecularly defined axis, explicitly out of scope for this program's current stage per spec §7 |
| **timing** | Timing is part of the fate definition itself (spec: "timing is part of the fate definition, not an extra label") | Multi-state/competing-risks models make transition **time** a first-class part of the estimand (time-to-division, time-to-death, cumulative incidence *by* time `t` — Putter et al. 2007 [R30]); Spencer et al. 2013's restriction window makes timing-of-commitment itself the object of study, not an auxiliary variable [R6] |
| **"outcome composition"** (retired as not a valid simplex) | A lineage can simultaneously produce descendants that keep dividing and descendants that are lost — a single categorical mixture per lineage is structurally wrong | Multi-state transition process / explicitly enumerated trajectory summaries (spec §3.3); statistically, this is exactly what competing-risks and multi-state models are built for — multiple, non-exclusive, time-indexed transition probabilities per unit rather than one label (Putter et al. 2007 [R30]; Cornwell et al. 2016 [R31]) |

---

## UNVERIFIED — could not retrieve

The following were surfaced by search but I could not independently confirm DOI/PMID and
full-text content with a direct tool-call fetch in this pass. They carry **zero
evidential weight** per the SHARED_BRIEF's non-negotiable evidence rules and must not
enter any evidence table or gate verdict without independent re-verification:

- A bioRxiv paper on Bayesian phylodynamic inference for single-cell CRISPR/Cas9 lineage-
  tracing barcode data with dependent target sites (title located via search, associated
  with a Royal Society *Phil. Trans. B* DOI pattern; not independently fetched/confirmed).
- A bioRxiv paper assessing the inference of single-cell phylogenies and population
  dynamics from CRISPR lineage recordings (title and general claim about sampling-induced
  systematic overestimation of division rate located via search snippet only; full text
  not fetched).
- A bioRxiv paper on detecting branching-rate heterogeneity in multifurcating trees for
  lineage-tracing data (title located via search only).
- The precise page range and exact author-order for Vitale et al. 2023 NCCD apoptosis
  update ([R10]) beyond volume 30, issue 5, pages 1097-1154 — full text was not
  retrievable (Nature paywall redirect loop each time), so any definitional sentence from
  that document specifically (as opposed to the 2018 general NCCD document, which *was*
  fetched in full) should be treated as `abstract-only` until re-verified.
- Any existing published design that is a genuine Candidate-B rank-1 "condition-level
  paired prospective anchor" (state at `t0`, independently measured dynamics over
  `[t0,T]`) for a genetic-perturbation, cancer-cell-line context specifically — none was
  located in this pass. Per spec §4.3, this is reported as **insufficient evidence**, not
  as proof that no such design exists.
- Whether Live-seq-class non-destructive sampling has been coupled to CRISPR/Cas9
  perturbation in any cell line, cancer or otherwise, since the original 2022 publication
  — not located in this pass; the original paper itself confirms it was not done as of
  publication [R25].

---

## Consolidated Reference List

All entries were retrieved via WebSearch/WebFetch/Crossref tool calls made in this run.

- **[R1]** Trapnell C. "Defining cell types and states with single-cell genomics." *Genome
  Res.* 2015;25(10):1491-1498. DOI `10.1101/gr.190595.115`. PMID 26430159. PMC4579334.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4579334/
- **[R2]** Weinreb C, Wolock S, Tusi BK, Socolovsky M, Klein AM. "Fundamental limits on
  dynamic inference from single-cell snapshots." *Proc Natl Acad Sci USA*.
  2018;115(10):E2467-E2476. DOI `10.1073/pnas.1714723115`. PMID 29463712.
  https://www.pnas.org/doi/10.1073/pnas.1714723115
- **[R3]** Wagner DE, Klein AM. "Lineage tracing meets single-cell omics: opportunities
  and challenges." *Nat Rev Genet*. 2020;21(7):410-427. DOI `10.1038/s41576-020-0223-2`.
  PMC7307462. https://pmc.ncbi.nlm.nih.gov/articles/PMC7307462/
- **[R4]** Weinreb C, Rodriguez-Fraticelli A, Camargo FD, Klein AM. "Lineage tracing on
  transcriptional landscapes links state to fate during differentiation." *Science*.
  2020;367(6479):eaaw3381. DOI `10.1126/science.aaw3381`. PMID 31974159.
  https://pubmed.ncbi.nlm.nih.gov/31974159/
- **[R5]** Pennycook BR, Barr AR. "Restriction point regulation at the crossroads between
  quiescence and cell proliferation." *FEBS Lett*. 2020;594(13):2046-2060. DOI
  `10.1002/1873-3468.13867`. https://febs.onlinelibrary.wiley.com/doi/10.1002/1873-3468.13867
- **[R6]** Spencer SL, Cappell SD, Tsai FC, Overton KW, Wang CL, Meyer T. "The
  proliferation-quiescence decision is controlled by a bifurcation in CDK2 activity at
  mitotic exit." *Cell*. 2013;155(2):369-383. DOI `10.1016/j.cell.2013.08.062`. PMID
  24075009. https://pubmed.ncbi.nlm.nih.gov/24075009/
- **[R7]** Green DR, Kroemer G. "The pathophysiology of mitochondrial cell death."
  *Science*. 2004;305(5684):626-629. DOI `10.1126/science.1099320`. PMID 15286356.
  https://pubmed.ncbi.nlm.nih.gov/15286356/
- **[R8]** Spencer SL, Gaudet S, Albeck JG, Burke JM, Sorger PK. "Non-genetic origins of
  cell-to-cell variability in TRAIL-induced apoptosis." *Nature*. 2009;459(7245):428-432.
  DOI `10.1038/nature08012`. PMID 19363473. https://pubmed.ncbi.nlm.nih.gov/19363473/
- **[R9]** Galluzzi L, Vitale I, Aaronson SA, et al. "Molecular mechanisms of cell death:
  recommendations of the Nomenclature Committee on Cell Death 2018." *Cell Death Differ*.
  2018;25(3):486-541. DOI `10.1038/s41418-017-0012-4`. PMID 29362479. **Full text
  fetched.** https://www.nature.com/articles/s41418-017-0012-4 ;
  https://pubmed.ncbi.nlm.nih.gov/29362479/
- **[R10]** Vitale I, Pietrocola F, Guilbaud E, et al. "Apoptotic cell death in disease —
  Current understanding of the NCCD 2023." *Cell Death Differ*. 2023;30(5):1097-1154. DOI
  `10.1038/s41418-023-01153-w`. https://www.nature.com/articles/s41418-023-01153-w
  (`abstract+summary` only — full text not retrievable this pass)
- **[R11]** Yang D, Jones MG, Naranjo S, et al. "Lineage tracing reveals the phylodynamics,
  plasticity, and paths of tumor evolution." *Cell*. 2022;185(11):1905-1923.e25. DOI
  `10.1016/j.cell.2022.04.015`. https://www.cell.com/cell/fulltext/S0092-8674(22)00462-7
- **[R12]** van den Brink SC, Sage F, Vértesy Á, et al. "Single-cell sequencing reveals
  dissociation-induced gene expression in tissue subpopulations." *Nat Methods*.
  2017;14(10):935-936. DOI `10.1038/nmeth.4437`. PMID 28960196.
  https://pubmed.ncbi.nlm.nih.gov/28960196/
- **[R13]** Denisenko E, Guo BB, Jones M, et al. "Systematic assessment of tissue
  dissociation and storage biases in single-cell and single-nucleus RNA-seq workflows."
  *Genome Biol*. 2020;21:130. DOI `10.1186/s13059-020-02048-6`.
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02048-6
- **[R14]** Hong R, Koga Y, Bandyadka S, et al. "Comprehensive generation, visualization,
  and reporting of quality control metrics for single-cell RNA sequencing data." *Nat
  Commun*. 2022;13:1688. DOI `10.1038/s41467-022-29212-9`. **Full text fetched.**
  https://www.nature.com/articles/s41467-022-29212-9 ;
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8967915/
- **[R15]** Yates J, Kraft A, Boeva V. "Filtering cells with high mitochondrial content
  depletes viable metabolically altered malignant cell populations in cancer single-cell
  studies." *Genome Biol*. 2025;26:91. DOI `10.1186/s13059-025-03559-w`. PMID 40205439.
  PMC11983838. **Full text fetched.** https://pmc.ncbi.nlm.nih.gov/articles/PMC11983838/
- **[R16]** Coller HA, Sang L, Roberts JM. "A new description of cellular quiescence."
  *PLoS Biol*. 2006;4(3):e83. DOI `10.1371/journal.pbio.0040083`. PMID 16509772.
  https://pubmed.ncbi.nlm.nih.gov/16509772/
- **[R17]** van Deursen JM. "The role of senescent cells in ageing." *Nature*.
  2014;509(7501):439-446. DOI `10.1038/nature13193`. https://www.nature.com/articles/nature13193
- **[R18]** Sharpless NE, Sherr CJ. "Forging a signature of in vivo senescence." *Nat Rev
  Cancer*. 2015;15(7):397-408. DOI `10.1038/nrc3960`. PMID 26105537.
  https://pubmed.ncbi.nlm.nih.gov/26105537/
- **[R19]** Ashraf HM, Fernandez B, Spencer SL. "The intensities of canonical senescence
  biomarkers integrate the duration of cell-cycle withdrawal." *Nat Commun*.
  2023;14:4423. DOI `10.1038/s41467-023-40132-0`. PMC10374620.
  https://www.nature.com/articles/s41467-023-40132-0
- **[R20]** Fernandez B, Passanisi VJ, Ashraf HM, Spencer SL. "Single-cell RNA sequencing
  reveals a quiescence-senescence continuum and distinct senotypes following
  chemotherapy." *Nat Commun*. 2025. DOI `10.1038/s41467-025-66836-z`.
  https://www.nature.com/articles/s41467-025-66836-z
- **[R21]** Sharma SV, Lee DY, Li B, et al. "A chromatin-mediated reversible drug-tolerant
  state in cancer cell subpopulations." *Cell*. 2010;141(1):69-80. DOI
  `10.1016/j.cell.2010.02.027`. PMID 20371346. https://pubmed.ncbi.nlm.nih.gov/20371346/
- **[R22]** Russo M, Chen M, Mariella E, et al. "Cancer drug-tolerant persister cells: from
  biological questions to clinical opportunities." *Nat Rev Cancer*. 2024;24(10):694-717.
  DOI `10.1038/s41568-024-00737-z`. https://www.nature.com/articles/s41568-024-00737-z
- **[R23]** Balaban NQ, Merrin J, Chait R, Kowalik L, Leibler S. "Bacterial persistence as
  a phenotypic switch." *Science*. 2004;305(5690):1622-1625. DOI
  `10.1126/science.1099390`. PMID 15308767. https://pubmed.ncbi.nlm.nih.gov/15308767/
- **[R24]** Hangauer MJ, Viswanathan VS, Ryan MJ, et al. "Drug-tolerant persister cancer
  cells are vulnerable to GPX4 inhibition." *Nature*. 2017;551(7679):247-250. DOI
  `10.1038/nature24297`. https://www.nature.com/articles/nature24297
- **[R25]** Chen W, Guillaume-Gentil O, Rainer PY, et al. "Live-seq enables temporal
  transcriptomic recording of single cells." *Nature*. 2022;608(7924):733-740. DOI
  `10.1038/s41586-022-05046-9`. PMID 35978187. PMC9402441. **Full text fetched.**
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9402441/
- **[R26]** Hormoz S, Singer ZS, Linton JM, Antebi YE, Shraiman BI, Elowitz MB. "Inferring
  cell-state transition dynamics from lineage trees and endpoint single-cell
  measurements." *Cell Syst*. 2016;3(5):419-433.e8. DOI `10.1016/j.cels.2016.10.015`. PMID
  27883889. https://pubmed.ncbi.nlm.nih.gov/27883889/
- **[R27]** Dempster JM, Boyle I, Vazquez F, Root DE, Boehm JS, Hahn WC, Tsherniak A,
  McFarland JM. "Chronos: a cell population dynamics model of CRISPR experiments that
  improves inference of gene fitness effects." *Genome Biol*. 2021;22:343. DOI
  `10.1186/s13059-021-02540-7`. PMID 34930405. PMC8686573. **Full text fetched.**
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8686573/
- **[R28]** Hafner M, Niepel M, Chung M, Sorger PK. "Growth rate inhibition metrics correct
  for confounders in measuring sensitivity to cancer drugs." *Nat Methods*.
  2016;13(6):521-527. DOI `10.1038/nmeth.3853`. PMID 27135972. PMC4887336. **Full text
  fetched.** https://pmc.ncbi.nlm.nih.gov/articles/PMC4887336/
- **[R29]** Schiebinger G, Shu J, Tabaka M, et al. "Optimal-transport analysis of
  single-cell gene expression identifies developmental trajectories in reprogramming."
  *Cell*. 2019;176(4):928-943.e22. DOI `10.1016/j.cell.2019.01.006`. PMID 30712874.
  https://pubmed.ncbi.nlm.nih.gov/30712874/
- **[R30]** Putter H, Fiocco M, Geskus RB. "Tutorial in biostatistics: competing risks and
  multi-state models." *Stat Med*. 2007;26(11):2389-2430. DOI `10.1002/sim.2712`. PMID
  17031868. https://onlinelibrary.wiley.com/doi/10.1002/sim.2712
- **[R31]** Cornwell JA, Hallett RM, Auf der Mauer S, Motazedian A, Schroeder T, Draper JS,
  Harvey RP, Nordon RE. "Quantifying intrinsic and extrinsic control of single-cell fates
  in cancer and stem/progenitor cell pedigrees with competing risks analysis." *Sci Rep*.
  2016;6:27100. DOI `10.1038/srep27100`. PMID 27250534. PMC4890426. **Full text fetched.**
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4890426/

**Seed papers from SHARED_BRIEF, re-verified here:**

- Live-seq = [R25] above (volume corrected from brief's "Nature 609" to the Crossref-
  confirmed 608(7924):733-740; DOI is identical and authoritative).
- Death-seq: Colville A, Liu JY, et al. "Death-seq identifies regulators of cell death and
  senolytic therapies." *Cell Metab*. 2023. DOI `10.1016/j.cmet.2023.08.008`. PMID
  37699398. PMC10597643. **Full text fetched.** Confirmed: positive selection for
  detached/dying cells contrasted with negative-selection dropout screens; screens death
  enhancers specifically under senescence-inducing/drug contexts (Doxo-SEN + ABT-263 or
  natural senescent death), not a general division-vs-loss decomposition for an arbitrary
  perturbation; authors note any knockout affecting cell attachment can confound the assay
  ("an assay for cell detachment accompanying cell death"). https://pmc.ncbi.nlm.nih.gov/articles/PMC10597643/
- Chronos = [R27] above.
- GR/DIP = [R28] above.

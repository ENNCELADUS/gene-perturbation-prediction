# Gate 2B — Phenomenon Prevalence: Candidate B (Population Level)

**Question:** Under matched net fitness, does reproducible divergence in population
dynamics exist, and is it large enough to matter?

**Governing documents:** SHARED_BRIEF.md; SIGNIFICANCE_CRITERIA.md (FROZEN 2026-07-13);
`docs/superpowers/specs/2026-07-13-cell-fate-prediction-research-direction-design.md`
§4 Candidate B, §8/Gate 2. All thresholds below are taken verbatim from
SIGNIFICANCE_CRITERIA.md; none are invented here.

All citations below were retrieved via WebSearch/WebFetch/Crossref/EuropePMC tool calls
in this run. URLs actually fetched are listed per entry. No claim is made from memory.

---

## Evidence table

| # | Paper | Perturbation type | B-tier | T1/T2/T3 | What it shows | Criterion it bears on |
|---|---|---|---|---|---|---|
| 1 | Pau, Walter, Neumann, Hériché, Ellenberg, Huber. "Dynamical modelling of phenotypes in a genome-wide RNAi live-cell imaging assay." *BMC Bioinformatics* 14:308 (2013). DOI `10.1186/1471-2105-14-308`, PMID 24131777, PMCID PMC3827932. Builds on Neumann et al., "Phenotypic profiling of the human genome by time-lapse microscopy reveals cell division genes," *Nature* 464:721 (2010), DOI `10.1038/nature08869`, PMID 20360735, PMCID PMC3108885 (Mitocheck). | RNAi (siRNA), genome-scale (~51,766 constructs / 17,293 genes), HeLa-H2B-GFP | **1** by design (state at seeding/`t0`, continuous imaging dynamics over 48 h per condition) — but **not matched to an independent net-fitness scalar**, so cannot itself certify a *matched-pair* prevalence number | T1 (per-frame nuclear morphology classification) aggregated into a **T2-flavored** per-gene population-dynamics model (penetrance + timing of quiescence/arrest/polynucleation/death) | ~2,190 of ~51,766 constructs (~4%) show reproducibly-timed cell-cycle disruption, sorted into **qualitatively distinct** dynamics classes: quiescence, mitotic arrest, polynucleation, cell death (per-category counts extracted via tool are approximate — see caveat below). Classification reproducibility high: 94.2% cross-validated separation of control spots, mean relative model-fit error 3.2%; 98.7% of genes had ≥2 independent siRNA reagents. Mitotic-arrest timing correlates with subsequent death timing at r=0.80 among the subset (~36 siRNAs) showing both. | Best available **existence-proof of category-level divergence** in a genome-scale genetic-perturbation panel with a genuine paired-prospective design. **Wrong denominator for COMMON** (not net-fitness-matched pairs). Reproducibility ≈ R1 (multi-reagent, single screen), not demonstrated R2 (independent dataset). The r=0.80 arrest→death coupling is mild **counter-evidence** against full independence of the two processes. |
| 2 | Gross, Mohammadi, Sanchez-Aguila, Zhan, Liby, Dane, Meyer, Heiser. "Analysis and modeling of cancer drug responses using cell cycle phase-specific rate effects." *Nat Commun* 14:3450 (2023). DOI `10.1038/s41467-023-39122-z`, PMID 37301933, PMCID PMC10257663. | Small-molecule drugs (lapatinib, gemcitabine, paclitaxel, palbociclib, doxorubicin), HER2+ breast cancer lines (AU565 + 3 validation lines) | 1 (paired prospective; per-drug condition, reporter-based state + continuous dynamics over 96 h) | T2 — mechanistic ODE model separately estimates phase progression rates (α, β) **and** phase-specific death rates (γ1, γ2) directly from a cell-cycle reporter (HDHB-GFP), not inferred from net counts alone | **Explicit existence proof of the wedge in a real system**, quoted verbatim: "these drugs yield similar final numbers through distinct impacts on the cell cycle." Lapatinib (G1 arrest, minimal death) and gemcitabine (S-G2 extension + substantial death) reach similar final cell numbers (~0.5× control) via opposite decompositions; paclitaxel reaches a comparable endpoint via ~56% relative cell death. A standard endpoint-only Bliss-additivity model **mispredicts** a lapatinib+gemcitabine combination unless phase-specific dynamics are modeled. | **LARGE** (qualitative — near-zero death vs. majority death at matched endpoint) and directly refutes "net count is a sufficient statistic" **in this drug system**. Not genetic perturbation; n=5 drugs, no panel-wide prevalence; no explicit noise-bar (replicate SD) reported for the rate parameters. |
| 3 | Colville, Liu, et al. "Death-seq identifies regulators of cell death and senolytic therapies." *Cell Metab* 35 (2023). DOI `10.1016/j.cmet.2023.08.008`, PMID 37699398, PMCID PMC10597643. | Genome-wide CRISPR KO, IMR-90 fibroblasts, doxorubicin-induced senescence + ABT-263 | 4 (terminal-state classifier: single endpoint live-vs-dying pool split, no state-at-`t0` + dynamics pairing, no net-fitness matching) | **T1** explicitly, not T2 — a positive-selection screen splitting live/floating pools at one 24 h endpoint under one fixed background treatment | 31 genes (10% FDR) shift sgRNA representation between live and dying pools. Does **not** decompose an arbitrary perturbation's own division-vs-death dynamics; no matched-net-fitness comparison; no dynamics or regrowth/clonogenic follow-up (validation is Annexin V, XTT, caspase 3/7 — all endpoint assays). | Confirms the field's existing death-targeted readouts are Tier-4/T1 by design. Essentially **uninformative** for COMMON/LARGE/CONSEQUENTIAL as specified (design mismatch), though it establishes that a death-vs-live split *can* be engineered as an independent readout in principle. |
| 4 | Hafner, Niepel, Chung, Sorger. "Growth rate inhibition metrics correct for confounders in measuring sensitivity to cancer drugs." *Nat Methods* 13:521 (2016). DOI `10.1038/nmeth.3853`, PMCID PMC4887336. | Small-molecule drugs, LINCS cell-line panel | 3–4 (cross-sectional condition comparison; GRmax is **computed from the same net cell-count series** it is meant to explain) | Nominally T2, but see caveat | Paclitaxel is cytotoxic (GRmax<0) in HER2amp/TNBC lines vs. cytostatic in HR+/non-malignant lines — but net GR values also differ across these contexts, so this is a **different-net-effect** comparison, not matched. No panel-wide prevalence of cytostatic/cytotoxic divergence **at matched net effect** is reported. Paper does not state whether GR metrics alone (without an independent death assay) can separate death from arrest. | Illustrates the **cytostatic/cytotoxic net-statement trap** named in the task: the classification is derived from the same bulk time-course it purports to decompose. Gate-1-relevant caution, weak/indirect support at best for COMMON. |
| 5 | Harris, Frick, Garbett, Hardeman, Paudel, Lopez, Quaranta, Tyson. "An unbiased metric of antiproliferative drug effect in vitro." *Nat Methods* 13:497 (2016). DOI `10.1038/nmeth.3852`, PMCID PMC4887341. | Small-molecule drugs, MDA-MB-231 and other lines | 3 (cross-sectional condition comparison; DIP rate is a single net rate parameter from population counts) | Nominally T2, but explicitly a net statistic | Paper **explicitly states** DIP rate "cannot differentiate" no-proliferation/no-death from high-division-matched-by-high-death without an independent death assay. Example given: rotenone ("partially cytostatic") vs. phenformin ("cytotoxic") at similar potency in MDA-MB-231 — a small (n=2), illustrative instance of divergent character at roughly matched effect. Reproducibility reported only as "repeated at least twice," technical duplicates; no formal cross-replicate statistics. | Directly names the **identifiability limit** underlying the whole wedge (Gate-1-relevant). The rotenone/phenformin pair is a genuine but n=1-pair, drug-only, non-systematic hint toward COMMON — not evidentiary at the required scale. |
| 6 | Dempster, Boyle, Vazquez, Root, Boehm, Hahn, Tsherniak, McFarland. "Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects." *Genome Biol* 22:343 (2021). DOI `10.1186/s13059-021-02540-7`, PMCID PMC8686573. | Genome-scale CRISPR KO, DepMap panel | N/A — measurement/identifiability tool, not a phenomenon anchor | N/A | Fits a single relative growth-rate effect per gene per cell line from sgRNA-abundance time series; **does not** separately identify a death-rate parameter (per SHARED_BRIEF/spec framing, consistent with what this run found). This is the model that produces the `F_net` scalar Candidate B's matching criterion is defined against. | Establishes that the field's default net-fitness readout **cannot by construction** answer Gate 2B's question; the decomposition has to come from an independent source. Read at abstract/search-summary depth, not full-text-verified line-by-line in this run — flag accordingly. |
| 7 | Williams, Gervasio, Turkal, Stuhlfire, Wang, Mauch, Plawat, Nguyen, Paw, Hairani, Lathrop, Harris, Page, Hangauer. "DNA fragmentation factor B suppresses interferon to enable cancer persister cell regrowth." *Nat Cell Biol* 27:2143–2151 (2025). DOI `10.1038/s41556-025-01810-x`, PMCID PMC12717002. | **Genetic** (CRISPR KO of DFFB) crossed with drug-induced persister state, 3 independent systems: A375 melanoma (dabrafenib+trametinib), PC9 NSCLC (erlotinib/osimertinib), BT474 breast (lapatinib) | **2** (condition-level anchor with partial time resolution: baseline viability/persister-formation state, then long-term regrowth measured 5–11 weeks later) | **T2**, reproduced across 3 independent cell-line/drug systems (n=3 biological replicates each, t-tests) | DFFB-KO vs. WT explicitly reported as **"independent of tumour cell viability, initial drug response or persister cell formation"** (i.e., matched at the short-term net-outcome level) yet DFFB-KO persister cells were **"severely hindered"** in forming DTEP colonies over weeks — an endpoint (long-term regrowth/colony formation) **outside** the trajectory definition. | **Closest on-point evidence found for CONSEQUENTIAL.** Matched short-term outcome, divergent long-term regrowth, by a genetic modifier, reproduced across 3 independent systems (R2, arguably approaching R3 across cell lines/drug classes). Caveat: the "state" is a stable genotype, not a measured early single-cell state *distribution*; matching is to viability/persister-formation, not to a tau-defined DepMap/Chronos GeneEffect. **Adjacent, not on-target.** |
| 8 | Nano, Mondo, Harwood, Balasanyan, Montell. "Cell survival following direct executioner-caspase activation." *PNAS* 120:e2216531120 (2023). DOI `10.1073/pnas.2216531120`, PMCID PMC9942801. | Optogenetic/chemogenetic direct caspase-3 activation (CaspaseLOV) in HeLa (endogenous CASP3 CRISPR-KO background) — a synthetic perturbation of the death-effector pathway itself, not a genetic loss-of-function screen | A1-analog (same-cell, non-destructive prospective: live caspase-reporter imaging, then the *same* cells' fate tracked ≥20 h) | T2 | At a **matched** intermediate stimulus dose, the population splits into ~15–30% death / 70–85% survival. Logistic regression (Tjur's R²) of fate on caspase kinetics: fold-change 0.02, rate 0.16, AUC ≈0.00, combined "death score" 0.00 (p=0.75) under baseline conditions — **near-zero prospective information** from the proximate death-pathway state. Sister cells from one mitosis (identical genetic/environmental context) diverge in fate in 33 tracked pairs; in 5/33 the *surviving* sister had *higher* caspase than its dying sibling. Under added phototoxic/proteotoxic stress, predictive power rose to R²=0.70. | **Primary counter-evidence entry.** A plausible early state readout, at a matched-dose population, explains little of the fate split under baseline conditions — context-dependent, not absent. Per-cell (Candidate-A-relevant primarily); survival/death only, no division axis; synthetic optogenetic system, not a genetic screen. |
| 9 | Ishikawa et al. "RENGE infers gene regulatory networks using time-series single-cell RNA-seq data with CRISPR perturbations." *Commun Biol* (2023). DOI `10.1038/s42003-023-05594-4`, PMCID PMC10754834. | CRISPR KO of 23 pluripotency transcription factors, hiPSC | N/A — no fitness/division/death axis | N/A | 4 timepoints (days 2–5 post-transduction), ~75 cells/gRNA, purely transcriptional network inference in surviving/selected (BFP+) cells. **No** viability, death, or division tracking. | Does not bear on any of the three criteria. Logged only to document that time-resolved single-cell genetic-perturbation designs are technically feasible at modest scale (~75 cells/condition, 4 timepoints) — relevant to bounded-pilot feasibility, not to the phenomenon question. |

**Numeric-precision caveat (Mitocheck, row 1):** category counts (168/289/390/171) were
extracted by an AI-summarization tool from the full text and do not cleanly sum to the
"~2,190 reproducibly-timed" headline figure quoted elsewhere in the same paper. Treat the
per-category breakdown as **approximate**, pending direct verification against the
paper's primary tables, while the qualitative finding (multiple distinct, reproducible
dynamics classes exist genome-wide) is corroborated by the reported classification
accuracy (94.2%) and model fit error (3.2%) statistics, which were stated as direct
quotes rather than summarized arithmetic.

---

## Per-criterion verdicts

### COMMON (≥20% of net-fitness-matched perturbation pairs, matched at `tau`, show R2-reproducible divergence)

**Verdict: insufficient evidence.** No study found — genetic or pharmacological — computes
this exact quantity: take perturbations with non-trivial fitness effect, pair them within
`tau` (1× replicate SD of GeneEffect), and ask what fraction diverge in trajectory at R2.
The single design-appropriate genome-scale genetic-perturbation anchor (Mitocheck, row 1)
reports a *different* prevalence — the fraction of *all* tested reagents (not
net-fitness-matched pairs) showing reproducibly-timed, qualitatively distinct dynamics
classes (~4%) — which cannot be substituted for the frozen quantity without committing
the denominator trap the task explicitly warns against. Drug-pharmacology evidence (rows
2, 4, 5) supplies existence-proof-level and n=1–2-pair illustrations, not a prevalence
estimate, and is off-target modality (small molecule, not genetic). This is not a powered
absence: no adequately powered, correctly-matched study looked and found <5%. The gap is
that the right study has not been run, not that a wrong-null was returned.

### LARGE (≥0.20 absolute founder-loss or ≥30% relative division-rate difference, AND >3× replicate SD)

**Verdict: insufficient evidence.** No genetic-perturbation study reports founder-referenced
fraction-lost or division-rate differences at matched net fitness with an accompanying
noise bar. The one system where a clearly large-looking divergence is directly
demonstrated with a mechanistic (not net-count-inferred) decomposition is drug
pharmacology (Gross et al. 2023, row 2): near-zero death vs. majority death at matched
final cell number. This shows the phenomenon *can* be large when present, but supplies no
noise-bar statistic and is not on the genetic-perturbation unit.

### CONSEQUENTIAL (predicts a downstream endpoint outside the trajectory definition, by stated margin, beyond `F_net` alone)

**Verdict: insufficient evidence, but the closest positive-direction signal in the entire
review.** Williams et al. 2025 (row 7) matches persister populations on short-term
viability/drug-response/persister-formation and shows a genetic factor (DFFB) produces a
qualitatively large, reproducible (3 independent cell-line/drug systems) divergence in
long-term regrowth — precisely the "regrowth after perturbation withdrawal" /
"resistance-emergence" row of the CONSEQUENTIAL table, on an endpoint outside the
trajectory definition. It falls short of certifying CONSEQUENTIAL only because: (a) the
matching variable is short-term viability, not a tau-defined DepMap/Chronos GeneEffect;
(b) the "state" is a stable genotype rather than an early single-cell state distribution;
(c) no quantitative margin over an `F_net`-only baseline (e.g., Spearman or fold-separation
number) is reported against the specific thresholds in SIGNIFICANCE_CRITERIA. It is strong
enough to materially shape the bounded pilot design below.

---

## Overall Gate 2B verdict

**Insufficient evidence → bounded validation pilot, not a stop.**

No criterion has an adequately powered, on-target (genetic perturbation, matched net
fitness, R2 reproducibility) study demonstrating either presence at the frozen thresholds
or absence. What the literature *does* establish, robustly:

1. The measurement-level premise is well-founded and explicitly stated in the field's own
   foundational papers (Harris et al. 2016, row 5): net population-count metrics (DIP
   rate, and by the same mathematics GR metrics) **cannot**, on their own, distinguish "no
   division/no death" from "matched high division and high death." This is not a new
   observation of this review — it is already in print.
2. When an independent, mechanistic (not net-count-derived) decomposition is applied in a
   real cellular system, the wedge phenomenon is **directly observed and can be large**
   (Gross et al. 2023, row 2) — but only for drugs, not genetic perturbations, and without
   a systematic prevalence estimate.
3. The closest thing to a consequential, reproducible, matched-short-term / divergent-
   long-term genetic result (Williams et al. 2025, row 7) exists in persister/regrowth
   biology, not in a DepMap/Chronos-matched genetic-perturbation-library design.
4. No genome-scale CRISPR/CRISPRi screen was found that (a) independently measures
   division and death (or population dynamics more broadly) per gene, (b) matches genes on
   net fitness, and (c) reports the prevalence/effect-size/consequence of divergence at
   that matched fitness. This is a genuine gap, not a null result.

Per the Gate 2 verdict rule, absence-of-the-right-study routes to *insufficient evidence*,
never to *powered absence*, and never to *stop*.

---

## Bounded validation pilot (sketch)

**System:** K562 (matches the project's proof-of-concept cell line and existing
DepMap/Chronos GeneEffect data; avoids introducing a second cell-line variable).

**Pair selection:** Using the actual K562 CRISPR-screen replicate data (per
SIGNIFICANCE_CRITERIA's `tau` definition), select ~10–20 matched pairs (`|F_net(g1) -
F_net(g2)| <= tau`, reported additionally at `0.5*tau` and `2*tau` per the sensitivity
requirement) spanning a range of non-trivial `|F_net|` — some near strongly essential,
some mildly essential — rather than only the strongest hits, since prevalence claims
require sampling across the effect-size range, not just the tail.

**Measurement:** Arrayed (not pooled) CRISPRi knockdown of the selected genes in K562,
with either (a) an IncuCyte-style live-cell imaging time course (cell count + a death dye,
e.g., a fixable/non-fixable viability stain compatible with continued imaging) at
≥4 timepoints over a defined `T` (matching or modestly exceeding the DepMap screen
window), or (b) a FUCCI/cell-cycle-phase reporter line if phase-resolved decomposition
(as in Gross et al. 2023) is feasible at this scale — this directly reuses the one design
in this review's evidence base that produced an unambiguous decomposition rather than a
net-count inference.

**Regrowth arm:** After `T`, withdraw the CRISPRi induction (dox washout) or passage/dilute
the surviving population and track outgrowth over an additional interval, directly
borrowing the Williams et al. 2025 logic (row 7) to test the CONSEQUENTIAL criterion on an
endpoint outside the trajectory definition (regrowth rate; colony-forming efficiency if
feasible).

**Power / stop rule:** Before collecting data, use the pilot's own technical/biological
replicate spread to compute the minimum detectable effect at the frozen thresholds (0.20
absolute founder-loss difference or 30% relative division-rate difference, at >3×
replicate SD). If the pilot is powered to detect the threshold and finds divergence in
`<5%` of matched pairs, that is a legitimate contribution toward *powered absence* for a
future, larger Gate 2B revisit. If `5–20%`, scope any follow-on work to the specific
sub-population showing divergence, not to perturbations in general (per the `5–20%`
partial routing). If `>=20%` and the regrowth arm clears the `>=0.10` absolute
Spearman / `>=2×` regrowth-rate-separation margin, escalate to a larger production study.
**Explicit budget ceiling:** this is a small-array, single-cell-line, single-modality
pilot (≤20–40 genes, a few weeks of imaging plus one passage/regrowth interval) — it is
not powered to make a generality (R3) claim, only to move Gate 2B from *insufficient* to a
resolvable *positive* or *powered-absence* signal for K562/CRISPRi specifically.

---

## Counter-evidence

- **Mitocheck (row 1):** among siRNAs producing both mitotic arrest and death, arrest
  timing and death timing are correlated at **r=0.80** — indicating the two processes are
  *not* independent within a perturbation, which tempers (does not refute) the premise
  that arrest and death vary freely of one another. This is a within-perturbation temporal
  coupling, not a matched-net-effect independence test, so it is partial counter-evidence
  at most.
- **Nano et al. 2023 / PNAS (row 8):** the clearest single falsifier-flavored result found.
  At a matched stimulus dose, a plausible early "state" readout (caspase activation
  kinetics) has near-zero-to-low prospective information (Tjur R² 0.00–00.16) about
  individual cell fate under baseline conditions; genetically and environmentally
  identical sister cells diverge in fate in a way not explained by measured caspase level
  in 5/33 discordant pairs. Predictive power rises sharply (R²=0.70) only under added
  exogenous stress — showing the phenomenon is real but **context-gated**, not absent.
  Caveats: single-cell resolution (bears more directly on Candidate A), survival/death
  only (no division axis), and a synthetic optogenetic perturbation of the death pathway
  itself rather than a genetic loss-of-function screen.
- **CRISPR-dropout-screen methodology literature** (search-summary level only, not
  full-text-verified in this run): multiple sources describe early-to-intermediate
  pooled-screen depletion signal as reflecting proliferation-rate slowdown more than acute
  death for a substantial share of hits (e.g., an NEDD8 example where knockout "only
  delays cell division initially" with proliferation later unaffected). If broadly true,
  this would push the *modal* genetic-perturbation fitness effect toward the
  division-suppression end, which would argue for a **lower**, not higher, COMMON
  prevalence of division/death divergence specifically. Flagged as weaker-confidence
  (search-summary only) and not incorporated into the formal verdict above.
- **Measurement papers themselves (rows 4, 5, 6)** state the identifiability limit as a
  reason decomposition is *hard to see*, not as evidence the underlying phenomenon is
  absent. This distinction is preserved throughout: a missing decomposition tool is a
  Gate-1 problem, not Gate-2B evidence of absence.

---

## UNVERIFIED — could not retrieve or not fetched in depth

- "Population Consequences of Single-Cell Damage Dynamics: An Overlooked Role of
  Mortality Under Stress" (bioRxiv, 2025). URL attempted:
  `https://www.biorxiv.org/content/10.1101/2025.09.25.678661.full.pdf` — **HTTP 403,
  could not retrieve.** Title and apparent relevance (population-level mortality readouts
  potentially masking single-cell damage heterogeneity) found only via WebSearch snippet.
  Zero evidential weight; not used anywhere above.
- "Inducible CRISPR–Cas9 screening platform to interrogate non-proliferative cellular
  states" (*Nature Protocols*, companion methods paper in the Death-seq lineage) — found
  via WebSearch title/URL only; full text not fetched in this run. Not used as evidence.
- "Robust high-throughput kinetic analysis of apoptosis with real-time high-content
  live-cell imaging" (*Cell Death & Disease*, 2016; PMC5261025) — found via WebSearch
  title/URL only; full text not fetched. Not used as evidence.
- The claim (surfaced in one WebSearch snippet, subsequently traced to Nano et al. 2023
  PNAS and confirmed there directly) that "optogenetic caspase-3 activation... neither
  rate, peak, nor total caspase activity predicted survival" is **verified** — see row 8 —
  but the original snippet mis-attributed it to a different persister-biology context
  before the correct primary source was located and fetched.

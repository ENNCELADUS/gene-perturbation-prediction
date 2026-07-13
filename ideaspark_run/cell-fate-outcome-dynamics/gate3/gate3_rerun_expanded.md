# Gate 3 — RERUN (Expanded): Jost 2020 Deep Read, Missing Neighbours, Gate 2 Cross-Feed, Expected-Information-Value

Scope of this rerun: closes the four gaps named in DECISION_MEMO.md v2 §3 and the
reviewer verdict — a deep read of Jost et al. 2020, the Dixit/Replogle/Nadal-Ribelles
missing neighbours, cross-feeding Gate 2's finds into the prior-art matrix, and the
expected-information-value question (retrospective reanalysis vs. new Study 0). This
document **extends** `gate3/gate3_prior_art_matrix.md`; it does not repeat that matrix's
20 rows.

Method note, per SHARED_BRIEF non-negotiable rules: every paper below was retrieved via
direct tool calls (Crossref, Europe PMC search + full-text HTML fetch from
`pmc.ncbi.nlm.nih.gov`) in this run. Full-text was retrieved and grepped/read for Jost,
Dixit, Replogle, Nadal-Ribelles, Panagopoulos, and the multi-level-barcoding paper — not
abstract-only. Quoted sentences below are copied verbatim from the fetched full text.
Mitocheck/Pau figures are carried forward from Gate 2b (`gate2b_population_phenomenon.md`),
which already deep-read them; their DOI/PMID/PMCID were independently re-verified via
Crossref/Europe PMC in this run (see §2 table) rather than trusted from memory.

---

## 1. Jost et al. 2020 deep read — the burden-axis verdict

**Citation.** Jost M, Santos DA, Saunders RA, Horlbeck MA, Hawkins JS, Scaria SM, Norman
TM, Hussmann JA, Liem CR, Gross CA, Weissman JS. "Titrating gene expression using
libraries of systematically attenuated CRISPR guide RNAs." *Nat Biotechnol* 38:355–364
(2020). DOI `10.1038/s41587-019-0387-5`, PMID `31932729`, PMCID `PMC7065968`. Retrieved
full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC7065968/ (confirmed open access via
Europe PMC `fullTextUrlList`, `availabilityCode: OA`).

**System, in numbers.** K562 CRISPRi. A large-scale mismatched-sgRNA screen across
~2,400 essential genes established rules for graded knockdown; a compact 128-sgRNA /
25-gene "allelic series" library (4–5 mismatched variants per gene, spanning full to
low activity) was then profiled by Perturb-seq: **19,587 single cells** with assigned
sgRNA identity. Growth phenotype (their `γ`) is "calculated using relative γ
measurements from the Perturb-seq cell pool **after 5 days of growth**" — i.e., computed
in the **same experiment, same window** as the transcriptome. This is **Analysis R**
(§5.1 of the governing spec), not Analysis P: Jost has no DepMap/Achilles reference
anywhere in the text (confirmed by direct grep of the full text — zero hits for
"DepMap" or "Achilles"). Jost's `γ` also carries the same identifiability limit as
Chronos: it is a single net population growth rate, not a division/death decomposition.

### (a) Is `strength → response magnitude → growth effect` a single monotonic axis?

**Partially, and only within a gene's own series.** Direct quote: *"Within each series,
two metrics of phenotype, bulk population growth phenotype and transcriptional
response, were well-correlated, despite substantial differences in the absolute
magnitudes of the transcriptional responses with different series (Fig. 6f, S10e–g)."*
No numeric r is given in the main text for this correlation (exact values would require
extracting supplementary-figure source data, not done in this pass — flagged in
§UNVERIFIED). But the qualitative claim is explicit and it is *for the response↔growth
relationship*, not for the knockdown-fraction↔phenotype relationship, which the paper
explicitly separates and treats very differently:

*"By contrast, the relationships between either metric of phenotype and target gene
expression were strongly gene-specific (Fig. 6g, Fig. S10h–j). For HSPA5 and GATA1, for
example, a reduction in mRNA levels by ~50% was sufficient to induce a near-maximal
transcriptional response and growth defect, whereas for most other genes a larger
reduction was required."*

So: growth tracks transcriptional-response *magnitude* reasonably well within a gene's
own titration series (support for a within-gene burden axis) — but the *mapping from
knockdown fraction to that magnitude* is sigmoidal and gene-specific, not one shared
axis across genes. The abstract's own headline finding is explicitly non-linear:
*"Staging cells along a continuum of gene expression levels combined with single-cell
RNA-seq readout revealed sharp transitions in cellular behaviors at gene-specific
expression thresholds."*

### (b) Does the full transcriptomic state carry information beyond a magnitude scalar?

**Yes, on four separate lines of evidence, all quoted directly from the results:**

1. **Variance itself is state-dependent, and is not accounted for by the mean.**
   *"[T]he magnitude of this transcriptional signature increased with increasing sgRNA
   activity on both the population (Fig. 6d) and the single-cell level (Fig. 6e),
   although populations with intermediate-activity sgRNAs had larger cell-to-cell
   variation in response magnitude. Similarly, the transcriptional responses to
   knockdown of other genes scaled with sgRNA activity and exhibited larger variance for
   intermediate-activity sgRNAs."* A single "expected magnitude" scalar is least
   informative exactly in the partial-knockdown regime most relevant to real genetic LOF.

2. **Qualitatively distinct pathway identities, not just distance-along-one-line.**
   Clustering mean per-sgRNA transcriptional profiles and UMAP projection of single
   cells "recapitulated the clustering" into a ribosomal-protein/POLR1D cluster and a
   separate ISR-activating cluster (HSPA9, HSPE1, EIF2S1). *Within* an individual gene's
   series cells do move outward along one direction with increasing knockdown ("Within
   individual series, cells projected further outward in UMAP space with increasing
   sgRNA activity") — but *across* genes, different perturbations occupy qualitatively
   different regions of transcriptome space at comparable severity, i.e., "how strong"
   does not determine "which state."

3. **Genuine single-cell bimodality at matched nominal perturbation strength.**
   The ATP5E case is the sharpest evidence found in this search: *"The frequency of ISR
   activation increased with lower ATP5E mRNA levels, but even at the lowest levels some
   cells did not exhibit ISR activation."* At a fixed knockdown level, individual cells
   split into an ISR-on vs. ISR-off population — a discrete branching outcome that a
   knockdown-strength or response-magnitude scalar cannot resolve, because it is
   genuinely bimodal, not a continuum around a mean.

4. **Sharp, gene-specific thresholds rather than smooth proportional scaling**
   (quoted above) — the growth/death transition is sigmoidal and its inflection point
   varies by an order of magnitude across genes, so "knockdown fraction" is not
   interchangeable with "growth cost" across genes without gene-specific calibration.

### Verdict

**Mixed, and it cuts in a specific, informative direction.** Jost 2020 supports a
*within-gene* burden axis (transcriptional-response magnitude tracks that gene's own
growth phenotype reasonably well) — so response burden is real, substantial signal, not
a strawman. But it simultaneously and explicitly documents that the full transcriptomic
state is **not reducible to that magnitude scalar**: variance is itself
knockdown-strength-dependent and largest exactly at intermediate (partial-LOF) doses;
different genes produce qualitatively distinct pathway-level signatures at matched
severity; and single cells at the *same* nominal perturbation strength diverge into
discrete ISR-on/off subpopulations. This is the single most direct, purpose-adjacent
empirical test of the specificity hypothesis (§5.2) found in this literature — and it
**leaves the hypothesis alive, with real supporting evidence**, rather than settling it
in either direction. Two hard caveats limit how far this can be pushed: (i) it is
Analysis R (same-window `γ`, no independent anchor) — it cannot itself certify
"beyond an *independent* net-fitness anchor"; and (ii) `γ` is a single net rate, so Jost
is silent on division/death/recovery decomposition — it never touches the Decision
Memo §1 "real question" at all, only the specificity question of §5.2.

---

## 2. Expanded matrix rows

Same column schema as `gate3_prior_art_matrix.md`.

| # | Paper (cite + ID/URL) | Year/venue | Unit | Early state? | Outcome anchor | Evidence tier | Task (T1/T2/T3) | Beat net-fitness/burden baseline? | What it did NOT do |
|---|---|---|---|---|---|---|---|---|---|
| 22 | Jost M, Santos DA, Saunders RA, et al. "Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs." DOI `10.1038/s41587-019-0387-5`, PMID `31932729`, PMCID `PMC7065968`. https://pmc.ncbi.nlm.nih.gov/articles/PMC7065968/ | 2020, Nat Biotechnol | Perturbation-condition / dose-series (K562 CRISPRi, per-gene allelic series) | **Graded**, not "early" — a single day-5 population per dose level, not a time course | Population growth phenotype (`γ`) from the **same** Perturb-seq pool, **same window** as the transcriptome — Analysis R, no independent anchor | **3** (cross-sectional condition comparison; within-series it approaches a partial dose-response, still not the paired-prospective tier-1/2 design) | None of T1/T2/T3 cleanly — a dose-response mapping, not a fate prediction | **Partially, and specifically documented**: response magnitude tracks growth within a series (burden real), but bimodal single-cell ISR on/off at matched knockdown, gene-specific sigmoidal thresholds, and cross-gene qualitative clustering are evidence the state is not reducible to magnitude alone | No independent (DepMap-style) fitness anchor; no division/death/recovery decomposition (`γ` is net, like Chronos); no lineage/clone tracking; not "early" relative to fate commitment, only relative to dose |
| 23 | Dixit A, Parnas O, Li B, et al. "Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens." DOI `10.1016/j.cell.2016.11.038`, PMID `27984732`, PMCID `PMC5181115`. https://pmc.ncbi.nlm.nih.gov/articles/PMC5181115/ | 2016, Cell | Perturbation-condition (K562 Perturb-seq, 13 cell-cycle regulators + TFs) | Single snapshot; concurrent with fitness readout, not early-vs-late | sgRNA abundance fold-change (fitness), from the **same** pooled screen — Analysis R, no independent anchor | **3** (cross-sectional condition comparison) | **Explicit qualitative divergence at matched/same-direction fitness**: quoted section header *"Perturbations of cell cycle regulators reveal distinct profiles associated with similar fitness effects and mitotic arrest."* CABP7 and CIT both **increase** fitness but via opposite cell-cycle-phase signature shifts (CABP7: ↓G2/M-M, ↑M/G1, plus mitochondrial-respiration/NFkB/mitotic-division program; CIT: ↑G1/S-S via a distinct histone-gene program). CABP7 has "a distinct transcriptional phenotype" despite a morphology matching CIT/PTGER2/RACGAP1's binuclear phenotype. | Never quantifies this beyond n=13 genes/33 guides; no DepMap-style independent anchor; no division/death decomposition; no burden-scalar residualization test; not framed as a fate/trajectory question at all |
| 24 | Replogle JM, Saunders RA, Pogson AN, et al. "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq." DOI `10.1016/j.cell.2022.05.013`, PMID `35688146`, PMCID `PMC9380471`. https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/ | 2022, Cell | Perturbation-condition (K562, genome-scale CRISPRi, day 6–8) | **Single, late** snapshot (day 6–8 post-transduction) — explicitly a survivor population | Growth phenotype (`γ`, log2 guide enrichment/doubling, day 6→16) from the **same** pooled screen — Analysis R | **3** (cross-sectional condition comparison) | **Quantified**: *"The strength of transcriptional response was correlated with the growth phenotype (Spearman's ρ = −0.51) with 86.6% of essential genetic perturbations leading to a significant transcriptional response."* Also: *"a substantial number of genetic perturbations that cause a transcriptional phenotype have a negligible growth phenotype (n = 771 [of 9,608]; ... indicating that many genetic perturbations influence cell state but not growth or survival."* | ρ=−0.51 leaves ~74% of variance (ρ²≈0.26) unexplained by response magnitude alone — real but partial burden signal; still no independent (DepMap) anchor in this correlation; no division/death decomposition; DepMap is referenced only for **gene selection** ("20Q1 Cancer Dependency Map common essential genes"), never as a modeled/correlated fitness target |
| 25 | Nadal-Ribelles M, Lieb G, Solé C, et al. "Transcriptional heterogeneity shapes stress-adaptive responses in yeast." DOI `10.1038/s41467-025-57911-6`, PMID `40097446`, PMCID `PMC11914649`. https://pmc.ncbi.nlm.nih.gov/articles/PMC11914649/ | 2025, Nat Commun | Population/genotype (yeast, YKOC deletion mutants) | Cross-sectional per mutant, at peak of osmostress response; not time-resolved relative to a later fate | Competitive/endpoint fitness (FACS-based competition assay, 48 h; or growth-ratio pre-stressed vs. naive) — **independent of** the scRNA-seq assay itself, but same-study, not an external database anchor | **3** (cross-sectional condition comparison), but with an explicit **distributional (variance), not mean-only, predictor** — closer to Candidate B's spirit than any other row in either matrix | **Yes, on the distributional axis specifically**: scores standard deviation of the induced-signature response per mutant as a variability metric distinct from its mean; "hyper-responsive" cells (top 2% of an unstable reporter) show "greater competitive fitness... under stress conditions" but lower fitness under normal conditions; 8/13 of the most-*variable* mutants show faster stress adaptation than WT | **Yeast, not mammalian; no CRISPR/DepMap anchor at all; not a genetic loss-of-function fitness readout in the DepMap sense (fitness = adaptation speed under an imposed osmotic challenge, closer to a drug/stress-response paradigm than baseline growth); no matched-mean/controlled-burden design (mutants are selected by variability score, not matched on mean induction level first); no trajectory (division/death/recovery) decomposition at all.** **This paper corrects the Decision Memo's own "3,500 deletions" figure** — full text states the transcription-focused analysis profiled **325 mutants** from the Yeast Knockout Collection (YKOC), not 3,500 (searched exhaustively; no "3,500" string appears anywhere in the full text). |
| 26 | Neumann B, Walter T, Hériché JK, et al. "Phenotypic profiling of the human genome by time-lapse microscopy reveals cell division genes." DOI `10.1038/nature08869`, PMID `20360735`, PMCID `PMC3108885`. + Pau G, Walter A, Neumann B, et al. "Dynamical modelling of phenotypes in a genome-wide RNAi live-cell imaging assay." DOI `10.1186/1471-2105-14-308`, PMID `24131777`, PMCID `PMC3827932` (Mitocheck). https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/, https://pmc.ncbi.nlm.nih.gov/articles/PMC3827932/ | 2010/2013, Nature / BMC Bioinformatics | Cell/lineage (HeLa-H2B-GFP, genome-scale RNAi, continuous 48 h live imaging) | **Yes**, state-at-seeding → continuous imaging is Tier-1-shaped by design | Reproducibly-timed penetrance/timing of quiescence, mitotic arrest, polynucleation, death per gene — an **internal**, not independently pre-existing, dynamics model | **1 by design**, but **not matched to an independent net-fitness scalar** (per Gate 2b/Decision Memo §0 correction) | **T1 per-frame, aggregated to a T2-flavored per-gene dynamics model** | N/A — this is an existence-proof of category-level, qualitatively distinct dynamics classes (~4% of ~51,766 constructs), not a burden-vs-fitness residualization test | RNAi (siRNA), HeLa, not CRISPR/K562/cancer-dependency; **wrong denominator for "common"** — ~4% is fraction of *all reagents*, not of *net-fitness-matched pairs* (Decision Memo §0's own correction, carried forward here); mitotic-arrest/death timing correlate at r=0.80 *within* the ~36 siRNAs already showing both, which is conditional, not general, coupling evidence |
| 27 | Panagopoulos A, et al. "Multigenerational cell tracking of DNA replication and heritable DNA damage." DOI `10.1038/s41586-025-08986-0`, PMID `40399682`, PMCID `PMC12176655`. https://pmc.ncbi.nlm.nih.gov/articles/PMC12176655/ | 2025, Nature | Lineage (U2OS/RPE-1, endogenously tagged 53BP1/PCNA, ≤4-generation pedigrees) | **Yes** — G1 DNA-damage/replication state, tracked forward across generations | None — no independent fitness/dependency anchor at all; outcome is a qualitative fate class (polyploidization route, genome-integrity outcome), not a growth/dependency score | **A1** (Candidate A hierarchy) | **T2** | Not applicable — no burden-scalar comparison attempted | **siRNA (TP53/CDKN1A/AMBRA1) and overexpression (HRAS/cyclin-E1), not CRISPR knockout**; dual-CRISPR editing used only for the endogenous fluorescent knock-in reporters, not as the perturbation; no effect-size/R²/founder-fraction statistic reported (qualitative correlation only, per Gate 2a); n≈20 lineages/≤80 granddaughters (U2OS), ≤10 lineages (RPE-1) — underpowered; no fitness anchor of any kind |
| 28 | "Tracing cellular heterogeneity in pooled genetic screens via multi-level barcoding." DOI `10.1186/s12864-019-5480-0`, PMID `30727954`, PMCID `PMC6364396`. https://pmc.ncbi.nlm.nih.gov/articles/PMC6364396/ | 2019, BMC Genomics | Sub-clone (Jurkat, two-level barcode: clone ID + sub-clonal barcode, genuine CRISPR knockout/CRISPRi of TRAIL-apoptosis genes) | **No** — no molecular state readout of any kind, only barcode abundance | Barcode fold-change trajectory (days 0/4/9/14) under TRAIL-receptor-antibody challenge — an internal trajectory, not an independent fitness anchor | Clone-outcome-only, closest to **A3-adjacent "outcome divergence"** — no `S_early` variable exists to rank | **T2** on the abundance trajectory; **no T1/state variable to compare against** | N/A — nothing to test against, since no state is measured | **Silent on the state-association half of the question entirely** — its own abstract's contribution is that within-perturbation (identical sgRNA) sub-clonal divergence is real and reproducible (Wilcoxon p<10⁻¹²), which is the cleanest genuine-CRISPR existence-proof of within-perturbation divergence found in either matrix, but it cannot speak to whether an early state predicts it; outcome is a drug-like TRAIL-antibody death challenge layered on the KO, not baseline genetic-LOF growth |

---

## 3. Which novelty claims are now CLOSED, and what remains open

Closing a claim is more valuable than protecting one; several genuinely close here.

### CLOSED

1. **"No paper has ever combined graded genetic perturbation strength, Perturb-seq, and
   a growth phenotype in one K562 system."** **CLOSED** by Jost 2020 — it does exactly
   this (4 of the original 5 "never co-occur" attributes: transcriptome + CRISPR(i) +
   K562 + growth phenotype, missing only the DepMap-specific dependency framing). This
   further confirms — a second time, independently of the reviewer's original
   correction — that the v1 "attribute-stacking" framing of novelty was too weak a bar;
   a near-complete stack already exists in a 2020 paper the first Gate 3 pass missed
   entirely.

2. **"No quantitative transcriptome-response-vs-growth-phenotype correlation exists for
   K562 genetic perturbation."** **CLOSED** by Replogle 2022: Spearman ρ = −0.51,
   genome-wide, with an explicit dissociated subset (771/9,608 perturbations with
   significant transcriptional phenotype but negligible growth effect). This number now
   exists and is citable; it should replace any residual "unquantified" framing of the
   burden-vs-fitness relationship in K562.

3. **"Distinct transcriptional profiles under matched/same-direction fitness effects
   have never been shown for genetic (not drug) perturbation."** **CLOSED** by Dixit
   2016's cell-cycle-regulator panel: two perturbations with the *same sign* of fitness
   effect (CABP7, CIT — both increase fitness) act through materially different cell-cycle
   and downstream transcriptional programs. This is a genuine, if small (n=13 genes),
   existence proof of the wedge's qualitative version in a genetic system.

4. **"Population-state distribution (heterogeneity/variance, not mean) has never been
   linked to fitness in any genetic-perturbation system."** **CLOSED, with a modality
   caveat**, by Nadal-Ribelles 2025: variability of the stress-response signature,
   scored explicitly as a distinct axis from its mean, associates with adaptive fitness
   in a 325-mutant yeast deletion panel. Candidate B's *distributional* framing is
   anticipated — in yeast, under an imposed stress challenge, not baseline genetic LOF
   growth in a mammalian/cancer line. This closes the "such a linkage has never been
   attempted" claim; it does **not** close the mammalian/cancer/CRISPR/DepMap-anchored
   version of the same claim.

5. **"Whether the pseudobulk transcriptome carries information beyond a burden scalar
   for predicting an independent fitness anchor is untested."** **CLOSED — and already
   answered — by this project's own exp02** (`docs/experiment/02_replogle_k562_viability_axis_audit.md`,
   `docs/experiment/model-card/02_replogle_k562_viability_axis_audit.md`), which residualizes
   Replogle K562 pseudobulk transcriptome against NAR viability scores and response-burden
   summaries before predicting DepMap K562 GeneEffect: NAR alone reaches Spearman 0.244,
   NAR+burden reaches 0.443, the full pseudobulk PCA-RandomForest comparator reaches
   0.494, and — critically — residualizing OUT *both* NAR and burden still leaves
   Spearman 0.469, "below the baseline but still meaningful." This is a real,
   already-computed, DepMap-anchored (Analysis P by the spec's own §5.1 definition)
   test of the specificity hypothesis, and it already comes out supportive.

### OPEN — none of the seven new papers, nor exp02, closes these

1. **The Decision Memo §1 "real question" itself** (does net fitness conceal
   reproducible division/death/recovery dynamics in genetic LOF) is untouched by every
   paper in this rerun. Jost's `γ`, Dixit's sgRNA fold-change, Replogle's `γ`,
   Nadal-Ribelles's competition fitness, and DepMap's Chronos GeneEffect are all single
   net rates. None separates division suppression from cell loss from recovery. A
   reanalysis of any of these datasets, however sophisticated, cannot manufacture a
   trajectory label the underlying assay never captured.

2. **An independent (Analysis P), matched-modality anchor with a genuine trajectory
   outcome, in a genetic-perturbation cancer-line system, does not exist anywhere in
   this search.** exp02 gets to Analysis P (DepMap is pre-existing and independent of
   the Perturb-seq run) but is modality-mismatched (CRISPRi transcriptome vs. Cas9-KO
   fitness, per Decision Memo §5) and predicts a scalar, not a trajectory.

3. **Genuinely early state relative to fate commitment** remains unresolved: Jost is a
   single day-5 snapshot per dose; Dixit and Replogle are single late snapshots
   (day 7–8); Nadal-Ribelles is a peak-of-response snapshot. None is a time course
   capable of separating "early, predictive" state from "late, consequence-of-fate"
   state — the same live Gate-1 constraint (§4.4 of the spec) that Replogle 2022 already
   carried in the original matrix.

4. **Mitocheck's category-existence finding cannot be upgraded to a matched-fitness
   prevalence number** (Decision Memo §0's denominator trap, reconfirmed here) —
   it remains a "raises concern, decides nothing" entry, now with two more genetic
   systems (Dixit, Jost) that are qualitatively consistent with distinct-routes-same-outcome
   but are far too small (n=13, n=25 genes) to move a prevalence estimate.

5. **Panagopoulos and multi-level-barcoding each supply one side of Candidate A's
   requirement and are missing the other** — Panagopoulos has the lineage-level state
   + fate design but no fitness anchor, no CRISPR modality, and no statistics;
   multi-level-barcoding has genuine CRISPR-KO and reproducible within-perturbation
   divergence but no state measurement at all. Neither closes Candidate A; together
   they sharpen exactly what a real Candidate-A design still needs (both halves, in one
   system).

---

## 4. Revised novelty statement

In a cancer cell line under genetic (CRISPR/CRISPRi) loss-of-function, no published
study — and no reanalysis run so far, including this project's own exp02 — measures an
early single-cell transcriptomic state, an independently pre-existing net-fitness
anchor, and a subsequent multi-state division/persistence/recovery/extinction
trajectory, and asks whether the state carries information about that trajectory beyond
a response-burden scalar and the fitness anchor. The nearest approaches close the
*specificity* half of this claim more than Gate 3's first pass credited: Replogle 2022
shows a real but partial genome-wide burden-vs-growth correlation (ρ=−0.51, with 771/9,608
perturbations dissociating the two entirely); Jost 2020 shows response magnitude tracks
growth within a gene's own titration series while simultaneously documenting
non-monotonic, gene-specific thresholds and genuine single-cell bimodality at matched
knockdown; Dixit 2016 shows two same-direction-fitness perturbations achieving it through
materially different transcriptional routes; and this project's own exp02 already shows
that residualizing a K562 pseudobulk transcriptome against both a generic viability
signature and a response-burden scalar leaves Spearman 0.469 against an independent
(DepMap) fitness anchor — barely below the unresidualized 0.494 ceiling. Taken together,
"does population-level transcriptomic state carry information beyond burden for
predicting net fitness" is no longer an open question in K562-adjacent systems — it is
answered, affirmatively, modestly. What remains genuinely untouched by every paper and
every reanalysis found in this rerun is the trajectory-decomposition question at the
center of the program: every fitness/growth readout used anywhere in this literature —
Jost's γ, Dixit's sgRNA fold-change, Replogle's γ, Nadal-Ribelles's competition fitness,
DepMap's Chronos GeneEffect — is a single net rate, and none of the datasets offers more
than one or two snapshots of transcriptomic state. The defensible novelty surface is
therefore narrower and more specific than either v1's attribute-stacking claim or Gate
3's first-pass "specific conjunction" framing: it is not that the ingredients have never
co-occurred (Jost comes close to that), but that no one has used any of these near-miss
systems to ask whether transcriptomic state predicts *how* a given net fitness was
reached — division suppressed, cells lost, or both partially and recovered — because no
existing dataset (public or internal) captures that decomposition at all.

---

## 5. Expected-information-value comparison: retrospective reanalysis vs. new Study 0

### What is already, literally, done

A reanalysis correlating K562 Perturb-seq transcriptomic features against an
independent DepMap/Chronos anchor is **not hypothetical** — it is this project's own
exp02 (`docs/experiment/02_replogle_k562_viability_axis_audit.md`), already run, already
costed at analyst-time, already showing the specificity-supportive numbers quoted
above. This changes the framing of Priority 4 from "should we try a reanalysis" to
"how far can the reanalysis class be pushed, and does it change what Study 0 should
measure."

### Cheap extensions of the same reanalysis class, not yet done

1. Apply exp02's NAR/burden-residualization audit to **Jost 2020's own titration
   series** (public supplementary tables) — ask whether burden-residualized
   transcriptome still predicts a gene's own growth phenotype *within its dose series*,
   a cleaner within-gene test than exp02's cross-gene one.
2. Apply the same audit to **Dixit 2016's 13-gene cell-cycle panel** — does a
   residualized-transcriptome model separate CABP7 (distinct route) from CIT/PTGER2
   (shared route) better than a burden-only model, using the paper's own supplementary
   data.
3. Generalize **Nadal-Ribelles's mean-vs-variance test** to Replogle 2022's raw
   single-cell (not pseudobulk) K562 data, which already contains per-cell resolution
   per perturbation — this would be the first "distribution beyond mean" test on
   mammalian genetic LOF data, directly on already-public data, without new wet-lab work.

### What this reanalysis class categorically cannot do

1. **Survivorship selection.** Every dataset in this rerun (Jost day 5, Dixit day 7/14,
   Replogle day 6–8) profiles cells that already survived to that timepoint. Any cell
   whose transcriptomic state predicted early death is structurally absent from the
   observed distribution — this is not a fixable analysis problem, it is a property of
   the assay.
2. **Late, single-snapshot design.** None of these datasets is a time course. A
   reanalysis can correlate one snapshot's state with one scalar fitness number; it
   cannot construct, validate, or falsify a trajectory estimand, because no trajectory
   was ever measured.
3. **No decomposition to serve as a label.** Every fitness/growth readout available for
   reanalysis — Jost's γ, Dixit's fold-change, Replogle's γ, DepMap's Chronos
   GeneEffect — is a single net rate. However cleverly residualized, a reanalysis cannot
   manufacture a division/death/recovery label the underlying screen never captured.
   This means the reanalysis class can address **Gate 3/specificity** ("does state beat
   a burden scalar for a scalar target") but categorically **cannot** address **Gate 2's
   phenomenon question** ("does matched-fitness trajectory divergence exist") — that
   requires primary trajectory data (imaging, repeated counts, or a mechanistic
   division/death-separating assay), which is exactly what Study 0 is designed to probe
   feasibility for.
4. **Same-window confound persists for every own-experiment γ.** Jost's, Dixit's, and
   Replogle's own growth readouts are Analysis R (concurrent with the transcriptome);
   only the exp02-style comparison against the fully independent, pre-existing DepMap
   anchor reaches Analysis P — and even that comparison is CRISPRi-vs-Cas9-KO
   modality-mismatched (Decision Memo §5), so it tests "beyond an external reference,"
   not "matched on the true net fitness of the same intervention."

### Recommendation: reanalysis extension first, Study 0 second — with a stated reason, not a default

Run the three reanalysis extensions above **before** committing Study 0's wet-lab
budget, for a concrete reason: they cost analyst-days against already-public,
already-downloaded data, versus weeks-to-months and real reagent/imaging cost for Study
0. They directly and cheaply advance exactly the question that currently gates whether
building expensive trajectory infrastructure is worth it — the specificity hypothesis —
and exp02 already shows a promising signal there. But the reanalysis is not a substitute
for Study 0 and must not be read as one: it cannot touch the phenomenon-prevalence
question or the trajectory-decomposition question at all, by construction (points 1–3
above). Its role is strictly sequencing, not replacement:

- A robust "beyond burden, beyond an independent anchor" result across Jost + Dixit +
  Replogle raises the expected value of paying for Study 0's trajectory-capable
  infrastructure — it says the transcriptome-side of the wager is likely to pay off if
  a trajectory ever gets measured.
- A null result (transcriptome reduces to burden once properly residualized across
  multiple independent datasets) would justify narrowing or deprioritizing the
  specificity claim before any wet-lab spend, at zero marginal cost beyond analyst time
  already available.

Either way, Study 0's own claim ceiling (DECISION_MEMO.md §7) is unaffected: it still
cannot select between candidates, still cannot reach R2, and still cannot license a Gate
2 positive. The reanalysis extension is a cheap, decision-enabling prerequisite that the
spec's §9 table already permits ("Predefined small-scale descriptive reanalysis") — it
should be scheduled and reported before Study 0's protocol is finalized, not in parallel
with no ordering, so that its result can actually inform Study 0's scope and budget ask.

---

## UNVERIFIED

- **Jost 2020 Fig. 6f/6g numeric correlation values.** The main text states growth
  phenotype and transcriptional-response magnitude were "well-correlated" within each
  series but reports no numeric r in the body text; extracting the exact value would
  require the figure's underlying source data (not retrieved in this pass). Treat the
  within-series burden-axis claim as qualitatively, not quantitatively, verified.
- **The Decision Memo's "3,500 yeast deletions" figure for Nadal-Ribelles 2025 appears
  to be incorrect.** The retrieved full text states the transcription-focused Perturb-seq
  analysis profiled **325 mutants** from the Yeast Knockout Collection; no "3,500"
  string appears anywhere in the full text (searched exhaustively). This should be
  corrected wherever the Decision Memo or downstream framing repeats the "3,500" figure.
- **Public accessibility of Replogle 2022 raw single-cell data and Jost 2020
  supplementary tables was assumed, not click-through-confirmed.** Both papers state
  data-availability commitments (GEO/Zenodo-style deposits per their methods), but this
  run did not verify that the specific files needed for the three proposed reanalysis
  extensions (§5) are currently downloadable at a working URL. This should be confirmed
  before scheduling analyst time against them.
- **exp02's numbers are taken from this repository's own internal documentation**
  (`docs/experiment/02_replogle_k562_viability_axis_audit.md` and its model card), not
  independently re-run or code-audited in this pass. They are treated as authoritative
  project state per this task's framing, but they are an internal record, not an
  externally peer-reviewed source, and carry that caveat.
- **Jost 2020's Fig. 3e–g "heterogeneity... at the level of growth phenotype" for
  SNRPD2 mismatched-sgRNA variants** was noticed in passing (large-scale screen section)
  but not deep-read — it is a different heterogeneity claim (across-guide, not
  across-cell) and was out of scope for this rerun's four priorities. Flagged as a
  possible further lead, not evaluated.
- **Dixit 2016's broader TF panel (14 K562 TFs) and BMDC LPS results** were not
  deep-read for this rerun beyond the cell-cycle-regulator section directly relevant to
  Priority 2; only the cell-cycle-regulator finding was verified in depth.

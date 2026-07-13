# Gate 3 — Nearest Prior Art and Exact Novelty

Scope: cell-fate-outcome-dynamics program. Question tested against every row: does an
early post-perturbation single-cell molecular state carry incremental prospective
information about subsequent division/persistence/recovery/extinction, **beyond an
independently measured net fitness**, and **not reducible to a scalar
response-burden/injury score**?

Method note: every row below was retrieved via WebSearch/WebFetch/Europe PMC/bioRxiv
API tool calls made in this run (listed with DOI/PMID/PMCID and a retrieved URL). Rows
marked "abstract-only" mean I read a search snippet or fetched summary but did not
retrieve full-text methods; treat their tier/task assignment as provisional. Nothing
below is cited from memory.

## The nearest-prior-work matrix

| # | Paper (cite + ID/URL) | Year/venue | Unit | Early state? | Outcome anchor | Evidence tier | Task (T1/T2/T3) | Beat net-fitness/burden baseline? | What it did NOT do |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Iyer A, Alva A, Granada AE, Chakrabarti S. "Inheritable cell-states shape drug-persister correlations and population dynamics in cancer cells." DOI `10.1371/journal.pcbi.1013446`. https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013446 | 2025, PLOS Comput Biol | Cell/lineage (live-imaging-tracked, HCT116 & U2OS) | **Yes** — pre-treatment intermitotic time (IMT), measured non-destructively before cisplatin exposure | Survival vs. death of the **same tracked lineage**, observed by continued live imaging | **A1** (Candidate A) — same-cell/lineage prospective, non-destructive | **T2**, genuinely prospective | **Yes, explicitly** — shows "primed for death/survival" ancestor states have indistinguishable pre-treatment cell-cycle kinetics (no fitness cost), i.e. state carries info a growth-rate scalar does not | State = cell-cycle timing only, not a genome-wide/transcriptomic molecular state. Perturbation = cisplatin (drug), not genetic/CRISPR. No independent pre-existing net-fitness screen (F_net is internal/matched — an Analysis-R design, not Analysis-P). Outcome is binary survive/die, not the full division/persistence/recovery/extinction multi-state estimand. No cancer-dependency (DepMap/Chronos) anchor at all. |
| 2 | Weinreb C, Rodriguez-Fraticelli A, Camargo FD, Klein AM. "Lineage tracing on transcriptional landscapes links state to fate during differentiation." DOI `10.1126/science.aaw3381`, PMID `31974159`, PMCID `PMC7608074`. https://www.science.org/doi/10.1126/science.aaw3381 | 2020, Science | Clone (LARRY barcode) | **Yes** — scRNA-seq of one time point per clone (via sibling) | Later differentiation-fate identity of siblings/descendants | **A2** (Candidate A) — sibling/clone proxy | **T2**, formally prospective, with the field's most careful quantification of an accuracy ceiling (~50–60%) | N/A — **no net-fitness anchor exists in this system at all**; the "beyond fitness" test is never posed because there is no fitness label, only differentiation-type choice | Genuinely transcriptome-wide molecular state, but: unperturbed normal hematopoiesis, not CRISPR/genetic perturbation; outcome is differentiation-type classification, not division/death/dependency; never touches a DepMap-style anchor; explicitly reports the residual (~40–50%) is NOT explained by top predictive genes, i.e. documents rather than closes the information gap. |
| 3 | Weinreb C, Wolock S, Tusi BK, Socolovsky M, Klein AM. "Fundamental limits on dynamic inference from single-cell snapshots." DOI `10.1073/pnas.1714723115`, PMID `29463712`, PMCID `PMC5878004`. https://pmc.ncbi.nlm.nih.gov/articles/PMC5878004/ | 2018, PNAS | Population/clone (methodological) | N/A — theory/identifiability paper (population balance analysis) | None directly; defines what is identifiable from destructive snapshots | Methodological — bounds all rows below that use snapshot-only trajectory inference | None (T-none) | N/A | Provides no empirical fate result; used here to bound the honesty of OT/velocity-based "fate" claims (rows 8–9): snapshot-only inference cannot uniquely recover dynamics without added structural assumptions. |
| 4 | Shaffer SM, Dunagin MC, Torborg SR, et al. "Rare cell variability and drug-induced reprogramming as a mode of cancer drug resistance." DOI `10.1038/nature22794`, PMID `28607484`, PMCID `PMC5542814`. https://pmc.ncbi.nlm.nih.gov/articles/PMC5542814/ | 2017, Nature | Population (FACS-sorted marker-high subpopulation, melanoma) | **Yes** — EGFR protein marker level, prospectively FACS-sorted before vemurafenib | Colony formation count after 3 weeks of drug | **A3** (population/marker-average; not clone-barcode-verified in this paper) | **T2** prospective | Not framed against a burden scalar (predates that framing); shows 7.9-fold enrichment from a single marker | Single antibody marker, not transcriptome-wide; drug not genetic perturbation; not barcode-verified as clonal (see row 5, its own follow-up); no fitness-independence test. |
| 5 | Emert BL, Cote CJ, Torre EA, et al. "Variability within rare cell states enables multiple paths toward drug resistance." DOI `10.1038/s41587-021-00837-3`. https://pmc.ncbi.nlm.nih.gov/articles/PMC8277666/ | 2021, Nat Biotechnol | Clone (Rewind: barcode + RNA FISH) | **Yes** — drug-naive precursor transcriptional/signaling marker panel | Descendant clone's resistance fate under drug | **A2** — sibling/clone proxy, barcode-linked | **T2** prospective | No explicit burden-scalar residualization | Targeted RNA-FISH marker panel, not genome-wide scRNA-seq; drug (vemurafenib), not genetic/CRISPR; melanoma, not K562; no DepMap-style anchor; no explicit test of specificity vs. a generic-stress scalar. |
| 6 | Frick PL, Paudel BB, Tyson DR, Quaranta V. "Quantifying heterogeneity and dynamics of clonal fitness in response to perturbation." DOI `10.1002/jcp.24888`, PMID `25600161`, PMCID `PMC5580929`. https://pmc.ncbi.nlm.nih.gov/articles/PMC5580929/ | 2015, J Cell Physiol | Clone (single-cell-derived colony, PC9/A375) | **Yes** — EGFR immunofluorescence on the clone | Proliferation rate of that clone's derived subline | **A2/A3** | **T2** | Not tested against an independent net-fitness anchor | Single marker, not transcriptome; drug (cycloheximide/trametinib/PLX4720/ABT-737), not genetic; does not decompose division vs. loss as separate trajectory components; imaging-only, no molecular profiling. |
| 7 | Balaban NQ, Merrin J, Chait R, Kowalik L, Leibler S. "Bacterial persistence as a phenotypic switch." DOI `10.1126/science.1099390`, PMID `15308767`. https://www.science.org/doi/10.1126/science.1099390 | 2004, Science | Cell (E. coli, microfluidic time-lapse) | **Yes** — pre-existing growth-rate/dormancy phenotype, observed before antibiotic exposure | Survival through antibiotic treatment, same tracked cell | **A1** — true same-cell prospective, foundational existence proof | **T2**, same-cell prospective | Conceptually yes — persistence is explicitly not an aggregate-dropout phenomenon; this is the origin of "phenotypic-switch beyond population fitness" | No molecular/transcriptomic state at all (state = growth-rate phenotype only); prokaryotic, not mammalian/cancer; no genetic perturbation; binary survive/die only, no recovery/persistence gradation. |
| 8 | Schiebinger G, Shu J, Tabaka M, et al. "Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming" (Waddington-OT). DOI `10.1016/j.cell.2019.01.006`, PMID `30712874`, PMCID `PMC6402800`. https://pmc.ncbi.nlm.nih.gov/articles/PMC6402800/ (abstract/summary level) | 2019, Cell | Condition/population (repeated cross-sectional snapshots) | Cross-sectional per time point; no same-cell early state | None independently observed — "fate" is an OT-**inferred coupling** under assumptions (growth rates estimated partly from proliferation/apoptosis **signature scores**, not observed division/death) | **5 — signature-only/assumption-laden inference** (both hierarchies) | Claims T2 language ("trajectories," "fates") but performs assumption-laden reconstruction, not validated same-unit prospective observation | No — no burden-residualization; growth-rate prior is itself signature-derived | Does not observe any single cell's actual future (destructive scRNA-seq per timepoint); growth-rate proxy is exactly the kind of signature-only apoptosis/proliferation-marker inference the governing spec (§6) warns cannot be treated as ground truth; no genetic-perturbation/dependency application; no independent net-fitness anchor. |
| 9 | Lange M, Bergen V, Klein M, et al. "CellRank for directed single-cell fate mapping." DOI `10.1038/s41592-021-01346-6`, PMID `35027767`, PMCID `PMC8828480`. https://pmc.ncbi.nlm.nih.gov/articles/PMC8828480/ | 2022, Nat Methods | Cell (Markov chain over one/few snapshots + RNA velocity) | No — "initial states" are inferred from the same snapshot's transition structure | "Terminal states" and fate probabilities computed from present-state graph structure; validated in some applications against LARRY (row 2) ground truth | **4 — terminal-state classifier** by default; inherits A2 only when validated against barcode ground truth | **T1 presented as T2** in most applications — textbook example of the conflation the governing spec flags | No | Does not use lineage barcoding as ground truth by default; no genetic-perturbation/dependency application; "fate probability" language is not equivalent to a validated prospective claim outside the few benchmarked datasets. |
| 10 | Norman TM, Horlbeck MA, Replogle JM, et al. "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes." DOI `10.1126/science.aax4438`, PMID `31395745`, PMCID `PMC6746554`. https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/ | 2019, Science | Perturbation-condition (K562 Perturb-seq) | Single post-perturbation snapshot, not framed as "early" vs. later | Growth-based genetic-interaction (GI) map used only as an **external annotation** to select/interpret manifold structure — never as a predicted target | **3 — cross-sectional condition comparison** (Candidate B) | None of T1/T2/T3 cleanly — descriptive manifold geometry, growth is a label not a prediction target | **Not attempted** — growth and transcriptome are parallel readouts, never modeled as predictor→target | Does not predict growth/fitness from the transcriptome; does not test single-cell distribution as informative about future dynamics; exactly the comparison this program wants to run, left undone. |
| 11 | Replogle JM, Saunders RA, Pogson AN, et al. "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq." DOI `10.1016/j.cell.2022.05.013`, PMID `35688146`, PMCID `PMC9380471`. https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/ | 2022, Cell | Perturbation-condition (K562, genome-scale + essential-gene CRISPRi) | **Single, late** snapshot (day 6–8 post-transduction) — explicitly not an early-state design relative to fate commitment | Growth phenotype (gamma, log2 guide enrichment/doubling) from a **separate bulk dropout screen**, reported alongside, not modeled from the transcriptome | **3 — cross-sectional condition comparison** | None attempted | **Not attempted** | Does not model transcriptome→growth; even the richest existing K562 Perturb-seq dataset supplies only a single late timepoint, so it cannot by itself demonstrate a genuinely "early" molecular state relative to fate commitment (a live Gate-1 constraint, not resolved by this paper). |
| 12 | Rosenski J, Shifman S, Kaplan T. "Predicting gene knockout effects from expression data." DOI `10.1186/s12920-023-01446-6`, PMID `36803845`, PMCID `PMC9938619`. https://pmc.ncbi.nlm.nih.gov/articles/PMC9938619/ | 2023, BMC Med Genomics | Cell line (cross-sectional, baseline expression) | No — **baseline** expression of "modifier genes," not post-perturbation, not temporal | DepMap essentiality of a **different** target gene, same cell line | N/A to A/B hierarchies (different question: cross-cell-line context-dependence of a static label, not post-perturbation trajectory) | None (static association) | Reports accurate prediction for ~3,000/~18,000 genes using ~10 modifier genes; no burden-scalar comparison in this program's sense | Does not use post-perturbation transcriptome at all; no temporal/trajectory dimension whatsoever; establishes that even the simpler baseline-expression→essentiality task is only partially solved, underscoring the difficulty of the harder task this program proposes. |
| 13 | Chiu YC, Zheng S, Wang LJ, et al. "Predicting and characterizing a cancer dependency map of tumors with deep learning" (DeepDEP). DOI `10.1126/sciadv.abh1275`, PMCID `PMC8378822`. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8378822/ | 2021, Science Advances | Cell line/tumor (baseline multi-omics) | No — baseline, not post-perturbation | DepMap dependency score | N/A | None | Claimed deep-learning gains over conventional ML | Same limitations as row 12; its own reported advantage over simpler baselines was later shown not to hold (row 14). |
| 14 | Chang D, Zhang X, Myers C. "Ridge regression baseline model outperforms deep learning method for cancer genetic dependency prediction." DOI `10.1101/2023.11.29.569083` (bioRxiv preprint). https://www.biorxiv.org/content/10.1101/2023.11.29.569083v1 | 2023, bioRxiv (preprint) | Cell line (same task as DeepDEP) | No | DepMap dependency score | N/A | None | **This IS the finding**: ridge regression (ρ=0.88) matches/beats DeepDEP (ρ=0.87) once the prediction problem is correctly framed | Does not touch post-perturbation dynamics; a direct precedent for "added model complexity doesn't beat a linear/scalar baseline" in exactly the transcriptome→dependency space, mirroring this program's own exp02 finding (burden scalar 0.426 vs. full model 0.494) and row 18 below. |
| 15 | Ahlmann-Eltze C, Huber W, Anders S. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." DOI `10.1038/s41592-025-02772-6`, PMID `40759747`, PMCID `PMC12328236`. https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/ | 2025, Nat Methods | Perturbation-condition (predicting the post-perturbation **transcriptome itself**) | N/A — this predicts the molecular state, not a fate derived from it | None — expression-to-expression benchmark, no phenotype/fate link | N/A | N/A (upstream of the wedge entirely) | **Explicit negative result**: none of 5 foundation models + 2 other DL models beat a mean/linear baseline | Does not touch fate/dependency prediction at all; establishes that the upstream "generate B" step (predicting post-perturbation transcriptome) that any generate-B/virtual-cell strategy would need is itself not solved better than trivial baselines — directly relevant to the spec's §7 "generate B" hypothesis review. |
| 16 | Roohani Y, Huang K, Leskovec J. "Predicting transcriptional outcomes of novel multigene perturbations with GEARS." DOI `10.1038/s41587-023-01905-6`, PMID `37592036`, PMCID `PMC11180609`. https://pmc.ncbi.nlm.nih.gov/articles/PMC11180609/ | 2024, Nat Biotechnol | Perturbation-condition | N/A | None — predicts post-perturbation transcriptome profile only | N/A | N/A | Not evaluated against a fitness/dependency baseline | Never predicts viability/division/persistence/extinction or a DepMap-style label from the perturbed profile; stops at the expression level. |
| 17 | Lotfollahi M, Klimovskaia Susmelj A, De Donno C, et al. "Predicting cellular responses to complex perturbations in high-throughput screens" (CPA). DOI `10.15252/msb.202211517`, PMID `37154091`, PMCID `PMC10258562`. https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/ | 2023, Mol Syst Biol | Perturbation-condition | N/A | None — predicts transcriptomic profile across dose/cell type/drug | N/A | N/A | Not attempted | Same gap as row 16: no phenotypic/fitness/dependency outcome anywhere in the evaluation. |
| 18 | Adduri AK, et al. (Arc Institute). "Predicting cellular responses to perturbation across diverse contexts with State." bioRxiv `10.1101/2025.06.26.661135` (preprint; author list from search snippet, not independently confirmed in full). https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1 | 2025, bioRxiv (preprint) | Perturbation-condition | N/A | None in the released benchmark — evaluated via Perturbation Discrimination Score / Differential Expression Score / MAE, all expression-similarity metrics, per the companion Virtual Cell Challenge paper (Cell, DOI `10.1016/j.cell.2025.06.001`-family; https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0) | N/A | N/A | Not attempted in the public benchmark | Largest current perturbation forward model; still stops at expression-similarity metrics, not phenotype/fate; author list and some details abstract/search-snippet only — flagged below. |
| 19 | Papalexi E, Mimitou EP, Butler AW, et al. "Characterizing the molecular regulation of inhibitory immune checkpoints with multimodal single-cell screens" (Mixscape). DOI `10.1038/s41588-021-00778-2`, PMCID `PMC8011839`. https://pmc.ncbi.nlm.nih.gov/articles/PMC8011839/ | 2021, Nat Genet | Cell (same-snapshot classification) | No — classifies current perturbation-response state | None — no future observed | **4 — terminal/current-state classifier** | **T1, explicitly** (does not claim otherwise) | N/A | Makes no fate claim at all; included as the clean textbook T1 tool against which any "state predicts fate" claim built on Perturb-seq data must be distinguished — the exact conflation risk the governing spec names. |
| 20 | Posas F, Nadal-Ribelles M, Sole C, et al. "Perturbation-driven transcriptional heterogeneity impacts cell fitness." bioRxiv `10.1101/2024.05.31.596868` (preprint; **abstract-only**, single-source retrieval, not cross-verified against a second database). https://www.biorxiv.org/content/10.1101/2024.05.31.596868v1 | 2024, bioRxiv (preprint) | Population (yeast, 3,500 mutants) | Cross-sectional per mutant/condition, not time-resolved | Fitness of the mutant population (mechanism of anchoring not confirmed from abstract alone) | **3 — cross-sectional condition comparison** (provisional) | None of T1/T2/T3 cleanly | Shows population-level transcriptional-state heterogeneity (not just mean) associates with fitness — closest population-level analogue found to Candidate B's spirit | Not mammalian, no CRISPR/DepMap anchor, not time-resolved/paired-prospective, no explicit burden-scalar control; **abstract-only**, flagged for follow-up verification. |
| 21 | Raju PC. "Geometric coherence of single-cell CRISPR perturbations reveals regulatory architecture and predicts cellular stress." arXiv preprint, submitted 2026-04-17. https://arxiv.org/abs/2604.16642 (**abstract-only**; single-author, non-peer-reviewed preprint) | 2026, arXiv (preprint) | Perturbation-condition, K562-class CRISPR datasets | Same-snapshot geometric coherence metric, not a separate early timepoint | **Concurrent** UPR pathway activation (same window), not a future/independent outcome | Not applicable — this is a same-window residual/incremental-information test, not a prospective design | **T1** (same-snapshot residual predicting a concurrent stress marker), explicitly not prospective | Reports "significant incremental prediction... beyond both [perturbation-response score] and magnitude" (p<10⁻¹⁸) for **concurrent** stress, not future fate | The closest explicit "incremental information beyond a scalar" framing found in the literature (Family 7), but it is same-window (T1), not prospective (T2); does not touch division/persistence/extinction or an independent net-fitness anchor. |

## Prose answers

### 1. Who is closest, and exactly how close?

No single paper combines the program's full conjunction (transcriptome-wide early
state + genetic/CRISPR perturbation + cancer cell line + independent net-fitness anchor
+ prospective trajectory outcome). The nearest work splits into two clusters that are
each close on a different axis, and the honest answer names both:

- **On modality and rigor of the prospective-fate methodology**: Weinreb et al. 2020
  Science (LARRY, row 2) is nearest. It is the only paper that combines a genuinely
  transcriptome-wide early molecular state, a real evidence-tier-A2 clonal design, an
  explicitly prospective (T2) task, and a documented quantitative accuracy ceiling
  (~50–60%). **The precise delta**: it is done in unperturbed normal hematopoiesis, not
  a CRISPR-perturbed cancer line; its outcome is differentiation-**type** choice, not a
  division/persistence/recovery/extinction trajectory; and — critically — it has **no
  net-fitness anchor at all**, so it never poses the "beyond fitness" question this
  program exists to answer. It demonstrates the phenomenon class (state has real but
  bounded prospective information) without touching the specific claim.

- **On the fitness-independence logic itself**: Iyer et al. 2025 PLOS Comput Biol (row
  1) is nearest. It is A1-tier (true same-lineage, live-imaged, non-destructive), in
  human cancer cell lines (HCT116, U2OS), under a lethal perturbation (cisplatin), and
  it **explicitly tests and reports** that pre-existing state predicts survival/death
  with no accompanying fitness-rate difference — the closest existing execution of the
  utility+specificity hypotheses in spirit. **The precise delta**: its "state" is
  cell-cycle timing, not a molecular/transcriptomic profile; its perturbation is a
  small-molecule drug, not a genetic/CRISPR knockdown; and its fitness comparison is an
  internal matched design (Analysis-R style), not conditioning on an independently
  pre-existing screen (Analysis-P / DepMap-Chronos style) as this program's primary
  claim requires.

Between the two, if forced to name one paper, **Weinreb et al. 2020** is the better
"nearest" answer because the program's estimand is explicitly a *molecular* state, and
LARRY is the only row that gets that dimension right at real evidence-tier rigor — but
the reviewer should not let that obscure that Iyer et al. 2025 is closer on the actual
logic of the claim being tested.

### 2. Is there an un-scooped claim left?

**Still available (un-scooped)**: "In a CRISPR/CRISPRi-perturbed cancer cell line,
anchored to an independently pre-measured DepMap/Chronos GeneEffect score, an early
post-perturbation single-cell transcriptomic state provides incremental prospective
information about the subsequent division/persistence/recovery/extinction trajectory
beyond that independent scalar, and this information does not reduce to a
response-magnitude/generic-injury scalar." No row in the matrix attempts this full
conjunction. Rows 10–11 (Norman, Replogle) show the exact comparison the program wants
to run — transcriptome vs. growth phenotype in K562 Perturb-seq — sitting right there,
unexploited, with growth used only as a side annotation, never as a modeled target from
which a residual is computed.

**Already taken (weakest available claim)**: "A pre-existing single-cell
state — measured before a lethal or growth-suppressing perturbation — predicts which
cells subsequently survive vs. die, independent of/beyond an aggregate growth-rate or
fitness scalar" is **already demonstrated**, at evidence-tier A1/A2, in multiple systems
(Balaban 2004 in bacteria; Iyer et al. 2025, Shaffer et al. 2017, Emert et al. 2021,
Frick et al. 2015 in human cancer cell lines under drug perturbation). A generic,
system-agnostic version of this claim carries little novelty by itself; the phenomenon
class is established. Novelty must come from the specific conjunction (molecular state,
genetic perturbation, DepMap-style anchor, full multi-state trajectory), not from
re-asserting that pre-existing state can matter beyond fitness in general.

### 3. Is the program scooped?

**Verdict: partially-scooped.**

The generic phenomenon underlying the wedge — that a pre-existing single-cell state can
predict divergent fate under a fitness-reducing perturbation independent of an aggregate
growth-rate/fitness readout — is not an open biological unknown in the abstract; it is
positively and repeatedly demonstrated at A1/A2 tier across bacteria (Balaban 2004) and
human cancer cell lines under drug perturbation (Iyer 2025, Emert 2021, Shaffer 2017,
Frick 2015). Gate 2's phenomenon-existence question should be updated toward "positive
cross-system precedent exists" rather than treated as fully open, and any framing of
this program that implies "it is unknown whether such state-dependent divergence can
exist independent of fitness at all" would overstate the novelty and should be avoided.

However, the **specific conjunction** this program needs — transcriptome-wide early
state, genetic/CRISPR perturbation (not a drug), a cancer-dependency line, an
independently pre-existing net-fitness anchor (DepMap/Chronos, not an internal matched
comparison), and a full division/persistence/recovery/extinction trajectory estimand
(not a binary survive/die call) — was not found combined in any single paper across
this search. The two nearest clusters (LARRY-style clonal transcriptomics; persister/
resistance-state cancer-cell-line literature) each get one side of the conjunction right
and the other side wrong, and the papers that sit exactly on the intended comparison
(Norman 2019, Replogle 2022 — Perturb-seq transcriptome alongside a growth phenotype in
K562) never model one from the other. That gap is real and appears genuinely open. This
is why the verdict is partially-scooped rather than scooped: the phenomenon claim is
weakened by strong adjacent precedent, but the specific target claim remains
undemonstrated.

## UNVERIFIED — could not retrieve / not independently confirmed

- **scGPT** (Cui et al.) and **scFoundation** (Hao et al.) — referenced repeatedly as
  benchmarked foundation models in the Ahlmann-Eltze et al. 2025 and Virtual Cell
  Challenge search results (rows 15, 18), but I did not independently fetch or confirm
  their own primary citations (title/venue/DOI/PMID) in this run. Carries zero
  evidential weight until verified directly.
- **CINEMA-OT** (causal identification of perturbation effects) — surfaced once in a
  Nature Methods search result alongside CellRank/Mixscape searches; not investigated
  further, not included in the matrix, and should not be treated as reviewed.
- Full author list and full-text methods for **Adduri et al. 2025 (State, Arc
  Institute)** (row 18) — obtained only via WebSearch snippet, not a direct fetch of the
  bioRxiv abstract/full text; treat the author list as provisional.
- Full-text methods for **Posas/Nadal-Ribelles et al. 2024** (row 20) and **Raju 2026**
  (row 21) — both retrieved via a single WebFetch summarization pass on a PDF/abstract
  page, not cross-checked against a second source (Europe PMC / Crossref / bioRxiv API
  all failed or were not attempted for these two). Marked abstract-only in the matrix;
  their tier/task assignments are provisional and should not be upgraded without a
  direct full-text read.
- A dedicated **per-cell prospective apoptosis/death prediction from transcriptomic
  state** literature (Family 5, in a genetic-perturbation or cancer-dependency context)
  was searched for explicitly and **not found**. What surfaced instead were bulk-tumor
  "cell-death gene signature" prognostic models (patient-population survival, not
  per-cell fate) — a different task entirely, at best Tier-5/T1. This is an
  **insufficient-evidence** finding, not a claim that such work does not exist; it was
  not located in this search.

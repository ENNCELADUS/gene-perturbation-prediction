# Gate 2A — Phenomenon Prevalence, Candidate A (Lineage/Clone Level)

**Question (spec §4.2):** Within the same perturbation and context, does reproducible
lineage-level trajectory heterogeneity exist, and is early lineage state associated
with it?

**Scope of this search:** literature-only, per SHARED_BRIEF. All entries below were
retrieved via WebSearch + WebFetch (Europe PMC `fullTextXML` or `search` core records)
in this run. DOI/PMID/PMCID given for every table row; anything not independently
retrieved is in `## UNVERIFIED`, never in the table or verdicts.

---

## Evidence table

| Paper | Unit / design | Perturbation type | Tier | Task | What it actually shows | Criterion |
|---|---|---|---|---|---|---|
| Spencer et al. 2009, *Nature* 459:428–432. DOI `10.1038/nature08012`, PMID 19363473, PMC2858974. https://pmc.ncbi.nlm.nih.gov/articles/PMC2858974/ | HeLa & MCF10A; continuous time-lapse imaging of individual cells through division and death; sister-pair correlation analysis | TRAIL (protein ligand), ± cycloheximide | **A1** for same-cell continuous tracking; sibling-correlation analysis is the mechanism used to demonstrate heritability (A2-flavored) | **T2** (prospective: death timing, `Td`, of a tracked cell) | Sister `Td` correlated `R²=0.93` if born <7 h before treatment; random pairs `R²=0.04`. Correlation decays exponentially, half-life ~11 h, lost (`R²≤0.05`) by ~50 h / ~2 generations. Protein state, not genotype, is the heritable variable | COMMON (existence); memory/heritability decay |
| "Hidden heterogeneity and circadian-controlled cell fate inferred from single cell lineages," *Nat. Commun.* 9:5372 (2018). DOI `10.1038/s41467-018-07788-5`, PMID 30560953, PMC6299096. https://pmc.ncbi.nlm.nih.gov/articles/PMC6299096/ | HCT116 p53-VKI colon cancer; continuous pedigree tracking by time-lapse imaging | Cisplatin (12.5 µM) | **A1/A2** (continuous imaging of every cell to its own fate; sibling analysis reported) | **T2** | Sisters share fate (death/survival) ~80% of time vs. ~53% expected under independence; correlation decays to baseline by 4 divisions (3rd cousins ≈ unrelated pairs). Fate set by state **inherited from mother pre-drug**, not by division order relative to drug addition | COMMON; LARGE (concordance-based, not founder-fraction); memory decay |
| Paek et al. 2016, *Cell* 165:631–642. DOI `10.1016/j.cell.2016.03.025`, PMID 27062928, PMC5217463. https://www.cell.com/cell/fulltext/S0092-8674(16)30321-X | Colon cancer; live-cell p53 reporter imaging (design detail corroborated by secondary summaries only — **abstract-level for our own verification**) | Chemotherapy (etoposide-class DNA damage) | **A1** (abstract-confirmed) | **T2** | Cell death is not a fixed p53-threshold event; rate/timing of p53 accumulation, not final level, predicts fractional killing. Surviving and dying cells reach similar p53 levels | COMMON; mechanism |
| Panagopoulos et al. 2025, *Nature* 642:785–795. DOI `10.1038/s41586-025-08986-0`, PMID 40399682, PMC12176655. https://pmc.ncbi.nlm.nih.gov/articles/PMC12176655/ | U2OS & RPE-1; endogenously tagged 53BP1/PCNA, multigenerational (≤4-generation) live-imaging pedigrees | **Genetic**: HRAS/cyclin-E1 overexpression, TP53/CDKN1A(p21)/AMBRA1 **siRNA** knockdown (not CRISPR); + APH/ATRi/IR/etoposide/pevonedistat | **A1** | **T2** | G1 DNA-damage state (53BP1/γH2AX/p53/p21 levels) is heritable and "correlates with" divergent S-phase commitment / replication-stress fate in descendant sister cells across generations. **No effect-size, R², or founder-fraction statistic reported** — qualitative correlation only. n≈20 lineages / ≤80 granddaughters per condition (U2OS), ≤10 lineages (RPE-1) | COMMON — closest genetic-perturbation A1 anchor found, but **underpowered and unquantified** |
| Emert et al. 2021 ("Rewind"), *Nat. Biotechnol.* 39:865–876. DOI `10.1038/s41587-021-00837-3`, PMID 33619394, PMC8277666. https://pmc.ncbi.nlm.nih.gov/articles/PMC8277666/ | BRAF^V600E melanoma; barcode + RNA-FISH on drug-naive precursor, descendant fate scored later | Vemurafenib + trametinib (drug) | **A2** | **T2** | Rare precursor state (~1:1,000–1:10,000 frequency) and persistent MAPK signaling shortly after drug predict later resistant-clone fate of that lineage | COMMON; CONSEQUENTIAL row "resistance emergence" |
| Fennell et al. 2022 ("SPLINTR"), *Nature* 601:125–131. DOI `10.1038/s41586-021-04206-7`, PMID 34880496. https://www.nature.com/articles/s41586-021-04206-7 | Mouse AML models (MLL-AF9/NrasG12D background); expressed-barcode clonal tracing pre/post chemo | Cytarabine chemotherapy (drug); genetic driver is the disease *model*, not the tested perturbation | **A2** | **T2** | Clonal dominance/output is a heritable, cell-intrinsic property; LSC clonal output predicts chemosensitivity | COMMON; CONSEQUENTIAL row "resistance emergence" |
| "Multifunctional barcoding with ClonMapper...," *Nat. Cancer* 2:1782–1801 (2021). DOI `10.1038/s43018-021-00222-8`, PMID 34939038, PMC8691751. https://www.nature.com/articles/s43018-021-00222-8 | CLL cell line; barcode + scRNA-seq + CRISPRa-inducible clone recall (COLBERT), clones retrieved before/during/after treatment | Ibrutinib/venetoclax (drug); CRISPRa used as a **retrieval tool**, not the studied perturbation | **A2** | **T2** | Distinct clonal transcriptional signatures link to distinct chemotherapy-survivorship trajectories of the same barcoded lineage | COMMON |
| Oren et al. 2021 ("Watermelon"), *Nature* 596:576–582. DOI `10.1038/s41586-021-03796-6`, PMID 34381210, PMC9209846. https://www.nature.com/articles/s41586-021-03796-6 | Multiple cancer lines; expressed barcode + H2B-mCherry dilution (lineage + proliferation history) | Chemo/targeted drugs (multiple) | **A2** | **T2** | Cycling vs. non-cycling persister fate traces to **pre-existing** lineage-specific transcriptional/metabolic programs (antioxidant, fatty-acid oxidation), not generic stress alone | COMMON; specificity hypothesis (§5.2 of spec) |
| Umkehrer et al. 2021 ("CaTCH"), *Nat. Biotechnol.* 39:174–178. DOI `10.1038/s41587-020-0614-0`, PMID 32719478, PMC7616981. | Melanoma, in vivo targeted therapy; CRISPRa-inducible reporter for live clone recall at 0.001% frequency | Targeted therapy (drug); CRISPRa again a **tool**, not the tested perturbation | **A2** (when applied) | **T2** | Technology paper — confirms retrospective founder-to-fate linkage is feasible at very low clone frequency, but is not itself a phenomenon finding | COMMON (methods note only) |
| "Tracing cellular heterogeneity in pooled genetic screens via multi-level barcoding," *BMC Genomics* 20:107 (2019). DOI `10.1186/s12864-019-5480-0`, PMID 30727954, PMC6364396. https://pmc.ncbi.nlm.nih.gov/articles/PMC6364396/ | Jurkat T cells; two-level barcode (clone ID + sub-clonal barcode) on **CRISPR knockout / CRISPRi** of TRAIL-apoptosis genes | **Genuine CRISPR knockout/knockdown** | Clone-outcome-only — no early-state arm at all; closest to **A3-adjacent "outcome divergence"** | **T2** on abundance trajectory (days 0/4/9/14), but **no `S_early` variable measured** | Sub-clonal barcodes carrying the **identical sgRNA** show significantly divergent fold-change trajectories (Wilcoxon p<10⁻¹²) — i.e., reproducible within-perturbation divergence for a real genetic perturbation. Cannot speak to whether early state predicts it, because no early molecular readout was taken | **COMMON — the one genuine CRISPR-perturbation existence data point found**, but silent on the "state-associated" half of the question |
| "Wildtype heterogeneity contributes to clonal variability in genome edited cells," *Sci. Rep.* 12:18211 (2022). DOI `10.1038/s41598-022-22885-8`, PMID 36307508, PMC9616811. https://pmc.ncbi.nlm.nih.gov/articles/PMC9616811/ | mIMCD-3 cells; CRISPR knockout of Pkd1, monoclonal WT and KO lines compared | **Genuine CRISPR knockout** | **Terminal-state / clone-average** (cross-sectional; honestly described as such, not misrepresented as prospective) | **T1** | Same-guide KO clones differ substantially in proteome/drug response/morphology; attributable in part to **pre-existing WT clonal heterogeneity**, not the edit. No variance decomposition given | COMMON (source of heterogeneity), not a T1-as-T2 violation |
| "Inheritable cell-states shape drug-persister correlations and population dynamics in cancer cells," *PLOS Comput. Biol.* (2025). DOI `10.1371/journal.pcbi.1013446`, PMID 40971961, PMC12469175. https://pmc.ncbi.nlm.nih.gov/articles/PMC12469175/ | Re-analysis + stochastic modeling (M0–M3) of published HCT116 and U2OS live-imaging cisplatin datasets | Cisplatin (re-analysis) | **A3 / inferential** (not new primary data) | Meta-level | Confirms 2–3 generation inheritance window; **explicitly documents that measuring barcode-diversity change is a misleading proxy for the timing of persister fate decisions** — a formal statement of the T1-vs-T2 pitfall this review is watching for | Memory decay; methodological caution |
| Sigal et al. 2006, *Nature* 444:643–646. DOI `10.1038/nature05316`, PMID 17122776. https://www.nature.com/articles/nature05316 | Human cell lines, 20 endogenous YFP-tagged proteins, **unperturbed baseline** | None (baseline noise) | Background, not perturbation-response | — | Protein-level "memory" persists >2 generations (>40 h) before mixing to population distribution — sets the generic decay timescale against which perturbation-response heritability (rows above) should be read | Memory/heritability baseline |

---

## Per-criterion verdicts

### COMMON (prevalence)

The frozen definition ("≥20% of net-fitness-matched **perturbation pairs**...") is written
in Candidate-B's between-perturbation matching language. Candidate A's question is
within-perturbation (do lineages that all got the *same* perturbation diverge?), so the
literal denominator ("how many perturbations were tested this way, and in what fraction
was divergence found") does not exist as a systematic quantity in the literature — only a
non-random set of ~9 published perturbation contexts that a lab specifically looked at
lineage-level divergence for.

- **Qualitative existence:** every study that used a genuine lineage-linked design (imaging
  or barcode) to look for within-perturbation divergence found it — across ≥4 independent
  live-imaging datasets (Spencer/TRAIL, Paek/chemo, HCT116-cisplatin, Panagopoulos/DNA
  damage) and ≥4 independent barcode-lineage datasets (Emert, Fennell, ClonMapper,
  Watermelon), spanning drug, ligand, and genetic (siRNA/overexpression, and one CRISPR
  knockout/CRISPRi case) perturbation types, multiple cancer types. This clears the **R2
  reproducibility bar** for the qualitative phenomenon.
- **Quantitative bar as frozen:** cannot be evaluated — no systematic panel of
  perturbations with a "tested and found no divergence" arm exists to compute a
  percentage, and publication/ascertainment bias (labs publish positive findings) makes
  any literature-derived percentage uninterpretable in either direction.
- **Call: existence — positive (R2). Frozen quantitative threshold — insufficient
  evidence**, because the denominator this review needs is not constructible from
  literature.

### LARGE (effect size)

No paper in the table reports the frozen quantities (≥0.20 absolute difference in
**founder**-referenced fraction lost, or ≥30% relative division-rate difference, each
>3× replicate SD).

- Imaging papers report correlation/concordance statistics (R², %-concordance-vs-chance),
  which are not the same estimand as a founders-lost fraction difference and are not
  reported with the required 3×-replicate-SD comparison.
- Barcode+sequencing papers (Emert, Watermelon, ClonMapper, Fennell) show
  near-binary outcomes (a lineage becomes a resistant colony or does not) that, if
  reframed, would likely be large in magnitude — but none report a founder-referenced
  denominator or a replicate-SD noise floor, and several (Emert, Watermelon) define
  "resistant fraction" over barcodes recovered as expanded/surviving clones, which is
  vulnerable to the **denominator trap**: cells lost pre-collection or in dissociation are
  invisible, so the quantity reported is not cleanly fraction-of-founders-lost.
- The continuous-imaging papers (Spencer; HCT116-cisplatin; Panagopoulos) are the
  methodologically correct design for founder-referenced accounting — because every cell
  is tracked to a directly observed division/arrest/death outcome, nothing is lost to
  unlogged attrition — but none of them actually computes or reports the frozen LARGE
  quantities in that form.
- **Call: insufficient evidence.** No adequately powered, founder-referenced,
  replicate-SD-quantified study exists for either drug or genetic perturbation.

### CONSEQUENTIAL (value, outside the trajectory definition)

Of the four qualifying downstream-endpoint rows in SIGNIFICANCE_CRITERIA.md, "resistance
emergence / persister outgrowth" is the one populated by evidence here: Emert/Rewind,
Watermelon/Oren, ClonMapper, and Fennell/SPLINTR all show early/precursor lineage state
reproducibly (R2, across ≥3 independent systems: melanoma, CLL, AML) predicting later
resistant/persister/chemosensitive outcomes of that same lineage.

- However: (1) none of these papers benchmark the improvement against an
  `F_net`-only baseline as the criterion requires — resistance emergence is detected, not
  shown to *exceed* what net fitness alone would predict; (2) all are **drug** or
  chemotherapy perturbations in cancer models, none in a CRISPR-knockout / DepMap-style
  genetic-dependency context; (3) the long-term regrowth, clonogenic-survival, and
  intervention-prioritization-turnover rows have no populating evidence at all in this
  search.
- **Call: insufficient evidence** for CONSEQUENTIAL as frozen. The "resistance emergence"
  row is the closest to being satisfiable and the most transferable target for a pilot,
  but the margin-over-`F_net` comparison and the genetic-perturbation setting are both
  unaddressed.

---

## Overall Gate 2A verdict: **INSUFFICIENT EVIDENCE → bounded validation pilot (not a stop)**

Not "positive" (LARGE and CONSEQUENTIAL, as frozen, are unaddressed by any study found).
Not "powered absence" (no adequately powered study tests the frozen thresholds and comes
back negative — the gap is that the necessary studies don't exist in this quantitative or
genetic-perturbation form, which is explicitly *not* falsification per spec §4.3).
The qualitative core phenomenon — within-perturbation lineage divergence, associated with
an inherited early state that decays over roughly 2–4 generations — is about as strongly
R2-reproduced as literature-only evidence can make it, but almost entirely from **drug**
perturbation (TRAIL, cisplatin, chemo, targeted therapy) and imaging/barcode designs never
built for DepMap-relevant CRISPR knockouts or for the frozen founder-referenced effect-size
metric.

### Bounded validation pilot sketch

- **System:** K562 (project's proof-of-concept line), continuous live-cell pedigree
  imaging (fluorescent viability/damage + division reporter, non-destructive — the design
  class that correctly avoids the denominator trap) OR a Rewind/ReSisTrace-style
  sister-split barcode design if imaging infrastructure is unavailable.
- **Perturbation panel:** 5–10 CRISPR knockouts spanning a DepMap GeneEffect range,
  explicitly matched in pairs at `tau`, `0.5·tau`, `2·tau` (per SIGNIFICANCE_CRITERIA §4.4)
  so COMMON and LARGE can be evaluated at their frozen definitions, not a proxy.
- **Scale:** power-calculated (not assumed) for ≥100 founder lineages per gene, tracked to
  a fixed horizon `T` spanning ≥4 generations, so a 0.20-absolute founders-lost difference
  is detectable against measured replicate SD (>3× bar).
- **Stop rule:** if <5% of matched pairs show R1 divergence meeting the LARGE bar →
  powered absence for Candidate A, stop. If 5–20% → scope to the implicated sub-class only.
  If ≥20% and confirmed R2 in an independent replicate → proceed to test CONSEQUENTIAL via
  a clonogenic-regrowth assay on the same tracked lineages, explicitly held out from
  trajectory construction.
- **Budget:** one imaging/barcoding campaign + one independent confirmatory replicate,
  reviewed before any production modeling commitment (per spec §9 permitted-work table).

---

## UNVERIFIED — could not retrieve / not independently read this run

These surfaced only in WebSearch result snippets or as citations inside other papers; I
did not fetch their primary text and they carry **zero evidential weight** above:

- GESTALT (genome editing of synthetic target arrays for lineage tracing) — developmental
  zebrafish system, mentioned only in a review snippet.
- Al'Khafaji, Deatherage & Brock, *ACS Synth. Biol.* 2018 — original COLBERT paper.
- "Cellular barcoding tracks heterogeneous clones through selective pressures and
  phenotypic transitions," PMC10339273.
- "Multi-omic lineage tracing predicts the transcriptional, epigenetic and genetic
  determinants of cancer evolution," *Nat. Commun.* 2024.
- "Clonal dynamics shaped by diverse drug-tolerant persister states in melanoma
  resistance," *Mol. Cancer* 2026.
- "Mother cells control daughter cell proliferation in intestinal organoids to minimize
  proliferation fluctuations," *eLife* 2022.
- "Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach"
  (Frontiers/bioRxiv 2020/2021).
- Review articles surfaced but not read in full: *Nat. Rev. Cancer* "Mastering the use of
  cellular barcoding..."; *Nat. Rev. Genet.* "Charting single-cell lineages..."; *Annu.
  Rev. Cancer Biol.* "New Tools for Lineage Tracing in Cancer In Vivo"; *Genome Res.*
  "Advancements in prospective single-cell lineage barcoding."
- The exact identity of "Chakrabarti et al." (ref. 18) and the U2OS dataset (ref. 37) cited
  inside the PLOS Comput. Biol. 2025 re-analysis — plausibly the HCT116/Nat. Commun. 2018
  paper and a Lahav/Loewer-lab U2OS p53 dataset respectively, but I did not confirm this
  attribution independently.

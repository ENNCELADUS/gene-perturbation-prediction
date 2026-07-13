# Gate 4 — Prospective Incremental Information: Do Linking Designs Exist?

Governing question (spec §8/Gate 4): *Do designs exist linking state to FUTURE outcome,
per candidate's hierarchy? Enforce T1-vs-T2 strictly.*

Scope note: this memo audits **designs**, not results. Every entry states what was
independently retrieved (DOI/PMID/PMCID + URL), whether it was read past the abstract,
and what tier it reaches under §8.1's hierarchies. Nothing here is cited from memory.

---

## 1. Design table

| # | Design | Linked unit | Future outcome independently measured? | Tier (§8.1) | Throughput | Genetic (CRISPR) perturbation done? | Supports P or only R? |
|---|---|---|---|---|---|---|---|
| 1 | **Live-seq** (FluidFM cytoplasmic biopsy + downstream imaging) | Same single cell | **Yes** — the biopsied cell's own later phenotype (time-lapse imaging, Tnf-mCherry reporter trajectory) is observed after the biopsy | **A1** (existence proof) | 4–5 extractions/hr; ~15 min/sample; 641 samples → 588 libraries → 294 QC-pass (5 replicates); ASPC: 44 cells → 8 paired QC-pass profiles | **No.** LPS-macrophage response and adipose-stromal differentiation only. No CRISPR coupling found anywhere in the paper or across 100 citing works surveyed (2024–2026) | Could support **P** if paired with an independent net-fitness anchor — not attempted |
| 2 | **Lineage barcode clonal-split / "sister cell"** (SIS-seq, ReSisTrace, CellTag-multi, LARRY, STRACK) | Sibling cell(s) sharing a clone barcode | **Yes for the sibling(s)**, never for the sequenced founder itself | **A2** | SIS-seq: single-clone scale; ReSisTrace: genome-wide, ovarian cancer line, per clone-pair; CellTag-multi: thousands of clones; LARRY: barcodes recovered in 26.4–47.8% of sequenced cells | **No CRISPR-knockout example found.** ReSisTrace = drug/NK-immune; SIS-seq/LARRY = unperturbed differentiation; CellTag-multi = reprogramming-factor overexpression; STRACK = Cre-activated oncogenic point mutation (genetic, not CRISPR-KO) | Design is **P-capable in principle** (barcode + CRISPR library + clonal split is buildable with existing components) — not yet instantiated this way |
| 3 | **Live imaging w/ molecular reporter (FUCCI/caspase/degron) → fixation → in-situ profiling** | Same tracked cell | **Not established.** Reporter tools (FUCCI, DEVD-caspase FRET, executioner-caspase reporters) exist and track fate live, but no paper found that couples a *rich* (transcriptome-scale) early molecular state of a tracked cell to its own later, independently confirmed fate via this pipeline | Not established (would default to tier 4–5 if attempted with a snapshot) | N/A | No | **Insufficient evidence** — could not find the design, not shown infeasible |
| 4 | **Time-resolved / inducible genetic perturbation** (dTAG, AID degrons; inducible Cas9) | Same cell/population, temporally decoupled from division | Supplies **early perturbation onset**, not by itself a future-outcome pairing | Enables an early state; realized designs remain population dropout (tier 4, aggregate) | SLC25A23 senolytic screen: genome-wide Brunello library, dox-iCas9, 10-day dropout in senescent A549 | **Yes** (dropout-screen sense) — dox-inducible Cas9 in non-dividing/senescent cells is real and published; **no** single-cell/lineage prospective pairing yet | Tool exists to enable **P**; not yet combined with a per-cell/lineage prospective readout |
| 5 | **Condition-level paired prospective anchor** (Candidate B rank-1: independent `F_net` + independently measured `[t0,T]` population dynamics, for a genetic perturbation) | Condition/population | **None found** despite extensive search | — | — | No | This is exactly what analysis **P** requires — **absent**, not merely under-sampled |
| 6 | **Optical pooled screening** (guide ID + imaging) — Feldman OPS, CRISPRmap, Perturb-FISH, Perturb-Multi, DuMPLING, DynaScreen | Single cell, genotyped after (or alongside) imaging | **Mixed.** DuMPLING/DynaScreen: genuinely prospective at single-cell level (phenotype tracked *before* genotype is revealed) but the tracked "state" is a shallow physiological signal (growth rate/division size; FRET-FLIM cAMP kinetics), not a transcriptome. CRISPRmap/Perturb-FISH/Perturb-Multi: richer molecular readout (imaging + in-situ RNA/IF) but **simultaneous**, not prospective | Feldman OPS: tier 3–4 (cross-sectional, morphology). CRISPRmap/Perturb-FISH/Perturb-Multi: tier 3 (cross-sectional, richer state). DuMPLING/DynaScreen: tier 2 (partial time resolution) for a **narrow, non-transcriptomic** state | DuMPLING: 235 CRISPRi knockdowns, >500 cell cycles/knockdown, *E. coli*. DynaScreen: 318 genes × 4 gRNAs, HeLa (proof of concept). Feldman OPS: genome-scale. CRISPRmap/Perturb-FISH/Perturb-Multi: hundreds–genome-scale | **Yes**, all six are CRISPR-based | DuMPLING/DynaScreen: **P-capable but state-poor**. CRISPRmap/Perturb-FISH/Perturb-Multi: **R only** (cross-sectional genotype–phenotype maps) |

---

## 2. Design-by-design evidence

### 2.1 Live-seq — the A1 existence proof

Chen W, Guillaume-Gentil O, Rainer PY, et al. "Live-seq enables temporal transcriptomic
recording of single cells." *Nature* 609, 2022. DOI `10.1038/s41586-022-05046-9`, PMID
35978187, PMCID PMC9402441. Retrieved via Europe PMC full-text XML
(`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9402441/fullTextXML`), read in full
(not abstract-only).

Concrete numbers extracted directly from the full text:

- **Throughput**: "Live-seq sampling is currently throughput-limited with 4–5 extractions
  per hour due to downstream processing and following the fate of individual cells by
  live imaging." Manual timing: ~15 min/sample (5 min load buffer + approach cell, 5 min
  extract biopsy, 5 min transfer to lysis buffer).
- **Yield/success**: Across 5 replicates, 641 samples acquired → 588 libraries generated
  → **294 passed QC** (>1,000 genes, <30% mitochondrial reads) — the paper separately
  states "around 40% of the samples passing our data quality control criteria."
- **Biopsy size vs. fate perturbation (explicitly benchmarked)**: extraction removed
  40–70% of total cell volume. Cell viability post-biopsy was **85–89%** across three
  cell types, "only slightly lower than that after conventional trypsin-based cell
  dissociation (90–95%, data not shown)." Extracted cells recovered pre-extraction volume
  within 100–320 min and resumed growth comparable to controls; the authors explicitly
  state they "cannot rule out that Live-seq introduces a small cell cycle delay." For
  sequential ASPC sampling: 44 cells sampled, 2/44 died vs. 3/41 unextracted controls died
  (95% vs. 93% survival) — statistically indistinguishable in the paper's own framing.
- **Cell types demonstrated**: RAW264.7 macrophages (and a Tnf-mCherry reporter subline,
  RAW-G9), IBA brown-adipose stromal cells, primary mouse adipose stem/progenitor cells
  (ASPCs), HeLa (species-mixing control only). **No cancer cell line, no CRISPR
  perturbation.**
- **The A1 pairing itself**: ground-state transcriptome of individual RAW macrophages was
  recorded, then the *same* cells were time-lapse imaged through LPS exposure; n = 91 in
  the Nfkbia-BFP validation; of 40 cells subjected to both Live-seq and Tnf-mCherry
  tracking, 17 passed QC. This is the paper's own A1 pairing — same-cell, prospective,
  non-CRISPR.

**Successor**: Cai L, Lin S, Qiu M, et al. "scBiopsy-seq: a platform for temporal
single-cell RNA-seq analysis." bioRxiv, DOI `10.1101/2025.03.26.645409` (Fudan
University; retrieved via bioRxiv API, preprint only — `published: NA` as of retrieval).
Reports >10K genes/extraction at ~90% success rate (a substantial improvement over
Live-seq's ~40%), and was applied to a **BRD4 degrader** (chemical-genetic, not CRISPR)
time course. **No CRISPR coupling found for this successor either.** Citation search
(Semantic Scholar, 100 citing papers of Live-seq, 2024–2026) surfaced no paper coupling
any FluidFM-style non-destructive biopsy to a CRISPR perturbation.

### 2.2 Lineage barcoding / clonal-split ("sister cell") designs — the A2 ceiling

- **SIS-seq**: original preprint bioRxiv `10.1101/403113`; published as "Combining
  single-cell tracking and omics improves blood stem cell fate regulator identification,"
  *Blood* 2022, DOI `10.1182/blood.2022016880`, PMID 35820055, PMCID PMC9523371 (Rieger
  lab). Design: clones subdivided at day 2.5 into 3 parts — 1 for RNA-seq, 2 into sister
  culture wells — fate scored later by flow cytometry, correlated computationally with the
  sequenced sibling's transcriptome. **Unperturbed hematopoietic differentiation, no
  genetic perturbation.**
- **ReSisTrace**: "Tracing back primed resistance in cancer via sister cells," *Nature
  Communications* 15:1158, 2024. DOI `10.1038/s41467-024-45478-7`, PMID 38326354, PMCID
  PMC10850087. Sister cells barcoded, allowed one division, one half sequenced
  pre-treatment, the other half treated; applied to a high-grade serous ovarian cancer
  line against carboplatin, olaparib (PARP inhibitor), and NK-cell cytotoxicity. **The
  "treatment" here is chemical/immune, not CRISPR** — but the underlying scaffold (barcode
  clone → split → sequence one half → perturb/observe the other) is perturbation-agnostic
  and could in principle take a CRISPR knockout as the treatment arm. This has not been
  done.
- **CellTag-multi**: "Single-cell lineage capture across genomic modalities with
  CellTag-multi reveals fate-specific gene regulatory changes," *Nature Biotechnology*
  2024, DOI `10.1038/s41587-023-01931-4`, PMID 37749269, PMCID PMC11180607 (Morris lab).
  Clones split into up to 4 subclones at different timepoints/modalities ("sibling
  sequencing"); day-2.5 vs. day-5 subclone comparison shows early functional priming
  before fate commitment is visible transcriptomically. Context: fibroblast
  reprogramming (a genetic perturbation via reprogramming-factor overexpression, **not a
  CRISPR knockout**).
- **LARRY / Weinreb et al.**: "Lineage tracing on transcriptional landscapes links state
  to fate during differentiation," *Science* 2020, DOI `10.1126/science.aaw3381`, PMID
  31974159, PMCID PMC7608074. Barcodes recovered in 26.4–47.8% of sequenced cells across
  datasets, ~16% shared between parent/daughter. **Key finding directly relevant to the
  wedge**: clone-splitting/transplant experiments showed that cell-autonomous fate bias
  explained *more* variance in eventual fate choice than the initial transcriptional state
  alone could explain — i.e., a transcriptome snapshot underdetermines fate, which is
  positive A2-tier evidence *for* the program's motivating premise. No genetic
  perturbation (developmental hematopoiesis).
- **STRACK**: Singh I, Fernandez-Perez D, Sanchez Sanchez P, Rodriguez-Fraticelli AE.
  "Pre-existing stem cell heterogeneity dictates clonal responses to the acquisition of
  leukemic driver mutations," *Cell Stem Cell* 2025, DOI `10.1016/j.stem.2025.01.012`,
  PMID 40010350. Traces clonal HSC transcriptomic state **before** and clonal fate
  **after** activation of an oncogenic driver mutation (Dnmt3a-R878H, Npm1c; mouse,
  recombinase-activated conditional allele). **This is the closest match found to
  "A2/A3 tier + genetic perturbation"** — but the perturbation is a Cre-activated point
  mutation in vivo, not a CRISPR knockout screen in a cancer-dependency context.

**No A2-tier design found anywhere uses a CRISPR-knockout library as the perturbation
whose clonal fate is causally attributed.** All four found scaffolds (ReSisTrace,
SIS-seq/LARRY, CellTag-multi, STRACK) are otherwise complete, validated designs that
differ only in the nature of the perturbing agent.

### 2.3 Live imaging + molecular reporter + fixation/profiling

Searched extensively for FUCCI/caspase/degron-reporter designs coupled to a rich in-situ
molecular profiling step on the same tracked cell. Found abundant reporter technology
(FUCCI2a cell-cycle reporter, PMID 25486356; DEVD-based FRET caspase-3 reporters, PMC
6025355; real-time executioner-caspase reporter platforms, *Cell Death Discovery* 2025,
`10.1038/s41420-025-02662-y`) but **no paper coupling these to a subsequent rich
transcriptomic in-situ readout of the same tracked cell's early state**. This is an
**insufficient-evidence** finding (§4.3) — absence of a retrieved design is not evidence
the design is infeasible.

### 2.4 Time-resolved / inducible genetic perturbation

Degron systems (dTAG, auxin-inducible degron/AID) are mature, comparative benchmarking
exists (PMC11601833, 4-system comparison in human PSCs), but no paper was found pairing a
degron/inducible-Cas9 system directly with a Perturb-seq-style early transcriptomic
snapshot in a cancer-dependency context. The concrete realized application found:

- Xu et al. (title not independently verified beyond PNAS listing), "An antibiotic that
  mediates immune destruction of senescent cancer cells," *PNAS* 2024, DOI
  `10.1073/pnas.2417724121`, PMCID PMC11670111 — genome-wide Brunello library,
  doxycycline-inducible Cas9, 10-day dropout screen in senescent (non-dividing) vs.
  proliferating A549 cells; identifies SLC25A23 as a senescence-selective vulnerability.
- Companion protocol: "Inducible CRISPR–Cas9 screening platform to interrogate
  non-proliferative cellular states," *Nature Protocols*, DOI `10.1038/s41596-025-01251-8`,
  PMID 41062702 (2026). Generalizes temporal control of Cas9 induction independent of cell
  division — directly relevant to spec §4.4's concern that Replogle Perturb-seq's single,
  late, division-linked timepoint conflates timing with fate commitment.

Both remain **population dropout screens** (T1/aggregate net-fitness), not per-cell or
per-lineage prospective (T2) designs. The temporal-decoupling *tool* exists; it has not
been paired with a single-cell/lineage prospective fate readout.

### 2.5 Condition-level paired prospective anchor (Candidate B rank-1)

Searched for a design that (a) measures an early single-cell state distribution for a
genetic (CRISPR) perturbation, (b) independently measures population dynamics over
`[t0,T]` for the *same* perturbation, and (c) anchors net fitness to an *independent,
pre-existing* screen (per §5.1's Analysis P) rather than the same future window. GR/DIP
metrics (Hafner et al. 2016, PMC4887336) were built for and have been applied almost
exclusively to small-molecule dose-response, not CRISPR knockouts paired with early
transcriptomic snapshots. **No design of this kind was found.** This is the single most
consequential negative finding in this memo: it is exactly the anchor Candidate B's rank-1
tier and Analysis P require.

### 2.6 Optical pooled screening

- Feldman D, et al. "Optical Pooled Screens in Human Cells," *Cell* 2019, DOI
  `10.1016/j.cell.2019.09.016`, PMID 31626775, PMCID PMC6886477 — foundational OPS;
  guide identity + imaged morphology, genome-scale, cross-sectional/endpoint.
- CRISPRmap: "Sequencing-free optical pooled screens mapping multi-omic phenotypes in
  cells and tissue," bioRxiv `10.1101/2023.12.26.572587` (retrieved via bioRxiv API;
  `published: NA` — still a preprint). Guide ID + multiplexed immunofluorescence + in-situ
  RNA — richer molecular state, but simultaneous with genotyping, not prospective.
- Perturb-FISH extension: "Simultaneous CRISPR screening and spatial transcriptomics
  reveal intracellular, intercellular, and functional transcriptional circuits," *Cell*
  2025, DOI `10.1016/j.cell.2025.02.012`, PMCID PMC12135205. Couples CRISPRi knockdown to
  live calcium-activity imaging *and* gene expression in iPSC-derived astrocytes; the
  paper's own framing is explicitly "simultaneous," not sequential state→fate.
- Perturb-Multi: Saunders RA, et al., *Cell* 2025, DOI `10.1016/j.cell.2025.05.022`,
  PMCID PMC12324982. Guide ID + imaging (expression + morphology) + sequencing in mouse
  liver tissue — again simultaneous multimodal mapping.
- **DuMPLING**: "Time-resolved imaging-based CRISPRi screening," *Nature Methods* 2020,
  DOI `10.1038/s41592-019-0629-y`, PMID 31740817. 235 CRISPRi knockdowns in *E. coli*, in
  situ genotyping performed **after** time-lapse imaging of on average >500 cell cycles
  per knockdown. This is genuinely prospective at the single-cell level (phenotype
  tracked, then genotype revealed) but the tracked "state" is growth rate/division
  size/replication timing — not a molecular/transcriptomic readout — and the organism is
  bacterial.
- **DynaScreen**: "Beyond Static Screens: A High-Throughput Pooled Imaging CRISPR
  Platform for Dynamic Phenotype Discovery," bioRxiv `10.1101/2025.07.11.664338`
  (Netherlands Cancer Institute; retrieved via bioRxiv API). Mammalian (HeLa), tracks live
  FRET-FLIM cAMP-signaling dynamics per cell, then photo-tags + FACS-sorts + sequences to
  reveal genotype. Genuinely prospective/time-resolved at the single-cell level, but state
  = a single biosensor signal (cAMP kinetics), proof-of-concept scale (318 genes × 4
  gRNAs).

None of these six give a rich early transcriptomic state paired with an independently
measured, later fate for the same cell. DuMPLING/DynaScreen solve the *temporal
ordering* problem (phenotype before genotype) but not the *state richness* problem;
CRISPRmap/Perturb-FISH/Perturb-Multi solve the *state richness* problem but not the
temporal one.

---

## 3. Papers that are T1 presented as T2

1. **Okaniwa T, Kryukov K, Shiroguchi K. "Finding differentially expressed genes between
   cell fates predicted by image-based deep learning."** *Biophysics and Physicobiology*
   22:e220022, 2025. DOI `10.2142/biophysico.bppb-v22.0022`, PMID 41189733, PMCID
   PMC12582640. Retrieved and read in full via Europe PMC full-text XML (not
   abstract-only). Design: heat-stressed mammalian cells are time-lapse imaged; a deep
   learning model is trained on image trajectories to **predict** eventual fate (survival
   vs. death); cells are then **picked at an early timepoint (e.g., 5 h) and destructively
   sequenced** using the paper's ALPS robot; DEGs are computed between transcriptomes of
   cells whose fate was *predicted* by the imaging classifier. **This is T1 dressed as
   T2**: because the profiled cell is destroyed at the early timepoint, its own true
   future is never independently observed — the "future outcome" attached to each
   sequenced transcriptome is a **classifier-imputed label**, not an independently
   measured outcome for that unit. The paper is transparent about this in its methods
   (worth crediting), but a downstream reader could easily mistake "DEGs between predicted
   fates" for "DEGs that predict fate" — precisely the substitution flagged by the spec's
   "tell." Non-genetic perturbation (heat stress).
2. **General pattern, not a single paper**: any downstream use of DepMap/Chronos
   `GeneEffect` as if it were a cell-death or fate label is a population-level version of
   the same T1-as-T2 substitution — an aggregate net-fitness summary standing in for a
   decomposed dynamic. The spec's own §12 already guards against this; no single named
   paper was found committing it inside this review's search radius, but the risk is
   structural to the field's habitual language, not paper-specific.
3. **High-mito/low-UMI "dying cell" signature usage** — the spec's own citation for this
   caveat, "Comprehensive generation, visualization, and reporting of quality control
   metrics for single-cell RNA sequencing data," *Nature Communications* 2022, DOI
   `10.1038/s41467-022-29212-9`, PMID 35354805, PMCID PMC8967915, was independently
   re-retrieved via Europe PMC (abstract confirmed: the paper is a general QC-metrics
   tool, not itself a T1-as-T2 offender, but its subject matter — mito%/UMI as QC
   artifacts rather than fate labels — is exactly the substitution warned against; it is
   evidence *for* the caveat, not an instance of the error).

No other candidate papers surfaced in this search radius made an explicit, checkable claim
of prospective fate prediction from a snapshot without either (a) genuine lineage/sibling
pairing or (b) a disclosed model-imputed proxy label. This is reported as **insufficient
search coverage**, not as "no such papers exist" — a systematic sweep of the CRISPR-screen
literature for exactly this substitution was out of scope for the time available.

---

## 4. The three-way finding, per candidate

### Candidate A (lineage/clone level)

- **A1 tier (same-cell prospective).** Live-seq is real, published, and technically
  works — cell viability post-biopsy (85–89%) is close to conventional dissociation
  (90–95%), and growth dynamics recover within hours. **Not technically infeasible.** It
  has never been coupled to a CRISPR perturbation, by anyone, in any organism found in
  this search. This is **finding 3 — a genuine methodological opportunity** — with an
  honest caveat: current throughput (4–5 cells/hr, ~40–90% QC pass depending on
  platform generation) makes genome-scale pooled CRISPR screening at A1 tier currently
  impractical; an arrayed, small-panel CRISPR perturbation (tens of genes) coupled to
  Live-seq or scBiopsy-seq is within reach today. This scaling gap is a resourcing/
  engineering constraint (partially finding 2, in the sense that no lab has generated
  this data and we cannot acquire FluidFM/scBiopsy-seq access ourselves without a wet-lab
  collaboration), not evidence of infeasibility.
- **A2 tier (sibling/clone proxy).** Four independent, validated scaffolds exist
  (ReSisTrace, SIS-seq, CellTag-multi, STRACK) that are perturbation-agnostic in their
  core logic — barcode a clone, split it, sequence one branch early, observe the other
  branch's fate. **None uses a CRISPR-knockout library as the perturbing agent.** Building
  this requires no new hardware, only a lentiviral barcode/CRISPR library and standard
  clonal culture — well within reach of a molecular biology lab. This is **finding 3 — the
  most actionable near-term methodological opportunity in this entire review** for
  Candidate A, should it be selected.
- **A3 tier (clone-average).** Trivially achievable with any barcoded CRISPR screen with
  repeated timepoint sampling; not a bottleneck.

### Candidate B (population level)

- **Rank 1 (condition-level paired prospective anchor).** Not found for genetic
  perturbation despite extensive search (GR/DIP + CRISPR, Perturb-seq + growth-curve
  imaging + independent DepMap/Chronos anchor). GR/DIP metrics exist and are validated,
  but only for small-molecule dose-response. **This is finding 3 — the central genuine
  methodological opportunity for Candidate B**, and not a resourcing question: nothing
  found suggests the components (arrayed or pooled CRISPR library + time-resolved
  imaging/growth assay + an independent DepMap anchor + an early Perturb-seq or bulk RNA
  snapshot) are technically blocked from being combined. It simply has not been done.
- **Rank 2–3 (partial time resolution / cross-sectional).** DuMPLING and DynaScreen sit
  at rank 2 for a narrow, non-transcriptomic state; CRISPRmap/Perturb-FISH/Perturb-Multi
  sit at rank 3 (richer state, but cross-sectional). These exist, at real scale, with
  CRISPR perturbation — **finding: exists, but capped below rank 1 and therefore cannot
  by itself support Analysis P**.

**Do not conflate the two negative findings above.** Neither absence is because the
underlying technology is impossible (finding 1 does not apply to either candidate).
Neither is primarily a case of "the data exists elsewhere but we lack access" (finding 2)
— nothing found suggests a data-unavailable-to-us situation; the designs schematically
do not yet exist anywhere. Both route to **finding 3: genuine methodological
opportunity**, which per the spec's own framing is *the best possible outcome for the
program* — it means the reason no incremental-information claim can yet be made is that
nobody has built the measurement, not that the biology forbids it.

---

## 5. Temporal contract (§5.1): does anything found support Analysis P, or only R?

- **Live-seq (A1)**: supports **P** in principle for the specific cell whose transcriptome
  is measured — its own future is observed after, not derived from, the transcriptome.
  But no existing instance pairs this with an *independent, pre-existing* `F_net` (e.g.,
  DepMap/Chronos) for a genetic perturbation; the published demonstrations use LPS/
  differentiation, which have no DepMap-style external fitness anchor at all. So today it
  is P-capable in structure but untested in the program's actual P sense.
- **A2 scaffolds (ReSisTrace, SIS-seq, CellTag-multi, STRACK)**: also P-capable in
  structure (the sibling's fate is observed independently of the sequenced sibling's
  state) but none of the realized instances anchor to an independent, pre-existing
  net-fitness screen; as run, they are descriptive/associational, closer to **R** in
  practice even though the design itself does not require same-window `F_net`.
- **CRISPRmap/Perturb-FISH/Perturb-Multi**: structurally **R only** — state and outcome
  are measured in the same window/same assay, so any "beyond net fitness" claim built on
  them would be a retrospective conditional decomposition at best.
- **DuMPLING/DynaScreen**: structurally **P-capable** (phenotype observed before genotype
  reveal), but the "state" is not the kind of early molecular/transcriptomic state the
  spec's `S_early` refers to, so they cannot support the specificity hypothesis (§5.2)
  even where they support temporal ordering.
- **Net effect on claim ceiling**: **no design found in this review can currently support
  Analysis P as specified** (independent, pre-existing `F_net` + genuinely early molecular
  state + independently measured future outcome, for a genetic perturbation). Every
  usable existing scaffold either lacks the independent pre-existing anchor, lacks the
  molecular-state richness, or lacks the prospective pairing altogether. **Any near-term
  empirical work this program undertakes on Candidate A or B would need to build the P
  anchor itself; absent that, all claims default to Analysis R** (retrospective
  conditional decomposition) and must be labeled as such per §5.1 and §12.

---

## UNVERIFIED

- Any claim that Live-seq, scBiopsy-seq, DuMPLING, or DynaScreen have since (after this
  review's retrieval date) been coupled to a CRISPR/dependency screen — not searched
  exhaustively past ~2026-07; a narrower, dated follow-up search is recommended before
  this memo is treated as final on that point.
- Whether any non-English-language or conference-only (non-indexed) literature reports an
  A1- or Candidate-B-rank-1-tier design for genetic perturbation — out of scope for the
  retrieval tools used here (Europe PMC, Crossref, Semantic Scholar, bioRxiv API,
  WebSearch).
- A systematic sweep of the CRISPR-Perturb-seq literature specifically for the "T1
  presented as T2" substitution (item 2 in §3) was not performed exhaustively; only one
  concrete instance (Okaniwa et al. 2025) was confirmed by full-text reading. Treat §3 as
  illustrative, not a complete catalogue.
- The two Nature Protocols entries on inducible CRISPR–Cas9 for non-proliferative states
  (`10.1038/s41596-025-01251-8` and a companion `10.1038/s41596-025-01252-7` surfaced in
  search but not independently opened) may be companion pieces (protocol +
  primer/commentary); only the first was directly confirmed via Crossref/PubMed in this
  run.

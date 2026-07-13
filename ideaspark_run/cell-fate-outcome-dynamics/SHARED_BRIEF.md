# Shared Brief — Cell-Fate Outcome-Dynamics Literature Review

**Every subagent MUST read this file and the governing spec before starting.**

Governing spec (read it in full, it is the contract):
`docs/superpowers/specs/2026-07-13-cell-fate-prediction-research-direction-design.md`

---

## The wedge (the thing under review)

> The same net fitness loss can arise from completely different cellular dynamics.

Strong division suppression with little loss, normal division with substantial loss,
early loss followed by survivor regrowth, and transient arrest followed by recovery can
all yield the same aggregate readout. This is **mathematically true and biologically
unproven**. The review exists to find out whether it is *common, large, and
consequential* in real genetic perturbations.

## The two candidate research questions (carried in PARALLEL — do not select between them)

**Candidate A — lineage / clone level.** Within a fixed context and time horizon, does
an early post-perturbation molecular state predict the subsequent division,
persistence/recovery, and extinction trajectory of **its linked lineage**, beyond an
independently measured net fitness?

**Candidate B — population level.** Under comparable, independently measured net
fitness, does the early single-cell state **distribution** provide incremental
information about **independently measured** future population dynamics?

## Evidence tiers — a finding's weight is CAPPED by its tier

Candidate A tiers (§4.1):

| Tier | Design | Highest claim supported |
|---|---|---|
| A1 | Same-cell prospective — non-destructive state measurement on a cell whose own future is then observed | Per-cell prospective fate prediction |
| A2 | Sibling / clone proxy — one clone member sequenced, siblings' futures observed | Clone-level prospective association. NOT per-cell fate. |
| A3 | Clone-average — clone-level early-state summary vs clone-level outcome | Clone-average association only |

Candidate B hierarchy (§8.1): 1 = condition-level paired prospective anchor (state at
`t0`, dynamics over `[t0,T]`); 2 = partial time resolution; 3 = cross-sectional
condition comparison; 4 = terminal-state classifier; 5 = signature-only inference.

**Governing principle:** direct prospective evidence *at the same unit as the estimand*
outranks proxy evidence at another unit. A terminal-state classifier can NEVER establish
a prospective claim, however strong its metrics.

## The three prediction tasks — DO NOT CONFLATE (§3.5)

- **T1** — is this cell *currently* dying/arrested? Terminal-state classification. A
  state readout, **not fate**.
- **T2** — probability of division / persistence / recovery / extinction within
  `[t, t+D]`? **Prospective prediction.** Requires longitudinal or lineage pairing.
- **T3** — what would this same unit have done under a *different* perturbation? Strict
  counterfactual.

> **Much of the apparent literature will be T1 presented as if it were T2. Catching this
> is a primary job of this review.** For every paper you log, state which task it
> actually performs, not which it claims.

## Two kinds of "loss" — NEVER conflate (§3.4)

- **Biological loss / lineage extinction** — the lineage genuinely ends. A biological outcome.
- **Assay attrition** — the unit is not observed: died pre-collection, lost in
  dissociation, failed capture, removed by QC. A measurement process.

## SEED PAPERS (verified, use as anchors)

1. **Live-seq** — Chen W, Guillaume-Gentil O, et al. "Live-seq enables temporal
   transcriptomic recording of single cells." *Nature* 609, 2022.
   DOI `10.1038/s41586-022-05046-9`.
   Non-destructive single-cell transcriptome extraction via fluidic force microscopy;
   couples a cell's ground-state transcriptome to its OWN downstream phenotype, including
   preregistering macrophage transcriptomes then time-lapse imaging the same cells after
   LPS. **This is an existence proof of Tier A1.** It was demonstrated on macrophage LPS
   response and adipose stromal differentiation — NOT on genetic (CRISPR) perturbation in
   a cancer line. Establish precisely what it can and cannot support: throughput, cell
   types, whether anyone has coupled it to CRISPR perturbation, whether the
   biopsy itself perturbs fate.

2. **Death-seq** — Colville A, Liu JY, et al. "Death-seq identifies regulators of cell
   death and senolytic therapies." *Cell Metabolism* 35, 2023.
   DOI `10.1016/j.cmet.2023.08.008`. PMID 37699398. PMC10597643.
   A **positive-selection** CRISPR screen for cell death, built explicitly because
   dropout screens "are generally underpowered because of the short timescales of cell
   death as well as the difficulty of scaling non-dividing cells." This is an existing
   readout that targets death directly rather than inferring it from net depletion.
   Establish what it identifies and what it does NOT (it screens for death *enhancers*
   under a drug; does it decompose division vs loss for an arbitrary genetic
   perturbation?).

## Reviewer-supplied prior readouts to verify by reading (§2)

- **Chronos** — fits an explicit population-dynamics model converting sgRNA abundance
  change into a relative growth-rate effect of knockout.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8686573/
- **GR / DIP metrics** — Hafner et al. 2016; given time course and initial counts, can
  distinguish fully cytostatic from net cytotoxic responses under stated assumptions.
  https://pmc.ncbi.nlm.nih.gov/articles/4887336/

---

## NON-NEGOTIABLE EVIDENCE RULES

1. **Every citation must come from an actual tool call in your run** (WebSearch,
   WebFetch, or an API such as Europe PMC / Crossref / Semantic Scholar / OpenAlex).
   Record the DOI or PMID or PMCID **and the URL you actually retrieved**.
2. **You may NOT cite from memory.** If you believe a paper exists but could not retrieve
   it, put it in a separate `UNVERIFIED — could not retrieve` list. It carries **zero
   evidential weight** and must never enter an evidence table or a gate verdict.
3. **Do not paraphrase an abstract into a claim it does not make.** If you did not read
   past the abstract, say so — mark the record `abstract-only`. Method details (number of
   timepoints, whether the same cell was re-observed, what the denominator was) are
   exactly where abstract-only reading fails.
4. **Absence of data is NOT falsification** (§4.3). If you cannot find evidence, the
   finding is *insufficient evidence*, which routes to a bounded pilot — never to a stop.
   Say "I could not find X" — never "X does not exist."
5. Obey every claim boundary in §12 of the spec. In particular: DepMap GeneEffect is not
   a cell-death label; high-mito/low-UMI is not a dying-cell signature; a
   QC-relaxation-induced cluster is not a recovered dying population.

## Useful retrieval endpoints (publishers often 403 WebFetch; use these)

```
Europe PMC:  https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=<q>&resultType=core&format=json
             (full text: .../rest/<PMCID>/fullTextXML)
Crossref:    https://api.crossref.org/works/<DOI>
OpenAlex:    https://api.openalex.org/works?search=<q>&filter=from_publication_date:YYYY-MM-DD
Semantic Sch: https://api.semanticscholar.org/graph/v1/paper/search?query=<q>&fields=title,year,venue,abstract,externalIds,tldr
bioRxiv:     https://api.biorxiv.org/details/biorxiv/<DOI>
```

## Output contract

Write your deliverable to the exact path given in your task prompt using the `Write`
tool. Then return **≤ 250 words** to the parent: the output path, your verdict/routing
signal, and the two or three findings that would most change the parent's decision.
Do not paste your full deliverable into your reply.

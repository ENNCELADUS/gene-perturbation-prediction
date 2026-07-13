# Research Direction: Perturbation Outcome Dynamics Behind Net Fitness

Status date: 2026-07-13 (revised after reviewer major-revision)
Type: research direction + literature review plan (not an implementation spec)
Origin: PI meeting notes on transcriptomics -> cellular phenotype / cell death
prediction; brainstorming session over repository state; reviewer major-revision
pass.

Revision note: this document replaces an earlier draft that asserted (a) population
screens cannot separate death from arrest, (b) per-cell death probability is
counterfactual, (c) unobserved `B` must therefore be generated, and (d) one fate
model unifies dependency, conditional essentiality, and SL. All four were
overstated. They are corrected or downgraded to hypotheses below. See Section 8.

## 1. The Preserved Wedge

One claim survives the revision intact, and it is the reason to continue:

```text
The same net fitness loss can arise from completely different cellular dynamics.
```

Strong division suppression with little death, normal division with substantial
death, early death followed by survivor regrowth, and transient arrest followed by
recovery can all produce the same aggregate readout. Everything below is an attempt
to state precisely what follows from that, and what does not.

## 2. Corrected Premise

The previous premise ("population screens measure depletion; no population screen
can separate death from arrest") is wrong as stated.

- **Chronos** already fits an explicit population-dynamics model, converting sgRNA
  abundance changes into relative growth-rate effects of gene knockout. It does not
  uniquely decompose reduced division from increased death, but it is materially
  more precise than "measures depletion."
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC8686573/)
- **GR / DIP-style metrics** with time-course information and initial cell counts
  can, under stated assumptions, distinguish fully cytostatic from net cytotoxic
  responses. (Hafner et al. 2016, https://pmc.ncbi.nlm.nih.gov/articles/4887336/)

The accurate premise:

> A single endpoint-abundance or net-fitness readout is generally insufficient to
> uniquely determine the underlying division, recovery, persistent-arrest, and
> cell-loss dynamics.

And the accurate gap statement:

> Existing pooled fitness readouts do not provide perturbation-specific,
> single-cell-resolved, prospective decomposition of cellular outcomes.

Not: "prior work failed to distinguish cytostatic from cytotoxic responses."

## 3. Ontology and Estimands (L0)

The previous draft's output — `proliferating / arrested / dying / mechanism /
timing` — mixes levels and cannot form a mutually exclusive distribution.
`proliferating` is a rate, `arrested` is a state (possibly reversible), `dying` is a
process, `death` is an event, the mechanisms may overlap, and `timing` is not an
extra label but part of the fate definition. A cell can arrest and recover, or
arrest and later die.

Any usable estimand must therefore fix:

| Element | Requirement |
| --- | --- |
| Time horizon `T` | All outcomes are defined over `[t0, t0 + T]` from perturbation onset. No horizon, no estimand. |
| Unit | The founder cell / lineage / clone, **not** the observed cell. Division means one founder maps to many observed cells. |
| Division | Division history or division rate over the horizon. |
| Arrest | Reversible vs. persistent arrest, distinguished; recovery is an explicit outcome, not the absence of one. |
| Cell loss | Cumulative loss over the horizon, distinct from instantaneous dying state. |
| Censoring | Right censoring at `T` is explicit. |
| Mechanism | Overlapping multi-label attributes, not a forced single-label simplex. |
| Denominator | Fractions are of **what**? Cells observed at `T` are survivors of a branching-plus-death process; fraction-of-observed is not fraction-of-founders. |

The denominator row is not pedantry. It is where identifiability and observation
bias meet, and it is the reason Section 4 is the first gate.

### 3.1 Three Distinct Prediction Tasks

These must never be conflated again:

| Task | Statement | Type |
| --- | --- | --- |
| T1 | Is this cell **currently** in a dying state? | Terminal-state classification. A state readout, not fate. |
| T2 | What is the probability this cell divides / arrests / recovers / is lost within `[t, t+D]`? | **Prospective prediction.** Estimable, but requires longitudinal or lineage pairing. Not a counterfactual. |
| T3 | What would this same initial cell have done under a *different* perturbation? | Strict intervention counterfactual. |

The earlier claim that "per-cell death probability is counterfactual" collapsed T2
into T3 and is withdrawn. T2 is the task of interest and it is a prediction problem,
not an identification-from-nothing problem — provided the longitudinal design exists.
Lineage/barcoding work (e.g. Rewind, https://www.nature.com/articles/s41587-021-00837-3)
shows state and future behavior can be linked through such designs, which is
simultaneously evidence that a **snapshot alone does not equal fate**.

## 4. First Gate: Identifiability

This is the largest logical hole in the previous draft and is now the first gate.

A single aggregate fitness scalar is consistent with infinitely many
division/death histories:

```text
same net fitness  <-  strong arrest, little death
                  <-  normal division, substantial death
                  <-  early death followed by survivor regrowth
                  <-  transient arrest followed by recovery
```

Therefore:

- Aggregate consistency (`aggregate(outcome composition) == observed fitness`) is
  **necessary calibration**. It is **not** identification.
- Retreating from a per-cell probability to a **bag-level fraction does not make the
  problem identifiable**. The previous draft implied it did. It does not.
- Any decomposition requires **independent anchors** on at least division, loss, and
  time — not a fitness scalar alone.

The literature review must therefore first produce a **measurement -> identifiable
quantity map**: which combinations of readouts (endpoint abundance; time course;
initial counts; division tracking via dye dilution or lineage-barcode counts; direct
death readout via live imaging, dead-cell stain, or caspase reporter) identify which
quantities, under which assumptions. Any modeling proposal that is not anchored in
that map is unfalsifiable.

## 5. Central Research Question (first-stage candidate)

> **Among genetic perturbations with comparable net fitness effects, do early
> single-cell molecular states prospectively distinguish later division suppression,
> persistent arrest, recovery, and cell loss?**

Main hypothesis:

> After controlling for aggregate effect severity and generic stress, early cellular
> state distributions still contain reproducible information about later outcome
> composition.

Null hypothesis:

> All apparent fate information is explained by effect magnitude, timing, generic
> stress, and observation bias.

Properties that make this the right first question: it assumes neither that the
transcriptome is useful, nor that virtual cells are needed, nor that the
decomposition is identifiable; it exploits the preserved wedge directly (same net
fitness, different dynamics); and a **negative result is scientifically meaningful**.

The literature review is expected to return 2-3 candidate questions of this
character, of which exactly one is selected as primary. The above is the leading
candidate entering the survey, not a settled choice.

## 6. Rival Explanations to Control

Any positive result must be shown not to reduce to these.

| Rival | Source | Status |
| --- | --- | --- |
| Effect magnitude / response burden | exp02: a generic response-magnitude scalar reaches Spearman 0.426 against a 0.494 full-feature baseline | Mandatory reported covariate |
| Generic stress program | Meeting notes; exp02 program scores | Mandatory reported covariate |
| Observation / selection bias | Section 7 | Mandatory reported covariate |
| Two-way additive structure: `GeneEffect(c,g) ~ gene_mean(g) + line_mean(c)` | Reviewer; PI's "model may only learn cell line identity" concern; exp09's non-pan-essential collapse (CV3 AUROC 0.645 -> 0.583, AUPR 0.651 -> 0.490) | **Scoped**: this null gates claims about *context-specific residual* effects. It does **not** adjudicate whether outcome decomposition is scientifically valuable. It is run when, and only when, a context-specificity claim is made. |

The scoping of the additive null is a correction: the previous draft made it the
universal first gate, which conflated a context-specificity control with an
identifiability question.

## 7. Observation-Process Selection (downgraded from "QC survivorship bias")

The previous draft asserted that high mitochondrial fraction and low UMI count are
"the definitional signature of a dying cell." That is wrong. They also arise from
dissociation stress, genuinely low-RNA biological states, ambient RNA, and technical
failure; mitochondria-rich clusters may reflect sample preparation rather than
biology (https://www.nature.com/articles/s41467-022-29212-9). Worse, cells that died
before collection, dissociation, or droplet capture are **never observed at all**, and
no computational QC relaxation can recover them.

The correct formulation:

> State-dependent acquisition, dissociation, capture, and QC may induce
> **missing-not-at-random** observation of perturbation outcomes.

Two consequences:

1. "A cluster appears after relaxing QC" **cannot** be interpreted as "the dying
   population has been recovered." That inference is invalid and the previous draft
   proposed it as an experiment.
2. The honest immediate probe is a **loss-accounting** analysis, not a QC-relaxation
   analysis: quantify what fraction of cells is removed at each stage per
   perturbation, and test whether the *removal rate* covaries with perturbation
   strength or with GeneEffect. A dependence there is direct evidence of
   state-dependent observation loss; independence bounds it.

Existing datapoint worth carrying in: exp01 found `n_cells_only` gives Spearman 0.000
and AUROC 0.498 against GeneEffect. So the *surviving* cell count carries no
dependency signal in that setup. That is a different quantity from the **QC-failure
fraction**, and the distinction is exactly what the loss-accounting analysis tests.

## 8. Downgraded Assumptions (now hypotheses, not premises)

| Previously asserted | Now |
| --- | --- |
| The transcriptome can decompose division / arrest / death dynamics | **Central hypothesis under test** (Section 5). Not an assumption. A snapshot measures state; it may reflect early fate commitment, generic stress severity, the consequence of already-executing death, or a residual state after survivorship selection. |
| `B` is unobserved for most contexts, therefore `B` must be generated | **Invalid inference; withdrawn.** Generating `B` is warranted only if `B` carries independently verifiable fate-relevant information. Virtual-cell response generation is a **tool hypothesis**, not a structural justification for the research direction. |
| One fate model unifies dependency, conditional essentiality, and SL as corollaries | **Long-term hypothesis, not a corollary.** Dependency / conditional fitness is retained as potential downstream relevance. Death mechanism is a second-stage question. |
| Bag-level output solves the identifiability problem | **Withdrawn.** More honest than an uninterpreted per-cell latent, but not identifiable without independent anchors (Section 4). |

## 9. Synthetic Lethality: Separate Memo

SL is a **joint-intervention effect relative to a combination null model**. It is not
the same scientific question as single-perturbation outcome decomposition, and it must
stop driving the definition of this project. It moves to its own research memo.

Substance to carry into that memo (needed for the next PI discussion regardless):

- The PI's question "how do we derive SL from DepMap" was already partially answered
  here by experiment 09's cross-cell-line selectivity contrast
  (`sel(a->b) = mean[d_{c,b} | a-intact] - mean[d_{c,b} | a-defective]`, with defect
  called by a composite OR over damaging mutation, hotspot, CN loss, low expression).
- It produced a consistent classification lift, largest on CV3 (+0.050 AUROC), but on
  the non-pan-essential slice the CV3 lift largely vanished (AUROC 0.583, AUPR 0.490).
  The recorded verdict: most cold-start lift is essentiality structure, not
  pair-specific co-dependency.
- So the live question is **not** "which statistic," it is: *how do we construct a
  null that removes pan-essentiality so the residual is genuinely interaction?*
- The same null problem governs any multi-gene AIVC route. SL is defined as a
  deviation from an expected combined effect, so `P(loss | perturb a + b)` is
  meaningless without an explicit single-perturbation null to subtract:
  `interaction(a,b) = outcome(a,b) - psi(outcome(a), outcome(b))`. The choice of
  `psi` (additive, multiplicative, or learned) is the crux.

## 10. Literature Review: Go/No-Go Funnel

The previous L1-L8 list is replaced. Gates run in order; a failed gate stops the
funnel rather than being routed around.

### L0 — Ontology and Estimands

Define, with citations: state; fate; death; cell loss; quiescence; senescence;
recovery; time horizon; denominator. Output is the vocabulary every later gate uses.

### Gate 1 — What Can Existing Readouts Identify?

Chronos and CRISPR fitness scoring; GR / DIP metrics; birth-death decomposition; and
which **combinations** of measurements separate division from loss, under which
assumptions. Output: the measurement -> identifiable-quantity map (Section 4).

**No-go condition:** if existing readouts already deliver perturbation-specific,
prospective outcome decomposition, the wedge is closed and the program stops.

### Gate 2 — Does State Predict Future Outcome?

Search same-cell, lineage, clone, and matched-population designs pairing transcriptome
with **future** fate. Strictly separate **prospective prediction** (T2) from
**terminal-state classification** (T1); much of the apparent literature will be T1.

**No-go condition:** if no design links state to future outcome, T2 is not estimable
with available data and the program must either acquire such data or stop.

### Gate 3 — Observation-Process Bias

Acquisition, dissociation, capture, QC, and pre-capture dead-cell disappearance —
as one selection process, not a mito/UMI threshold question. Output: what is knowable
about the missing-not-at-random structure, and what bounds exist.

### Gate 4 — Expand Only After Novelty Is Confirmed

Only if Gates 1-3 support the problem: virtual-cell forward models (including the
live critique that deep perturbation models barely beat trivial baselines — this
directly threatens any generate-`B` route); dependency and conditional essentiality
prior art; death mechanism signatures; cross-context generalization. SL is handled
in the separate memo, not here.

## 11. Literature Review Deliverables

1. Ontology / estimand memo.
2. Measurement -> identifiable-quantity map.
3. Nearest-prior-work matrix.
4. Evidence table **supporting and challenging** the core premise.
5. 2-3 candidate research questions.
6. For each: falsifier, claim boundary, explicit out-of-scope definition.
7. Selection of **one** primary research question.

## 12. Claim Boundaries

Extending `CLAUDE.md`'s terminology guardrails:

- Do not say population screens cannot separate death from arrest. Say a single
  endpoint net-fitness readout does not uniquely determine the underlying dynamics.
- Do not call DepMap GeneEffect a cell-death label. It is a relative growth-rate
  effect under an explicit population-dynamics model.
- Do not equate high-mito / low-UMI cells with dying cells.
- Do not describe a QC-relaxation-induced cluster as a recovered dying population.
- Do not call a prospective outcome prediction (T2) a counterfactual; reserve that
  for T3.
- Do not claim an outcome decomposition is identified when only aggregate
  consistency has been shown.
- Do not claim synthetic lethality without an explicit combination null and
  interaction residual.

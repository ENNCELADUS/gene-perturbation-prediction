# Research Direction: Outcome Dynamics Behind Net Fitness

Status date: 2026-07-13 (revision 2, after focused-revision review)
Type: research direction + literature review plan. **Not** an implementation spec.
No modeling work begins before the decision point in Section 9.

Origin: PI meeting notes on transcriptomics -> cellular phenotype / cell death;
brainstorming over repository state; two reviewer passes (major revision, then
focused revision).

## 0. Revision History and Withdrawn Claims

Revision 1 corrected: Chronos/GeneEffect is not raw depletion; aggregate consistency
is not identification; bag-level output is not automatically identifiable; T1/T2/T3
separated; transcriptome-decomposes-dynamics, generate-`B`, and one-model-unifies-all
downgraded to hypotheses; QC reformulated as observation-process selection; SL moved
out of the core question.

Revision 2 (this document) fixes six further defects:

| # | Defect in revision 1 | Fix |
| --- | --- | --- |
| 1 | Primary unit inconsistent: Section 3 said lineage, T2 said cell, Section 5 said perturbation-level distribution. Three different research questions sharing one "fate prediction" conclusion. | **Two candidate questions, explicitly separated** (Section 4). The survey selects one. |
| 2 | "Outcome composition" is not a valid simplex. Division suppression is continuous; persistent arrest is window-defined; recovery is a transition; a lineage can simultaneously produce dividing and dying descendants. | Replaced by **trajectory / multi-state process estimands** (Section 3). The word "composition" is retired. |
| 3 | The gap statement was technical ("pooled readouts do not provide single-cell-resolved prospective decomposition"), not biological. "Same net fitness, different dynamics" is mathematically true but not yet a finding. | **New Gate 2: phenomenon prevalence and value** (Section 8). The gap is restated biologically. |
| 4 | All rivals treated as covariates. But net fitness is a summary of the outcome, not a confounder; generic stress may be a true fate precursor, not a nuisance; MNAR cannot be fixed by adding a covariate. | **Utility and specificity hypotheses split** (Section 5); MNAR removed from the composite null and made an identifiability boundary (Section 6). |
| 5 | Gate 1's no-go ("if existing readouts can decompose, the program stops") was wrong — an independent decomposition is the *ground truth* the prospective question needs. Gate 2 conflated "no published design" with "not estimable." | **Funnel reordered and no-go criteria rewritten** (Section 8). |
| 6 | Observation-bias claims too strong: independence of removal rate does not bound pre-capture loss. | **Claims weakened; specific probes moved out** of this spec into a measurement memo (Section 6). |

## 1. The Preserved Wedge

```text
The same net fitness loss can arise from completely different cellular dynamics.
```

Strong division suppression with little loss, normal division with substantial loss,
early loss followed by survivor regrowth, and transient arrest followed by recovery
can all yield the same aggregate readout.

This is **mathematically true and biologically unproven**. Whether such divergence is
common, reproducible, large, and consequential in real genetic perturbations is an
empirical question, and it is Gate 2. Until Gate 2 passes, this program has a
motivating observation, not a finding.

## 2. Premise and Gap

What existing readouts already do (**to verify by reading, not from summary** — these
are reviewer-supplied and have not yet been read in full):

- **Chronos** fits an explicit population-dynamics model, converting sgRNA abundance
  change into a relative growth-rate effect of knockout. It does not uniquely
  decompose reduced division from increased loss.
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC8686573/)
- **GR / DIP-style metrics**, given time course and initial counts, can distinguish
  fully cytostatic from net cytotoxic responses under stated assumptions.
  (Hafner et al. 2016, https://pmc.ncbi.nlm.nih.gov/articles/4887336/)

Accurate premise:

> A single endpoint-abundance or net-fitness readout is generally insufficient to
> uniquely determine the underlying division, recovery, persistent-arrest, and
> cell-loss dynamics.

Accurate **biological** gap (this replaces the technical gap statement):

> It is unknown whether genetic perturbations with similar net fitness effects induce
> distinct, reproducible outcome trajectories, and whether early molecular states
> prospectively distinguish those trajectories.

Accompanying **value** question, which the program must answer to matter:

> Does such a decomposition explain consequences that net fitness cannot — recovery,
> persistence, lineage extinction?

Without the value question, a successful program is a technically correct measurement
decomposition of limited interest.

## 3. Ontology and Estimands (L0)

### 3.1 Retired Vocabulary

`proliferating / arrested / dying / mechanism / timing` mixes levels and is not a
simplex. `proliferating` is a rate; `arrested` is a window-defined state, possibly
reversible; `dying` is a process; death is an event; mechanisms overlap; timing is
part of the fate definition, not an extra label. **The term "outcome composition" is
retired** — a lineage can simultaneously produce descendants that keep dividing and
descendants that are lost.

### 3.2 Required Structure

Any usable estimand fixes:

| Element | Requirement |
| --- | --- |
| Time horizon `T` | All outcomes defined over `[t0, t0+T]` from perturbation onset. No horizon, no estimand. |
| "Early" | Defined **relative to `T` and to fate commitment**, not to sample collection. Must be stated numerically per dataset. |
| Unit | See Section 4 — the open decision. |
| Censoring | Right censoring at `T` is explicit. |
| Denominator | Fractions are of *what*? Observed cells at `T` are survivors of a branching-plus-loss process; fraction-of-observed is not fraction-of-founders. |

### 3.3 Estimands as Trajectories, Not Categories

**Lineage/founder unit** — natural endpoints over `[t0, t0+T]`:

- division history (number/rate of divisions);
- alive-but-non-dividing through `T`;
- recovery transition (arrest -> resumed division);
- **lineage extinction**;
- descendant abundance at `T`.

**Population unit** — the object is a **multi-state transition process** or an
explicitly enumerated set of trajectory summaries, not a categorical mixture.

### 3.4 Two Kinds of "Loss" — Never Conflate

| Term | Meaning |
| --- | --- |
| **Biological loss / lineage extinction** | The lineage genuinely ends. A biological outcome. |
| **Assay attrition** | The cell or lineage is not observed: died pre-collection, lost in dissociation, failed capture, or removed by QC. A measurement process. |

Revision 1 used "cell loss" for both. Every future use must be disambiguated.

### 3.5 Three Distinct Prediction Tasks

| Task | Statement | Type |
| --- | --- | --- |
| T1 | Is this cell **currently** in a dying/arrested state? | Terminal-state classification. A state readout, **not fate**. |
| T2 | What is the probability of division / persistence / recovery / extinction within `[t, t+D]`? | **Prospective prediction.** Requires longitudinal or lineage pairing. Not a counterfactual. |
| T3 | What would this same initial unit have done under a **different** perturbation? | Strict intervention counterfactual. |

Much of the apparent literature will be T1 presented as if it were T2. Gate 4 exists
to enforce the distinction.

## 4. The Two Candidate Research Questions

The primary unit is **not yet selected**. Both are carried, with separate estimands,
evidence requirements, and falsifiers. The survey selects one at the Section 9
decision point.

### Candidate A — Lineage level (cell-fate science)

> Within a fixed biological context and time horizon, does an early post-perturbation
> molecular state predict the subsequent division, persistence/recovery, and
> extinction trajectory of **its linked lineage**, beyond perturbation-average net
> fitness?

- Unit: founder cell / lineage / clone.
- Estimand: Section 3.3, lineage endpoints.
- Evidence requirement: a design **linking a state measurement to that same
  lineage's future** — lineage barcoding with longitudinal sampling, or
  imaging-paired sequencing.
- Supports: a genuine per-lineage prospective fate claim.
- Known risk: no such data for genetic perturbation in K562 is present in this
  repository, and its existence at usable scale is unverified. Candidate A may
  require data generation or collaboration.

### Candidate B — Population level

> Under comparable aggregate net fitness, does the early single-cell state
> **distribution** provide incremental information about **independently measured**
> future population dynamics?

- Unit: the perturbation condition.
- Estimand: multi-state population trajectory summaries (Section 3.3).
- Evidence requirement: an **independently measured** future-dynamics anchor.
  Endpoint viability alone does **not** qualify — it is an aggregate, not a
  trajectory.
- Supports: a population-level informational claim only.
- **Explicitly forfeits any per-cell or per-lineage fate-prediction claim.**
- Known risk: matched-population evidence can only support population-level
  association. It can never be upgraded into prospective per-cell fate prediction.

### 4.1 Terms Requiring Operational Definition Before Either Runs

- **"Comparable net fitness"**: the matching tolerance, and whether matching is on the
  point estimate or its uncertainty.
- **"Reproducible"**: across replicates, across datasets, or across contexts. These are
  different claims of different strength.
- **"Early"**: numerically, per dataset, relative to `T`. Note as a live constraint
  that Replogle Perturb-seq is a **single, late** timepoint several days
  post-transduction; whether any existing genetic-perturbation dataset provides a
  genuinely early state is a Gate 1 question, not an assumption.

## 5. Hypotheses

Revision 1 lumped every rival into one composite null. That was wrong: net fitness is
a **summary of the outcome**, not an ordinary confounder; generic stress may be a
**true fate precursor**, not a nuisance to residualize away; and timing may itself be
perturbation biology. Two hypotheses, tested separately:

### Utility hypothesis

> The early state provides **incremental prospective information beyond aggregate net
> fitness**.

Predictive null:

```text
Y_future  ⟂  S_early  |  F_net, X
```

where `Y_future` is an **independently measured** future outcome (never derived from
`S_early`), `F_net` is the aggregate net-fitness summary, and `X` contains predefined
context, timing, and measurement variables.

### Specificity hypothesis

> That information **cannot be reduced to a scalar response-burden or generic-injury
> score**.

Tested by residualizing on burden/stress scalars, not by discarding them. This keeps
generic stress biologically meaningful rather than declaring it noise. Grounding: exp02
found a generic response-magnitude scalar reaches Spearman 0.426 against a 0.494
full-feature baseline — burden is a large, real component, not an artifact.

### Scoped separately, not in the null

- **Two-way additive structure** (`GeneEffect(c,g) ~ gene_mean(g) + line_mean(c)`):
  gates claims of **context-specific residual** effects only. It does not adjudicate
  whether outcome decomposition is valuable. Run it when, and only when, a
  context-specificity claim is made. Grounding: exp09's non-pan-essential slice
  (CV3 AUROC 0.645 -> 0.583, AUPR 0.651 -> 0.490).
- **MNAR observation bias**: an **identifiability boundary**, not a covariate.
  Section 6.

## 6. Observation Process as an Identifiability Boundary

State-dependent acquisition, dissociation, capture, and QC may induce
**missing-not-at-random** observation of perturbation outcomes. Cells that die before
collection are **never observed**, and no computational QC relaxation recovers them.

Claims that must not be made:

- High mitochondrial fraction / low UMI is **not** a definitional signature of a dying
  cell. It also arises from dissociation stress, genuinely low-RNA states, ambient RNA,
  and technical failure; mito-rich clusters may reflect sample preparation
  (https://www.nature.com/articles/s41467-022-29212-9).
- "A cluster appears after relaxing QC" **cannot** be read as "the dying population was
  recovered."
- Independence between removal rate and perturbation strength **does not bound**
  pre-capture loss. Every perturbation could lose the same *fraction* of cells while
  losing biologically *different* cells, and truly dead cells may vanish before any
  logged stage. Revision 1 asserted this bound; it is withdrawn.
- A correlation between removal rate and GeneEffect is at most **evidence compatible
  with perturbation-dependent attrition**. It is not direct evidence of
  state-dependent biological loss.

MNAR is therefore an **identifiability boundary on what any estimand can claim**, not a
term in the predictive null. Concrete loss-accounting probes, QC ablations, and the
relevant exp01 datapoints belong in a **separate measurement memo**, not in this
science spec.

## 7. Downgraded Assumptions (hypotheses, not premises)

| Previously asserted | Now |
| --- | --- |
| The transcriptome can decompose division / arrest / loss dynamics | **The central hypothesis under test.** A snapshot measures state; it may reflect early fate commitment, generic stress severity, the consequence of already-executing death, or a residual after survivorship selection. |
| `B` is unobserved for most contexts, therefore `B` must be generated | **Invalid inference; withdrawn.** Generating `B` is warranted only if `B` carries independently verifiable outcome-relevant information. Virtual-cell response generation is a **tool hypothesis**, not a justification for the direction. |
| One model unifies dependency, conditional essentiality, and SL | **Long-term hypothesis.** Dependency / conditional fitness is retained as potential downstream relevance only. |
| Bag-level output solves identifiability | **Withdrawn.** More honest than an uninterpreted per-cell latent, but not identifiable without independent anchors. |
| Death mechanism (apoptosis / ferroptosis / necroptosis) | **Second-stage question, out of scope here.** The plausible supervision route — perturbagens of known death mechanism — is recorded for later, not pursued now. |

## 8. Literature Review: Go/No-Go Funnel

### Evidence Hierarchy (predefined, applied at every gate)

```text
same-cell / lineage prospective anchor
  >  condition-level paired anchor
  >  terminal-state classifier
  >  signature-only inference
```

A finding's weight is capped by its tier. A terminal-state classifier can never
establish a prospective claim, however strong its metrics.

### L0 — Ontology, Primary Unit, Latent-to-Observable Map

Define with citations: state; fate; death; biological loss vs. assay attrition;
quiescence; senescence; recovery; time horizon; denominator. Enumerate, for each
candidate unit, which latent quantities map to which observables.

### Gate 1 — Measurement and Observation Validity

Chronos and CRISPR fitness scoring; GR / DIP metrics; birth-death decomposition; which
**combinations** of readouts separate division from loss, under which assumptions;
what is knowable about the MNAR structure and what bounds exist. Also: does any
existing genetic-perturbation dataset supply a genuinely **early** state?

Output: the **measurement -> identifiable-quantity map**.

**No-go:** if the target outcome cannot be defined and identified through **any**
credible independent anchor, **narrow the estimand or stop.**

*Not* a no-go: existing readouts already decomposing dynamics. An independent
decomposition is the **ground truth the prospective question requires** — it removes
measurement novelty, not the biological question. Revision 1 had this backwards.

### Gate 2 — Phenomenon Prevalence and Biological Importance

Are dynamic differences under **matched net fitness** common, reproducible, large
enough to matter, and consequential beyond what net fitness already captures
(recovery, persistence, extinction)?

**No-go:** if divergence under matched net fitness is rare, small, or
inconsequential, the wedge is not a research program.

### Gate 3 — Nearest Prior Art and Exact Novelty

Who has come closest, on which unit, with which anchor, at which evidence tier. Output
is a nearest-prior-work matrix, not a reading list.

### Gate 4 — Prospective Incremental Information

Do same-cell / lineage / clone / matched-population designs exist that link state to
**future** outcome? Enforce T1-vs-T2 strictly.

**Separate three distinct findings** — revision 1 collapsed them:

1. no published linkage design exists because it is **technically infeasible**;
2. none exists because the **data is unavailable** to us;
3. none exists and this is a genuine **methodological opportunity**.

Only (1) is a stop. (2) is a resourcing question. (3) is the best possible outcome.

Also here: the live critique that deep perturbation forward models barely beat trivial
baselines — it directly threatens any generate-`B` route.

### Decision

`proceed` | `narrow-or-pivot` | `stop`.

## 9. Deliverables and Decision Point

1. Ontology / estimand memo (L0), including the **selected primary unit**.
2. Measurement -> identifiable-quantity map.
3. Evidence table **supporting and challenging** the core premise.
4. Nearest-prior-work matrix.
5. 2-3 candidate research questions, each with falsifier, claim boundary, and explicit
   out-of-scope definition.
6. Selection of **one** primary research question.
7. Decision: `proceed` | `narrow-or-pivot` | `stop`.

**No modeling, feature engineering, or data acquisition begins before item 7.** The
earlier decision to "acquire dose-response data with real viability" is **suspended**:
endpoint viability is an aggregate and supplies no prospective anchor under either
candidate unit. The data target is re-decided at the decision point.

## 10. Separate Memos (out of scope here)

- **Synthetic lethality.** SL is a joint-intervention effect relative to a combination
  null. It is not the same question as single-perturbation outcome decomposition and
  must stop driving this project's definition. The memo carries: exp09's cross-cell-line
  selectivity result and its collapse on the non-pan-essential slice; the resulting
  question ("how do we build a null that removes pan-essentiality so the residual is
  genuinely interaction?"); and the combination-null problem governing any multi-gene
  route, `interaction(a,b) = outcome(a,b) - psi(outcome(a), outcome(b))`, where the
  choice of `psi` is the crux.
- **Measurement memo.** Loss accounting, QC ablation design, attrition probes, and the
  exp01 cell-count datapoint.

## 11. Claim Boundaries

Extending `CLAUDE.md`'s terminology guardrails:

- Do not say population screens cannot separate death from arrest. Say a single
  endpoint net-fitness readout does not uniquely determine the underlying dynamics.
- Do not call DepMap GeneEffect a cell-death label. It is a relative growth-rate effect
  under an explicit population-dynamics model.
- Do not use "outcome composition"; state a trajectory or multi-state process estimand.
- Do not write "loss" without disambiguating biological extinction from assay attrition.
- Do not equate high-mito / low-UMI cells with dying cells.
- Do not describe a QC-relaxation-induced cluster as a recovered dying population.
- Do not claim observation bias is bounded by an independence check.
- Do not call a prospective outcome prediction (T2) a counterfactual; reserve that for T3.
- Do not upgrade a population-level (Candidate B) result into a per-cell fate claim.
- Do not claim an outcome decomposition is identified when only aggregate consistency
  has been shown.
- Do not claim synthetic lethality without an explicit combination null and interaction
  residual.

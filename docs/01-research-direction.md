# Research Direction: Outcome Dynamics Behind Net Fitness

**Status:** revision 4 (2026-07-14) — the active research contract for this project.
**Type:** research direction + literature-review contract. **Not** an implementation spec.
**Supersedes:** [`docs/superpowers/specs/2026-07-13-cell-fate-prediction-research-direction-design.md`](superpowers/specs/2026-07-13-cell-fate-prediction-research-direction-design.md) (revision 3, frozen).
**Provenance:** revision 4 applies the eight amendments required by [`ideaspark_run/cell-fate-outcome-dynamics/DECISION_MEMO.md`](../ideaspark_run/cell-fate-outcome-dynamics/DECISION_MEMO.md) §9, which were never applied to revision 3.

Origin: PI meeting notes on transcriptomics -> cellular phenotype / cell death; brainstorming over
repository state; three reviewer passes; the L0 -> Gate 4 literature review and its decision memo.

## 0. Revision History and Withdrawn Claims

**Revision 1** corrected: Chronos/GeneEffect is not raw depletion; aggregate
consistency is not identification; bag-level output is not automatically identifiable;
T1/T2/T3 separated; transcriptome-decomposes-dynamics, generate-`B`, and
one-model-unifies-all downgraded to hypotheses; QC reformulated as observation-process
selection; SL moved out of the core question.

**Revision 2** fixed: the primary unit was inconsistent across sections (now two
explicit candidates); "outcome composition" was not a valid simplex (now trajectory /
multi-state estimands); the gap was technical, not biological (added a phenomenon-and-
value gate); all rivals were lumped into one null (split into utility and specificity
hypotheses, MNAR moved to an identifiability boundary); Gate 1's no-go was backwards
(an existing independent decomposition is ground truth, not a stop); the claim that
removal-rate independence bounds pre-capture loss was withdrawn.

**Revision 3** fixes:

| # | Defect in revision 2 | Fix |
| --- | --- | --- |
| 1 | Candidate A grouped "founder / lineage / clone" as one unit. Destructive sequencing plus lineage barcoding does **not** give same-founder longitudinal observation. | **Three evidence tiers within A**, each with its own claim ceiling (§4.1). |
| 2 | The single evidence hierarchy ranked condition-level anchors below lineage anchors — unfair to B, for which condition-level *is* the matching unit. Falsifiers were referenced but never written. | **Per-candidate evidence hierarchies** on a unit-matching principle, **per-candidate phenomena**, and **explicit falsifiers** (§4, §8). |
| 3 | The value gate justified decomposition by recovery / persistence / extinction — all of which are *inside* the trajectory estimand. Circular. | **Value tested on downstream endpoints outside the trajectory definition**, with predefined significance criteria (§2.1). |
| 4 | "Beyond net fitness" left the temporal relation of `F_net` to `Y_future` unspecified. If both come from the same window, the result is a retrospective conditional decomposition, not prospective prediction. | **Temporal contract stated and chosen** (§5.1); causal claim boundary added (§5.3). |
| 5 | A literature-only funnel plus a total ban on empirical work deadlocks: literature cannot establish that divergence is common/large/consequential in the target setting. Absence of evidence was being read as evidence of absence. | **Gate 2 trichotomy** (positive / powered-absence / insufficient), where insufficient -> **bounded pilot**, not stop (§8). §9 ban narrowed to production work. |
| 6 | §9 item 1 required L0 to contain the selected primary unit, but item 6 was where selection happened. Circular. | **L0 builds parallel maps for A and B; selection happens only at the decision point** (§8, §9). |
| 7 | Gate 4 carried the virtual-cell-baseline critique before generate-`B` was ever selected as a strategy. | Moved to a **post-selection tool-strategy review** (§10). |

**Revision 4** (this document, provenance: [`ideaspark_run/cell-fate-outcome-dynamics/DECISION_MEMO.md`](../ideaspark_run/cell-fate-outcome-dynamics/DECISION_MEMO.md) §9) applies eight amendments required by the L0 -> Gate 4 literature review's decision memo but never applied to revision 3:

| # | Section | Change |
| --- | --- | --- |
| 1 | §1 | The wedge is demonstrated for selected drug perturbations (Gross 2023); prevalence and importance remain unresolved for genetic loss-of-function — replacing the "mathematically true and biologically unproven" framing. |
| 2 | §2 | Chronos is structurally incapable of separating division from death because that decomposition lies outside its estimand — a readout limitation, not a tool defect. Retracts the earlier "unflagged identifiability hole" framing. |
| 3 | §4.1 | A1 exists (Live-seq) but is an existence proof, throughput-bound, not a deployable platform (~4-5 extractions/hour, ~300 high-quality transcriptomes in the entire study, 85-89% post-biopsy viability, small cell-cycle delays not excluded). A2 remains the ceiling for anything pooled. |
| 4 | §4 (Candidate B) | Adds an explicit falsifier: genetic LOF may act predominantly through division suppression, making the wedge substantially drug-specific. Enters as a hypothesis to test, not a finding. |
| 5 | §5.1 | DepMap (Cas9 knockout, Chronos-scored) is a different intervention from a CRISPRi experiment. DepMap supports "incremental beyond an external DepMap reference," not "matched on the true net fitness of the same intervention." Any study matching on an external `F_net` must measure its own achieved net effect. |
| 6 | §5.2 | Adds a context-dependence boundary condition (Nano 2023: Tjur R² 0.00-0.16 at baseline, rising to 0.70 under added stress). Corrects the exp02 grounding sentence: exp02 is evidence for the specificity hypothesis, not a headwind — with corrected Spearman figures (0.244 / 0.443 / 0.494 / 0.503). |
| 7 | §7 | Strengthens the generate-`B` row: Ahlmann-Eltze 2025 shows no deep perturbation model beats a simple linear baseline at generating `B` in the first place. |
| 8 | §12 | Appends four claim boundaries: the Live-seq viability caveat, the drug-vs-gene wedge transfer restriction, `|ΔF_net| ≤ 1 SD` is not equivalence, and Norman/Replogle are not prospective fate comparisons. |

Each amendment is applied in the section named. Revision 3 is frozen and retained as a
historical record.

## 1. The Preserved Wedge

```text
The same net fitness loss can arise from completely different cellular dynamics.
```

Strong division suppression with little loss, normal division with substantial loss,
early loss followed by survivor regrowth, and transient arrest followed by recovery
can all yield the same aggregate readout.

The wedge is **demonstrated for selected drug perturbations** (Gross 2023,
https://pmc.ncbi.nlm.nih.gov/articles/PMC10257663/); **its prevalence and importance
remain unresolved for genetic loss-of-function.** Whether such divergence is common,
reproducible, large, and consequential in real genetic perturbations is an empirical
question (Gate 2). Until it passes, this program has a motivating observation, not a
finding.

## 2. Premise, Gap, and Value

What existing readouts already do (**to verify by reading** — reviewer-supplied, not
yet read in full):

- **Chronos** fits an explicit population-dynamics model, converting sgRNA abundance
  change into a relative growth-rate effect of knockout. It is **structurally
  incapable of separating division from death, because that decomposition lies outside
  its estimand** — a **readout limitation, not a tool defect**. A tool is not defective
  for failing to identify a quantity it never claimed to identify.
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC8686573/)
- **GR / DIP-style metrics**, given time course and initial counts, can distinguish
  fully cytostatic from net cytotoxic responses under stated assumptions.
  (Hafner et al. 2016, https://pmc.ncbi.nlm.nih.gov/articles/4887336/)

Accurate premise:

> A single endpoint-abundance or net-fitness readout is generally insufficient to
> uniquely determine the underlying division, recovery, persistent-arrest, and
> cell-loss dynamics.

Accurate **biological** gap:

> It is unknown whether genetic perturbations with similar net fitness effects induce
> distinct, reproducible outcome trajectories, and whether early molecular states
> prospectively distinguish those trajectories.

### 2.1 The Value Question — Tested Outside the Trajectory Definition

Revision 2 justified the decomposition by its ability to explain recovery,
persistence, and extinction. Those are **components of the trajectory estimand
itself**; using them to prove the estimand is worth having is circular.

The value question must therefore be answered on **downstream endpoints that are not
part of trajectory construction**:

- long-term regrowth;
- rebound after perturbation withdrawal;
- durable clonogenic survival;
- resistance emergence;
- change in intervention prioritisation.

> **Value question.** Does a fate-resolved trajectory predict downstream outcomes that
> are not already contained in net fitness, and not already contained in the trajectory
> definition itself?

**Predefined significance criteria.** The words *common*, *large*, and *consequential*
must be assigned **minimum biological significance thresholds in the review memo before
evidence is collected**. They are not left to post-hoc judgement.

## 3. Ontology and Estimands (L0)

### 3.1 Retired Vocabulary

`proliferating / arrested / dying / mechanism / timing` mixes levels and is not a
simplex. `proliferating` is a rate; `arrested` is a window-defined, possibly reversible
state; `dying` is a process; death is an event; mechanisms overlap; timing is part of
the fate definition, not an extra label. **"Outcome composition" is retired** — a
lineage can simultaneously produce descendants that keep dividing and descendants that
are lost.

### 3.2 Required Structure

| Element | Requirement |
| --- | --- |
| Time horizon `T` | All outcomes defined over `[t0, t0+T]` from perturbation onset. No horizon, no estimand. |
| "Early" | Defined relative to `T` **and to fate commitment**, not to sample collection. Stated numerically per dataset. |
| Unit | See §4 — the open decision, resolved only at §9. |
| Censoring | Right censoring at `T` is explicit. |
| Denominator | Fractions are of *what*? Observed units at `T` are survivors of a branching-plus-loss process; fraction-of-observed is not fraction-of-founders. |

### 3.3 Estimands as Trajectories, Not Categories

**Lineage/clone unit** — endpoints over `[t0, t0+T]`: division history; alive-but-
non-dividing through `T`; recovery transition (arrest -> resumed division); **lineage
extinction**; descendant abundance at `T`.

**Population unit** — a **multi-state transition process** or an explicitly enumerated
set of trajectory summaries. Not a categorical mixture.

### 3.4 Two Kinds of "Loss" — Never Conflate

| Term | Meaning |
| --- | --- |
| **Biological loss / lineage extinction** | The lineage genuinely ends. A biological outcome. |
| **Assay attrition** | The unit is not observed: died pre-collection, lost in dissociation, failed capture, or removed by QC. A measurement process. |

### 3.5 Three Distinct Prediction Tasks

| Task | Statement | Type |
| --- | --- | --- |
| T1 | Is this cell **currently** in a dying/arrested state? | Terminal-state classification. A state readout, **not fate**. |
| T2 | What is the probability of division / persistence / recovery / extinction within `[t, t+D]`? | **Prospective prediction.** Requires longitudinal or lineage pairing. Not a counterfactual. |
| T3 | What would this same initial unit have done under a **different** perturbation? | Strict intervention counterfactual. |

Much of the apparent literature will be T1 presented as if it were T2.

## 4. The Two Candidate Research Questions

The primary unit is **not selected here**. Both candidates are carried in parallel with
separate estimands, evidence hierarchies, phenomena, and falsifiers. Selection happens
only at §9.

### Candidate A — Lineage / clone level

> Within a fixed biological context and time horizon, does an early post-perturbation
> molecular state predict the subsequent division, persistence/recovery, and extinction
> trajectory of **its linked lineage**, beyond an independently measured net fitness?

Estimand: §3.3, lineage endpoints.

#### 4.1 Evidence Tiers Within Candidate A — Different Claim Ceilings

Destructive sequencing plus lineage barcoding does **not** observe the same founder
before and after. It typically captures the state of a *clone member*, clone-level early-
state distributions, and the later behaviour of *siblings/descendants* sharing a barcode.
These are not the same evidence, and they do not license the same claim:

| Tier | Design | Highest claim supported |
| --- | --- | --- |
| A1 | **Same-cell prospective** — non-destructive state measurement on a cell whose own future is then observed (e.g. imaging-paired, or a genuinely non-destructive readout) | Per-cell prospective fate prediction |
| A2 | **Sibling / clone proxy** — one clone member is sequenced; siblings' futures are observed | Clone-level prospective association. **Not** per-cell fate. |
| A3 | **Clone-average** — clone-level early-state summary vs. clone-level outcome | Clone-average association only |

**A1 exists but is not a deployable platform.** Live-seq is a genuine existence proof
of same-cell prospective, minimally destructive measurement — but it is
throughput-bound: roughly 4-5 extractions per hour, and only ~300 high-quality
transcriptomes across the entire study. Post-biopsy viability is 85-89%, and the paper
does **not** exclude small cell-cycle delays; the biopsy itself is a small but
**non-zero** perturbation. Absent a system that closes this throughput and viability
gap, **A2 remains the realistic ceiling for anything pooled**, and the honest claim for
pooled designs is clone-level prospective association, not per-cell fate prediction.

#### 4.2 Candidate A — Phenomenon, Falsifier, Risk

- **Phenomenon required (Gate 2A):** within the same perturbation and context, does
  reproducible lineage-level trajectory heterogeneity exist, and is early lineage state
  associated with it?
- **Falsifier:** *under a reliable linked-lineage design and predefined detection
  limits, early state provides no reproducible incremental information about subsequent
  lineage trajectories.*
- **Risk:** no such data for genetic perturbation in K562 is present in this repository,
  and its existence at usable scale is unverified. May require data generation or
  collaboration.

### Candidate B — Population level

> Under comparable, independently measured net fitness, does the early single-cell state
> **distribution** provide incremental information about **independently measured**
> future population dynamics?

Estimand: §3.3, multi-state population trajectory summaries.

- **Phenomenon required (Gate 2B):** under matched net fitness, does reproducible
  divergence in population dynamics exist, and is it large enough to matter?
- **Falsifier:** *under matched-net-fitness conditions there is no sufficiently large
  and reproducible dynamics divergence, or the early state distribution provides no
  incremental information about it.*
- **Falsifier (drug-specificity risk):** *genetic loss-of-function may act
  predominantly through division suppression rather than increased loss, making the
  wedge substantially drug-specific.* This enters the program as a **hypothesis to
  test**, not a finding — the drug literature (§1) answers the wedge question
  affirmatively; the genetic literature has never asked it.
- **Explicitly forfeits** any per-cell or per-lineage fate claim, permanently.
  Matched-population evidence supports population-level association and can never be
  upgraded.
- **Risk:** requires an independently measured future-*dynamics* anchor. Endpoint
  viability alone does not qualify.

### 4.3 Falsification Is Not Absence of Data

**Lack of data does not falsify either candidate.** A candidate is falsified only by a
sufficiently powered study under a design capable of detecting the effect. "We could not
find the data" routes to §8's insufficient-evidence branch, not to a stop.

### 4.4 Terms Requiring Operational Definition Before Either Runs

- **"Comparable net fitness"** — matching tolerance; whether matching is on the point
  estimate or on its uncertainty.
- **"Reproducible"** — across replicates, datasets, or contexts. Different claims of
  different strength.
- **"Early"** — numerically, per dataset, relative to `T`. Live constraint: Replogle
  Perturb-seq is a **single, late** timepoint several days post-transduction. Whether
  *any* existing genetic-perturbation dataset supplies a genuinely early state is a
  Gate 1 question, not an assumption.

## 5. Hypotheses

Net fitness is a **summary of the outcome**, not an ordinary confounder. Generic stress
may be a **true fate precursor**, not a nuisance to residualize away. Timing may itself
be perturbation biology. Hence two hypotheses, tested separately.

### Utility hypothesis

> The early state provides **incremental prospective information beyond net fitness**.

Predictive null:

```text
Y_future  ⟂  S_early  |  F_net, X
```

`Y_future` is an **independently measured** future outcome, never derived from
`S_early`. `X` contains predefined context, timing, and measurement variables.

### 5.1 Temporal Contract for `F_net` — Choose One, State Which

If `F_net` is computed from the **same future window** as `Y_future`, then conditioning
on it uses future information. That analysis is still meaningful — it asks *"given the
same final net fitness, does state distinguish the underlying trajectories?"* — but it
is a **retrospective conditional decomposition**, not deployment-style prospective
prediction.

**Choice made in this spec (overturnable):**

| Analysis | `F_net` source | Claim ceiling |
| --- | --- | --- |
| **P (primary)** | An **independent, pre-existing** screen or replicate — DepMap/Chronos is exactly this relative to a new Perturb-seq experiment | **Prospective**: early state adds **incremental information beyond an external DepMap fitness reference** at prediction time |
| **R (secondary, clearly labelled)** | Same future window as `Y_future`; used for post-hoc matched analysis only | **Retrospective conditional decomposition.** Must never be reported as prospective prediction |

Any reported "beyond net fitness" result states which analysis produced it.

**DepMap is not the same intervention.** DepMap Achilles is genome-wide **Cas9
knockout**, scored by Chronos. A CRISPRi Perturb-seq experiment is a **different
intervention** — differing in perturbation strength, penetrance (hypomorph vs null),
kinetics, screening horizon, guide efficacy, and K562 subclone/culture conditions.
Analysis P therefore supports the claim *"incremental information beyond an external
DepMap fitness reference"* and does **not** support the stronger claim *"matched on the
true net fitness of the same intervention."* Any study that matches nominal pairs on an
external `F_net` **must measure its own achieved net effect** under the intervention
actually applied; unverified matching is a confound that can manufacture false
positives — a CRISPRi/knockout transport mismatch masquerading as trajectory
divergence. `|ΔF_net| ≤ 1 replicate SD` between nominally matched pairs is **not**
equivalence: SD is not SE, and equivalence requires a TOST against a predefined margin,
not a non-significant difference (§12).

### 5.2 Specificity hypothesis

> That information **cannot be reduced to a scalar response-burden or generic-injury
> score.**

Tested by residualizing on burden/stress scalars, not by discarding them — generic
stress stays biologically meaningful.

**Boundary condition.** Fate information is **context-dependent**, and a proximate
death-effector readout alone is insufficient to predict individual fate. Nano et al.
2023 (https://pmc.ncbi.nlm.nih.gov/articles/PMC9942801/) found that, at a matched
caspase-3 activation dose, prospective information about individual fate is near-zero
at baseline (Tjur R² 0.00-0.16), rising to R² = 0.70 only under added stress. This does
not speak directly to transcriptome-wide early state in K562 CRISPRi, but it warns
against assuming any single proximate readout is sufficient across regimes.

**Grounding (corrected).** exp02 is **evidence for** the specificity hypothesis, not a
headwind against it — the earlier "thin residual" reading was a misreading of our own
data. A generic response-magnitude (NAR) scalar alone reaches Spearman 0.244; NAR +
burden reaches 0.443; the best full-feature pseudobulk baseline (`delta_all`) reaches
0.494; the NAR-residualized transcriptome reaches **0.503** — higher than the
unresidualized baseline. See
[`docs/results/prior-internal-evidence.md`](results/prior-internal-evidence.md) for the
full table.

### 5.3 Causal Claim Boundary

Incremental predictive information does **not** establish that the early state is
causal, fate-committed, mechanistic, or manipulable. Controlling for response burden
supports **specificity**, not **mechanism**. No mechanistic or interventional language
is licensed by a positive utility or specificity result alone.

### 5.4 Scoped separately, not in the null

- **Two-way additive structure** (`GeneEffect(c,g) ~ gene_mean(g) + line_mean(c)`) gates
  claims of **context-specific residual** effects only. Run it when, and only when, a
  context-specificity claim is made. Grounding: exp09's non-pan-essential slice (CV3
  AUROC 0.645 -> 0.583, AUPR 0.651 -> 0.490).
- **MNAR observation bias** is an **identifiability boundary**, not a covariate (§6).

## 6. Observation Process as an Identifiability Boundary

State-dependent acquisition, dissociation, capture, and QC may induce
**missing-not-at-random** observation. Cells that die before collection are **never
observed**; no computational QC relaxation recovers them.

Claims that must not be made:

- High mito / low UMI is **not** a definitional signature of a dying cell. It also
  arises from dissociation stress, genuinely low-RNA states, ambient RNA, and technical
  failure; mito-rich clusters may reflect sample preparation
  (https://www.nature.com/articles/s41467-022-29212-9).
- "A cluster appears after relaxing QC" **cannot** be read as "the dying population was
  recovered."
- Independence between removal rate and perturbation strength **does not bound**
  pre-capture loss. Every perturbation could lose the same *fraction* of cells while
  losing biologically *different* cells; truly dead cells may vanish before any logged
  stage.
- A removal-rate/GeneEffect correlation is at most **evidence compatible with
  perturbation-dependent attrition** — not direct evidence of state-dependent biological
  loss.

Concrete loss-accounting probes, QC ablation designs, and the relevant exp01 datapoints
belong in a **separate measurement memo**, not in this science spec.

## 7. Downgraded Assumptions (hypotheses, not premises)

| Previously asserted | Now |
| --- | --- |
| The transcriptome can decompose division / arrest / loss dynamics | **The central hypothesis under test.** A snapshot measures state; it may reflect early fate commitment, generic stress severity, the consequence of already-executing death, or a residual after survivorship selection. |
| `B` is unobserved for most contexts, therefore `B` must be generated | **Invalid inference; withdrawn — and materially weaker than previously stated.** Generating `B` is warranted only if `B` carries independently verifiable outcome-relevant information. Ahlmann-Eltze 2025 (https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/) shows **no deep perturbation model beats a simple linear baseline at generating `B` in the first place**, undercutting the case for a generate-`B` strategy before outcome-relevance is even assessed. A **tool hypothesis**, reviewed only post-selection (§10). |
| One model unifies dependency, conditional essentiality, and SL | **Long-term hypothesis.** Dependency / conditional fitness retained as potential downstream relevance only. |
| Bag-level output solves identifiability | **Withdrawn.** More honest than an uninterpreted per-cell latent, but not identifiable without independent anchors. |
| Death mechanism (apoptosis / ferroptosis / necroptosis) | **Second-stage question, out of scope.** The plausible supervision route — perturbagens of known death mechanism — is recorded for later, not pursued now. |

## 8. Literature Review: Go/No-Go Funnel

### 8.1 Evidence Hierarchies — One Per Candidate

The governing principle is **not** `lineage > condition`. It is:

> **Direct prospective evidence at the same unit as the estimand** outranks **proxy
> evidence at another unit.**

| Rank | Candidate A (lineage/clone) | Candidate B (population) |
| --- | --- | --- |
| 1 | Same-cell prospective (A1) | Condition-level **paired prospective** anchor (state at `t0`, dynamics measured over `[t0, T]`) |
| 2 | Sibling / clone proxy (A2) | Condition-level anchor with partial time resolution |
| 3 | Clone-average (A3) | Cross-sectional condition comparison |
| 4 | Terminal-state classifier | Terminal-state classifier |
| 5 | Signature-only inference | Signature-only inference |

A finding's weight is capped by its tier under **its own** hierarchy. A terminal-state
classifier can never establish a prospective claim, however strong its metrics.

### L0 — Ontology, Latent-to-Observable Maps (both candidates in parallel)

Define with citations: state; fate; death; biological loss vs. assay attrition;
quiescence; senescence; recovery; time horizon; denominator. Build **parallel**
latent-to-observable maps for A and B. **L0 does not select the primary unit** —
selection happens at §9 item 6.

### Gate 1 — Measurement and Observation Validity

Chronos and CRISPR fitness scoring; GR / DIP metrics; birth-death decomposition; which
**combinations** of readouts separate division from loss, under which assumptions; what
is knowable about MNAR structure and what bounds exist. Also: does any existing
genetic-perturbation dataset supply a genuinely **early** state?

Output: the **measurement -> identifiable-quantity map**, per candidate.

**No-go:** if the target outcome cannot be defined and identified through **any**
credible independent anchor -> **narrow the estimand or stop.**

*Not* a no-go: existing readouts already decomposing dynamics. An independent
decomposition is **the ground truth the prospective question requires** — it removes
measurement novelty, not the biological question.

### Gate 2 — Phenomenon Prevalence and Biological Importance

Per candidate: Gate 2A (reproducible lineage-level trajectory heterogeneity within a
perturbation, associated with early lineage state) and Gate 2B (reproducible,
consequential divergence in population dynamics under matched net fitness). Judged
against the **predefined significance criteria** of §2.1, plus the value question on
downstream endpoints outside the trajectory definition.

**Literature alone usually cannot settle this.** Therefore Gate 2 returns a
**trichotomy**, not a binary:

| Finding | Route |
| --- | --- |
| **Positive evidence** | Proceed |
| **Sufficiently powered evidence of absence** | **Stop** (for that candidate) |
| **Insufficient evidence** | **Bounded validation pilot** with explicit budget and stop rules. **Not a stop.** Absence of evidence is not evidence of absence. |

### Gate 3 — Nearest Prior Art and Exact Novelty

Who has come closest, on which unit, with which anchor, at which evidence tier. Output:
a nearest-prior-work matrix, not a reading list.

### Gate 4 — Prospective Incremental Information

Do designs exist linking state to **future** outcome, per candidate's hierarchy? Enforce
T1-vs-T2 strictly.

Where no linkage design is found, **separate three distinct findings**:

1. none exists because it is **technically infeasible** -> stop;
2. none exists because the **data is unavailable to us** -> resourcing question;
3. none exists and this is a genuine **methodological opportunity** -> the best possible
   outcome.

### Decision

`proceed` | `narrow-or-pivot` | `stop`, per candidate.

## 9. Deliverables, Decision Point, and What May Run Before It

### Deliverables

1. Ontology / estimand memo (L0), with **parallel** maps for A and B.
2. Measurement -> identifiable-quantity map, per candidate.
3. Evidence table **supporting and challenging** the core premise.
4. Nearest-prior-work matrix.
5. 2-3 candidate research questions, each with falsifier, claim boundary, and explicit
   out-of-scope definition.
6. **Selection of one primary research question and its unit.**
7. Decision: `proceed` | `narrow-or-pivot` | `stop`.

### What May and May Not Run Before Item 6

> **No production modeling or large-scale data acquisition begins before selection.
> Model-agnostic, bounded work explicitly required to resolve a decision gate is
> permitted.**

| Permitted (decision-enabling) | Prohibited before selection |
| --- | --- |
| Dataset / metadata / access audit | Production modeling |
| Assay and linkage feasibility investigation | Model-specific feature engineering |
| Identifiability and power calculations | Foundation-model benchmarking |
| Predefined small-scale descriptive reanalysis | Large-scale or irreversible data acquisition |
| Bounded phenomenon-validation pilot, with explicit budget and stop rules, when Gate 2 returns *insufficient evidence* | — |

### Data-Acquisition Status

The earlier decision to "acquire dose-response data with real viability" is
**suspended, not cancelled.** Endpoint viability is an aggregate and supplies no
prospective trajectory anchor on its own. **Combined with time-resolved counts, division
tracking, and an independent death readout, it may still become a valid measurement
component.** Re-decided at the decision point.

## 10. Post-Selection Reviews (not part of the funnel)

- **Tool-strategy review** — runs **only if** the decision selects a generate-`B` /
  virtual-cell strategy. Carries the live critique that deep perturbation forward models
  barely beat trivial baselines. Reviewing it before generate-`B` is selected is
  premature.

## 11. Separate Memos (out of scope here)

- **Synthetic lethality.** SL is a joint-intervention effect relative to a combination
  null. Not the same question as single-perturbation outcome decomposition; it must stop
  driving this project's definition. The memo carries: exp09's cross-cell-line
  selectivity result and its collapse on the non-pan-essential slice; the resulting
  question ("how do we build a null that removes pan-essentiality so the residual is
  genuinely interaction?"); and the combination-null problem governing any multi-gene
  route, `interaction(a,b) = outcome(a,b) - psi(outcome(a), outcome(b))`, where the
  choice of `psi` is the crux.
- **Measurement memo.** Loss accounting, QC ablation design, attrition probes, and the
  exp01 cell-count datapoint.

## 12. Claim Boundaries

Extending `CLAUDE.md`'s terminology guardrails:

- Do not say population screens cannot separate death from arrest. Say a single endpoint
  net-fitness readout does not uniquely determine the underlying dynamics.
- Do not call DepMap GeneEffect a cell-death label. It is a relative growth-rate effect
  under an explicit population-dynamics model.
- Do not use "outcome composition"; state a trajectory or multi-state process estimand.
- Do not write "loss" without disambiguating biological extinction from assay attrition.
- Do not equate high-mito / low-UMI cells with dying cells.
- Do not describe a QC-relaxation-induced cluster as a recovered dying population.
- Do not claim observation bias is bounded by an independence check.
- Do not call a prospective outcome prediction (T2) a counterfactual; reserve that for T3.
- Do not report a same-window `F_net` result (Analysis R) as prospective prediction.
- Do not upgrade a sibling/clone-proxy (A2) or clone-average (A3) result into a per-cell
  fate claim.
- Do not upgrade a population-level (Candidate B) result into a per-cell fate claim.
- Do not infer causation, fate commitment, mechanism, or manipulability from incremental
  predictive information.
- Do not treat absence of data as falsification.
- Do not claim an outcome decomposition is identified when only aggregate consistency has
  been shown.
- Do not claim synthetic lethality without an explicit combination null and interaction
  residual.
- Do not cite Live-seq as "non-destructive" without the 85-89% post-biopsy viability
  caveat.
- Do not present a drug-derived wedge result (e.g. Gross 2023) as evidence for genetic
  perturbation.
- Do not treat `|ΔF_net| ≤ 1 SD` as fitness equivalence. SD is not SE; equivalence
  requires a TOST against a predefined margin.
- Do not describe Norman or Replogle as prospective fate comparisons; both supply only
  late survivor transcriptomes and aggregate growth.

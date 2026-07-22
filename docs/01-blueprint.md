# Research Blueprint: Generalizable Synthetic-Lethality Discovery by Virtual-Cell Composition

**Status:** established. This is the research contract — locked.
**Type:** research direction and claim boundaries. **Not** an implementation spec.
**Scope note:** this contract defines the active program. Prior evidence remains
under `docs/archive/`, `ideaspark_run/`, and `docs/results/`; those artifacts do
not define the current scope.
**Companions:** [`02-acceptance-criteria.md`](02-acceptance-criteria.md) (what
counts as passing) · [`03-literature-review.md`](03-literature-review.md) (related
work) · [`04-roadmap.md`](04-roadmap.md) (execution order).

## 1. The Problem

```text
Discover synthetic-lethal gene pairs that generalize to genes withheld from
SL-pair/graph training and to cancer cell lines not used to fit or select the
model.
```

Synthetic lethality (SL) is a pairwise and context-dependent fitness
interaction: disrupting either gene alone can be tolerated while disrupting both
is selectively deleterious in a particular cellular background. The candidate
space is too large for exhaustive screening, so the operational task is
**candidate prioritization for experimental follow-up**.

This program builds a general SL discovery model with two distinct evaluation
requirements:

1. **Benchmark-general gene-pair discovery.** Compare against the Feng et al.
   2024 model zoo on the official, SynLethDB-derived 9,845-gene benchmark. CV2 and
   CV3 test generalization to genes withheld from SL-pair/graph training.
2. **Cross-cell-line SL generalization.** Learn a context-conditioned score
   `s(a,b | c)` and evaluate it on cancer cell lines excluded from all model
   fitting and selection.

These requirements are complementary, not interchangeable. The main Feng2024
benchmark is not cell-line-specific and therefore cannot, by itself, establish
cross-cell-line generalization. Conversely, a result on one held-out cell line
does not establish competitive performance against the general SL-prediction
SOTA.

## 2. Premise, Gap, and Contribution

**Premise established locally.** Post-perturbation transcriptomic response carries
signal about single-gene fitness. The current exp05 forward model
(`src/aivc_model/`) maps a control state and gene identity to a predicted response
and then to DepMap GeneEffect. Its current implementation and evidence are K562
based; they are an initial backbone, not the scope of the SL problem.

**Gap.** Single-gene GeneEffect is not SL. It has no double-knockout quantity and
cannot identify an interaction merely by combining the two marginal
essentialities. A valid SL model must emit an explicit pairwise interaction and
must distinguish a general pair prior from a cell-line-specific effect.

**Contribution.** Compose a perturbation-response-trained virtual cell into a
pairwise interaction through an explicit non-interaction null, then test whether
that interaction:

- beats strong SL-prediction SOTA on the official Feng2024 cold-start splits;
- transfers across held-out cell lines when conditioned on cellular context;
- survives controls for pan-essentiality and context-free pair priors; and
- corresponds to measured genetic interactions rather than only curated labels.

Graph-free inductive reach is a design property, not the novelty claim.
CILANTRO-SL, RFM-SL, PARIS, and ESM4SL already occupy parts of that space. The
defensible methodological claim is the **perturbation-response-trained
composition, explicit interaction null, and context-resolved mechanistic
validation**.

## 3. Objects and Definitions

| Symbol | Meaning |
| --- | --- |
| $\mathcal{G}_F$ | The official Feng2024 9,845-gene benchmark universe. It is not a K562-specific assay. |
| $\mathcal{C}$ | Cancer cell-line contexts available for model development or held-out evaluation. |
| $q(a,b)$ | Context-agnostic SL pair score used for the Feng2024 benchmark. Swap-invariant. |
| $s(a,b\mid c)$ | Cell-line-conditioned SL score. Swap-invariant in $a,b$, but allowed to vary with $c$. |
| $F(X_c,g)$ | Virtual-cell forward model: control state from cell line $c$ plus perturbation gene $g$ to a predicted response. |
| $h_C(F(X_c,g))$ | Predicted single-gene GeneEffect or other declared fitness readout in context $c$. |
| $\psi$ | Explicit non-interaction null, such as additive or min/HSA, against which a joint effect is measured. |
| $D_{ab}$ | Feng2024/SynLethDB-derived pair label. A curated benchmark label, not a cell-line-specific SL measurement. |
| $Y_{ab,c}^{GI}$ | Measured genetic-interaction quantity for pair $(a,b)$ in cell line $c$. |

**GeneEffect boundary.** DepMap GeneEffect is a single-gene relative growth-rate
effect under a population-dynamics model. It is not a single-cell death label,
not a double-knockout observation, and not itself an SL label.

### 3.1 What each evaluation establishes

| Evaluation | Establishes | Does not establish |
| --- | --- | --- |
| Feng2024 CV1 | Pair-holdout diagnostic and topology sensitivity | Unseen-gene or cell-line generalization |
| Feng2024 CV2/CV3 | Semi-cold/complete-cold generalization to unseen genes | Cross-cell-line generalization |
| Held-out-cell-line evaluation | Transfer of `s(a,b | c)` to unseen cellular contexts | Competitive SOTA performance unless run against the same eligible baselines |
| Measured-GI evaluation | Correspondence with an observed interaction in the assayed context | Multi-cell-line mechanistic generality outside the assayed contexts |

## 4. Model and Composition Mechanism

The model has a context-free pair component and a context-conditioned interaction
component:

$$
s(a,b\mid c) = q(a,b) + \Delta_c(a,b).
$$

This decomposition is an evaluation guard, not an assumption that both terms are
nonzero. `q(a,b)` supports the official Feng2024 comparison. Before contextual
fitting, a final `q` artifact is built by a prespecified all-admissible-data retrain
or fixed fold ensemble using train-only-selected hyperparameters — never by
choosing the best test fold — and then frozen. `Δ_c(a,b)` must add
held-out-cell-line information beyond that fixed pair prior; joint fitting that
can move arbitrary signal between the two terms is inadmissible.

The SL graph may provide training labels for an explicitly reported calibrated
head, but it may not construct model features. Pure zero-shot composition and any
label-calibrated version are always reported separately.

### 4.1 Bridge A — counterfactual co-dependency

In cell line $c$, simulate loss of $a$ and ask whether $b$ becomes more essential:

$$
s_A(a,b\mid c) = \tfrac{1}{2}\left[
(\hat c_{b,c}-\hat c_{b\mid a,c}) +
(\hat c_{a,c}-\hat c_{a\mid b,c})
\right].
$$

The bridge uses single-gene fitness supervision but requires sequential
perturbation composition. It is an extrapolation until validated against measured
interactions.

### 4.2 Bridge B — virtual double knockout

Predict joint fitness and subtract an explicit null:

$$
s_B(a,b\mid c) =
\psi(\hat c_{a,c},\hat c_{b,c}) - \hat c_{ab,c},
$$

oriented so that joint-worse-than-null gives a larger SL score. Both additive and
min/HSA nulls are evaluated. Deep perturbation models often compress synergy, so
the virtual double-knockout is not presumed to work; simple linear/additive
ablations are mandatory.

### 4.3 From contextual scores to the Feng2024 score

The official benchmark has no verified cell-line label. A Feng2024 submission
therefore uses a declared context-agnostic score `q(a,b)`. If it aggregates
context-conditioned predictions, the aggregation rule and the cell lines used
must be fixed from training data only and applied identically to every fold. It
must not be described as a cell-line-specific prediction.

## 5. Hypotheses

### H1 — benchmark competitiveness

> On Feng2024 CV2 and CV3, `q(a,b)` improves per-anchor ranking over the best
> reproduced eligible SOTA and the dependency-only floor.

### H2 — cross-cell-line generalization

> On cell lines excluded from training and model selection, `s(a,b | c)` improves
> SL ranking over both direct transfer of `q(a,b)` and the strongest eligible
> context-free/context-only baselines.

Pooling held-out cell lines cannot rescue a failure on an individual line. Every
prespecified eligible held-out line is binding. Evidence on a small set of named
lines supports transfer to those lines; a population-level cross-cell-line claim
additionally requires a powered analysis that treats cell lines as inferential
units.

### H3 — pair and context specificity

> The gain remains after removing pan-essential pairs and cannot be reproduced by
> gene marginals, cell-line identity, lineage, or a context-free pair prior alone.

### H4 — mechanistic correspondence

> The composed interaction score correlates with measured continuous genetic
> interactions in the same cell-line context.

The K562 arm of Horlbeck 2018 and the small Adamson UPR set can test the mechanism
in K562. They cannot alone establish multi-cell-line generality. A stronger claim
additionally requires at least one eligible non-K562 measured-GI context; the
Horlbeck Jurkat arm is a candidate subject to a data/provenance audit.

## 6. Success Contract

Numeric and statistical rules live in
[`02-acceptance-criteria.md`](02-acceptance-criteria.md). At contract level:

1. **Feng2024 SOTA comparison:** reproduce SLMGAE and KR4SL, plus the relevant
   official model ladder, under the unmodified official benchmark splits and
   metrics. CV2/CV3 are primary; CV1 is diagnostic.
2. **Held-out-cell-line transfer:** train on multiple cell lines and evaluate on
   cell lines excluded from fitting, preprocessing decisions, calibration, and
   threshold selection. Audit task-data and available foundation-checkpoint
   pretraining provenance separately.
3. **Pair/context controls:** report non-pan-essential results and ablate the
   context-free pair prior, gene marginals, and context-only features.
4. **Mechanistic validation:** compare the interaction residual with measured GI
   in the matching cell line, with calibration and evaluation pairs disjoint.
5. **Integrity:** use train-only selection, fixed manifests, five official folds
   for Feng2024, and uncertainty intervals appropriate to anchors and cell lines.

## 7. Scope and Non-Goals

**Fixed scope:** a general SL partner-discovery model, evaluated first against the
official Feng2024 benchmark and then for transfer to held-out cancer cell lines.
K562 is the first implemented perturbation/fitness context and a mechanistic anchor,
not the target population.

**Non-goals:**

- claiming cross-cell-line generalization from Feng2024 CV2/CV3;
- calling a K562-mappable Feng subset a K562 SL assay;
- treating GeneEffect, essentiality, or a benchmark label as measured SL;
- using the SL graph to construct features;
- using held-out cell-line labels for preprocessing, checkpoint selection,
  calibration, or threshold tuning;
- treating randomly sampled unknown pairs as confirmed non-SL;
- aligning CRISPRa response data to knockout labels without a modality caveat;
- making patient or clinical-generalization claims without a separate protocol.

## 8. Claim Boundaries

- **Benchmark generalization and cell-line generalization are different axes.**
  Always name which axis was tested.
- **CV2/CV3 support unseen-gene claims only.** CV1 is a degree-gameable diagnostic.
- **“Unseen gene” means unseen to SL-pair/graph training unless otherwise
  qualified.** Report exposure to response data, GeneEffect, pretrained
  representations, and other auxiliary inputs separately.
- **A held-out-cell-line claim requires a cell line absent from all fitting and
  selection decisions.** A frozen K562-to-HCT116 single-gene GeneEffect audit is
  useful backbone evidence but is not pairwise SL generalization.
- **Candidate prioritization is not target validation.** Feng2024 labels are
  curated and Rand negatives are unconfirmed.
- **An explicit interaction is required.** Never infer SL from two single-gene
  essentiality scores alone.
- **A pan-essentiality lift is not an SL result.** The non-pan-essential slice is
  binding.
- **A benchmark rank is not a mechanism.** Mechanistic claims require matched,
  measured GI.
- **Measured GI is context-specific.** K562 correspondence does not prove
  non-K562 correspondence.
- **The virtual double knockout is an extrapolation.** It must beat simple
  additive/linear ablations before the machinery is credited.
- **A single fold or test-selected checkpoint is not a result.**

## 9. Locked Decisions

Changing any item below changes the research program.

1. **The task is general SL discovery, not K562-only prediction.** Learn a
   context-agnostic `q(a,b)` and a cell-line-conditioned `s(a,b | c)`.
2. **The first comparison is the official Feng2024 benchmark.** Its main
   9,845-gene pair labels are not K562-specific.
3. **Gene and cell-line generalization are evaluated separately.** CV2/CV3 test
   unseen genes; held-out-cell-line splits test unseen contexts.
4. **K562 is an initial backbone and validation context, not the program scope.**
5. **The mechanism is an explicit pairwise interaction.** Single-gene GeneEffect
   remains a floor and source of supervision, never the SL definition.
6. **Bridge A and Bridge B are compared head-to-head**, with declared
   non-interaction nulls and simple baselines.
7. **The SL graph never enters feature construction.** Any label-calibrated head
   is separated from zero-shot composition.
8. **The final context-free `q(a,b)` is frozen before contextual fitting.** It is
   retrained on all admissible calibration data or formed by a prespecified fold
   ensemble after train-only hyperparameter selection; no test-fold checkpoint is
   selected. The same artifact is the transfer baseline for every held-out line.
9. **The Feng2024 SOTA bar is set by reproduced strong methods**, centered on
   SLMGAE and KR4SL; KG4SL remains a reference rather than the assumed leader.
10. **Cross-cell-line claims require untouched held-out cell lines** and must beat
   context-free transfer baselines per line.
11. **The win must survive non-pan-essential and context-specificity controls.**
12. **Measured epistasis is the mechanistic anchor.** K562 data support only a
    K562 mechanistic claim; a multi-cell-line mechanistic claim needs non-K562 evidence.
13. **Acceptance criteria are frozen before formal evaluation** and cannot move
    to fit a result.

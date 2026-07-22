# Acceptance Criteria (Pre-Evaluation Contract)

**Status:** claim structure frozen. Dataset-specific minimum effect sizes,
eligibility thresholds, power targets, and hierarchical estimators are the only
open registrations; Phase 0 must freeze them before any formal model result is
opened. They are **not revisable to fit a result**.
**What this is:** the bar a result must clear to establish benchmark
competitiveness, cross-cell-line generalization, pair/context specificity, and
mechanistic correspondence.
**Governs:** [`01-blueprint.md`](01-blueprint.md),
[`04-roadmap.md`](04-roadmap.md), every result note, and every paper claim.

## 1. Claim axes

| Claim | Criterion | Required evaluation |
| --- | --- | --- |
| Competitive general SL discovery | **BEAT-SOTA** | Feng2024 CV2 and CV3 |
| Generalization across cancer cell lines | **CELL-LINE-GENERALIZATION** | untouched held-out cell lines |
| Pair- and context-specific signal | **SPECIFICITY** | non-pan-essential and context ablations |
| Biologically grounded interaction | **MECHANISTIC** | matched measured GI |
| Admissible result | **INTEGRITY** | leakage, selection, fold, and uncertainty audit |

No single axis substitutes for another. In particular, Feng2024 CV2/CV3 do not
test held-out cell lines, and K562 measured-GI correspondence does not establish
multi-cell-line generality.

## 2. Operational definitions

### 2.1 Feng2024 benchmark harness

Use the official Feng2024/SynLethDB-derived 9,845-gene universe, official cached
five-fold CV1/CV2/CV3 splits, declared negative-sampling regime, and official
`cal_metrics` implementation. The primary comparison is `Rand` 1:1; `Exp`, `Dep`,
and other positive:negative ratios are sensitivity analyses and are never pooled
with the primary result.

The main benchmark is **not cell-line-specific**. No result on it may be labelled
K562 SL or cross-cell-line generalization. A locally derived K562-DepMap-filtered
subset is a coverage/ablation dataset only and cannot replace the official SOTA
comparison.

### 2.2 Metrics

**Per-anchor NDCG@10** is the primary ranking metric. Report MAP@10,
NDCG@{20,50}, AUROC, AUPR, and F1 alongside it. Classification gains do not count
as ranking wins, and vice versa.

### 2.3 SOTA reference

The formal bar is the best eligible method reproduced under the identical
official harness. The required ladder includes **SLMGAE** and **KR4SL**, plus
KG4SL and the other official methods needed to verify the best comparator.

Published Feng2024 Rand 1:1 results are orientation, not a substitute for local
reproduction. Known local K562-filtered results and any corrupted published
ranking columns are inadmissible as the formal general-benchmark bar.

### 2.4 Held-out cell line

A cell line is held out only if its response data, GeneEffect labels, SL/GI
labels, and derived statistics were not used for:

- representation or model fitting;
- feature selection or normalization decisions;
- checkpoint or hyperparameter selection;
- calibration, threshold choice, or null selection; or
- manual selection of the reported model variant.

The manifest assigns every foundation checkpoint or pretrained representation one
of three exposure states for each target line: **verified absent**, **known
present**, or **unknown/unauditable**. Only verified absent permits an
unseen-context claim. Known present and unknown exposure permit differently
qualified task-data-held-out claims.

Development requires at least two training cell lines and at least two
prespecified eligible held-out lines. Every eligible held-out line is binding and
reported individually. With only a small number of lines, the verdict is limited
to those named contexts. A broader population-level claim requires a prospective
power analysis and an estimator that treats cell lines, not anchors, as the
inferential units. If suitable pairwise labels are unavailable, the criterion is
**not evaluable**, not passed by single-gene GeneEffect transfer.

### 2.5 Cell-line label eligibility

A pairwise dataset is eligible only after a label contract freezes, before model
evaluation:

- the positive or continuous relevance rule and sign convention;
- the candidate universe and treatment of unmeasured pairs (never implicit
  negatives);
- minimum anchors, positives per anchor, and coverage required by a prospective
  power/simulation analysis;
- assay, intervention, time-scale, study, and batch compatibility;
- source-level provenance against SynLethDB/Feng2024 and all calibration labels;
  and
- a study/batch-confounding sensitivity analysis.

If a held-out line or assay contributed labels to SynLethDB or another calibration
source, **all labels derived from that line/assay are purged from `q` and
contextual calibration before fitting**. Formal evaluation then uses pair-disjoint
records from the held-out assay. If source lineage cannot be mapped or the purge
cannot be verified, the affected line is ineligible for the formal verdict.

### 2.6 Pan-essential and context-free controls

A pair is pan-essential if either gene is on the declared DepMap
common-essential list for the release in use. The non-pan-essential slice contains
pairs for which neither gene is common-essential.

After train-only hyperparameter selection, the final Feng2024 `q(a,b)` artifact is
created by a prespecified retrain on all admissible calibration data or a fixed
ensemble of all fold models — never by choosing the best test-fold checkpoint.
It is frozen before contextual fitting and used unchanged for every held-out
line. The contextual model may only learn the increment beyond that fixed score.
A cell-line-conditioned gain must exceed:

1. `q(a,b)` transferred unchanged;
2. gene-marginal dependency/essentiality features;
3. context-only features such as cell-line or lineage identity; and
4. the strongest eligible published/reproduced contextual baseline available for
   the same data contract.

### 2.7 Measured genetic interaction

MECHANISTIC uses a continuous interaction score from a combinatorial perturbation
assay in the same cell line as the prediction. The currently planned K562 evidence
is the K562 arm of Horlbeck 2018 plus Adamson UPR. It does not prove
cross-cell-line mechanism. At least one eligible non-K562 measured-GI context is
required for a multi-cell-line mechanistic claim; Horlbeck's Jurkat arm is a
candidate only after its data contract and independence are verified.

Jost/Replogle dual-sgRNA is single-gene knockdown-efficacy data, not a GI dataset.
Norman CRISPRa is auxiliary and must retain the modality caveat.

## 3. BEAT-SOTA — official benchmark win

> On both Feng2024 CV2 and CV3, the model must exceed the best reproduced
> eligible SOTA and the dependency-only floor in per-anchor NDCG@10.

| Claim | Requirement |
| --- | --- |
| Win **EXISTS** | The paired improvement over the best SOTA has a 95% CI excluding zero on both CV2 and CV3. |
| Win is **PRACTICALLY MEANINGFUL** | The 95% CI lower bound exceeds a prespecified `delta_win` on both CV2 and CV3. `delta_win` is justified by a prospective candidate-ranking utility/power simulation and registered before formal model evaluation. |

Use synchronized paired resampling of anchors within each fold, equal-weighted
fold effects, and repeated training seeds. Report anchor uncertainty and
between-seed variability separately; the exact hierarchical estimator is frozen
with the power simulation. A point estimate above SOTA with a CI touching zero is
reported as “above the reproduced mean; difference not established.” A win on
CV2 but not CV3 is “one-SL-gene-cold only.” CV1 is always a diagnostic and is
inadmissible as the headline win.

## 4. CELL-LINE-GENERALIZATION — unseen-context win

> On untouched held-out cell lines, `s(a,b | c)` must improve per-anchor
> NDCG@10 over the strongest control defined in §2.6.

| Claim | Requirement |
| --- | --- |
| Named-context transfer **EXISTS** | Improvement has a 95% CI excluding zero on every prespecified eligible held-out line. The macro-average is reported but cannot rescue a line. |
| Named-context transfer is **PRACTICALLY MEANINGFUL** | Every line has a positive point estimate and the macro-average lower CI exceeds a registered `delta_context`, justified by prospective utility/power simulation before formal evaluation. |
| Population-level cross-cell-line evidence | The line-level hierarchical effect has a 95% CI excluding zero under a prospectively powered design that treats cell lines as the inferential units. |

Report the number of anchors, positive pairs, candidate partners, label source,
lineage, and coverage for every cell line. Pooled micro-averages are secondary;
they cannot hide failure on one line. With too few lines for line-level inference,
claims are limited to the named held-out contexts. A K562-to-HCT116 single-gene GeneEffect
transport result is backbone evidence only and does not satisfy this criterion.

## 5. SPECIFICITY — the gain is pairwise and contextual

Both parts bind:

1. **Non-pan-essential:** the relevant BEAT-SOTA and
   CELL-LINE-GENERALIZATION improvements retain a 95% CI excluding zero after
   removing pan-essential pairs.
2. **Context increment:** on held-out lines, `s(a,b | c) - q(a,b)` improves over
   zero and over the context-only and gene-marginal controls.

If the full-set gain disappears on the non-pan-essential slice, report
“pan-essentiality signal; pair-specific SL not established.” If the contextual
increment disappears, report “general pair prior transferred; cell-line-specific
SL not established.”

## 6. MECHANISTIC — correspondence with measured GI

> The composed interaction residual must correlate with measured GI in the
> matching cellular context on evaluation pairs disjoint from calibration data.

| Claim | Requirement |
| --- | --- |
| Context correspondence **EXISTS** | Spearman has a 95% CI excluding zero in the prespecified SL direction within the assayed cell line. |
| Context correspondence is **PRACTICALLY MEANINGFUL** | The CI clears a prespecified `rho_min` justified by prospective power and biological utility, registered before formal evaluation. |
| Multi-cell-line mechanism | EXISTS in K562 and at least one eligible non-K562 context, with concordant direction and no pooled-only rescue. |

The small Adamson UPR set is qualitative and cannot alone certify EXISTS.
Without an eligible fitness-GI anchor, MECHANISTIC is “not evaluable.” Calibration
and measured-GI evaluation pairs/genes must be disjoint according to a manifest
frozen before labels are inspected.

## 7. INTEGRITY — admissibility

Every formal result must satisfy all of the following:

1. **Train-only selection.** Test folds and held-out cell lines are never used for
   model or protocol selection.
2. **Fixed manifests.** Gene, pair, cell-line, modality, and label-source roles are
   materialized and hashed before the formal run.
3. **Pretraining lineage disclosed.** Known foundation-checkpoint exposure to
   held-out genes, cell lines, assays, and label sources is audited; unknown
   exposure narrows the claim.
4. **Five official folds.** Feng2024 results report all five folds; partial folds
   are diagnostics only.
5. **Identical harness.** Method and comparators differ only in the scorer and
   declared model inputs, not splits, seeds, candidates, or metric code.
6. **Zero-shot separated.** Pure composition and any SL-label-calibrated head are
   separate rows.
7. **No cross-axis relabelling.** SL-pair/graph-gene cold-start, cell-line
   transfer, and mechanistic correspondence are named separately.
8. **Coverage disclosed.** Filtering is reported at the gene, pair, anchor, and
   cell-line levels; a coverage-filtered subset is not presented as the full
   benchmark.
9. **Estimator registered.** Paired resampling units, fold/seed aggregation, and
   cell-line-level inference are fixed before formal results are opened.

## 8. Verdicts

| Conditions met | Allowed verdict |
| --- | --- |
| BEAT-SOTA + SPECIFICITY + INTEGRITY | **Beats SOTA for genes unseen to SL-pair/graph training** |
| Named-context transfer + SPECIFICITY + INTEGRITY; checkpoint non-exposure verified | **Generalizes SL ranking to the named unseen cell-line contexts** |
| Named-context transfer + SPECIFICITY + INTEGRITY; checkpoint exposure known present | **Transfers SL ranking to task-data-held-out named cell lines; checkpoint pretrained on the target lines** |
| Named-context transfer + SPECIFICITY + INTEGRITY; checkpoint exposure unknown | **Transfers SL ranking to task-data-held-out named cell lines; pretraining exposure unknown** |
| Population-level cross-cell-line evidence + SPECIFICITY + INTEGRITY; checkpoint non-exposure verified | **Cross-cell-line generalization supported under the registered line-level design** |
| Population-level cross-cell-line evidence + SPECIFICITY + INTEGRITY; checkpoint exposure known present | **Cross-cell-line task-data-held-out transfer supported; checkpoint pretrained on target lines** |
| Population-level cross-cell-line evidence + SPECIFICITY + INTEGRITY; checkpoint exposure unknown | **Cross-cell-line task-data-held-out transfer supported; pretraining exposure unknown** |
| BEAT-SOTA plus an applicable cell-line verdict | **General SL discovery model with unseen-SL-gene and held-out-context evidence**, inheriting the most restrictive checkpoint-exposure qualifier |
| BEAT-SOTA + population-level cross-cell-line evidence + multi-cell-line MECHANISTIC | **Mechanistically grounded general SL discovery model**, inheriting the most restrictive checkpoint-exposure qualifier |
| Only K562 MECHANISTIC | **Mechanistic correspondence in K562** |
| Any required dataset unavailable | Relevant axis **not evaluable** |
| Neither predictive axis clears its bar | **Negative** |

Composite verdicts never erase exposure status. If target lines differ in
exposure, the combined claim uses the most restrictive state: known present is
reported explicitly, otherwise unknown dominates verified absent. “Unseen
context” is reserved for verified-absent pretraining exposure.

## 9. Named risks

- Feng2024 labels are curated and negatives are noisy; success remains candidate
  prioritization, not target validation.
- Published ranking columns may be inconsistent; the formal bar depends on a
  verified reproduction under one harness.
- Cell-line-specific SL labels may be sparse or heterogeneous. Sparse coverage
  widens uncertainty; it does not justify pooling away context.
- Deep perturbation models may underestimate synergy. Bridge B must be compared
  with additive/min and GenePert-style linear baselines.
- A non-K562 GI anchor may not be obtainable. In that case the multi-cell-line
  mechanistic claim remains explicitly unestablished.

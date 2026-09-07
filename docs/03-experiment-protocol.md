# Experiment Protocol: Held-Out-Cell-Line SL Ranking

Updated 2026-09-07. This is the **separate, unimplemented SL-pair protocol** under
[the research blueprint](01-blueprint.md). The current GeneEffect model has completed
training and testing; its [joint-training design](specs/2026-09-06-modular-joint-training-design.md)
and [results](results/joint_geneeffect_seed0/README.md) govern that experiment.
No SL head, out-of-fold SL feature set or SL performance result exists.

## 1. Objective and prediction unit

For each held-out cell line $c$, rank canonical unordered gene pairs $(a,b)$ using
its basal single-cell transcriptome and gene features. Output a positive-is-SL
score $s(a,b\mid c)=s(b,a\mid c)$. No measured GeneEffect or SL labels from $c$
may enter the deployable feature path.

The primary question is whether context-dependent predicted single-gene profiles
add SL-ranking value beyond pair/gene identity and pan-essentiality. Pair labels
score this question; the separate GeneEffect benchmark diagnoses the dependency
predictor. Neither outcome estimates a double-knockout interaction phenotype.

## 2. Benchmark and outstanding prerequisites

The [tracked split](../configs/benchmarks/context_screen_v2_split.json) and
[basal registry](../configs/benchmarks/context_screen_v2_basal_registry.json)
are the authorities. The [data card](data/sl-context-screen.md) documents source
filters, label inference and artifact provenance.

| Split | Context | ModelID | Positive | Negative |
| --- | --- | --- | ---: | ---: |
| train | K562 | ACH-000551 | 1,669 | 10,270 |
| train | Jurkat | ACH-000995 | 95 | 9,124 |
| train | OVCAR8 | ACH-000696 | 89 | 643 |
| train | HAP1 | ACH-002475 | 56,993 | 20 |
| train | HT29 | ACH-000552 | 235 | 7,322 |
| validation | A549 | ACH-000681 | 392 | 1,581 |
| test | 22RV1 | ACH-000956 | 38 | 580 |
| test | PC9 | ACH-000779 | 134 | 2,364 |
| test | HeLa | ACH-001086 | 170 | 2,364 |

The published table contains 94,083 rows across nine contexts. Its split removed
128 source-row groups crossing sides (359 rows); no retained source row or
canonical pair crosses sides. `source_row_id` only identifies one raw aggregate
record, not a study, so this does not prove study independence.

Before a first SL run, resolve and document the existing prerequisites:

- Complete per-context filter attribution for evidence-count mismatch and
  non-atomic context rejection; current failure counts overlap and are not additive.
- Verify model-ready basal inputs for each included context. The card records
  HAP1/22RV1 as source-registered only; registration alone does not prove usable caches.
- Fix the candidate-pair universe and report batch/source confounding and class
  counts. A549 validation is degenerate: all positives contain TRA2A and no
  negative does. HAP1 has only 20 negatives; 22RV1 has only 38 positives.
- Record PC9/HeLa as one aggregate-label cluster, while retaining separate scores.
  Their labels are SL-only: neither ModelID has GeneEffect in the pinned release.
- Freeze backbone/head selection, inner context groups, hyperparameters, random
  seeds and the minimum context-effect margin before accessing SL test labels.

These are unresolved experimental choices and data checks, not completed steps.
Do not silently change membership or remove difficult contexts. Any changed
benchmark must have an explicit version and its own documented comparison surface.

## 3. Inputs and supervision

### 3.1 Basal cells

Use the exact registered model's basal cells: raw UMI or documented count inputs,
not an informal-name match or substituted bulk profile. Fit transforms on eligible
training contexts and restore them for evaluation. Frozen Tx1 caches retain their
recorded pretraining and preprocessing provenance.

### 3.2 Response anchors

| ModelID | Cell line |
| --- | --- |
| ACH-000551 | K562 |
| ACH-000971 | HCT116 |
| ACH-000995 | Jurkat |
| ACH-000739 | HepG2 |

Use genetic-perturbation responses and non-targeting controls. Inner-fold exclusion
also removes an excluded context's response supervision. Response-condition holdout
is not cell-line holdout and does not imply unseen-gene evaluation.

### 3.3 Dependency labels

Use the pinned DepMap 26Q1 GeneEffect release. Fit gene means, variable-gene
membership and normalization on the eligible training cohort of each fit only.
The 226-line benchmark is a separate diagnostic experiment; its supervised
membership does not determine eligibility under the SL split.

### 3.4 Pair labels

The sole pair-label input is `data/SL_Benchmark_Formal/sl_integrated_pairs.csv`,
processed into `data/SL_Benchmark_Formal/derived/context_screen_v2/` according to
the data card. Preserve natural class counts, canonical pairs, `source_row_id`,
`screen_cluster`, `split` and `label_confidence=silver_inferred`.
Screened non-hits are not universal non-SL labels. Do not import GeneEffect,
co-dependency, Feng or measured-GI labels as additional SL ground truth.

## 4. Backbone fitting and selection

Use the current joint GeneEffect formulation: a fixed training gene mean plus a
predicted residual, Huber regression every update and recurring balanced response
supervision. The [joint design](specs/2026-09-06-modular-joint-training-design.md)
defines the implemented loss, epoch validation and minimum GeneEffect-loss selector.
The SL experiment must specify its own eligible dependency cohort and validation
surface before fitting; it must not import the 226-line split by convenience.

SL validation/test contexts are excluded from supervised fitting and fitted
preprocessing. Validation may select but never join the training cohort. Record
response loss and both component terms alongside dependency error and correlation;
improving one task is not evidence of improving the other.

The completed standalone `best.pt` is diagnostic evidence only. It neither supplies
the required SL exclusions nor the inner-fold models below. Inner fits that exclude
a response anchor need an explicit eligible-anchor configuration; the fixed
four-anchor standalone trainer cannot be reused unchanged for those folds.

## 5. GeneEffect Evaluation

The [GeneEffect metric definitions](01-blueprint.md#6-what-the-geneeffect-metrics-measure)
distinguish cross-gene absolute correlation from per-gene cross-context residual
correlation. One GeneEffect-covered SL test context cannot establish the latter;
the separate 27-line test is diagnostic evidence, not an SL test score.
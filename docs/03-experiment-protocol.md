# Experiment Protocol: STATE GeneEffect to SLIdR Benchmark

**Status:** fixed exploratory named-context protocol; no test results have been
inspected under this split.

## 1. Objective

Evaluate a two-stage pipeline on cell lines excluded from model fitting and
checkpoint selection:

```text
basal single-cell state + gene
  -> STATE-based GeneEffect prediction
  -> train-free SLIdR
  -> context-matched SL pair ranking
```

The benchmark answers two separate questions:

1. Does the backbone predict held-out-cell-line GeneEffect?
2. When predicted GeneEffect replaces measured GeneEffect in SLIdR, does the
   resulting ranking recover experimental SL hits in the same held-out line?

GeneEffect is a single-gene fitness quantity, not an SL label. Only the second
stage is evaluated against pairwise SL labels.

This is a new split, not a continuation of the historical 28/5/9 Tx1
GeneEffect registration. A549 was previously available to the GeneEffect head,
and seven lines previously used as test lines now enter training. Consequently,
all GeneEffect-head fitting and checkpoint selection must be rerun under this
protocol. Existing heads or selected checkpoints from the historical split are
inadmissible. A response-model checkpoint may be reused only if its provenance
shows that it was fitted exclusively on the four anchors in Section 2.2 and did
not use A549 or HT29 data.

## 2. Data Contract

### 2.1 Basal input and GeneEffect target

- Tahoe lines use their Tahoe-100M DMSO cells as basal single-cell input.
- K562, HCT116, Jurkat, and HepG2 use non-targeting/control Perturb-seq cells as
  basal input.
- GeneEffect supervision and evaluation use one frozen DepMap 26Q1
  `CRISPRGeneEffect.csv` release and ModelID-based joins.
- The source inventory is
  `../results/phase_a_tx1_20260724/cell_line_manifest.csv`. The split below
  supersedes that file's historical `role` column but not its identity, basal
  source, or coverage fields.

Every train, validation, and test line in this protocol has both a basal input
and DepMap GeneEffect. The Tx1 encoder was pretrained on Tahoe-100M and therefore
has known pretraining exposure to the Tahoe test lines. Their GeneEffect and SL
labels remain task-held-out; results must be described with this pretraining
exposure qualifier.

### 2.2 Perturb-seq supervision

The STATE response model is trained only on the four genetic-perturbation
anchors:

| ModelID | Cell line | Basal control |
| --- | --- | --- |
| ACH-000551 | K562 | non-targeting Perturb-seq cells |
| ACH-000971 | HCT116 | non-targeting Perturb-seq cells |
| ACH-000995 | Jurkat | non-targeting Perturb-seq cells |
| ACH-000739 | HepG2 | non-targeting Perturb-seq cells |

The initial benchmark does not add public Perturb-seq datasets beyond the
already curated anchor inputs. In particular, no A549 or HT29 perturbation data
may enter response-model training.

### 2.3 SL labels

The sole pair-label source is:

```text
data/SL_Benchmark_Formal/sl_integrated_pairs.csv
```

Use the preprocessing contract in
[`data/sl-context-screen-v1.md`](data/sl-context-screen-v1.md) and the generated
`derived/context_screen_v1/sl_context_pairs.csv`. Only the A549 and HT29 rows are
test labels. Labels from Feng2024, Horlbeck K562/Jurkat, DepMap co-dependency, or
any other source are not merged into this evaluation.

The negative class is an experimental screen non-hit in the named context, not
a universal non-SL assertion. The context assignments remain
`silver_inferred` as documented in the dataset card.

Before SLIdR eligibility filtering, A549 contains 392 positive and 1,701
negative labeled pairs; HT29 contains 235 positive and 7,412 negative labeled
pairs. These full-table counts describe the source label surface, not the final
SLIdR evaluation universe.

## 3. Fixed Cell-Line Split

The split unit is the cell line. No GeneEffect or SL label from a validation or
test line may be used to fit the response model or GeneEffect head.

| Role | Cell lines | Use |
| --- | ---: | --- |
| Train | 35 | fit the backbone only |
| Validation | 5 | select one checkpoint |
| Test | 2 | final GeneEffect and SL evaluation |

### 3.1 Training lines

The GeneEffect head uses 31 Tahoe lines plus the four Perturb-seq anchors. The
response module receives genetic Perturb-seq supervision only from the four
anchors.

| ModelID | Cell line | ModelID | Cell line |
| --- | --- | --- | --- |
| ACH-000178 | Hs 766T | ACH-000348 | RPMI-7951 |
| ACH-000389 | H4 | ACH-000496 | NCI-H1792 |
| ACH-000790 | SHP-77 | ACH-000793 | KATO III |
| ACH-000890 | SW 1271 | ACH-000950 | LoVo |
| ACH-000120 | CHP-212 | ACH-000138 | CFPAC-1 |
| ACH-000139 | Panc 03.27 | ACH-000148 | Hs 578T |
| ACH-000164 | PANC-1 | ACH-000222 | AsPC-1 |
| ACH-000311 | NCI-H2122 | ACH-000396 | J82 |
| ACH-000437 | SW 1088 | ACH-000493 | SNU-423 |
| ACH-000521 | NCI-H2030 | ACH-000558 | A-172 |
| ACH-000580 | C32 | ACH-000757 | A427 |
| ACH-000861 | HOP-62 | ACH-000900 | NCI-H23 |
| ACH-000916 | NCI-H1573 | ACH-000932 | SNU-1 |
| ACH-000957 | LS 180 | ACH-000958 | SW48 |
| ACH-000997 | HCT-15 | ACH-001039 | COLO 205 |
| ACH-001333 | C-33 A | ACH-000551 | K562 |
| ACH-000971 | HCT116 | ACH-000995 | Jurkat |
| ACH-000739 | HepG2 |  |  |

### 3.2 Validation lines

These five Tahoe lines are reused from the frozen validation registration at
`../configs/experiments/12_tx1_st_geneeffect/phase_d/validation_lines.json`:

| ModelID | Cell line | Lineage |
| --- | --- | --- |
| ACH-000463 | NCI-H460 | Lung |
| ACH-000601 | MIA PaCa-2 | Pancreas |
| ACH-000853 | NCI-H661 | Lung |
| ACH-000943 | RKO | Bowel |
| ACH-001190 | SK-MEL-2 | Skin |

Validation GeneEffect selects exactly one checkpoint. Validation is not used to
fit SLIdR thresholds or to select a favorable SL result.

### 3.3 Test lines

| ModelID | Benchmark context | Lineage | Stage-1 GeneEffect | Stage-2 SL |
| --- | --- | --- | --- | --- |
| ACH-000681 | A549 | Lung | evaluate | evaluate |
| ACH-000552 | HT29 | Bowel | evaluate | evaluate |

HeLa is excluded because it has neither a 26Q1 GeneEffect target nor a compatible
basal single-cell input in the frozen STATE/Tahoe manifest. PC9 and RPE1 are also
outside this protocol.

## 4. Backbone Training and Checkpoint Selection

1. Fit the STATE response module on the four Perturb-seq anchors.
2. Fit the GeneEffect head on the 31 Tahoe training lines and four anchors using
   basal input and training-line GeneEffect only.
3. At each checkpoint, predict GeneEffect for all five validation lines without
   adaptation.
4. Select the single checkpoint with the best validation-line macro mean of
   per-line residual Spearman correlation. Residuals subtract the training-line
   gene mean from both prediction and target; the gene mean is computed without
   validation or test labels.
5. Freeze the checkpoint and all preprocessing, gene filters, missing-value
   rules, and SLIdR parameters before generating A549 or HT29 predictions.

The frozen checkpoint generates GeneEffect predictions for all 42 lines in the
split, not only the two test lines. These predictions form the cohort matrix
required by SLIdR. No measured validation/test GeneEffect is substituted into
the primary predicted-GeneEffect matrix.

Raw GeneEffect Spearman, MAE, and RMSE may be monitored as secondary validation
metrics, but they do not override the residual-Spearman checkpoint rule.

## 5. Stage-1 Test: GeneEffect Prediction

Run the frozen checkpoint once on A549 and HT29. For each line, evaluate over the
prespecified intersection of model-output genes and non-missing 26Q1 GeneEffect
genes.

**Primary metric:** per-line residual Spearman correlation, using gene means
computed from training lines only.

**Secondary metrics:** raw Spearman correlation, MAE, RMSE, and predicted-versus-
observed variance ratio. Report A549 and HT29 separately and their unweighted
macro mean. Do not tune, calibrate, or select genes using either test line's
GeneEffect values.

## 6. Stage-2 Test: Train-Free SLIdR

SLIdR consumes the frozen GeneEffect predictions together with frozen DepMap
mutation/copy-number inputs. It is not trained on the SL label table.

The two target-line cohorts are fixed to the basal-covered lines below:

- **Lung:** NCI-H1792, SHP-77, SW 1271, NCI-H2122, NCI-H2030, A427,
  HOP-62, NCI-H23, NCI-H1573, NCI-H460, NCI-H661, and A549.
- **Bowel:** LoVo, LS 180, SW48, HCT-15, COLO 205, HCT116, RKO, and HT29.

Every cohort member must have basal input, a frozen GeneEffect prediction,
mutation/copy-number calls, and the partner gene in the common gene universe.
For a driver direction to be scoreable, its lineage cohort must contain at
least three altered and three reference lines under the frozen alteration rule.

For each target line:

1. Use the target's mutation/copy-number state only to determine which endpoint
   of an unordered pair is an eligible natural driver.
2. Compute the directional driver-to-partner SLIdR score from the target's
   lineage cohort. The alteration thresholds, multiple-testing rule, and score
   orientation must be frozen before test scoring.
3. If neither endpoint is an eligible driver, mark the pair `unscored`; do not
   impute a score. If both directions are eligible, retain both directional
   scores and use the better-ranked direction as the pair score under one frozen
   rule.
4. Report eligible-pair coverage before reporting ranking performance.

Pair eligibility is fixed before either GeneEffect arm is scored. It depends
only on the target alteration state, the three-versus-three cohort-size rule,
gene-name mapping, and availability in both frozen GeneEffect matrices. No
predicted or measured GeneEffect value, essentiality threshold, or SL label may
change this common pair universe.

This filtered set is declared the **mutation-gated SLIdR sub-benchmark**. It is
not treated as performance on the full unordered context-screen table. Report
the post-gating positive and negative counts alongside the source counts in
Section 2.3. If either test line retains fewer than 10 positives or 10 negatives,
its Stage-2 result is `not evaluable`; it may not be rescued by pooling the two
lines or by relaxing the gate after labels are inspected.

Two SLIdR inputs are then evaluated on this identical eligible pair set:

- **Primary:** predicted GeneEffect from the frozen STATE backbone.
- **Oracle diagnostic:** measured 26Q1 GeneEffect. This measures the loss caused
  by the GeneEffect prediction stage and is not a deployable result.

The SLIdR implementation is train-free, but its thresholds and filtering choices
are still hyperparameters. They must be copied from one declared implementation
or fixed without consulting A549/HT29 SL labels.

## 7. SL Metrics and Controls

The primary SL metric is AUPR computed separately for A549 and HT29 over scored
pairs. AUROC is secondary. Report the unweighted two-line macro mean only as a
descriptive summary; two lines do not support a population-level cross-cell-line
claim.

Also report:

- number and fraction of SL-label pairs eligible for SLIdR scoring;
- positive/negative counts among eligible pairs;
- predicted-GeneEffect SLIdR minus measured-GeneEffect SLIdR performance;
- a context-blind pair-score baseline evaluated on the same pair universe; and
- a mutation/copy-number-only baseline evaluated on the same eligible pairs.

Because the current SL table has no retained same-pair label reversals across
contexts, this benchmark tests transfer to named held-out contexts but cannot by
itself demonstrate recovery of context-dependent label reversal.

SLIdR estimates a directional, mutation-gated cohort association rather than a
target-specific virtual double-knockout interaction. Therefore this protocol can
support a named-context mutation-gated dependency-ranking result, but it does
not by itself satisfy the explicit pair-interaction or measured-GI mechanism
claim in [`01-blueprint.md`](01-blueprint.md).

## 8. Leakage Rules

- A549 and HT29 are absent from response-model fitting, GeneEffect-head fitting,
  checkpoint selection, calibration, feature selection, and threshold tuning.
- Validation lines select the backbone checkpoint only; they are never promoted
  into training after selection.
- Test GeneEffect may be read only by the Stage-1 evaluator and the explicitly
  labeled measured-GeneEffect oracle.
- Test SL labels may be read only by the final Stage-2 evaluator.
- Cell lines are joined by DepMap ModelID, never by informal name alone.
- Unknown or missing SLIdR scores are not converted to negatives.
- All test metrics are reported for both lines; neither line may be dropped after
  inspecting its result.

## 9. Required Outputs

One benchmark run must preserve:

```text
split_manifest.json
checkpoint_selection.json
geneeffect_predictions.csv
geneeffect_metrics.json
slidr_scores.csv
slidr_metrics.json
run_manifest.json
```

The run manifest records the git commit, input and checkpoint hashes, DepMap
release, exact line lists, gene universe, SLIdR configuration, and random seeds.
`slidr_scores.csv` records target ModelID, canonical pair, eligible direction(s),
predicted-GeneEffect score, measured-GeneEffect oracle score, label, and every
exclusion reason. Planned metrics are not results; results enter `docs/results/`
only after the frozen run completes.

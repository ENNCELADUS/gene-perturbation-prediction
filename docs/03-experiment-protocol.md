# Experiment Protocol: Held-Out-Cell-Line SL Ranking

Updated 2026-09-07. This is the **separate, unimplemented SL-pair protocol** under
[the research blueprint](01-blueprint.md). The current GeneEffect model has completed
training and testing once, seed 0; its
[joint-training design](specs/2026-09-06-modular-joint-training-design.md)
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
Do not silently change membership or remove difficult contexts. Because no model
has run, a registry expansion may replace the nine-context split in place; after
the first model run, any membership change requires a new benchmark version with
its own documented comparison surface. RPE1 stays excluded for exact-model basal
mismatch.

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

## 4. Backbone model, fitting and selection

![Context-conditioned GeneEffect architecture](../figures/geneeffect_architecture.png)

*The implemented GeneEffect backbone that the SL head builds on. (a) Frozen Tx1-3B encodes
128 sampled basal cells of line $c$ into $H_c\in\mathbb{R}^{128\times2560}$ and frozen ESM-2
encodes gene $g$ into $e_g\in\mathbb{R}^{1280}$; both are cached at preparation time.
(b) A trainable adapter maps $e_g$ to the perturbation token $p_g\in\mathbb{R}^{2024}$, and the
trainable STATE transition model, initialised from the ST-HVG-Replogle checkpoint and run on
64-cell windows, predicts post-perturbation HVG expression $\hat Y_{g,c}\in\mathbb{R}^{128\times2000}$.
(c) Response descriptors compare $\hat Y_{g,c}$ with the matched basal HVG expression $X_c$:
the 4000-d mean/variance shift $\Delta$ is reduced to 256 dimensions by one fixed seeded
projection, plus six scalar summaries $s$. Concatenated with the fixed covariates
$q_{g,c}$ (3 basal statistics of $g$ in $c$), $e_g$ and $z_c$ (mean- and variance-pooled $H_c$,
5120-d) and three coverage masks, the 6668-d vector feeds the residual head, an MLP
6668 → 256 → 256 → 1 over train-fitted standardised blocks. The prediction is
$\hat y_{g,c}=\mu_{\text{train}}(g)+\hat\delta(g,c)$. (d) The joint objective. Source:
[`figures/geneeffect_architecture.drawio`](../figures/geneeffect_architecture.drawio);
vector exports sit beside it.*

The backbone is the current joint GeneEffect formulation: a fixed training gene mean
plus a predicted residual, Huber regression (delta 1) at every update and a balanced
four-anchor response replay (mean-shift MSE plus energy distance, weight 1) at every
fourth update. The [joint design](specs/2026-09-06-modular-joint-training-design.md)
defines the implemented loss, once-per-epoch validation and the minimum
`val_geneeffect_loss` selector; the blueprint
[§3–§4](01-blueprint.md#3-implemented-geneeffect-model) states the contract.
The SL experiment must specify its own eligible dependency cohort and validation
surface before fitting; it must not import the 226-line split by convenience.

SL validation/test contexts are excluded from supervised fitting and fitted
preprocessing. Validation may select but never join the training cohort. Record
response loss and both component terms alongside dependency error and correlation;
improving one task is not evidence of improving the other.

The completed seed-0 `best.pt` (epoch 3, test observed on 2026-09-07) is diagnostic
evidence only: it beats the context-blind gene mean by 0.0773% Huber and trails the
Tx1 context-PCA ridge on residual correlation
([result](results/joint_geneeffect_seed0/README.md)). It neither supplies the
required SL exclusions nor the inner-fold models of §5, and its 226-line test set
must not become an SL tuning surface. Inner fits that exclude a response anchor need
an explicit eligible-anchor configuration; the fixed four-anchor trainer cannot be
reused unchanged for those folds.

## 5. Out-of-fold feature generation

Every backbone-derived block of an **SL training** row — the profile, the
target-context values and $\mu_{\text{train}}$ — must come from one model that excluded
that row's context group from response supervision, dependency fitting and fitted
preprocessing, with the gene mean refit without it. Partition the train-side SL
contexts into inner groups, declare them before any run, and fit one model per group
under the finally selected hyperparameters; models from the hyperparameter search
are not reused. K562 and Jurkat are both SL training contexts and response anchors,
so their inner fits run with a reduced anchor set.

The profile reference cohort $\mathcal{C}_{\text{ref}}$ is the GeneEffect-labelled
cohort with **all nine SL benchmark contexts removed**; it is identical for every arm.
Fit the feature standardiser only on complete out-of-fold vectors. Mixing an
in-sample profile block with an out-of-fold context block inside one vector is
prohibited, and there is no non-out-of-fold fallback for Arm A.

The dominant cost is materialising $\hat\delta(g,c')$ for every benchmark gene across
every $\mathcal{C}_{\text{ref}}$ context once per inner model, on the order of
genes × contexts × cells-per-bag STATE forward passes over the once-built Tx1 basal
cache; predicted responses are reduced online, never cached to disk. Pin the bag size
(128 cells), the cell-sampling seed and the cache layout, and record a measured
GPU-hour estimate for one inner model before launching the full set.

## 6. SL head and controls

Per gene, summarise the residual profile over $\mathcal{C}_{\text{ref}}$ by nine frozen
statistics $\Sigma$: mean, population standard deviation, the 0.10/0.25/0.50/0.75/0.90
quantiles, and the fractions below −0.5 and below −1.0. Target-context features use
absolute GeneEffect $\hat y$; the pan-essentiality block uses the fixed
$\mu_{\text{train}}$ so it can be ablated separately.

```text
prof_g     = Sigma( ( deltahat(g, c') )_{c' in C_ref} )

phi(a,b|c) = [ prof_a + prof_b , |prof_a - prof_b| ,        18  invariant
               rho(deltahat(a,.), deltahat(b,.)) ,           1  invariant
               yhat_ac + yhat_bc , |yhat_ac - yhat_bc| ,     2  context
               psi_min(yhat_ac, yhat_bc) ,                   1  context
               mu_a + mu_b , |mu_a - mu_b| ]                 2  invariant

s(a,b|c)   = sigmoid( f_theta( phi(a,b|c) ) )
```

Twenty-four dimensions, three of which vary with $c$; every block is invariant under
$a\leftrightarrow b$. $\psi_{\min}=\min(\hat y_{a,c},\hat y_{b,c})$ is the declared
non-interaction null (highest single agent; GeneEffect is negative-is-lethal), so
$\psi$ used as a ranking score is negated. An additive null is excluded because it
duplicates the sum feature. Uncovered genes are masked explicitly, never zero-filled;
in Arm B a pair with a missing profile leaves the common universe.

The head minimises binary cross-entropy over train-side SL contexts,
**context-balanced** so each context contributes equally, with positives and negatives
reweighted inside each context; otherwise HAP1 alone (57,013 of 86,460 train rows,
20 negatives) dominates. Three arms score one identical pair universe: **Arm A**
out-of-fold predicted profiles, **Arm B** measured DepMap GeneEffect over the same
$\mathcal{C}_{\text{ref}}$ columns, **Arm B-full** measured over all DepMap columns as
the ceiling.

Controls, all reported as per-context lift on the same universe; none may remove a
context after its result is seen:

| ID | Control | Shortcut removed |
| --- | --- | --- |
| C1 | pair identity / train-context label frequency | pair memorisation |
| C1b | anchor-gene frequency, `max(r_a, r_b)` | gene-level memorisation C1 misses |
| C2 | $\psi$ alone, predicted and measured | ranking that is only the null |
| C3 | the $\mu_{\text{train}}$ block alone | pan-essentiality |
| C5 | the strongest residual control of the GeneEffect baseline ladder replacing the backbone | a backbone beating no simple prior |
| C6 | identically trained head with the three context dimensions ablated | a context claim with no context information |
| C4 | Arm B, Arm B-full | reference only, never a bar |

C1 is a lift, not an absolute bar: on the v1 benchmark it reached AUPR 1.0 on Jurkat,
HeLa and PC9, so the split, not a threshold, handles those contexts. C6 also includes
fixed deterministic context derangements and a permutation null. Declare the minimum
per-context incremental AUPR − prior attributable to the context block before reading
any test label; statistical distinguishability alone does not clear it, and below it no
context claim is licensed at any absolute AUPR.

## 7. Metrics

**GeneEffect.** The [metric definitions](01-blueprint.md#6-what-the-geneeffect-metrics-measure)
distinguish cross-gene absolute correlation from per-gene cross-context residual
correlation. The latter is scored on the separate 226-line benchmark under the
[Exp13 residual spec](specs/2026-08-17-exp13-geneeffect-residual-protocol.md), which this
document does not duplicate; that score is diagnostic, not an SL test score. Among SL
test contexts only 22RV1 is GeneEffect-covered; its per-context cross-gene Spearman is
reportable and cannot support a context claim. Report response loss and its two terms
for every inner model.

**SL.** Coverage and post-filter class counts precede every performance number. Report
AUPR − prior per test context and as a macro; AUROC is secondary. Stratify each AUPR
by whether both, one or neither endpoint appeared in SL training. The de-duplicated
diagnostic macro weights observable label clusters: PC9/HeLa share one cluster but keep
separate scores. Uncertainty uses a two-way dyadic bootstrap over both endpoints, 2,000
replicates per context; a one-way anchor-gene bootstrap is invalid because pairs have
two endpoints. Report Arm A − Arm B both in full and restricted to the context block;
only the restricted form isolates out-of-sample GeneEffect cost, since 21 of 24
dimensions are context-invariant in both arms.

## 8. Leakage rules

- Test contexts are absent from response training, dependency training,
  $\mu_{\text{train}}$, SL-head training, hyperparameter and checkpoint selection,
  standardiser fitting, calibration and thresholding.
- Validation contexts select only; they are never promoted into training.
- Join contexts by DepMap ModelID through the checked-in map and fail loudly on an
  unmapped context. Never join on informal name: DepMap's `CellLineName` for K562 is `K-562`.
- Fit the standardiser, any calibrator and any threshold on the training side only.
  **Per-context z-scoring of $\hat y$ is forbidden**: it consumes the test context's own
  distribution and erases the quantity under test.
- One common, label-independent pair universe across arms. Missing scores stay
  missing, never imputed to zero and never counted as negatives.
- Report every test context; none may be dropped after its result is inspected.
- Qualify every result with the Tx1 Tahoe-100M pretraining exposure of the test lines.

## 9. Required outputs

```text
context_screen_v2_split.json   copy of the tracked split file, with its hash
checkpoint_selection.json      per inner model: selection surface, epoch, hashes
geneeffect_predictions.csv     out-of-fold yhat, deltahat and mu_train per (gene, context)
geneeffect_metrics.json
sl_features.parquet
sl_scores.csv
sl_metrics.json
run_manifest.json
```

`run_manifest.json` records the git commit, input and checkpoint hashes, the DepMap
release, exact context lists and inner groups, the gene universe and variable-gene set,
all hyperparameters with the surface each was selected on, and the training, collation
and projection seeds. `sl_scores.csv` records context ModelID, canonical pair, arm,
score, label, endpoint-seen stratum and every exclusion reason. Planned metrics are not
results; a claim enters [`results/`](results/) only after the frozen run completes and
its integrity checks pass.

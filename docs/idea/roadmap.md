# Roadmap: Perturbation Transcriptomes to SL Target Prioritization

Status date: 2026-05-25

This project should be read as a staged research program, not as an end-to-end
synthetic-lethality predictor at the current stage. The current validated link is:

```text
single-cell perturbation response distribution
    -> population-level dependency ranking
```

The intended final link is:

```text
basal cancer context + candidate perturbation
    -> observed or predicted response distribution
    -> population-level dependency score
    -> context-specific SL-like target ranking
    -> known-SL / experimental validation
```

## Core Logic

| Node | Meaning | Current data source |
| --- | --- | --- |
| A: cancer context + perturbation | Cell line and target gene, e.g. K562 + gene knockdown | Replogle K562 CRISPRi perturbations; later DepMap/CCLE contexts |
| B: post-perturbation response distribution | Measured or predicted bag of single cells after perturbation | Replogle single-cell Perturb-seq; pseudobulk delta is the current baseline summary |
| C: dependency phenotype | Population-level fitness consequence of target loss | DepMap CRISPR GeneEffect |
| D: SL-like candidate | Target whose dependency is selective for a cancer context | Future multi-cell-line DepMap/CCLE/TCGA filtering |
| E: true SL evidence | Context-target or gene-pair interaction evidence | Future known SL databases, combinatorial screens, or wet-lab validation |

The first claim is deliberately narrow: perturbation-induced transcriptomes may
contain predictive signal about downstream dependency. DepMap GeneEffect is a
population-level fitness label, not a single-cell death label and not a strict
synthetic-lethality label. Therefore the supervised unit is a perturbation-level
bag, not an individual cell:

```text
B_c,g = distribution of post-perturbation cells
phi(B_c,g) -> GeneEffect(c, g)
```

## Phase 0: Data Alignment

Goal:

```text
Perturb-seq response for (cell line, perturbation gene)
    aligned to
DepMap GeneEffect for the same (cell line, gene)
```

Current implementation:

- Primary proof-of-concept: Replogle K562 CRISPRi essential screen.
- DepMap model: K562 / `ACH-000551`.
- Matched key: `(cell line, perturbation gene)`.
- Training rows: `1917` Replogle genes with numeric K562 GeneEffect labels.
- External validation source: Adamson K562 CRISPRi Perturb-seq.
- Auxiliary sources reserved for later response-only checks: Dixit K562 and Norman 2019.
- Norman is CRISPRa and should remain an auxiliary response benchmark, not direct knockout-dependency supervision.

Status: done for Replogle K562 v0.

## Phase 1: Observed Transcriptome to Dependency

Question:

```text
Given the observed post-perturbation response distribution B,
can we predict the population-level dependency phenotype C?
```

Completed experiment:

- Doc: `docs/experiment/baselines/replogle_k562_b_to_c_baseline/README.md`.
- Setup: Replogle K562 pseudobulk delta expression -> DepMap K562 CRISPR GeneEffect.
- Evaluation: `internal_cv_all`, 5-fold CV x 1 repeat, seed `42`.
- Best main row: `delta_all + pca50_random_forest`.

Key results:

| Model / check | Spearman | AUROC GE < -1.0 | Interpretation |
| --- | ---: | ---: | --- |
| `delta_all + pca50_random_forest` | 0.494 | 0.742 | Observed transcriptome contains moderate dependency signal. |
| `response_burden + ridge` | 0.426 | 0.705 | Generic response magnitude carries substantial signal. |
| `n_cells_only` | 0.000 | 0.498 | Signal is not explained by cell-count alone. |
| Target-masked Ridge vs full Ridge | -0.005 Spearman change | - | Signal is not dominated by direct target-gene expression leakage. |

Current conclusion:

The observed B->C bridge is viable, but current models compress each
perturbation's single-cell distribution into pseudobulk features. The next
modeling step should treat all cells under `(cell line, perturbation gene)` as a
bag and learn a set-to-label / multiple-instance predictor:

```text
{post-perturbation cells for (c, g)}
    -> response embedding
    -> GeneEffect(c, g)
```

This avoids pretending that each cell has an independent GeneEffect label and
lets the model test whether heterogeneity, rare stressed subpopulations, and
distribution shape add signal beyond mean delta expression.

Recommended first architecture: cell encoder + attention/summary pooling +
response-burden/program covariates + perturbation-gene embedding -> ranking and
GeneEffect regression head; add a context encoder when moving beyond K562.

## Phase 1b: Viability-Axis and Signal-Decomposition Audit

Question:

```text
Is the B->C signal mostly a generic viability / proliferation / response-burden axis?
```

Completed experiment:

- Doc: `docs/experiment/baselines/replogle_k562_viability_axis_audit_5x1_main.md`.
- Main audit: NAR Achilles/CTRP viability-axis scores.
- Follow-up: NAR + response-burden nuisance residualization and curated program scores.

Key results:

| Model / check | Spearman | AUROC GE < -1.0 | Interpretation |
| --- | ---: | ---: | --- |
| NAR viability score only | 0.244 | 0.623 | NAR viability axis alone does not explain the baseline. |
| NAR score + response burden | 0.443 | 0.714 | Generic response burden explains a large fraction of the signal. |
| NAR-residualized PCA50 RandomForest | 0.503 | 0.744 | Removing NAR viability scores preserves performance. |
| NAR+burden-residualized PCA50 RandomForest | 0.469 | 0.731 | Response burden is a real signal component. |
| Residual PCs + nuisance scores Ridge | 0.491 | 0.740 | Best view is mixed burden signal plus residual transcriptomic structure. |
| Sparse residualized Lasso | 0.113 | 0.559 | Raw sparse-gene modeling is not currently promising. |

Current conclusion:

The signal is not just a known NAR viability axis. It is also not clean,
perturbation-specific dependency biology. The safest interpretation is:

```text
current B->C signal =
    generic perturbation response / burden
    + residual transcriptomic structure
```

For Set-MIL, response burden, cell count, cell-cycle/state composition, and
program scores should be explicit covariates or nuisance controls, not hidden
shortcuts.

## Phase 2: External Response Validation

Question:

```text
Does the Replogle-trained B->C signal transfer beyond the original dataset?
```

Completed experiment:

- Doc: `docs/experiment/baselines/adamson_external_ensemble/README.md`.
- Setup: train model variants on Replogle K562 5-fold CV, then evaluate the
  mean prediction across fold models on combined Adamson K562.
- External feature pack: Adamson pilot, UPR epistasis, and UPR Perturb-seq,
  aggregated to `85` gene-level rows.
- Primary scope: `external_ensemble:adamson_k562`.
- Strict sensitivity scope:
  `external_ensemble_target_heldout:adamson_k562`, using only fold models whose
  Replogle train split did not contain the Adamson target gene.

Key results:

| Model / check | Adamson Spearman | Adamson R2 | AUROC GE < -1.0 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `pca50_ridge_alpha100` primary ensemble | 0.500 | 0.263 | 0.886 | Best external-transfer row. |
| `pca50_ridge_alpha100` target-heldout | 0.490 | 0.247 | 0.863 | Similar performance under stricter target-exposure sensitivity. |
| `pca50_random_forest_leaf3` primary ensemble | 0.385 | -0.143 | 0.791 | Replogle CV winner transfers worse than PCA Ridge. |
| `xgboost_depth4_lr0p03` primary ensemble | 0.024 | -0.170 | 0.586 | Nonlinear tree boosting does not transfer in this setting. |

Current conclusion:

The B->C bridge passes the first same-cell-line external-transfer gate. The
strongest current downstream model is not the Replogle CV winner; it is the more
regularized `pca50_ridge_alpha100`, which is less expressive but transfers
better to Adamson. This supports continuing the roadmap, with the caveat that
Adamson has only `85` gene-level rows and is UPR-biased.

The recommended next step after Phase 2 is **not** to move directly into AIVC or
other predicted transcriptomes. The safer next gate is an observed-transcriptome
MIL / Set-learning version of B->C:

```text
Replogle single-cell bag under perturbation g
    -> DepMap GeneEffect(g)
```

This directly tests whether single-cell heterogeneity adds dependency signal
beyond the current pseudobulk summaries before introducing forward-model error.

Status: completed for K562 Replogle -> Adamson observed-transcriptome transfer.

## Phase 2.5: Observed Single-Cell Set-MIL Dependency Predictor

Question:

```text
Does the full observed single-cell response distribution improve dependency
prediction beyond pseudobulk mean delta and response-burden baselines?
```

Primary supervised unit:

```text
Replogle single-cell bag under perturbation g
    -> DepMap GeneEffect(K562, g)
```

Required comparisons:

- pseudobulk delta + `pca50_ridge_alpha100`;
- pseudobulk delta + `pca50_random_forest_leaf3`;
- response burden + Ridge;
- target-masked pseudobulk baselines;
- `n_cells_only` negative control.

Required controls:

- keep one GeneEffect label per perturbation bag; do not assign independent
  dependency labels to individual cells;
- make cell count, response burden, cell-cycle/state composition, and program
  scores explicit covariates or nuisance controls;
- preserve target-heldout sensitivity where possible, especially for any
  perturbation-gene embedding;
- report internal Replogle CV and Adamson transfer, not just in-dataset metrics.

Decision rule:

```text
If Set-MIL does not significantly beat pseudobulk or burden baselines,
do not prioritize AIVC predicted-transcriptome experiments yet.
```

Rationale:

Observed pseudobulk B->C already shows signal, and external response validation
shows some cross-dataset stability. The next uncertainty is whether
single-cell heterogeneity, rare stressed subpopulations, or distribution shape
provide extra dependency signal. If they do not, then predicted transcriptomes
are a high-risk next layer because forward perturbation error will amplify the
downstream dependency uncertainty.

Status: next recommended modeling phase.

## Phase 3: Predicted Transcriptomes

Question:

```text
Can a forward perturbation model predict B well enough that predicted B still supports C?
```

Entry condition:

```text
Phase 2.5 shows that observed single-cell Set-MIL adds meaningful signal
beyond pseudobulk delta, response burden, and target-masked baselines.
```

Planned route:

```text
basal K562 state + candidate perturbation
    -> predicted post-perturbation transcriptome
    -> trained dependency predictor
    -> dependency ranking
```

Candidate forward models:

- simple additive / linear perturbation baselines;
- GEARS-style perturbation prediction;
- scGPT / STATE / other transcriptome foundation-model embeddings;
- Tahoe/LINCS-style resources only after the gene-perturbation benchmark is stable.

Required evaluation:

- compare observed-B -> C against predicted-B -> C;
- quantify how forward-model error affects dependency ranking;
- report whether top dependency candidates are stable under predicted transcriptomes.
- use the best observed-B downstream predictor as the primary downstream model;
- keep `pca50_ridge_alpha100` as the conservative pseudobulk comparator because
  it has the best Adamson external transfer, and keep
  `pca50_random_forest_leaf3` as an internal-CV comparator.

Status: deferred until the observed single-cell Set-MIL gate is tested. Current
experiments use observed transcriptomes only.

## Phase 4: Dependency to SL-Like Prioritization

Question:

```text
How do we move from DepMap GeneEffect to SL-like target ranking?
```

DepMap GeneEffect by itself is:

```text
(cell line, target gene) -> population fitness effect
```

SL requires context specificity:

```text
(cancer context A, target gene B) -> selective lethality / strong dependency
```

Planned bridge:

1. Expand from K562-only to multi-cell-line DepMap/CCLE contexts.
2. Define context groups such as mutation, copy-number loss, lineage, pathway state, or biomarker status.
3. Score target selectivity:

```text
dependency in A+ cell lines
    stronger than
dependency in matched A- / normal-like / less-vulnerable controls
```

4. Penalize pan-essential genes unless the therapeutic-window argument is explicit.
5. Use transcriptomic response features to explain or rank context-specific dependencies.

Output should be called `SL-like candidate prioritization`, not true SL discovery,
until validated against interaction evidence.

Status: design stage only.

## Phase 5: True SL Evidence

Question:

```text
Do top-ranked context-target candidates match real SL evidence?
```

Validation options:

- curated SL databases such as SynLethDB / SLKB;
- published validated context-target examples, e.g. BRCA1/2-HRD with PARP-axis targeting;
- combinatorial CRISPR or double-perturbation screens;
- drug-response plus biomarker-context evidence;
- targeted wet-lab validation if available.

Preferred validation target:

```text
For context A and target B:
    A alone is viable or already present in cancer
    B inhibition alone is not broadly lethal
    A + B produces selective fitness loss
```

Status: not completed.

## Current Research Verdict

The roadmap still holds, but only with staged claims:

| Claim | Current support |
| --- | --- |
| Observed perturbation transcriptomes predict K562 dependency better than trivial controls. | Supported by Phase 1. |
| The signal is not merely cell count or direct target-expression leakage. | Supported by Phase 1 checks. |
| The signal is partly generic response burden / viability-like biology. | Supported by Phase 1b. |
| The signal also contains residual transcriptomic structure beyond NAR viability scores. | Supported by Phase 1b. |
| The observed-transcriptome B->C signal transfers from Replogle to Adamson K562. | Supported by Phase 2, strongest with PCA Ridge. |
| Single-cell heterogeneity improves B->C prediction beyond pseudobulk summaries. | Not yet tested; this is the next A->B->C modeling target. |
| AIVC / forward-predicted transcriptomes should be the immediate next experiment. | Not supported; first test observed single-cell Set-MIL against pseudobulk and burden baselines. |
| Replogle internal CV winner is also the best external-transfer model. | Not supported; PCA RandomForest wins internal CV but PCA Ridge transfers better. |
| Predicted transcriptomes can replace observed transcriptomes for dependency ranking. | Not yet tested. |
| DepMap GeneEffect labels are true SL labels. | Not supported; they are dependency labels. |
| The pipeline can prioritize SL targets. | Plausible future layer, requires multi-context selectivity and external SL validation. |

The concise current conclusion is:

```text
Perturbation-induced transcriptomic responses contain measurable dependency
signal, but the current K562 evidence comes from pseudobulk summaries of
single-cell Perturb-seq. The next A->B->C question is whether a Set-MIL
dependency ranker can use the full observed response distribution to beat
pseudobulk PCA Ridge / RandomForest, response burden, and target-masked
baselines without losing Adamson transfer. If it cannot, moving directly to AIVC
predicted transcriptomes is high risk because forward-model error will further
amplify downstream uncertainty. SL claims still require forward-model error
analysis, multi-context selectivity, and true SL evidence.
```

## Immediate Next Steps

1. Build a Replogle Set-MIL dependency predictor: each perturbation gene is a
   bag of post-perturbation cells, and the label is the single DepMap GeneEffect
   for `(K562, target gene)`.
2. Compare Set-MIL against the frozen pseudobulk baselines:
   `pca50_ridge_alpha100`, `pca50_random_forest_leaf3`, `response_burden +
   ridge`, target-masked models, and `n_cells_only`.
3. Make the Set-MIL head ranking-aware: report GeneEffect regression, dependency
   ranking, GeneEffect `< -1.0` classification, and top-k enrichment.
4. Add nuisance controls for response burden, cell count, cell-cycle/state
   composition, and program scores so attention/subpopulation gains are not just
   generic stress shortcuts.
5. Add a graph-regularized perturbation-gene encoder after the plain Set-MIL
   baseline is established, using GO/PPI/coessentiality/pathway or protein
   complex priors to test held-out target generalization.
6. Keep the predicted-B pilot and multi-context SL-like ranking as later layers
   after the observed single-cell distribution -> dependency bridge is tested.

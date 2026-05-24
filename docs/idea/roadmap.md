# Roadmap: Perturbation Transcriptomes to SL Target Prioritization

Status date: 2026-05-24

This project should be read as a staged research program, not as an end-to-end
synthetic-lethality predictor at the current stage. The current validated link is:

```text
observed post-perturbation transcriptome
    -> DepMap GeneEffect / dependency ranking
```

The intended final link is:

```text
basal cancer context + candidate perturbation
    -> predicted post-perturbation transcriptome
    -> dependency score
    -> context-specific SL-like target ranking
    -> known-SL / experimental validation
```

## Core Logic

| Node | Meaning | Current data source |
| --- | --- | --- |
| A: cancer context + perturbation | Cell line and target gene, e.g. K562 + gene knockdown | Replogle K562 CRISPRi perturbations; later DepMap/CCLE contexts |
| B: post-perturbation transcriptome | Measured or predicted response state after perturbation | Perturb-seq pseudobulk delta expression |
| C: dependency phenotype | Population-level fitness consequence of target loss | DepMap CRISPR GeneEffect |
| D: SL-like candidate | Target whose dependency is selective for a cancer context | Future multi-cell-line DepMap/CCLE/TCGA filtering |
| E: true SL evidence | Context-target or gene-pair interaction evidence | Future known SL databases, combinatorial screens, or wet-lab validation |

The first claim is deliberately narrow: perturbation-induced transcriptomes may
contain predictive signal about downstream dependency. DepMap GeneEffect is a
population-level fitness label, not a single-cell death label and not a strict
synthetic-lethality label.

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
- Auxiliary sources reserved for later validation: Adamson K562, Dixit K562, Norman 2019.
- Norman is CRISPRa and should remain an auxiliary response benchmark, not direct knockout-dependency supervision.

Status: done for Replogle K562 v0.

## Phase 1: Observed Transcriptome to Dependency

Question:

```text
Given the observed post-perturbation transcriptome B,
can we predict the DepMap dependency phenotype C?
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

The B->C bridge is viable as a first-stage benchmark. It is not strong enough to
claim mechanistic SL discovery, but it is clearly better than trivial controls.

## Phase 1b: Viability-Axis and Signal-Decomposition Audit

Question:

```text
Is the B->C signal mostly a generic viability / proliferation / response-burden axis?
```

Completed experiment:

- Doc: `docs/experiment/replogle_k562_viability_axis_audit_5x1_main.md`.
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

## Phase 2: External Response Validation

Question:

```text
Does the Replogle-trained B->C signal transfer beyond the original dataset?
```

Next required experiments:

- Train on Replogle K562 and test on Adamson K562 where identifiers and modality are compatible.
- Use Dixit only as a small TF sanity check because it covers few unique targets.
- Keep Norman separate because CRISPRa activation does not directly match DepMap CRISPR knockout labels.
- Re-run burden, target-leakage, and nuisance-axis diagnostics on any external validation set.

Decision gate:

- If external K562 transfer preserves meaningful ranking signal, continue toward learned embeddings and forward perturbation models.
- If transfer collapses, treat Phase 1 as dataset-specific and prioritize representation / batch / modality robustness before SL claims.

Status: not completed.

## Phase 3: Predicted Transcriptomes

Question:

```text
Can a forward perturbation model predict B well enough that predicted B still supports C?
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

Status: not completed. Current experiments use observed transcriptomes only.

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
| Predicted transcriptomes can replace observed transcriptomes for dependency ranking. | Not yet tested. |
| DepMap GeneEffect labels are true SL labels. | Not supported; they are dependency labels. |
| The pipeline can prioritize SL targets. | Plausible future layer, requires multi-context selectivity and external SL validation. |

The concise current conclusion is:

```text
Perturbation-induced transcriptomic responses contain measurable dependency
signal, but the current K562 evidence is best interpreted as dependency
prediction with mixed generic response-burden and residual transcriptomic
structure. The path to SL target prioritization remains viable only after
external validation, forward-model error analysis, and context-specific
selectivity validation.
```

## Immediate Next Steps

1. Run Replogle -> Adamson K562 external validation.
2. Freeze a robust response representation: delta PCA, response-burden features, pathway scores, or embeddings.
3. Quantify CRISPRi-vs-CRISPR-KO and dataset-transfer effects.
4. Test predicted-B -> C using simple forward baselines before foundation models.
5. Build a multi-cell-line DepMap context-specific dependency table for SL-like ranking.
6. Validate top SL-like candidates against curated or experimental SL evidence.

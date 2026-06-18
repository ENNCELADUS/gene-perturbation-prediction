# K562 SL Pair Cross-Cell-Line Selectivity

Run status: implementation and CV1/CV2/CV3 5-fold official-metric run completed
2026-06-18 (module `src/sl_benchmark_baseline/`, selectivity mode). This is the
**exp06-lineage** cross-cell-line route; parent floor is exp06 (dependency-only).
Results in `results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/`
(gitignored).

## Goal

Test whether a cross-cell-line **Selectivity** contrast lifts gene-pair SL link
prediction over the exp06 single-K562 dependency-only floor, while staying inside the
DepMap GeneEffect-dependent evidence family (no transcriptome; that is exp07/08).

DepMap `CRISPRGeneEffect` is a single-gene KO screen keyed by
`cell_line x single_gene_KO`, so cancer-SL evidence comes from cross-cell-line
comparison: in cell lines where `gene_a` is defective, is `KO(gene_b)` more lethal than
in `gene_a`-intact lines? If so, `(gene_a, gene_b)` is SL-like.

```text
(gene_a, gene_b) + cross-cell-line Selectivity(a,b) -> P(SL) in [0, 1]
```

This is a **benchmark-adapter feature**, NOT a validated K562 SL assay. No
context-specific SL biological claim is made. `Rand` negatives only.

## Data

Benchmark input (unchanged from exp06):
`data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv`
(CV1/CV2/CV3, 1:1 balanced, 9,471-gene K562 universe, seed 17).

Cross-cell-line inputs (DepMap Public 26Q1, `data/sl_dependency_v0/raw/depmap/`,
gitignored). Cell-line axis = the 1,208 GeneEffect lines; K562 = `ACH-000551`. Genes
joined by Entrez id from `SYMBOL (ENTREZ)` headers.

| File | Role | Encoding |
| --- | --- | --- |
| `CRISPRGeneEffect.csv` | dependency `d_{c,b}` | float (Chronos) |
| `OmicsSomaticMutationsMatrixDamaging.csv` | damaging mutation | binary 0/1 |
| `OmicsSomaticMutationsMatrixHotspot.csv` | hotspot mutation (554 genes) | binary 0/1 |
| `PortalOmicsCNGeneLog2.csv` | copy-number loss | log2 (~1.0 neutral) |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | low expression | log2(TPM+1) |

## Method

**Composite-OR defective call.** For cell line `c` and anchor gene `a`,
`defective(c, a)` is true if ANY channel fires: damaging mutation, hotspot mutation,
copy-number loss (`CN_log2 < 0.8`), or low expression (`expr <= per-gene 10th
percentile`). Missing channels and NaN never fire. This composite clears the `n >= 20`
coverage bar (both groups) for 9,459/9,471 genes; damaging-mutation alone would clear
only ~8%, so the OR is necessary.

**Selectivity contrast** (larger = more SL-like):

```text
sel(a -> b) = mean[ d_{c,b} | a-intact ] - mean[ d_{c,b} | a-defective ]
```

Anchors with `< 20` defective or `< 20` intact lines get `sel = 0` and a coverage flag.
Optional pan-essentiality penalty `sel -= lambda * max(0, -pan_essential(b))`, lambda
default 0 (the model learns pan-essentiality from a feature instead).

**Swap-invariant features** (SL labels are symmetric; mirror exp06's sum/|diff|):

```text
sel_mean    = (sel(a->b) + sel(b->a)) / 2
sel_absdiff = |sel(a->b) - sel(b->a)|
pan_essential_min = min(pan_essential(a), pan_essential(b))
```

**Models.** `A_xcl` (logreg) and `B_xcl` (xgboost) use exp06's 5 GeneEffect scalars
(min/max/sum/product/|diff|) PLUS the 3 selectivity features (8 total). Baseline `A`/
`B`/`C` re-run unchanged. Standardization fit on train-fold only. The Selectivity
feature is computed from the external DepMap matrix and does not depend on the SL
train/test split — same leakage profile as exp06's K562 scalar (low risk under `Rand`).

**Diagnostics** (report-only): `non_pan_essential` slice (pairs where neither gene is
broadly essential, essential_fraction <= 0.5) and `covered_pairs` slice (both genes
cleared the `n >= 20` selectivity bar). If `_xcl` lift vanishes on the non-pan-essential
slice, the feature is re-encoding essentiality rather than SL.

Run command:

```bash
uv run python -m sl_benchmark_baseline \
  --input-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
  --output-dir results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run \
  --depmap-dir data/sl_dependency_v0/raw/depmap \
  --split-types CV1 CV2 CV3 --folds 0 1 2 3 4 --ranking-k 10 20 50
```

## Results

Full CV1/CV2/CV3 × 5-fold run (`run/summary.csv`, 405 rows). All values are
5-fold means on the `full_universe` slice unless noted. `A`/`B`/`C` reproduce the
exp06 floor (B f1 CV1 = 0.7304, matches exp06's 0.730).

**Classification (AUROC / AUPR), A_xcl vs A and B_xcl vs B:**

| Metric | Split | A | A_xcl | Δ | B | B_xcl | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUROC | CV1 | 0.621 | 0.651 | +0.030 | 0.795 | 0.818 | +0.023 |
| AUROC | CV2 | 0.620 | 0.650 | +0.030 | 0.704 | 0.742 | +0.039 |
| AUROC | CV3 | 0.617 | 0.643 | +0.027 | 0.596 | 0.645 | **+0.050** |
| AUPR | CV1 | 0.648 | 0.670 | +0.022 | 0.812 | 0.842 | +0.031 |
| AUPR | CV2 | 0.648 | 0.670 | +0.022 | 0.732 | 0.769 | +0.037 |
| AUPR | CV3 | 0.645 | 0.666 | +0.021 | 0.609 | 0.651 | **+0.042** |

**Ranking (NDCG@10):**

| Split | B | B_xcl | Δ |
| --- | --- | --- | --- |
| CV1 | 0.050 | 0.160 | +0.110 |
| CV2 | 0.042 | 0.086 | +0.044 |
| CV3 | 0.002 | 0.001 | −0.001 |

**Read.** The cross-cell-line Selectivity feature gives a consistent classification
lift over the dependency-only floor on all three splits, and the lift is **largest on
CV3** (the both-genes-cold-start surface where exp06's XGBoost collapsed to
AUROC 0.596). Top-k ranking improves markedly on CV1/CV2 but is flat/null on CV3:
cross-cell-line evidence helps *classify* cold-start pairs but does not fix cold-start
top-k *ranking*. The XGBoost head (`B_xcl`) uses the feature far more than the linear
head (`A_xcl`), consistent with a nonlinear interaction between the GeneEffect scalars
and the Selectivity contrast.

**Confound diagnostic (non-pan-essential slice, B_xcl AUROC / AUPR):**

| Split | AUROC full | AUROC non-pan-ess | AUPR full | AUPR non-pan-ess |
| --- | --- | --- | --- | --- |
| CV1 | 0.818 | 0.784 | 0.842 | 0.778 |
| CV2 | 0.742 | 0.701 | 0.769 | 0.680 |
| CV3 | 0.645 | 0.583 | 0.651 | 0.490 |

When pairs where either gene is broadly essential are removed, the lift **shrinks but
does not vanish** (CV1/CV2 stay well above the A baseline; CV3 AUROC 0.583 is near the
dependency-only floor and CV3 AUPR drops to 0.490). Conclusion: part of the Selectivity
signal is pan-essentiality-linked, but a genuine pair-specific component remains on
CV1/CV2. On CV3 the honest verdict is weaker — most of the cold-start classification
lift is attributable to essentiality structure, not pair-specific co-dependency.


## Parity gate

Baseline `A`/`B`/`C` reproduce the locked exp06 floor bit-for-bit when re-run through the
shared harness (asserted by `tests/test_sl_parity_gate.py::test_selectivity_mode_preserves_abc_parity`
on synthetic data; numerically spot-checked against
`results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv`
on the real run).

## Guardrails & caveats

- `Rand` negatives only — `Dep`/`Exp` negatives are built from DepMap covariation and
  would leak directly into the Selectivity feature. The run aborts on non-`Rand` input.
- Benchmark-adapter feature, not a validated K562 SL assay; no context-specific SL
  biological claim. Do not equate with true synthetic lethality.
- Same modality family as exp06 (DepMap GeneEffect-derived); the 3 omics define only the
  defective stratification, not a new label.


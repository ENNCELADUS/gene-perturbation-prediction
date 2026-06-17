# K562 SL Pair Dependency-Only MVP

Run status: official-metric implementation and CV1/CV2/CV3 rerun completed on
2026-06-17. Module `src/sl_benchmark_baseline/`; the previous
`results/experiments/06_k562_sl_pair_dependency_only_mvp/all_cv_run/` used obsolete
flat pair-level ranking metrics and should not be compared with the paper.

## Goal

First baseline for **gene-pair synthetic-lethality (SL) link prediction** on the
K562-mappable benchmark. Predicts the per-pair label `D = sl_label(gene_a, gene_b)`
using **only** the two genes' K562 DepMap GeneEffect scalars (`C`-style evidence) as
input. This is the floor that later observed-B / predicted-B / frozen-AIVC pair
features must beat. It is **not** DepMap GeneEffect regression (experiments 01-05).
See `CONTEXT.md` glossary entries `C` and `D`.

```text
(gene_a, gene_b) -> P(SL) in [0, 1]
```

## Data

Run input: `data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv`
(877,418 rows; 1:1 balanced; CV1/CV2/CV3). Its CV1 partition (331,730 rows) is
content-identical on key columns to the canonical CV1-only
`data/k562_SL_benchmark_minimal.csv`. Built by
`scripts/build_k562_sl_benchmark.py`; both genes carry a numeric K562 DepMap
GeneEffect value for `ACH-000551`. Negatives are `Rand` (random unknown pairs;
noisy non-SL, not confirmed).

CV split difficulty (standard SynLethDB / Feng 2024 convention):

| Split | Holdout | Difficulty |
| --- | --- | --- |
| CV1 | pair-level; both genes may recur in train | easiest |
| CV2 | one gene unseen in train | intermediate |
| CV3 | both genes unseen in train (cold-start) | hardest |

CV1 results are **not** evidence of held-out-gene generalization; CV2/CV3 are the
generalization surfaces.

## Method

**Features** (`features.py`): five swap-invariant functions of the two GeneEffect
scalars `ea, eb` — `min, max, sum, product, |diff|` — standardized on train-fold
statistics only. No raw `ea`/`eb` and no pair ordering is exposed.

**Models** (`models.py`):

| Id | Model | Inputs | Role |
| --- | --- | --- | --- |
| `A` | Symmetric logistic regression | 5 standardized features | Honest dependency-only floor |
| `B` | XGBoost (200 trees, depth 4) | same 5 features | Nonlinear "both-essential" interactions |
| `C` | Preferential-attachment degree probe | per-gene train-positive degree | Control: CV gameability from pair-degree alone |

`C` scores a test pair by `pos_degree[a] * pos_degree[b]` from training positives
only (min-max normalized per fold). It sees gene identity; A and B do not.

**Protocol** (`evaluate.py`): per `(split_type, fold_id)`, fit on `train` rows,
build a full score matrix over the K562-filtered candidate-gene universe, mask
train-positive pairs for ranking, then compute metrics aligned to
`data/SL_benchmark/src/preprocess.py:cal_metrics`. Seed 17.

Classification metrics are computed on sampled positive and negative test pairs:
AUROC, PR-curve AUC (`auc(recall, precision)`), and maximum F1 over the PR
curve. Ranking metrics are official per-anchor candidate-partner metrics, not a
flat test-pair list: NDCG@{10,20,50}, Recall@{10,20,50}, Precision@{10,20,50},
and MAP@{10,20,50}. Official-code Precision@k uses
`hits / min(k, n_positive_partners_for_anchor)`.

## Results

Official-protocol artifacts from the 2026-06-17 rerun:

| Split | Input CSV | Output directory |
| --- | --- | --- |
| CV1 | `data/SL_benchmark/derived/k562_depmap_rand_1to1/CV1_Rand_1to1_k562_depmap_pairs_balanced.csv` | `results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv1/` |
| CV2 | `data/SL_benchmark/derived/k562_depmap_rand_1to1/CV2_Rand_1to1_k562_depmap_pairs_balanced.csv` | `results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv2/` |
| CV3 | `data/SL_benchmark/derived/k562_depmap_rand_1to1/CV3_Rand_1to1_k562_depmap_pairs_balanced.csv` | `results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv3/` |

Each output directory contains `fold_metrics.csv`, `summary.csv`, and
`manifest.json`. The combined table is
`results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv`.
All three manifests record `candidate_gene_count = 9471`, `seed = 17`,
`ranking_k = [10, 20, 50]`, and
`official_metric_source = data/SL_benchmark/src/preprocess.py:cal_metrics`.

Paper-comparable result table:
`(NSM_Rand, CV_i, 1:1), i = 1, 2, 3`. Values are five-fold means rounded to
three decimals, matching the paper table format. Full mean/std values for all
metrics are in `official_metrics_summary.csv`.

| Models | F1 score CV1 | F1 score CV2 | F1 score CV3 | NDCG@10 CV1 | NDCG@10 CV2 | NDCG@10 CV3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: Symmetric logistic regression | 0.667 | 0.668 | 0.669 | 0.004 | 0.005 | 0.004 |
| B: XGBoost dependency-only | 0.730 | 0.676 | 0.670 | 0.050 | 0.042 | 0.002 |
| C: Degree probe | 0.823 | 0.667 | 0.667 | 0.197 | 0.001 | 0.001 |

## Interpretation

CV1 is dominated by topology and broad dependency signal. The identity-based
degree probe (`C`) is strongest on CV1 classification and ranking, confirming
that pair-level splits can be gamed by train-positive degree. This is a
diagnostic control, not a deployable biological model.

CV2 is the useful dependency-only floor for partly held-out genes. Nonlinear
GeneEffect features (`B`) beat logistic features (`A`) on classification
(AUROC 0.704 vs 0.620; AUPR 0.732 vs 0.648), but official ranking remains low
(NDCG@10 0.042). The degree probe collapses to AUROC 0.5 and near-zero ranking,
as expected when held-out genes have no train-positive degree signal.

CV3 is close to cold-start failure for dependency-only features. `B` falls to
AUROC 0.596 and NDCG@10 0.002; `A` remains weakly above chance in classification
but does not rank useful partners. Future observed-B / predicted-B / frozen-AIVC
pair features should be judged primarily by whether they improve CV2/CV3 under
this official protocol.

## Caveats

- The old flat-ranking output is obsolete. Paper comparison requires the
  official per-anchor ranking metrics implemented on 2026-06-17.
- The candidate universe for this K562 MVP is the K562-filtered gene universe
  present in the input table, not the full official 9845-gene universe with
  genes lacking K562 GeneEffect features.
- `Rand` negatives are unconfirmed non-SL. No SL biological claim is made; this is a
  benchmark-adapter floor.
- Using `GeneEffect(K562, g)` against `Rand` negatives is low leakage risk;
  switching to `Exp`/`Dep` negatives would leak against this feature and requires a
  separate leakage review.

## Next

1. Add observed-B / Deep Sets pair features and evaluate on **CV2/CV3**; beating
   A (0.620) and B (0.704) on CV2 is the first evidence transcriptomic signal adds
   information beyond essentiality.
2. Calibration / reliability check for A on CV2/CV3.
3. Compare future pair-feature models to the paper's Rand 1:1 table only with
   the official per-anchor ranking protocol and the same CV split boundary.

## Scope guard

CV1/CV2/CV3 only (no further splits); `Rand` negatives only; dependency-only
inputs (no observed-B/predicted-B/AIVC features yet); standalone module, no new
`vcc-dep-baseline` subcommand; pair-level adapter, no new cell lines and no SL
biological claims.

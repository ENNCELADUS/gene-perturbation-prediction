# K562 SL Pair Perturb-seq-Augmented Baseline

Run status: Tier-1 real-data run completed on 2026-06-18. Module
`src/sl_benchmark_baseline/` augmented mode ran CV1/CV2/CV3 with Replogle K562
gwps `pca_delta_meanpool`, zero fallback, and no coverage flag. Artifacts live
under
`results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag/`.
The with-coverage-flag, global-mean-fallback, and Tier-2 `scvi_delta_meanpool`
ablations remain pending.

## Goal

Second baseline for **gene-pair synthetic-lethality (SL) link prediction** on the
K562-mappable benchmark. Predicts the per-pair label `D = sl_label(gene_a, gene_b)`
using exp06's GeneEffect features **augmented with observed Replogle K562 gwps
Perturb-seq response embeddings per gene**. Measures head-to-head lift over exp06
under the identical official ranking + classification protocol.

This tests whether **observed single-cell transcriptomic perturbation response**
adds information beyond population-level DepMap essentiality for SL candidate
prioritization. It is **not** a transcriptome → GeneEffect regression (experiments
01-05) and does not train a forward perturbation model (experiment 05). See
`CONTEXT.md` glossary entries `C` and `D`.

```text
(gene_a, gene_b) + transcript_emb(gene_a) + transcript_emb(gene_b) -> P(SL) in [0, 1]
```

## The Coverage Problem (the crux)

Replogle K562 gwps covers **6,070 / 9,471 genes = 64.09%** of the K562-filtered
SL-benchmark candidate universe (see `docs/data/k562-perturbseq-sl-coverage.md`).
All other local CRISPRi sources combined add only **+4 genes** (6,074 total).

But this is a **pair** task. Under random pairing, P(both genes covered) ≈
0.64² ≈ **41% of possible pairs**. The completed no-flag run records an empirical
both-covered fraction of **51.17%** over the actual CV1/CV2/CV3 Rand 1:1
benchmark rows, so roughly half of evaluated rows still use a fallback embedding
for at least one gene. The pair-level fraction, not the 64.09% per-gene coverage,
is the practical constraint for transcriptomic pair features.

The chosen formulation (Option A: full benchmark + fallback) handles this by:
- Keeping every pair and the full 9,471-gene universe unchanged.
- Covered genes → real transcript embedding + `coverage_flag = 1`.
- Uncovered genes → fallback embedding (`zero` or `global_mean`) +
  `coverage_flag = 0`.
- The transcriptome block is purely **additive** over exp06, so any lift is
  attributable to the added features under identical CV splits and metric protocol.

Alternative formulations (both-covered subset only; two-tier primary/secondary
reporting) were considered and rejected for the MVP. See Stage-1 design discussion
for rationale.

## Data

Run input: same `data/SL_benchmark/derived/k562_depmap_rand_1to1/` CSVs as exp06
(CV1/CV2/CV3 Rand 1:1, 877,418 rows, 1:1 balanced). Same 9,471-gene candidate
universe, same seen-pair masking, same official metric protocol.

Transcriptome source:
**Replogle K562 gwps** (`data/external/replogle_k562_gwps_2023.h5ad`):
- 1,989,578 cells × 8,248 genes; 9,866 unique perturbation labels (9,866 parsed
  single-gene target conditions); 6,070 targets in the SL candidate universe.
- **CRISPRi** (loss-of-function), modality-aligned with DepMap CRISPR knockout.
- Per-gene response bag = set of control-centroid-subtracted per-cell embeddings
  for all surviving cells under that perturbation.
- See `docs/data/replogle-k562-gwps.md`.

CV split difficulty (unchanged from exp06):

| Split | Holdout | Difficulty |
| --- | --- | --- |
| CV1 | pair-level; both genes may recur in train | easiest |
| CV2 | one gene unseen in train | intermediate |
| CV3 | both genes unseen in train (cold-start) | hardest |

CV1 results are **not** evidence of held-out-gene generalization; CV2/CV3 are the
generalization surfaces.

## Method

### Features

Per gene `g`, construct a **transcript embedding** `e_g`:
- **Covered gene** (g ∈ gwps, 6,070 genes): real pooled embedding from the gwps
  response bag, plus `coverage_flag = 1`.
- **Uncovered gene** (g ∉ gwps, 3,401 genes): fallback embedding, plus
  `coverage_flag = 0`. Two fallback strategies are supported:
  - `zero`: a zero vector.
  - `global_mean`: the mean of all covered genes' pooled embeddings across the
    full candidate universe (NOT a per-fold train-set mean). This is stable
    across folds because gwps coverage is fixed and label-free — no SL `D` label
    touches the embedding or the mean — so it is treated as a fixed preprocessing
    step (equivalent to an unsupervised feature extractor), not a fold-local
    statistic. See `align_to_universe` in `embeddings.py`.

**Swap-invariant pair features** (exp06 is order-invariant, so new features must
preserve symmetry):
- **GeneEffect block** (5 features, unchanged from exp06): `min, max, sum, product,
  |diff|` of `(GeneEffect_a, GeneEffect_b)`.
- **Transcript embedding block** (3 × `embedding_dim` features): `e_a + e_b`,
  `|e_a - e_b|`, `e_a ⊙ e_b` (element-wise product).
- **Coverage indicator** (optional): `cov_min = min(flag_a, flag_b)` and
  `cov_max = max(flag_a, flag_b)`, preserving swap invariance while
  distinguishing none-covered, one-covered, and both-covered pairs.

All features standardized on train-fold statistics only (matching exp06's
`Standardizer`).

### Models — Two Tiers

**Tier 1 — MVP floor (directly comparable to exp06):**

| Model | Inputs | Role |
| --- | --- | --- |
| `A_transcript` | GeneEffect block + transcript block + optional coverage flags | Symmetric logistic regression, same head as exp06-A |
| `B_transcript` | GeneEffect block + transcript block + optional coverage flags | XGBoost (200 trees, depth 4), same head as exp06-B |

Both models are the **exact same estimators** as exp06 A/B, only the feature matrix
width changes. Any lift is attributable to the transcript block alone.

**Tier 2 — frozen exp03 representation:**

Reuse exp03's winning recipe (**scVI128 → frozen-GMM occupancy** or **pooled scVI
delta**) applied to gwps bags, but **pooled to one per-gene vector** instead of
exp03's bag→scalar regression. The pooled vector becomes `e_g`. No new
representation training; the embedding method is fixed from exp03, only the
pooling target changes (per-gene vector for pair features vs. scalar GeneEffect
prediction).

Concretely:
- Train scVI on gwps control + target cells (or reuse exp03's scVI model if the
  gene panel is compatible).
- For each covered gene's response bag: scVI latent → subtract control latent
  centroid → frozen-GMM occupancy features (64-dimensional if using GMM-K64) OR
  pool the centered latent directly (128-dimensional) → `e_g`.
- Uncovered genes: fallback as in Tier 1.
- Feed `e_g` into the Tier-1 LogReg/XGB models.

**Reporting variants for both tiers:**
- **With coverage flag:** include swap-invariant `cov_min` and `cov_max` columns.
- **Without coverage flag:** omit the coverage indicators entirely, so the model
  cannot shortcut via coverage-as-proxy-for-degree.

Tier-1 floor establishes whether transcriptome adds lift with the simplest
representation (PCA/HVG mean-pool). Tier-2 tests whether exp03's proven
representation transfers to the pair-ranking task.

### Protocol

Per `(split_type, fold_id)`:
1. Build a label-free per-gene transcript embedding table from the supplied
   `bags.npz` and align it to the fixed 9,471-gene candidate universe. Covered
   genes receive pooled observed embeddings; uncovered genes receive the selected
   label-free fallback. No SL labels touch this preprocessing step.
2. Build swap-invariant pair features for train/test pairs.
3. Fit LogReg/XGB on train pairs.
4. Build full score matrix over the 9,471-gene universe, mask train-positive pairs,
   compute exp06's `official_classification_metrics` and `official_ranking_metrics`
   (AUROC, AUPR, F1, NDCG@k, Recall@k, Precision@k, MAP@k for k ∈ {10,20,50}).
5. **Covered-pair diagnostic slice** (reporting-level cut, not a task change):
   Re-compute metrics restricted to test pairs where **both genes are gwps-covered**,
   to isolate the transcriptome signal from fallback dilution. This is a secondary
   metric block; the primary block is the full universe (head-to-head with exp06).

Seed: 17 (matching exp06). Standardization: train-fold only. Official metric
source: `data/SL_benchmark/src/preprocess.py:cal_metrics`.

**Primary baseline = exp06 re-run through the identical harness** (transcript block
zeroed / removed) to ensure the comparison is a true ablation of the same code path
and random state, not a cross-run artifact.

### How to run

The augmented mode is implemented in `src/sl_benchmark_baseline/` and triggered by
supplying `--bags-npz`. The reusable invocation is recorded in
`configs/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_cv1_cv2_cv3.yaml`
(the module reads CLI flags, not the YAML directly).

Completed Tier-1 no-coverage-flag run:

```bash
uv run python -m sl_benchmark_baseline \
  --input-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
  --bags-npz results/experiments/07_k562_sl_pair_perturbseq_augmented/gwps_pca_mean_bags/bags.npz \
  --embedding-method pca_delta_meanpool \
  --fallback-strategy zero \
  --no-coverage-flag \
  --split-types CV1 CV2 CV3 --folds 0 1 2 3 4 --ranking-k 10 20 50 \
  --output-dir results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag
```

Ablations (spec MLE flags A/C) are flag flips, not code changes:
- **With coverage flag:** rerun with `--include-coverage-flag` and a `_with_flag`
  output dir.
- **Tier 2 (frozen exp03 scVI representation):** rerun with the `scvi_delta` bags NPZ
  and `--embedding-method scvi_delta_meanpool`.
- **Fallback sensitivity:** rerun with `--fallback-strategy global_mean`.

Without `--bags-npz` the module behaves exactly like the exp06 dependency-only
baseline (models A/B/C, no transcript block).

## Results

Completed artifacts:

| Artifact | Path |
| --- | --- |
| Run manifest | `results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag/manifest.json` |
| Fold metrics | `results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag/fold_metrics.csv` |
| Mean/std summary | `results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag/summary.csv` |
| Mean-pooled gwps bags | `results/experiments/07_k562_sl_pair_perturbseq_augmented/gwps_pca_mean_bags/bags.npz` |
| Bag feature summary | `results/experiments/07_k562_sl_pair_perturbseq_augmented/gwps_pca_mean_bags/feature_summary.json` |

Run manifest highlights:

| Field | Value |
| --- | --- |
| Candidate genes | 9,471 |
| Input rows | 877,418 Rand 1:1 CV1/CV2/CV3 rows |
| Seed / folds / ranking k | 17 / 0-4 / 10, 20, 50 |
| Embedding method | `pca_delta_meanpool` over `single_cell_pc_delta` |
| Embedding dimension | 128 |
| Fallback | `zero` |
| Coverage flag | disabled |
| Covered genes | 6,070 / 9,471 = 64.09% |
| Both-covered benchmark rows | 51.17% |
| Official metric source | `data/SL_benchmark/src/preprocess.py:cal_metrics` |

The gwps feature cache contains 6,070 compact mean-pooled bags, 1,166,742 target
cells, 75,328 control cells, 2,000 HVGs, and 128-dimensional PCA-delta features
(`feature_summary.json`).

Primary full-universe metrics are five-fold means rounded to three decimals.
`A` and `B` are the in-harness exp06 dependency-only baselines rerun alongside the
augmented models; `A_transcript` and `B_transcript` add the observed gwps
transcript block.

| Split | Model | F1 | AUROC | AUPR | NDCG@10 | MAP@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CV1 | A | 0.667 | 0.621 | 0.648 | 0.004 | 0.003 |
| CV1 | A_transcript | 0.692 | 0.757 | 0.787 | 0.027 | 0.016 |
| CV1 | B | 0.730 | 0.795 | 0.812 | 0.050 | 0.040 |
| CV1 | B_transcript | 0.769 | 0.852 | 0.873 | 0.174 | 0.138 |
| CV2 | A | 0.668 | 0.620 | 0.648 | 0.005 | 0.003 |
| CV2 | A_transcript | 0.669 | 0.683 | 0.716 | 0.020 | 0.013 |
| CV2 | B | 0.676 | 0.704 | 0.732 | 0.042 | 0.034 |
| CV2 | B_transcript | 0.702 | 0.751 | 0.787 | 0.094 | 0.079 |
| CV3 | A | 0.669 | 0.617 | 0.645 | 0.004 | 0.002 |
| CV3 | A_transcript | 0.667 | 0.608 | 0.630 | 0.008 | 0.006 |
| CV3 | B | 0.670 | 0.596 | 0.609 | 0.002 | 0.001 |
| CV3 | B_transcript | 0.676 | 0.630 | 0.657 | 0.001 | 0.001 |

Full-universe lift over the matched dependency-only baseline:

| Split | Comparison | ΔF1 | ΔAUROC | ΔAUPR | ΔNDCG@10 | ΔMAP@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CV1 | A_transcript - A | +0.024 | +0.136 | +0.139 | +0.023 | +0.013 |
| CV1 | B_transcript - B | +0.039 | +0.057 | +0.061 | +0.123 | +0.098 |
| CV2 | A_transcript - A | +0.001 | +0.063 | +0.068 | +0.015 | +0.009 |
| CV2 | B_transcript - B | +0.026 | +0.048 | +0.054 | +0.052 | +0.045 |
| CV3 | A_transcript - A | -0.001 | -0.008 | -0.015 | +0.004 | +0.003 |
| CV3 | B_transcript - B | +0.006 | +0.035 | +0.048 | -0.001 | -0.001 |

Covered-pair diagnostic slice for augmented models only:

| Split | Model | F1 | AUROC | AUPR | NDCG@10 | MAP@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CV1 | A_transcript | 0.766 | 0.774 | 0.842 | 0.025 | 0.015 |
| CV1 | B_transcript | 0.802 | 0.837 | 0.896 | 0.139 | 0.109 |
| CV2 | A_transcript | 0.750 | 0.697 | 0.785 | 0.020 | 0.014 |
| CV2 | B_transcript | 0.761 | 0.749 | 0.835 | 0.081 | 0.070 |
| CV3 | A_transcript | 0.747 | 0.613 | 0.713 | 0.011 | 0.008 |
| CV3 | B_transcript | 0.750 | 0.645 | 0.740 | 0.002 | 0.001 |

## Interpretation

The no-coverage-flag Tier-1 run shows that observed gwps Perturb-seq features add
clear signal on CV2. `B_transcript` improves the strongest dependency-only model
from NDCG@10 0.042 to 0.094 and MAP@10 0.034 to 0.079, while also improving AUROC
and AUPR. This is the most important positive result because CV2 suppresses the
CV1 degree shortcut while still allowing one seen gene.

CV3 remains the hard cold-start surface. `A_transcript` improves official ranking
over `A` (NDCG@10 0.004 to 0.008; MAP@10 0.002 to 0.006) but loses slight
classification signal. `B_transcript` improves classification over `B` but does
not improve CV3 top-k ranking. The current result therefore supports
"transcriptome helps under one-gene-held-out CV2" more strongly than
"transcriptome solves both-gene-cold-start CV3."

The covered-pair slice is stronger than the full-universe slice for most
classification metrics, especially CV2 AUPR/AUROC, which is consistent with real
transcript embeddings carrying signal. It does not fully rescue CV3 top-k ranking,
so fallback dilution is not the only limitation.

Because this completed run omits coverage flags, the CV2 lift cannot be explained
by an explicit coverage-as-degree shortcut. The with-coverage-flag run is still
useful as a leakage/confound diagnostic, but it is not needed to establish the
flag-free Tier-1 result.

## Success Criteria

**Primary:** Tier 1 or Tier 2 beat the in-harness exp06 baseline (models A/B) on
**CV2 and CV3** official ranking (NDCG@k, MAP@k). CV1 is degree-gameable (exp06
showed the degree probe wins CV1), so CV2/CV3 are the clean generalization
surfaces.

**Diagnostic:** Lift is concentrated on the **covered-pair slice**, confirming the
gain is transcriptomic and not an artifact of the coverage indicator or fallback
strategy.

**Current status:** Tier-1 no-flag satisfies the CV2 ranking criterion for both
logistic and XGBoost heads, and improves CV3 ranking only for the logistic head.
The result is therefore positive for one-gene-held-out ranking, but not a clean
solution to the both-gene-cold-start CV3 setting.

## MLE-Reviewer Flags

**A. Coverage-flag leakage/confound risk.** gwps coverage correlates with how
well-studied a gene is, which correlates with SL-graph degree — and exp06 showed
the **degree probe wins CV1 (NDCG@10 0.197 vs 0.050 for XGBoost)**. The coverage
indicator could let the model relearn that topology shortcut rather than use
transcriptome signal. Mitigation: report **with and without** the coverage flag as
separate rows, and weight conclusions toward CV2/CV3 (where degree shortcuts are
suppressed by the held-out-gene split structure). If the flag-free model shows
lift, the transcriptome signal is genuine.

**B. Pair-level coverage dilution.** Option A (full benchmark + fallback) can
blur a real transcriptomic signal because many rows still contain at least one
fallback embedding. The **covered-pair diagnostic slice** isolates that issue
from transcriptome uselessness. In the completed no-flag run, the empirical
both-covered benchmark-row fraction is 51.17%; the covered-pair slice improves
classification metrics but does not fully solve CV3 top-k ranking.

**C. Fallback-strategy sensitivity.** Zero-impute vs. global-mean-impute are
different missingness models. Tier 1 should ablate both supported fallback
strategies to check robustness. If results are sensitive to fallback choice, that
is evidence the model is learning from the fallback structure, not the
transcriptome.

## Caveats

- The candidate universe for this K562 MVP is the K562-filtered 9,471-gene universe
  from exp06, not the full official 9,845-gene SynLethDB universe.
- `Rand` negatives are unconfirmed non-SL. No SL biological claim is made; this is
  a benchmark-adapter extension of exp06.
- Pair-level coverage differs by reference set: ~41% is the random-pair estimate
  from the 64.09% gene coverage, while the completed benchmark-row manifest
  records 51.17% both-covered rows.
- gwps is the sole transcriptome source because other CRISPRi sources add only +4
  SL-universe genes. Future experiments could revisit the union or add Adamson
  transfer, but this MVP prioritizes simplicity.
- Tier 2 (frozen exp03 representation) reuses the *method*, not necessarily the
  exact same trained scVI/GMM checkpoint (gene panels may differ between
  Replogle "essential" and gwps). The representation recipe is fixed, but the
  models may be re-fit on gwps data.

## Next

1. Run the with-coverage-flag ablation to quantify whether the explicit coverage
   indicator adds useful signal or reintroduces degree/coverage confounding.
2. Run the `global_mean` fallback ablation to check whether the CV2 lift is robust
   to the missingness model for uncovered genes.
3. Build the Tier-2 `scvi_delta_meanpool` bags and rerun the same no-flag protocol
   to test whether exp03-style representation quality improves CV3 ranking.
4. Calibration / reliability check for transcript-augmented models on CV2/CV3.
5. Compare to the paper's Rand 1:1 table only with the official per-anchor ranking
   protocol and the same CV split boundaries.

## Scope Guard

CV1/CV2/CV3 only (no further splits); `Rand` negatives only; gwps transcriptome
only (no union, no Adamson training); two tiers (LogReg/XGB + frozen exp03
representation), no end-to-end learned set encoder; full benchmark + fallback
(Option A), no both-covered-subset-only restriction; pair-level adapter, no new
cell lines and no SL biological claims.

**Explicitly NOT in scope:**
- Predicted/generated transcriptome bags (no forward model; uncovered genes use
  feature fallback, not imputed response).
- Context-specific SL validation (stays a benchmark adapter; no "SL target"
  language).
- Changing the `D` label, `C = GeneEffect(K562,g)` definition, or the exp03
  transcriptome→GeneEffect mapping (all are fixed inputs).
- Changing the Feng-2024 benchmark, the official metric protocol, or CV1/CV2/CV3
  split definitions.

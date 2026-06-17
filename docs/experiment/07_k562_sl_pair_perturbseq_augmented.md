# K562 SL Pair Perturb-seq-Augmented Baseline

Run status: MVP implementation completed 2026-06-17. Module
`src/sl_benchmark_baseline/` augmented mode (Tier 1 `pca_delta_meanpool`, Tier 2
`scvi_delta_meanpool` via a different bags NPZ). Code and unit tests are green on
synthetic fixtures; the real-data CV1/CV2/CV3 run is pending the gwps-derived
`bags.npz` artifact (multi-GB, gitignored, built by exp03 `build-cell-bags`).

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

But this is a **pair** task. Under independence, P(both genes covered) ≈ 0.64² ≈
**41% of pairs**. The remaining **~59% of evaluated pairs** have at least one gene
with no observed response bag. The Feng 2024 SL-benchmark documentation and prior
coverage audits quantify per-gene coverage but not pair-level coverage — **41% is
the real constraint for transcriptomic pair features, not 64%.**

The chosen formulation (Option A: full benchmark + fallback) handles this by:
- Keeping every pair and the full 9,471-gene universe unchanged.
- Covered genes → real transcript embedding + `coverage_flag = 1`.
- Uncovered genes → fallback embedding (zero or training-mean) + `coverage_flag = 0`.
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
- **Coverage indicator** (1 or 2 features): `flag_a · flag_b` (both-covered
  indicator) or both `flag_a`, `flag_b` separately.

All features standardized on train-fold statistics only (matching exp06's
`Standardizer`).

### Models — Two Tiers

**Tier 1 — MVP floor (directly comparable to exp06):**

| Model | Inputs | Role |
| --- | --- | --- |
| `LogReg_transcript` | GeneEffect block + transcript block + coverage flag(s) | Symmetric logistic regression, same head as exp06-A |
| `XGB_transcript` | GeneEffect block + transcript block + coverage flag(s) | XGBoost (200 trees, depth 4), same head as exp06-B |

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
- **With coverage flag:** include `flag_a · flag_b` or separate flags.
- **Without coverage flag:** omit the coverage indicators entirely, so the model
  cannot shortcut via coverage-as-proxy-for-degree.

Tier-1 floor establishes whether transcriptome adds lift with the simplest
representation (PCA/HVG mean-pool). Tier-2 tests whether exp03's proven
representation transfers to the pair-ranking task.

### Protocol

Per `(split_type, fold_id)`:
1. Fit transcript embeddings on **train genes only** (for Tier 2, scVI/GMM fit on
   train-covered genes; uncovered genes get fallback). Test genes at inference use
   the frozen embedding model.
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
supplying `--bags-npz`. The canonical invocation is recorded in
`configs/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_cv1_cv2_cv3.yaml`
(the module reads CLI flags, not the YAML directly). Tier 1, with coverage flag:

```bash
uv run python -m sl_benchmark_baseline \
  --input-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
  --bags-npz results/experiments/03_replogle_k562_single_cell_deepsets_adamson/cell_bags/single_cell_pc_delta/bags.npz \
  --embedding-method pca_delta_meanpool \
  --fallback-strategy zero \
  --include-coverage-flag \
  --split-types CV1 CV2 CV3 --folds 0 1 2 3 4 --ranking-k 10 20 50 \
  --output-dir results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_with_flag
```

Ablations (spec MLE flags A/C) are flag flips, not code changes:
- **Without coverage flag:** rerun with `--no-coverage-flag` and a `_no_flag` output dir.
- **Tier 2 (frozen exp03 scVI representation):** rerun with the `scvi_delta` bags NPZ
  and `--embedding-method scvi_delta_meanpool`.
- **Fallback sensitivity:** rerun with `--fallback-strategy global_mean`.

Without `--bags-npz` the module behaves exactly like the exp06 dependency-only
baseline (models A/B/C, no transcript block).

## Expected Results (Stage 1 — no results yet)

Artifacts will mirror exp06's structure:
- Per split: `results/experiments/07_k562_sl_pair_perturbseq_augmented/cv{1,2,3}/`
  with `fold_metrics.csv`, `summary.csv`, `manifest.json`.
- Combined: `official_metrics_summary.csv`.
- Manifests record: `candidate_gene_count = 9471`, `seed = 17`, `ranking_k = [10,
  20, 50]`, `gwps_coverage_gene_count = 6070`, `gwps_coverage_pair_fraction ≈ 0.41`,
  `embedding_method`, `fallback_strategy`, `coverage_flag_included`.

Table layout will add LogReg_transcript / XGB_transcript rows to exp06's baseline
table for CV1/CV2/CV3, with both full-universe and covered-pair-slice metric
blocks.

## Success Criteria

**Primary:** Tier 1 or Tier 2 beat the in-harness exp06 baseline (models A/B) on
**CV2 and CV3** official ranking (NDCG@k, MAP@k). CV1 is degree-gameable (exp06
showed the degree probe wins CV1), so CV2/CV3 are the clean generalization
surfaces.

**Diagnostic:** Lift is concentrated on the **covered-pair slice**, confirming the
gain is transcriptomic and not an artifact of the coverage indicator or fallback
strategy.

**Negative result is publishable:** If Tier 1/2 do not beat exp06 on CV2/CV3,
the conclusion is "observed Perturb-seq at 64% per-gene / 41% per-pair coverage
does not add information beyond DepMap GeneEffect for K562 SL candidate ranking."
This is a valid Stage-3 finding and informs whether to pursue predicted-B (exp04)
or frozen-AIVC (exp05) pair features.

## MLE-Reviewer Flags

**A. Coverage-flag leakage/confound risk.** gwps coverage correlates with how
well-studied a gene is, which correlates with SL-graph degree — and exp06 showed
the **degree probe wins CV1 (NDCG@10 0.197 vs 0.050 for XGBoost)**. The coverage
indicator could let the model relearn that topology shortcut rather than use
transcriptome signal. Mitigation: report **with and without** the coverage flag as
separate rows, and weight conclusions toward CV2/CV3 (where degree shortcuts are
suppressed by the held-out-gene split structure). If the flag-free model shows
lift, the transcriptome signal is genuine.

**B. Pair-level coverage dilution.** Option A (full benchmark + fallback) cannot
distinguish "transcriptome useless" from "signal exists but is diluted by 59%
fallback pairs." The **covered-pair diagnostic slice** isolates the former from
the latter. If lift appears only on the covered-pair slice and vanishes on the
full universe, the conclusion is "transcriptome helps but 41% coverage is
insufficient for population-level gains."

**C. Fallback-strategy sensitivity.** Zero-impute vs. training-mean-impute vs.
PCA-project-to-origin are different missingness models. Tier 1 should ablate at
least two fallback strategies to check robustness. If results are sensitive to
fallback choice, that is evidence the model is learning from the fallback
structure, not the transcriptome.

## Caveats

- The candidate universe for this K562 MVP is the K562-filtered 9,471-gene universe
  from exp06, not the full official 9,845-gene SynLethDB universe.
- `Rand` negatives are unconfirmed non-SL. No SL biological claim is made; this is
  a benchmark-adapter extension of exp06.
- Pair-level coverage (~41%) is a derived estimate under independence; the actual
  fraction depends on whether coverage is random or correlated across genes
  (e.g., co-essential pairs may be co-covered).
- gwps is the sole transcriptome source because other CRISPRi sources add only +4
  SL-universe genes. Future experiments could revisit the union or add Adamson
  transfer, but this MVP prioritizes simplicity.
- Tier 2 (frozen exp03 representation) reuses the *method*, not necessarily the
  exact same trained scVI/GMM checkpoint (gene panels may differ between
  Replogle "essential" and gwps). The representation recipe is fixed, but the
  models may be re-fit on gwps data.

## Next

1. If Tier 1/2 beat exp06 on CV2/CV3: add Adamson K562 held-out transfer (mirroring
   exp03's external-validation design) to check whether the transcriptome→SL-ranking
   signal generalizes across Perturb-seq batches.
2. If Tier 1/2 do NOT beat exp06: this is evidence that Stage-2 set-MIL (exp03)
   does not bridge to Stage-3 SL pair ranking, and predicted-B / frozen-AIVC pair
   features (exp04/05) are high-risk.
3. Calibration / reliability check for transcript-augmented models on CV2/CV3.
4. Compare to the paper's Rand 1:1 table only with the official per-anchor ranking
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

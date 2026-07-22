# SL_benchmark 2024 Synthetic-Lethality Benchmark

## Role

Primary general synthetic-lethality pair benchmark for comparison with the
Feng2024 SOTA model zoo. Use the official 9,845-gene CV1/CV2/CV3 caches and metric
implementation for the formal context-agnostic score `q(a,b)`.

This is not a DepMap GeneEffect regression dataset and not a single-cell
perturbation dataset. The supervised object is an undirected gene pair:

```text
(gene_a, gene_b) -> binary SL interaction label or pair score
```

The main benchmark is not cell-line-specific. Contextual features may be
aggregated into a declared context-agnostic score, but the benchmark label remains
a generic curated gene-pair label and cannot establish cross-cell-line
generalization.

## Sources

- Paper: Feng et al., "Benchmarking machine learning methods for synthetic
  lethality prediction in cancer", Nature Communications 15, 9058, published
  2024-10-20.
- Local paper PDF:
  `docs/literature/literature/02_data/benchmarks/2024_nature_communications_benchmarking_machine_learning_methods_for_synthetic_lethality_prediction_in_cancer.pdf`
- Official GitHub:
  `https://github.com/JieZheng-ShanghaiTech/SL_benchmark.git`
- Local checkout: `data/SL_benchmark`
- Local checkout commit: `04274e801b820a81f8dd92eb362d154086b80d9a`
- Dataset DOI in the official README:
  `https://doi.org/10.5281/zenodo.14025191`

## Local Files Checked on 2026-06-16

Local root: `data/SL_benchmark`

| Path | Role |
| --- | --- |
| `README.md` | Official benchmark usage, environment, data download, and model list. |
| `src/main.py` | Official entrypoint; exposes `-m`, `-ns`, `-ds`, `-pn`, and `--save_mat`. |
| `src/preprocess.py` | Split construction, negative sampling, independent-test loading, and metrics. |
| `src/summary_all_matrics.csv` | Published-style aggregate result table, 12 models x 3 negative sampling methods. |
| `data/fin_entities.csv` | Unified entity table; columns are `unified_id`, `entity_type`, `entity_type_name`, `entity_name`. |
| `data/predicted_by_model.csv` | Model-ranked pair predictions; columns are `id_a`, `id_b`, `model`, `rank`. |
| `data/data_split/*.npy` | Cached 5-fold positive and negative train/test pair matrices for CV1/CV2/CV3. |
| `data/data_split/*_indep_test.pkl` | Cached independent-test variants, not cell-line-specific by filename. |
| `data/data_small.tar.gz.partaa` ... `partae` | Downloaded small archive parts. |
| `data/data_small.tar.gz.md5`, `data/data_small.tar.gz.part.md5` | Official checksums for the small archive and parts. |

The current local tree does not contain `data/preprocessed_data/`, so raw
tables such as `human_sl_9845.csv`, `human_sl_6460.csv`,
`sorted_neg_ids_scores_exp.npy`, and `sorted_neg_ids_scores_dep.npy` are not
available at the checked path. Use the existing `data_split` caches for
benchmark-split inspection, or reassemble/extract the archive if raw
preprocessed tables are needed.

No local file matching `*k562*`, `*a549*`, `*293t*`, or `*hela*` was present
under `data/SL_benchmark` on 2026-06-16. The official code supports
cell-line-specific independent-test suffixes such as `_cell_k562`, but this
local checkout does not currently include those files.

## Benchmark Definition

The paper constructs a benchmark dataset from SynLethDB 2.0 and related
biological resources. After filtering, it reports:

- 9845 unique genes.
- 35913 positive SL gene pairs.
- Positive labels come from SynLethDB-derived known SL pairs; a subset is
  experimentally identified by CRISPR, RNAi, or text mining, while another
  subset is computationally predicted.
- The official benchmark also uses GO, PPI, pathway, protein-complex,
  protein-sequence, knowledge-graph, gene-expression, and dependency-score data
  depending on the model.

The local `data_split` files encode pair matrices over a fixed `9845 x 9845`
gene universe. Pairs should be treated as undirected unless a downstream task
explicitly creates an ordered anchor-target view.

## Splits

The official benchmark uses 5-fold cross-validation with three data splitting
methods:

| Split | Meaning | Use in this project |
| --- | --- | --- |
| `CV1` | Pair-level holdout. Both genes in a test pair may appear in other training pairs. | Easiest sanity check for the adapter, but not evidence of held-out-gene generalization. |
| `CV2` | Gene split where only one gene in a tested pair is present in the training set. | Better approximation of ranking partners for a partly known gene. |
| `CV3` | Gene split where neither gene in a tested pair is present in the training set. | Cold-start setting; hardest and most relevant for unseen-gene generalization claims. |

Local cache structure for `*.npy` split files:

```text
array shape: (2, 4, 5)
axis 0: positive samples, negative samples
axis 1: graph_train, graph_test, train_pair, test_pair
axis 2: 5 folds
```

For example, `data/data_split/CV1_1.npy` has approximately 28730 training
positive pairs and 7183 test positive pairs per fold, with the same number of
negative pairs for the `1:1` setting.

## Negative Sampling

The paper evaluates four positive-negative ratios:

```text
1:1, 1:5, 1:20, 1:50
```

The official CLI names these ratios with `-pn 1`, `-pn 5`, `-pn 20`, or
`-pn 50`.

It also evaluates three negative sampling methods:

| Method | Official CLI value | Meaning |
| --- | --- | --- |
| Random | `Rand` | Randomly sample unknown/non-SL pairs as negatives. |
| Expression-informed | `Exp` | Use DepMap cross-cell-line gene-expression correlation to select harder or more biologically controlled negatives. |
| Dependency-informed | `Dep` | Use DepMap CRISPR dependency-score correlation to select negatives. |

For this project, `Exp` and `Dep` need leakage warnings whenever model features or
context supervision also use DepMap expression or dependency information. The
primary formal comparison is therefore `Rand` 1:1; the other regimes are declared
sensitivity analyses.

## Metrics

The benchmark evaluates both pair classification and gene ranking:

- Classification: AUROC, AUPR, F1.
- Ranking: NDCG@10/20/50, Recall@10/20/50, Precision@10/20/50.
- The official code also computes MAP@10/20/50 in `src/preprocess.py`.

Official `src/preprocess.py:cal_metrics` expects a floating-point score matrix.
Classification is computed on sampled positive and negative test pairs from
that matrix. AUPR is trapezoidal PR-curve AUC, and F1 is the maximum F1 over
the PR curve, not fixed-threshold F1.

Ranking is per-anchor candidate-partner retrieval. For each anchor gene that
appears in test positives, the evaluator ranks candidate partner genes from the
full score matrix. Training positive pairs are masked to a very negative score
before ranking. NDCG@k uses `sklearn.metrics.ndcg_score`; Recall@k is
`hits / n_positive_partners_for_anchor`; official-code Precision@k is
`hits / min(k, n_positive_partners_for_anchor)`, which differs from conventional
`hits / k` when anchors have fewer than `k` positives. MAP@k averages per-anchor
average precision over hits found in the top `k`.

## How to Use in This Project

Use this benchmark as the formal general pair-label target and SOTA comparison,
not as a replacement for cell-line-resolved SL/GI evaluation.

Recommended formal route:

```text
official SL_benchmark data_split pair cache
    -> convert unified_id pairs to gene symbols with fin_entities.csv
    -> build sl_pairs table with split, fold, label, negative sampling method
    -> construct context-agnostic q(a,b) without changing the pair universe
    -> train/evaluate alongside reproduced SOTA under identical cal_metrics
```

Suggested canonical adapter schema:

```text
pair_id
gene_a_symbol
gene_b_symbol
gene_a_unified_id
gene_b_unified_id
pair_is_ordered = false
sl_label
negative_sampling_method
positive_negative_ratio
split_type
fold_id
split_role
source_file
```

The historical K562 coverage ablation joins response/dependency features as:

```text
pair_id
gene_a_symbol
gene_b_symbol
sl_label
candidate_has_k562_response
candidate_gene_effect_true
candidate_gene_effect_pred
candidate_observed_response_features
candidate_frozen_state_features
anchor_has_k562_response
anchor_gene_effect_true
anchor_observed_response_features
```

Retain that K562 `dependency_only` row as a prior floor, but do not let its gene
coverage redefine the formal pair universe. The general benchmark ladder must
also include gene-marginal/context-free controls that can score the full official
universe under the same folds.

## K562 DepMap-Filtered Rand 1:1 CV1/CV2/CV3 Subset

The helper script `scripts/build_k562_sl_benchmark.py` filters the official
CV1/CV2/CV3 Rand 1:1 split caches to pairs whose two genes both have numeric
K562 DepMap GeneEffect values for `ACH-000551`, then writes per-split balanced
tables and an all-CV concatenation for training and evaluation.

Historical K562 coverage-ablation datasets (all under
`data/SL_benchmark/derived/k562_depmap_rand_1to1/`):

```text
CV1_Rand_1to1_k562_depmap_pairs_balanced.csv
CV2_Rand_1to1_k562_depmap_pairs_balanced.csv
CV3_Rand_1to1_k562_depmap_pairs_balanced.csv
all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv   # historical K562 baseline input
```

Run:

```bash
uv run python scripts/build_k562_sl_benchmark.py
```

The default data API consumes the all-CV concatenation (which carries a
`split_type` column for CV1/CV2/CV3); the per-split balanced CSVs are available
for single-split runs. The script also writes unbalanced variants, summaries,
and metadata in the same directory.

The dependency-only adapter is run with:

```bash
uv run python -m sl_benchmark_baseline \
  --input-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/CV1_Rand_1to1_k562_depmap_pairs_balanced.csv \
  --output-dir results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv1 \
  --split-types CV1 --folds 0 1 2 3 4 --ranking-k 10 20 50
```

Repeat with `CV2` and `CV3` input/output paths for the other split types. The
2026-06-17 official-metric outputs are:

```text
results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv1/
results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv2/
results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_cv3/
results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv
```

These outputs use a 9471-gene K562-filtered candidate universe and official
`src/preprocess.py:cal_metrics` semantics. The older
`results/experiments/06_k562_sl_pair_dependency_only_mvp/all_cv_run/` ranking metrics
used flat pair-level ranking and are obsolete for paper comparison.

Canonical columns:

```text
pair_id
fold_id
split_role
sl_label
gene_a_unified_id
gene_b_unified_id
gene_a_symbol
gene_b_symbol
gene_a_k562_gene_effect
gene_b_k562_gene_effect
```

On 2026-06-16, the generated metadata was:

```text
model_id: ACH-000551
n_k562_depmap_genes: 17787
n_sl_gene_entities: 25260
n_sl_genes_with_k562_depmap: 17538
```

Canonical CV1 row counts:

```text
rows: 331730
positive: 165865
negative: 165865
```

This filtered dataset is K562-mappable by DepMap gene coverage. It is an ablation
and prior-baseline surface, not the formal general benchmark and not a
cell-line-specific K562 SL assay unless a real `*_cell_k562.pkl` split is available
and verified.

## Official Reproduction Notes

The official repo is a standalone Python 3.7 / Conda / CUDA-era project. Do not
install its pinned PyTorch Geometric wheels into this repo's `.venv`.

Official-style reproduction should be isolated:

```bash
cd data/SL_benchmark
conda env create -f SL-Benchmark.yml
conda activate SLBench
pip install ./torch_spline_conv-latest+cu102-cp37-cp37m-linux_x86_64.whl
pip install ./torch_sparse-latest+cu102-cp37-cp37m-linux_x86_64.whl
pip install ./torch_scatter-latest+cu102-cp37-cp37m-linux_x86_64.whl
pip install ./torch_cluster-latest+cu102-cp37-cp37m-linux_x86_64.whl
cd src
python main.py -m SLMGAE -ns Rand -ds CV1 -pn 1
```

For this repo's own adapter code, use the project environment convention:

```bash
uv run python <adapter_or_inspection_script>.py
```

## Cautions

- Do not call `DepMap GeneEffect(K562, g)` an SL label.
- Do not call a K562 essential gene an SL target without pair/context evidence.
- Treat randomly sampled unknown pairs as noisy negatives, not experimentally
  confirmed non-SL pairs.
- Treat `CV1` as a pair split, not a held-out-gene split.
- Prefer `CV2` or `CV3` for any claim about generalization to unseen genes.
- Do not call CV2/CV3 cross-cell-line generalization; the main benchmark has no
  held-out-cell-line axis.
- Verify actual `*_cell_k562.pkl` files before claiming K562-specific SL
  benchmark execution. The local checkout checked on 2026-06-16 does not
  contain those files.
- Track whether each result uses `Rand`, `Exp`, or `Dep`, because the negative
  sampling method changes the biological and leakage interpretation.

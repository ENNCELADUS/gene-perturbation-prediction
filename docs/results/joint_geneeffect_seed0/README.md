# 30030 selected-checkpoint test and baselines

Verified 2026-09-07 on apg3op6hp3v99-0, checkout `/2023533015/VCC_Project`, revision `a0af4999b73ba6539af58b415291fbac9ea6d90e`.

Run: `joint_seed0_20260906T174818Z_b1024`. Test uses validation-selected best.pt, epoch 3 (stored index 2), step 13250. Training was not resumed. Test evaluation is completed, both worker PIDs exited, both GPUs released. All six baseline variants exported predictions, metrics and detailed per-line/per-gene tables.

Checkpoint SHA256: `37405454717cb12164497542623722cd31e9dc793ebbfef3d79a3ef868e63c59`.

All methods have exactly the same 478501 observed (ModelID, gene_symbol) keys across 27 test cell lines; 1748 missing labels out of 480249 possible pairs remain excluded. Target values agree within 1.72e-7 (float precision). Residual correlations use the same 4447 train-defined variable genes. Baseline transforms and fits use supervised train only. Seed 0 only; no cross-seed variance or significance claim.

| Method | Huber ↓ | RMSE ↓ | Absolute Pearson ↑ | Absolute Spearman ↑ | Residual Pearson ↑ | Residual Spearman ↑ | Loss reduction vs gene mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint_best_epoch3 | 0.01611905 | 0.18038516 | 0.91050399 | 0.73338939 | 0.05414127 | 0.05280360 | 0.0773% |
| context_pca_ridge[hvg] | 0.01635306 | 0.18167029 | 0.90920119 | 0.72759752 | 0.07736289 | 0.08323587 | -1.3733% |
| context_pca_ridge[tx1] | 0.01625512 | 0.18106113 | 0.90983343 | 0.72777157 | 0.12160618 | 0.11573946 | -0.7662% |
| copy_prior | 0.03389791 | 0.26316988 | 0.81686034 | 0.54343217 | undefined | undefined | -110.1347% |
| gene_mean | 0.01613152 | 0.18044272 | 0.91047146 | 0.73393976 | undefined | undefined | 0.0000% |
| nearest_line[hvg] | 0.02853913 | 0.24133962 | 0.84602658 | 0.58732115 | 0.05900657 | 0.06130214 | -76.9154% |
| nearest_line[tx1] | 0.02868782 | 0.24203011 | 0.84576280 | 0.59698253 | 0.06077659 | 0.05973719 | -77.8371% |

## Interpretation

1. Joint Huber loss improves over gene mean by only 0.0773%; absolute Spearman is slightly lower. The high absolute Pearson is already present in the context-blind prior.
2. Joint residual Pearson/Spearman (0.05414/0.05280) are below Tx1 PCA-ridge (0.12161/0.11574), and below the other contextual baselines in this run. This does not establish useful context-modeling superiority.
3. Gene-mean and copy-prior have undefined residual correlations because their per-gene predictions are constant across contexts; these are not zeros. Baseline bootstrap warnings about nonfinite values are consistent with these undefined metrics; all expected exports are present.
4. Response MSE/energy/total are 85.58490/370.35131/455.93621 over 3285 held-out conditions from four training anchors. These reuse the response holdout and do not represent responses measured on the 27 GeneEffect test lines.

## Next experiments

Use validation for further model decisions: compare matched-batch response_weight=0 versus 1, and inspect residual prediction variance/calibration. Treat this test set as observed; do not use repeated test comparisons to select the next model. These are GeneEffect results, not SL interaction evidence; Tx1 pretraining exposure limits remain.

## Artifacts

Remote model: `outputs/geneeffect_joint/joint_seed0_20260906T174818Z_b1024/evaluation/best/test/`.
Remote baselines: `outputs/geneeffect_joint/joint_seed0_20260906T174818Z_b1024/baselines/test/`.
Remote launch record/logs: `outputs/launches/test_baselines_20260907T0622Z/`.
Curated evidence: [metrics and provenance](evidence.json), [test comparison](test_comparison.csv), [learning curves](learning_curves.csv). Full raw snapshots and plots remain under ignored `outputs/analysis/30030_20260907/`.

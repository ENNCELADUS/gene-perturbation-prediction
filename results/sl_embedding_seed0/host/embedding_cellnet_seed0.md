# Gene-knowledge, cell-set model: embedding_cellnet_seed0

**Table 1.** The model on the multi-study SL benchmark, mean over lines (17 validation, 16 test; training lines in sample). Higher is better. Validation selected the epoch and the phase; test was scored once.

| Metric | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Recall in the top 10% | 0.8829 | 0.4859 | 0.4727 |
| AUROC | 0.9606 | 0.7832 | 0.7656 |
| AUPRC over the positive rate | 717.8154 | 9.7783 | 7.9996 |

**Table 2.** Stage-one numbers on the SL lines with GeneEffect (12 / 11 lines, validation / test; out of sample for stage one), as every stage-one reader is scored: the predicted GeneEffect as a residual over the training gene mean. The second row is the fit lines' gene mean alone, which the untrained model predicts. Higher is better. They decide nothing.

| Model | per-line residual Spearman, validation | per-line residual Spearman, test | per-gene residual Spearman, validation | per-gene residual Spearman, test | residual skill score, validation | residual skill score, test | whole-line Pearson, validation | whole-line Pearson, test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gene-knowledge, cell-set model | 0.0327 | 0.0848 | 0.0183 | 0.0633 | -0.0260 | -0.0171 | 0.9225 | 0.9150 |
| Gene mean of the fit lines | -0.0530 | -0.0213 | — | — | -0.0121 | -0.0080 | 0.9235 | 0.9162 |

**Table 3.** Stage two: validation recall in the top 10% of each phase at its best epoch. The chosen phase is in bold.

| Phase | GeneEffect loss weight | Best epoch / epochs run | Validation |
| --- | ---: | ---: | ---: |
| frozen | — | 1 / 6 | 0.1092 |
| trained further | 1 | 4 / 9 | **0.4859** |

**Table 4.** Stage one: the GeneEffect loss on the SL training lines (out of sample; lower is better). Epoch 0 is the untrained model.

| Epoch | Training loss | Loss on the SL training lines |
| ---: | ---: | ---: |
| 0 | — | 1.0503 |
| 1 | 0.9716 | 1.0595 |
| 2 | 0.8953 | 1.0581 |
| 3 | 0.8627 | 1.0584 |

Stage-one fit lines: 928 (366 with cells); left out: 74 lines with an SL label and 12 lines of their patients. Early-stop lines: 47. Cells: 652,399. Genes: 17,787; with each source: gene_text 17,607, esmc_600m 17,708, string_space 17,503. Stage-two training rows: 1,146,517.


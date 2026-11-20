# Gene-knowledge, cell-set model: embedding_direct_tx1_gene_seed0_v2

**Table 1.** The model on the multi-study SL benchmark, mean over lines (17 validation, 16 test; training lines in sample). Higher is better. Validation selected the epoch and the phase; test was scored once.

| Metric | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Recall in the top 10% | 0.8457 | 0.5528 | 0.4961 |
| AUROC | 0.9506 | 0.8170 | 0.7858 |
| AUPRC over the positive rate | 583.3370 | 12.8514 | 9.0050 |

**Table 2.** The line term on the multi-study SL benchmark, mean over lines. The score with the line term is the model's score in the line plus the line term's deviation from its mean over the SL training lines with training rows. The last row reads the model and the line term with another line's tokens and gene-in-line features (no refit). The model on the mean input is the context-ablated reference. Higher is better; the best of each column is in bold. Validation selected the epoch; test was scored once.

| Model | Recall in the top 10%, validation | Recall in the top 10%, test | AUROC, validation | AUROC, test | AUPRC over the positive rate, validation | AUPRC over the positive rate, test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gene-knowledge, cell-set model | **0.5528** | **0.4961** | **0.8170** | **0.7858** | **12.8514** | **9.0050** |
| The model on the mean input | 0.4658 | 0.4299 | 0.7804 | 0.7612 | 5.5660 | 4.6092 |
| The model with the line term | 0.5073 | 0.4838 | 0.8032 | 0.7772 | 9.7994 | 7.7838 |
| The model and the line term on another line's input | 0.4734 | 0.4579 | 0.7893 | 0.7656 | 12.6497 | 8.4284 |

**Table 3.** Stage-one numbers on the SL lines with GeneEffect (12 / 11 lines, validation / test; out of sample for stage one), as every stage-one reader is scored: the predicted GeneEffect as a residual over the training gene mean. The second row is the fit lines' gene mean alone. Higher is better. They decide nothing.

| Model | per-line residual Spearman, validation | per-line residual Spearman, test | per-gene residual Spearman, validation | per-gene residual Spearman, test | residual skill score, validation | residual skill score, test | whole-line Pearson, validation | whole-line Pearson, test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gene-knowledge, cell-set model | 0.0325 | 0.0333 | 0.0347 | 0.0306 | -0.1356 | -0.2231 | 0.9150 | 0.9007 |
| Gene mean of the fit lines | -0.0530 | -0.0213 | — | — | -0.0121 | -0.0080 | 0.9235 | 0.9162 |

**Table 4.** Stage two: validation recall in the top 10% of each phase at its best epoch. The chosen phase is in bold. The line term, when the config has one, trains last on the chosen phase.

| Phase | GeneEffect loss weight | Best epoch / epochs run | Validation |
| --- | ---: | ---: | ---: |
| frozen | — | 4 / 9 | 0.4247 |
| trained further | 1 | 2 / 7 | **0.5528** |
| line term | — | 1 / 6 | 0.5073 |

**Table 5.** Stage one: the GeneEffect loss on the SL training lines (out of sample; lower is better). Epoch 0 is the untrained model, which predicts one value for every gene.

| Epoch | Training loss | Loss on the SL training lines |
| ---: | ---: | ---: |
| 0 | — | 1.0224 |
| 1 | 0.2784 | 0.2073 |
| 2 | 2.0113 | 0.3068 |
| 3 | 0.1895 | 0.2268 |
| 4 | 0.1728 | 0.2144 |

Stage-one fit lines: 928 (366 with cells); left out: 74 lines with an SL label and 12 lines of their patients. Early-stop lines: 47. Cells: 652,399. Genes: 17,787; with each source: gene_text 17,607, esmc_600m 17,708, string_space 17,503. Stage-two training rows: 1,146,517.


# Gene-knowledge, cell-set model: embedding_direct_mean_token_seed0_v2

**Table 1.** The model on the multi-study SL benchmark, mean over lines (17 validation, 16 test; training lines in sample). Higher is better. Validation selected the epoch and the phase; test was scored once.

| Metric | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Recall in the top 10% | 0.9225 | 0.5447 | 0.5050 |
| AUROC | 0.9747 | 0.8129 | 0.7889 |
| AUPRC over the positive rate | 841.5031 | 18.9441 | 8.9827 |

**Table 2.** The line term on the multi-study SL benchmark, mean over lines. The score with the line term is the model's score in the line plus the line term's deviation from its mean over the SL training lines with training rows. The last row reads the model and the line term with another line's tokens and gene-in-line features (no refit). The model on the mean input is the context-ablated reference. Higher is better; the best of each column is in bold. Validation selected the epoch; test was scored once.

| Model | Recall in the top 10%, validation | Recall in the top 10%, test | AUROC, validation | AUROC, test | AUPRC over the positive rate, validation | AUPRC over the positive rate, test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gene-knowledge, cell-set model | 0.5447 | 0.5050 | 0.8129 | 0.7889 | 18.9441 | 8.9827 |
| The model on the mean input | 0.3985 | 0.3987 | 0.7325 | 0.7219 | 4.8323 | 5.6942 |
| The model with the line term | **0.5488** | **0.5053** | **0.8129** | **0.7901** | **19.0293** | **9.1216** |
| The model and the line term on another line's input | 0.4867 | 0.4729 | 0.7933 | 0.7663 | 12.9977 | 8.6457 |

**Table 3.** Stage-one numbers on the SL lines with GeneEffect (12 / 11 lines, validation / test; out of sample for stage one), as every stage-one reader is scored: the predicted GeneEffect as a residual over the training gene mean. The second row is the fit lines' gene mean alone. Higher is better. They decide nothing.

| Model | per-line residual Spearman, validation | per-line residual Spearman, test | per-gene residual Spearman, validation | per-gene residual Spearman, test | residual skill score, validation | residual skill score, test | whole-line Pearson, validation | whole-line Pearson, test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gene-knowledge, cell-set model | 0.0695 | 0.0701 | 0.0567 | 0.0724 | -0.0517 | -0.0713 | 0.9205 | 0.9117 |
| Gene mean of the fit lines | -0.0530 | -0.0213 | — | — | -0.0121 | -0.0080 | 0.9235 | 0.9162 |

**Table 4.** Stage two: validation recall in the top 10% of each phase at its best epoch. The chosen phase is in bold. The line term, when the config has one, trains last on the chosen phase.

| Phase | GeneEffect loss weight | Best epoch / epochs run | Validation |
| --- | ---: | ---: | ---: |
| frozen | — | 23 / 28 | 0.4639 |
| trained further | 1 | 5 / 10 | **0.5447** |
| line term | — | 8 / 13 | 0.5488 |

**Table 5.** Stage one: the GeneEffect loss on the SL training lines (out of sample; lower is better). Epoch 0 is the untrained model, which predicts one value for every gene.

| Epoch | Training loss | Loss on the SL training lines |
| ---: | ---: | ---: |
| 0 | — | 1.0224 |
| 1 | 0.2687 | 0.2031 |
| 2 | 0.1821 | 0.1963 |
| 3 | 4.0518 | 0.2424 |
| 4 | 0.1973 | 0.2038 |
| 5 | 0.1737 | 0.1981 |

Stage-one fit lines: 928 (366 with cells); left out: 74 lines with an SL label and 12 lines of their patients. Early-stop lines: 47. Cells: 0. Genes: 17,787; with each source: gene_text 17,607, esmc_600m 17,708, string_space 17,503. Stage-two training rows: 1,146,517.


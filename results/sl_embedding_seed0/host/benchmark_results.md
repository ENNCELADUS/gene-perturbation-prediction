# SL benchmark baselines: benchmark_embedding_seed0

**Table 1.** Recall in the top 10% on the multi-study SL benchmark, mean over lines (each line: the mean over its studies of the mean over their genetic screens with both classes). Higher is better; a random ranking gives 0.10. Best in bold on validation and test. The training lines are in sample: the training SL fraction reads their own labels and every fitted model was fitted on them; — : the model has no score for every training row.

| Model | Training (44 lines) | Validation (15 lines) | Test (13 lines) |
| --- | ---: | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | 0.324 | 0.381 | 0.298 |
| Backbone head on the buffer-aware reader's terms | 0.407 | 0.394 | 0.361 |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 0.411 | 0.383 | 0.355 |
| Pair identity (hit rate in the study's training lines) | 0.905 | **0.540** | **0.541** |
| Pan-essentiality (gene mean GeneEffect) | 0.318 | 0.380 | 0.297 |
| Measured GeneEffect (reference, lines with GeneEffect) | 0.319 | 0.425 | 0.296 |
| Labels as the score (check: the ceiling) | 1.000 | 1.000 | 1.000 |
| Random score (check: about 0.10) | 0.093 | 0.102 | 0.124 |
| KG4SL (knowledge graph, official code, trained on the training lines) | — | 0.367 | 0.338 |
| Cilantro-SL (Geneformer deletions, official code, refitted) | — | 0.123 | 0.134 |
| PARIS (random-forest dependency pairs, Python port of the R code) | — | 0.219 | 0.213 |
| SLIdR (mutation-dependency test, Python port of the R code) | — | 0.095 | 0.115 |
| De Kegel 2021 paralog classifier (released scores) | — | 0.281 | 0.293 |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | — | 0.355 | 0.395 |
| MiT4SL (cell-line-specific graph model, official code, refitted) | — | 0.154 | 0.138 |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | 0.782 | 0.500 | 0.485 |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's | 0.883 | 0.486 | 0.473 |
| Gene-knowledge model without cell tokens (each line reads its mean token alone) | 0.898 | 0.508 | 0.501 |

**Table 2.** Difference from pair identity in recall in the top 10%, mean over lines, with a 95% bootstrap interval over lines. Higher is better.

| Model − pair identity | Validation | Test |
| --- | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | -0.167 [-0.301, -0.062] | -0.239 [-0.399, -0.100] |
| Backbone head on the buffer-aware reader's terms | -0.153 [-0.225, -0.080] | -0.165 [-0.226, -0.099] |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | -0.165 [-0.235, -0.092] | -0.172 [-0.234, -0.109] |
| Pan-essentiality (gene mean GeneEffect) | -0.169 [-0.296, -0.035] | -0.241 [-0.378, -0.104] |
| Measured GeneEffect (reference, lines with GeneEffect) | -0.203 [-0.357, -0.074] | -0.289 [-0.482, -0.134] |
| KG4SL (knowledge graph, official code, trained on the training lines) | -0.164 [-0.251, -0.067] | -0.174 [-0.261, -0.079] |
| Cilantro-SL (Geneformer deletions, official code, refitted) | -0.415 [-0.564, -0.282] | -0.402 [-0.522, -0.259] |
| PARIS (random-forest dependency pairs, Python port of the R code) | -0.326 [-0.437, -0.197] | -0.319 [-0.422, -0.215] |
| SLIdR (mutation-dependency test, Python port of the R code) | -0.443 [-0.585, -0.286] | -0.423 [-0.555, -0.265] |
| De Kegel 2021 paralog classifier (released scores) | -0.253 [-0.332, -0.170] | -0.227 [-0.297, -0.150] |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | -0.183 [-0.253, -0.113] | -0.124 [-0.175, -0.068] |
| MiT4SL (cell-line-specific graph model, official code, refitted) | -0.389 [-0.524, -0.246] | -0.397 [-0.527, -0.272] |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | -0.028 [-0.063, +0.005] | -0.020 [-0.047, +0.007] |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's | -0.040 [-0.083, +0.000] | -0.026 [-0.063, +0.004] |
| Gene-knowledge model without cell tokens (each line reads its mean token alone) | -0.022 [-0.051, +0.007] | -0.012 [-0.049, +0.020] |

**Table 3.** Recall in the top 10% for each stratum and subset of held-out pairs, ranked inside it in each screen, mean over lines (validation / test). Higher is better. Held-out rows per subset: partner pairs 81,657 (15.0%); no gene a dependency alone 286,650 (52.7%).

| Model | Seen pair, label agrees with the training lines | Seen pair, label differs from the training lines | New pair, both genes seen | Partner pairs | No gene a dependency alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | 0.411 / 0.335 | 0.586 / 0.581 | 0.325 / 0.314 | 0.301 / 0.323 | 0.389 / 0.279 |
| Backbone head on the buffer-aware reader's terms | 0.537 / 0.552 | 0.392 / 0.440 | 0.419 / 0.534 | 0.356 / 0.346 | 0.431 / 0.376 |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 0.451 / 0.544 | 0.393 / 0.404 | 0.381 / 0.499 | 0.428 / 0.362 | 0.413 / 0.377 |
| Pair identity (hit rate in the study's training lines) | 1.000 / 1.000 | 0.005 / 0.022 | 0.300 / 0.297 | 0.600 / 0.565 | 0.604 / 0.564 |
| Pan-essentiality (gene mean GeneEffect) | 0.412 / 0.337 | 0.586 / 0.632 | 0.329 / 0.382 | 0.294 / 0.299 | 0.376 / 0.285 |
| Measured GeneEffect (reference, lines with GeneEffect) | 0.472 / 0.286 | 0.628 / 0.667 | 0.281 / 0.416 | 0.374 / 0.268 | 0.381 / 0.232 |
| Labels as the score (check: the ceiling) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Random score (check: about 0.10) | 0.129 / 0.121 | 0.477 / 0.512 | 0.224 / 0.220 | 0.172 / 0.092 | 0.105 / 0.118 |
| KG4SL (knowledge graph, official code, trained on the training lines) | 0.538 / 0.579 | 0.431 / 0.432 | 0.310 / 0.323 | 0.370 / 0.365 | 0.378 / 0.364 |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 0.129 / 0.139 | 0.545 / 0.574 | 0.174 / 0.222 | 0.126 / 0.143 | 0.133 / 0.145 |
| PARIS (random-forest dependency pairs, Python port of the R code) | 0.251 / 0.275 | 0.482 / 0.432 | 0.322 / 0.402 | 0.260 / 0.275 | 0.207 / 0.236 |
| SLIdR (mutation-dependency test, Python port of the R code) | 0.103 / 0.107 | 0.571 / 0.638 | 0.161 / 0.259 | 0.091 / 0.134 | 0.108 / 0.120 |
| De Kegel 2021 paralog classifier (released scores) | 0.359 / 0.458 | 0.499 / 0.496 | 0.293 / 0.392 | 0.338 / 0.313 | 0.331 / 0.320 |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 0.399 / 0.547 | 0.441 / 0.518 | 0.330 / 0.411 | 0.483 / 0.485 | 0.439 / 0.416 |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 0.135 / 0.200 | 0.555 / 0.552 | 0.148 / 0.270 | 0.130 / 0.161 | 0.185 / 0.151 |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | 0.673 / 0.749 | 0.320 / 0.423 | 0.475 / 0.354 | 0.455 / 0.471 | 0.544 / 0.497 |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's | 0.671 / 0.809 | 0.297 / 0.342 | 0.483 / 0.426 | 0.430 / 0.547 | 0.544 / 0.497 |
| Gene-knowledge model without cell tokens (each line reads its mean token alone) | 0.725 / 0.820 | 0.263 / 0.346 | 0.472 / 0.403 | 0.512 / 0.560 | 0.556 / 0.552 |

**Table 4.** Recall in the top 10% (AUROC) of each model that scores only part of the held-out rows, on those rows only, beside the training SL fraction and the labels (the ceiling) on the same rows; mean over lines. Higher is better.

| Model | Rows covered | Validation, model | Validation, training SL fraction | Validation, labels | Test, model | Test, training SL fraction | Test, labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Backbone head on the buffer-aware reader's terms | 238,048 (43.8%) | 0.337 (0.68) | 0.501 (0.77) | 1.000 (1.00) | 0.364 (0.66) | 0.532 (0.77) | 1.000 (1.00) |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 238,048 (43.8%) | 0.345 (0.68) | 0.501 (0.77) | 1.000 (1.00) | 0.353 (0.66) | 0.532 (0.77) | 1.000 (1.00) |
| KG4SL (knowledge graph, official code, trained on the training lines) | 523,075 (96.2%) | 0.370 (0.69) | 0.540 (0.77) | 1.000 (1.00) | 0.339 (0.65) | 0.541 (0.77) | 1.000 (1.00) |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 6,804 (1.3%) | 0.511 (0.70) | 0.489 (0.74) | 1.000 (1.00) | 0.419 (0.75) | 0.584 (0.80) | 1.000 (1.00) |
| PARIS (random-forest dependency pairs, Python port of the R code) | 511,485 (94.1%) | 0.219 (0.57) | 0.540 (0.77) | 1.000 (1.00) | 0.221 (0.58) | 0.541 (0.77) | 1.000 (1.00) |
| SLIdR (mutation-dependency test, Python port of the R code) | 94,794 (17.4%) | 0.132 (0.49) | 0.553 (0.75) | 1.000 (1.00) | 0.163 (0.51) | 0.469 (0.71) | 1.000 (1.00) |
| De Kegel 2021 paralog classifier (released scores) | 112,567 (20.7%) | 0.457 (0.79) | 0.794 (0.89) | 1.000 (1.00) | 0.426 (0.79) | 0.720 (0.89) | 1.000 (1.00) |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 108,052 (19.9%) | 0.723 (0.92) | 0.787 (0.89) | 1.000 (1.00) | 0.685 (0.90) | 0.726 (0.88) | 1.000 (1.00) |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 472,797 (87.0%) | 0.161 (0.60) | 0.601 (0.80) | 1.000 (1.00) | 0.143 (0.56) | 0.580 (0.79) | 1.000 (1.00) |

**Table 5.** Paired difference in recall in the top 10% (model minus control), mean over lines, with a 95% bootstrap interval over lines and the line count. Higher is better.

| Model − control | Pairs | Validation | Test |
| --- | --- | ---: | ---: |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Gene-knowledge model without cell tokens (each line reads its mean token alone) | All pairs | -0.008 [-0.029, +0.013] (17) | -0.016 [-0.053, +0.015] (16) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge model without cell tokens (each line reads its mean token alone) | All pairs | -0.023 [-0.056, +0.005] (17) | -0.028 [-0.072, +0.006] (16) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | All pairs | -0.014 [-0.037, +0.008] (17) | -0.012 [-0.044, +0.018] (16) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | All pairs | +0.145 [+0.084, +0.203] (17) | +0.090 [+0.037, +0.147] (16) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − KG4SL (knowledge graph, official code, trained on the training lines) | All pairs | +0.133 [+0.059, +0.211] (17) | +0.147 [+0.072, +0.227] (16) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Backbone head on the buffer-aware reader's terms | All pairs | +0.106 [+0.051, +0.165] (17) | +0.124 [+0.059, +0.193] (16) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Gene-knowledge model without cell tokens (each line reads its mean token alone) | Seen pair, label agrees with the training lines | -0.052 [-0.130, +0.005] (15) | -0.071 [-0.176, -0.002] (12) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge model without cell tokens (each line reads its mean token alone) | Seen pair, label agrees with the training lines | -0.054 [-0.137, +0.008] (15) | -0.011 [-0.032, +0.007] (12) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | Seen pair, label agrees with the training lines | -0.002 [-0.046, +0.046] (15) | +0.060 [-0.008, +0.170] (12) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | Seen pair, label agrees with the training lines | +0.274 [+0.160, +0.393] (15) | +0.202 [+0.061, +0.313] (12) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − KG4SL (knowledge graph, official code, trained on the training lines) | Seen pair, label agrees with the training lines | +0.135 [+0.024, +0.236] (15) | +0.170 [+0.031, +0.294] (12) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Backbone head on the buffer-aware reader's terms | Seen pair, label agrees with the training lines | +0.136 [-0.065, +0.278] (15) | +0.197 [+0.074, +0.295] (12) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Gene-knowledge model without cell tokens (each line reads its mean token alone) | Seen pair, label differs from the training lines | +0.058 [+0.002, +0.117] (15) | +0.077 [+0.037, +0.115] (13) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge model without cell tokens (each line reads its mean token alone) | Seen pair, label differs from the training lines | +0.034 [-0.011, +0.084] (15) | -0.004 [-0.063, +0.044] (13) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | Seen pair, label differs from the training lines | -0.023 [-0.109, +0.070] (15) | -0.082 [-0.137, -0.033] (13) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | Seen pair, label differs from the training lines | -0.121 [-0.215, -0.024] (15) | -0.094 [-0.265, +0.033] (13) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − KG4SL (knowledge graph, official code, trained on the training lines) | Seen pair, label differs from the training lines | -0.110 [-0.236, +0.012] (15) | -0.009 [-0.158, +0.122] (13) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Backbone head on the buffer-aware reader's terms | Seen pair, label differs from the training lines | -0.072 [-0.164, +0.004] (15) | -0.017 [-0.127, +0.081] (13) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Gene-knowledge model without cell tokens (each line reads its mean token alone) | New pair, both genes seen | +0.002 [-0.027, +0.032] (8) | -0.049 [-0.171, +0.075] (10) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge model without cell tokens (each line reads its mean token alone) | New pair, both genes seen | +0.010 [-0.045, +0.079] (8) | +0.023 [-0.214, +0.308] (10) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | New pair, both genes seen | +0.008 [-0.040, +0.083] (8) | +0.072 [-0.125, +0.330] (10) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | New pair, both genes seen | +0.145 [+0.013, +0.287] (8) | -0.057 [-0.194, +0.074] (10) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − KG4SL (knowledge graph, official code, trained on the training lines) | New pair, both genes seen | +0.165 [+0.009, +0.430] (8) | +0.031 [-0.162, +0.213] (10) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Backbone head on the buffer-aware reader's terms | New pair, both genes seen | +0.056 [-0.144, +0.343] (8) | -0.180 [-0.421, +0.027] (10) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Gene-knowledge model without cell tokens (each line reads its mean token alone) | Partner pairs | -0.057 [-0.138, +0.003] (16) | -0.089 [-0.229, +0.025] (15) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge model without cell tokens (each line reads its mean token alone) | Partner pairs | -0.082 [-0.156, -0.024] (16) | -0.013 [-0.182, +0.126] (15) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | Partner pairs | -0.025 [-0.058, +0.005] (16) | +0.076 [-0.005, +0.176] (15) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | Partner pairs | -0.028 [-0.149, +0.076] (16) | -0.014 [-0.095, +0.055] (15) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − KG4SL (knowledge graph, official code, trained on the training lines) | Partner pairs | +0.085 [-0.063, +0.207] (16) | +0.107 [-0.010, +0.202] (15) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Backbone head on the buffer-aware reader's terms | Partner pairs | +0.099 [+0.047, +0.153] (16) | +0.125 [+0.026, +0.200] (15) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Gene-knowledge model without cell tokens (each line reads its mean token alone) | No gene a dependency alone | -0.012 [-0.048, +0.022] (12) | -0.056 [-0.100, -0.013] (11) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge model without cell tokens (each line reads its mean token alone) | No gene a dependency alone | -0.012 [-0.055, +0.030] (12) | -0.055 [-0.113, -0.010] (11) |
| Gene-knowledge, cell-set model with CellNet's cell embeddings in place of Tx1's − Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) | No gene a dependency alone | -0.000 [-0.046, +0.052] (12) | +0.000 [-0.046, +0.045] (11) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | No gene a dependency alone | +0.105 [+0.042, +0.169] (12) | +0.081 [-0.002, +0.174] (11) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − KG4SL (knowledge graph, official code, trained on the training lines) | No gene a dependency alone | +0.166 [+0.046, +0.285] (12) | +0.133 [+0.030, +0.234] (11) |
| Gene-knowledge, cell-set model (pair head on gene-in-line embeddings, pretrained on GeneEffect) − Backbone head on the buffer-aware reader's terms | No gene a dependency alone | +0.113 [+0.056, +0.170] (12) | +0.121 [+0.043, +0.202] (11) |

Label agreement (the stage-two ceiling), within a line across lists: 167,827 pairs, agreement 0.983, SL given SL 0.249; within a study across lines: 337,952 pairs, agreement 0.961, SL given SL 0.298.

Other line groups, AUROC and AUPRC are in `metrics.json`; each line in `lines.csv`; each screen, the non-genetic ones included, in `lists.csv`.

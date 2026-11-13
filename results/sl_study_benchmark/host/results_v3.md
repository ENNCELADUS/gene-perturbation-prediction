# SL benchmark baselines: benchmark_baselines_all_seed0_v3

**Table 1.** Recall in the top 10% on the multi-study SL benchmark, mean over held-out lines (each line: the mean over its studies of the mean over their genetic screens with both classes). Higher is better; a random ranking gives 0.10. Best in bold.

| Model | Validation (15 lines) | Test (13 lines) |
| --- | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | 0.381 | 0.298 |
| Conditional score (backbone: B's predicted change when A is lost) | 0.305 | 0.318 |
| Conditional score, context-ablated (training mean profile) | 0.305 | 0.324 |
| Backbone head (logistic, on the conditional terms) | 0.384 | 0.353 |
| Backbone head, context-ablated (training mean profile) | 0.368 | 0.347 |
| Conditional score, buffer-aware reader | 0.303 | 0.320 |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | 0.310 | 0.321 |
| Backbone head on the buffer-aware reader's terms | 0.394 | 0.361 |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 0.383 | 0.355 |
| Pair identity (hit rate in the study's training lines) | **0.540** | **0.541** |
| Pan-essentiality (gene mean GeneEffect) | 0.380 | 0.297 |
| Measured GeneEffect (reference, lines with GeneEffect) | 0.425 | 0.296 |
| Labels as the score (check: the ceiling) | 1.000 | 1.000 |
| Random score (check: about 0.10) | 0.102 | 0.124 |
| KG4SL (knowledge graph, official code, trained on the training lines) | 0.367 | 0.338 |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 0.123 | 0.134 |
| PARIS (random-forest dependency pairs, Python port of the R code) | 0.219 | 0.213 |
| SLIdR (mutation-dependency test, Python port of the R code) | 0.095 | 0.115 |
| De Kegel 2021 paralog classifier (released scores) | 0.281 | 0.293 |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | 0.296 | 0.304 |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 0.355 | 0.395 |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 0.154 | 0.138 |

**Table 2.** Difference from pair identity in recall in the top 10%, mean over lines, with a 95% bootstrap interval over lines. Higher is better.

| Model − pair identity | Validation | Test |
| --- | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | -0.167 [-0.294, -0.049] | -0.239 [-0.379, -0.098] |
| Conditional score (backbone: B's predicted change when A is lost) | -0.235 [-0.309, -0.160] | -0.201 [-0.274, -0.134] |
| Conditional score, context-ablated (training mean profile) | -0.235 [-0.312, -0.156] | -0.205 [-0.268, -0.136] |
| Backbone head (logistic, on the conditional terms) | -0.163 [-0.226, -0.098] | -0.175 [-0.246, -0.109] |
| Backbone head, context-ablated (training mean profile) | -0.183 [-0.253, -0.101] | -0.181 [-0.248, -0.113] |
| Conditional score, buffer-aware reader | -0.238 [-0.320, -0.160] | -0.198 [-0.273, -0.119] |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | -0.229 [-0.304, -0.148] | -0.207 [-0.275, -0.139] |
| Backbone head on the buffer-aware reader's terms | -0.153 [-0.228, -0.084] | -0.165 [-0.235, -0.096] |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | -0.165 [-0.234, -0.092] | -0.172 [-0.246, -0.104] |
| Pan-essentiality (gene mean GeneEffect) | -0.169 [-0.303, -0.064] | -0.241 [-0.394, -0.095] |
| Measured GeneEffect (reference, lines with GeneEffect) | -0.203 [-0.355, -0.069] | -0.289 [-0.452, -0.124] |
| KG4SL (knowledge graph, official code, trained on the training lines) | -0.164 [-0.254, -0.062] | -0.174 [-0.261, -0.087] |
| Cilantro-SL (Geneformer deletions, official code, refitted) | -0.415 [-0.556, -0.264] | -0.402 [-0.536, -0.265] |
| PARIS (random-forest dependency pairs, Python port of the R code) | -0.326 [-0.439, -0.206] | -0.319 [-0.425, -0.219] |
| SLIdR (mutation-dependency test, Python port of the R code) | -0.443 [-0.599, -0.298] | -0.423 [-0.562, -0.283] |
| De Kegel 2021 paralog classifier (released scores) | -0.253 [-0.333, -0.173] | -0.227 [-0.302, -0.144] |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | -0.219 [-0.311, -0.136] | -0.190 [-0.279, -0.106] |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | -0.183 [-0.256, -0.111] | -0.124 [-0.179, -0.069] |
| MiT4SL (cell-line-specific graph model, official code, refitted) | -0.389 [-0.505, -0.253] | -0.397 [-0.524, -0.254] |

**Table 3.** Recall in the top 10% for each stratum and subset of held-out pairs, ranked inside it in each screen, mean over lines (validation / test). Higher is better. Held-out rows per subset: partner pairs 81,657 (15.0%); no gene a dependency alone 286,650 (52.7%).

| Model | Seen pair, label agrees with the training lines | Seen pair, label differs from the training lines | New pair, both genes seen | Partner pairs | No gene a dependency alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | 0.411 / 0.333 | 0.586 / 0.581 | 0.324 / 0.314 | 0.301 / 0.321 | 0.390 / 0.279 |
| Conditional score (backbone: B's predicted change when A is lost) | 0.433 / 0.494 | 0.435 / 0.461 | 0.242 / 0.524 | 0.347 / 0.368 | 0.337 / 0.341 |
| Conditional score, context-ablated (training mean profile) | 0.399 / 0.489 | 0.440 / 0.431 | 0.247 / 0.526 | 0.369 / 0.385 | 0.335 / 0.328 |
| Backbone head (logistic, on the conditional terms) | 0.512 / 0.516 | 0.433 / 0.413 | 0.409 / 0.538 | 0.353 / 0.330 | 0.420 / 0.371 |
| Backbone head, context-ablated (training mean profile) | 0.441 / 0.519 | 0.398 / 0.396 | 0.376 / 0.530 | 0.370 / 0.348 | 0.392 / 0.385 |
| Conditional score, buffer-aware reader | 0.443 / 0.502 | 0.445 / 0.503 | 0.365 / 0.494 | 0.355 / 0.383 | 0.325 / 0.351 |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | 0.408 / 0.512 | 0.473 / 0.476 | 0.366 / 0.497 | 0.361 / 0.388 | 0.332 / 0.333 |
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
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | 0.344 / 0.440 | 0.476 / 0.580 | 0.215 / 0.344 | 0.377 / 0.391 | 0.453 / 0.400 |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 0.399 / 0.547 | 0.441 / 0.518 | 0.330 / 0.411 | 0.483 / 0.485 | 0.439 / 0.416 |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 0.135 / 0.200 | 0.555 / 0.552 | 0.148 / 0.270 | 0.130 / 0.161 | 0.185 / 0.151 |

**Table 4.** Recall in the top 10% (AUROC) of each model that scores only part of the held-out rows, on those rows only, beside the training SL fraction and the labels (the ceiling) on the same rows; mean over lines. Higher is better.

| Model | Rows covered | Validation, model | Validation, training SL fraction | Validation, labels | Test, model | Test, training SL fraction | Test, labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG4SL (knowledge graph, official code, trained on the training lines) | 523,075 (96.2%) | 0.370 (0.69) | 0.540 (0.77) | 1.000 (1.00) | 0.339 (0.65) | 0.541 (0.77) | 1.000 (1.00) |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 6,804 (1.3%) | 0.511 (0.70) | 0.489 (0.74) | 1.000 (1.00) | 0.419 (0.75) | 0.584 (0.80) | 1.000 (1.00) |
| Conditional score (backbone: B's predicted change when A is lost) | 238,048 (43.8%) | 0.299 (0.62) | 0.501 (0.77) | 1.000 (1.00) | 0.346 (0.65) | 0.532 (0.77) | 1.000 (1.00) |
| Conditional score, context-ablated (training mean profile) | 238,048 (43.8%) | 0.298 (0.63) | 0.501 (0.77) | 1.000 (1.00) | 0.333 (0.65) | 0.532 (0.77) | 1.000 (1.00) |
| Backbone head (logistic, on the conditional terms) | 238,048 (43.8%) | 0.332 (0.68) | 0.501 (0.77) | 1.000 (1.00) | 0.333 (0.67) | 0.532 (0.77) | 1.000 (1.00) |
| Backbone head, context-ablated (training mean profile) | 238,048 (43.8%) | 0.336 (0.68) | 0.501 (0.77) | 1.000 (1.00) | 0.351 (0.66) | 0.532 (0.77) | 1.000 (1.00) |
| Conditional score, buffer-aware reader | 238,048 (43.8%) | 0.290 (0.63) | 0.501 (0.77) | 1.000 (1.00) | 0.342 (0.64) | 0.532 (0.77) | 1.000 (1.00) |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | 238,048 (43.8%) | 0.294 (0.63) | 0.501 (0.77) | 1.000 (1.00) | 0.324 (0.65) | 0.532 (0.77) | 1.000 (1.00) |
| Backbone head on the buffer-aware reader's terms | 238,048 (43.8%) | 0.337 (0.68) | 0.501 (0.77) | 1.000 (1.00) | 0.364 (0.66) | 0.532 (0.77) | 1.000 (1.00) |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 238,048 (43.8%) | 0.345 (0.68) | 0.501 (0.77) | 1.000 (1.00) | 0.353 (0.66) | 0.532 (0.77) | 1.000 (1.00) |
| PARIS (random-forest dependency pairs, Python port of the R code) | 511,485 (94.1%) | 0.219 (0.57) | 0.540 (0.77) | 1.000 (1.00) | 0.221 (0.58) | 0.541 (0.77) | 1.000 (1.00) |
| SLIdR (mutation-dependency test, Python port of the R code) | 94,794 (17.4%) | 0.132 (0.49) | 0.553 (0.75) | 1.000 (1.00) | 0.163 (0.51) | 0.469 (0.71) | 1.000 (1.00) |
| De Kegel 2021 paralog classifier (released scores) | 112,567 (20.7%) | 0.457 (0.79) | 0.794 (0.89) | 1.000 (1.00) | 0.426 (0.79) | 0.720 (0.89) | 1.000 (1.00) |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | 100,966 (18.6%) | 0.686 (0.83) | 0.765 (0.87) | 1.000 (1.00) | 0.645 (0.88) | 0.739 (0.88) | 1.000 (1.00) |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 108,052 (19.9%) | 0.723 (0.92) | 0.787 (0.89) | 1.000 (1.00) | 0.685 (0.90) | 0.726 (0.88) | 1.000 (1.00) |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 472,797 (87.0%) | 0.161 (0.60) | 0.601 (0.80) | 1.000 (1.00) | 0.143 (0.56) | 0.580 (0.79) | 1.000 (1.00) |

**Table 5.** Paired difference in recall in the top 10% (model minus control), mean over lines, with a 95% bootstrap interval over lines and the line count. Higher is better.

| Model − control | Pairs | Validation | Test |
| --- | --- | ---: | ---: |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | All pairs | -0.006 [-0.022, +0.008] (17) | -0.001 [-0.024, +0.020] (16) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | All pairs | +0.011 [-0.007, +0.028] (17) | +0.007 [-0.013, +0.028] (16) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | All pairs | -0.000 [-0.011, +0.008] (17) | -0.006 [-0.027, +0.009] (16) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | All pairs | +0.016 [-0.004, +0.036] (17) | +0.005 [-0.013, +0.026] (16) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.036 [-0.095, +0.196] (15) | -0.010 [-0.031, +0.009] (12) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.087 [+0.005, +0.230] (15) | +0.008 [-0.010, +0.025] (12) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.035 [-0.091, +0.203] (15) | +0.004 [-0.010, +0.022] (12) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.072 [-0.022, +0.217] (15) | -0.003 [-0.027, +0.018] (12) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | Seen pair, label differs from the training lines | -0.029 [-0.079, +0.013] (15) | +0.027 [-0.004, +0.056] (13) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | Seen pair, label differs from the training lines | -0.001 [-0.030, +0.026] (15) | +0.037 [-0.019, +0.100] (13) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | Seen pair, label differs from the training lines | -0.005 [-0.035, +0.025] (15) | +0.030 [+0.004, +0.064] (13) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | Seen pair, label differs from the training lines | +0.035 [-0.027, +0.115] (15) | +0.018 [-0.027, +0.076] (13) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | New pair, both genes seen | -0.001 [-0.015, +0.010] (8) | -0.003 [-0.052, +0.055] (10) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | New pair, both genes seen | +0.037 [-0.004, +0.100] (8) | +0.034 [-0.013, +0.105] (10) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | New pair, both genes seen | -0.005 [-0.012, +0.000] (8) | -0.002 [-0.048, +0.052] (10) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | New pair, both genes seen | +0.034 [-0.012, +0.100] (8) | +0.008 [-0.043, +0.073] (10) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | Partner pairs | -0.006 [-0.022, +0.008] (16) | -0.005 [-0.032, +0.019] (15) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | Partner pairs | -0.072 [-0.199, +0.009] (16) | -0.016 [-0.052, +0.017] (15) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | Partner pairs | -0.023 [-0.049, -0.000] (16) | -0.016 [-0.050, +0.009] (15) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | Partner pairs | -0.017 [-0.044, +0.009] (16) | -0.018 [-0.054, +0.010] (15) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | No gene a dependency alone | -0.008 [-0.030, +0.013] (12) | +0.018 [+0.002, +0.034] (11) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | No gene a dependency alone | +0.017 [+0.002, +0.036] (12) | -0.001 [-0.020, +0.015] (11) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | No gene a dependency alone | +0.001 [-0.011, +0.014] (12) | +0.013 [+0.001, +0.028] (11) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | No gene a dependency alone | +0.028 [+0.010, +0.053] (12) | -0.014 [-0.051, +0.009] (11) |

Label agreement (the stage-two ceiling), within a line across lists: 167,827 pairs, agreement 0.983, SL given SL 0.249; within a study across lines: 337,952 pairs, agreement 0.961, SL given SL 0.298.

Other line groups, AUROC and AUPRC are in `metrics.json`; each line in `lines.csv`; each screen, the non-genetic ones included, in `lists.csv`.

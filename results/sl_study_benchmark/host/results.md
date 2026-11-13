# SL benchmark baselines: benchmark_baselines_all_seed0_v2

**Table 1.** Recall in the top 10% on the multi-study SL benchmark, mean over held-out lines (each line: the mean over its studies of the mean over their genetic screens with both classes). Higher is better; a random ranking gives 0.10. Best in bold.

| Model | Validation (15 lines) | Test (13 lines) |
| --- | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | 0.371 | 0.276 |
| Conditional score (backbone: B's predicted change when A is lost) | 0.295 | 0.288 |
| Conditional score, context-ablated (training mean profile) | 0.296 | 0.295 |
| Backbone head (logistic, on the conditional terms) | 0.374 | 0.322 |
| Backbone head, context-ablated (training mean profile) | 0.358 | 0.314 |
| Conditional score, buffer-aware reader | 0.293 | 0.287 |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | 0.300 | 0.291 |
| Backbone head on the buffer-aware reader's terms | 0.383 | 0.328 |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 0.373 | 0.321 |
| Pair identity (hit rate in the study's training lines) | **0.524** | **0.495** |
| Pan-essentiality (gene mean GeneEffect) | 0.369 | 0.276 |
| Measured GeneEffect (reference, lines with GeneEffect) | 0.412 | 0.269 |
| Labels as the score (check: the ceiling) | 0.964 | 0.920 |
| Random score (check: about 0.10) | 0.093 | 0.106 |
| KG4SL (knowledge graph, official code, trained on the training lines) | 0.352 | 0.306 |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 0.113 | 0.115 |
| PARIS (random-forest dependency pairs, Python port of the R code) | 0.208 | 0.189 |
| SLIdR (mutation-dependency test, Python port of the R code) | 0.086 | 0.097 |
| De Kegel 2021 paralog classifier (released scores) | 0.270 | 0.267 |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | 0.283 | 0.270 |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 0.343 | 0.359 |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 0.144 | 0.120 |

**Table 2.** Difference from pair identity in recall in the top 10%, mean over lines, with a 95% bootstrap interval over lines. Higher is better.

| Model − pair identity | Validation | Test |
| --- | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | -0.162 [-0.289, -0.045] | -0.219 [-0.337, -0.095] |
| Conditional score (backbone: B's predicted change when A is lost) | -0.229 [-0.306, -0.151] | -0.192 [-0.264, -0.128] |
| Conditional score, context-ablated (training mean profile) | -0.229 [-0.308, -0.150] | -0.194 [-0.257, -0.128] |
| Backbone head (logistic, on the conditional terms) | -0.158 [-0.221, -0.093] | -0.167 [-0.235, -0.103] |
| Backbone head, context-ablated (training mean profile) | -0.177 [-0.249, -0.096] | -0.175 [-0.242, -0.108] |
| Conditional score, buffer-aware reader | -0.233 [-0.315, -0.155] | -0.192 [-0.267, -0.115] |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | -0.224 [-0.299, -0.142] | -0.198 [-0.266, -0.130] |
| Backbone head on the buffer-aware reader's terms | -0.148 [-0.223, -0.078] | -0.160 [-0.227, -0.093] |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | -0.159 [-0.230, -0.084] | -0.167 [-0.238, -0.099] |
| Pan-essentiality (gene mean GeneEffect) | -0.164 [-0.298, -0.060] | -0.220 [-0.344, -0.092] |
| Measured GeneEffect (reference, lines with GeneEffect) | -0.194 [-0.340, -0.062] | -0.258 [-0.393, -0.114] |
| KG4SL (knowledge graph, official code, trained on the training lines) | -0.165 [-0.253, -0.068] | -0.166 [-0.252, -0.083] |
| Cilantro-SL (Geneformer deletions, official code, refitted) | -0.410 [-0.546, -0.261] | -0.379 [-0.495, -0.258] |
| PARIS (random-forest dependency pairs, Python port of the R code) | -0.322 [-0.434, -0.203] | -0.302 [-0.398, -0.210] |
| SLIdR (mutation-dependency test, Python port of the R code) | -0.437 [-0.590, -0.292] | -0.399 [-0.525, -0.265] |
| De Kegel 2021 paralog classifier (released scores) | -0.249 [-0.331, -0.169] | -0.212 [-0.276, -0.134] |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | -0.217 [-0.308, -0.134] | -0.186 [-0.276, -0.099] |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | -0.181 [-0.254, -0.109] | -0.121 [-0.177, -0.065] |
| MiT4SL (cell-line-specific graph model, official code, refitted) | -0.384 [-0.502, -0.251] | -0.373 [-0.493, -0.241] |

**Table 3.** Recall in the top 10% for each stratum and subset of held-out pairs, ranked inside it in each screen, mean over lines (validation / test). Higher is better. Held-out rows per subset: partner pairs 81,657 (15.0%); no gene a dependency alone 286,650 (52.7%).

| Model | Seen pair, label agrees with the training lines | Seen pair, label differs from the training lines | New pair, both genes seen | Partner pairs | No gene a dependency alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage one's predicted GeneEffect (linear expression model) | 0.402 / 0.314 | 0.100 / 0.122 | 0.309 / 0.231 | 0.280 / 0.288 | 0.373 / 0.249 |
| Conditional score (backbone: B's predicted change when A is lost) | 0.422 / 0.472 | 0.082 / 0.072 | 0.234 / 0.306 | 0.311 / 0.318 | 0.327 / 0.301 |
| Conditional score, context-ablated (training mean profile) | 0.389 / 0.468 | 0.082 / 0.067 | 0.239 / 0.308 | 0.323 / 0.322 | 0.324 / 0.287 |
| Backbone head (logistic, on the conditional terms) | 0.502 / 0.496 | 0.098 / 0.066 | 0.400 / 0.322 | 0.332 / 0.279 | 0.408 / 0.329 |
| Backbone head, context-ablated (training mean profile) | 0.432 / 0.499 | 0.062 / 0.065 | 0.368 / 0.306 | 0.344 / 0.288 | 0.381 / 0.338 |
| Conditional score, buffer-aware reader | 0.431 / 0.481 | 0.084 / 0.079 | 0.257 / 0.295 | 0.322 / 0.330 | 0.315 / 0.308 |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | 0.397 / 0.491 | 0.087 / 0.075 | 0.258 / 0.296 | 0.326 / 0.334 | 0.321 / 0.292 |
| Backbone head on the buffer-aware reader's terms | 0.527 / 0.533 | 0.061 / 0.071 | 0.406 / 0.332 | 0.335 / 0.295 | 0.418 / 0.332 |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 0.442 / 0.526 | 0.060 / 0.067 | 0.373 / 0.303 | 0.402 / 0.309 | 0.401 / 0.333 |
| Pair identity (hit rate in the study's training lines) | 0.962 / 0.926 | 0.001 / 0.002 | 0.100 / 0.100 | 0.548 / 0.478 | 0.587 / 0.511 |
| Pan-essentiality (gene mean GeneEffect) | 0.403 / 0.317 | 0.100 / 0.125 | 0.311 / 0.250 | 0.273 / 0.267 | 0.359 / 0.254 |
| Measured GeneEffect (reference, lines with GeneEffect) | 0.464 / 0.273 | 0.096 / 0.104 | 0.253 / 0.298 | 0.356 / 0.221 | 0.366 / 0.203 |
| Labels as the score (check: the ceiling) | 0.962 / 0.926 | 0.278 / 0.243 | 0.842 / 0.718 | 0.923 / 0.864 | 0.953 / 0.890 |
| Random score (check: about 0.10) | 0.119 / 0.103 | 0.081 / 0.087 | 0.103 / 0.106 | 0.167 / 0.073 | 0.094 / 0.094 |
| KG4SL (knowledge graph, official code, trained on the training lines) | 0.521 / 0.542 | 0.079 / 0.081 | 0.278 / 0.228 | 0.346 / 0.297 | 0.359 / 0.325 |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 0.118 / 0.117 | 0.102 / 0.097 | 0.105 / 0.111 | 0.108 / 0.112 | 0.119 / 0.118 |
| PARIS (random-forest dependency pairs, Python port of the R code) | 0.241 / 0.255 | 0.082 / 0.072 | 0.253 / 0.229 | 0.231 / 0.217 | 0.192 / 0.201 |
| SLIdR (mutation-dependency test, Python port of the R code) | 0.094 / 0.087 | 0.105 / 0.111 | 0.092 / 0.109 | 0.074 / 0.109 | 0.095 / 0.094 |
| De Kegel 2021 paralog classifier (released scores) | 0.350 / 0.439 | 0.092 / 0.084 | 0.223 / 0.240 | 0.302 / 0.268 | 0.317 / 0.283 |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | 0.334 / 0.421 | 0.087 / 0.099 | 0.146 / 0.225 | 0.345 / 0.313 | 0.438 / 0.354 |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 0.389 / 0.528 | 0.071 / 0.083 | 0.260 / 0.292 | 0.431 / 0.402 | 0.424 / 0.368 |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 0.124 / 0.181 | 0.101 / 0.092 | 0.130 / 0.171 | 0.123 / 0.121 | 0.168 / 0.125 |

**Table 4.** Recall in the top 10% (AUROC) of each model that scores only part of the held-out rows, on those rows only, beside the training SL fraction and the labels (the ceiling) on the same rows; mean over lines. Higher is better.

| Model | Rows covered | Validation, model | Validation, training SL fraction | Validation, labels | Test, model | Test, training SL fraction | Test, labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG4SL (knowledge graph, official code, trained on the training lines) | 523,075 (96.2%) | 0.354 (0.69) | 0.523 (0.77) | 0.964 (1.00) | 0.306 (0.65) | 0.494 (0.77) | 0.919 (1.00) |
| Cilantro-SL (Geneformer deletions, official code, refitted) | 6,804 (1.3%) | 0.357 (0.70) | 0.324 (0.74) | 0.764 (1.00) | 0.289 (0.75) | 0.394 (0.80) | 0.759 (1.00) |
| Conditional score (backbone: B's predicted change when A is lost) | 238,048 (43.8%) | 0.219 (0.62) | 0.392 (0.77) | 0.818 (1.00) | 0.238 (0.65) | 0.382 (0.77) | 0.782 (1.00) |
| Conditional score, context-ablated (training mean profile) | 238,048 (43.8%) | 0.220 (0.63) | 0.392 (0.77) | 0.818 (1.00) | 0.239 (0.65) | 0.382 (0.77) | 0.782 (1.00) |
| Backbone head (logistic, on the conditional terms) | 238,048 (43.8%) | 0.246 (0.68) | 0.392 (0.77) | 0.818 (1.00) | 0.223 (0.67) | 0.382 (0.77) | 0.782 (1.00) |
| Backbone head, context-ablated (training mean profile) | 238,048 (43.8%) | 0.250 (0.68) | 0.392 (0.77) | 0.818 (1.00) | 0.237 (0.66) | 0.382 (0.77) | 0.782 (1.00) |
| Conditional score, buffer-aware reader | 238,048 (43.8%) | 0.210 (0.63) | 0.392 (0.77) | 0.818 (1.00) | 0.231 (0.64) | 0.382 (0.77) | 0.782 (1.00) |
| Conditional score, buffer-aware reader, context-ablated (training mean profile) | 238,048 (43.8%) | 0.215 (0.63) | 0.392 (0.77) | 0.818 (1.00) | 0.225 (0.65) | 0.382 (0.77) | 0.782 (1.00) |
| Backbone head on the buffer-aware reader's terms | 238,048 (43.8%) | 0.257 (0.68) | 0.392 (0.77) | 0.818 (1.00) | 0.244 (0.66) | 0.382 (0.77) | 0.782 (1.00) |
| Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | 238,048 (43.8%) | 0.262 (0.68) | 0.392 (0.77) | 0.818 (1.00) | 0.238 (0.66) | 0.382 (0.77) | 0.782 (1.00) |
| PARIS (random-forest dependency pairs, Python port of the R code) | 511,485 (94.1%) | 0.208 (0.57) | 0.524 (0.77) | 0.964 (1.00) | 0.195 (0.58) | 0.495 (0.77) | 0.914 (1.00) |
| SLIdR (mutation-dependency test, Python port of the R code) | 94,794 (17.4%) | 0.098 (0.49) | 0.506 (0.75) | 0.885 (1.00) | 0.102 (0.51) | 0.374 (0.71) | 0.823 (1.00) |
| De Kegel 2021 paralog classifier (released scores) | 112,567 (20.7%) | 0.372 (0.79) | 0.686 (0.89) | 0.899 (1.00) | 0.372 (0.79) | 0.595 (0.89) | 0.865 (1.00) |
| Kebabci 2026 paralog classifier (reads measured GeneEffect, reference, official code, refitted) | 100,966 (18.6%) | 0.661 (0.83) | 0.651 (0.87) | 0.888 (1.00) | 0.539 (0.88) | 0.601 (0.88) | 0.829 (1.00) |
| Kebabci 2026 paralog classifier (stage one's predicted GeneEffect in the held-out lines, official code, refitted) | 108,052 (19.9%) | 0.639 (0.92) | 0.691 (0.89) | 0.912 (1.00) | 0.599 (0.90) | 0.615 (0.88) | 0.876 (1.00) |
| MiT4SL (cell-line-specific graph model, official code, refitted) | 472,797 (87.0%) | 0.149 (0.60) | 0.582 (0.80) | 0.959 (1.00) | 0.122 (0.56) | 0.527 (0.79) | 0.907 (1.00) |

**Table 5.** Paired difference in recall in the top 10% (model minus control), mean over lines, with a 95% bootstrap interval over lines and the line count. Higher is better.

| Model − control | Pairs | Validation | Test |
| --- | --- | ---: | ---: |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | All pairs | -0.006 [-0.022, +0.008] (17) | -0.003 [-0.025, +0.016] (16) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | All pairs | +0.010 [-0.008, +0.028] (17) | +0.006 [-0.013, +0.028] (16) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | All pairs | -0.001 [-0.011, +0.008] (17) | -0.007 [-0.029, +0.007] (16) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | All pairs | +0.016 [-0.004, +0.035] (17) | +0.007 [-0.011, +0.027] (16) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.034 [-0.097, +0.196] (15) | -0.010 [-0.031, +0.009] (12) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.085 [+0.003, +0.227] (15) | +0.007 [-0.011, +0.024] (12) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.033 [-0.092, +0.202] (15) | +0.005 [-0.010, +0.023] (12) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | Seen pair, label agrees with the training lines | +0.071 [-0.023, +0.216] (15) | -0.003 [-0.027, +0.018] (12) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | Seen pair, label differs from the training lines | -0.004 [-0.010, +0.002] (15) | +0.004 [-0.002, +0.009] (13) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | Seen pair, label differs from the training lines | +0.001 [-0.004, +0.005] (15) | +0.004 [-0.005, +0.013] (13) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | Seen pair, label differs from the training lines | -0.000 [-0.004, +0.004] (15) | +0.006 [+0.001, +0.011] (13) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | Seen pair, label differs from the training lines | +0.036 [-0.003, +0.106] (15) | +0.001 [-0.006, +0.009] (13) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | New pair, both genes seen | -0.001 [-0.015, +0.010] (8) | -0.002 [-0.047, +0.051] (10) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | New pair, both genes seen | +0.033 [-0.004, +0.095] (8) | +0.029 [-0.015, +0.093] (10) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | New pair, both genes seen | -0.005 [-0.012, +0.000] (8) | -0.002 [-0.044, +0.047] (10) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | New pair, both genes seen | +0.031 [-0.013, +0.098] (8) | +0.016 [-0.022, +0.068] (10) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | Partner pairs | -0.004 [-0.018, +0.009] (16) | -0.004 [-0.029, +0.020] (15) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | Partner pairs | -0.067 [-0.194, +0.012] (16) | -0.014 [-0.044, +0.015] (15) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | Partner pairs | -0.012 [-0.027, +0.003] (16) | -0.005 [-0.021, +0.013] (15) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | Partner pairs | -0.012 [-0.033, +0.010] (16) | -0.009 [-0.033, +0.009] (15) |
| Conditional score, buffer-aware reader − Conditional score, buffer-aware reader, context-ablated (training mean profile) | No gene a dependency alone | -0.007 [-0.029, +0.014] (12) | +0.016 [+0.001, +0.031] (11) |
| Backbone head on the buffer-aware reader's terms − Backbone head on the buffer-aware reader's terms, context-ablated (training mean profile) | No gene a dependency alone | +0.017 [+0.001, +0.035] (12) | -0.001 [-0.018, +0.014] (11) |
| Conditional score (backbone: B's predicted change when A is lost) − Conditional score, context-ablated (training mean profile) | No gene a dependency alone | +0.002 [-0.010, +0.014] (12) | +0.013 [+0.001, +0.029] (11) |
| Backbone head (logistic, on the conditional terms) − Backbone head, context-ablated (training mean profile) | No gene a dependency alone | +0.027 [+0.009, +0.053] (12) | -0.010 [-0.040, +0.010] (11) |

Label agreement (the stage-two ceiling), within a line across lists: 167,827 pairs, agreement 0.983, SL given SL 0.249; within a study across lines: 337,952 pairs, agreement 0.961, SL given SL 0.298.

Other line groups, AUROC and AUPRC are in `metrics.json`; each line in `lines.csv`; each screen, the non-genetic ones included, in `lists.csv`.

# SL backbone: conditional_buffer_seed0

**Table 1.** Stage-one metrics of the linear expression model fitted on the SL split's training side, on the SL lines with GeneEffect (training lines out of fold), beside the same prediction with its lines permuted and the ceiling: one DepMap screen's residual as the prediction of another screen's residual of the same line, on the lines with two screens (124 / 4 / 8 lines, train / val / test).

| Metric | Train, reader | Train, permuted | Train, two screens | Validation, reader | Validation, permuted | Validation, two screens | Test, reader | Test, permuted | Test, two screens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per-line residual Spearman | 0.1781 | -0.0013 | 0.2455 | 0.1834 | 0.0408 | 0.2654 | 0.1944 | 0.0377 | 0.3665 |
| per-gene residual Spearman | 0.1886 | -0.0047 | 0.2362 | 0.1707 | 0.0216 | 0.1941 | 0.1567 | 0.0175 | 0.3446 |
| residual skill score | 0.0299 | -0.0161 | -0.2339 | 0.0295 | -0.0106 | -0.0730 | 0.0271 | -0.0050 | 0.0277 |
| calibration slope | 1.8376 | -0.0388 | 0.3143 | 1.5968 | 0.1115 | 0.4320 | 1.7022 | 0.0457 | 0.5181 |
| top-20 recall of a line's strongest dependencies | 0.0707 | 0.0064 | 0.1056 | 0.0792 | 0.0042 | 0.1000 | 0.0545 | 0.0045 | 0.1625 |
| genes with per-gene Spearman at or above 0.3 | 999.8000 | 0 | 1,496 | 1,613 | 879 | 2,250 | 1,653 | 870 | 2,664 |
| whole-line Pearson (sanity) | 0.9154 | 0.9111 | 0.8288 | 0.9269 | 0.9237 | 0.8163 | 0.9195 | 0.9165 | 0.8618 |

**Table 2.** The head's validation recall in the top 10% at each penalty.

| Penalty | Validation recall in the top 10% |
| ---: | ---: |
| 0.01 | 0.383 |
| 0.1 | 0.383 |
| 1 | 0.383 |
| 10 | 0.383 |
| 100 | 0.383 |

Fit lines: 979. Rows whose genes the reader can edit (train / val / test): 49.9% / 52.0% / 36.5%.


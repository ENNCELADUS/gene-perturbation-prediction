# P1-B response diagnostics

| State | Role | MSE | Energy | Total |
|---|---|---:|---:|---:|
| B-continue | external | 25.378 | 148.1737 | 173.5517 |
| B-continue | train | 5.414395 | 25.83421 | 31.2486 |
| B-continue | val | 5.607499 | 25.12381 | 30.73131 |
| B-init | external | 57.41222 | 422.3561 | 479.7683 |
| B-init | train | 157.8725 | 732.9228 | 890.7953 |
| B-init | val | 162.2795 | 738.3331 | 900.6126 |
| B-interface | external | 20.06163 | 115.8951 | 135.9568 |
| B-interface | train | 5.732379 | 28.03812 | 33.7705 |
| B-interface | val | 5.897379 | 27.15835 | 33.05573 |
| B-joint | external | 27.24425 | 173.3117 | 200.5559 |
| B-joint | train | 102.3275 | 429.1639 | 531.4914 |
| B-joint | val | 105.9621 | 434.0451 | 540.0071 |
| B-unfreeze | external | 26.48032 | 153.5008 | 179.9812 |
| B-unfreeze | train | 4.196776 | 20.65442 | 24.8512 |
| B-unfreeze | val | 4.896327 | 21.63906 | 26.53538 |

Raw metrics: raw_summary.csv; equal-anchor means: equal_anchor_summary.csv. Paired differences: paired_differences.csv; positive identity advantage means correct identity predicts better. Its per-state intervals against zero are in identity_intervals.csv. Cross-context summaries and paired differences are exported separately. Curves and common updates describe actual early-stopping budgets.

Single training seed 0; intervals use 1,000 paired perturbation resamples conditional on selected checkpoints. They do not estimate new-anchor or initialization variability. Jurkat is adaptation-held-out only; ST/Tx1 pretraining exposure remains unresolved. Native preprocessing is unavailable, so no native predictions are reported. A response improvement is not GeneEffect evidence.

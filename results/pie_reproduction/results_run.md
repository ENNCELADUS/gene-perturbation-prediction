# PIE reproduction `pie_reproduction_20261016`

Released checkpoints (`arcinstitute/PIE_replogle_xdataset@f6da960`, `arcinstitute/PIE_replogle_wdataset@055c7a2`) run through the vendored PIE (`21a526a`) on PIE's splits (`396ab95`), against the metrics released with them. Differences are reproduced minus released. Nothing is decided here.

## `replogle_wdataset/hepg2` on `unseen_ctx`

| Metric (`all`) | Released | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.443248 | 0.443248 | -4.17e-07 |
| sig_jaccard | 0.161653 | 0.161654 | +1.25e-06 |
| direction_match | 0.840114 | 0.840114 | +0.00e+00 |
| spearman_nsig | 0.737815 | 0.737809 | -5.60e-06 |
| spearman_lfc | 0.711455 | 0.711454 | -5.36e-07 |
| discrimination_score_l1 | 0.812804 | 0.812806 | +1.33e-06 |

Largest |difference| over 2 contexts: -5.60e-06 (spearman_nsig, `hepg2`).

## `replogle_wdataset/jurkat` on `unseen_ctx`

| Metric (`all`) | Released | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.375403 | 0.375403 | -2.98e-08 |
| sig_jaccard | 0.171954 | 0.171954 | +4.47e-08 |
| direction_match | 0.816767 | 0.816767 | +0.00e+00 |
| spearman_nsig | 0.775437 | 0.775441 | +4.71e-06 |
| spearman_lfc | 0.615647 | 0.615647 | +0.00e+00 |
| discrimination_score_l1 | 0.756747 | 0.756751 | +4.18e-06 |

Largest |difference| over 2 contexts: +4.71e-06 (spearman_nsig, `jurkat`).

## `replogle_wdataset/k562` on `unseen_ctx`

| Metric (`all`) | Released | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.422211 | 0.422211 | +0.00e+00 |
| sig_jaccard | 0.184112 | 0.184112 | +0.00e+00 |
| direction_match | 0.839847 | 0.839847 | +0.00e+00 |
| spearman_nsig | 0.757922 | 0.757922 | +0.00e+00 |
| spearman_lfc | 0.662161 | 0.662161 | +0.00e+00 |
| discrimination_score_l1 | 0.820516 | 0.820518 | +2.56e-06 |

Largest |difference| over 2 contexts: +2.56e-06 (discrimination_score_l1, `k562`).

## `replogle_wdataset/rpe1` on `unseen_ctx`

| Metric (`all`) | Released | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.421119 | 0.421119 | +7.75e-07 |
| sig_jaccard | 0.190553 | 0.190553 | +4.47e-08 |
| direction_match | 0.797341 | 0.797347 | +5.90e-06 |
| spearman_nsig | 0.684258 | 0.684255 | -2.74e-06 |
| spearman_lfc | 0.637869 | 0.637870 | +4.77e-07 |
| discrimination_score_l1 | 0.690169 | 0.690115 | -5.38e-05 |

Largest |difference| over 2 contexts: -5.38e-05 (discrimination_score_l1, `rpe1`).

## `replogle_xdataset` on `test_seen`

| Metric (`all`) | Released | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.380246 | 0.380246 | -6.71e-08 |
| sig_jaccard | 0.152301 | 0.152301 | +0.00e+00 |
| direction_match | 0.740068 | 0.740068 | +0.00e+00 |
| spearman_nsig | 0.623144 | 0.623144 | +0.00e+00 |
| spearman_lfc | 0.512979 | 0.512971 | -8.50e-06 |
| discrimination_score_l1 | 0.631207 | 0.631216 | +8.54e-06 |

Largest |difference| over 5 contexts: +3.75e-05 (discrimination_score_l1, `hepg2`).

## `replogle_xdataset` on `test_unseen`

| Metric (`all`) | Released | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.246348 | 0.246348 | +1.49e-08 |
| sig_jaccard | 0.119641 | 0.119641 | -3.73e-08 |
| direction_match | 0.646620 | 0.646621 | +8.20e-07 |
| spearman_nsig | 0.501628 | 0.501628 | +0.00e+00 |
| spearman_lfc | 0.269830 | 0.269794 | -3.60e-05 |
| discrimination_score_l1 | 0.530854 | 0.530850 | -4.55e-06 |

Largest |difference| over 5 contexts: -1.44e-04 (spearman_lfc, `rpe1`).

## Per-pair metrics

Pairs defined on both sides; a pair undefined on exactly one side is a NaN mismatch.

| Target | Metric | Pairs | NaN mismatch | Max abs difference | Mean abs difference |
| --- | --- | ---: | ---: | ---: | ---: |
| `replogle_wdataset/hepg2` `unseen_ctx` | binary_auprc | 867 | 0 | 4.27e-04 | 8.73e-07 |
| `replogle_wdataset/hepg2` `unseen_ctx` | sig_jaccard | 867 | 0 | 8.72e-04 | 1.26e-06 |
| `replogle_wdataset/hepg2` `unseen_ctx` | direction_match | 867 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/hepg2` `unseen_ctx` | spearman_lfc | 758 | 0 | 3.26e-04 | 6.53e-07 |
| `replogle_wdataset/hepg2` `unseen_ctx` | discrimination_score_l1 | 867 | 0 | 1.15e-03 | 3.99e-06 |
| `replogle_wdataset/jurkat` `unseen_ctx` | binary_auprc | 978 | 0 | 1.06e-05 | 1.30e-08 |
| `replogle_wdataset/jurkat` `unseen_ctx` | sig_jaccard | 978 | 0 | 4.92e-05 | 5.03e-08 |
| `replogle_wdataset/jurkat` `unseen_ctx` | direction_match | 978 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/jurkat` `unseen_ctx` | spearman_lfc | 912 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/jurkat` `unseen_ctx` | discrimination_score_l1 | 978 | 0 | 2.04e-03 | 4.18e-06 |
| `replogle_wdataset/k562` `unseen_ctx` | binary_auprc | 884 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/k562` `unseen_ctx` | sig_jaccard | 884 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/k562` `unseen_ctx` | direction_match | 884 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/k562` `unseen_ctx` | spearman_lfc | 792 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_wdataset/k562` `unseen_ctx` | discrimination_score_l1 | 884 | 0 | 2.26e-03 | 5.12e-06 |
| `replogle_wdataset/rpe1` `unseen_ctx` | binary_auprc | 983 | 0 | 4.21e-04 | 8.65e-07 |
| `replogle_wdataset/rpe1` `unseen_ctx` | sig_jaccard | 983 | 0 | 1.12e-04 | 1.89e-07 |
| `replogle_wdataset/rpe1` `unseen_ctx` | direction_match | 983 | 0 | 7.09e-03 | 8.56e-06 |
| `replogle_wdataset/rpe1` `unseen_ctx` | spearman_lfc | 950 | 0 | 3.95e-04 | 1.49e-06 |
| `replogle_wdataset/rpe1` `unseen_ctx` | discrimination_score_l1 | 983 | 0 | 1.12e-02 | 1.41e-03 |
| `replogle_xdataset` `test_seen` | binary_auprc | 2286 | 0 | 1.97e-04 | 9.18e-08 |
| `replogle_xdataset` `test_seen` | sig_jaccard | 2286 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_xdataset` `test_seen` | direction_match | 2286 | 0 | 0.00e+00 | 0.00e+00 |
| `replogle_xdataset` `test_seen` | spearman_lfc | 2085 | 0 | 1.76e-02 | 9.51e-06 |
| `replogle_xdataset` `test_seen` | discrimination_score_l1 | 2286 | 0 | 1.02e-02 | 2.02e-05 |
| `replogle_xdataset` `test_unseen` | binary_auprc | 3173 | 0 | 9.66e-05 | 9.83e-08 |
| `replogle_xdataset` `test_unseen` | sig_jaccard | 3173 | 0 | 1.27e-04 | 4.00e-08 |
| `replogle_xdataset` `test_unseen` | direction_match | 3173 | 0 | 2.33e-03 | 1.33e-06 |
| `replogle_xdataset` `test_unseen` | spearman_lfc | 2903 | 0 | 1.19e-01 | 4.37e-05 |
| `replogle_xdataset` `test_unseen` | discrimination_score_l1 | 3173 | 0 | 5.40e-03 | 4.99e-06 |

## `replogle_wdataset` `unseen_ctx`, 4 folds

Mean ± sample SD (ddof=1) of the folds' `all` rows.

| Metric | Released | Reproduced | Model card |
| --- | ---: | ---: | ---: |
| binary_auprc | 0.4155 ± 0.0286 | 0.4155 ± 0.0286 | 0.4155 ± 0.0286 |
| spearman_lfc | 0.6568 ± 0.0411 | 0.6568 ± 0.0411 | 0.6568 ± 0.0411 |

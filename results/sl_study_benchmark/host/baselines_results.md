# SL benchmark baselines: benchmark_baselines_20261104

Recall in the top 10% of each list (a random ranking gives 0.10): the known positives of a line's lists, pooled, then the mean over lines; the difference from pair identity per line, with a 95% interval from resampling lines. AUROC is the mean percentile of the known positives; AUPRC lift is AUPRC over the positive rate. A list's negatives are tested and not called, not true negatives. The measured GeneEffect is scored only on lines with GeneEffect. Stage two trains mostly on in-sample stage-one inputs, so the number for lines without GeneEffect is a low estimate; that group is small and mostly non-cancer.

| Split | Group | Scorer | Lines | Positives | Recall top 10% | − pair identity | 95% interval | AUROC | AUPRC lift |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| val | all | measured_geneeffect | 14 | 2137 | 0.331 | -0.275 | [-0.400, -0.146] | 0.677 | 6.051 |
| val | all | pair_identity | 15 | 2084 | 0.588 | +0.000 | [0.000, 0.000] | 0.808 | 244.001 |
| val | all | pan_essentiality | 16 | 2381 | 0.322 | -0.267 | [-0.420, -0.128] | 0.684 | 4.074 |
| val | all | stage_one_production | 16 | 2381 | 0.319 | -0.270 | [-0.430, -0.135] | 0.683 | 4.151 |
| val | with_geneeffect | measured_geneeffect | 14 | 2137 | 0.331 | -0.275 | [-0.406, -0.159] | 0.677 | 6.051 |
| val | with_geneeffect | pair_identity | 13 | 1878 | 0.632 | +0.000 | [0.000, 0.000] | 0.838 | 279.588 |
| val | with_geneeffect | pan_essentiality | 14 | 2148 | 0.333 | -0.299 | [-0.475, -0.149] | 0.694 | 4.327 |
| val | with_geneeffect | stage_one_production | 14 | 2148 | 0.331 | -0.301 | [-0.478, -0.131] | 0.693 | 4.423 |
| val | without_geneeffect | pair_identity | 2 | 206 | 0.301 | +0.000 | [0.000, 0.000] | 0.613 | 12.690 |
| val | without_geneeffect | pan_essentiality | 2 | 233 | 0.240 | -0.060 | [-0.102, -0.019] | 0.616 | 2.302 |
| val | without_geneeffect | stage_one_production | 2 | 233 | 0.235 | -0.065 | [-0.111, -0.019] | 0.614 | 2.247 |
| val | cancer | measured_geneeffect | 14 | 2137 | 0.331 | -0.275 | [-0.394, -0.158] | 0.677 | 6.051 |
| val | cancer | pair_identity | 15 | 2084 | 0.588 | +0.000 | [0.000, 0.000] | 0.808 | 244.001 |
| val | cancer | pan_essentiality | 16 | 2381 | 0.322 | -0.267 | [-0.422, -0.139] | 0.684 | 4.074 |
| val | cancer | stage_one_production | 16 | 2381 | 0.319 | -0.270 | [-0.412, -0.122] | 0.683 | 4.151 |
| test | all | measured_geneeffect | 13 | 7198 | 0.266 | -0.268 | [-0.423, -0.132] | 0.661 | 3.830 |
| test | all | pair_identity | 14 | 5643 | 0.465 | +0.000 | [0.000, 0.000] | 0.754 | 96.607 |
| test | all | pan_essentiality | 16 | 7644 | 0.259 | -0.217 | [-0.347, -0.078] | 0.650 | 3.570 |
| test | all | stage_one_production | 16 | 7644 | 0.268 | -0.209 | [-0.339, -0.071] | 0.652 | 3.655 |
| test | with_geneeffect | measured_geneeffect | 13 | 7198 | 0.266 | -0.268 | [-0.432, -0.123] | 0.661 | 3.830 |
| test | with_geneeffect | pair_identity | 12 | 5587 | 0.521 | +0.000 | [0.000, 0.000] | 0.793 | 112.492 |
| test | with_geneeffect | pan_essentiality | 13 | 7198 | 0.275 | -0.265 | [-0.407, -0.142] | 0.688 | 3.357 |
| test | with_geneeffect | stage_one_production | 13 | 7198 | 0.283 | -0.256 | [-0.398, -0.129] | 0.690 | 3.472 |
| test | without_geneeffect | pair_identity | 2 | 56 | 0.129 | +0.000 | [0.000, 0.000] | 0.516 | 1.293 |
| test | without_geneeffect | pan_essentiality | 3 | 446 | 0.192 | +0.071 | [-0.100, 0.241] | 0.486 | 4.491 |
| test | without_geneeffect | stage_one_production | 3 | 446 | 0.203 | +0.069 | [-0.100, 0.239] | 0.489 | 4.451 |
| test | cancer | measured_geneeffect | 13 | 7198 | 0.266 | -0.268 | [-0.417, -0.133] | 0.661 | 3.830 |
| test | cancer | pair_identity | 13 | 5588 | 0.489 | +0.000 | [0.000, 0.000] | 0.771 | 103.916 |
| test | cancer | pan_essentiality | 14 | 7199 | 0.255 | -0.252 | [-0.381, -0.130] | 0.638 | 3.188 |
| test | cancer | stage_one_production | 14 | 7199 | 0.263 | -0.244 | [-0.375, -0.119] | 0.640 | 3.295 |
| test | non_cancer | pair_identity | 1 | 55 | 0.158 | +0.000 | [0.000, 0.000] | 0.532 | 1.585 |
| test | non_cancer | pan_essentiality | 2 | 445 | 0.289 | +0.241 | [0.241, 0.241] | 0.729 | 6.237 |
| test | non_cancer | stage_one_production | 2 | 445 | 0.305 | +0.239 | [0.239, 0.239] | 0.733 | 6.176 |

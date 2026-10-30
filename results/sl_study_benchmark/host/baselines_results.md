# SL benchmark baselines: benchmark_baselines_seed0

AUPRC over the list's positive rate (1 is a random ranking), averaged within each line, then over lines; 95% interval from resampling lines.

| Scorer | Split | Lines | Lists | AUPRC lift | 95% interval | AUROC |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| measured_geneeffect | test | 12 | 37 | 2.772 | [2.123, 3.516] | 0.633 |
| pair_identity | test | 10 | 29 | 56.622 | [6.825, 149.472] | 0.809 |
| pan_essentiality | test | 12 | 37 | 2.824 | [2.253, 3.460] | 0.657 |
| prior_a_blind | test | 12 | 37 | 2.948 | [2.358, 3.575] | 0.658 |
| measured_geneeffect | val | 12 | 16 | 8.251 | [5.129, 11.507] | 0.778 |
| pair_identity | val | 12 | 15 | 18.200 | [11.613, 26.025] | 0.872 |
| pan_essentiality | val | 12 | 16 | 5.776 | [3.791, 8.146] | 0.783 |
| prior_a_blind | val | 12 | 16 | 5.825 | [3.895, 8.149] | 0.783 |

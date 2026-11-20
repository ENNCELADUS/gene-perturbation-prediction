| Line input, pairs | Validation recall in the top 10% | Validation AUROC | Validation lines | Test recall in the top 10% | Test AUROC | Test lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| The line's own input, all pairs | 0.5085 | 0.7887 | 17 | 0.5008 | 0.7760 | 16 |
| The fit lines' mean input, all pairs | 0.4599 | 0.7852 | 17 | 0.4847 | 0.7761 | 16 |
| The input of another SL line (seeded permutation), all pairs | 0.4866 | 0.7768 | 17 | 0.4902 | 0.7721 | 16 |
| The line's own input, seen pairs whose label agrees with the training lines (303,670 held-out rows) | 0.7254 | 0.8585 | 15 | 0.8198 | 0.9143 | 12 |
| The fit lines' mean input, seen pairs whose label agrees with the training lines | 0.6478 | 0.8483 | 15 | 0.7718 | 0.8985 | 12 |
| The line's own input, seen pairs whose label differs from the training lines (8,164 held-out rows) | 0.2628 | 0.2433 | 15 | 0.3461 | 0.2785 | 13 |
| The fit lines' mean input, seen pairs whose label differs from the training lines | 0.3376 | 0.2986 | 15 | 0.4396 | 0.3364 | 13 |
| The line's own input, new pairs, both genes seen (231,745 held-out rows) | 0.4723 | 0.7963 | 8 | 0.4033 | 0.6523 | 10 |
| The fit lines' mean input, new pairs, both genes seen | 0.2958 | 0.6591 | 8 | 0.3286 | 0.6203 | 10 |

Pearson correlation of the two scores (own input, mean input) over the held-out rows: 0.2606

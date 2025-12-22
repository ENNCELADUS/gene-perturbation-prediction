# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 2 experiments across 2 runs.

## Results Table

| experiment   | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:-------------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| logreg       | True   |        0.3533 |        0.6431 |           0.6251 |           0.8113 | 0.476  |   0.5081 |
| logreg       | False  |        0.3905 |        0.6817 |           0.6714 |           0.833  | 0.515  |   0.5482 |
| pca          | True   |        0.1872 |        0.4248 |           0.4099 |           0.6176 | 0.2891 |   0.311  |
| pca          | False  |        0.1871 |        0.432  |           0.4184 |           0.6215 | 0.2915 |   0.3149 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K

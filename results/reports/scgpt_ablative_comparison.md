# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 3 experiments across 3 runs.

## Results Table

| experiment      | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:----------------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| scgpt           | True   |        0.0628 |        0.1868 |           0.2012 |           0.393  | 0.1163 |   0.1249 |
| scgpt           | False  |        0.0644 |        0.1954 |           0.208  |           0.4048 | 0.1217 |   0.1311 |
| scgpt_head_only | True   |        0.0756 |        0.2324 |           0.2511 |           0.4385 | 0.1434 |   0.1545 |
| scgpt_head_only | False  |        0.0795 |        0.2393 |           0.2557 |           0.4442 | 0.1491 |   0.1607 |
| scgpt_lora_head | True   |        0.0718 |        0.2357 |           0.2455 |           0.4403 | 0.1422 |   0.1554 |
| scgpt_lora_head | False  |        0.0711 |        0.2375 |           0.2462 |           0.4396 | 0.1423 |   0.1563 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K

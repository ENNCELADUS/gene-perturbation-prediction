# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 3 experiments across 4 runs.

## Results Table

| experiment         | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:-------------------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| scgpt              | True   |        0.0692 |        0.2008 |           0.2134 |           0.4038 | 0.1257 |   0.1355 |
| scgpt              | False  |        0.074  |        0.2101 |           0.2274 |           0.4196 | 0.1332 |   0.1432 |
| scgpt_finetune_cls | True   |        0.1856 |        0.4284 |           0.3808 |           0.5825 | 0.2885 |   0.3116 |
| scgpt_finetune_cls | True   |        0.0333 |        0.1059 |           0.1187 |           0.2659 | 0.0651 |   0.0685 |
| scgpt_finetune_cls | False  |        0.1763 |        0.4207 |           0.3733 |           0.5764 | 0.2801 |   0.303  |
| scgpt_finetune_cls | False  |        0.0346 |        0.1071 |           0.1234 |           0.265  | 0.0665 |   0.0697 |
| scgpt_head_only    | True   |        0.1349 |        0.3575 |           0.3666 |           0.5427 | 0.2296 |   0.2486 |
| scgpt_head_only    | False  |        0.1358 |        0.3559 |           0.3648 |           0.5425 | 0.2297 |   0.2484 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K

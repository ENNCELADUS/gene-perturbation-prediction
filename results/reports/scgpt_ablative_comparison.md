# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 3 experiments across 4 runs.

## Results Table

| experiment         | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:-------------------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| scgpt              | True   |        0.0684 |        0.1987 |           0.2097 |           0.403  | 0.1252 |   0.1343 |
| scgpt              | False  |        0.0722 |        0.2089 |           0.2268 |           0.4175 | 0.1314 |   0.1417 |
| scgpt_finetune_cls | True   |        0.1532 |        0.3802 |           0.3578 |           0.5524 | 0.2498 |   0.27   |
| scgpt_finetune_lora | True   |        0.0142 |        0.0533 |           0.0587 |           0.2307 | 0.0337 |   0.0331 |
| scgpt_finetune_cls | False  |        0.1449 |        0.3695 |           0.3466 |           0.5445 | 0.2404 |   0.26   |
| scgpt_finetune_lora | False  |        0.013  |        0.051  |           0.0578 |           0.2282 | 0.0327 |   0.0316 |
| scgpt_head_only    | True   |        0.1385 |        0.3591 |           0.3645 |           0.5446 | 0.2319 |   0.2512 |
| scgpt_head_only    | False  |        0.1313 |        0.3534 |           0.3616 |           0.5391 | 0.227  |   0.2455 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K

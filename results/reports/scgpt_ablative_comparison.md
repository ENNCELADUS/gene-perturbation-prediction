# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of 4 experiments across 4 runs.

## Results Table

| experiment         | mask   |   exact_hit@1 |   exact_hit@5 |   relevant_hit@1 |   relevant_hit@5 |    mrr |   ndcg@5 |
|:-------------------|:-------|--------------:|--------------:|-----------------:|-----------------:|-------:|---------:|
| scgpt              | True   |        0.0656 |        0.1971 |           0.2079 |           0.406  | 0.1228 |   0.1321 |
| scgpt              | False  |        0.0743 |        0.2116 |           0.2276 |           0.417  | 0.1341 |   0.1442 |
| scgpt_finetune_cls | True   |        0.1842 |        0.429  |           0.3809 |           0.5818 | 0.288  |   0.3114 |
| scgpt_finetune_cls | False  |        0.1736 |        0.4205 |           0.3714 |           0.5742 | 0.2783 |   0.3016 |
| scgpt_head_only    | True   |        0.124  |        0.3385 |           0.3512 |           0.5317 | 0.2165 |   0.2341 |
| scgpt_head_only    | False  |        0.1298 |        0.3443 |           0.3611 |           0.5369 | 0.2221 |   0.2399 |
| scgpt_lora_head    | True   |        0.1183 |        0.3187 |           0.3222 |           0.5089 | 0.2042 |   0.2207 |
| scgpt_lora_head    | False  |        0.1164 |        0.3251 |           0.3247 |           0.5153 | 0.2071 |   0.2245 |

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K

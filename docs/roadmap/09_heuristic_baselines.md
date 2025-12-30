# Heuristic Baselines for Reverse Perturbation Prediction

One training-free heuristic baseline for the reverse perturbation prediction task: **TGA (Target-Gene Activation)**.

## Overview

Both baselines exploit a key biological principle: in CRISPRa experiments, target genes are directly activated and typically show **increased expression**. By identifying upregulated genes in query cells (vs. control), we can predict which perturbation condition was applied.

## Algorithm Summary

| Baseline | Core Idea |
|----------|-----------|
| **TGA** | Compute z-score `(query - ctrl) / std` for each gene; rank conditions by sum of target gene scores |

---

## Results on Norman Dataset

### Metrics

- **Hit@K**: Fraction of queries where the true condition appears in top-K predictions
- **MRR** (Mean Reciprocal Rank): Average of `1/rank` where rank is the position of the true condition; higher = better (max 1.0)

### Overall Performance

| Baseline | Hit@1 | Hit@5 | Hit@10 | MRR |
|----------|-------|-------|--------|-----|
| **TGA** | **60.6%** | 80.3% | 86.4% | 0.698 |

### Stratified by Test Strata

| Stratum | n | TGA Hit@1 | TGA Hit@5 |
|---------|---|-----------|-----------|
| **seen2** (both genes seen) | 15 | 80.0% | 86.7% |
| **seen1** (one gene seen) | 31 | 74.2% | 83.9% |
| **seen0** (neither gene seen) | 5 | 40.0% | 60.0% |
| **single_unseen** | 15 | 20.0% | 73.3% |

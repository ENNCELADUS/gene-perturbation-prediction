## Route B1 Task and Loss

- Task:
  $$
  \text{given } (x, y) \;\Rightarrow\; \hat p = g(x, y)
  $$
  then rank genes to recover perturbation targets.
- Inputs($x$, $y$): expression tokens per perturbed cell plus matched controls.
- Output($\hat p$): per-gene logits used for ranking against target genes.
- Targets(ground truth): multi-hot vector from condition string (all perturbed genes marked 1).
- Loss: weighted sum with config weights. BCE is computed only on the positive set plus sampled negatives (not all genes) to mitigate long-tail imbalance.
  $$
  \mathcal{L} = 0.7\,\mathcal{L}_{\text{rank}} + 0.1\,\mathcal{L}_{\text{bce}}
  $$
  - $\mathcal{L}_{\text{rank}}$: Ranking loss that constrains target logits to be higher than sampled negatives to push correct targets into the top-K.
  - $\mathcal{L}_{\text{bce}}$: Auxiliary multi-label calibration using sigmoid-BCE on a subset of genes to match multi-hot labels while avoiding gradient dominance from the full gene space.

- Metrics: relevant_hit@k (any target in top-k), exact_hit@k (all targets in top-k), and MRR from best-ranked target.

---

## scGPT Pipeline

![scGPT pipeline](./scGPT_pipeline_standalone.png)

---

## Results

| Metric | scGPT | pca_knn | random_forest | xgboost |
| --- | --- | --- | --- | --- |
| mrr | 0.1975 | **0.3602** | 0.3232 | 0.3323 |
| exact_hit@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| relevant_hit@1 | 0.1039 | **0.3158** | 0.2105 | 0.2368 |
| recall@1 | 0.0520 | **0.1579** | 0.1053 | 0.1184 |
| exact_hit@5 | 0.0526 | **0.1053** | 0.0789 | 0.0789 |
| relevant_hit@5 | 0.2922 | 0.4211 | **0.4737** | 0.4211 |
| recall@5 | 0.1631 | 0.2632 | **0.2763** | 0.2500 |
| exact_hit@10 | 0.0963 | 0.1053 | **0.1316** | 0.1053 |
| relevant_hit@10 | 0.4155 | 0.4211 | **0.5000** | **0.5000** |
| recall@10 | 0.2492 | 0.2632 | **0.3158** | 0.3026 |
| exact_hit@20 | **0.2123** | 0.1053 | 0.1842 | 0.1842 |
| relevant_hit@20 | **0.5731** | 0.4211 | 0.5000 | 0.5526 |
| recall@20 | **0.3927** | 0.2632 | 0.3421 | 0.3684 |
| exact_hit@40 | **0.3476** | 0.1053 | 0.1842 | 0.1842 |
| relevant_hit@40 | **0.6828** | 0.4211 | 0.5263 | 0.5526 |
| recall@40 | **0.5152** | 0.2632 | 0.3553 | 0.3684 |

---

## Summary

- Validated the feasibility of scGPT finetuning (Route B1 direct gene scoring) on the Norman dataset and completed a comparative evaluation against baselines.

## Future Work

- **Improve drug target benchmarks**: Tahoe drug target annotations are currently insufficient (only 52 labeled drugs), requiring more comprehensive labeling for robust evaluation.
- **Incorporate prior knowledge**: Integrate domain knowledge to narrow down the candidate target pool in practical applications, rather than ranking across the entire genome.
- **Enhance ranking sensitivity**: Address the lack of sensitivity in the current scGPT head by introducing stronger relative ranking constraints and improved score calibration methods.

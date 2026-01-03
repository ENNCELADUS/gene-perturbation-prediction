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
  - $\mathcal{L}_{\text{rank}}$: Ranking objective that constrains target logits to be higher than sampled negatives to push correct targets into the top-K.
  - $\mathcal{L}_{\text{bce}}$: Auxiliary multi-label calibration using sigmoid-BCE on a subset of genes to match multi-hot labels while avoiding gradient dominance from the full gene space.

- Metrics: relevant_hit@k (any target in top-k), exact_hit@k (all targets in top-k), and MRR from best-ranked target.

---

## scGPT Pipeline

![scGPT pipeline](./scGPT_pipeline_standalone.png)

---

## Results

| Metric | scGPT | pca_knn | tga |
| --- | --- | --- | --- |
| mrr | 0.1975 | **0.3602** | 0.0002 |
| exact_hit@1 | 0.0000 | 0.0000 | 0.0000 |
| relevant_hit@1 | 0.1039 | **0.3158** | 0.0000 |
| exact_hit@5 | 0.0526 | **0.1053** | 0.0000 |
| relevant_hit@5 | 0.2922 | **0.4211** | 0.0000 |
| exact_hit@10 | 0.0963 | **0.1053** | 0.0000 |
| relevant_hit@10 | 0.4155 | **0.4211** | 0.0000 |
| exact_hit@20 | **0.2123** | 0.1053 | 0.0000 |
| relevant_hit@20 | **0.5731** | 0.4211 | 0.0000 |
| exact_hit@40 | **0.3476** | 0.1053 | 0.0000 |
| relevant_hit@40 | **0.6828** | 0.4211 | 0.0000 |

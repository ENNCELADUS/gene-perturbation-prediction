# Metrics

Evaluation reports ranked retrieval metrics over candidate perturbation genes or
candidate perturbation combinations.

## Gene-Level Metrics

Gene-level evaluation starts from a ranked list of candidate genes for each
query cell or condition.

Common metrics:

- `hit@K`: at least one true target gene appears in the top K genes.
- `exact_hit@K`: all true target genes appear in the top K genes.
- `mrr`: reciprocal rank of the first relevant target gene, averaged over
  queries.
- `ndcg@K`: ranking quality with higher credit for relevant genes near the top.

For a two-gene perturbation such as `CNN1+MAPK1`, gene-level `exact_hit@K`
means both `CNN1` and `MAPK1` are present in the top K ranked genes. This is not
the same as exact condition retrieval unless the ranked genes are explicitly
composed into candidate perturbation combinations.

## Combination-Level Metrics

Combination-level evaluation starts from ranked candidate perturbation
conditions, for example:

```text
CNN1+MAPK1
FOSB+UBASH3B
```

Common metrics:

- `correct_hit@K`: the exact true perturbation condition appears in the top K
  candidate conditions.
- `relevant_hit@K`: at least one top K candidate condition shares one or more
  genes with the true perturbation condition.

These metrics match the semantics used by retrieval-style inverse perturbation
prediction papers.

## Masked Evaluation

For CRISPRa datasets, the perturbed target gene can be visibly upregulated in
the observed expression profile. Masked evaluation reduces this shortcut by
zeroing or hiding target-gene expression before ranking.

Configs expose evaluation options under:

```yaml
evaluation_config:
  mask: 10
```

Use unmasked scores for debugging and masked scores for the main leakage-aware
claim.

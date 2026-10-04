# Linear context prior, seed 0

Run 2026-10-04 on the H20 container on port 30838 (worktree `/2023533015/VCC_Project_context_prior`, branch
`feat/context-prior` at `4334eae`), CPU only. Design: [spec](../../docs/specs/2026-10-04-context-generalization-design.md);
protocol §10 of the [GeneEffect protocol](../../docs/03-geneeffect-protocol.md). The DepMap inputs on the host match
the SHA256s pinned in `configs/benchmarks/extra_bulk_lines_26Q1.json`. One config is one run at seed 0; validation
chooses; each run scored its chosen prior once on test. Nothing here is synthetic-lethality evidence.

| Run | Config | Summary |
| --- | --- | --- |
| `prior_20261004` | `configs/context_prior/prior.yaml` (all 919 extra labelled lines) | [summary](summary_prior_20261004.md) |
| `prior_no_haematopoietic_20261004` | `configs/context_prior/prior_no_haematopoietic.yaml` (the 133 Lymphoid and Myeloid extras dropped) | [summary](summary_prior_no_haematopoietic_20261004.md) |
| `oracle_20261004` | `prior.yaml --oracle-only`, on the Mac (bulk input only) | [summary](summary_oracle_20261004.md) |

## Result

**Validation chooses the run without the haematopoietic extras.** Its prior is a ridge on 128 expression components
(penalty 10 × n), trained on 953 labelled lines (167 single-cell training lines with bulk RNA plus 786 solid-tumour
extras) and read from bridged pseudo-bulk at query time.

| Selective Spearman | Linear context prior, no haematopoietic extras | Linear context prior, all extras (fell back to single-cell lines) | Tx1 context-PCA ridge |
| --- | ---: | ---: | ---: |
| Validation | **0.2232** | 0.1658 | 0.1296 |
| Test | **0.2227** | 0.1721 | 0.1206 |

| Prior minus Tx1 context-PCA ridge (27-line paired bootstrap, 95%) | No haematopoietic extras | All extras |
| --- | --- | --- |
| Validation | +0.094 [0.058, 0.124] | +0.036 [0.012, 0.062] |
| Test | +0.102 [0.066, 0.131] | +0.052 [0.013, 0.087] |

Residual Pearson over variable genes, chosen prior: 0.214 validation, 0.218 test (Tx1 ridge 0.133, 0.122). Selective
AUPR lift: 0.196 and 0.186 (Tx1 ridge 0.134, 0.126). The same test pairs (478,501) are scored for every method.

## Learning curve and the extra-lines decision

Validation selective Spearman, expression components alone; mean over three patient-grouped subsets below the full
set. "Bulk" scores validation lines' own bulk RNA (off-contract upper bound); "bridged" scores their bridged
pseudo-bulk.

| Labelled lines | Bulk | Bridged |
| ---: | ---: | ---: |
| 100 | 0.111 | 0.112 |
| 167 (single-cell training lines) | 0.182 | 0.166 |
| 400 | 0.187 | 0.168 |
| 700 | 0.218 | 0.197 |
| 1,086 (all) | 0.235 | 0.212 |

The decision compares the all-lines prior with the single-cell-line prior, both expression components plus 50
data-selected genes, on bridged input:

| Run | Gain | Interval | Decision |
| --- | --- | --- | --- |
| All extras (1,086 lines) | +0.017 | [−0.006, 0.039] | fails; bulk input +0.048 [0.021, 0.071], so the run flagged the bridge as failing |
| No haematopoietic extras (953 lines) | +0.051 | [0.027, 0.069] | passes; bulk input +0.055 [0.031, 0.075] |

The haematopoietic extras are what failed the decision: validation and test hold no haematopoietic line, and with
them the data-selected genes lower the bridged score (0.212 with components alone to 0.191). Without them the
bridged and bulk scores of the chosen prior differ by 0.013 (0.223 vs 0.236).

## Block selection

In both runs only the expression components were kept. On bridged input the gene-level blocks hurt: in the chosen
run own expression −0.012 to −0.004 and reduced rank 64 −0.029 to −0.007; pathway scores, predicted genotype,
partners and data-selected genes had intervals spanning zero. In the single-cell-only run own expression and
partners cost 0.03 to 0.07. The bridge's median per-gene correlation across out-of-fold training lines is 0.43,
which is the likely reason gene-level features carry over badly while components survive.

## Caveats

- **Scale.** The SD ratio of the chosen prior is 0.016: the ridge is heavily shrunk, so its Huber equals the gene
  mean's (0.0162 vs 0.0163). The selector is scale-free; the prior ranks lines, it does not size effects.
- **Grid edges.** The single-cell-only run chose the largest components penalty (100); the oracle curve at 1,086 lines
  chose the smallest (0.01). A wider grid could move both runs.
- **Shared space.** 9,711 genes, those every validation and test line's source measures; 60 training lines lacked
  some (up to 2,843 values) and took the training mean for them (`space.json`).
- **Test views.** Each run scored its own chosen prior on test once; the choice between the two runs was made on
  validation. These are seed-0 numbers without cross-seed variance.
- **Claim boundary.** Single-gene dependency evidence only, with a training-data change (extra DepMap bulk-RNA lines)
  scored on the unchanged validation and test lines; validation lines' bulk RNA entered only the oracle rows.

## Next steps the results point to

1. A better bridge (the design's contrastive-PCA alignment), since bulk input still beats bridged input and
   gene-level blocks fail through the bridge.
2. A wider penalty grid for the components stage.
3. The single-cell correction on top of the chosen prior (the next plan): what Tx1, `q_sc` and STATE add.

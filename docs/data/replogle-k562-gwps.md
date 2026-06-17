# Replogle K562 GWPS CRISPRi Perturb-seq

## Role

Genome-scale K562 CRISPRi Perturb-seq source for expanding observed
post-perturbation response coverage beyond the smaller K562 essential subset.
Use this as the primary source for K562 loss-of-function response bags when
joining perturbation-gene evidence to the K562-filtered SL benchmark candidate
universe.

## Downloaded File

Local path:
`data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad`

- Source: Replogle/Nadig processed K562 GWPS single-cell h5ad.
- Size: 65830941948 bytes.
- Downloaded on 2026-06-17.

## AnnData Shape and Fields

- Shape: 1989578 cells x 8248 genes.
- Perturbation label: `obs["gene"]`.
- Unique perturbation labels: 9867.
- Control label: `non-targeting`.
- Control cells: 75328.
- Other observed metadata fields include `gene_id`, `transcript`,
  `gene_transcript`, `sgID_AB`, `UMI_count`, `core_scale_factor`, and
  `core_adjusted_UMI_count`.

## SL Benchmark Coverage

Coverage was computed on 2026-06-17 against the local K562-filtered
SL_benchmark candidate universe from the balanced CV1/CV2/CV3 CSV files:
9471 unique genes.

| Statistic | Value |
| --- | ---: |
| Raw unique conditions | 9867 |
| Parsed target conditions | 9866 |
| Single-gene conditions | 9866 |
| Conditions with a gene in SL universe | 6070 |
| Unique target genes | 9866 |
| Unique target genes in SL universe | 6070 |
| SL gene coverage | 64.09% |
| Cells in parsed target conditions | 1914250 |
| Cells in SL-covered conditions | 1166742 |

Generated tables:

- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_summary.csv`
- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_conditions.csv`

This dataset accounts for almost all observed K562 Perturb-seq coverage of the
SL candidate universe in the 2026-06-17 local audit. It should be treated
separately from Replogle K562 essential in source-specific ablations because
the target library and coverage profile are different.

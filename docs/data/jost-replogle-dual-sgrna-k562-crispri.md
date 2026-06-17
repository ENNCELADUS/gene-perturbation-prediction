# Jost/Replogle Dual-sgRNA K562 CRISPRi Perturb-seq

## Role

K562 CRISPRi Perturb-seq supplement for target-specific response coverage and
guide-efficacy checks. Use it as a small K562 loss-of-function supplement, not
as a genome-wide replacement for Replogle K562 GWPS.

## Downloaded File

Local path:
`data/sl_dependency_v0/raw/jost_replogle_dual_sgrna/GSE205310_RAW.tar`

- Source: GEO `GSE205310`.
- Size: 1688791040 bytes.
- Downloaded on 2026-06-17.

Archive contents:

```text
GSM6210116_dual.barcodes.tsv.gz
GSM6210116_dual.features.tsv.gz
GSM6210116_dual.matrix.mtx.gz
GSM6210116_dual_cell_identities.csv.gz
GSM6210117_dolcetto.barcodes.tsv.gz
GSM6210117_dolcetto.features.tsv.gz
GSM6210117_dolcetto.matrix.mtx.gz
GSM6210117_dolcetto_cell_identities.csv.gz
```

## Fields Used for Coverage

The coverage statistics below use `guide_identity` from the two
`*_cell_identities.csv.gz` files. Gene symbols are parsed from guide identities
such as `PDLIM2_-_22436722.23-P2_posB` and
`CD2BP2_GGGGACCGCCCGAATCCCCG`.

## SL Benchmark Coverage

Coverage was computed on 2026-06-17 against the local K562-filtered
SL_benchmark candidate universe from the balanced CV1/CV2/CV3 CSV files:
9471 unique genes.

| Source partition | Raw unique conditions | Parsed target conditions | Unique target genes | Unique target genes in SL universe | SL gene coverage | Cells in parsed conditions | Cells in SL-covered conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GSM6210116_dual` | 510 | 510 | 403 | 82 | 0.87% | 60049 | 16721 |
| `GSM6210117_dolcetto` | 462 | 442 | 150 | 100 | 1.06% | 61655 | 39567 |

Generated tables:

- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_summary.csv`
- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_conditions.csv`

This source is K562 and CRISPRi-compatible, but the incremental SL-universe
coverage is small compared with Replogle K562 GWPS. Use it for supplement,
guide-design checks, and source-robustness ablations.

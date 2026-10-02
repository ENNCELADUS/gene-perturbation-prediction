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

## Perturbation Fields

Each cell's perturbation is `guide_identity` in the two `*_cell_identities.csv.gz`
files. Gene symbols are parsed from guide identities
such as `PDLIM2_-_22436722.23-P2_posB` and
`CD2BP2_GGGGACCGCCCGAATCCCCG`.

# Dixit K562 Perturb-seq

## Role

Early K562 Perturb-seq reference for Stage 1 response-to-dependency alignment.
Use it as a small transcription-factor CRISPR-KO response set, not as a
genome-wide dependency screen.

## Downloaded File

Remote path:
`/home/richard/projects/VCC/data/sl_dependency_v0/raw/dixit/dixit_2016.h5ad`

- Source: `https://exampledata.scverse.org/pertpy/dixit_2016.h5ad`
- Size: 1515276836 bytes
- MD5: `6aba165990d58c7d7bf5299726af80c4`
- SHA256:
  `dfe239d29abd55c2058444a46c5d8e18d8108c0428c1e612c581236405aa5cc9`
- Downloaded on 2026-05-12 Asia/Shanghai

## Fields and Alignment

- Shape: 99722 cells x 18531 genes
- Perturbation label: `obs["perturbation_name"]`
- K562 screen clusters: `tfs_7`, `tfs_13`, `tfs_highmoi`
- Cell-line column: absent
- DepMap model used: `ACH-000551`
- Matched DepMap component rows: 1334 / 1416

Remote overlap table:
`/home/richard/projects/VCC/data/sl_dependency_v0/interim/k562_dixit_depmap_overlap.csv`

The pertpy processed file has 99722 cells, not the Single Cell Portal display
count of 57120 cells.

## SL Benchmark Coverage

Coverage was recomputed on 2026-06-17 against the local K562-filtered
SL_benchmark candidate universe from the balanced CV1/CV2/CV3 CSV files:
9471 unique genes.

| Statistic | Value |
| --- | ---: |
| Raw unique conditions | 641 |
| Parsed target conditions | 305 |
| Single-gene conditions | 23 |
| Multi-gene conditions | 282 |
| Conditions with a gene in SL universe | 295 |
| Conditions with all parsed genes in SL universe | 139 |
| Unique target genes | 11 |
| Unique target genes in SL universe | 8 |
| SL gene coverage | 0.08% |
| Cells in parsed target conditions | 68923 |
| Cells in SL-covered conditions | 54628 |

Generated coverage tables:

- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_summary.csv`
- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_conditions.csv`

Dixit is useful as a K562 CRISPR-KO transcription-factor sanity check, but it
does not materially expand the broad SL candidate-gene coverage.

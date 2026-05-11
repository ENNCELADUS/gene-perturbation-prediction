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

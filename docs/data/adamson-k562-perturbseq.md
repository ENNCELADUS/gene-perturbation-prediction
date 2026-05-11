# Adamson K562 Perturb-seq

## Role

K562 Perturb-seq / CRISPRi response benchmark for Stage 1. This is more
modality-compatible with DepMap CRISPR gene-effect labels than Norman CRISPRa,
but it is much smaller than Replogle genome-scale K562.

## Downloaded Files

Remote root:
`/home/richard/projects/VCC/data/sl_dependency_v0/raw/adamson/`

| File | Cells x genes | Size | SHA256 |
| --- | ---: | ---: | --- |
| `adamson_2016_pilot.h5ad` | 5768 x 35635 | 34557246 | `119e3c1cf7dede4e13f887b86f9bcd797a9dc29213ee57d36aa80012d93f1c1c` |
| `adamson_2016_upr_epistasis.h5ad` | 15006 x 32738 | 139059637 | `6c6eca0f53f8887b86597e2a4ff512ff2b2d3d9c78ee7deec9a6e7d6ae859d01` |
| `adamson_2016_upr_perturb_seq.h5ad` | 65337 x 32738 | 471286951 | `e70fcd49808cab8d724de8d5a332940911206e1c8ef44cc7b568d048ed795c85` |

Source URLs use the pertpy mirror:
`https://exampledata.scverse.org/pertpy/`.

## Fields and Alignment

- Cell-line label: `obs["cell_line"]`, value `K562`
- Perturbation label: `obs["perturbation"]`
- DepMap model used: `ACH-000551`

Matched DepMap component rows:

- Pilot: 7 / 7
- UPR epistasis: 20 / 21
- UPR Perturb-seq: 98 / 110

Remote overlap tables:

- `interim/k562_adamson_pilot_depmap_overlap.csv`
- `interim/k562_adamson_upr_epistasis_depmap_overlap.csv`
- `interim/k562_adamson_upr_perturb_seq_depmap_overlap.csv`
- `interim/k562_adamson_depmap_overlap.csv`

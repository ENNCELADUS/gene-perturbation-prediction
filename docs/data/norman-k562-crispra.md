# Norman K562 CRISPRa Perturb-seq

## Role

K562 CRISPRa combinatorial Perturb-seq benchmark. Use it as an auxiliary
response-model benchmark. Do not treat it as direct knockout / CRISPRi
supervision for DepMap CRISPR gene-effect labels.

## Existing Remote Files

The dataset was already present on the remote PC.

| File | Size | SHA256 |
| --- | ---: | --- |
| `/home/richard/projects/VCC/data/norman/perturb_processed.h5ad` | 2228610012 | `23ffb0fac6a847ff927cf7509d80d85052bfefbfb97610786a2dafaaefa0b6a0` |
| `/home/richard/projects/VCC/data/raw/norman.zip` | 161M | `c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877` |

The zip contains `norman/go.csv` and `norman/perturb_processed.h5ad`.

## Fields and Alignment

- Shape: 91205 cells x 5045 genes
- Perturbation label: `obs["condition"]`
- Control column: `obs["control"]`
- Gene symbol field: `var["gene_name"]`
- DepMap model used: `ACH-000551`
- Matched DepMap component rows: 387 / 414

Remote overlap table:
`/home/richard/projects/VCC/data/sl_dependency_v0/interim/k562_norman_depmap_overlap.csv`

Metadata caveat: this existing GEARS-style h5ad reports `cell_type=A549`, even
though Norman 2019 is the K562 CRISPRa dataset. Treat current overlap as a
benchmark-file overlap with that caveat.

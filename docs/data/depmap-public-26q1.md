# DepMap Public 26Q1

## Role

Primary dependency-label source for Stage 1 K562 alignment. Use continuous
CRISPR gene-effect scores as population-level fitness labels, not single-cell
death labels.

## Downloaded Files

Remote root:
`/home/richard/projects/VCC/data/sl_dependency_v0/raw/depmap/`

| File | Size | SHA256 |
| --- | ---: | --- |
| `CRISPRGeneEffect.csv` | 440646050 | `e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e` |
| `Model.csv` | 697455 | `ea4e0b2a3bc806f81df62689a5ae75f1a100135727a3d7b8a4c7ccc8815183f8` |

Downloaded on 2026-05-11 from the DepMap downloads API:
`https://depmap.org/portal/api/download/files`.
Signed file URLs were resolved at download time and intentionally not persisted.

## K562 Mapping

Selected model: `ACH-000551`

- Cell line name: `K-562`
- CCLE name: `K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE`
- Other K562-like candidates are recorded remotely in
  `interim/depmap_k562_model_candidates.csv`.

## Notes

- Gene-effect columns are parsed as gene symbol plus Entrez ID.
- This dataset supplies labels for `(cell line, perturbation gene)`.
- Do not describe these labels as synthetic lethality without additional
  context-specific evidence.

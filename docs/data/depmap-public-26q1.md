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
| `PortalOmicsCNGeneLog2.csv` | 402982032 | `2687bd530681d84209192783d2b848c8dc02a0036924ad0b32c840087ab71c26` |
| `OmicsSomaticMutationsMatrixHotspot.csv` | 6902131 | `a86cd8c92b86d5507e63103f06a507913897c85262e0a0d7e44fa22768dd4dc9` |
| `OmicsSomaticMutationsMatrixDamaging.csv` | 238858587 | `45709b1b842dd6dcd05e504cd42ca977fff760c480c60ff1184c1a5935b595b0` |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 305039593 | `0377be80c525fde98cbd2c6e8b06bdf2a4014a9683eb70182c1f8649d711021a` |

Downloaded on 2026-05-11 from the DepMap downloads API:
`https://depmap.org/portal/api/download/files`.
Signed file URLs were resolved at download time and intentionally not persisted.

Context files for SL-like selectivity were added on 2026-06-18 from the same
DepMap downloads API. They support a minimal context definition:

```text
gene_a defective(c) =
    damaging mutation in gene_a
    OR hotspot mutation in gene_a
    OR deep copy-number loss of gene_a
    OR very low expression of gene_a
```

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

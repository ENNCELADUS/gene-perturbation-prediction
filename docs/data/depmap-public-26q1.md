# DepMap Public 26Q1

## Role

The pinned GeneEffect label source (`CRISPRGeneEffect.csv`) of the 226-line GeneEffect
benchmark and the SL protocol, joined by ModelID. Use continuous CRISPR gene-effect
scores as population-level fitness labels, not single-cell death labels.

## Downloaded Files

Remote root:
`/2023533015/VCC_Project/data/sl_dependency_v0/raw/depmap/`

| File | Size | SHA256 |
| --- | ---: | --- |
| `CRISPRGeneEffect.csv` | 440646050 | `e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e` |
| `Model.csv` | 697455 | `ea4e0b2a3bc806f81df62689a5ae75f1a100135727a3d7b8a4c7ccc8815183f8` |
| `PortalOmicsCNGeneLog2.csv` | 402982032 | `2687bd530681d84209192783d2b848c8dc02a0036924ad0b32c840087ab71c26` |
| `OmicsSomaticMutationsMatrixHotspot.csv` | 6902131 | `a86cd8c92b86d5507e63103f06a507913897c85262e0a0d7e44fa22768dd4dc9` |
| `OmicsSomaticMutationsMatrixDamaging.csv` | 238858587 | `45709b1b842dd6dcd05e504cd42ca977fff760c480c60ff1184c1a5935b595b0` |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 305039593 | `0377be80c525fde98cbd2c6e8b06bdf2a4014a9683eb70182c1f8649d711021a` |
| `ScreenGeneEffect.csv` | 528962158 | `f77ca708f884b89586f4d94450b6ca8d921a115548900022e17a8facd55a5099` |
| `CRISPRScreenMap.csv` | 37743 | `9bec9f1fcc7ce5219c41c5f4d91be054f0828e8c289fca9f7b7d79337bc21a97` |

The two per-screen files for noise diagnosis ([protocol §14](../03-geneeffect-protocol.md#14-noise-diagnosis)) were
downloaded on 2026-10-12. The browser selected **DepMap Public 26Q1** and resolved each file's signed download URL;
Chrome blocked attachment navigation (`ERR_BLOCKED_BY_CLIENT`), so `wget` retrieved those browser-issued URLs directly
on H20. Both downloads exited successfully, and the table records their on-host SHA-256 hashes.
`ScreenGeneEffect.csv` has an unnamed first column holding ScreenIDs and gene columns in `SYMBOL (Entrez ID)` format;
`CRISPRScreenMap.csv` has columns `ScreenID,ModelID`. These are screen-level dependency labels, not SL labels.
The full CSV read on H20 verified 1,458 unique screen rows with 18,531 numeric gene columns and 1,451 map rows;
every mapped ScreenID exists in the effect matrix. The noise diagnosis read them on 2026-10-12 (run
`noise_20261012`, [record](../../results/noise_diagnosis_seed0/README.md)): 202 lines outside validation and test have
two screens.

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
- Do not describe these labels, or dependency patterns derived from them, as
  synthetic lethality or as SL labels: SL labels are experimental SL-database
  entries only ([blueprint §3.1](../01-blueprint.md#31-synthetic-lethality-two-operational-definitions)).

# DepMap Public 26Q1

## Role

The pinned GeneEffect label source (`CRISPRGeneEffect.csv`) of the 226-line GeneEffect
benchmark and the SL protocol, joined by ModelID. Use continuous CRISPR gene-effect
scores as population-level fitness labels, not single-cell death labels.

## Downloaded Files

Remote root:
`/2023533015/VCC_Project/data/sl_dependency_v0/raw/depmap/`: `CRISPRGeneEffect.csv`, `Model.csv`,
`PortalOmicsCNGeneLog2.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, `OmicsSomaticMutationsMatrixDamaging.csv`,
`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`.

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

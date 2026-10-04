# Extra Bulk Lines, DepMap 26Q1

**Status:** Membership of the DepMap lines outside the 226-member
[GeneEffect benchmark](cell-line-geneeffect-226.md) that may join the training side of the
linear context prior ([protocol §10](../03-geneeffect-protocol.md#10-linear-context-prior),
[design](../specs/2026-10-04-context-generalization-design.md)). Nothing here is SL evidence.

## Purpose

The 226 benchmark gives 170 labelled single-cell training lines. Bulk RNA exists for many
more DepMap lines, and 26Q1 CRISPRGeneEffect labels for most of them. These lines add
contexts that the single-cell atlas does not cover. Their bulk RNA fits context encoders
(expression components, predicted genotype); their GeneEffect labels fit the prior only if
the learning-curve rule of protocol §10 passes. They carry no single cells, so a query line
never arrives as one of them.

## Membership

The file is the sole membership authority for the extra lines, as the split JSON is for the 226.

| Group | Lines | Definition |
| --- | ---: | --- |
| Labelled | 919 | Outside the 226, with 26Q1 bulk RNA and 26Q1 GeneEffect |
| Unlabelled | 576 | Outside the 226, with 26Q1 bulk RNA and no GeneEffect |
| Excluded | 10 | Outside the 226, sharing a `PatientID` with a validation or test line |

The builder prints these counts when it writes the file; its printed counts are the
authority, and this card follows them. The unlabelled count is the 1,664 training-side
lines with bulk RNA, minus the 169 training members of the 226 that have it, minus the 919
labelled lines.

**Exclusion rule.** A line outside the 226 whose `PatientID` (Model.csv) equals that of a
validation or test line is excluded from every fit, labelled or not, and each exclusion is
recorded with its reason and the held-out line it shares a patient with. Extras that share a
patient with a training line are kept; they stay on the training side, in that line's
cross-fitting fold. The file is a pinned list, not a rule recomputed from labels at run time.

**Haematopoietic lineages.** 133 of the labelled lines are haematopoietic. They are kept in
the first run. The `training_side.exclude_lineages` key of a prior config drops lineages from the
training side (`Lymphoid` and `Myeloid` in `configs/context_prior/bridge_remedies.yaml`; Model.csv
`OncotreeLineage`). Validation and test hold no haematopoietic line, and the first runs showed that
keeping them lowered the bridged score ([record](../../results/context_prior_seed0/README.md)).

## Rebuild

```bash
uv run python -m src.data.prepare.build_extra_bulk_lines \
    --model data/sl_dependency_v0/raw/depmap/Model.csv \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --bulk data/sl_dependency_v0/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --out configs/benchmarks/extra_bulk_lines_26Q1.json
```

`src/data/extra_lines.py:load_extra_lines` reads the file against the split and raises on
duplicate ModelIDs or any overlap between labelled, unlabelled, excluded and the split.

## Artifacts

- `configs/benchmarks/extra_bulk_lines_26Q1.json`: membership, the exclusion reasons, the
  policy text and the path and SHA-256 of the three source files (Model.csv,
  CRISPRGeneEffect.csv, the protein-coding bulk TPM matrix).
- `src/data/prepare/build_extra_bulk_lines.py`: the builder.
- `src/data/extra_lines.py`: the loader.

## Use

Validation lines' bulk RNA is read only by the oracle diagnostic; test lines' bulk RNA is
never read ([GeneEffect protocol rules](../03-geneeffect-protocol.md#11-rules)). Results that use the
extra lines are a training-data change, scored on the unchanged validation and test lines.

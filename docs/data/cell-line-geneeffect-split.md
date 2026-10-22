# Final Cell-Line Split

**Status:** Since 2026-10-12 the one split every stage-one fit and score reads:
`configs/benchmarks/cell_line_geneeffect_split.json` and its table (`.csv`). It is built from three files that stay
as its provenance: the [226 split](cell-line-geneeffect-226.md) (the held-out partition and the first training
lines), the [single-cell extras](single-cell-atlas-search.md) and the [bulk extras](extra-bulk-lines-26q1.md).
Joint configs name it as `paths.split`; the linear prior reads it through its joint config. Nothing here is SL
evidence.

## Roles

Each line has one role.

| Role | Lines | Definition |
| --- | ---: | --- |
| `train` | 379 | Single-cell training lines: the 226's 171 training lines, then the 208 built single-cell extras. PC9 and HeLa (`unlabeled_train`) have no 26Q1 GeneEffect, so 377 are labelled |
| `val` | 27 | The 226's validation lines, unchanged |
| `test` | 28 | The 226's test lines (A549 since 2026-10-11) |
| `bulk_labelled` | 713 | Training-side lines read through bulk RNA only: the bulk file's labelled lines without built single cells |
| `bulk_unlabelled` | 576 | Bulk RNA without 26Q1 GeneEffect, encoder rows only (52 have single cells but no label) |
| `excluded` | 12 | Lines sharing a `PatientID` with a validation or test line, with the reason; never read (the two such single-cell candidates are among them) |

The builder refuses an unbuilt single-cell extra, an extra the bulk file excludes or lists as unlabelled, and a
single-cell training line sharing a patient with a held-out line (patients from the 226 table and the extras'
records); the bulk-only exclusions come from the bulk file, built against the same held-out lines.

## What fits on what

- Gene means, variable and selective genes, the residual SD, the context PCA, every trained model and every control
  fit on the 377 labelled single-cell training lines ([GeneEffect protocol §3.3, §11](../03-geneeffect-protocol.md#11-rules)).
- The linear prior also fits on the `bulk_labelled` lines outside its config's dropped lineages (lymphoid and myeloid)
  and encodes the `bulk_unlabelled` ones; its bridge pairs the single-cell training lines that have bulk RNA
  ([protocol §10](../03-geneeffect-protocol.md#10-linear-context-prior)).
- Validation selects; test is scored once per config.

## Table

`cell_line_geneeffect_split.csv`: `split`, `model_id`, `patient_id`, `lineage`, `expression_source` (the 226 table's
column; an extra's source key), one row per single-cell line (434), training lines first. The revision summary's
per-lineage table and the readout comparison read it.

## Rebuild

```bash
uv run python -m src.data.builders.build_cell_line_geneeffect_split \
    --split-226 configs/benchmarks/cell_line_geneeffect_226_split.json \
    --single-cell configs/benchmarks/extra_single_cell_lines_26Q1.json \
    --bulk configs/benchmarks/extra_bulk_lines_26Q1.json \
    --out configs/benchmarks/cell_line_geneeffect_split.json
```

A test checks that the tracked file is what the builder writes from the tracked inputs. Rebuild it whenever an input
changes.

## History

Before 2026-10-12 the 226 split was the membership authority and the extras had their own membership files; results
recorded before then fit their definitions on the 226's 169 labelled training lines (172/27/27 before 2026-10-11,
171/27/28 after). A checkpoint records the split membership it was fitted under, so every earlier checkpoint is refused
under this split.

# Final Cell-Line Split

**Status:** Since 2026-10-12 the one split every stage-one fit and score reads:
`configs/benchmarks/cell_line_geneeffect_split.json` and its table (`.csv`). It is built from three files that stay
as its provenance: the [226 split](cell-line-geneeffect-226.md) (the held-out partition and the first training
lines), the [single-cell extras](single-cell-atlas-search.md) and the [bulk extras](extra-bulk-lines-26q1.md).
Joint configs name it as `paths.split`; the linear model reads it through its joint config. Nothing here is SL
evidence.

## Roles

Each line has one role.

| Role | Lines | Definition |
| --- | ---: | --- |
| `train` | 400 | Single-cell training lines: the 226's 171 training lines, then the built single-cell extras. 16 (`unlabeled_train`: PC9, HeLa and the SL lines without GeneEffect) have no 26Q1 GeneEffect, so 384 are labelled |
| `val` | 27 | The 226's validation lines (PK-45H in place of PA-TU-8988S since 2026-11-04) |
| `test` | 28 | The 226's test lines (ABC-1 and HT-55 in place of A549 and SW 620 since 2026-11-04) |
| `bulk_labelled` | 708 | Training-side lines read through bulk RNA only: the bulk file's labelled lines without built single cells |
| `bulk_unlabelled` | 570 | Bulk RNA without 26Q1 GeneEffect, encoder rows only |
| `excluded` | 7 | Lines sharing a `PatientID` with a validation or test line, with the reason; never read. None has GeneEffect or bulk RNA |

The builder refuses an unbuilt single-cell extra, an extra the bulk file excludes, and a single-cell training line
sharing a patient with a held-out line (patients from the 226 table and the extras' records). A built extra leaves
`bulk_labelled` and `bulk_unlabelled`. The bulk-only exclusions come from the bulk file, built against the same
held-out lines.

## What fits on what

- Gene means, variable and selective genes, the residual SD, the context PCA, every trained model and every control
  fit on the 377 labelled single-cell training lines ([GeneEffect protocol §3.3, §11](../03-geneeffect-protocol.md#11-rules)).
- The linear model also fits on the `bulk_labelled` lines outside its config's dropped lineages (lymphoid and myeloid)
  and encodes the `bulk_unlabelled` ones; its bridge pairs the single-cell training lines that have bulk RNA
  ([protocol §10](../03-geneeffect-protocol.md#10-linear-model)).
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
171/27/28 after). Until 2026-11-04 a checkpoint recorded the split membership and was refused under another; that check
is removed, and the checkpoints written before the 2026-11-04 build do not belong to the split it writes.

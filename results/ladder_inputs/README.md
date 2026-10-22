# Stage-one inputs and the frozen-PIE pass

Task 2 of the [stage-one design](../../docs/specs/2026-10-16-pie-geneeffect-ladder-design.md): the fixed inputs every
later task reads, under `data/geneeffect_joint/v2/ladder/` and `data/tx1_basal_embeddings/v2_all/` on the H20
container on port 30838, from `configs/stage_one/ladder_inputs.yaml` at `c219707` (`hpc/run.sh ladder-inputs` and
`hpc/run.sh pie-precompute`, recipes in `hpc/README.md`). Contents and definitions are in the
[data card](../../docs/data/ladder-inputs.md) and protocol §15. Finished 2026-10-17 18:12 UTC.

## What was built

| Step | Finished (UTC) | Counts |
| --- | --- | --- |
| lines | 2026-10-16 18:46 | frame 981 (377 single-cell training, 28 of them lymphoid or myeloid; 604 bulk-only), 343 paired, 27 validation, 28 test; 17,787 genes, decode panel 17,607, selective 4,323 (4,269 in the panel) |
| cells | 2026-10-16 19:05 | 434 lines, 573,532 cells, 19,277 source genes |
| mean | 2026-10-16 19:06 | 1,036 lines, 8,810 space genes |
| summaries | 2026-10-16 19:06 | 434 lines × 70 features |
| gene sources, Cellosaurus text | 2026-10-17 05:20 | coverage in the data card |
| Tx1 cells | 2026-10-17 18:12 | 432 lines, real and shuffled, none missing; 7–8.5 cells per second per GPU |
| PIE rescore with the DepMap source masked | 2026-10-16 18:59 | below |
| PIE pilot and basis | 2026-10-17 06:34 | 200 random panel genes (44 selective) × 1,036 lines |
| PIE full pass | 2026-10-17 16:26 | 17,407 genes in four shards plus the pilot's 200; 7.5 s per gene per GPU over 1,036 lines (about 7.2 ms per pair) |
| PIE assemble | 2026-10-17 16:38 | 17,607 genes × 1,036 lines |

## What masking PIE's DepMap source costs PIE

Replogle, xdataset checkpoint, `all` row:

| Split | Metric | Unmasked (Task 1) | Masked |
| --- | --- | ---: | ---: |
| `test_seen` | AUPRC | 0.3802 | 0.3594 |
| `test_seen` | Spearman LFC | 0.5130 | 0.4504 |
| `test_unseen` | AUPRC | 0.2463 | 0.2313 |
| `test_unseen` | Spearman LFC | 0.2698 | 0.1145 |

The full table, all six metrics, is in `pie/pilot/pilot.md` on the host. The mask costs most on unseen perturbations
(Spearman LFC 0.27 → 0.11), where the DepMap token stood in for missing response evidence.

## The pilot

Variance across lines within a gene against variance between genes, over the 200 pilot genes:

| Delta | Within gene | Between genes | Ratio |
| --- | ---: | ---: | ---: |
| ΔH | 88.37 | 40.31 | 2.19 |
| ΔE | 7.611 | 3.992 | 1.91 |
| ΔL | 636.8 | 238.7 | 2.67 |
| own feature `h_g` | 794.1 | 1258 | 0.63 |
| own predicted change | 0.001707 | 0.009207 | 0.19 |

The 512-component profile basis keeps 0.886 of the centred variance of the pilot's 196,200 frame-line pairs. Agreement
with PIE's own data path, worst over the pilot and four full shards: our float32 decode differs from PIE's float32
delta-p logits by 0; our compiled bf16 decode by at most 0.394, against PIE's own bf16 forward at 0.410.

## Reading

Every input Tasks 3 to 9 need exists, and the frozen PIE's deltas vary across lines within a gene by more than they
vary between genes for every pooled delta. That ratio counts any line-to-line movement, informative or not; whether
it predicts GeneEffect is Task 3's question. ΔL, which reads no expression, varies most across lines: its line content
comes from the Cellosaurus text alone.

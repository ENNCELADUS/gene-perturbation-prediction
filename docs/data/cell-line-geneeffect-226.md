# Cell-Line GeneEffect Benchmark

**Status:** The held-out partition and the first training lines. Since 2026-10-12 it is an
input of the [final split](cell-line-geneeffect-split.md), which adds the 208 single-cell extras
to its training lines and is what every fit reads; validation and test are this split's. Before
then it was the only split, used by the final run of the historical staged GeneEffect protocol
(Exp13, [result](../../results/exp13_stage2_full/README.md)) and by the seed-0 run of the
[GeneEffect protocol](../03-geneeffect-protocol.md), whose test has been observed once
([result](../../results/joint_geneeffect_seed0/README.md)).

## Fixed split

| Cohort | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Original basal/SL-context union | 46 | 0 | 1 |
| New single-cell atlas | 125 | 27 | 27 |
| Total membership | 171 | 27 | 28 |

It fixes the validation and test lines of the final split. The 47 original contexts are fixed in
train except A549 (`ACH-000681`), fixed in test since 2026-10-11 because all three screens of
the SL benchmark test it ([SL protocol](../04-sl-ranking-protocol.md)); the split was rebuilt in
place, and the earlier 172/27/27 version, with A549 in train, is in git history. The 179 new
contexts retain the deterministic patient-grouped MILP partition; no PatientID crosses split
sides and GeneEffect values did not determine membership. Moving A549 leaves the
coverage gene universe unchanged (17,931 genes), so the prepared caches did not change.

## Labels and inputs

DepMap 26Q1 GeneEffect covers 224 of 226 members. PC9 (`ACH-000779`) and HeLa
(`ACH-001086`) remain train members as required but have no GeneEffect row, so they
cannot contribute supervised loss. Train therefore has 171 members and 169 labeled
contexts. Raw UMI, registered basal and processed CPM input semantics remain recorded
per row at the available source level; `registered_basal` is not a numeric matrix
semantic. **Resolved 2026-08-18:** the 152 `kinker_sccle` lines were published as
`processed_cpm`, and Tx1 does not read CPM like the counts underneath it (the shift survives
pooling; [result](../../results/exp13_stage0/README.md)). Raw UMI counts from
SCP542 now back all 152 lines — protocol §6, branch 1. Membership is unchanged.

Before 2026-10-12 this split JSON was the sole membership authority; the final split is now.
Fit residual targets, normalization,
feature transforms and all model parameters on train only; validation selects the
checkpoint by minimum GeneEffect Huber loss and test remains one-shot. GeneEffect
is single-gene dependency, not SL.
The two unlabeled train members are excluded from every supervised residual, context
model and nearest-label donor fit; they remain part of the requested train membership.
The split authority declares exactly these two IDs in `unlabeled_train`; any other
missing train label, or any missing validation/test label, is a hard runtime error.

## Artifacts

- `configs/benchmarks/cell_line_geneeffect_226_split.json`: the 226 split, an input of the final split.
- `configs/benchmarks/cell_line_geneeffect_226_split.csv`: row-level cohort, patient,
  source semantics and label availability.
- `configs/benchmarks/cell_line_geneeffect_226_split_audit.json`: counts and pinned hashes.
- `src/data/prepare/build_cell_line_geneeffect_226_split.py`: reproducible builder.
- HPC: `data/sl_dependency_v0/processed/cell_line_geneeffect_226/benchmark_split_v1/`.

The benchmark has 17,931 genes with at least five finite train labels and at least
three finite labels in validation and test. This is coverage, not the final
train-defined residual-variance gene universe. Coverage counts describe the split;
performance lives in the result notes.

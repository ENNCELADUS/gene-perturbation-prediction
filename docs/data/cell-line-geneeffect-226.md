# Cell-Line GeneEffect Benchmark

**Status:** One fixed pre-run split; no model result has been produced. Protocol:
[`../04-exp13-geneeffect-residual-protocol.md`](../04-exp13-geneeffect-residual-protocol.md).

## Fixed split

| Cohort | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Original basal/SL-context union | 47 | 0 | 0 |
| New single-cell atlas | 125 | 27 | 27 |
| Total membership | 172 | 27 | 27 |

This is the only train/validation/test split. All 47 original contexts are fixed in
train. The 179 new contexts retain the deterministic patient-grouped MILP partition;
no PatientID crosses split sides and GeneEffect values did not determine membership.

## Labels and inputs

DepMap 26Q1 GeneEffect covers 224 of 226 members. PC9 (`ACH-000779`) and HeLa
(`ACH-001086`) remain train members as required but have no GeneEffect row, so they
cannot contribute supervised loss. Train therefore has 172 members and 170 labeled
contexts. Raw UMI, registered basal and processed CPM input semantics remain recorded
per row at the available source level; `registered_basal` is not a numeric matrix
semantic. **Open question:** 152 of the 179 new-atlas lines are `kinker_sccle`
`processed_cpm` (19/27 test, 23/27 val); whether the Tx1-X1 collator produces a trustworthy
embedding from CPM rather than raw counts is unresolved — protocol §6 names the three
admissible branches and rules out "swap only `z_c`" as a fallback.

The split JSON is the sole membership authority. Fit residual targets, normalization,
feature transforms and all model parameters on train only; validation selects the
configuration and test remains one-shot. GeneEffect is single-gene dependency, not SL.
The two unlabeled train members are excluded from every supervised residual, context
model and nearest-label donor fit; they remain part of the requested train membership.
The split authority declares exactly these two IDs in `unlabeled_train`; any other
missing train label, or any missing validation/test label, is a hard runtime error.

## Artifacts

- `configs/benchmarks/cell_line_geneeffect_226_split.json`: the only split authority.
- `configs/benchmarks/cell_line_geneeffect_226_split.csv`: row-level cohort, patient,
  source semantics and label availability.
- `configs/benchmarks/cell_line_geneeffect_226_split_audit.json`: counts and pinned hashes.
- `scripts/build_cell_line_geneeffect_226_split.py`: reproducible builder.
- HPC: `data/sl_dependency_v0/processed/cell_line_geneeffect_226/benchmark_split_v1/`.

The benchmark has 17,931 genes with at least five finite train labels and at least
three finite labels in validation and test. This is coverage, not the final
train-defined residual-variance gene universe. GPU embeddings and scores remain unrun.

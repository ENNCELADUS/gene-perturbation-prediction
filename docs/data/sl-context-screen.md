# SL Context Screen Benchmark

**Status:** current build is `context_screen_v2`, with a row-level context split updated 2026-08-16.
No model has run. The raw-filter audit is incomplete (see Audits). PC9 and HeLa now have
Tx1-contract-verified basal artifacts; they are SL-label-only test contexts because the pinned
26Q1 GeneEffect file has no row for either ModelID. HAP1/22RV1 remain source-registered only.

This is the only card for this dataset. It supersedes the separate v1 and v2 cards; the v1
artifact remains on disk unmodified as the pre-provenance snapshot.

The pre-split surface has 184,962 rows, 172,838 unique pairs, 15,694 genes, 10 contexts,
11,999 multi-context pairs and 0 cross-context label changes. After registered-context
selection and row-level leakage removal the published table has 94,083 rows, 82,344 pairs,
15,517 genes and nine contexts.

## Role

The sole pair-label table and split authority for context-conditioned SL ranking:

```text
(gene_a, gene_b, cell_line) -> experimental screen hit/non-hit
```

The negative class is an experimentally screened non-hit **in the named context**. It is
stronger than a randomly sampled unknown pair, but it must never be described as universal
biological non-SL.

## Sole Input and Provenance

```text
data/SL_Benchmark_Formal/sl_integrated_pairs.csv
```

The builder reads no other pair-label source — not `sl_pairs_high_quality_labeled.csv`, Horlbeck,
Feng or DepMap co-dependency labels. The manifest records the input SHA-256. The sole-input claim
is about pair labels, **not** split-registration evidence or a complete upstream-lineage audit: the integrated
CSV has no study identifiers and cannot establish its upstream sources.

`source_row_id` is the zero-based raw CSV row index. It links contexts exploded from one
aggregate row; it **cannot** link separate records from the same experiment, for the same
reason — no study or evidence identifier exists in the source.

Registered contexts are locked by
[`../../configs/benchmarks/context_screen_v2_split.json`](../../configs/benchmarks/context_screen_v2_split.json),
which records DepMap ModelIDs, basal sources, and hashes for the 26Q1 GeneEffect file and
the nine-context basal registry. Contexts are never joined by informal name.

## Preprocessing Contract

The procedure adopts Feng et al. (Nat. Commun. 2024, DOI `10.1038/s41467-024-52900-7`)
data-preparation ideas only: standardized human gene names, canonical undirected pairs,
explicit binary labels, and a preference for experimental-screen negatives over sampled
unknown pairs. A raw row is retained only when:

1. `organisms == human`, both endpoints are `approved` or `updated`, human evidence is
   present, and `qc_flag` is empty;
2. `conflict == 0`, `sources == screen`, `evidence_types == experimental_screen`;
3. a positive row is `experimental` with every evidence item positive, or a negative row is
   `experimental_negative` with every evidence item negative;
4. `pair_human_ortholog` holds two distinct genes, uppercased and sorted into a canonical
   unordered pair;
5. `n_evidence == n_cell_lines == number of semicolon-separated context tokens`;
6. every retained context is an atomic cell-line-like token — `CTX:*`, aggregates such as
   `MULTIPLE`, non-human `mESC`, and composite strings are excluded.

Condition 5 plus a unanimous row label is the **only** basis for assigning the row label to
each exploded context, so every output row records
`context_assignment=unanimous_row_evidence_count_match` and
`label_confidence=silver_inferred`. This is a conservative inference, not reconstruction of
an unavailable per-evidence table.

After explosion, any pair–context key carrying both labels is dropped and duplicate same-label
keys are collapsed. No class balancing or negative sampling: natural imbalance is retained.

## Split Construction

`scripts/build_sl_context_benchmark.py` then:

1. retains pre-split contexts with at least 10 rows in each class;
2. keeps configured contexts with hash-pinned basal metadata; train and validation require
   GeneEffect, while an explicitly `sl_only` context is allowed only on test;
3. applies the tracked explicit assignments, with response anchors pinned to train;
4. removes every complete `source_row_id` group appearing on more than one split side.

Rule 4 is row-level: it prevents source-row leakage without transitively forcing whole
contexts onto one side. Same-side shared rows remain and are reported.

| Split | Context | ModelID | Positive | Negative | Total |
| --- | --- | --- | ---: | ---: | ---: |
| train | K562 | ACH-000551 | 1,669 | 10,270 | 11,939 |
| train | JURKAT | ACH-000995 | 95 | 9,124 | 9,219 |
| train | OVCAR8 | ACH-000696 | 89 | 643 | 732 |
| train | HAP1 | ACH-002475 | 56,993 | 20 | 57,013 |
| train | HT29 | ACH-000552 | 235 | 7,322 | 7,557 |
| validation | A549 | ACH-000681 | 392 | 1,581 | 1,973 |
| test | 22RV1 | ACH-000956 | 38 | 580 | 618 |
| test (SL only) | PC9 | ACH-000779 | 134 | 2,364 | 2,498 |
| test (SL only) | HELA | ACH-001086 | 170 | 2,364 | 2,534 |

K562 and JURKAT are response anchors. The explicit assignment puts HT29 on train and
22RV1/PC9/HELA on test. The split removed 128 crossing source groups (359 rows); afterwards
zero source rows and zero canonical pairs cross split sides.

## Audits

Before the split, K562/JURKAT share 9,219 source rows and HELA/PC9 share 2,523, both at
label agreement 1.0 — one aggregate row reporting "tested in 2 lines, unanimous" was
exploded into two context names. In the final table all 9,219 JURKAT rows still share a
source row with K562, but both are train.

**A549 is degenerate:** all 392 positives contain TRA2A and no A549 negative does, so
`1[TRA2A in {a,b}]` scores AUPR 1.0 on it. HT29's top positive gene is CDK6 at 0.297872 of
positives; HAP1 has only 20 negatives and a 0.999649 positive prior.

`filter_audit.csv` reports independent, overlapping per-condition failure counts — they
identify which conditions fail, but are **not** additive causal attribution. It currently
omits per-context losses for the `n_evidence == n_cell_lines == token_count` rule and for
atomic-context rejection; the manifest reports 829 count-mismatch rows and 6,832 invalid
tokens in aggregate only.

## Excluded Context

RPE1 (5,798 positives / 84,722 negatives) remains excluded. The available parental RPE1
single-cell run cannot be bound exactly to any DepMap RPE1 subclone, while the exact SS48
model has only bulk expression. The benchmark does not substitute parental cells or bulk
profiles for the required same-model basal single cells.

PC9 preprocessing retains 2,381/2,389 untreated cells after a 1,000-count, 500-gene and
20% mitochondrial-count gate. HeLa preprocessing retains all 720 cells, repairs the known
Excel-mangled symbols, maps unique GENCODE-v32 gene names to Ensembl IDs and aligns the
three matrices; the source omits mitochondrial genes, so mitochondrial QC is unavailable.

## Generated Files

Build: `uv run python scripts/build_sl_context_benchmark.py`.
PC9/HeLa basal build: `scripts/build_pc9_hela_basal.py` on the HPC.
Historical unsplit v1 remains under `data/SL_Benchmark_Formal/derived/context_screen_v1/` with `sl_context_pairs.csv`, `context_inventory.csv`, and `manifest.json`.
The pair table has no split/fold; `source_row_min_fdr` is aggregate-row, not per-context. Its inventory reports unique
genes and genes with both labels; the manifest freezes source/output hashes, filtering rules,
Feng-alignment decisions and row counts.

```text
configs/benchmarks/context_screen_v2_split.json and context_screen_v2_basal_registry.json   tracked authorities

data/SL_Benchmark_Formal/derived/context_screen_v2/
  sl_context_pairs.csv        final rows with split, scope, screen_cluster and source_row_id
  context_inventory.csv       pre-split context eligibility
  filter_audit.csv            independent per-context positive failure counts
  context_statistics.csv      final class, prior and positive-anchor statistics
  manifest.json               input/output/config hashes and split audit
```

Everything under `derived/` is gitignored. The tracked split manifest and the dataset
`split` column must agree exactly.

## Scope and Cautions

- No pre-split recurring pair changes label across contexts, so this table **cannot** test
  recovery of context-dependent label reversal.
- Validation is the degenerate A549 context. Test has 22RV1 plus SL-only PC9 and HeLa;
  GeneEffect residual metrics are evaluable only for 22RV1.
- PC9 and HeLa are one aggregate-label screen cluster, not independent test units; headline
  the de-duplicated diagnostic macro weights observable clusters; report each context too.
- Priors are extremely imbalanced and span a wide range. Report coverage and class counts
  first, then per-context AUPR minus prior; AUROC is secondary.
- `source_row_id` prevents only observable aggregate-row leakage. It cannot establish
  independence between separate records originating in the same unidentified study.
- The included contexts have not passed a ranking-candidate-universe, batch-confounding,
  power, or prospective-evaluation gate.
- Do not join context features by informal cell-line name without the verified ModelID map.

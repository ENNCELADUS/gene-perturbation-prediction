# SL Context Screen Benchmark

**Status:** current build is `context_screen_v2`, with a row-level context split updated 2026-08-16.
No model has run. The raw-filter audit is incomplete (see Audits). Basal single-cell candidates
were downloaded for the six original non-split contexts (`data/sl_dependency_v0/raw/context_basal_candidates/`). OVCAR8, HAP1 and
22RV1 are now registered; HAP1/22RV1 still need Tx1-ready exports before model execution.

This is the only card for this dataset. It supersedes the separate v1 and v2 cards; the v1
artifact remains on disk unmodified as the pre-provenance snapshot.

The pre-split surface has 184,962 rows, 172,838 unique pairs, 15,694 genes, 10 contexts,
11,999 multi-context pairs and 0 cross-context label changes. After registered-context
selection and row-level leakage removal the published table has 89,049 rows, 79,824 pairs,
15,171 genes and seven contexts.

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
the seven-context basal registry. Contexts are never joined by informal name.

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
2. keeps configured contexts having **both** DepMap GeneEffect and hash-pinned basal metadata;
   there is no 50/50 split gate, though leakage removal must retain both classes;
3. pins response anchors to train, sorts remaining contexts by ModelID, and allocates one
   test and one validation context;
4. removes every complete `source_row_id` group appearing on more than one split side.

Rule 4 is row-level: it prevents source-row leakage without transitively forcing whole
contexts onto one side. Same-side shared rows remain and are reported.

| Split | Context | ModelID | Positive | Negative | Total |
| --- | --- | --- | ---: | ---: | ---: |
| train | K562 | ACH-000551 | 1,668 | 10,268 | 11,936 |
| train | JURKAT | ACH-000995 | 95 | 9,124 | 9,219 |
| train | OVCAR8 | ACH-000696 | 89 | 635 | 724 |
| train | 22RV1 | ACH-000956 | 38 | 580 | 618 |
| train | HAP1 | ACH-002475 | 56,989 | 20 | 57,009 |
| validation | A549 | ACH-000681 | 392 | 1,610 | 2,002 |
| test | HT29 | ACH-000552 | 229 | 7,312 | 7,541 |

K562 and JURKAT are response anchors and therefore train. Sorting the remaining ModelIDs
puts HT29 in test and A549 in validation. The split removed 114 crossing source groups —
304 rows, including 106 HT29, 92 OVCAR8 and 91 A549 rows. Afterwards zero source rows
and zero canonical pairs cross split sides.

## Audits

Before the split, K562/JURKAT share 9,219 source rows and HELA/PC9 share 2,523, both at
label agreement 1.0 — one aggregate row reporting "tested in 2 lines, unanimous" was
exploded into two context names. In the final table all 9,219 JURKAT rows still share a
source row with K562, but both are train.

**A549 is degenerate:** all 392 positives contain TRA2A and no A549 negative does, so
`1[TRA2A in {a,b}]` scores AUPR 1.0 on it. HT29's top positive gene is CDK6 at 0.301 of
positives; HAP1 has only 20 negatives and a 0.999649 positive prior.

`filter_audit.csv` reports independent, overlapping per-condition failure counts — they
identify which conditions fail, but are **not** additive causal attribution. It currently
omits per-context losses for the `n_evidence == n_cell_lines == token_count` rule and for
atomic-context rejection; the manifest reports 829 count-mismatch rows and 6,832 invalid
tokens in aggregate only.

## Candidate Contexts Not in the Split

Verified 2026-08-16 against the downloaded basal candidates and DepMap 26Q1 `Model.csv` /
`CRISPRGeneEffect.csv`. Registration needs three things at once — labels passing the 10/10
pre-split gate, hash-pinned basal metadata, and a pinned 26Q1 GeneEffect row; model execution separately requires a verified expression artifact/cache.

| Context | Labels pos/neg | Basal candidate | 26Q1 GeneEffect | Blocker |
| --- | ---: | --- | --- | --- |
| RPE1 | 5,798 / 84,722 | 11,485 non-targeting cells; total raw counts in `layers["matrix"]` | 5 of 6 subclones screened | **no plain-RPE1 ModelID** |
| PC9 | 135 / 2,388 | 2,389 untreated cells, raw counts | `ACH-000779` **absent** | no row in pinned 26Q1 `CRISPRGeneEffect.csv` |
| HELA | 171 / 2,395 | 720 manually isolated cells; SMART-Seq2-derived processed integer counts; GEO reports UMI counting | `ACH-001086` **absent** | no row in pinned 26Q1 `CRISPRGeneEffect.csv` |

RPE1 is the highest-value remaining unblock by pair count, but DepMap
has no generic RPE1. Its six Non-Cancerous single-cell-derived subclones are `RPE1-ss6`, `-ss48`,
`-ss51`, `-ss77`, `-ss111`, and `-ss119`; five have GeneEffect rows, while `RPE1-ss111` does not.
Binding the Perturb-seq run to one subclone or averaging five is a consequential modelling
decision that must be declared, not resolved by prohibited name matching.

Two further preprocessing notes. RPE1 defaults to `X == layers["spliced"]`; total raw counts are
in `layers["matrix"]`, so the chosen representation must be declared. HeLa used manual mouth-pipette
isolation and a Tang-modified SMART-Seq2-derived protocol; GEO describes processing as UMI counting.
All three files have different gene sets; the third has two duplicate gene keys requiring deduplication.

## Generated Files

Build: `uv run python scripts/build_sl_context_benchmark.py`.
Historical unsplit v1 remains under `data/SL_Benchmark_Formal/derived/context_screen_v1/` with `sl_context_pairs.csv`, `context_inventory.csv`, and `manifest.json`.
The pair table has no split/fold; `source_row_min_fdr` is aggregate-row, not per-context. Its inventory reports unique
genes and genes with both labels; the manifest freezes source/output hashes, filtering rules,
Feng-alignment decisions and row counts.

```text
configs/benchmarks/context_screen_v2_split.json and context_screen_v2_basal_registry.json   tracked authorities

data/SL_Benchmark_Formal/derived/context_screen_v2/
  sl_context_pairs.csv        final rows with model_id, split and source_row_id
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
- Validation is the degenerate A549 context; test is HT29 alone.
- Priors are extremely imbalanced and span a wide range. Report coverage and class counts
  first, then per-context AUPR minus prior; AUROC is secondary.
- `source_row_id` prevents only observable aggregate-row leakage. It cannot establish
  independence between separate records originating in the same unidentified study.
- The included contexts have not passed a ranking-candidate-universe, batch-confounding,
  power, or prospective-evaluation gate.
- Do not join context features by informal cell-line name without the verified ModelID map.

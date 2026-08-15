# SL Context Screen Benchmark v2

**Status:** row-level context split built 2026-08-15; no model has run. The raw-filter audit
remains incomplete.

The pre-split v2 surface reproduces v1 exactly: 184,962 rows, 172,838 unique pairs,
15,694 genes, 10 contexts, 11,999 multi-context pairs and 0 cross-context label changes.
After executable-context selection and row-level leakage removal, the published table has
30,726 rows, 21,507 pairs, 3,101 genes and four contexts.

## Role

The sole pair-label table and split authority for context-conditioned SL ranking:

```text
(gene_a, gene_b, cell_line) -> experimental screen hit/non-hit
```

The negative class is a screened non-hit in the named context, not universal biological
non-SL. v2 adds raw-row provenance, audits and a fixed context split to v1; it does not
change any retained v1 label.

## Inputs and Provenance

The only pair-label input is:

```text
data/SL_Benchmark_Formal/sl_integrated_pairs.csv
```

No Feng2024, Horlbeck or DepMap co-dependency labels are merged. The generated manifest
records its SHA-256. `source_row_id` is the zero-based raw CSV row index and links contexts
exploded from one aggregate row. It cannot link separate records from the same experiment,
because the source carries no study or evidence identifier.

Executable contexts are locked by [`../../configs/benchmarks/context_screen_v2_split.json`](../../configs/benchmarks/context_screen_v2_split.json),
which records DepMap ModelIDs, basal sources and hashes for the 26Q1 GeneEffect file and
42-line basal manifest. Contexts are never joined by informal name.

## Construction

`scripts/build_sl_context_benchmark.py`:

1. retains human experimental-screen rows with approved/updated endpoints and unanimous
   positive or negative evidence;
2. canonicalizes unordered gene pairs, explodes identifiable contexts, removes conflicting
   pair-context keys and retains pre-split contexts with at least 10 rows in each class;
3. requires at least 50 positives and 50 negatives for a split context and keeps only
   contexts with both DepMap GeneEffect and basal single-cell input;
4. pins response anchors to train, sorts other contexts by ModelID, then allocates one test
   and one validation context;
5. removes every complete `source_row_id` group appearing on more than one split side.

The last rule is row-level: it prevents source-row leakage without transitively forcing
whole contexts onto one side. Same-side shared rows remain visible and are reported.

## Published Split

| Split | Context | ModelID | Positive | Negative | Total |
| --- | --- | --- | ---: | ---: | ---: |
| train | K562 | ACH-000551 | 1,668 | 10,268 | 11,936 |
| train | JURKAT | ACH-000995 | 95 | 9,124 | 9,219 |
| validation | A549 | ACH-000681 | 392 | 1,618 | 2,010 |
| test | HT29 | ACH-000552 | 234 | 7,327 | 7,561 |

K562 and JURKAT are response anchors and therefore train. Sorting the remaining ModelIDs
places HT29 (`ACH-000552`) in test and A549 (`ACH-000681`) in validation. The split removes
86 crossing source groups: 172 rows total (A549 83 negatives; HT29 85 negatives and one
positive; K562 two negatives and one positive). After removal, zero source rows and zero
canonical pairs cross split sides.

## Published Audits

Before the split, K562/JURKAT share 9,219 source rows and HELA/PC9 share 2,523; both pairs
have label agreement 1.0. In the final table all 9,219 JURKAT rows still share a source row
with K562, but both contexts are train.

A549 remains degenerate: all 392 positives contain TRA2A and no A549 negative does, so
`1[TRA2A in {a,b}]` has AUPR 1.0. HT29's top positive gene is CDK6 at 0.299 of positives.

`filter_audit.csv` contains independent, overlapping failure counts. They identify which
conditions fail but are not additive causal attribution. It currently omits per-context
losses for `n_evidence == n_cell_lines == token_count` and atomic-context rejection; the
manifest reports 829 count-mismatch rows and 6,832 invalid tokens only in aggregate.

## Generated Files

```text
configs/benchmarks/context_screen_v2_split.json   tracked split authority

data/SL_Benchmark_Formal/derived/context_screen_v2/
  sl_context_pairs.csv        final rows with model_id, split and source_row_id
  context_inventory.csv       pre-split context eligibility
  filter_audit.csv            independent per-context positive failure counts
  context_statistics.csv      final class, prior and positive-anchor statistics
  manifest.json               input/output/config hashes and split audit
```

Everything under `derived/` is gitignored. The tracked split manifest and dataset `split`
column must agree exactly.

## Scope and Cautions

- No pre-split recurring pair changes label across contexts, so v2 cannot test recovery of
  context-dependent label reversal.
- Validation is degenerate A549; test is HT29.
- Priors remain highly imbalanced. Report coverage and class counts first, then per-context
  AUPR minus prior; AUROC is secondary.
- `source_row_id` prevents only observable aggregate-row leakage. It cannot establish
  independence between separate records from the same unidentified source study.

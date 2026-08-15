# SL Context Screen Benchmark v2

**Status:** specified, **not built**. No v2 artifact exists on disk. This card is the build
contract required by [`../03-experiment-protocol.md`](../03-experiment-protocol.md) §2; the
measured numbers below are v1 statistics that motivate the rebuild, not v2 results.

## Role

The pair–cell-line label table for the context-conditioned SL benchmark, and the **authority
for the published train/val/test split**. The supervised object is unchanged from v1:

```text
(gene_a, gene_b, cell_line) -> experimental screen hit/non-hit
```

The negative class is an experimentally screened non-hit in the named context. It is
stronger than a randomly sampled unknown pair and must not be described as universal
biological non-SL.

v2 differs from [`sl-context-screen-v1.md`](sl-context-screen-v1.md) in exactly two ways: it
carries row provenance, and it carries the split. Do not overwrite v1.

## Sole Input

```text
data/SL_Benchmark_Formal/sl_integrated_pairs.csv
```

No other label source is merged — not Feng2024, not Horlbeck, not DepMap co-dependency. The
input SHA-256 is recorded in the generated manifest. As with v1, the sole-input claim is
about the direct build dependency, not a complete upstream-lineage audit: the integrated CSV
does not carry study identifiers.

## Why v1 Is Rebuilt

Three properties of v1. The first two were computed from `sl_context_pairs.csv`; the third
comes from `context_inventory.csv`, because `sl_context_pairs.csv` holds only the ten
contexts that passed the `>= 10/10` gate and therefore cannot show a zero-positive context
at all. The rebuild must record the exact artifact hash and query behind each figure.

**Duplicate screens.** K562/JURKAT share 9,219 rows — 100% of JURKAT, Jaccard 0.772 — and
HELA/PC9 share 2,523 rows — 100% of PC9, Jaccard 0.983 — both at label agreement exactly
1.0000. These rows carry `source_n_evidence == 2, source_row_count == 1`: one aggregated
source row reported "tested in 2 lines, unanimous" and the builder copied its single label
to each exploded context. **97.9% of v1's 11,999 cross-context recurring pairs are this
artifact.** Consequence: those context names are not independent evaluation units, and a
split that separates them holds out nothing.

**A degenerate anchor in A549.** All 392 A549 positives contain TRA2A, and no A549 row
containing TRA2A is negative, so the indicator `1[TRA2A in {a,b}]` scores AUPR 1.0 on that
context. Positive-anchor concentration varies widely across contexts and must be published
so users can see it rather than discover it.

**Missing positives in repeated patterns** (from `context_inventory.csv`). Nine contexts
carry 933–941 negatives and zero
positives (GI1, HS936T, HS944T, HSC5, IPC298, PATU8988S, PK1, MEL202, MELJUSO); five carry
exactly 684 negatives and zero positives (A427, CAL27, CAL33, MCF10A, MCF7); THP1 carries
1,332 positives and zero negatives. Identical counts across unrelated lineages indicate a
filter or explosion artifact rather than biology.

## Build Contract

Re-run `scripts/build_sl_context_benchmark.py` into `derived/context_screen_v2/`, adding:

1. **`source_row_id`** on every exploded row. This links contexts exploded from one
   aggregate source row. It **cannot** link separate rows produced by the same underlying
   experimental screen, because the source carries no study or evidence identifier. It is
   therefore used for one purpose only — keeping duplicated contexts on the same side of the
   split — and no independence claim may rest on it.
2. **A per-filter, per-context drop audit** covering `sources == screen`,
   `evidence_types == experimental_screen`, `conflict == 0`, the all-evidence-unanimous rule,
   `n_evidence == n_cell_lines == n_context_tokens`, and the atomic-context token rule.
   The audit must specifically account for the zero-positive contexts above.

The v1 preprocessing contract otherwise carries over unchanged, including canonical
unordered pairs, the removal of any pair–context key holding both labels, and
`context_assignment=unanimous_row_evidence_count_match` with every row marked
`label_confidence=silver_inferred`. No class balancing or negative sampling is applied; the
natural imbalance is part of the dataset.

## The Published Split

The split ships as a `split` column in the table, but its **canonical, tracked** copy is
`configs/benchmarks/context_screen_v2_split.json`. `/data/` is gitignored in full, so a
manifest living only beside the CSV would be neither distributed nor verifiable; the
in-dataset column is a mirror and a mismatch is a hard error. It is constructed once by this
rule and no experiment may redefine it:

- a context is **eligible** with at least 50 positives and at least 50 negatives;
- a context is **executable** if it has DepMap GeneEffect and basal single-cell input;
- **only executable contexts enter the benchmark on any side.** Arm A must predict a profile
  for every context it touches, so an eligible-but-non-executable context is unusable in
  train and validation just as it is in test;
- the four Perturb-seq anchors (K562, HCT116, Jurkat, HepG2) are **pinned to train**, since
  they carry the only response supervision;
- contexts sharing a `source_row_id` group stay on the same side;
- assignment is deterministic: sort remaining executable groups by ModelID and allocate to
  test, validation and train under counts fixed in the manifest before any context's
  difficulty is inspected.

HELA can never be executable: it has neither a 26Q1 GeneEffect target nor a compatible basal
input, and the planned acquisitions supply basal cells, not GeneEffect. RPE1 and HAP1
DepMap-CRISPR membership must be verified, not assumed. On v1 counts HAP1 fails eligibility
outright (56,994 positives against 20 negatives).

Of the ten contexts clearing v1's permissive `>= 10/10` preprocessing gate, only **K562,
Jurkat, A549 and HT29** appear in the 42-line basal manifest at
`results/phase_a_tx1_20260724/cell_line_manifest.csv`. Basal acquisition for the remainder is
a prerequisite, and each acquired context records its source and accession here.

## Generated Files

```text
configs/benchmarks/context_screen_v2_split.json   TRACKED — canonical split

data/SL_Benchmark_Formal/derived/context_screen_v2/
  sl_context_pairs.csv        + source_row_id, + split (mirror)
  filter_audit.csv
  context_statistics.csv      class counts, prior, distinct positive genes, top-gene share
  manifest.json               source and output hashes, filter rules, row counts
```

Everything under `derived/` is gitignored; only the split manifest is tracked. All must be
frozen before any model run.

## Scope and Cautions

- **The label marginals of this table have been inspected.** Eligibility thresholds and the
  known degeneracies above were derived after looking at v1 counts. This is a
  development-grade surface, declared as such in [`../01-blueprint.md`](../01-blueprint.md)
  §8, not laundered by additional selection rules.
- No retained pair changes label across contexts, so the table cannot evaluate recovery of
  context-dependent label reversal. The duplicate-screen finding makes this caveat stronger,
  not weaker.
- Contexts remain extremely imbalanced and the priors span roughly an eighteen-fold range.
  Use AUPR minus prior and context-macro aggregation; never infer quality from AUROC alone.
- Do not join contexts to omics or dependency data by informal cell-line name. Use the
  checked-in `context -> ModelID` map and fail loudly on an unmapped context.

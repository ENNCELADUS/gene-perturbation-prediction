# SL Context Screen Benchmark v1

**Status:** historical. Superseded by [`sl-context-screen-v2.md`](sl-context-screen-v2.md),
which adds row provenance and the published split. v1 lacks both, and three of its
properties — duplicate screens exploded into separate context names, a single-gene anchor
in A549, and contexts with zero positives — are why v2 exists. The v1 artifact stays on disk
and is not rebuilt or edited.

## Role

This is an **unsplit pair–cell-line label table** derived only from
`data/SL_Benchmark_Formal/sl_integrated_pairs.csv`. It is intended as the input
surface for a future context-conditioned SL benchmark. It does not contain
CV1/CV2/CV3 folds or train/validation/test assignments.

The supervised object is:

```text
(gene_a, gene_b, cell_line) -> experimental screen hit/non-hit
```

The negative class means an experimentally screened non-hit in the named
cell-line context. It is stronger than a randomly sampled unknown pair, but it
must not be described as universal biological non-SL.

## Sole Input

```text
data/SL_Benchmark_Formal/sl_integrated_pairs.csv
```

The builder does not directly read `sl_pairs_high_quality_labeled.csv`, Horlbeck
data, the Feng benchmark caches, DepMap, or any other label source. The exact
input SHA-256 is recorded in the generated manifest. The integrated CSV does not
carry enough study identifiers to independently prove that every upstream source
excluded a particular published dataset; the sole-input claim is therefore about
the direct build dependency, not a complete upstream-lineage audit.

## Preprocessing Contract

The procedure adopts the relevant data-preparation ideas from Feng et al.
(Nature Communications 2024, DOI `10.1038/s41467-024-52900-7`): standardized
human gene names, canonical undirected pairs, explicit binary labels, and a
preference for experimental-screen negatives over sampled unknown pairs.

A raw row is retained only when:

1. `organisms == human`, both endpoints have `approved` or `updated` human-gene
   status, human evidence is present, and `qc_flag` is empty;
2. `conflict == 0`, `sources == screen`, and
   `evidence_types == experimental_screen`;
3. a positive row is experimental and every evidence item is positive, or a
   negative row is `experimental_negative` and every evidence item is negative;
4. `pair_human_ortholog` contains two distinct genes, which are uppercased and
   sorted into a canonical unordered pair;
5. `n_evidence == n_cell_lines == number of semicolon-separated context tokens`;
6. every retained context is an atomic cell-line-like token. `CTX:*`, aggregate
   values such as `MULTIPLE`, non-human `mESC`, and descriptive/composite strings
   are excluded.

Condition 5 plus a unanimous row label is the only basis for assigning the row
label to each exploded context. The raw table lacks evidence IDs and
per-context scores, so the output records
`context_assignment=unanimous_row_evidence_count_match`. This is a conservative
inference, not direct reconstruction of an unavailable per-evidence table.
Accordingly, every output row is marked `label_confidence=silver_inferred`.

After explosion, any canonical pair–context key with both labels is removed.
Duplicate same-label keys are collapsed. A context enters the pair-classification
candidate table only when it has at least 10 positive and 10 negative pairs. This
is a permissive preprocessing threshold, not proof of formal benchmark or ranking
eligibility. The context inventory also reports unique genes and genes appearing
with both labels so later task design can audit anchor coverage.

No class balancing or negative sampling is applied. This deliberately differs
from Feng's sampled 1:1/1:5/1:20/1:50 regimes because this source already contains
experimental screen non-hits. Natural class imbalance is part of this dataset.

## Build

```bash
uv run python scripts/build_sl_context_benchmark.py
```

Generated files are gitignored under:

```text
data/SL_Benchmark_Formal/derived/context_screen_v1/
  sl_context_pairs.csv
  context_inventory.csv
  manifest.json
```

`sl_context_pairs.csv` contains no split or fold columns. Its
`source_row_min_fdr` field is the source aggregate-row value, not a per-context
FDR. `manifest.json` freezes the source/output hashes, filtering rules,
Feng-alignment decisions, and row counts.

## Scope and Cautions

- This table is a candidate input for designing a context-conditioned benchmark,
  subject to the context-level counts and inferred context assignment above. The
  included contexts have not yet passed a formal ranking-candidate-universe,
  batch-confounding, power, or prospective evaluation gate.
- In the v1 build, 11,999 pairs occur in more than one retained context, but no
  retained pair changes label across contexts. The table therefore supports
  context-stratified evaluation and later held-out-context tests, but it cannot
  evaluate whether a model recovers context-dependent label reversal for the
  same pair.
- It does not by itself demonstrate cross-cell-line generalization; that requires
  a later, explicitly frozen held-out-context protocol.
- Extremely imbalanced contexts remain extremely imbalanced. Use AUPR and
  context-macro aggregation later; do not infer quality from AUROC alone.
- Do not join context features by informal cell-line names without a separately
  verified identifier mapping.

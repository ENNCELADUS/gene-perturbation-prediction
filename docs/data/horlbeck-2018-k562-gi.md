# Horlbeck 2018 K562 CRISPRi genetic-interaction map

**Status:** acquired, normalized, and coverage-audited on 2026-07-22.

## Role

This is the continuous K562 fitness-GI anchor for measured-epistasis evaluation.
It can support a K562 mechanism test only. It does not establish a non-K562 or
multi-cell-line mechanism.

## Primary source and frozen artifact

- Paper: Horlbeck et al., *Cell* 2018, [doi:10.1016/j.cell.2018.06.010](https://doi.org/10.1016/j.cell.2018.06.010).
- Author-deposited data: Mendeley Data version 1,
  [doi:10.17632/rdzk59n6j4.1](https://doi.org/10.17632/rdzk59n6j4.1), CC BY 4.0.
- Mendeley file: `GI_map_treeview.zip`, file ID `5542388b-e895-4f29-af07-bfe2eecd0655`.
- Local raw archive:
  `data/sl_dependency_v0/raw/horlbeck_2018/GI_map_treeview.zip`.
- Extracted K562 matrix: `K562_gene.cdt`.

The paper describes a 472-gene library design with 111,628 unique combinations.
The author-deposited, post-QC K562 gene-level matrix contains 448 genes and
100,128 unique off-diagonal pairs. The matrix is complete, finite, and exactly
symmetric.

## Frozen GI field and sign

The source `K562_gene.cdt` is a square Java TreeView matrix. The numeric matrix
cells, after excluding the diagonal and retaining one unordered copy per pair,
are the gene-level genetic-interaction score. The normalized column name is
frozen as:

`gi_score`

Do not substitute the single/double growth phenotype `gamma`, a GI-profile
correlation, or a cluster score for this field.

- Negative `gi_score`: synergistic interaction; increasingly negative means
  stronger synthetic-sick/lethal behavior.
- Positive `gi_score`: buffering/suppressive interaction.
- The continuous score is primary. When a binary strong-SL label is required,
  the frozen rule is the strict inequality `gi_score < -3.0`.

The matrix contains 1,523 pairs under that strong-SL rule. The `+/-3` strong-GI
cutoff is defined from negative-control interactions in the paper; the strict
negative rule matches the Horlbeck K562 usage in later SL benchmarking.

## Symbol normalization

The 2018 matrix includes legacy symbols. Coverage uses a frozen HGNC complete-set
download from 2026-07-22. Resolution order is exact symbol, unique case-insensitive match, then a unique
approved HGNC previous/alias-symbol mapping. Ambiguous or unresolved mappings
remain uncovered; they are not manually forced.

## Exp05 Replogle coverage contract

The primary coverage definition requires both genes to:

1. have a single-gene condition in `replogle_k562_gwps` with at least 8 observed
   cells, matching the exp05 `min_cells_per_gene`; and
2. occur in the frozen exp05 pool manifest
   `k562_pool_depmap_fixed_seed42.csv`.

Under that definition, 408 of 448 Horlbeck genes are covered, yielding 83,028
of 100,128 pairs with both genes covered (82.92%). If the DepMap-label-qualified
fixed-pool restriction is removed and the question is only whether Replogle has
at least 8 observed cells, 436 genes and 94,830 pairs are covered (94.71%). The
former is the exp05-ready bound; the latter is the broader observed-response
bound.

## Reproduction

The builder is not part of the tracked code; it is kept in the local, gitignored
archive and runs from the repository root:

```bash
uv run python archive/2026-09-05-inactive-routes/scripts/prepare_horlbeck_2018.py
```

Generated, gitignored artifacts, under `data/sl_dependency_v0/processed/horlbeck_2018/`: `k562_gene_pairs.csv`,
`k562_gene_coverage.csv` and `coverage_summary.json`.

The pair table contains the frozen `gi_score`, strong-SL flag, canonical symbols,
cell counts, and both broad observed-response and strict exp05 coverage flags.

## Limitations

- This is CRISPRi partial loss-of-function in K562, not a knockout assay and not
  a general cell-line result.
- The screened genes are enriched for growth-relevant genes, so coverage and GI
  prevalence do not represent the genome uniformly.
- Coverage establishes data availability, not statistical power for a particular
  estimator. Effect-size and uncertainty requirements still need registration.

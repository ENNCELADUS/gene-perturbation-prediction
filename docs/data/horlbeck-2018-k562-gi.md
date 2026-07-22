# Horlbeck 2018 K562 CRISPRi genetic-interaction map

**Status:** acquired, checksum-verified, normalized, and coverage-audited on
2026-07-22.

## Role

This is the continuous K562 fitness-GI anchor for measured-epistasis evaluation.
It can support a K562 mechanism test only. It does not establish a non-K562 or
multi-cell-line mechanism.

## Primary source and frozen artifact

- Paper: Horlbeck et al., *Cell* 2018, [doi:10.1016/j.cell.2018.06.010](https://doi.org/10.1016/j.cell.2018.06.010).
- Author-deposited data: Mendeley Data version 1,
  [doi:10.17632/rdzk59n6j4.1](https://doi.org/10.17632/rdzk59n6j4.1), CC BY 4.0.
- Mendeley file: `GI_map_treeview.zip`, file ID
  `5542388b-e895-4f29-af07-bfe2eecd0655`, 9,280,985 bytes.
- Published and locally verified SHA-256:
  `d389b3ebdd2c88f86cc446f272f3ff6526df635d666cd17d67440bf10c0f9115`.
- Local raw archive:
  `data/sl_dependency_v0/raw/horlbeck_2018/GI_map_treeview.zip`.
- Extracted K562 matrix: `K562_gene.cdt`, SHA-256
  `e3f7c4caa860cda8daec94be56e6900d9737781499e5357eaf7b5f18e7534c19`.

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
download from 2026-07-22, SHA-256
`25042e23a296bfa046bf7bc65ee110139cc03a9a55fb7f15ab34b19226cf6417`.
Resolution order is exact symbol, unique case-insensitive match, then a unique
approved HGNC previous/alias-symbol mapping. Ambiguous or unresolved mappings
remain uncovered; they are not manually forced.

## Exp05 Replogle coverage contract

The primary coverage definition requires both genes to:

1. have a single-gene condition in `replogle_k562_gwps` with at least 8 observed
   cells, matching the exp05 `min_cells_per_gene`; and
2. occur in the frozen exp05 pool manifest
   `k562_pool_depmap_fixed_seed42.csv`.

Input hashes are:

- Replogle condition table:
  `76f8f055549c17a466d13f8c81d7788a8d8ae08eac0817703d4848c653d63397`;
- exp05 fixed pool manifest:
  `c33dbe6cbdd78d0e590a9810eab8a47f74b1a7d1ff3d9872ceca26d65cd3d6ef`.

Under that definition, 408 of 448 Horlbeck genes are covered, yielding 83,028
of 100,128 pairs with both genes covered (82.92%). If the DepMap-label-qualified
fixed-pool restriction is removed and the question is only whether Replogle has
at least 8 observed cells, 436 genes and 94,830 pairs are covered (94.71%). The
former is the exp05-ready bound; the latter is the broader observed-response
bound.

## Reproduction

Run:

```bash
uv run python scripts/prepare_horlbeck_2018.py
```

Generated, gitignored artifacts are:

- `data/sl_dependency_v0/processed/horlbeck_2018/k562_gene_pairs.csv`, SHA-256
  `8cc5d6eba748b5284126f2f13bd8e8eccf3bdeea11a9d3bfa7571e5ce40f12ad`;
- `data/sl_dependency_v0/processed/horlbeck_2018/k562_gene_coverage.csv`,
  SHA-256
  `ce54ebe8631ffb47027f600db83c24252f29f3fd7afbc1ad67b026cc925e241a`;
- `data/sl_dependency_v0/processed/horlbeck_2018/coverage_summary.json`,
  SHA-256
  `95044a3607093be4d41e3dda9f69d57faa35d8eb60f917bd331b685bb0075e92`.

The pair table contains the frozen `gi_score`, strong-SL flag, canonical symbols,
cell counts, and both broad observed-response and strict exp05 coverage flags.

## Limitations

- This is CRISPRi partial loss-of-function in K562, not a knockout assay and not
  a general cell-line result.
- The screened genes are enriched for growth-relevant genes, so coverage and GI
  prevalence do not represent the genome uniformly.
- Coverage establishes data availability, not statistical power for a particular
  estimator. Effect-size and uncertainty requirements still need registration.

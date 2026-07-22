# Horlbeck K562 GI acquisition and exp05 coverage

**Status:** completed on 2026-07-22; no mechanistic model evaluation has run.

The author-deposited Horlbeck 2018 K562 gene-level GI map was acquired from
Mendeley Data, matched its published SHA-256, and passed completeness and exact
symmetry checks. The processed matrix has 448 genes and 100,128 unique
off-diagonal gene pairs.

The frozen continuous field is `gi_score`, taken from the numeric cells of
`K562_gene.cdt`. Negative scores are synergistic/synthetic-sick-lethal and
positive scores are buffering. The optional strong-SL rule is
`gi_score < -3.0`, which identifies 1,523 pairs.

## Coverage result

| Coverage surface | Horlbeck genes | Pairs with both genes | Pair fraction |
| --- | ---: | ---: | ---: |
| Replogle GWPS, at least 8 observed cells | 436 / 448 | 94,830 / 100,128 | 94.71% |
| Replogle GWPS plus frozen exp05 pool membership | 408 / 448 | 83,028 / 100,128 | 82.92% |

The second row is the binding exp05-ready coverage bound. The first row shows
that most additional loss comes from the label-qualified fixed pool rather than
from absence of observed Replogle cells.

Full provenance, hashes, symbol-resolution rules, and reproduction instructions
are in the [dataset card](../data/horlbeck-2018-k562-gi.md).

## Interpretation

The K562 measured-GI test has substantial pair coverage and is not blocked by
gross Replogle-vocabulary absence. This does not by itself establish estimator
power: the formal test still needs a frozen candidate universe, effect-size and
uncertainty criteria, and gene/pair separation from any calibration labels.

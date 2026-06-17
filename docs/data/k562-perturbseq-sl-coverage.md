# K562 Perturb-seq Coverage of the SL Candidate Universe

## Role

This page records how local K562 Perturb-seq / CRISPRi response sources cover
the K562-filtered `SL_benchmark` candidate gene universe. It is a coverage
audit for joining observed perturbation-response bags to gene-pair SL benchmark
records; it is not an SL label source.

## Inputs

Coverage was computed on 2026-06-17 from:

- SL universe: union of `gene_a_symbol` and `gene_b_symbol` in
  `data/CV1_Rand_1to1_k562_depmap_pairs_balanced.csv`,
  `data/CV2_Rand_1to1_k562_depmap_pairs_balanced.csv`, and
  `data/CV3_Rand_1to1_k562_depmap_pairs_balanced.csv`.
- SL universe size: 9471 unique genes.
- Perturbation sources under `data/sl_dependency_v0/raw/`.

Generated outputs:

- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_summary.csv`
- `data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_conditions.csv`

## Per-Source Coverage

| Source | Parsed target conditions | Conditions with any SL gene | Unique target genes | Unique target genes in SL universe | SL coverage | Cells in parsed conditions | Cells in SL-covered conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Replogle K562 essential | 2057 | 1528 | 2057 | 1528 | 16.13% | 299694 | 227433 |
| Replogle K562 GWPS | 9866 | 6070 | 9866 | 6070 | 64.09% | 1914250 | 1166742 |
| Dixit 2016 K562 | 305 | 295 | 11 | 8 | 0.08% | 68923 | 54628 |
| Adamson 2016 pilot | 7 | 5 | 7 | 5 | 0.05% | 3983 | 2873 |
| Adamson 2016 UPR epistasis | 16 | 11 | 15 | 10 | 0.11% | 11331 | 4955 |
| Adamson 2016 UPR Perturb-seq | 110 | 83 | 90 | 65 | 0.69% | 55328 | 40010 |
| Jost/Replogle dual-sgRNA `dual` | 510 | 153 | 403 | 82 | 0.87% | 60049 | 16721 |
| Jost/Replogle dual-sgRNA `dolcetto` | 442 | 295 | 150 | 100 | 1.06% | 61655 | 39567 |

## Union Coverage

| Source set | Unique SL genes covered | Coverage of 9471-gene universe |
| --- | ---: | ---: |
| Existing Replogle essential + Adamson sources | 1554 | 16.41% |
| New Replogle GWPS + Dixit + Jost/Replogle sources | 6070 | 64.09% |
| All readable local sources on 2026-06-17 | 6074 | 64.13% |
| Incremental new-source coverage over existing sources | 4520 | 47.72 percentage points |

The practical coverage gain is dominated by Replogle K562 GWPS. Dixit and
Jost/Replogle are useful as K562 non-CRISPRa supplements and sanity checks, but
they add little broad candidate-universe coverage.

## Parsing Rules and Caveats

- Replogle sources use `obs["gene"]` as the perturbation condition.
- Dixit uses `obs["perturbation_name"]`; multi-gene labels separated by `+`
  were expanded to component genes.
- Adamson uses `obs["perturbation"]`; guide suffixes such as `_pDS...` were
  stripped, and UPR aliases were mapped as `PERK -> EIF2AK3` and
  `IRE1 -> ERN1`.
- Jost/Replogle uses `guide_identity` from `*_cell_identities.csv.gz`; guide
  suffixes were parsed to gene symbols.
- Control-like labels such as `non-targeting`, `control`, `INTERGENIC...`,
  `neg_ctrl`, `Gal4...`, and `(mod)` labels were excluded from target coverage.
- Norman CRISPRa is intentionally excluded from this coverage page because it
  is not loss-of-function / CRISPRi-compatible with the K562 DepMap alignment
  used in this repo.
- Gasperini enhancer CRISPRi was not present under
  `data/sl_dependency_v0/raw/gasperini/` at the time of this audit and is not
  included.

# SL Study Inventory

**Status:** discovery inventory built 2026-10-29 for the
[multi-study SL benchmark](../specs/2026-10-29-multi-study-sl-benchmark-design.md). Nothing is downloaded or labelled
yet. Every ModelID and every kind on a `candidate` row is a proposal for review. Nothing here is SL evidence.

## Files

| File | Rows | Content |
| --- | ---: | --- |
| `configs/benchmarks/sl_studies.csv` | 2,261 | One row per study: `kind`, `platform`, per-line score, authors' call, data location, licence, `in_synlethdb3`, `in_slkb`, `found_via`, `curation_status`. For a curated row, `note` holds `decision=` |
| `configs/benchmarks/sl_study_lines.csv` | 846 | One row per (study, line): the line as the source spells it, `engineered_genotype`, `proposed_model_id`, `match_evidence` |
| `configs/benchmarks/single_cell_atlas_candidates.csv` | +41 | The basal scRNA sources found for the screened lines, appended with an empty `review` and `wave`, so the builder's `propose` step reviews them before use |

The search scripts, full-text pulls and per-line search logs are local only, in
`docs/notes/2026-10-29-sl-inventory/`.

## Sources searched

- **SynLethDB 3.0** (Zenodo 22843223, 2026-09-19, CC BY 4.0). All 375 screen PMIDs are resolved. 636 SL rows without
  a PMID cannot be traced to a paper. The database has no scores, joins several lines or PMIDs in one field, has no
  screen newer than 2022, and has 0 SL rows for Ito's lines.
- **SLKB** (last updated 2023-05-23). Its 11 studies were taken from the NAR paper's Table 1. The 230 MB SQL dump was
  not fetched.
- **BioGRID ORCS 2.0.18.** 2,217 screens were read. 115 publications have a genetic background or a drug with a named
  target.
- **Europe PMC** (6 queries, 2014–2026) and citation chasing of Harle, Flister, Klingbeil, Ito, Kebabci and SLKB. The
  combinatorial, paralog and citation searches stopped adding studies. The broad drug-anchor and low-throughput
  searches did not.

## Counts

`kind` over all rows: drug_anchor 1,051, low_throughput 734, rnai 273, combinatorial 124 and anchor 79. Curation status:
746 rows are `excluded_flagged` and 1,312 are `candidate` (kind from abstract wording only).

The 203 combinatorial and anchor rows were curated from full texts:

| Decision | Rows |
| --- | ---: |
| `keep` (makes SL calls or publishes scores for each line) | 90 |
| `nonhuman` | 31 |
| `duplicate_preprint` or `duplicate_of` | 22 |
| `computational` | 19 |
| `method_only` | 15 |
| `not_sl` | 14 |
| `review` | 6 |
| `reanalysis` (Zamanighomi 2019, Ajmal 2025, Chou 2025; no line rows of their own) | 3 |
| `reclassify_low_throughput` | 3 |

Of the 90 kept screens, 73 have line rows with a ModelID: 330 (study, line) rows over 174 lines, and 55 studies have two
or more lines. Basal scRNA coverage of these 174 lines:

| scRNA status | Lines |
| --- | ---: |
| Training line of the final split | 78 |
| Validation line | 5 |
| Test line | 11 |
| Other split line (PC9, HeLa) | 2 |
| Built extra line or reviewed atlas row (`ok`) | 12 |
| New raw-count source found 2026-10-29 (not reviewed) | 8 |
| Only a weak new source | 1 |
| None found | 57 |

So 116 kept-screen lines have basal scRNA or a raw-count candidate, against 34 lines in the three-screen benchmark. They
form 247 (study, line) lists, and 43 lines have two or more studies.

## Pitfalls

- **Overcounted lines.** For the 65 kept studies outside the 25 hand-read ones, the lines are every human line that the
  full text names at least twice, not the screen panel the paper states. Validation-only lines are therefore included,
  and a line named once can be missing. `match_evidence` records the mention count.
- **ModelIDs are proposals.** They come from hand alias lists. A source's spelling becomes a ModelID only through a
  reviewed row.
- **Lines without scRNA.** After two searches, these screened lines have no public basal scRNA: every uveal melanoma line
  except MP41, every head-and-neck line, most of Flister's extra lines, and lines such as SiHa, MV4-11, RT112, OV90 and
  KYSE150. They cannot be scored until a source exists.
- **Not curated yet.** The drug-anchor, RNAi and low-throughput rows are still `candidate` rows, and 6 kept screens have
  no line row (closed access or truncated text).

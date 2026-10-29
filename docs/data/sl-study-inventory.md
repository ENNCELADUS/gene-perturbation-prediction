# SL Study Inventory

**Status:** discovery inventory built 2026-10-29 for the
[multi-study SL benchmark](../specs/2026-10-29-multi-study-sl-benchmark-design.md). Nothing is downloaded or labelled
yet. The line rows and scRNA rows of the kept screens were reviewed on 2026-10-29 (see Review). The kind on each
`candidate` row is still a proposal. Nothing here is SL evidence.

## Files

| File | Rows | Content |
| --- | ---: | --- |
| `configs/benchmarks/sl_studies.csv` | 2,261 | One row per study: `kind`, `platform`, per-line score, authors' call, data location, licence, `in_synlethdb3`, `in_slkb`, `found_via`, `curation_status`. For a curated row, `note` holds `decision=` |
| `configs/benchmarks/sl_study_lines.csv` | 896 | One row per (study, line): the line as the source spells it, `engineered_genotype`, `proposed_model_id`, `match_evidence`; for kept screens also `model_review`, the reviewed `model_id`, `cellosaurus`, `screened` and `review_evidence` (see Review) |
| `configs/benchmarks/sl_study_tables.csv` | 518 | One row per result file of a kept screen: URL, bytes, SHA-256, H20 path, route, status and a summary of its sheets and columns (see Per-line tables) |
| `configs/benchmarks/single_cell_atlas_candidates.csv` | +41 | The basal scRNA sources found for the screened lines, reviewed by `propose` and then by hand (see Review); the review evidence is at the end of each `note` |

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

## Review (2026-10-29)

One Codex agent (gpt-6.1-sol) checked each line row of the kept screens against Model.csv, Cellosaurus and the
full text, and each new scRNA row against GEO. Claude adjudicated the verdicts and applied them.

- **Line rows** (414 of the kept screens). `model_review`: 325 `correct`, 1 `wrong` (corrected), 71 `not_in_depmap`
  (for example 293T, HeLa S3, generic RPE1), 13 `not_a_line` (text tokens such as author initials, and empty
  placeholders) and 4 `unsure`. `model_id` is set only for `correct` and `wrong` rows. `screened`: 222 `yes`, 105
  `validation_only`, 35 `no` and 52 `unsure`. The hand-read panels of Ito, Flister and CHyMErA were set to `yes`.
- **Lines the table lacked.** The review named 50 screened lines that the table did not have, for example 34 lines of
  Wang 2017's leukaemia panel. They were added with `model_review = added_unreviewed`. A ModelID is proposed only when
  exactly one Model.csv name matches (44 rows). Harle's COLO-792 and WM3702 have `screened = qc_excluded`, because the
  authors removed them in QC.
- **scRNA rows** (41). The `review` values are now 14 `ok`, 9 `ambiguous`, 5 `derived`, 1 `no_match` and 12
  `dropped`. Rows were dropped for 3D or differentiated culture, sorted fractions, knockouts, or values that are not
  counts (RPKM, FPKM, scaled). `condition_class` and `raw_counts` were corrected. HEL in GSE319999 was re-mapped to
  HEL 92.1.7 (ACH-000005) and stays `ambiguous`.

After the review, 156 lines were screened (`screened = yes`) in 65 studies. 93 of them have basal scRNA:

| scRNA status | Lines |
| --- | ---: |
| Training line of the final split | 67 |
| Validation line | 3 |
| Test line | 7 |
| Other split line (PC9, HeLa) | 2 |
| Built extra line or reviewed atlas row (`ok`) | 8 |
| New scRNA row reviewed `ok` | 6 |

These 93 lines form 172 (study, line) lists, and 29 lines have two or more studies. The 116 lines in Counts above
include lines used only for validation.

## Per-line tables (2026-10-29)

The files that hold each kept screen's per-line results (scores, FDR, calls) are on the H20 in
`data/sl_studies/raw/<study_id>/`: 509 files, 2.64 GB. The earlier copies of Harle, Ito and Klingbeil in
`data/sl_screens/raw/` are used as they are. Every tabular supplement was fetched, library and oligo tables included;
the `content` column tells them apart. No file was parsed, labelled or filtered. The SHA-256 of every file on the H20
matches `sl_study_tables.csv`. Files that the H20 could not reach, or reached too slowly, came through the Mac and were
deleted there after the check. 38 files behind the PMC browser check or a publisher's 403 were downloaded by hand in a
browser (`route = manual_browser`).

- **Complete (68 studies):** a per-line score or call table is on the H20. `per_line_score_found` was read from the
  sheet headers, not from the values.
- **Partial (16):** only part of the data is there, for example only the shared SLKB dump (Wong 2016, Tang 2022),
  only BioGRID ORCS files, only read counts, or only PDF tables (Wang 2020, Guo 2019, Shao 2025, Pinault 2025,
  Zhao 2019, Ragupathi 2025, Ramadan 2026). Zhao 2017's supplements hold ChIP-seq and expression tables, not its
  screen. Neggers 2020's screen data is probably in a RAR file that was not opened.
- **None (6):** Magesh 2024 and Park 2026 have raw reads only (39.8 GB and 9.4 GB in SRA, not fetched).
  Boettcher 2017, Doberstein 2018 and Hayward 2026 have no supplement on bioRxiv. Chatr-aryamontri 2025's data site
  gave no file.

Not opened: the SLKB SQL dump (241 MB, shared by its 11 studies), Alborzinia 2023's `.7z`, van 2023's `.xlsb` and
Neggers 2020's RAR. Shen 2017's `MOESM2_ESM.xlsx` is a PDF. Zhao 2018's DOI was corrected to
10.1016/j.molcel.2018.01.017 (PMID 29452643).

## Pitfalls

- **Use `screened` and `model_id`, not `proposed_model_id`.** The proposals include lines used only for validation and
  near-names. The 50 added rows and the 17 rows still `unsure` are not reviewed.
- **Lines without scRNA.** After two searches, these screened lines have no public basal scRNA: every uveal melanoma line
  except MP41, every head-and-neck line, most of Flister's extra lines, and lines such as SiHa, MV4-11, RT112, OV90 and
  KYSE150. They cannot be scored until a source exists.
- **Not curated yet.** The drug-anchor, RNAi and low-throughput rows are still `candidate` rows, and 6 kept screens have
  no line row (closed access or truncated text).

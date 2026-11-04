# Multi-study cell-line SL benchmark: design

**Status:** design agreed 2026-10-29 (interview with the user); discovery inventory built 2026-10-29
; labels and SL split built 2026-10-30 ([data card](../data/sl-benchmark.html)). It replaces the three-screen benchmark of blueprint §5 and SL protocol §10.

## 1. Why a new benchmark

The task (blueprint §3) scores SL ranking per held-out cell line, so the number of lines with labels and basal
scRNA sets how much a result can say. The current benchmark (Ito 2021, Klingbeil 2024, Harle 2025) gives 2
validation and 3 test lines. No public benchmark collects every SL measurement in every line that can be found.

| Benchmark | Evaluates on | Limit for this task |
| --- | --- | --- |
| Cilantro-SL (Hua, Haber and Ma 2026) | SynLethDB 2.0 pairs, 5-fold pair holdout and gene holdout, one pooled AP | No line held out; one label per pair, copied to every line where both genes are among its 2,048 most expressed; the `cell_line` field is not read |
| Feng 2024 | SynLethDB, CV1–CV3; cross-line Jurkat → A549 | The two lines share no pair |
| MiT4SL 2025 | SLKB, leave one of 6 lines out | Screens merged within a line; all 6 lines are training lines here |
| SLWise 2023 | 3 lines, line-to-line transfer | Classes balanced by subsampling; all 3 are training lines here |
| Kebabci 2026 screen set (current) | Within-screen labels across lines, continuous scores | 3 paralog screens; 5 held-out lines |

SL databases do not solve the coverage problem. SynLethDB 3.0 (Zenodo 22843223, 2026-09-19, CC BY 4.0) and SLKB
(updated 2023-05-23, 11 studies) take their data from the same primary screens. SynLethDB 3.0 has no scores. Its
newest screen PMID is from 2022, and many of its rows join several lines or PMIDs in one field. Ito's 11 lines
have 4,962 non-SL rows each and 0 SL rows. The project's `data/SL_Benchmark_Formal/sl_integrated_pairs.csv` has
the same gaps. Its `conflict == 0` filter also removes the 8,866 screen pairs whose labels differ between lines,
which is the signal this task needs. The databases are therefore used as indexes for discovery. They are not
used as label sources.

## 2. Decisions

1. **Source unit.** Each label comes from one record: one study, one line, one pair and one score. SLKB's `rawSL`
   and `calcSL` exports supply this unit for its studies. Primary supplements supply it for all other studies.
   Labels are never merged across studies.
2. **Discovery first, no eligibility rules yet.** Fetch every line of every study, with its attributes (scRNA source,
   GeneEffect, study, pairs tested). Rules come later, after people read the inventory.
3. **Experiment kinds.** Every kind that measures SL is included: combinatorial double-knockout, anchor or isogenic,
   RNAi, drug-anchor and low-throughput validation. Each record stores its kind and platform. Computational
   predictions are excluded. A text-mined row counts only through the experiment its PMID reports.
4. **scRNA.** For each fetched line, look up the project's atlas candidates, then search for new basal scRNA
   sources. Register each source as a `proposed` row in the format of `single_cell_atlas_candidates.csv`. Downloads
   and processing come after the split.
5. **Labels.** Use the authors' call where the study publishes one. Otherwise, use one rule per study, fixed and
   recorded before any model is scored. Every row keeps its continuous score. A pair whose gene is depleted alone
   in a line is unlabelled in that line. A study with positives only cannot form a list, and its pairs are reported
   by name outside the headline.
6. **Metric** (changed 2026-11-04). A *screen* is the set of pairs that one study tested in one line with one
   setup. Its negatives are "tested, not called", not true negatives, so the metrics are positive-unlabelled ones.
   The headline is recall in the top 10%: rank the pairs of each screen, credit each known positive in the top 10%
   (a tie block shares its credit; a random ranking gives 0.10), take the mean over a study's screens, then over a
   line's studies, then over the lines. Only genetic screens with both classes enter it (the rules' `condition`
   column; treatment, gain and group screens are scored and kept in the run files). Report the model minus the
   pair-identity control per line, with a bootstrap interval over lines. AUROC and AUPRC over the positive rate
   are averaged the same way and stay in the run files. Controls: pair identity (the pair's hit rate in the same
   study's training lines), pan-essentiality and a context-ablated model.
7. **Split** (changed 2026-11-04). A split of lines placed by hand (2026-11-04), for stage two only: 17 validation and 16 test lines chosen so that training, validation and test hold about 80%, 10% and 10% of the kept SL pairs, across lineages; every other line trains. Stage one's sides do not bind it. Lines from one patient stay on one side. Each line records its group (`with_geneeffect`: stage one's production fit reads its GeneEffect; `without_geneeffect`: it does not), and results are stratified by it; validation and test each hold 5 lines without GeneEffect. Validation selects the model; test is scored once.
8. **Pairs scored.** The headline scores all pairs of a list. The split of pairs into seen and unseen pairs (seen
   means the pair was tested in a training line) is a later ablation.

## 3. Discovery result

The [data card](../data/sl-benchmark.html) has the files and counts. Of 2,261 studies, 90 combinatorial or
anchor screens are kept after a full-text pass. After the review of 2026-10-29, 156 lines were screened in them, and 93
of these have basal scRNA (10 of them current validation or test lines). They form 172 (study, line) lists. The
three-screen benchmark has 34 lines. scRNA, not SL labels, limits the line count: 63 screened lines have no basal
source.

## 4. Not yet decided

- Eligibility rules for lines and lists (minimum positives and negatives per list). The baselines score every list
  that holds both classes.

Settled on 2026-10-30 ([data card](../data/sl-benchmark.html)): the tables live in
`configs/benchmarks/` (`sl_label_rules.csv`, `cell_line_sl_split.json`); the builders are `build_sl_study_benchmark`
and `build_sl_split`. The split's validation and test lines are the placements in `configs/benchmarks/sl_split_placements.csv` (since 2026-11-04). A row with no
published SL call or cut takes the authors' own cut where the paper states one; a study that publishes no SL pairs, or
no cut that can be applied to its table, is left out: on 2026-11-04 its rule rows were removed from the rules file and its
files from the H20 (git history keeps the rows).

Settled on 2026-11-04 (user decisions; [data card](../data/sl-benchmark.html)):

- Training (stage two): one model scores (basal profile, pair); the screen and the study are not inputs. Binary
  cross-entropy on every genetic two-class screen row, with weight 1 / (lines · studies of the line · screens of the
  study in the line · 2 · rows of the row's class in its screen), scaled to mean 1. Conflicting labels stay as
  separate rows. Validation recall in the top 10% selects the checkpoint.
- A line needs basal scRNA and study labels to be in the benchmark. Measured GeneEffect only supervises stage one.
- A line without a ModelID gets its Cellosaurus accession as its ID, or `SL-<study>-<line>` when Cellosaurus has no
  record of it. The reviewed table `configs/benchmarks/sl_line_identities.csv` records each line's spellings, ID,
  parent, patient group and lineage. Every DepMap join continues to use the ModelID.
- SL lines that have basal scRNA move into the final split, so that preparation reads them: lines with GeneEffect
  into `train`, lines without GeneEffect (lines without a ModelID included) into `train` and `unlabeled_train`.
  All earlier checkpoints and the prepared root's derived parts become unusable. The check that refused a checkpoint
  under another split membership is removed.
- hTERT-RPE1 (`CVCL_4388`) is on the SL train side by hand, with its patient group (`PT-JWRsco`, the six RPE1-ss DepMap
  subclones). The reviewed placement is one row of `configs/benchmarks/sl_split_placements.csv`; the split records it
  under `placed` and in its inputs. The draw of every other line does not change.
- Stage one and stage two are separate. The stage-one benchmark keeps the final split (validation selects, test is
  scored once). Stage one's production fit is one fit of the chosen configuration on every labelled line, with no
  validation set and no cross-fitting (`src/experiments/prior/production.py`). It writes one GeneEffect table for
  every single-cell line, and stage two reads it. If an SL validation or test line without GeneEffect shares a
  patient with labelled lines, the production fit leaves those labelled lines out and records them.
- The SL report gives every row on validation and on test, for all lines, the lines with GeneEffect (the
  screened-line use case: stage one's input is in sample) and the lines without GeneEffect (the new-patient use
  case: stage one's input is out of sample), and the cancer and non-cancer lines, each with its line count. The
  control with measured GeneEffect is scored only on the lines with GeneEffect. Limits: stage two trains mostly on
  in-sample inputs, so the number for the lines without GeneEffect is a low estimate; that group is small and
  mostly non-cancer.
- A screened line that shares a patient with a drawn line goes to the same side; only unscreened lines of that
  patient are excluded (the first build lost HeLa S3 and eHAP under the earlier rule).

Built on 2026-11-04 ([data card](../data/sl-benchmark.html), [baselines](../../results/sl_study_benchmark/summary.html)):
135 lines, 82 with a score (train 50, validation 16, test 16); 5 scored held-out lines have no GeneEffect (validation
2, test 3). The production fit of stage one leaves out HAP1 (the patient of eHAP, an SL test line). On test, pair
identity scores recall 0.465 in the top 10% and stage one 0.268, the same as each gene's mean (0.259).

# Multi-study cell-line SL benchmark: design

**Status:** design agreed 2026-10-29 (interview with the user); discovery inventory built 2026-10-29
([data card](../data/sl-study-inventory.md)). Not built. It will replace the three-screen benchmark of blueprint §5 and SL protocol §10 when built.

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
6. **Metric.** A *list* is the set of pairs that one study tested in one line. For each list, the score is the
   AUPRC of the model's ranking against the study's call, divided by the list's positive rate. The headline is the
   mean over test lines of the mean over each line's lists, so a line that many studies tested counts once.
   Controls: pair identity (the pair's hit rate in the same study's training lines), pan-essentiality and a
   context-ablated model.
7. **Split.** A fixed random split of lines, about 60% training, 20% validation and 20% test, drawn within each
   lineage. Lines from one patient stay on one side. Stage one follows this split, so all stage-one rows are
   scored again on it. A line without GeneEffect can be on any side: stage one predicts its GeneEffect from scRNA.
   Validation selects the model; test is scored once.
8. **Pairs scored.** The headline scores all pairs of a list. The split of pairs into seen and unseen pairs (seen
   means the pair was tested in a training line) is a later ablation.

## 3. Discovery result

The [data card](../data/sl-study-inventory.md) has the files and counts. Of 2,261 studies, 90 combinatorial or
anchor screens are kept after a full-text pass. After the review of 2026-10-29, 156 lines were screened in them, and 93
of these have basal scRNA (10 of them current validation or test lines). They form 172 (study, line) lists. The
three-screen benchmark has 34 lines. scRNA, not SL labels, limits the line count: 63 screened lines have no basal
source.

## 4. Not yet decided

- Eligibility rules for lines and lists (minimum positives and negatives per list).
- The split rule's lineage strata and exact fractions, after the inventory.
- Where tracked inventory tables live, and the builder.

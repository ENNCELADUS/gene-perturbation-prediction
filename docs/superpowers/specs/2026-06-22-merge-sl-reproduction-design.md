# Design: Merge the four-model SL reproduction into the report

Date: 2026-06-22
Status: Approved (design sections), pending spec review
Related: `docs/superpowers/specs/2026-06-22-paper-story-design.md`,
`personal_report_sl_gene_perturbation.md`, `docs/report/neurips_2026.tex`

## 1. Problem

A teammate reproduced four published synthetic-lethality (SL) prediction
methods (DDGCN, GRSMF, SL2MF, SLGNN) under the K562 CV1/CV2/CV3 protocol and
produced an official comparison table (F1 + NDCG@10 per split, plus cross-CV
mean AUROC / AUPR / F1 / NDCG@10). This work is summarized in
`personal_report_sl_gene_perturbation.md`. We need to merge those numbers into
the report's story as **direct-comparison baselines** without violating the
paper's honesty guardrails.

## 2. Verified alignment (why a shared table is valid)

Confirmed against on-disk artifacts before designing:

- **Identical metric code.** Both the vendored published models and our
  `src/sl_benchmark_baseline` call the same `cal_metrics` (per-anchor
  `ndcg_score`, k=10) in `data/SL_benchmark/src/preprocess.py:818`. NDCG@10 is
  the same quantity on both sides.
- **Same protocol.** CV1/CV2/CV3, 1:1 negative sampling, 5-fold aggregation on
  both sides (their Slurm 5-fold array; our 5-fold mean±std).
- **Same candidate universe** (user-confirmed): the teammate trained the four
  models on the derived K562-DepMap-filtered splits
  (`data/SL_benchmark/derived/k562_depmap_rand_1to1/`), so the per-anchor
  candidate gene set matches exp06 exactly. No universe caveat required.
- **Canonical numbers:** the teammate's reported official table (in
  `personal_report_sl_gene_perturbation.md` §6). The in-repo
  `data/SL_benchmark/src/summary_all_matrics.csv` (dep/exp/random feature
  variants) is treated as separate/exploratory and is NOT used.

## 3. Honest framing (Approach 1: orthogonal-signal benchmark)

The merge is framed as **rigor + an honest gap**, never as us winning:

- Published methods are **label-graph** methods: they consume the SL
  association matrix / knowledge graph as input.
- Our methods are **label-free functional-signal** methods (exp06
  dependency-only B; exp07 + observed transcriptome): no SL graph as input.
- On **classification** (AUROC/F1) our functional methods are in the pack
  (exp06-B mean-AUROC ~0.70 vs DDGCN/GRSMF 0.752, SLGNN 0.721).
- On **ranking** (NDCG@10) the label-graph methods clearly lead (GRSMF 0.317
  mean, DDGCN 0.257) vs our functional methods (exp06-B ~0.03, exp07
  transcriptome ~0.09). This holds even at cold-start CV3 — **no cold-start
  win is claimed**.
- Teammate's own insight folded into prose: best classifier (SLGNN) != best
  ranker (GRSMF), so SL evaluation needs both F1 and NDCG@10.

The ranking gap is positioned as exactly what the generative direction (exp08)
targets — consistent with the paper's existing "preliminary, does not beat
baselines" spine.

## 4. The new table (`tab_benchmark`)

A single comparison table, two row-groups separated by a `\cmidrule`:

Columns (mirror the teammate's official schema):
`CV1 F1 | CV1 N@10 | CV2 F1 | CV2 N@10 | CV3 F1 | CV3 N@10 | Mean AUROC |
Mean AUPR | Mean F1 | Mean N@10`.

Row-group A — **Label-graph methods** (consume SL matrix): DDGCN, GRSMF,
SL2MF, SLGNN (verbatim from the teammate official table).

Row-group B — **Functional-signal methods** (label-free): Dependency-only
(exp06 Model B), + transcriptome (exp07 `B_transcript`, `full_universe`
slice).

Style: booktabs (`\toprule/\midrule/\bottomrule`, no vertical rules), caption
above, direction arrows in headers, 3 dp. No "best"-bolding that implies our
methods win; the only emphasis is the field-leading ranker (GRSMF Mean N@10)
to underline the honest gap.

## 5. Single-source data path (same discipline as `foundation_values.csv`)

- New `docs/report/tables/benchmark_published.csv`: the four published rows
  verbatim from the teammate official table, with a `source` provenance
  column crediting the reproduction (internal; PDF stays anonymized).
- New `build_benchmark()` writer in `docs/report/scripts/make_tables.py`:
  - reads `benchmark_published.csv` for the four label-graph rows;
  - reads exp06/exp07 artifacts directly via the existing `read_metric`
    (with its `slice` argument; exp07 uses `slice="full_universe"`);
  - adds a small `mean_over_cvs(values)` helper = simple average of the three
    per-CV means, matching the teammate's "Mean" column definition.
- Registered in the writer dispatch dict and `--all`.

## 6. Tests (gate, extend `tests/test_make_tables.py`)

New anchored assertions:
- GRSMF Mean N@10 == 0.317 (from published CSV).
- SLGNN CV3 N@10 == 0.000 (ranking collapse anchor).
- exp06 Model B mean-over-CV AUROC computed ~ 0.698 (our floor, recomputed,
  tolerance-checked).
- exp07 `B_transcript` CV2 N@10 (`full_universe`) == 0.094 (already anchored;
  reused).
- `tab_benchmark.tex` is generated, booktabs-clean (`\toprule` present, no
  `|`), and contains both row-group labels ("Label-graph", "Functional").

## 7. Placement and surrounding edits

- `sections/experiments.tex`: new subsection **after** the dependency-only
  floor and **before** the proof-of-concept transcriptome subsection
  (Foundation -> Floor -> Benchmark-vs-published -> Transcriptome ->
  Generative-preliminary -> Decomposition). `\input{tables/tab_benchmark}`.
  The existing `tab_transcriptome` stays (it is the controlled within-feature
  ablation; the new table is cross-method context).
- `sections/related_work.tex`: upgrade the "families surveyed by
  \citet{wang2022slreview}" sentence to note we reproduce them under our
  shared K562 protocol (Table~\ref{tab:benchmark}). Stays F7-safe (literature
  methods + our reproduction; no claim they are our contribution).
- `docs/report/references.bib`: add CrossRef-verified original-paper entries
  for DDGCN, GRSMF, SL2MF, SLGNN; `\cite` them in the new table caption /
  prose. (User-approved: add all four.)
- `docs/report/scripts/make_figures.py` + `figures/fig_difficulty_ladder.pdf`:
  overlay GRSMF (best published ranker) as a dashed NDCG@10 reference line, so
  the figure visualizes the gap our methods must close. (User-approved.)
- `docs/superpowers/claim_evidence_map.md`: add a row for the new
  cross-method comparison claim and its evidence (tab_benchmark + provenance
  CSV + teammate reproduction).

## 8. Guardrails honored

- No "outperforms / beats / SOTA / state-of-the-art" language.
- Universe identical (confirmed) -> caption states "same K562-DepMap splits,
  same per-anchor `cal_metrics`"; no apples-to-oranges disclaimer needed.
- exp08 preliminary status untouched.
- Attribution: anonymized PDF cannot name the teammate, but the provenance
  CSV, commit message, and claim-evidence map credit the reproduction.
- All reported numbers flow through `make_tables.py` and are pytest-gated; no
  hand-typed numbers in the `.tex` body.

## 9. Out of scope (YAGNI)

- The `dep`/`exp`/`random` feature-fusion probe (Approach 3) is NOT included;
  provenance of `summary_all_matrics.csv` is unclear and the user chose the
  teammate-official numbers as canonical.
- No re-running of the vendored models; we consume the teammate's reported
  numbers.
- No changes to exp08 experiments.

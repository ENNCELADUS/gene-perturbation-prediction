# Cross-Line Loss and Local Gene-in-Line Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the gene-knowledge, cell-set model's line term with a cross-line ranking loss, give it local features of the two genes in the line, and measure whether the SL score orders held-out lines correctly.

**Architecture:** The model, the runner and the cache format stay. The runner gets a loss between two training lines of one study for the same pair, a joint training mode and a line-head input without line tokens. `gene_in_line.py` gets seven CPU cache steps. The harness gets the cross-line ordering and a mean over several other-line controls.

**Tech Stack:** Python 3.11, PyTorch, pandas, NumPy; `uv`; pytest with synthetic fixtures; H20 for data and GPU.

**Spec:** `docs/specs/2026-11-24-context-specific-sl-architecture-design.md`, §5 and §6. Not in this plan: the row with 80 training lines (§6, last row). The label table holds the 36 bulk-RNA lines with a two-class genetic screen (32 `bulk_labelled`, 4 `bulk_unlabelled`; measured on the H20, 2026-11-24), but `benchmark_rows` drops every line outside train, validation and test, so that row needs its own plan.

## Global Constraints

- Run from the repo root as modules; imports are `src.*`. `src/data` and `src/model` never import `training`, `eval` or `experiments`.
- Fail closed: every new config key is required (`require_keys`); no `.get(key, default)` config reads; no `load_state_dict(..., strict=False)`.
- One config is one seed-0 run: validation recall in the top 10% selects, test is scored once. No code gates a run on a model-quality number.
- Every scaling value, percentile and threshold is fitted on SL training lines only.
- Tests use synthetic fixtures; run one file with `uv run python -m pytest <file> -q > <scratch file> 2>&1` and read the file (the `rtk` hook breaks foreground pytest output). Lint: `.venv/bin/ruff check <files>`; format: `.venv/bin/ruff format <files you touched>` only.
- Baseline before this plan: 840 passed, 22 skipped.
- Names: no bare labels in code comments, docs or report text; say what the thing is. Do not write "pair identity" in new text.
- New docs text is in Simplified Technical English. Do not rewrite text you do not change.
- **Workers do not commit and do not touch files outside their track's list.** The three tracks run at the same time in one working tree; the coordinator commits.

## Review Focus

1. A config with `cross_line` and a training set without any cross-line example: the run must stop with a clear error, not train on an empty draw (Task A2).
2. A line that has both calls for one pair in two screens of a study: it must never form an example with itself (Tasks A2, B1).
3. A gene in the damaging-mutation matrix that the hotspot matrix does not list: its row is present with hotspot 0, not absent (Task C3).
4. A line without a cell file or without DepMap data: it gets a file of NaN rows and the run continues (Tasks C1, C3).
5. `eval.other_lines` equal to or above the number of benchmark lines: an error, and no line ever reads its own input as "another line" (Task A4).

---

## Track A: the model and the runner

Files of this track (one worker, tasks in order):
- Modify: `src/model/sl_embedding.py`, `src/experiments/sl/embedding.py`
- Modify: `configs/sl/embedding.yaml`, `embedding_mean_token.yaml`, `embedding_mean_token_slow_stage_one.yaml`, `embedding_cellnet.yaml`, `embedding_direct.yaml`, `embedding_direct_mean_token.yaml`, `embedding_direct_pie_change.yaml`, `embedding_direct_tx1_gene.yaml`
- Test: `tests/model/test_sl_embedding.py`, `tests/experiments/sl/test_embedding.py`

### Task A1: config keys and the line head's `genes` input

**Interfaces:**
- Produces: `GeneLineModel(sources, expression_dim, cell_dim, hidden, head_hidden, gene_in_line, line_input: str)`; `GeneLineModel.source_vectors() -> Tensor` (genes + 1, width; the gene encoder's output without `deviation`, then a zero row); `GeneLineModel.line_vectors() -> Tensor` (`gene_vectors()` for `line_input == "model"`, `source_vectors()` for `"genes"`).
- Produces config keys: `line_term.joint` (bool), `line_term.input` (`model` or `genes`), `line_term.cross_line` (`null` or `{weight, examples_per_step}`), `eval.other_lines` (int ≥ 1).

- [ ] **Step 1: Write the failing tests** in `tests/model/test_sl_embedding.py`:
  - `test_the_genes_input_reads_no_line_token_and_no_gene_deviation`: with `line_input="genes"` and a trained (non-zero) last layer of `line_head`, `line_term` gives the same values when the `mean` token changes and when `model.deviation` changes; it changes when a gene-in-line feature changes.
  - `test_the_genes_input_is_the_same_for_both_orders_and_reads_a_zero_anchor`: `line_term(u_a, u_b, ...) == line_term(u_b, u_a, ...)` on combinatorial rows; a row with the zero row of `line_vectors()` as gene A is finite.
  - `test_an_unknown_line_input_is_refused`: `GeneLineModel(..., line_input="cells")` raises `ValueError`.
- [ ] **Step 2: Run them; expect failures** (`TypeError: unexpected keyword 'line_input'`).
- [ ] **Step 3: Implement in `src/model/sl_embedding.py`.** `line_head` input width is `7 * WIDTH + 1` for `model` and `6 * WIDTH + 1` for `genes`. For `genes`, `line_term` builds `[u_a + u_b ; |u_a − u_b| ; u_a ⊙ u_b ; combinatorial ; e_a + e_b ; |e_a − e_b| ; e_a ⊙ e_b]` from the vectors it is given and ignores `mean`, `cells`, `mask`. The caller passes rows of `line_vectors()`. Update the module docstring.
- [ ] **Step 4: Implement in `src/experiments/sl/embedding.py`.** Add the three keys to `LINE_TERM` and `other_lines` to `SECTIONS["eval"]`; `load_config` checks `input` in `{"model", "genes"}`, `cross_line` is `None` or has exactly `weight` and `examples_per_step`, `other_lines >= 1`. `build_model` passes `line_input` (`"model"` when `line_term` is `None`). `line_term_of` takes the line head's gene vectors as a new argument `line_u: Tensor` (rows of `model.line_vectors()`, computed once by each caller: `line_score`, `line_term_scores`, `line_term_loss`).
- [ ] **Step 5: Update the eight configs.** Every file: `eval.other_lines: 1`. The four with a `line_term` section: `joint: false`, `input: model`, `cross_line: null`. Add a test `test_a_line_term_without_the_new_keys_is_refused` beside `test_the_config_loads_and_a_missing_key_is_refused`.
- [ ] **Step 6: Run both test files; expect all pass** (the old line-term tests must pass unchanged with `input: model`).

### Task A2: the cross-line loss with the model frozen

**Interfaces:**
- Produces: `cross_line_examples(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]`; `PairRows.cross: list[tuple[Tensor, Tensor]]` (CPU, long); `cross_line_loss(model, data, pairs, config, generator, train_f: bool) -> Tensor`; `line_score(..., train_f: bool = False)`; `line_term_loss(..., train_f: bool = False)`.

- [ ] **Step 1: Write the failing tests** in `tests/experiments/sl/test_embedding.py`:
  - `test_cross_line_examples_pair_two_lines_of_one_study_for_one_pair`: on a hand-made frame, every example is (row with label 1, row with label 0) of the same `study_id`, `design`, `gene_a`, `gene_b` and different `model_id`; rows outside `training_rows` never appear; a line with both labels for one pair in two screens gives no example with itself; a study without an example has no entry.
  - `test_a_score_that_ignores_the_line_gives_log_two`: on the untrained model (its token maps and line head end in zero layers), `cross_line_loss(...)` equals `math.log(2)` within `1e-6`.
  - `test_the_cross_line_loss_trains_only_the_line_term_when_the_model_is_frozen`: after `fit_line_term` with `cross_line: {weight: 0.5, examples_per_step: 32}`, the pair head and the backbone are bit-equal to before and the line head is not.
  - `test_cross_line_without_an_example_is_refused`: a world whose training lines never differ on a pair, with `cross_line` set, raises `ValueError` that names `cross_line`.
- [ ] **Step 2: Run them; expect failures** (`AttributeError: cross_line_examples`).
- [ ] **Step 3: Implement.**
  - `cross_line_examples`: over the rows of `training_rows(rows)`, grouped by `["study_id", "design", "gene_a", "gene_b"]`; one entry for each study with an example, studies in sorted order; arrays of row positions `(positive, negative)`.
  - `cross_line_loss`: draw `examples_per_step` examples, each with a study drawn evenly and then an example drawn evenly in it. For each distinct line among the drawn rows, draw its tokens once (`one_line` with `cells.per_step` and `cells.dropout`) and compute `f + D` for its rows (`line_score`). Return the mean of `softplus(−(g⁺ − g⁻))`. The mean over the reference lines is not computed: it cancels.
  - `line_score(..., train_f)`: `f` under `torch.no_grad()` only when `train_f` is false.
  - `line_term_loss`: when `config["line_term"]["cross_line"]` is set, add `weight * cross_line_loss(...)`; pass `train_f` through.
  - `pair_rows` raises the error of Review Focus 1 when `cross_line` is set and no study has an example. `pair_rows` takes `config` for this (update its callers).
- [ ] **Step 4: Run the file; expect all pass.**

### Task A3: joint training

**Interfaces:**
- Consumes: `line_term_loss(..., train_f=True)`, `line_term_scores`.
- Produces: `fit_pairs(model, data, pairs, config, generator, geneeffect_weight, with_line_term: bool = False)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_joint_training_moves_the_model_and_the_line_term_together`: with `line_term.joint: true`, one whole-model phase changes the backbone, the pair head and the line head.
  - `test_joint_training_has_no_separate_line_term_phase`: `train(...)` with `joint: true` returns `result["line_term"]` whose `history` is the chosen phase's history and whose `scores` equal `line_term_scores` of the returned model; `fit_line_term` is not called (patch it to raise).
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement.** In `train`, when the config has a line term with `joint: true`: the frozen phase is as now (the untrained line term is 0, so `s = f`); each whole-model phase calls `fit_pairs(..., with_line_term=True)`. There the step loss is `line_term_loss(..., train_f=True)` plus the GeneEffect loss at its weight; the optimizer has the pair head and `model.line_parameters()` at `stage_two.learning_rate` and the backbone at `backbone_learning_rate`; validation scores are `line_term_scores`. After the phase is chosen, `result["line_term"]` is built from the chosen state without `fit_line_term`. With `joint: false` nothing changes. Update the module docstring (the line term no longer always trains last).
- [ ] **Step 4: Run the file; expect all pass.**

### Task A4: the other-line control over several lines

**Interfaces:**
- Produces: `other_lines(data, config) -> list[dict[int, int]]` (length `eval.other_lines`; entry `k` maps each benchmark line to the line `k + 1` places before it in the seeded order). Control names: `other_line`, `other_line_2`, …, `other_line_<n>`; with a line term also `line_term_other_line`, `line_term_other_line_2`, …. Score files: `scores/embedding_<name>.parquet`.

- [ ] **Step 1: Write the failing tests:**
  - `test_each_other_line_is_another_line_and_the_first_is_the_old_control`: with `other_lines: 3`, no map sends a line to itself, the three maps differ for every line, and the first equals `np.roll(order, 1)` of the seeded order.
  - `test_more_other_lines_than_benchmark_lines_is_refused`: `other_lines` equal to the number of benchmark lines raises `ValueError`.
  - `test_a_run_writes_one_table_for_each_other_line`: `line_input_controls` returns `other_line` and `other_line_2`; `train` returns `line_term` entries for each.
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement** in `other_lines`, `line_input_controls`, `train`, `rescore` and `run` (every control goes through `controls_named` and `write_scores`; `sl_line_term` keeps `line_term_other_line`, the first). Table 2 of `results_markdown` does not change.
- [ ] **Step 4: Run both test files of the track; expect all pass. Run `.venv/bin/ruff check` and `ruff format` on the four touched Python files.**

---

## Track B: the harness

Files of this track (one worker):
- Modify: `src/experiments/sl/benchmark_baselines.py`; every `configs/sl/benchmark_*.yaml` (eight files: add the two new keys, empty)
- Test: `tests/experiments/sl/test_benchmark_baselines.py`

### Task B1: the cross-line ordering

**Interfaces:**
- Produces: `cross_line_ordering(frame: pd.DataFrame, score: np.ndarray, reps: int, seed: int) -> dict[str, dict[str, Any]]`, keyed by split. Each value: `{"value": float, "interval": [low, high], "examples": int, "pairs": int, "studies": int, "by_study": {study_id: {"value", "examples", "pairs"}}}`. A split without an example has no key.
- Produces config key `cross_line_ordering: [scorer, ...]` (required; may be empty) and `metrics["cross_line_ordering"] = {scorer: {split: ...}}`.

- [ ] **Step 1: Write the failing tests:**
  - `test_a_score_that_is_the_same_in_every_line_orders_at_one_half`: a score that depends on the pair alone gives `value == 0.5` exactly; the labels as the score give `1.0`; the negated labels give `0.0`.
  - `test_cross_line_ordering_pairs_lines_of_one_side_study_and_pair`: lines of validation never pair with lines of test or training; lines of two studies never pair; a (line, pair) cell whose rows hold both labels is dropped; `examples` is the sum over pairs of positive lines times negative lines; a tie counts one half.
  - `test_the_study_mean_weighs_each_study_once`: two studies with 1 and 100 examples and values 1.0 and 0.0 give `0.5`.
  - `test_the_harness_reports_the_ordering_for_the_named_scorers`: a run of the module's synthetic harness fixture with `cross_line_ordering: [labels]` writes `metrics["cross_line_ordering"]["labels"]` with `val`, `test` and `train`, and a table in `results.md`.
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement.**
  - Rows: `condition == HEADLINE` and a finite score. One value for each `(split, study_id, design, gene_a, gene_b, model_id)`: the mean score; drop the cell when its rows hold both labels.
  - A pair group is `(split, study_id, design, gene_a, gene_b)` with a positive line and a negative line. Its wins are the count of (positive line, negative line) with the higher positive score, plus half the ties.
  - A study's value is its wins over its examples; a split's value is the mean over its studies. The interval is the 2.5% and 97.5% quantiles over `reps` draws that resample each study's pair groups with replacement (`np.random.default_rng(seed)`).
  - In `run`: for each named scorer, call it on the held rows and on the training rows (when `scorers_for(training, strict=False)` has the scorer). `load_config` refuses a name that is not in `NAMES`.
  - `results_markdown`: one table, "Cross-line ordering", models as rows; columns training, validation, test, each `value [low, high] (examples, studies)`; caption: "Share of (line that called the pair SL, line that did not) pairs of one study and gene pair that the score orders correctly, mean over studies; 0.5 for a score that is the same in every line; training lines are in sample."
- [ ] **Step 4: Run the file; expect all pass.**

### Task B2: the mean of several scorers

**Interfaces:**
- Produces: `with_scorer_means(lines: pd.DataFrame, means: Mapping[str, Sequence[str]]) -> pd.DataFrame`; config key `scorer_means: {name: [scorer, ...]}` (required; may be empty).

- [ ] **Step 1: Write the failing tests:**
  - `test_a_scorer_mean_is_the_mean_of_its_scorers_per_line`: for a line with recall 0.2 and 0.4 under two scorers, the derived scorer has 0.3 (the same for `auroc` and `auprc_lift`); `positives`, `studies` and `screens` come from the first listed scorer; a line that one of the scorers lacks has no derived row.
  - `test_a_paired_difference_can_name_a_scorer_mean`: the harness fixture with `scorer_means: {m: [a, b]}` and `paired_differences: [[a, m]]` reports that difference.
  - `test_a_scorer_mean_of_an_unknown_scorer_is_refused`: `load_config` raises when the derived name is not in `NAMES` or a listed scorer is not an `extra_scores` key.
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement.** In `run`, apply `with_scorer_means` to each stratum's line records right after `line_records(...)` and to the training lines' records, before `summary` and `paired_differences`. Add the two keys, empty, to the eight harness configs.
- [ ] **Step 4: Run the file, then `tests/experiments/sl` whole; expect all pass. Run ruff check and format on the touched Python files.**

---

## Track C: the gene-in-line caches

Files of this track (one worker):
- Modify: `src/experiments/sl/gene_in_line.py`, `configs/sl/gene_in_line.yaml`, `hpc/run.sh` (the usage line of `sl-gene-in-line`), `docs/data/gene-in-line-features.md`
- Test: `tests/experiments/sl/test_gene_in_line.py`

Facts measured on the H20 (2026-11-24); the tests build small files of the same layout:
- `data/geneeffect_joint/v2/ladder/cells/genes.json` is `{"genes": [...]}`, 19,277 source genes; `cells/<ModelID>.npz` holds `pseudobulk`, `detected`, `sd`, `detected_expected`, `sd_expected` (each `(19277,)`, float32). `pseudobulk` is `log1p` of the mean CP10K.
- `OmicsSomaticMutationsMatrixDamaging.csv` and `OmicsSomaticMutationsMatrixHotspot.csv`: one row for each sequencing; columns `SequencingID`, `ModelID`, `ModelConditionID`, `IsDefaultEntryForModel` (`Yes` / `No`), `IsDefaultEntryForMC`, then genes as `SYMBOL (entrez)`; values 0, 1, 2. The hotspot matrix has 555 gene columns, the damaging matrix 19,579.
- `PortalOmicsCNGeneLog2.csv`: first column the ModelID, then genes as `SYMBOL (entrez)`; the value is `log2(relative copy number + 1)` (median 1.0).
- All three are in `data/sl_dependency_v0/raw/depmap/`.

New config keys of `configs/sl/gene_in_line.yaml` (all required): `ladder: data/geneeffect_joint/v2/ladder`, `linear_model: configs/linear_model/default_linear_model.yaml` (its reference tables give the paralogs and complexes), `depmap: {damaging: ..., hotspot: ..., copy_number: ...}`.

Rules for every new step: the gene axis and the lines are those of `axis_and_lines` (the upper-cased symbols of the label table; every split line with a labelled row). Each step writes `<out>/<step>/genes.json` (`open_cache`), one `<ModelID>.npy` for every line (float16 through `as_float16`; a NaN row where the gene has no value; a file of NaN rows for a line without the source) and `report.json` (lines, lines with the source, genes with a value in at least one line). A step writes every line again on each call. The SL training lines of a step are the lines of the split's `train` side that have the source.

### Task C1: `wide_expression` and `cell_spread`

**Interfaces:**
- Produces: `on_axis(values: np.ndarray, source: Sequence[str], genes: Sequence[str]) -> np.ndarray` (source genes × width → axis genes × width; upper-cased match, first occurrence, NaN rows elsewhere); `low_threshold(pseudobulk: np.ndarray, train: np.ndarray) -> np.ndarray` (the 10th percentile of each source gene over the training rows); `read_cell_files(ladder: Path, lines: Sequence[str]) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]` (the source genes; for each line with a file, its five arrays); steps `wide_expression`, `cell_spread` in `STEPS`.

- [ ] **Step 1: Write the failing tests:**
  - `test_on_axis_matches_upper_cased_symbols_and_leaves_nan_rows`.
  - `test_wide_expression_holds_the_value_and_the_silenced_flag`: column 0 is `pseudobulk`; column 1 is 1.0 where the value is at or below the gene's 10th percentile over the training lines, else 0.0; the percentile ignores a validation line with an extreme value.
  - `test_cell_spread_holds_the_detected_share_and_the_excess_sd`: columns `detected` and `sd − sd_expected`.
  - `test_a_line_without_a_cell_file_gets_nan_rows` (Review Focus 4): the file exists, every row is NaN, and the report counts the line as without the source.
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement** the two steps, `load_config` (new keys) and `main` (`--steps` choices from `STEPS`; a dispatch dict in place of the `if`/`else`). Update `test_the_config_loads_and_an_unknown_step_is_refused` for the new keys.
- [ ] **Step 4: Run the file; expect all pass.**

### Task C2: the buffer caches

**Interfaces:**
- Consumes: `read_cell_files`, `on_axis`, `low_threshold`; `partner_index(genes, space, reference)` and `knowledge_features(expression, index, low_threshold)` of `src/linear_model/gene_features.py` (columns: own, paralog minimum, paralog sum, closest paralog, complex mean, share of low partners).
- Produces: steps `paralog_buffer` (paralog minimum, paralog sum, closest paralog), `complex_buffer` (complex mean), `partner_low` (share of low partners).

- [ ] **Step 1: Write the failing tests:**
  - `test_the_buffer_caches_hold_the_partner_columns`: with a hand-made reference (gene A has paralogs B and C; A and D share a complex), `paralog_buffer` of A holds `min`, `sum` and the first-listed paralog's value of the line's wide expression; `complex_buffer` of A holds D's value; `partner_low` holds the share of A's partners at or below their training 10th percentile.
  - `test_a_gene_without_a_paralog_is_absent_from_the_paralog_cache_only`: its `paralog_buffer` row is NaN and its `complex_buffer` row is finite.
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement.** The expression is the lines × source genes matrix of `pseudobulk` (lines with a cell file); the space is the upper-cased source genes; the target genes are the axis genes; the reference is loaded as the linear model loads it from `config["linear_model"]` (find the loader that `src/linear_model/linear_model.py` uses; upper-case the reference symbols if the tables are not upper-case). One call of `knowledge_features` serves the three steps.
- [ ] **Step 4: Run the file; expect all pass.**

### Task C3: `depmap_mutation` and `depmap_copy_number`

**Interfaces:**
- Produces: `read_mutation_matrix(path: Path) -> pd.DataFrame` (ModelID × upper-cased symbol; rows with `IsDefaultEntryForModel == "Yes"`; symbols through `parse_gene_symbol`, the first column kept when two columns give one symbol); `read_copy_number(path: Path) -> pd.DataFrame` (the same axes); steps `depmap_mutation` (damaging, hotspot) and `depmap_copy_number` (the stored value; deep-loss flag).

- [ ] **Step 1: Write the failing tests:**
  - `test_the_mutation_matrix_keeps_the_default_entry_of_each_line`.
  - `test_a_gene_outside_the_hotspot_matrix_has_hotspot_zero` (Review Focus 3): a gene in the damaging matrix and not in the hotspot matrix has a finite row `[damaging, 0.0]`; a gene in neither is a NaN row.
  - `test_copy_number_holds_the_value_and_the_deep_loss_flag`: the flag is 1.0 where `2 ** value − 1 < 2 ** −0.8` (a log2 copy-number ratio below −0.8), else 0.0.
  - `test_a_line_without_depmap_data_gets_nan_rows` (Review Focus 4): a line that the matrices do not hold (`CVCL_0058`).
- [ ] **Step 2: Run them; expect failures.**
- [ ] **Step 3: Implement** the two readers and the two steps.
- [ ] **Step 4: Update the docs.** `docs/data/gene-in-line-features.md`: one row or paragraph for each new cache (values, source, when absent, the command). `hpc/run.sh`: the usage line lists the nine steps. `configs/sl/gene_in_line.yaml`: the new keys with one comment line each and the commands in the header comment.
- [ ] **Step 5: Run the file; expect all pass. Run ruff check and format on the touched Python files.**

---

## Task D: the study's configs, names and docs (coordinator, after the three tracks)

**Files:**
- Create: `configs/sl/cross_line_frozen.yaml`, `cross_line_joint.yaml`, `cross_line_genes_input.yaml`, `cross_line_wide.yaml`, `cross_line_depmap_dna.yaml`, `configs/sl/benchmark_cross_line.yaml`
- Modify: `configs/sl/embedding_direct_mean_token.yaml` (`eval.other_lines: 4`, for the rescore of the reference run), `src/experiments/sl/benchmark_baselines.py` (`NAMES`), `tests/experiments/sl/test_embedding.py` (the study's configs differ only where this plan says), `docs/04-sl-ranking-protocol.md` (§20), `hpc/README.md`, `CLAUDE.md`, `README.md`, the spec (§5.1 and §5.3: where the counts and the training recall are reported)

Each run config is `embedding_direct_mean_token.yaml` with these differences, cumulative down the list:

| Config | Difference |
| --- | --- |
| `cross_line_frozen` | `line_term.cross_line: {weight: 0.5, examples_per_step: 256}`; `eval.other_lines: 4` |
| `cross_line_joint` | `line_term.joint: true` |
| `cross_line_genes_input` | `line_term.input: genes` |
| `cross_line_wide` | `line_term.own_expression: false`; `gene_in_line`: `wide_expression`, `cell_spread`, `paralog_buffer`, `complex_buffer`, `partner_low` under `data/sl_gene_in_line/v1/` |
| `cross_line_depmap_dna` | `gene_in_line` also `depmap_mutation`, `depmap_copy_number` |

Harness scorers of a run `<run>`: `<run>` (the score with the line term), `<run>_model` (the pair head alone), `<run>_mean_input`, `<run>_other_line_1` … `_4`, and the derived `<run>_other_lines` (their mean). `cross_line_ordering` names every `<run>`, every `<run>_other_line_1`, `direct_mean_token_line_term` and `labels`. `paired_differences`: each `<run>` against `<run>_other_lines`, against `<run>_mean_input`, against the row above it, and against `kebabci_paralog_predicted`.

- [ ] Write the configs, the `NAMES` entries and the config test; run the full suite and ruff; commit; push; pull on the H20.

## Launch (H20; caches on port 30838, runs on port 30757)

```bash
# caches, CPU, one call for each step
for step in wide_expression cell_spread paralog_buffer complex_buffer partner_low depmap_mutation depmap_copy_number; do
  hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps $step
done
# the reference run's four other-line controls (no training)
hpc/run.sh sl-embedding configs/sl/embedding_direct_mean_token.yaml --run-id embedding_direct_mean_token_seed0_v2 --line-input-controls
# the five runs, one GPU each
hpc/run.sh sl-embedding configs/sl/<config>.yaml --run-id <config>_seed0
# the harness
hpc/run.sh sl-baselines configs/sl/benchmark_cross_line.yaml --run-id benchmark_cross_line_seed0
```

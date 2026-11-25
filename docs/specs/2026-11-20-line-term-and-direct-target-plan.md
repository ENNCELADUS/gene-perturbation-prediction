# The Line Term and the Direct GeneEffect Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SL score of the gene-knowledge, cell-set model change with the cell line, and measure what the
line's single cells add to that change in unseen lines.

**Architecture:** Two changes to the model of SL protocol §18. Stage one trains on GeneEffect itself, not on its
difference from the training gene mean, so the untrained model can no longer be the checkpoint that validation
selects. The SL score gets a line term: a second head whose output is a deviation from its mean over the SL training
lines, added to the model's own score in the line. The line term reads gene-in-line features from caches: the gene's
own expression, Tx1-3B's contextual gene embedding, and PIE's predicted change after the gene's loss.

**Tech Stack:** Python 3.11–3.12, PyTorch, pandas, `uv`, pytest, ruff; Tx1-3B (`tahoe_x1`) and PIE outputs on the H20.

**Spec:** §1 to §7 of this file (the study design, from the discussion of 2026-11-20). The model it changes is
[SL protocol §18](../04-sl-ranking-protocol.md) and the [plan of 2026-11-19](2026-11-19-embedding-transfer-plan.md);
its results are in the [run record](../../results/sl_embedding_seed0/summary.html).

Date: 2026-11-20. Status: run on 2026-11-20 (SL protocol §19, [record](../../results/sl_line_term_seed0/summary.html)): the
direct target gives stage two trained weights and the score now changes with the line; the gain over the residual
target holds on validation only; the line term and the cell tokens add nothing. The line term's base was changed
after the first runs (§3).

## 1. Question

Do a line's single cells make the SL score change with the line, and does that change hold in unseen lines?

## 2. Why

Findings of the runs of 2026-11-19 and 2026-11-20 (run record, Tables 2, 6, 7, 8):

- **The model does not use the line.** With the fit lines' mean input in place of the line's own, the full model
  scores 0.509 / 0.498 against 0.500 / 0.485 (validation / test).
- **The cell tokens add nothing.** Tx1-3B cell tokens minus no cell tokens: −0.008 [−0.029, +0.015] /
  −0.016 [−0.052, +0.018]. CellNet's cell tokens give the same result, so the encoder is not the cause.
- **The pretrained weights never reached stage two.** The early stop kept the model with the lowest GeneEffect loss on
  the SL training lines. That was the untrained model in all four runs (1.0503 before training, higher after every
  epoch). Its line tokens are zero, so the head on the frozen model read a constant and scored 0.1092.
- **The cause is the residual target.** The target was the difference from the training gene mean, and the untrained
  model predicts exactly that mean. Any training error then puts a trained epoch above the untrained model. Stage-one
  accuracy is no goal (blueprint §5): a predicted GeneEffect that is not accurate can still train a representation
  that helps SL.
- **The GeneEffect loss alone does not make the score line-specific.** It was in stage two at weight 1, and the model
  still did not read the line. The pair's identity can fill the whole score.
- **Only the cell-level output of the foundation model was used.** `src/model/tx1.py` sets
  `return_gene_embeddings = False`. The foundation model is frozen and its cell vectors are cached.

## 3. The two changes

**Stage one trains on GeneEffect itself.** The target is `(GeneEffect − c) / s`, with one centre `c` and one scale `s`
for every gene, fitted on the fit lines. The model must then learn how essential each gene is, as well as how a gene
differs between lines. The bias of the mean-token map gives the model a term that is the same in every line, so no
new module is needed. The early stop does not change: the kept checkpoint has the lowest loss on the SL training
lines, which stage one does not fit. With this target the untrained model predicts one value for every gene. Its loss
is then far above that of a trained epoch (the gene mean alone gives a whole-line Pearson of about 0.92), so it cannot
be kept, and stage two starts from trained weights without a special rule. No fixed list of pretraining lengths is
used. The residual target is removed from the code. Its rows are the finished runs of 2026-11-19, in which stage two
started from the untrained model.

**The line term.** The SL score of a row (line `c`, pair A, B) becomes

```text
s(c, A, B) = f(c, A, B) + D(c, A, B) − mean over r in R of D(r, A, B)
```

- `f(c, A, B)`: the model's own score, the pair head of the chosen phase read in line `c`. At a training step `f` and
  `D` of the row's line read the same drawn tokens; at scoring, all the line's cells.
- `D(r, A, B)`: the line head, a second MLP. It reads the pair features of the pair in line `r` (the same features the
  pair head reads) and the gene-in-line features of A and B in line `r` (§4).
- `R`: the SL training lines with training rows (genetic screens with both classes). At a training step `R` is a
  draw of `line_term.reference_lines` of them.
- The line term trains last, with binary cross-entropy on `s`. Only the line head and its input projections train, so
  `f` stays fixed.
- The line term is a deviation from the training lines' mean, so the gene pair alone cannot fill it. The line's gain
  is `s − f`.
- Changed on 2026-11-20 after the first runs: the base was the pair score on the mean input; that score is weak for a
  model that reads the line (recall in the top 10% of 0.466 / 0.430 against 0.553 / 0.496 for the model with Tx1-3B
  cell tokens), and the line term on it scored 0.288 / 0.279.

## 4. Gene-in-line features

Each source gives one vector for each (line, gene). A linear projection takes it to 32 wide, the sources are summed,
and the line head reads `[e_A + e_B ; |e_A − e_B| ; e_A ⊙ e_B]`.

| Source | Vector of gene `g` in line `c` | Data |
| --- | --- | --- |
| Own expression | the line's scaled bridged pseudo-bulk value of `g` | built at load time |
| Tx1-3B gene level | the mean over the line's cells of Tx1-3B's contextual embedding of `g` (2,560 wide), and the share of the cells whose input sequence holds `g` | new cache, one Tx1-3B pass |
| PIE change | PIE's predicted change of the line's profile after the loss of `g`, in 64 components | new cache from `data/sl_pie/v1/pie/change/` (no GPU) |

Every cache has one format: `<cache>/genes.json` (gene symbols) and `<cache>/<ModelID>.npy` (genes × width, float16, a
NaN row where the line has no value). Each column is centred and scaled on the SL training lines. A gene without a
value reads zeros and a presence flag of 0.

## 5. Rows of the study

One config each, seed 0, run id `<config name>_seed0_v2` (the `_seed0` runs of 2026-11-20 used the old base, the pair
score on the mean input). Every run writes five score tables: the model, its two line-input controls, the model with
the line term, and the model and the line term on another line's input.

| Config (`configs/sl/`) | Cell tokens | Gene-in-line features |
| --- | --- | --- |
| `embedding_direct.yaml` | Tx1-3B | own expression |
| `embedding_direct_mean_token.yaml` | none | own expression |
| `embedding_direct_pie_change.yaml` | Tx1-3B | own expression, PIE change |
| `embedding_direct_tx1_gene.yaml` | Tx1-3B | own expression, Tx1-3B gene level |

The rows with the residual target are the finished runs `embedding_seed0`, `embedding_mean_token_seed0` and
`embedding_cellnet_seed0`; they are not run again.

## 6. Measures

- Recall in the top 10%, AUROC and AUPRC over the positive rate, on all pairs, validation and test. Validation
  selects; test is scored once.
- The line's gain: the model with the line term minus the model without it (`f`), minus the context-ablated model,
  and minus itself on another line's input.
- What the pretraining on the direct target adds: `embedding_direct` and `embedding_direct_mean_token` minus the
  finished runs of 2026-11-19 with the same inputs.
- What the cells add: a row with Tx1-3B cell tokens minus the same row without cell tokens; the Tx1-3B gene-level row
  and the PIE row minus `embedding_direct`.
- The same differences on the harness's row sets: pairs on which the training lines differ, pairs that no training
  line called SL, new pairs.
- The stage-one numbers of each model, after the SL tables.

## 7. Limits, and what is not in this plan

- **Not in this plan.** CellNet's gene-level outputs (they enter later through a cache of the same format, if its
  interface gives them), in-silico deletion with Tx1-3B, and adapter training of Tx1-3B. Each needs a result of this
  plan first. No other foundation model is added (decision of 2026-11-20).
- **The direct target is mostly the gene mean.** The line-specific part gets a smaller share of the gradient, and the
  early stop mostly follows the fit of the gene means.
- **Power.** The row set "training lines differ" has 13 / 10 held-out lines. A gain below about 0.05 will not show.
- **The labels.** The line-specific part repeats at 0.35 to 0.39 inside Harle 2025 and at 0.04 to 0.12 between two
  studies (SL protocol §15, §16).
- **Pretraining exposure.** PIE was trained on 42 of the project's lines. A549 is one of them; it is a training line
  of the SL split. Which SL validation or test lines are among the 42 is not recorded. Tx1-3B's exposure to
  our lines is not recorded.
- **Old checkpoints.** A `best.pt` written before this change has no line head and does not load in the new code.
  The finished runs keep their score tables; use their commit to rescore them.

## Global Constraints

- Python 3.11–3.12 under `uv`; run modules with `-m` from the repo root; imports are `src.*`.
- Baseline first: `uv run python -m pytest tests -q > pytest.txt 2>&1` before the first edit. Always redirect pytest
  to a file (the `rtk` hook breaks foreground pytest).
- Lint: `.venv/bin/ruff check .` and `.venv/bin/ruff format <files you touched>`; never `ruff format .`.
- Config reads fail closed: `require_keys`, no `.get(key, default)`. No bare `load_state_dict(..., strict=False)`.
- GeneEffect is a target only, never a model input.
- Every centre, scale and component is fitted on training-side lines only (the fit lines for stage one, the SL
  training lines for the gene-in-line features).
- Validation selects the epoch and the phase; test is scored once. One config is one experiment at seed 0.
- Experiment code reports and blocks no run on a model-quality number.
- Name every model, head and run by what it is in docs, reports and scorer names.
- New docs text is in Simplified Technical English. Commits use Conventional Commits.
- Large files stay on the H20. Code moves by Git only (`hpc/pull.sh`).

## Review Focus

1. A benchmark line without a file in a gene-in-line cache: the run stops with `FileNotFoundError`; it does not fill
   zeros (Task 3, `read_in_line` test).
2. A gene that a cache has no value for (not on its axis, or a NaN row because no cell's sequence holds its token):
   zeros with presence 0, and the count of covered genes in `counts` (Task 3 test).
3. An anchor row whose gene A has no gene vector (`a = −1`): the line term reads the zero feature row and gives a
   finite score (Task 2 and Task 4 tests use `-1`).
4. A line without cells, as the row's line or as a reference line beside lines with cells: the line term gives a
   finite score (Task 2 test with `None` cells; Task 4 test with no cells).
5. A gene without a GeneEffect value in some fit lines (NaN): the target's centre and scale ignore the missing
   values (Task 1 test).

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `src/model/sl_embedding.py` | the model | `features`, `line_term`, `line_parameters`; `backbone` leaves the line term out |
| `src/experiments/sl/embedding.py` | the experiment | the direct target, gene-in-line features, the line-term phase, its scores and report |
| `src/model/tx1.py` | Tx1-3B loading | gene-level output: `gene_sums`, `build_tx1_gene_encoder` |
| `src/experiments/sl/gene_in_line.py` | new: writes the two gene-in-line caches | create |
| `src/experiments/sl/benchmark_baselines.py` | the harness | scorer names in `NAMES` |
| `hpc/run.sh`, `hpc/README.md` | commands | `sl-gene-in-line`; launch blocks |
| `configs/sl/` | configs | four study configs, `gene_in_line.yaml`, `benchmark_line_term.yaml`; the four finished configs get the new keys |
| `tests/model/test_sl_embedding.py`, `tests/model/test_tx1.py`, `tests/experiments/sl/test_embedding.py`, `tests/experiments/sl/test_gene_in_line.py` | tests | extend; the last one is new |
| `docs/04-sl-ranking-protocol.md`, `docs/data/gene-in-line-features.md`, `src/README.md`, `CLAUDE.md`, `README.md`, `model-status.html` | docs | new protocol section §19, a new data card, status |

---

### Task 1: Stage one trains on GeneEffect itself

**Files:**
- Modify: `src/experiments/sl/embedding.py` (`Data`, `load_data`, `geneeffect_table`, `results_markdown`, the module docstring)
- Modify: the four finished configs (`embedding.yaml`, `embedding_mean_token.yaml`, `embedding_cellnet.yaml`,
  `embedding_mean_token_slow_stage_one.yaml`): one comment line each
- Test: `tests/experiments/sl/test_embedding.py`

**Interfaces:**
- Produces: `target_scale(labelled: pd.DataFrame) -> tuple[float, float]`: the mean and the SD (`ddof=0`) over every
  finite GeneEffect value of the fit lines (lines × genes).
- Produces: `Data.centre: float` and `Data.scale: float` replace `Data.sigma_g`. `Data.m_g` stays the fit lines' gene
  mean (the report's reference row).
- No config key changes. `fit_dependency` and its early stop do not change.

- [ ] **Step 1: Make the branch and baseline the suite**

Run: `git switch -c sl-line-term; uv run python -m pytest tests -q > pytest.txt 2>&1; tail -3 pytest.txt`
Expected: the present pass count; keep the file to compare later (do not commit it).

- [ ] **Step 2: Write the failing test**

In `tests/experiments/sl/test_embedding.py` add `import copy` (later tasks use it), replace `sigma_g=np.ones(30)` in
`world()` with `centre=0.0, scale=1.0`, and add:

```python
def test_the_target_is_geneeffect_itself_on_one_scale():
    frame = pd.DataFrame(
        {"A": [0.0, -2.0], "B": [1.0, 1.0], "C": [np.nan, 3.0]}, index=["L1", "L2"]
    )
    centre, scale = embedding.target_scale(frame)
    assert np.isclose(centre, 0.6) and np.isclose(scale, 2.64**0.5)
```

In `test_the_model_learns_geneeffect_and_ranks_held_out_pairs` add, after the `history` assertions:

```python
    # the kept checkpoint is a trained one, so the frozen head reads embeddings (on an
    # untrained model it reads a constant, and its recall is about 0.1)
    assert min(history, key=lambda r: r["stop_loss"])["epoch"] >= 1
    assert result["phases"][0]["val_recall_top10"] > 0.2
```

- [ ] **Step 3: Run it to see it fail**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -15 pytest_task.txt`
Expected: FAIL (`target_scale` is not defined; `Data` has no `centre`).

- [ ] **Step 4: Implement**

- `load_data`: the targets are `(labelled − centre) / scale` with `target_scale(labelled.loc[fit])`; no per-gene
  centring and no per-gene scale. `m_g` stays the fit lines' column mean.
- `geneeffect_table`: `data.centre + z * data.scale`.
- `results_markdown`: Table 2's second row is "the fit lines' gene mean alone" (drop "which the untrained model
  predicts"); Table 4's caption says "Epoch 0 is the untrained model, which predicts one value for every gene."
- The module docstring's stage-one text: the new target.
- Each finished config gets the comment
  `# The run of 2026-11-19 used the residual target (code at 232b4f0); the present code trains on GeneEffect itself.`
  (`embedding_mean_token.yaml`: `ad355ee`; `embedding_cellnet.yaml`: `64dbc5c`).

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -3 pytest_task.txt`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format src/experiments/sl/embedding.py tests/experiments/sl/test_embedding.py
git add src/experiments/sl/embedding.py tests/experiments/sl/test_embedding.py configs/sl
git commit -m "feat(sl): stage one trains on GeneEffect itself, so validation keeps a trained checkpoint"
```

---

### Task 2: The line term in the model

**Files:**
- Modify: `src/model/sl_embedding.py`
- Modify: `src/experiments/sl/embedding.py` (`build_model` passes `gene_in_line={}` until Task 3)
- Test: `tests/model/test_sl_embedding.py`

**Interfaces:**
- Produces: `GeneLineModel(sources, expression_dim, cell_dim, hidden, head_hidden, gene_in_line: Mapping[str, int])`;
  `gene_in_line` maps a feature source to its width (an empty mapping: no source).
- Produces: `features(u_a, u_b, combinatorial, mean, cells, mask) -> Tensor` (rows, `4 * WIDTH + 1`): the pair
  features that `pair` builds today. `pair` keeps its signature and returns `head(features(...))`.
- Produces: `line_term(u_a, u_b, combinatorial, in_line_a: Mapping[str, Tensor], in_line_b: Mapping[str, Tensor], mean, cells, mask) -> Tensor`
  (rows,). `in_line_a[name]` is (rows, width of `name`). The output is the same for (A, B) and (B, A).
- Produces: `line_parameters() -> list[nn.Parameter]`: the parameters of `self.line_head` and `self.in_line`.
  `backbone()` returns every parameter outside `head`, `line_head` and `in_line`.
- Module names (the tests of Task 4 read state-dict keys): `self.line_head` (`nn.Sequential`: `Linear(7 * WIDTH + 1, head_hidden)`,
  `GELU`, `Linear(head_hidden, 1)`, the last layer zero) and `self.in_line` (`nn.ModuleDict`: `name -> Linear(width, WIDTH)`).

- [ ] **Step 1: Write the failing test**

In the file's `model()` helper pass `gene_in_line={"own": 3}` (and `gene_in_line={}` in the test that builds a fresh
model). Add:

```python
def test_line_term_starts_at_zero_and_is_the_same_for_both_orders():
    built = model()
    expression, cells, mask = inputs(lines=1)
    u = built.gene_vectors()
    a, b = torch.tensor([0, 2, -1, 5]), torch.tensor([1, 4, 3, 5])
    flag = torch.tensor([True, True, False, True])
    mean, tokens = built.tokens(expression, cells)
    own = torch.randn(GENES + 1, 3)
    line = (mean[0], tokens[0], mask[0])
    start = built.line_term(u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, *line)
    assert start.shape == (4,) and torch.all(start == 0)
    torch.nn.init.normal_(built.line_head[-1].weight, std=0.5)
    forward = built.line_term(u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, *line)
    backward = built.line_term(u[b], u[a], flag, {"own": own[b]}, {"own": own[a]}, *line)
    assert forward.abs().sum() > 0 and torch.allclose(forward, backward, atol=1e-5)
    other = built.line_term(u[a], u[b], flag, {"own": own[b]}, {"own": own[b]}, *line)
    assert not torch.allclose(forward, other)  # the gene-in-line features are read
    no_cells = built.line_term(
        u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, mean[0], None, None
    )
    assert torch.isfinite(no_cells).all()
    with_line = {id(p) for p in built.line_parameters()}
    assert with_line and not with_line & {id(p) for p in built.backbone()}
    assert not with_line & {id(p) for p in built.head.parameters()}
```

In `test_gene_vectors_end_with_a_zero_row_and_backbone_leaves_the_head_out`, the count becomes
`len(built.backbone()) + len(head) + len(built.line_parameters()) == len(list(built.parameters()))`.

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python -m pytest tests/model/test_sl_embedding.py -q > pytest_task.txt 2>&1; tail -15 pytest_task.txt`
Expected: FAIL (`gene_in_line` is an unexpected argument).

- [ ] **Step 3: Implement**

`line_term` computes `e_g = Σ over sources of in_line[name](x_g[name])` for A and for B (zeros (rows, WIDTH) when the
model has no source) and returns
`line_head(cat([features(...), e_a + e_b, (e_a − e_b).abs(), e_a * e_b], dim=-1)).squeeze(-1)`. Add the model's
formula for the line term to the module docstring.

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run python -m pytest tests/model/test_sl_embedding.py tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -3 pytest_task.txt`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/sl_embedding.py src/experiments/sl/embedding.py tests/model/test_sl_embedding.py
git commit -m "feat(sl): a line head on the pair features and the gene-in-line features of the two genes"
```

---

### Task 3: Gene-in-line features in the data

**Files:**
- Modify: `src/experiments/sl/embedding.py` (`load_config`, `Data`, `load_data`, `build_model`)
- Test: `tests/experiments/sl/test_embedding.py`

**Interfaces:**
- Produces: top-level config key `line_term`: `null`, or a mapping with the keys
  `{"reference_lines", "max_epochs", "patience", "learning_rate", "own_expression", "gene_in_line"}`
  (`gene_in_line`: source name to cache directory, may be empty; `own_expression`: bool).
- Produces: `read_in_line(cache: Path, lines: Sequence[str], genes: Sequence[str], fit: Sequence[str]) -> np.ndarray`:
  (lines, genes + 1, width + 1), float16. The last column is the presence flag; the last gene row is zero.
- Produces: `own_expression(expression: pd.DataFrame, lines: Sequence[str], genes: Sequence[str]) -> np.ndarray`:
  (lines, genes + 1, 2), float16: the line's value of the gene in the scaled expression frame, and the presence flag.
- Produces: `Data.in_line: dict[str, Tensor]` (source name to (feature lines, genes + 1, width), float16, on the
  device) and `Data.in_line_at: Tensor` (lines,), the row of each line in those tensors, −1 for a line without
  benchmark rows. The feature lines are the lines with benchmark rows.
- Consumes: `GeneLineModel(..., gene_in_line=...)` from Task 2: `build_model` passes
  `{name: t.shape[2] for name, t in data.in_line.items()}`.

- [ ] **Step 1: Write the failing test**

Add `import json`, `"line_term": None` to `CONFIG`, and to `world()`'s `Data(...)`:
`in_line={}, in_line_at=torch.full((len(lines),), -1)`. Add:

```python
def test_gene_in_line_features_align_scale_on_the_fit_lines_and_fail_closed(
    tmp_path: Path,
):
    (tmp_path / "genes.json").write_text(json.dumps(["B", "A", "X"]))
    nan = [np.nan, np.nan]
    for name, values in (
        ("L1", [[1.0, 10.0], [3.0, 30.0], nan]),
        ("L2", [[3.0, 30.0], [5.0, 50.0], [9.0, 90.0]]),
        ("L3", [[5.0, 50.0], nan, [0.0, 0.0]]),
    ):
        np.save(tmp_path / f"{name}.npy", np.array(values, np.float16))
    out = embedding.read_in_line(
        tmp_path, ["L1", "L2", "L3"], ["A", "B", "Y"], fit=["L1", "L2"]
    )
    assert out.shape == (3, 4, 3) and out.dtype == np.float16
    root = 2**0.5  # the fit lines' values of the model's genes: 3, 1, 5, 3
    assert np.allclose(out[:, 0, 0], [0, root, 0], atol=1e-3)  # A; L3 has no value
    assert np.allclose(out[:, 1, 0], [-root, 0, root], atol=1e-3)  # B
    assert out[:, :, 2].tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]]
    assert not out[:, 2:].any()  # a gene off the cache's axis, and the zero row
    with pytest.raises(FileNotFoundError, match="L4"):
        embedding.read_in_line(tmp_path, ["L1", "L4"], ["A"], fit=["L1"])


def test_own_expression_is_the_lines_value_of_the_gene():
    frame = pd.DataFrame({"A": [0.5, -1.0], "C": [2.0, 3.0]}, index=["L1", "L2"])
    out = embedding.own_expression(frame, ["L2", "L1"], ["A", "B"])
    assert out.shape == (2, 3, 2)
    assert out[:, 0].tolist() == [[-1.0, 1.0], [0.5, 1.0]]
    assert not out[:, 1:].any()  # B is not an expression gene; the zero row
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -15 pytest_task.txt`
Expected: FAIL (`read_in_line` is not defined).

- [ ] **Step 3: Implement**

- `read_in_line`: a missing `<cache>/<ModelID>.npy` raises `FileNotFoundError` that lists the lines. Each feature
  column is centred and scaled (SD with `ddof=0`, 1 where 0) on the present entries of the `fit` lines over the
  model's genes. Absent entries are 0.
- `load_data`, when `config["line_term"]` is a mapping: the feature lines are the lines with benchmark rows; the fit
  lines of the scaling are those with `split == "train"`. Source `"expression"` (if `own_expression`) comes from the
  scaled expression frame; each `gene_in_line` entry comes from `read_in_line`. Add
  `counts_record["in_line_coverage"]`: for each source, the number of (line, gene) entries with presence 1.
  With `line_term: null`, `in_line` is empty. The Tx1-3B source is about 7 GB on the device (77 lines × 17,787 genes
  × 2,562 values, float16); keep it float16 and convert only the rows of a step.
- The four finished configs get `line_term: null`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -3 pytest_task.txt`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/sl/embedding.py tests/experiments/sl/test_embedding.py configs/sl
git commit -m "feat(sl): gene-in-line features of the SL lines, from the expression and from caches"
```

---

### Task 4: The line-term phase, its scores and its report

**Files:**
- Modify: `src/experiments/sl/embedding.py` (`train`, `run`, `rescore`, `results_markdown`, the module docstring)
- Test: `tests/experiments/sl/test_embedding.py`

**Interfaces:**
- Consumes: `GeneLineModel.line_term`, `line_parameters` (Task 2); `Data.in_line`, `Data.in_line_at` (Task 3).
- Produces: `line_term_scores(model, data, pairs, rows: np.ndarray, config, read: Callable[[int], int] | None = None) -> np.ndarray`:
  the score `s` of §3 for each row position. `R` is `sorted(pairs.draws)`. `read(line)` names the line whose tokens
  and gene-in-line features stand in for `line`; `R` and `p` do not change with it.
- Produces: `fit_line_term(model, data, pairs, config, generator) -> tuple[dict[str, Tensor], float, list[dict[str, Any]]]`:
  the weights, the validation recall in the top 10% and the history of the best epoch.
- Produces: `train(...)["line_term"]`: `None` without a line term, else a dict with the keys
  `{"val_recall_top10", "history", "scores", "other_line"}`. `train` leaves the model at the line term's best epoch.
- Produces: score tables `scores/embedding_line_term.parquet` and `scores/embedding_line_term_other_line.parquet`;
  `metrics.json` keys `"sl_line_term"` and `"line_term"` (validation recall and history).

- [ ] **Step 1: Write the failing tests**

In `world()` give the SL lines one feature source, so that the line term has an input:
`own = np.concatenate([z[sl_at][:, :, None], np.ones((16, 30, 1))], axis=2)` with `sl_at` the positions of the 16 SL
lines in `lines`, a zero gene row appended, as `in_line={"own": torch.tensor(..., dtype=torch.float16)}`, and
`in_line_at` set to 0…15 at those positions. Add:

```python
LINE_CONFIG = CONFIG | {
    "line_term": {"reference_lines": 3, "max_epochs": 4, "patience": 2,
                  "learning_rate": 1.0e-2, "own_expression": True, "gene_in_line": {}}
}  # fmt: skip


def test_line_term_is_a_deviation_from_the_training_lines():
    data, _ = world()
    torch.manual_seed(0)
    model = embedding.build_model(LINE_CONFIG, data)
    pairs = embedding.pair_rows(data)
    every = np.arange(len(data.rows))
    torch.nn.init.normal_(model.mean_token[-1].weight, std=0.5)
    on_mean = embedding.line_input_controls(model, data, pairs, LINE_CONFIG)["mean_input"]
    # an untrained line head gives 0: the score is the pair score on the mean input
    start = embedding.line_term_scores(model, data, pairs, every, LINE_CONFIG)
    assert np.allclose(start, on_mean, atol=1e-5)
    torch.nn.init.normal_(model.line_head[-1].weight, std=0.5)
    own = embedding.line_term_scores(model, data, pairs, every, LINE_CONFIG)
    assert np.isfinite(own).all() and not np.allclose(own, on_mean)
    # a line head whose input is the same in every line gives no deviation
    data.expression[:] = data.expression[0]
    data.offsets[:] = 0  # no line has a cell
    data.in_line["own"][:] = data.in_line["own"][0]
    same = embedding.line_term_scores(model, data, pairs, every, LINE_CONFIG)
    assert np.allclose(same, on_mean, atol=1e-4)


def test_the_line_term_trains_alone_and_train_reports_it():
    data, _ = world()
    torch.manual_seed(0)
    model = embedding.build_model(LINE_CONFIG, data)
    result = embedding.train(model, data, LINE_CONFIG)
    line = result["line_term"]
    assert set(line) == {"val_recall_top10", "history", "scores", "other_line"}
    assert np.isfinite(line["scores"]).all()
    assert not np.allclose(line["scores"], line["other_line"])
    assert not np.allclose(line["scores"], result["controls"]["mean_input"])
    before = copy.deepcopy(model.state_dict())
    state, _, history = embedding.fit_line_term(
        model, data, embedding.pair_rows(data), LINE_CONFIG,
        torch.Generator().manual_seed(1),
    )  # fmt: skip
    moved = {k for k in before if not torch.equal(before[k], state[k])}
    assert moved and all(k.startswith(("line_head.", "in_line.")) for k in moved)
    assert history and all("val_recall_top10" in r for r in history)
```

In `test_the_model_learns_geneeffect_and_ranks_held_out_pairs` (its config has `line_term: None`) add
`assert result["line_term"] is None`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -15 pytest_task.txt`
Expected: FAIL (`line_term_scores` is not defined).

- [ ] **Step 3: Implement**

- A helper gives `D` for row positions read in one line: the gene-in-line features are
  `data.in_line[name][data.in_line_at[line]][genes].float()`, and index −1 reads the zero row.
- `line_term_scores`: `p` from `model.pair` on the mean input (as `line_input_controls` builds it); `D` of the row's
  line (or of `read(line)`); minus the mean of `D` over `R`. Compute each reference line's tokens once, with every
  cell of the line; chunk rows by `eval.rows_per_chunk`.
- The step loss: lines and rows drawn as `pair_loss` draws them; `R` is `line_term.reference_lines` lines drawn with
  replacement from `sorted(pairs.draws)`, one draw for the step; tokens use `cells.per_step` and `cells.dropout`;
  binary cross-entropy on `s`.
- `fit_line_term`: only `model.line_parameters()` have `requires_grad`; AdamW with `line_term.learning_rate` and
  `stage_two.weight_decay`; `stage_two.steps_per_epoch` steps an epoch; `line_term.max_epochs` and
  `line_term.patience`; validation recall in the top 10% of `line_term_scores` selects the epoch. Restore
  `requires_grad` at the end.
- `train`: after the chosen phase's scores and controls, when `config["line_term"]` is a mapping, run
  `fit_line_term`, load its weights and add the result. The other-line control uses the same seeded order as
  `line_input_controls`.
- `run`: write the two new score tables and the two `metrics.json` keys. `rescore` writes the line term's other-line
  table too when the config has a line term.
- `results_markdown`: a new table after Table 1, models as rows (the model; the model on the mean input; the model
  with the line term; the line term on another line's input), columns recall in the top 10%, AUROC and AUPRC over the
  positive rate on validation and test; and one row for the line term in Table 3.

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py tests/model/test_sl_embedding.py -q > pytest_task.txt 2>&1; tail -3 pytest_task.txt`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/sl/embedding.py tests/experiments/sl/test_embedding.py
git commit -m "feat(sl): the line term trains last as a deviation from the training lines, with its score tables"
```

---

### Task 5: The gene-in-line caches

**Files:**
- Modify: `src/model/tx1.py` (`load_local_safetensors` gets the argument `gene_embeddings: bool`; update its callers)
- Create: `src/experiments/sl/gene_in_line.py`, `configs/sl/gene_in_line.yaml`, `docs/data/gene-in-line-features.md`
- Modify: `hpc/run.sh` (usage line, the `case` list, `sl-gene-in-line) exec "$python_bin" -m src.experiments.sl.gene_in_line "$@" ;;`
  with the environment that `ladder-inputs --steps tx1` uses), `hpc/README.md`, `src/README.md`
- Test: `tests/model/test_tx1.py`, `tests/experiments/sl/test_gene_in_line.py`

**Interfaces:**
- Produces: `gene_sums(gene_ids: Tensor, gene_emb: Tensor, wanted: Tensor) -> tuple[Tensor, Tensor]` in
  `src/model/tx1.py`: for each token id of `wanted` (genes,), the sum of its embedding over the cells whose sequence
  holds it (genes, width) and the number of those cells (genes,). `gene_ids` is (cells, sequence), `gene_emb`
  (cells, sequence, width).
- Produces: `build_tx1_gene_encoder(model_dir: Path, batch_size: int, max_length: int, cells_per_pass: int) -> Callable[[ad.AnnData], tuple[np.ndarray, np.ndarray]]`:
  for the genes of `adata.var`, the mean embedding (genes × 2,560, float32, NaN where no cell holds the gene) and the
  share of the cells that hold it (genes,).
- Produces: `fit_components(vectors: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]` in
  `gene_in_line.py`: the centre (panel,) and an orthonormal basis (panel, components) of the vectors.
- Produces: the command `hpc/run.sh sl-gene-in-line CONFIG --steps tx1_gene|pie_change [--shard I/N]`
  and the caches `<out>/tx1_gene/` and `<out>/pie_change/` in the format of §4.
- Config `configs/sl/gene_in_line.yaml`: `kind: sl_gene_in_line`, `seed: 0`,
  `joint: configs/revision/frozen_huber.yaml` (its `paths.tx1_model_dir`, `paths.basal_raw_umi` and
  `preparation.var_ensembl_col`), `labels: data/sl_studies/benchmark_v2/labels.parquet`,
  `split: configs/benchmarks/cell_line_sl_split.json`, `out: data/sl_gene_in_line/v1`,
  `tx1: {batch_size: 32, max_length: 2048, max_cells: 256, cells_per_pass: 64}`,
  `pie: {root: data/sl_pie/v1, components: 64, sample: 200000}`.

- [ ] **Step 1: Write the failing tests**

`tests/model/test_tx1.py`:

```python
def test_gene_sums_add_each_gene_over_the_cells_that_hold_it():
    from src.model.tx1 import gene_sums

    ids = torch.tensor([[7, 3, 0], [3, 9, 0], [5, 0, 0]])
    emb = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    sums, counts = gene_sums(ids, emb, torch.tensor([3, 9, 4]))
    assert counts.tolist() == [2, 1, 0]
    assert sums.tolist() == [[8.0, 10.0], [8.0, 9.0], [0.0, 0.0]]
```

`tests/experiments/sl/test_gene_in_line.py`:

```python
import numpy as np
import pytest

from src.experiments.sl import gene_in_line


def test_components_span_the_training_vectors():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 9)) + 3.0
    centre, basis = gene_in_line.fit_components(vectors, 2)
    assert centre.shape == (9,) and basis.shape == (9, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-4)
    assert np.allclose((vectors - centre) @ basis @ basis.T, vectors - centre, atol=1e-3)


def test_the_config_loads_and_an_unknown_step_is_refused():
    from pathlib import Path

    config = gene_in_line.load_config(Path("configs/sl/gene_in_line.yaml"))
    assert config["kind"] == "sl_gene_in_line" and config["pie"]["components"] == 64
    with pytest.raises(SystemExit):
        gene_in_line.main(["configs/sl/gene_in_line.yaml", "--steps", "cellnet"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/model/test_tx1.py tests/experiments/sl/test_gene_in_line.py -q > pytest_task.txt 2>&1; tail -15 pytest_task.txt`
Expected: FAIL (the module and `gene_sums` do not exist).

- [ ] **Step 3: Implement**

- `load_local_safetensors(model_dir, gene_embeddings)`: sets `model_config["return_gene_embeddings"]` to the argument.
  With it on, the model's output holds `gene_ids` (cells, sequence) and `gene_emb` (cells, sequence, width), the
  transformer's output for each token (`tahoe_x1/model/model.py`, read 2026-11-20).
- `build_tx1_gene_encoder`: as `build_tx1_encoder`, but it encodes `cells_per_pass` cells at a time
  (`trainer.predict` holds every batch's token outputs, about 1.3 GB for 64 cells), adds each pass with `gene_sums`
  and frees it. A missing `gene_emb` or `gene_ids` key raises `KeyError`; do not fall back to the cell embedding.
- `gene_in_line.py`: the gene axis is the sorted set of `gene_a` and `gene_b` of the label table; the lines are the
  split's lines with a labelled row. Each step writes `genes.json` once and skips a line whose file exists; files are
  written with `src.data.io.write_npy` (atomic).
  - `tx1_gene`: read the line's raw cells with `read_basal_source`, keep at most `tx1.max_cells` with
    `tx1_cell_indices(…, seed)`, map each gene of the axis to its `var` row by the symbol column
    (`src.data.basal.symbol_column`; the first row of a repeated symbol), and write genes × 2,561 (the mean embedding,
    then the share), a NaN row for a gene with no value. `--shard I/N` divides the lines. A `report.json` for each
    shard gives, for each line, its cells and the number of genes with a value.
  - `pie_change`: the lines are those of `<pie.root>/lines.parquet`; a gene's vector in a line is its row of
    `<pie.root>/pie/change/<GENE>.npy` (lines × panel; PIE's symbols are upper case). `fit_components` is fitted on a
    seeded sample of at most `pie.sample` (line, gene) vectors of the lines with role `train`; every vector is then
    written in `pie.components` components. A gene without a change file is a NaN row.
- The data card `docs/data/gene-in-line-features.md`: role, sources, the format of §4, the fit rule, and the known
  limits (a gene has a Tx1-3B value only in the cells whose 2,048-gene sequence holds it; PIE has seen 42 of our
  lines).

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run python -m pytest tests/model/test_tx1.py tests/experiments/sl/test_gene_in_line.py -q > pytest_task.txt 2>&1; tail -3 pytest_task.txt`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/tx1.py src/experiments/sl/gene_in_line.py configs/sl/gene_in_line.yaml hpc/run.sh hpc/README.md src/README.md docs/data/gene-in-line-features.md tests/model/test_tx1.py tests/experiments/sl/test_gene_in_line.py
git commit -m "feat(sl): caches of gene-in-line features from Tx1-3B's gene embeddings and from PIE's predicted changes"
```

---

### Task 6: The study's configs, the harness and the protocol

**Files:**
- Create: the four configs of §5, `configs/sl/benchmark_line_term.yaml`
- Modify: `src/experiments/sl/benchmark_baselines.py` (`NAMES`), `docs/04-sl-ranking-protocol.md` (new §19; a note in
  §18's stage-one bullet that §19 replaces the residual target), `hpc/README.md` (the launch block), `src/README.md` (the
  `embedding` entry)
- Test: `tests/experiments/sl/test_embedding.py`

**Interfaces:**
- Consumes: the `line_term` config key (Task 3); the cache paths of Task 5.
- Produces: scorer keys for the harness. For each run prefix `P` in `direct`, `direct_mean_token`,
  `direct_pie_change`, `direct_tx1_gene`: `P` (the model), `P_mean_input`, `P_other_line`,
  `P_line_term`, `P_line_term_other_line`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_study_configs_differ_only_where_the_plan_says():
    base = embedding.load_config(Path("configs/sl/embedding_direct.yaml"))
    finished = embedding.load_config(Path("configs/sl/embedding.yaml"))
    assert {k: v for k, v in base.items() if k != "line_term"} == {
        k: v for k, v in finished.items() if k != "line_term"
    }
    assert finished["line_term"] is None
    assert base["line_term"] == {
        "reference_lines": 8, "max_epochs": 30, "patience": 5, "learning_rate": 1.0e-3,
        "own_expression": True, "gene_in_line": {},
    }  # fmt: skip
    changes = {
        "embedding_direct_mean_token": {("cell_embeddings",): None},
        "embedding_direct_pie_change": {("line_term", "gene_in_line"): {
            "pie_change": "data/sl_gene_in_line/v1/pie_change"}},
        "embedding_direct_tx1_gene": {("line_term", "gene_in_line"): {
            "tx1_gene": "data/sl_gene_in_line/v1/tx1_gene"}},
    }  # fmt: skip
    for name, change in changes.items():
        expected = copy.deepcopy(base)
        for keys, value in change.items():
            target = expected
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        assert embedding.load_config(Path(f"configs/sl/{name}.yaml")) == expected
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python -m pytest tests/experiments/sl/test_embedding.py -q > pytest_task.txt 2>&1; tail -15 pytest_task.txt`
Expected: FAIL (no such config).

- [ ] **Step 3: Write the configs, the names and the docs**

- `embedding_direct.yaml`: `configs/sl/embedding.yaml` with the `line_term` mapping of the test. Each config starts
  with a comment that says what it is, its launch command and this plan's path. The other three differ as the test
  says.
- `benchmark_line_term.yaml`: a copy of `benchmark_embedding.yaml`. `extra_scores` keeps the baselines and the three
  finished runs' tables, and adds the five tables of each of the four runs
  (`outputs/sl/embedding/<config name>_seed0_v2/scores/embedding*.parquet`). `paired_differences`: for each prefix `P`,
  `[P_line_term, P_mean_input]`, `[P_line_term, P_line_term_other_line]` and `[P_line_term, P]`; then
  `[direct, direct_mean_token]`, `[direct_line_term, direct_mean_token_line_term]`,
  `[direct_pie_change_line_term, direct_line_term]`, `[direct_tx1_gene_line_term, direct_line_term]`,
  `[direct, embedding]`, `[direct_mean_token, embedding_mean_token]` and
  `[direct_line_term, kebabci_paralog_predicted]`.
- `NAMES`: one entry for each scorer key. The model name is "Gene-knowledge, cell-set model, direct GeneEffect
  target" (`direct`), with "without cell tokens", "PIE change in the line term" or "Tx1-3B gene embeddings in the line
  term" as the config says. The suffixes are ", context-ablated (the fit lines' mean input)", " on another line's
  input", " with the line term" and " with the line term on another line's input".
- SL protocol §19, "The direct GeneEffect target and the line term (2026-11-20)": the content of §3 to §6 of this plan as rules
  (model, stage one, the line term, the features, outputs, controls), with the code and config paths, and "Runs:
  none yet".

- [ ] **Step 4: Run the whole suite and the lint**

Run: `uv run python -m pytest tests -q > pytest_after.txt 2>&1; tail -3 pytest_after.txt; .venv/bin/ruff check .`
Expected: no failure that `pytest.txt` (the baseline) does not have; ruff prints no finding.

- [ ] **Step 5: Commit**

```bash
git add configs/sl src/experiments/sl/benchmark_baselines.py docs/04-sl-ranking-protocol.md hpc/README.md src/README.md tests/experiments/sl/test_embedding.py
git commit -m "feat(sl): the configs and the harness rows of the line-term study, and its protocol section"
```

---

### Task 7: The runs on the H20 and the record

Follow the `hpc-execution` skill. The user names the container port before the first launch. Every `sl-embedding`
run needs one GPU.

**Files:**
- Create: `results/sl_line_term_seed0/summary.html` (format: `.claude/skills/results-report`), with the copied host
  reports under `results/sl_line_term_seed0/host/`
- Modify: `docs/04-sl-ranking-protocol.md` (§19's runs and results), `CLAUDE.md` (the current state), `README.md`,
  `model-status.html`, this plan's status line

- [ ] **Step 1: Push the branch and pull it on the H20**

Run locally: `git push`. Run in the H20 checkout: `hpc/pull.sh`.
Expected: the H20 checkout is at the commit of Task 6.

- [ ] **Step 2: Launch the two runs that need no cache**

```bash
for name in embedding_direct embedding_direct_mean_token; do
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup hpc/run.sh sl-embedding configs/sl/$name.yaml --run-id ${name}_seed0_v2 \
    > outputs/launches/${name}_seed0_v2.log 2>&1 < /dev/null &
done   # one free GPU index in $GPU for each run
```

Expected: each log ends with `gene-knowledge, cell-set model: outputs/sl/embedding/<run id>`, and each run directory
holds `metrics.json`, `results.md` and five tables under `scores/`.

- [ ] **Step 3: Write the PIE cache (CPU) and pilot the Tx1-3B cache on one line**

```bash
hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps pie_change
CUDA_VISIBLE_DEVICES=0 hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps tx1_gene --shard 0/77
```

Expected: `data/sl_gene_in_line/v1/pie_change/` holds `genes.json` and one file for each PIE line; the pilot writes
one line's file and a `report.json`. Read in the report the number of genes with a value and the wall time, and
record both in the data card. A `KeyError` for `gene_emb` means that Tx1-3B's predict path does not return the token
outputs: stop and report it.

- [ ] **Step 4: Write the Tx1-3B cache for every line, one process for each GPU**

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup hpc/run.sh sl-gene-in-line configs/sl/gene_in_line.yaml --steps tx1_gene \
    --shard $i/4 > outputs/launches/gene_in_line_tx1_$i.log 2>&1 < /dev/null &
done
```

Expected: one `.npy` for each SL-split line with a labelled row under `data/sl_gene_in_line/v1/tx1_gene/`.

- [ ] **Step 5: Launch the two runs that read a cache, then the harness**

```bash
for name in embedding_direct_pie_change embedding_direct_tx1_gene; do
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup hpc/run.sh sl-embedding configs/sl/$name.yaml --run-id ${name}_seed0_v2 \
    > outputs/launches/${name}_seed0_v2.log 2>&1 < /dev/null &
done
hpc/run.sh sl-baselines configs/sl/benchmark_line_term.yaml --run-id benchmark_line_term_seed0_v2
```

Expected: the harness writes its results under `outputs/sl/benchmark_baselines/benchmark_line_term_seed0_v2/`.

- [ ] **Step 6: Write the record and update the docs**

Copy each run's `results.md` and `metrics.json` and the harness report to `results/sl_line_term_seed0/host/`. Write
`summary.html` with the `results-report` skill: SL tables first (models as rows; recall in the top 10%, AUROC, AUPRC;
validation and test), then the line's gain and what the cells add with their intervals, then the row sets, then the
kept stage-one epoch of each run, then the stage-one numbers. Do not show the training SL fraction. Update SL
protocol §19, the current-state section of `CLAUDE.md` (replace, do not append), `README.md`, `model-status.html`
and this plan's status line.

- [ ] **Step 7: Commit, and merge when the work is established**

```bash
git add results/sl_line_term_seed0 docs CLAUDE.md README.md model-status.html
git commit -m "docs(sl): the record of the line-term study"
```

Merge the branch into `main` and push when the suite and the lint pass and the runs have finished on the H20.

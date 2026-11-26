# Published SL Baselines on the Benchmark: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score five more published SL models (SLMGAE, ESM4SL, MVGCN-iSL, MuSL, PromptCASL) on the multi-study SL
benchmark with their official code, on the H20, and report them beside the four published baselines that stay
(KG4SL, MiT4SL, the Kebabci 2026 paralog classifier, Cilantro-SL). PARIS, SLIdR and the De Kegel classifier leave
the reported set; their code stays (stage two reads PARIS).

**Architecture:** Each model gets one harness directory `hpc/<model>/` (`setup.sh`, `requirements.txt`, `prepare.py`
where inputs must be built, `train.py`). The official code is vendored: its code files are copied into
`third_party/<model>/` at a pinned commit, tracked in Git with a `VENDORED.md`, and imported; our code only builds
the inputs from the benchmark and writes `scores.parquet`. The existing harness `src/experiments/sl/benchmark_baselines.py` reads those tables.

**Tech Stack:** Python 3.10/3.11 environments built by `uv` on the H20; TensorFlow 2.15 in `compat.v1` mode (SLMGAE);
torch 2.6.0 + CUDA 12.4 (ESM4SL, MVGCN-iSL, MuSL); PyG 2.6.1; fair-esm; magic-impute, scprep, scanpy.

**Spec:** No separate spec. The design is the section "Design" below, from the conversation of 2026-11-26. Read
`docs/04-sl-ranking-protocol.md` §8 and §18, `hpc/kg4sl/train.py` and `hpc/mit4sl/train.py` before you start.

## Global Constraints

- All five official code bases are Python. No translation is necessary. GOSemSim (R), which made SLMGAE's released GO
  matrices, is not vendored: Wang's measure is written in Python from its formula and compared with the released matrices.
- Vendoring (as `third_party/pie/`): copy the official code files and the licence at the pinned commit into
  `third_party/<model>/`; no data file above 1 MB (`setup.sh` fetches those on the H20 from the same commit with a
  SHA-256 check, into `data/<model>/official/`); add `!/third_party/<model>/` to `.gitignore`. `VENDORED.md` gives the
  source, the commit, the licence, the files left out and every local change. Official model classes, losses,
  training steps and default settings are imported. A necessary change to a vendored file is made in place, is the
  smallest possible, and has a line in `VENDORED.md`; a shim in our `train.py` is better when it does the same.
- Fit lines: the split's `train` and `bulk_labelled` sides (`FIT_SIDES`): the training lines with single cells and
  the 33 labelled lines with bulk data only. A baseline is not limited to the lines with single cells. Cilantro-SL,
  MiT4SL and the Kebabci classifier already fit on both; KG4SL is fitted again (Task 7). MuSL alone fits on `train`
  only: its input is the distribution of a gene pair over a line's cells, which a bulk-only line does not have. Held-out lines: `val`, `test`. Training rows: `condition == "genetic"`.
- No measured GeneEffect of a validation or test line enters any input. A held-out line reads stage one's prediction
  `outputs/sl/conditional/conditional_seed0/predictions.parquet` (454 lines x 17,787 genes).
- Every graph loses each edge between the two genes of a benchmark pair (the rule of `hpc/kg4sl/train.py`).
- Checkpoint selection reads validation lines only. Test is scored once.
- Output contract: `outputs/sl/<model>/seed0/scores.parquet` with `gene_a, gene_b, score, covered` (pair-level models)
  or `model_id, gene_a, gene_b, score, covered` (models that read the line); one row for each stored orientation of
  each row of the `train`, `val` and `test` lines. A row the model cannot score has `covered = False` and the median
  score of the covered held-out rows. `metrics.json` records settings, counts, coverage and input SHA-256 values.
- No gate on model quality. Agreement and coverage numbers are telemetry in `metrics.json`.
- Tests use synthetic fixtures, load harness files with `importlib` as `tests/hpc/test_kebabci_features.py` does, and
  use `pytest.importorskip` for a package the dev environment lacks.
- Code moves by Git only (`git push origin main`, then `hpc/pull.sh` on the H20). Data files (weights, tables) are
  copied with a SHA-256 check. Conventional Commits. `.venv/bin/ruff format` on touched files only.
- H20: port 30757 answered on 2026-11-26; port 30838 closed the connection. State the port before each long run.
  Launch long runs with `nohup ... > outputs/launches/<id>.log 2>&1 &`.

## Review Focus

1. A gene outside a model's inputs (no GO term, no PPI edge, no protein sequence, no expression): the row must tie at
   the median with `covered = False`. It must not stop the run or get a real-looking score. Test in Task 1.
2. One pair stored as (A, B) in one study and (B, A) in another: the table must hold both orientations, or the harness
   stops with "has no row for N held rows". Test in Task 1.
3. Label leak through a graph: BioGRID genetic interactions hold SL calls of benchmark screens. Test in Task 1
   (`drop_pair_edges`) and used in Tasks 2, 3 and 5.
4. Measured GeneEffect of a held-out line in MVGCN-iSL's node features or its co-essentiality graph. Test in Task 2.
5. A line with few cells (the smallest labelled line has 18): MAGIC and the quantile features of MuSL. Test in Task 5.

## Design

Facts measured on the H20 on 2026-11-26 (commit `238c43c`): 24,793 genes and 1,123,862 distinct pairs in the labelled
rows; fit pairs with a genetic row: 867,801, of which 16,924 are SL in at least half of their rows; all 86 labelled
`train`, `val` and `test` lines have a raw-UMI h5ad (18 to 58,324 cells, median 291); 1.5 TB RAM, 4 x H20 (97 GB).

| Model | Official code (pinned commit) | What changes from the official setting, and why |
| --- | --- | --- |
| SLMGAE | `DiNg1011/SLMGAE` `ba0c7016` | The released gene list (6,375 genes) covers 21.7% of the held-out rows, so the three support views (GO-BP and GO-CC similarity, k = 45; BioGRID PPI) are built again for the benchmark genes. One fit in place of five folds; the official 300 epochs, last epoch. |
| ESM4SL | `JieZheng-ShanghaiTech/ESM4SL` `348b2bf9` | None in the model. The official CLI runs on our CSV files. Sequences longer than 2,000 residues are cut before ESM-2 (the model reads 2,000 residues at most). The released config has no line input (`DATASET.CELL_LINE: null`), so the score is the same in every line. |
| MVGCN-iSL | `kunjiefan/MVGCNiSL` `85b7b880` | The official model is one line. Here one model trains on all fit lines: one official training step for each line in each epoch. The line-specific SL graph is left out (a held-out line has none). The co-essentiality graph is computed again on the fit lines. |
| MuSL | `JieZheng-ShanghaiTech/MuSL` `f8021cfc` | The official cell-line setting is one line. Here one model trains on all `train` lines; each row reads its own line's imputed cells. At most 2,000 cells for each line. |
| PromptCASL | `JieZheng-ShanghaiTech/PromptCASL` `afcf7e92` | The official leave-one-cell-line-out run trains one model for each target line. Here one model trains on the fit lines and scores every held-out line. The three omics tables are built for the benchmark lines. |

Pair-level training samples (SLMGAE, ESM4SL): one sample for each pair, SL when at least half of its genetic rows in
the fit lines are SL (the rule of `hpc/kg4sl/train.py`).

Controls for the three new models that read the line (the "recall: own − mean input" column of `model-status.html`):
MVGCN-iSL on the fit lines' mean node features; MuSL on one pooled matrix of fit-line cells; PromptCASL on the fit
lines' mean omics values.

## File Structure

| File | Responsibility |
| --- | --- |
| `hpc/sl_common.py` (new) | Benchmark rows, pair labels, balanced draw, graph-edge removal, score-table writers. No heavy import. |
| `third_party/{mvgcn_isl,slmgae,esm4sl,musl,promptcasl}/` (new, tracked) | Vendored official code with `VENDORED.md`. |
| `hpc/mvgcn_isl/{setup.sh,requirements.txt,prepare.py,train.py}` (new) | MVGCN-iSL environment, inputs and run. |
| `hpc/slmgae/{setup.sh,requirements.txt,go_similarity.py,prepare.py,train.py}` (new) | SLMGAE environment, GO views and run. |
| `hpc/promptcasl/{setup.sh,requirements.txt,prepare.py,train.py}` (new) | PromptCASL environment, omics tables, prompts and run. |
| `hpc/kg4sl/train.py` (modify) | KG4SL fits on `FIT_SIDES`. |
| `hpc/esm4sl/{setup.sh,requirements.txt,prepare.py,collect.py}` (new) | ESM4SL environment, sequences and CSV files, score table from the official output. |
| `hpc/musl/{setup.sh,requirements.txt,prepare.py,train.py}` (new) | MuSL environment, imputed line matrices, features and run. |
| `tests/hpc/test_sl_common.py`, `test_mvgcn_isl.py`, `test_slmgae_go.py`, `test_esm4sl_prepare.py`, `test_musl_prepare.py`, `test_promptcasl_prepare.py` (new) | Tests of our input code. |
| `src/experiments/sl/benchmark_baselines.py` (modify `NAMES`) | Eight new scorer names. |
| `configs/sl/benchmark_published.yaml` (new) | Harness config of the nine published baselines and the three controls. |
| `.gitignore`, `hpc/README.md`, `docs/04-sl-ranking-protocol.md`, `model-status.html`, `CLAUDE.md`, `README.md` (modify) | Environments ignored; commands, protocol record and status. |

---

### Task 1: Shared benchmark rows and score writers

**Files:**
- Create: `hpc/sl_common.py`
- Test: `tests/hpc/test_sl_common.py`

**Interfaces (produced, used by every later task):**
- `FIT_SIDES = ("train", "bulk_labelled")`, `HELD_SIDES = ("val", "test")`, `SCORED_SIDES = ("train", "val", "test")`
- `benchmark_rows(labels: Path, split: Path) -> pd.DataFrame`: labelled rows of the lines of `FIT_SIDES + HELD_SIDES`;
  columns `model_id, gene_a, gene_b, label (int), condition, side, gene_1, gene_2` (`gene_1 <= gene_2`).
- `pair_labels(rows: pd.DataFrame, sides: tuple[str, ...]) -> pd.DataFrame`: genetic rows of `sides`; one row for each
  `(gene_1, gene_2)`; `label = 1` when the mean label is at least 0.5.
- `balanced(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame`: all rows with `label == 1` and as many
  rows with `label == 0`, drawn without replacement.
- `drop_pair_edges(edges: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame`: `edges` (`gene1, gene2`) without the
  edges whose two genes are a benchmark pair, in either order.
- `write_pair_scores(out: Path, rows: pd.DataFrame, scores: pd.Series) -> dict`: `scores` indexed by
  `(gene_1, gene_2)`; writes `out/scores.parquet`; returns `{"scored_pairs", "scored_pairs_covered",
  "held_rows_covered_share", "uncovered_score"}`.
- `write_line_scores(out: Path, rows: pd.DataFrame, scores: pd.Series, name: str = "scores") -> dict`: `scores`
  indexed by `(model_id, gene_1, gene_2)`; writes `out/<name>.parquet`; same return keys.

- [ ] **Step 1: Baseline the suite.** `uv run python -m pytest tests -q > pytest_baseline.txt 2>&1`; keep the file
  until the last task, then delete it.
- [ ] **Step 2: Create the branch.** `git switch -c published-baselines`
- [ ] **Step 3: Write the failing tests** in `tests/hpc/test_sl_common.py` (load `hpc/sl_common.py` with `importlib`):

```python
ROWS = pd.DataFrame({
    "model_id": ["T1", "T1", "T2", "V1", "V1", "X1"],
    "gene_a":   ["B",  "A",  "A",  "B",  "C",  "A"],
    "gene_b":   ["A",  "C",  "B",  "A",  "Z",  "B"],
    "label":    [1.0,  0.0,  0.0,  1.0,  0.0,  None],
    "condition": ["genetic"] * 6,
})
SPLIT = {"train": ["T1"], "bulk_labelled": ["T2"], "val": ["V1"], "test": [], "unlabeled_train": ["X1"]}

def test_rows_keep_labelled_split_lines_and_sort_the_pair(tmp_path):
    rows = common.benchmark_rows(*write(tmp_path, ROWS, SPLIT))
    assert list(rows["side"]) == ["train", "train", "bulk_labelled", "val", "val"]
    assert list(rows["gene_1"] + rows["gene_2"]) == ["AB", "AC", "AB", "AB", "CZ"]

def test_pair_is_sl_when_half_of_its_fit_rows_are_sl(tmp_path):
    pairs = common.pair_labels(common.benchmark_rows(*write(tmp_path, ROWS, SPLIT)), common.FIT_SIDES)
    assert dict(zip(pairs["gene_1"] + pairs["gene_2"], pairs["label"])) == {"AB": 1, "AC": 0}

def test_drop_pair_edges_removes_both_orders(tmp_path):
    edges = pd.DataFrame({"gene1": ["A", "B", "A"], "gene2": ["B", "A", "Q"]})
    kept = common.drop_pair_edges(edges, common.benchmark_rows(*write(tmp_path, ROWS, SPLIT)))
    assert kept.values.tolist() == [["A", "Q"]]

def test_pair_scores_cover_each_stored_orientation_and_tie_the_rest(tmp_path):
    rows = common.benchmark_rows(*write(tmp_path, ROWS, SPLIT))
    scores = pd.Series({("A", "B"): 0.9, ("A", "C"): 0.1})          # (C, Z): gene outside the model
    report = common.write_pair_scores(tmp_path, rows, scores)
    table = pd.read_parquet(tmp_path / "scores.parquet").set_index(["gene_a", "gene_b"])
    assert table.loc[("B", "A"), "score"] == 0.9 and table.loc[("A", "C"), "score"] == 0.1
    assert not table.loc[("C", "Z"), "covered"]
    assert table.loc[("C", "Z"), "score"] == 0.9                     # median of the covered held-out rows
    assert report["held_rows_covered_share"] == 0.5

def test_line_scores_keep_the_line(tmp_path):
    rows = common.benchmark_rows(*write(tmp_path, ROWS, SPLIT))
    scores = pd.Series({("T1", "A", "B"): 0.2, ("V1", "A", "B"): 0.7})
    common.write_line_scores(tmp_path, rows, scores)
    table = pd.read_parquet(tmp_path / "scores.parquet").set_index(["model_id", "gene_a", "gene_b"])
    assert table.loc[("V1", "B", "A"), "score"] == 0.7 and not table.loc[("V1", "C", "Z"), "covered"]
```

  `balanced` test: 2 SL and 5 non-SL rows give 4 rows with 2 SL. The writers score the rows of `SCORED_SIDES` only.
- [ ] **Step 4: Run** `uv run python -m pytest tests/hpc/test_sl_common.py -q > pytest.txt 2>&1`. Expected: FAIL
  (`hpc/sl_common.py` is absent).
- [ ] **Step 5: Implement `hpc/sl_common.py`** (pandas, numpy and pyarrow only; each harness imports it after
  `sys.path.insert(0, str(ROOT / "hpc"))`).
- [ ] **Step 6: Run the test file again.** Expected: PASS.
- [ ] **Step 7: Commit.** `git add hpc/sl_common.py tests/hpc/test_sl_common.py && git commit -m "feat(sl): shared benchmark rows and score writers for the published baselines"`

---

### Task 2: MVGCN-iSL

**Files:**
- Create: `hpc/mvgcn_isl/setup.sh`, `hpc/mvgcn_isl/requirements.txt`, `hpc/mvgcn_isl/prepare.py`, `hpc/mvgcn_isl/train.py`
- Modify: `.gitignore` (add `/.venv-mvgcn/`)
- Test: `tests/hpc/test_mvgcn_isl.py`

**Official code used:** `code/model.py: GCN_pool, EarlyStopping`; `code/main.py: train_model, test_model`;
`code/utils.py: load_graph_data, calculate_coexpression, generate_torch_edges, get_link_labels`. Official defaults
(`main.py`): `GCN_pool`, max pooling, `out_channels` 64, `LR` 1e-4, AdamW, `batch_size` 512, `balanced` 1,
`epochs` 500, `patience` 150, node features `raw_omics` (exp, mut, cnv, ess), independent graphs `PPI-genetic`,
`PPI-physical`, `co-exp`, `co-ess`.

**Decisions:**
- Environment `.venv-mvgcn`, Python 3.10: the pins of `hpc/mit4sl/requirements.txt` for torch, torch-geometric, numpy,
  pandas, pyarrow, scipy, scikit-learn, tqdm, plus `networkx` and `matplotlib`. No torch-scatter (`GCNConv` does not need it).
- Shims in our files, not a patch: `pd.DataFrame.append = lambda self, other: pd.concat([self, other])` (pandas 2
  dropped it; `load_graph_data` calls it); `main.args = argparse.Namespace(pooling="max", batch_size=512, balanced=1,
  pos_weight=50, model="GCN_pool")` (the official functions read the module global `args`); `MPLBACKEND=Agg`.
  The official loaders read `../data/...`: run them with the working directory `third_party/mvgcn_isl/code`.
- Node index: the sorted union of the benchmark genes and the genes of the four graphs. One index for all lines.
- Graphs: `load_graph_data("PPI-genetic")`, `("PPI-physical")`, `("co-exp")` (released files) and co-essentiality
  computed again: write the fit lines' measured GeneEffect (`CRISPRGeneEffect.csv`, lines of `FIT_SIDES`) to
  `third_party/mvgcn_isl/data/CCLE/CCLE_ess.csv` and call the official `calculate_coexpression("ess", 0.2)` (the
  released `coexpression_ess_0.2.csv` is not fetched: it reads the held-out lines' GeneEffect). Then
  `drop_pair_edges` on each graph. For each line, a graph keeps the edges whose two genes have expression > 0 in
  that line (the official filter).
- Node features of a line, in the official order `exp, mut, cnv, ess`, then `sklearn.preprocessing.scale` over the nodes:
  - `exp`: `2**x - 1` of `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`. That table lacks 5 training, 2
    validation and 1 test line (measured 2026-11-26): such a line reads its pseudo-bulk in counts per million, the
    sum over the cells of `data/sl_dependency_v0/processed/basal_raw_umi/<ModelID>.h5ad`. The source of each line
    goes to `metrics.json`;
  - `mut`: 1 when `OmicsSomaticMutationsMatrixDamaging.csv` or `OmicsSomaticMutationsMatrixHotspot.csv` is > 0;
  - `cnv`: `2**x - 1` of `PortalOmicsCNGeneLog2.csv`;
  - `ess`: measured GeneEffect for a fit line; stage one's prediction for a held-out line; both cut to the predicted
    genes. A gene that a table lacks is 0 before scaling (the official rule). Read the tables with
    `src.data.depmap.read_depmap_matrix` (all files are under `data/sl_dependency_v0/raw/depmap/`).
- A line that the copy-number or the mutation table lacks reads 0 for that feature (the official zero fill): 21
  labelled lines have no copy number (13 training, 6 validation, 2 test). A fit line without measured GeneEffect
  (6 training lines) is left out of the fit. Every held-out line has a score. Counts go to `metrics.json`.
- Mean input: each held-out line reads the mean over the fit lines of the four unscaled features, then the same
  scaling and expression filter.
- Epoch: the fit lines in a new random order; for each line `generate_torch_edges(rows_of_line, True, True, device)`
  and the official `train_model`. Validation loss: the mean over validation lines of the official `test_model` loss
  on a fixed balanced draw. `EarlyStopping(patience=150, reverse=True)`; at most 500 epochs; the checkpoint with
  the lowest validation loss. `torch.manual_seed(0)`, `np.random.seed(0)`.
- Score: the sigmoid of `model.decode` on every row of the `train`, `val` and `test` lines, in batches of 100,000.

**Interfaces:**
- Consumes: `sl_common.benchmark_rows`, `drop_pair_edges`, `write_line_scores`.
- Produces: `prepare.py`: `node_features(tables: dict[str, pd.DataFrame], line: str, genes: pd.Index) -> np.ndarray`
  (genes x 4, unscaled); `gene_effect_input(measured, predicted, held_lines) -> pd.DataFrame`;
  `line_edges(edges: np.ndarray, expressed: np.ndarray) -> np.ndarray`. Files: `data/mvgcn_isl/benchmark/`
  (`genes.json`, `graphs.npz`, `features.npz` with one array for each line and `mean_input`, `rows.parquet`).
  `train.py` writes `outputs/sl/mvgcn_isl/seed0/{scores.parquet,scores_mean_input.parquet,metrics.json,best.pt}`.

- [ ] **Step 1: Write the failing tests** in `tests/hpc/test_mvgcn_isl.py`:

```python
def test_held_out_line_reads_predicted_gene_effect_only():
    measured = pd.DataFrame({"G1": [-1.0, -2.0], "G2": [0.0, 0.5], "G3": [9.0, 9.0]}, index=["FIT", "HELD"])
    predicted = pd.DataFrame({"G1": [-0.3], "G2": [0.1]}, index=["HELD"])
    table = prepare.gene_effect_input(measured, predicted, held_lines=["HELD"])
    assert list(table.columns) == ["G1", "G2"]                     # cut to the predicted genes
    assert table.loc["HELD"].tolist() == [-0.3, 0.1] and table.loc["FIT"].tolist() == [-1.0, 0.0]

def test_node_features_follow_the_official_order_and_zero_fill():
    tables = {"exp": frame(L=[3.0, 0.0]), "mut": frame(L=[1.0, 0.0]), "cnv": frame(L=[1.0, 2.0]),
              "ess": pd.DataFrame({"G1": [-0.5]}, index=["L"])}     # G2 has no GeneEffect
    x = prepare.node_features(tables, "L", pd.Index(["G1", "G2", "G9"]))
    assert x.tolist() == [[3.0, 1.0, 1.0, -0.5], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

def test_line_graph_keeps_edges_between_expressed_genes():
    edges = np.array([[0, 1, 2], [1, 2, 0]])
    assert prepare.line_edges(edges, expressed=np.array([True, True, False])).tolist() == [[0], [1]]
```

  `frame(L=[a, b])` is a test helper: a lines x genes frame with the row `L` and the columns `G1, G2`. The tables
  that `node_features` reads hold the transformed values (TPM, 0 or 1, relative copy number, GeneEffect).
- [ ] **Step 2: Run** `uv run python -m pytest tests/hpc/test_mvgcn_isl.py -q > pytest.txt 2>&1`. Expected: FAIL.
- [ ] **Step 3: Vendor and write `setup.sh`.** Copy `code/` and `LICENSE` of `kunjiefan/MVGCNiSL` at
  `85b7b880be9a86ae0b721bef0b26cc4363026882` to `third_party/mvgcn_isl/`. `setup.sh` (after `hpc/mit4sl/setup.sh`)
  fetches `data/BIOGRID-9606.csv` to `data/sl_baselines/biogrid/` and `data/coexpression_exp_0.5.csv` to
  `data/mvgcn_isl/official/` from that commit, links `third_party/mvgcn_isl/data` to a directory under
  `data/mvgcn_isl/` that holds them (the official loaders read `../data`), builds `.venv-mvgcn`, and ends with an
  import check of `model.GCN_pool` and `torch.cuda.is_available()`.
- [ ] **Step 4: Implement `prepare.py`** (`--out data/mvgcn_isl/benchmark`, `--held-gene-effect`, defaults for the
  DepMap paths) and **`train.py`** (`--data`, `--out`, `--device`, `--epochs 500`).
- [ ] **Step 5: Run the tests.** Expected: PASS. Run `.venv/bin/ruff check hpc/mvgcn_isl tests/hpc/test_mvgcn_isl.py`.
- [ ] **Step 6: Commit.** `git commit -m "feat(sl): MVGCN-iSL harness, one model over the fit lines with predicted GeneEffect in held-out lines"`

---

### Task 3: SLMGAE

**Files:**
- Create: `third_party/slmgae/`, `hpc/slmgae/setup.sh`, `requirements.txt`, `go_similarity.py`, `prepare.py`, `train.py`
- Modify: `.gitignore` (add `/.venv-slmgae/`)
- Test: `tests/hpc/test_slmgae_go.py`

**Official code used:** `code/models.py: SLMGAE`; `code/objective.py: Optimizer`; `code/utils.py: preprocess_graph,
sparse_to_tuple, construct_feed_dict, load_dense_feature`. Official flags (`train.py`): `learning_rate` 0.001,
`epochs` 300, `hidden1` 512, `hidden2` 256, `nn_size` 45, `dropout` 0.2, `Alpha` 2.0, `Coe` 2.0, `Beta` 4.0.

**Decisions:**
- Environment `.venv-slmgae`: the pins of `hpc/kg4sl/requirements.txt` plus `scipy==1.13.1`. CPU (the host has
  1.5 TB RAM). Local change to the vendored files, listed in `VENDORED.md`: `import tensorflow as tf` becomes
  `import tensorflow.compat.v1 as tf` in `code/{utils,layers,models,objective,metrics,inits}.py`. `train.py` calls `tf.disable_v2_behavior()` and defines
  the official flags with the official defaults before it builds the model.
- The two released GO matrices are Git LFS files (109 MB each), needed only for the telemetry below. `setup.sh`
  fetches them to `data/slmgae/official/` from
  `https://media.githubusercontent.com/media/DiNg1011/SLMGAE/master/data/<name>` and checks SHA-256: `Human_GOsim.txt` `dc9b077133c9f468fda94ba0c8cfc1fcc8c05ddd04d7c69de426e4dd6d7b1a1b`,
  `Human_GOsim_CC.txt` `b227e4a64e2056f355b212aecd569a738278e38c562c0eec544d8c71efe527ad`. If that host does not
  answer on the H20, copy the two files from the laptop with the same check.
- GO inputs: `data/sl_baselines/go/goa_human.gaf.218.gz` (on the H20) and `go-basic.obo` of the same month from
  `https://release.geneontology.org/2023-07-27/ontology/go-basic.obo` (the H20 reaches it), saved to
  `data/sl_baselines/go/go-basic.2023-07-27.obo`; its SHA-256 goes to `metrics.json`.
- Wang's measure (`go_similarity.py`). Annotations: GAF rows of the aspect (`P` or `C`) without a `NOT` qualifier.
  Edge weights: `is_a` 0.8, `part_of` 0.6. For a term A with ancestor set T_A (A included):
  `S_A(A) = 1`, `S_A(t) = max over children t' of t in T_A of w(t', t) * S_A(t')`, `SV(A) = sum of S_A`.
  `sim(A, B) = sum over t in T_A ∩ T_B of (S_A(t) + S_B(t)) / (SV(A) + SV(B))`. Two genes with term sets of sizes m
  and n: best-match average `(sum_i max_j sim_ij + sum_j max_i sim_ij) / (m + n)`. A gene without a term has
  similarity 0 to every gene. Write it with sparse matrices (term x ancestor S-values; gene x term), not with a loop
  over gene pairs: 24,000 genes give 2.9e8 pairs.
- `knn_graph(sim, k=45)`: the official `build_KNN_mateix` then `array2coo(t=0)` then symmetric, vectorized (the
  official loops take 5.8e8 Python steps at this size).
- Nodes: the benchmark genes that have a GO-BP term, a GO-CC term, a BioGRID physical edge or a fit SL pair. Views in
  the official order: GO-BP graph, GO-CC graph, PPI graph (physical rows of
  `data/sl_baselines/biogrid/BIOGRID-9606.csv`, which this `setup.sh` also fetches, from `kunjiefan/MVGCNiSL` at
  `85b7b880`; then `drop_pair_edges`), then
  the main view: the fit pairs with `label == 1`, symmetric. Node features: the main view (the official rule).
- Loss index (the official `train_index`, as an array, not a Python set): every pair `i < j` of nodes but the pairs
  that only held-out lines tested. An untested pair counts as non-SL, as in the official code.
- One fit, 300 epochs, the last epoch (the official script has no validation). Every 25 epochs write the AUROC on
  `pair_labels(rows, ("val",))` to the curve in `metrics.json` (telemetry). `np.random.seed(0)`,
  `tf.set_random_seed(0)`, 64 intra-op threads.
- Score of a pair: the mean of `adj_rec[i, j]` and `adj_rec[j, i]`. A pair with a gene outside the nodes is not covered.
- Telemetry in `metrics.json`: for the released 6,375 genes, the mean Jaccard index between a gene's neighbours in
  our k = 45 graph and in the official one (the official `load_dense_feature` with `knn=True, nnSize=45` on the
  fetched matrix), for BP and for CC; node and edge counts of each view; seconds per epoch.

**Interfaces:**
- Consumes: `sl_common.benchmark_rows, pair_labels, drop_pair_edges, write_pair_scores`.
- Produces: `go_similarity.py`: `read_obo(path) -> dict[str, list[tuple[str, float]]]` (term to weighted parents);
  `term_similarity(parents, terms: list[str]) -> np.ndarray`; `gene_similarity(term_sim: np.ndarray,
  gene_terms: scipy.sparse.csr_matrix) -> np.ndarray`; `knn_graph(sim: np.ndarray, k: int) -> scipy.sparse.coo_matrix`.
  `prepare.py` writes `data/slmgae/benchmark/{genes.json,views.npz,pairs.parquet}`; `train.py` writes
  `outputs/sl/slmgae/seed0/{scores.parquet,metrics.json}`.

- [ ] **Step 1: Write the failing tests** in `tests/hpc/test_slmgae_go.py`:

```python
PARENTS = {"R": [], "A": [("R", 0.8)], "B": [("R", 0.8)], "C": [("A", 0.8)]}   # all is_a

def test_wang_term_similarity_on_a_small_graph():
    sim = go.term_similarity(PARENTS, ["A", "B", "C"])
    assert sim[0, 1] == pytest.approx(1.6 / 3.6)          # A, B share R: (0.8 + 0.8) / (1.8 + 1.8)
    assert sim[0, 2] == pytest.approx(3.24 / 4.24)        # A, C share A and R
    assert np.allclose(np.diag(sim), 1.0) and np.allclose(sim, sim.T)

def test_gene_similarity_is_the_best_match_average():
    sim = go.term_similarity(PARENTS, ["A", "B", "C"])
    gene_terms = scipy.sparse.csr_matrix(np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]))  # g0 {A}, g1 {B, C}, g2 {}
    out = go.gene_similarity(sim, gene_terms)
    assert out[0, 1] == pytest.approx((3.24 / 4.24 + 1.6 / 3.6 + 3.24 / 4.24) / 3)
    assert out[2].tolist() == [0.0, 0.0, 0.0]

def test_part_of_weighs_0_6(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text("[Term]\nid: GO:1\n\n[Term]\nid: GO:2\nis_a: GO:1 ! x\n\n"
                   "[Term]\nid: GO:3\nrelationship: part_of GO:1 ! x\n\n[Term]\nid: GO:4\nis_obsolete: true\n")
    assert go.read_obo(obo) == {"GO:1": [], "GO:2": [("GO:1", 0.8)], "GO:3": [("GO:1", 0.6)]}

def test_knn_graph_equals_the_official_construction():
    rng = np.random.default_rng(0)
    sim = rng.random((30, 30)); sim = (sim + sim.T) / 2; np.fill_diagonal(sim, 0)
    top = np.zeros_like(sim)
    for i in range(30):                                   # official build_KNN_mateix
        keep = np.argsort(sim[i])[::-1][:5]; top[i, keep] = sim[i, keep]
    expected = ((top + top.T) > 0).astype(float)          # official array2coo, then symmetric
    assert np.array_equal(go.knn_graph(sim, 5).toarray() > 0, expected > 0)
```

- [ ] **Step 2: Run** `uv run python -m pytest tests/hpc/test_slmgae_go.py -q > pytest.txt 2>&1`. Expected: FAIL.
- [ ] **Step 3: Implement `go_similarity.py`.** Run the tests. Expected: PASS.
- [ ] **Step 4: Vendor and write `setup.sh`.** Copy `code/` and `README.md` of `DiNg1011/SLMGAE` at
  `ba0c7016922fbec569f8adda5153c28a5392ec58` to `third_party/slmgae/` (the repository has no licence file: say so in
  `VENDORED.md`) and make the `compat.v1` change. `setup.sh` fetches and checks the two GO matrices, the GO ontology
  file and the BioGRID table, builds `.venv-slmgae`, and ends with an import check of `models.SLMGAE`.
- [ ] **Step 5: Implement `prepare.py`** (`--out data/slmgae/benchmark`) and **`train.py`** (`--data`, `--out`,
  `--epochs 300`, `--check-official-go`).
- [ ] **Step 6: Run the tests and ruff.** Expected: PASS, no finding.
- [ ] **Step 7: Commit.** `git commit -m "feat(sl): SLMGAE harness with GO and PPI views for the benchmark genes"`

---

### Task 4: ESM4SL

**Files:**
- Create: `hpc/esm4sl/setup.sh`, `requirements.txt`, `prepare.py`, `collect.py`
- Modify: `.gitignore` (add `/.venv-esm4sl/`)
- Test: `tests/hpc/test_esm4sl_prepare.py`

**Official code used, unchanged, through its command line:** `data_preprocess/esm2_gen.py` (per-residue ESM-2 650M,
layer 33, one `<gene id>.pt` for each gene) and `train.py --config-file esm4sl/configuration/attn/new.yaml` (fit,
checkpoint with the best `(AUROC + AUPRC) / 2` on the validation file, then a test pass that writes
`csv_log/version_<n>/test_logits.csv` with `gene1, gene2, probs`). Official settings (`new.yaml`): batch 16, base
learning rate 5e-5, 100 epochs, class-balanced sampler, 2,000 residues at most.

**Decisions:**
- Environment `.venv-esm4sl`, Python 3.10: torch 2.6.0 (CUDA 12.4 index, as `hpc/mit4sl/requirements.txt`) and the
  packages of the official `environment.sh` (`pytorch-lightning`, `fair-esm`, `omegaconf`, `fvcore`, `easydict`,
  `transformers`, `peft`, `biopython`, `scikit-learn`, `pandas`, `scipy`, `tensorboard`, `wandb`, `matplotlib`,
  `seaborn`). Write the versions that `uv` resolves into `requirements.txt`.
- ESM-2 weights: the H20 gets HTTP 403 from `dl.fbaipublicfiles.com`. Download on the laptop
  `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt` and
  `https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt`, copy them to
  `/2023533015/.cache/torch/hub/checkpoints/` with a SHA-256 check, delete the laptop copies, and run with
  `TORCH_HOME=/2023533015/.cache/torch`.
- Sequences: one table from `https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&format=tsv&fields=accession,gene_primary,sequence`
  (the H20 reaches it), saved to `data/esm4sl/uniprot_human_reviewed.tsv`; SHA-256 and date go to `metrics.json`.
  A gene symbol takes its longest reviewed sequence, cut to its first 2,000 residues. A benchmark gene without one
  is not covered.
- Gene id: the position of the symbol in the sorted covered symbols. `prepare.py` writes under `data/esm4sl/`:
  `mapping/all_id_seq.pkl` (a list of `(id, sequence)`, the input of `esm2_gen.py --root data/esm4sl`), `genes.csv`
  (`id, symbol`), and `benchmark/{sl_train.csv,sl_val.csv,sl_test.csv}` with the official columns `0, 1, 2`:
  - train: `balanced(pair_labels(rows, FIT_SIDES))`, seed 0 (34,000 rows give about 2,100 steps for each epoch;
    all 867,801 pairs would give 5.4 million steps in 100 epochs);
  - validation: `balanced(pair_labels(rows, ("val",)))`, seed 0;
  - test: every covered pair of the `train`, `val` and `test` rows, with `pair_labels(rows, SCORED_SIDES)` as the
    label column (the official test pass needs both classes; these labels are not used for any choice).
- `collect.py` reads the newest `test_logits.csv`, maps ids to symbols and calls `write_pair_scores`.

**Interfaces:**
- Consumes: `sl_common.benchmark_rows, pair_labels, balanced, write_pair_scores`.
- Produces: `prepare.py`: `gene_sequences(table: pd.DataFrame, symbols: pd.Index, limit: int = 2000) -> pd.Series`
  (symbol to sequence); `pair_file(pairs: pd.DataFrame, ids: pd.Series) -> pd.DataFrame` (columns `"0", "1", "2"`).
  `collect.py --run outputs/sl/esm4sl/seed0 --out outputs/sl/esm4sl/seed0` writes `scores.parquet`, `metrics.json`.

- [ ] **Step 1: Write the failing tests** in `tests/hpc/test_esm4sl_prepare.py`:

```python
def test_gene_takes_its_longest_reviewed_sequence_cut_to_the_limit():
    table = pd.DataFrame({"Entry": ["P1", "P2", "P3"], "Gene Names (primary)": ["TP53", "TP53", "BRCA1"],
                          "Sequence": ["MEE", "MEEPQ", "MDL"]})
    out = prepare.gene_sequences(table, pd.Index(["BRCA1", "NOGENE", "TP53"]), limit=4)
    assert out.to_dict() == {"BRCA1": "MDL", "TP53": "MEEP"}

def test_pair_file_has_the_official_columns_and_drops_uncovered_pairs():
    pairs = pd.DataFrame({"gene_1": ["A", "A"], "gene_2": ["B", "Z"], "label": [1, 0]})
    out = prepare.pair_file(pairs, pd.Series({"A": 0, "B": 1}))
    assert out.to_dict("list") == {"0": [0], "1": [1], "2": [1]}
```

- [ ] **Step 2: Run** `uv run python -m pytest tests/hpc/test_esm4sl_prepare.py -q > pytest.txt 2>&1`. Expected: FAIL.
- [ ] **Step 3: Vendor and write `setup.sh`.** Copy `esm4sl/`, `coach_pl/`, `data_preprocess/esm2_gen.py`,
  `train.py`, `test.py` and `LICENSE` of `JieZheng-ShanghaiTech/ESM4SL` at
  `348b2bf9a03f89b8a80d9becb329522a5bb13053` to `third_party/esm4sl/` (no `data/`). `setup.sh` builds `.venv-esm4sl`
  and ends with an import check (`import esm, esm4sl, pytorch_lightning`, working directory `third_party/esm4sl`).
- [ ] **Step 4: Implement `prepare.py` and `collect.py`.** Run the tests and ruff. Expected: PASS, no finding.
- [ ] **Step 5: Commit.** `git commit -m "feat(sl): ESM4SL harness, official CLI on the benchmark pairs"`

---

### Task 5: MuSL

**Files:**
- Create: `hpc/musl/setup.sh`, `requirements.txt`, `prepare.py`, `train.py`
- Modify: `.gitignore` (add `/.venv-musl/`)
- Test: `tests/hpc/test_musl_prepare.py`

**Official code used:** `src/module.py: MuSL, EdgeDataset, train_step`; `src/utils.py: customAdamW,
calculate_features, get_normalize_params, normalize_with_params`; the imputation steps of
`prepare_single_cell_data.ipynb`. Official cell-line command (README): learning rate 1e-4, weight decay 1e-3,
batch 512, patience 10, `--use_cnn --use_gnn --use_stat --cal_state --use_custom_adamw --use_pre_embeddings`;
defaults of `get_default_config`: `hidden_dim` 128, `unified_dim` 256, `target_node_dim` 1024, 32 bins, 100 epochs,
`ReduceLROnPlateau` with patience 3; loss `CrossEntropyLoss`; early stop on the AUPR of `final_pred`.

**Decisions:**
- Environment `.venv-musl`, Python 3.10: torch 2.6.0 (CUDA 12.4), torch-geometric 2.6.1, and the packages of
  `environment.yml` that `train_model.py`, `src/` and the notebook import (`scanpy`, `magic-impute`, `scprep`,
  `scikit-learn`, `pandas`, `scipy`, `wandb`). Write the resolved versions into `requirements.txt`.
- Released node embeddings: `protein_embeddings.pt` (411 MB) of Zenodo record 17098066. The H20 does not reach
  Zenodo: download on the laptop, copy to `data/musl/official/` with a SHA-256 check (record it in `metrics.json`),
  delete the laptop copy. A node without an embedding gets a zero vector; the count goes to `metrics.json`.
- PPI graph: the released `merged_ppi_gene_names.csv` (`gene1, gene2`) in `data/musl/official/`, then `drop_pair_edges`.
- Nodes: the sorted benchmark genes of the `train`, `val` and `test` rows.
- Line matrix (genes x cells), from `data/sl_dependency_v0/processed/basal_raw_umi/<ModelID>.h5ad` (raw UMI,
  `var["gene_symbol"]`): draw at most 2,000 cells (`np.random.default_rng(0)`), then the notebook's steps:
  `scprep.filter.filter_library_size(cutoff=1500)`, `scprep.normalize.library_size_normalize`,
  `scprep.transform.sqrt`, `magic.MAGIC().fit_transform`. If fewer than 10 cells pass the library filter, the line
  skips that filter. MAGIC's default PCA has 100 components: a line with 100 cells or fewer runs
  `magic.MAGIC(n_pca=None)`; the count of such lines goes to `metrics.json`. A gene that the h5ad lacks is a zero row (the official `get_cell_expression`). The matrix is
  saved as `data/musl/benchmark/lines/<ModelID>.npy`.
- Pooled input (control): 2,000 cells drawn evenly from the `train` lines' imputed matrices, seed 0, one matrix.
- Samples. Fit: for each `train` line, its genetic SL rows and as many of its non-SL rows (drawn once, seed 0).
  Validation: the same rule on each validation line. Scored: every row of the `val` and `test` lines. The training
  lines are not scored (their 1.4 million rows cost hours of feature time); the harness then leaves this scorer out
  of the training columns.
- 35 statistical features: the official `calculate_features` on a line's matrix and row pairs, in chunks of 20,000
  rows; `torch.nan_to_num`; normalization parameters from the fit rows (`get_normalize_params`), applied to all rows
  (`normalize_with_params`). For the pooled control, features of each distinct held-out pair on the pooled matrix.
- Model: `MuSL(num_nodes, node_embedding_dim, hidden_dim=128, target_node_dim=1024, node_embeddings, unified_dim=256,
  enable_cross_attention=False, enable_adaptive_fusion=False, enable_contrastive_learning=False, use_cnn=True,
  use_gnn=True, use_stat=True)`. One `EdgeDataset(edges, labels, exp_data, state, bins=32)` for each line, joined
  with `torch.utils.data.ConcatDataset`; the official `train_step`; `customAdamW`. Each epoch reads the validation
  AUPR of `softmax(final_pred)[:, 1]` over all validation samples; stop after 10 epochs without a gain; keep the
  best epoch's weights. `torch.manual_seed(0)`.
- Score: `softmax(final_pred)[:, 1]`.

**Interfaces:**
- Consumes: `sl_common.benchmark_rows, drop_pair_edges, write_line_scores`.
- Produces: `prepare.py`: `impute(counts: pd.DataFrame, rng, max_cells: int = 2000, min_cells: int = 10) ->
  pd.DataFrame` (cells x genes); `line_matrix(imputed: pd.DataFrame, genes: pd.Index) -> np.ndarray` (genes x cells,
  float32); `line_samples(rows: pd.DataFrame, rng) -> pd.DataFrame`. `train.py` writes
  `outputs/sl/musl/seed0/{scores.parquet,scores_pooled_input.parquet,metrics.json,best.pt}`.

- [ ] **Step 1: Write the failing tests** in `tests/hpc/test_musl_prepare.py` (`pytest.importorskip("magic")` for
  the first test only):

```python
def test_a_line_with_18_cells_is_imputed_without_the_library_filter():
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.poisson(0.5, (18, 40)), columns=[f"G{i}" for i in range(40)])  # library size < 1500
    out = prepare.impute(counts, np.random.default_rng(0))
    assert out.shape == (18, 40) and np.isfinite(out.to_numpy()).all()

def test_line_matrix_is_genes_by_cells_with_zero_rows_for_absent_genes():
    imputed = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["G1", "G2"])      # 3 cells
    out = prepare.line_matrix(imputed, pd.Index(["G2", "G9"]))
    assert out.tolist() == [[2.0, 4.0, 6.0], [0.0, 0.0, 0.0]] and out.dtype == np.float32

def test_fit_samples_are_all_sl_rows_and_as_many_non_sl_rows_of_each_line():
    rows = pd.DataFrame({"model_id": ["L1"] * 5 + ["L2"] * 3, "label": [1, 0, 0, 0, 0, 1, 1, 0],
                         "gene_1": list("AAAAABBB"), "gene_2": list("BCDEFCDE")})
    out = prepare.line_samples(rows, np.random.default_rng(0))
    assert out.groupby("model_id")["label"].agg(["sum", "size"]).values.tolist() == [[1, 2], [2, 3]]
```

  (A line with fewer non-SL rows than SL rows keeps all its non-SL rows: `L2` gives 2 SL and 1 non-SL.)
- [ ] **Step 2: Run** `uv run python -m pytest tests/hpc/test_musl_prepare.py -q > pytest.txt 2>&1`. Expected: FAIL.
- [ ] **Step 3: Vendor and write `setup.sh`.** Copy `src/`, `train_model.py`, `prepare_single_cell_data.ipynb` and
  `LICENSE` of `JieZheng-ShanghaiTech/MuSL` at `f8021cfc618fafae8c330b694d0fa7c46db5f1a5` to `third_party/musl/` (no
  `processed_data/`). `setup.sh` fetches `processed_data/merged_ppi_gene_names.csv` from that commit to
  `data/musl/official/`, builds `.venv-musl`, and ends with an import check of `src.module.MuSL` and `magic`.
- [ ] **Step 4: Implement `prepare.py`** (`--out data/musl/benchmark`, `--workers`) and **`train.py`** (`--data`,
  `--out`, `--device`, `--epochs 100`). Run the tests and ruff. Expected: PASS, no finding.
- [ ] **Step 5: Commit.** `git commit -m "feat(sl): MuSL harness, one model over the training lines' imputed single cells"`

---

### Task 6: PromptCASL

**Files:**
- Create: `third_party/promptcasl/`, `hpc/promptcasl/setup.sh`, `requirements.txt`, `prepare.py`, `train.py`
- Modify: `.gitignore` (add `/.venv-promptcasl/`)
- Test: `tests/hpc/test_promptcasl_prepare.py`

**Official code used:** `src/data_prepare.py` (prompt templates), `scripts/build_reactome_pair_evidence.py` and
`src/pair_pathway.py` (pathway evidence), `scripts/train_conve_primekg.py` and `scripts/build_kg_neighbors.py`
(PrimeKG gene vectors and neighbour tables), `src/preprocess_datasets.py`, `src/model.py`, `src/train_model.py`,
`src/main_loco.py`. Official leave-one-cell-line-out command (README):
`python -m src.main_loco --loco_mode full --target_cell_line <line> --omics_mode pair_pathway_evidence --device 0`
with the defaults of `main_loco.py` (batch 24, 50 epochs, early-stop patience 6, `pathway_topk` 16).

**Decisions:**
- Vendor `src/`, `scripts/`, `data/build_splits.py`, `env.yaml` and `LICENSE` at
  `afcf7e9221f873a6f13e4db8edb570c7810f9bc9`; no `ana/`, no released `data/` files. Environment `.venv-promptcasl`,
  Python 3.10: torch 2.6.0 (CUDA 12.4) and the packages of `env.yaml` that `src/` and `scripts/` import.
- The model reads no GeneEffect. Its line inputs are the three tables of `PROMPTCASL_CCLE_DIR`, which `prepare.py`
  builds for the fit and held-out lines in the official file format (`GE.csv`, `CNGene.csv`, `MU.csv`,
  `name2ens.json`): expression from the DepMap table, or the line's pseudo-bulk for the 8 lines that the table lacks
  (the rule of Task 2); copy number and mutations from the DepMap tables. A line that the copy-number or the
  mutation table lacks keeps that value missing. Read `src/data_prepare.py` and use the released prompt version that
  writes a gene's prompt when its copy number or mutation is missing; name it in the docstring and in `metrics.json`.
- The line description of the prompt follows the official format, from `Model.csv` (line name and disease).
- External files (the H20 does not reach Zenodo): GenePT gene embeddings
  (`GenePT_gene_protein_embedding_model_3_text.pickle`), from the laptop with a SHA-256 check; PrimeKG triples (the
  MiT4SL data under `data/mit4sl/official/` hold PrimeKG; else the laptop); the Reactome gene sets of MSigDB C2:CP;
  BiomedBERT from `hf-mirror.com`. All under `data/promptcasl/official/`; paths go through the official
  `PROMPTCASL_*` environment variables.
- Labels: the official split tables are replaced by ours in the official column format: fit rows = for each fit
  line its genetic SL rows and as many of its non-SL rows (seed 0; the official balanced setting); validation rows
  = the same rule on the validation lines; scored rows = every row of the `val` and `test` lines. PrimeKG edges
  inside a benchmark pair are removed (`drop_pair_edges`) before the neighbour tables are built.
- One model: the official training loop on the fit rows, the checkpoint chosen on the validation rows by the
  official monitor, then scores for the scored rows. If the official dataset step stores language-model token
  embeddings for each row, the scored rows are built and scored in chunks that fit on disk; the training lines are
  not scored.
- Mean input: each held-out line reads the fit lines' mean expression and copy number, no mutation and a line
  description without name and disease; the same weights; `scores_mean_input.parquet`.

**Interfaces:**
- Consumes: `sl_common.benchmark_rows, drop_pair_edges, write_line_scores`.
- Produces: `prepare.py`: `omics_tables(tables: dict[str, pd.DataFrame], lines: list[str]) -> dict[str,
  pd.DataFrame]` (the three official tables; a missing value stays `NaN`); `label_table(rows: pd.DataFrame) ->
  pd.DataFrame` (the official columns of `data/folds_assignment/*_full.csv`). `train.py` writes
  `outputs/sl/promptcasl/seed0/{scores.parquet,scores_mean_input.parquet,metrics.json}`.

- [ ] **Step 1: Write the failing tests** in `tests/hpc/test_promptcasl_prepare.py`: a line without copy number
  keeps `NaN` in `CNGene.csv` and still gets a prompt for each gene of its rows (call the vendored verbalizer; skip
  with `importorskip` when a package is absent); `label_table` writes `SL` / `Not SL` in `SL_or_not` and
  `gene_1|gene_2` in `gene_pair`; a line without DepMap expression reads its pseudo-bulk.
- [ ] **Step 2: Run** `uv run python -m pytest tests/hpc/test_promptcasl_prepare.py -q > pytest.txt 2>&1`. Expected: FAIL.
- [ ] **Step 3: Vendor and write `setup.sh`** (builds `.venv-promptcasl`, fetches BiomedBERT, import check of
  `src.model` with the working directory `third_party/promptcasl`).
- [ ] **Step 4: Implement `prepare.py` and `train.py`.** Run the tests and ruff. Expected: PASS, no finding.
- [ ] **Step 5: Commit.** `git commit -m "feat(sl): PromptCASL harness, one model over the fit lines with the benchmark lines' omics tables"`

---

### Task 7: Harness names, config and the KG4SL fit lines

**Files:**
- Modify: `src/experiments/sl/benchmark_baselines.py` (`NAMES`, after the `"mit4sl"` entry)
- Create: `configs/sl/benchmark_published.yaml`
- Test: `tests/experiments/sl/test_benchmark_baselines.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_published_baseline_config_names_known_scorers():
    config = benchmark_baselines.load_config(REPO_ROOT / "configs/sl/benchmark_published.yaml")
    assert {"slmgae", "esm4sl", "mvgcn_isl", "mvgcn_isl_mean_input", "musl", "musl_pooled_input",
            "promptcasl", "promptcasl_mean_input"} <= set(config["extra_scores"])
    assert "paris" not in config["extra_scores"]
    assert ["mvgcn_isl", "mvgcn_isl_mean_input"] in config["paired_differences"]
```

- [ ] **Step 2: Run it.** Expected: FAIL (no config file).
- [ ] **Step 3: Add the names:**

```python
    "slmgae": "SLMGAE (multi-view graph autoencoder, official code, trained on the fit lines)",
    "esm4sl": "ESM4SL (ESM-2 protein language model, official code, trained on the fit lines)",
    "mvgcn_isl": "MVGCN-iSL (multi-view graph network on line omics, stage one's predicted "
    "GeneEffect in the held-out lines, official code, one model over the fit lines)",
    "mvgcn_isl_mean_input": "MVGCN-iSL, context-ablated (the fit lines' mean node features)",
    "musl": "MuSL (pair expression histograms from the line's single cells, official code, "
    "one model over the training lines)",
    "musl_pooled_input": "MuSL, context-ablated (cells pooled over the training lines)",
    "promptcasl": "PromptCASL (BiomedBERT prompt of the line's omics and pathway evidence, official "
    "code, one model over the fit lines)",
    "promptcasl_mean_input": "PromptCASL, context-ablated (the fit lines' mean omics, no line description)",
```

  Change the `"kg4sl"` name to end with "trained on the fit lines".

- [ ] **Step 4: Write `configs/sl/benchmark_published.yaml`:** the fixed keys of `configs/sl/benchmark_baselines.yaml`
  (`stage_one: outputs/sl/conditional/conditional_seed0/predictions.parquet`), and
  - `extra_scores`: `kg4sl` (`outputs/sl/kg4sl/fit_lines_seed0/scores.parquet`), `cilantro_sl`, `mit4sl`,
    `kebabci_paralog_predicted` (paths as in `benchmark_baselines.yaml`); `promptcasl` and `promptcasl_mean_input`
    (`outputs/sl/promptcasl/seed0/scores.parquet`, `scores_mean_input.parquet`); `slmgae: outputs/sl/slmgae/seed0/scores.parquet`; `esm4sl:
    outputs/sl/esm4sl/seed0/scores.parquet`; `mvgcn_isl` and `mvgcn_isl_mean_input`
    (`outputs/sl/mvgcn_isl/seed0/scores.parquet`, `scores_mean_input.parquet`); `musl` and `musl_pooled_input`
    (`outputs/sl/musl/seed0/scores.parquet`, `scores_pooled_input.parquet`); `cross_joint:
    outputs/sl/embedding/cross_line_joint_seed0_select/scores/embedding_line_term.parquet` (our best row);
  - `paired_differences`: `[mvgcn_isl, mvgcn_isl_mean_input]`, `[musl, musl_pooled_input]`,
    `[promptcasl, promptcasl_mean_input]`, and `[cross_joint, X]` for `X` in `slmgae, esm4sl, mvgcn_isl, musl,
    promptcasl, mit4sl, kebabci_paralog_predicted, kg4sl`;
  - `scorer_means: {}`; `cross_line_ordering: [mvgcn_isl, mvgcn_isl_mean_input, musl, musl_pooled_input, promptcasl,
    promptcasl_mean_input, mit4sl, kebabci_paralog_predicted, cross_joint]`.
- [ ] **Step 4b: KG4SL on the fit lines.** In `hpc/kg4sl/train.py`, `benchmark_rows` keeps the lines of
  `train`, `bulk_labelled`, `val` and `test`, and `labelled_pairs("train")` reads the rows of `train` and
  `bulk_labelled`; the scored pairs stay those of `train`, `val` and `test`. Update its docstring.
- [ ] **Step 5: Run** the test file and the full suite to `pytest.txt`; compare with `pytest_baseline.txt`.
  Expected: the new tests pass; no new failure. Run `.venv/bin/ruff check .`
- [ ] **Step 6: Commit and push the branch.** `git commit -m "feat(sl): harness config of the published baselines"`,
  `git push -u origin published-baselines` (the push before GPU work). The merge into `main` is Task 9, after the
  H20 runs.

---

### Task 8: Runs on the H20

**Execution (2026-11-26):** one subagent for each of Tasks 2 to 6, in parallel, each on its own branch
`published-baselines-<model>` and its own Git worktree. Each makes its harness run on the H20 in its own clone
`/2023533015/VCC_baselines/<model>` (links to the main checkout's `data/` and `outputs/`), with the short run of its
block below, and pushes its branch. The primary agent merges the five branches, does Task 7, and starts the full
runs in the main checkout.

One container, port 30757, `cd /2023533015/VCC_Project && hpc/pull.sh published-baselines`; read `git status --short`
first and keep the host's uncommitted changes; confirm the printed commit. Copy the
data files from the laptop first (ESM-2 weights, `protein_embeddings.pt`; the GO matrices if `setup.sh` could not
fetch them). Times are estimates; measure with the short run named in each block before the full run.

- [ ] **Step 1: MVGCN-iSL** (GPU 0, about 1 h).

```bash
hpc/mvgcn_isl/setup.sh
PYTHONPATH=third_party/mvgcn_isl/code MPLBACKEND=Agg .venv-mvgcn/bin/python hpc/mvgcn_isl/prepare.py \
  --out data/mvgcn_isl/benchmark --held-gene-effect outputs/sl/conditional/conditional_seed0/predictions.parquet
PYTHONPATH=third_party/mvgcn_isl/code OMP_NUM_THREADS=16 .venv-mvgcn/bin/python hpc/mvgcn_isl/train.py \
  --data data/mvgcn_isl/benchmark --out outputs/sl/mvgcn_isl/seed0 --device 0
```

  Expected: `scores.parquet`, `scores_mean_input.parquet`, `metrics.json` with `held_rows_covered_share` and the
  expression source of each line. Short run first: `--epochs 2`.
- [ ] **Step 2: SLMGAE** (CPU, after Step 1's `setup.sh`).

```bash
hpc/slmgae/setup.sh
.venv-slmgae/bin/python hpc/slmgae/prepare.py --out data/slmgae/benchmark
PYTHONPATH=third_party/slmgae/code .venv-slmgae/bin/python hpc/slmgae/train.py \
  --data data/slmgae/benchmark --out outputs/sl/slmgae/seed0 --check-official-go
```

  Short run first: `--epochs 2`; read seconds per epoch and peak memory (`/usr/bin/time -v`) before the 300 epochs.
- [ ] **Step 3: ESM4SL** (GPU 1 for the embeddings, then training; test pass on 1.1 million pairs at batch 16).

```bash
hpc/esm4sl/setup.sh
.venv-esm4sl/bin/python hpc/esm4sl/prepare.py --out data/esm4sl
cd third_party/esm4sl && TORCH_HOME=/2023533015/.cache/torch ../../.venv-esm4sl/bin/python \
  data_preprocess/esm2_gen.py --root ../../data/esm4sl --gpu_id 1 --batch_size 4
CUDA_VISIBLE_DEVICES=1 ../../.venv-esm4sl/bin/python train.py --config-file esm4sl/configuration/attn/new.yaml \
  --num-gpus 1 SEED 0 OUTPUT_DIR ../../outputs/sl/esm4sl/seed0 \
  DATASET.ESM_ROOT ../../data/esm4sl/pretrained_emb/esm2 \
  DATASET.TRAIN_FILE ../../data/esm4sl/benchmark/sl_train.csv \
  DATASET.VAL_FILE ../../data/esm4sl/benchmark/sl_val.csv \
  DATASET.TEST_FILE ../../data/esm4sl/benchmark/sl_test.csv
cd ../.. && .venv-esm4sl/bin/python hpc/esm4sl/collect.py --run outputs/sl/esm4sl/seed0 --out outputs/sl/esm4sl/seed0
```

  Short run first: `TRAINER.MAX_EPOCHS 1` with the first 2,000 rows of each file; read steps per second. If 100
  epochs need more than 24 h, use `--num-gpus 4` (the official option; the learning rate follows the device count).
- [ ] **Step 4: MuSL** (CPU for `prepare.py`, GPU 2 for training).

```bash
hpc/musl/setup.sh
OMP_NUM_THREADS=1 .venv-musl/bin/python hpc/musl/prepare.py --out data/musl/benchmark --workers 64
PYTHONPATH=third_party/musl OMP_NUM_THREADS=16 .venv-musl/bin/python hpc/musl/train.py \
  --data data/musl/benchmark --out outputs/sl/musl/seed0 --device 2
```

  Short run first: `prepare.py --lines 3`, `train.py --epochs 1`.
- [ ] **Step 5: PromptCASL** (GPU 3).

```bash
hpc/promptcasl/setup.sh
.venv-promptcasl/bin/python hpc/promptcasl/prepare.py --out data/promptcasl/benchmark
OMP_NUM_THREADS=16 .venv-promptcasl/bin/python hpc/promptcasl/train.py \
  --data data/promptcasl/benchmark --out outputs/sl/promptcasl/seed0 --device 3
```

  Short run first: `prepare.py --lines 3`, `train.py --epochs 1`.
- [ ] **Step 5b: KG4SL on the fit lines** (about 1 min).
  `PYTHONPATH=third_party/kg4sl/src .venv-kg4sl/bin/python hpc/kg4sl/train.py --out outputs/sl/kg4sl/fit_lines_seed0`
- [ ] **Step 5c: Harness.**
  `hpc/run.sh sl-baselines configs/sl/benchmark_published.yaml --run-id benchmark_published_seed0`
  Expected: `outputs/sl/benchmark_baselines/benchmark_published_seed0/` with `benchmark_results.md` and
  `benchmark_metrics.json`; a table without a row for a held row stops the run with the path of that table.
- [ ] **Step 6: Record the fixes.** Each change that a run needed (a version pin, a shim) is committed on the laptop,
  pushed and pulled; no edit on the H20 checkout.

---

### Task 9: Report and documents

- [ ] **Step 1: Run record.** Copy the harness report files to `results/sl_published_baselines_seed0/host/` (local,
  not tracked) and write `results/sl_published_baselines_seed0/summary.html` with the `results-report` skill: models as
  rows; recall in the top 10%, recall at 10% false positives, AUROC, chance-corrected AUPRC, cross-line ordering, own
  minus control input; validation and test; coverage of each model; the SLMGAE GO agreement.
- [ ] **Step 2: `docs/04-sl-ranking-protocol.md`:** a new section "Published baselines, second set (2026-11-26)" with
  the table of section "Design" above, the settings, the commit and the results.
- [ ] **Step 3: `hpc/README.md`:** one paragraph and the commands of Task 8 for each of the five harnesses, after the
  MiT4SL paragraph; the data files that come from the laptop with their SHA-256 values.
- [ ] **Step 4: `model-status.html`:** Table 1 and Table 2 "Published baselines": add SLMGAE, ESM4SL, MVGCN-iSL, MuSL,
  PromptCASL and MiT4SL (test metrics only); remove the PARIS and De Kegel rows; KG4SL takes its new fit; update the
  lead and the key figure if the best published baseline changes. `docs/data/sl-benchmark.html`: one sentence that a
  baseline also fits on the 33 labelled bulk-only lines.
- [ ] **Step 5: `CLAUDE.md` (current state) and `README.md`:** the best published baseline and its recall; keep
  `CLAUDE.md` under 100 lines.
- [ ] **Step 6: Suite, lint, commit, push.** Full suite to `pytest.txt`, `.venv/bin/ruff check .`, delete
  `pytest.txt` and `pytest_baseline.txt`, `git commit -m "docs(sl): five more published baselines on the SL benchmark"`,
  `git switch main && git merge --ff-only published-baselines && git push origin main`, then
  `git branch -d published-baselines && git push origin --delete published-baselines`, and `hpc/pull.sh` on the H20.

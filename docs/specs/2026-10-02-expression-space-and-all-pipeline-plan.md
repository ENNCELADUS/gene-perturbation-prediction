# One Expression Space and the `all` Run — Implementation Plan

> **For agentic workers:** executed with superpowers:dispatching-parallel-agents. Each task is owned by
> one agent; an agent edits **only** the files its task lists. Steps use checkbox (`- [ ]`) syntax.
> Easy tasks run on Sonnet 5.5, hard tasks on Opus 5.5.

**Goal:** STATE gets inputs and targets in its own log-normalised space, the joint model runs STATE on
its own HVG basal path, a six-arm response-model comparison measures what STATE adds, and one command
(`hpc/run.sh all`) runs preparation through validation evaluation.

**Architecture:** Preparation writes every expression quantity once, in log space, under one prepared
root; training and evaluation read it unchanged. The joint model is the existing GeneEffect model with
STATE's released basal encoder restored. A new comparison module and a new orchestrator sit on top.
Closed diagnostic harnesses are deleted; the remaining formal path is simplified to six guards.

**Tech stack:** Python 3.11, PyTorch, accelerate, anndata/scanpy, numpy/pandas, arc-state (pinned),
pytest, Ruff. Run everything from the repo root as `uv run python -m …`.

**Spec:** `docs/specs/2026-10-02-expression-space-and-all-pipeline-design.md` — read it first.

## Global constraints

- Tx1 reads raw UMI. Every other expression quantity: `log1p(x * T / L_cell)` then the STATE HVG
  slice, `L_cell` = the cell's UMI sum over **all** genes of its source matrix (not the HVG panel).
- `T` = median `L_cell` over non-targeting cells of Jurkat `ACH-000995` and HepG2 `ACH-000739`
  response sources, hard-coded as those two ModelIDs in `src/experiments/prepare.py`, never a config key.
- STATE's released checkpoint (`basal_encoder` 2000→328, `pert_encoder` 2024→328, `project_out`
  328→2000) loads with **zero** missing, unexpected or shape-skipped keys in the joint model.
- No response-condition holdout anywhere. GeneEffect validation (27 lines) is the only validation split.
- Joint learning rates: head 1e-4, ESM2 adapter 1e-4, STATE 1e-5.
- Tx1 cache on-disk format (`<ModelID>/embeddings.npy`, `hvg.npy`, `obs.parquet`) is read, never rewritten.
- Guards kept (and only these as hard checks): `assert_fit_eligible`; targets on fold-fit mean,
  predictions on fold-independent mean; undefined (NaN) correlations for constant predictors, never 0;
  checkpoint loads raise on zero loaded keys; config rejects unknown keys; `load_inputs` refuses a
  manifest without `expression_space`.
- No `.get(key, default)` config reads in new code. No bare `load_state_dict(..., strict=False)`.
- Prose, comments, docs and commit messages name things descriptively; never `P1-A/B/C`, `V0–V3`,
  `A0–A3`, `Tier n`, or step numbers as names.
- Tests: `uv run python -m pytest tests/<file>.py -q > /tmp/<name>.txt 2>&1; tail -5 /tmp/<name>.txt`
  (the rtk hook corrupts foreground pytest). Lint: `.venv/bin/ruff check <files>`; format only files you
  touched: `.venv/bin/ruff format <files>`. Never `ruff format .`.
- Commits: Conventional Commits, end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Commit only your own files (`git add <paths>`), never `git add -A`.

## Shared contracts (every task codes against these)

### Config `configs/geneeffect_joint.yaml` (owned by Task 6; others read it)

```yaml
seeds: {train: 0, collator: 0, projection: 0}
train:
  max_epochs: 50
  patience: 5
  dependency_batch_size: 1024
  response_batch_size: 64
  response_interval: 4
  response_weight: 1.0
  state_learning_rate: 0.00001
  adapter_learning_rate: 0.0001
  head_learning_rate: 0.0001
  weight_decay: 0.01
comparison:
  epochs: 50
  hidden: 512
  learning_rate: 0.0001
  state_learning_rate: 0.00001
  batch_size: 64
  shuffles: 10
  bootstrap: 1000
precision: bf16
output_root: outputs/geneeffect_joint
prepared_root: data/geneeffect_joint/v2
features: {cells_per_context: 128, hvg_dim: 2000, esm2_dim: 1280, variable_gene_min_observations: 5, variable_gene_percentile: 75}
model:
  cell_sentence_len: 64
  esm2_adapter_hidden: 512
  head_hidden: 256
  head_layers: 2
  head_blocks: {use_delta_proj: true, use_s: true, use_q_sc: true, use_e_g: true, use_z_c: true}
preparation:
  response_max_cells_per_gene: 128
  response_total_cells_per_line: null
  response_sampling_seed: 42
  tx1_batch_size: 32
  tx1_max_length: 2048
  var_ensembl_col: ensembl_id
  hvg_gene_symbol_col: auto
paths:
  split: configs/benchmarks/cell_line_geneeffect_226_split.json
  gene_effect: data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv
  source_registry: configs/experiments/13_geneeffect_226/basal_source_registry.csv
  tx1_registration: configs/benchmarks/provenance/phase_a_tx1_20260724/phase_a_registration.json
  cell_line_manifest: configs/benchmarks/provenance/phase_a_tx1_20260724/cell_line_manifest.csv
  tx1_model_dir: data/models/tahoe_x1_3b/3b-model
  tx1_cache: data/tx1_basal_embeddings/v1
  esm2_embeddings: data/esm2/exp13_gene_universe_esm2_650M.npz
  state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
  state_model_dir: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562
  perturbseq_sources: configs/experiments/13_geneeffect_226/perturbseq_sources.json
```

`selection` is gone (selection on `val_geneeffect_loss` is fixed in code). `q_sc_cache`,
`response_cache` and `common_gene_panel` paths are gone: they live under `prepared_root`.
`src/experiments/config.py`: `load_config(path) -> dict` and `validate_config(config) -> dict` reject
missing and unknown keys only.

### Prepared inputs (`src/data/prepared.py`, owned by Task 5)

```python
@dataclass(frozen=True)
class PreparedLine:
    controls_tx1: np.ndarray   # [cells_per_context, 2560] float32 Tx1 embeddings of the selected basal cells
    basal_hvg: np.ndarray      # [cells_per_context, 2000] float32, log space, STATE HVG order, same cells
    q_sc: QScFeatures          # unchanged class from src/data/q_sc.py; mean/var now in log space

@dataclass(frozen=True)
class PreparedInputs:
    split: FixedSplit
    labels: pd.DataFrame                 # model_id, gene_symbol, gene_effect, residual
    genes: tuple[str, ...]               # common gene panel
    train_gene_means: pd.Series          # indexed by genes, fit on supervised train only
    variable_genes: frozenset[str]
    hvg_order: tuple[str, ...]           # STATE's 2000 HVGs
    esm2_symbols: tuple[str, ...]
    esm2_vectors: np.ndarray
    lines: Mapping[str, PreparedLine]
    response_targets: ResponseTargetsCache   # .keys -> tuple[(model_id, gene)], .target_bag(i) -> [n,2000] log space
    response_anchors: tuple[str, ...]        # 4 ModelIDs
    target_sum: float

    def preprocessing_state(self) -> dict[str, object]: ...  # gene_means, variable_genes, esm2_symbols, esm2_vectors, target_sum

def load_inputs(config, *, preprocessing=None, include_test=False) -> PreparedInputs: ...
```

Anchor control bags are `lines[anchor].basal_hvg`; anchor Tx1 cells are `lines[anchor].controls_tx1`.
Prepared root layout: `prepared_inputs.json` (written last; holds `expression_space =
{"transform": "log1p_normalize_total", "target_sum": T, "library_size": "all_genes",
"target_sum_sources": ["ACH-000995", "ACH-000739"]}`), `common_gene_panel.csv`, `lines/<ModelID>.npz`
(`controls_tx1`, `basal_hvg`, `q_sc_values`, `q_sc_available`), `response/` (targets cache), ESM2 union.

### Expression transform (`src/data/expression.py`, owned by Task 2)

```python
def library_sizes(matrix) -> np.ndarray            # [cells] float64 row sums over all columns; dense or scipy sparse
def log_normalize(matrix, library_size: np.ndarray, target_sum: float) -> np.ndarray  # float32 log1p(x*T/L); rows with L==0 stay 0
def median_library_size(*sizes: np.ndarray) -> float  # median of the concatenation
```

### Model and training (owned by Task 6)

```python
# src/model/state.py
def load_released_state(checkpoint_path: Path, *, cell_set_len: int) -> nn.Module
    # StateTransitionPerturbationModel with every released weight loaded; raises on any missing,
    # unexpected or shape-mismatched key.
class StateResponse(nn.Module):           # STATE driven by ESM2 perturbation tokens
    def __init__(self, state: nn.Module, perturbations: nn.Module): ...
    def forward(self, basal_chunks: tuple[Tensor, ...], genes: tuple[str, ...]) -> tuple[Tensor, ...]
        # each chunk [cell_set_len, 2000] log-space basal cells -> predicted perturbed [cell_set_len, 2000]; batch index 0
# src/model/perturbation.py — Esm2PerturbationAdapter unchanged (symbols, table, hidden, pert_dim=2024)
# src/model/response.py — mean_delta_mse(predicted, observed, control_mean), energy_distance(left, right) unchanged
def response_loss(predicted: Tensor, observed: Tensor, control: Tensor) -> Tensor  # mean_delta_mse + energy_distance
# src/model/geneeffect.py — GeneEffectE2EModel.condition_features(batch) -> FeatureBatch and FeatureBatch unchanged
# src/experiments/geneeffect.py
def run_training(config: Mapping, run_dir: Path) -> Path   # trains (resumes from run_dir/last.pt if present); returns best.pt
def evaluate_checkpoint(checkpoint: Path, *, split: str) -> EvalResult
def export_evaluation(result: EvalResult, out_dir: Path) -> None   # predictions.parquet, metrics.json, per_line.csv, per_gene.csv
def restore_model(saved: Mapping, inputs: PreparedInputs) -> GeneEffectE2EModel   # public (was _restore_model)
```

`src.train` CLI: `python -m src.train --config CONFIG --run-dir DIR` (works under `accelerate launch`
and as a single CPU process). `src.evaluate` CLI unchanged: `--checkpoint`, `--split`.

### Comparison, baselines, readout, orchestrator

```python
# src/model/response_mlp.py (Task 3)
class ResponseMLP(nn.Module):
    def __init__(self, cell_dim: int, gene_dim: int, hidden: int, out_dim: int = 2000): ...
    def forward(self, cells: Tensor, basal_hvg: Tensor, gene: Tensor) -> Tensor
        # cells [n, cell_dim] (Tx1 embedding or log HVG), basal_hvg [n, 2000], gene [gene_dim] ESM2 vector
        # returns basal_hvg + f([cells ; adapter(gene)]); final layer zero-initialised
# src/experiments/response_comparison.py (Task 7)
def run_comparison(config: Mapping, out_dir: Path, *, device: str) -> pd.DataFrame
    # writes sanity.json first, then comparison.csv, curves.csv, verdicts.json; skips if verdicts.json exists
# src/experiments/baselines.py (Task 8)
def run_baselines(config, *, split: str, out_dir: Path) -> R1Result   # unchanged signature
# src/experiments/readout.py (Tasks 1, 8)
def run_readout(checkpoint: Path, out_dir: Path, *, device: str) -> dict[str, float]
    # extracts features, trains the explicit context-slope readout, writes val metrics; skips if done
# src/experiments/all.py (Task 9)
def run_all(config_path: Path, *, run_id: str | None) -> Path   # returns run dir holding summary.md
```

Run directory: `outputs/geneeffect_joint/<run_id>/{comparison/, train/, evaluation/val/,
baselines/val/, readout/, summary.md}`. `hpc/run.sh` commands: `all CONFIG [--run-id ID]` and
`test CHECKPOINT`.

## Review focus

1. A basal source whose matrix is already normalised (non-integer) → preparation must raise, not
   log-transform it twice. Owner: Task 5, test `test_prepare_rejects_non_integer_source`.
2. A cell with zero UMI in the whole library → stays all-zero, no NaN. Owner: Task 2,
   `test_zero_library_row_stays_zero`.
3. A response gene absent from STATE's one-hot vocabulary → the released-checkpoint arm scores only
   covered genes and reports the coverage count, never crashes. Owner: Task 7,
   `test_released_arm_skips_uncovered_genes`.
4. `all` interrupted mid-training → rerun with the same `--run-id` resumes from `train/last.pt`, and
   finished steps are skipped. Owner: Task 9, `test_all_resumes_and_skips`.
5. An old v1 prepared root or seed-0 checkpoint passed to the new code → clear error naming the missing
   `expression_space`, not a silent run. Owner: Task 5, `test_load_inputs_refuses_v1_manifest`.

---

## Wave 1 — five parallel tasks (no shared files)

### Task 1 (easy): delete the closed diagnostic harnesses; relocate the readout entry point

**Files:**
- Delete: `src/experiments/{p1a,p1b,p1b_preparation,p1c,p1c_preparation,profile_joint}.py`,
  `src/data/{p1b,p1c}.py`, `src/model/{p1b,p1c}.py`, `src/training/p1b.py`,
  `src/eval/{p1b,p1b_comparison,p1c,p1c_tier0}.py`, `hpc/p1c_pipeline.sh`,
  `tests/test_p1a.py`, `tests/test_p1b.py`, `tests/test_p1c.py`
- Create: `src/experiments/readout.py`
- Modify: `hpc/run.sh` (remove `p1a|p1b|p1c` from usage, case list and dispatch)

**Interfaces:** Produces `src/experiments/readout.py` with `extract_cache(checkpoint, destination, *,
device="cpu", batch_size=32)`, `iter_features(...)` and a `main()` offering `extract`, `train`,
`evaluate`, `compare` subcommands — moved verbatim from `src/experiments/p1a.py` (Task 8 adapts it).

- [ ] Step 1: `grep -rn "p1a\|p1b\|p1c\|profile_joint" src tests hpc --include=*.py --include=*.sh` and
  list every importer outside the files being deleted. Expected: only `hpc/run.sh` and the readout
  modules' docstrings. If anything else imports them, stop and report it.
- [ ] Step 2: Create `src/experiments/readout.py` with the contents of `src/experiments/p1a.py`; change
  its module docstring to "Readout head with the explicit gene-specific context slope on a frozen joint
  backbone." and replace user-facing arm labels in help text with descriptive names ("shared MLP head",
  "explicit context-slope head") while keeping the arm keys the code uses.
- [ ] Step 3: `git rm` the files listed above; edit `hpc/run.sh`.
- [ ] Step 4: `uv run python -c "import src.experiments.readout"` succeeds; `bash -n hpc/run.sh` succeeds.
- [ ] Step 5: Commit `refactor: delete closed diagnostic harnesses, move readout entry point`.

### Task 2 (easy): expression transform

**Files:** Create `src/data/expression.py`, `tests/test_expression.py`.

**Interfaces:** Produces the three functions in *Shared contracts → Expression transform*.

- [ ] Step 1: Write the failing tests.

```python
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

from src.data.expression import library_sizes, log_normalize, median_library_size


def _toy():
    return np.array([[1, 0, 3, 6], [0, 2, 2, 0], [5, 5, 0, 10]], dtype=np.float32)


def test_matches_scanpy_whole_library_then_hvg_slice():
    x = _toy()
    expected = ad.AnnData(x.copy())
    sc.pp.normalize_total(expected, target_sum=100.0)
    sc.pp.log1p(expected)
    got = log_normalize(x[:, [0, 2]], library_sizes(x), 100.0)
    np.testing.assert_allclose(got, expected.X[:, [0, 2]], rtol=1e-6)


def test_library_size_uses_all_genes_not_the_slice():
    x = _toy()
    np.testing.assert_array_equal(library_sizes(x), [10, 4, 20])
    assert not np.allclose(log_normalize(x[:, :2], library_sizes(x), 10.0),
                           log_normalize(x[:, :2], library_sizes(x[:, :2]), 10.0))


def test_sparse_input_matches_dense():
    x = _toy()
    np.testing.assert_allclose(library_sizes(sparse.csr_matrix(x)), library_sizes(x))
    np.testing.assert_allclose(
        log_normalize(sparse.csr_matrix(x), library_sizes(x), 50.0),
        log_normalize(x, library_sizes(x), 50.0))


def test_zero_library_row_stays_zero():
    x = np.zeros((2, 3), dtype=np.float32)
    x[1] = [1, 1, 2]
    out = log_normalize(x, library_sizes(x), 10.0)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_array_equal(out[0], 0.0)


def test_median_library_size_pools_sources():
    assert median_library_size(np.array([1.0, 3.0]), np.array([10.0])) == 3.0
```

- [ ] Step 2: Run `uv run python -m pytest tests/test_expression.py -q > /tmp/expr.txt 2>&1; tail -5 /tmp/expr.txt`. Expected: import error.
- [ ] Step 3: Implement.

```python
"""STATE's expression space: whole-library normalize_total, then log1p (arc-state preprocess_train)."""

import numpy as np
from scipy import sparse


def library_sizes(matrix) -> np.ndarray:
    """UMI total per cell over every column of the source matrix."""
    return np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()


def log_normalize(matrix, library_size: np.ndarray, target_sum: float) -> np.ndarray:
    """``log1p(x * target_sum / library_size)``; cells with no UMI stay zero."""
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    scale = np.zeros_like(library_size, dtype=np.float64)
    nonzero = library_size > 0
    scale[nonzero] = target_sum / library_size[nonzero]
    return np.log1p(dense * scale[:, None]).astype(np.float32)


def median_library_size(*sizes: np.ndarray) -> float:
    """Median library size over the pooled cells of every given source."""
    return float(np.median(np.concatenate(sizes)))
```

- [ ] Step 4: Rerun; expected 5 passed. Ruff check and format the two files.
- [ ] Step 5: Commit `feat(data): STATE expression space transform`.

### Task 3 (easy): STATE-free response MLP

**Files:** Create `src/model/response_mlp.py`, `tests/test_response_mlp.py`.

**Interfaces:** Produces `ResponseMLP` per *Shared contracts*.

- [ ] Step 1: Failing tests.

```python
import torch

from src.model.response_mlp import ResponseMLP


def test_untrained_model_is_exactly_no_change():
    torch.manual_seed(0)
    model = ResponseMLP(cell_dim=7, gene_dim=5, hidden=16, out_dim=4)
    basal = torch.rand(3, 4)
    out = model(torch.randn(3, 7), basal, torch.randn(5))
    assert torch.equal(out, basal)


def test_gene_changes_output_after_one_step():
    torch.manual_seed(0)
    model = ResponseMLP(cell_dim=7, gene_dim=5, hidden=16, out_dim=4)
    cells, basal = torch.randn(3, 7), torch.rand(3, 4)
    loss = (model(cells, basal, torch.randn(5)) - 1.0).square().mean()
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    a = model(cells, basal, torch.ones(5))
    b = model(cells, basal, -torch.ones(5))
    assert not torch.allclose(a, b)


def test_per_cell_output_shape():
    model = ResponseMLP(cell_dim=7, gene_dim=5, hidden=16, out_dim=4)
    assert model(torch.randn(9, 7), torch.rand(9, 4), torch.randn(5)).shape == (9, 4)
```

- [ ] Step 2: Run; expect import error.
- [ ] Step 3: Implement.

```python
"""STATE-free response model: a per-cell expression shift from the cell and the gene embedding."""

import torch
from torch import nn


class ResponseMLP(nn.Module):
    """``basal + f([cell ; adapter(gene)])``; the last layer starts at zero, so the untrained model is no-change."""

    def __init__(self, cell_dim: int, gene_dim: int, hidden: int, out_dim: int = 2000):
        super().__init__()
        self.gene = nn.Sequential(nn.Linear(gene_dim, hidden), nn.GELU())
        self.shift = nn.Sequential(
            nn.Linear(cell_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
        nn.init.zeros_(self.shift[-1].weight)
        nn.init.zeros_(self.shift[-1].bias)

    def forward(self, cells: torch.Tensor, basal_hvg: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        token = self.gene(gene).expand(cells.shape[0], -1)
        return basal_hvg + self.shift(torch.cat((cells, token), dim=1))
```

- [ ] Step 4: Rerun; expect 3 passed. Ruff.
- [ ] Step 5: Commit `feat(model): STATE-free response MLP`.

### Task 4 (easy): documents

**Files:** Modify `docs/01-blueprint.md`, `docs/03-geneeffect-protocol.md`, `AGENTS.md`, `CLAUDE.md`,
`hpc/README.md`, `docs/literature/notes/01_Papers/03_models/foundation_models/2025_Tahoe-x1_3B_perturbation_FM.md`.

- [ ] Step 1: Blueprint → high-level task description only: keep §1 task, §2 data and generalisation,
  §5 SL composition (shortened to intent), claim boundaries, and a status paragraph linking the protocol
  and results. Move §3 equations, §4 rates/batches/selection, §6 metric table and §7 tables into the
  protocol (they may already exist there; do not duplicate — link). Date line "Updated 2026-10-02".
- [ ] Step 2: Protocol: add §3.4 "Expression space" (the transform, `T` from the Nadig Jurkat/HepG2
  non-targeting median, library size over all genes, Tx1 raw); rewrite §4–§5 for STATE on its own HVG
  basal path, Tx1 to the head only, rates 1e-4/1e-4/1e-5, response as auxiliary on all conditions of all
  four anchors, one validation split; replace §9 with the six-arm response-model comparison and the
  `all` run (link the spec). Keep §8 as the diagnostics record; fix its "§3.4" link to point at the new
  section. Caption of Figure 1 updated to the new wiring.
- [ ] Step 3: `AGENTS.md`, `CLAUDE.md`, `hpc/README.md`: commands become `hpc/run.sh all CONFIG
  [--run-id ID]` and `hpc/run.sh test CHECKPOINT`; remove diagnostic harness commands and sections;
  remove the stale "568 pass and 2 fail" baseline sentence; silent-failure list gains "a raw-count cache
  fed to STATE — `load_inputs` now refuses manifests without `expression_space`"; architecture section
  says STATE reads log-space HVG basal cells through its released encoder.
- [ ] Step 4: Tahoe-x1 note lines 37–38: replace with Fig. 6B zero-shot Pearson ΔC (unseen plate /
  same plate) ST+HVG ≈0.49/0.39, ST+Tx1-3B ≈0.41/0.38, ST+SE-600M ≈0.41/0.36, perturbation mean
  ≈0.40/0.40, values read from the figure; state that 0.67/0.64, 0.73/0.68, 0.74/0.67 are Fig. 5
  separability values.
- [ ] Step 5: Commit `docs: one expression space, STATE on its own basal path, the all run`.

### Task 10 (easy): architecture figure

**Files:** Replace `figures/geneeffect_architecture.svg` (hand-authored SVG, real `<text>` elements,
under 60 KB, light background, readable at 800 px wide).

- [ ] Step 1: Draw: (a) frozen Tx1-3B on raw UMI → per-cell embeddings `H_c` [128×2560] → pooled
  context `z_c`; frozen ESM-2 → `e_g` [1280]. Basal cells → log-normalised HVG `X_c` [128×2000]. (b)
  ESM2 adapter → perturbation token `p_g` [2024] → STATE (released basal encoder 2000→328, perturbation
  encoder 2024→328, transformer, decoder 328→2000) on `X_c` → `Ŷ_{g,c}` [128×2000]. (c) Response
  descriptors `Δ`, `s` from `Ŷ − X_c`; with `z_c`, `e_g`, `q_{g,c}` → residual head → `δ̂`; plus
  `μ_train(g)` → GeneEffect. Mark trainable (STATE 1e-5, adapter 1e-4, head 1e-4) vs frozen.
- [ ] Step 2: Check it renders: `uv run python -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('figures/geneeffect_architecture.svg')"`.
- [ ] Step 3: Commit `docs(figure): STATE on its own HVG basal path`.

## Wave 2 — two parallel tasks

### Task 5 (hard): preparation and prepared inputs in one expression space

**Files (sole owner):** `src/experiments/prepare.py`, `src/data/{prepared,response,response_cache,
response_streaming,basal,tx1_cache,q_sc,gene_bags}.py`, `src/data/prepare/{build_exp13_q_sc_cache,
build_exp13_tx1_cache}.py` (delete if superseded by `prepare`), tests
`tests/test_{tx1_basal,tx1_embed_cache,tx1_response_data,tx1_response_gene_bags_cache,
tx1_response_streaming,joint_data,build_exp13_q_sc_cache,build_exp13_tx1_cache}.py` (rewrite or delete),
new `tests/test_prepare.py`. Do **not** edit `tests/conftest.py` (Task 6 owns it); put fixtures in
your test files.

**Interfaces:** Consumes Task 2's `src.data.expression`. Produces *Prepared inputs* exactly as in
Shared contracts, and the prepared-root layout.

- [ ] Step 1: Read `src/experiments/prepare.py`, `src/data/prepared.py`, `src/data/response.py`,
  `src/data/basal.py`, `src/data/tx1_cache.py`, `src/data/q_sc.py`. Write `tests/test_prepare.py` first
  with a synthetic world: two tiny raw-count h5ad response sources (Jurkat/HepG2 ModelIDs) and basal
  sources for 4 lines, a fake Tx1 cache in the existing format, a 6-gene "STATE HVG order". Tests:
  `test_target_sum_is_median_of_jurkat_hepg2_controls`; `test_basal_hvg_is_log_space_over_all_genes`
  (compare to `log_normalize(raw[:, hvg], library_sizes(raw), T)` for the same cells as the Tx1 cache
  `obs` order); `test_response_targets_are_log_space`; `test_q_sc_mean_variance_in_log_space`
  (detected fraction equals raw `> 0` fraction); `test_no_condition_holdout_in_manifest`;
  `test_load_inputs_refuses_v1_manifest`; `test_prepare_rejects_non_integer_source`;
  `test_prepare_skips_when_manifest_exists`; `test_train_means_fit_on_supervised_train_only`
  (uses `assert_fit_eligible`).
- [ ] Step 2: Run them; expect failures.
- [ ] Step 3: Implement. One pass per basal line: materialise the line's raw AnnData through the
  existing `basal.py` builders, compute `library_sizes` over all genes, select the cached cells by the
  Tx1 cache `obs` index (assert same order), write `lines/<ModelID>.npz` with `controls_tx1`,
  `basal_hvg` (log), `q_sc_values`, `q_sc_available` (q_sc over all basal cells in log space). Response
  targets: compute library sizes before the HVG slice in `_align_to_checkpoint_order`, apply
  `log_normalize`, write under `prepared_root/response/`; no holdout. `T` from the two hard-coded
  ModelIDs' non-targeting cells. Manifest `prepared_inputs.json` written last with `expression_space`.
  `load_inputs` reads `lines/*.npz` (no Tx1 cache open at train time), drops holdout fields, adds
  `target_sum`. Tx1 encoding of missing lines stays in `tx1_cache.py`, simplified: keep the reader
  and encoder, remove hash sidecars, SHA re-verification, recursive artifact checks and duplicated
  width/dtype assertions. Simplify `basal.py`/`response.py` the same way; keep behaviour.
- [ ] Step 4: Run your tests to green; `.venv/bin/ruff check` your files.
- [ ] Step 5: Commit `refactor(data): one log expression space in preparation, simpler caches`.

### Task 6 (hard): joint model, training and evaluation on STATE's own basal path

**Files (sole owner):** `configs/geneeffect_joint.yaml`, `src/experiments/{config,geneeffect}.py`,
`src/train.py`, `src/evaluate.py`, `src/model/{state,initialization,features,geneeffect,response,
losses,normalization,head,perturbation}.py`, `src/data/{batches,datasets}.py`,
`src/training/{trainer,sampling,checkpoint,distributed}.py`, `src/eval/{geneeffect,response}.py`,
`tests/conftest.py`, tests `tests/test_{geneeffect_e2e,geneeffect_features,geneeffect_head,
geneeffect_sampler,joint_checkpoint,joint_cli,joint_distributed,joint_evaluation,joint_integration,
joint_launcher,joint_objective,joint_resume,joint_sampling,joint_training,response_training,state_core,
tx1_predicted_response,distributed}.py` (rewrite to behaviour tests or delete).

**Interfaces:** Consumes *Prepared inputs* (build `PreparedInputs` directly in fixtures; do not call
Task 5's preparation). Produces *Model and training* contracts and the config.

- [ ] Step 1: Write behaviour tests first in `tests/test_joint.py` with a STATE-shaped CPU fixture
  (keep `conftest.py` environment setup): `test_state_own_basal_path_loads_every_released_weight`
  (fixture checkpoint with 2000→328 basal encoder; zero missing/unexpected/shape-skipped);
  `test_zero_key_checkpoint_load_raises`; `test_response_batches_cover_all_conditions` (no holdout);
  `test_optimizer_rates` (head 1e-4, adapter 1e-4, STATE 1e-5); `test_selection_uses_val_geneeffect_loss`;
  `test_resume_continues_from_last` ; `test_residual_targets_use_fold_fit_mean_predictions_use_fixed_mean`;
  `test_constant_prediction_residual_correlation_is_nan`; `test_config_rejects_unknown_key`;
  `test_cpu_single_process_training_writes_best_and_last`.
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement. `build_joint_model`: `input_dim` stays 2000; control chunks are
  `lines[...].basal_hvg`; remove the "intentional shape skip" allowance; expose `load_released_state`
  and `StateResponse` in `src/model/state.py`. Sampling: all response conditions, balanced across the
  four anchors, every `response_interval` updates. Remove response-holdout validation from the epoch
  loop; validation = GeneEffect on `split.val` only. `run_training(config, run_dir)` public; `src.train`
  takes `--config --run-dir`, resumes from `run_dir/last.pt` when present. Config validator: missing and
  unknown keys only. Delete ceremony: pinned seed/selection checks, duplicated validate() methods that
  re-check shapes on every batch, config-conflict resume machinery beyond "same config or fail".
  Fix or delete the two pre-existing failures in `tests/test_geneeffect_sampler.py`.
- [ ] Step 4: Run your tests to green; `.venv/bin/ruff check` your files.
- [ ] Step 5: Commit `refactor(joint): STATE on its own HVG basal path, one validation split, simpler training`.

## Wave 3 — two parallel tasks

### Task 7 (hard): six-arm response-model comparison

**Files:** Create `src/experiments/response_comparison.py`, `tests/test_response_comparison.py`.

**Interfaces:** Consumes `load_inputs`/`PreparedInputs` (Task 5), `load_released_state`,
`StateResponse`, `Esm2PerturbationAdapter`, `response_loss`, `energy_distance`, `mean_delta_mse`
(Task 6), `ResponseMLP` (Task 3). Produces `run_comparison` per Shared contracts.

- [ ] Step 1: Tests with a synthetic `PreparedInputs` (4 anchors, 12 genes, 8 cells, width 6) and a
  tiny STATE-shaped fixture: `test_no_change_ratio_is_one`; `test_zero_init_arms_start_at_no_change`;
  `test_folds_never_train_on_held_out_anchor`; `test_released_arm_skips_uncovered_genes`;
  `test_identity_share_zero_for_gene_blind_arm` (global mean effect); `test_bootstrap_interval_contains_point`;
  `test_pooled_with_and_without_hct116`; `test_skips_when_verdicts_exist`; `test_outputs_written`.
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement. Arms: no-change; global mean effect (mean over source-anchor conditions of
  `mean(target) − mean(control)`, added to every basal cell); released STATE checkpoint with one-hot
  perturbations from `state_model_dir/pert_onehot_map.pt` (genes outside it skipped, counted); STATE as
  in the joint model (`StateResponse`, STATE 1e-5, adapter 1e-4); MLP on HVG (`ResponseMLP(2000, …)`);
  MLP on Tx1 (`ResponseMLP(2560, …)`). Fold = hold out one anchor; train `comparison.epochs` epochs on
  all conditions of the other three, balanced, AdamW, seed 0, no early stopping. Per condition loss =
  `response_loss(pred, target, control)`. Report per arm×fold: held-out mean loss ÷ no-change,
  identity share = (mean loss under `shuffles` fixed gene derangements − loss) / loss, source training
  ratio, per-epoch held-out ratio (curves.csv). Pooled ratio = mean of per-fold ratios, all four and
  without HCT116 `ACH-000971`; 95% interval from `bootstrap` gene resamples (seed 0). Write
  `sanity.json` (released arm ratio per anchor) before training. `verdicts.json`: STATE-as-joint vs
  MLP-on-HVG and MLP-on-Tx1 vs MLP-on-HVG — difference of pooled ratios with interval, and "overlap"/
  "separated". CLI `python -m src.experiments.response_comparison --config C --out-dir D [--device]`.
- [ ] Step 4: Tests green; Ruff.
- [ ] Step 5: Commit `feat(experiments): six-arm response-model comparison`.

### Task 8 (easy): baselines and readout on the new inputs

**Files:** `src/experiments/baselines.py`, `src/baselines/residual.py`, `src/experiments/readout.py`,
`src/experiments/tx1_gmm_ridge.py` (only if it breaks), tests `tests/test_run_r1_residual_ladder.py`,
`tests/test_tx1_gmm_ridge.py`, new `tests/test_readout_entry.py`.

**Interfaces:** Consumes `PreparedInputs` (Task 5), `restore_model`, `load_checkpoint` (Task 6).
Produces `run_baselines` (unchanged signature, HVG view now log space) and `run_readout`.

- [ ] Step 1: Tests: `test_baselines_run_on_synthetic_inputs` (gene mean, copy prior, nearest line,
  context-PCA ridge on Tx1 and on HVG all present; gene-mean residual correlation NaN);
  `test_run_readout_writes_val_metrics_and_skips_when_done` (synthetic checkpoint from Task 6's fixture).
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Adapt `run_baselines` to `inputs.train_gene_means`, `lines[...].controls_tx1/basal_hvg`;
  strip formal checks in `residual.py` that are not one of the six guards (axis-invariance checks,
  exact-method-coverage validation) while keeping its numbers. In `readout.py` use `restore_model`; add
  `run_readout(checkpoint, out_dir, *, device)` = extract train/val features → fit the explicit
  context-slope readout (head seed 0) → evaluate on val → write `metrics.json`; skip if it exists.
- [ ] Step 4: Tests green; Ruff.
- [ ] Step 5: Commit `refactor: baselines and readout on the log-space inputs`.

## Wave 4 — one task

### Task 9 (hard): the `all` run, summary and end-to-end smoke test

**Files:** Create `src/experiments/all.py`, `tests/test_all.py`; modify `hpc/run.sh`.

**Interfaces:** Consumes `prepare_inputs` (Task 5), `run_comparison` (Task 7), `src.train` CLI /
`run_training`, `evaluate_checkpoint`, `export_evaluation` (Task 6), `run_baselines`, `run_readout`
(Task 8). Produces `run_all` and `hpc/run.sh all|test`.

- [ ] Step 1: Tests on a synthetic fixture (tiny prepared world, STATE-shaped fixture, epochs 1, CPU):
  `test_all_end_to_end_writes_summary` (summary.md contains `T`, the sanity line, the comparison table,
  both verdicts, validation rows for the joint model, readout and every baseline);
  `test_all_resumes_and_skips` (kill after training writes last.pt → rerun with same run id resumes;
  completed steps not recomputed); `test_all_never_touches_test_split`.
- [ ] Step 2: Run; expect failures.
- [ ] Step 3: Implement `run_all`: prepare (skip if manifest) → comparison (skip if verdicts) →
  training via `accelerate launch --num_processes <visible GPUs> --module src.train --config C
  --run-dir RUN/train` when CUDA is visible, else in-process `run_training` (skip if `train/best.pt`
  and `train/done.json`) → val evaluation, baselines, readout (each skipped if its metrics exist) →
  `summary.md`. Default run id `all_<UTC timestamp>`; print it first. `hpc/run.sh`: `all CONFIG
  [--run-id ID]` → `python -m src.experiments.all`; keep `test CHECKPOINT`; drop `prepare`/`train`
  subcommands.
- [ ] Step 4: Full suite: `uv run python -m pytest tests -q > /tmp/full.txt 2>&1; tail -15 /tmp/full.txt`
  green; `.venv/bin/ruff check .` clean.
- [ ] Step 5: Commit `feat: hpc/run.sh all runs preparation through validation evaluation`.

## After Wave 4 (primary agent)

Integration review of the whole diff, full suite and Ruff, Codex review of the wave
(`codex review --base 0ee5166` with a clean `CODEX_HOME`), adjudicate findings, merge to `main`,
push, pull on H20 port 30838, launch `hpc/run.sh all configs/geneeffect_joint.yaml` in the background,
report once `T` and the sanity line are written.

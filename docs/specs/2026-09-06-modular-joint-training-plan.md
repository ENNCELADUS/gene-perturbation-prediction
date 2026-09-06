# Modular Joint GeneEffect Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Implementation status:** Completed locally on 2026-09-06. The integrated suite passes 551 tests with one CUDA-only skip; Ruff, wheel imports and 24 command help checks pass. No research-data training or GPU run was launched.

**Goal:** Replace the staged Exp13 runtime with one joint GeneEffect trainer, recurring four-anchor response supervision, loss-selected checkpoints and a topology-style repository layout.

**Architecture:** Reuse the current data semantics and five-block model, moving their computation into independent data/model modules. One trainer owns optimization and checkpoint state; one evaluator computes all validation/test losses and metrics. Preparation opens or builds fixed inputs without a recursive artifact-certification pipeline.

**Tech Stack:** Python 3.11–3.12, PyTorch, Accelerate, NumPy, pandas, SciPy, scikit-learn, PyArrow, PyYAML, pytest, Ruff, Hatch and uv; existing project dependencies only.

**Spec:** [2026-09-06-modular-joint-training-design.md](2026-09-06-modular-joint-training-design.md), including the owner's latest validation/seed corrections. Read both documents before execution. Source evidence was inspected at `2593ad1`; the preceding code snapshot is `e6341d2`.

## Global constraints

- Training / cell-collation / projection seed: **0 / 0 / 0**.
- Every optimizer update contains a dependency batch; updates 0, 4, 8, … also contain a balanced four-anchor response batch. Initial response weight is 1.0.
- Validate exactly once at the end of every completed training epoch.
- Early stopping and `best.pt` selection both **minimize `val_geneeffect_loss`**. Strict improvement resets patience; ties/increases increment it. No composite selector or improvement tolerance.
- `val_geneeffect_loss` is Huber, delta 1, averaged over valid validation pairs. `val_total_loss = val_geneeffect_loss + response_weight * val_response_loss`.
- Report total loss, GeneEffect loss, both response loss terms, their sum, Pearson, Spearman, RMSE, MAE and coverage every epoch, with axes and averaging from the spec.
- Keep the fixed 170 labeled train / 27 validation / 27 test GeneEffect split, the two unlabeled train exclusions and the fixed response-condition holdout.
- Tx1 is frozen; STATE, the ESM2 adapter and the head are trainable. Trainable response features remain in the autograd graph.
- Retain only boundary checks needed for correct inputs, fitting isolation, checkpoint loading, numerical/distributed correctness and I/O. No seals, digest chains or artifact promotion gates.
- Use `rtk proxy uv run` from the repository root for Python, pytest and Ruff; preserve `tests/conftest.py` initialization.
- Worktree changes, commits and tests in this plan are local. No pushing, remote artifact moves, training launches or scientific claims are authorized by this plan.
- Do not edit `CLAUDE.md`, frozen benchmark membership, historical result values or the local archive. Do not leave compatibility wrappers or the old active training route.

---

## File ownership and final structure

Paths in the first column below are relative to the current `src/aivc_model/`. A split means move the named responsibilities, update all consumers and remove the original definitions; it does not mean copy a second implementation.

| Current file(s) | Final owner |
| --- | --- |
| `benchmark_split.py`; `FixedSplit` from `residual_ladder.py` | `src/data/splits.py` |
| `residual_target.py` | `src/data/residual_target.py` |
| `geneeffect_data.py` | `src/data/geneeffect.py`; q_sc computation/storage in `src/data/q_sc.py` |
| `tx1_basal.py`, `tx1_response_streaming.py` | `src/data/basal.py`, `src/data/response_streaming.py` |
| `tx1_embed_cache.py` | `src/data/tx1_cache.py`; encoder construction in `src/model/tx1.py` |
| `tx1_response_data.py`, `tx1_response_gene_bags_cache.py` | `src/data/response.py`, `src/data/response_cache.py` |
| `gene_embeddings.py` | Table/loading in `src/data/embeddings.py`; `PertAdapter` in `src/model/perturbation.py` |
| `gene_splits.py`, `esm2_provenance.py` | Retained preparation helpers in `src/data/gene_splits.py`, `src/data/esm2_provenance.py` |
| `state_core.py` | `GeneBags` in `src/data/gene_bags.py`; gene-order helpers in `src/data/gene_order.py`; STATE/perturbation adapters in `src/model/state.py`, `src/model/perturbation.py` |
| `state_warm_start.py`, model construction in `tx1_predicted_response.py` | `src/model/initialization.py` |
| `geneeffect_e2e.py` | Batch records in `src/data/batches.py`; composition in `src/model/geneeffect.py` |
| `geneeffect_features.py` | Computation/projection in `src/model/features.py`; standardizer in `src/model/normalization.py` |
| `geneeffect_head.py` | Head in `src/model/head.py`; retained pooling and new Huber loss in `src/model/losses.py` |
| Forward/loss primitives in `response_training.py` | `src/model/response.py` |
| `residual_metrics.py`, `residual_ladder.py` | `src/eval/metrics.py`, `src/baselines/residual.py` |
| `distributed.py` | `src/training/distributed.py`, retaining only needed setup/coordinated error handling |
| Stage-specific runners, trainers, configs, feature stores and artifact seals | Replaced by `src/training/trainer.py`, `src/training/checkpoint.py`, `src/experiments/geneeffect.py`; old implementations deleted after cutover |

Additional new files: `src/data/prepared.py`, `src/data/datasets.py`, `src/training/sampling.py`, `src/eval/geneeffect.py`, `src/eval/response.py`, `src/experiments/config.py`, `src/experiments/prepare.py`, `src/experiments/baselines.py`, `src/train.py`, `src/evaluate.py`, `hpc/run.sh`, `hpc/README.md`, `configs/geneeffect_joint.yaml`.

All retained top-level Python preparation scripts move to `src/data/prepare/` with their existing basenames, except `cell_line_split_common.py` becomes `src/data/split_build.py`, and the shared loader functions in `verify_tx1_obsm_width.py` and `build_tx1_basal_embeddings.py` move to `src/model/tx1.py`. Their remaining CLI bodies call that module. The four existing historical scripts move to `src/experiments/historical/` with their basenames unchanged. `run_r1_residual_ladder.py` becomes the thin baseline command in `src/experiments/baselines.py`. The old training, sealing and DDP launcher scripts are deleted at cutover. `scripts/download_tahoe_source_shards.sh` remains an operational utility; the atlas JSON moves to `configs/data/cell_line_atlas_raw_umi_27.json`.

Keep the existing flat test filenames where their behavior survives; add only the new behavior tests named below. Do not reorganize unrelated tests or add a generic pipeline framework.

### Task 1: Establish the new package without breaking retained behavior

**Files:** Move current source to the ownership map above; temporarily keep not-yet-replaced stage machinery under `src/experiments/exp13_legacy/` using its current basenames. Modify `pyproject.toml`, retained Python scripts and all affected test imports. Create package `__init__.py` files and `src/README.md`. The temporary directory contains the relocated old code, not wrappers around it, and Task 7 removes it.

**Interfaces:** Preserve retained public signatures while moving them. Extract `FixedSplit` first so `src/data/splits.py` never imports the baseline evaluator. Data/model modules must not import `src.training`, `src.eval` or `src.experiments`. Move old checkpoint-seal loading and historical scoring functions out of model construction into the temporary experiment directory.

- [x] Run the retained-suite baseline and save its counts in the implementation summary:

  ```bash
  rtk proxy uv run --no-cache --offline --no-sync python -m pytest tests -q
  rtk proxy uv run --no-cache --offline --no-sync ruff check src scripts tests
  ```

  If a command fails, distinguish environment/asset failures from failing assertions before editing.

- [x] Move modules and split the mixed data/model symbols according to the table. Update imports and string-based monkeypatch targets in the same change. Keep checkpoint state-dictionary parameter names unchanged for moved retained components. Set package inclusion to:

  ```toml
  [tool.hatch.build.targets.wheel]
  packages = ["src"]
  ```

- [x] Move `GeneBags` and `OnlineConditionBatch` without changing their field definitions. Move pure response functions (`predict_bags`, `predict_bag`, `mean_delta_mse`, `energy_distance`) out of the trainer so model feature construction imports `src.model.response`.
- [x] Run the retained suite again. Resolve import cycles in the owning module rather than inserting lazy compatibility imports. Run a wheel build with `uv build --wheel` and verify that it includes the `src` package; report a missing build dependency instead of calling an import check a package-build result.
- [x] Commit the runnable extraction as `refactor: separate data model evaluation and experiment modules`.

### Task 2: Provide simple prepared inputs and batch loaders

**Files:** Create `src/data/prepared.py`, `src/data/datasets.py`, `src/training/sampling.py`; modify `src/data/batches.py`, `src/data/response_cache.py`, `src/data/tx1_cache.py`, `src/data/geneeffect.py`. Add `tests/test_joint_data.py`, `tests/test_joint_sampling.py`; update retained input/cache tests.

**Consumes:** Moved `FixedSplit`, `OnlineConditionBatch`, `GeneBags`, `load_line_cache`, `load_hvg_gene_order`, `load_esm2_embeddings`, `fit_gene_means`, response assembly and residual metrics' existing gene/line conventions.

**Produces:** `PreparedInputs`, `DependencyBatch`, `ResponseBatch`; `load_inputs(config: Mapping[str, Any], *, preprocessing: Mapping[str, Any] | None = None, include_test: bool = False) -> PreparedInputs`; `make_training_loaders(inputs, config, epoch, accelerator) -> tuple[DataLoader, Iterator[ResponseBatch]]`; `make_evaluation_loaders(inputs, config, split, accelerator) -> tuple[DataLoader, DataLoader]`. Loader builders live in `src/training/sampling.py` and `src/data/datasets.py`, respectively. Accelerator may be `None` in CPU unit tests.

- [x] Define these shared records; `OnlineConditionBatch` keeps its already inspected fields:

  ```python
  @dataclass(frozen=True)
  class PreparedInputs:
      split: FixedSplit
      labels: pd.DataFrame
      genes: tuple[str, ...]
      train_gene_means: pd.Series
      variable_genes: frozenset[str]
      tx1_cache: Path
      q_sc_cache: Path
      response_cache: Path
      hvg_order: tuple[str, ...]
      response_holdout: frozenset[tuple[str, str]]  # ModelID, gene

  @dataclass(frozen=True)
  class DependencyBatch:
      conditions: OnlineConditionBatch
      residual: torch.Tensor
      gene_mean: torch.Tensor
      valid: torch.Tensor

  @dataclass(frozen=True)
  class ResponseBatch:
      model_ids: tuple[str, ...]
      genes: tuple[str, ...]
      controls_tx1: tuple[torch.Tensor, ...]
      observed_hvg: tuple[torch.Tensor, ...]
      control_hvg: tuple[torch.Tensor, ...]
  ```

  `labels` has `model_id`, `gene_symbol`, `gene_effect`, `residual`; masks use boolean tensors. Add `.to(device, non_blocking=False)` to the batch records, moving nested tensor fields while preserving identifiers. This makes device transfer explicit and testable with Accelerate.

- [x] Build dependency rows only from finite labels and the selected split; for a fresh run, fit gene means and variable-gene membership on labeled train lines. When `preprocessing` is supplied by resume/evaluation, restore its means/membership without calling fitting functions. Default loading exposes only train/validation labels; only explicit test evaluation uses `include_test=True`. Load cached basal/HVG rows by ModelID and gene order. Reuse the paired 128-cell selection and explicit masks; no trainable features are stored. Preserve the fixed response holdout, using the existing 10%/seed-13 partition only when reconstructing that historical partition, not as a new training seed.
- [x] Implement anchor-balanced replay with separate shuffled index pools. Each call draws 16 conditions from each anchor for the initial batch size of 64. Exhausted pools reshuffle and cycle; epoch/rank RNG streams derive from training seed 0. Assert at loader construction that the response batch size is divisible by four. Dependency sampling uses standard distributed full batches; evaluation keeps the tail and gathers actual rows with padding removed.
- [x] Write tests constructing a small `FixedSplit` and cached arrays: holdout IDs never appear in training batches; changing validation labels cannot change train means or normalized training inputs; all replay batches have equal anchor counts; held-out response keys never appear; restarting epoch 2 with the same seed/rank reproduces its first batches. Check preserved GeneEffect labels for response-held-out conditions without admitting their response targets.
- [x] Make cache opening a read operation: consume the existing array layout and small metadata without hashing raw sources, rebuilding missing data or creating per-rank caches. A missing/mismatched cache reports the concrete path and the preparation command. Keep raw-source building in the preparation tool. Run:

  ```bash
  rtk proxy uv run --no-cache --offline --no-sync python -m pytest tests/test_joint_data.py tests/test_joint_sampling.py tests/test_benchmark_split.py tests/test_residual_target.py tests/test_tx1_response_data.py tests/test_tx1_embed_cache.py -q
  ```

- [x] Commit as `refactor: load fixed inputs and sample balanced response replay`.

### Task 3: Make the model support direct joint training with seed 0

**Files:** Modify `src/model/geneeffect.py`, `src/model/features.py`, `src/model/normalization.py`, `src/model/initialization.py`, `src/model/losses.py`, `src/model/response.py`. Add `tests/test_joint_objective.py`; update `tests/test_geneeffect_e2e.py`, `tests/test_geneeffect_features.py`, `tests/test_state_core.py`.

**Consumes:** Existing `GeneEffectE2EModel.forward(OnlineConditionBatch, response: ResponseForwardBatch | None) -> E2EForwardOutput`; Task 2 batch types.

**Produces:** `geneeffect_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor`; `response_terms(predicted: Sequence[Tensor], batch: ResponseBatch) -> dict[str, Tensor]`, with per-condition vectors under `mean_delta_mse` and `energy_distance`; a model constructible without any Stage 1 artifact. Rename the internal `PrecomputedFeatureBatch` to `FeatureBatch` and update consumers because it now represents live features, not a warmup store.

- [x] Implement the masked regression objective, rejecting non-finite predictions on labeled rows and an empty labeled batch before reducing:

  ```python
  return F.huber_loss(
      prediction.float()[valid], target.float()[valid], delta=1.0, reduction="mean"
  )
  ```

  For `response_terms`, call the retained mean-delta and energy functions on each predicted/observed bag and its control mean. Return per-condition values so training can average a balanced batch and evaluation can aggregate unequal anchor counts correctly.

- [x] Change the projection constructor default and its constant to 0, pass `config["seeds"]["projection"]` explicitly at construction and pass the collator seed explicitly to STATE forwarding. Remove state/schema/component hash checks from projection and normalizer serialization; retain shapes, finite values and positive scales. Save the actual matrix/statistics in checkpoints.
- [x] Initialize a fresh training model through the inspected upstream-checkpoint construction path; initialize the gene adapter and head normally. For resume/evaluation, construct from saved architecture metadata and load the saved state strictly, without needing to reload upstream weights. Remove Stage 1 artifact imports from the model constructor. Extract raw-feature generation from the existing forward before standardization, exposing `GeneEffectE2EModel.condition_features(batch: OnlineConditionBatch) -> FeatureBatch`. The normal forward uses this same method without detaching it.
- [x] Implement the bounded startup standardizer fit on rank zero: select up to 32 train rows per line with seed 0, stream all five blocks without gradients, broadcast statistics and restore RNG state. Skip on resume. No full feature dataset is generated or written.
- [x] Add direct checks for the changed behavior:

  ```python
  def test_huber_matches_absolute_geneeffect():
      pred = torch.tensor([0.2, -0.4], requires_grad=True)
      target = torch.tensor([0.0, -0.1])
      mean = torch.tensor([-0.7, 0.3])
      valid = torch.tensor([True, True])
      torch.testing.assert_close(
          geneeffect_loss(pred, target, valid),
          geneeffect_loss(pred + mean, target + mean, valid),
      )

  def test_default_projection_uses_zero_seed():
      np.testing.assert_array_equal(
          FixedSparseProjection().components, FixedSparseProjection(seed=0).components
      )
  ```

  Reuse the tiny STATE fixture from `tests/test_geneeffect_e2e.py` for two gradient tests: regression alone updates both backbone/adapter and head; adding a response batch adds reconstruction gradients through the same wrapped model call. Verify that feature generation remains differentiable and normalization sees training rows only.
- [x] Run the four affected model/objective test files and Ruff, then commit as `feat: support direct joint objectives and zero-seed initialization`.

### Task 4: Implement epoch evaluation and the GeneEffect-loss selector

**Files:** Create `src/eval/geneeffect.py`, `src/eval/response.py`, `src/training/checkpoint.py`; modify `src/eval/metrics.py`. Add `tests/test_joint_evaluation.py`, `tests/test_joint_checkpoint.py`.

**Consumes:** Task 2 evaluation loaders and Task 3 model/objective. Reuse the finite-pair and constant-vector handling in the inspected `residual_metrics.py` when adding Pearson.

**Produces:** `EvalResult` below; `evaluate_model(model, inputs: PreparedInputs, config: Mapping[str, Any], *, split: str, accelerator=None) -> EvalResult`; `compose_metrics(geneeffect: Mapping[str, float | int | None], response: Mapping[str, float | int | None], *, response_weight: float, prefix: str) -> dict[str, float | int | None]`; `TrainState`; `record_validation(state: TrainState, metrics: Mapping[str, Any], epoch: int) -> bool`.

- [x] Define results and checkpoint-selection state:

  ```python
  @dataclass
  class EvalResult:
      metrics: dict[str, float | int | None]
      predictions: pd.DataFrame
      per_line: pd.DataFrame
      per_gene: pd.DataFrame
      response: pd.DataFrame

  @dataclass
  class TrainState:
      next_epoch: int = 0
      global_step: int = 0
      best_loss: float = math.inf
      best_epoch: int = -1
      bad_epochs: int = 0

  def record_validation(state, metrics, epoch):
      loss = float(metrics["val_geneeffect_loss"])
      if not math.isfinite(loss):
          raise ValueError("val_geneeffect_loss must be finite")
      improved = loss < state.best_loss
      if improved:
          state.best_loss, state.best_epoch, state.bad_epochs = loss, epoch, 0
      else:
          state.bad_epochs += 1
      state.next_epoch = epoch + 1
      return improved
  ```

- [x] Aggregate GeneEffect Huber by loss sum / valid pair count, not batch means. Aggregate each response term within ModelID, then average the four anchors. Add the loss terms using the explicit response weight, without dividing validation loss by replay frequency. In distributed evaluation use Accelerate's supported gather-for-metrics path and test tail padding removal. Do not average rank-level correlation coefficients.
- [x] Compute all fields named in design §4: absolute-GeneEffect per-line Pearson/Spearman, variable-gene residual per-gene Pearson/Spearman, RMSE/MAE and coverage. `predictions` contains `model_id`, `gene_symbol`, `gene_effect`, `residual`, `geneeffect_prediction`, `residual_prediction`; `response` contains one row per held-out condition with `model_id`, `gene_symbol` and both response terms. Derive detailed tables and macro metrics from these aligned rows. Serialize undefined scalars as JSON null, retaining undefined counts. Non-finite predicted values on scored rows are errors, not silently missing scores.
- [x] Add concrete conflicting-metric and aggregation cases:

  ```python
  def test_only_geneeffect_loss_controls_selection():
      state = TrainState()
      assert record_validation(state, {"val_geneeffect_loss": 0.4}, 0)
      assert not record_validation(state, {
          "val_geneeffect_loss": 0.5, "val_total_loss": 0.1,
          "val_residual_spearman_macro_per_gene": 0.99,
      }, 1)
      assert state.best_epoch == 0 and state.bad_epochs == 1
      assert record_validation(state, {
          "val_geneeffect_loss": 0.3, "val_total_loss": 9.0,
      }, 2)
      assert state.best_epoch == 2 and state.bad_epochs == 0
      assert not record_validation(state, {"val_geneeffect_loss": 0.3}, 3)
      assert state.bad_epochs == 1
  ```

  In evaluator tests use residual errors `[0, 0, 2]` split into batches of 2 and 1: Huber must be `1.5 / 3 = 0.5`. For response conditions with anchor means `[1, 3, 5, 7]` and unequal anchor counts, the aggregate must be 4. Verify a constant prediction has a valid Huber loss and undefined correlations without blocking checkpoint selection. A `compose_metrics` example with GeneEffect loss 0.2, response loss 0.8 and weight 0.5 must produce total loss 0.6.
- [x] Run the new evaluator/selector tests and retained residual-metric tests. Commit as `feat: validate all loss terms and select minimum GeneEffect loss`.

### Task 5: Integrate the single trainer, DDP and epoch-boundary resume

**Files:** Create `src/training/trainer.py`; extend `src/training/checkpoint.py`, `src/training/distributed.py`. Add `tests/test_joint_training.py`, `tests/test_joint_distributed.py`, `tests/test_joint_resume.py`.

**Consumes:** Task 2 loaders, Task 3 model/loss functions and Task 4 evaluator/selection state.

**Produces:** `fit(model, inputs: PreparedInputs, config: Mapping[str, Any], run_dir: Path, accelerator, *, restored: Mapping[str, Any] | None = None) -> TrainState`; `save_checkpoint(path, model, optimizer, state: TrainState, config, preprocessing, accelerator) -> None`; `load_checkpoint(path: Path) -> dict[str, Any]`. Model and optimizer remain ordinary PyTorch/Accelerate objects, not a new trainer framework.

- [x] Implement the loop around the prepared model, one optimizer and three learning-rate groups. Use these exact control-flow decisions:

  ```python
  replay = state.global_step % config["train"]["response_interval"] == 0
  dependency_batch = dependency_batch.to(accelerator.device)
  response_batch = next(response_iterator).to(accelerator.device) if replay else None
  response_input = None if response_batch is None else ResponseForwardBatch(
      response_batch.controls_tx1, response_batch.genes
  )
  output = model(dependency_batch.conditions, response=response_input)
  dep_loss = geneeffect_loss(
      output.delta_hat, dependency_batch.residual, dependency_batch.valid
  )
  total = dep_loss
  if response_batch is not None:
      terms = response_terms(output.response_predicted, response_batch)
      total = total + config["train"]["response_weight"] * (
          terms["mean_delta_mse"].mean() + terms["energy_distance"].mean()
      )
  accelerator.backward(total)
  accelerator.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
  state.global_step += 1
  ```

  All ranks take the same replay branch. Use FP32 losses and BF16 model autocast, no static-graph flag and no train-time unwrapping. Missing response-step metrics are null, not 0. Check finite losses/gradients; do not require nonzero gradient magnitude as a runtime quality gate.
- [x] After the final training batch of each epoch, call `evaluate_model(..., split="val")` exactly once, print/log every scalar loss/metric/count from its result, apply `record_validation`, save `best.pt` if improved and always save `last.pt`. Stop when `bad_epochs >= patience`. Validation's correlation or response metrics cannot postpone this decision. Restore train mode before the next epoch.
- [x] Save complete architecture/configuration, model state, ESM2 buffers/order, train gene means, projection, normalization, optimizer, `TrainState`, world size, AMP state and every rank's RNG state. Encode NumPy RNG arrays as tensors/lists and retain ordinary Python/Torch RNG state; load checkpoint data without pickling model classes. Rank zero writes a temporary file and atomically replaces the target. Epoch/rank deterministic loaders restart at `state.next_epoch`, while replay uses restored `global_step`.
- [x] Add a tiny two-epoch CPU run using the existing small STATE model and new data fixtures. Spy on the common evaluator to verify exactly two validation calls, all required logged fields and a best checkpoint selected by GeneEffect loss when other metrics disagree. Stop/resume after epoch 1 and compare the next update against the uninterrupted run, including optimizer tensors and replay position.
- [x] Add one two-process CPU/Gloo test for each update type against the same effective single-process batch. Use the real model wrapper, batch transfer, optimizer and gather path; use deterministic no-dropout inputs to isolate synchronization. Preserve the normal pytest initialization instead of running a test file as a script. Verify padded evaluation rows are counted once.
- [x] Run the new trainer, distributed, resume, evaluator and model tests. Commit as `feat: train jointly with recurring reconstruction and ordinary checkpoints`.

### Task 6: Connect real preparation, CLI execution and independent testing

**Files:** Create `src/experiments/config.py`, `src/experiments/prepare.py`, `src/experiments/geneeffect.py`, `src/experiments/baselines.py`, `src/train.py`, `src/evaluate.py`, `hpc/run.sh`, `hpc/README.md`, `configs/geneeffect_joint.yaml`. Move preparation scripts using the ownership rules above. Add `tests/test_joint_cli.py`, `tests/test_joint_launcher.py`, `tests/test_joint_integration.py`.

**Consumes:** `load_inputs`, `fit`, `evaluate_model`, checkpoint loading and retained raw-data/cache builders.

**Produces:** `load_config(path: Path) -> dict[str, Any]`; `prepare_inputs(config: Mapping[str, Any]) -> Path`; `run_training(config_path: Path, *, run_id: str | None, resume: Path | None) -> Path`; `evaluate_checkpoint(checkpoint: Path, *, split: str) -> EvalResult`; the commands below. Require exactly one of `run_id` and `resume`; resume uses its checkpoint's saved configuration and rejects a conflicting supplied config.

- [x] Put the approved defaults in one config, including explicit seed and selector fields:

  ```yaml
  seeds: {train: 0, collator: 0, projection: 0}
  train:
    max_epochs: 50
    patience: 5
    dependency_batch_size: 256
    response_batch_size: 64
    response_interval: 4
    response_weight: 1.0
    state_learning_rate: 0.000001
    adapter_learning_rate: 0.00001
    head_learning_rate: 0.0001
    weight_decay: 0.01
  selection: {metric: val_geneeffect_loss, mode: min}
  precision: bf16
  output_root: outputs/geneeffect_joint
  ```

  Add the real input fields from the current config: split, GeneEffect CSV, source registry, Tx1 cache, q_sc cache, ESM2 table/common gene panel, STATE checkpoint/model directory, response-source JSON and response cache. Add `prepared_root: data/geneeffect_joint/v1`. Retain existing external input paths when their files are not being moved; use Task 7's new path for tracked Phase-A provenance. Do not add a Stage 1 checkpoint, seal, lambda-calibration, frozen-feature-store or run-eligibility field. `load_config` checks unknown/missing settings and numeric domains, and requires the stated selector for this training protocol.
- [x] `prepare_inputs` runs source/label alignment and response-target assembly once, preserving the common panel and fixed response holdout, and writes small input/cache metadata. It must prepare an ESM2-resolvable union sufficient for dependency and response conditions and report excluded conditions. Cache readers open those arrays without rescanning raw files. Test that invoking a training rank with a missing cache fails without calling raw assembly.
- [x] Make `run_training` compose model/data/trainer only. Record revision, seeds, input identifiers and separate `training`/`evaluation` statuses in `run.json`. On an exception save its phase/type/message, retain valid checkpoints and propagate the failure. Do not make successful training depend on testing or export.
- [x] `evaluate_checkpoint` loads model and preprocessing from the checkpoint, passes that state to `load_inputs` with `include_test=(split == "test")`, builds the requested loaders and calls `evaluate_model`. It must not call the preprocessing fit functions. Write named prediction columns to Parquet and scalar metrics to JSON; return per-line/per-gene/response tables for ordinary exports. Use `test_` prefixes for test. Re-running evaluation does not alter training status or model weights. Baseline fitting uses `src.baselines.residual` on train only and exports against the same panel through `src.experiments.baselines`.
- [x] Implement thin command dispatch with no separate qualification ladder:

  ```bash
  hpc/run.sh prepare configs/geneeffect_joint.yaml
  hpc/run.sh train configs/geneeffect_joint.yaml --run-id joint_seed0
  hpc/run.sh train configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/joint_seed0/last.pt
  hpc/run.sh test outputs/geneeffect_joint/joint_seed0/best.pt
  uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/joint_seed0/best.pt --split val
  uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split test --out-dir outputs/geneeffect_joint/baselines_seed0
  ```

  On the H20 environment, use the existing `.venv-tx1/bin/python` (or explicit `PYTHON_BIN`) and its `-m accelerate.commands.launch --module src.train` entry. Detect visible GPUs through that environment's Torch; respect its visibility mask. `prepare` is single-process, `train` launches the detected workers, and `test` is an ordinary checkpoint evaluation invocation. Help must work without importing models or accessing GPUs. Do not copy the reference repository's unrelated SSH endpoint, fixed checkout or model-family dispatch.
- [x] Test parser conflicts, help, dispatch argument preservation, no hidden raw-data rebuild and independent failure states. Use fake launcher executables to inspect arguments without launching GPUs. In the tiny end-to-end fixture, train, reload, evaluate and deliberately fail the export: `training` stays completed, `evaluation` records failure, and a second export succeeds without optimizer steps. Run the three new CLI/integration test files plus moved preparation tests.
- [x] Commit as `feat: expose prepare train and checkpoint test commands`.

### Task 7: Retire the old runtime and migrate tracked results and documentation

**Files:** Remove `src/experiments/exp13_legacy/` and its now-unused stage-only tests; remove `scripts/train_geneeffect_e2e.py`, `scripts/train_geneeffect_response_model.py`, `scripts/seal_stage1_response_artifact.py`, `scripts/run_stage1_response_ddp.sh` after their replacements pass. Remove retired `stage1_response.yaml` and `stage2_e2e.yaml` from active config. Modify `.gitignore`, README, `src/README.md`, `scripts/README.md`, project `AGENTS.md`, the HPC runbook/skill, blueprint and affected experiment/data documentation. Move only the tracked artifacts below.

**Interfaces:** All active Python commands/imports use `src.*`. Old commands resolve in the historical `e6341d2` snapshot. Model, data and experiment types are the concrete types defined in Tasks 1–6, with no re-export shim.

- [x] Remove obsolete warmup, sealed-artifact and gradient-calibration code/tests after confirming their useful numerical behavior is covered in the retained/new tests. Retain any still-used gene-order, masked-loss, input-semantics, DDP or checkpoint-key checks in their final owner. Do not delete a test merely because it fails after a move.
- [x] Move the four tracked Phase-A files byte-for-byte to `configs/benchmarks/provenance/phase_a_tx1_20260724/`. Move the two Stage 0 JSON files and its note to `docs/results/exp13_stage0/`, naming the note `README.md`. Move the Stage 2 note into the existing `docs/results/exp13_stage2_full/README.md`. Record pre/post content hashes for this local file-move verification only; do not install those hashes as runtime gates.
- [x] Update active consumers and links, including all four retained historical preparation commands. Update `.gitignore` with `/outputs/` and the exact new small-provenance CSV exceptions, removing obsolete `results/` exceptions. Preserve embedded historical manifest paths/values. Leave ignored datasets, existing caches, archive files and remote runs in place.
- [x] Update the current blueprint/protocol to describe the new objective and **minimum validation GeneEffect loss** selector; retain per-gene residual correlations as scientific reporting metrics. Label old Exp13 results/protocol as historical, state that the new run is unexecuted, and distinguish it from the separate SL protocol. Do not copy the old seed into any new training default.
- [x] Inspect the final change and run the full retained/new suite once after integration:

  ```bash
  rtk proxy uv run --no-cache --offline --no-sync python -m pytest tests -q
  rtk proxy uv run --no-cache --offline --no-sync ruff check src tests
  rtk proxy uv build --wheel
  rtk proxy git diff --check
  rtk proxy rg -n 'aivc_model|exp13_legacy|stage1_artifact|stage2_artifacts|calibrate_lambda_dep|20_260_828|20260828' src scripts configs/geneeffect_joint.yaml tests
  ```

  Review search hits by role: no retired namespace, artifact gate or old seed remains in the active route. Historical prepared inputs or deliberately tested non-default seeds are not rewritten to satisfy a textual zero-hit quota. Verify wheel imports outside the source checkout, all active CLI help commands and local documentation links. Report skipped real-asset tests and any unexecuted GPU path explicitly.
- [x] Commit the completed cutover as `refactor: retire staged Exp13 runtime and organize results` and deliver the changed paths, test outcomes and exact later launch commands. Do not launch them as part of this plan.

## Plan self-review and execution handoff

Design §§1–2 are implemented by Tasks 1–3; §3 by Tasks 2, 3 and 5; §4 by Tasks 4–6; §5 by Tasks 1 and 6; §6 by Tasks 2, 6 and 7; §7 by the per-task tests and final checks. The latest owner changes have direct tests in Tasks 3–5 and config/contract changes in Tasks 6–7.

This document is the implementation plan, not evidence that implementation or tests have run. Execute in dependency order and preserve unrelated working-tree changes. Inline execution with `superpowers:executing-plans` is sufficient; subagents are an execution option, not a required review ceremony.

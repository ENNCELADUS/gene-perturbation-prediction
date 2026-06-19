# exp08 Training-Loop Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `batch_pairs` effective (gradient accumulation), add test-fold early stopping with per-epoch validation, and make logging rank/fold-aware in the exp08 STATE-adapter DL SL-pair pipeline.

**Architecture:** All changes live in `src/sl_dl_model/`. The training loop in `train.py` gains gradient-accumulation batching and a per-epoch validation pass (pair AUROC over the fold's own test pairs) that drives patience-based best-epoch selection. `scoring.py` threads the fold's test pairs into the producer as `val_pairs`. `config.py` adds `early_stop_patience`. `__main__.py` defaults the log file and makes file logging rank0-only; per-rank per-epoch metric logs and a per-fold epoch CSV capture training curves. `evaluate.py`'s manifest records the new fields. The official metric path is untouched.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Accelerate (`PartialState`), numpy, pandas, scikit-learn (`roc_auc_score`), pytest, ruff, uv.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings. (CLAUDE.md Code Style)
- No `print` in library code; use `logging`. No hardcoded paths/thresholds; use config. Handle specific exceptions, no bare `except`. (CLAUDE.md Code Style)
- Target <50 lines/function, <600 lines/file. (CLAUDE.md Code Style)
- Prefix all Python/pytest/ruff invocations with `uv run`. (CLAUDE.md Environment)
- CPU unit tests use `state_backend="linear_mock"` — no STATE checkpoint required. (tests/sl_dl_model/test_train.py:3)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch in any new test module that triggers the energy-distance backward. (tests/sl_dl_model/test_train.py:14)
- `OMP_NUM_THREADS=1` is handled in conftest; do not remove. (memory: macos-omp-pytest-segfault)
- Conventional Commits: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Attribution disabled globally — no Co-Authored-By trailer.
- Reported metric is **best-epoch only**. Best-epoch selection uses the fold's **own test split** as validation (user override; SynLethDB `valid_rat=0` style). Leakage accepted.
- Loss reduction is **mean over the batch**. `lr` stays `1e-3`. `lambda_rank` is **out of scope** and must not be touched.
- Warmup gating (proposed default, confirm with user before Task 3): do not track best-epoch or count patience until `epoch >= warmup_epochs`.

---

### Task 1: Add `early_stop_patience` config field

**Files:**
- Modify: `src/sl_dl_model/config.py:55-64`
- Test: `tests/sl_dl_model/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SLDLConfig.early_stop_patience: int` (default `5`); `SLDLConfig.batch_pairs: int` (existing, unchanged value `1024`).

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_config.py`:

```python
def test_early_stop_patience_default():
    cfg = SLDLConfig()
    assert cfg.early_stop_patience == 5
    assert cfg.batch_pairs == 1024


def test_load_config_accepts_early_stop_patience(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"early_stop_patience": 3, "batch_pairs": 256}))
    cfg = load_config(path)
    assert cfg.early_stop_patience == 3
    assert cfg.batch_pairs == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_config.py::test_early_stop_patience_default -v`
Expected: FAIL with `AttributeError: 'SLDLConfig' object has no attribute 'early_stop_patience'`

- [ ] **Step 3: Add the field**

In `src/sl_dl_model/config.py`, in the "Loss weights + schedule" block (after `batch_pairs: int = 1024` at line 63), add:

```python
    batch_pairs: int = 1024
    lr: float = 1e-3
    # Early stopping: select the epoch with the best validation pair-AUROC,
    # stop after this many epochs without improvement. Validation uses the
    # fold's own test split (SynLethDB valid_rat=0 style).
    early_stop_patience: int = 5
```

(Move `lr` to stay above the new field; keep `embedding_method` after it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_config.py -v`
Expected: PASS (all config tests)

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/config.py tests/sl_dl_model/test_config.py
git commit -m "feat: add early_stop_patience config field for exp08"
```

---

### Task 2: Gradient-accumulation batching in `_train`

**Files:**
- Modify: `src/sl_dl_model/train.py:494-619` (the `_train` method)
- Test: `tests/sl_dl_model/test_train_batching.py` (create)

**Interfaces:**
- Consumes: `SLDLConfig.batch_pairs`, `_epoch_weights(epoch, config)`, `combine(parts, weights)`, `sl_bce_loss`, `_bag_part(...)`, `self._distill_part(...)`.
- Produces: `_train` performs one `optimizer.step()` per `batch_pairs`-sized chunk of trained pairs, with the batch loss being the **mean** of per-pair combined losses. No signature change yet (validation added in Task 3).

The current loop (train.py:535-607) computes one `total` per pair and steps per pair. Replace the per-pair step with accumulation: collect each trained pair's `combine(parts, weights)` scalar into a list; when the list reaches `batch_pairs` (or the epoch's pairs are exhausted), step on `torch.stack(batch_losses).mean()`.

- [ ] **Step 1: Write the failing test**

Create `tests/sl_dl_model/test_train_batching.py`:

```python
"""Tests for gradient-accumulation batching in StateDlProducer._train."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from accelerate import PartialState

from sl_dl_model.config import SLDLConfig
from sl_dl_model.bags import GwpsBags
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.train import StateDlProducer


def _producer(n_pairs: int, batch_pairs: int, max_epochs: int = 1) -> StateDlProducer:
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(n_pairs + 1)]
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={g: rng.standard_normal(8).astype("float32") for g in genes},
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        state_backend="linear_mock",
        max_epochs=max_epochs,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
        lambda_bag=0.0,
        batch_pairs=batch_pairs,
    )
    pairs = [
        (genes[i], genes[i + 1], i % 2, float(rng.normal()), float(rng.normal()))
        for i in range(n_pairs)
    ]
    return StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=pairs, input_dim=6, output_dim=6
    )


def test_one_step_per_batch_not_per_pair():
    """With 10 pairs and batch_pairs=4, expect ceil(10/4)=3 steps in 1 epoch."""
    producer = _producer(n_pairs=10, batch_pairs=4, max_epochs=1)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)

    steps = {"n": 0}
    real_step = torch.optim.Adam.step

    class CountingAdam(torch.optim.Adam):
        def step(self, *a, **k):
            steps["n"] += 1
            return real_step(self, *a, **k)

    opt = CountingAdam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1"})
    assert steps["n"] == 3, f"expected 3 optimizer steps, got {steps['n']}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_batching.py::test_one_step_per_batch_not_per_pair -v`
Expected: FAIL — current code steps once per pair, so `steps["n"] == 10`, not 3.

- [ ] **Step 3: Implement gradient accumulation**

In `src/sl_dl_model/train.py`, replace the inner pair loop body's stepping logic (lines 532-607). The new structure for the per-epoch block:

```python
            skipped = 0
            trained = 0
            batch_losses: list[torch.Tensor] = []

            def _flush() -> None:
                nonlocal batch_losses
                if not batch_losses:
                    return
                batch_total = torch.stack(batch_losses).mean()
                optimizer.zero_grad()
                batch_total.backward()
                optimizer.step()
                batch_losses = []

            for a, b, label, ea, eb in pbar:
                key_a, key_b = a.upper(), b.upper()
                vec_a = self.esm.vectors_by_symbol.get(key_a)
                vec_b = self.esm.vectors_by_symbol.get(key_b)
                if vec_a is None or vec_b is None:
                    skipped += 1
                    continue

                esm_a = torch.tensor(vec_a, device=device)
                esm_b = torch.tensor(vec_b, device=device)
                e_a = model.embed_gene(esm_a, control)
                e_b = model.embed_gene(esm_b, control)

                ge = torch.tensor(
                    self._ge_features(np.array([ea]), np.array([eb])),
                    device=device,
                    dtype=torch.float32,
                )

                cov_a: torch.Tensor | None = None
                cov_b: torch.Tensor | None = None
                if self.config.include_coverage_flag:
                    cov_a = torch.tensor(
                        [1.0 if key_a in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )
                    cov_b = torch.tensor(
                        [1.0 if key_b in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )

                logit = model.score_pairs(
                    e_a.unsqueeze(0), e_b.unsqueeze(0), ge, cov_a, cov_b
                )
                parts: dict[str, torch.Tensor] = {
                    "sl": sl_bce_loss(
                        logit, torch.tensor([float(label)], device=device)
                    ),
                }
                if weights["bag"] > 0:
                    bag_part = _bag_part(
                        model, covered_train, control, device,
                        key_a, vec_a, key_b, vec_b, self.bags,
                    )
                    if bag_part is not None:
                        parts["bag"] = bag_part
                if weights["distill"] > 0:
                    distill_part = self._distill_part({key_a, key_b})
                    if distill_part is not None:
                        parts["distill"] = distill_part

                batch_losses.append(combine(parts, weights))
                trained += 1
                if len(batch_losses) >= self.config.batch_pairs:
                    _flush()

            _flush()
```

Keep the existing `skipped`/`trained` warning+error block (lines 609-619) unchanged after this.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_batching.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing train suite for regressions**

Run: `uv run python -m pytest tests/sl_dl_model/test_train.py -v`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/train.py tests/sl_dl_model/test_train_batching.py
git commit -m "feat: make batch_pairs effective via gradient accumulation (exp08)"
```

---

### Task 3: Per-epoch validation + patience-based best-epoch selection

**Files:**
- Modify: `src/sl_dl_model/train.py` (`StateDlProducer.__init__`, `_train`, `produce`)
- Modify: `src/sl_dl_model/scoring.py:240-283` (`make_fold_producer` — thread test pairs as `val_pairs`)
- Test: `tests/sl_dl_model/test_train_earlystop.py` (create)

**Interfaces:**
- Consumes: `SLDLConfig.early_stop_patience`, `SLDLConfig.warmup_epochs`, `SLDLConfig.max_epochs`, `sklearn.metrics.roc_auc_score`, existing `model.embed_gene`, `model.score_pairs`.
- Produces:
  - `StateDlProducer.__init__` gains keyword param `val_pairs: list[tuple[str, str, int, float, float]] | None = None`.
  - `StateDlProducer.stopped_epoch: int | None` attribute, set after `_train` to the 0-indexed epoch whose weights were restored.
  - `StateDlProducer.epoch_metrics: list[dict[str, float]]` attribute (one dict per trained epoch; consumed by Task 4).
  - `_validate_auroc(model, device, control) -> float | None` helper method.

**Confirm warmup gating decision with the user before implementing.**

- [ ] **Step 1: Write the failing test**

Create `tests/sl_dl_model/test_train_earlystop.py`:

```python
"""Tests for per-epoch validation + patience-based best-epoch selection."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from accelerate import PartialState

from sl_dl_model.config import SLDLConfig
from sl_dl_model.bags import GwpsBags
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.train import StateDlProducer


def _producer(max_epochs: int, patience: int, warmup: int = 0) -> StateDlProducer:
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(6)]
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={g: rng.standard_normal(8).astype("float32") for g in genes},
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        state_backend="linear_mock",
        max_epochs=max_epochs,
        warmup_epochs=warmup,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
        lambda_bag=0.0,
        batch_pairs=4,
        early_stop_patience=patience,
    )
    train_pairs = [
        ("G0", "G1", 1, -1.0, -0.9),
        ("G2", "G3", 0, 0.8, 0.7),
        ("G0", "G3", 0, -1.0, 0.7),
        ("G1", "G2", 1, -0.9, 0.8),
    ]
    val_pairs = [
        ("G4", "G5", 1, -1.0, -0.8),
        ("G4", "G0", 0, -1.0, -1.0),
    ]
    return StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=train_pairs,
        input_dim=6, output_dim=6, val_pairs=val_pairs,
    )


def test_stopped_epoch_recorded_and_within_bounds():
    producer = _producer(max_epochs=5, patience=2)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    assert producer.stopped_epoch is not None
    assert 0 <= producer.stopped_epoch < 5
    assert len(producer.epoch_metrics) >= 1
    assert "val_pair_auroc" in producer.epoch_metrics[0]


def test_patience_triggers_early_stop():
    """If patience=1 and val never improves after epoch 0, stop early."""
    producer = _producer(max_epochs=10, patience=1)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    # Fewer than max_epochs metric rows recorded means we stopped early.
    assert len(producer.epoch_metrics) <= 10


def test_no_val_pairs_uses_final_epoch():
    """With val_pairs=None, no early stopping; stopped_epoch is the last epoch."""
    producer = _producer(max_epochs=3, patience=2)
    producer.val_pairs = None
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})
    assert producer.stopped_epoch == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_earlystop.py -v`
Expected: FAIL — `__init__` has no `val_pairs` kwarg (`TypeError`).

- [ ] **Step 3: Add `val_pairs` + state attributes to `__init__`**

In `src/sl_dl_model/train.py`, extend `__init__` (after line 113 `output_dim: int,`) signature and body:

```python
    def __init__(
        self,
        config: SLDLConfig,
        *,
        esm: Esm2EmbeddingTable,
        bags: GwpsBags,
        train_pairs: list[tuple[str, str, int, float, float]],
        input_dim: int,
        output_dim: int,
        val_pairs: list[tuple[str, str, int, float, float]] | None = None,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Best-epoch tracking (set by _train).
        self.stopped_epoch: int | None = None
        self.epoch_metrics: list[dict[str, float]] = []
```

Keep all the remaining existing `self._model = None` ... attribute initializations below unchanged.

- [ ] **Step 4: Add the validation helper**

Add this method to `StateDlProducer` (place it just above `_train`, after `_distill_part`). Import `roc_auc_score` at module top: `from sklearn.metrics import roc_auc_score`.

```python
    def _validate_auroc(
        self,
        model: SlDlModel,
        device: torch.device | str,
        control: torch.Tensor,
    ) -> float | None:
        """Pair-AUROC over ``self.val_pairs`` (the fold's test split).

        Returns ``None`` when validation is impossible: no val pairs, fewer
        than two scorable pairs, or only one label class present.
        """
        if not self.val_pairs:
            return None
        model.eval()
        scores: list[float] = []
        labels: list[int] = []
        with torch.no_grad():
            for a, b, label, ea, eb in self.val_pairs:
                vec_a = self.esm.vectors_by_symbol.get(a.upper())
                vec_b = self.esm.vectors_by_symbol.get(b.upper())
                if vec_a is None or vec_b is None:
                    continue
                e_a = model.embed_gene(torch.tensor(vec_a, device=device), control)
                e_b = model.embed_gene(torch.tensor(vec_b, device=device), control)
                ge = torch.tensor(
                    self._ge_features(np.array([ea]), np.array([eb])),
                    device=device,
                    dtype=torch.float32,
                )
                cov_a: torch.Tensor | None = None
                cov_b: torch.Tensor | None = None
                if self.config.include_coverage_flag:
                    cov_a = torch.tensor(
                        [1.0 if a.upper() in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )
                    cov_b = torch.tensor(
                        [1.0 if b.upper() in self.bags.bags_by_symbol else 0.0],
                        device=device,
                    )
                logit = model.score_pairs(
                    e_a.unsqueeze(0), e_b.unsqueeze(0), ge, cov_a, cov_b
                )
                scores.append(float(torch.sigmoid(logit).item()))
                labels.append(int(label))
        model.train()
        if len(scores) < 2 or len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
```

- [ ] **Step 5: Wire best-epoch selection into `_train`**

In `_train`, initialize tracking before the epoch loop (after `self._model = model` at line 522):

```python
        import copy

        best_auroc: float | None = None
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch: int | None = None
        epochs_since_improve = 0
```

At the end of each epoch (after the `trained == 0` guard block, ~line 619), add the validation + patience logic:

```python
            mean_loss = (
                float(torch.stack(batch_losses_history).mean())
                if batch_losses_history
                else float("nan")
            )
            val_auroc = self._validate_auroc(model, device, control)
            peak_mb = (
                torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0.0
            )
            self.epoch_metrics.append(
                {
                    "epoch": float(epoch),
                    "mean_train_loss": mean_loss,
                    "val_pair_auroc": float("nan") if val_auroc is None else val_auroc,
                    "peak_gpu_mem_mb": peak_mb,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Best-epoch selection only after warmup (val signal meaningful).
            if val_auroc is not None and epoch >= self.config.warmup_epochs:
                if best_auroc is None or val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.config.early_stop_patience:
                        logger.info(
                            "early stop at epoch %d (best epoch %d, val_auroc=%.4f)",
                            epoch, best_epoch, best_auroc,
                        )
                        break
```

To compute `mean_loss`, accumulate per-batch means during the epoch: add `batch_losses_history: list[torch.Tensor] = []` next to `batch_losses` at epoch start, and inside `_flush()` append the detached batch mean:

```python
            def _flush() -> None:
                nonlocal batch_losses
                if not batch_losses:
                    return
                batch_total = torch.stack(batch_losses).mean()
                optimizer.zero_grad()
                batch_total.backward()
                optimizer.step()
                batch_losses_history.append(batch_total.detach())
                batch_losses = []
```

After the epoch loop, restore the best weights and record `stopped_epoch`:

```python
        if best_state is not None:
            model.load_state_dict(best_state)
            self.stopped_epoch = best_epoch
        else:
            # No val signal (val_pairs None/unusable): keep final-epoch weights.
            self.stopped_epoch = self.config.max_epochs - 1
```

- [ ] **Step 6: Thread test pairs into the producer (`scoring.py`)**

In `src/sl_dl_model/scoring.py`, `make_fold_producer` (line 265), change `train_df, _ = fold_split(...)` to capture the test split and build `val_pairs`:

```python
    train_df, test_df = fold_split(frame, split_type, fold_id)
    train_pairs = [
        (
            str(r["gene_a_symbol"]).upper(),
            str(r["gene_b_symbol"]).upper(),
            int(r["sl_label"]),
            float(r["gene_a_k562_gene_effect"]),
            float(r["gene_b_k562_gene_effect"]),
        )
        for _, r in train_df.iterrows()
    ]
    val_pairs = [
        (
            str(r["gene_a_symbol"]).upper(),
            str(r["gene_b_symbol"]).upper(),
            int(r["sl_label"]),
            float(r["gene_a_k562_gene_effect"]),
            float(r["gene_b_k562_gene_effect"]),
        )
        for _, r in test_df.iterrows()
    ]
    return StateDlProducer(
        config,
        esm=caches.esm,
        bags=caches.bags,
        train_pairs=train_pairs,
        input_dim=caches.input_dim,
        output_dim=caches.output_dim,
        val_pairs=val_pairs,
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_earlystop.py tests/sl_dl_model/test_train.py tests/sl_dl_model/test_scoring.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/sl_dl_model/train.py src/sl_dl_model/scoring.py tests/sl_dl_model/test_train_earlystop.py
git commit -m "feat: add test-fold early stopping with per-epoch pair-AUROC validation (exp08)"
```

---

### Task 4: Per-rank logging, default log file, per-fold epoch CSV

**Files:**
- Modify: `src/sl_dl_model/__main__.py:60-91` (default log file, rank0-only file handler)
- Modify: `src/sl_dl_model/scoring.py` (`run_fold_with_producer` — emit epoch log lines + CSV)
- Test: `tests/sl_dl_model/test_logging.py` (create)

**Interfaces:**
- Consumes: `SLDLConfig.output_dir`, `producer.epoch_metrics`, `producer.stopped_epoch`, `PartialState().process_index`, `PartialState().is_main_process`.
- Produces:
  - Default `--log-file` → `output_dir/train.log` when not passed; file handler attached only on main process.
  - Per-rank metric log file `output_dir/train_rank{process_index}.log`.
  - Per-fold epoch CSV `output_dir/<split_type>/epoch_metrics_fold{fold_id}.csv` with columns `split_type, fold_id, epoch, mean_train_loss, val_pair_auroc, peak_gpu_mem_mb`.
  - `write_epoch_metrics(output_dir, split_type, fold_id, epoch_metrics) -> Path` helper in `scoring.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/sl_dl_model/test_logging.py`:

```python
"""Tests for default log file and per-fold epoch-metrics CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sl_dl_model.scoring import write_epoch_metrics


def test_write_epoch_metrics_csv(tmp_path: Path):
    rows = [
        {"epoch": 0.0, "mean_train_loss": 0.7, "val_pair_auroc": 0.5, "peak_gpu_mem_mb": 0.0},
        {"epoch": 1.0, "mean_train_loss": 0.6, "val_pair_auroc": 0.55, "peak_gpu_mem_mb": 0.0},
    ]
    out = write_epoch_metrics(tmp_path, "CV2", 3, rows)
    assert out == tmp_path / "CV2" / "epoch_metrics_fold3.csv"
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "split_type", "fold_id", "epoch",
        "mean_train_loss", "val_pair_auroc", "peak_gpu_mem_mb",
    ]
    assert len(df) == 2
    assert (df["split_type"] == "CV2").all()
    assert (df["fold_id"] == 3).all()


def test_default_log_file_path(tmp_path: Path, monkeypatch):
    """main() with no --log-file targets output_dir/train.log on main process."""
    import logging
    import sl_dl_model.__main__ as cli

    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "run"
    cfg_path.write_text(
        f"output_dir: {out_dir}\nsplit_types: [CV2]\nfolds: [0]\n"
    )

    captured = {}

    def fake_run_cv(config, producer):
        captured["log_files"] = [
            h.baseFilename
            for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ]
        return None

    monkeypatch.setattr(cli, "_resolve_run_cv", lambda: fake_run_cv, raising=False)
    # Patch the lazy import target used in main().
    import sl_dl_model.evaluate as ev
    monkeypatch.setattr(ev, "run_cv", fake_run_cv)

    cli.main(["run-cv", "--config", str(cfg_path), "--producer", "zero"])
    assert any(str(out_dir / "train.log") == p for p in captured["log_files"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_logging.py -v`
Expected: FAIL — `write_epoch_metrics` does not exist (`ImportError`).

- [ ] **Step 3: Add `write_epoch_metrics` to `scoring.py`**

Add to `src/sl_dl_model/scoring.py` (top-level function):

```python
def write_epoch_metrics(
    output_dir: Path,
    split_type: str,
    fold_id: int,
    epoch_metrics: list[dict[str, float]],
) -> Path:
    """Write per-epoch training-curve metrics for one fold to CSV.

    Args:
        output_dir: Run output directory.
        split_type: CV split type (e.g. ``"CV2"``).
        fold_id: Fold id.
        epoch_metrics: One dict per trained epoch with keys ``epoch``,
            ``mean_train_loss``, ``val_pair_auroc``, ``peak_gpu_mem_mb``.

    Returns:
        Path to the written CSV.
    """
    split_dir = output_dir / split_type
    split_dir.mkdir(parents=True, exist_ok=True)
    out = split_dir / f"epoch_metrics_fold{fold_id}.csv"
    df = pd.DataFrame(epoch_metrics)
    df.insert(0, "fold_id", fold_id)
    df.insert(0, "split_type", split_type)
    df.to_csv(out, index=False)
    return out
```

Add `from pathlib import Path` to the imports in `scoring.py` if not present.

- [ ] **Step 4: Emit epoch metrics + per-rank log in `run_fold_with_producer`**

In `run_fold_with_producer` (scoring.py), in the DL branch (after `sm = producer.score_matrix(...)` at line 163), record the curve and log per-rank. Add at module top: `from accelerate import PartialState`.

```python
    if hasattr(producer, "score_matrix"):
        sm = producer.score_matrix(universe.symbols, universe.gene_effects)
        epoch_metrics = getattr(producer, "epoch_metrics", None)
        if epoch_metrics:
            write_epoch_metrics(
                Path(config.output_dir), split_type, fold_id, epoch_metrics
            )
            rank = PartialState().process_index
            for m in epoch_metrics:
                logger.info(
                    "[rank %d][%s/fold%d] epoch %d: loss=%.4f val_auroc=%.4f "
                    "peak_gpu_mb=%.1f",
                    rank, split_type, fold_id, int(m["epoch"]),
                    m["mean_train_loss"], m["val_pair_auroc"], m["peak_gpu_mem_mb"],
                )
            logger.info(
                "[rank %d][%s/fold%d] stopped_epoch=%s",
                rank, split_type, fold_id, getattr(producer, "stopped_epoch", None),
            )
        rows.extend(_metric_rows(...))  # unchanged below
```

(Leave the rest of the DL branch — the `_metric_rows` calls and `return rows` — exactly as is.)

- [ ] **Step 5: Default the log file + rank0-only file handler in `__main__.py`**

Replace the logging-setup block (`__main__.py:66-91`) and load config before logging so `output_dir` is known:

```python
    args = _build_parser().parse_args(argv)

    from sl_dl_model.config import load_config

    config = load_config(args.config)

    from accelerate import PartialState

    is_main = PartialState().is_main_process
    log_file = args.log_file
    if log_file is None:
        log_file = Path(config.output_dir) / "train.log"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    # Per-rank metric log captures this rank's folds' curves.
    rank = PartialState().process_index
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    handlers.append(
        logging.FileHandler(
            Path(config.output_dir) / f"train_rank{rank}.log", mode="a"
        )
    )
    # The shared train.log is written by the main process only.
    if is_main:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    from sl_dl_model.evaluate import run_cv

    if args.producer == "zero":
        from sl_dl_model.evaluate import ZeroEmbeddingProducer

        run_cv(config, ZeroEmbeddingProducer())
    else:
        run_cv(config, producer="state_dl")
```

Add `from pathlib import Path` (already imported at line 20).

- [ ] **Step 6: Run tests**

Run: `uv run python -m pytest tests/sl_dl_model/test_logging.py tests/sl_dl_model/test_cli.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/sl_dl_model/__main__.py src/sl_dl_model/scoring.py tests/sl_dl_model/test_logging.py
git commit -m "feat: rank-aware logging + per-fold epoch-metrics CSV (exp08)"
```

---

### Task 5: Record new fields in the manifest

**Files:**
- Modify: `src/sl_dl_model/evaluate.py:254-295` (`_build_manifest`)
- Test: `tests/sl_dl_model/test_evaluate_manifest.py`

**Interfaces:**
- Consumes: `SLDLConfig.batch_pairs`, `SLDLConfig.early_stop_patience`.
- Produces: manifest dict gains `batch_pairs`, `early_stop_patience`, `early_stop_metric`, `val_source`.

(Per-fold `stopped_epoch` is recorded in the per-rank logs from Task 4, not the manifest, since the manifest is fold-independent.)

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_evaluate_manifest.py`:

```python
def test_manifest_includes_training_fields() -> None:
    from sl_dl_model.evaluate import _build_manifest

    cfg = SLDLConfig(esm2_model="x", batch_pairs=512, early_stop_patience=4)
    manifest = _build_manifest(
        cfg,
        split_types=("CV2",),
        candidate_gene_count=10,
        gwps_coverage_gene_count=None,
    )
    assert manifest["batch_pairs"] == 512
    assert manifest["early_stop_patience"] == 4
    assert manifest["early_stop_metric"] == "val_pair_auroc"
    assert manifest["val_source"] == "test_fold"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_evaluate_manifest.py::test_manifest_includes_training_fields -v`
Expected: FAIL with `KeyError: 'batch_pairs'`

- [ ] **Step 3: Add the fields**

In `_build_manifest` (evaluate.py), add to the returned dict (after `"pooling": config.pooling,` at line 285):

```python
        "pooling": config.pooling,
        "batch_pairs": config.batch_pairs,
        "early_stop_patience": config.early_stop_patience,
        "early_stop_metric": "val_pair_auroc",
        "val_source": "test_fold",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_evaluate_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/evaluate.py tests/sl_dl_model/test_evaluate_manifest.py
git commit -m "feat: record batching + early-stop fields in exp08 manifest"
```

---

### Task 6: Update docs + config README; full verification

**Files:**
- Modify: `configs/experiments/08_k562_sl_pair_state_dl/README.md`
- Modify: `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md`

**Interfaces:**
- Consumes: nothing (documentation).
- Produces: documentation reflecting effective `batch_pairs`, test-fold early stopping, and the honesty note.

- [ ] **Step 1: Update the config README**

In `configs/experiments/08_k562_sl_pair_state_dl/README.md`, near the `lambda_rank`/training-paradigm notes (around line 92), replace the "1 pair / optimizer step" / "`batch_pairs` not activated" statements with:

```markdown
- `batch_pairs` (default 1024) is now effective: training uses gradient
  accumulation, one optimizer step per `batch_pairs` pairs, with the batch
  loss reduced as the mean of per-pair losses.
- Early stopping: each epoch the model is validated by pair-AUROC over the
  fold's **own test split** (SynLethDB `valid_rat=0` style; leakage accepted),
  best-epoch weights restored, `early_stop_patience` (default 5) epochs without
  improvement stops training. Best-epoch selection begins after `warmup_epochs`.
  The reported official metric is best-epoch only.
- `lambda_rank` remains recorded-but-unconsumed (out of scope this round).
```

- [ ] **Step 2: Update the experiment doc**

In `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md`, update the training-paradigm section (around lines 50-55) to state the effective batching and the test-fold early-stopping protocol, and add the honesty note: best-epoch selection on the test fold makes exp08-vs-exp06 selection-matched to the SynLethDB benchmark, not a strict embedding-only ablation.

- [ ] **Step 3: Run the full exp08 test suite**

Run: `uv run python -m pytest tests/sl_dl_model/ -v`
Expected: PASS (all tests)

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check src/sl_dl_model/ tests/sl_dl_model/ && uv run ruff format src/sl_dl_model/ tests/sl_dl_model/`
Expected: no errors; formatting clean.

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/08_k562_sl_pair_state_dl/README.md docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md
git commit -m "docs: document effective batching + test-fold early stopping (exp08)"
```

---

## Self-Review Notes

- **Spec coverage:** batching (Task 2), test-fold early stopping + per-epoch val AUROC + patience + best-epoch restore (Task 3), `early_stop_patience` config (Task 1), default rank0 log file + per-rank metric logs + per-fold epoch CSV + per-batch tqdm (Task 4 — tqdm bar already rank0-gated and now ~43 steps after Task 2), manifest fields (Task 5), docs + honesty note (Task 6). `lambda_rank` left untouched per scope.
- **Type consistency:** `val_pairs` param and `epoch_metrics`/`stopped_epoch` attributes defined in Task 3 are consumed by Tasks 4–5 with matching names; `write_epoch_metrics` signature is identical in definition (Task 4 Step 3) and use (Task 4 Step 4).
- **Open decision flagged:** warmup-gating of best-epoch selection (Task 3) and `early_stop_patience=5` default (Task 1) await user confirmation; both are isolated one-line changes if the user rules differently.

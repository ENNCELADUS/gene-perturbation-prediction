# exp08 Phase 3 NaN Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the epoch-0 NaN that crashes exp08 Phase 3 (`phase3_bag_supervision.yaml`) at validation, by adding NaN-safe numerics and defense-in-depth (gradient clipping + finite checks) to `src/sl_dl_model/`.

**Architecture:** Four targeted defenses, each mapped to a ranked hypothesis from the 2026-06-20 root-cause investigation (see `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md` "Phase 3 NaN Blocker"). H1: replace the `torch.cdist` energy distance (NaN in both forward mm-mode `sqrt(neg)` and backward self-distance `0/0`) with a NaN-safe quadratic-form variant local to `sl_dl_model.losses`. H3: replace `MeanStdPool`'s `std()` (`sqrt(0)` grad on constant features) with an eps-guarded form. H2/H4: add gradient clipping plus a per-step finite guard that skips an optimizer step whose loss or gradients are non-finite, so a transient blow-up cannot corrupt the trainable weights. Plus a defensive finite guard in validation so one bad fold returns "no signal" instead of killing a multi-fold cluster run.

**Tech Stack:** Python 3.11+, PyTorch, pytest, `uv`. Frozen Arc STATE backbone + trainable adapter (unchanged). exp05's shared `aivc_model.model._energy_distance` is **left untouched** (it is separately validated and shows no NaN); exp08 gets its own safe copy.

## Global Constraints

- All Python/pytest/ruff invocations are prefixed with `uv run`.
- Python 3.11+, strict type hints, Google-style docstrings, absolute imports.
- No `print` in library code; use the module `logger`.
- No hardcoded tuning thresholds — `max_grad_norm` is a real hyperparameter and goes in `SLDLConfig`. Pure numerical-stability epsilons (machine-epsilon guards inside a single op) are named module-level constants, not config.
- Do NOT modify `src/aivc_model/` — exp05 numerics must not shift.
- Tests live under `tests/sl_dl_model/`, use `state_backend="linear_mock"`, and rely on the existing `tests/conftest.py` (sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, `OMP_NUM_THREADS=1`).
- `torch.cdist` backward is unimplemented on MPS; the new safe energy distance must use only ops with CPU/MPS-fallback backward (the quadratic form does).

---

<!-- TASKS -->

## File Structure

- `src/sl_dl_model/losses.py` — **modify**: add `_safe_energy_distance` (H1), repoint `bag_loss` to it, drop the `aivc_model._energy_distance` import.
- `src/sl_dl_model/pooling.py` — **modify**: eps-guard the std in `MeanStdPool` (H3).
- `src/sl_dl_model/config.py` — **modify**: add `max_grad_norm` field (H2/H4 hyperparameter).
- `src/sl_dl_model/train.py` — **modify**: add module-level `safe_optimizer_step` (H2/H4), call it from `_flush`; add a finite guard to `_validate_auroc`.
- `tests/sl_dl_model/test_losses.py`, `test_pooling.py`, `test_config.py` — **modify**: unit tests.
- `tests/sl_dl_model/test_nan_guards.py` — **create**: step-guard, validation-guard, and end-to-end training tests.

---

### Task 1: NaN-safe energy distance (H1)

**Files:**
- Modify: `src/sl_dl_model/losses.py`
- Test: `tests/sl_dl_model/test_losses.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_safe_energy_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor` (module-private). `bag_loss(pred_bag, real_bag) -> torch.Tensor` signature unchanged but now NaN-safe in forward and backward.

- [ ] **Step 1: Write the failing tests**

Add to `tests/sl_dl_model/test_losses.py`:

```python
def test_bag_loss_grad_finite_on_identical_bags():
    # H1b: cdist(x, x) self-distance has a 0/0 backward; identical pred/real
    # makes every cross- and self-distance zero, the exact NaN-grad trigger.
    pred = torch.randn(8, 6, requires_grad=True)
    real = pred.detach().clone()
    loss = bag_loss(pred, real)
    assert torch.isfinite(loss).all(), "bag_loss value must be finite"
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), (
        "bag_loss gradient must be finite on identical bags"
    )


def test_bag_loss_grad_finite_with_duplicate_rows():
    # Duplicate rows create zero pairwise distances inside a single bag.
    pred = torch.zeros(5, 4, requires_grad=True)
    real = torch.randn(7, 4)
    loss = bag_loss(pred, real)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/sl_dl_model/test_losses.py -k grad_finite -v`
Expected: FAIL — `pred.grad` contains NaN (assert fails), because `bag_loss` still routes through `aivc_model.model._energy_distance` (`torch.cdist`).

- [ ] **Step 3: Implement the NaN-safe energy distance**

In `src/sl_dl_model/losses.py`, remove the line `from aivc_model.model import _energy_distance` and add near the top (after the `F` import):

```python
# Epsilon added under the sqrt so the Euclidean-distance gradient stays finite
# at zero pairwise distance (the cdist self-distance 0/0 NaN trap, H1b).
_ENERGY_EPS = 1e-8


def _safe_energy_distance(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """NaN-safe energy distance between two cell bags.

    Equivalent to ``2*E||X-Y|| - E||X-X'|| - E||Y-Y'||`` but computed without
    ``torch.cdist``: pairwise squared distances are formed via the quadratic
    form, clamped to be non-negative (kills the float ``sqrt(negative)`` NaN,
    H1a), and an epsilon is added under the ``sqrt`` so the gradient is finite
    at zero distance (kills the self-distance ``0/0`` NaN, H1b).

    Args:
        predicted: Predicted bag, shape ``(n, D)``.
        target: Real bag, shape ``(m, D)``.

    Returns:
        Non-negative scalar energy distance.
    """
    cross = _safe_pairwise_dist(predicted, target).mean()
    pred_self = _safe_pairwise_dist(predicted, predicted).mean()
    target_self = _safe_pairwise_dist(target, target).mean()
    return (2.0 * cross - pred_self - target_self).clamp_min(0.0)


def _safe_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean pairwise distances with a finite gradient at zero distance.

    Args:
        a: Tensor of shape ``(n, D)``.
        b: Tensor of shape ``(m, D)``.

    Returns:
        Distance matrix of shape ``(n, m)``.
    """
    a2 = (a * a).sum(dim=-1, keepdim=True)  # (n, 1)
    b2 = (b * b).sum(dim=-1, keepdim=True)  # (m, 1)
    d2 = a2 - 2.0 * (a @ b.transpose(-2, -1)) + b2.transpose(-2, -1)
    d2 = d2.clamp_min(0.0)
    return torch.sqrt(d2 + _ENERGY_EPS)
```

Then change `bag_loss` to call the local function:

```python
    energy = _safe_energy_distance(pred_bag, real_bag)
```

(replacing the old `energy = _energy_distance(pred_bag, real_bag)` line).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_losses.py -v`
Expected: PASS (all four original tests + the two new grad-finite tests). The existing `test_bag_loss_nonnegative_and_zero_for_identical` still passes (identical bags → energy ≈ 0).

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/losses.py tests/sl_dl_model/test_losses.py
git commit -m "fix: NaN-safe energy distance for exp08 bag loss (H1)"
```

### Task 2: eps-guarded MeanStdPool std (H3)

**Files:**
- Modify: `src/sl_dl_model/pooling.py`
- Test: `tests/sl_dl_model/test_pooling.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `MeanStdPool.forward(bag)` — same `(2D,)` output shape and semantics, but the std branch has a finite gradient when a feature is constant across cells.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_pooling.py`:

```python
def test_mean_std_pool_grad_finite_on_constant_feature():
    # std = sqrt(var); var=0 on a constant feature gives sqrt'(0)=inf grad (H3).
    bag = torch.zeros(10, 6, requires_grad=True)  # every feature constant
    out = MeanStdPool()(bag)
    out.sum().backward()
    assert torch.isfinite(bag.grad).all(), "pooling grad must be finite at var=0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_pooling.py -k grad_finite -v`
Expected: FAIL — `bag.grad` contains NaN/inf because `bag.std(dim=0)` backward divides by zero std.

- [ ] **Step 3: Implement the eps-guarded std**

In `src/sl_dl_model/pooling.py`, add a module-level constant after the imports:

```python
# Floor added to the variance before sqrt so the std gradient is finite when a
# feature is constant across the bag (sqrt'(0) is infinite otherwise, H3).
_STD_EPS = 1e-8
```

Replace the body of `MeanStdPool.forward`:

```python
        mean = bag.mean(dim=0)
        var = bag.var(dim=0, unbiased=False)
        std = torch.sqrt(var + _STD_EPS)
        return torch.cat([mean, std], dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_pooling.py -v`
Expected: PASS. `test_mean_std_pool_dim` still passes (output shape `(12,)` unchanged); the new grad-finite test passes.

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/pooling.py tests/sl_dl_model/test_pooling.py
git commit -m "fix: eps-guard MeanStdPool std for finite grad at zero variance (H3)"
```

### Task 3: `max_grad_norm` config field (H2/H4 hyperparameter)

**Files:**
- Modify: `src/sl_dl_model/config.py`
- Test: `tests/sl_dl_model/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SLDLConfig.max_grad_norm: float` (default `1.0`). Consumed by `safe_optimizer_step` in Task 4. `max_grad_norm <= 0` disables clipping.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_config.py`:

```python
def test_max_grad_norm_default_and_yaml_override(tmp_path):
    from sl_dl_model.config import SLDLConfig, load_config

    assert SLDLConfig().max_grad_norm == 1.0
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("max_grad_norm: 0.5\n")
    assert load_config(cfg_path).max_grad_norm == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_config.py -k max_grad_norm -v`
Expected: FAIL — `AttributeError: 'SLDLConfig' object has no attribute 'max_grad_norm'` (and `load_config` would raise `unknown config keys`).

- [ ] **Step 3: Add the field**

In `src/sl_dl_model/config.py`, inside the `SLDLConfig` dataclass, add after the `early_stop_patience` field (line ~68):

```python
    # Gradient clipping max-norm applied before every optimizer step (H2/H4
    # NaN defense). <= 0 disables clipping.
    max_grad_norm: float = 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_config.py -v`
Expected: PASS (new test + all existing config tests; `load_config` now accepts `max_grad_norm`).

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/config.py tests/sl_dl_model/test_config.py
git commit -m "feat: add max_grad_norm config field for exp08 grad clipping"
```

### Task 4: Gradient clipping + finite step guard in `_flush` (H2/H4)

**Files:**
- Modify: `src/sl_dl_model/train.py`
- Test: `tests/sl_dl_model/test_nan_guards.py` (create)

**Interfaces:**
- Consumes: `SLDLConfig.max_grad_norm` (Task 3); `bag_loss` (Task 1).
- Produces: module-level `safe_optimizer_step(model, optimizer, loss, max_grad_norm, *, logger_=logger) -> bool` in `sl_dl_model.train`. Returns `True` if the step was applied, `False` if it was skipped because the loss or the post-clip grad norm was non-finite. `_flush` calls it instead of the inline `backward`/`step`.

**Rationale:** Grad clipping bounds the H2 blow-up; the finite guard is the backstop for H4 (and any residual H1/H3) — a non-finite loss or gradient must never reach `optimizer.step()`, because one corrupted step poisons the weights for the rest of the epoch and only surfaces at validation.

- [ ] **Step 1: Write the failing test**

Create `tests/sl_dl_model/test_nan_guards.py`:

```python
"""NaN-defense guards for exp08 Phase 3 training (H1-H4)."""

from __future__ import annotations

import torch
from torch import nn, optim

from sl_dl_model.train import safe_optimizer_step


def test_safe_optimizer_step_applies_finite_step():
    model = nn.Linear(3, 1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = (model(torch.ones(2, 3)) - 2.0).pow(2).mean()
    applied = safe_optimizer_step(model, opt, loss, max_grad_norm=1.0)
    assert applied is True
    assert not torch.equal(before, model.weight), "finite step must update weights"
    assert torch.isfinite(model.weight).all()


def test_safe_optimizer_step_skips_nonfinite_loss():
    model = nn.Linear(3, 1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    # NaN loss that still carries grad_fn back to the params.
    loss = (model(torch.ones(2, 3)) * float("nan")).mean()
    applied = safe_optimizer_step(model, opt, loss, max_grad_norm=1.0)
    assert applied is False, "non-finite loss must skip the step"
    assert torch.equal(before, model.weight), "weights must be unchanged on skip"
    assert torch.isfinite(model.weight).all(), "weights must stay finite"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py -v`
Expected: FAIL — `ImportError: cannot import name 'safe_optimizer_step' from 'sl_dl_model.train'`.

- [ ] **Step 3: Implement `safe_optimizer_step`**

In `src/sl_dl_model/train.py`, add this module-level function (e.g. just above `_bag_part`):

```python
def safe_optimizer_step(
    model: SlDlModel,
    optimizer: optim.Optimizer,
    loss: torch.Tensor,
    max_grad_norm: float,
    *,
    logger_: logging.Logger = logger,
) -> bool:
    """Backward + clip + step, skipping the step if anything is non-finite.

    Guards exp08 Phase 3 training against NaN/Inf corruption (H2/H4): a
    non-finite loss or gradient must never reach ``optimizer.step()``, because
    one bad step poisons the weights for the rest of the epoch and only
    surfaces later at validation. ``optimizer.zero_grad()`` always runs so a
    skipped step leaves no stale gradients behind.

    Args:
        model: The model whose trainable parameters are being optimized.
        optimizer: The optimizer to step.
        loss: Scalar loss tensor (already reduced).
        max_grad_norm: Max gradient L2 norm for clipping; ``<= 0`` disables
            clipping (the finite check still runs).
        logger_: Logger for the skip warning.

    Returns:
        ``True`` if the optimizer step was applied; ``False`` if it was skipped
        because the loss or the post-clip gradient norm was non-finite.
    """
    optimizer.zero_grad()
    if not torch.isfinite(loss).all():
        logger_.warning("non-finite loss (%s); skipping optimizer step", loss)
        return False
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    if not torch.isfinite(grad_norm):
        logger_.warning(
            "non-finite gradient norm (%s); skipping optimizer step", grad_norm
        )
        optimizer.zero_grad()
        return False
    optimizer.step()
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py -v`
Expected: PASS (both step-guard tests).

- [ ] **Step 5: Wire `safe_optimizer_step` into `_flush`**

In `src/sl_dl_model/train.py`, replace the body of the nested `_flush` function inside `_train` (currently lines ~623-635). The current body is:

```python
            def _flush() -> None:
                nonlocal batch_loss_count, batch_loss_sum, batch_losses
                if not batch_losses:
                    return
                batch_total = torch.stack(batch_losses).mean()
                batch_size = len(batch_losses)
                optimizer.zero_grad()
                batch_total.backward()
                optimizer.step()
                batch_loss_sum += float(batch_total.detach().cpu()) * batch_size
                batch_loss_count += batch_size
                batch_losses = []
                pbar.update(1)
```

Replace it with:

```python
            def _flush() -> None:
                nonlocal batch_loss_count, batch_loss_sum, batch_losses
                if not batch_losses:
                    return
                batch_total = torch.stack(batch_losses).mean()
                batch_size = len(batch_losses)
                applied = safe_optimizer_step(
                    model, optimizer, batch_total, self.config.max_grad_norm
                )
                if applied:
                    batch_loss_sum += float(batch_total.detach().cpu()) * batch_size
                    batch_loss_count += batch_size
                batch_losses = []
                pbar.update(1)
```

(A skipped step still advances `pbar` and clears `batch_losses`, but does not pollute the mean-loss accumulator with the non-finite value.)

- [ ] **Step 6: Add the end-to-end training guard test**

Append to `tests/sl_dl_model/test_nan_guards.py`. This drives the full `linear_mock` producer with bag supervision on and asserts the trained weights and recorded metrics are finite — the integration-level reproduction of the original crash:

```python
def test_train_with_bag_supervision_keeps_params_finite():
    import numpy as np

    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in symbols
        },
    )
    # Identical pred/real-style bags + a constant-feature bag exercise the H1
    # self-distance and H3 zero-variance paths inside a real training loop.
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={
            "A": np.zeros((8, 6), dtype="float32"),
            "B": rng.standard_normal((8, 6)).astype("float32"),
        },
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=2,
        warmup_epochs=1,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
        lambda_bag=1.0,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    pairs = [
        ("A", "B", 1, -1.0, -0.5),
        ("C", "D", 0, 0.1, 0.2),
        ("A", "C", 0, -1.0, 0.1),
    ]
    producer = StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=pairs, input_dim=6, output_dim=6,
        val_pairs=pairs,
    )
    emb, _mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert np.isfinite(emb).all(), "produced embeddings must be finite"
    model = producer._model
    assert model is not None
    assert all(
        torch.isfinite(p).all() for p in model.parameters()
    ), "all trained params must remain finite"
    assert all(
        np.isfinite(row["mean_train_loss"]) for row in producer.epoch_metrics
    ), "recorded train losses must be finite"
```

- [ ] **Step 7: Run tests + commit**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py tests/sl_dl_model/test_train.py -v`
Expected: PASS (guards + the full existing train suite still green).

```bash
git add src/sl_dl_model/train.py tests/sl_dl_model/test_nan_guards.py
git commit -m "fix: grad-clip + finite step guard in exp08 training (H2/H4)"
```

### Task 5: Finite guard in `_validate_auroc` (defensive — keep the cluster job alive)

**Files:**
- Modify: `src/sl_dl_model/train.py`
- Test: `tests/sl_dl_model/test_nan_guards.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_validate_auroc` returns `None` (instead of raising inside `roc_auc_score`) when any computed score is non-finite.

**Rationale:** Tasks 1-4 should prevent NaN at the source. This task hardens the *detector itself*: even if a residual non-finite value appears, validation should return "no signal" for that fold rather than crashing the entire Accelerate run (which owns multiple `(split_type, fold)` jobs per rank). It converts a fatal crash into a logged, skipped fold.

- [ ] **Step 1: Write the failing test**

Append to `tests/sl_dl_model/test_nan_guards.py`:

```python
def test_validate_auroc_returns_none_on_nonfinite_scores(monkeypatch):
    import numpy as np

    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={s: rng.standard_normal(8).astype("float32")
                           for s in ["A", "B"]},
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={"A": rng.standard_normal((8, 6)).astype("float32")},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x", state_backend="linear_mock", pert_dim=5,
        adapter_hidden=16, pair_hidden=(16,), include_coverage_flag=False,
    )
    val = [("A", "B", 1, -1.0, -0.5), ("B", "A", 0, 0.1, 0.2)]
    producer = StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=val, input_dim=6, output_dim=6,
        val_pairs=val,
    )
    model = producer._build_model()
    control = torch.zeros(8, 6)
    # Force the pair head to emit a NaN logit so sigmoid -> NaN score.
    monkeypatch.setattr(
        model, "score_pairs",
        lambda *a, **k: torch.tensor([float("nan")]),
    )
    assert producer._validate_auroc(model, "cpu", control) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py -k validate_auroc -v`
Expected: FAIL — `roc_auc_score` raises `ValueError: Input contains NaN` (the original crash), instead of returning `None`.

- [ ] **Step 3: Add the finite guard**

In `src/sl_dl_model/train.py`, in `_validate_auroc`, replace the final guard+return (currently lines ~552-554):

```python
        if len(scores) < 2 or len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
```

with:

```python
        if len(scores) < 2 or len(set(labels)) < 2:
            return None
        if not np.isfinite(scores).all():
            logger.warning(
                "non-finite validation score(s); returning no val signal for "
                "this epoch (%d/%d non-finite)",
                int((~np.isfinite(scores)).sum()),
                len(scores),
            )
            return None
        return float(roc_auc_score(labels, scores))
```

(`np` is already imported at the top of `train.py`. `scores` is a `list[float]`; `np.isfinite(scores)` coerces it to an array.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py -v`
Expected: PASS (all guard tests including the new validation test).

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/train.py tests/sl_dl_model/test_nan_guards.py
git commit -m "fix: validation returns no-signal on non-finite scores (exp08)"
```

### Task 6: Full suite + lint + Phase 3 smoke gate

**Files:**
- No source changes — verification only.

**Interfaces:**
- Consumes: all of Tasks 1-5.
- Produces: a green test suite, clean ruff, and a smoke-run protocol that confirms the original crash is gone before committing the cluster to the full Phase 3 run.

- [ ] **Step 1: Run the full sl_dl_model test suite**

Run: `uv run python -m pytest tests/sl_dl_model/ -v`
Expected: PASS — all pre-existing tests plus the new `test_nan_guards.py`, and the new asserts in `test_losses.py` / `test_pooling.py` / `test_config.py`.

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check src/sl_dl_model/ tests/sl_dl_model/ && uv run ruff format --check src/sl_dl_model/ tests/sl_dl_model/`
Expected: no errors. If `format --check` reports diffs, run `uv run ruff format src/sl_dl_model/ tests/sl_dl_model/` and re-stage.

- [ ] **Step 3: Confirm the `aivc_model` energy distance is untouched**

Run: `git diff --name-only HEAD~5 -- src/aivc_model/`
Expected: empty output (exp05 numerics unchanged — `_energy_distance` was copied into `sl_dl_model.losses`, not modified in place).

Run the exp05 model test to confirm no regression: `uv run python -m pytest tests/test_aivc_model.py -v`
Expected: PASS.

- [ ] **Step 4: Phase 3 smoke run on the cluster (short)**

On the cluster (where the ESM2 cache, STATE checkpoint, and gwps bags exist), run a single fold for a couple of epochs to confirm epoch-0 no longer crashes. Use a temporary selection-narrowed copy of the config or CLI flags:

Run:
```bash
uv run python -m sl_dl_model run-cv \
  --config configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml \
  --producer state_dl --split-type CV2 --fold 0
```
(Use whatever single-fold selection flag the CLI exposes; the goal is one short fold.)

Expected: epoch 0 completes, `_validate_auroc` returns a finite number (logged as `val_auroc=<finite>`), and `epoch_metrics_fold0.csv` shows finite `mean_train_loss` and `val_pair_auroc`. No `ValueError: Input contains NaN`. Watch the per-rank log for any `non-finite loss`/`non-finite gradient norm`/`non-finite validation score` warnings: a few early warnings are acceptable (the guard is working); persistent warnings every step mean H2 is undertuned — lower `lr` or `max_grad_norm` in the config and re-smoke before the full run.

- [ ] **Step 5: Commit any config/doc tweaks from the smoke run**

If the smoke run required a config change (e.g. `max_grad_norm` or `lr`), commit it:

```bash
git add configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml
git commit -m "chore: tune exp08 phase3 grad-clip/lr after NaN-fix smoke run"
```

---

## Success Criteria & Pipeline Gate

The fix is **successful** (and Phase 3 / the pipeline is unblocked) when all hold:

1. `uv run python -m pytest tests/sl_dl_model/ -v` is fully green, including the H1-H4 guard tests.
2. `tests/test_aivc_model.py` still passes (exp05 untouched).
3. The Phase 3 smoke run (Task 6 Step 4) completes epoch 0 and validation with **finite** loss and AUROC, no `Input contains NaN`.

If 1-3 hold, proceed to the full Phase 3 run (`split_types: [CV2, CV3]`, folds 0-4) and resume the exp08 success evaluation (beat exp06 CV2/CV3 NDCG@k / MAP@k per `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md`). After the full run starts cleanly, update the "Phase 3 NaN Blocker" section of that doc and the `exp08-task-definition` memory to mark the blocker resolved.

If the smoke run **still** produces NaN after Tasks 1-5: do not patch further blindly. Per systematic-debugging Phase 4.5, the per-step warnings now pinpoint which layer first goes non-finite (loss vs grad vs validation). Use that signal to confirm which hypothesis (H1/H2/H3/H4) actually fired and re-open the investigation with that evidence.

## Self-Review

- **Spec coverage (H1-H4):** H1 → Task 1 (`_safe_energy_distance`). H2 → Tasks 3+4 (grad clip + config). H3 → Task 2 (eps-guarded std). H4 → Task 4 (finite step guard, the backstop after STATE overflow). Plus Task 5 (defensive validation guard) and Task 6 (verification + smoke gate). All four hypotheses are covered with both a source fix and a test.
- **Defense-in-depth ordering:** source-level NaN-safe ops (Tasks 1-2) → bounded gradients (Tasks 3-4) → never-step-on-NaN backstop (Task 4) → detector hardening (Task 5). Each layer is independently tested.
- **No placeholders:** every code step shows the full function/edit; every run step shows the command and expected pass/fail.
- **Type consistency:** `safe_optimizer_step(model, optimizer, loss, max_grad_norm, *, logger_=logger) -> bool` is defined in Task 4 Step 3 and called identically in Task 4 Step 5. `SLDLConfig.max_grad_norm: float` defined in Task 3, consumed in Task 4. `_safe_energy_distance` / `_safe_pairwise_dist` defined and used within Task 1.
- **Don't-touch constraint:** Task 6 Step 3 explicitly verifies `src/aivc_model/` is unchanged.







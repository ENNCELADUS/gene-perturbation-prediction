# Exp05 End-to-End STATE Response-GMM 4-GPU DDP Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the authoritative exp05 scVI/ridge/fixed-GMM path with a fully differentiable ESM2 adapter -> unfrozen STATE -> shared `Linear(2000, 128)` response encoder -> trainable diagonal GMM -> GeneEffect head model, trained with observed-B and C co-supervision under mandatory 4-GPU DDP.

**Architecture:** Each outer fold is trained sequentially by one four-rank DDP job; every rank works on a disjoint shard of the same fold, so this plan explicitly forbids fold-per-GPU execution. Predicted and observed response bags share one response encoder, one trainable GMM pooler, and one C head; observed responses are available only for inner-train supervision and post-freeze outer-test diagnostics. The authoritative path no longer fits or caches scVI latents, a ridge projector, or a fixed sklearn GMM.

**Tech Stack:** Python 3.11, PyTorch 2.8+, Hugging Face Accelerate DDP, BF16, Arc STATE, NumPy memmaps, pandas, pytest, Ruff.

## Global Constraints

- The canonical universe remains exactly 9,338 GWPS-DepMap genes from `data/sl_dependency_v0/splits/k562_gwps_depmap_outer5_seed42.csv` and its frozen SHA-256 authority.
- One gene's GeneEffect label, GWPS response cells, transition supervision, prompt cells, and fine-tuning samples must remain in the same frozen outer fold.
- Every outer fold must use an inner validation split derived only from that fold's outer-train genes.
- Outer-test observed responses must not affect the ESM2 adapter, STATE, response encoder, GMM, normalizer/imputer, C head, early stopping, checkpoint selection, or representation selection.
- Outer-test observed responses may be opened only after the selected checkpoint is frozen, for generation-quality evaluation and the frozen observed-B diagnostic.
- `DepMap GeneEffect` remains a population-level relative growth-rate/dependency label, not a cell-death, mechanism, or single-cell fate label.
- The authoritative exp05 path must not import, fit, load, wait for, hash, or write fold-local scVI artifacts.
- The authoritative exp05 path must not fit, load, hash, or write a ridge projector artifact.
- The response encoder is shared by predicted B and observed B and contains `Linear(2000, 128)` followed by `LayerNorm(128)` for scale stability.
- The GMM is a trainable diagonal mixture over the 128-dimensional per-cell response encodings; means, variances, and mixture logits are learned end to end.
- STATE is trainable from epoch 1. Use a smaller STATE learning rate instead of a freeze/unfreeze transition that would invalidate DDP reducer membership.
- Training requires exactly four CUDA ranks with one GPU per rank. All four ranks train the same fold; fold-per-GPU execution is forbidden.
- Per-device gene batch size is `1`, giving an effective global gene batch size of `4` with four DDP ranks and no idle GPU.
- Rank-zero-only filesystem mutations must be entered by all ranks through one error-broadcast helper; a rank-zero Python exception must reach all ranks before any later collective.
- Negative normalized expression values are allowed. NaN and Inf are forbidden and must be replaced deterministically with feature-wise finite means computed only from non-targeting controls.
- Legacy scVI helpers used by non-authoritative ablations may remain in place, but the repaired exp05 config and audited fold runner must be structurally unable to call them.

---

## File Structure

- Create `src/aivc_model/expression.py`: finite-value statistics, deterministic imputation, and chunked cache validation.
- Create `src/aivc_model/response.py`: shared response encoder and trainable diagonal GMM pooler.
- Create `src/aivc_model/distributed.py`: exact-four-rank validation and symmetric rank-zero action/error broadcast.
- Create `tests/test_aivc_expression.py`: expression sanitization contract.
- Create `tests/test_aivc_response.py`: response encoder, trainable GMM, shared-branch, and gradient tests.
- Create `tests/ddp_exp05_smoke.py`: real four-process Accelerate smoke witness.
- Create `scripts/run_exp05_ddp.sh`: scheduler-neutral authoritative four-GPU launcher.
- Modify `src/aivc_model/gwps_cache.py`: schema-v2 cache with control-derived fill values and finite arrays.
- Modify `src/aivc_model/prepare.py`: response-encoder config, fill-value propagation, and external-expression sanitization.
- Modify `src/aivc_model/model.py`: replace the scVI projector/fixed-GMM model wiring with the shared response branch.
- Modify `src/aivc_model/train.py`: remove audited scVI/ridge/fixed-GMM fitting, add differential optimizer groups, gradient clipping, shared observed-B supervision, and frozen post-selection diagnostics.
- Modify `src/aivc_model/cross_validate.py`: finite-cache preflight, mandatory four-rank runtime guard, and shared distributed error handling.
- Modify `tests/test_aivc_model.py`: update model/config/artifact expectations without weakening legacy coverage.
- Modify `tests/test_aivc_cross_validate.py`: preserve and strengthen fold leakage guards.
- Modify `configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml`: authoritative e2e response-GMM/DDP config.
- Modify `scripts/state.sh`: keep Slurm allocation but delegate the actual launch to `run_exp05_ddp.sh`.
- Modify `docs/experiment/05_aivc_a_to_b_to_c.md`: repaired architecture, loss, DDP, and evaluation semantics.
- Modify `docs/experiment/model-card/05_aivc_a_to_b_to_c.md`: remove scVI/ridge claims from the implemented authoritative STATE path.

---

### Task 1: Add the finite-expression contract

**Files:**
- Create: `src/aivc_model/expression.py`
- Create: `tests/test_aivc_expression.py`

**Interfaces:**
- Produces: `compute_finite_feature_means(matrix: object, row_indices: np.ndarray, column_indices: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray`.
- Produces: `replace_nonfinite(values: np.ndarray, fill_values: np.ndarray) -> np.ndarray`.
- Produces: `assert_finite_npy(path: Path, *, chunk_size: int = 4096) -> None`.

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_aivc_expression.py
from pathlib import Path

import numpy as np
import pytest

from aivc_model.expression import (
    assert_finite_npy,
    compute_finite_feature_means,
    replace_nonfinite,
)


def test_control_means_ignore_nonfinite_and_preserve_negative_values() -> None:
    matrix = np.asarray(
        [[-2.0, np.nan, 1.0], [2.0, 4.0, np.inf], [8.0, 6.0, 3.0]],
        dtype=np.float32,
    )
    means = compute_finite_feature_means(
        matrix,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        chunk_size=1,
    )
    np.testing.assert_allclose(means, [0.0, 4.0, 1.0])


def test_replace_nonfinite_uses_feature_means_only() -> None:
    values = np.asarray([[-1.0, np.nan], [np.inf, 5.0]], dtype=np.float32)
    result = replace_nonfinite(values, np.asarray([2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(result, [[-1.0, 3.0], [2.0, 5.0]])


def test_assert_finite_npy_reports_the_first_bad_chunk(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.asarray([[1.0], [np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match=r"bad.npy.*rows 1:2"):
        assert_finite_npy(path, chunk_size=1)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
rtk proxy uv run python -m pytest tests/test_aivc_expression.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'aivc_model.expression'`.

- [ ] **Step 3: Implement the finite-expression helpers**

```python
# src/aivc_model/expression.py
"""Deterministic finite-value handling for normalized expression matrices."""

from pathlib import Path

import numpy as np


def compute_finite_feature_means(
    matrix: object,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Compute per-feature means from finite control entries only."""
    sums = np.zeros(len(column_indices), dtype=np.float64)
    counts = np.zeros(len(column_indices), dtype=np.int64)
    for start in range(0, len(row_indices), chunk_size):
        rows = row_indices[start : start + chunk_size]
        chunk = matrix[rows, :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        values = np.asarray(chunk, dtype=np.float32)[:, column_indices]
        finite = np.isfinite(values)
        sums += np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
        counts += finite.sum(axis=0, dtype=np.int64)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"control cells have no finite values for features {missing[:10].tolist()}")
    return (sums / counts).astype(np.float32)


def replace_nonfinite(values: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """Return a float32 copy with NaN/Inf replaced column-wise."""
    array = np.asarray(values, dtype=np.float32).copy()
    if array.ndim != 2 or fill_values.shape != (array.shape[1],):
        raise ValueError("expression and fill-value shapes are inconsistent")
    rows, columns = np.nonzero(~np.isfinite(array))
    array[rows, columns] = fill_values[columns]
    if not np.isfinite(array).all():
        raise ValueError("expression sanitization did not produce finite values")
    return array


def assert_finite_npy(path: Path, *, chunk_size: int = 4096) -> None:
    """Validate a numeric NPY without materializing the full array."""
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        if not np.isfinite(np.asarray(matrix[start:stop])).all():
            raise ValueError(f"{path.name} contains nonfinite values in rows {start}:{stop}")
```

- [ ] **Step 4: Run focused tests and lint**

Run:

```bash
rtk proxy uv run python -m pytest tests/test_aivc_expression.py -q
rtk proxy uv run ruff check src/aivc_model/expression.py tests/test_aivc_expression.py
```

Expected: `3 passed`; Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit the finite-expression boundary**

```bash
rtk git add src/aivc_model/expression.py tests/test_aivc_expression.py
rtk git commit -m "feat: add finite expression contract for exp05"
```

---

### Task 2: Rebuild the GWPS cache as finite schema v2

**Files:**
- Modify: `src/aivc_model/gwps_cache.py:20-215,240-305`
- Modify: `src/aivc_model/prepare.py:230-290,440-550`
- Modify: `src/aivc_model/cross_validate.py:161-182`
- Modify: `tests/test_aivc_model.py:200-390`
- Modify: `tests/test_aivc_cross_validate.py:220-370`

**Interfaces:**
- Consumes: Task 1 finite-expression helpers.
- Produces: cache array `feature_fill_values.npy` with shape `(2000,)`.
- Produces: `GeneBags.feature_fill_values: np.ndarray`.
- Produces: schema-v2 manifest whose arrays are all finite before training starts.

- [ ] **Step 1: Add failing cache and preflight tests**

```python
def test_gwps_cache_replaces_nonfinite_from_control_only(tmp_path: Path) -> None:
    config = _tiny_gwps_cache_config(tmp_path)
    cache_dir = tmp_path / "cache"
    gwps_cache_module._build_gwps_cache(
        config,
        cache_dir,
        gwps_cache_module._CacheContract(gene_count=2, state_dim=2),
    )
    cells = np.load(cache_dir / "cells.npy")
    controls = np.load(cache_dir / "control_cells.npy")
    fills = np.load(cache_dir / "feature_fill_values.npy")
    assert np.isfinite(cells).all()
    assert np.isfinite(controls).all()
    assert np.isfinite(fills).all()
    assert json.loads((cache_dir / "manifest.json").read_text())["schema_version"] == 2


def test_preflight_rejects_nonfinite_prepared_cache(tmp_path: Path) -> None:
    config, manifest = _preflight_cache_fixture(tmp_path)
    cells_path = config.data.prepared_cache_dir / "cells.npy"
    cells = np.load(cells_path)
    cells[0, 0] = np.nan
    np.save(cells_path, cells)
    with pytest.raises(ValueError, match="cells.npy contains nonfinite"):
        cv._validate_prepared_cache(config, manifest)
```

- [ ] **Step 2: Run the focused tests and verify failures**

Run:

```bash
rtk proxy uv run python -m pytest \
  tests/test_aivc_model.py -k 'gwps_cache_replaces_nonfinite' \
  tests/test_aivc_cross_validate.py -k 'preflight_rejects_nonfinite' -q
```

Expected: failures because the cache is schema v1 and does not write `feature_fill_values.npy` or inspect numeric arrays.

- [ ] **Step 3: Update the cache writer and loader**

Apply these exact structural changes:

```python
# src/aivc_model/gwps_cache.py
from aivc_model.expression import (
    assert_finite_npy,
    compute_finite_feature_means,
    replace_nonfinite,
)

_SCHEMA_VERSION = 2
_ARRAY_FILENAMES = (
    "cells.npy",
    "offsets.npy",
    "genes.npy",
    "gene_outer_folds.npy",
    "batch_labels.npy",
    "control_cells.npy",
    "control_batch.npy",
    "feature_names.npy",
    "feature_fill_values.npy",
)

# After control_rows is computed and before either matrix is written:
fill_values = compute_finite_feature_means(
    adata.X,
    control_rows,
    indices,
    chunk_size=_ROW_CHUNK_SIZE,
)
_write_matrix_rows(
    cache_dir / "cells.npy",
    adata.X,
    selected_rows,
    indices,
    fill_values,
)
_write_matrix_rows(
    cache_dir / "control_cells.npy",
    adata.X,
    control_rows,
    indices,
    fill_values,
)
_write_array(cache_dir / "feature_fill_values.npy", fill_values)

def _write_matrix_rows(
    path: Path,
    matrix: object,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    fill_values: np.ndarray,
) -> None:
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(row_indices), len(column_indices)),
    )
    for start in range(0, len(row_indices), _ROW_CHUNK_SIZE):
        stop = min(start + _ROW_CHUNK_SIZE, len(row_indices))
        rows = row_indices[start:stop]
        order = np.argsort(rows)
        chunk = matrix[rows[order], :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        values = np.asarray(chunk)[np.argsort(order)][:, column_indices]
        target[start:stop] = replace_nonfinite(values, fill_values)
    target.flush()
```

Load `feature_fill_values.npy`, validate shape `(2000,)`, and pass it into the new `GeneBags.feature_fill_values` field. Do not reject negative values.

- [ ] **Step 4: Sanitize external Adamson matrices with the same reference fill vector**

In `load_external_gene_bags`, apply:

```python
from aivc_model.expression import replace_nonfinite

bags = tuple(
    replace_nonfinite(np.asarray(bag, dtype=np.float32), reference.feature_fill_values)
    for bag in bags
)
control_input = replace_nonfinite(
    np.asarray(control_input, dtype=np.float32),
    reference.feature_fill_values,
)
```

Add `feature_fill_values` unchanged to every `GeneBags.for_genes` and sealed-view copy. Missing external features must continue to use the reference control mean; this step additionally replaces NaN/Inf after alignment.

- [ ] **Step 5: Make preflight scan both expression arrays**

```python
# src/aivc_model/cross_validate.py inside _validate_prepared_cache
from aivc_model.expression import assert_finite_npy

fills = np.load(cache_dir / "feature_fill_values.npy", allow_pickle=False)
if fills.shape != (STATE_FEATURE_COUNT,) or not np.isfinite(fills).all():
    raise ValueError("GWPS cache feature fill values must be 2000 finite values")
assert_finite_npy(cache_dir / "cells.npy")
assert_finite_npy(cache_dir / "control_cells.npy")
```

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
rtk proxy uv run python -m pytest tests/test_aivc_expression.py tests/test_aivc_cross_validate.py tests/test_aivc_model.py -q
rtk proxy uv run ruff check src/aivc_model/expression.py src/aivc_model/gwps_cache.py src/aivc_model/prepare.py src/aivc_model/cross_validate.py
```

Expected: all selected tests pass; Ruff reports no errors.

- [ ] **Step 7: Commit cache schema v2**

```bash
rtk git add src/aivc_model/gwps_cache.py src/aivc_model/prepare.py src/aivc_model/cross_validate.py tests/test_aivc_model.py tests/test_aivc_cross_validate.py
rtk git commit -m "fix: make exp05 expression cache finite"
```

---

### Task 3: Add the shared response encoder and trainable diagonal GMM

**Files:**
- Create: `src/aivc_model/response.py`
- Create: `tests/test_aivc_response.py`

**Interfaces:**
- Produces: `ResponseEncoder(input_dim: int, latent_dim: int) -> nn.Module`.
- Produces: `TrainableDiagonalGMM(latent_dim: int, n_components: int, covariance_floor: float, init_scale: float) -> nn.Module`.
- Produces: `TrainableDiagonalGMM.forward(bag: Tensor, control_bag: Tensor) -> Tensor` with output width `2*K + 2*D + 1`.
- Produces: `TrainableDiagonalGMM.occupancy(bag: Tensor) -> Tensor` and `negative_log_likelihood(bag: Tensor) -> Tensor`.

- [ ] **Step 1: Write failing response-module tests**

```python
# tests/test_aivc_response.py
import torch

from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM


def test_response_encoder_is_shared_linear_2000_to_128() -> None:
    encoder = ResponseEncoder(2000, 128)
    output = encoder(torch.randn(7, 2000))
    assert output.shape == (7, 128)
    assert encoder.linear.in_features == 2000
    assert encoder.linear.out_features == 128


def test_trainable_gmm_parameters_receive_finite_gradients() -> None:
    torch.manual_seed(42)
    gmm = TrainableDiagonalGMM(128, 8, 1e-4, 0.02)
    bag = torch.randn(16, 128, requires_grad=True)
    control = torch.randn(16, 128, requires_grad=True)
    loss = gmm(bag, control).square().mean() + gmm.negative_log_likelihood(bag)
    loss.backward()
    for parameter in (gmm.means, gmm.raw_variances, gmm.mixture_logits):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_control_occupancy_is_recomputed_after_gmm_update() -> None:
    torch.manual_seed(42)
    gmm = TrainableDiagonalGMM(4, 3, 1e-4, 0.02)
    bag = torch.randn(6, 4)
    control = torch.randn(6, 4)
    before = gmm(bag, control).detach().clone()
    with torch.no_grad():
        gmm.means.add_(0.5)
    after = gmm(bag, control).detach()
    assert not torch.equal(before, after)
```

- [ ] **Step 2: Verify the missing-module failure**

Run:

```bash
rtk proxy uv run python -m pytest tests/test_aivc_response.py -q
```

Expected: collection fails because `aivc_model.response` does not exist.

- [ ] **Step 3: Implement the response components**

```python
# src/aivc_model/response.py
"""End-to-end response encoding and differentiable mixture pooling."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class ResponseEncoder(nn.Module):
    """Shared per-cell normalized-expression encoder."""

    def __init__(self, input_dim: int = 2000, latent_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(expression))


class TrainableDiagonalGMM(nn.Module):
    """Trainable diagonal Gaussian responsibilities plus bag summaries."""

    def __init__(
        self,
        latent_dim: int,
        n_components: int,
        covariance_floor: float,
        init_scale: float,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)
        self.covariance_floor = float(covariance_floor)
        self.means = nn.Parameter(
            torch.randn(self.n_components, self.latent_dim) * float(init_scale)
        )
        self.raw_variances = nn.Parameter(
            torch.full((self.n_components, self.latent_dim), 0.54132485)
        )
        self.mixture_logits = nn.Parameter(torch.zeros(self.n_components))

    @property
    def variances(self) -> torch.Tensor:
        return F.softplus(self.raw_variances) + self.covariance_floor

    @property
    def output_dim(self) -> int:
        return 2 * self.n_components + 2 * self.latent_dim + 1

    def _component_log_prob(self, bag: torch.Tensor) -> torch.Tensor:
        variances = self.variances
        diff = bag.unsqueeze(1) - self.means.unsqueeze(0)
        gaussian = -0.5 * (
            diff.square() / variances.unsqueeze(0)
            + variances.log().unsqueeze(0)
            + math.log(2.0 * math.pi)
        ).sum(dim=2)
        return gaussian + self.mixture_logits.log_softmax(dim=0).unsqueeze(0)

    def occupancy(self, bag: torch.Tensor) -> torch.Tensor:
        return self._component_log_prob(bag).softmax(dim=1).mean(dim=0)

    def negative_log_likelihood(self, bag: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(self._component_log_prob(bag), dim=1).mean()

    def forward(self, bag: torch.Tensor, control_bag: torch.Tensor) -> torch.Tensor:
        occupancy = self.occupancy(bag)
        control_occupancy = self.occupancy(control_bag)
        mean = bag.mean(dim=0)
        variance = bag.var(dim=0, unbiased=False)
        entropy = -(occupancy * occupancy.clamp_min(1e-8).log()).sum().view(1)
        return torch.cat(
            [occupancy, occupancy - control_occupancy, mean, variance, entropy],
            dim=0,
        )
```

- [ ] **Step 4: Run tests and lint**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_response.py -q
rtk proxy uv run ruff check src/aivc_model/response.py tests/test_aivc_response.py
```

Expected: `3 passed`; Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit response components**

```bash
rtk git add src/aivc_model/response.py tests/test_aivc_response.py
rtk git commit -m "feat: add trainable response GMM modules"
```

---

### Task 4: Rewire AivcModel to shared predicted/observed response branches

**Files:**
- Modify: `src/aivc_model/model.py:22-34,233-320,348-720`
- Modify: `tests/test_aivc_response.py`
- Modify: `tests/test_aivc_model.py:2200-2350,2980-3080`

**Interfaces:**
- Consumes: Task 3 `ResponseEncoder` and `TrainableDiagonalGMM`.
- Produces: `AivcModel.predict_response(control_cells, gene, batch_indices) -> tuple[predicted_expression, predicted_latent]`.
- Produces: `AivcModel.predict_c_from_response(expression_bag, control_expression_bag) -> Tensor`.
- Produces: loss keys `hvg_mean_delta`, `latent_mean_delta`, `pred_c`, `obs_c`, `occupancy`, `gmm_nll`, `pred_rank`, and `total`.

- [ ] **Step 1: Add failing branch and gradient tests**

```python
def test_observed_c_supervision_updates_shared_response_stack_not_state() -> None:
    model = _tiny_e2e_aivc_model()
    losses = model.losses_for_gene(
        **_tiny_gene_inputs(),
        weights=LossWeights(
            latent_mean_delta=0.0,
            latent_energy=0.0,
            hvg_mean_delta=0.0,
            hvg_energy=0.0,
            pred_c=0.0,
            obs_c=1.0,
            occupancy=0.0,
            gmm_nll=0.0,
        ),
    )
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in model.state_adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.response_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.response_pooler.parameters())
    assert any(parameter.grad is not None for parameter in model.c_head.parameters())


def test_predicted_c_supervision_reaches_unfrozen_state() -> None:
    model = _tiny_e2e_aivc_model()
    losses = model.losses_for_gene(
        **_tiny_gene_inputs(),
        weights=_weights(pred_c=1.0),
    )
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.state_adapter.parameters())
```

- [ ] **Step 2: Run the focused tests and verify constructor/loss failures**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_response.py tests/test_aivc_model.py -k 'observed_c_supervision or predicted_c_supervision' -q
```

Expected: failures because `AivcModel` still requires a ridge-initialized projector and fixed GMM and `LossWeights` has no `gmm_nll` field.

- [ ] **Step 3: Replace the model's response stack**

Make these constructor fields authoritative:

```python
class AivcModel(nn.Module):
    def __init__(
        self,
        *,
        state_adapter: StateForwardAdapter,
        perturbations: PerturbationVectorAdapter | Esm2PerturbationAdapter,
        response_encoder: ResponseEncoder,
        response_pooler: TrainableDiagonalGMM,
        c_head: MLPHead,
        control_expression_mean: np.ndarray,
    ) -> None:
        super().__init__()
        self.state_adapter = state_adapter
        self.perturbations = perturbations
        self.response_encoder = response_encoder
        self.response_pooler = response_pooler
        self.c_head = c_head
        self.register_buffer(
            "control_expression_mean",
            torch.as_tensor(control_expression_mean, dtype=torch.float32),
        )

    def predict_response(
        self,
        control_cells: torch.Tensor,
        gene: str,
        batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        perturbation = self.perturbations(gene)
        predicted_expression = self.state_adapter(
            control_cells,
            perturbation,
            gene,
            batch_indices,
        )
        return predicted_expression, self.response_encoder(predicted_expression)

    def predict_c_from_response(
        self,
        expression_bag: torch.Tensor,
        control_expression_bag: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.response_encoder(expression_bag)
        control_latent = self.response_encoder(control_expression_bag)
        return self.c_head(self.response_pooler(latent, control_latent))
```

Delete `ExpressionToLatentProjector` and `FixedGMMFeatureizer` imports from the authoritative model wiring. Leave their legacy definitions only if a non-authoritative ablation still imports them.

- [ ] **Step 4: Implement observed-B/C co-supervision**

Inside `_forward_one_gene`, encode predicted, observed, and control bags with the same module:

```python
predicted_latent = self.response_encoder(predicted_expression)
observed_latent = self.response_encoder(target_expression)
control_latent = self.response_encoder(batched_control)

pred_y = self.c_head(self.response_pooler(predicted_latent, control_latent))
obs_y = self.c_head(self.response_pooler(observed_latent, control_latent))

hvg_mean_delta = _mean_delta_loss(
    predicted_expression,
    target_expression,
    self.control_expression_mean,
)
latent_mean_delta = F.mse_loss(
    predicted_latent.mean(dim=0),
    observed_latent.detach().mean(dim=0),
)
pred_c = F.mse_loss(pred_y.view(()), y.view(()))
obs_c = F.mse_loss(obs_y.view(()), y.view(()))
occupancy = F.mse_loss(
    self.response_pooler.occupancy(predicted_latent),
    self.response_pooler.occupancy(observed_latent).detach(),
)
gmm_nll = self.response_pooler.negative_log_likelihood(observed_latent)
```

Add `gmm_nll: float` to `LossWeights` and include `weights.gmm_nll * gmm_nll` in `total`. The detached observed targets apply only to B-alignment losses; `obs_c` must retain gradients through observed latent, GMM, and C head.

- [ ] **Step 5: Run model tests**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_response.py tests/test_aivc_model.py -q
rtk proxy uv run ruff check src/aivc_model/model.py src/aivc_model/response.py tests/test_aivc_response.py tests/test_aivc_model.py
```

Expected: all tests pass and every reported loss is finite.

- [ ] **Step 6: Commit shared branch integration**

```bash
rtk git add src/aivc_model/model.py tests/test_aivc_response.py tests/test_aivc_model.py
rtk git commit -m "feat: co-supervise exp05 from observed B and GeneEffect"
```

---

### Task 5: Remove scVI, ridge, and fixed-GMM fitting from audited exp05

**Files:**
- Modify: `src/aivc_model/prepare.py:159-220,424-445`
- Modify: `src/aivc_model/train.py:620-705,801-880,1081-1115,1690-1745,2073-2160`
- Modify: `src/aivc_model/cross_validate.py:310-460`
- Modify: `tests/test_aivc_model.py`
- Modify: `tests/test_aivc_cross_validate.py`

**Interfaces:**
- Produces: `ResponseEncoderConfig(input_dim=2000, latent_dim=128)` in `AivcConfig.response_encoder`.
- Produces: `_build_e2e_model(config, data, extra_genes, canonical_gene_order) -> AivcModel` with no fitted precursor artifacts.
- Preserves: legacy scVI utilities for non-authoritative experiments, but audited exp05 has no call edge to them.

- [ ] **Step 1: Add failing audited-path exclusion tests**

```python
def test_audited_exp05_never_calls_scvi_ridge_or_fixed_gmm(monkeypatch, tmp_path) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("removed precursor path was called")

    monkeypatch.setattr(train_module, "_fit_audited_scvi_latents", forbidden)
    monkeypatch.setattr(train_module, "_fit_or_load_projector_cache", forbidden)
    monkeypatch.setattr(train_module, "_fit_or_load_fixed_gmm_cache", forbidden)
    _run_tiny_audited_fold(tmp_path)


def test_audited_fit_summary_has_e2e_artifacts_only(tmp_path) -> None:
    summary = _run_tiny_audited_fold(tmp_path)
    payload = json.loads(summary["fit_audit_summary"].read_text())
    assert "response_encoder_sha256" in payload
    assert "gmm_sha256" in payload
    assert "state_sha256" in payload
    assert "scvi_sha256" not in payload
    assert "projector_sha256" not in payload
```

- [ ] **Step 2: Run focused tests and verify removed-path calls**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_cross_validate.py -k 'never_calls_scvi or e2e_artifacts_only' -q
```

Expected: failures because audited training still fits scVI, ridge, and fixed GMM.

- [ ] **Step 3: Add the response-encoder config without breaking legacy configs**

```python
@dataclass(frozen=True)
class ResponseEncoderConfig:
    input_dim: int = 2000
    latent_dim: int = 128


@dataclass(frozen=True)
class AivcConfig:
    response_encoder: ResponseEncoderConfig | None


def _response_encoder_config(values: Any) -> ResponseEncoderConfig | None:
    if values is None:
        return None
    return ResponseEncoderConfig(
        input_dim=int(values.get("input_dim", 2000)),
        latent_dim=int(values.get("latent_dim", 128)),
    )
```

Parse `response_encoder` alongside the existing legacy `projector` field. In audited exp05, require `response_encoder == ResponseEncoderConfig(2000, 128)` and ignore `projector`; other configs keep their previous behavior.

- [ ] **Step 4: Build the end-to-end modules directly**

Replace the audited `_fit_audited_scvi_latents`, `_fit_or_load_projector_cache`, and `_fit_or_load_fixed_gmm_cache` block with:

```python
model = _build_e2e_model(
    config,
    train_data,
    extra_genes=(*fold_spec.val_genes, *fold_spec.test_genes, *external_genes),
    canonical_gene_order=canonical_gene_order,
)
```

Implement `_build_e2e_model` by loading the STATE checkpoint and ESM2 adapter exactly as `_build_model` currently does, then constructing:

```python
response_encoder = ResponseEncoder(
    config.response_encoder.input_dim,
    config.response_encoder.latent_dim,
)
response_pooler = TrainableDiagonalGMM(
    latent_dim=config.response_encoder.latent_dim,
    n_components=config.gmm.n_components,
    covariance_floor=config.gmm.covariance_floor,
    init_scale=config.gmm.init_scale,
)
c_head = MLPHead(
    input_dim=response_pooler.output_dim,
    hidden_units=config.model.c_hidden_units,
    dropout=config.model.dropout,
)
```

Do not create any `scvi_teacher_latents`, `scvi_teacher_model`, `ridge_projector_fit`, or `fixed_gmm_fit` directory.

- [ ] **Step 5: Replace the separate observed-B oracle fit with frozen shared-head evaluation**

After best-checkpoint loading and `selected_model.requires_grad_(False)`, open the sealed test response only through `observed_b_oracle_outer_test` and call:

```python
observed_y = selected_model.predict_c_from_response(
    observed_expression_bag,
    control_expression_bag,
)
```

Remove `_fit_observed_b_oracle` from the audited path and record the scope as `observed_b_shared_oracle_outer_test`. This is an input oracle using the frozen shared model, not a separately selected model.

- [ ] **Step 6: Update audited artifact hashes**

Use:

```python
fit_summary = {
    "adapter_sha256": _module_sha256(model.perturbations),
    "state_sha256": _module_sha256(model.state_adapter),
    "response_encoder_sha256": _module_sha256(model.response_encoder),
    "gmm_sha256": _module_sha256(model.response_pooler),
    "c_head_sha256": _module_sha256(model.c_head),
    "best_epoch": int(best_epoch),
    "checkpoint_sha256": _path_sha256(checkpoint_path),
    "source_fingerprint": source_fingerprint,
    **authority.metadata(),
}
```

- [ ] **Step 7: Run audited-path tests**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_cross_validate.py tests/test_aivc_model.py -q
rtk proxy uv run ruff check src/aivc_model/prepare.py src/aivc_model/train.py src/aivc_model/cross_validate.py
```

Expected: tests pass; a repository search of the authoritative audited function shows no scVI/ridge/fixed-GMM call.

- [ ] **Step 8: Commit precursor removal**

```bash
rtk git add src/aivc_model/prepare.py src/aivc_model/train.py src/aivc_model/cross_validate.py tests/test_aivc_model.py tests/test_aivc_cross_validate.py
rtk git commit -m "refactor: remove exp05 scvi and ridge precursors"
```

---

### Task 6: Unfreeze STATE safely with differential learning rates

**Files:**
- Modify: `src/aivc_model/prepare.py:205-220,2090-2130`
- Modify: `src/aivc_model/train.py:680-730,2040-2170,2219-2285`
- Modify: `src/aivc_model/model.py:348-410`
- Modify: `tests/test_aivc_model.py:2220-2350`

**Interfaces:**
- Produces: `_optimizer_parameter_groups(model: AivcModel, config: AivcConfig) -> list[dict[str, object]]`.
- Produces: train settings `state_learning_rate`, `learning_rate`, `max_grad_norm`, and `required_world_size`.
- Guarantees: STATE parameters have `requires_grad=True` before `accelerator.prepare` and remain part of DDP for the entire run.

- [ ] **Step 1: Add failing optimizer and gradient tests**

```python
def test_optimizer_uses_lower_state_learning_rate() -> None:
    model = _tiny_e2e_aivc_model()
    config = _tiny_e2e_config(state_learning_rate=2.5e-6, learning_rate=2.5e-5)
    groups = train_module._optimizer_parameter_groups(model, config)
    assert [group["lr"] for group in groups] == [2.5e-6, 2.5e-5]


def test_state_is_trainable_before_ddp_prepare() -> None:
    model = _tiny_e2e_aivc_model()
    assert all(parameter.requires_grad for parameter in model.state_adapter.parameters())


def test_gradient_clipping_is_called_before_optimizer_step(monkeypatch) -> None:
    events = []
    accelerator = _RecordingAccelerator(events)
    optimizer = _RecordingOptimizer(events)
    _run_one_tiny_epoch(accelerator=accelerator, optimizer=optimizer)
    assert events.index("clip") < events.index("step")
```

- [ ] **Step 2: Verify tests fail under forced ESM2 freezing**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_model.py -k 'lower_state_learning_rate or trainable_before_ddp or gradient_clipping' -q
```

Expected: failures because ESM2 forces STATE frozen and the optimizer has one learning rate.

- [ ] **Step 3: Extend `TrainConfig`**

Add these exact fields to the existing `TrainConfig` dataclass; retain its other fields unchanged:

```python
learning_rate: float = 2.5e-5
state_learning_rate: float = 2.5e-6
max_grad_norm: float = 1.0
required_world_size: int = 4
```

Reject `state_learning_rate <= 0`, `state_learning_rate > learning_rate`, `max_grad_norm <= 0`, and `required_world_size != 4` while loading the authoritative exp05 config.

- [ ] **Step 4: Remove forced STATE freezing and build two optimizer groups**

```python
def _optimizer_parameter_groups(
    model: AivcModel,
    config: AivcConfig,
) -> list[dict[str, object]]:
    state_parameters = [
        parameter for parameter in model.state_adapter.parameters() if parameter.requires_grad
    ]
    state_ids = {id(parameter) for parameter in state_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in state_ids
    ]
    if not state_parameters or not other_parameters:
        raise ValueError("e2e exp05 requires trainable STATE and downstream parameters")
    return [
        {"params": state_parameters, "lr": config.train.state_learning_rate},
        {"params": other_parameters, "lr": config.train.learning_rate},
    ]
```

Construct AdamW from these groups. Delete `freeze_state=(tokenizer == "esm2" or ...)` and the audited assertion that STATE must be frozen.

- [ ] **Step 5: Add clipping immediately after backward**

```python
accelerator.backward(total_loss)
accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
optimizer.step()
```

Do not change `requires_grad` after `accelerator.prepare`; DDP reducer membership stays stable from epoch 1 through checkpoint selection.

- [ ] **Step 6: Run tests and commit**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_model.py tests/test_aivc_response.py -q
rtk proxy uv run ruff check src/aivc_model/model.py src/aivc_model/train.py src/aivc_model/prepare.py
rtk git add src/aivc_model/model.py src/aivc_model/train.py src/aivc_model/prepare.py tests/test_aivc_model.py
rtk git commit -m "feat: unfreeze STATE with differential learning rates"
```

Expected: tests and Ruff pass; STATE receives finite gradients under `pred_c` supervision.

---

### Task 7: Make four-rank DDP mandatory and exception-symmetric

**Files:**
- Create: `src/aivc_model/distributed.py`
- Create: `tests/ddp_exp05_smoke.py`
- Modify: `src/aivc_model/train.py:1472-1490,620-810`
- Modify: `src/aivc_model/cross_validate.py:365-460,484-542`
- Modify: `tests/test_aivc_cross_validate.py:1-120`

**Interfaces:**
- Produces: `require_exact_world_size(accelerator: Accelerator, expected: int = 4) -> None`.
- Produces: `run_rank_zero_or_raise(accelerator: Accelerator, label: str, action: Callable[[], object]) -> None`.
- Produces: `assert_all_ranks_stepped(accelerator: Accelerator, local_steps: int) -> tuple[int, ...]`.
- Guarantees: all ranks enter every rank-zero action broadcast in identical order.

- [ ] **Step 1: Add failing distributed helper tests**

```python
def test_exp05_requires_exactly_four_ranks() -> None:
    with pytest.raises(RuntimeError, match="requires exactly 4 DDP ranks"):
        require_exact_world_size(SimpleNamespace(num_processes=1), expected=4)


def test_rank_zero_exception_is_raised_on_every_rank(monkeypatch) -> None:
    accelerator = _FakeAccelerator(is_main_process=True, num_processes=4)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, src, device: None,
    )
    with pytest.raises(RuntimeError, match="checkpoint write failed.*disk full"):
        run_rank_zero_or_raise(
            accelerator,
            "checkpoint write",
            lambda: (_ for _ in ()).throw(OSError("disk full")),
        )


def test_zero_optimizer_steps_on_any_rank_is_rejected() -> None:
    accelerator = _FakeAccelerator(
        is_main_process=True,
        num_processes=4,
        gathered=torch.tensor([8, 8, 0, 8]),
    )
    with pytest.raises(RuntimeError, match="rank optimizer-step counts.*0"):
        assert_all_ranks_stepped(accelerator, local_steps=8)
```

- [ ] **Step 2: Verify missing helper failures**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_cross_validate.py -k 'exactly_four_ranks or rank_zero_exception or zero_optimizer_steps' -q
```

Expected: import or name failures for the new distributed helpers.

- [ ] **Step 3: Implement the symmetric distributed helpers**

```python
# src/aivc_model/distributed.py
"""Small DDP invariants shared by exp05 orchestration and training."""

from collections.abc import Callable
import traceback

from accelerate import Accelerator
import torch


def require_exact_world_size(accelerator: Accelerator, expected: int = 4) -> None:
    if accelerator.num_processes != expected:
        raise RuntimeError(
            f"authoritative exp05 requires exactly {expected} DDP ranks; "
            f"got {accelerator.num_processes}"
        )


def run_rank_zero_or_raise(
    accelerator: Accelerator,
    label: str,
    action: Callable[[], object],
) -> None:
    error_text: str | None = None
    if accelerator.is_main_process:
        try:
            action()
        except Exception:
            error_text = traceback.format_exc()
    values = [error_text]
    if accelerator.num_processes > 1:
        device = accelerator.device if "nccl" in str(torch.distributed.get_backend()).lower() else torch.device("cpu")
        torch.distributed.broadcast_object_list(values, src=0, device=device)
    if values[0] is not None:
        raise RuntimeError(f"{label} failed on rank zero:\n{values[0]}")


def assert_all_ranks_stepped(
    accelerator: Accelerator,
    local_steps: int,
) -> tuple[int, ...]:
    local = torch.tensor([local_steps], device=accelerator.device, dtype=torch.int64)
    counts = tuple(int(value) for value in accelerator.gather(local).cpu().tolist())
    if len(counts) != accelerator.num_processes or min(counts) <= 0:
        raise RuntimeError(f"rank optimizer-step counts must all be positive: {counts}")
    return counts
```

- [ ] **Step 4: Apply world-size and symmetric-action guards**

Keep `--preflight-only` as a CPU/single-process path that completes and exits before Accelerator construction. For every training invocation, call `require_exact_world_size(accelerator, config.train.required_world_size)` immediately after Accelerator construction in `run_cross_validation`. Replace rank-zero-only run-directory creation, train-log write, best-checkpoint write, fold-output write, and final aggregation followed by a naked barrier with `run_rank_zero_or_raise` calls that every rank enters.

Use `DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)` because the loaded STATE checkpoint may contain a fixed set of auxiliary parameters not exercised by direct `predict_step`; `static_graph=True` avoids rediscovering that set on every step. Keep model, optimizer, train loader, and val loader inside one `accelerator.prepare(...)` call so Accelerate shards the same fold's gene loader across all four ranks. Increment `local_optimizer_steps` after every `optimizer.step()`, then call `assert_all_ranks_stepped(accelerator, local_optimizer_steps)` at epoch end and write the returned four counts to the fold train log as `rank_optimizer_steps`.

- [ ] **Step 5: Add a real four-process DDP smoke witness**

```python
# tests/ddp_exp05_smoke.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator


def main() -> None:
    accelerator = Accelerator(mixed_precision="bf16")
    if accelerator.num_processes != 4:
        raise RuntimeError(f"DDP smoke requires 4 ranks, got {accelerator.num_processes}")
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16) / 1024.0
    targets = features.mean(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(features, targets), batch_size=1, shuffle=False)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    local_steps = torch.zeros(1, device=accelerator.device)
    for inputs, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - labels).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        local_steps += 1
    gathered_steps = accelerator.gather(local_steps)
    flat = torch.cat([parameter.detach().reshape(-1) for parameter in accelerator.unwrap_model(model).parameters()])
    checksum = flat.sum().view(1).to(accelerator.device)
    gathered_checksums = accelerator.gather(checksum)
    if accelerator.is_main_process:
        assert gathered_steps.shape == (4,)
        assert (gathered_steps > 0).all()
        assert torch.allclose(gathered_checksums, gathered_checksums[0].expand_as(gathered_checksums))
        print("DDP_SMOKE_OK world_size=4 all_ranks_active=1 parameters_synced=1")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run unit tests; reserve the real smoke for the four-H20 host**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_cross_validate.py -k 'exactly_four_ranks or rank_zero_exception or zero_optimizer_steps' -q
rtk proxy uv run ruff check src/aivc_model/distributed.py tests/ddp_exp05_smoke.py
```

Expected locally: unit tests pass and Ruff passes.

- [ ] **Step 7: Commit DDP hardening**

```bash
rtk git add src/aivc_model/distributed.py src/aivc_model/train.py src/aivc_model/cross_validate.py tests/test_aivc_cross_validate.py tests/ddp_exp05_smoke.py
rtk git commit -m "fix: enforce symmetric four-gpu exp05 ddp"
```

---

### Task 8: Preserve the strict fold protocol with the shared observed-B branch

**Files:**
- Modify: `src/aivc_model/train.py:700-900,1120-1400,2280-2420`
- Modify: `src/aivc_model/gene_splits.py:20-120`
- Modify: `tests/test_aivc_cross_validate.py:669-1020,1120-1310`

**Interfaces:**
- Produces: access stages `response_encoder_fit`, `gmm_fit`, `state_fit`, `c_head_fit`, `early_stopping_prediction_only`, and `observed_b_shared_oracle_outer_test`.
- Guarantees: inner validation and outer-test main evaluation call only the control-plus-gene prediction path.

- [ ] **Step 1: Add failing leakage and selection tests**

```python
def test_inner_validation_never_reads_observed_response(monkeypatch, tmp_path) -> None:
    val = _sealed_validation_bags_that_raise_on_response_access()
    _run_prediction_only_selection(val, tmp_path)
    assert val.response_access_count == 0


def test_outer_test_prediction_is_invariant_to_observed_response(tmp_path) -> None:
    first = _run_frozen_outer_test(tmp_path / "first", response_transform=lambda x: x)
    second = _run_frozen_outer_test(
        tmp_path / "second",
        response_transform=lambda x: np.full_like(x, 999.0),
    )
    pd.testing.assert_frame_equal(
        first.query("evaluation_scope == 'internal_outer_test'").reset_index(drop=True),
        second.query("evaluation_scope == 'internal_outer_test'").reset_index(drop=True),
    )


def test_observed_b_shared_oracle_opens_only_after_checkpoint_freeze() -> None:
    sealed = _tiny_sealed_outer_test()
    with pytest.raises(PermissionError):
        sealed.open("observed_b_shared_oracle_outer_test", checkpoint_frozen=False)
    opened = sealed.open("observed_b_shared_oracle_outer_test", checkpoint_frozen=True)
    assert len(opened.genes) > 0
```

- [ ] **Step 2: Run tests and verify the new stage is rejected**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_cross_validate.py -k 'inner_validation_never or invariant_to_observed or shared_oracle_opens' -q
```

Expected: failures because the new access stage and shared oracle path do not exist.

- [ ] **Step 3: Update access policy and train/validation routes**

Authorize observed response access only for inner-train stages:

```python
for stage in (
    "adapter_fit",
    "state_fit",
    "response_encoder_fit",
    "gmm_fit",
    "c_head_fit",
    "transition_supervision",
):
    _authorize_data_access(train_data, stage)
```

Keep early stopping on `_evaluate_prediction_only_final`, which consumes control cells, perturbation identity, and validation C labels but not validation observed B. Add `observed_b_shared_oracle_outer_test` to the sealed post-freeze allowlist and remove the old separately fitted oracle stages from the authoritative artifact metadata.

- [ ] **Step 4: Record new fitted-artifact authorities**

Write metadata for:

```python
payloads = {
    "esm_adapter_fit": {"kind": "esm_adapter"},
    "state_fit": {"kind": "state"},
    "response_encoder_fit": {"kind": "response_encoder"},
    "gmm_fit": {"kind": "trainable_gmm"},
    "c_head_fit": {"kind": "c_head"},
}
```

Each payload must include `ArtifactAuthority.metadata()` with exact train, validation, and test gene lists/hashes. No payload may include an outer-test gene in `fit_genes`.

- [ ] **Step 5: Run strict protocol tests**

```bash
rtk proxy uv run python -m pytest tests/test_aivc_cross_validate.py -q
rtk proxy uv run ruff check src/aivc_model/train.py src/aivc_model/gene_splits.py tests/test_aivc_cross_validate.py
```

Expected: all strict fold, sealed-response, and one-row-per-scope assertions pass.

- [ ] **Step 6: Commit protocol preservation**

```bash
rtk git add src/aivc_model/train.py src/aivc_model/gene_splits.py tests/test_aivc_cross_validate.py
rtk git commit -m "test: lock exp05 shared-response leakage guards"
```

---

### Task 9: Lock the authoritative config and four-GPU launcher

**Files:**
- Create: `scripts/run_exp05_ddp.sh`
- Modify: `scripts/state.sh:20-42`
- Modify: `configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml`
- Create: `tests/test_exp05_ddp_launcher.py`

**Interfaces:**
- Produces: `scripts/run_exp05_ddp.sh [config_path]` that launches exactly four Accelerate processes.
- Produces: authoritative effective global gene batch size `4` (`1` gene per rank).

- [ ] **Step 1: Add failing launcher/config tests**

```python
from pathlib import Path
import yaml


def test_exp05_config_is_e2e_response_gmm_ddp() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml").read_text()
    )
    assert "projector" not in config
    assert config["response_encoder"] == {"input_dim": 2000, "latent_dim": 128}
    assert config["gmm"]["trainable"] is True
    assert config["data"]["prepared_cache_dir"].endswith(
        "k562_gwps_state2000_v2"
    )
    assert config["train"]["required_world_size"] == 4
    assert config["train"]["gene_batch_size"] == 1


def test_launcher_uses_exactly_four_processes() -> None:
    script = Path("scripts/run_exp05_ddp.sh").read_text()
    assert "--num_processes 4" in script
    assert "--mixed_precision bf16" in script
    assert "aivc_model.cross_validate" in script
```

- [ ] **Step 2: Run tests and verify expected failures**

```bash
rtk proxy uv run python -m pytest tests/test_exp05_ddp_launcher.py -q
```

Expected: the launcher file is missing and the config still contains `projector.teacher: scvi`.

- [ ] **Step 3: Replace the authoritative YAML sections**

```yaml
data:
  prepared_cache_dir: data/exp05_cache/k562_gwps_state2000_v2

response_encoder:
  input_dim: 2000
  latent_dim: 128

gmm:
  n_components: 64
  covariance_floor: 0.0001
  init_scale: 0.02
  trainable: true

loss:
  latent_mean_delta_weight: 0.1
  latent_energy_weight: 0.0
  hvg_mean_delta_weight: 0.01
  hvg_energy_weight: 0.0
  pred_c_weight: 2.0
  obs_c_weight: 0.25
  occupancy_weight: 0.1
  gmm_nll_weight: 0.01
  pred_rank_weight: 5.0
  pred_rank_tau: 0.25
  pred_rank_pair_margin: 0.25
  pred_rank_pair_weight_clip: 2.0
  b_loss_anneal_epochs: 5
  b_loss_anneal_final_fraction: 0.1

train:
  run_id: state_esm2_response_gmm_ddp_outer5
  seed: 42
  max_epochs: 20
  learning_rate: 0.000025
  state_learning_rate: 0.0000025
  weight_decay: 0.0001
  max_grad_norm: 1.0
  cell_set_len: 256
  gene_batch_size: 1
  required_world_size: 4
  device: auto
  float32_matmul_precision: high
```

Remove `data.scvi_obsm_key`, the complete `projector` section, and `train.freeze_state` from this YAML only.

- [ ] **Step 4: Add the scheduler-neutral launcher**

```bash
#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$REPO_ROOT/.venv/bin/accelerate}"

test -x "$PYTHON_BIN"
test -x "$ACCELERATE_BIN"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$ACCELERATE_BIN" launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m aivc_model.cross_validate \
  --config "$CONFIG_PATH"
```

Modify `scripts/state.sh` so its final command is `srun scripts/run_exp05_ddp.sh "$CONFIG_PATH"`. On the direct H20 host, invoke the same launcher without `srun`, setting `PYTHON_BIN=.venv-esm2/bin/python` and `ACCELERATE_BIN=.venv-esm2/bin/accelerate`.

- [ ] **Step 5: Run launcher tests and shell syntax checks**

```bash
rtk proxy uv run python -m pytest tests/test_exp05_ddp_launcher.py -q
rtk proxy bash -n scripts/run_exp05_ddp.sh
rtk proxy bash -n scripts/state.sh
rtk proxy uv run ruff check tests/test_exp05_ddp_launcher.py
```

Expected: tests pass and both shell syntax checks exit `0`.

- [ ] **Step 6: Commit config and launcher**

```bash
rtk git add scripts/run_exp05_ddp.sh scripts/state.sh configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml tests/test_exp05_ddp_launcher.py
rtk git commit -m "feat: lock exp05 to four-gpu response-gmm ddp"
```

---

### Task 10: Update experiment documentation and run the complete verification gate

**Files:**
- Modify: `docs/experiment/05_aivc_a_to_b_to_c.md`
- Modify: `docs/experiment/model-card/05_aivc_a_to_b_to_c.md`

**Interfaces:**
- Documents: exact model graph, loss graph, four-rank DDP semantics, fold access policy, and GeneEffect claim boundary.
- Verifies: all implementation, protocol, config, and launcher tasks as one authoritative path.

- [ ] **Step 1: Replace the implemented architecture description**

Use this exact model statement in both documents:

```text
For each inner-train perturbation gene, an ESM-2 adapter produces the STATE
perturbation token. The trainable STATE checkpoint predicts a 2,000-feature
post-perturbation response bag from non-targeting control cells. Predicted and
observed response cells pass through the same Linear(2000, 128) response encoder,
trainable diagonal-GMM pooler, and GeneEffect head. Observed B supplies auxiliary
response and GeneEffect supervision for inner-train genes only. Validation and
primary outer-test GeneEffect predictions use control cells plus perturbation
identity only.
```

- [ ] **Step 2: Document mandatory DDP semantics**

```text
Each outer fold is one four-GPU DDP training job. Rank 0 through rank 3 process
disjoint gene batches from the same fold and synchronize gradients every optimizer
step. The five outer folds run sequentially; GPUs are not assigned independent
fold-local models. Per-device gene batch size is one, so the global gene batch size
is four.
```

State explicitly that scVI, ridge projector artifacts, and fixed-GMM fit caches are absent from the authoritative run.

- [ ] **Step 3: Run the full local verification gate**

```bash
rtk proxy uv run python -m pytest
rtk proxy uv run ruff check .
rtk proxy uv run ruff format --check .
rtk git diff --check
```

Expected: the full test suite passes, Ruff checks pass, formatting is clean, and `git diff --check` emits no output.

- [ ] **Step 4: Run static removal checks**

```bash
rtk proxy rg -n "_fit_audited_scvi_latents|_fit_or_load_projector_cache|_fit_or_load_fixed_gmm_cache" src/aivc_model/train.py
rtk proxy rg -n "scvi_teacher|ridge_projector|fixed_gmm_fit" configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml
```

Expected: the first search may find legacy helper definitions but no call inside `_run_audited_training`; the second search exits with no matches.

- [ ] **Step 5: Commit documentation**

```bash
rtk git add docs/experiment/05_aivc_a_to_b_to_c.md docs/experiment/model-card/05_aivc_a_to_b_to_c.md
rtk git commit -m "docs: describe exp05 e2e response-gmm protocol"
```

---

### Task 11: Validate on the four-H20 host before launching the full five-fold run

**Files:**
- Runtime artifacts only under `results/state/` and `results/experiments/05_aivc_a_to_b_to_c/`.

**Interfaces:**
- Consumes: committed implementation from Tasks 1-10.
- Produces: a four-rank DDP witness, rebuilt finite cache, successful strict preflight, and a live four-GPU training process.

- [ ] **Step 1: Rebuild the invalid schema-v1 cache**

On `/2023533015/VCC_Project`:

```bash
PYTHONPATH=src .venv-esm2/bin/python scripts/build_exp05_gwps_cache.py \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml \
  --cache-dir data/exp05_cache/k562_gwps_state2000_v2
```

Expected: `GWPS cache ready` and a schema-v2 manifest containing `feature_fill_values.npy`. The committed YAML already points to `k562_gwps_state2000_v2`; do not overwrite or reuse the invalid schema-v1 cache.

- [ ] **Step 2: Run strict single-process preflight without starting training**

```bash
PYTHONPATH=src .venv-esm2/bin/python -m aivc_model.cross_validate \
  --config configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml \
  --preflight-only
```

Expected lines include:

```text
gwps_depmap_overlap=9338
canonical_split_genes=9338
esm2_resolved=9338/9338
state_expression_matches=2000/2000
```

- [ ] **Step 3: Run the real four-GPU DDP witness**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv-esm2/bin/accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  tests/ddp_exp05_smoke.py
```

Expected exactly once from rank zero:

```text
DDP_SMOKE_OK world_size=4 all_ranks_active=1 parameters_synced=1
```

- [ ] **Step 4: Launch a one-epoch four-GPU exp05 smoke config**

Copy the authoritative YAML to `/tmp/exp05_smoke.yaml`, change only `train.run_id` to `state_esm2_response_gmm_ddp_smoke` and `train.max_epochs` to `1`, then run:

```bash
PYTHON_BIN=.venv-esm2/bin/python \
ACCELERATE_BIN=.venv-esm2/bin/accelerate \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/run_exp05_ddp.sh /tmp/exp05_smoke.yaml
```

Expected: all four GPUs allocate memory, every rank executes optimizer steps, fold 0 writes one selected checkpoint, and no `scvi_teacher`, `ridge_projector_fit`, or `fixed_gmm_fit` directory appears.

- [ ] **Step 5: Verify the smoke artifacts and access audit**

```bash
test -f results/experiments/05_aivc_a_to_b_to_c/runs/state_esm2_response_gmm_ddp_smoke/fold_0/train_log.csv
test -f results/experiments/05_aivc_a_to_b_to_c/runs/state_esm2_response_gmm_ddp_smoke/fold_0/artifacts/fit_access_audit.csv
! find results/experiments/05_aivc_a_to_b_to_c/runs/state_esm2_response_gmm_ddp_smoke -type d \( -name 'scvi*' -o -name 'ridge_projector_fit' -o -name 'fixed_gmm_fit' \) | grep .
```

Expected: both required files exist and the forbidden-artifact command exits successfully with no output.

- [ ] **Step 6: Launch the authoritative five-fold run only after the smoke passes**

```bash
nohup env \
  PYTHON_BIN=.venv-esm2/bin/python \
  ACCELERATE_BIN=.venv-esm2/bin/accelerate \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  scripts/run_exp05_ddp.sh \
  configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml \
  > results/state/exp05_response_gmm_ddp.log 2>&1 < /dev/null &
echo $! > results/state/exp05_response_gmm_ddp.pid
```

Verify the recorded parent PID has four worker processes and that GPU 0-3 each show nonzero memory before reporting the launch as successful.

---

## Self-Review Checklist

- [x] Spec coverage: Tasks 4-6 implement the requested adapter -> unfrozen STATE -> shared response encoder -> trainable GMM -> C head graph and observed-B/C co-supervision.
- [x] scVI/ridge removal: Task 5 removes both from the authoritative audited path and Task 10 adds structural searches.
- [x] Four-GPU DDP: Tasks 6, 7, 9, and 11 require exactly four ranks training the same fold, with one gene per rank and a real multi-process witness.
- [x] No wasted fold allocation: the plan explicitly forbids fold-per-GPU and keeps all four ranks in every optimizer step.
- [x] Failure visibility: Task 7 replaces naked rank-zero mutation/barrier sequences with symmetric error broadcasts.
- [x] Leakage protocol: Task 8 keeps observed B inner-train-only and outer-test post-freeze-only.
- [x] Numeric validity: Tasks 1-2 replace only NaN/Inf using non-targeting controls and retain valid negative normalized values.
- [x] Type consistency: `ResponseEncoder`, `TrainableDiagonalGMM`, `ResponseEncoderConfig`, `feature_fill_values`, and distributed helper signatures are identical across producer and consumer tasks.
- [x] Placeholder scan: the plan contains no deferred implementation markers; every code-changing step includes concrete code or exact structural edits.

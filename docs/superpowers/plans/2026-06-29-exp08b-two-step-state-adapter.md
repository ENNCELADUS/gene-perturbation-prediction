# exp08b Two-Step State-Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build exp08b as a leakage-safe, fold-local, two-step response-generator and SL-head pipeline that decouples bag reconstruction from SL ranking.

**Architecture:** Keep exp08b inside `src/sl_dl_model` but do not mutate the existing exp08 `StateDlProducer` into a new end-to-end trainer. Step 1 writes fold-local generator artifacts and cached `MeanStdPool(pred_bag)` embeddings to disk. Step 2 loads those cached embeddings as frozen inputs, trains only `SymmetricPairHead`, and reuses the official `sl_benchmark_baseline` ranking metrics.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, accelerate `PartialState`, pytest, ruff, uv, RTK.

---

## Assumptions

- Use `python -m sl_dl_model train-generator` and `python -m sl_dl_model train-sl-head` as the two new entrypoints.
- Keep all exp08b code in `src/sl_dl_model` to reuse existing package tests, queue primitives, `StateEncoder`, `MeanStdPool`, `SymmetricPairHead`, `bag_loss`, and official metric wrappers.
- Do not delete or rewrite `src/sl_dl_model/train.py`; it remains the historical exp08 end-to-end path.
- Use `state_backend="linear_mock"` in unit tests. Real STATE checkpoint behavior is covered by config and artifact contracts, not by local checkpoint-heavy tests.
- Step 1 artifacts live under `config.output_dir / "step1_generator" / "<split>_fold<fold>"`.
- Step 2 official metric artifacts live under `config.output_dir / "step2_sl_head"`.
- The direct-ESM2-MLP control is a Step 1 generator variant. It emits a cell bag in STATE-output space and is evaluated through the exact same Step 2 path.

## Scope Check

The spec covers one coherent subsystem: exp08b re-architects exp08 into two queue-separated passes. It includes generator training, generator monitor metrics, a direct-ESM2 control, SL-head training, CLI wiring, configs, and regression tests. It should stay one plan because every part is required to produce one testable experiment pipeline.

## File Structure

- Create `src/sl_dl_model/exp08b_config.py`: exp08b-specific config dataclass and YAML loader.
- Create `src/sl_dl_model/exp08b_artifacts.py`: fold artifact paths, cached embedding NPZ read/write, manifest read/write.
- Create `src/sl_dl_model/pert_vocab.py`: trusted STATE `pert_onehot_map.pt` loader.
- Modify `src/sl_dl_model/train.py`: import the shared loader as `_load_pert_vocab` for backward-compatible exp08 tests, so exp08b does not import the old end-to-end trainer.
- Create `src/sl_dl_model/exp08b_generator.py`: generator-validation split, fixed warmup scale, STATE-adapter generator, direct-MLP generator, Step 1 training, Step 1 monitor rows.
- Create `src/sl_dl_model/exp08b_sl_head.py`: Step 2 cached-embedding producer and pair-head-only trainer. This file must not import `StateEncoder`, `PertAdapter`, `SlDlModel`, or `sl_dl_model.train`.
- Create `src/sl_dl_model/exp08b_queue.py`: step-aware wrappers around the existing filesystem queue primitives.
- Create `src/sl_dl_model/exp08b_runner.py`: shared, label-free orchestration helpers (`jobs`, `raise_if_step_incomplete`) imported by both step runners; imports neither the generator/STATE nor the pair head.
- Create `src/sl_dl_model/exp08b_step1_runner.py`: the `train-generator` queue pass; imports the generator + STATE, never the pair head or `sl_label`.
- Create `src/sl_dl_model/exp08b_step2_runner.py`: the `train-sl-head` queue pass; imports the pair head + scoring on a STATE-neutralized metric config, never the generator/STATE.
- Modify `src/sl_dl_model/__main__.py`: add `train-generator` and `train-sl-head` subcommands with lazy imports.
- Modify `src/sl_dl_model/fold_queue.py`: include exp08b fields in the fingerprint, and add an optional `extra` mapping to `write_result` so Step 2 can persist its consumed `cache_fp` inside the single `.result.json` resume artifact (no separate sidecar).
- Create `tests/sl_dl_model/test_exp08b_config.py`: config and artifact contract tests.
- Create `tests/sl_dl_model/test_exp08b_generator.py`: generator split, scale normalization, Step 1 training, and direct-MLP tests.
- Create `tests/sl_dl_model/test_exp08b_monitor.py`: pooled metrics, bag energy, and ESM2-nearest-neighbor baseline tests.
- Create `tests/sl_dl_model/test_exp08b_sl_head.py`: cached embedding producer, pair-head-only training, import separation tests.
- Create `tests/sl_dl_model/test_exp08b_queue_cli.py`: step-aware queue and CLI smoke tests.
- Modify `tests/test_no_collectives.py`: no change expected; it should automatically scan the new `src/sl_dl_model/*.py` files.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml`: primary exp08b config.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/direct_mlp.yaml`: direct-ESM2-MLP control config.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/nn_copy.yaml`: §5.2 NN-copy step-2-rung config.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_bag_only.yaml`: `lambda_distill=0` ablation.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_distill_only.yaml`: `lambda_bag=0` ablation.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_ema_scale.yaml`: EMA scale ablation config, marked as secondary.
- Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/README.md`: exact run commands and artifact layout.

---

### Task 1: Config And Artifact Contract

**Files:**
- Create: `src/sl_dl_model/exp08b_config.py`
- Create: `src/sl_dl_model/exp08b_artifacts.py`
- Test: `tests/sl_dl_model/test_exp08b_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/sl_dl_model/test_exp08b_config.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    fold_artifact_dir,
    generator_manifest_path,
    generator_weights_path,
    load_embedding_cache,
    load_generator_manifest,
    save_embedding_cache,
    write_generator_manifest,
)
from sl_dl_model.exp08b_config import Exp08bConfig, load_exp08b_config


def test_exp08b_defaults_keep_distill_full_weight() -> None:
    cfg = Exp08bConfig()
    assert cfg.lambda_distill == 1.0
    assert cfg.lambda_distill_after_warmup == 1.0
    assert cfg.lambda_bag == 1.0
    assert cfg.lambda_sl == 0.0
    assert cfg.generator_kind == "state_adapter"
    assert cfg.generator_val_fraction == 0.2
    assert cfg.bag_scale_mode == "fixed_warmup"
    assert cfg.embedding_method == "exp08b_state_adapter_meanstd"


def test_load_exp08b_config_coerces_paths_and_tuples(tmp_path: Path) -> None:
    path = tmp_path / "exp08b.yaml"
    payload = {
        "input_csv": "data/pairs.csv",
        "output_dir": "results/exp08b/run",
        "split_types": ["CV2", "CV3"],
        "folds": [0, 4],
        "ranking_k": [10, 50],
        "esm2_npz": "data/esm2.npz",
        "bags_npz": "data/bags.npz",
        "generator_kind": "direct_mlp",
        "direct_mlp_hidden": 32,
    }
    path.write_text(yaml.safe_dump(payload))

    cfg = load_exp08b_config(path)

    assert cfg.input_csv == Path("data/pairs.csv")
    assert cfg.output_dir == Path("results/exp08b/run")
    assert cfg.split_types == ("CV2", "CV3")
    assert cfg.folds == (0, 4)
    assert cfg.ranking_k == (10, 50)
    assert cfg.esm2_npz == Path("data/esm2.npz")
    assert cfg.bags_npz == Path("data/bags.npz")
    assert cfg.generator_kind == "direct_mlp"
    assert cfg.direct_mlp_hidden == 32


def test_load_exp08b_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"unknown_field": 1}))

    try:
        load_exp08b_config(path)
    except ValueError as exc:
        assert "unknown config keys" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown_field")


def test_artifact_paths_are_fold_local(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")

    fold_dir = fold_artifact_dir(cfg, "CV2", 3)

    assert fold_dir == tmp_path / "run" / "step1_generator" / "CV2_fold3"
    assert embedding_cache_path(cfg, "CV2", 3).name == "predicted_embeddings.npz"
    assert generator_manifest_path(cfg, "CV2", 3).name == "generator_manifest.json"
    assert generator_weights_path(cfg, "CV2", 3).name == "generator_weights.pt"


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")
    symbols = np.array(["A", "B"], dtype=object)
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    coverage = np.array([1, 0], dtype=np.int64)
    path = embedding_cache_path(cfg, "CV2", 0)

    save_embedding_cache(
        path,
        symbols=symbols,
        embeddings=embeddings,
        coverage_mask=coverage,
        embedding_method="exp08b_state_adapter_meanstd",
    )
    loaded = load_embedding_cache(path)

    assert loaded["embedding_method"] == "exp08b_state_adapter_meanstd"
    assert loaded["symbols"].tolist() == ["A", "B"]
    np.testing.assert_allclose(loaded["embeddings"], embeddings)
    np.testing.assert_array_equal(loaded["coverage_mask"], coverage)


def test_generator_manifest_roundtrip(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")
    path = generator_manifest_path(cfg, "CV3", 4)
    payload = {
        "split_type": "CV3",
        "fold_id": 4,
        "bag_scale": 3.5,
        "train_bag_gene_count": 8,
        "val_bag_gene_count": 2,
    }

    write_generator_manifest(path, payload)
    loaded = load_generator_manifest(path)

    assert json.dumps(loaded, sort_keys=True) == json.dumps(payload, sort_keys=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.exp08b_config'`.

- [ ] **Step 3: Implement config and artifact modules**

Create `src/sl_dl_model/exp08b_config.py`:

```python
"""Configuration for exp08b two-step STATE-adapter experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from sl_dl_model.config import SLDLConfig


@dataclass(frozen=True)
class Exp08bConfig(SLDLConfig):
    """Exp08b config with Step 1 and Step 2 specific fields."""

    output_dir: Path = Path(
        "results/experiments/08b_k562_sl_pair_two_step_state_adapter/run"
    )
    embedding_method: str = "exp08b_state_adapter_meanstd"

    # Step 1 generator.
    generator_kind: str = "state_adapter"
    generator_val_fraction: float = 0.2
    generator_val_seed: int = 17
    direct_mlp_hidden: int = 512
    bag_scale_mode: str = "fixed_warmup"
    bag_scale_min: float = 1e-3
    bag_scale_ema_decay: float = 0.95

    # Exp08b keeps the distill anchor at full weight for the full Step 1 run.
    lambda_sl: float = 0.0
    lambda_distill: float = 1.0
    lambda_distill_after_warmup: float = 1.0
    lambda_bag: float = 1.0
    warmup_epochs: int = 1

    # Step-scoped artifacts.
    step1_artifacts_subdir: str = "step1_generator"
    step2_results_subdir: str = "step2_sl_head"
    generator_embedding_filename: str = "predicted_embeddings.npz"
    generator_manifest_filename: str = "generator_manifest.json"
    generator_weights_filename: str = "generator_weights.pt"
    generator_monitor_filename: str = "generator_monitor.csv"


# generator_kind → side-by-side ladder label (spec §5.2). Step 2 metric rows
# must carry distinct model names so exp08b, the direct-ESM2-MLP control, and
# the NN-copy rung land as separate rows in the official-metric summary.
_GENERATOR_KIND_TO_MODEL_NAME = {
    "state_adapter": "exp08b",
    "direct_mlp": "direct_esm2_mlp",
    "nn_copy": "nn_copy",
}


def metric_model_name_for(generator_kind: str) -> str:
    """Map a ``generator_kind`` to its §5.2 ladder model label.

    Raises:
        ValueError: If ``generator_kind`` is not a known exp08b rung.
    """
    try:
        return _GENERATOR_KIND_TO_MODEL_NAME[generator_kind]
    except KeyError as exc:
        raise ValueError(f"unknown generator_kind: {generator_kind!r}") from exc


@dataclass(frozen=True)
class SlHeadConfig:
    """Slim Step-2 config: pair-head, scoring, and optimization fields only.

    Spec §7.1 forbids Step 2 from holding a STATE checkpoint. This dataclass
    carries exactly the fields ``CachedEmbeddingPairHeadProducer`` reads — no
    ``state_checkpoint``, ``esm2_npz``, ``gwps_h5ad``, or generator field — so
    the Step-2 producer module never even has a checkpoint path to leak.
    """

    pair_hidden: tuple[int, ...] = (256, 64)
    include_coverage_flag: bool = True
    lr: float = 1e-3
    max_epochs: int = 20
    batch_pairs: int = 1024
    max_grad_norm: float = 1.0

    @classmethod
    def from_exp08b(cls, config: "Exp08bConfig") -> "SlHeadConfig":
        """Project the relevant Step-2 fields out of a full exp08b config."""
        return cls(
            pair_hidden=tuple(config.pair_hidden),
            include_coverage_flag=bool(config.include_coverage_flag),
            lr=float(config.lr),
            max_epochs=int(config.max_epochs),
            batch_pairs=int(config.batch_pairs),
            max_grad_norm=float(config.max_grad_norm),
        )


_PATH_FIELDS = {
    "input_csv",
    "output_dir",
    "esm2_npz",
    "state_checkpoint",
    "gwps_h5ad",
    "gwps_overlap_csv",
    "bags_npz",
}
_TUPLE_FIELDS = {"split_types", "folds", "ranking_k", "pair_hidden"}


def load_exp08b_config(path: Path) -> Exp08bConfig:
    """Load an :class:`Exp08bConfig` from YAML."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(Exp08bConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key in _PATH_FIELDS and value is not None:
            kwargs[key] = Path(value)
        elif key in _TUPLE_FIELDS and value is not None:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return Exp08bConfig(**kwargs)
```

Create `src/sl_dl_model/exp08b_artifacts.py`:

```python
"""Artifact paths and IO helpers for exp08b."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sl_dl_model.exp08b_config import Exp08bConfig


def fold_artifact_dir(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the Step 1 artifact directory for one fold."""
    return (
        Path(config.output_dir)
        / config.step1_artifacts_subdir
        / f"{split_type}_fold{fold_id}"
    )


def embedding_cache_path(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the cached e_hat NPZ path for one fold."""
    return fold_artifact_dir(config, split_type, fold_id) / config.generator_embedding_filename


def generator_manifest_path(
    config: Exp08bConfig, split_type: str, fold_id: int
) -> Path:
    """Return the Step 1 generator manifest path for one fold."""
    return fold_artifact_dir(config, split_type, fold_id) / config.generator_manifest_filename


def generator_weights_path(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the frozen generator weights path for one fold."""
    return fold_artifact_dir(config, split_type, fold_id) / config.generator_weights_filename


def generator_monitor_path(config: Exp08bConfig, split_type: str, fold_id: int) -> Path:
    """Return the Step 1 monitor CSV path for one fold."""
    return fold_artifact_dir(config, split_type, fold_id) / config.generator_monitor_filename


def step2_output_dir(config: Exp08bConfig) -> Path:
    """Return the Step 2 official-metric output directory."""
    return Path(config.output_dir) / config.step2_results_subdir


def save_embedding_cache(
    path: Path,
    *,
    symbols: np.ndarray,
    embeddings: np.ndarray,
    coverage_mask: np.ndarray,
    embedding_method: str,
) -> None:
    """Write a fold-local cached embedding table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            symbols=np.asarray(symbols, dtype=object),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            coverage_mask=np.asarray(coverage_mask, dtype=np.int64),
            embedding_method=np.asarray(embedding_method, dtype=object),
        )
    os.replace(tmp, path)


def load_embedding_cache(path: Path) -> dict[str, Any]:
    """Load a fold-local cached embedding table."""
    with np.load(path, allow_pickle=True) as data:
        method = data["embedding_method"]
        return {
            "symbols": data["symbols"].astype(object),
            "embeddings": data["embeddings"].astype(np.float32),
            "coverage_mask": data["coverage_mask"].astype(np.int64),
            "embedding_method": str(method.item() if method.shape == () else method[0]),
        }


def write_generator_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a generator manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def load_generator_manifest(path: Path) -> dict[str, Any]:
    """Read a generator manifest."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return payload
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/sl_dl_model/exp08b_config.py src/sl_dl_model/exp08b_artifacts.py tests/sl_dl_model/test_exp08b_config.py
rtk git commit -m "feat: add exp08b config and artifact contract"
```

Expected: commit succeeds.

---

### Task 2: Generator Split And Bag-Scale Helpers

**Files:**
- Create: `src/sl_dl_model/pert_vocab.py`
- Create: `src/sl_dl_model/exp08b_generator.py`
- Modify: `src/sl_dl_model/train.py`
- Test: `tests/sl_dl_model/test_exp08b_generator.py`

- [ ] **Step 1: Write failing tests for leakage-safe split and fixed warmup scale**

Create `tests/sl_dl_model/test_exp08b_generator.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_generator import (
    EmaBagScale,
    FixedWarmupBagScale,
    build_bag_scale,
    select_generator_bag_sets,
)
from sl_dl_model.pert_vocab import load_pert_vocab


def test_generator_validation_split_comes_only_from_train_covered() -> None:
    train_symbols = {"A", "B", "C", "D", "E", "TEST_ONLY"}
    covered_symbols = {"A", "B", "C", "D", "E", "OUTSIDE_TRAIN"}

    train_bag, val_bag = select_generator_bag_sets(
        train_symbols=train_symbols,
        covered_symbols=covered_symbols,
        val_fraction=0.4,
        seed=17,
    )

    assert train_bag | val_bag == {"A", "B", "C", "D", "E"}
    assert train_bag.isdisjoint(val_bag)
    assert "TEST_ONLY" not in train_bag | val_bag
    assert "OUTSIDE_TRAIN" not in train_bag | val_bag
    assert len(val_bag) == 1


def test_generator_validation_split_is_deterministic() -> None:
    kwargs = {
        "train_symbols": {"A", "B", "C", "D", "E", "F"},
        "covered_symbols": {"A", "B", "C", "D", "E", "F"},
        "val_fraction": 0.2,
        "seed": 23,
    }

    first = select_generator_bag_sets(**kwargs)
    second = select_generator_bag_sets(**kwargs)

    assert first == second


def test_fixed_warmup_bag_scale_uses_median_and_clamp() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)
    for value in (torch.tensor(10.0), torch.tensor(2.0), torch.tensor(6.0)):
        scale.observe(value)

    chosen = scale.finalize()

    assert chosen == 6.0
    assert scale.value == 6.0
    assert torch.isclose(scale.normalize(torch.tensor(12.0)), torch.tensor(2.0))


def test_fixed_warmup_bag_scale_clamps_small_values() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)
    scale.observe(torch.tensor(0.0))
    scale.observe(torch.tensor(1e-8))

    chosen = scale.finalize()

    assert chosen == 1e-3
    assert torch.isclose(scale.normalize(torch.tensor(1e-3)), torch.tensor(1.0))


def test_fixed_warmup_bag_scale_requires_observations() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)

    try:
        scale.finalize()
    except ValueError as exc:
        assert "no bag losses observed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ema_bag_scale_updates_and_normalizes() -> None:
    scale = EmaBagScale(min_scale=1e-3, decay=0.5)

    scale.observe(torch.tensor(10.0))
    assert scale.value == 10.0
    scale.observe(torch.tensor(2.0))

    assert scale.value == 6.0
    assert torch.isclose(scale.normalize(torch.tensor(12.0)), torch.tensor(2.0))


def test_build_bag_scale_selects_fixed_or_ema() -> None:
    fixed = build_bag_scale(Exp08bConfig(bag_scale_mode="fixed_warmup"))
    ema = build_bag_scale(Exp08bConfig(bag_scale_mode="ema", bag_scale_ema_decay=0.9))

    assert isinstance(fixed, FixedWarmupBagScale)
    assert isinstance(ema, EmaBagScale)


def test_build_bag_scale_rejects_unknown_mode() -> None:
    try:
        build_bag_scale(Exp08bConfig(bag_scale_mode="mystery"))
    except ValueError as exc:
        assert "unknown bag_scale_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_pert_vocab_reads_checkpoint_sibling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    torch.save(
        {"A": np.eye(3, dtype=np.float32)[0]},
        checkpoint.parent.parent / "pert_onehot_map.pt",
    )

    loaded = load_pert_vocab(checkpoint)

    assert loaded is not None
    assert set(loaded) == {"A"}
    np.testing.assert_allclose(loaded["A"], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_load_pert_vocab_returns_none_when_sidecar_missing(tmp_path: Path) -> None:
    """Missing pert_onehot_map.pt must return None, NOT {}.

    The exp08 `_ensure_pert_vocab` raise-on-missing contract (train.py) keys off
    a `None` return to distinguish "absent" from "present but empty". Returning
    `{}` here would silently disable the distill anchor — the OOD-token fix the
    spec mandates at full weight (§3.2) — and would also break the existing
    `tests/sl_dl_model/test_train.py::test_distill_required_but_missing_vocab_raises`.
    """
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert load_pert_vocab(checkpoint) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `sl_dl_model.exp08b_generator` or `sl_dl_model.pert_vocab`.

- [ ] **Step 3: Implement pert-vocab, split, and scale helpers**

Create `src/sl_dl_model/pert_vocab.py`:

```python
"""Trusted STATE perturbation-vocab loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_pert_vocab(checkpoint: Path) -> dict[str, np.ndarray] | None:
    """Load ``pert_onehot_map.pt`` from ``checkpoint.parent.parent``.

    Returns ``None`` (not ``{}``) when the sibling file is absent. This
    ``None``-on-missing contract is load-bearing: it is exactly the contract
    the existing exp08 ``StateDlProducer._ensure_pert_vocab`` relies on to
    *raise* when distill is configured but the vocab is missing
    (`tests/sl_dl_model/test_train.py::test_distill_required_but_missing_vocab_raises`).
    Returning ``{}`` here would silently turn a required-distill run into a
    bag-only run. Callers that genuinely want "no distill on this backend"
    (e.g. ``linear_mock``) must substitute ``{}`` themselves.

    The file is a trusted project artifact produced with the STATE checkpoint,
    so ``weights_only=False`` is intentional for compatibility with the existing
    serialized NumPy objects.
    """
    vocab_path = Path(checkpoint).parent.parent / "pert_onehot_map.pt"
    if not vocab_path.exists():
        return None
    raw: dict[str, object] = torch.load(
        vocab_path,
        map_location="cpu",
        weights_only=False,
    )
    return {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in raw.items()}
```

Modify `src/sl_dl_model/train.py` to reuse the shared helper:

```python
from sl_dl_model.pert_vocab import load_pert_vocab as _load_pert_vocab
```

Delete the local `_load_pert_vocab(checkpoint: Path)` function from `src/sl_dl_model/train.py`. Existing tests that monkeypatch `sl_dl_model.train._load_pert_vocab` remain valid because the imported alias has the same name, and the exp08 raise-on-missing behavior is preserved because the shared loader keeps the `None`-on-missing return contract.

Create `src/sl_dl_model/exp08b_generator.py` with:

```python
"""Step 1 generator training utilities for exp08b."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from torch import nn, optim

from sl_dl_model.bags import GwpsBags
from sl_dl_model.encoder import StateEncoder, state_encoded_token, state_original_token
from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    generator_manifest_path,
    generator_weights_path,
    save_embedding_cache,
    write_generator_manifest,
)
from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.losses import bag_loss, distill_loss
from sl_dl_model.pert_vocab import load_pert_vocab
from sl_dl_model.pooling import MeanStdPool


def select_generator_bag_sets(
    *,
    train_symbols: set[str],
    covered_symbols: set[str],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split fold-train covered genes into bag-supervision and monitor sets."""
    eligible = sorted({s.upper() for s in train_symbols} & {s.upper() for s in covered_symbols})
    if not eligible:
        return set(), set()
    n_val = int(math.floor(len(eligible) * float(val_fraction)))
    if len(eligible) >= 2 and n_val < 1 and val_fraction > 0:
        n_val = 1
    n_val = min(n_val, max(0, len(eligible) - 1))

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(eligible, dtype=object)
    rng.shuffle(shuffled)
    val = {str(s) for s in shuffled[:n_val]}
    train = set(eligible) - val
    return train, val


class FixedWarmupBagScale:
    """Median bag-loss scale chosen from detached warmup observations."""

    def __init__(self, *, min_scale: float) -> None:
        self.min_scale = float(min_scale)
        self._observed: list[float] = []
        self.value: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether a fixed scale has been selected."""
        return self.value is not None

    def observe(self, loss: torch.Tensor) -> None:
        """Record one detached bag-loss value."""
        self._observed.append(float(loss.detach().cpu()))

    def finalize(self) -> float:
        """Select median observed scale, clamped to ``min_scale``."""
        if not self._observed:
            raise ValueError("no bag losses observed during warmup")
        finite = [x for x in self._observed if np.isfinite(x)]
        if not finite:
            raise ValueError("no finite bag losses observed during warmup")
        self.value = max(float(np.median(np.asarray(finite, dtype=float))), self.min_scale)
        return self.value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize a bag loss by the selected fixed scale."""
        if self.value is None:
            raise RuntimeError("bag scale has not been finalized")
        return loss / float(self.value)


class EmaBagScale:
    """EMA-normalized bag-loss scale for the normalization ablation."""

    def __init__(self, *, min_scale: float, decay: float) -> None:
        self.min_scale = float(min_scale)
        self.decay = float(decay)
        self.value: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether at least one finite scale has been observed."""
        return self.value is not None

    def observe(self, loss: torch.Tensor) -> None:
        """Update the EMA scale from one detached bag-loss value."""
        current = max(float(loss.detach().cpu()), self.min_scale)
        if not np.isfinite(current):
            return
        if self.value is None:
            self.value = current
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * current
        self.value = max(float(self.value), self.min_scale)

    def finalize(self) -> float:
        """Return the current EMA scale."""
        if self.value is None:
            raise ValueError("no finite bag losses observed for EMA scale")
        return self.value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize a bag loss by the current EMA scale."""
        if self.value is None:
            raise RuntimeError("bag scale has not been initialized")
        return loss / float(self.value)


def build_bag_scale(config: Exp08bConfig) -> FixedWarmupBagScale | EmaBagScale:
    """Build the configured bag-loss normalizer."""
    if config.bag_scale_mode == "fixed_warmup":
        return FixedWarmupBagScale(min_scale=config.bag_scale_min)
    if config.bag_scale_mode == "ema":
        return EmaBagScale(
            min_scale=config.bag_scale_min,
            decay=config.bag_scale_ema_decay,
        )
    raise ValueError(f"unknown bag_scale_mode: {config.bag_scale_mode!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py tests/sl_dl_model/test_train.py::test_distill_loss_wired_with_monkeypatch -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/sl_dl_model/pert_vocab.py src/sl_dl_model/exp08b_generator.py src/sl_dl_model/train.py tests/sl_dl_model/test_exp08b_generator.py
rtk git commit -m "feat: add exp08b generator split and scale helpers"
```

Expected: commit succeeds.

---

### Task 3: Step 1 STATE-Adapter Generator Training

**Files:**
- Modify: `src/sl_dl_model/exp08b_generator.py`
- Modify: `tests/sl_dl_model/test_exp08b_generator.py`

- [ ] **Step 1: Add failing Step 1 training tests**

Append to `tests/sl_dl_model/test_exp08b_generator.py`:

```python
from sl_dl_model.bags import GwpsBags
from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    generator_manifest_path,
    generator_weights_path,
    load_embedding_cache,
    load_generator_manifest,
)
from sl_dl_model.exp08b_generator import Step1GeneratorTrainer
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable


def _tiny_esm_and_bags() -> tuple[Esm2EmbeddingTable, GwpsBags, np.ndarray]:
    rng = np.random.default_rng(7)
    symbols = np.array(["A", "B", "C"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=4,
        vectors_by_symbol={
            "A": rng.standard_normal(4).astype(np.float32),
            "B": rng.standard_normal(4).astype(np.float32),
            "C": rng.standard_normal(4).astype(np.float32),
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((5, 3)).astype(np.float32),
        bags_by_symbol={
            "A": rng.standard_normal((5, 3)).astype(np.float32),
            "B": rng.standard_normal((5, 3)).astype(np.float32),
        },
        input_dim=3,
    )
    return esm, bags, symbols


def test_step1_trainer_writes_fold_local_cache_and_manifest(tmp_path: Path) -> None:
    esm, bags, symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="linear_mock",
        pert_dim=3,
        adapter_hidden=8,
        max_epochs=1,
        warmup_epochs=1,
        lambda_bag=1.0,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    result = trainer.train_fold(
        split_type="CV2",
        fold_id=0,
        symbols=symbols,
        train_symbols={"A", "B", "C"},
    )

    assert result.embedding_path == embedding_cache_path(cfg, "CV2", 0)
    assert result.manifest_path == generator_manifest_path(cfg, "CV2", 0)
    assert result.weights_path == generator_weights_path(cfg, "CV2", 0)
    assert result.embedding_path.exists()
    assert result.manifest_path.exists()
    assert result.weights_path.exists()

    cache = load_embedding_cache(result.embedding_path)
    assert cache["symbols"].tolist() == ["A", "B", "C"]
    assert cache["embeddings"].shape == (3, 6)
    assert cache["coverage_mask"].tolist() == [1, 1, 0]

    manifest = load_generator_manifest(result.manifest_path)
    assert manifest["split_type"] == "CV2"
    assert manifest["fold_id"] == 0
    assert manifest["generator_kind"] == "state_adapter"
    assert manifest["train_bag_gene_count"] == 1
    assert manifest["val_bag_gene_count"] == 1
    assert manifest["bag_scale"] >= 1e-3
    assert manifest["generator_weights_path"] == str(result.weights_path)


def test_step1_trainer_uses_partialstate_device_not_cuda_default() -> None:
    source = Path("src/sl_dl_model/exp08b_generator.py").read_text()

    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in source
    assert "PartialState().device" in source or "device=" in source


def test_distill_symbols_include_fold_train_vocab_independent_of_bag_split(tmp_path: Path) -> None:
    esm, bags, _symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(output_dir=tmp_path / "run", state_backend="linear_mock")
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )
    trainer._pert_vocab = {
        "A": np.eye(3, dtype=np.float32)[0],
        "B": np.eye(3, dtype=np.float32)[1],
        "UNCOVERED": np.eye(3, dtype=np.float32)[2],
    }

    distill_symbols = trainer.distill_symbols_for_fold(
        {"A", "B", "UNCOVERED", "NOT_IN_VOCAB"}
    )

    assert distill_symbols == {"A", "B", "UNCOVERED"}


def test_distill_required_but_missing_vocab_raises(tmp_path: Path) -> None:
    """Real backend + positive distill + missing pert_onehot_map.pt must fail loudly.

    The spec keeps the distill anchor at full weight as the OOD-token fix
    (§3.2); a real-backend run that requests distill but cannot load the STATE
    vocab must raise rather than silently degrade to bag-only.
    """
    esm, bags, _symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="state_checkpoint",
        state_checkpoint=tmp_path / "state" / "checkpoints" / "final.ckpt",
        lambda_distill=1.0,
        lambda_distill_after_warmup=1.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    try:
        trainer.distill_symbols_for_fold({"A", "B"})
    except RuntimeError as exc:
        assert "distill" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for missing required distill vocab")


def test_distill_not_required_when_weight_zero_does_not_raise(tmp_path: Path) -> None:
    """lambda_distill == 0 with a missing vocab is fine (distill not requested)."""
    esm, bags, _symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="state_checkpoint",
        state_checkpoint=tmp_path / "state" / "checkpoints" / "final.ckpt",
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    assert trainer.distill_symbols_for_fold({"A", "B"}) == set()


def test_step1_generator_source_does_not_read_sl_labels() -> None:
    source = Path("src/sl_dl_model/exp08b_generator.py").read_text()

    assert "sl_label" not in source
    assert "SymmetricPairHead" not in source


def test_step1_distill_only_does_not_crash_at_warmup_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    """lambda_bag == 0 must not raise when the warmup window observed no bags.

    Regression for the distill-only ablation: with lambda_bag == 0 the bag
    block never calls ``scale.observe``, so an unconditional ``finalize()`` at
    the warmup boundary would raise ``no bag losses observed``. The boundary
    finalize must be skipped, and the post-loop guard must default the unused
    scale to 1.0. ``_distill_term`` is monkeypatched to a param-free scalar so
    the linear_mock backend (empty pert-vocab, no STATE pert_encoder) still
    yields a trainable distill term.
    """
    esm, bags, symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="linear_mock",
        max_epochs=2,
        warmup_epochs=1,
        lambda_bag=0.0,
        lambda_distill=1.0,
        lambda_distill_after_warmup=1.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )
    trainer._pert_vocab = {
        "A": np.eye(3, dtype=np.float32)[0],
        "B": np.eye(3, dtype=np.float32)[1],
        "C": np.eye(3, dtype=np.float32)[2],
    }
    monkeypatch.setattr(
        trainer,
        "_distill_term",
        lambda generator, symbol, device: torch.tensor(1.0, requires_grad=True),
    )

    result = trainer.train_fold(
        split_type="CV2",
        fold_id=0,
        symbols=symbols,
        train_symbols={"A", "B", "C"},
    )

    manifest = load_generator_manifest(result.manifest_path)
    assert manifest["bag_scale"] == 1.0
    assert manifest["distill_gene_count"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py -q
```

Expected: FAIL with `ImportError: cannot import name 'Step1GeneratorTrainer'`.

- [ ] **Step 3: Add Step 1 trainer implementation**

Append this implementation to `src/sl_dl_model/exp08b_generator.py`:

```python
@dataclass(frozen=True)
class Step1TrainResult:
    """Paths and scalar summary for one trained Step 1 fold."""

    embedding_path: Path
    manifest_path: Path
    weights_path: Path
    bag_scale: float
    train_bag_gene_count: int
    val_bag_gene_count: int


class StateAdapterBagGenerator(nn.Module):
    """STATE-adapter generator that predicts a response bag for one gene."""

    def __init__(
        self,
        *,
        config: Exp08bConfig,
        esm_dim: int,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        checkpoint = None if config.state_backend == "linear_mock" else config.state_checkpoint
        self.encoder = StateEncoder(
            backend=config.state_backend,
            checkpoint=checkpoint,
            esm_dim=esm_dim,
            adapter_hidden=config.adapter_hidden,
            pert_dim=config.pert_dim,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.pool = MeanStdPool()

    def forward(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict a cell bag in STATE-output space."""
        return self.encoder(esm_vec, control)

    def pooled(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict and pool a cell bag."""
        return self.pool(self.forward(esm_vec, control))


class Step1GeneratorTrainer:
    """Train a fold-local response generator and cache all-gene embeddings."""

    def __init__(
        self,
        config: Exp08bConfig,
        *,
        esm: Esm2EmbeddingTable,
        bags: GwpsBags,
        input_dim: int,
        output_dim: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.device = device
        self.pool = MeanStdPool()
        self._pert_vocab: dict[str, np.ndarray] | None = None
        self._pert_vocab_loaded: bool = False

    def train_fold(
        self,
        *,
        split_type: str,
        fold_id: int,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> Step1TrainResult:
        """Train Step 1 on train-covered genes and write fold artifacts."""
        train_bag, val_bag = select_generator_bag_sets(
            train_symbols=train_symbols,
            covered_symbols=set(self.bags.bags_by_symbol),
            val_fraction=self.config.generator_val_fraction,
            seed=self.config.generator_val_seed + int(fold_id),
        )
        if self.config.generator_kind == "nn_copy":
            # §5.2 NN-copy step-2 rung: no trainable generator. Returns early
            # before any model/optimizer is built.
            return self._produce_nn_copy_fold(
                split_type=split_type,
                fold_id=fold_id,
                symbols=symbols,
                train_covered=train_bag | val_bag,
                val_bag=val_bag,
            )
        device = (
            torch.device(self.device)
            if self.device is not None
            else PartialState().device
        )
        generator = self._build_generator().to(device)
        optimizer = optim.Adam(
            (p for p in generator.parameters() if p.requires_grad),
            lr=self.config.lr,
        )
        control = torch.tensor(self.bags.control_template, device=device)
        scale = build_bag_scale(self.config)
        distill_symbols = self.distill_symbols_for_fold(train_symbols)

        for epoch in range(self.config.max_epochs):
            generator.train()
            epoch_losses: list[torch.Tensor] = []
            for symbol in sorted(train_bag | distill_symbols):
                esm_vec = self._esm_tensor(symbol, device)
                terms: list[torch.Tensor] = []

                if self.config.lambda_bag > 0 and symbol in train_bag:
                    pred = generator(esm_vec, control)
                    real = torch.tensor(self.bags.bags_by_symbol[symbol], device=device)
                    raw_bag = bag_loss(pred, real)
                    if isinstance(scale, FixedWarmupBagScale):
                        if epoch < self.config.warmup_epochs:
                            scale.observe(raw_bag)
                            bag_term = raw_bag
                        else:
                            if not scale.ready:
                                scale.finalize()
                            bag_term = scale.normalize(raw_bag)
                    else:
                        scale.observe(raw_bag)
                        bag_term = scale.normalize(raw_bag)
                    terms.append(float(self.config.lambda_bag) * bag_term)

                if self.config.lambda_distill > 0 and symbol in distill_symbols:
                    distill = self._distill_term(generator, symbol, device)
                    if distill is not None:
                        terms.append(float(self.config.lambda_distill) * distill)

                if not terms:
                    continue
                total = torch.stack(terms).sum()
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in generator.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                optimizer.step()
                epoch_losses.append(total.detach())

            # Only finalize the warmup scale if bag supervision actually ran
            # during warmup. With lambda_bag == 0 (distill-only ablation) or an
            # empty train_bag, no detached bag losses were observed, so
            # FixedWarmupBagScale.finalize() would raise; the post-loop guard
            # sets scale.value = 1.0 for those folds instead.
            if (
                isinstance(scale, FixedWarmupBagScale)
                and epoch + 1 == self.config.warmup_epochs
                and not scale.ready
                and self.config.lambda_bag > 0
                and train_bag
            ):
                scale.finalize()

            if not epoch_losses and (train_bag or distill_symbols):
                raise RuntimeError("Step 1 produced no trainable generator losses")

        if not scale.ready:
            if self.config.lambda_bag > 0 and train_bag:
                scale.finalize()
            else:
                scale.value = 1.0

        embeddings, coverage = self._embed_universe(generator, control, symbols, device)
        emb_path = embedding_cache_path(self.config, split_type, fold_id)
        weights_path = generator_weights_path(self.config, split_type, fold_id)
        save_embedding_cache(
            emb_path,
            symbols=symbols,
            embeddings=embeddings,
            coverage_mask=coverage,
            embedding_method=self.config.embedding_method,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_weights = weights_path.with_suffix(weights_path.suffix + f".tmp.{os.getpid()}")
        torch.save(
            {
                "generator_kind": self.config.generator_kind,
                "state_dict": generator.state_dict(),
            },
            tmp_weights,
        )
        os.replace(tmp_weights, weights_path)
        manifest_path = generator_manifest_path(self.config, split_type, fold_id)
        write_generator_manifest(
            manifest_path,
            {
                "split_type": split_type,
                "fold_id": int(fold_id),
                "generator_kind": self.config.generator_kind,
                "embedding_method": self.config.embedding_method,
                "bag_scale": float(scale.value),
                "bag_scale_mode": self.config.bag_scale_mode,
                "generator_weights_path": str(weights_path),
                "train_bag_gene_count": len(train_bag),
                "val_bag_gene_count": len(val_bag),
                "distill_gene_count": len(distill_symbols),
                "universe_gene_count": int(len(symbols)),
            },
        )
        return Step1TrainResult(
            embedding_path=emb_path,
            manifest_path=manifest_path,
            weights_path=weights_path,
            bag_scale=float(scale.value),
            train_bag_gene_count=len(train_bag),
            val_bag_gene_count=len(val_bag),
        )

    def _build_generator(self) -> nn.Module:
        if self.config.generator_kind != "state_adapter":
            raise ValueError(f"unknown generator_kind: {self.config.generator_kind!r}")
        return StateAdapterBagGenerator(
            config=self.config,
            esm_dim=self.esm.dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def _esm_tensor(self, symbol: str, device: torch.device) -> torch.Tensor:
        vec = self.esm.vectors_by_symbol.get(symbol.upper())
        if vec is None:
            if self.config.fallback_strategy == "global_mean" and self.esm.vectors_by_symbol:
                vec = np.mean(
                    np.vstack(list(self.esm.vectors_by_symbol.values())), axis=0
                ).astype(np.float32)
            else:
                vec = np.zeros(self.esm.dim, dtype=np.float32)
        return torch.tensor(np.asarray(vec, dtype=np.float32), device=device)

    def _load_distill_vocab(self) -> dict[str, np.ndarray]:
        """Load the STATE pert-vocab once, raising if required but missing.

        Mirrors the exp08 ``StateDlProducer._ensure_pert_vocab`` contract: on a
        real STATE backend with ``lambda_distill`` or
        ``lambda_distill_after_warmup`` > 0, a missing/unreadable
        ``pert_onehot_map.pt`` is a hard error — the spec keeps the distill
        anchor at full weight as the OOD-token fix (§3.2), so silently dropping
        it is forbidden. ``linear_mock`` and zero-distill runs return ``{}``.
        """
        # Honor a vocab injected by tests / callers before first load.
        if self._pert_vocab is not None:
            return self._pert_vocab
        if self._pert_vocab_loaded:
            return self._pert_vocab or {}
        self._pert_vocab_loaded = True

        if self.config.state_backend == "linear_mock":
            self._pert_vocab = {}
            return self._pert_vocab

        distill_requested = (
            self.config.lambda_distill > 0
            or self.config.lambda_distill_after_warmup > 0
        )
        try:
            self._pert_vocab = load_pert_vocab(self.config.state_checkpoint)
        except Exception as exc:
            if distill_requested:
                raise RuntimeError(
                    "distill loss is configured (lambda_distill>0) but the STATE "
                    "pert_onehot_map.pt could not be loaded; refusing to train "
                    "the exp08b generator silently without the distill anchor"
                ) from exc
            self._pert_vocab = {}
            return self._pert_vocab

        if self._pert_vocab is None:
            if distill_requested:
                raise RuntimeError(
                    "distill loss is configured (lambda_distill>0) but the STATE "
                    "pert_onehot_map.pt is missing next to the checkpoint; "
                    "refusing to train the exp08b generator silently without the "
                    "distill anchor (expected at "
                    "<checkpoint>.parent.parent/pert_onehot_map.pt)"
                )
            self._pert_vocab = {}
        return self._pert_vocab

    def distill_symbols_for_fold(self, train_symbols: set[str]) -> set[str]:
        """Return STATE pert-vocab symbols from the full fold-train set."""
        vocab = self._load_distill_vocab()
        train = {symbol.upper() for symbol in train_symbols}
        return train & set(vocab)

    def _distill_term(
        self,
        generator: nn.Module,
        symbol: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not isinstance(generator, StateAdapterBagGenerator):
            return None
        vocab = self._load_distill_vocab()
        if not vocab:
            return None
        onehot_arr = vocab.get(symbol.upper())
        if onehot_arr is None:
            return None
        state_model = generator.encoder.state.state_model
        if not hasattr(state_model, "pert_encoder"):
            return None
        esm_vec = self._esm_tensor(symbol, device)
        adapter_raw = generator.encoder.adapter(esm_vec.unsqueeze(0)).squeeze(0)
        adapter_tok = state_encoded_token(state_model, adapter_raw)
        onehot = torch.tensor(onehot_arr, device=device)
        target_tok = state_original_token(state_model, onehot)
        return distill_loss(adapter_tok.unsqueeze(0), target_tok.unsqueeze(0))

    def _embed_universe(
        self,
        generator: nn.Module,
        control: torch.Tensor,
        symbols: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        generator.eval()
        rows: list[np.ndarray] = []
        coverage: list[int] = []
        with torch.no_grad():
            for symbol_obj in symbols:
                symbol = str(symbol_obj).upper()
                e_hat = generator.pooled(self._esm_tensor(symbol, device), control)
                rows.append(e_hat.detach().cpu().numpy().astype(np.float32))
                coverage.append(1 if symbol in self.bags.bags_by_symbol else 0)
        return np.vstack(rows).astype(np.float32), np.asarray(coverage, dtype=np.int64)
```

- [ ] **Step 4: Run Step 1 tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/sl_dl_model/exp08b_generator.py tests/sl_dl_model/test_exp08b_generator.py
rtk git commit -m "feat: train exp08b step1 state adapter"
```

Expected: commit succeeds.

---

### Task 4: Step 1 Monitor Metrics And NN-Copy Baseline

**Files:**
- Create: `tests/sl_dl_model/test_exp08b_monitor.py`
- Modify: `src/sl_dl_model/exp08b_generator.py`

- [ ] **Step 1: Write failing monitor tests**

Create `tests/sl_dl_model/test_exp08b_monitor.py`:

```python
from __future__ import annotations

import numpy as np

from sl_dl_model.exp08b_generator import (
    bag_energy_metric,
    compute_monitor_rows,
    nearest_neighbor_copy_predictions,
    pooled_vector_metrics,
)


def test_pooled_vector_metrics_report_direction_and_magnitude() -> None:
    pred = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    real = np.array([1.0, 0.0, 2.0], dtype=np.float32)

    metrics = pooled_vector_metrics(pred, real)

    assert metrics["pooled_cosine"] > 0.999
    assert metrics["pooled_mse"] == 0.0
    assert metrics["pooled_l2"] == 0.0


def test_bag_energy_metric_is_zero_for_identical_bags() -> None:
    bag = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert bag_energy_metric(bag, bag) < 1e-6


def test_nearest_neighbor_copy_uses_train_covered_only() -> None:
    esm = {
        "TRAIN_A": np.array([1.0, 0.0], dtype=np.float32),
        "TRAIN_B": np.array([0.0, 1.0], dtype=np.float32),
        "VAL": np.array([0.9, 0.1], dtype=np.float32),
        "OUTSIDE": np.array([0.91, 0.09], dtype=np.float32),
    }
    bags = {
        "TRAIN_A": np.full((2, 2), 1.0, dtype=np.float32),
        "TRAIN_B": np.full((2, 2), 2.0, dtype=np.float32),
        "OUTSIDE": np.full((2, 2), 9.0, dtype=np.float32),
    }

    copied = nearest_neighbor_copy_predictions(
        val_symbols={"VAL"},
        train_covered_symbols={"TRAIN_A", "TRAIN_B"},
        esm_vectors=esm,
        real_bags=bags,
    )

    np.testing.assert_allclose(copied["VAL"], bags["TRAIN_A"])


def test_compute_monitor_rows_has_generator_and_nn_rows() -> None:
    pred_bags = {"VAL": np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)}
    real_bags = {"VAL": np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)}
    nn_bags = {"VAL": np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)}

    rows = compute_monitor_rows(
        epoch=2,
        split_type="CV2",
        fold_id=0,
        pred_bags=pred_bags,
        real_bags=real_bags,
        nn_copy_bags=nn_bags,
    )

    assert {row["predictor"] for row in rows} == {"generator", "esm2_nn_copy"}
    assert all(row["split_type"] == "CV2" for row in rows)
    assert all(row["fold_id"] == 0 for row in rows)
    assert all(row["epoch"] == 2 for row in rows)
    gen = [row for row in rows if row["predictor"] == "generator"][0]
    assert gen["pooled_cosine"] > 0.999
    assert gen["pooled_mse"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_monitor.py -q
```

Expected: FAIL with missing monitor functions.

- [ ] **Step 3: Add monitor metric functions**

Append to `src/sl_dl_model/exp08b_generator.py`:

```python
def _mean_std_pool_np(bag: np.ndarray) -> np.ndarray:
    arr = np.asarray(bag, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def pooled_vector_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """Return cosine, MSE, and L2 between pooled vectors."""
    pred_arr = np.asarray(pred, dtype=np.float32)
    real_arr = np.asarray(real, dtype=np.float32)
    denom = float(np.linalg.norm(pred_arr) * np.linalg.norm(real_arr))
    cosine = 0.0 if denom == 0.0 else float(np.dot(pred_arr, real_arr) / denom)
    diff = pred_arr - real_arr
    return {
        "pooled_cosine": cosine,
        "pooled_mse": float(np.mean(diff * diff)),
        "pooled_l2": float(np.linalg.norm(diff)),
    }


def bag_energy_metric(pred_bag: np.ndarray, real_bag: np.ndarray) -> float:
    """Return the safe energy-distance term between two cell bags."""
    pred = torch.tensor(np.asarray(pred_bag, dtype=np.float32))
    real = torch.tensor(np.asarray(real_bag, dtype=np.float32))
    from sl_dl_model.losses import _safe_energy_distance

    return float(_safe_energy_distance(pred, real).detach().cpu())


def nearest_neighbor_copy_predictions(
    *,
    val_symbols: set[str],
    train_covered_symbols: set[str],
    esm_vectors: dict[str, np.ndarray],
    real_bags: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Copy each val gene's nearest train-covered real bag in ESM2 space."""
    train = sorted({s.upper() for s in train_covered_symbols})
    copied: dict[str, np.ndarray] = {}
    for val_symbol in sorted({s.upper() for s in val_symbols}):
        val_vec = esm_vectors.get(val_symbol)
        if val_vec is None:
            continue
        best_symbol: str | None = None
        best_dist: float | None = None
        for train_symbol in train:
            train_vec = esm_vectors.get(train_symbol)
            train_bag = real_bags.get(train_symbol)
            if train_vec is None or train_bag is None:
                continue
            dist = float(np.linalg.norm(np.asarray(val_vec) - np.asarray(train_vec)))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_symbol = train_symbol
        if best_symbol is not None:
            copied[val_symbol] = np.asarray(real_bags[best_symbol], dtype=np.float32)
    return copied


def compute_monitor_rows(
    *,
    epoch: int,
    split_type: str,
    fold_id: int,
    pred_bags: dict[str, np.ndarray],
    real_bags: dict[str, np.ndarray],
    nn_copy_bags: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Compute Step 1 monitor rows for generator and NN-copy predictions."""
    rows: list[dict[str, object]] = []
    for symbol, real_bag in sorted(real_bags.items()):
        real_pool = _mean_std_pool_np(real_bag)
        for predictor, source in (
            ("generator", pred_bags),
            ("esm2_nn_copy", nn_copy_bags),
        ):
            pred_bag = source.get(symbol)
            if pred_bag is None:
                continue
            pred_pool = _mean_std_pool_np(pred_bag)
            pooled = pooled_vector_metrics(pred_pool, real_pool)
            rows.append(
                {
                    "split_type": split_type,
                    "fold_id": int(fold_id),
                    "epoch": int(epoch),
                    "gene_symbol": symbol,
                    "predictor": predictor,
                    **pooled,
                    "bag_energy": bag_energy_metric(pred_bag, real_bag),
                }
            )
    return rows
```

- [ ] **Step 4: Run monitor tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_monitor.py -q
```

Expected: PASS.

- [ ] **Step 5: Add monitor CSV flushing to Step 1 trainer**

Add imports near the top of `src/sl_dl_model/exp08b_generator.py`:

```python
import pandas as pd

from sl_dl_model.exp08b_artifacts import generator_monitor_path
```

Add these methods inside `Step1GeneratorTrainer`:

```python
    def _esm_vectors_for(self, symbols: set[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for symbol in symbols:
            vec = self.esm.vectors_by_symbol.get(symbol.upper())
            if vec is not None:
                out[symbol.upper()] = np.asarray(vec, dtype=np.float32)
        return out

    def _predict_bags(
        self,
        generator: nn.Module,
        control: torch.Tensor,
        symbols: set[str],
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        generator.eval()
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for symbol in sorted(symbols):
                esm_vec = self._esm_tensor(symbol, device)
                pred = generator(esm_vec, control)
                out[symbol.upper()] = pred.detach().cpu().numpy().astype(np.float32)
        generator.train()
        return out

    def _append_monitor_rows(
        self,
        *,
        split_type: str,
        fold_id: int,
        rows: list[dict[str, object]],
    ) -> None:
        if not rows:
            return
        path = generator_monitor_path(self.config, split_type, fold_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        write_header = not path.exists()
        frame.to_csv(path, mode="a", header=write_header, index=False)
```

Inside `train_fold`, after the epoch loop body and before `if epoch + 1 == self.config.warmup_epochs`, insert:

```python
            if val_bag:
                pred_bags = self._predict_bags(generator, control, val_bag, device)
                real_val_bags = {
                    symbol: self.bags.bags_by_symbol[symbol] for symbol in sorted(val_bag)
                }
                esm_vectors = self._esm_vectors_for(train_bag | val_bag)
                nn_bags = nearest_neighbor_copy_predictions(
                    val_symbols=val_bag,
                    train_covered_symbols=train_bag,
                    esm_vectors=esm_vectors,
                    real_bags=self.bags.bags_by_symbol,
                )
                rows = compute_monitor_rows(
                    epoch=epoch,
                    split_type=split_type,
                    fold_id=fold_id,
                    pred_bags=pred_bags,
                    real_bags=real_val_bags,
                    nn_copy_bags=nn_bags,
                )
                self._append_monitor_rows(
                    split_type=split_type,
                    fold_id=fold_id,
                    rows=rows,
                )
```

- [ ] **Step 6: Add Step 1 trainer monitor assertion**

Append to `test_step1_trainer_writes_fold_local_cache_and_manifest`:

```python
    monitor = result.manifest_path.parent / "generator_monitor.csv"
    assert monitor.exists()
    monitor_text = monitor.read_text()
    assert "pooled_cosine" in monitor_text
    assert "esm2_nn_copy" in monitor_text
```

- [ ] **Step 7: Run generator and monitor tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py tests/sl_dl_model/test_exp08b_monitor.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
rtk git add src/sl_dl_model/exp08b_generator.py tests/sl_dl_model/test_exp08b_generator.py tests/sl_dl_model/test_exp08b_monitor.py
rtk git commit -m "feat: add exp08b step1 monitor metrics"
```

Expected: commit succeeds.

---

### Task 5: Direct-ESM2-MLP And NN-Copy Generator Controls

Both §5.2 step-2 ladder rungs that are neither the floor (exp06) nor the ceiling
(exp07) live here: the **direct-ESM2-MLP** control (`MLP(ESM2)`, STATE bypassed)
and the **NN-copy** control (cache `ê_g` from the ESM2-nearest train-covered
gene's real pooled bag, no training). NN-copy appears twice in the spec: §4.3 as
a per-epoch step-1 *monitor* (already implemented in Task 4) and §5.2 as a
*step-2 ranking rung* reported side-by-side. Task 4 covered only the monitor;
this task adds the `nn_copy` generator kind so the §5.2 rung flows through the
identical Step 2 official-metric path as every other rung.

**Files:**
- Modify: `src/sl_dl_model/exp08b_generator.py`
- Modify: `tests/sl_dl_model/test_exp08b_generator.py`

- [ ] **Step 1: Write failing direct-MLP tests**

Append to `tests/sl_dl_model/test_exp08b_generator.py`:

```python
from sl_dl_model.exp08b_generator import DirectMlpBagGenerator


def test_direct_mlp_generator_broadcasts_delta_over_control_template() -> None:
    model = DirectMlpBagGenerator(esm_dim=4, hidden=8, output_dim=3)
    esm_vec = torch.randn(4)
    control = torch.randn(5, 3)

    pred = model(esm_vec, control)

    assert pred.shape == (5, 3)
    delta_rows = pred - control
    assert torch.allclose(delta_rows[0], delta_rows[1])


def test_step1_trainer_supports_direct_mlp_control(tmp_path: Path) -> None:
    esm, bags, symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        generator_kind="direct_mlp",
        state_backend="linear_mock",
        direct_mlp_hidden=8,
        max_epochs=1,
        warmup_epochs=1,
        lambda_bag=1.0,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    result = trainer.train_fold(
        split_type="CV2",
        fold_id=0,
        symbols=symbols,
        train_symbols={"A", "B", "C"},
    )

    cache = load_embedding_cache(result.embedding_path)
    assert cache["embeddings"].shape == (3, 6)
    manifest = load_generator_manifest(result.manifest_path)
    assert manifest["generator_kind"] == "direct_mlp"
```

- [ ] **Step 2: Run direct-MLP tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py::test_direct_mlp_generator_broadcasts_delta_over_control_template tests/sl_dl_model/test_exp08b_generator.py::test_step1_trainer_supports_direct_mlp_control -q
```

Expected: FAIL with `ImportError: cannot import name 'DirectMlpBagGenerator'`.

- [ ] **Step 3: Implement direct-MLP generator**

Append to `src/sl_dl_model/exp08b_generator.py` before `Step1GeneratorTrainer`:

```python
class DirectMlpBagGenerator(nn.Module):
    """Direct ESM2 control that predicts a broadcast delta over control cells."""

    def __init__(self, *, esm_dim: int, hidden: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
        self.pool = MeanStdPool()

    def forward(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Return ``control + delta(esm_vec)`` with delta broadcast over cells."""
        delta = self.net(esm_vec.unsqueeze(0)).squeeze(0)
        return control + delta.unsqueeze(0).expand_as(control)

    def pooled(self, esm_vec: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict and pool a cell bag."""
        return self.pool(self.forward(esm_vec, control))
```

Replace `_build_generator` in `Step1GeneratorTrainer` with:

```python
    def _build_generator(self) -> nn.Module:
        if self.config.generator_kind == "state_adapter":
            return StateAdapterBagGenerator(
                config=self.config,
                esm_dim=self.esm.dim,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
        if self.config.generator_kind == "direct_mlp":
            return DirectMlpBagGenerator(
                esm_dim=self.esm.dim,
                hidden=self.config.direct_mlp_hidden,
                output_dim=self.output_dim,
            )
        raise ValueError(f"unknown generator_kind: {self.config.generator_kind!r}")
```

- [ ] **Step 4: Run generator tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py -q
```

Expected: PASS.

- [ ] **Step 5: Write failing NN-copy step-2-rung tests**

The §5.2 baseline ladder lists **NN-copy** as a step-2 ranking rung reported
side-by-side with exp06/exp08b/exp07 — distinct from the §4.3 step-1 *monitor*
NN-copy baseline (Task 4). Implement it as a third `generator_kind` that caches
`ê_g = MeanStdPool(real_bag of ESM2-nearest train-covered gene)` for every
universe gene and flows through the identical Step 2 path. It trains no model.

Append to `tests/sl_dl_model/test_exp08b_generator.py`:

```python
def test_step1_trainer_nn_copy_caches_nearest_train_covered_pool(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    symbols = np.array(["A", "B", "TEST"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=2,
        vectors_by_symbol={
            "A": np.array([1.0, 0.0], dtype=np.float32),
            "B": np.array([0.0, 1.0], dtype=np.float32),
            "TEST": np.array([0.9, 0.1], dtype=np.float32),
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((4, 3)).astype(np.float32),
        bags_by_symbol={
            "A": np.full((4, 3), 1.0, dtype=np.float32),
            "B": np.full((4, 3), 2.0, dtype=np.float32),
        },
        input_dim=3,
    )
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        generator_kind="nn_copy",
        state_backend="linear_mock",
        embedding_method="exp08b_nn_copy_meanstd",
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    result = trainer.train_fold(
        split_type="CV2",
        fold_id=0,
        symbols=symbols,
        train_symbols={"A", "B", "TEST"},
    )

    cache = load_embedding_cache(result.embedding_path)
    assert cache["symbols"].tolist() == ["A", "B", "TEST"]
    assert cache["embeddings"].shape == (3, 6)
    # TEST is nearest to A in ESM2 space, so its pooled embedding copies A's bag.
    pool_a = np.concatenate(
        [np.full(3, 1.0, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    )
    np.testing.assert_allclose(cache["embeddings"][2], pool_a, atol=1e-5)
    assert cache["coverage_mask"].tolist() == [1, 1, 0]
    assert result.weights_path.exists()

    manifest = load_generator_manifest(result.manifest_path)
    assert manifest["generator_kind"] == "nn_copy"
    assert manifest["bag_scale"] == 1.0
```

- [ ] **Step 6: Run NN-copy tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py::test_step1_trainer_nn_copy_caches_nearest_train_covered_pool -q
```

Expected: FAIL with `ValueError: unknown generator_kind: 'nn_copy'` (or a
missing-method `AttributeError`).

- [ ] **Step 7: Implement the NN-copy fold producer**

`_build_generator` needs no change: the `nn_copy` branch added to `train_fold`
in Step 5 returns before `_build_generator` is ever called, so the `nn_copy`
kind never reaches the guard. Leave `_build_generator` exactly as Step 3 left
it (raising `ValueError` for unrecognized kinds is still correct, since only
`state_adapter` and `direct_mlp` ever build a module).

Append this method inside `Step1GeneratorTrainer`:

```python
    def _produce_nn_copy_fold(
        self,
        *,
        split_type: str,
        fold_id: int,
        symbols: np.ndarray,
        train_covered: set[str],
        val_bag: set[str],
    ) -> Step1TrainResult:
        """Cache ê_g = MeanStdPool(nearest train-covered real bag) per gene.

        The §5.2 NN-copy rung: no trainable model. For every universe gene,
        copy the real gwps bag of its ESM2-nearest fold-train-covered gene and
        pool it the same way exp08b pools its predicted bag. Writes the same
        embedding cache + manifest artifact contract; ``bag_scale`` is recorded
        as 1.0 (no normalization happens).
        """
        universe = [str(s).upper() for s in symbols]
        esm_vectors = self._esm_vectors_for(set(universe) | set(train_covered))
        copied = nearest_neighbor_copy_predictions(
            val_symbols=set(universe),
            train_covered_symbols=set(train_covered),
            esm_vectors=esm_vectors,
            real_bags=self.bags.bags_by_symbol,
        )
        rows: list[np.ndarray] = []
        coverage: list[int] = []
        emb_dim = 2 * int(self.output_dim)
        for symbol in universe:
            bag = copied.get(symbol)
            if bag is None:
                rows.append(np.zeros(emb_dim, dtype=np.float32))
            else:
                rows.append(_mean_std_pool_np(bag))
            coverage.append(1 if symbol in self.bags.bags_by_symbol else 0)
        embeddings = np.vstack(rows).astype(np.float32)
        coverage_arr = np.asarray(coverage, dtype=np.int64)

        emb_path = embedding_cache_path(self.config, split_type, fold_id)
        save_embedding_cache(
            emb_path,
            symbols=symbols,
            embeddings=embeddings,
            coverage_mask=coverage_arr,
            embedding_method=self.config.embedding_method,
        )
        weights_path = generator_weights_path(self.config, split_type, fold_id)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_weights = weights_path.with_suffix(weights_path.suffix + f".tmp.{os.getpid()}")
        torch.save({"generator_kind": "nn_copy", "state_dict": {}}, tmp_weights)
        os.replace(tmp_weights, weights_path)
        manifest_path = generator_manifest_path(self.config, split_type, fold_id)
        write_generator_manifest(
            manifest_path,
            {
                "split_type": split_type,
                "fold_id": int(fold_id),
                "generator_kind": self.config.generator_kind,
                "embedding_method": self.config.embedding_method,
                "bag_scale": 1.0,
                "bag_scale_mode": self.config.bag_scale_mode,
                "generator_weights_path": str(weights_path),
                "train_bag_gene_count": len(train_covered),
                "val_bag_gene_count": len(val_bag),
                "distill_gene_count": 0,
                "universe_gene_count": int(len(symbols)),
            },
        )
        return Step1TrainResult(
            embedding_path=emb_path,
            manifest_path=manifest_path,
            weights_path=weights_path,
            bag_scale=1.0,
            train_bag_gene_count=len(train_covered),
            val_bag_gene_count=len(val_bag),
        )
```

The `nn_copy` branch in `train_fold` (added in this task) returns this result
before any optimizer is built, and it reuses `nearest_neighbor_copy_predictions`
and `_mean_std_pool_np` from Task 4 — no duplicate NN logic.

- [ ] **Step 8: Run generator tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_generator.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
rtk git add src/sl_dl_model/exp08b_generator.py tests/sl_dl_model/test_exp08b_generator.py
rtk git commit -m "feat: add exp08b direct-mlp and nn-copy step2 controls"
```

Expected: commit succeeds.

---

### Task 6: Step 2 Cached-Embedding Pair Head

**Files:**
- Create: `src/sl_dl_model/exp08b_sl_head.py`
- Modify: `src/sl_dl_model/scoring.py`
- Create: `tests/sl_dl_model/test_exp08b_sl_head.py`

- [ ] **Step 1: Write failing Step 2 tests**

Create `tests/sl_dl_model/test_exp08b_sl_head.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from sl_dl_model.exp08b_artifacts import save_embedding_cache
from sl_dl_model.exp08b_config import SlHeadConfig
from sl_dl_model.exp08b_sl_head import CachedEmbeddingPairHeadProducer


def _write_cache(path: Path) -> None:
    save_embedding_cache(
        path,
        symbols=np.array(["A", "B", "C"], dtype=object),
        embeddings=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        coverage_mask=np.array([1, 1, 0], dtype=np.int64),
        embedding_method="exp08b_state_adapter_meanstd",
    )


def test_cached_embedding_producer_returns_frozen_table(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.npz"
    _write_cache(cache_path)
    cfg = SlHeadConfig(
        max_epochs=1,
        batch_pairs=2,
        pair_hidden=(8,),
        include_coverage_flag=True,
    )
    producer = CachedEmbeddingPairHeadProducer(
        cfg,
        cache_path=cache_path,
        metric_model_name="exp08b",
        train_pairs=[
            ("A", "B", 1, -1.0, -0.5),
            ("A", "C", 0, -1.0, 0.2),
            ("B", "C", 0, -0.5, 0.2),
        ],
    )

    emb, mask = producer.produce(np.array(["A", "B", "C"], dtype=object), {"A", "B"})

    assert emb.shape == (3, 2)
    assert mask.tolist() == [1, 1, 0]
    assert producer.metric_model_name == "exp08b"


def test_cached_pair_head_scores_full_matrix(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.npz"
    _write_cache(cache_path)
    cfg = SlHeadConfig(
        max_epochs=2,
        batch_pairs=2,
        pair_hidden=(8,),
        include_coverage_flag=False,
    )
    producer = CachedEmbeddingPairHeadProducer(
        cfg,
        cache_path=cache_path,
        metric_model_name="direct_esm2_mlp",
        train_pairs=[
            ("A", "B", 1, -1.0, -0.5),
            ("A", "C", 0, -1.0, 0.2),
            ("B", "C", 0, -0.5, 0.2),
        ],
    )
    producer.produce(np.array(["A", "B", "C"], dtype=object), {"A", "B", "C"})

    scores = producer.score_matrix(
        np.array(["A", "B", "C"], dtype=object),
        np.array([-1.0, -0.5, 0.2], dtype=float),
    )

    assert scores.shape == (3, 3)
    assert np.allclose(np.diag(scores), 0.0)
    assert np.isfinite(scores).all()


def test_step2_module_has_no_generator_imports() -> None:
    source = Path("src/sl_dl_model/exp08b_sl_head.py").read_text()
    tree = ast.parse(source)
    forbidden = {
        "StateEncoder",
        "PertAdapter",
        "SlDlModel",
        "StateAdapterBagGenerator",
        "Step1GeneratorTrainer",
        "Exp08bConfig",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert forbidden.isdisjoint(names | attrs)
    assert "sl_dl_model.train" not in source
    assert "state_checkpoint" not in source
    # The slim Step-2 config has no checkpoint/generator field, so the producer
    # physically cannot hold a STATE checkpoint path (spec §7.1).
    assert "esm2_npz" not in source
    assert "gwps_h5ad" not in source


def test_step2_uses_partialstate_device_not_cuda_default() -> None:
    source = Path("src/sl_dl_model/exp08b_sl_head.py").read_text()

    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in source
    assert "PartialState().device" in source or "device=" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_sl_head.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.exp08b_sl_head'`.

- [ ] **Step 3: Implement cached pair-head producer**

Create `src/sl_dl_model/exp08b_sl_head.py`:

```python
"""Step 2 SL-head trainer for exp08b cached embeddings."""

from __future__ import annotations

import numpy as np
import torch
from accelerate import PartialState
from torch import optim

from sl_benchmark_baseline.features import Standardizer, build_pair_features
from sl_dl_model.exp08b_artifacts import load_embedding_cache
from sl_dl_model.exp08b_config import SlHeadConfig
from sl_dl_model.losses import sl_bce_loss
from sl_dl_model.pair_head import SymmetricPairHead


class CachedEmbeddingPairHeadProducer:
    """Producer that trains only ``SymmetricPairHead`` on cached e_hat vectors.

    Holds a slim :class:`SlHeadConfig` (never the full ``Exp08bConfig``) so this
    module has no ``state_checkpoint`` path to leak (spec §7.1). ``run_fold_with_producer``
    reads ``metric_model_name`` to label this rung's metric rows distinctly
    (``exp08b`` / ``direct_esm2_mlp`` / ``nn_copy``).
    """

    def __init__(
        self,
        config: SlHeadConfig,
        *,
        cache_path,
        train_pairs: list[tuple[str, str, int, float, float]],
        metric_model_name: str,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.cache_path = cache_path
        self.train_pairs = train_pairs
        self.metric_model_name = metric_model_name
        self.device = device
        self._symbols: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._coverage_mask: np.ndarray | None = None
        self._head: SymmetricPairHead | None = None
        self._standardizer: Standardizer | None = None

    def produce(
        self,
        symbols: np.ndarray,
        train_symbols: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load cached embeddings, align to universe order, and train the head."""
        payload = load_embedding_cache(self.cache_path)
        cached_symbols = [str(s).upper() for s in payload["symbols"]]
        index = {symbol: i for i, symbol in enumerate(cached_symbols)}
        rows: list[np.ndarray] = []
        cov: list[int] = []
        for symbol_obj in symbols:
            symbol = str(symbol_obj).upper()
            if symbol not in index:
                raise ValueError(f"cached embedding table missing gene {symbol}")
            row = index[symbol]
            rows.append(payload["embeddings"][row])
            cov.append(int(payload["coverage_mask"][row]))
        self._symbols = np.asarray([str(s).upper() for s in symbols], dtype=object)
        self._embeddings = np.vstack(rows).astype(np.float32)
        self._coverage_mask = np.asarray(cov, dtype=np.int64)
        self._train_head()
        return self._embeddings, self._coverage_mask

    def score_matrix(self, symbols: np.ndarray, gene_effects: np.ndarray) -> np.ndarray:
        """Score all universe pairs with the trained pair head."""
        if self._head is None or self._embeddings is None or self._coverage_mask is None:
            self.produce(symbols, set())
        assert self._head is not None
        assert self._embeddings is not None
        assert self._coverage_mask is not None
        assert self._standardizer is not None

        device = next(self._head.parameters()).device
        table = torch.tensor(self._embeddings, device=device, dtype=torch.float32)
        cov = torch.tensor(self._coverage_mask, device=device, dtype=torch.float32)
        n = len(symbols)
        scores = np.zeros((n, n), dtype=float)
        self._head.eval()
        with torch.no_grad():
            for i in range(n):
                ea = np.full(n, float(gene_effects[i]))
                eb = np.asarray(gene_effects, dtype=float)
                ge = torch.tensor(
                    self._standardizer.transform(build_pair_features(ea, eb)),
                    device=device,
                    dtype=torch.float32,
                )
                e_a = table[i].unsqueeze(0).expand(n, -1)
                cov_a = cov[i].expand(n) if self.config.include_coverage_flag else None
                cov_b = cov if self.config.include_coverage_flag else None
                logits = self._head(e_a, table, ge, cov_a, cov_b)
                scores[i] = torch.sigmoid(logits).detach().cpu().numpy()
        np.fill_diagonal(scores, 0.0)
        return scores

    def _train_head(self) -> None:
        if self._embeddings is None or self._symbols is None or self._coverage_mask is None:
            raise RuntimeError("produce must load embeddings before training")
        symbol_index = {str(s).upper(): i for i, s in enumerate(self._symbols)}
        emb_dim = int(self._embeddings.shape[1])
        device = (
            torch.device(self.device)
            if self.device is not None
            else PartialState().device
        )
        head = SymmetricPairHead(
            emb_dim=emb_dim,
            hidden=self.config.pair_hidden,
            include_coverage_flag=self.config.include_coverage_flag,
        ).to(device)
        optimizer = optim.Adam(head.parameters(), lr=self.config.lr)

        ea = np.array([p[3] for p in self.train_pairs], dtype=float)
        eb = np.array([p[4] for p in self.train_pairs], dtype=float)
        self._standardizer = Standardizer.fit(build_pair_features(ea, eb))

        table = torch.tensor(self._embeddings, device=device, dtype=torch.float32)
        cov = torch.tensor(self._coverage_mask, device=device, dtype=torch.float32)
        rows = list(self.train_pairs)
        for _epoch in range(self.config.max_epochs):
            for start in range(0, len(rows), self.config.batch_pairs):
                batch = rows[start : start + self.config.batch_pairs]
                idx_a = [symbol_index[a.upper()] for a, *_ in batch]
                idx_b = [symbol_index[b.upper()] for _, b, *_ in batch]
                labels = torch.tensor(
                    [float(label) for *_ab, label, _ea, _eb in batch],
                    device=device,
                    dtype=torch.float32,
                )
                raw_ge = build_pair_features(
                    np.array([ea for *_prefix, ea, _eb in batch], dtype=float),
                    np.array([eb for *_prefix, _ea, eb in batch], dtype=float),
                )
                ge = torch.tensor(
                    self._standardizer.transform(raw_ge),
                    device=device,
                    dtype=torch.float32,
                )
                idx_a_t = torch.tensor(idx_a, device=device, dtype=torch.long)
                idx_b_t = torch.tensor(idx_b, device=device, dtype=torch.long)
                cov_a = cov[idx_a_t] if self.config.include_coverage_flag else None
                cov_b = cov[idx_b_t] if self.config.include_coverage_flag else None
                logits = head(table[idx_a_t], table[idx_b_t], ge, cov_a, cov_b)
                loss = sl_bce_loss(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), self.config.max_grad_norm)
                optimizer.step()
        self._head = head
```

- [ ] **Step 4: Make `run_fold_with_producer` honor a producer model label**

The current scoring path hardcodes `model="state_dl"` for any producer that
exposes `score_matrix` (`src/sl_dl_model/scoring.py:237,256`), so all three
§5.2 rungs (`exp08b`, `direct_esm2_mlp`, `nn_copy`) would collapse into a single
`state_dl` row and the ladder could not be reported side-by-side (spec §5.2).
Add a `metric_model_name` opt-in: producers that set the attribute label their
own rows; producers that don't keep the legacy `state_dl` label, so the existing
exp08 `StateDlProducer` path is unchanged.

First add the failing test. Append to `tests/sl_dl_model/test_exp08b_sl_head.py`:

```python
def test_run_fold_with_producer_labels_rows_by_metric_model_name(tmp_path: Path) -> None:
    """Each §5.2 rung must produce distinctly-labeled metric rows."""
    import pandas as pd

    from sl_dl_model.scoring import run_fold_with_producer

    cache_path = tmp_path / "embeddings.npz"
    save_embedding_cache(
        cache_path,
        symbols=np.array(["A", "B", "C", "D"], dtype=object),
        embeddings=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
            dtype=np.float32,
        ),
        coverage_mask=np.array([1, 1, 0, 0], dtype=np.int64),
        embedding_method="exp08b_state_adapter_meanstd",
    )
    frame = pd.DataFrame(
        {
            "pair_id": ["p0", "p1", "p2", "p3"],
            "split_type": ["CV2", "CV2", "CV2", "CV2"],
            "fold_id": [0, 0, 0, 0],
            "split_role": ["train", "train", "test", "test"],
            "sl_label": [1, 0, 1, 0],
            "gene_a_symbol": ["A", "A", "C", "C"],
            "gene_b_symbol": ["B", "C", "D", "A"],
            "gene_a_k562_gene_effect": [-1.0, -1.0, 0.2, 0.2],
            "gene_b_k562_gene_effect": [-0.5, 0.2, 0.3, -1.0],
        }
    )
    cfg = SlHeadConfig(max_epochs=1, batch_pairs=4, pair_hidden=(8,), include_coverage_flag=False)

    labels: set[str] = set()
    for model_name in ("exp08b", "direct_esm2_mlp", "nn_copy"):
        producer = CachedEmbeddingPairHeadProducer(
            cfg,
            cache_path=cache_path,
            metric_model_name=model_name,
            train_pairs=[("A", "B", 1, -1.0, -0.5), ("A", "C", 0, -1.0, 0.2)],
            device="cpu",
        )
        rows = run_fold_with_producer(frame, "CV2", 0, cfg, producer)
        row_models = {row["model"] for row in rows}
        assert row_models == {model_name}, f"expected only {model_name}, got {row_models}"
        labels |= row_models

    assert labels == {"exp08b", "direct_esm2_mlp", "nn_copy"}
    assert "state_dl" not in labels
```

Then modify `src/sl_dl_model/scoring.py`. In `run_fold_with_producer`, inside the
`if hasattr(producer, "score_matrix"):` block, replace the two hardcoded
`"state_dl"` literals passed to `_metric_rows` with a resolved label. Add this
line just before the `rows.extend(...)` for the full-universe slice:

```python
        metric_model = getattr(producer, "metric_model_name", "state_dl")
```

Then change both `_metric_rows(split_type, "state_dl", fold_id, ...)` calls in
that block to `_metric_rows(split_type, metric_model, fold_id, ...)` (the
`"full_universe"` and `"covered_pairs"` slices). Leave the sklearn `_transcript`
path below untouched.

- [ ] **Step 5: Run Step 2 tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_sl_head.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add src/sl_dl_model/exp08b_sl_head.py src/sl_dl_model/scoring.py tests/sl_dl_model/test_exp08b_sl_head.py
rtk git commit -m "feat: add exp08b cached pair head trainer with ladder labels"
```

Expected: commit succeeds.

---

### Task 7: Two-Pass Queue And CLI Entrypoints

**Files:**
- Create: `src/sl_dl_model/exp08b_queue.py`
- Create: `src/sl_dl_model/exp08b_runner.py` (shared label-free orchestration helpers — imports neither the generator/STATE nor the pair head)
- Create: `src/sl_dl_model/exp08b_step1_runner.py` (Step 1 / `train-generator` pass; imports the generator + STATE, never the pair head or `sl_label`)
- Create: `src/sl_dl_model/exp08b_step2_runner.py` (Step 2 / `train-sl-head` pass; imports the pair head + scoring, never the generator/STATE)
- Modify: `src/sl_dl_model/__main__.py`
- Modify: `src/sl_dl_model/fold_queue.py`
- Create: `tests/sl_dl_model/test_exp08b_queue_cli.py`

- [ ] **Step 1: Write failing queue and CLI tests**

Create `tests/sl_dl_model/test_exp08b_queue_cli.py`:

```python
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_queue import step_failed_path, step_result_path


def test_step_queue_paths_include_step_name(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")

    assert step_result_path(cfg, "generator", "CV2", 0).name == "CV2_fold0.result.json"
    assert step_result_path(cfg, "generator", "CV2", 0).parent.name == "generator"
    assert step_failed_path(cfg, "sl_head", "CV3", 4).parent.name == "sl_head"


def test_cli_help_lists_exp08b_entrypoints() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "sl_dl_model", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert out.returncode == 0
    assert "train-generator" in out.stdout
    assert "train-sl-head" in out.stdout


def test_train_sl_head_missing_cache_fails_fast(tmp_path: Path) -> None:
    import yaml

    csv = tmp_path / "pairs.csv"
    csv.write_text(
        "pair_id,split_type,fold_id,split_role,sl_label,gene_a_symbol,gene_b_symbol,"
        "gene_a_k562_gene_effect,gene_b_k562_gene_effect\n"
        "p0,CV2,0,train,1,A,B,-1.0,-0.5\n"
        "p1,CV2,0,test,0,A,C,-1.0,0.2\n"
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "input_csv": str(csv),
                "output_dir": str(tmp_path / "run"),
                "split_types": ["CV2"],
                "folds": [0],
                "ranking_k": [10],
            }
        )
    )

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "sl_dl_model",
            "train-sl-head",
            "--config",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert out.returncode != 0
    assert "missing Step 1 cache" in out.stderr or "missing Step 1 cache" in out.stdout


def test_runner_exception_handlers_quarantine_and_continue() -> None:
    # Per-fold try/except quarantine-and-continue lives in the two step runners.
    for module in (
        "src/sl_dl_model/exp08b_step1_runner.py",
        "src/sl_dl_model/exp08b_step2_runner.py",
    ):
        source = Path(module).read_text()
        tree = ast.parse(source)
        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "Exception"
        ]

        assert handlers, f"no broad handler found in {module}"
        for handler in handlers:
            assert any(isinstance(node, ast.Continue) for node in ast.walk(handler))
            assert not any(isinstance(node, ast.Raise) for node in ast.walk(handler))


def test_step1_runner_does_not_import_pair_head_or_sl_label() -> None:
    """Spec §7.1: the Step-1 entrypoint knows nothing about SL labels / pair head.

    Guard the actual train-generator import path, not just exp08b_generator.py:
    importing exp08b_step1_runner must not pull in the pair head, scoring,
    SlHeadConfig, or the CachedEmbeddingPairHeadProducer, and the module source
    must not read sl_label. The label-aware ``sl_dl_model.scoring`` module is
    forbidden outright (P2): it reads ``sl_label`` and lazily constructs the old
    SL-coupled ``StateDlProducer`` path, so even importing its label-free
    ``train_symbols_for_fold`` helper would re-couple Step 1 to it. Step 1 inlines
    its own ``_train_symbols`` instead.
    """
    source = Path("src/sl_dl_model/exp08b_step1_runner.py").read_text()
    for forbidden in (
        "SymmetricPairHead",
        "CachedEmbeddingPairHeadProducer",
        "SlHeadConfig",
        "run_fold_with_producer",
        "exp08b_sl_head",
        "sl_label",
        "sl_dl_model.scoring",
        "train_symbols_for_fold",
    ):
        assert forbidden not in source, f"{forbidden} leaked into Step-1 runner"


def test_step2_runner_does_not_import_generator_or_state() -> None:
    """Spec §7.1: the Step-2 entrypoint must not import the generator or STATE."""
    source = Path("src/sl_dl_model/exp08b_step2_runner.py").read_text()
    for forbidden in (
        "Step1GeneratorTrainer",
        "StateEncoder",
        "PertAdapter",
        "SlDlModel",
        "exp08b_generator",
        "state_checkpoint",
        "_load_state_dl_caches",
    ):
        assert forbidden not in source, f"{forbidden} leaked into Step-2 runner"


def test_step2_fold_fingerprint_changes_when_step1_cache_rewritten(tmp_path: Path) -> None:
    """Step 2 resume must bind the Step-1 cache it consumes, not config alone.

    Regression for [P1]: ``fq.fingerprint`` ignores the on-disk ê_g table, so a
    Step-1 regeneration under the same config / output_dir would otherwise let
    Step 2 reuse stale rows. The per-fold fingerprint must change when the
    embedding cache is rewritten with new contents.
    """
    import numpy as np

    from sl_dl_model.exp08b_artifacts import (
        embedding_cache_path,
        generator_manifest_path,
        save_embedding_cache,
        write_generator_manifest,
    )
    from sl_dl_model.exp08b_queue import step2_fold_fingerprint

    cfg = Exp08bConfig(output_dir=tmp_path / "run")
    emb_path = embedding_cache_path(cfg, "CV2", 0)
    write_generator_manifest(
        generator_manifest_path(cfg, "CV2", 0),
        {"split_type": "CV2", "fold_id": 0, "bag_scale": 1.0},
    )
    save_embedding_cache(
        emb_path,
        symbols=np.array(["A", "B"], dtype=object),
        embeddings=np.array([[1.0], [2.0]], dtype=np.float32),
        coverage_mask=np.array([1, 0], dtype=np.int64),
        embedding_method="exp08b_state_adapter_meanstd",
    )
    before = step2_fold_fingerprint(cfg, "CV2", 0)

    # Rewrite the Step-1 cache with different contents (bumps size/mtime).
    save_embedding_cache(
        emb_path,
        symbols=np.array(["A", "B", "C"], dtype=object),
        embeddings=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        coverage_mask=np.array([1, 0, 1], dtype=np.int64),
        embedding_method="exp08b_state_adapter_meanstd",
    )
    after = step2_fold_fingerprint(cfg, "CV2", 0)

    assert before != after


def test_step2_result_cache_fp_round_trips_inside_result_json(tmp_path: Path) -> None:
    """The consumed cache_fp lives INSIDE .result.json, not a sidecar (P3).

    Spec §4.4: ``.result.json`` is the sole result-side cross-run resume state.
    ``fq.write_result(..., extra={"cache_fp": ...})`` persists the binding inside
    that file, and ``read_step2_result_cache_fp`` reads it back. A result written
    without the binding (older / non-exp08b path) reads back ``None`` so the fold
    recomputes.
    """
    from sl_dl_model import fold_queue as fq
    from sl_dl_model.exp08b_queue import read_step2_result_cache_fp

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # No result yet → None.
    assert read_step2_result_cache_fp(results_dir, "CV2", 0) is None

    # Result with the cache_fp binding → reads it back; rows/fingerprint intact.
    fq.write_result(
        results_dir, "CV2", 0, [{"metric": "ndcg", "value": 1.0}],
        fingerprint="run_fp", extra={"cache_fp": "fp_abc"},
    )
    assert read_step2_result_cache_fp(results_dir, "CV2", 0) == "fp_abc"
    assert fq.read_result_rows(results_dir, "CV2", 0, fingerprint="run_fp") == [
        {"metric": "ndcg", "value": 1.0}
    ]

    # Result without the binding (extra omitted) → None, so the fold recomputes.
    fq.write_result(results_dir, "CV3", 1, [], fingerprint="run_fp")
    assert read_step2_result_cache_fp(results_dir, "CV3", 1) is None


def test_step2_metric_model_name_reads_generator_kind_from_manifest(
    tmp_path: Path,
) -> None:
    """Step 2 labels rows from the Step-1 manifest, not config.generator_kind (P2).

    Spec §7.1: the two steps communicate only through the ê_g table + manifest, so
    the row label must reflect the generator that actually wrote the cache. Even
    when ``config.generator_kind`` is re-pointed, the label follows the manifest.
    """
    from sl_dl_model.exp08b_artifacts import (
        generator_manifest_path,
        write_generator_manifest,
    )
    from sl_dl_model.exp08b_queue import step2_metric_model_name

    # config says state_adapter, but the manifest on disk says direct_mlp.
    cfg = Exp08bConfig(output_dir=tmp_path / "run", generator_kind="state_adapter")
    write_generator_manifest(
        generator_manifest_path(cfg, "CV2", 0),
        {"split_type": "CV2", "fold_id": 0, "generator_kind": "direct_mlp"},
    )
    # The label follows the manifest (direct_esm2_mlp), not config (exp08b).
    assert step2_metric_model_name(cfg, "CV2", 0) == "direct_esm2_mlp"


def test_step2_metric_config_is_state_neutral(tmp_path: Path) -> None:
    """Step 2's metric fingerprint must not depend on STATE/ESM2/GWPS inputs (P2).

    The spec forbids Step 2 from holding a STATE checkpoint and says the steps
    communicate only via the ê_g table + manifest. ``step2_metric_config`` must
    blank the generator-only inputs so two configs that differ ONLY in those
    fields produce the same ``fq.fingerprint`` and the same neutralized config.
    """
    from sl_dl_model import fold_queue as fq
    from sl_dl_model.exp08b_queue import step2_metric_config

    base = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_checkpoint=Path("/ckpt_a/checkpoints/final.ckpt"),
        esm2_npz=Path("/esm/a.npz"),
        gwps_h5ad=Path("/gwps/a.h5ad"),
        bags_npz=Path("/bags/a.npz"),
    )
    other = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_checkpoint=Path("/ckpt_b/checkpoints/final.ckpt"),
        esm2_npz=Path("/esm/b.npz"),
        gwps_h5ad=Path("/gwps/b.h5ad"),
        bags_npz=Path("/bags/b.npz"),
    )

    neutral_base = step2_metric_config(base)
    assert neutral_base.state_checkpoint is None
    assert neutral_base.esm2_npz is None
    assert neutral_base.gwps_h5ad is None
    assert neutral_base.bags_npz is None
    assert neutral_base.state_backend == "linear_mock"
    # Output is redirected into the Step-2 results subtree.
    assert neutral_base.output_dir.name == base.step2_results_subdir

    # The neutralized fingerprint ignores the generator-only inputs entirely.
    assert fq.fingerprint(step2_metric_config(base)) == fq.fingerprint(
        step2_metric_config(other)
    )


def test_step2_stale_failed_marker_ignored_after_cache_appears(tmp_path: Path) -> None:
    """A fold quarantined before its Step-1 cache existed must retry (P3).

    A premature train-sl-head run writes a ``.failed`` marker bound to the
    pre-cache ``cache_fp``. Once Step 1 writes the cache, ``step2_fold_fingerprint``
    changes, so the recorded failed ``cache_fp`` no longer matches and the fold is
    eligible to re-run.
    """
    import numpy as np

    from sl_dl_model import fold_queue as fq
    from sl_dl_model.exp08b_artifacts import (
        embedding_cache_path,
        generator_manifest_path,
        save_embedding_cache,
        write_generator_manifest,
    )
    from sl_dl_model.exp08b_queue import (
        read_step2_failed_cache_fp,
        step2_fold_fingerprint,
        step2_metric_config,
    )

    cfg = Exp08bConfig(output_dir=tmp_path / "run")
    metric_config = step2_metric_config(cfg)
    results_dir = fq.fold_results_dir(metric_config)
    fp = fq.fingerprint(metric_config)

    # Quarantine the fold against the pre-cache fingerprint (no ê_g table yet).
    cache_fp_before = step2_fold_fingerprint(cfg, "CV2", 0)
    fq.write_failed(
        results_dir,
        "CV2",
        0,
        {"split_type": "CV2", "fold_id": 0, "error": "missing", "cache_fp": cache_fp_before},
        fingerprint=fp,
    )
    assert read_step2_failed_cache_fp(results_dir, "CV2", 0) == cache_fp_before

    # Step 1 now produces the cache + manifest, changing the per-fold fingerprint.
    write_generator_manifest(
        generator_manifest_path(cfg, "CV2", 0),
        {"split_type": "CV2", "fold_id": 0, "bag_scale": 1.0},
    )
    save_embedding_cache(
        embedding_cache_path(cfg, "CV2", 0),
        symbols=np.array(["A", "B"], dtype=object),
        embeddings=np.array([[1.0], [2.0]], dtype=np.float32),
        coverage_mask=np.array([1, 0], dtype=np.int64),
        embedding_method="exp08b_state_adapter_meanstd",
    )
    cache_fp_after = step2_fold_fingerprint(cfg, "CV2", 0)

    # The marker is stale: its recorded cache_fp no longer matches, so the
    # runner's "honor failed only if cache_fp matches" gate lets the fold retry.
    assert cache_fp_after != cache_fp_before
    assert read_step2_failed_cache_fp(results_dir, "CV2", 0) != cache_fp_after
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_queue_cli.py -q
```

Expected: FAIL with missing `exp08b_queue` or missing CLI subcommands.

- [ ] **Step 3: Implement step-aware queue wrappers**

Create `src/sl_dl_model/exp08b_queue.py`:

```python
"""Step-aware filesystem queue wrappers for exp08b."""

from __future__ import annotations

from pathlib import Path

from sl_dl_model import fold_queue as fq
from sl_dl_model.exp08b_config import Exp08bConfig

# The ONLY result-affecting config fields Step 2's official metric depends on:
# the fold/scoring structure and the SlHeadConfig pair-head fields. Everything
# else in fq._FINGERPRINT_FIELDS / _FINGERPRINT_PATH_FIELDS is a Step-1
# generator concern (STATE/ESM2/GWPS paths, bag/distill weights, adapter width,
# control-template size, embedding_method, the generator_* / bag_scale_* knobs)
# and is canonicalized away by step2_metric_config so it can never enter the
# Step-2 hash. Deriving the canonical set from fq._FINGERPRINT_FIELDS (rather
# than enumerating the generator fields) means any future generator fingerprint
# field is auto-neutralized instead of silently leaking into Step 2 (spec §7.1:
# the steps communicate only through the ê_g table + manifest).
_STEP2_FINGERPRINT_ALLOWLIST = frozenset(
    {
        "split_types",
        "folds",
        "ranking_k",
        "seed",
        "fallback_strategy",
        "include_coverage_flag",
        "pair_hidden",
        "lr",
        "max_epochs",
        "batch_pairs",
        "max_grad_norm",
    }
)


def step_results_dir(config: Exp08bConfig, step: str) -> Path:
    """Return the queue result directory for one exp08b step."""
    return fq.fold_results_dir(config) / step


def step_result_path(
    config: Exp08bConfig,
    step: str,
    split_type: str,
    fold_id: int,
) -> Path:
    """Return the success result path for one step job."""
    return fq.result_path(step_results_dir(config, step), split_type, fold_id)


def step_failed_path(
    config: Exp08bConfig,
    step: str,
    split_type: str,
    fold_id: int,
) -> Path:
    """Return the failed marker path for one step job."""
    return fq.failed_path(step_results_dir(config, step), split_type, fold_id)


def step2_metric_config(config: Exp08bConfig) -> Exp08bConfig:
    """Return a Step-2 metric config that depends only on the allowlist.

    Spec §7.1: Step 2 consumes only the cached ``ê_g`` table + SL pairs and must
    not depend on any Step-1 generator input. ``fq.fingerprint`` hashes *every*
    field in ``fq._FINGERPRINT_FIELDS`` / ``fq._FINGERPRINT_PATH_FIELDS`` — which
    includes not just the STATE/ESM2/GWPS *path* fields but also the generator
    *scalars* (``lambda_bag``, ``lambda_distill``, ``adapter_hidden``,
    ``pert_dim``, ``control_template_size``, ``warmup_epochs``, ``pooling``,
    ``embedding_method``, plus the exp08b ``generator_*`` / ``bag_scale_*``
    fields added to the fingerprint in Step 6). A path-only neutralization would
    leave all of those scalars in the Step-2 hash (P2).

    So this canonicalizes **every** fingerprint-affecting field that is *not* in
    :data:`_STEP2_FINGERPRINT_ALLOWLIST` to a fixed sentinel (``None``). Deriving
    the canonical set from ``fq._FINGERPRINT_FIELDS`` (rather than enumerating
    the generator fields) means any future generator fingerprint field is
    auto-neutralized rather than silently leaking into Step 2. The result is that
    ``fq.fingerprint(step2_metric_config(config))`` depends only on the
    allowlisted fold / scoring / pair-head fields + the content-hashed
    ``input_csv``. The runner passes this neutralized config (never the full one)
    into the fingerprint, scoring, and assembly, so no STATE/ESM2/GWPS input
    reaches the Step-2 metric path.

    ``output_dir`` is redirected to the Step-2 results subtree so Step-1 and
    Step-2 results never collide.
    """
    from dataclasses import replace

    from sl_dl_model.exp08b_artifacts import step2_output_dir

    # Every fingerprint-affecting field outside the allowlist → None. The path
    # fields (state_checkpoint, esm2_npz, gwps_h5ad, gwps_overlap_csv, bags_npz)
    # are in _FINGERPRINT_PATH_FIELDS and none are allowlisted, so they all blank
    # to None here.
    neutral: dict[str, object] = {
        name: None
        for name in (*fq._FINGERPRINT_FIELDS, *fq._FINGERPRINT_PATH_FIELDS)
        if name not in _STEP2_FINGERPRINT_ALLOWLIST
    }
    # state_backend == "linear_mock" both lands a stable sentinel in the hash and
    # short-circuits the STATE-sidecar (var_dims.pkl / pert_onehot_map.pt)
    # signatures inside fq.fingerprint, so no checkpoint-adjacent file is stat'd.
    neutral["state_backend"] = "linear_mock"
    return replace(
        config,
        output_dir=step2_output_dir(config),
        **neutral,
    )


def step2_fold_fingerprint(
    config: Exp08bConfig,
    split_type: str,
    fold_id: int,
) -> str:
    """Return a Step-2 fingerprint that also binds the Step-1 cache it consumes.

    Step 2's only inputs are the STATE-neutral config fields (via
    :func:`step2_metric_config`, so STATE/ESM2/GWPS paths never enter the hash)
    and the on-disk Step-1 ``ê_g`` table + generator manifest for this fold
    (spec §7.1: the two steps communicate only through those artifacts). The
    plain ``fq.fingerprint(config)`` ignores those artifacts, so a Step-1
    regeneration under the same config / ``output_dir`` would let Step 2 reuse
    stale ``.result.json`` rows. Folding the cache + manifest
    ``(path, size, mtime_ns)`` stat signatures into the fingerprint busts that
    stale reuse the moment Step 1 rewrites either file.

    Args:
        config: The exp08b run configuration (path helpers resolve against its
            original ``output_dir``; the config-portion is STATE-neutralized).
        split_type: CV split type for this fold.
        fold_id: Fold id for this fold.

    Returns:
        A 16-hex-char fingerprint string.
    """
    import hashlib

    from sl_dl_model.exp08b_artifacts import (
        embedding_cache_path,
        generator_manifest_path,
    )

    h = hashlib.sha256()
    h.update(f"config={fq.fingerprint(step2_metric_config(config))}".encode())
    h.update(
        f"embeddings={fq._path_signature(embedding_cache_path(config, split_type, fold_id))}".encode()
    )
    h.update(
        f"manifest={fq._path_signature(generator_manifest_path(config, split_type, fold_id))}".encode()
    )
    return h.hexdigest()[:16]


def step2_metric_model_name(
    config: Exp08bConfig,
    split_type: str,
    fold_id: int,
) -> str:
    """Return the §5.2 ladder label for a fold, read from the Step-1 manifest.

    Spec §7.1: the two steps communicate **only** through the on-disk ``ê_g``
    table + manifest, so Step 2 must label its metric rows from what Step 1
    actually produced — not from the live ``config.generator_kind``. A
    ``train-sl-head`` invocation whose ``config`` disagrees with the generator
    that wrote the cache (e.g. a hand-edited ``generator_kind`` re-pointed at an
    existing cache) would otherwise mislabel exp08b / direct-ESM2-MLP / NN-copy
    rows in the official-metric summary. Reading ``generator_kind`` from
    ``generator_manifest.json`` makes the row label provably match the cache it
    scored.

    Raises:
        ValueError: If the manifest lacks a string ``generator_kind`` or it is
            not a known §5.2 rung (delegated to :func:`metric_model_name_for`).
    """
    from sl_dl_model.exp08b_artifacts import (
        generator_manifest_path,
        load_generator_manifest,
    )
    from sl_dl_model.exp08b_config import metric_model_name_for

    manifest = load_generator_manifest(
        generator_manifest_path(config, split_type, fold_id)
    )
    kind = manifest.get("generator_kind")
    if not isinstance(kind, str):
        raise ValueError(
            f"generator manifest for {split_type}/fold{fold_id} has no string "
            f"'generator_kind': {kind!r}"
        )
    return metric_model_name_for(kind)


def read_step2_result_cache_fp(
    results_dir: Path,
    split_type: str,
    fold_id: int,
) -> str | None:
    """Return the Step-1 cache fingerprint a fold's ``.result.json`` was built on.

    Spec §4.4: ``.result.json`` is the **sole** cross-run resume state, so the
    consumed cache fingerprint is stored *inside* that file (under a ``cache_fp``
    key written via ``fq.write_result(..., extra={"cache_fp": ...})``) rather than
    in a separate sidecar. ``None`` means no result exists, it is unreadable, or
    it predates this binding — in any of which cases the Step-2 result must be
    recomputed even when its config fingerprint matches, because the Step-1 cache
    changed or was never bound.
    """
    path = fq.result_path(results_dir, split_type, fold_id)
    if not path.exists():
        return None
    try:
        payload = fq.read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("cache_fp")
    return value if isinstance(value, str) else None


def read_step2_failed_cache_fp(
    results_dir: Path,
    split_type: str,
    fold_id: int,
) -> str | None:
    """Return the Step-1 cache fingerprint a fold's ``.failed`` marker was bound to.

    Step-2 failed markers carry a ``cache_fp`` field recording the
    :func:`step2_fold_fingerprint` value current when the failure was written.
    ``None`` means no marker exists, it is unreadable, or it predates this
    binding. Used so a fold quarantined *before* its Step-1 cache existed (e.g.
    a premature ``train-sl-head`` run that hit the missing-cache error) is retried
    once Step 1 produces the cache, instead of staying permanently quarantined.
    """
    path = fq.failed_path(results_dir, split_type, fold_id)
    if not path.exists():
        return None
    try:
        payload = fq.read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("cache_fp")
    return value if isinstance(value, str) else None
```

- [ ] **Step 4: Implement the shared runner helpers and the two step runners**

The Step-1 and Step-2 passes live in **separate modules** so neither leaks the
other's imports at module load (spec §7.1: Step 1 "knows nothing about SL labels
or the pair head"; Step 2 "must not import `StateEncoder`/`PertAdapter` or hold a
STATE checkpoint"). Shared, label-free orchestration helpers live in
`exp08b_runner.py`; the Step-1 entrypoint imports the generator (and thus STATE)
but never the pair head, and the Step-2 entrypoint imports the pair head but
never the generator/STATE.

Create `src/sl_dl_model/exp08b_runner.py` (shared helpers only — imports neither
the generator/STATE nor the pair head, so importing it is side-effect-free for
both passes):

```python
"""Shared, label-free orchestration helpers for the exp08b two-pass runners.

This module is imported by BOTH step runners, so it must not import the Step-1
generator/STATE stack or the Step-2 pair-head/scoring stack. Each step runner
owns its own step-specific imports (spec §7.1 entrypoint boundary).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sl_dl_model import fold_queue as fq
from sl_dl_model.exp08b_config import Exp08bConfig

logger = logging.getLogger(__name__)


def jobs(frame: pd.DataFrame, config: Exp08bConfig) -> list[tuple[str, int]]:
    """Return the ``(split_type, fold_id)`` job list available in ``frame``."""
    split_types = config.split_types or ("CV1", "CV2", "CV3")
    available = set(frame["split_type"].unique())
    return [(s, f) for s in split_types if s in available for f in config.folds]


def raise_if_step_incomplete(
    *,
    results_dir: Path,
    job_list: list[tuple[str, int]],
    fingerprint: str,
    step: str,
) -> None:
    """Raise after the queue pass if any job failed or never produced a result."""
    failed: list[str] = []
    missing: list[str] = []
    for split_type, fold_id in job_list:
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fingerprint):
            continue
        failed_path = fq.failed_path(results_dir, split_type, fold_id)
        if fq.is_failed(results_dir, split_type, fold_id, fingerprint=fingerprint):
            payload = fq.read_json(failed_path)
            failed.append(
                f"{split_type}/fold{fold_id}: {payload.get('error', failed_path)}"
            )
        else:
            missing.append(f"{split_type}/fold{fold_id}")
    if failed or missing:
        raise RuntimeError(
            f"{step} queue incomplete; failed={failed}; missing={missing}"
        )
```

Create `src/sl_dl_model/exp08b_step1_runner.py` (Step 1 only — imports the
generator/STATE stack, never the pair head, scoring, `SlHeadConfig`, or
`sl_label`):

```python
"""Step 1 (train-generator) queue pass. Knows nothing about SL labels."""

from __future__ import annotations

import logging

import pandas as pd
from accelerate import PartialState

from sl_benchmark_baseline.data import fold_split, load_benchmark
from sl_dl_model import fold_queue as fq
from sl_dl_model.evaluate import _load_state_dl_caches
from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_generator import Step1GeneratorTrainer
from sl_dl_model.exp08b_queue import step_results_dir
from sl_dl_model.exp08b_runner import jobs, raise_if_step_incomplete

logger = logging.getLogger(__name__)


def _train_symbols(train_df: pd.DataFrame) -> set[str]:
    """Return the upper-cased gene symbols in this fold's train pairs.

    Label-free by construction: it reads only ``gene_a_symbol`` /
    ``gene_b_symbol`` and never ``sl_label``. Inlined here (rather than imported
    from ``sl_dl_model.scoring``) so the Step-1 entrypoint has no source-level
    dependency on the label-aware scoring module — which reads ``sl_label`` and
    lazily constructs the old SL-coupled ``StateDlProducer`` path (spec §7.1:
    Step 1 "knows nothing about SL labels or the pair head"; enforced by the
    Step-1 import guard forbidding ``sl_dl_model.scoring``).
    """
    a = train_df["gene_a_symbol"].astype(str).str.upper()
    b = train_df["gene_b_symbol"].astype(str).str.upper()
    return set(a) | set(b)


def run_train_generator(config: Exp08bConfig) -> None:
    """Run the Step 1 generator queue pass."""
    frame = load_benchmark(config.input_csv)
    caches = _load_state_dl_caches(config)
    state = PartialState()
    token = fq.run_token()
    fp = fq.fingerprint(config)
    results_dir = step_results_dir(config, "generator")
    results_dir.mkdir(parents=True, exist_ok=True)
    universe_symbols = sorted(
        set(frame["gene_a_symbol"].astype(str).str.upper())
        | set(frame["gene_b_symbol"].astype(str).str.upper())
    )

    for split_type, fold_id in jobs(frame, config):
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        if fq.is_failed(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        if not fq.try_claim(results_dir, split_type, fold_id, run_token=token):
            continue
        try:
            train_df, _test_df = fold_split(frame, split_type, fold_id)
            trainer = Step1GeneratorTrainer(
                config,
                esm=caches.esm,
                bags=caches.bags,
                input_dim=caches.input_dim,
                output_dim=caches.output_dim,
                device=state.device,
            )
            result = trainer.train_fold(
                split_type=split_type,
                fold_id=fold_id,
                symbols=pd.Series(universe_symbols, dtype=object).to_numpy(),
                train_symbols=_train_symbols(train_df),
            )
            fq.write_result(
                results_dir,
                split_type,
                fold_id,
                [
                    {
                        "split_type": split_type,
                        "fold_id": fold_id,
                        "embedding_path": str(result.embedding_path),
                        "manifest_path": str(result.manifest_path),
                        "bag_scale": result.bag_scale,
                    }
                ],
                fingerprint=fp,
            )
        except Exception as exc:
            fq.write_failed(
                results_dir,
                split_type,
                fold_id,
                {"split_type": split_type, "fold_id": fold_id, "error": repr(exc)},
                fingerprint=fp,
            )
            logger.exception(
                "[rank %d] generator fold %s/%d failed; quarantined and continuing",
                state.process_index,
                split_type,
                fold_id,
            )
            continue
    logger.info("[rank %d] train-generator pass complete", state.process_index)
    if state.is_main_process:
        raise_if_step_incomplete(
            results_dir=results_dir,
            job_list=jobs(frame, config),
            fingerprint=fp,
            step="generator",
        )
```

Create `src/sl_dl_model/exp08b_step2_runner.py` (Step 2 only — imports the pair
head + scoring, never the generator/STATE; passes the STATE-neutralized metric
config into scoring and assembly so no checkpoint path reaches the metric path):

```python
"""Step 2 (train-sl-head) queue pass. Never imports the generator or STATE."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from accelerate import PartialState

from sl_benchmark_baseline.data import fold_split, load_benchmark
from sl_dl_model import fold_queue as fq
from sl_dl_model.evaluate import _assemble
from sl_dl_model.exp08b_artifacts import embedding_cache_path
from sl_dl_model.exp08b_config import Exp08bConfig, SlHeadConfig
from sl_dl_model.exp08b_queue import (
    read_step2_failed_cache_fp,
    read_step2_result_cache_fp,
    step2_fold_fingerprint,
    step2_metric_config,
    step2_metric_model_name,
)
from sl_dl_model.exp08b_runner import jobs, raise_if_step_incomplete
from sl_dl_model.exp08b_sl_head import CachedEmbeddingPairHeadProducer
from sl_dl_model.scoring import run_fold_with_producer

logger = logging.getLogger(__name__)


def _train_pairs(train_df: pd.DataFrame) -> list[tuple[str, str, int, float, float]]:
    return [
        (
            str(row["gene_a_symbol"]).upper(),
            str(row["gene_b_symbol"]).upper(),
            int(row["sl_label"]),
            float(row["gene_a_k562_gene_effect"]),
            float(row["gene_b_k562_gene_effect"]),
        )
        for _, row in train_df.iterrows()
    ]


def run_train_sl_head(config: Exp08bConfig) -> None:
    """Run the Step 2 SL-head queue pass and assemble official metrics.

    The metric/results config is STATE-neutralized via ``step2_metric_config``:
    scoring, the result/failed fingerprint, and ``_assemble`` all use a config
    whose ``fq.fingerprint`` depends only on SL/scoring/output fields + the input
    CSV (no STATE checkpoint, ESM2, GWPS, or sidecars). The full ``config`` is
    used only to resolve the Step-1 artifact *paths* this fold consumes, never
    passed into scoring.
    """
    frame = load_benchmark(config.input_csv)
    state = PartialState()
    token = fq.run_token()
    metric_config = step2_metric_config(config)
    fp = fq.fingerprint(metric_config)
    results_dir = fq.fold_results_dir(metric_config)
    results_dir.mkdir(parents=True, exist_ok=True)
    job_list = jobs(frame, config)

    for split_type, fold_id in job_list:
        cache_fp = step2_fold_fingerprint(config, split_type, fold_id)
        # A prior result is reusable only if BOTH the (allowlist-only) config
        # fingerprint matches AND the Step-1 cache it consumed is unchanged. The
        # consumed cache_fp lives INSIDE the .result.json (spec §4.4: the result
        # file is the sole result-side resume state — no separate sidecar). This
        # guards against a Step-1 regeneration under the same config/output_dir
        # silently reusing stale Step-2 rows (spec §7.1: the two steps communicate
        # only through the on-disk ê_g table + manifest).
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fp) and (
            read_step2_result_cache_fp(results_dir, split_type, fold_id) == cache_fp
        ):
            continue
        # A quarantined fold is honored only if it failed against the SAME Step-1
        # cache fingerprint. A premature train-sl-head run quarantines a fold for a
        # missing cache; once Step 1 produces the cache, cache_fp changes and the
        # stale .failed marker is ignored so the fold retries (P3).
        if (
            fq.is_failed(results_dir, split_type, fold_id, fingerprint=fp)
            and read_step2_failed_cache_fp(results_dir, split_type, fold_id) == cache_fp
        ):
            continue
        if not fq.try_claim(results_dir, split_type, fold_id, run_token=token):
            continue
        try:
            cache_path = embedding_cache_path(config, split_type, fold_id)
            if not Path(cache_path).exists():
                raise RuntimeError(
                    f"missing Step 1 cache for {split_type}/fold{fold_id}: {cache_path}"
                )
            train_df, _test_df = fold_split(frame, split_type, fold_id)
            producer = CachedEmbeddingPairHeadProducer(
                SlHeadConfig.from_exp08b(config),
                cache_path=cache_path,
                train_pairs=_train_pairs(train_df),
                # Label rows from what Step 1 actually wrote (the manifest's
                # generator_kind), NOT the live config.generator_kind, so a
                # config re-pointed at an existing cache cannot mislabel the
                # exp08b / direct-ESM2-MLP / NN-copy rung (P2; spec §7.1).
                metric_model_name=step2_metric_model_name(config, split_type, fold_id),
                device=state.device,
            )
            rows = run_fold_with_producer(
                frame, split_type, fold_id, metric_config, producer
            )
            # Persist the consumed cache_fp INSIDE the single .result.json (spec
            # §4.4: the result file is the sole result-side resume state — no
            # separate sidecar). A later Step-1 regeneration changes cache_fp, so
            # the done-gate above recomputes instead of reusing stale rows.
            fq.write_result(
                results_dir,
                split_type,
                fold_id,
                rows,
                fingerprint=fp,
                extra={"cache_fp": cache_fp},
            )
        except Exception as exc:
            # Record cache_fp on the failed marker so a fold quarantined before
            # its Step-1 cache existed retries once Step 1 produces it (P3).
            fq.write_failed(
                results_dir,
                split_type,
                fold_id,
                {
                    "split_type": split_type,
                    "fold_id": fold_id,
                    "error": repr(exc),
                    "cache_fp": cache_fp,
                },
                fingerprint=fp,
            )
            logger.exception(
                "[rank %d] sl-head fold %s/%d failed; quarantined and continuing",
                state.process_index,
                split_type,
                fold_id,
            )
            continue

    if state.is_main_process:
        split_types = tuple(sorted({s for s, _f in job_list}))
        try:
            _assemble(metric_config, job_list, split_types, frame, shared=None)
        except RuntimeError:
            raise_if_step_incomplete(
                results_dir=results_dir,
                job_list=job_list,
                fingerprint=fp,
                step="sl_head",
            )
            raise
```

- [ ] **Step 5: Add CLI subcommands**

In `src/sl_dl_model/__main__.py`, add subparsers inside `_build_parser()`:

```python
    gen = sub.add_parser(
        "train-generator",
        help="Run exp08b Step 1 fold-local generator training.",
    )
    gen.add_argument("--config", type=Path, required=True, help="Path to Exp08bConfig YAML.")
    gen.add_argument("--log-file", type=Path, default=None, help="Optional log file.")

    head = sub.add_parser(
        "train-sl-head",
        help="Run exp08b Step 2 SL-head training on cached Step 1 embeddings.",
    )
    head.add_argument("--config", type=Path, required=True, help="Path to Exp08bConfig YAML.")
    head.add_argument("--log-file", type=Path, default=None, help="Optional log file.")
```

In `main()`, before loading the old `SLDLConfig`, route exp08b commands. Each
branch lazy-imports **only its own** step-runner module, so the `train-generator`
path never imports the Step-2 pair-head/scoring stack and the `train-sl-head`
path never imports the Step-1 generator/STATE stack (spec §7.1 entrypoint
boundary — a static guard over the actual import path is added in Step 7):

```python
    if args.command in {"train-generator", "train-sl-head"}:
        from sl_dl_model.exp08b_config import load_exp08b_config

        config = load_exp08b_config(args.config)
        log_file = args.log_file or Path(config.output_dir) / f"{args.command}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
        )
        if args.command == "train-generator":
            from sl_dl_model.exp08b_step1_runner import run_train_generator

            run_train_generator(config)
        else:
            from sl_dl_model.exp08b_step2_runner import run_train_sl_head

            run_train_sl_head(config)
        return
```

- [ ] **Step 6: Add exp08b fields to queue fingerprint**

In `src/sl_dl_model/fold_queue.py`, extend `_FINGERPRINT_FIELDS` with:

```python
    "generator_kind",
    "generator_val_fraction",
    "generator_val_seed",
    "direct_mlp_hidden",
    "bag_scale_mode",
    "bag_scale_min",
    "bag_scale_ema_decay",
    "step1_artifacts_subdir",
    "step2_results_subdir",
    "generator_embedding_filename",
    "generator_manifest_filename",
    "generator_weights_filename",
    "generator_monitor_filename",
```

This is safe for `SLDLConfig` because `fingerprint()` already uses `getattr(config, name, None)`.

Then let `write_result` carry an optional extra payload, so Step 2 can record
the consumed Step-1 cache fingerprint **inside** `.result.json` instead of in a
separate sidecar (spec §4.4: `.result.json` is the sole cross-run resume state).
Replace `write_result` with:

```python
def write_result(
    results_dir: Path,
    split: str,
    fold: int,
    rows: object,
    fingerprint: str,
    extra: dict | None = None,
) -> None:
    """Atomically write a fold's success result with its run fingerprint.

    Args:
        results_dir: The fold-results directory.
        split: CV split type.
        fold: Fold id.
        rows: The metric rows for this fold.
        fingerprint: Current run fingerprint (see :func:`fingerprint`).
        extra: Optional top-level keys merged into the result JSON alongside
            ``fingerprint`` and ``rows`` (e.g. exp08b's ``cache_fp`` binding the
            Step-1 cache this result consumed). ``fingerprint`` and ``rows`` are
            written last so ``extra`` can never shadow them.
    """
    payload = {**(extra or {}), "fingerprint": fingerprint, "rows": rows}
    atomic_write_json(result_path(results_dir, split, fold), payload)
```

This is backward-compatible: every existing caller omits `extra` and gets the
identical `{"fingerprint", "rows"}` file.

- [ ] **Step 7: Run queue, CLI, and no-collectives tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_queue_cli.py tests/test_no_collectives.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
rtk git add src/sl_dl_model/__main__.py src/sl_dl_model/fold_queue.py src/sl_dl_model/exp08b_queue.py src/sl_dl_model/exp08b_runner.py src/sl_dl_model/exp08b_step1_runner.py src/sl_dl_model/exp08b_step2_runner.py tests/sl_dl_model/test_exp08b_queue_cli.py
rtk git commit -m "feat: add exp08b two-pass queue entrypoints"
```

Expected: commit succeeds.

---

### Task 8: Experiment Configs, Docs, And Verification

**Files:**
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml`
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/direct_mlp.yaml`
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/nn_copy.yaml`
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_bag_only.yaml`
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_distill_only.yaml`
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_ema_scale.yaml`
- Create: `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/README.md`
- Test: config loader smoke with existing tests.

- [ ] **Step 1: Add primary config**

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml`:

```yaml
# exp08b primary: fold-local two-step STATE-adapter response generator.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08b_k562_sl_pair_two_step_state_adapter/default_cv2_cv3
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
generator_kind: state_adapter
generator_val_fraction: 0.2
generator_val_seed: 17
bag_scale_mode: fixed_warmup
bag_scale_min: 0.001
lambda_sl: 0.0
lambda_distill: 1.0
lambda_distill_after_warmup: 1.0
lambda_bag: 1.0
warmup_epochs: 1
max_epochs: 20
early_stop_patience: 5
batch_pairs: 64
lr: 0.0003
max_grad_norm: 1.0
include_coverage_flag: false
fallback_strategy: zero
embedding_method: exp08b_state_adapter_meanstd
```

- [ ] **Step 2: Add §5.2 step-2 control configs (direct-MLP and NN-copy)**

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/direct_mlp.yaml`:

```yaml
# exp08b direct-ESM2-MLP control: STATE forward bypassed.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08b_k562_sl_pair_two_step_state_adapter/direct_mlp_cv2_cv3
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
generator_kind: direct_mlp
direct_mlp_hidden: 512
generator_val_fraction: 0.2
generator_val_seed: 17
bag_scale_mode: fixed_warmup
bag_scale_min: 0.001
lambda_sl: 0.0
lambda_distill: 0.0
lambda_distill_after_warmup: 0.0
lambda_bag: 1.0
warmup_epochs: 1
max_epochs: 20
early_stop_patience: 5
batch_pairs: 64
lr: 0.0003
max_grad_norm: 1.0
include_coverage_flag: false
fallback_strategy: zero
embedding_method: exp08b_direct_mlp_meanstd
```

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/nn_copy.yaml`:

```yaml
# exp08b §5.2 NN-copy step-2 rung: cache ê_g = MeanStdPool of the ESM2-nearest
# fold-train-covered gene's real gwps bag; no generator is trained. STATE
# checkpoint is referenced only to satisfy config contracts; train-generator
# returns early for generator_kind == nn_copy and never loads it.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08b_k562_sl_pair_two_step_state_adapter/nn_copy_cv2_cv3
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
generator_kind: nn_copy
generator_val_fraction: 0.2
generator_val_seed: 17
bag_scale_mode: fixed_warmup
bag_scale_min: 0.001
lambda_sl: 0.0
lambda_distill: 0.0
lambda_distill_after_warmup: 0.0
lambda_bag: 0.0
warmup_epochs: 1
max_epochs: 20
early_stop_patience: 5
batch_pairs: 64
lr: 0.0003
max_grad_norm: 1.0
include_coverage_flag: false
fallback_strategy: zero
embedding_method: exp08b_nn_copy_meanstd
```

- [ ] **Step 3: Add ablation configs**

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_bag_only.yaml`:

```yaml
# exp08b ablation: bag objective only.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_bag_only_cv2_cv3
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
generator_kind: state_adapter
generator_val_fraction: 0.2
generator_val_seed: 17
bag_scale_mode: fixed_warmup
bag_scale_min: 0.001
lambda_sl: 0.0
lambda_distill: 0.0
lambda_distill_after_warmup: 0.0
lambda_bag: 1.0
warmup_epochs: 1
max_epochs: 20
batch_pairs: 64
lr: 0.0003
max_grad_norm: 1.0
include_coverage_flag: false
fallback_strategy: zero
embedding_method: exp08b_state_adapter_bag_only_meanstd
```

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_distill_only.yaml`:

```yaml
# exp08b ablation: distill anchor only.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_distill_only_cv2_cv3
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
generator_kind: state_adapter
generator_val_fraction: 0.2
generator_val_seed: 17
bag_scale_mode: fixed_warmup
bag_scale_min: 0.001
lambda_sl: 0.0
lambda_distill: 1.0
lambda_distill_after_warmup: 1.0
lambda_bag: 0.0
warmup_epochs: 1
max_epochs: 20
batch_pairs: 64
lr: 0.0003
max_grad_norm: 1.0
include_coverage_flag: false
fallback_strategy: zero
embedding_method: exp08b_state_adapter_distill_only_meanstd
```

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_ema_scale.yaml`:

```yaml
# exp08b ablation: EMA-normalized bag scale.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08b_k562_sl_pair_two_step_state_adapter/ablation_ema_scale_cv2_cv3
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
state_checkpoint: model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
generator_kind: state_adapter
generator_val_fraction: 0.2
generator_val_seed: 17
bag_scale_mode: ema
bag_scale_min: 0.001
bag_scale_ema_decay: 0.95
lambda_sl: 0.0
lambda_distill: 1.0
lambda_distill_after_warmup: 1.0
lambda_bag: 1.0
warmup_epochs: 1
max_epochs: 20
batch_pairs: 64
lr: 0.0003
max_grad_norm: 1.0
include_coverage_flag: false
fallback_strategy: zero
embedding_method: exp08b_state_adapter_ema_scale_meanstd
```

- [ ] **Step 4: Document exact commands**

Create `configs/experiments/08b_k562_sl_pair_two_step_state_adapter/README.md`:

````markdown
# exp08b Two-Step STATE-Adapter Configs

Run Step 1 first:

```bash
rtk uv run python -m sl_dl_model train-generator --config configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml
```

Then run Step 2:

```bash
rtk uv run python -m sl_dl_model train-sl-head --config configs/experiments/08b_k562_sl_pair_two_step_state_adapter/default.yaml
```

Step 1 writes fold-local generator artifacts under:

```text
results/experiments/08b_k562_sl_pair_two_step_state_adapter/default_cv2_cv3/step1_generator/CV*_fold*/
```

Step 2 writes official metric artifacts under:

```text
results/experiments/08b_k562_sl_pair_two_step_state_adapter/default_cv2_cv3/step2_sl_head/
```

Use `direct_mlp.yaml` and `nn_copy.yaml` for the two §5.2 step-2 control rungs
(both flow through the identical `train-generator` → `train-sl-head` passes;
`nn_copy.yaml` skips generator training and caches the ESM2-nearest
train-covered real bag). Use `ablation_bag_only.yaml`,
`ablation_distill_only.yaml`, and `ablation_ema_scale.yaml` for attribution runs.
````

- [ ] **Step 5: Load every config in a smoke test**

Append to `tests/sl_dl_model/test_exp08b_config.py`:

```python
def test_exp08b_repo_configs_load() -> None:
    root = Path("configs/experiments/08b_k562_sl_pair_two_step_state_adapter")
    for path in sorted(root.glob("*.yaml")):
        cfg = load_exp08b_config(path)
        assert cfg.input_csv.name.endswith(".csv")
        assert cfg.output_dir.parts[-2] == "08b_k562_sl_pair_two_step_state_adapter"
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_config.py tests/sl_dl_model/test_exp08b_generator.py tests/sl_dl_model/test_exp08b_monitor.py tests/sl_dl_model/test_exp08b_sl_head.py tests/sl_dl_model/test_exp08b_queue_cli.py tests/test_no_collectives.py -q
```

Expected: PASS.

- [ ] **Step 7: Run lint and full tests**

Run:

```bash
rtk uv run ruff check src/sl_dl_model tests/sl_dl_model tests/test_no_collectives.py
rtk uv run python -m pytest tests/sl_dl_model tests/test_no_collectives.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
rtk git add configs/experiments/08b_k562_sl_pair_two_step_state_adapter tests/sl_dl_model/test_exp08b_config.py
rtk git commit -m "docs: add exp08b configs and runbook"
```

Expected: commit succeeds.

---

## Final Verification

Run the focused exp08b suite:

```bash
rtk uv run python -m pytest tests/sl_dl_model/test_exp08b_config.py tests/sl_dl_model/test_exp08b_generator.py tests/sl_dl_model/test_exp08b_monitor.py tests/sl_dl_model/test_exp08b_sl_head.py tests/sl_dl_model/test_exp08b_queue_cli.py tests/test_no_collectives.py -q
```

Run the package lint:

```bash
rtk uv run ruff check src/sl_dl_model tests/sl_dl_model tests/test_no_collectives.py
```

Run the existing sl_dl_model suite:

```bash
rtk uv run python -m pytest tests/sl_dl_model tests/test_no_collectives.py -q
```

Do not launch the real STATE checkpoint experiment locally as part of this implementation plan. The real run is a GPU/HPC experiment; local success criteria are the unit tests, no-collectives guard, config load smoke, and artifact path contracts.

## Self-Review

Spec coverage:

- Two decoupled steps: Tasks 3, 6, and 7 create separate generator and SL-head entrypoints with disk-only communication.
- Fold-local Step 1 leakage rule: Task 2 restricts generator bag sets to `fold_train_genes & gwps_covered`; Task 7 passes fold-local train symbols.
- Device placement contract: Tasks 3, 6, and 7 use `PartialState().device` or pass `state.device`; no trainer chooses `cuda:0` by probing `torch.cuda.is_available()`.
- Fixed warmup and EMA bag scale: Task 2 implements both `FixedWarmupBagScale` and `EmaBagScale`; Task 3 routes through `build_bag_scale`; Task 8 includes the EMA ablation config.
- Distill anchor full weight and independent symbol set: Task 1 config defaults keep `lambda_distill` and `lambda_distill_after_warmup` equal to 1.0; Task 3 computes `distill_symbols = fold_train & pert_vocab`, independent of the bag-validation holdout.
- Pert-vocab loader boundary: Task 2 creates `sl_dl_model.pert_vocab` and updates `train.py` to use a backward-compatible alias, so exp08b does not import the old end-to-end trainer.
- Generator artifact contract: Tasks 1 and 3 include `generator_weights.pt` alongside the cached embedding table and manifest.
- Step 1 monitor metrics and NN-copy baseline (§4.3): Task 4 implements pooled cosine, MSE, L2, bag energy, and the ESM2 nearest-neighbor-copy monitor baseline.
- §5.2 baseline ladder — both controls run through the full Step 2 ranking path, not just the §4.3 monitor: Task 5 implements the direct-ESM2-MLP control (`control + MLP(ESM2)` cell bag, `generator_kind="direct_mlp"`) **and** the NN-copy step-2 rung (`generator_kind="nn_copy"`, caches `MeanStdPool` of the nearest train-covered real bag for every universe gene). Both emit the identical embedding-cache + manifest artifact contract and flow through the same `train-sl-head` pass; Task 8 adds `direct_mlp.yaml` and `nn_copy.yaml`. exp06 floor and exp07 ceiling are reported by their own harnesses.
- Step 2 official metric reuse: Task 6 exposes a `score_matrix` producer, and Task 7 calls `run_fold_with_producer`.
- Import-level Step 2 separation: Task 6 includes an AST test forbidding generator and checkpoint imports.
- Filesystem queue and no collectives: Task 7 uses `fold_queue` wrappers, quarantines failed folds with `.failed` markers, continues the queue pass, and reruns `tests/test_no_collectives.py`.
- Config ladder and runbook: Task 8 adds primary, direct-MLP, NN-copy, bag-only, distill-only, and EMA-scale configs plus exact commands.

Resolved review findings:

- **Distill-only warmup-boundary crash (HIGH):** Task 3 `train_fold` guards the in-loop `FixedWarmupBagScale.finalize()` with `lambda_bag > 0 and train_bag` so the `lambda_bag=0` ablation (and empty-`train_bag` folds) fall through to the post-loop guard that defaults `scale.value = 1.0`. Regression covered by `test_step1_distill_only_does_not_crash_at_warmup_boundary` (Task 3).
- **§5.2 NN-copy step-2 rung (MEDIUM):** previously NN-copy existed only as the §4.3 monitor; Task 5 now adds `generator_kind="nn_copy"` (a non-trainable Step 1 producer caching `MeanStdPool(nearest train-covered real bag)`) so the §5.2 ladder rung gets the same Step 2 ranking treatment as direct-ESM2-MLP. Config `nn_copy.yaml` added in Task 8.
- **`_assemble(shared=None)` (verify, no change):** confirmed runtime-safe — `_write_assembly_artifacts` → `_gwps_coverage_count` returns `None` when `shared is None`, so Step 2 assembly with no caches just records a null gwps-coverage count in the manifest.
- **Distill silently disappears when the STATE vocab sidecar is missing (P1):** Task 2 `pert_vocab.load_pert_vocab` returns `None` (not `{}`) when `pert_onehot_map.pt` is absent, preserving the exp08 `StateDlProducer._ensure_pert_vocab` raise-on-missing contract (test `test_load_pert_vocab_returns_none_when_sidecar_missing`). Task 3 `Step1GeneratorTrainer._load_distill_vocab` mirrors that contract: a real STATE backend with `lambda_distill` or `lambda_distill_after_warmup` > 0 and a missing/unreadable vocab raises `RuntimeError` rather than degrading to bag-only (test `test_distill_required_but_missing_vocab_raises`); `linear_mock` and zero-distill runs return `{}`.
- **Step 2 resume fingerprint ignores the Step 1 cache it consumes (P1):** Task 7 `exp08b_queue.step2_fold_fingerprint` folds the consumed `predicted_embeddings.npz` + `generator_manifest.json` `(path, size, mtime_ns)` signatures into a per-fold fingerprint. That `cache_fp` is persisted **inside the fold's `.result.json`** via `fq.write_result(..., extra={"cache_fp": ...})` (read back by `read_step2_result_cache_fp`), keeping `.result.json` the sole result-side resume artifact (spec §4.4) rather than adding a separate sidecar. `run_train_sl_head` skips a fold only when the config fingerprint matches **and** the recorded `cache_fp` matches the current cache fingerprint, so a Step 1 regeneration under the same config/`output_dir` busts stale `.result.json` reuse. Tests: `test_step2_fold_fingerprint_changes_when_step1_cache_rewritten`, `test_step2_result_cache_fp_round_trips_inside_result_json`.
- **Step 2 metric rows all labeled `state_dl` (P2):** Task 1 adds `metric_model_name_for(generator_kind)` (`state_adapter→exp08b`, `direct_mlp→direct_esm2_mlp`, `nn_copy→nn_copy`); Task 6 makes `run_fold_with_producer` read an optional `producer.metric_model_name` (defaulting to `state_dl` so the exp08 path is unchanged) and label both the `full_universe` and `covered_pairs` slices with it. Test `test_run_fold_with_producer_labels_rows_by_metric_model_name` asserts the three rungs land as distinct rows.
- **Step 2 producer holds the full config including the STATE checkpoint (P2):** Task 1 adds a slim frozen `SlHeadConfig` (pair-head, scoring, optimization fields only — no `state_checkpoint`, `esm2_npz`, `gwps_h5ad`, or generator field) with `from_exp08b`. Task 6 `CachedEmbeddingPairHeadProducer` takes `SlHeadConfig`, never `Exp08bConfig`. The import-separation AST test forbids `Exp08bConfig` and the literals `esm2_npz` / `gwps_h5ad`.
- **Step 2 runner/fingerprint still carried the full `Exp08bConfig` through scoring + assembly (P2, second pass):** Task 7 adds `exp08b_queue.step2_metric_config`, which projects the generator-only inputs (`state_checkpoint`, `esm2_npz`, `gwps_h5ad`, `gwps_overlap_csv`, `bags_npz`, `state_backend`) to neutral sentinels and redirects `output_dir` to the Step-2 subtree. `run_train_sl_head` uses this neutralized config for the result/failed fingerprint, `_assemble`, and `run_fold_with_producer` — so `fq.fingerprint` (and the embedded `step2_fold_fingerprint` config-portion) depend only on SL/scoring/output fields + the input CSV, never STATE/ESM2/GWPS paths or sidecars. The full `config` is used solely to resolve the Step-1 artifact *paths* the fold consumes. Test `test_step2_metric_config_is_state_neutral` asserts two configs differing only in generator inputs share one fingerprint.
- **Step 1 entrypoint leaked Step 2 / label-aware imports at module load (P2, second pass):** the single `exp08b_runner.py` is split into three modules — `exp08b_runner.py` (shared, label-free `jobs` / `raise_if_step_incomplete`; imports neither stack), `exp08b_step1_runner.py` (imports the generator/STATE stack, never the pair head, scoring, `SlHeadConfig`, or `sl_label`), and `exp08b_step2_runner.py` (imports the pair head + scoring, never the generator/STATE). The CLI lazy-imports only the matching step-runner inside each command branch, so the `train-generator` import path never pulls in Step 2 and vice versa. Guards: `test_step1_runner_does_not_import_pair_head_or_sl_label` and `test_step2_runner_does_not_import_generator_or_state` scan the actual entrypoint modules (not just `exp08b_generator.py`).
- **Step 2 cache fingerprinting did not apply to `.failed` markers (P3):** Step-2 failed markers now carry a `cache_fp` field (the `step2_fold_fingerprint` current at failure). `run_train_sl_head` honors a quarantine only when the recorded `cache_fp` matches the current one (`read_step2_failed_cache_fp`), so a fold quarantined by a premature `train-sl-head` run (missing-cache error) is retried once Step 1 produces the cache and the fingerprint changes. Test `test_step2_stale_failed_marker_ignored_after_cache_appears`.
- **Step-2 fingerprint still carried generator *scalars*, not just paths (P2, third pass):** `step2_metric_config` previously blanked only the STATE/ESM2/GWPS *path* fields, but `fq.fingerprint` also hashes every scalar in `fq._FINGERPRINT_FIELDS` — including the generator knobs (`lambda_bag`, `lambda_distill`, `adapter_hidden`, `pert_dim`, `control_template_size`, `warmup_epochs`, `pooling`, `embedding_method`) and the exp08b `generator_*` / `bag_scale_*` fields added to the fingerprint in Task 7 Step 6. It now canonicalizes **every** field in `fq._FINGERPRINT_FIELDS` ∪ `fq._FINGERPRINT_PATH_FIELDS` that is not in `_STEP2_FINGERPRINT_ALLOWLIST` (fold / scoring / pair-head fields only) to `None`, deriving the canonical set from `fold_queue` so any future generator fingerprint field is auto-neutralized instead of silently leaking. `test_step2_metric_config_is_state_neutral` is unchanged but now also covers scalar neutralization through the shared-fingerprint assertion.
- **Step 2 labeled rows from `config.generator_kind` rather than the Step-1 manifest (P2, third pass):** Task 7 adds `exp08b_queue.step2_metric_model_name(config, split_type, fold_id)`, which reads `generator_kind` from the fold's `generator_manifest.json` and maps it via `metric_model_name_for`. `run_train_sl_head` labels the producer from this manifest-derived name, so a `train-sl-head` invocation whose `config.generator_kind` disagrees with the generator that wrote the cache cannot mislabel exp08b / direct-ESM2-MLP / NN-copy rows. The two steps now exchange the rung identity through the manifest only (spec §7.1). Test `test_step2_metric_model_name_reads_generator_kind_from_manifest`.
- **`.cache_fp.json` sidecar diverged from the single-`.result.json` resume contract (P3, second pass):** the separate cache-fingerprint sidecar is removed. `fq.write_result` gains an optional `extra` mapping (backward-compatible: existing callers get the identical `{"fingerprint", "rows"}` file), and Step 2 persists the consumed `cache_fp` *inside* the fold's `.result.json` via `extra={"cache_fp": ...}`, read back by `read_step2_result_cache_fp`. `.result.json` is once again the sole result-side cross-run resume artifact (spec §4.4). The Step-1-cache-change resume gate is unchanged in behavior; only its storage location moved. Test `test_step2_result_cache_fp_round_trips_inside_result_json`.

Placeholder scan:

- No red-flag placeholder markers, no open-ended test requests, and no code ellipses are used.

Type consistency:

- `Exp08bConfig`, `Step1GeneratorTrainer`, `DirectMlpBagGenerator`, `CachedEmbeddingPairHeadProducer`, `embedding_cache_path`, `generator_manifest_path`, and `generator_weights_path` are introduced before use in later tasks.
- `generator_kind` takes exactly three values — `state_adapter`, `direct_mlp`, `nn_copy` — used consistently across Task 1 config, Task 3/5 `train_fold` branching, and the Task 8 configs.
- `_produce_nn_copy_fold` (Task 5) reuses `nearest_neighbor_copy_predictions` and `_mean_std_pool_np` from Task 4 rather than duplicating NN logic, and returns the same `Step1TrainResult` type as the trained generators.
- The cached embedding NPZ keys are consistently `symbols`, `embeddings`, `coverage_mask`, and `embedding_method`.
- Queue code consistently uses step names `generator` and `sl_head`.

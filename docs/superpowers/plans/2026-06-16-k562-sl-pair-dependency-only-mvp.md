# K562 SL Pair Dependency-Only MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `sl_benchmark_baseline` module that runs three dependency-only baselines (A symmetric logistic regression, B XGBoost, C preferential-attachment frequency probe) over the CV1 K562 SL-pair benchmark and emits per-fold classification and ranking metrics.

**Architecture:** A lean package under `src/sl_benchmark_baseline/` separate from `dependency_baseline` (it predicts the per-pair label `D`, not the per-gene scalar `C`). Each fold slices train/test from `data/k562_SL_benchmark_minimal.csv`, builds 5 symmetric features of the two genes' DepMap GeneEffect scalars (standardizer fit on train only), fits the three models, and computes AUROC/AUPR/F1 plus NDCG/Recall/Precision@{10,20,50}. Results aggregate to mean ± std across the 5 CV1 folds with a provenance manifest.

**Tech Stack:** Python 3.11+, scikit-learn 1.7.2 (LogisticRegression, metrics, StandardScaler-equivalent done by hand for symmetry), xgboost 2.1.4 (dev group), pandas, numpy. Build via hatchling. Tests via pytest under `uv run`.

---

## Spec Reference

This plan implements [`docs/experiment/06_k562_sl_pair_dependency_only_mvp.md`](../../experiment/06_k562_sl_pair_dependency_only_mvp.md). Read it before starting. MVP scope only: three baselines, CV1 only, no observed-B/AIVC features.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/sl_benchmark_baseline/__init__.py` | Package marker + version. |
| `src/sl_benchmark_baseline/config.py` | `SLBaselineConfig` frozen dataclass; all defaults and hyperparameters centralized. |
| `src/sl_benchmark_baseline/data.py` | Load minimal CSV, schema validation, per-fold train/test slicing. |
| `src/sl_benchmark_baseline/features.py` | 5 symmetric pair features + train-fit `Standardizer`. |
| `src/sl_benchmark_baseline/metrics.py` | Classification + pair-level ranking metrics. |
| `src/sl_benchmark_baseline/models.py` | Models A, B, C with a shared `fit`/`predict_proba` interface and a `FoldData` container. |
| `src/sl_benchmark_baseline/evaluate.py` | Per-fold CV loop, aggregation, manifest, output writing. |
| `src/sl_benchmark_baseline/__main__.py` | Thin CLI entrypoint (`uv run python -m sl_benchmark_baseline`). |
| `tests/test_sl_benchmark_baseline.py` | All unit + integration tests for the module. |
| `pyproject.toml` | Register the new package in the hatch wheel `packages` list. |

**Shared types used across tasks (define once, reuse):**

- `FEATURE_NAMES = ("f_min", "f_max", "f_sum", "f_product", "f_absdiff")` — in `features.py`.
- `REQUIRED_COLUMNS` tuple — in `data.py`.
- `FoldData` dataclass with fields `df: pd.DataFrame`, `features: np.ndarray`, `labels: np.ndarray` — in `models.py`.
- Every model exposes `name: str`, `fit(self, train: FoldData) -> None`, `predict_proba(self, test: FoldData) -> np.ndarray` returning scores in `[0, 1]`.

---

## Task 0: Scaffold package and register with build

**Files:**
- Create: `src/sl_benchmark_baseline/__init__.py`
- Modify: `pyproject.toml` (the `[tool.hatch.build.targets.wheel]` `packages` list)
- Test: `tests/test_sl_benchmark_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sl_benchmark_baseline.py`:

```python
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def test_package_imports_and_has_version() -> None:
    import sl_benchmark_baseline

    assert isinstance(sl_benchmark_baseline.__version__, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_package_imports_and_has_version -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/__init__.py`:

```python
"""Dependency-only baselines for K562 gene-pair synthetic-lethality link prediction."""

__all__ = ["__version__"]

__version__ = "0.1.0"
```

In `pyproject.toml`, change the wheel packages list from:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/dependency_baseline", "src/aivc_model"]
```

to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/dependency_baseline", "src/aivc_model", "src/sl_benchmark_baseline"]
```

Then re-sync so the editable install picks up the new package:

Run: `uv sync`
Expected: completes without error; the project reinstalls.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_package_imports_and_has_version -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/__init__.py pyproject.toml tests/test_sl_benchmark_baseline.py
git commit -m "feat: scaffold sl_benchmark_baseline package"
```

**Expected artifact:** importable `sl_benchmark_baseline` package registered in the hatch wheel build.

---

## Task 1: Config dataclass

**Files:**
- Create: `src/sl_benchmark_baseline/config.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sl_benchmark_baseline.py`:

```python
def test_config_defaults_and_override() -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig

    config = SLBaselineConfig()
    assert config.input_csv == Path("data/k562_SL_benchmark_minimal.csv")
    assert config.folds == (0, 1, 2, 3, 4)
    assert config.ranking_k == (10, 20, 50)
    assert config.seed == 17

    overridden = SLBaselineConfig(seed=99, folds=(0, 1))
    assert overridden.seed == 99
    assert overridden.folds == (0, 1)
    # frozen: mutation raises
    try:
        overridden.seed = 5  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("SLBaselineConfig must be frozen")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_config_defaults_and_override -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.config'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/config.py`:

```python
"""Configuration for the K562 SL-pair dependency-only baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SLBaselineConfig:
    """Defaults and hyperparameters for the SL-pair baseline run.

    Attributes:
        input_csv: Canonical minimal CV1 benchmark CSV.
        output_dir: Run directory for metrics and manifest.
        folds: CV1 fold ids to evaluate.
        ranking_k: Cutoffs for NDCG/Recall/Precision@k.
        seed: Global seed for deterministic model fits.
        logreg_c: Inverse regularization strength for model A.
        logreg_max_iter: Max solver iterations for model A.
        xgb_n_estimators: Number of trees for model B.
        xgb_max_depth: Max tree depth for model B.
        xgb_learning_rate: Learning rate for model B.
    """

    input_csv: Path = Path("data/k562_SL_benchmark_minimal.csv")
    output_dir: Path = Path(
        "results/experiments/06_k562_sl_pair_dependency_only_mvp/run"
    )
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    seed: int = 17
    logreg_c: float = 1.0
    logreg_max_iter: int = 1000
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
```

(`field` import is intentionally unused now but harmless; if ruff flags it, drop the `field` import.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_config_defaults_and_override -v`
Expected: PASS.

- [ ] **Step 5: Refactor check + lint**

Run: `uv run ruff check src/sl_benchmark_baseline/config.py`
Expected: no errors. If `field` is flagged as unused import, remove it: change `from dataclasses import dataclass, field` to `from dataclasses import dataclass`.

- [ ] **Step 6: Commit**

```bash
git add src/sl_benchmark_baseline/config.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add SLBaselineConfig dataclass"
```

**Expected artifact:** frozen `SLBaselineConfig` with centralized defaults.

---

## Task 2: Data loading, validation, and fold slicing

**Files:**
- Create: `src/sl_benchmark_baseline/data.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

First add a shared synthetic-CSV helper at the top of the test file body (after the imports, before the first test). This helper is reused by later tasks:

```python
def _write_synthetic_benchmark(path: Path) -> Path:
    """Write a tiny CV1-shaped benchmark CSV with 2 folds and both classes.

    Each fold has 8 train rows (4 pos, 4 neg) and 6 test rows (3 pos, 3 neg).
    Gene effects are chosen so positives skew more negative (more essential).
    """
    rows = []
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(12)]
    pair_counter = 0
    for fold_id in (0, 1):
        for role, n_each in (("train", 4), ("test", 3)):
            for label in (1, 0):
                for _ in range(n_each):
                    a = genes[rng.integers(0, len(genes))]
                    b = genes[rng.integers(0, len(genes))]
                    while b == a:
                        b = genes[rng.integers(0, len(genes))]
                    # positives more essential (more negative gene effect)
                    base = -1.0 if label == 1 else 0.2
                    ea = base + rng.normal(0, 0.1)
                    eb = base + rng.normal(0, 0.1)
                    rows.append(
                        {
                            "pair_id": f"P{pair_counter}",
                            "fold_id": fold_id,
                            "split_role": role,
                            "sl_label": label,
                            "gene_a_symbol": a,
                            "gene_b_symbol": b,
                            "gene_a_k562_gene_effect": ea,
                            "gene_b_k562_gene_effect": eb,
                        }
                    )
                    pair_counter += 1
    frame = pd.DataFrame(rows)
    csv_path = path / "synthetic_sl.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path
```

Then append the test:

```python
def test_load_benchmark_validates_and_fold_split(tmp_path: Path) -> None:
    from sl_benchmark_baseline.data import (
        REQUIRED_COLUMNS,
        fold_split,
        load_benchmark,
    )

    csv_path = _write_synthetic_benchmark(tmp_path)
    df = load_benchmark(csv_path)
    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert set(df["sl_label"].unique()) == {0, 1}

    train_df, test_df = fold_split(df, fold_id=0)
    assert set(train_df["split_role"].unique()) == {"train"}
    assert set(test_df["split_role"].unique()) == {"test"}
    assert (train_df["fold_id"] == 0).all()
    assert (test_df["fold_id"] == 0).all()


def test_load_benchmark_rejects_missing_column(tmp_path: Path) -> None:
    from sl_benchmark_baseline.data import load_benchmark

    bad = pd.DataFrame({"pair_id": ["P0"], "fold_id": [0]})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    try:
        load_benchmark(bad_path)
    except ValueError as error:
        assert "missing" in str(error).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing columns")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "load_benchmark or fold_split" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.data'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/data.py`:

```python
"""Load and slice the K562 SL-pair benchmark CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "fold_id",
    "split_role",
    "sl_label",
    "gene_a_symbol",
    "gene_b_symbol",
    "gene_a_k562_gene_effect",
    "gene_b_k562_gene_effect",
)


def load_benchmark(path: Path) -> pd.DataFrame:
    """Load the minimal benchmark CSV and validate its schema.

    Args:
        path: Path to ``k562_SL_benchmark_minimal.csv`` or a compatible CSV.

    Returns:
        The validated benchmark DataFrame.

    Raises:
        ValueError: If required columns are missing, labels are not binary,
            split roles are unexpected, or gene-effect values contain NaN.
    """
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"benchmark CSV missing columns: {missing}")
    labels = set(frame["sl_label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"sl_label must be in {{0, 1}}, got {sorted(labels)}")
    roles = set(frame["split_role"].unique())
    if not roles.issubset({"train", "test"}):
        raise ValueError(f"split_role must be train/test, got {sorted(roles)}")
    effect_cols = ["gene_a_k562_gene_effect", "gene_b_k562_gene_effect"]
    if frame[effect_cols].isna().any().any():
        raise ValueError("gene-effect columns must not contain NaN")
    return frame


def fold_split(
    frame: pd.DataFrame, fold_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice train and test rows for one CV1 fold.

    Args:
        frame: Validated benchmark DataFrame.
        fold_id: CV1 fold id to extract.

    Returns:
        A ``(train_df, test_df)`` tuple, each reset-indexed.
    """
    fold = frame[frame["fold_id"] == fold_id]
    train_df = fold[fold["split_role"] == "train"].reset_index(drop=True)
    test_df = fold[fold["split_role"] == "test"].reset_index(drop=True)
    return train_df, test_df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "load_benchmark or fold_split" -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/data.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add benchmark loader, schema validation, and fold slicing"
```

**Expected artifact:** `load_benchmark` + `fold_split` with strict schema validation.

---

## Task 3: Symmetric features and train-fit standardizer

**Files:**
- Create: `src/sl_benchmark_baseline/features.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_pair_features_are_swap_invariant() -> None:
    from sl_benchmark_baseline.features import FEATURE_NAMES, build_pair_features

    ea = np.array([-1.0, 0.5])
    eb = np.array([0.2, -0.3])
    forward = build_pair_features(ea, eb)
    swapped = build_pair_features(eb, ea)
    assert forward.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(forward, swapped)
    # spot check first row: min, max, sum, product, absdiff
    np.testing.assert_allclose(
        forward[0], [-1.0, 0.2, -0.8, -0.2, 1.2], rtol=1e-6
    )


def test_standardizer_fits_on_train_only() -> None:
    from sl_benchmark_baseline.features import Standardizer

    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    standardizer = Standardizer.fit(train)
    transformed = standardizer.transform(train)
    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(transformed.std(axis=0), [1.0, 1.0], atol=1e-9)
    # constant column does not divide by zero
    const = np.array([[5.0], [5.0]])
    const_std = Standardizer.fit(const).transform(const)
    np.testing.assert_allclose(const_std, [[0.0], [0.0]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "pair_features or standardizer" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.features'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/features.py`:

```python
"""Symmetric pair features and a train-fit standardizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "f_min",
    "f_max",
    "f_sum",
    "f_product",
    "f_absdiff",
)


def build_pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Build swap-invariant features from two gene-effect vectors.

    Args:
        ea: Gene-effect values for gene a, shape ``(n,)``.
        eb: Gene-effect values for gene b, shape ``(n,)``.

    Returns:
        Feature matrix of shape ``(n, 5)`` ordered by ``FEATURE_NAMES``.
    """
    ea = np.asarray(ea, dtype=float)
    eb = np.asarray(eb, dtype=float)
    return np.column_stack(
        [
            np.minimum(ea, eb),
            np.maximum(ea, eb),
            ea + eb,
            ea * eb,
            np.abs(ea - eb),
        ]
    )


@dataclass(frozen=True)
class Standardizer:
    """Zero-mean unit-variance standardizer fit on training data only."""

    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        """Fit per-column mean and std; zero-std columns map to std 1.0."""
        features = np.asarray(features, dtype=float)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return cls(mean_=mean, std_=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization to a feature matrix."""
        features = np.asarray(features, dtype=float)
        return (features - self.mean_) / self.std_
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "pair_features or standardizer" -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/features.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add symmetric pair features and train-fit standardizer"
```

**Expected artifact:** `build_pair_features` (5 symmetric columns) + `Standardizer`.

---

## Task 4: Classification metrics

**Files:**
- Create: `src/sl_benchmark_baseline/metrics.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_classification_metrics_perfect_and_keys() -> None:
    from sl_benchmark_baseline.metrics import classification_metrics

    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    out = classification_metrics(y_true, scores)
    assert set(out) == {"auroc", "aupr", "f1@0.5"}
    assert out["auroc"] == 1.0
    assert out["aupr"] == 1.0
    assert out["f1@0.5"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_classification_metrics_perfect_and_keys -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/metrics.py`:

```python
"""Classification and pair-level ranking metrics for SL-pair scoring."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Compute AUROC, AUPR, and F1 at threshold 0.5.

    Args:
        y_true: Binary labels, shape ``(n,)``.
        scores: Predicted probabilities in ``[0, 1]``, shape ``(n,)``.

    Returns:
        Mapping with keys ``auroc``, ``aupr``, ``f1@0.5``.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "aupr": float(average_precision_score(y_true, scores)),
        "f1@0.5": float(f1_score(y_true, preds, zero_division=0)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_classification_metrics_perfect_and_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/metrics.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add classification metrics (auroc, aupr, f1)"
```

**Expected artifact:** `classification_metrics` returning the three classification scores.

---

## Task 5: Pair-level ranking metrics with deterministic tie-breaking

**Files:**
- Modify: `src/sl_benchmark_baseline/metrics.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_ranking_metrics_keys_and_topk() -> None:
    from sl_benchmark_baseline.metrics import ranking_metrics

    # 5 items, 2 positives ranked at top by score
    y_true = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    pair_ids = ["P0", "P1", "P2", "P3", "P4"]
    out = ranking_metrics(y_true, scores, pair_ids, ks=(2, 5))
    assert "ndcg@2" in out and "recall@2" in out and "precision@2" in out
    # top-2 are both positives
    assert out["precision@2"] == 1.0
    assert out["recall@2"] == 1.0
    assert out["ndcg@2"] == 1.0
    # at k=5, recall captures all positives
    assert out["recall@5"] == 1.0
    assert out["precision@5"] == 2 / 5


def test_ranking_metrics_breaks_ties_by_pair_id() -> None:
    from sl_benchmark_baseline.metrics import ranking_metrics

    # all scores tied; positive has the lexicographically smaller pair_id,
    # so deterministic tie-break by pair_id puts it first.
    y_true = np.array([1, 0, 0])
    scores = np.array([0.5, 0.5, 0.5])
    pair_ids = ["A", "B", "C"]
    out = ranking_metrics(y_true, scores, pair_ids, ks=(1,))
    assert out["precision@1"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "ranking_metrics" -v`
Expected: FAIL with `ImportError: cannot import name 'ranking_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/sl_benchmark_baseline/metrics.py`:

```python
def _ranked_relevance(
    y_true: np.ndarray, scores: np.ndarray, pair_ids: list[str]
) -> list[int]:
    """Order items by descending score, breaking ties by ascending pair_id."""
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), str(pair_ids[i])),
    )
    return [int(y_true[i]) for i in order]


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    pair_ids: list[str],
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Compute pair-level NDCG/Recall/Precision@k over the flat test list.

    Items are ranked by descending score with ties broken deterministically by
    ascending ``pair_id``. Positives are the relevant items.

    Args:
        y_true: Binary labels, shape ``(n,)``.
        scores: Predicted scores (any monotonic scale), shape ``(n,)``.
        pair_ids: Stable identifiers used for tie-breaking, length ``n``.
        ks: Rank cutoffs.

    Returns:
        Mapping with ``ndcg@k``, ``recall@k``, ``precision@k`` for each ``k``.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    sorted_rel = _ranked_relevance(y_true, scores, pair_ids)
    total_pos = int(sum(sorted_rel))
    ideal_rel = sorted(sorted_rel, reverse=True)
    out: dict[str, float] = {}
    for k in ks:
        topk = sorted_rel[:k]
        hits = sum(topk)
        out[f"precision@{k}"] = hits / k if k > 0 else 0.0
        out[f"recall@{k}"] = hits / total_pos if total_pos > 0 else 0.0
        dcg = sum(
            rel / math.log2(rank + 2) for rank, rel in enumerate(topk)
        )
        idcg = sum(
            rel / math.log2(rank + 2)
            for rank, rel in enumerate(ideal_rel[:k])
        )
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "ranking_metrics" -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/metrics.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add pair-level ranking metrics with deterministic ties"
```

**Expected artifact:** `ranking_metrics` producing NDCG/Recall/Precision@{ks}.

---

## Task 6: Models A, B, C with shared interface

**Files:**
- Create: `src/sl_benchmark_baseline/models.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def _make_fold_data(labels: np.ndarray, features: np.ndarray, df: pd.DataFrame):
    from sl_benchmark_baseline.models import FoldData

    return FoldData(df=df, features=features, labels=labels)


def test_models_emit_probabilities_in_unit_interval() -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.models import build_models

    rng = np.random.default_rng(1)
    n_train, n_test = 40, 12
    # separable: positives have more negative gene effects
    train_labels = np.array([1] * 20 + [0] * 20)
    train_feats = np.where(
        train_labels[:, None] == 1, -1.0, 0.5
    ) + rng.normal(0, 0.05, size=(n_train, 5))
    train_df = pd.DataFrame(
        {
            "pair_id": [f"T{i}" for i in range(n_train)],
            "sl_label": train_labels,
            "gene_a_symbol": ["GA"] * n_train,
            "gene_b_symbol": ["GB"] * n_train,
        }
    )
    test_labels = np.array([1] * 6 + [0] * 6)
    test_feats = np.where(
        test_labels[:, None] == 1, -1.0, 0.5
    ) + rng.normal(0, 0.05, size=(n_test, 5))
    test_df = pd.DataFrame(
        {
            "pair_id": [f"S{i}" for i in range(n_test)],
            "sl_label": test_labels,
            "gene_a_symbol": ["GA"] * n_test,
            "gene_b_symbol": ["GB"] * n_test,
        }
    )
    train = _make_fold_data(train_labels, train_feats, train_df)
    test = _make_fold_data(test_labels, test_feats, test_df)

    models = build_models(SLBaselineConfig())
    assert {m.name for m in models} == {"A", "B", "C"}
    for model in models:
        model.fit(train)
        scores = model.predict_proba(test)
        assert scores.shape == (n_test,)
        assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_frequency_probe_uses_train_positive_degree() -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.models import FoldData, build_models

    # gene HUB appears in many train positives; RARE never does.
    train_df = pd.DataFrame(
        {
            "pair_id": [f"T{i}" for i in range(4)],
            "sl_label": [1, 1, 1, 0],
            "gene_a_symbol": ["HUB", "HUB", "HUB", "RARE"],
            "gene_b_symbol": ["X1", "X2", "X3", "X4"],
        }
    )
    test_df = pd.DataFrame(
        {
            "pair_id": ["S0", "S1"],
            "sl_label": [1, 0],
            "gene_a_symbol": ["HUB", "RARE"],
            "gene_b_symbol": ["HUB", "RARE"],
        }
    )
    dummy_feats_train = np.zeros((4, 5))
    dummy_feats_test = np.zeros((2, 5))
    train = FoldData(df=train_df, features=dummy_feats_train, labels=np.array([1, 1, 1, 0]))
    test = FoldData(df=test_df, features=dummy_feats_test, labels=np.array([1, 0]))

    probe = next(m for m in build_models(SLBaselineConfig()) if m.name == "C")
    probe.fit(train)
    scores = probe.predict_proba(test)
    # HUB-HUB pair (high degree product) scores strictly above RARE-RARE (degree 0)
    assert scores[0] > scores[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "models_emit or frequency_probe" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.models'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/models.py`:

```python
"""Baseline models A (logreg), B (xgboost), C (frequency probe)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sl_benchmark_baseline.config import SLBaselineConfig


@dataclass
class FoldData:
    """Per-fold inputs shared across models.

    Attributes:
        df: Fold rows, including ``pair_id``, ``sl_label``, gene symbols.
        features: Standardized symmetric features, shape ``(n, 5)``.
        labels: Binary labels aligned with ``features``, shape ``(n,)``.
    """

    df: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray


class LogRegModel:
    """Model A: symmetric logistic regression on the 5 standardized features."""

    name = "A"

    def __init__(self, config: SLBaselineConfig) -> None:
        self._model = LogisticRegression(
            C=config.logreg_c,
            max_iter=config.logreg_max_iter,
            random_state=config.seed,
        )

    def fit(self, train: FoldData) -> None:
        self._model.fit(train.features, train.labels)

    def predict_proba(self, test: FoldData) -> np.ndarray:
        return self._model.predict_proba(test.features)[:, 1]


class XGBModel:
    """Model B: gradient-boosted trees on the same 5 features."""

    name = "B"

    def __init__(self, config: SLBaselineConfig) -> None:
        from xgboost import XGBClassifier

        self._model = XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.seed,
            eval_metric="logloss",
        )

    def fit(self, train: FoldData) -> None:
        self._model.fit(train.features, train.labels)

    def predict_proba(self, test: FoldData) -> np.ndarray:
        return self._model.predict_proba(test.features)[:, 1]


class FrequencyProbeModel:
    """Model C: preferential-attachment probe from train-positive degree.

    Scores a test pair by ``pos_degree[a] * pos_degree[b]`` using only training
    positives, then min-max normalizes the fold's scores into ``[0, 1]`` so the
    output is comparable with probability outputs of A and B. AUROC/AUPR and
    ranking are invariant to this monotonic rescaling; F1@0.5 uses it directly.
    """

    name = "C"

    def __init__(self, config: SLBaselineConfig) -> None:
        self._pos_degree: Counter[str] = Counter()

    def fit(self, train: FoldData) -> None:
        positives = train.df[train.df["sl_label"] == 1]
        self._pos_degree = Counter()
        for symbol in positives["gene_a_symbol"]:
            self._pos_degree[str(symbol)] += 1
        for symbol in positives["gene_b_symbol"]:
            self._pos_degree[str(symbol)] += 1

    def predict_proba(self, test: FoldData) -> np.ndarray:
        raw = np.array(
            [
                self._pos_degree[str(a)] * self._pos_degree[str(b)]
                for a, b in zip(
                    test.df["gene_a_symbol"],
                    test.df["gene_b_symbol"],
                    strict=True,
                )
            ],
            dtype=float,
        )
        span = raw.max() - raw.min()
        if span == 0.0:
            return np.zeros_like(raw)
        return (raw - raw.min()) / span


def build_models(config: SLBaselineConfig) -> list:
    """Construct the three baseline models in canonical order A, B, C."""
    return [
        LogRegModel(config),
        XGBModel(config),
        FrequencyProbeModel(config),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -k "models_emit or frequency_probe" -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/models.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add baseline models A/B/C with shared FoldData interface"
```

**Expected artifact:** `build_models` returning A/B/C, each emitting `[0,1]` scores.

---

## Task 7: Per-fold CV loop, aggregation, and outputs

**Files:**
- Create: `src/sl_benchmark_baseline/evaluate.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_run_cv_writes_outputs_and_aggregates(tmp_path: Path) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    csv_path = _write_synthetic_benchmark(tmp_path)
    output_dir = tmp_path / "run"
    config = SLBaselineConfig(
        input_csv=csv_path,
        output_dir=output_dir,
        folds=(0, 1),
        ranking_k=(2, 5),
    )
    summary = run_cv(config)

    # outputs exist
    assert (output_dir / "fold_metrics.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "manifest.json").exists()

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["model"].unique()) == {"A", "B", "C"}
    assert set(fold_metrics["fold_id"].unique()) == {0, 1}
    assert {"model", "fold_id", "metric", "value"}.issubset(fold_metrics.columns)
    # classification + ranking metrics present
    metric_names = set(fold_metrics["metric"].unique())
    assert {"auroc", "aupr", "f1@0.5"}.issubset(metric_names)
    assert {"ndcg@2", "recall@2", "precision@2"}.issubset(metric_names)

    # summary has mean and std per (model, metric)
    assert {"model", "metric", "mean", "std"}.issubset(summary.columns)
    assert len(summary) == len(fold_metrics["metric"].unique()) * 3

    # manifest records provenance
    import json

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "input_csv_sha256" in manifest
    assert "leakage_notes" in manifest
    assert "ranking_semantics" in manifest
    assert manifest["seed"] == config.seed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_run_cv_writes_outputs_and_aggregates -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/evaluate.py`:

```python
"""Per-fold CV loop, aggregation, and output writing for the SL baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.data import fold_split, load_benchmark
from sl_benchmark_baseline.features import Standardizer, build_pair_features
from sl_benchmark_baseline.metrics import (
    classification_metrics,
    ranking_metrics,
)
from sl_benchmark_baseline.models import FoldData, build_models

LEAKAGE_NOTES = (
    "GeneEffect(K562, g) as a feature against Rand negatives is low leakage "
    "risk. This becomes high risk under Exp/Dep negative sampling. CV1 is a "
    "pair-level split: results are not held-out-gene generalization."
)
RANKING_SEMANTICS = (
    "Ranking metrics are pair-level over the flat test list; this differs from "
    "the official per-gene-anchor candidate ranking and is not claimed "
    "equivalent. Ties are broken by pair_id."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _build_fold_data(frame: pd.DataFrame, standardizer: Standardizer) -> FoldData:
    raw = build_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
    )
    return FoldData(
        df=frame,
        features=standardizer.transform(raw),
        labels=frame["sl_label"].to_numpy(dtype=int),
    )


def run_fold(
    frame: pd.DataFrame, fold_id: int, config: SLBaselineConfig
) -> list[dict[str, object]]:
    """Fit all models on one fold and return long-form metric rows."""
    train_df, test_df = fold_split(frame, fold_id)
    train_raw = build_pair_features(
        train_df["gene_a_k562_gene_effect"].to_numpy(),
        train_df["gene_b_k562_gene_effect"].to_numpy(),
    )
    standardizer = Standardizer.fit(train_raw)
    train = _build_fold_data(train_df, standardizer)
    test = _build_fold_data(test_df, standardizer)
    pair_ids = test_df["pair_id"].astype(str).tolist()

    rows: list[dict[str, object]] = []
    for model in build_models(config):
        model.fit(train)
        scores = model.predict_proba(test)
        metrics = classification_metrics(test.labels, scores)
        metrics.update(
            ranking_metrics(test.labels, scores, pair_ids, config.ranking_k)
        )
        for metric, value in metrics.items():
            rows.append(
                {
                    "model": model.name,
                    "fold_id": fold_id,
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(["model", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary.sort_values(["model", "metric"]).reset_index(drop=True)


def run_cv(config: SLBaselineConfig) -> pd.DataFrame:
    """Run the full CV1 loop, write outputs, and return the summary table."""
    frame = load_benchmark(config.input_csv)
    all_rows: list[dict[str, object]] = []
    for fold_id in config.folds:
        all_rows.extend(run_fold(frame, fold_id, config))
    fold_metrics = pd.DataFrame(all_rows)
    summary = _summarize(fold_metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    manifest = {
        "input_csv": str(config.input_csv),
        "input_csv_sha256": _file_sha256(config.input_csv),
        "folds": list(config.folds),
        "ranking_k": list(config.ranking_k),
        "seed": config.seed,
        "models": ["A", "B", "C"],
        "leakage_notes": LEAKAGE_NOTES,
        "ranking_semantics": RANKING_SEMANTICS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_run_cv_writes_outputs_and_aggregates -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/evaluate.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add per-fold CV loop, aggregation, and manifest outputs"
```

**Expected artifacts:** `fold_metrics.csv`, `summary.csv`, `manifest.json` written under the run dir.

---

## Task 8: CLI entrypoint

**Files:**
- Create: `src/sl_benchmark_baseline/__main__.py`
- Test: `tests/test_sl_benchmark_baseline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_cli_main_runs_and_writes_summary(tmp_path: Path) -> None:
    from sl_benchmark_baseline.__main__ import main

    csv_path = _write_synthetic_benchmark(tmp_path)
    output_dir = tmp_path / "cli_run"
    exit_code = main(
        [
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--folds",
            "0",
            "1",
            "--ranking-k",
            "2",
            "5",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "summary.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_cli_main_runs_and_writes_summary -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.__main__'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/__main__.py`:

```python
"""CLI entrypoint: uv run python -m sl_benchmark_baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.evaluate import run_cv

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K562 SL-pair dependency-only baseline (CV1)."
    )
    defaults = SLBaselineConfig()
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(defaults.folds)
    )
    parser.add_argument(
        "--ranking-k", type=int, nargs="+", default=list(defaults.ranking_k)
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the baseline CV loop from CLI flags. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    config = SLBaselineConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        folds=tuple(args.folds),
        ranking_k=tuple(args.ranking_k),
        seed=args.seed,
    )
    summary = run_cv(config)
    logger.info("Wrote summary with %d rows to %s", len(summary), config.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py::test_cli_main_runs_and_writes_summary -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/__main__.py tests/test_sl_benchmark_baseline.py
git commit -m "feat: add sl_benchmark_baseline CLI entrypoint"
```

**Expected artifact:** runnable `uv run python -m sl_benchmark_baseline`.

---

## Task 9: Full-suite verification and lint

**Files:** none (verification only)

- [ ] **Step 1: Run the full module test suite**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline.py -v`
Expected: all tests PASS.

- [ ] **Step 2: Confirm no regressions elsewhere**

Run: `uv run python -m pytest -q`
Expected: full suite PASS (no regressions from the pyproject packages change).

- [ ] **Step 3: Lint and format**

Run: `uv run ruff check src/sl_benchmark_baseline tests/test_sl_benchmark_baseline.py`
Expected: no errors. Then `uv run ruff format src/sl_benchmark_baseline tests/test_sl_benchmark_baseline.py`.

- [ ] **Step 4: Smoke-run against the real benchmark (optional but recommended)**

Run:
```bash
uv run python -m sl_benchmark_baseline \
  --output-dir results/experiments/06_k562_sl_pair_dependency_only_mvp/run
```
Expected: writes `fold_metrics.csv`, `summary.csv`, `manifest.json` under that dir. Inspect `summary.csv`: chance AUROC ≈ 0.5 baseline, with A/B above chance and C reported as the degree control. (The `experiments/` tree is gitignored per repo convention; do not commit artifacts.)

- [ ] **Step 5: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: lint and format sl_benchmark_baseline"
```

**Expected artifact:** green test suite, clean lint, and a real-data run directory.

---

## Self-Review Notes

- **Spec coverage:** Data contract → Task 2; symmetric features + train-only standardizer → Task 3; models A/B/C → Task 6; classification metrics → Task 4; ranking metrics with `pair_id` tie-break → Task 5; per-fold protocol + aggregation + manifest (sha256, leakage notes, ranking caveat) → Task 7; module layout + `python -m` entrypoint → Tasks 0 and 8; outputs `fold_metrics.csv`/`summary.csv`/`manifest.json` → Task 7. CV1-only, Rand-only, no AIVC features are enforced by consuming only the minimal CSV (no new data sources introduced).
- **Interface consistency:** `FoldData(df, features, labels)` defined in Task 6 and consumed identically in Task 7; every model exposes `name`, `fit(train)`, `predict_proba(test) -> [0,1]`. `FEATURE_NAMES`/`build_pair_features`/`Standardizer` (Task 3) used in Task 7. `classification_metrics`/`ranking_metrics` signatures (Tasks 4–5) match the calls in Task 7.
- **Known judgment calls baked into the plan:** (1) Model C min-max normalizes its degree-product score into `[0,1]` so F1@0.5 is defined and comparable; AUROC/AUPR/ranking are invariant to this monotonic rescale. (2) Constant-score folds (all degree 0) return zeros, deterministically. (3) Ranking is pair-level over the flat test list, explicitly flagged as non-equivalent to the official per-anchor ranking in the manifest. These match the spec's stated semantics.

# exp08 — STATE-Adapter End-to-End DL Model for K562 SL-Pair Ranking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen-STATE + trainable-adapter DL model for K562 SL-pair ranking that beats the exp06 dependency-only baseline on CV2/CV3 official per-anchor NDCG/MAP by adding a transcriptomic-response signal that generalizes to held-out genes.

**Architecture:** Replace STATE's one-hot perturbation encoder with a trainable MLP adapter fed by ESM2 gene embeddings (1280-d); freeze STATE's 8-layer Llama backbone; train a pooling head (bag→e_g) and symmetric pair head (e_a,e_b→P(SL)) on a 3-part loss (token-distill anchor + real-bag supervision + SL BCE). All 9,471 genes flow through one frozen STATE coordinate system; real gwps bags supervise the covered train genes only (leakage-free CV2/CV3). Orchestrated by HuggingFace Accelerate (DDP default) + tqdm. Reuses exp06/07's official metric harness verbatim.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers (ESM2) + Accelerate, sklearn, numpy, pandas, Arc STATE checkpoint (`model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/`), exp06/07 baseline harness (`src/sl_benchmark_baseline/{metrics,data,evaluate}.py`), gwps h5ad (`data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad`).

## Global Constraints

- Python ≥3.11; strict type hints; Google-style docstrings; absolute imports.
- Frozen dataclass config from YAML; no hardcoded paths/thresholds.
- Prefix all Python/pytest/script invocations with `uv run`.
- Use `logging` not `print` in library code.
- Max 50 lines/function, 600 lines/file (split if exceeded).
- Seed 17 for all random operations.
- ESM2 model: `facebook/esm2_t33_650M_UR50D` (1280-d) from `transformers`.
- STATE checkpoint: `model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt`.
- Reuse `sl_benchmark_baseline/{metrics,data,evaluate,embeddings,features,config}.py` — do not reimplement; import verbatim.
- Commit messages: conventional commits format (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`).

---

## File Structure

New package `src/sl_dl_model/` — each file one responsibility:

| File | Responsibility |
| --- | --- |
| `config.py` | `SLDLConfig` frozen dataclass + YAML loader |
| `gene_embeddings.py` | Load precomputed ESM2 `.npz` → `Esm2EmbeddingTable` (symbol→1280-d) |
| `encoder.py` | `PertAdapter` (ESM2→328) + frozen-STATE wrapper → predicted response bag |
| `pooling.py` | `bag → e_g` permutation-invariant pool (mean+std default) |
| `pair_head.py` | `SymmetricPairHead` (e_a,e_b)+GeneEffect → logit; the trainable scorer |
| `losses.py` | 3-part loss assembly (reuses aivc_model losses) |
| `bags.py` | Build per-gene real gwps bags + shared K562 control template from h5ad |
| `model.py` | `SlDlModel`: encoder + pooling + pair head composed end-to-end |
| `train.py` | Accelerate/DDP training loop, tqdm, `StateDlProducer` (per-fold fit → e_g table + score matrix) |
| `evaluate.py` | `EmbeddingProducer` protocol + `run_cv`; writes metrics/manifest |
| `scoring.py` | `run_fold_with_producer`: producer → universe embeddings → official metric rows |
| `__main__.py` | CLI: `run-cv` subcommand (`--producer {zero,state_dl}`) |

Standalone script: `scripts/precompute_esm2_embeddings.py` (UniProt fetch + ESM2).
Config dir: `configs/experiments/08_k562_sl_pair_state_dl/`.
Slurm wrapper: `scripts/sl_dl_model.sh`.
Tests: `tests/sl_dl_model/test_*.py`.

Reused verbatim (imported, not copied): `sl_benchmark_baseline/metrics.py`,
`data.py`, `features.py`, and `evaluate.py`'s `GeneUniverse` / `_build_gene_universe`
/ `_build_augmented_score_matrix` / `_metric_rows` / `_pair_indices`. exp08 produces
a per-gene embedding table consumed exactly like `embeddings.load_gene_embeddings`.

---

## Phase 0 — Harness parity (no STATE, no ESM2)

Goal: stand up `src/sl_dl_model/` with config + CLI that runs the existing exp06/07
scoring harness and reproduces exp06 numbers. This proves the harness wiring before
any DL is added. **Gate:** exp06 CV2/CV3 numbers reproduced within fold noise.

### Task 0.1: Package scaffold + config dataclass

**Files:**
- Create: `src/sl_dl_model/__init__.py`
- Create: `src/sl_dl_model/config.py`
- Test: `tests/sl_dl_model/test_config.py`

**Interfaces:**
- Produces: `SLDLConfig` frozen dataclass; `load_config(path: Path) -> SLDLConfig`.

> **Test layout note:** existing tests are flat files in `tests/` (e.g.
> `tests/test_sl_benchmark_baseline_config.py`), but this plan groups exp08 tests under
> `tests/sl_dl_model/`. Either is fine for pytest (no `__init__.py` needed — the repo's
> `conftest.py` and `src`-layout install make `sl_dl_model` importable once registered).
> The plan uses `tests/sl_dl_model/` for grouping; create the directory on first test.

- [ ] **Step 0: Register the package in pyproject.toml (REQUIRED — editable install)**

Without this the `sl_dl_model` package is not importable. Modify `pyproject.toml:63`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/dependency_baseline", "src/aivc_model", "src/sl_benchmark_baseline", "src/sl_dl_model"]
```

Then re-sync so the editable install picks up the new package:

Run: `uv sync`
Expected: completes without error; `uv run python -c "import sl_dl_model"` works after
Step 3 creates `__init__.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_config.py
from pathlib import Path

import yaml

from sl_dl_model.config import SLDLConfig, load_config


def test_defaults_match_exp06_protocol():
    cfg = SLDLConfig()
    assert cfg.seed == 17
    assert cfg.ranking_k == (10, 20, 50)
    assert cfg.folds == (0, 1, 2, 3, 4)
    assert cfg.esm2_model == "facebook/esm2_t33_650M_UR50D"


def test_load_config_roundtrip(tmp_path: Path):
    payload = {
        "input_csv": "data/x.csv",
        "output_dir": "results/exp08/run",
        "split_types": ["CV2", "CV3"],
        "esm2_npz": "data/esm2.npz",
        "warmup_epochs": 3,
        "lambda_sl": 1.0,
        "lambda_distill": 0.5,
        "lambda_bag": 1.0,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    cfg = load_config(path)
    assert cfg.split_types == ("CV2", "CV3")
    assert cfg.esm2_npz == Path("data/esm2.npz")
    assert cfg.lambda_distill == 0.5


def test_load_config_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"nonsense_key": 1}))
    try:
        load_config(path)
    except ValueError as exc:
        assert "unknown config keys" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/__init__.py
"""STATE-adapter DL model for K562 SL-pair ranking (exp08)."""
```

```python
# src/sl_dl_model/config.py
"""Configuration for the exp08 STATE-adapter DL SL-pair model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SLDLConfig:
    """Defaults and hyperparameters for the exp08 DL run."""

    input_csv: Path = Path(
        "data/SL_benchmark/derived/k562_depmap_rand_1to1/"
        "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv"
    )
    output_dir: Path = Path("results/experiments/08_k562_sl_pair_state_dl/run")
    split_types: tuple[str, ...] | None = None
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ranking_k: tuple[int, ...] = (10, 20, 50)
    seed: int = 17

    # Gene embedding (ESM2)
    esm2_model: str = "facebook/esm2_t33_650M_UR50D"
    esm2_npz: Path | None = None
    fallback_strategy: str = "zero"
    include_coverage_flag: bool = True

    # STATE encoder
    state_checkpoint: Path = Path(
        "model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt"
    )
    state_backend: str = "state_checkpoint"
    gwps_h5ad: Path = Path(
        "data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad"
    )
    gwps_overlap_csv: Path = Path(
        "data/sl_dependency_v0/interim/k562_replogle_depmap_overlap.csv"
    )
    bags_npz: Path | None = None
    control_template_size: int = 256
    cells_per_bag: int = 256

    # Adapter / pooling / pair head
    pert_dim: int = 328
    adapter_hidden: int = 512
    pooling: str = "mean_std"
    pair_hidden: tuple[int, ...] = (256, 64)

    # Loss weights + schedule
    lambda_sl: float = 1.0
    lambda_distill: float = 0.5
    lambda_distill_after_warmup: float = 0.1
    lambda_bag: float = 1.0
    lambda_rank: float = 0.0
    warmup_epochs: int = 3
    max_epochs: int = 20
    batch_pairs: int = 1024
    lr: float = 1e-3

    embedding_method: str = "state_adapter_esm2_meanstd"

    @property
    def augmented(self) -> bool:
        """exp08 always runs the augmented (transcript) scoring path."""
        return True


_PATH_FIELDS = {
    "input_csv", "output_dir", "esm2_npz", "state_checkpoint",
    "gwps_h5ad", "gwps_overlap_csv", "bags_npz",
}
_TUPLE_FIELDS = {"split_types", "folds", "ranking_k", "pair_hidden"}


def load_config(path: Path) -> SLDLConfig:
    """Load an :class:`SLDLConfig` from YAML, coercing paths and tuples."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(SLDLConfig)}
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
    return SLDLConfig(**kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/__init__.py src/sl_dl_model/config.py tests/sl_dl_model/test_config.py pyproject.toml
git commit -m "feat: add exp08 sl_dl_model package scaffold and config"
```

### Task 0.2: Embedding-table protocol + exp06-parity CV runner

**Files:**
- Create: `src/sl_dl_model/evaluate.py`
- Test: `tests/sl_dl_model/test_evaluate_parity.py`

**Interfaces:**
- Consumes: `SLDLConfig` (Task 0.1); `sl_benchmark_baseline.evaluate._build_gene_universe`,
  `_build_augmented_score_matrix`, `_metric_rows`, `_pair_indices`, `_covered_pair_mask`,
  `_summarize`; `sl_benchmark_baseline.features.{build_augmented_pair_features, Standardizer}`;
  `sl_benchmark_baseline.data.{load_benchmark, fold_split}`;
  `sl_benchmark_baseline.embeddings.GeneEmbeddingTable`.
- Produces:
  - `EmbeddingProducer` protocol: `produce(symbols: np.ndarray, train_symbols: set[str]) -> tuple[np.ndarray, np.ndarray]` returning `(embeddings (n_gene, dim), coverage_mask (n_gene,))`.
  - `run_cv(config: SLDLConfig, producer: EmbeddingProducer) -> pd.DataFrame`.
  - `ZeroEmbeddingProducer`: returns all-zero embeddings + all-zero mask (GeneEffect-only baseline = exp06-equivalent in-harness).

This is the seam: exp08's DL becomes an `EmbeddingProducer` in later phases. The DL
does NOT touch metric/scoring code; it only produces a better per-gene embedding table.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_evaluate_parity.py
import numpy as np
import pandas as pd

from sl_dl_model.config import SLDLConfig
from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv


def _toy_frame() -> pd.DataFrame:
    rows = []
    genes = [f"G{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    eff = {g: float(rng.normal()) for g in genes}
    pid = 0
    for split in ("CV2",):
        for fold in (0, 1):
            for role in ("train", "test"):
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        a, b = genes[i], genes[j]
                        rows.append({
                            "pair_id": f"p{pid}", "fold_id": fold,
                            "split_type": split, "split_role": role,
                            "sl_label": (i + j) % 2,
                            "gene_a_symbol": a, "gene_b_symbol": b,
                            "gene_a_k562_gene_effect": eff[a],
                            "gene_b_k562_gene_effect": eff[b],
                        })
                        pid += 1
    return pd.DataFrame(rows)


def test_zero_producer_runs_and_emits_full_universe_metrics(tmp_path):
    csv = tmp_path / "toy.csv"
    _toy_frame().to_csv(csv, index=False)
    cfg = SLDLConfig(input_csv=csv, output_dir=tmp_path / "run",
                     split_types=("CV2",), folds=(0, 1),
                     include_coverage_flag=False)
    summary = run_cv(cfg, ZeroEmbeddingProducer())
    assert (tmp_path / "run" / "fold_metrics.csv").exists()
    assert (tmp_path / "run" / "manifest.json").exists()
    metrics = set(summary["metric"])
    assert "auroc" in metrics and "ndcg@10" in metrics
    assert set(summary["slice"]) >= {"full_universe"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_evaluate_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/evaluate.py
"""Per-fold CV runner: turns a per-gene embedding table into official metrics.

Reuses the exp06/07 baseline scoring harness verbatim. exp08's DL plugs in as an
:class:`EmbeddingProducer`; the metric/scoring path is never reimplemented here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from sl_benchmark_baseline.data import fold_split, load_benchmark
from sl_benchmark_baseline.evaluate import (
    _build_augmented_score_matrix,
    _build_gene_universe,
    _covered_pair_mask,
    _metric_rows,
    _pair_indices,
    _summarize,
)
from sl_benchmark_baseline.embeddings import GeneEmbeddingTable
from sl_benchmark_baseline.features import Standardizer, build_augmented_pair_features
from sl_dl_model.config import SLDLConfig

logger = logging.getLogger(__name__)


class EmbeddingProducer(Protocol):
    """Produces a per-gene embedding table + coverage mask for a fold."""

    def produce(
        self, symbols: np.ndarray, train_symbols: set[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings (n_gene, dim), coverage_mask (n_gene,))``."""
        ...


class ZeroEmbeddingProducer:
    """All-zero embedding (GeneEffect-only): exp06-equivalent in-harness baseline."""

    dim: int = 1

    def produce(
        self, symbols: np.ndarray, train_symbols: set[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(symbols)
        return np.zeros((n, self.dim), dtype=float), np.zeros(n, dtype=int)


def _augmented_raw_from_universe(
    frame: pd.DataFrame, universe, include_coverage_flag: bool
) -> np.ndarray:
    pair_idx = _pair_indices(frame, universe)
    a_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1]
    return build_augmented_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
        universe.embeddings[a_idx],
        universe.embeddings[b_idx],
        universe.coverage_mask[a_idx],
        universe.coverage_mask[b_idx],
        include_coverage_flag=include_coverage_flag,
    )


def run_cv(config: SLDLConfig, producer: "EmbeddingProducer | str") -> pd.DataFrame:
    """Run CV across split_types x folds, write metrics, return summary.

    ``producer`` is either a reusable :class:`EmbeddingProducer` instance (e.g.
    :class:`ZeroEmbeddingProducer`) or the string ``"state_dl"``. The string path
    loads the shared ESM2 + gwps-bags caches once and builds a per-fold
    ``StateDlProducer`` inside each fold (see Task 2.4 for ``_make_fold_producer``).
    """
    from sl_dl_model.scoring import make_fold_producer, run_fold_with_producer

    frame = load_benchmark(config.input_csv)
    split_types = config.split_types or ("CV1", "CV2", "CV3")
    available = set(frame["split_type"].unique())
    split_types = tuple(s for s in split_types if s in available)

    shared = None
    if producer == "state_dl":
        shared = _load_state_dl_caches(config)  # Task 2.4

    all_rows: list[dict[str, object]] = []
    for split_type in split_types:
        for fold_id in config.folds:
            fold_producer = (
                make_fold_producer(config, shared, frame, split_type, fold_id)
                if producer == "state_dl"
                else producer
            )
            all_rows.extend(
                run_fold_with_producer(
                    frame, split_type, fold_id, config, fold_producer
                )
            )
    fold_metrics = pd.DataFrame(all_rows)
    summary = _summarize(fold_metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    manifest = {
        "input_csv": str(config.input_csv),
        "split_types": list(split_types),
        "folds": list(config.folds),
        "ranking_k": list(config.ranking_k),
        "seed": config.seed,
        "embedding_method": config.embedding_method,
        "fallback_strategy": config.fallback_strategy,
        "include_coverage_flag": config.include_coverage_flag,
        "esm2_model": config.esm2_model,
        "state_checkpoint": str(config.state_checkpoint),
        "state_pert_vocab_overlap": 1542,
        "candidate_gene_count": 9471,
        "gwps_coverage_gene_count": 6070,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary
```

- [ ] **Step 4: Run test to verify it fails on missing scoring module**

Run: `uv run python -m pytest tests/sl_dl_model/test_evaluate_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.scoring'`
(this is expected — Task 0.3 provides `run_fold_with_producer`).

- [ ] **Step 5: Commit (defer green to Task 0.3)**

```bash
git add src/sl_dl_model/evaluate.py tests/sl_dl_model/test_evaluate_parity.py
git commit -m "feat: add exp08 embedding-producer protocol and CV runner skeleton"
```

### Task 0.3: Per-fold scoring with a producer + leakage-safe train_symbols

**Files:**
- Create: `src/sl_dl_model/scoring.py`
- Test: `tests/sl_dl_model/test_scoring.py`

**Interfaces:**
- Consumes: `EmbeddingProducer` (Task 0.2); the same `sl_benchmark_baseline.evaluate`
  helpers; `fold_split`.
- Produces: `run_fold_with_producer(frame, split_type, fold_id, config, producer) -> list[dict]`
  and `train_symbols_for_fold(train_df) -> set[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_scoring.py
import numpy as np
import pandas as pd

from sl_dl_model.config import SLDLConfig
from sl_dl_model.evaluate import ZeroEmbeddingProducer
from sl_dl_model.scoring import run_fold_with_producer, train_symbols_for_fold


def _toy_frame():
    rows = []
    genes = [f"G{i}" for i in range(6)]
    eff = {g: float(i) - 3 for i, g in enumerate(genes)}
    pid = 0
    for role in ("train", "test"):
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                rows.append({
                    "pair_id": f"p{pid}", "fold_id": 0, "split_type": "CV2",
                    "split_role": role, "sl_label": (i + j) % 2,
                    "gene_a_symbol": genes[i], "gene_b_symbol": genes[j],
                    "gene_a_k562_gene_effect": eff[genes[i]],
                    "gene_b_k562_gene_effect": eff[genes[j]],
                })
                pid += 1
    return pd.DataFrame(rows)


def test_train_symbols_excludes_test_only_genes():
    df = pd.DataFrame({
        "split_role": ["train", "train", "test"],
        "gene_a_symbol": ["A", "B", "E"],
        "gene_b_symbol": ["B", "C", "F"],
    })
    train_only = df[df["split_role"] == "train"]
    syms = train_symbols_for_fold(train_only)
    assert syms == {"A", "B", "C"}
    assert "E" not in syms and "F" not in syms


def test_run_fold_emits_metric_rows():
    cfg = SLDLConfig(include_coverage_flag=False, ranking_k=(10,))
    rows = run_fold_with_producer(_toy_frame(), "CV2", 0, cfg,
                                  ZeroEmbeddingProducer())
    assert rows
    metrics = {r["metric"] for r in rows}
    assert "auroc" in metrics and "ndcg@10" in metrics
    assert all(r["split_type"] == "CV2" for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.scoring'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/scoring.py
"""Per-fold scoring: producer -> universe embeddings -> official metric rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sl_benchmark_baseline.data import fold_split
from sl_benchmark_baseline.evaluate import (
    GeneUniverse,
    _build_gene_universe,
    _build_augmented_score_matrix,
    _covered_pair_mask,
    _metric_rows,
    _pair_indices,
)
from sl_benchmark_baseline.features import Standardizer, build_augmented_pair_features
from sl_dl_model.config import SLDLConfig


def train_symbols_for_fold(train_df: pd.DataFrame) -> set[str]:
    """Upper-case gene symbols appearing in this fold's TRAIN pairs only."""
    a = train_df["gene_a_symbol"].astype(str).str.upper()
    b = train_df["gene_b_symbol"].astype(str).str.upper()
    return set(a) | set(b)


def _augmented_raw(frame, universe, include_flag):
    idx = _pair_indices(frame, universe)
    a, b = idx[:, 0], idx[:, 1]
    return build_augmented_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
        universe.embeddings[a], universe.embeddings[b],
        universe.coverage_mask[a], universe.coverage_mask[b],
        include_coverage_flag=include_flag,
    )


def run_fold_with_producer(
    frame: pd.DataFrame,
    split_type: str,
    fold_id: int,
    config: SLDLConfig,
    producer,
) -> list[dict[str, object]]:
    """Produce a fold embedding table, score the universe, return metric rows.

    The DL scorer trains inside ``producer.produce`` on train pairs only, then this
    function reuses the exp07 augmented score matrix + official metrics verbatim.
    """
    train_df, test_df = fold_split(frame, split_type, fold_id)
    base_universe = _build_gene_universe(frame)
    train_symbols = train_symbols_for_fold(train_df)
    embeddings, coverage = producer.produce(base_universe.symbols, train_symbols)
    universe = GeneUniverse(
        keys=base_universe.keys,
        symbols=base_universe.symbols,
        gene_effects=base_universe.gene_effects,
        index_by_key=base_universe.index_by_key,
        embeddings=embeddings,
        coverage_mask=coverage,
    )

    test_pos = test_df[test_df["sl_label"] == 1]
    test_neg = test_df[test_df["sl_label"] == 0]
    train_pos = train_df[train_df["sl_label"] == 1]
    pos_index = _pair_indices(test_pos, universe)
    neg_index = _pair_indices(test_neg, universe)
    seen_index = _pair_indices(train_pos, universe)

    aug_std = Standardizer.fit(
        _augmented_raw(train_df, universe, config.include_coverage_flag)
    )

    # The producer already encodes embeddings into the universe; the pair scorer it
    # exposes is applied via the augmented score matrix path. For Phase 0 (Zero
    # producer) and feature-style producers, we fit the exp07 logistic/XGB models.
    from sl_benchmark_baseline.models import build_augmented_models

    proxy_config = _as_baseline_config(config)
    train_aug = _fold_data(train_df, universe, aug_std, config)
    rows: list[dict[str, object]] = []
    pos_cov = pos_index[_covered_pair_mask(pos_index, universe)]
    neg_cov = neg_index[_covered_pair_mask(neg_index, universe)]
    for model in build_augmented_models(proxy_config):
        if not model.name.endswith("_transcript"):
            continue
        model.fit(train_aug)
        sm = _build_augmented_score_matrix(
            model, universe, aug_std, config.include_coverage_flag
        )
        rows.extend(_metric_rows(split_type, model.name, fold_id, "full_universe",
                                 sm, pos_index, neg_index, seen_index,
                                 config.ranking_k))
        if len(pos_cov) > 0 and len(neg_cov) > 0:
            rows.extend(_metric_rows(split_type, model.name, fold_id,
                                     "covered_pairs", sm, pos_cov, neg_cov,
                                     seen_index, config.ranking_k))
    return rows


def _fold_data(frame, universe, std, config):
    from sl_benchmark_baseline.models import FoldData
    raw = _augmented_raw(frame, universe, config.include_coverage_flag)
    return FoldData(df=frame, features=std.transform(raw),
                    labels=frame["sl_label"].to_numpy(dtype=int))


def _as_baseline_config(config: SLDLConfig):
    from sl_benchmark_baseline.config import SLBaselineConfig
    return SLBaselineConfig(
        input_csv=config.input_csv, output_dir=config.output_dir,
        split_types=config.split_types, folds=config.folds,
        ranking_k=config.ranking_k, seed=config.seed,
        fallback_strategy=config.fallback_strategy,
        include_coverage_flag=config.include_coverage_flag,
        bags_npz=config.bags_npz,
    )
```

- [ ] **Step 4: Run both evaluate + scoring tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_scoring.py tests/sl_dl_model/test_evaluate_parity.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/scoring.py tests/sl_dl_model/test_scoring.py
git commit -m "feat: add exp08 per-fold producer scoring reusing exp07 harness"
```

> **NOTE for Phase 0 gate:** the Zero/feature producers reuse the exp07 logistic/XGB
> models so Phase 0 can reproduce exp06+exp07 numbers. In Phase 2+ the DL pair head
> replaces this model loop — see Task 2.x where `run_fold_with_producer` gains a
> `score_matrix_fn` from the DL scorer. Keep the `_transcript` model path for the
> coverage-flag ablation baseline.

### Task 0.4: CLI `run-cv` + Phase 0 gate (exp06/07 parity)

**Files:**
- Create: `src/sl_dl_model/__main__.py`
- Create: `configs/experiments/08_k562_sl_pair_state_dl/phase0_parity.yaml`
- Test: `tests/sl_dl_model/test_cli.py`

**Interfaces:**
- Consumes: `run_cv`, `ZeroEmbeddingProducer` (Task 0.2), `load_config` (Task 0.1).
- Produces: CLI entrypoint `uv run python -m sl_dl_model run-cv --config <path>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_cli.py
import subprocess
import sys


def test_cli_help_lists_run_cv():
    out = subprocess.run(
        [sys.executable, "-m", "sl_dl_model", "--help"],
        capture_output=True, text=True,
    )
    assert "run-cv" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_cli.py -v`
Expected: FAIL (no `__main__` / no `run-cv` in help)

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/__main__.py
"""CLI for the exp08 STATE-adapter DL SL-pair model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sl_dl_model.config import load_config
from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sl_dl_model")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-cv", help="Run CV and write official metrics.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--producer", choices=["zero", "state_dl"], default="state_dl",
        help="zero = GeneEffect-only exp06-parity baseline.",
    )
    run.add_argument("--log-file", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    if args.producer == "zero":
        producer = ZeroEmbeddingProducer()
        run_cv(config, producer)
    else:
        # state_dl producer is constructed per-fold inside scoring (needs fold
        # train pairs + shared ESM2/bags caches); run_cv dispatches on a flag.
        run_cv(config, producer="state_dl")


if __name__ == "__main__":
    main()
```

```yaml
# configs/experiments/08_k562_sl_pair_state_dl/phase0_parity.yaml
# exp06/07 in-harness parity baseline (GeneEffect-only via zero producer).
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08_k562_sl_pair_state_dl/phase0_parity
split_types: [CV1, CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
include_coverage_flag: false
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Phase 0 gate — reproduce exp06 numbers (manual verification)**

Run: `uv run python -m sl_dl_model run-cv --config configs/experiments/08_k562_sl_pair_state_dl/phase0_parity.yaml --producer zero`

Then compare to exp06 reference:
```bash
uv run python -c "
import pandas as pd
s = pd.read_csv('results/experiments/08_k562_sl_pair_state_dl/phase0_parity/summary.csv')
b = s[(s.metric=='ndcg@10') & (s['slice']=='full_universe')]
print(b[['split_type','model','mean','std']].to_string())
"
```
Expected: the GeneEffect-only model's CV2 NDCG@10 ≈ 0.042 ± 0.008, CV3 ≈ 0.002,
matching `results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv`
(XGB rows). **If CV2/CV3 differ beyond fold noise, STOP — the harness wiring is wrong.**

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/__main__.py configs/experiments/08_k562_sl_pair_state_dl/phase0_parity.yaml tests/sl_dl_model/test_cli.py
git commit -m "feat: add exp08 run-cv CLI and Phase 0 parity config"
```

---

## Phase 1 — ESM2 gene embeddings + frozen-STATE plumbing

Goal: precompute ESM2 per-gene embeddings; wire `PertAdapter → frozen STATE →
pooling → e_g`; train the adapter on the distill anchor only. **Gate:** adapter
reproduces STATE's original token on held-out in-vocab genes (low distill MSE);
`e_g` is finite and varies across genes.

### Task 1.1: ESM2 precompute script (UniProt fetch + embed)

**Files:**
- Create: `scripts/precompute_esm2_embeddings.py`
- Create: `src/sl_dl_model/gene_embeddings.py`
- Test: `tests/sl_dl_model/test_gene_embeddings.py`

**Interfaces:**
- Produces:
  - `Esm2EmbeddingTable` dataclass: `dim: int`, `vectors_by_symbol: dict[str, np.ndarray]`.
  - `load_esm2_embeddings(npz: Path) -> Esm2EmbeddingTable`.
  - `align_esm2_to_universe(table, symbols, fallback_strategy) -> tuple[np.ndarray, np.ndarray]`
    (mirrors `sl_benchmark_baseline.embeddings.align_to_universe`).
  - Script writes `.npz` with keys: `symbols (object)`, `vectors (float32 (n, 1280))`,
    `resolved (bool)` per gene.

> **Network note:** the UniProt fetch in this script runs ONCE on a network-connected
> node. Training/eval read only the cached `.npz` — fully offline. Use stdlib
> `urllib.request` (no new dependency). Cache the symbol→sequence JSON so re-runs are
> incremental. Per-symbol query: UniProt REST search, human (taxonomy 9606), reviewed,
> take the canonical sequence of the top hit.

- [ ] **Step 1: Write the failing test (offline — validates cache format + alignment)**

```python
# tests/sl_dl_model/test_gene_embeddings.py
import numpy as np

from sl_dl_model.gene_embeddings import (
    Esm2EmbeddingTable,
    align_esm2_to_universe,
    load_esm2_embeddings,
)


def _write_npz(path):
    symbols = np.array(["TP53", "KRAS", "EGFR"], dtype=object)
    vectors = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    resolved = np.array([True, True, False])
    np.savez(path, symbols=symbols, vectors=vectors, resolved=resolved)


def test_load_and_align(tmp_path):
    npz = tmp_path / "esm2.npz"
    _write_npz(npz)
    table = load_esm2_embeddings(npz)
    assert table.dim == 4
    assert set(table.vectors_by_symbol) == {"TP53", "KRAS"}  # unresolved dropped

    symbols = np.array(["KRAS", "TP53", "UNKNOWN"], dtype=object)
    emb, mask = align_esm2_to_universe(table, symbols, "zero")
    assert emb.shape == (3, 4)
    assert mask.tolist() == [1, 1, 0]
    assert np.allclose(emb[2], 0.0)  # fallback for uncovered


def test_align_global_mean_fallback(tmp_path):
    npz = tmp_path / "esm2.npz"
    _write_npz(npz)
    table = load_esm2_embeddings(npz)
    symbols = np.array(["TP53", "ZZZ"], dtype=object)
    emb, mask = align_esm2_to_universe(table, symbols, "global_mean")
    assert mask.tolist() == [1, 0]
    assert not np.allclose(emb[1], 0.0)  # global-mean fallback non-zero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_gene_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.gene_embeddings'`

- [ ] **Step 3: Write minimal implementation (loader + alignment)**

```python
# src/sl_dl_model/gene_embeddings.py
"""Load precomputed ESM2 per-gene embeddings and align to a gene universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Esm2EmbeddingTable:
    """Per-gene ESM2 embedding vectors keyed by upper-case symbol."""

    dim: int
    vectors_by_symbol: dict[str, np.ndarray]


def load_esm2_embeddings(npz: Path) -> Esm2EmbeddingTable:
    """Load a precomputed ESM2 ``.npz``; drop unresolved genes."""
    with np.load(npz, allow_pickle=True) as payload:
        symbols = np.asarray(payload["symbols"], dtype=object)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        resolved = np.asarray(payload["resolved"], dtype=bool)
    table: dict[str, np.ndarray] = {}
    for symbol, vector, ok in zip(symbols, vectors, resolved, strict=True):
        if bool(ok):
            table[str(symbol).upper()] = vector
    return Esm2EmbeddingTable(dim=int(vectors.shape[1]), vectors_by_symbol=table)


def align_esm2_to_universe(
    table: Esm2EmbeddingTable, symbols: np.ndarray, fallback_strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Align ESM2 vectors to universe order with a coverage mask.

    Mirrors :func:`sl_benchmark_baseline.embeddings.align_to_universe`.
    """
    if fallback_strategy not in {"zero", "global_mean"}:
        raise ValueError(f"unknown fallback_strategy: {fallback_strategy}")
    covered = [
        table.vectors_by_symbol[str(s).upper()]
        for s in symbols
        if str(s).upper() in table.vectors_by_symbol
    ]
    if fallback_strategy == "global_mean" and covered:
        fallback = np.mean(np.vstack(covered), axis=0)
    else:
        fallback = np.zeros(table.dim, dtype=float)
    embeddings = np.zeros((len(symbols), table.dim), dtype=float)
    mask = np.zeros(len(symbols), dtype=int)
    for row, symbol in enumerate(symbols):
        key = str(symbol).upper()
        if key in table.vectors_by_symbol:
            embeddings[row] = table.vectors_by_symbol[key]
            mask[row] = 1
        else:
            embeddings[row] = fallback
    return embeddings, mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_gene_embeddings.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the precompute script (run once, offline thereafter)**

```python
# scripts/precompute_esm2_embeddings.py
"""Fetch UniProt sequences for SL-universe genes and embed with ESM2.

Run once on a network node:
    uv run python scripts/precompute_esm2_embeddings.py \
        --benchmark-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
        --out data/esm2/k562_sl_universe_esm2_650M.npz \
        --seq-cache data/esm2/symbol_to_sequence.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

logger = logging.getLogger("precompute_esm2")
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def universe_symbols(benchmark_csv: Path) -> list[str]:
    frame = pd.read_csv(
        benchmark_csv, usecols=["gene_a_symbol", "gene_b_symbol"]
    )
    symbols = set(frame["gene_a_symbol"].str.upper()) | set(
        frame["gene_b_symbol"].str.upper()
    )
    return sorted(symbols)


def fetch_sequence(symbol: str) -> str | None:
    """Return the canonical human protein sequence for a gene symbol, or None."""
    query = (
        f'(gene:{symbol}) AND (organism_id:9606) AND (reviewed:true)'
    )
    params = urllib.parse.urlencode(
        {"query": query, "format": "fasta", "size": 1}
    )
    url = f"{UNIPROT_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - network best-effort, logged
        logger.warning("fetch failed for %s: %s", symbol, exc)
        return None
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines) or None


def load_or_fetch_sequences(symbols: list[str], cache: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    if cache.exists():
        seqs = json.loads(cache.read_text())
    for i, symbol in enumerate(symbols):
        if symbol in seqs:
            continue
        seq = fetch_sequence(symbol)
        if seq:
            seqs[symbol] = seq
        if i % 100 == 0:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(seqs))
            logger.info("resolved %d/%d sequences", len(seqs), len(symbols))
        time.sleep(0.1)  # be polite to UniProt
    cache.write_text(json.dumps(seqs))
    return seqs


def embed_sequences(
    symbols: list[str], seqs: dict[str, str], model_name: str
) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    dim = model.config.hidden_size
    vectors = np.zeros((len(symbols), dim), dtype=np.float32)
    resolved = np.zeros(len(symbols), dtype=bool)
    with torch.no_grad():
        for row, symbol in enumerate(symbols):
            seq = seqs.get(symbol)
            if not seq:
                continue
            toks = tokenizer(seq[:1022], return_tensors="pt").to(device)
            out = model(**toks).last_hidden_state[0]  # (L, dim)
            vectors[row] = out.mean(dim=0).cpu().numpy()
            resolved[row] = True
            if row % 200 == 0:
                logger.info("embedded %d/%d", row, len(symbols))
    return vectors, resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq-cache", type=Path, required=True)
    parser.add_argument(
        "--model", default="facebook/esm2_t33_650M_UR50D"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    symbols = universe_symbols(args.benchmark_csv)
    logger.info("universe size: %d genes", len(symbols))
    seqs = load_or_fetch_sequences(symbols, args.seq_cache)
    vectors, resolved = embed_sequences(symbols, seqs, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        symbols=np.array(symbols, dtype=object),
        vectors=vectors,
        resolved=resolved,
    )
    logger.info(
        "wrote %s (%d resolved / %d)", args.out, int(resolved.sum()), len(symbols)
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the script wiring offline (no network)**

Run:
```bash
uv run python -c "
from scripts.precompute_esm2_embeddings import universe_symbols
import pandas as pd, tempfile, os
df = pd.DataFrame({'gene_a_symbol':['tp53'],'gene_b_symbol':['kras']})
p = tempfile.mktemp(suffix='.csv'); df.to_csv(p, index=False)
print(universe_symbols(p))
"
```
Expected: `['KRAS', 'TP53']`

- [ ] **Step 7: Commit**

```bash
git add scripts/precompute_esm2_embeddings.py src/sl_dl_model/gene_embeddings.py tests/sl_dl_model/test_gene_embeddings.py
git commit -m "feat: add exp08 ESM2 precompute script and embedding loader"
```

### Task 1.2: gwps bags + shared K562 control template

**Files:**
- Create: `src/sl_dl_model/bags.py`
- Test: `tests/sl_dl_model/test_bags.py`

**Interfaces:**
- Produces:
  - `GwpsBags` dataclass: `control_template: np.ndarray (T, D)`,
    `bags_by_symbol: dict[str, np.ndarray]` (each `(n_cells, D)`),
    `input_dim: int`, `batch_by_symbol: dict[str, np.ndarray] | None`.
  - `build_gwps_bags(config: SLDLConfig, rng_seed: int = 17) -> GwpsBags`
    (reads h5ad backed; uses `obs.gene`, `non-targeting` control, `X_hvg` obsm,
    `gem_group` batch col — verified against exp05 config).
  - `save_bags_npz(bags, path)` / `load_bags_npz(path) -> GwpsBags` (cache, since the
    h5ad read is expensive).

> Reuse the chunked, backed-h5ad reading approach from `aivc_model.prepare.load_gene_bags`
> (do not load the full 1.99M-cell matrix into memory). Subsample each gene's cells to
> `config.cells_per_bag` and control cells to `config.control_template_size`, seeded.

- [ ] **Step 1: Write the failing test (synthetic AnnData, no real h5ad)**

```python
# tests/sl_dl_model/test_bags.py
import anndata as ad
import numpy as np
import pandas as pd

from sl_dl_model.bags import build_gwps_bags, load_bags_npz, save_bags_npz
from sl_dl_model.config import SLDLConfig


def _toy_h5ad(path):
    n, d = 200, 6
    rng = np.random.default_rng(0)
    genes = (["non-targeting"] * 80 + ["AAAS"] * 60 + ["KRAS"] * 60)
    obs = pd.DataFrame({"gene": genes, "gem_group": ["b0"] * n})
    adata = ad.AnnData(X=rng.normal(size=(n, d)).astype("float32"), obs=obs)
    adata.obsm["X_hvg"] = rng.normal(size=(n, d)).astype("float32")
    adata.write_h5ad(path)


def test_build_and_cache_bags(tmp_path):
    h5ad = tmp_path / "toy.h5ad"
    _toy_h5ad(h5ad)
    cfg = SLDLConfig(gwps_h5ad=h5ad, control_template_size=16, cells_per_bag=16)
    bags = build_gwps_bags(cfg, rng_seed=17)
    assert bags.input_dim == 6
    assert bags.control_template.shape == (16, 6)
    assert set(bags.bags_by_symbol) == {"AAAS", "KRAS"}
    assert bags.bags_by_symbol["KRAS"].shape[1] == 6

    npz = tmp_path / "bags.npz"
    save_bags_npz(bags, npz)
    loaded = load_bags_npz(npz)
    assert set(loaded.bags_by_symbol) == {"AAAS", "KRAS"}
    assert np.allclose(loaded.control_template, bags.control_template)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_bags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.bags'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/bags.py
"""Build per-gene gwps response bags and a shared K562 control template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from sl_dl_model.config import SLDLConfig


@dataclass(frozen=True)
class GwpsBags:
    """Per-gene response bags + a shared control template in STATE input space."""

    control_template: np.ndarray
    bags_by_symbol: dict[str, np.ndarray]
    input_dim: int


def _embed_matrix(adata: ad.AnnData, embed_key: str | None) -> np.ndarray:
    if embed_key and embed_key in adata.obsm:
        return np.asarray(adata.obsm[embed_key], dtype=np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def build_gwps_bags(config: SLDLConfig, rng_seed: int = 17) -> GwpsBags:
    """Read gwps h5ad and build subsampled per-gene bags + control template."""
    rng = np.random.default_rng(rng_seed)
    adata = ad.read_h5ad(config.gwps_h5ad)
    matrix = _embed_matrix(adata, "X_hvg")
    genes = adata.obs["gene"].astype(str).to_numpy()
    control_label = "non-targeting"

    control_rows = np.where(genes == control_label)[0]
    if len(control_rows) > config.control_template_size:
        control_rows = rng.choice(
            control_rows, size=config.control_template_size, replace=False
        )
    control_template = matrix[np.sort(control_rows)]

    bags: dict[str, np.ndarray] = {}
    for symbol in np.unique(genes):
        if symbol == control_label:
            continue
        rows = np.where(genes == symbol)[0]
        if len(rows) == 0:
            continue
        if len(rows) > config.cells_per_bag:
            rows = rng.choice(rows, size=config.cells_per_bag, replace=False)
        bags[str(symbol).upper()] = matrix[np.sort(rows)]
    return GwpsBags(
        control_template=control_template,
        bags_by_symbol=bags,
        input_dim=int(matrix.shape[1]),
    )


def save_bags_npz(bags: GwpsBags, path: Path) -> None:
    """Cache bags to a flat NPZ (ragged offsets)."""
    symbols = sorted(bags.bags_by_symbol)
    arrays = [bags.bags_by_symbol[s] for s in symbols]
    offsets = np.cumsum([0] + [a.shape[0] for a in arrays])
    flat = (
        np.vstack(arrays)
        if arrays
        else np.zeros((0, bags.input_dim), dtype=np.float32)
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        control_template=bags.control_template,
        symbols=np.array(symbols, dtype=object),
        flat=flat.astype(np.float32),
        offsets=offsets.astype(np.int64),
        input_dim=np.int64(bags.input_dim),
    )


def load_bags_npz(path: Path) -> GwpsBags:
    """Load cached bags from NPZ."""
    with np.load(path, allow_pickle=True) as payload:
        control = np.asarray(payload["control_template"], dtype=np.float32)
        symbols = np.asarray(payload["symbols"], dtype=object)
        flat = np.asarray(payload["flat"], dtype=np.float32)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        input_dim = int(payload["input_dim"])
    bags = {
        str(symbols[i]): flat[offsets[i]:offsets[i + 1]]
        for i in range(len(symbols))
    }
    return GwpsBags(control_template=control, bags_by_symbol=bags,
                    input_dim=input_dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_bags.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/bags.py tests/sl_dl_model/test_bags.py
git commit -m "feat: add exp08 gwps bag builder and control template"
```

### Task 1.3: PertAdapter + frozen-STATE encoder

**Files:**
- Create: `src/sl_dl_model/encoder.py`
- Test: `tests/sl_dl_model/test_encoder.py`

**Interfaces:**
- Consumes: `aivc_model.model.{StateForwardAdapter, load_state_model, LinearMockStateModel}`.
- Produces:
  - `PertAdapter(nn.Module)`: `__init__(esm_dim, hidden, pert_dim)`,
    `forward(esm: Tensor (B, esm_dim)) -> Tensor (B, pert_dim)`.
  - `StateEncoder(nn.Module)`: wraps a frozen STATE model + `StateForwardAdapter`;
    `forward(esm_vec: Tensor (esm_dim,), control_cells: Tensor (T, input_dim)) -> Tensor (T, output_dim)`
    (one gene's predicted response bag). Backbone params frozen
    (`requires_grad=False`); only `PertAdapter` trains.
  - `state_original_token(state_model, onehot: Tensor (vocab,)) -> Tensor (pert_dim,)`
    — applies the checkpoint's own `pert_encoder` to a one-hot, the distill target.

> Use `state_backend="linear_mock"` in tests (no checkpoint needed). The mock's
> `forward(batch, padded)` consumes `ctrl_cell_emb` + `pert_emb`; pass the adapter
> output as `pert_emb` via `StateForwardAdapter`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_encoder.py
import torch

from sl_dl_model.encoder import PertAdapter, StateEncoder


def test_pert_adapter_shapes():
    adapter = PertAdapter(esm_dim=8, hidden=16, pert_dim=5)
    out = adapter(torch.randn(3, 8))
    assert out.shape == (3, 5)


def test_state_encoder_forward_with_mock_backend():
    enc = StateEncoder(
        backend="linear_mock", checkpoint=None,
        esm_dim=8, adapter_hidden=16, pert_dim=5,
        input_dim=6, output_dim=6,
    )
    esm_vec = torch.randn(8)
    control = torch.randn(10, 6)
    bag = enc(esm_vec, control)
    assert bag.shape == (10, 6)


def test_backbone_frozen_adapter_trainable():
    enc = StateEncoder(
        backend="linear_mock", checkpoint=None,
        esm_dim=8, adapter_hidden=16, pert_dim=5, input_dim=6, output_dim=6,
    )
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert all(n.startswith("adapter") for n in trainable)
    assert any(n.startswith("adapter") for n in trainable)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.encoder'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/encoder.py
"""Trainable ESM2->pert adapter on top of a frozen Arc STATE backbone."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from aivc_model.model import StateForwardAdapter, load_state_model


class PertAdapter(nn.Module):
    """Map an ESM2 gene embedding to a STATE pert token (replaces pert_encoder)."""

    def __init__(self, esm_dim: int, hidden: int, pert_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pert_dim),
        )

    def forward(self, esm: torch.Tensor) -> torch.Tensor:
        return self.net(esm)


class StateEncoder(nn.Module):
    """Frozen STATE backbone fed by a trainable ESM2 pert-adapter."""

    def __init__(
        self,
        *,
        backend: str,
        checkpoint: Path | None,
        esm_dim: int,
        adapter_hidden: int,
        pert_dim: int,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.adapter = PertAdapter(esm_dim, adapter_hidden, pert_dim)
        state_model = load_state_model(
            backend=backend, checkpoint_path=checkpoint,
            input_dim=input_dim, output_dim=output_dim, pert_dim=pert_dim,
            emit_checkpoint_output=False,
        )
        for param in state_model.parameters():
            param.requires_grad = False
        state_model.eval()
        self.state = StateForwardAdapter(state_model)

    def train(self, mode: bool = True) -> "StateEncoder":
        """Keep the frozen STATE backbone in eval; let the adapter follow mode."""
        super().train(mode)
        self.state.eval()
        self.adapter.train(mode)
        return self

    def forward(
        self, esm_vec: torch.Tensor, control_cells: torch.Tensor
    ) -> torch.Tensor:
        pert = self.adapter(esm_vec.unsqueeze(0)).squeeze(0)
        return self.state(control_cells, pert, gene="adapter")


def state_original_token(
    state_model: nn.Module, onehot: torch.Tensor
) -> torch.Tensor:
    """Apply the checkpoint's own pert_encoder to a one-hot (distill target)."""
    encoder = getattr(state_model, "pert_encoder", None)
    if encoder is None:
        raise AttributeError("state_model has no pert_encoder for distillation")
    with torch.no_grad():
        return encoder(onehot.float())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_encoder.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/encoder.py tests/sl_dl_model/test_encoder.py
git commit -m "feat: add exp08 PertAdapter and frozen-STATE encoder"
```

### Task 1.4: Pooling head (bag → e_g)

**Files:**
- Create: `src/sl_dl_model/pooling.py`
- Test: `tests/sl_dl_model/test_pooling.py`

**Interfaces:**
- Produces:
  - `MeanStdPool(nn.Module)`: `forward(bag: Tensor (n_cells, D)) -> Tensor (2D,)`.
  - `build_pool(name: str, dim: int) -> nn.Module` (default `"mean_std"`;
    raises on unknown). `output_dim(name, dim) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_pooling.py
import torch

from sl_dl_model.pooling import MeanStdPool, build_pool, output_dim


def test_mean_std_pool_dim():
    pool = MeanStdPool()
    out = pool(torch.randn(20, 6))
    assert out.shape == (12,)


def test_build_pool_and_output_dim():
    assert output_dim("mean_std", 6) == 12
    pool = build_pool("mean_std", 6)
    assert isinstance(pool, MeanStdPool)


def test_unknown_pool_raises():
    try:
        build_pool("nope", 6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_pooling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.pooling'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/pooling.py
"""Permutation-invariant pooling of a predicted response bag into e_g."""

from __future__ import annotations

import torch
from torch import nn


class MeanStdPool(nn.Module):
    """Concatenate per-feature mean and std over the cell dimension."""

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        mean = bag.mean(dim=0)
        std = bag.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)


def build_pool(name: str, dim: int) -> nn.Module:
    """Return a pooling module by name."""
    if name == "mean_std":
        return MeanStdPool()
    raise ValueError(f"unknown pooling: {name}")


def output_dim(name: str, dim: int) -> int:
    """Return the pooled embedding dimension for a pooling name."""
    if name == "mean_std":
        return 2 * dim
    raise ValueError(f"unknown pooling: {name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_pooling.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Phase 1 gate (manual, requires real STATE checkpoint + ESM2 cache)**

After Tasks 1.1–1.4, verify the encoder produces sane `e_g` end-to-end with the real
checkpoint. This is a manual check (no committed test, since it needs the 16MB+
checkpoint and ESM2 cache):
```bash
uv run python -c "
import torch, numpy as np
from sl_dl_model.encoder import StateEncoder
from sl_dl_model.pooling import MeanStdPool
# load real checkpoint dims from var_dims.pkl; smoke a forward
print('manual Phase 1 gate: confirm e_g finite + gene-varying')
"
```
Expected: `e_g` vectors are finite and differ across two distinct ESM2 inputs.
**Gate:** if `e_g` is constant across genes or non-finite, STOP and debug the encoder
before Phase 2.

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/pooling.py tests/sl_dl_model/test_pooling.py
git commit -m "feat: add exp08 mean-std pooling head"
```

---

## Phase 2 — SL classifier (pair head + BCE)

Goal: add the symmetric pair head and the SL BCE loss; assemble the
`StateDlProducer` that trains per fold and emits a per-gene `e_g` table + a DL score
matrix. **Gate:** beat exp06 on CV2 classification (AUROC > 0.704 / AUPR > 0.732).

### Task 2.1: Symmetric pair head

**Files:**
- Create: `src/sl_dl_model/pair_head.py`
- Test: `tests/sl_dl_model/test_pair_head.py`

**Interfaces:**
- Produces:
  - `SymmetricPairHead(nn.Module)`: `__init__(emb_dim, geneeffect_dim=5, hidden=(256,64), include_coverage_flag=False)`;
    `forward(e_a, e_b, ge_features, cov_a=None, cov_b=None) -> logit (B,)`.
  - The transcript block is built swap-invariantly inside: `[e_a+e_b, |e_a-e_b|, e_a*e_b]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_pair_head.py
import torch

from sl_dl_model.pair_head import SymmetricPairHead


def test_pair_head_output_shape():
    head = SymmetricPairHead(emb_dim=12, hidden=(16,))
    e_a = torch.randn(4, 12)
    e_b = torch.randn(4, 12)
    ge = torch.randn(4, 5)
    logit = head(e_a, e_b, ge)
    assert logit.shape == (4,)


def test_pair_head_is_swap_invariant():
    torch.manual_seed(0)
    head = SymmetricPairHead(emb_dim=12, hidden=(16,)).eval()
    e_a = torch.randn(4, 12)
    e_b = torch.randn(4, 12)
    # GeneEffect block must also be swap-invariant; reuse same ge for both orders
    ge = torch.randn(4, 5)
    with torch.no_grad():
        ab = head(e_a, e_b, ge)
        ba = head(e_b, e_a, ge)
    assert torch.allclose(ab, ba, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_pair_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.pair_head'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/pair_head.py
"""Swap-invariant pair scorer over per-gene embeddings + GeneEffect features."""

from __future__ import annotations

import torch
from torch import nn


class SymmetricPairHead(nn.Module):
    """Score a gene pair from swap-invariant transcript + GeneEffect features."""

    def __init__(
        self,
        emb_dim: int,
        geneeffect_dim: int = 5,
        hidden: tuple[int, ...] = (256, 64),
        include_coverage_flag: bool = False,
    ) -> None:
        super().__init__()
        self.include_coverage_flag = include_coverage_flag
        transcript_dim = 3 * emb_dim
        cov_dim = 2 if include_coverage_flag else 0
        in_dim = transcript_dim + geneeffect_dim + cov_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.GELU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        e_a: torch.Tensor,
        e_b: torch.Tensor,
        ge_features: torch.Tensor,
        cov_a: torch.Tensor | None = None,
        cov_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        blocks = [e_a + e_b, (e_a - e_b).abs(), e_a * e_b, ge_features]
        if self.include_coverage_flag:
            if cov_a is None or cov_b is None:
                raise ValueError("coverage flags required when enabled")
            cov_min = torch.minimum(cov_a, cov_b).unsqueeze(-1)
            cov_max = torch.maximum(cov_a, cov_b).unsqueeze(-1)
            blocks += [cov_min, cov_max]
        features = torch.cat(blocks, dim=-1)
        return self.net(features).squeeze(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_pair_head.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/pair_head.py tests/sl_dl_model/test_pair_head.py
git commit -m "feat: add exp08 symmetric pair head"
```

### Task 2.2: Loss assembly (distill + bag + SL BCE)

**Files:**
- Create: `src/sl_dl_model/losses.py`
- Test: `tests/sl_dl_model/test_losses.py`

**Interfaces:**
- Consumes: `aivc_model.model.{_energy_distance, _pairwise_ranknet_loss}`.
- Produces:
  - `sl_bce_loss(logits, labels) -> Tensor`.
  - `distill_loss(adapter_tokens, target_tokens) -> Tensor` (MSE).
  - `bag_loss(pred_bag, real_bag) -> Tensor` (mean-delta MSE + energy distance).
  - `combine(parts: dict[str, Tensor], weights: dict[str, float]) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_losses.py
import torch

from sl_dl_model.losses import bag_loss, combine, distill_loss, sl_bce_loss


def test_sl_bce_decreases_with_correct_logits():
    labels = torch.tensor([1.0, 0.0, 1.0])
    good = sl_bce_loss(torch.tensor([5.0, -5.0, 5.0]), labels)
    bad = sl_bce_loss(torch.tensor([-5.0, 5.0, -5.0]), labels)
    assert good < bad


def test_distill_zero_when_equal():
    t = torch.randn(4, 8)
    assert distill_loss(t, t.clone()).item() < 1e-8


def test_bag_loss_nonnegative_and_zero_for_identical():
    bag = torch.randn(16, 6)
    assert bag_loss(bag, bag.clone()).item() < 1e-4


def test_combine_weights():
    parts = {"sl": torch.tensor(2.0), "distill": torch.tensor(4.0)}
    weights = {"sl": 1.0, "distill": 0.5}
    total = combine(parts, weights)
    assert abs(total.item() - 4.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.losses'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_dl_model/losses.py
"""Three-part loss for exp08: SL BCE + adapter distill + bag supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aivc_model.model import _energy_distance


def sl_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on SL pair logits."""
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def distill_loss(
    adapter_tokens: torch.Tensor, target_tokens: torch.Tensor
) -> torch.Tensor:
    """MSE between adapter output and STATE's original one-hot pert token."""
    return F.mse_loss(adapter_tokens, target_tokens)


def bag_loss(pred_bag: torch.Tensor, real_bag: torch.Tensor) -> torch.Tensor:
    """Mean-delta MSE + energy distance between predicted and real bags."""
    mean_delta = F.mse_loss(pred_bag.mean(dim=0), real_bag.mean(dim=0))
    energy = _energy_distance(pred_bag, real_bag)
    return mean_delta + energy


def combine(
    parts: dict[str, torch.Tensor], weights: dict[str, float]
) -> torch.Tensor:
    """Weighted sum of named loss parts (missing parts contribute 0)."""
    total: torch.Tensor | None = None
    for name, value in parts.items():
        weight = float(weights.get(name, 0.0))
        term = weight * value
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no loss parts provided")
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_losses.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/losses.py tests/sl_dl_model/test_losses.py
git commit -m "feat: add exp08 three-part loss assembly"
```

### Task 2.3: Training module + StateDlProducer (Accelerate/DDP, tqdm)

**Files:**
- Create: `src/sl_dl_model/model.py`
- Create: `src/sl_dl_model/train.py`
- Test: `tests/sl_dl_model/test_train.py`

**Interfaces:**
- Consumes: `StateEncoder`, `PertAdapter` (Task 1.3); `build_pool`, `output_dim`
  (Task 1.4); `SymmetricPairHead` (Task 2.1); `sl_bce_loss`, `distill_loss`,
  `bag_loss`, `combine` (Task 2.2); `GwpsBags` (Task 1.2); `Esm2EmbeddingTable`,
  `align_esm2_to_universe` (Task 1.1); `EmbeddingProducer` (Task 0.2).
- Produces:
  - `SlDlModel(nn.Module)`: holds `StateEncoder`, pooling, `SymmetricPairHead`;
    `embed_gene(esm_vec, control) -> e_g`; `score_pairs(e_a, e_b, ge, cov_a, cov_b) -> logits`.
  - `StateDlProducer` implementing `EmbeddingProducer.produce` AND exposing
    `score_matrix(universe) -> np.ndarray` (the DL pair head's full 9,471² scores).
  - `train_one_fold(model, fold_pairs, esm, bags, config, accelerator) -> None`.

> **Schedule (from spec §6.4):** epochs `< warmup_epochs` use
> `lambda_distill` and `lambda_bag` with `lambda_sl=0`; epochs `>= warmup_epochs`
> use `lambda_sl`, `lambda_distill_after_warmup` (reduced, stays on), and `lambda_bag`.
> **Leakage rule (spec §5):** `bag_loss` only for genes in `train_symbols ∩ covered`;
> distill only for genes in `train_symbols ∩ in_vocab`. Held-out genes are embedded at
> eval purely via `adapter(ESM2)` + frozen STATE — never supervised.

- [ ] **Step 1: Write the failing test (mock backend, tiny fold)**

```python
# tests/sl_dl_model/test_train.py
import numpy as np
import torch

from sl_dl_model.config import SLDLConfig
from sl_dl_model.model import SlDlModel


def _model(esm_dim=8, input_dim=6):
    return SlDlModel(
        backend="linear_mock", checkpoint=None, esm_dim=esm_dim,
        adapter_hidden=16, pert_dim=5, input_dim=input_dim, output_dim=input_dim,
        pooling="mean_std", pair_hidden=(16,), include_coverage_flag=False,
    )


def test_embed_gene_shape():
    model = _model().eval()
    e_g = model.embed_gene(torch.randn(8), torch.randn(10, 6))
    assert e_g.shape == (12,)  # mean_std over 6-d output


def test_score_pairs_shape_and_backprop():
    model = _model()
    e_a = torch.randn(4, 12, requires_grad=True)
    e_b = torch.randn(4, 12)
    ge = torch.randn(4, 5)
    logits = model.score_pairs(e_a, e_b, ge)
    assert logits.shape == (4,)
    logits.sum().backward()
    assert e_a.grad is not None


def test_producer_emits_universe_table(tmp_path):
    # Build tiny ESM2 + bags caches, run produce() on a 4-gene universe.
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.train import StateDlProducer

    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8, vectors_by_symbol={s: np.random.randn(8).astype("float32")
                                  for s in ["A", "B", "C", "D"]}
    )
    bags = GwpsBags(
        control_template=np.random.randn(8, 6).astype("float32"),
        bags_by_symbol={"A": np.random.randn(8, 6).astype("float32"),
                        "B": np.random.randn(8, 6).astype("float32")},
        input_dim=6,
    )
    cfg = SLDLConfig(esm2_model="x", max_epochs=1, warmup_epochs=1,
                     pert_dim=5, adapter_hidden=16, pair_hidden=(16,),
                     include_coverage_flag=False, state_backend="linear_mock")
    pairs = [("A", "B", 1, -1.0, -0.5), ("C", "D", 0, 0.1, 0.2),
             ("A", "C", 0, -1.0, 0.1)]
    producer = StateDlProducer(cfg, esm=esm, bags=bags, train_pairs=pairs,
                               input_dim=6, output_dim=6)
    emb, mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert emb.shape[0] == 4
    assert mask.shape == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_dl_model.model'`

- [ ] **Step 3: Write the model module**

```python
# src/sl_dl_model/model.py
"""The exp08 model: frozen-STATE encoder + pooling + symmetric pair head."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sl_dl_model.encoder import StateEncoder
from sl_dl_model.pair_head import SymmetricPairHead
from sl_dl_model.pooling import build_pool, output_dim


class SlDlModel(nn.Module):
    """End-to-end SL-pair model: e_g = pool(STATE(adapter(esm), control))."""

    def __init__(
        self,
        *,
        backend: str,
        checkpoint: Path | None,
        esm_dim: int,
        adapter_hidden: int,
        pert_dim: int,
        input_dim: int,
        output_dim: int,
        pooling: str = "mean_std",
        pair_hidden: tuple[int, ...] = (256, 64),
        include_coverage_flag: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            backend=backend, checkpoint=checkpoint, esm_dim=esm_dim,
            adapter_hidden=adapter_hidden, pert_dim=pert_dim,
            input_dim=input_dim, output_dim=output_dim,
        )
        self.pool = build_pool(pooling, output_dim)
        self.emb_dim = output_dim_fn(pooling, output_dim)
        self.pair_head = SymmetricPairHead(
            emb_dim=self.emb_dim, hidden=pair_hidden,
            include_coverage_flag=include_coverage_flag,
        )

    def embed_gene(
        self, esm_vec: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        bag = self.encoder(esm_vec, control)
        return self.pool(bag)

    def score_pairs(
        self,
        e_a: torch.Tensor,
        e_b: torch.Tensor,
        ge_features: torch.Tensor,
        cov_a: torch.Tensor | None = None,
        cov_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.pair_head(e_a, e_b, ge_features, cov_a, cov_b)


def output_dim_fn(pooling: str, dim: int) -> int:
    """Pooled embedding dimension (wrapper around pooling.output_dim)."""
    return output_dim(pooling, dim)
```

- [ ] **Step 4: Write the training/producer module**

```python
# src/sl_dl_model/train.py
"""Per-fold training loop (Accelerate/DDP, tqdm) and the embedding producer."""

from __future__ import annotations

import logging

import numpy as np
import torch
from accelerate import Accelerator
from torch import optim
from tqdm.auto import tqdm

from sl_dl_model.config import SLDLConfig
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable, align_esm2_to_universe
from sl_dl_model.bags import GwpsBags
from sl_dl_model.losses import bag_loss, combine, distill_loss, sl_bce_loss
from sl_dl_model.model import SlDlModel
from sl_benchmark_baseline.features import build_pair_features

logger = logging.getLogger(__name__)


def _epoch_weights(epoch: int, config: SLDLConfig) -> dict[str, float]:
    if epoch < config.warmup_epochs:
        return {"sl": 0.0, "distill": config.lambda_distill,
                "bag": config.lambda_bag}
    return {"sl": config.lambda_sl,
            "distill": config.lambda_distill_after_warmup,
            "bag": config.lambda_bag}


class StateDlProducer:
    """Train the DL model on a fold's train pairs; emit per-gene e_g + score matrix."""

    def __init__(
        self,
        config: SLDLConfig,
        *,
        esm: Esm2EmbeddingTable,
        bags: GwpsBags,
        train_pairs: list[tuple[str, str, int]],
        input_dim: int,
        output_dim: int,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.train_pairs = train_pairs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._model: SlDlModel | None = None
        self._emb_cache: dict[str, np.ndarray] = {}

    def _build_model(self) -> SlDlModel:
        return SlDlModel(
            backend=self.config.state_backend,
            checkpoint=(None if self.config.state_backend == "linear_mock"
                        else self.config.state_checkpoint),
            esm_dim=self.esm.dim, adapter_hidden=self.config.adapter_hidden,
            pert_dim=self.config.pert_dim, input_dim=self.input_dim,
            output_dim=self.output_dim, pooling=self.config.pooling,
            pair_hidden=self.config.pair_hidden,
            include_coverage_flag=self.config.include_coverage_flag,
        )

    def produce(
        self, symbols: np.ndarray, train_symbols: set[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train on this fold, then embed all universe genes via frozen STATE."""
        torch.manual_seed(self.config.seed)
        accelerator = Accelerator()
        model = self._build_model()
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=self.config.lr
        )
        model, optimizer = accelerator.prepare(model, optimizer)
        self._train(model, optimizer, accelerator, train_symbols)
        self._model = accelerator.unwrap_model(model)

        # Embed all universe genes through frozen STATE (held-out genes included).
        device = accelerator.device
        control = torch.tensor(self.bags.control_template, device=device)
        pooled_dim = self._model.emb_dim
        embeddings = np.zeros((len(symbols), pooled_dim), dtype=float)
        mask = np.zeros(len(symbols), dtype=int)
        self._model.eval()
        with torch.no_grad():
            for row, symbol in enumerate(tqdm(symbols, desc="embed-universe",
                                              disable=not accelerator.is_main_process)):
                key = str(symbol).upper()
                vec = self.esm.vectors_by_symbol.get(key)
                if vec is None:
                    continue  # uncovered ESM2 -> stays zero (fallback)
                esm_vec = torch.tensor(vec, device=device)
                e_g = self._model.embed_gene(esm_vec, control)
                embeddings[row] = e_g.cpu().numpy()
                mask[row] = 1
        return embeddings, mask

    def _train(self, model, optimizer, accelerator, train_symbols) -> None:
        device = accelerator.device
        control = torch.tensor(self.bags.control_template, device=device)
        covered_train = {
            s for s in train_symbols if s in self.bags.bags_by_symbol
        }
        for epoch in range(self.config.max_epochs):
            weights = _epoch_weights(epoch, self.config)
            model.train()
            pbar = tqdm(self.train_pairs, desc=f"epoch {epoch}",
                        disable=not accelerator.is_main_process)
            for a, b, label, ea, eb in pbar:
                key_a, key_b = a.upper(), b.upper()
                vec_a = self.esm.vectors_by_symbol.get(key_a)
                vec_b = self.esm.vectors_by_symbol.get(key_b)
                if vec_a is None or vec_b is None:
                    continue
                inner = accelerator.unwrap_model(model)
                e_a = inner.embed_gene(torch.tensor(vec_a, device=device), control)
                e_b = inner.embed_gene(torch.tensor(vec_b, device=device), control)
                ge = torch.tensor(
                    build_pair_features(np.array([ea]), np.array([eb])),
                    device=device, dtype=torch.float32,
                )
                logit = inner.score_pairs(e_a.unsqueeze(0), e_b.unsqueeze(0), ge)
                parts = {
                    "sl": sl_bce_loss(logit, torch.tensor([float(label)],
                                                          device=device)),
                }
                # bag supervision: covered train genes only (leakage rule)
                for key, vec in ((key_a, vec_a), (key_b, vec_b)):
                    if key in covered_train and weights["bag"] > 0:
                        pred = inner.encoder(torch.tensor(vec, device=device),
                                             control)
                        real = torch.tensor(self.bags.bags_by_symbol[key],
                                            device=device)
                        parts["bag"] = parts.get("bag", 0.0) + bag_loss(pred, real)
                total = combine(parts, weights)
                optimizer.zero_grad()
                accelerator.backward(total)
                optimizer.step()
```

> **NOTE for implementer (performance + correctness):** the per-pair loop above is the
> minimal correct form. Two refinements the implementer should apply: (1) batch pairs
> (`config.batch_pairs`) rather than one at a time; (2) cache `e_g` per unique gene per
> optimizer step — genes, not pairs, drive STATE cost, so embedding each gene once per
> step and indexing into that cache is far cheaper. The `ea`/`eb` GeneEffect values are
> already threaded through `train_pairs` as 5-tuples `(a, b, label, ea, eb)`; the score
> -matrix path (Task 2.4) reads `universe.gene_effects`. Keep the pair head's GeneEffect
> block construction identical between train and score-matrix so the scorer is consistent.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_train.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/model.py src/sl_dl_model/train.py tests/sl_dl_model/test_train.py
git commit -m "feat: add exp08 SlDlModel and StateDlProducer training loop"
```

### Task 2.4: DL score matrix + wire producer into scoring

**Files:**
- Modify: `src/sl_dl_model/train.py` (add `score_matrix` to `StateDlProducer`)
- Modify: `src/sl_dl_model/scoring.py` (use DL score matrix when producer provides one)
- Test: `tests/sl_dl_model/test_dl_score_matrix.py`

**Interfaces:**
- Produces: `StateDlProducer.score_matrix(universe) -> np.ndarray (n_gene, n_gene)`,
  diagonal zeroed, built by the trained pair head over cached `e_g`.
- Modifies `run_fold_with_producer`: if `producer` has `score_matrix`, use it for the
  `state_dl` model row instead of the sklearn `_transcript` models.

- [ ] **Step 1: Write the failing test**

```python
# tests/sl_dl_model/test_dl_score_matrix.py
import numpy as np

from sl_dl_model.config import SLDLConfig
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.bags import GwpsBags
from sl_dl_model.train import StateDlProducer


def test_score_matrix_diag_zero_and_shape():
    symbols = np.array(["A", "B", "C"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8, vectors_by_symbol={s: np.random.randn(8).astype("float32")
                                  for s in ["A", "B", "C"]})
    bags = GwpsBags(control_template=np.random.randn(8, 6).astype("float32"),
                    bags_by_symbol={}, input_dim=6)
    cfg = SLDLConfig(max_epochs=1, warmup_epochs=0, pert_dim=5, adapter_hidden=16,
                     pair_hidden=(16,), include_coverage_flag=False,
                     state_backend="linear_mock")
    pairs = [("A", "B", 1, -1.0, -0.5), ("B", "C", 0, -0.5, 0.2)]
    producer = StateDlProducer(cfg, esm=esm, bags=bags, train_pairs=pairs,
                               input_dim=6, output_dim=6)
    gene_effects = np.array([-1.0, -0.5, 0.2])
    sm = producer.score_matrix(symbols, gene_effects)
    assert sm.shape == (3, 3)
    assert np.allclose(np.diag(sm), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_dl_score_matrix.py -v`
Expected: FAIL (`StateDlProducer` has no `score_matrix`, or signature mismatch)

- [ ] **Step 3: Implement `score_matrix`**

Add to `StateDlProducer` in `src/sl_dl_model/train.py`. Update `train_pairs` typing to
carry `(a, b, label, ea, eb)` and fix the training `ge` placeholder per the NOTE:

```python
    def score_matrix(
        self, symbols: np.ndarray, gene_effects: np.ndarray
    ) -> np.ndarray:
        """Score all candidate pairs with the trained pair head over cached e_g."""
        import torch
        from sl_benchmark_baseline.features import build_pair_features

        if self._model is None:
            self.produce(symbols, {a.upper() for a, *_ in self.train_pairs}
                         | {b.upper() for _, b, *_ in self.train_pairs})
        model = self._model
        device = next(model.parameters()).device
        control = torch.tensor(self.bags.control_template, device=device)
        n = len(symbols)
        e_table = torch.zeros((n, model.emb_dim), device=device)
        model.eval()
        with torch.no_grad():
            for i, symbol in enumerate(symbols):
                vec = self.esm.vectors_by_symbol.get(str(symbol).upper())
                if vec is not None:
                    e_table[i] = model.embed_gene(
                        torch.tensor(vec, device=device), control)
            score = np.zeros((n, n), dtype=float)
            for i in range(n):
                ea = np.full(n, gene_effects[i])
                eb = gene_effects
                ge = torch.tensor(build_pair_features(ea, eb),
                                  device=device, dtype=torch.float32)
                e_a = e_table[i].unsqueeze(0).expand(n, -1)
                logits = model.score_pairs(e_a, e_table, ge)
                score[i] = torch.sigmoid(logits).cpu().numpy()
        np.fill_diagonal(score, 0.0)
        return score
```

- [ ] **Step 4: Wire into `run_fold_with_producer` + add per-fold producer factory**

First add the cache loader to `src/sl_dl_model/evaluate.py` and the per-fold factory
to `src/sl_dl_model/scoring.py` (referenced by `run_cv`'s `state_dl` path):

```python
# src/sl_dl_model/evaluate.py  (append)
from dataclasses import dataclass


@dataclass(frozen=True)
class StateDlCaches:
    """Shared, fold-independent caches for the state_dl producer."""

    esm: object  # Esm2EmbeddingTable
    bags: object  # GwpsBags
    input_dim: int
    output_dim: int


def _load_state_dl_caches(config: SLDLConfig) -> StateDlCaches:
    """Load ESM2 + gwps-bags caches once (shared across folds)."""
    from sl_dl_model.gene_embeddings import load_esm2_embeddings
    from sl_dl_model.bags import build_gwps_bags, load_bags_npz

    if config.esm2_npz is None:
        raise ValueError("state_dl producer requires config.esm2_npz")
    esm = load_esm2_embeddings(config.esm2_npz)
    if config.bags_npz is not None and Path(config.bags_npz).exists():
        bags = load_bags_npz(config.bags_npz)
    else:
        bags = build_gwps_bags(config, rng_seed=config.seed)
    return StateDlCaches(esm=esm, bags=bags, input_dim=bags.input_dim,
                         output_dim=bags.input_dim)
```

```python
# src/sl_dl_model/scoring.py  (append)
def make_fold_producer(config, caches, frame, split_type, fold_id):
    """Build a fold-specific StateDlProducer from shared caches + fold train pairs."""
    from sl_dl_model.train import StateDlProducer

    train_df, _ = fold_split(frame, split_type, fold_id)
    train_df = train_df  # train pairs only — leakage rule
    train_pairs = [
        (
            str(r["gene_a_symbol"]).upper(), str(r["gene_b_symbol"]).upper(),
            int(r["sl_label"]),
            float(r["gene_a_k562_gene_effect"]),
            float(r["gene_b_k562_gene_effect"]),
        )
        for _, r in train_df.iterrows()
    ]
    return StateDlProducer(
        config, esm=caches.esm, bags=caches.bags, train_pairs=train_pairs,
        input_dim=caches.input_dim, output_dim=caches.output_dim,
    )
```

Then update `run_fold_with_producer` so the DL score matrix is used when available:

```python
    if hasattr(producer, "score_matrix"):
        sm = producer.score_matrix(universe.symbols, universe.gene_effects)
        rows.extend(_metric_rows(split_type, "state_dl", fold_id, "full_universe",
                                 sm, pos_index, neg_index, seen_index,
                                 config.ranking_k))
        if len(pos_cov) > 0 and len(neg_cov) > 0:
            rows.extend(_metric_rows(split_type, "state_dl", fold_id,
                                     "covered_pairs", sm, pos_cov, neg_cov,
                                     seen_index, config.ranking_k))
        return rows
```

Note: `_covered_pair_mask` requires `universe.coverage_mask`. Compute `pos_cov`/
`neg_cov` after the producer has set the universe embeddings/mask (the DL producer's
`produce` sets `mask=1` for genes with an ESM2 vector). For the `score_matrix`-only
path, call `producer.produce(universe.symbols, train_symbols)` first to populate the
universe mask, then build `pos_cov`/`neg_cov`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_dl_score_matrix.py tests/sl_dl_model/test_scoring.py -v`
Expected: PASS

- [ ] **Step 6: Create the Phase 2 config**

```yaml
# configs/experiments/08_k562_sl_pair_state_dl/phase2_bce.yaml
# exp08 Phase 2 — SL BCE only (lambda_bag=0): classification gate before bag-sup.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08_k562_sl_pair_state_dl/phase2_bce
split_types: [CV2]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
lambda_sl: 1.0
lambda_distill: 0.5
lambda_distill_after_warmup: 0.1
lambda_bag: 0.0
warmup_epochs: 2
max_epochs: 20
batch_pairs: 1024
lr: 0.001
include_coverage_flag: false
fallback_strategy: zero
```

- [ ] **Step 7: Phase 2 gate (manual, real data)**

Run the DL on CV2 with a real ESM2 cache + STATE checkpoint; confirm classification
beats exp06:
```bash
uv run python -m sl_dl_model run-cv --config configs/experiments/08_k562_sl_pair_state_dl/phase2_bce.yaml --producer state_dl
```
**Gate:** CV2 `auroc` > 0.704 and `aupr` > 0.732 on the `full_universe` slice. If not,
inspect the covered-pair slice — if covered AUROC is high but full is not, the signal
is real but diluted (continue to Phase 3); if covered AUROC is also low, debug before
proceeding.

- [ ] **Step 8: Commit**

```bash
git add src/sl_dl_model/train.py src/sl_dl_model/scoring.py src/sl_dl_model/evaluate.py configs/experiments/08_k562_sl_pair_state_dl/phase2_bce.yaml tests/sl_dl_model/test_dl_score_matrix.py
git commit -m "feat: add exp08 DL score matrix, per-fold producer factory, Phase 2 config"
```

---

## Phase 3 — Bag supervision (primary gate)

Goal: bag supervision is already wired in `train.py` (Task 2.3); this phase confirms
the config enables `lambda_bag > 0` and gates on the primary success criterion.
**Gate:** beat exp06 on CV2/CV3 official ranking (NDCG@k, MAP@k), with lift
concentrated on the covered-pair slice.

### Task 3.1: Config + gate verification

**Files:**
- Create: `configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml`

**Interfaces:**
- Verifies: `lambda_bag=1.0`, `warmup_epochs > 0`, full pipeline CV2/CV3.

- [ ] **Step 1: Write the config**

```yaml
# configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml
# exp08 Phase 3 — full 3-part loss with bag supervision (primary gate).
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08_k562_sl_pair_state_dl/phase3_bag_sup
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
lambda_sl: 1.0
lambda_distill: 0.5
lambda_distill_after_warmup: 0.1
lambda_bag: 1.0
warmup_epochs: 3
max_epochs: 20
batch_pairs: 1024
lr: 0.001
include_coverage_flag: false
fallback_strategy: zero
```

- [ ] **Step 2: Run Phase 3 gate**

Run:
```bash
uv run python -m sl_dl_model run-cv --config configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml --producer state_dl
```

Expected: `results/experiments/08_k562_sl_pair_state_dl/phase3_bag_sup/summary.csv`
with CV2/CV3 metrics for `state_dl` model. Inspect covered-pair slice first (the lift
should be larger there); then check the full-universe slice. **Primary gate:** CV2
`ndcg@10` > 0.042 and/or CV3 `ndcg@10` > 0.002. If CV2/CV3 are within exp06 fold
noise, document the null result and stop; if covered slice shows strong signal but
full is diluted, the premise is validated (covered genes benefit, uncovered dilute).

- [ ] **Step 3: Commit**

```bash
git add configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml
git commit -m "feat: add exp08 Phase 3 bag-supervision config and gate"
```

---

## Phase 4 — Robustness & ablations

Goal: coverage-flag ablation, pooling swap, optional RankNet, reporting polish. This
is where exp08's honesty checks (spec §7) run and the final artifact structure locks.

### Task 4.1: Coverage-flag ablation config

**Files:**
- Create: `configs/experiments/08_k562_sl_pair_state_dl/ablation_coverage_flag.yaml`

**Interfaces:**
- Flips `include_coverage_flag: true`; otherwise identical to Phase 3 config.

- [ ] **Step 1: Write the config**

```yaml
# configs/experiments/08_k562_sl_pair_state_dl/ablation_coverage_flag.yaml
# exp08 coverage-flag ablation (honesty check against degree proxy).
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
output_dir: results/experiments/08_k562_sl_pair_state_dl/ablation_coverage_flag
split_types: [CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
esm2_npz: data/esm2/k562_sl_universe_esm2_650M.npz
state_backend: state_checkpoint
gwps_h5ad: data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad
bags_npz: data/exp08_cache/k562_gwps_bags.npz
lambda_sl: 1.0
lambda_distill: 0.5
lambda_distill_after_warmup: 0.1
lambda_bag: 1.0
warmup_epochs: 3
max_epochs: 20
batch_pairs: 1024
lr: 0.001
include_coverage_flag: true  # <-- ablation flip
fallback_strategy: zero
```

- [ ] **Step 2: Run and compare**

Run: `uv run python -m sl_dl_model run-cv --config configs/.../ablation_coverage_flag.yaml --producer state_dl`

Compare to Phase 3 (no flag): `uv run python -c "import pandas as pd; ..."`
Expected: the coverage-flag version may show higher metrics if coverage correlates
with SL-graph degree (a shortcut). Report both; the no-flag version is the honest one.

- [ ] **Step 3: Commit**

```bash
git add configs/experiments/08_k562_sl_pair_state_dl/ablation_coverage_flag.yaml
git commit -m "feat: add exp08 coverage-flag ablation config"
```

### Task 4.2: Slurm wrapper + README

**Files:**
- Create: `scripts/sl_dl_model.sh`
- Create: `configs/experiments/08_k562_sl_pair_state_dl/README.md`

**Interfaces:**
- Slurm script wraps `accelerate launch` + `uv run python -m sl_dl_model run-cv`.
- README documents the Phase 0–4 configs, when to run each, and how to interpret gates.

- [ ] **Step 1: Write the Slurm wrapper**

```bash
# scripts/sl_dl_model.sh
#!/usr/bin/env bash
#SBATCH --job-name=sl_dl_model
#SBATCH --output=logs/sl_dl_model_%j.out
#SBATCH --error=logs/sl_dl_model_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

CONFIG="${1:?usage: $0 <config.yaml>}"
PRODUCER="${2:-state_dl}"

accelerate launch --multi_gpu --num_processes=4 \
    -m sl_dl_model run-cv --config "$CONFIG" --producer "$PRODUCER"
```

- [ ] **Step 2: Write the README**

```markdown
# exp08 — STATE-Adapter DL Model for K562 SL-Pair Ranking

Configs for the 5-phase exp08 implementation (see `docs/superpowers/specs/2026-06-17-exp08-state-dl-sl-ranking-design.md`).

## Prerequisites

1. Run the ESM2 precompute script once (network node):
   ```bash
   uv run python scripts/precompute_esm2_embeddings.py \
       --benchmark-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
       --out data/esm2/k562_sl_universe_esm2_650M.npz \
       --seq-cache data/esm2/symbol_to_sequence.json
   ```

2. Build and cache gwps bags (local or interactive node):
   ```bash
   uv run python -c "
   from pathlib import Path
   from sl_dl_model.bags import build_gwps_bags, save_bags_npz
   from sl_dl_model.config import SLDLConfig
   cfg = SLDLConfig()
   bags = build_gwps_bags(cfg)
   save_bags_npz(bags, Path('data/exp08_cache/k562_gwps_bags.npz'))
   "
   ```

## Phase 0 — Harness parity (exp06 in-harness baseline)

Run: `uv run python -m sl_dl_model run-cv --config phase0_parity.yaml --producer zero`

**Gate:** CV2 NDCG@10 ≈ 0.042, CV3 ≈ 0.002 (match exp06 XGB). If not, harness is wrong.

## Phase 2 — SL classifier (BCE)

Run: `sbatch ../../scripts/sl_dl_model.sh phase2_bce.yaml state_dl`

**Gate:** CV2 AUROC > 0.704, AUPR > 0.732. If not, debug encoder/pair head before Phase 3.

## Phase 3 — Bag supervision (primary)

Run: `sbatch ../../scripts/sl_dl_model.sh phase3_bag_supervision.yaml state_dl`

**Gate (primary):** CV2/CV3 NDCG@k and MAP@k beat exp06; lift concentrated on covered-pair slice. This is the pass/fail for exp08.

## Phase 4 — Ablations

- Coverage-flag: `ablation_coverage_flag.yaml` (report both; no-flag is the honest one).
- Pooling swap: duplicate `phase3_bag_supervision.yaml`, set `pooling: gmm` (not implemented yet; mean_std default).
- RankNet: set `lambda_rank: 1.0` in a duplicate config if BCE underperformed on NDCG.

## Interpreting Results

Read `summary.csv` → filter `metric=="ndcg@10"` and `slice=="full_universe"` → compare mean ± std against exp06 (0.042 ± 0.008 for CV2 XGB). Lift within fold noise is null; lift concentrated on covered-pair slice validates the premise but documents the uncovered-gene dilution.
```

- [ ] **Step 3: Make Slurm wrapper executable and commit**

```bash
chmod +x scripts/sl_dl_model.sh
git add scripts/sl_dl_model.sh configs/experiments/08_k562_sl_pair_state_dl/README.md
git commit -m "docs: add exp08 Slurm wrapper and experiment README"
```

### Task 4.3: Update top-level docs + spec linkage

**Files:**
- Modify: `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md` (create if missing)
- Modify: `CONTEXT.md` (add exp08 pointer)

**Interfaces:**
- The experiment doc follows the pattern of `docs/experiment/06_*.md` and `07_*.md`:
  rationale, task definitions, data, model, evaluation, results-to-beat, config paths.

- [ ] **Step 1: Write experiment doc**

```markdown
# Experiment 08 — STATE-Adapter DL Model for K562 SL-Pair Ranking

**Status:** Implementation complete (Phase 0–4).
**Design spec:** `docs/superpowers/specs/2026-06-17-exp08-state-dl-sl-ranking-design.md`.
**Configs:** `configs/experiments/08_k562_sl_pair_state_dl/`.
**Package:** `src/sl_dl_model/`.

## Rationale

exp06 (dependency-only) and exp07 (real-bag features) establish the SL-pair baseline.
exp08 asks: can a frozen-STATE encoder + a trainable adapter fed by ESM2 gene
embeddings produce a transcriptomic signal that beats exp06 on CV2/CV3 per-anchor
ranking, and generalizes to held-out genes?

The local STATE checkpoint is a closed-vocabulary one-hot model (2,024 perturbation
genes, 16.3% of the SL universe). exp08 replaces STATE's one-hot `pert_encoder` with
a trainable adapter that consumes ESM2 protein embeddings, keeping the 8-layer Llama
backbone frozen. All 9,471 genes flow through one coordinate system; real gwps bags
supervise the covered train genes only (leakage-free CV2/CV3).

## Task Definitions

Identical to exp06/07 (spec §1). Classification: `(a,b) → sl_label`. Ranking
(primary): anchor `a` → rank all 9,471 candidate partners, evaluate against held-out
positives, seen-masking + diagonal-zero.

## Data

- SL pairs: `data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv`
  (CV1/CV2/CV3 Rand 1:1, 9,471-gene universe).
- ESM2 embeddings: `data/esm2/k562_sl_universe_esm2_650M.npz` (precomputed,
  `scripts/precompute_esm2_embeddings.py`, UniProt + HF `esm2_t33_650M_UR50D`).
- gwps bags: `data/exp08_cache/k562_gwps_bags.npz` (cached from
  `data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad`,
  6,070 covered genes).
- STATE checkpoint: `model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/`.

## Model

**Architecture:** `PertAdapter` (ESM2 1280-d → 328-d pert token) → frozen STATE
backbone (8-layer Llama) → predicted response bag → `MeanStdPool` (bag → e_g) →
`SymmetricPairHead` ([e_a+e_b, |e_a−e_b|, e_a⊙e_b] + GeneEffect block → logit).

**Training:** 3-part loss (SL BCE + adapter token-distill + real-bag supervision),
warmup schedule, Accelerate/DDP. Bag supervision for covered train genes only
(leakage-free held-out gene eval). Seed 17, max 20 epochs, lr 1e-3.

## Evaluation

Reuses `sl_benchmark_baseline.metrics.official_*_metrics` verbatim. Primary: CV2/CV3
per-anchor NDCG@k and MAP@k. Honesty checks: covered-pair diagnostic slice,
coverage-flag ablation, effect-size ± std reporting.

## Results to Beat (exp06 XGB, 5-fold mean)

| Split | AUROC | AUPR | NDCG@10 | MAP@10 |
| --- | ---: | ---: | ---: | ---: |
| CV2 | 0.704 | 0.732 | 0.042 | 0.034 |
| CV3 | 0.596 | — | 0.002 | — |

exp08 must beat these on the full-universe slice with lift concentrated on the
covered-pair slice. Within-noise lift is null.

## Implementation Phases

See `configs/experiments/08_k562_sl_pair_state_dl/README.md`.
```

- [ ] **Step 2: Add exp08 pointer to CONTEXT.md**

Append under the exp06/exp07 section:

```markdown
- **exp08 (STATE-adapter DL):** `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md`.
  Frozen STATE + trainable ESM2 pert-adapter; 3-part loss (BCE + distill + bag);
  DDP. Targets CV2/CV3 NDCG/MAP lift over exp06.
```

- [ ] **Step 3: Commit**

```bash
git add docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md CONTEXT.md
git commit -m "docs: add exp08 experiment documentation and CONTEXT pointer"
```

---

## Final Verification & Self-Review

After all tasks, run the full test suite and confirm every phase config is present.

- [ ] **Step 1: Run full test suite**

Run: `uv run python -m pytest tests/sl_dl_model/ -v`
Expected: all tests PASS.

- [ ] **Step 2: Confirm config coverage**

Run: `ls -1 configs/experiments/08_k562_sl_pair_state_dl/`
Expected files:
- `phase0_parity.yaml`
- `phase2_bce.yaml`
- `phase3_bag_supervision.yaml`
- `ablation_coverage_flag.yaml`
- `README.md`

- [ ] **Step 3: Verify CLI help**

Run: `uv run python -m sl_dl_model --help`
Expected: `run-cv` listed; `--producer {zero,state_dl}` documented.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete exp08 STATE-adapter DL SL-pair ranking pipeline"
```

---

## Summary

**Implemented:**
- 10 new modules under `src/sl_dl_model/` (config, gene_embeddings, encoder, pooling,
  pair_head, losses, bags, model, train, evaluate, scoring, __main__).
- Precompute script `scripts/precompute_esm2_embeddings.py` (UniProt + ESM2).
- 5 phase configs (0: parity, 1: plumbing gate, 2: BCE gate, 3: bag-sup primary gate,
  4: ablations).
- Full test coverage (12+ test modules).
- Slurm wrapper + experiment README + top-level docs.

**Verification gates:**
- Phase 0: exp06 CV2/CV3 numbers reproduced within fold noise.
- Phase 1: `e_g` finite and gene-varying (manual smoke).
- Phase 2: CV2 AUROC > 0.704 / AUPR > 0.732.
- Phase 3 (primary): CV2/CV3 NDCG@k / MAP@k beat exp06, lift on covered-pair slice.

**Next step:** run Phase 3 on real data (requires ESM2 cache + STATE checkpoint +
gwps bags NPZ, all gitignored). If Phase 3 gate passes, exp08 succeeds; if not,
document the null result and the covered-slice diagnostic as the honest outcome.



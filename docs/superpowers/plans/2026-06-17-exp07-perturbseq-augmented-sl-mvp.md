# Exp07 Perturb-seq-Augmented K562 SL-Pair Baseline (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Augment the exp06 dependency-only SL-pair baseline with observed Replogle gwps Perturb-seq per-gene embeddings, and measure head-to-head lift over exp06 under the identical official ranking + classification protocol.

**Architecture:** Extend the existing `src/sl_benchmark_baseline/` module with an *augmented mode* triggered when a pooled cell-bag NPZ is supplied. A new `embeddings.py` pools an exp03 cell-bags NPZ (mean over each gene's delta-cell bag) into one per-gene vector, aligns it to the SL candidate universe with a coverage mask and label-free fallback, and feeds swap-invariant transcript pair-features alongside the existing 5 GeneEffect features into the same LogReg/XGB heads. In augmented mode the run emits both the dependency-only baseline (A, B) and the transcript-augmented models (A_transcript, B_transcript), plus a covered-pair diagnostic metric slice. When no NPZ is supplied the module behaves exactly like exp06 (existing tests stay green).

**Tech Stack:** Python 3.11+, numpy, pandas, scikit-learn, xgboost, anndata (test fixtures only), pytest, uv, ruff.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings (project `CLAUDE.md`).
- Target <50 lines per function, <600 lines per file; no `print` in library code (use `logging`); no bare `except`; no hardcoded paths/thresholds (use config).
- Prefix every Python/pytest/ruff invocation with `uv run`.
- Standalone module `src/sl_benchmark_baseline/`; no new `vcc-dep-baseline` subcommand.
- Benchmark, metric protocol, and CV1/CV2/CV3 splits are UNCHANGED from exp06. Seed `17`. `ranking_k = (10, 20, 50)`. Candidate universe = the K562-filtered gene universe present in the input CSV (9,471 genes in production).
- MVP pooling op is **mean-pool of bag delta vectors** only. Tier 1 = PCA-delta bags NPZ; Tier 2 = scVI-delta bags NPZ — same code, different `bags_npz` artifact. Frozen-GMM-occupancy pooling is OUT of MVP scope (follow-up).
- Fallback for uncovered genes is label-free (`zero` or `global_mean` of covered-gene vectors), computed once at universe build — no fold-coupling, no leakage.
- Augmented mode runs models A, B (dependency-only baseline) and A_transcript, B_transcript. The degree probe C is NOT run in augmented mode (already characterized in exp06).
- Scope guard (from spec): no predicted/generated transcriptome bags; no context-specific SL claims; no changing the `D` label / `C` definition / exp03 mapping; gwps transcriptome only.

---

## File Structure

**Create:**
- `src/sl_benchmark_baseline/embeddings.py` — pool a cell-bags NPZ into per-gene vectors; align to the candidate universe; build coverage mask + fallback. Holds `GeneEmbeddingTable`.
- `tests/test_sl_benchmark_baseline_embeddings.py` — embedding pooling, alignment, coverage mask, fallback.
- `configs/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_cv1_cv2_cv3.yaml` — production run config (paths + embedding params).

**Modify:**
- `src/sl_benchmark_baseline/config.py` — add 4 optional augmentation fields (defaults keep exp06 behavior).
- `src/sl_benchmark_baseline/features.py` — add `transcript_feature_names`, `build_transcript_pair_features`, `build_augmented_pair_features`.
- `src/sl_benchmark_baseline/models.py` — add `LogRegTranscriptModel`, `XGBTranscriptModel`, `build_augmented_models`.
- `src/sl_benchmark_baseline/evaluate.py` — augmented `GeneUniverse` (carry embeddings + coverage), augmented score-matrix path, covered-pair slice in `run_fold`, manifest fields, `run_cv` branch.
- `src/sl_benchmark_baseline/__main__.py` — add `--bags-npz`, `--embedding-method`, `--fallback-strategy`, `--include-coverage-flag` CLI flags.
- `tests/conftest.py` — add `synthetic_bags_npz` fixture (tiny in-memory pooled bags).
- `tests/test_sl_benchmark_baseline_features.py` — transcript pair-feature tests.
- `tests/test_sl_benchmark_baseline_models.py` — transcript model interface tests.
- `tests/test_sl_benchmark_baseline_evaluate.py` — augmented run + slice + manifest tests.
- `docs/experiment/07_k562_sl_pair_perturbseq_augmented.md` — flip Run status to implemented; record results.

**Build order:** config → features → embeddings → models → evaluate → CLI → config-yaml/docs. Each task is independently testable and green before the next.

---

### Task 1: Augmentation config fields

**Files:**
- Modify: `src/sl_benchmark_baseline/config.py:9-43`
- Test: `tests/test_sl_benchmark_baseline_config.py`

**Interfaces:**
- Consumes: nothing (leaf).
- Produces: `SLBaselineConfig` gains optional fields `bags_npz: Path | None = None`, `embedding_method: str = "pca_delta_meanpool"`, `fallback_strategy: str = "zero"`, `include_coverage_flag: bool = True`. `augmented` is a read-only property: `bags_npz is not None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_config.py`:

```python
def test_config_augmentation_defaults_preserve_exp06():
    config = SLBaselineConfig()
    assert config.bags_npz is None
    assert config.augmented is False
    assert config.embedding_method == "pca_delta_meanpool"
    assert config.fallback_strategy == "zero"
    assert config.include_coverage_flag is True


def test_config_augmented_property_true_when_bags_set(tmp_path):
    config = SLBaselineConfig(bags_npz=tmp_path / "bags.npz")
    assert config.augmented is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_config.py::test_config_augmentation_defaults_preserve_exp06 -v`
Expected: FAIL with `AttributeError: 'SLBaselineConfig' object has no attribute 'bags_npz'`.

- [ ] **Step 3: Write minimal implementation**

In `src/sl_benchmark_baseline/config.py`, add fields after line 43 (inside the dataclass, after `xgb_learning_rate`) and an `augmented` property. Add to the docstring Attributes block too:

```python
    bags_npz: Path | None = None
    embedding_method: str = "pca_delta_meanpool"
    fallback_strategy: str = "zero"
    include_coverage_flag: bool = True

    @property
    def augmented(self) -> bool:
        """True when a cell-bags NPZ is supplied (exp07 augmented mode)."""
        return self.bags_npz is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_config.py -v`
Expected: PASS (all config tests, including pre-existing frozen/override tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/config.py tests/test_sl_benchmark_baseline_config.py
git commit -m "feat: add exp07 augmentation config fields"
```

### Task 2: Transcript pair-features (swap-invariant)

**Files:**
- Modify: `src/sl_benchmark_baseline/features.py:9-38`
- Test: `tests/test_sl_benchmark_baseline_features.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces:
  - `transcript_feature_names(dim, include_coverage_flag) -> tuple[str, ...]` — names for the transcript block: `sum_0..sum_{d-1}`, `absdiff_0..`, `prod_0..` (length `3*dim`), plus `cov_min`, `cov_max` when `include_coverage_flag`. (Lowercase: it is a function, matching the project's `build_*` convention; the existing `FEATURE_NAMES` stays a constant.)
  - `build_transcript_pair_features(emb_a, emb_b, flag_a, flag_b, include_coverage_flag) -> np.ndarray` shape `(n, 3*dim + (2 if flag else 0))`. Columns: `[emb_a+emb_b | |emb_a-emb_b| | emb_a*emb_b | (flag_a, flag_b)?]`.
  - `build_augmented_pair_features(ea, eb, emb_a, emb_b, flag_a, flag_b, include_coverage_flag) -> np.ndarray` shape `(n, 5 + 3*dim + cov)` = `[build_pair_features(ea,eb) | build_transcript_pair_features(...)]`.

Note: coverage columns use `flag_a + flag_b` and `flag_a * flag_b`? NO — keep them as the two raw per-gene flags but order-sort them so the pair feature is swap-invariant: column `cov_a = min(flag_a,flag_b)`, `cov_b = max(flag_a,flag_b)`. This preserves both-covered (1,1), one-covered (0,1), none (0,0) while staying symmetric.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_features.py`:

```python
import numpy as np
from sl_benchmark_baseline.features import (
    build_pair_features,
    build_transcript_pair_features,
    build_augmented_pair_features,
    transcript_feature_names,
)


def test_transcript_pair_features_are_swap_invariant():
    emb_a = np.array([[1.0, 2.0], [0.5, -1.0]])
    emb_b = np.array([[3.0, -1.0], [2.0, 4.0]])
    flag_a = np.array([1.0, 1.0])
    flag_b = np.array([0.0, 1.0])
    forward = build_transcript_pair_features(
        emb_a, emb_b, flag_a, flag_b, include_coverage_flag=True
    )
    swapped = build_transcript_pair_features(
        emb_b, emb_a, flag_b, flag_a, include_coverage_flag=True
    )
    np.testing.assert_allclose(forward, swapped)


def test_transcript_pair_features_shape_and_names():
    emb_a = np.zeros((4, 8))
    emb_b = np.zeros((4, 8))
    flag = np.ones(4)
    feats = build_transcript_pair_features(
        emb_a, emb_b, flag, flag, include_coverage_flag=True
    )
    assert feats.shape == (4, 3 * 8 + 2)
    assert len(transcript_feature_names(8, include_coverage_flag=True)) == 3 * 8 + 2
    assert len(transcript_feature_names(8, include_coverage_flag=False)) == 3 * 8


def test_transcript_pair_features_omit_coverage_flag():
    emb_a = np.zeros((2, 5))
    emb_b = np.zeros((2, 5))
    flag = np.ones(2)
    feats = build_transcript_pair_features(
        emb_a, emb_b, flag, flag, include_coverage_flag=False
    )
    assert feats.shape == (2, 15)


def test_augmented_pair_features_concatenate_geneeffect_then_transcript():
    ea = np.array([-1.0, 0.2])
    eb = np.array([-0.8, 0.3])
    emb_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    emb_b = np.array([[0.0, 1.0], [2.0, 2.0]])
    flag = np.ones(2)
    feats = build_augmented_pair_features(
        ea, eb, emb_a, emb_b, flag, flag, include_coverage_flag=True
    )
    # 5 gene-effect + (3*2 transcript) + 2 coverage = 13
    assert feats.shape == (2, 13)
    np.testing.assert_allclose(feats[:, :5], build_pair_features(ea, eb))


def test_augmented_pair_features_are_swap_invariant():
    ea = np.array([-1.0]); eb = np.array([0.4])
    emb_a = np.array([[1.0, -2.0]]); emb_b = np.array([[0.5, 3.0]])
    flag_a = np.array([1.0]); flag_b = np.array([0.0])
    forward = build_augmented_pair_features(
        ea, eb, emb_a, emb_b, flag_a, flag_b, include_coverage_flag=True
    )
    swapped = build_augmented_pair_features(
        eb, ea, emb_b, emb_a, flag_b, flag_a, include_coverage_flag=True
    )
    np.testing.assert_allclose(forward, swapped)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_features.py::test_transcript_pair_features_are_swap_invariant -v`
Expected: FAIL with `ImportError: cannot import name 'build_transcript_pair_features'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/sl_benchmark_baseline/features.py`:

```python
def transcript_feature_names(
    dim: int, include_coverage_flag: bool
) -> tuple[str, ...]:
    """Column names for the transcript pair-feature block."""
    names = (
        [f"sum_{i}" for i in range(dim)]
        + [f"absdiff_{i}" for i in range(dim)]
        + [f"prod_{i}" for i in range(dim)]
    )
    if include_coverage_flag:
        names += ["cov_min", "cov_max"]
    return tuple(names)


def build_transcript_pair_features(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    flag_a: np.ndarray,
    flag_b: np.ndarray,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Build swap-invariant transcript features from two per-gene embeddings.

    Args:
        emb_a: Per-pair embedding for gene a, shape ``(n, dim)``.
        emb_b: Per-pair embedding for gene b, shape ``(n, dim)``.
        flag_a: Coverage indicator for gene a, shape ``(n,)``.
        flag_b: Coverage indicator for gene b, shape ``(n,)``.
        include_coverage_flag: Whether to append swap-invariant coverage columns.

    Returns:
        Feature matrix of shape ``(n, 3*dim + (2 if include_coverage_flag else 0))``.
    """
    emb_a = np.asarray(emb_a, dtype=float)
    emb_b = np.asarray(emb_b, dtype=float)
    blocks = [emb_a + emb_b, np.abs(emb_a - emb_b), emb_a * emb_b]
    if include_coverage_flag:
        flag_a = np.asarray(flag_a, dtype=float)
        flag_b = np.asarray(flag_b, dtype=float)
        blocks.append(
            np.column_stack(
                [np.minimum(flag_a, flag_b), np.maximum(flag_a, flag_b)]
            )
        )
    return np.column_stack(blocks)


def build_augmented_pair_features(
    ea: np.ndarray,
    eb: np.ndarray,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    flag_a: np.ndarray,
    flag_b: np.ndarray,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Concatenate the GeneEffect block and the transcript block."""
    return np.column_stack(
        [
            build_pair_features(ea, eb),
            build_transcript_pair_features(
                emb_a, emb_b, flag_a, flag_b, include_coverage_flag
            ),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_features.py -v`
Expected: PASS (new tests + pre-existing swap-invariance/standardizer tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/features.py tests/test_sl_benchmark_baseline_features.py
git commit -m "feat: add swap-invariant transcript pair features"
```

### Task 3: Gene-embedding table (pool bags, align to universe, fallback)

**Files:**
- Create: `src/sl_benchmark_baseline/embeddings.py`
- Create: `tests/test_sl_benchmark_baseline_embeddings.py`
- Modify: `tests/conftest.py` (add `synthetic_bags_npz` fixture)

**Interfaces:**
- Consumes: a cell-bags NPZ written by exp03 `build_cell_bags` containing keys `cell_delta_pcs` `(total_cells, dim)` float32, `bag_offsets` `(n_bags+1,)` int64, `perturbation_gene` `(n_bags,)` object. (Tier 1 = PCA-delta NPZ; Tier 2 = scVI-delta NPZ; same keys.)
- Produces:
  - `GeneEmbeddingTable` frozen dataclass: `dim: int`, `vectors_by_symbol: dict[str, np.ndarray]` (covered genes only).
  - `load_gene_embeddings(bags_npz: Path) -> GeneEmbeddingTable` — mean-pool each gene's bag rows into one `(dim,)` vector keyed by uppercased gene symbol.
  - `align_to_universe(table, symbols, fallback_strategy) -> tuple[np.ndarray, np.ndarray]` returns `(embeddings (n_gene, dim), coverage_mask (n_gene,) float)`. Covered rows = pooled vector + flag 1.0; uncovered = fallback + flag 0.0. `fallback_strategy="zero"` → zeros; `"global_mean"` → mean of covered vectors (zeros if none covered). Fallback uses NO labels.

- [ ] **Step 1: Write the failing fixture + test**

Add to `tests/conftest.py` (after the existing fixtures):

```python
def _write_synthetic_bags_npz(path: Path) -> Path:
    """Write a tiny cell-bags NPZ: 3 covered genes, 2-dim embeddings."""
    # gene G0: 2 cells, gene G1: 1 cell, gene G2: 3 cells
    cell_delta_pcs = np.array(
        [
            [1.0, 0.0], [3.0, 2.0],      # G0 -> mean [2, 1]
            [5.0, 5.0],                  # G1 -> mean [5, 5]
            [0.0, 0.0], [2.0, 0.0], [4.0, 6.0],  # G2 -> mean [2, 2]
        ],
        dtype=np.float32,
    )
    bag_offsets = np.array([0, 2, 3, 6], dtype=np.int64)
    perturbation_gene = np.asarray(["G0", "G1", "G2"], dtype=object)
    npz_path = path / "synthetic_bags.npz"
    np.savez_compressed(
        npz_path,
        cell_delta_pcs=cell_delta_pcs,
        bag_offsets=bag_offsets,
        perturbation_gene=perturbation_gene,
    )
    return npz_path


@pytest.fixture
def synthetic_bags_npz(tmp_path: Path) -> Path:
    """Provide a path to a freshly written synthetic cell-bags NPZ."""
    return _write_synthetic_bags_npz(tmp_path)


def _write_augmented_benchmark_csv(path: Path) -> Path:
    """Deterministic benchmark guaranteeing both-covered and mixed test pairs.

    Covered genes C0-C3 (present in the augmented bags NPZ); uncovered U0-U1.
    Per fold (0, 1), each role/label includes at least one both-covered pair and
    one mixed pair, so the covered_pairs slice is always non-empty.
    """
    effects = {"C0": -1.2, "C1": -1.0, "C2": -0.9, "C3": -1.1, "U0": 0.2, "U1": 0.3}
    # (role, label, gene_a, gene_b)
    spec = [
        ("train", 1, "C0", "C1"), ("train", 1, "C2", "C3"),
        ("train", 0, "C0", "U0"), ("train", 0, "C1", "U1"),
        ("test", 1, "C0", "C2"), ("test", 1, "C1", "U0"),
        ("test", 0, "C1", "C3"), ("test", 0, "C2", "U1"),
    ]
    rows = []
    pair_counter = 0
    for fold_id in (0, 1):
        for role, label, a, b in spec:
            rows.append(
                {
                    "pair_id": f"P{pair_counter}",
                    "fold_id": fold_id,
                    "split_role": role,
                    "sl_label": label,
                    "gene_a_symbol": a,
                    "gene_b_symbol": b,
                    "gene_a_k562_gene_effect": effects[a],
                    "gene_b_k562_gene_effect": effects[b],
                }
            )
            pair_counter += 1
    csv_path = path / "synthetic_augmented_sl.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_augmented_bags_npz(path: Path) -> Path:
    """Bags NPZ covering only C0-C3 (matches _write_augmented_benchmark_csv)."""
    cell_delta_pcs = np.array(
        [
            [1.0, 0.0], [3.0, 2.0],   # C0 -> [2, 1]
            [5.0, 5.0],               # C1 -> [5, 5]
            [0.0, 0.0], [4.0, 4.0],   # C2 -> [2, 2]
            [1.0, 3.0],               # C3 -> [1, 3]
        ],
        dtype=np.float32,
    )
    bag_offsets = np.array([0, 2, 3, 5, 6], dtype=np.int64)
    perturbation_gene = np.asarray(["C0", "C1", "C2", "C3"], dtype=object)
    npz_path = path / "synthetic_augmented_bags.npz"
    np.savez_compressed(
        npz_path,
        cell_delta_pcs=cell_delta_pcs,
        bag_offsets=bag_offsets,
        perturbation_gene=perturbation_gene,
    )
    return npz_path


@pytest.fixture
def synthetic_augmented_benchmark_csv(tmp_path: Path) -> Path:
    """Deterministic benchmark CSV with guaranteed covered/mixed test pairs."""
    return _write_augmented_benchmark_csv(tmp_path)


@pytest.fixture
def synthetic_augmented_bags_npz(tmp_path: Path) -> Path:
    """Bags NPZ covering exactly the covered genes of the augmented benchmark."""
    return _write_augmented_bags_npz(tmp_path)
```

Create `tests/test_sl_benchmark_baseline_embeddings.py`:

```python
"""Tests for per-gene embedding pooling, alignment, and fallback."""

from __future__ import annotations

import numpy as np

from sl_benchmark_baseline.embeddings import (
    align_to_universe,
    load_gene_embeddings,
)


def test_load_gene_embeddings_mean_pools_each_bag(synthetic_bags_npz):
    table = load_gene_embeddings(synthetic_bags_npz)
    assert table.dim == 2
    np.testing.assert_allclose(table.vectors_by_symbol["G0"], [2.0, 1.0])
    np.testing.assert_allclose(table.vectors_by_symbol["G1"], [5.0, 5.0])
    np.testing.assert_allclose(table.vectors_by_symbol["G2"], [2.0, 2.0])


def test_align_to_universe_marks_coverage_and_zero_fallback(synthetic_bags_npz):
    table = load_gene_embeddings(synthetic_bags_npz)
    symbols = np.asarray(["G1", "UNCOVERED", "G0"])
    emb, mask = align_to_universe(table, symbols, fallback_strategy="zero")
    assert emb.shape == (3, 2)
    np.testing.assert_allclose(mask, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(emb[0], [5.0, 5.0])
    np.testing.assert_allclose(emb[1], [0.0, 0.0])
    np.testing.assert_allclose(emb[2], [2.0, 1.0])


def test_align_to_universe_global_mean_fallback(synthetic_bags_npz):
    table = load_gene_embeddings(synthetic_bags_npz)
    symbols = np.asarray(["G0", "G1", "G2", "UNCOVERED"])
    emb, mask = align_to_universe(table, symbols, fallback_strategy="global_mean")
    # global mean of covered = mean([2,1],[5,5],[2,2]) = [3, 8/3]
    np.testing.assert_allclose(emb[3], [3.0, 8.0 / 3.0])
    np.testing.assert_allclose(mask, [1.0, 1.0, 1.0, 0.0])


def test_align_to_universe_global_mean_is_coverage_stable(synthetic_bags_npz):
    # global_mean is computed over covered genes only, so adding extra uncovered
    # symbols must not change the fallback vector (label-free, fold-stable).
    table = load_gene_embeddings(synthetic_bags_npz)
    emb_small, _ = align_to_universe(
        table, np.asarray(["G0", "G1", "G2"]), fallback_strategy="global_mean"
    )
    emb_big, _ = align_to_universe(
        table, np.asarray(["G0", "G1", "G2", "U0", "U1"]),
        fallback_strategy="global_mean",
    )
    expected = np.vstack([emb_small[0], emb_small[1], emb_small[2]]).mean(axis=0)
    np.testing.assert_allclose(emb_big[3], expected)  # U0 fallback
    np.testing.assert_allclose(emb_big[4], expected)  # U1 fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.embeddings'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sl_benchmark_baseline/embeddings.py`:

```python
"""Per-gene transcript embeddings pooled from an exp03 cell-bags NPZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeneEmbeddingTable:
    """Mean-pooled per-gene embedding vectors keyed by upper-case symbol."""

    dim: int
    vectors_by_symbol: dict[str, np.ndarray]


def load_gene_embeddings(bags_npz: Path) -> GeneEmbeddingTable:
    """Mean-pool each gene's delta-cell bag into one per-gene vector.

    Args:
        bags_npz: Path to an exp03 cell-bags NPZ with ``cell_delta_pcs``,
            ``bag_offsets``, and ``perturbation_gene`` keys.

    Returns:
        A :class:`GeneEmbeddingTable` over covered gene symbols.
    """
    with np.load(bags_npz, allow_pickle=True) as payload:
        cells = np.asarray(payload["cell_delta_pcs"], dtype=float)
        offsets = np.asarray(payload["bag_offsets"], dtype=int)
        genes = np.asarray(payload["perturbation_gene"], dtype=object)
    vectors: dict[str, np.ndarray] = {}
    for index, symbol in enumerate(genes):
        start, stop = offsets[index], offsets[index + 1]
        if stop <= start:
            continue
        vectors[str(symbol).upper()] = cells[start:stop].mean(axis=0)
    return GeneEmbeddingTable(dim=cells.shape[1], vectors_by_symbol=vectors)


def align_to_universe(
    table: GeneEmbeddingTable,
    symbols: np.ndarray,
    fallback_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align pooled embeddings to a universe order with a coverage mask.

    Args:
        table: Pooled per-gene embeddings.
        symbols: Universe gene symbols in canonical order, shape ``(n_gene,)``.
        fallback_strategy: ``"zero"`` or ``"global_mean"`` for uncovered genes.

    Returns:
        ``(embeddings (n_gene, dim), coverage_mask (n_gene,))``.

    Raises:
        ValueError: If ``fallback_strategy`` is not recognized.
    """
    if fallback_strategy not in {"zero", "global_mean"}:
        raise ValueError(f"unknown fallback_strategy: {fallback_strategy}")
    covered = [table.vectors_by_symbol[str(s).upper()]
               for s in symbols if str(s).upper() in table.vectors_by_symbol]
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

Note: `global_mean` is computed over all covered genes present in `symbols`. Because `symbols` is always the full candidate universe (built before any fold split) and gwps coverage is fixed (not fold-dependent), this fallback is stable across folds and is label-free (no SL `D` label touches the embedding or the mean). Document this in the docstring.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_embeddings.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/embeddings.py tests/test_sl_benchmark_baseline_embeddings.py tests/conftest.py
git commit -m "feat: add per-gene embedding pooling, universe alignment, and augmented fixtures"
```

### Task 4: Transcript-augmented models (A_transcript, B_transcript)

**Files:**
- Modify: `src/sl_benchmark_baseline/models.py`
- Test: `tests/test_sl_benchmark_baseline_models.py`

**Interfaces:**
- Consumes: `FoldData` (existing). Augmented models hold their own sklearn/XGB estimator and consume `FoldData.features` exactly like A/B — i.e., **the augmented feature matrix is assembled in `evaluate.py` (Task 5) and handed in via `FoldData.features`**, so the models stay thin and width-agnostic. The only difference from A/B is `name`.
- Produces:
  - `LogRegTranscriptModel(config)` with `name = "A_transcript"`, same `.fit`/`.predict_proba` as `LogRegModel`.
  - `XGBTranscriptModel(config)` with `name = "B_transcript"`, same as `XGBModel`.
  - `build_augmented_models(config) -> list` returning `[LogRegModel, XGBModel, LogRegTranscriptModel, XGBTranscriptModel]` (baseline A/B + transcript A_transcript/B_transcript; no degree probe C).

Design note: because `LogRegModel`/`XGBModel` already place no shape constraint on `features`, the transcript variants are literally subclasses that override `name`. The feature-width difference is realized by the evaluate layer building two `FoldData` objects per fold — one with 5-col GeneEffect features (for A/B) and one with augmented features (for A_transcript/B_transcript). Task 5 wires this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_models.py`:

```python
import numpy as np
import pandas as pd
from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.models import (
    FoldData,
    LogRegTranscriptModel,
    XGBTranscriptModel,
    build_augmented_models,
)


def _augmented_fold(n_features: int, n: int = 12) -> FoldData:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, n_features))
    labels = np.array([1, 0] * (n // 2))
    df = pd.DataFrame(
        {
            "gene_a_symbol": [f"G{i}" for i in range(n)],
            "gene_b_symbol": [f"H{i}" for i in range(n)],
            "sl_label": labels,
        }
    )
    return FoldData(df=df, features=features, labels=labels)


def test_transcript_models_emit_unit_interval_proba():
    config = SLBaselineConfig()
    train = _augmented_fold(n_features=11)
    for model_cls in (LogRegTranscriptModel, XGBTranscriptModel):
        model = model_cls(config)
        model.fit(train)
        proba = model.predict_proba(train)
        assert proba.shape == (12,)
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0


def test_transcript_model_names():
    assert LogRegTranscriptModel(SLBaselineConfig()).name == "A_transcript"
    assert XGBTranscriptModel(SLBaselineConfig()).name == "B_transcript"


def test_build_augmented_models_excludes_degree_probe():
    models = build_augmented_models(SLBaselineConfig())
    names = [m.name for m in models]
    assert names == ["A", "B", "A_transcript", "B_transcript"]
    assert "C" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_models.py::test_transcript_model_names -v`
Expected: FAIL with `ImportError: cannot import name 'LogRegTranscriptModel'`.

- [ ] **Step 3: Write minimal implementation**

In `src/sl_benchmark_baseline/models.py`, add subclasses after `XGBModel` (after line 69) and a factory after `build_models` (after line 126):

```python
class LogRegTranscriptModel(LogRegModel):
    """Model A_transcript: logistic regression on augmented features."""

    name = "A_transcript"


class XGBTranscriptModel(XGBModel):
    """Model B_transcript: XGBoost on augmented features."""

    name = "B_transcript"


def build_augmented_models(config: SLBaselineConfig) -> list:
    """Construct exp07 augmented-mode models: baseline A/B + transcript A/B."""
    return [
        LogRegModel(config),
        XGBModel(config),
        LogRegTranscriptModel(config),
        XGBTranscriptModel(config),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_models.py -v`
Expected: PASS (new tests + pre-existing A/B/C tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/models.py tests/test_sl_benchmark_baseline_models.py
git commit -m "feat: add transcript-augmented SL models"
```

### Task 5: Augmented gene universe + augmented score matrix

**Files:**
- Modify: `src/sl_benchmark_baseline/evaluate.py:36-44` (GeneUniverse), `:70-98` (`_build_gene_universe`), and add an augmented score-matrix helper near `:111-149`.
- Test: `tests/test_sl_benchmark_baseline_evaluate.py`

**Interfaces:**
- Consumes: `GeneEmbeddingTable`, `align_to_universe` (Task 3); `build_augmented_pair_features` (Task 2); `FoldData` (Task 4).
- Produces:
  - `GeneUniverse` gains two optional fields: `embeddings: np.ndarray | None = None` `(n_gene, dim)`, `coverage_mask: np.ndarray | None = None` `(n_gene,)`. Existing callers that omit them keep working.
  - `_build_gene_universe(frame, embedding_table=None, fallback_strategy="zero")` — when `embedding_table` is given, align it to `universe.symbols` and populate `embeddings`/`coverage_mask`; else both stay `None`.
  - `_build_augmented_score_matrix(model, universe, standardizer) -> np.ndarray` `(n_gene, n_gene)` — like `_build_score_matrix` but features come from `build_augmented_pair_features` using `universe.embeddings`/`coverage_mask`, standardized by the augmented `standardizer`. Diagonal zeroed.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_evaluate.py`:

```python
def test_build_gene_universe_populates_embeddings_and_coverage(
    synthetic_benchmark_csv: Path, synthetic_bags_npz: Path
) -> None:
    import numpy as np
    from sl_benchmark_baseline.data import load_benchmark
    from sl_benchmark_baseline.embeddings import load_gene_embeddings
    from sl_benchmark_baseline.evaluate import _build_gene_universe

    frame = load_benchmark(synthetic_benchmark_csv)
    table = load_gene_embeddings(synthetic_bags_npz)  # covers G0,G1,G2
    universe = _build_gene_universe(frame, embedding_table=table,
                                    fallback_strategy="zero")
    assert universe.embeddings is not None
    assert universe.embeddings.shape == (len(universe.symbols), table.dim)
    assert universe.coverage_mask.shape == (len(universe.symbols),)
    # G0/G1/G2 are in the synthetic benchmark gene pool G0..G11
    covered = {s for s, m in zip(universe.symbols, universe.coverage_mask) if m == 1.0}
    assert covered.issubset(set(universe.symbols))
    assert {"G0", "G1", "G2"}.issubset(covered)


def test_build_gene_universe_without_embeddings_is_none(
    synthetic_benchmark_csv: Path,
) -> None:
    from sl_benchmark_baseline.data import load_benchmark
    from sl_benchmark_baseline.evaluate import _build_gene_universe

    frame = load_benchmark(synthetic_benchmark_csv)
    universe = _build_gene_universe(frame)
    assert universe.embeddings is None
    assert universe.coverage_mask is None


def test_augmented_score_matrix_is_square_with_zero_diagonal(
    synthetic_benchmark_csv: Path, synthetic_bags_npz: Path
) -> None:
    import numpy as np
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.data import load_benchmark
    from sl_benchmark_baseline.embeddings import load_gene_embeddings
    from sl_benchmark_baseline.evaluate import (
        _build_augmented_score_matrix,
        _build_gene_universe,
    )
    from sl_benchmark_baseline.features import (
        Standardizer,
        build_augmented_pair_features,
    )
    from sl_benchmark_baseline.models import LogRegTranscriptModel

    frame = load_benchmark(synthetic_benchmark_csv)
    table = load_gene_embeddings(synthetic_bags_npz)
    universe = _build_gene_universe(frame, embedding_table=table,
                                    fallback_strategy="zero")
    # build a tiny train feature set to fit the standardizer and a model
    ea = frame["gene_a_k562_gene_effect"].to_numpy()
    eb = frame["gene_b_k562_gene_effect"].to_numpy()
    dim = table.dim
    emb_a = np.zeros((len(frame), dim)); emb_b = np.zeros((len(frame), dim))
    flag = np.ones(len(frame))
    raw = build_augmented_pair_features(ea, eb, emb_a, emb_b, flag, flag, True)
    standardizer = Standardizer.fit(raw)
    model = LogRegTranscriptModel(SLBaselineConfig())
    from sl_benchmark_baseline.models import FoldData
    model.fit(FoldData(df=frame, features=standardizer.transform(raw),
                       labels=frame["sl_label"].to_numpy(dtype=int)))
    matrix = _build_augmented_score_matrix(model, universe, standardizer, True)
    n = len(universe.symbols)
    assert matrix.shape == (n, n)
    assert np.allclose(np.diag(matrix), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_evaluate.py::test_build_gene_universe_without_embeddings_is_none -v`
Expected: FAIL with `TypeError: _build_gene_universe() ... unexpected keyword argument 'embedding_table'` (or `AttributeError` on `embeddings`).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_benchmark_baseline/evaluate.py`:

(a) Add imports near the top (after line 15):
```python
from sl_benchmark_baseline.embeddings import GeneEmbeddingTable, align_to_universe
from sl_benchmark_baseline.features import build_augmented_pair_features
```

(b) Extend `GeneUniverse` (lines 36-44) — add two fields with defaults:
```python
    embeddings: np.ndarray | None = None
    coverage_mask: np.ndarray | None = None
```

(c) Replace the entire `_build_gene_universe` function (lines 70-98) with this complete drop-in:
```python
def _build_gene_universe(
    frame: pd.DataFrame,
    embedding_table: GeneEmbeddingTable | None = None,
    fallback_strategy: str = "zero",
) -> GeneUniverse:
    gene_a_key, gene_b_key = _gene_key_columns(frame)
    gene_a = pd.DataFrame(
        {
            "key": frame[gene_a_key],
            "symbol": frame["gene_a_symbol"],
            "gene_effect": frame["gene_a_k562_gene_effect"],
        }
    )
    gene_b = pd.DataFrame(
        {
            "key": frame[gene_b_key],
            "symbol": frame["gene_b_symbol"],
            "gene_effect": frame["gene_b_k562_gene_effect"],
        }
    )
    genes = (
        pd.concat([gene_a, gene_b], ignore_index=True)
        .drop_duplicates("key")
        .sort_values("key")
        .reset_index(drop=True)
    )
    keys = tuple(genes["key"].tolist())
    symbols = genes["symbol"].astype(str).to_numpy()
    embeddings = None
    coverage_mask = None
    if embedding_table is not None:
        embeddings, coverage_mask = align_to_universe(
            embedding_table, symbols, fallback_strategy
        )
    return GeneUniverse(
        keys=keys,
        symbols=symbols,
        gene_effects=genes["gene_effect"].to_numpy(dtype=float),
        index_by_key={key: index for index, key in enumerate(keys)},
        embeddings=embeddings,
        coverage_mask=coverage_mask,
    )
```

(d) Add the augmented score-matrix helper after `_build_score_matrix` (after line 149). The signature takes `include_coverage_flag` as an explicit 4th parameter (do NOT use any inline expression for it):
```python
def _build_augmented_score_matrix(
    model: object,
    universe: GeneUniverse,
    standardizer: Standardizer,
    include_coverage_flag: bool,
) -> np.ndarray:
    """Score all candidate pairs using augmented GeneEffect+transcript features."""
    if universe.embeddings is None or universe.coverage_mask is None:
        raise ValueError("augmented score matrix requires universe embeddings")
    n_gene = len(universe.symbols)
    score_matrix = np.zeros((n_gene, n_gene), dtype=float)
    all_idx = np.arange(n_gene)
    for start in range(0, n_gene, SCORE_MATRIX_CHUNK_ROWS):
        stop = min(start + SCORE_MATRIX_CHUNK_ROWS, n_gene)
        rows = np.arange(start, stop)
        a_idx = np.repeat(rows, n_gene)
        b_idx = np.tile(all_idx, len(rows))
        raw = build_augmented_pair_features(
            universe.gene_effects[a_idx],
            universe.gene_effects[b_idx],
            universe.embeddings[a_idx],
            universe.embeddings[b_idx],
            universe.coverage_mask[a_idx],
            universe.coverage_mask[b_idx],
            include_coverage_flag=include_coverage_flag,
        )
        features = standardizer.transform(raw)
        pair_df = pd.DataFrame(
            {
                "gene_a_symbol": universe.symbols[a_idx],
                "gene_b_symbol": universe.symbols[b_idx],
            }
        )
        fold_data = FoldData(
            df=pair_df, features=features, labels=np.zeros(len(features), dtype=int)
        )
        score_matrix[start:stop, :] = model.predict_proba(fold_data).reshape(
            len(rows), n_gene
        )
    np.fill_diagonal(score_matrix, 0.0)
    return score_matrix
```

The Task 5 test must call this with `include_coverage_flag=True` as the 4th argument: `_build_augmented_score_matrix(model, universe, standardizer, True)`. Update the test's last line accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_evaluate.py -v`
Expected: PASS — new universe/score-matrix tests AND the 3 pre-existing run_cv tests (which call `_build_gene_universe(frame)` with no embeddings → `None`).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/evaluate.py tests/test_sl_benchmark_baseline_evaluate.py
git commit -m "feat: add augmented gene universe and augmented score matrix"
```

### Task 6: Augmented `run_fold` with covered-pair diagnostic slice

**Files:**
- Modify: `src/sl_benchmark_baseline/evaluate.py:152-197` (`run_fold`) and `:200-206` (`_summarize`).
- Test: `tests/test_sl_benchmark_baseline_evaluate.py`

**Interfaces:**
- Consumes: `build_augmented_models` (Task 4), `_build_augmented_score_matrix` (Task 5), `Standardizer`, `build_augmented_pair_features`, `official_*_metrics`.
- Produces:
  - Every metric row gains a `"slice"` key. In the non-augmented (exp06) path the value is always `"full_universe"` (so existing 5-column-shaped tests still pass after `_summarize` adds `slice` to the groupby — see note).
  - In augmented mode, `run_fold` fits all 4 models. Baseline A/B use the 5-col GeneEffect `standardizer`/score matrix (existing `_build_score_matrix`). Transcript A_transcript/B_transcript use an augmented `standardizer` and `_build_augmented_score_matrix`. For the transcript models, metrics are emitted twice: `slice="full_universe"` (all test pos/neg) and `slice="covered_pairs"` (test pos/neg restricted to pairs where both genes' `coverage_mask == 1`).
  - `_summarize` groups by `["split_type", "model", "slice", "metric"]`.

Important backward-compat note (verified by inspection against the existing tests, not just asserted): existing evaluate tests assert `len(summary) == n_metrics * 3` and `not summary.duplicated(["split_type","model","metric"]).any()`, and check columns with `.issubset`. After this task, every row carries a `slice` key and `_summarize` groups by `["split_type","model","slice","metric"]`. In the non-augmented (exp06) path there is exactly one slice value (`"full_universe"`) per (model, metric), so the grouped row count stays `n_metrics * 3 * 1` and no `(split_type, model, metric)` duplicates appear; the extra `slice` column is permitted by the `.issubset` checks. The `test_nonaugmented_run_cv_unchanged_models` test below pins this. No edits to the three pre-existing evaluate tests are required — confirm they still pass in Step 4.

The `covered_pairs` slice is emitted only for the transcript models and only when at least one test-positive pair has both genes covered (guarded by `len(pos_cov) > 0`); the deterministic `synthetic_augmented_*` fixtures (Task 3) guarantee such a pair exists, so the `"covered_pairs" in slices` assertion is stable.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_evaluate.py`:

```python
def test_augmented_run_cv_emits_transcript_models_and_covered_slice(
    synthetic_augmented_benchmark_csv: Path,
    synthetic_augmented_bags_npz: Path,
    tmp_path: Path,
) -> None:
    import json
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "aug_run"
    config = SLBaselineConfig(
        input_csv=synthetic_augmented_benchmark_csv,
        output_dir=output_dir,
        bags_npz=synthetic_augmented_bags_npz,
        folds=(0, 1),
        ranking_k=(2, 5),
    )
    summary = run_cv(config)

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    # baseline A/B + transcript variants; degree probe C excluded in augmented mode
    assert set(fold_metrics["model"].unique()) == {
        "A", "B", "A_transcript", "B_transcript"
    }
    assert "slice" in fold_metrics.columns
    slices = set(fold_metrics["slice"].unique())
    assert "full_universe" in slices
    assert "covered_pairs" in slices
    # covered_pairs slice only emitted for transcript models
    covered = fold_metrics[fold_metrics["slice"] == "covered_pairs"]
    assert set(covered["model"].unique()).issubset({"A_transcript", "B_transcript"})
    assert {"split_type", "model", "slice", "metric"}.issubset(summary.columns)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["augmented"] is True
    assert manifest["embedding_method"] == config.embedding_method


def test_nonaugmented_run_cv_unchanged_models(
    synthetic_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "base_run"
    config = SLBaselineConfig(
        input_csv=synthetic_benchmark_csv, output_dir=output_dir,
        folds=(0, 1), ranking_k=(2, 5),
    )
    run_cv(config)
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["model"].unique()) == {"A", "B", "C"}
    assert set(fold_metrics["slice"].unique()) == {"full_universe"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_evaluate.py::test_augmented_run_cv_emits_transcript_models_and_covered_slice -v`
Expected: FAIL — `run_cv` does not yet accept augmented config (KeyError on `slice` column, or models still `{A,B,C}`).

- [ ] **Step 3: Write minimal implementation**

Refactor `run_fold` in `src/sl_benchmark_baseline/evaluate.py` to branch on `config.augmented`. Replace the body (lines 159-197) with a dispatcher and two helpers. Add a `_metric_rows` helper to DRY the row assembly:

```python
def _metric_rows(
    split_type: str, model_name: str, fold_id: int, slice_name: str,
    score_matrix: np.ndarray, pos_index: np.ndarray, neg_index: np.ndarray,
    seen_index: np.ndarray, ks: tuple[int, ...],
) -> list[dict[str, object]]:
    metrics = official_classification_metrics(score_matrix, pos_index, neg_index)
    metrics.update(
        official_ranking_metrics(score_matrix, pos_index, seen_index=seen_index, ks=ks)
    )
    return [
        {"split_type": split_type, "model": model_name, "fold_id": fold_id,
         "slice": slice_name, "metric": metric, "value": float(value)}
        for metric, value in metrics.items()
    ]
```

Then `run_fold`:
```python
def run_fold(frame, split_type, fold_id, config, universe):
    """Fit all models on one fold and return long-form metric rows."""
    train_df, test_df = fold_split(frame, split_type, fold_id)
    base_std = Standardizer.fit(
        build_pair_features(
            train_df["gene_a_k562_gene_effect"].to_numpy(),
            train_df["gene_b_k562_gene_effect"].to_numpy(),
        )
    )
    train_base = _build_fold_data(train_df, base_std)
    test_pos = test_df[test_df["sl_label"] == 1]
    test_neg = test_df[test_df["sl_label"] == 0]
    train_pos = train_df[train_df["sl_label"] == 1]
    pos_index = _pair_indices(test_pos, universe)
    neg_index = _pair_indices(test_neg, universe)
    seen_index = _pair_indices(train_pos, universe)

    if not config.augmented:
        rows: list[dict[str, object]] = []
        for model in build_models(config):
            model.fit(train_base)
            score_matrix = _build_score_matrix(model, universe, base_std)
            rows.extend(_metric_rows(
                split_type, model.name, fold_id, "full_universe",
                score_matrix, pos_index, neg_index, seen_index, config.ranking_k))
        return rows
    return _run_fold_augmented(
        train_df, train_base, base_std, pos_index, neg_index, seen_index,
        split_type, fold_id, config, universe)
```

Add `_run_fold_augmented` (keep it <50 lines; it builds the augmented standardizer, fits the 4 models, and emits full + covered slices):
```python
def _augmented_fold_data(frame, universe, standardizer, config):
    pair_idx = _pair_indices(frame, universe)
    a_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1]
    raw = build_augmented_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
        universe.embeddings[a_idx], universe.embeddings[b_idx],
        universe.coverage_mask[a_idx], universe.coverage_mask[b_idx],
        include_coverage_flag=config.include_coverage_flag,
    )
    return FoldData(df=frame, features=standardizer.transform(raw),
                    labels=frame["sl_label"].to_numpy(dtype=int))


def _augmented_raw(frame, universe, config):
    """Unstandardized augmented features for a frame (used to fit a standardizer)."""
    pair_idx = _pair_indices(frame, universe)
    a_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1]
    return build_augmented_pair_features(
        frame["gene_a_k562_gene_effect"].to_numpy(),
        frame["gene_b_k562_gene_effect"].to_numpy(),
        universe.embeddings[a_idx], universe.embeddings[b_idx],
        universe.coverage_mask[a_idx], universe.coverage_mask[b_idx],
        include_coverage_flag=config.include_coverage_flag,
    )


def _covered_pair_mask(index: np.ndarray, universe) -> np.ndarray:
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    return (universe.coverage_mask[index[:, 0]] == 1) & (
        universe.coverage_mask[index[:, 1]] == 1)
```

`_run_fold_augmented` assembles the augmented standardizer from train rows, then for each of `build_augmented_models(config)` decides feature path by name suffix (`_transcript` → augmented matrix + both slices; baseline `A`/`B` → base matrix + full slice only). The covered-pairs slice is emitted only when non-empty, and always uses the unfiltered `seen_index` (so every train-positive pair is masked from candidate rankings, matching the full-universe slice):
```python
def _run_fold_augmented(train_df, train_base, base_std, pos_index, neg_index,
                        seen_index, split_type, fold_id, config, universe):
    aug_std = Standardizer.fit(_augmented_raw(train_df, universe, config))
    train_aug = _augmented_fold_data(train_df, universe, aug_std, config)
    pos_cov = pos_index[_covered_pair_mask(pos_index, universe)]
    neg_cov = neg_index[_covered_pair_mask(neg_index, universe)]
    rows: list[dict[str, object]] = []
    for model in build_augmented_models(config):
        if model.name.endswith("_transcript"):
            model.fit(train_aug)
            sm = _build_augmented_score_matrix(
                model, universe, aug_std, config.include_coverage_flag)
            rows.extend(_metric_rows(split_type, model.name, fold_id,
                        "full_universe", sm, pos_index, neg_index,
                        seen_index, config.ranking_k))
            if len(pos_cov) > 0:
                rows.extend(_metric_rows(split_type, model.name, fold_id,
                            "covered_pairs", sm, pos_cov, neg_cov,
                            seen_index, config.ranking_k))
        else:
            model.fit(train_base)
            sm = _build_score_matrix(model, universe, base_std)
            rows.extend(_metric_rows(split_type, model.name, fold_id,
                        "full_universe", sm, pos_index, neg_index,
                        seen_index, config.ranking_k))
    return rows
```

Update `_build_augmented_score_matrix` signature to take `include_coverage_flag` (from the Task 5 design fix). Update `_summarize` (line 200) groupby to `["split_type", "model", "slice", "metric"]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_evaluate.py -v`
Expected: PASS — new augmented tests AND all 3 pre-existing run_cv tests (non-augmented path unchanged, `slice` column present with single value `full_universe`).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/evaluate.py tests/test_sl_benchmark_baseline_evaluate.py
git commit -m "feat: add augmented run_fold with covered-pair diagnostic slice"
```

### Task 7: Manifest augmentation fields + `run_cv` wiring

**Files:**
- Modify: `src/sl_benchmark_baseline/evaluate.py:228-259` (`run_cv`).
- Test: covered by the Task 6 augmented test (`manifest["augmented"]`, `manifest["embedding_method"]`); add one focused assertion below.

**Interfaces:**
- Consumes: `load_gene_embeddings` (Task 3), `config.augmented`, `config.embedding_method`, `config.fallback_strategy`, `config.include_coverage_flag`, `config.bags_npz`.
- Produces: `run_cv` loads the embedding table when augmented, passes it to `_build_gene_universe`, and writes augmentation fields into the manifest. `models` manifest field reflects the actual model set.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_evaluate.py`:

```python
def test_augmented_manifest_records_coverage_fields(
    synthetic_augmented_benchmark_csv: Path,
    synthetic_augmented_bags_npz: Path,
    tmp_path: Path,
) -> None:
    import json
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "aug_manifest_run"
    config = SLBaselineConfig(
        input_csv=synthetic_augmented_benchmark_csv, output_dir=output_dir,
        bags_npz=synthetic_augmented_bags_npz, folds=(0,), ranking_k=(2, 5),
        fallback_strategy="global_mean", include_coverage_flag=False,
    )
    run_cv(config)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["augmented"] is True
    assert manifest["bags_npz"].endswith("synthetic_augmented_bags.npz")
    assert manifest["fallback_strategy"] == "global_mean"
    assert manifest["include_coverage_flag"] is False
    assert "gwps_coverage_gene_count" in manifest
    assert manifest["models"] == ["A", "B", "A_transcript", "B_transcript"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_evaluate.py::test_augmented_manifest_records_coverage_fields -v`
Expected: FAIL with `KeyError: 'augmented'` (manifest lacks the field).

- [ ] **Step 3: Write minimal implementation**

In `run_cv` (line 228), load embeddings and build the universe conditionally; extend the manifest. Replace lines 230-231 and the manifest dict:

```python
    frame = load_benchmark(config.input_csv)
    embedding_table = None
    if config.augmented:
        from sl_benchmark_baseline.embeddings import load_gene_embeddings

        embedding_table = load_gene_embeddings(config.bags_npz)
    universe = _build_gene_universe(
        frame, embedding_table=embedding_table,
        fallback_strategy=config.fallback_strategy,
    )
```

Add after the existing manifest keys (before `write_text`):
```python
    coverage_count = (
        int(universe.coverage_mask.sum()) if universe.coverage_mask is not None else 0
    )
    model_names = (
        ["A", "B", "A_transcript", "B_transcript"] if config.augmented else ["A", "B", "C"]
    )
    manifest["models"] = model_names
    manifest["augmented"] = config.augmented
    manifest["bags_npz"] = None if config.bags_npz is None else str(config.bags_npz)
    manifest["embedding_method"] = config.embedding_method
    manifest["fallback_strategy"] = config.fallback_strategy
    manifest["include_coverage_flag"] = config.include_coverage_flag
    manifest["gwps_coverage_gene_count"] = coverage_count
    manifest["gwps_coverage_fraction"] = (
        coverage_count / len(universe.symbols) if len(universe.symbols) else 0.0
    )
```
(Remove the now-redundant hardcoded `"models": ["A", "B", "C"]` line from the original manifest dict.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_evaluate.py -v`
Expected: PASS (all evaluate tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/evaluate.py tests/test_sl_benchmark_baseline_evaluate.py
git commit -m "feat: record exp07 augmentation fields in run manifest"
```

---

### Task 8: CLI flags for augmented mode

**Files:**
- Modify: `src/sl_benchmark_baseline/__main__.py:15-42`
- Test: `tests/test_sl_benchmark_baseline_cli.py`

**Interfaces:**
- Consumes: `SLBaselineConfig` augmentation fields (Task 1), `run_cv` (Task 7).
- Produces: CLI accepts `--bags-npz PATH`, `--embedding-method STR`, `--fallback-strategy {zero,global_mean}`, `--include-coverage-flag/--no-coverage-flag`. When `--bags-npz` omitted, behavior is identical to exp06.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_benchmark_baseline_cli.py`:

```python
def test_cli_augmented_run_writes_transcript_models(
    synthetic_augmented_benchmark_csv, synthetic_augmented_bags_npz, tmp_path
):
    import pandas as pd
    from sl_benchmark_baseline.__main__ import main

    output_dir = tmp_path / "cli_aug"
    code = main([
        "--input-csv", str(synthetic_augmented_benchmark_csv),
        "--output-dir", str(output_dir),
        "--bags-npz", str(synthetic_augmented_bags_npz),
        "--folds", "0",
        "--ranking-k", "2", "5",
        "--fallback-strategy", "zero",
    ])
    assert code == 0
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert "A_transcript" in set(fold_metrics["model"].unique())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_cli.py::test_cli_augmented_run_writes_transcript_models -v`
Expected: FAIL with `SystemExit: 2` / `unrecognized arguments: --bags-npz`.

- [ ] **Step 3: Write minimal implementation**

In `_parse_args` (after line 27) add:
```python
    parser.add_argument("--bags-npz", type=Path, default=defaults.bags_npz)
    parser.add_argument(
        "--embedding-method", type=str, default=defaults.embedding_method
    )
    parser.add_argument(
        "--fallback-strategy", choices=("zero", "global_mean"),
        default=defaults.fallback_strategy,
    )
    parser.add_argument(
        "--include-coverage-flag", dest="include_coverage_flag",
        action="store_true", default=defaults.include_coverage_flag,
    )
    parser.add_argument(
        "--no-coverage-flag", dest="include_coverage_flag", action="store_false",
    )
```
In `main` (the `config = SLBaselineConfig(...)` block), add the new kwargs:
```python
        bags_npz=args.bags_npz,
        embedding_method=args.embedding_method,
        fallback_strategy=args.fallback_strategy,
        include_coverage_flag=args.include_coverage_flag,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_benchmark_baseline_cli.py -v`
Expected: PASS (new + pre-existing CLI tests).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/__main__.py tests/test_sl_benchmark_baseline_cli.py
git commit -m "feat: add exp07 augmented-mode CLI flags"
```

---

### Task 9: Production config, full-suite verification, docs

**Files:**
- Create: `configs/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_cv1_cv2_cv3.yaml`
- Modify: `docs/experiment/07_k562_sl_pair_perturbseq_augmented.md` (Run status + how-to-run)

**Interfaces:**
- Consumes: nothing new (config is documentation/runbook; the SL baseline reads CLI flags, not YAML, so this YAML is a recorded invocation, mirroring how exp06 documents its run table).

- [ ] **Step 1: Write the config + runbook YAML**

Create `configs/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_cv1_cv2_cv3.yaml`:

```yaml
# Exp07 Perturb-seq-augmented SL-pair baseline (MVP).
# This module reads CLI flags (see commands below), not this YAML directly;
# the file records the canonical production invocation for reproducibility.
input_csv: data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv
bags_npz: results/experiments/03_replogle_k562_single_cell_deepsets_adamson/cell_bags/single_cell_pc_delta/bags.npz
embedding_method: pca_delta_meanpool   # tier 1; tier 2 swaps in an scvi_delta bags.npz
fallback_strategy: zero
include_coverage_flag: true
split_types: [CV1, CV2, CV3]
folds: [0, 1, 2, 3, 4]
ranking_k: [10, 20, 50]
seed: 17
output_dir: results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_with_flag

# Canonical invocation (Tier 1, with coverage flag):
# uv run python -m sl_benchmark_baseline \
#   --input-csv  <input_csv> \
#   --bags-npz   <bags_npz> \
#   --embedding-method pca_delta_meanpool \
#   --fallback-strategy zero \
#   --include-coverage-flag \
#   --split-types CV1 CV2 CV3 --folds 0 1 2 3 4 --ranking-k 10 20 50 \
#   --output-dir results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_with_flag
#
# Ablation (no coverage flag): rerun with --no-coverage-flag and a _no_flag output dir.
# Tier 2: rerun with the scvi_delta bags.npz and --embedding-method scvi_delta_meanpool.
```

- [ ] **Step 2: Run the FULL SL baseline suite + lint**

Run:
```bash
uv run python -m pytest tests/test_sl_benchmark_baseline_config.py tests/test_sl_benchmark_baseline_features.py tests/test_sl_benchmark_baseline_embeddings.py tests/test_sl_benchmark_baseline_models.py tests/test_sl_benchmark_baseline_evaluate.py tests/test_sl_benchmark_baseline_cli.py tests/test_sl_benchmark_baseline_data.py tests/test_sl_benchmark_baseline_metrics.py tests/test_sl_benchmark_baseline_init.py -v
uv run ruff check src/sl_benchmark_baseline/ tests/
uv run ruff format --check src/sl_benchmark_baseline/
```
Expected: all SL baseline tests PASS; ruff reports no errors. Fix any lint findings (line length, import order) before proceeding.

- [ ] **Step 3: Update the experiment doc**

In `docs/experiment/07_k562_sl_pair_perturbseq_augmented.md`, change the `Run status:` line (top of file) from `Stage 1 only — problem definition ... Implementation not yet started.` to:

```
Run status: MVP implementation completed YYYY-MM-DD (replace with date). Module
`src/sl_benchmark_baseline/` augmented mode (Tier 1 pca_delta_meanpool, Tier 2
scvi_delta_meanpool). Real-data CV1/CV2/CV3 run pending data availability.
```

Add a `## How to run` subsection under `## Method` pointing at the config file and the canonical invocation (copy the two commands from the YAML comment). Do NOT fabricate result numbers — leave the Results section's table as "pending real-data run" until an actual run is executed.

- [ ] **Step 4: Verify docs + config are consistent**

Run: `uv run python -c "import yaml; yaml.safe_load(open('configs/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_cv1_cv2_cv3.yaml'))"`
Expected: no exception (valid YAML).

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/07_k562_sl_pair_perturbseq_augmented/ docs/experiment/07_k562_sl_pair_perturbseq_augmented.md
git commit -m "docs: add exp07 production config and update run status"
```

---

## What This Plan Does NOT Cover (deferred)

- **Executing the real-data CV1/CV2/CV3 run** and recording actual NDCG/MAP/AUROC numbers. The plan builds and unit-tests the augmented machinery on synthetic fixtures; the production run depends on the gwps-derived `bags.npz` artifact (multi-GB, gitignored, built by exp03 `build-cell-bags`). Run it after Task 9 with the documented invocation, then fill the Results table.
- **Tier 2 representation training** (scVI on gwps). Tier 2 reuses the *same code path* with a different `bags.npz`; producing that NPZ is an exp03 `build-cell-bags` invocation with `single_cell_scvi_delta`, out of this module's scope.
- **Frozen-GMM-occupancy pooling.** MVP uses mean-pool only. GMM-occupancy is a follow-up that would add a pooling-strategy branch to `embeddings.py`.
- **Fallback-strategy ablation runs** (spec MLE flag C) and **with/without-flag comparison** (spec flag A) are *runbook* steps (rerun the CLI with different flags), not code — both flag values are already supported by Tasks 1/8.

---

## Self-Review

**1. Spec coverage** (against `docs/experiment/07_k562_sl_pair_perturbseq_augmented.md`):

| Spec section | Covered by |
|---|---|
| Full benchmark + fallback (Option A) | Task 3 (fallback + coverage mask), Task 6 (full_universe slice on full universe) |
| gwps-only transcriptome | Task 3 (single `bags_npz` input); config has one NPZ field |
| Per-gene embedding, covered→real+flag1 / uncovered→fallback+flag0 | Task 3 `align_to_universe` |
| Swap-invariant pair features: GeneEffect ++ [e_a+e_b, |e_a-e_b|, e_a⊙e_b] ++ coverage | Task 2 `build_augmented_pair_features` (coverage as swap-invariant min/max) |
| Tier 1: LogReg + XGBoost, same exp06 heads | Task 4 subclasses of LogRegModel/XGBModel |
| Tier 2: frozen exp03 representation pooled per-gene | Task 3 (NPZ-agnostic pooling) + Task 9 YAML (scvi_delta NPZ swap) |
| With/without coverage flag reporting | Task 1 + Task 8 (`include_coverage_flag` / `--no-coverage-flag`) |
| Covered-pair diagnostic slice | Task 6 (`covered_pairs` slice) |
| Primary baseline = exp06 re-run in same harness | Task 6 (A/B run alongside transcript models in augmented mode, identical metric code) |
| Unchanged metric/splits/universe/seed | Tasks 5-7 reuse `metrics.py`, `data.py`, seed 17, ranking_k unchanged |
| Manifest coverage fields | Task 7 |
| Scope guard: no generated bags, no SL claims, no label change | No task touches forward models, label definitions, or metric semantics |

No spec requirement is left without a task. The real-data *run* (not the code) is explicitly deferred above, consistent with the spec's "Stage 1 only / Implementation not yet started" status and the project's Plan→Confirm→Code rule.

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" left in code steps. Every code step shows complete code. The one date placeholder (`YYYY-MM-DD`) in Task 9 Step 3 is an intentional run-time value, flagged as "replace with date." The Results table is intentionally left "pending real-data run" — fabricating numbers would violate the verification guideline.

**3. Type consistency:**
- `build_augmented_pair_features(ea, eb, emb_a, emb_b, flag_a, flag_b, include_coverage_flag)` — same signature in Task 2 (def), Task 5 (score matrix), Task 6 (`_augmented_fold_data`/`_augmented_raw`). ✅
- `transcript_feature_names(dim, include_coverage_flag)` — lowercase function (renamed from the ALL_CAPS draft to match the `build_*` convention; `FEATURE_NAMES` remains a constant). Defined Task 2, used only in Task 2 tests. ✅
- `align_to_universe(table, symbols, fallback_strategy) -> (embeddings, coverage_mask)` — Task 3 def, Task 5 caller. `coverage_mask` is `dtype=int` (1/0). ✅
- `GeneUniverse.embeddings` / `.coverage_mask` — added Task 5 (complete drop-in), consumed Tasks 5/6/7. ✅
- `_build_augmented_score_matrix(model, universe, standardizer, include_coverage_flag)` — 4 params everywhere (Task 5 def, Task 5 test, Task 6 call). The first-draft inline `include_coverage_flag` expression was removed. ✅
- `_run_fold_augmented(train_df, train_base, base_std, pos_index, neg_index, seen_index, split_type, fold_id, config, universe)` — 10 params, no dead `test_pos`/`test_neg`; the `run_fold` dispatcher call matches. ✅
- Model names `"A_transcript"`, `"B_transcript"` — Task 4 def, Task 6 `.endswith("_transcript")` dispatch, Task 7 manifest list, Task 8 CLI test. ✅
- `slice` row key + `_summarize` groupby `["split_type","model","slice","metric"]` — Task 6. Existing tests use `.issubset`/`duplicated` and still hold (single slice in exp06 mode). ✅
- Covered-pair slice uses the unfiltered `seen_index` (same as full-universe) and is emitted only when `len(pos_cov) > 0`. Deterministic `synthetic_augmented_*` fixtures guarantee non-empty covered pairs. ✅

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-exp07-perturbseq-augmented-sl-mvp.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?









# Experiment 09 — Cross-Cell-Line Selectivity SL Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-cell-line "Selectivity" SL-pair model route (`A_xcl`,
`B_xcl`) to `src/sl_benchmark_baseline/`, derived from DepMap GeneEffect plus
mutation/copy-number/expression omics, comparable against the exp06
dependency-only floor on CV1/CV2/CV3.

**Architecture:** Mirror the existing exp07 `augmented` mode. A new
`selectivity` mode (enabled by a DepMap dir in config) computes a directional
Selectivity contrast `sel(a->b)` over the DepMap cell-line panel, symmetrizes it
into swap-invariant pair features, appends them to exp06's 5 GeneEffect scalars,
and adds `A_xcl`/`B_xcl` model columns. exp06's `A`/`B`/`C` re-run unchanged
through the shared harness (a parity gate asserts they reproduce the locked
floor). Results land in a new exp09 run dir.

**Tech Stack:** Python 3.11+, numpy, pandas, scikit-learn, xgboost, pytest,
ruff, uv. Library design follows the existing `sl_benchmark_baseline` package.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings.
- `uv run` prefix for all python/pytest/ruff invocations.
- No `print` in library code; use `logging`.
- No hardcoded paths or thresholds in library code; every threshold
  (`cn_loss_thr`, `expr_low_quantile`, `sel_n_min`, `sel_lambda`) flows from
  `SLBaselineConfig`.
- Handle specific exceptions; no bare `except`.
- Target <50 lines per function, <600 lines per file.
- **`Rand` negatives ONLY.** When selectivity mode is on, the run must reject any
  benchmark whose `negative_sampling_method` is not all `Rand`.
- Benchmark, CV1/CV2/CV3 splits, metric protocol, 9,471-gene universe, seed 17
  are UNCHANGED from exp06.
- No SL biological claim in code, comments, or docs: this is a benchmark-adapter
  feature, not a validated K562 SL assay.
- Selectivity computed from the external DepMap matrices does not depend on the
  SL train/test split — same leakage profile as exp06's K562 scalar.

## Reference Facts (verified during design)

- DepMap dir: `data/sl_dependency_v0/raw/depmap/` (gitignored).
- `CRISPRGeneEffect.csv`: 1,208 lines × 18,531 genes; index = `ACH-` model id;
  columns = `SYMBOL (ENTREZ)`; K562 = `ACH-000551`; ~4% NaN.
- `OmicsSomaticMutationsMatrixDamaging.csv` / `...Hotspot.csv`: binary 0/1;
  `ModelID` is a **column** (rows are integer-indexed); metadata cols to drop:
  `Unnamed: 0`, `SequencingID`, `ModelID`, `ModelConditionID`,
  `IsDefaultEntryForModel`, `IsDefaultEntryForMC`. Hotspot covers only 554 genes.
- `PortalOmicsCNGeneLog2.csv`: index = `ACH-` model id; log2, ~1.0 neutral; loss
  threshold default 0.8.
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`: `ModelID` is a column;
  log2(TPM+1); low-expression = per-gene bottom decile.
- Benchmark CSVs carry `gene_a_entrez_id`, `gene_b_entrez_id`,
  `negative_sampling_method` columns in addition to exp06's required columns.
- Composite-OR defective call clears `n>=20` (both groups) for 9,459/9,471
  benchmark genes; only ~12 hit fallback.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/sl_benchmark_baseline/selectivity.py` | Load DepMap GeneEffect + 3 omics; build composite defective mask; compute directional `sel(a->b)` over a gene universe; expose pan-essentiality + coverage. | Create |
| `src/sl_benchmark_baseline/features.py` | Add `build_selectivity_pair_features` + names. | Modify |
| `src/sl_benchmark_baseline/models.py` | Add `LogRegSelectivityModel`, `XGBSelectivityModel`, `build_selectivity_models`. | Modify |
| `src/sl_benchmark_baseline/config.py` | Add DepMap dir + threshold fields + `selectivity` property. | Modify |
| `src/sl_benchmark_baseline/data.py` | Add `Rand`-only guard helper. | Modify |
| `src/sl_benchmark_baseline/evaluate.py` | Thread selectivity table into `GeneUniverse`; selectivity score-matrix + fold path; diagnostic slices. | Modify |
| `src/sl_benchmark_baseline/__main__.py` | Add selectivity CLI flags. | Modify |
| `tests/test_sl_selectivity.py` | Unit tests for selectivity math, fallback, symmetry, features, guard. | Create |
| `tests/test_sl_parity_gate.py` | Assert A/B/C reproduce locked exp06 floor in selectivity mode. | Create |
| `docs/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md` | Experiment write-up. | Create |

The plan builds bottom-up: pure feature math (Task 1) → selectivity engine
(Tasks 2-4) → config/guard wiring (Task 5) → models (Task 6) → evaluate
integration (Task 7) → CLI (Task 8) → parity gate + smoke (Task 9) → write-up
(Task 10).

---

### Task 1: Selectivity pair features (pure, no I/O)

**Files:**
- Modify: `src/sl_benchmark_baseline/features.py` (append after `build_augmented_pair_features`)
- Test: `tests/test_sl_selectivity.py` (create)

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces:
  - `SELECTIVITY_FEATURE_NAMES: tuple[str, ...]` = `("sel_mean", "sel_absdiff", "pan_essential_min")`
  - `build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b) -> np.ndarray` returning shape `(n, 3)`: columns `[(sel_ab+sel_ba)/2, |sel_ab-sel_ba|, min(pan_a, pan_b)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sl_selectivity.py
"""Tests for cross-cell-line selectivity features and engine."""

from __future__ import annotations

import numpy as np

from sl_benchmark_baseline.features import (
    SELECTIVITY_FEATURE_NAMES,
    build_selectivity_pair_features,
)


def test_build_selectivity_pair_features_shape_and_values():
    sel_ab = np.array([1.0, 2.0])
    sel_ba = np.array([3.0, -2.0])
    pan_a = np.array([-0.5, -1.0])
    pan_b = np.array([-0.2, -3.0])
    out = build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b)
    assert out.shape == (2, 3)
    assert len(SELECTIVITY_FEATURE_NAMES) == 3
    # row 0: mean=(1+3)/2=2, absdiff=|1-3|=2, pan_min=min(-0.5,-0.2)=-0.5
    np.testing.assert_allclose(out[0], [2.0, 2.0, -0.5])
    # row 1: mean=0, absdiff=4, pan_min=-3.0
    np.testing.assert_allclose(out[1], [0.0, 4.0, -3.0])


def test_build_selectivity_pair_features_is_swap_invariant():
    sel_ab = np.array([1.0])
    sel_ba = np.array([3.0])
    pan_a = np.array([-0.5])
    pan_b = np.array([-0.2])
    forward = build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b)
    swapped = build_selectivity_pair_features(sel_ba, sel_ab, pan_b, pan_a)
    np.testing.assert_allclose(forward, swapped)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_selectivity_pair_features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# Append to src/sl_benchmark_baseline/features.py

SELECTIVITY_FEATURE_NAMES: tuple[str, ...] = (
    "sel_mean",
    "sel_absdiff",
    "pan_essential_min",
)


def build_selectivity_pair_features(
    sel_ab: np.ndarray,
    sel_ba: np.ndarray,
    pan_a: np.ndarray,
    pan_b: np.ndarray,
) -> np.ndarray:
    """Build swap-invariant selectivity features from directional contrasts.

    Args:
        sel_ab: Directional selectivity ``sel(a -> b)``, shape ``(n,)``.
        sel_ba: Directional selectivity ``sel(b -> a)``, shape ``(n,)``.
        pan_a: Pan-essentiality of gene a (mean GeneEffect), shape ``(n,)``.
        pan_b: Pan-essentiality of gene b (mean GeneEffect), shape ``(n,)``.

    Returns:
        Feature matrix of shape ``(n, 3)`` ordered by
        ``SELECTIVITY_FEATURE_NAMES``.
    """
    sel_ab = np.asarray(sel_ab, dtype=float)
    sel_ba = np.asarray(sel_ba, dtype=float)
    pan_a = np.asarray(pan_a, dtype=float)
    pan_b = np.asarray(pan_b, dtype=float)
    return np.column_stack(
        [
            (sel_ab + sel_ba) / 2.0,
            np.abs(sel_ab - sel_ba),
            np.minimum(pan_a, pan_b),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/features.py tests/test_sl_selectivity.py
git commit -m "feat: add swap-invariant selectivity pair features"
```

### Task 2: DepMap matrix loaders (GeneEffect + omics)

**Files:**
- Create: `src/sl_benchmark_baseline/selectivity.py`
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: nothing (pandas/numpy I/O).
- Produces:
  - `parse_entrez_columns(columns) -> dict[int, str]`: map Entrez id -> column
    label for `SYMBOL (ENTREZ)` headers; columns without a parenthesized int are
    skipped.
  - `load_gene_effect_matrix(path) -> tuple[pd.Index, dict[int, np.ndarray]]`:
    returns `(cell_line_index, {entrez: gene_effect_vector_over_lines})`.
  - `load_ach_indexed_matrix(path, cell_line_index) -> dict[int, np.ndarray]`:
    for `ACH-`-indexed matrices (CN); reindexed onto `cell_line_index` (missing
    lines -> NaN).
  - `load_modelid_column_matrix(path, cell_line_index) -> dict[int, np.ndarray]`:
    for matrices where `ModelID` is a column (damaging/hotspot/expression);
    drops metadata columns, sets `ModelID` as index, reindexes onto
    `cell_line_index`.

Note these read large CSVs; tests use tiny synthetic CSVs written to `tmp_path`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
import pandas as pd

from sl_benchmark_baseline.selectivity import (
    load_ach_indexed_matrix,
    load_gene_effect_matrix,
    load_modelid_column_matrix,
    parse_entrez_columns,
)


def test_parse_entrez_columns():
    cols = ["A1BG (1)", "TP53 (7157)", "weird_no_entrez"]
    mapping = parse_entrez_columns(cols)
    assert mapping == {1: "A1BG (1)", 7157: "TP53 (7157)"}


def test_load_gene_effect_matrix(tmp_path):
    path = tmp_path / "ge.csv"
    pd.DataFrame(
        {"GENEA (10)": [-0.1, -0.9], "GENEB (20)": [0.0, -1.5]},
        index=["ACH-1", "ACH-2"],
    ).to_csv(path)
    lines, vecs = load_gene_effect_matrix(path)
    assert list(lines) == ["ACH-1", "ACH-2"]
    np.testing.assert_allclose(vecs[10], [-0.1, -0.9])
    np.testing.assert_allclose(vecs[20], [0.0, -1.5])


def test_load_ach_indexed_matrix_reindexes(tmp_path):
    path = tmp_path / "cn.csv"
    pd.DataFrame({"GENEA (10)": [1.0, 0.5]}, index=["ACH-2", "ACH-9"]).to_csv(path)
    lines = pd.Index(["ACH-1", "ACH-2"])
    vecs = load_ach_indexed_matrix(path, lines)
    # ACH-1 absent -> NaN; ACH-2 -> 1.0; ACH-9 dropped (not in lines)
    assert np.isnan(vecs[10][0])
    np.testing.assert_allclose(vecs[10][1], 1.0)


def test_load_modelid_column_matrix_reindexes(tmp_path):
    path = tmp_path / "mut.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "SequencingID": ["s1", "s2"],
            "ModelID": ["ACH-2", "ACH-9"],
            "ModelConditionID": ["mc1", "mc2"],
            "IsDefaultEntryForModel": [True, True],
            "IsDefaultEntryForMC": [True, True],
            "GENEA (10)": [1, 0],
        }
    ).to_csv(path, index=False)
    lines = pd.Index(["ACH-1", "ACH-2"])
    vecs = load_modelid_column_matrix(path, lines)
    assert np.isnan(vecs[10][0])  # ACH-1 absent
    np.testing.assert_allclose(vecs[10][1], 1.0)  # ACH-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k "load or parse_entrez" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sl_benchmark_baseline.selectivity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/sl_benchmark_baseline/selectivity.py
"""Cross-cell-line Selectivity engine over DepMap GeneEffect + omics."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ENTREZ_RE = re.compile(r"\((\d+)\)\s*$")
_OMICS_META_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",
    "SequencingID",
    "ModelID",
    "ModelConditionID",
    "IsDefaultEntryForModel",
    "IsDefaultEntryForMC",
)


def parse_entrez_columns(columns: list[str]) -> dict[int, str]:
    """Map Entrez id -> column label for ``SYMBOL (ENTREZ)`` headers."""
    mapping: dict[int, str] = {}
    for col in columns:
        match = _ENTREZ_RE.search(str(col))
        if match is not None:
            mapping[int(match.group(1))] = col
    return mapping


def load_gene_effect_matrix(path: Path) -> tuple[pd.Index, dict[int, np.ndarray]]:
    """Load the GeneEffect matrix as ``(cell_line_index, {entrez: vector})``."""
    frame = pd.read_csv(path, index_col=0)
    entrez_map = parse_entrez_columns(list(frame.columns))
    vectors = {
        entrez: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        for entrez, col in entrez_map.items()
    }
    return frame.index, vectors


def load_ach_indexed_matrix(
    path: Path, cell_line_index: pd.Index
) -> dict[int, np.ndarray]:
    """Load an ``ACH-``-indexed omics matrix reindexed onto the GeneEffect lines."""
    frame = pd.read_csv(path, index_col=0)
    frame = frame.reindex(cell_line_index)
    entrez_map = parse_entrez_columns(list(frame.columns))
    return {
        entrez: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        for entrez, col in entrez_map.items()
    }


def load_modelid_column_matrix(
    path: Path, cell_line_index: pd.Index
) -> dict[int, np.ndarray]:
    """Load an omics matrix whose ``ModelID`` is a column, reindexed onto lines."""
    frame = pd.read_csv(path)
    if "ModelID" not in frame.columns:
        raise ValueError(f"{path} has no ModelID column")
    frame = frame.drop(
        columns=[c for c in _OMICS_META_COLUMNS if c in frame.columns],
        errors="ignore",
    ).set_index(frame["ModelID"])
    frame = frame.reindex(cell_line_index)
    entrez_map = parse_entrez_columns(list(frame.columns))
    return {
        entrez: pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        for entrez, col in entrez_map.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k "load or parse_entrez" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/selectivity.py tests/test_sl_selectivity.py
git commit -m "feat: add DepMap GeneEffect and omics matrix loaders"
```

### Task 3: Composite defective mask

**Files:**
- Modify: `src/sl_benchmark_baseline/selectivity.py` (append)
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: per-entrez vectors from Task 2 loaders; thresholds.
- Produces:
  - `build_defective_mask(entrez, n_lines, damaging, hotspot, cn_log2, expr, cn_loss_thr, expr_low_quantile) -> np.ndarray`
    returning a boolean vector of length `n_lines`. A line is defective if ANY
    channel fires: damaging==1, hotspot==1, `cn_log2 < cn_loss_thr`, or
    `expr <= per-gene quantile(expr_low_quantile)`. A missing channel (entrez
    absent from that dict) contributes nothing. NaN entries never fire.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
from sl_benchmark_baseline.selectivity import build_defective_mask


def test_build_defective_mask_or_across_channels():
    n = 5
    entrez = 10
    damaging = {10: np.array([1.0, 0.0, 0.0, 0.0, 0.0])}
    hotspot = {10: np.array([0.0, 1.0, 0.0, 0.0, 0.0])}
    cn = {10: np.array([1.0, 1.0, 0.5, 1.0, np.nan])}  # line 2 is a loss
    expr = {10: np.array([9.0, 9.0, 9.0, 0.0, 9.0])}  # line 3 lowest decile
    mask = build_defective_mask(
        entrez, n, damaging, hotspot, cn, expr,
        cn_loss_thr=0.8, expr_low_quantile=0.10,
    )
    # lines 0(dmg),1(hotspot),2(cn loss),3(low expr) defective; line 4 not
    assert mask.tolist() == [True, True, True, True, False]


def test_build_defective_mask_missing_channels():
    n = 3
    entrez = 99  # absent from all dicts
    mask = build_defective_mask(entrez, n, {}, {}, {}, {}, 0.8, 0.10)
    assert mask.tolist() == [False, False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k defective -v`
Expected: FAIL with `ImportError: cannot import name 'build_defective_mask'`.

- [ ] **Step 3: Write minimal implementation**

```python
# Append to src/sl_benchmark_baseline/selectivity.py

def build_defective_mask(
    entrez: int,
    n_lines: int,
    damaging: dict[int, np.ndarray],
    hotspot: dict[int, np.ndarray],
    cn_log2: dict[int, np.ndarray],
    expr: dict[int, np.ndarray],
    cn_loss_thr: float,
    expr_low_quantile: float,
) -> np.ndarray:
    """Composite-OR defective call for one gene across the cell-line panel.

    A line is defective if any available channel fires: damaging mutation,
    hotspot mutation, copy-number loss (``cn_log2 < cn_loss_thr``), or low
    expression (``expr <= per-gene quantile``). Missing channels and NaN values
    never fire.

    Args:
        entrez: Entrez id of the anchor gene.
        n_lines: Number of cell lines (length of the output mask).
        damaging: Entrez -> binary damaging-mutation vector.
        hotspot: Entrez -> binary hotspot-mutation vector.
        cn_log2: Entrez -> log2 copy-number vector.
        expr: Entrez -> log2(TPM+1) expression vector.
        cn_loss_thr: Copy-number loss threshold (strictly less than).
        expr_low_quantile: Quantile defining low expression (<=).

    Returns:
        Boolean mask of shape ``(n_lines,)``.
    """
    mask = np.zeros(n_lines, dtype=bool)
    if entrez in damaging:
        mask |= np.nan_to_num(damaging[entrez], nan=0.0) >= 1.0
    if entrez in hotspot:
        mask |= np.nan_to_num(hotspot[entrez], nan=0.0) >= 1.0
    if entrez in cn_log2:
        vec = cn_log2[entrez]
        mask |= np.where(np.isnan(vec), False, vec < cn_loss_thr)
    if entrez in expr:
        vec = expr[entrez]
        finite = vec[~np.isnan(vec)]
        if finite.size > 0:
            thr = float(np.quantile(finite, expr_low_quantile))
            mask |= np.where(np.isnan(vec), False, vec <= thr)
    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k defective -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/selectivity.py tests/test_sl_selectivity.py
git commit -m "feat: add composite-OR defective mask"
```

### Task 4: SelectivityTable engine (full sel matrix + pan-essentiality + coverage)

**Files:**
- Modify: `src/sl_benchmark_baseline/selectivity.py` (append)
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: Task 2 loaders, Task 3 `build_defective_mask`.
- Produces:
  - `SelectivityTable` frozen dataclass with: `entrez_order: tuple[int, ...]`,
    `sel_matrix: np.ndarray` (n×n, `sel[i,j] = sel(gene_i -> gene_j)`),
    `pan_essential: np.ndarray` (n,), `coverage_flag: np.ndarray` (n,) int,
    `index_by_entrez: dict[int, int]`.
  - `build_selectivity_table(entrez_order, gene_effect_vectors, damaging, hotspot, cn_log2, expr, cn_loss_thr, expr_low_quantile, n_min) -> SelectivityTable`.

**Math:** With `D` the (n_anchor × n_lines) boolean defective matrix, `GE0` the
(n_lines × n_gene) gene-effect matrix with NaN→0, and `valid` the (n_lines ×
n_gene) finite-mask: `def_sum = D @ GE0`, `def_cnt = D @ valid`, `total_sum`/
`total_cnt` are column sums, `intact = total - def`, and
`sel = intact_sum/intact_cnt - def_sum/def_cnt`. Anchors with
`|C+| < n_min or |C-| < n_min` (line-level group sizes) get `sel(a->:) = 0` and
`coverage_flag = 0`. `pan_essential[i] = nanmean(GE[:, i])`.

**Memory note (production):** for n=9,471, `sel_matrix` is ~359 MB float32; the
two intermediate matmul outputs are similar. Cast `GE0`/`valid`/`D` to float32
and free intermediates. Peak ~1.5 GB — acceptable; build once and cache.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
from sl_benchmark_baseline.selectivity import (
    SelectivityTable,
    build_selectivity_table,
)


def _toy_inputs():
    # 3 cell lines, genes 10 and 20
    ge = {
        10: np.array([-1.0, -0.5, 0.0]),
        20: np.array([0.2, -2.0, -1.0]),
    }
    damaging = {
        10: np.array([1.0, 0.0, 0.0]),  # line 0 defective for gene 10
        20: np.array([0.0, 1.0, 0.0]),  # line 1 defective for gene 20
    }
    return ge, damaging


def test_build_selectivity_table_values():
    ge, damaging = _toy_inputs()
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors=ge,
        damaging=damaging,
        hotspot={},
        cn_log2={},
        expr={},
        cn_loss_thr=0.8,
        expr_low_quantile=0.10,
        n_min=1,
    )
    assert isinstance(table, SelectivityTable)
    i, j = table.index_by_entrez[10], table.index_by_entrez[20]
    # sel(10->20) = mean(intact d20) - mean(def d20)
    #             = mean([-2.0,-1.0]) - mean([0.2]) = -1.5 - 0.2 = -1.7
    np.testing.assert_allclose(table.sel_matrix[i, j], -1.7, atol=1e-6)
    # sel(20->10) = mean([-1.0,0.0]) - mean([-0.5]) = -0.5 - (-0.5) = 0.0
    np.testing.assert_allclose(table.sel_matrix[j, i], 0.0, atol=1e-6)
    # pan_essential
    np.testing.assert_allclose(table.pan_essential[i], -0.5, atol=1e-6)
    np.testing.assert_allclose(table.pan_essential[j], -0.9333333, atol=1e-6)
    assert table.coverage_flag.tolist() == [1, 1]


def test_build_selectivity_table_n_min_fallback():
    ge, damaging = _toy_inputs()
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors=ge,
        damaging=damaging,
        hotspot={}, cn_log2={}, expr={},
        cn_loss_thr=0.8, expr_low_quantile=0.10,
        n_min=2,  # gene10 has only 1 defective line -> fallback
    )
    i = table.index_by_entrez[10]
    assert table.coverage_flag[i] == 0
    np.testing.assert_allclose(table.sel_matrix[i, :], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k selectivity_table -v`
Expected: FAIL with `ImportError: cannot import name 'build_selectivity_table'`.

- [ ] **Step 3: Write minimal implementation**

```python
# Append to src/sl_benchmark_baseline/selectivity.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectivityTable:
    """Full directional Selectivity matrix over a gene universe.

    Attributes:
        entrez_order: Universe gene Entrez ids in row/column order.
        sel_matrix: ``(n, n)`` where ``sel_matrix[i, j] = sel(gene_i -> gene_j)``.
        pan_essential: ``(n,)`` mean GeneEffect per gene across all lines.
        coverage_flag: ``(n,)`` 1 if anchor had >= n_min defective and intact
            lines, else 0 (and its ``sel`` row is zeroed).
        essential_fraction: ``(n,)`` fraction of finite lines with GeneEffect
            below ``essential_effect_thr`` (used by the non-pan-essential
            diagnostic slice).
        index_by_entrez: Entrez id -> row index.
    """

    entrez_order: tuple[int, ...]
    sel_matrix: np.ndarray
    pan_essential: np.ndarray
    coverage_flag: np.ndarray
    essential_fraction: np.ndarray
    index_by_entrez: dict[int, int]


def build_selectivity_table(
    entrez_order: tuple[int, ...],
    gene_effect_vectors: dict[int, np.ndarray],
    damaging: dict[int, np.ndarray],
    hotspot: dict[int, np.ndarray],
    cn_log2: dict[int, np.ndarray],
    expr: dict[int, np.ndarray],
    cn_loss_thr: float,
    expr_low_quantile: float,
    n_min: int,
    essential_effect_thr: float = -0.5,
) -> SelectivityTable:
    """Build the directional Selectivity matrix for a gene universe."""
    n_gene = len(entrez_order)
    sample = next(iter(gene_effect_vectors.values()))
    n_lines = sample.shape[0]

    ge = np.full((n_lines, n_gene), np.nan, dtype=np.float32)
    for col, entrez in enumerate(entrez_order):
        if entrez in gene_effect_vectors:
            ge[:, col] = gene_effect_vectors[entrez]
    valid = (~np.isnan(ge)).astype(np.float32)
    ge0 = np.nan_to_num(ge, nan=0.0).astype(np.float32)
    with np.errstate(invalid="ignore"):
        pan_essential = np.nanmean(ge, axis=0)
    pan_essential = np.nan_to_num(pan_essential, nan=0.0)
    below = ((ge < essential_effect_thr) & ~np.isnan(ge)).sum(axis=0)
    line_counts = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        essential_fraction = np.where(line_counts > 0, below / line_counts, 0.0)

    defective = np.zeros((n_gene, n_lines), dtype=np.float32)
    for row, entrez in enumerate(entrez_order):
        defective[row] = build_defective_mask(
            entrez, n_lines, damaging, hotspot, cn_log2, expr,
            cn_loss_thr, expr_low_quantile,
        )

    def_sum = defective @ ge0
    def_cnt = defective @ valid
    total_sum = ge0.sum(axis=0, keepdims=True)
    total_cnt = valid.sum(axis=0, keepdims=True)
    intact_sum = total_sum - def_sum
    intact_cnt = total_cnt - def_cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        dep_def = np.where(def_cnt > 0, def_sum / def_cnt, 0.0)
        dep_intact = np.where(intact_cnt > 0, intact_sum / intact_cnt, 0.0)
    sel_matrix = (dep_intact - dep_def).astype(np.float32)

    n_def_lines = defective.sum(axis=1)
    n_intact_lines = n_lines - n_def_lines
    coverage_flag = (
        (n_def_lines >= n_min) & (n_intact_lines >= n_min)
    ).astype(int)
    sel_matrix[coverage_flag == 0, :] = 0.0

    return SelectivityTable(
        entrez_order=tuple(entrez_order),
        sel_matrix=sel_matrix,
        pan_essential=pan_essential.astype(float),
        coverage_flag=coverage_flag,
        essential_fraction=essential_fraction.astype(float),
        index_by_entrez={e: i for i, e in enumerate(entrez_order)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k selectivity_table -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/selectivity.py tests/test_sl_selectivity.py
git commit -m "feat: add SelectivityTable engine with n_min fallback"
```

### Task 5: Config fields, `selectivity` property, and Rand-only guard

**Files:**
- Modify: `src/sl_benchmark_baseline/config.py`
- Modify: `src/sl_benchmark_baseline/data.py` (append guard helper)
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `SLBaselineConfig` gains: `depmap_dir: Path | None = None`,
    `cn_loss_thr: float = 0.8`, `expr_low_quantile: float = 0.10`,
    `sel_n_min: int = 20`, `sel_lambda: float = 0.0`, and property
    `selectivity -> bool` (`depmap_dir is not None`).
  - `data.assert_rand_only(frame) -> None`: raises `ValueError` if any
    `negative_sampling_method` value is not `"Rand"`. No-op if the column is
    absent (older CSVs).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
import pytest

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.data import assert_rand_only


def test_config_selectivity_property():
    assert SLBaselineConfig().selectivity is False
    cfg = SLBaselineConfig(depmap_dir=Path("data/sl_dependency_v0/raw/depmap"))
    assert cfg.selectivity is True
    assert cfg.cn_loss_thr == 0.8
    assert cfg.sel_n_min == 20


def test_assert_rand_only():
    rand = pd.DataFrame({"negative_sampling_method": ["Rand", "Rand"]})
    assert_rand_only(rand)  # no raise
    no_col = pd.DataFrame({"x": [1]})
    assert_rand_only(no_col)  # no raise
    dep = pd.DataFrame({"negative_sampling_method": ["Rand", "Dep"]})
    with pytest.raises(ValueError, match="Rand"):
        assert_rand_only(dep)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k "config_selectivity or rand_only" -v`
Expected: FAIL with `AttributeError`/`ImportError` (`selectivity`/`assert_rand_only` missing).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_benchmark_baseline/config.py`, add fields after
`include_coverage_flag: bool = True` (before the `augmented` property), and add
the new property after `augmented`:

```python
    depmap_dir: Path | None = None
    cn_loss_thr: float = 0.8
    expr_low_quantile: float = 0.10
    sel_n_min: int = 20
    sel_lambda: float = 0.0
```

```python
    @property
    def selectivity(self) -> bool:
        """True when a DepMap dir is supplied (exp09 selectivity mode)."""
        return self.depmap_dir is not None
```

Also extend the class docstring Attributes with one line each (depmap_dir,
cn_loss_thr, expr_low_quantile, sel_n_min, sel_lambda).

In `src/sl_benchmark_baseline/data.py`, append:

```python
def assert_rand_only(frame: pd.DataFrame) -> None:
    """Reject non-``Rand`` negative sampling (leakage guard for selectivity).

    Args:
        frame: Benchmark DataFrame.

    Raises:
        ValueError: If a ``negative_sampling_method`` column is present and any
            value is not ``"Rand"``.
    """
    if "negative_sampling_method" not in frame.columns:
        return
    methods = set(frame["negative_sampling_method"].unique())
    if methods != {"Rand"}:
        raise ValueError(
            "selectivity mode requires Rand negatives only; found "
            f"negative_sampling_method values: {sorted(methods)}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k "config_selectivity or rand_only" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/config.py src/sl_benchmark_baseline/data.py tests/test_sl_selectivity.py
git commit -m "feat: add selectivity config fields and Rand-only guard"
```

### Task 6: Selectivity models and factory

**Files:**
- Modify: `src/sl_benchmark_baseline/models.py`
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: `LogRegModel`, `XGBModel`, `FoldData` (existing).
- Produces:
  - `LogRegSelectivityModel(LogRegModel)` with `name = "A_xcl"`.
  - `XGBSelectivityModel(XGBModel)` with `name = "B_xcl"`.
  - `build_selectivity_models(config) -> list`: returns
    `[LogRegModel, XGBModel, FrequencyProbeModel, LogRegSelectivityModel, XGBSelectivityModel]`
    — baseline A/B/C (for the parity gate) plus the two `_xcl` columns.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
from sl_benchmark_baseline.models import (
    LogRegSelectivityModel,
    XGBSelectivityModel,
    build_selectivity_models,
)


def test_selectivity_models_names_and_factory():
    cfg = SLBaselineConfig()
    assert LogRegSelectivityModel(cfg).name == "A_xcl"
    assert XGBSelectivityModel(cfg).name == "B_xcl"
    names = [m.name for m in build_selectivity_models(cfg)]
    assert names == ["A", "B", "C", "A_xcl", "B_xcl"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k selectivity_models -v`
Expected: FAIL with `ImportError: cannot import name 'LogRegSelectivityModel'`.

- [ ] **Step 3: Write minimal implementation**

In `src/sl_benchmark_baseline/models.py`, add after `XGBTranscriptModel`:

```python
class LogRegSelectivityModel(LogRegModel):
    """Model A_xcl: logistic regression on GeneEffect + selectivity features."""

    name = "A_xcl"


class XGBSelectivityModel(XGBModel):
    """Model B_xcl: XGBoost on GeneEffect + selectivity features."""

    name = "B_xcl"
```

And add a factory after `build_augmented_models`:

```python
def build_selectivity_models(config: SLBaselineConfig) -> list:
    """exp09 models: baseline A/B/C + cross-cell-line A_xcl/B_xcl."""
    return [
        LogRegModel(config),
        XGBModel(config),
        FrequencyProbeModel(config),
        LogRegSelectivityModel(config),
        XGBSelectivityModel(config),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k selectivity_models -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/models.py tests/test_sl_selectivity.py
git commit -m "feat: add A_xcl/B_xcl selectivity models and factory"
```

### Task 7: DepMap-dir loader + universe-aligned selectivity arrays

**Files:**
- Modify: `src/sl_benchmark_baseline/selectivity.py` (append)
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: Task 2 loaders, Task 4 `build_selectivity_table`, `SLBaselineConfig`.
- Produces:
  - `STANDARD_FILES: dict[str, str]` mapping role -> filename (gene_effect,
    damaging, hotspot, cn, expr).
  - `build_selectivity_table_from_depmap(depmap_dir, entrez_order, config) -> SelectivityTable`:
    reads the 5 CSVs from `depmap_dir` and builds the table for `entrez_order`.
  - `UniverseSelectivity` frozen dataclass: `sel_matrix: np.ndarray` (n_uni ×
    n_uni, universe-aligned, lambda-penalized), `pan_essential: np.ndarray`
    (n_uni), `coverage_flag: np.ndarray` (n_uni int),
    `essential_fraction: np.ndarray` (n_uni).
  - `align_selectivity_to_universe(table, entrez_universe, sel_lambda) -> UniverseSelectivity`:
    gathers table rows/cols into universe order (entrez missing from table ->
    sel 0, pan 0, coverage 0, essential_fraction 0); applies
    `sel[i,j] -= sel_lambda * max(0, -pan[j])`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
from sl_benchmark_baseline.selectivity import (
    UniverseSelectivity,
    align_selectivity_to_universe,
)


def test_align_selectivity_to_universe_gather_and_missing():
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors={
            10: np.array([-1.0, -0.5, 0.0]),
            20: np.array([0.2, -2.0, -1.0]),
        },
        damaging={
            10: np.array([1.0, 0.0, 0.0]),
            20: np.array([0.0, 1.0, 0.0]),
        },
        hotspot={}, cn_log2={}, expr={},
        cn_loss_thr=0.8, expr_low_quantile=0.10, n_min=1,
    )
    # universe order: [20, 10, 999(missing)]
    uni = align_selectivity_to_universe(table, [20, 10, 999], sel_lambda=0.0)
    assert isinstance(uni, UniverseSelectivity)
    assert uni.sel_matrix.shape == (3, 3)
    # sel(20->10) lives at [0,1] in universe order; equals table sel(20->10)=0.0
    np.testing.assert_allclose(uni.sel_matrix[0, 1], 0.0, atol=1e-6)
    # sel(10->20) at [1,0] = -1.7
    np.testing.assert_allclose(uni.sel_matrix[1, 0], -1.7, atol=1e-6)
    # missing gene row/col all zero, coverage 0
    np.testing.assert_allclose(uni.sel_matrix[2, :], 0.0)
    assert uni.coverage_flag[2] == 0
    assert uni.coverage_flag[0] == 1 and uni.coverage_flag[1] == 1


def test_align_selectivity_lambda_penalty():
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors={
            10: np.array([-1.0, -0.5, 0.0]),
            20: np.array([0.2, -2.0, -1.0]),
        },
        damaging={10: np.array([1.0, 0.0, 0.0]), 20: np.array([0.0, 1.0, 0.0])},
        hotspot={}, cn_log2={}, expr={},
        cn_loss_thr=0.8, expr_low_quantile=0.10, n_min=1,
    )
    uni0 = align_selectivity_to_universe(table, [10, 20], sel_lambda=0.0)
    uni1 = align_selectivity_to_universe(table, [10, 20], sel_lambda=1.0)
    # pan_essential(gene20) ~ -0.9333; penalty on col j=20: -1*max(0,0.9333)
    j = 1
    delta = uni1.sel_matrix[:, j] - uni0.sel_matrix[:, j]
    np.testing.assert_allclose(delta, -0.9333333, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k "align_selectivity" -v`
Expected: FAIL with `ImportError: cannot import name 'align_selectivity_to_universe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# Append to src/sl_benchmark_baseline/selectivity.py

STANDARD_FILES: dict[str, str] = {
    "gene_effect": "CRISPRGeneEffect.csv",
    "damaging": "OmicsSomaticMutationsMatrixDamaging.csv",
    "hotspot": "OmicsSomaticMutationsMatrixHotspot.csv",
    "cn": "PortalOmicsCNGeneLog2.csv",
    "expr": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
}


def build_selectivity_table_from_depmap(
    depmap_dir: Path,
    entrez_order: tuple[int, ...],
    config: "object",
) -> SelectivityTable:
    """Load the 5 DepMap CSVs and build the Selectivity table for a universe.

    Args:
        depmap_dir: Directory containing the standard DepMap files.
        entrez_order: Universe gene Entrez ids in canonical order.
        config: Object exposing ``cn_loss_thr``, ``expr_low_quantile``,
            ``sel_n_min`` (e.g. ``SLBaselineConfig``).

    Returns:
        A :class:`SelectivityTable`.
    """
    depmap_dir = Path(depmap_dir)
    lines, gene_effect = load_gene_effect_matrix(
        depmap_dir / STANDARD_FILES["gene_effect"]
    )
    damaging = load_modelid_column_matrix(
        depmap_dir / STANDARD_FILES["damaging"], lines
    )
    hotspot = load_modelid_column_matrix(depmap_dir / STANDARD_FILES["hotspot"], lines)
    cn_log2 = load_ach_indexed_matrix(depmap_dir / STANDARD_FILES["cn"], lines)
    expr = load_modelid_column_matrix(depmap_dir / STANDARD_FILES["expr"], lines)
    return build_selectivity_table(
        entrez_order=entrez_order,
        gene_effect_vectors=gene_effect,
        damaging=damaging,
        hotspot=hotspot,
        cn_log2=cn_log2,
        expr=expr,
        cn_loss_thr=config.cn_loss_thr,
        expr_low_quantile=config.expr_low_quantile,
        n_min=config.sel_n_min,
    )


@dataclass(frozen=True)
class UniverseSelectivity:
    """Selectivity arrays gathered into the benchmark gene-universe order."""

    sel_matrix: np.ndarray
    pan_essential: np.ndarray
    coverage_flag: np.ndarray
    essential_fraction: np.ndarray


def align_selectivity_to_universe(
    table: SelectivityTable,
    entrez_universe: list[int],
    sel_lambda: float,
) -> UniverseSelectivity:
    """Gather a SelectivityTable into universe order and apply the lambda penalty.

    Genes whose Entrez id is absent from ``table`` get zero sel rows/cols, zero
    pan-essentiality, zero essential-fraction, and coverage 0. The penalty
    subtracts ``sel_lambda * max(0, -pan_essential[j])`` from every entry in
    column ``j``.

    Args:
        table: Source SelectivityTable.
        entrez_universe: Universe gene Entrez ids in canonical order.
        sel_lambda: Pan-essentiality soft-penalty coefficient (0 disables it).

    Returns:
        A :class:`UniverseSelectivity`.
    """
    n = len(entrez_universe)
    rows = np.array(
        [table.index_by_entrez.get(int(e), -1) for e in entrez_universe], dtype=int
    )
    present = rows >= 0
    safe_rows = np.where(present, rows, 0)
    sel = table.sel_matrix[np.ix_(safe_rows, safe_rows)].astype(float)
    sel[~present, :] = 0.0
    sel[:, ~present] = 0.0
    pan = np.where(present, table.pan_essential[safe_rows], 0.0)
    cov = np.where(present, table.coverage_flag[safe_rows], 0).astype(int)
    ess = np.where(present, table.essential_fraction[safe_rows], 0.0)
    if sel_lambda != 0.0:
        penalty = sel_lambda * np.maximum(0.0, -pan)
        sel = sel - penalty[np.newaxis, :]
    return UniverseSelectivity(
        sel_matrix=sel,
        pan_essential=pan,
        coverage_flag=cov,
        essential_fraction=ess,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k "align_selectivity" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the whole selectivity suite + lint, then commit**

```bash
uv run python -m pytest tests/test_sl_selectivity.py -v
uv run ruff check src/sl_benchmark_baseline/selectivity.py
git add src/sl_benchmark_baseline/selectivity.py tests/test_sl_selectivity.py
git commit -m "feat: add depmap-dir loader and universe-aligned selectivity"
```

### Task 8: Evaluate integration — universe wiring + selectivity features

**Files:**
- Modify: `src/sl_benchmark_baseline/evaluate.py`
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: `UniverseSelectivity`, `build_selectivity_pair_features`,
  `build_pair_features`, `Standardizer`, `FoldData`.
- Produces (module-level in `evaluate.py`):
  - `GeneUniverse` gains two optional fields: `entrez: np.ndarray | None = None`,
    `selectivity: "UniverseSelectivity | None" = None`.
  - `_selectivity_raw(frame, universe, config) -> np.ndarray`: unstandardized
    `(n, 8)` block = `build_pair_features(ea, eb)` (5) concatenated with
    `build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b)` (3), where
    `sel_ab = universe.selectivity.sel_matrix[a_idx, b_idx]` etc.
  - `_build_selectivity_score_matrix(model, universe, standardizer) -> np.ndarray`.

This task wires the data through but does not yet branch `run_cv` (Task 8b).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
from sl_benchmark_baseline.evaluate import GeneUniverse, _selectivity_raw
from sl_benchmark_baseline.selectivity import UniverseSelectivity as _US


def _toy_universe_with_selectivity():
    symbols = np.array(["GA", "GB"])
    sel = _US(
        sel_matrix=np.array([[0.0, -1.7], [0.0, 0.0]]),
        pan_essential=np.array([-0.5, -0.93]),
        coverage_flag=np.array([1, 1]),
        essential_fraction=np.array([0.2, 0.8]),
    )
    return GeneUniverse(
        keys=(10, 20),
        symbols=symbols,
        gene_effects=np.array([-0.5, -1.0]),
        index_by_key={10: 0, 20: 1},
        entrez=np.array([10, 20]),
        selectivity=sel,
    )


def test_selectivity_raw_block_width_and_values():
    universe = _toy_universe_with_selectivity()
    frame = pd.DataFrame(
        {
            "gene_a_unified_id": [10],
            "gene_b_unified_id": [20],
            "gene_a_symbol": ["GA"],
            "gene_b_symbol": ["GB"],
            "gene_a_k562_gene_effect": [-0.5],
            "gene_b_k562_gene_effect": [-1.0],
            "sl_label": [1],
        }
    )
    cfg = SLBaselineConfig()
    raw = _selectivity_raw(frame, universe, cfg)
    assert raw.shape == (1, 8)
    # selectivity block (cols 5,6,7): sel_mean=(-1.7+0.0)/2=-0.85,
    # absdiff=1.7, pan_min=min(-0.5,-0.93)=-0.93
    np.testing.assert_allclose(raw[0, 5:], [-0.85, 1.7, -0.93], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k selectivity_raw -v`
Expected: FAIL with `ImportError: cannot import name '_selectivity_raw'` (or `TypeError` on unexpected `entrez`/`selectivity` kwargs).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_benchmark_baseline/evaluate.py`, extend the `GeneUniverse` dataclass
fields (after `coverage_mask`):

```python
    entrez: np.ndarray | None = None
    selectivity: "UniverseSelectivity | None" = None
```

Add the import at the top (with the other `sl_benchmark_baseline` imports):

```python
from sl_benchmark_baseline.selectivity import (
    UniverseSelectivity,
    align_selectivity_to_universe,
    build_selectivity_table_from_depmap,
)
```

Add `build_selectivity_pair_features` to the existing `features` import block.

Then add the raw-feature helper (place near `_augmented_raw`):

```python
def _selectivity_raw(
    frame: pd.DataFrame, universe: GeneUniverse, config: SLBaselineConfig
) -> np.ndarray:
    """Unstandardized GeneEffect(5) + selectivity(3) block for a frame."""
    if universe.selectivity is None:
        raise ValueError("selectivity score requires universe.selectivity")
    pair_idx = _pair_indices(frame, universe)
    a_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1]
    sel = universe.selectivity
    sel_ab = sel.sel_matrix[a_idx, b_idx]
    sel_ba = sel.sel_matrix[b_idx, a_idx]
    pan_a = sel.pan_essential[a_idx]
    pan_b = sel.pan_essential[b_idx]
    return np.column_stack(
        [
            build_pair_features(
                frame["gene_a_k562_gene_effect"].to_numpy(),
                frame["gene_b_k562_gene_effect"].to_numpy(),
            ),
            build_selectivity_pair_features(sel_ab, sel_ba, pan_a, pan_b),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k selectivity_raw -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/evaluate.py tests/test_sl_selectivity.py
git commit -m "feat: wire selectivity into GeneUniverse and pair features"
```

### Task 8b: Evaluate integration — score matrix, fold path, run_cv branch

**Files:**
- Modify: `src/sl_benchmark_baseline/evaluate.py`
- Test: covered by Task 9's smoke test (this task is integration glue; its unit
  surface is small and exercised end-to-end in Task 9).

**Interfaces:**
- Consumes: Task 8 helpers, `build_selectivity_models`, `assert_rand_only`.
- Produces:
  - `_build_selectivity_score_matrix(model, universe, standardizer) -> np.ndarray`
  - `_run_fold_selectivity(...) -> list[dict]` emitting `A`, `B`, `C`,
    `A_xcl`, `B_xcl` rows with slices `full_universe`, plus `non_pan_essential`
    and `covered_pairs` for the `_xcl` models.
  - `run_cv` branches: `if config.selectivity:` builds the universe with
    selectivity, calls `assert_rand_only(frame)`, runs the selectivity fold path.

- [ ] **Step 1: Add the selectivity score-matrix builder**

Place after `_build_augmented_score_matrix`:

```python
def _build_selectivity_score_matrix(
    model: object,
    universe: GeneUniverse,
    standardizer: Standardizer,
) -> np.ndarray:
    """Score all candidate pairs using GeneEffect+selectivity features."""
    if universe.selectivity is None:
        raise ValueError("selectivity score matrix requires universe.selectivity")
    sel = universe.selectivity
    n_gene = len(universe.symbols)
    score_matrix = np.zeros((n_gene, n_gene), dtype=float)
    all_idx = np.arange(n_gene)
    for start in range(0, n_gene, SCORE_MATRIX_CHUNK_ROWS):
        stop = min(start + SCORE_MATRIX_CHUNK_ROWS, n_gene)
        rows = np.arange(start, stop)
        a_idx = np.repeat(rows, n_gene)
        b_idx = np.tile(all_idx, len(rows))
        raw = np.column_stack(
            [
                build_pair_features(
                    universe.gene_effects[a_idx], universe.gene_effects[b_idx]
                ),
                build_selectivity_pair_features(
                    sel.sel_matrix[a_idx, b_idx],
                    sel.sel_matrix[b_idx, a_idx],
                    sel.pan_essential[a_idx],
                    sel.pan_essential[b_idx],
                ),
            ]
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

- [ ] **Step 2: Add the non-pan-essential pair mask helper**

Place near `_covered_pair_mask`:

```python
def _non_pan_essential_mask(
    index: np.ndarray, universe: GeneUniverse, max_essential_fraction: float = 0.5
) -> np.ndarray:
    """Pairs where neither gene is broadly essential (essential_fraction <= thr)."""
    if len(index) == 0 or universe.selectivity is None:
        return np.zeros(len(index), dtype=bool)
    ess = universe.selectivity.essential_fraction
    return (ess[index[:, 0]] <= max_essential_fraction) & (
        ess[index[:, 1]] <= max_essential_fraction
    )


def _selectivity_covered_mask(index: np.ndarray, universe: GeneUniverse) -> np.ndarray:
    """Pairs whose two genes both cleared the selectivity n_min coverage bar."""
    if len(index) == 0 or universe.selectivity is None:
        return np.zeros(len(index), dtype=bool)
    cov = universe.selectivity.coverage_flag
    return (cov[index[:, 0]] == 1) & (cov[index[:, 1]] == 1)
```

- [ ] **Step 3: Add the selectivity fold path**

```python
def _run_fold_selectivity(
    train_df: pd.DataFrame,
    train_base: FoldData,
    base_std: Standardizer,
    pos_index: np.ndarray,
    neg_index: np.ndarray,
    seen_index: np.ndarray,
    split_type: str,
    fold_id: int,
    config: SLBaselineConfig,
    universe: GeneUniverse,
) -> list[dict[str, object]]:
    """Fit baseline A/B/C + A_xcl/B_xcl, emitting full + diagnostic slices."""
    sel_std = Standardizer.fit(_selectivity_raw(train_df, universe, config))
    train_sel = FoldData(
        df=train_df,
        features=sel_std.transform(_selectivity_raw(train_df, universe, config)),
        labels=train_df["sl_label"].to_numpy(dtype=int),
    )
    rows: list[dict[str, object]] = []
    for model in build_selectivity_models(config):
        if model.name.endswith("_xcl"):
            model.fit(train_sel)
            sm = _build_selectivity_score_matrix(model, universe, sel_std)
            rows.extend(
                _metric_rows(
                    split_type, model.name, fold_id, "full_universe", sm,
                    pos_index, neg_index, seen_index, config.ranking_k,
                )
            )
            for slice_name, mask_fn in (
                ("non_pan_essential", _non_pan_essential_mask),
                ("covered_pairs", _selectivity_covered_mask),
            ):
                pos_s = pos_index[mask_fn(pos_index, universe)]
                neg_s = neg_index[mask_fn(neg_index, universe)]
                if len(pos_s) > 0 and len(neg_s) > 0:
                    rows.extend(
                        _metric_rows(
                            split_type, model.name, fold_id, slice_name, sm,
                            pos_s, neg_s, seen_index, config.ranking_k,
                        )
                    )
                else:
                    logger.warning(
                        "split %s fold %s: %s slice skipped for %s "
                        "(pos=%d, neg=%d)",
                        split_type, fold_id, slice_name, model.name,
                        len(pos_s), len(neg_s),
                    )
        else:
            model.fit(train_base)
            sm = _build_score_matrix(model, universe, base_std)
            rows.extend(
                _metric_rows(
                    split_type, model.name, fold_id, "full_universe", sm,
                    pos_index, neg_index, seen_index, config.ranking_k,
                )
            )
    return rows
```

- [ ] **Step 4: Branch `run_fold` and `run_cv`**

In `run_fold`, replace the final `return _run_fold_augmented(...)` block so the
dispatch covers all three modes:

```python
    if config.selectivity:
        return _run_fold_selectivity(
            train_df, train_base, base_std, pos_index, neg_index, seen_index,
            split_type, fold_id, config, universe,
        )
    if not config.augmented:
        rows = []
        for model in build_models(config):
            model.fit(train_base)
            score_matrix = _build_score_matrix(model, universe, base_std)
            rows.extend(
                _metric_rows(
                    split_type, model.name, fold_id, "full_universe",
                    score_matrix, pos_index, neg_index, seen_index,
                    config.ranking_k,
                )
            )
        return rows
    return _run_fold_augmented(
        train_df, train_base, base_std, pos_index, neg_index, seen_index,
        split_type, fold_id, config, universe,
    )
```

(Remove the original `if not config.augmented:` block that previously preceded
the augmented return, since this replaces it.)

In `_build_gene_universe`, add an `entrez` column capture so the universe can
carry Entrez ids. After building `genes`, add `gene_a_entrez`/`gene_b_entrez`
to the two source frames (guarded by column presence) and set
`entrez = genes["entrez"].to_numpy()` when available, else `None`. Concretely,
change the `gene_a`/`gene_b` DataFrame construction to also include
`"entrez": frame["gene_a_entrez_id"]` / `frame["gene_b_entrez_id"]` when those
columns exist, carry `entrez` through the `drop_duplicates`, and pass
`entrez=...` to the `GeneUniverse(...)` constructor.

In `run_cv`, before `universe = _build_gene_universe(...)`, add the selectivity
branch:

```python
    selectivity = None
    if config.selectivity:
        from sl_benchmark_baseline.data import assert_rand_only

        assert_rand_only(frame)
```

After the universe is built (selectivity needs `universe.entrez`), build and
attach the table:

```python
    if config.selectivity:
        if universe.entrez is None:
            raise ValueError(
                "selectivity mode requires gene_a_entrez_id/gene_b_entrez_id "
                "columns in the benchmark CSV"
            )
        table = build_selectivity_table_from_depmap(
            config.depmap_dir, tuple(int(e) for e in universe.entrez), config
        )
        selectivity = align_selectivity_to_universe(
            table, [int(e) for e in universe.entrez], config.sel_lambda
        )
        universe = replace(universe, selectivity=selectivity)
```

Add `from dataclasses import replace` to the imports. Also extend the
`model_names` manifest logic to include the selectivity case:

```python
    if config.selectivity:
        model_names = ["A", "B", "C", "A_xcl", "B_xcl"]
    elif config.augmented:
        model_names = ["A", "B", "A_transcript", "B_transcript"]
    else:
        model_names = ["A", "B", "C"]
```

And record selectivity settings in the manifest:

```python
    manifest["selectivity"] = config.selectivity
    if config.selectivity:
        manifest["depmap_dir"] = str(config.depmap_dir)
        manifest["cn_loss_thr"] = config.cn_loss_thr
        manifest["expr_low_quantile"] = config.expr_low_quantile
        manifest["sel_n_min"] = config.sel_n_min
        manifest["sel_lambda"] = config.sel_lambda
        manifest["selectivity_coverage_gene_count"] = (
            int(selectivity.coverage_flag.sum()) if selectivity is not None else 0
        )
```

- [ ] **Step 5: Run lint + existing exp06/07 tests to confirm no regression**

Run:
```bash
uv run ruff check src/sl_benchmark_baseline/evaluate.py
uv run python -m pytest tests/ -k "sl_" -v
```
Expected: PASS (existing exp06/07 SL tests still green; selectivity unit tests
green).

- [ ] **Step 6: Commit**

```bash
git add src/sl_benchmark_baseline/evaluate.py
git commit -m "feat: add selectivity fold path, score matrix, and run_cv branch"
```

### Task 9: CLI flags for selectivity mode

**Files:**
- Modify: `src/sl_benchmark_baseline/__main__.py`
- Test: `tests/test_sl_selectivity.py` (append)

**Interfaces:**
- Consumes: `_parse_args`, `SLBaselineConfig`.
- Produces: new flags `--depmap-dir`, `--cn-loss-thr`, `--expr-low-quantile`,
  `--sel-n-min`, `--sel-lambda`, threaded into the `SLBaselineConfig(...)` built
  in `main`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_sl_selectivity.py
from sl_benchmark_baseline.__main__ import _parse_args


def test_cli_parses_selectivity_flags():
    args = _parse_args(
        [
            "--depmap-dir", "data/sl_dependency_v0/raw/depmap",
            "--cn-loss-thr", "0.7",
            "--expr-low-quantile", "0.15",
            "--sel-n-min", "25",
            "--sel-lambda", "0.5",
        ]
    )
    assert str(args.depmap_dir) == "data/sl_dependency_v0/raw/depmap"
    assert args.cn_loss_thr == 0.7
    assert args.expr_low_quantile == 0.15
    assert args.sel_n_min == 25
    assert args.sel_lambda == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k cli_parses_selectivity -v`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'depmap_dir'`.

- [ ] **Step 3: Write minimal implementation**

In `_parse_args`, after the `--no-coverage-flag` argument:

```python
    parser.add_argument("--depmap-dir", type=Path, default=defaults.depmap_dir)
    parser.add_argument("--cn-loss-thr", type=float, default=defaults.cn_loss_thr)
    parser.add_argument(
        "--expr-low-quantile", type=float, default=defaults.expr_low_quantile
    )
    parser.add_argument("--sel-n-min", type=int, default=defaults.sel_n_min)
    parser.add_argument("--sel-lambda", type=float, default=defaults.sel_lambda)
```

In `main`, extend the `SLBaselineConfig(...)` constructor call:

```python
        depmap_dir=args.depmap_dir,
        cn_loss_thr=args.cn_loss_thr,
        expr_low_quantile=args.expr_low_quantile,
        sel_n_min=args.sel_n_min,
        sel_lambda=args.sel_lambda,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_sl_selectivity.py -k cli_parses_selectivity -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_benchmark_baseline/__main__.py tests/test_sl_selectivity.py
git commit -m "feat: add selectivity CLI flags"
```

### Task 10: Parity gate + end-to-end smoke test

Two guarantees: (a) the shared harness still reproduces exp06's A/B/C numbers in
selectivity mode (parity), and (b) selectivity mode runs end-to-end on a tiny
synthetic DepMap fixture, producing `A_xcl`/`B_xcl` rows with the diagnostic
slices. The smoke test uses synthetic CSVs (the real 420MB DepMap files are
gitignored and absent in CI).

**Files:**
- Create: `tests/test_sl_parity_gate.py`
- Modify: `tests/conftest.py` (add `synthetic_selectivity_fixture`)
- Test: both of the above.

**Interfaces:**
- Consumes: `run_cv`, `SLBaselineConfig`, fixtures.
- Produces: `synthetic_selectivity_fixture(tmp_path) -> dict` with keys
  `benchmark_csv: Path` and `depmap_dir: Path`.

- [ ] **Step 1: Add the synthetic selectivity fixture to `tests/conftest.py`**

```python
@pytest.fixture
def synthetic_selectivity_fixture(tmp_path: Path) -> dict:
    """Tiny benchmark CSV with entrez ids + matching tiny DepMap CSVs.

    6 genes (entrez 10..60), 30 cell lines, CV1 with 2 folds. Returns a dict
    with ``benchmark_csv`` and ``depmap_dir`` paths.
    """
    rng = np.random.default_rng(3)
    entrez = [10, 20, 30, 40, 50, 60]
    symbols = [f"G{e}" for e in entrez]
    lines = [f"ACH-{i:04d}" for i in range(30)]

    # DepMap dir with the 5 standard files
    depmap_dir = tmp_path / "depmap"
    depmap_dir.mkdir()
    ge = pd.DataFrame(
        rng.normal(-0.3, 0.6, size=(len(lines), len(entrez))),
        index=lines,
        columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)],
    )
    ge.to_csv(depmap_dir / "CRISPRGeneEffect.csv")
    cn = pd.DataFrame(
        rng.uniform(0.6, 1.4, size=(len(lines), len(entrez))),
        index=lines,
        columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)],
    )
    cn.to_csv(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    for fname, density in (
        ("OmicsSomaticMutationsMatrixDamaging.csv", 0.3),
        ("OmicsSomaticMutationsMatrixHotspot.csv", 0.1),
    ):
        mat = (rng.uniform(0, 1, size=(len(lines), len(entrez))) < density).astype(int)
        frame = pd.DataFrame(
            mat, columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)]
        )
        frame.insert(0, "ModelID", lines)
        frame.insert(0, "SequencingID", [f"s{i}" for i in range(len(lines))])
        frame.to_csv(depmap_dir / fname, index=False)
    expr = pd.DataFrame(
        rng.uniform(0, 7, size=(len(lines), len(entrez))),
        columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)],
    )
    expr.insert(0, "ModelID", lines)
    expr.to_csv(depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
                index=False)

    # Benchmark CSV: 2 folds, both classes, entrez + negative_sampling_method
    rows = []
    counter = 0
    for fold_id in (0, 1):
        for role, n_each in (("train", 4), ("test", 3)):
            for label in (1, 0):
                for _ in range(n_each):
                    ia, ib = rng.integers(0, len(entrez), size=2)
                    while ib == ia:
                        ib = rng.integers(0, len(entrez))
                    base = -1.0 if label == 1 else 0.2
                    rows.append(
                        {
                            "pair_id": f"P{counter}",
                            "negative_sampling_method": "Rand",
                            "fold_id": fold_id,
                            "split_role": role,
                            "sl_label": label,
                            "gene_a_symbol": symbols[ia],
                            "gene_b_symbol": symbols[ib],
                            "gene_a_unified_id": int(entrez[ia]),
                            "gene_b_unified_id": int(entrez[ib]),
                            "gene_a_entrez_id": int(entrez[ia]),
                            "gene_b_entrez_id": int(entrez[ib]),
                            "gene_a_k562_gene_effect": base + rng.normal(0, 0.1),
                            "gene_b_k562_gene_effect": base + rng.normal(0, 0.1),
                        }
                    )
                    counter += 1
    csv_path = tmp_path / "bench_sel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"benchmark_csv": csv_path, "depmap_dir": depmap_dir}
```

- [ ] **Step 2: Write the parity + smoke tests**

```python
# tests/test_sl_parity_gate.py
"""Parity gate (A/B/C unchanged) and end-to-end smoke for selectivity mode."""

from __future__ import annotations

import pandas as pd

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.evaluate import run_cv


def test_selectivity_mode_preserves_abc_parity(synthetic_selectivity_fixture, tmp_path):
    fx = synthetic_selectivity_fixture
    common = dict(
        input_csv=fx["benchmark_csv"], folds=(0, 1), ranking_k=(2, 5), seed=17
    )
    base = run_cv(SLBaselineConfig(output_dir=tmp_path / "base", **common))
    sel = run_cv(
        SLBaselineConfig(
            output_dir=tmp_path / "sel", depmap_dir=fx["depmap_dir"], **common
        )
    )
    # A/B/C rows on the full_universe slice must match bit-for-bit.
    keys = ["split_type", "model", "metric"]
    base_abc = (
        base[base["model"].isin(["A", "B", "C"])]
        .set_index(keys)["mean"]
        .sort_index()
    )
    sel_full = sel[sel["slice"] == "full_universe"] if "slice" in sel.columns else sel
    sel_abc = (
        sel_full[sel_full["model"].isin(["A", "B", "C"])]
        .set_index(keys)["mean"]
        .sort_index()
    )
    pd.testing.assert_series_equal(base_abc, sel_abc, check_exact=False, atol=1e-9)


def test_selectivity_mode_emits_xcl_and_slices(synthetic_selectivity_fixture, tmp_path):
    fx = synthetic_selectivity_fixture
    out = tmp_path / "run"
    run_cv(
        SLBaselineConfig(
            input_csv=fx["benchmark_csv"],
            output_dir=out,
            depmap_dir=fx["depmap_dir"],
            folds=(0, 1),
            ranking_k=(2, 5),
        )
    )
    fold_metrics = pd.read_csv(out / "fold_metrics.csv")
    assert {"A", "B", "C", "A_xcl", "B_xcl"}.issubset(set(fold_metrics["model"]))
    xcl = fold_metrics[fold_metrics["model"] == "A_xcl"]
    assert "full_universe" in set(xcl["slice"])
    # at least one diagnostic slice present (data permitting)
    assert {"non_pan_essential", "covered_pairs"} & set(xcl["slice"])


def test_selectivity_rejects_non_rand_negatives(synthetic_selectivity_fixture, tmp_path):
    fx = synthetic_selectivity_fixture
    frame = pd.read_csv(fx["benchmark_csv"])
    frame.loc[0, "negative_sampling_method"] = "Dep"
    bad = tmp_path / "bad.csv"
    frame.to_csv(bad, index=False)
    import pytest

    with pytest.raises(ValueError, match="Rand"):
        run_cv(
            SLBaselineConfig(
                input_csv=bad, output_dir=tmp_path / "x", depmap_dir=fx["depmap_dir"],
                folds=(0,), ranking_k=(2,),
            )
        )
```

- [ ] **Step 3: Run the tests**

Run: `uv run python -m pytest tests/test_sl_parity_gate.py -v`
Expected: PASS (3 passed). If the parity test reports a mismatch, the
selectivity branch is mutating the A/B/C path — fix before continuing.

Note: the `assert_series_equal` parity check is the *structural* gate (A/B/C
identical with vs without selectivity mode on the same data). The *numeric*
exp06 floor (F1 CV1=0.730 etc.) is reproduced only on the real benchmark; add a
manual verification note in Task 11 rather than asserting it in CI.

- [ ] **Step 4: Run full suite + lint**

```bash
uv run ruff check . && uv run ruff format --check .
uv run python -m pytest tests/ -q
```
Expected: PASS (no regressions in exp06/07/08 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_sl_parity_gate.py tests/conftest.py
git commit -m "test: add selectivity parity gate and e2e smoke tests"
```

### Task 11: Real-data run + experiment write-up

**Files:**
- Create: `docs/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md`
- Output: `results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/`

This task produces the actual experiment results and documents them. It requires
the gitignored DepMap files and the benchmark CSVs on disk.

- [ ] **Step 1: Run exp09 on the real benchmark**

```bash
uv run python -m sl_benchmark_baseline \
  --input-csv data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
  --output-dir results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run \
  --depmap-dir data/sl_dependency_v0/raw/depmap \
  --split-types CV1 CV2 CV3 \
  --folds 0 1 2 3 4 \
  --ranking-k 10 20 50
```
Expected: writes `fold_metrics.csv`, `summary.csv`, `manifest.json`. Runtime is
dominated by building 5 score matrices per fold over the 9,471-gene universe.

- [ ] **Step 2: Verify the exp06 parity numerically (manual gate)**

```bash
uv run python - <<'PY'
import pandas as pd
sel = pd.read_csv("results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/summary.csv")
exp06 = pd.read_csv("results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv")
# Spot-check B (XGBoost) F1 on CV1 reproduces the locked 0.730 floor.
b_cv1_f1 = sel[(sel.model=="B") & (sel.split_type=="CV1") & (sel.metric=="f1")
               & (sel.slice=="full_universe")]["mean"]
print("exp09 B CV1 f1:", float(b_cv1_f1.iloc[0]))
print("expected ~0.730 (exp06 locked floor)")
PY
```
Expected: B CV1 F1 ≈ 0.730 (matches exp06). If it diverges, STOP — the harness
changed the baseline; do not proceed to the write-up.

- [ ] **Step 3: Write the experiment doc**

Create `docs/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md` with
these sections (fill the metric tables from `summary.csv`):

1. **Header & framing** — parent exp06; cross-cell-line Selectivity route;
   benchmark-adapter feature, NOT a validated K562 SL assay; `Rand` negatives
   only; 9,471-gene universe; seed 17.
2. **Inputs** — the 5 DepMap files (cite the §3 table from the design spec) and
   the composite-OR defective definition.
3. **Method** — Selectivity contrast (larger = more SL-like), `n_min`=20
   fallback, `sel_mean`/`sel_absdiff`/`pan_essential_min` features, `A_xcl`/
   `B_xcl` columns.
4. **Results** — `A_xcl` vs `A`, `B_xcl` vs `B` on CV1/CV2/CV3 official metrics
   (F1, AUROC, AUPR, NDCG@10), full_universe slice. Highlight CV3.
5. **Diagnostics** — non_pan_essential and covered_pairs slices; whether lift
   survives. State the verdict: real lift vs honest null.
6. **Guardrails & caveats** — leakage profile (sel computed from external matrix,
   split-independent; `Rand`-only); no SL biological claim.

- [ ] **Step 4: Commit the write-up**

```bash
git add docs/experiment/09_k562_sl_pair_cross_cell_line_selectivity.md
git commit -m "docs: add exp09 cross-cell-line selectivity results write-up"
```

(The `results/` dir is gitignored; do not commit it.)

## Self-Review Notes

- Spec coverage: §4 defective call → Task 3; §5 Selectivity + fallback +
  symmetrization → Tasks 1,4,7; §6 model columns → Tasks 6,8; §7 diagnostics →
  Task 8b; §8 guardrails (Rand-only, thresholds-in-config) → Task 5; §9 code
  touch points → Tasks 1-9; §10 tests (parity, unit, leakage guard, smoke) →
  Tasks 1-10; §11 results location + §12 success criteria → Task 11.
- Type consistency: `SelectivityTable` (Task 4, with `essential_fraction`) →
  `UniverseSelectivity` (Task 7) → `GeneUniverse.selectivity` (Task 8) →
  `_selectivity_raw`/score matrix/slices (Task 8,8b). Feature width 8 = 5
  GeneEffect + 3 selectivity throughout. Model names `A_xcl`/`B_xcl` consistent
  across Tasks 6, 8b, 10, 11.
- No placeholders: every code step shows complete code; every run step shows the
  exact command and expected output.













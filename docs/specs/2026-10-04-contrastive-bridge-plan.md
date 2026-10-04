# Contrastive-PCA Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the expression directions that pseudo-bulk and bulk do not share before the bridge, choose how many on validation, and rerun the linear context prior with all extra lines and without the haematopoietic ones.

**Architecture:** The design's fallback for a failing bridge (spec §3.2, §8.1), Celligner's contrastive PCA, adapted to our paired data. On the 167 single-cell training lines that have both profiles, the directions with more variance in pseudo-bulk than in bulk (pseudo-bulk-specific: sampling, platform) and those with more variance in bulk than in pseudo-bulk (bulk-specific: culture batch) are the top eigenvectors of the difference of the two covariance matrices. Their union is projected out of every bulk and pseudo-bulk row, each source about its own paired mean; the expression components, the bridge and everything after are then fitted on the aligned data. Because the profiles are paired, Celligner's cluster-mean removal and mutual-nearest-neighbour step are not needed: both covariances are over the same lines. The alignment reads no label and only training lines, so like the expression components it is fitted once, outside the cross-fitting folds. A new first step of the run chooses the counts on validation by the existing bootstrap rule; no alignment stays the default.

**Tech Stack:** numpy (QR, eigh, SVD), pandas, the existing `src/context_prior` package and runner.

**Spec:** `docs/specs/2026-10-04-context-generalization-design.md` (§3.2 bridge, §8.1 decision); results that motivate it: `results/context_prior_seed0/README.md`.

## Global Constraints

- `<scratch>` is the session's scratchpad directory; `<date>` is the run date as `YYYYMMDD`.
- Every Python invocation is `uv run python -m ...` from the repo root; tests run as `uv run python -m pytest <files> -q -p no:cacheprovider > <scratch file> 2>&1; tail -5 <scratch file>`; lint with `.venv/bin/ruff check src tests` and `.venv/bin/ruff format <touched files>` only.
- `src/context_prior/` imports `src.data` only; configs are strict (unknown or missing keys raise).
- The alignment is fitted on training lines only (`split.supervised_train` with bulk RNA, the bridge's paired lines); validation and test pseudo-bulk and test bulk never enter it.
- Validation chooses: pseudo-bulk-specific count first (bulk-specific at 0), then the bulk-specific count given it, each by point estimate; the alignment is kept only if the 95% paired bootstrap interval (27 validation lines, 1,000 resamples, seed 0) of its `val_selective_spearman` gain over no alignment excludes zero.
- One config is one run at seed 0; each run scores its chosen prior once on test.
- Work on branch `feat/contrastive-bridge` from `main` (`2439070`); commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- H20 access: `ssh -J richard@100.91.229.50 -p 30838 root@10.15.171.204`; code reaches the host as a git bundle (git push through the jump host fails); the host has no internet.
- Name things by what they are (the pseudo-bulk-specific directions, the aligned bridge); no bare labels.

## Review Focus

- A requested count larger than the number of directions with positive excess variance: fewer directions are removed, and the log records how many — pinned in the alignment task.
- Pseudo-bulk-specific and bulk-specific directions that overlap: the union basis drops dependent columns instead of removing a direction twice — pinned in the alignment task.
- Bulk rows outside the paired fit (extra lines, the oracle's validation rows): aligned about the paired bulk mean, the same affine map as the paired rows — pinned in the alignment task.
- A grid without 0 in either list: the run refuses it, so "no alignment" is always a candidate — pinned in the runner task.
- `--oracle-only` (no pseudo-bulk): no alignment step, unchanged behaviour — pinned by the existing oracle-mode runner test.

---

### Task 1: Contrastive alignment

**Files:**
- Create: `src/context_prior/alignment.py`
- Test: `tests/test_context_prior_alignment.py`

**Interfaces:**
- Produces: `contrastive_directions(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray` (genes x k, k <= count, orthonormal columns); `Alignment(genes: tuple[str, ...], directions: np.ndarray, pseudo_mean: np.ndarray, bulk_mean: np.ndarray)` with `apply(frame: pd.DataFrame, *, source: str) -> pd.DataFrame` (`source` is `"pseudo"` or `"bulk"`) and property `count -> int`; `fit_alignment(pseudobulk: pd.DataFrame, bulk: pd.DataFrame, *, pseudo_components: int, bulk_components: int) -> Alignment`.

- [ ] **Step 1: Create the branch**

```bash
git switch -c feat/contrastive-bridge main
```

- [ ] **Step 2: Write the failing tests**

```python
"""Contrastive directions between paired pseudo-bulk and bulk, and their removal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.alignment import (
    contrastive_directions,
    fit_alignment,
)

GENES = [f"G{i}" for i in range(30)]
RNG = np.random.default_rng(7)
#: The pseudo-bulk-only direction: spread over every gene, not one gene's axis.
NOISE_AXIS = RNG.normal(size=30) / np.linalg.norm(RNG.normal(size=30))


def paired(seed=0, lines=60):
    """Bulk = shared signal; pseudo-bulk = the same signal plus a strong
    pseudo-bulk-only direction and noise."""
    rng = np.random.default_rng(seed)
    axis = NOISE_AXIS / np.linalg.norm(NOISE_AXIS)
    shared = rng.normal(size=(lines, 3)) @ rng.normal(size=(3, 30))
    pseudo = shared + 5.0 * rng.normal(size=(lines, 1)) * axis + 0.1 * rng.normal(size=(lines, 30))
    bulk = shared + 0.1 * rng.normal(size=(lines, 30))
    index = [f"L{i}" for i in range(lines)]
    return (
        pd.DataFrame(pseudo, index=index, columns=GENES),
        pd.DataFrame(bulk, index=index, columns=GENES),
    )


def mean_gene_correlation(left, right):
    return np.mean([np.corrcoef(left[g], right[g])[0, 1] for g in GENES])


def test_directions_find_the_pseudo_bulk_only_axis():
    pseudo, bulk = paired()
    directions = contrastive_directions(pseudo.to_numpy(), bulk.to_numpy(), 1)
    assert directions.shape == (30, 1)
    axis = NOISE_AXIS / np.linalg.norm(NOISE_AXIS)
    assert abs(directions[:, 0] @ axis) > 0.95
    assert np.allclose(directions.T @ directions, np.eye(1))


def test_only_positive_excess_variance_counts():
    x = np.random.default_rng(1).normal(size=(40, 10))
    # Cov(x) - Cov(2x) = -3 Cov(x) has no positive eigenvalue: nothing to remove.
    assert contrastive_directions(x, 2 * x, 3).shape == (10, 0)


def test_zero_counts_are_the_identity():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=0, bulk_components=0)
    assert alignment.count == 0
    assert alignment.apply(pseudo, source="pseudo").equals(pseudo)


def test_removal_makes_the_sources_agree_and_is_affine_for_other_rows():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=1, bulk_components=0)
    before = mean_gene_correlation(pseudo, bulk)
    after = mean_gene_correlation(
        alignment.apply(pseudo, source="pseudo"), alignment.apply(bulk, source="bulk")
    )
    assert after > before + 0.05 and after > 0.95
    other = bulk.iloc[:5] + 1.0  # rows outside the fit take the same affine map
    shifted = alignment.apply(other, source="bulk") - alignment.apply(
        bulk.iloc[:5], source="bulk"
    )
    keep = np.eye(30) - alignment.directions @ alignment.directions.T
    assert np.allclose(shifted.to_numpy(), (other - bulk.iloc[:5]).to_numpy() @ keep)


def test_overlapping_direction_sets_are_removed_once():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=3, bulk_components=3)
    gram = alignment.directions.T @ alignment.directions
    assert np.allclose(gram, np.eye(alignment.count))
    assert alignment.count <= 6


def test_misaligned_frames_and_unknown_source_raise():
    pseudo, bulk = paired()
    with pytest.raises(ValueError, match="paired"):
        fit_alignment(pseudo, bulk.iloc[::-1], pseudo_components=1, bulk_components=0)
    alignment = fit_alignment(pseudo, bulk, pseudo_components=1, bulk_components=0)
    with pytest.raises(ValueError, match="source"):
        alignment.apply(pseudo, source="tumour")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_context_prior_alignment.py -q -p no:cacheprovider > <scratch>/alignment.txt 2>&1; tail -5 <scratch>/alignment.txt`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement** `src/context_prior/alignment.py`

```python
"""Contrastive alignment of pseudo-bulk and bulk before the bridge.

Celligner's contrastive PCA adapted to paired profiles: on lines measured both
ways, the directions with more variance in one source than in the other are the
top eigenvectors of the difference of their covariance matrices. Their union is
projected out of every row of both sources, each about its own paired mean, so
what the bridge and the prior see is the variation the two sources share. Both
covariances are over the same lines, so no cluster-mean removal or nearest-
neighbour matching is needed. Reads no label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Singular values below this fraction of the largest mark a dependent column.
_RANK_TOLERANCE = 1e-8


def contrastive_directions(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray:
    """Genes x k orthonormal directions (k <= ``count``) with the largest positive
    excess variance of ``first`` over ``second`` (rows are lines, both centred
    here). Computed in the span of the data, so the cost is in lines, not genes."""
    if count < 0:
        raise ValueError("count must be non-negative")
    genes = first.shape[1]
    if count == 0:
        return np.zeros((genes, 0))
    a = first - first.mean(axis=0)
    b = second - second.mean(axis=0)
    basis, triangle = np.linalg.qr(np.vstack([a, b]).T)  # genes x m, m x m
    left, right = triangle[:, : len(a)], triangle[:, len(a) :]
    excess = left @ left.T / len(a) - right @ right.T / len(b)
    values, vectors = np.linalg.eigh(excess)
    order = np.argsort(values)[::-1][:count]
    order = order[values[order] > 0]
    return basis @ vectors[:, order]


@dataclass(frozen=True)
class Alignment:
    """Directions removed from both sources, and each source's paired mean."""

    genes: tuple[str, ...]
    directions: np.ndarray  # genes x count, orthonormal
    pseudo_mean: np.ndarray
    bulk_mean: np.ndarray

    @property
    def count(self) -> int:
        return int(self.directions.shape[1])

    def apply(self, frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
        """Project the directions out of ``frame`` rows about the source's mean."""
        if source == "pseudo":
            mean = self.pseudo_mean
        elif source == "bulk":
            mean = self.bulk_mean
        else:
            raise ValueError(f"source must be 'pseudo' or 'bulk', not {source!r}")
        if self.count == 0:
            return frame
        values = frame.loc[:, list(self.genes)].to_numpy(dtype=np.float64)
        projected = ((values - mean) @ self.directions) @ self.directions.T
        return pd.DataFrame(values - projected, index=frame.index, columns=list(self.genes))


def fit_alignment(
    pseudobulk: pd.DataFrame,
    bulk: pd.DataFrame,
    *,
    pseudo_components: int,
    bulk_components: int,
) -> Alignment:
    """Fit on paired rows (same lines, same genes, same order) of the two sources."""
    if list(pseudobulk.index) != list(bulk.index) or list(pseudobulk.columns) != list(bulk.columns):
        raise ValueError("pseudo-bulk and bulk must be paired on lines and genes")
    p = pseudobulk.to_numpy(dtype=np.float64)
    b = bulk.to_numpy(dtype=np.float64)
    union = np.hstack(
        [
            contrastive_directions(p, b, pseudo_components),
            contrastive_directions(b, p, bulk_components),
        ]
    )
    if union.shape[1]:
        left, singular, _ = np.linalg.svd(union, full_matrices=False)
        union = left[:, singular > _RANK_TOLERANCE * singular[0]]
    return Alignment(tuple(pseudobulk.columns), union, p.mean(axis=0), b.mean(axis=0))
```

- [ ] **Step 5: Run tests to verify they pass**

Run the Step 3 command. Expected: `6 passed`. If `test_removal_makes_the_sources_agree...` fails on the threshold, check that the removed direction is the planted axis before changing anything; do not lower the threshold.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff format src/context_prior/alignment.py tests/test_context_prior_alignment.py
git add src/context_prior/alignment.py tests/test_context_prior_alignment.py
git commit -m "feat(context-prior): contrastive alignment of paired pseudo-bulk and bulk" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The alignment step in the run

**Files:**
- Modify: `src/experiments/context_prior.py` (split `load_run_data`; add `NormalizedData`, `load_normalized`, `assemble`, `fit_bridge_alignment`, `select_alignment`, `run_alignment_choice`; wire `main`; a summary section)
- Modify: `src/experiments/config.py` (`_PRIOR_GROUPS["bridge"]`)
- Modify: `configs/context_prior/prior.yaml`, `configs/context_prior/prior_no_haematopoietic.yaml`
- Test: `tests/test_context_prior_run.py` (append)

**Interfaces:**
- Consumes: `fit_alignment`, `Alignment` (Task 1); the runner's existing `RunData`, `scaled_residual`, `selective_score`, `paired_gain`, `_best`, `_score_key`, `_fit_all`, `_write_json`, `_read_json`, `_number`, `_interval`.
- Produces: `NormalizedData` (fields below); `load_normalized(config, *, oracle_only) -> NormalizedData`; `assemble(base: NormalizedData, alignment: Alignment | None) -> RunData`; `load_run_data(config, *, oracle_only) -> RunData` (unchanged signature: `assemble(load_normalized(...), None)`); `fit_bridge_alignment(base, choice: Mapping[str, int]) -> Alignment | None`; `select_alignment(evaluate, interval, grid) -> tuple[dict, list[dict]]`; `run_alignment_choice(base) -> tuple[dict, list[dict]]`; run output `alignment.json` = `{"choice": {...}, "log": [...]}`.

- [ ] **Step 1: Config.** In `src/experiments/config.py` add `"bridge": "pseudo_components bulk_components",` to `_PRIOR_GROUPS`. In both prior YAMLs add, after the `selection` group:

```yaml
bridge:
  pseudo_components: [0, 2, 4, 8, 16]
  bulk_components: [0, 2, 4, 8]
```

and widen the penalty grid, whose edges both earlier runs hit, to `penalties: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`.

- [ ] **Step 2: Write the failing test** (append to `tests/test_context_prior_run.py`)

```python
def test_select_alignment_order_and_keep_rule():
    scores = {(0, 0): 0.20, (2, 0): 0.22, (4, 0): 0.23, (4, 2): 0.235, (4, 4): 0.21}
    calls = []

    def evaluate(setting):
        calls.append(setting)
        return scores.get(setting, 0.0), pd.DataFrame({"A": [scores.get(setting, 0.0)]})

    grid = {"pseudo_components": [0, 2, 4], "bulk_components": [0, 2, 4]}
    choice, log = run.select_alignment(evaluate, lambda better, simpler: [0.01, 0.05], grid)
    assert choice == {"pseudo_components": 4, "bulk_components": 2,
                      "chosen": [4, 2], "interval": [0.01, 0.05], "kept": True}
    assert len(calls) == len(set(calls))  # every setting evaluated once
    assert {(e["pseudo_components"], e["bulk_components"]) for e in log} == set(calls)
    choice, _ = run.select_alignment(evaluate, lambda better, simpler: [-0.01, 0.05], grid)
    assert (choice["pseudo_components"], choice["bulk_components"], choice["kept"]) == (0, 0, False)
    with pytest.raises(ValueError, match="0"):
        run.select_alignment(evaluate, lambda b, s: [0.0, 0.0],
                             {"pseudo_components": [2, 4], "bulk_components": [0]})
```

Run: `uv run python -m pytest tests/test_context_prior_run.py -q -p no:cacheprovider > <scratch>/runner.txt 2>&1; tail -5 <scratch>/runner.txt`
Expected: FAIL with `AttributeError: ... 'select_alignment'` (the config test passes once Step 1 is in).

- [ ] **Step 3: Split the data loading.** Replace `load_run_data` with the three functions below. `load_normalized` is the current body of `load_run_data` up to and including the two `quantile_normalize` calls, stopping before the bridge, the components and `PriorInputs`; it keeps every guard and drop it has now (test bulk and excluded lines dropped at read, validation bulk only as oracle rows, the measured-gene space and `fill_unmeasured`).

```python
@dataclass(frozen=True)
class NormalizedData:
    """What every bridge variant shares: labels, definitions, and quantile-
    normalised bulk (training side, then the oracle's validation rows) and
    pseudo-bulk (the 226 lines; None with ``oracle_only``)."""

    config: Mapping[str, Any]
    joint: Mapping[str, Any]
    split: FixedSplit
    gene_effect: pd.DataFrame
    definitions: Definitions
    labelled: tuple[str, ...]
    single_cell_train: tuple[str, ...]
    models: pd.DataFrame
    side: tuple[str, ...]
    oracle_lines: tuple[str, ...]
    bulk: pd.DataFrame
    pseudobulk: pd.DataFrame | None
    filled: pd.Series | None
    reference: Reference

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        return scaled_residual(self.gene_effect, lines, self.definitions)


def assemble(base: NormalizedData, alignment: Alignment | None) -> RunData:
    """Align (when given), then fit the expression components and the bridge."""
    bulk, pseudobulk = base.bulk, base.pseudobulk
    if alignment is not None:
        bulk = alignment.apply(bulk, source="bulk")
        pseudobulk = alignment.apply(pseudobulk, source="pseudo")
    expression = bulk.loc[list(base.side)]
    queries = {"oracle": bulk.loc[list(base.oracle_lines)]}
    bridge = None
    if pseudobulk is not None:
        paired = list(base.single_cell_train)
        bridge = fit_bridge(pseudobulk.loc[paired], expression.loc[paired])
        queries["val"] = bridge.apply(pseudobulk.loc[list(base.split.val)])
    inputs = PriorInputs(
        expression=expression,
        residual=scaled_residual(base.gene_effect, base.labelled, base.definitions),
        components=fit_expression_components(
            expression, int(base.config["prior"]["components"])
        ),
        reference=base.reference,
        lineage=base.models.loc[list(base.side), "lineage"],
        patients=base.models["patient_id"].to_dict(),
    )
    return RunData(
        config=base.config,
        joint=base.joint,
        split=base.split,
        gene_effect=base.gene_effect,
        definitions=base.definitions,
        inputs=inputs,
        labelled=base.labelled,
        single_cell_train=base.single_cell_train,
        lineage=base.models["lineage"],
        pseudobulk=pseudobulk,
        bridge=bridge,
        queries=queries,
        filled=base.filled,
    )


def load_run_data(config: Mapping[str, Any], *, oracle_only: bool) -> RunData:
    """Everything the steps share, without alignment."""
    return assemble(load_normalized(config, oracle_only=oracle_only), None)
```

`load_normalized` returns `NormalizedData(config=config, joint=joint, split=split, gene_effect=gene_effect, definitions=definitions, labelled=labelled, single_cell_train=single_cell_train, models=models, side=tuple(side), oracle_lines=tuple(oracle_lines), bulk=normalized, pseudobulk=<normalised pseudo-bulk or None>, filled=filled, reference=load_reference(Path(paths["reference"]), blocked={*split.val, *split.test, *extra.excluded}))`, where `normalized` is `quantile_normalize(bulk.loc[[*side, *oracle_lines], space], reference_profile)` as now. Add the imports `from src.context_prior.alignment import Alignment, fit_alignment` and `from src.context_prior.reference import Reference` (keep `load_reference`).

- [ ] **Step 4: The choice.** Add after `assemble`:

```python
def fit_bridge_alignment(
    base: NormalizedData, choice: Mapping[str, int]
) -> Alignment | None:
    """The alignment for ``choice``, fitted on the bridge's paired training lines;
    None for no alignment."""
    pseudo, bulk = int(choice["pseudo_components"]), int(choice["bulk_components"])
    if pseudo == bulk == 0:
        return None
    paired = list(base.single_cell_train)
    return fit_alignment(
        base.pseudobulk.loc[paired],
        base.bulk.loc[paired],
        pseudo_components=pseudo,
        bulk_components=bulk,
    )


def select_alignment(
    evaluate: Callable[[tuple[int, int]], tuple[float, pd.DataFrame]],
    interval: Callable[[pd.DataFrame, pd.DataFrame], list[float]],
    grid: Mapping[str, Sequence[int]],
) -> tuple[dict, list[dict]]:
    """Pseudo-bulk-specific count first (bulk-specific 0), then the bulk-specific
    count given it, each by point estimate (ties keep the earlier, smaller count);
    kept only if the gain interval over no alignment excludes zero."""
    if 0 not in grid["pseudo_components"] or 0 not in grid["bulk_components"]:
        raise ValueError("both alignment grids must contain 0 (no alignment)")
    scored: dict[tuple[int, int], tuple[float, pd.DataFrame]] = {}

    def score(setting: tuple[int, int]) -> tuple[float, pd.DataFrame]:
        if setting not in scored:
            scored[setting] = evaluate(setting)
        return scored[setting]

    pseudo = max(grid["pseudo_components"], key=lambda k: _score_key(score((k, 0))[0]))
    bulk = max(grid["bulk_components"], key=lambda k: _score_key(score((pseudo, k))[0]))
    chosen, none = (pseudo, bulk), (0, 0)
    gain = None if chosen == none else interval(score(chosen)[1], score(none)[1])
    kept = gain is not None and gain[0] > 0
    final = chosen if kept else none
    log = [
        {"pseudo_components": p, "bulk_components": b, "score": float(value)}
        for (p, b), (value, _) in scored.items()
    ]
    return {
        "pseudo_components": final[0],
        "bulk_components": final[1],
        "chosen": list(chosen),
        "interval": gain,
        "kept": kept,
    }, log


def run_alignment_choice(base: NormalizedData) -> tuple[dict, list[dict]]:
    """Score each alignment by the expression-components prior on bridged
    validation input, its penalty chosen per alignment on validation."""
    truth = base.truth(list(base.split.val))
    penalties = base.config["selection"]["penalties"]
    first: dict[str, RunData] = {}

    def evaluate(setting: tuple[int, int]) -> tuple[float, pd.DataFrame]:
        choice = {"pseudo_components": setting[0], "bulk_components": setting[1]}
        data = assemble(base, fit_bridge_alignment(base, choice))
        first.setdefault("data", data)
        fits = []
        for penalty in penalties:
            spec = PriorSpec((Stage("expression_components", penalty),))
            prediction = total(_fit_all(data, spec).predict(data.queries["val"]))
            fits.append(
                (selective_score(prediction, truth, base.definitions.selective), penalty, prediction)
            )
        score, _, prediction = _best(fits)
        print(f"alignment {setting}: {score:.4f}", flush=True)
        return score, prediction

    def interval(better: pd.DataFrame, simpler: pd.DataFrame) -> list[float]:
        return paired_gain(first["data"], better, simpler, truth)["interval"]

    return select_alignment(evaluate, interval, base.config["bridge"])
```

- [ ] **Step 5: Wire `main`.** Replace `data = load_run_data(config, oracle_only=args.oracle_only)` with:

```python
    base = load_normalized(config, oracle_only=args.oracle_only)
    alignment = None
    if not args.oracle_only:
        if not (run_dir / "alignment.json").is_file():
            choice, log = run_alignment_choice(base)
            _write_json(run_dir / "alignment.json", {"choice": choice, "log": log})
        choice = _read_json(run_dir / "alignment.json")["choice"]
        alignment = fit_bridge_alignment(base, choice)
    data = assemble(base, alignment)
    del base
```

- [ ] **Step 6: Summary.** In `write_summary`, immediately before the learning-curve section, add:

```python
    if (run_dir / "alignment.json").is_file():
        record = _read_json(run_dir / "alignment.json")
        choice = record["choice"]
        out += [
            "",
            "## Bridge alignment",
            "",
            "Contrastive directions projected out of bulk and pseudo-bulk before the "
            "bridge; validation selective Spearman of the expression-components prior "
            "on bridged input.",
            "",
            "| Pseudo-bulk-specific | Bulk-specific | Score |",
            "| --- | --- | --- |",
        ]
        out += [
            f"| {e['pseudo_components']} | {e['bulk_components']} | {_number(e['score'])} |"
            for e in record["log"]
        ]
        verdict = "kept" if choice["kept"] else "not kept: no alignment"
        out += [
            "",
            f"Best {choice['chosen']}, gain over none {_interval(choice['interval'])}: "
            f"{verdict}.",
        ]
```

- [ ] **Step 7: Run the runner and prior tests, then the suite**

Run: `uv run python -m pytest tests/test_context_prior_run.py tests/test_context_prior_alignment.py -q -p no:cacheprovider > <scratch>/runner.txt 2>&1; tail -5 <scratch>/runner.txt`, then the full suite to a file and `.venv/bin/ruff check src tests`.
Expected: all pass, including the existing oracle-mode `load_run_data` test (it now runs through `load_normalized` and `assemble`).

- [ ] **Step 8: Commit**

```bash
.venv/bin/ruff format src/experiments/context_prior.py tests/test_context_prior_run.py
git add src/experiments/context_prior.py src/experiments/config.py configs/context_prior/prior.yaml configs/context_prior/prior_no_haematopoietic.yaml tests/test_context_prior_run.py
git commit -m "feat(context-prior): choose a contrastive bridge alignment on validation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Documents

**Files:**
- Modify: `docs/specs/2026-10-04-context-generalization-design.md` (§3.2 bridge bullet), `docs/03-geneeffect-protocol.md` (§10 steps and outputs), `hpc/README.md` (`prior` outputs)

- [ ] **Step 1:** Spec §3.2: replace "Celligner-style cPCA alignment enters only if the measured bridge loss is large" with the implemented rule: contrastive directions from the 167 paired training lines (pseudo-bulk-specific and bulk-specific, union projected out of both sources about each source's paired mean), counts chosen on validation by the bootstrap rule with no alignment as the default, fitted once outside the folds because it reads no label.
- [ ] **Step 2:** Protocol §10: add the alignment as the first step after pseudo-bulk preparation and `alignment.json` to the outputs; note the widened penalty grid.
- [ ] **Step 3:** `hpc/README.md`: add `alignment.json` to the `prior` output list.
- [ ] **Step 4: Commit** with `docs(context-prior): the contrastive bridge alignment`.

---

### Task 4: Runs on the H20 host and the record

- [ ] **Step 1: Ship the branch.** On the Mac: `git bundle create <scratch>/cb.bundle 4334eae..feat/contrastive-bridge`; stream it over SSH (`ssh ... 'cat > /tmp/cb.bundle && cd /2023533015/VCC_Project && git fetch /tmp/cb.bundle feat/contrastive-bridge:feat/contrastive-bridge'` with the bundle on stdin), then `git worktree add /2023533015/VCC_Project_contrastive_bridge feat/contrastive-bridge` and, inside it, symlink `data`, `model`, `.venv-tx1` and `configs/experiments/13_geneeffect_226/basal_source_registry.csv` from `/2023533015/VCC_Project` (the registry is git-ignored and pseudo-bulk preparation's guard reads it; the pseudo-bulk artifact already exists and is reused).
- [ ] **Step 2: Launch both runs, detached** (each 32 threads; the host has 224 cores):

```bash
cd /2023533015/VCC_Project_contrastive_bridge && mkdir -p outputs/launches
(OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 setsid nohup hpc/run.sh prior configs/context_prior/prior.yaml --run-id prior_cpca_<date> > outputs/launches/prior_cpca_<date>.log 2>&1 < /dev/null &)
(OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 setsid nohup hpc/run.sh prior configs/context_prior/prior_no_haematopoietic.yaml --run-id prior_cpca_no_haematopoietic_<date> > outputs/launches/prior_cpca_no_haematopoietic_<date>.log 2>&1 < /dev/null &)
```

- [ ] **Step 3: Confirm** `alignment.json`, `decision.json`, `selection.json`, `metrics.json` and `summary.md` in each run directory; read results in small slices (the jump host drops sessions whose output exceeds a few kilobytes).
- [ ] **Step 4: Record** a "Contrastive bridge" section in `results/context_prior_seed0/README.md` with: the alignment table and choice for each config; bridge quality before and after; the extra-lines decision with all extras (does it now pass?); which blocks survive selection (do own expression and partners now help?); validation and test tables against the Tx1 ridge and against the earlier chosen prior (0.223 / 0.223); copy each `summary.md` next to the README. Commit, then merge `feat/contrastive-bridge` into `main` once the user has seen the results.

# Exp08 GWPS Bag NaN Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop Phase 3 NaN crashes by cleaning non-finite GWPS bag entries at build time, verifying finiteness at cache load, and asserting at the training boundary.

**Architecture:** Strategy C — three guards along the bag lifecycle. `build_gwps_bags()` zero-fills non-finite entries (the single cleaning site) and logs what it touched; `load_bags_npz()` fail-fasts on any non-finite cache (verifies the invariant, surfaces stale pre-fix caches); `_bag_part()` carries a cheap finite assert (localizes any future bypass to a gene). A `combine()` zero-weight guard fixes the warmup `0.0 * NaN` masking trap.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, pytest, `uv`, ruff.

## Global Constraints

- Branch: `fix/exp08-phase3-nan-guards` (continue; do not branch off main).
- Prefix every Python/pytest/ruff invocation with `uv run`.
- Do NOT modify `src/aivc_model/` (exp05 numerics must stay intact).
- No `print` in library code; use `logging`. No bare `except`.
- Python 3.11+, strict type hints, Google-style docstrings, <50 lines/function.
- Fill value is `0.0` for nan and +/-inf (`np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`).
- Existing 96 sl_dl_model tests must stay green.
- Commit only the listed files per task (avoid `git add .`).

---

### Task 1: Build-time zero-fill cleaning

**Files:**
- Modify: `src/sl_dl_model/bags.py` (add `_zero_fill_nonfinite`; clean in `build_gwps_bags`)
- Test: `tests/sl_dl_model/test_bags.py`

**Interfaces:**
- Consumes: existing `build_gwps_bags(config, rng_seed) -> GwpsBags`, `_embed_matrix`.
- Produces: `_zero_fill_nonfinite(array: np.ndarray, label: str) -> tuple[np.ndarray, int]` returning `(cleaned, n_nonfinite)`. `build_gwps_bags` now guarantees `control_template` and every `bags_by_symbol` value is all-finite.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_bags.py`:

```python
def _toy_h5ad_with_nonfinite(path):
    """80 control + AAAS bag; plant NaN/inf into one control and one AAAS cell."""
    n, d = 120, 6
    rng = np.random.default_rng(3)
    genes = ["non-targeting"] * 80 + ["AAAS"] * 40
    obs = pd.DataFrame({"gene": genes})
    x = rng.normal(size=(n, d)).astype("float32")
    x[0, 1] = np.nan            # control cell, one entry
    x[0, 4] = np.inf
    x[80, 2] = np.nan           # AAAS cell, one entry
    x[81, 5] = -np.inf
    adata = ad.AnnData(X=x, obs=obs)
    adata.write_h5ad(path)


def test_build_zero_fills_nonfinite_entries(tmp_path, caplog):
    h5ad = tmp_path / "nonfinite.h5ad"
    _toy_h5ad_with_nonfinite(h5ad)
    cfg = SLDLConfig(gwps_h5ad=h5ad, control_template_size=80, cells_per_bag=40)
    with caplog.at_level(logging.WARNING, logger="sl_dl_model.bags"):
        bags = build_gwps_bags(cfg, rng_seed=17)
    assert np.isfinite(bags.control_template).all()
    assert np.isfinite(bags.bags_by_symbol["AAAS"]).all()
    # imputation, not drop: row counts preserved
    assert bags.control_template.shape == (80, 6)
    assert bags.bags_by_symbol["AAAS"].shape == (40, 6)
    msgs = [r.message.lower() for r in caplog.records]
    assert any("non-finite" in m for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_bags.py::test_build_zero_fills_nonfinite_entries -v`
Expected: FAIL — `assert np.isfinite(...).all()` is False (NaN/inf flow through).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/bags.py`, add the helper after `_dense_slice` (before `_embed_matrix`):

```python
def _zero_fill_nonfinite(array: np.ndarray, label: str) -> tuple[np.ndarray, int]:
    """Replace non-finite entries with 0.0, returning the count touched.

    STATE HVG input is normalized expression where 0 is the natural
    "no signal" baseline, so zero-fill keeps imputed entries from skewing the
    energy-distance / mean-delta bag losses.

    Args:
        array: Float array, possibly containing NaN/+-inf.
        label: Identifier used by the caller for logging context.

    Returns:
        Tuple of the cleaned float32 array and the number of non-finite entries.
    """
    mask = ~np.isfinite(array)
    n_nonfinite = int(mask.sum())
    if n_nonfinite == 0:
        return np.asarray(array, dtype=np.float32), 0
    cleaned = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(cleaned, dtype=np.float32), n_nonfinite
```

Then in `build_gwps_bags`, clean the control template right after it is sliced
(replace the `control_template = matrix[np.sort(control_rows)]` line):

```python
    control_template, control_nonfinite = _zero_fill_nonfinite(
        matrix[np.sort(control_rows)], "control_template"
    )
```

Track per-gene cleaning inside the bag loop (replace the
`bags[str(symbol).upper()] = matrix[np.sort(rows)]` line):

```python
        key = str(symbol).upper()
        bag, bag_nonfinite = _zero_fill_nonfinite(matrix[np.sort(rows)], key)
        bags[key] = bag
        if bag_nonfinite > 0:
            affected_genes += 1
            total_nonfinite += bag_nonfinite
```

Initialize the accumulators before the loop (`affected_genes = 0`,
`total_nonfinite = control_nonfinite`), and after the existing single-cell
warning add:

```python
    if total_nonfinite > 0:
        logger.warning(
            "Zero-filled %d non-finite GWPS expression entries across %d gene "
            "bag(s) plus the control template; upstream h5ad %s contained NaN/inf.",
            total_nonfinite,
            affected_genes,
            config.gwps_h5ad,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_bags.py::test_build_zero_fills_nonfinite_entries -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/bags.py tests/sl_dl_model/test_bags.py
git commit -m "fix: zero-fill non-finite GWPS bag entries at build (exp08)"
```

### Task 2: Load-time fail-fast verification

**Files:**
- Modify: `src/sl_dl_model/bags.py` (add `_assert_finite_bags`; call in `load_bags_npz`)
- Test: `tests/sl_dl_model/test_bags.py`

**Interfaces:**
- Consumes: existing `load_bags_npz(path) -> GwpsBags`.
- Produces: `_assert_finite_bags(control: np.ndarray, bags_by_symbol: dict[str, np.ndarray]) -> None`, raising `ValueError` (message contains `"non-finite"` and `"setup_exp08_assets.py bags"`) when any array is non-finite. `load_bags_npz` raises rather than returning a poisoned cache.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_bags.py`:

```python
def test_load_raises_on_nonfinite_cache(tmp_path):
    """A stale pre-fix cache with NaN must fail-fast at load, not silently load."""
    d = 6
    symbols = np.array(["AAAS", "KRAS"], dtype=object)
    flat = np.ones((10, d), dtype=np.float32)
    flat[7, 2] = np.nan  # poison a KRAS cell
    offsets = np.array([0, 5, 10], dtype=np.int64)
    control = np.ones((4, d), dtype=np.float32)
    npz_path = tmp_path / "stale_bags.npz"
    np.savez(
        npz_path,
        control_template=control,
        symbols=symbols,
        flat=flat,
        offsets=offsets,
        input_dim=np.int64(d),
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_bags_npz(npz_path)


def test_load_passes_on_clean_cache(tmp_path):
    """Round-tripping a cleanly built cache loads without error."""
    h5ad = tmp_path / "toy.h5ad"
    _toy_h5ad(h5ad)
    cfg = SLDLConfig(gwps_h5ad=h5ad, control_template_size=16, cells_per_bag=16)
    bags = build_gwps_bags(cfg, rng_seed=17)
    npz = tmp_path / "clean.npz"
    save_bags_npz(bags, npz)
    loaded = load_bags_npz(npz)  # must not raise
    assert np.isfinite(loaded.control_template).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_bags.py::test_load_raises_on_nonfinite_cache -v`
Expected: FAIL — no `ValueError` raised (poisoned cache loads silently).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/bags.py`, add the helper before `load_bags_npz`:

```python
def _assert_finite_bags(
    control: np.ndarray, bags_by_symbol: dict[str, np.ndarray]
) -> None:
    """Raise if any cached bag or the control template is non-finite.

    The build path is the single cleaning site; this verifies the invariant on
    load so a stale pre-fix cache fails loudly instead of poisoning training.

    Args:
        control: Control template array.
        bags_by_symbol: Per-gene response bags.

    Raises:
        ValueError: If any array contains NaN/inf, naming up to 10 symbols.
    """
    offenders: list[str] = []
    if not np.isfinite(control).all():
        offenders.append("control_template")
    for symbol, bag in bags_by_symbol.items():
        if not np.isfinite(bag).all():
            offenders.append(symbol)
    if offenders:
        shown = ", ".join(sorted(offenders)[:10])
        raise ValueError(
            f"GWPS bag cache contains non-finite values in: {shown}"
            f"{' ...' if len(offenders) > 10 else ''}. This is a stale pre-fix "
            "cache; rebuild it with "
            "`uv run python scripts/setup_exp08_assets.py bags`."
        )
```

In `load_bags_npz`, call it just before the `return GwpsBags(...)`:

```python
    _assert_finite_bags(control, bags)
    return GwpsBags(
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_bags.py::test_load_raises_on_nonfinite_cache tests/sl_dl_model/test_bags.py::test_load_passes_on_clean_cache -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/bags.py tests/sl_dl_model/test_bags.py
git commit -m "fix: fail-fast on non-finite GWPS bag cache at load (exp08)"
```

### Task 3: Train-boundary finite assert in `_bag_part`

**Files:**
- Modify: `src/sl_dl_model/train.py` (`_bag_part`, around line 954)
- Test: `tests/sl_dl_model/test_nan_guards.py`

**Interfaces:**
- Consumes: existing module-level `_bag_part(model, covered_train, control, device, key_a, vec_a, key_b, vec_b, bags) -> torch.Tensor | None`; `_bag_producer` helper (already in the test file); `producer._build_model()`.
- Produces: no signature change. `_bag_part` now `assert`s every `real` bag tensor is finite, raising `AssertionError` naming the offending gene key.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_nan_guards.py`:

```python
def test_bag_part_asserts_on_nonfinite_real():
    import numpy as np
    import pytest
    import torch

    from sl_dl_model.train import _bag_part

    producer = _bag_producer(max_epochs=1, warmup_epochs=0)
    model = producer._build_model()
    # Poison gene A's real bag directly (a path that bypasses load_bags_npz).
    producer.bags.bags_by_symbol["A"][0, 0] = np.nan
    control = torch.tensor(producer.bags.control_template)
    vec_a = producer.esm.vectors_by_symbol["A"]
    vec_b = producer.esm.vectors_by_symbol["B"]
    with pytest.raises(AssertionError, match="A"):
        _bag_part(
            model,
            covered_train={"A"},
            control=control,
            device="cpu",
            key_a="A",
            vec_a=vec_a,
            key_b="B",
            vec_b=vec_b,
            bags=producer.bags,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py::test_bag_part_asserts_on_nonfinite_real -v`
Expected: FAIL — no `AssertionError` (NaN passes through into `bag_loss`).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/train.py`, inside `_bag_part`, after the `real = torch.tensor(...)` line add the assert:

```python
        real = torch.tensor(bags.bags_by_symbol[key], device=device)
        assert torch.isfinite(real).all(), f"non-finite real bag for {key}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_nan_guards.py::test_bag_part_asserts_on_nonfinite_real -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/train.py tests/sl_dl_model/test_nan_guards.py
git commit -m "fix: assert finite real bag at _bag_part training boundary (exp08)"
```

---

### Task 4: `combine()` zero-weight guard + full suite verification

**Files:**
- Modify: `src/sl_dl_model/losses.py` (`combine`)
- Test: `tests/sl_dl_model/test_losses.py`

**Interfaces:**
- Consumes: existing `combine(parts: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor`.
- Produces: no signature change. A part whose weight is exactly `0.0` contributes a true `0.0` (never `0.0 * NaN = NaN`), so warmup with a non-finite-but-zero-weighted part stays finite.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_losses.py`:

```python
def test_combine_zero_weight_ignores_nonfinite():
    # Warmup: SL weight 0 while a (hypothetically) non-finite SL term exists.
    parts = {"sl": torch.tensor(float("nan")), "bag": torch.tensor(2.0)}
    weights = {"sl": 0.0, "bag": 1.0}
    total = combine(parts, weights)
    assert torch.isfinite(total).all()
    assert abs(total.item() - 2.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_losses.py::test_combine_zero_weight_ignores_nonfinite -v`
Expected: FAIL — `total` is NaN (`0.0 * nan`).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/losses.py`, replace the loop body in `combine`:

```python
    total: torch.Tensor | None = None
    for name, value in parts.items():
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            # Skip zero-weighted parts entirely so a non-finite-but-unused term
            # (e.g. SL during warmup) cannot poison the total via 0.0 * NaN.
            continue
        term = weight * value
        total = term if total is None else total + term
    if total is None:
        # Every part was zero-weighted; return a finite scalar zero on the
        # right device/dtype derived from an arbitrary part.
        any_value = next(iter(parts.values()))
        return torch.zeros((), dtype=any_value.dtype, device=any_value.device)
    return total
```

(Remove the old `assert total is not None` / `return total` tail; the
`if not parts: raise ValueError` guard at the top stays.)

- [ ] **Step 4: Run the targeted test, then the full sl_dl_model suite**

Run: `uv run python -m pytest tests/sl_dl_model/test_losses.py::test_combine_zero_weight_ignores_nonfinite -v`
Expected: PASS.

Run: `uv run ruff check src/sl_dl_model tests/sl_dl_model && uv run ruff format --check src/sl_dl_model tests/sl_dl_model`
Expected: clean (no diffs).

Run: `uv run python -m pytest tests/sl_dl_model -q`
Expected: all pass (96 prior + 6 new = 102).

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/losses.py tests/sl_dl_model/test_losses.py
git commit -m "fix: combine() skips zero-weighted parts to avoid 0*NaN warmup trap (exp08)"
```

---

## Operational follow-up (not a code task)

The cluster cache `k562_gwps_bags.npz` predates this fix and contains NaN.
Before the Phase 3 smoke run, rebuild it once:

```bash
uv run python scripts/setup_exp08_assets.py bags
```

The load-time verify (Task 2) will otherwise fail-fast on the stale cache — by
design. Smoke command after rebuild:

```bash
uv run python -m sl_dl_model run-cv \
  --config configs/experiments/08_k562_sl_pair_state_dl/phase3_bag_supervision.yaml \
  --producer state_dl --split-type CV2 --fold 0
```

## Self-Review

- **Spec coverage:** build clean → Task 1; load verify → Task 2; train assert → Task 3; `combine()` zero-weight trap → Task 4; operational cache rebuild → follow-up note. All design sections map to a task.
- **Decisions honored:** zero-fill (`np.nan_to_num` nan/posinf/neginf=0.0), keep-the-cell (row counts asserted in Task 1), strategy C single-cleaning-site (only `build_gwps_bags` cleans; load+train only verify).
- **Type consistency:** `_zero_fill_nonfinite -> tuple[np.ndarray, int]`, `_assert_finite_bags(control, bags_by_symbol) -> None`, `_bag_part` signature unchanged, `combine` signature unchanged — all consistent across tasks.
- **No placeholders:** every code step shows complete code and exact commands.

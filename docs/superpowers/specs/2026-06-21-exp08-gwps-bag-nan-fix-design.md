# Exp08 Phase 3 — GWPS Bag NaN Fix Design

**Date**: 2026-06-21
**Status**: Approved
**Branch (impl)**: continues `fix/exp08-phase3-nan-guards`
**Related**: `docs/experiment/08_k562_sl_pair_perturbseq_state_dl.md`,
prior guards in `src/sl_dl_model/{losses,pooling,train}.py`

## Problem

Phase 3 training crashes at epoch-0 validation with `roc_auc_score` "Input
contains NaN". Root cause was reproduced on local real data and is **not** in
STATE forward, the pert adapter, pooling, the logit head, or the optimizer.

The real GWPS bag cache already contains NaN. The h5ad source
(`K562_gwps_normalized_singlecell_01.h5ad`) has non-finite expression entries;
`build_gwps_bags()` writes them through to `k562_gwps_bags.npz` with no finite
check. In Phase 3, `bag_loss(pred_bag, real_bag)` hits a NaN-containing
`real_bag`, the batch loss becomes NaN, `safe_optimizer_step()` correctly skips
the step, no optimizer step is applied in the whole epoch, and the new
fail-fast raises `no optimizer step applied in epoch 0`.

## Reproduction (local, real data)

CV2 fold 0, first batch: 1024 pairs, 763 unique genes, 530 with a GWPS bag.
Of those 530 covered genes, **160 (~30%) have a non-finite bag**.

First triggering pair: batch index 2, `PBX3-BLM`.
- `PBX3` real bag finite; `BLM` real bag has 5 NaN entries.
- Real `state_checkpoint` forward on `PBX3-BLM`: `pred_bag_PBX3`,
  `pred_bag_BLM`, pooled embeddings, logit, SL loss, `bag_loss_PBX3` all
  finite; **`bag_loss_BLM` NaN → `combined_total` NaN**.

The NaN is upstream data, not a write bug. Cache and h5ad non-finite masks
match exactly:
- `BLM` cache shape `(106, 2000)`; NaN at `[1,138] [1,234] [1,400] [1,994]
  [1,1271]`.
- Maps to h5ad row `27959`, checkpoint genes `RIT1`, `TP53I3`, `SHQ1`,
  `RTKN2`, `SSH1`.

The NaN is **sparse** (5 entries in 106×2000) — this drives the cleaning
choice toward per-entry imputation rather than dropping cells or genes.

<!-- CHAIN -->

## Failure chain

```
h5ad has NaN expression entries
  → build_gwps_bags() has no finite check, writes them to k562_gwps_bags.npz
  → Phase 3 bag_loss(pred_bag, real_bag) sees NaN in real_bag
  → batch loss NaN
  → safe_optimizer_step() correctly skips
  → no optimizer step in epoch 0
  → fail-fast: "no optimizer step applied in epoch 0"
```

## Decisions

1. **Cleaning strategy: per-entry imputation, keep the cell** (not drop-cell,
   not drop-gene). NaN is sparse; dropping discards real cells / ~30% of
   covered genes.
2. **Fill value: zero-fill** (`np.nan_to_num`, with +/-inf also → 0). STATE HVG
   input is normalized expression where 0 is the natural "no signal" baseline;
   imputed entries get averaged against thousands of real values so the
   energy-distance / mean-delta terms are barely perturbed.
3. **Defense layering: C = build cleans, load verifies, train asserts.**
   Cleaning happens in exactly one place (build). Load and train *verify* the
   invariant rather than re-establishing it, so the two paths cannot drift, and
   a stale pre-fix cache fails loudly instead of being silently patched
   (preserving data provenance).

## Architecture — three guards along the bag lifecycle

| Boundary | Location | Behavior | Catches |
|---|---|---|---|
| Build (clean) | `build_gwps_bags()` in `bags.py` | Per-entry zero-fill of non-finite values in `control_template` and every gene bag; log affected gene / cell / entry counts | Upstream h5ad corruption (root cause) |
| Load (verify) | `load_bags_npz()` + on-the-fly path in `evaluate.py` | `np.isfinite` check; raise `ValueError` listing offending symbols + rebuild hint | Stale pre-fix cache on the cluster |
| Train (assert) | `_bag_part()` in `train.py` | Cheap `torch.isfinite(real).all()` assert after `torch.tensor(...)` | A future third construction path bypassing load |

The energy-distance / mean-delta NaN-safety guards already in `losses.py`
(`_safe_energy_distance`, `_safe_pairwise_dist`) and the std-pool guard in
`pooling.py` stay as-is — they protect against `cdist` / `sqrt` math traps on
*finite* input and are orthogonal to corrupt-input handling.

## Components

### Build cleaning (`bags.py`)
- New helper `_zero_fill_nonfinite(array, label) -> tuple[np.ndarray, int]`:
  returns `(cleaned, n_nonfinite)` using `np.nan_to_num(array, nan=0.0,
  posinf=0.0, neginf=0.0)`; counts non-finite entries via `~np.isfinite`.
- `build_gwps_bags()`: clean `control_template` and each gene bag at
  construction; accumulate `affected_genes`, `total_nonfinite_entries`,
  `affected_cells`; emit a single `logger.warning` summary when > 0 listing
  count of affected genes and entries (cap the gene list in the message).

### Load verification (`bags.py`)
- New helper `_assert_finite_bags(control, bags_by_symbol)`: raises `ValueError`
  with the offending symbols (capped list) and the rebuild command when any
  array is non-finite.
- `load_bags_npz()`: call it before returning.
- `evaluate.py` on-the-fly path (`build_gwps_bags(...)` result): build already
  cleans, so no extra check needed there; the on-the-fly result is finite by
  construction. (Load-path verification covers the cache case.)

### Train assert (`train.py`)
- `_bag_part()`: after `real = torch.tensor(bags.bags_by_symbol[key], ...)`,
  `assert torch.isfinite(real).all(), f"non-finite real bag for {key}"`. Cheap,
  localizes a failure to the exact gene at the training boundary.

### Secondary trap (`losses.py`)
- `combine()`: `0.0 * NaN = NaN` during warmup. Guard so a zero weight
  contributes exactly 0 regardless of the value (skip the term when
  `weight == 0.0`). Not the actual trigger, but fixes warmup-phase masking.

## Error handling

- Build: never raises on non-finite input — it cleans and logs. Raises only on
  the pre-existing conditions (missing `gene` column, no control cells).
- Load: raises `ValueError` (not assert) with offending symbols + rebuild hint
  when a cache is non-finite — actionable, points at `setup_exp08_assets.py
  bags`.
- Train: `assert` (cheap, internal invariant) — this path should be
  unreachable once build+load hold; the assert exists only to localize a future
  regression to a specific gene.

## Testing (TDD, RED → GREEN)

- `test_build_zero_fills_nonfinite_entries`: synthetic h5ad with planted NaN /
  inf in a control cell and a gene bag → built bags are all finite, counts
  logged.
- `test_build_preserves_cell_count`: a bag with sparse NaN keeps its row count
  (imputation, not drop).
- `test_load_raises_on_nonfinite_cache`: hand-write an NPZ with a NaN in `flat`
  → `load_bags_npz` raises `ValueError` naming the symbol.
- `test_load_passes_on_clean_cache`: round-trip a clean build → load succeeds.
- `test_bag_part_asserts_on_nonfinite_real` (or equivalent at the tensor
  boundary): poisoned `bags_by_symbol` entry → assert fires with the gene key.
- `test_combine_zero_weight_ignores_nonfinite`: `combine({"bag": nan}, {"bag":
  0.0})` returns finite 0, not NaN.

Existing 96 sl_dl_model tests must stay green; `src/aivc_model/` untouched.

## Operational follow-up (out of the spec's code scope)

The cluster cache `k562_gwps_bags.npz` was built pre-fix and contains NaN. It
must be rebuilt once via `uv run python scripts/setup_exp08_assets.py bags`
before the Phase 3 smoke run; the load-time verify will otherwise fail-fast
(by design).

## Out of scope

- Fixing the upstream h5ad (`K562_gwps_normalized_singlecell_01.h5ad`) — we
  treat its NaN as given and clean at ingestion.
- Changing the energy-distance / pooling / optimizer guards (already landed).
- Median or distribution-aware imputation (zero-fill chosen).

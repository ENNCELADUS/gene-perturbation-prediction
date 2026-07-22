"""Integration gates and seed-replicate aggregation for Bridge A.

Additive wrapper around ``aivc_model.bridge_a`` / ``bridge_a_independent``:
keeps the core forward-path wiring in those modules under the repository's
600-line file budget while giving ``scripts/bridge_a_forward.py`` and the
test suite a single, importable place for:

* ``assert_perturbed_arm_integrity`` -- the perturbed-arm integration gate a
  broken or degenerate a-perturbed forward path must FAIL. Checks
  gene-specific source indexing, non-identity vs. control, shape/batch
  alignment, finite/non-constant outputs across query genes, deterministic
  replay, AND (Finding B) that swapping the basal state from control to
  a-perturbed -- holding the pooler reference latent FIXED -- actually
  changes the prediction (catches a forward that silently ignores its basal
  input but still varies by query gene).
* ``aggregate_seed_replicate_pairs`` / ``aggregate_seed_replicate_c_hat`` --
  fold multi-seed pilot replicates into per-pair/per-arm seed mean and
  variance, sharing ``bridge_a_independent.validate_seed_replicate_tables``'s
  Finding-C completeness guards so an incomplete or duplicated replicate set
  cannot silently read as legitimate (or zero-variance). The c_hat key is
  ``(basal_state, query_gene, n_windows, reference_convention)``, not just
  ``(basal_state, query_gene)``: pooling across window budgets or reference
  conventions would silently average different estimands together.
* ``project_full_sweep`` -- projects the wall-clock/memory cost of the full
  candidate-universe sweep from measured per-forward timing; never runs it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aivc_model.bridge_a import (
    REFERENCE_CONVENTIONS,
    BridgeAContext,
    panel_latent,
    predict_c_hat,
    reference_latent,
)
from aivc_model.bridge_a_independent import (
    IndependentPanelSpec,
    control_panel_for_window_count,
    independent_perturbed_panel,
    validate_seed_replicate_tables,
)
from aivc_model.train import _EvaluationControlPanel

_LOGGER = logging.getLogger(__name__)

# float32 round-off is ~1e-7; a real basal-dependent forward through a
# nonlinear STATE model + response encoder + pooler produces differences many
# orders of magnitude larger, so this margin flags genuine basal-
# insensitivity (Finding B) without false-triggering on numerical noise.
_DEFAULT_BASAL_SENSITIVITY_TOLERANCE = 1e-6

# The frozen candidate universe (scripts/lock_bridge_a_universe.py): a
# complete graph over the exp05-observed-covered Horlbeck K562 genes. This is
# ``project_full_sweep``'s DEFAULT projection size only -- pass ``gene_count``
# explicitly to project a reduced/alternate universe (e.g.
# ``--drop-nonqueryable``); the pair count is always DERIVED, never hardcoded.
_FULL_SWEEP_GENE_COUNT = 408


def _check_deterministic_replay(
    basal_gene: str,
    panel_first: _EvaluationControlPanel,
    panel_replay: _EvaluationControlPanel,
) -> None:
    """Raise unless panel construction and forward tensors replay bit-identically."""
    if not np.array_equal(panel_first.selected_indices, panel_replay.selected_indices):
        raise RuntimeError(
            f"{basal_gene}: panel construction is not deterministic under a fixed seed"
        )
    if not torch.equal(
        torch.cat(panel_first.control_chunks, dim=0),
        torch.cat(panel_replay.control_chunks, dim=0),
    ):
        raise RuntimeError(
            f"{basal_gene}: forward panel tensors are not deterministic under "
            "a fixed seed"
        )


def _check_not_identical_to_control(
    basal_gene: str, concat_first: torch.Tensor, control_concat: torch.Tensor
) -> None:
    """Raise if the a-perturbed STATE input equals the control panel's (degenerate)."""
    if concat_first.shape == control_concat.shape and torch.equal(
        concat_first, control_concat
    ):
        raise RuntimeError(
            f"{basal_gene}-perturbed STATE input is IDENTICAL to the "
            "window-count-matched control panel -- the a-perturbed arm is not "
            "wired to gene-specific observed cells"
        )


def _check_panel_shape_alignment(
    basal_gene: str,
    panel_first: _EvaluationControlPanel,
    cell_set_len: int,
    input_dim: int,
) -> None:
    """Raise on a panel chunk-shape or batch/gem_group misalignment."""
    for chunk, batch_chunk in zip(
        panel_first.control_chunks,
        panel_first.batch_index_chunks,
        strict=True,
    ):
        if tuple(chunk.shape) != (cell_set_len, input_dim):
            raise RuntimeError(
                f"{basal_gene}: panel chunk shape {tuple(chunk.shape)} != "
                f"({cell_set_len}, {input_dim})"
            )
        if batch_chunk is not None and int(batch_chunk.shape[0]) != int(chunk.shape[0]):
            raise RuntimeError(
                f"{basal_gene}: batch/gem_group labels misaligned with cell chunk"
            )


def _check_source_indexing(
    basal_gene: str, concat_first: torch.Tensor, raw_rows: set[bytes]
) -> None:
    """Raise if any panel cell is not a member of this gene's observed bag."""
    n_foreign = sum(
        1
        for row in concat_first.detach().cpu().numpy()
        if row.tobytes() not in raw_rows
    )
    if n_foreign:
        raise RuntimeError(
            f"{basal_gene}: {n_foreign}/{concat_first.shape[0]} panel cells are "
            "not members of this gene's observed bag -- source-gene indexing "
            "is wrong"
        )


def _check_query_variation(
    basal_gene: str, outputs: list[float], query_genes: list[str]
) -> None:
    """Raise if c_hat is non-finite or constant across the tested query genes."""
    if not all(np.isfinite(outputs)):
        raise RuntimeError(f"{basal_gene}: non-finite c_hat in {outputs}")
    if outputs[0] == outputs[1]:
        raise RuntimeError(
            f"{basal_gene}: c_hat is constant across query genes {query_genes} "
            f"({outputs}) -- the response path looks degenerate"
        )


def _check_basal_sensitivity(
    basal_gene: str,
    control_outputs: list[float],
    perturbed_outputs: list[float],
    query_genes: list[str],
    tolerance: float,
) -> list[float]:
    """Raise unless swapping the basal state changes c_hat beyond tolerance (B)."""
    all_finite = all(np.isfinite(control_outputs)) and all(
        np.isfinite(perturbed_outputs)
    )
    if not all_finite:
        raise RuntimeError(
            f"{basal_gene}: non-finite c_hat in the basal-sensitivity check "
            f"(control={control_outputs}, perturbed={perturbed_outputs})"
        )
    deltas = [
        abs(perturbed_outputs[i] - control_outputs[i]) for i in range(len(query_genes))
    ]
    if not any(delta > tolerance for delta in deltas):
        raise RuntimeError(
            f"{basal_gene}: swapping the basal state from control to "
            f"{basal_gene}-perturbed did not change c_hat beyond "
            f"tolerance={tolerance:.1e} for ANY of {query_genes} "
            f"(control={control_outputs}, perturbed={perturbed_outputs}) -- "
            "the forward path looks insensitive to the basal input"
        )
    return deltas


def assert_perturbed_arm_integrity(
    context: BridgeAContext,
    genes: Sequence[str],
    coverage: dict[str, int],
    spec: IndependentPanelSpec,
    reference_convention: str,
    *,
    basal_sensitivity_tolerance: float = _DEFAULT_BASAL_SENSITIVITY_TOLERANCE,
) -> dict[str, object]:
    """Fail loudly if the a-perturbed forward path is broken or degenerate.

    For every gene in ``genes`` (used once as the a-perturbed basal state, at
    its independent-N window budget -- Finding A) this verifies gene-specific
    source indexing, non-identity vs. the window-count-matched control panel,
    panel shape/batch alignment, finite/non-constant c_hat across >= 2 query
    genes, deterministic replay, and BASAL-SENSITIVITY (Finding B): holding
    the pooler reference latent FIXED at the shared control latent for BOTH
    arms (isolating this from the Finding-1 reference-latent-convention
    effect), swapping the basal from control to this gene's a-perturbed
    panel must change c_hat beyond ``basal_sensitivity_tolerance`` for at
    least one query gene -- a forward that ignores ``control_chunks``
    entirely but still returns deterministic, query-specific outputs would
    pass every other check and must fail here.

    Args:
        context: A loaded ``BridgeAContext``.
        genes: >= 3 distinct gene symbols (1 basal + >= 2 query per basal).
        coverage: ``canonical_symbol -> observed cell count``
            (``bridge_a_panels.load_k562_gene_coverage``).
        spec: The independent-N window sizing to gate with.
        reference_convention: ``"self"`` or ``"control"``; the gate runs
            under whichever convention the caller is about to pilot with.
        basal_sensitivity_tolerance: Minimum ``|c_hat|`` change required from
            swapping the basal state (Finding B) before it counts as a real
            difference rather than numerical noise.

    Returns:
        A JSON-serializable summary of what was checked.

    Raises:
        ValueError: If ``genes`` has fewer than 3 distinct entries, or
            ``reference_convention`` is not recognized.
        RuntimeError: If any integrity invariant is violated.
    """
    unique_genes = list(dict.fromkeys(str(gene).upper() for gene in genes))
    if len(unique_genes) < 3:
        raise ValueError(
            "assert_perturbed_arm_integrity needs >= 3 distinct genes "
            f"(1 basal + 2 query), got {unique_genes}"
        )
    if reference_convention not in REFERENCE_CONVENTIONS:
        raise ValueError(f"unknown reference_convention {reference_convention!r}")

    checked: list[dict[str, object]] = []
    control_panel_cache: dict[int, tuple[_EvaluationControlPanel, torch.Tensor]] = {}
    with torch.no_grad():
        for basal_gene in unique_genes:
            index = context.gene_to_index[basal_gene]
            raw_bag = np.asarray(context.data.input_bags[index], dtype=np.float32)
            if raw_bag.shape[0] < 1:
                raise RuntimeError(
                    f"gene {basal_gene} has an empty observed-cell bag; the "
                    "perturbed arm cannot be gated"
                )
            raw_rows = {row.tobytes() for row in raw_bag}

            panel_first, budget = independent_perturbed_panel(
                context, basal_gene, coverage, spec
            )
            panel_replay, _ = independent_perturbed_panel(
                context, basal_gene, coverage, spec
            )
            _check_deterministic_replay(basal_gene, panel_first, panel_replay)
            concat_first = torch.cat(panel_first.control_chunks, dim=0)

            n_windows = budget.n_windows
            if n_windows not in control_panel_cache:
                shared_control = control_panel_for_window_count(
                    context, n_windows, spec
                )
                control_panel_cache[n_windows] = (
                    shared_control,
                    panel_latent(context.model, shared_control),
                )
            shared_control, shared_control_latent = control_panel_cache[n_windows]
            control_concat = torch.cat(shared_control.control_chunks, dim=0)

            _check_not_identical_to_control(basal_gene, concat_first, control_concat)
            _check_panel_shape_alignment(
                basal_gene, panel_first, spec.cell_set_len, context.data.input_dim
            )
            _check_source_indexing(basal_gene, concat_first, raw_rows)

            basal_latent = panel_latent(context.model, panel_first)
            ref_latent = reference_latent(
                reference_convention, shared_control_latent, basal_latent
            )
            query_genes = [gene for gene in unique_genes if gene != basal_gene][:2]

            outputs = [
                predict_c_hat(context.model, panel_first, ref_latent, query)
                for query in query_genes
            ]
            _check_query_variation(basal_gene, outputs, query_genes)

            control_outputs = [
                predict_c_hat(
                    context.model, shared_control, shared_control_latent, query
                )
                for query in query_genes
            ]
            perturbed_outputs_fixed_reference = [
                predict_c_hat(context.model, panel_first, shared_control_latent, query)
                for query in query_genes
            ]
            basal_sensitivity_deltas = _check_basal_sensitivity(
                basal_gene,
                control_outputs,
                perturbed_outputs_fixed_reference,
                query_genes,
                basal_sensitivity_tolerance,
            )

            checked.append(
                {
                    "basal_gene": basal_gene,
                    "n_windows": n_windows,
                    "effective_n": budget.effective_n,
                    "bootstrapped": budget.bootstrapped,
                    "query_genes_checked": query_genes,
                    "c_hat_sample": outputs,
                    "basal_sensitivity_delta": basal_sensitivity_deltas,
                }
            )
    _LOGGER.info(
        "perturbed-arm gate PASSED for %d genes under reference_convention=%s "
        "(basal_sensitivity_tolerance=%.1e)",
        len(checked),
        reference_convention,
        basal_sensitivity_tolerance,
    )
    return {
        "passed": True,
        "reference_convention": reference_convention,
        "basal_sensitivity_tolerance": basal_sensitivity_tolerance,
        "genes_checked": checked,
    }


def aggregate_seed_replicate_pairs(
    pair_tables: Sequence[pd.DataFrame],
    seeds: Sequence[int],
    *,
    seed_variance_threshold: float = 0.5,
) -> pd.DataFrame:
    """Fold repeat-seed independent-N pair-table replicates into per-pair seed
    mean/std (Finding 2's re-validation report), with Finding-C completeness
    guards (``validate_seed_replicate_tables``) so an incomplete or
    duplicated replicate set cannot silently read as a legitimate (or
    zero-variance) result.

    Args:
        pair_tables: One table per requested seed, in the SAME order as
            ``seeds``, each carrying that single seed in a ``seed`` column
            (see
            ``bridge_a_independent.bridge_a_pairs_from_independent_c_hat_table``
            plus ``.assign(seed=...)``).
        seeds: The seeds actually requested -- >= 2 DISTINCT values,
            validated against ``pair_tables`` (not merely counted).
        seed_variance_threshold: A pair is flagged ``seed_variance_material``
            when ``s_a_seed_std > seed_variance_threshold *
            abs(s_a_seed_mean)``, i.e. seed noise is not small relative to
            the reported effect.

    Returns:
        One row per pair with ``s_a_seed_mean``, ``s_a_seed_std``,
        ``delta_a_to_b_seed_mean/std``, ``delta_b_to_a_seed_mean/std``,
        ``n_seed_replicates``, ``seed_variance_material``, and (when
        present) the per-pair window-budget provenance columns carried
        through unchanged (asserted seed-invariant).

    Raises:
        ValueError: See ``validate_seed_replicate_tables``.
        RuntimeError: If any pair does not have exactly ``len(seeds)``
            distinct-seed replicates, or pair-level provenance columns vary
            across seed replicates for the same pair.
    """
    requested_seeds = list(seeds)
    combined = validate_seed_replicate_tables(
        pair_tables, requested_seeds, ("gene_a", "gene_b"), "pair table"
    )
    grouped = combined.groupby(["gene_a", "gene_b"], sort=False)

    replicate_counts = grouped["seed"].nunique().rename("n_seed_replicates")
    incomplete = replicate_counts[replicate_counts != len(requested_seeds)]
    if not incomplete.empty:
        raise RuntimeError(
            f"{len(incomplete)} pair(s) do not have exactly "
            f"{len(requested_seeds)} distinct-seed replicates: "
            f"{incomplete.to_dict()}"
        )

    summary = grouped[["delta_a_to_b", "delta_b_to_a", "s_a"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_seed_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().merge(
        replicate_counts.reset_index(), on=["gene_a", "gene_b"], validate="one_to_one"
    )

    provenance_columns = [
        column
        for column in (
            "n_windows_a_to_b",
            "n_windows_b_to_a",
            "effective_n_a_to_b",
            "effective_n_b_to_a",
            "bootstrapped_a_to_b",
            "bootstrapped_b_to_a",
            "n_cells_a",
            "n_cells_b",
            "high_effective_n",
        )
        if column in combined.columns
    ]
    if provenance_columns:
        varying = grouped[provenance_columns].nunique()
        non_constant = [
            column for column in provenance_columns if (varying[column] > 1).any()
        ]
        if non_constant:
            raise RuntimeError(
                f"pair-level provenance columns {non_constant} vary across seed "
                "replicates for the same pair -- window-budget/coverage lookups "
                "should be seed-independent"
            )
        first_values = grouped[provenance_columns].first().reset_index()
        summary = summary.merge(
            first_values, on=["gene_a", "gene_b"], validate="one_to_one"
        )

    summary["seed_variance_material"] = summary["s_a_seed_std"] > (
        seed_variance_threshold * summary["s_a_seed_mean"].abs()
    )
    return summary


_C_HAT_SEED_KEY_COLUMNS = (
    "basal_state",
    "query_gene",
    "n_windows",
    "reference_convention",
)


def _infer_single_seed(table: pd.DataFrame, label: str) -> int:
    """Read the one ``seed`` value a single-seed replicate table carries.

    Raises:
        ValueError: If ``table`` lacks a ``seed`` column or is not single-valued.
    """
    if "seed" not in table.columns:
        raise ValueError(f"each {label} must carry a 'seed' column to infer seeds from")
    observed = table["seed"].unique()
    if len(observed) != 1:
        raise ValueError(f"cannot infer one seed for a {label}: {observed.tolist()}")
    return int(observed[0])


def aggregate_seed_replicate_c_hat(
    c_hat_tables: Sequence[pd.DataFrame],
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Fold repeat-seed ``compute_independent_c_hat_table`` outputs into per-arm
    seed mean/std (Finding 2), sharing ``aggregate_seed_replicate_pairs``'s
    Finding-C completeness guards (``validate_seed_replicate_tables``).

    Groups by ``_C_HAT_SEED_KEY_COLUMNS`` -- NOT merely ``(basal_state,
    query_gene)``. Control c_hat is cached/emitted per ``(query_gene,
    n_windows)`` and the a-perturbed arm is computed once per
    ``reference_convention`` (Finding 1); omitting either from the key would
    pool DIFFERENT window budgets or conventions into one row, so the
    reported ``c_hat_seed_std`` would not be seed variance.

    Args:
        c_hat_tables: One table per requested seed, in the SAME order as
            ``seeds``, each carrying that single seed in a ``seed`` column
            (see ``compute_independent_c_hat_table`` plus
            ``.assign(seed=...)``).
        seeds: The seeds actually requested -- >= 2 DISTINCT values,
            validated against ``c_hat_tables`` (not merely counted). If
            ``None`` (default), inferred from each table's own single
            ``seed`` value -- kept for ``scripts/bridge_a_forward.py``'s
            existing caller; pass it explicitly where available.

    Returns:
        One row per ``_C_HAT_SEED_KEY_COLUMNS`` key with ``c_hat_seed_mean``,
        ``c_hat_seed_std``, and ``n_seed_replicates``. ``c_hat_seed_std`` is
        NEVER filled to ``0.0`` for an undefined single-observation std: the
        completeness guard below already forces every returned row to have
        exactly ``len(seeds) >= 2`` observations.

    Raises:
        ValueError: See ``validate_seed_replicate_tables``.
        RuntimeError: If any key does not have exactly ``len(seeds)``
            distinct-seed replicates.
    """
    requested_seeds = (
        [_infer_single_seed(table, "c_hat table") for table in c_hat_tables]
        if seeds is None
        else list(seeds)
    )
    key_columns = list(_C_HAT_SEED_KEY_COLUMNS)
    combined = validate_seed_replicate_tables(
        c_hat_tables, requested_seeds, key_columns, "c_hat table"
    )
    grouped = combined.groupby(key_columns, sort=False)

    replicate_counts = grouped["seed"].nunique().rename("n_seed_replicates")
    incomplete = replicate_counts[replicate_counts != len(requested_seeds)]
    if not incomplete.empty:
        raise RuntimeError(
            f"{len(incomplete)} c_hat arm(s) do not have exactly "
            f"{len(requested_seeds)} distinct-seed replicates: "
            f"{incomplete.to_dict()}"
        )

    summary = grouped["c_hat"].agg(["mean", "std"]).reset_index()
    summary = summary.rename(
        columns={"mean": "c_hat_seed_mean", "std": "c_hat_seed_std"}
    )
    return summary.merge(
        replicate_counts.reset_index(), on=key_columns, validate="one_to_one"
    )


def load_gi_lookup(pairs_csv: Path) -> dict[tuple[str, str], tuple[float, bool]]:
    """Load the candidate-universe Horlbeck GI lookup keyed by unordered pair."""
    pairs = pd.read_csv(pairs_csv)
    lookup: dict[tuple[str, str], tuple[float, bool]] = {}
    for row in pairs.itertuples(index=False):
        key = tuple(sorted((str(row.gene_a_canonical), str(row.gene_b_canonical))))
        lookup[key] = (float(row.gi_score), bool(row.is_strong_sl))
    return lookup


def score_pairs_against_gi(
    pairs: pd.DataFrame,
    gi_lookup: dict[tuple[str, str], tuple[float, bool]],
) -> pd.DataFrame:
    """Attach ``gi_score``/``is_strong_sl`` to a ``(gene_a, gene_b)`` frame."""
    gi_scores: list[float] = []
    strong_sl: list[bool] = []
    for row in pairs.itertuples(index=False):
        key = tuple(sorted((row.gene_a, row.gene_b)))
        match = gi_lookup.get(key)
        gi_scores.append(math.nan if match is None else match[0])
        strong_sl.append(False if match is None else match[1])
    return pairs.assign(gi_score=gi_scores, is_strong_sl=strong_sl)


def project_full_sweep(
    forward_seconds: list[float],
    device: torch.device,
    max_windows: int,
    gene_count: int = _FULL_SWEEP_GENE_COUNT,
) -> dict[str, object]:
    """Project the wall-clock and peak-memory cost of the full candidate-pair sweep.

    The full candidate universe is a complete graph over ``gene_count`` genes
    (the LOCKED universe by default -- pass the actual gene count explicitly
    to project a reduced/alternate universe, e.g. ``--drop-nonqueryable``):
    ``gene_count * (gene_count - 1)`` a-perturbed-basal forwards, plus an
    UPPER BOUND of ``gene_count * max_windows`` control-arm forwards, since
    ``c_hat[query | control@w]`` is cached by ``(query_gene, window_count)``
    and reused across every basal gene sharing that window count (Finding A).
    Does not run the sweep.

    Args:
        forward_seconds: Measured per-forward wall-clock seconds to project from.
        device: The device the measurement ran on (for peak-memory reporting).
        max_windows: The independent-N window cap (bounds control-arm cost).
        gene_count: The candidate universe's gene count to project over --
            defaults to the locked ``_FULL_SWEEP_GENE_COUNT``-gene universe.

    Raises:
        ValueError: If ``gene_count < 2`` (no pairs to form).
    """
    if gene_count < 2:
        raise ValueError(f"gene_count must be >= 2 to form any pairs, got {gene_count}")
    seconds = np.asarray(forward_seconds, dtype=float)
    pair_count = gene_count * (gene_count - 1) // 2
    a_perturbed_forwards = gene_count * (gene_count - 1)
    control_cache_forwards_upper_bound = gene_count * max_windows
    total_forwards = a_perturbed_forwards + control_cache_forwards_upper_bound
    mean_seconds = float(seconds.mean()) if seconds.size else math.nan
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else math.nan
    )
    return {
        "measured_forward_count": int(seconds.size),
        "measured_forward_seconds_mean": mean_seconds,
        "measured_forward_seconds_median": (
            float(np.median(seconds)) if seconds.size else math.nan
        ),
        "measured_forward_seconds_p95": (
            float(np.percentile(seconds, 95)) if seconds.size else math.nan
        ),
        "full_sweep_pair_count": pair_count,
        "full_sweep_gene_count": gene_count,
        "full_sweep_a_perturbed_forwards": a_perturbed_forwards,
        "full_sweep_control_cache_forwards_upper_bound": (
            control_cache_forwards_upper_bound
        ),
        "full_sweep_total_forwards_upper_bound": total_forwards,
        "projected_full_sweep_wallclock_hours_single_gpu": (
            total_forwards * mean_seconds / 3600.0 if seconds.size else math.nan
        ),
        "peak_gpu_memory_mib_this_process": peak_memory_mib,
        "note": (
            "Projection assumes one query gene per forward, sequential on a "
            "single GPU, at the measured mean per-forward wall-clock time "
            "(mixed across the control gate's --panel-size and the pilot's "
            "independent-N panels); control-arm cost is an UPPER BOUND "
            "(gene_count * max_windows) since c_hat[query | control@w] is "
            "cached and reused across basal genes sharing a window count -- "
            "it does not run the sweep."
        ),
    }

"""Independent-N-matched Bridge A forward orchestration (Finding A).

Builds every a-perturbed panel, and its window-count-matched control panel,
at the INDEPENDENT-cell-count window budget computed by
``bridge_a_panels.compute_window_budget`` -- see that module's docstring for
the windowed / capped / sub-window-bootstrap-and-flag policy this replaces
the old flat-nominal-cell-target panel matching with. Also exports
``validate_seed_replicate_tables`` (Finding-C), shared by
``bridge_a_gates.aggregate_seed_replicate_pairs``/``..._c_hat``.

Control-panel caching (cost control)
--------------------------------------
For a fixed window count ``w``, ``c_hat[query | control@w]`` does not depend
on WHICH gene supplied the ``w``-window a-perturbed budget -- only on
``query`` and ``w``. ``compute_independent_c_hat_table`` caches control
c_hat by ``(query_gene, w)`` so it is computed once and reused across every
basal gene sharing that window count, bounding control-arm cost by
``len(genes) * max_windows`` regardless of how many basal genes are scored.

Coverage-table vs. observed-bag reconciliation
-------------------------------------------------
The coverage table (``bridge_a_panels.load_k562_gene_coverage``) was built to
audit Horlbeck/Replogle coverage, independently of the exp05 fixed pool's own
data assembly (primary GWPS bags, pooled with an external K562 non-Replogle
supplement -- Adamson/Dixit -- by ``prepare.merge_gene_bag_pool`` whenever a
gene appears in both sources). On the live exp05 pool the two disagree for
54/408 genes (13%): most commonly the exp05 bag is capped at exactly 256
cells (4 windows) while the coverage table reports more (the exp05
assembly's own per-gene cap), and less commonly (~5 genes) the exp05 bag has
MORE cells than the coverage table -- because ``merge_gene_bag_pool``
concatenated that gene's Replogle cells with non-Replogle supplement cells,
and the (Replogle-only) coverage table never counted the latter.

``independent_perturbed_panel`` reconciles these in two steps (see
``_eligible_perturbed_rows``), never sampling from more cells than either
step certifies:

1. Source restriction. ``merge_gene_bag_pool`` tags every pooled cell's batch
   label with a ``"replogle:..."``/``"non_replogle:..."`` source prefix.
   When a gene's bag carries BOTH tags with >= 1 Replogle row, the
   non-Replogle rows are dropped BEFORE anything else -- an independent-N
   panel budgeted off the Replogle-only coverage count must never draw a
   non-Replogle cell. When a gene's bag is tagged but carries ZERO Replogle
   rows (entirely foreign supplement), there is no Replogle-source
   a-perturbed state to build a panel from at all, and this FAILS CLOSED
   with a ``ValueError`` rather than silently substituting Adamson/Dixit
   cells. Untagged (legacy, no source prefix at all) bags are left
   unrestricted.
2. ``n_a = min(coverage[gene], eligible_bag_size)`` -- the CONSERVATIVE bound
   that never claims more independent cells than the (now source-qualified)
   pool actually has, logged at ``WARNING`` when the two disagree. When the
   eligible bag is STRICTLY LARGER than ``n_a``, the kept rows are a SEEDED
   random choice without replacement (the same seed that drives the panel's
   own downstream sampling) rather than the pool's first ``n_a`` rows --
   bag row order is inherited from the h5ad and may be clustered (e.g. by
   ``gem_group``), so a fixed truncation would bias which cells are used and
   suppress genuine cross-seed variance. Either way the eligible bag ends up
   at exactly ``n_a`` rows, so the window budget's ``bootstrapped`` flag and
   the panel builder's actual with-/without-replacement sampling decision
   (which compares its request against the size of the array it is handed)
   always agree.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from aivc_model.bridge_a import (
    CONTROL_BASAL_STATE,
    REFERENCE_CONVENTIONS,
    BridgeAContext,
    delta_a_to_b,
    panel_latent,
    predict_c_hat,
    reference_latent,
    symmetrized_codependency,
)
from aivc_model.bridge_a_panels import WindowBudget, compute_window_budget
from aivc_model.train import _EvaluationControlPanel, _build_evaluation_control_panel

_LOGGER = logging.getLogger(__name__)

# ``prepare.merge_gene_bag_pool`` stamps every pooled cell's batch label with this
# source prefix (``"replogle"`` or ``"replogle:<original batch>"``) for primary GWPS
# cells, and ``"non_replogle"``/``"non_replogle:<source>:<original batch>"`` for K562
# non-Replogle supplement (Adamson/Dixit) cells -- see the module docstring.
_REPLOGLE_SOURCE_PREFIX = "replogle"
_NON_REPLOGLE_SOURCE_PREFIX = "non_replogle"


@dataclass(frozen=True)
class IndependentPanelSpec:
    """STATE window sizing for independent-N-matched Bridge A panels.

    Unlike ``bridge_a.PanelSpec`` (a flat cell target shared by every arm),
    every a-perturbed panel built with this spec gets its OWN window budget
    from ``compute_window_budget``; the matched control panel for the same
    direction is built at the SAME window count. ``max_windows`` bounds cost
    and the influence of outlier-large gene bags.
    """

    cell_set_len: int
    macro_batch_windows: int
    seed: int
    max_windows: int

    def __post_init__(self) -> None:
        if self.cell_set_len < 1:
            raise ValueError(f"cell_set_len must be >= 1, got {self.cell_set_len}")
        if self.macro_batch_windows < 1:
            raise ValueError(
                f"macro_batch_windows must be >= 1, got {self.macro_batch_windows}"
            )
        if self.max_windows < 1:
            raise ValueError(f"max_windows must be >= 1, got {self.max_windows}")


def _reconcile_observed_cell_count(
    gene: str,
    coverage_n_cells: int,
    raw_available: int,
) -> int:
    """Reconcile the coverage table's cell count against the observed bag size.

    The two are computed by independent pipelines (see module docstring) and
    disagree for a real, non-negligible slice of the exp05 pool. Takes the
    smaller of the two -- CONSERVATIVE: never claims more independent cells
    than the pool actually has available to sample from -- and logs a
    warning when they disagree rather than silently trusting either source.
    """
    n_cells = min(coverage_n_cells, raw_available)
    if n_cells != coverage_n_cells:
        _LOGGER.warning(
            "%s: coverage table reports n_cells=%d but the exp05 pool's "
            "observed bag has %d cells; using the smaller, conservative "
            "count (%d) for the independent-N window budget",
            gene,
            coverage_n_cells,
            raw_available,
            n_cells,
        )
    return n_cells


def _replogle_source_mask(batch_labels: np.ndarray) -> np.ndarray:
    """Boolean mask selecting Replogle-source rows from per-cell source labels."""
    return np.asarray(
        [str(label).startswith(_REPLOGLE_SOURCE_PREFIX) for label in batch_labels],
        dtype=bool,
    )


def _recognized_source_mask(batch_labels: np.ndarray) -> np.ndarray:
    """Boolean mask selecting rows carrying EITHER recognized source prefix.

    ``merge_gene_bag_pool`` tags every pooled cell it touches with a
    ``"replogle"``- or ``"non_replogle"``-prefixed source label (see the module
    docstring). A row matching neither prefix is an UNTAGGED legacy label (no
    gene in this pool was merged from more than one source), not evidence of a
    non-Replogle cell.
    """
    return np.asarray(
        [
            str(label).startswith(_REPLOGLE_SOURCE_PREFIX)
            or str(label).startswith(_NON_REPLOGLE_SOURCE_PREFIX)
            for label in batch_labels
        ],
        dtype=bool,
    )


def _eligible_perturbed_rows(
    gene: str,
    raw_bag: np.ndarray,
    raw_batch: np.ndarray | None,
    coverage_n_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Restrict and reconcile one gene's a-perturbed bag to its ACTUAL sampling pool.

    Two corrections are applied, in order, so the returned ``n_cells`` and the
    returned bag/batch arrays are always mutually consistent -- the reconciled
    metadata can never under-report what the panel actually draws from:

    1. Source restriction, by inspecting ``raw_batch`` against the recognized
       ``"replogle:..."``/``"non_replogle:..."`` source prefixes (see the module
       docstring):

       - No recognized prefix anywhere in ``raw_batch`` (or ``raw_batch`` is
         ``None``): untagged legacy labels -- no gene in this pool was merged
         from more than one source, so there is nothing to restrict by. The
         eligible population is the pool's own row order, capped to
         ``n_cells`` below -- an arbitrary-but-deterministic assumption, not a
         claim that the retained rows are more "correct" than any other subset.
       - At least one recognized prefix present, and at least one Replogle row:
         a MIXED bag (``merge_gene_bag_pool`` merged in a non-Replogle K562
         supplement for this gene). The non-Replogle rows are dropped before
         anything else -- an independent-N panel budgeted off the
         Replogle-only coverage count must never draw a non-Replogle cell.
       - At least one recognized prefix present, and ZERO Replogle rows: the
         ENTIRE bag is foreign (non-Replogle supplement). There is no
         Replogle-source a-perturbed state to sample from at all, so this
         FAILS CLOSED with a ``ValueError`` rather than silently substituting
         Adamson/Dixit cells for the required Replogle perturbed state.
    2. Coverage/bag-size reconciliation (``_reconcile_observed_cell_count``): the
       CONSERVATIVE ``min`` of the coverage table and the (now source-qualified)
       bag size is taken and logged.

    When the source-qualified pool is STRICTLY LARGER than the reconciled
    ``n_cells``, the kept rows are a SEEDED random choice without replacement
    (``np.random.default_rng(seed)``) rather than the pool's first ``n_cells``
    rows -- bag row order is inherited from the h5ad and may be clustered
    (e.g. by ``gem_group``), so a fixed first-``n_cells`` truncation would
    bias which cells are used and make cross-seed variance artificially
    vanish. ``seed`` is the SAME seed that governs the panel's own downstream
    with-/without-replacement sampling, so this preselection varies across
    seeds but replays deterministically for a fixed seed. The returned
    bag/batch always have exactly ``n_cells`` rows -- truncated, not merely
    filtered -- so ``compute_window_budget(n_cells, ...)``'s ``bootstrapped``
    decision and ``_build_evaluation_control_panel``'s actual with-/without-
    replacement sampling (which compares its requested panel size against the
    size of the array it is handed) agree by construction.

    Args:
        gene: The a-perturbed basal gene symbol (for warning/log/error
            messages).
        raw_bag: The gene's full observed-cell bag, ``(n_raw, input_dim)``.
        raw_batch: The gene's per-cell batch/source label array aligned to
            ``raw_bag``, or ``None`` if the pool carries no batch/source labels.
        coverage_n_cells: The coverage table's cell count for this gene.
        seed: The seed used for the eligible-pool preselection, when the
            source-qualified pool exceeds ``n_cells`` -- the SAME seed that
            drives the panel's own downstream sampling.

    Returns:
        ``(eligible_bag, eligible_batch, n_cells)``: the bag (and batch array, if
        ``raw_batch`` was given) reduced to exactly ``n_cells`` rows, and the
        reconciled cell count used to compute the window budget.

    Raises:
        ValueError: If every source-tagged row in ``raw_batch`` is
            non-Replogle (no Replogle-source cells are available for this
            gene's a-perturbed basal).
    """
    restricted_bag, restricted_batch = raw_bag, raw_batch
    if raw_batch is not None:
        recognized_mask = _recognized_source_mask(raw_batch)
        if recognized_mask.any():
            replogle_mask = _replogle_source_mask(raw_batch)
            if not replogle_mask.any():
                raise ValueError(
                    f"{gene}: a-perturbed bag carries only non-Replogle source "
                    "cells (no Replogle-source cells are available for this "
                    "gene) -- an independent-N panel budgeted off the "
                    "Replogle-only coverage count cannot be built from foreign "
                    "(Adamson/Dixit) cells"
                )
            if not replogle_mask.all():
                _LOGGER.info(
                    "%s: merged bag carries a mixed Replogle/non-Replogle source tag; "
                    "restricting the independent-N eligible population to %d/%d "
                    "Replogle-source cells",
                    gene,
                    int(replogle_mask.sum()),
                    raw_batch.shape[0],
                )
                restricted_bag = raw_bag[replogle_mask]
                restricted_batch = raw_batch[replogle_mask]
    n_cells = _reconcile_observed_cell_count(
        gene, coverage_n_cells, int(restricted_bag.shape[0])
    )
    eligible_size = int(restricted_bag.shape[0])
    if eligible_size > n_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(eligible_size, size=n_cells, replace=False)
        restricted_bag = restricted_bag[keep]
        if restricted_batch is not None:
            restricted_batch = restricted_batch[keep]
    eligible_batch = (
        restricted_batch[:n_cells] if restricted_batch is not None else None
    )
    return restricted_bag[:n_cells], eligible_batch, n_cells


def independent_perturbed_panel(
    context: BridgeAContext,
    gene: str,
    coverage: dict[str, int],
    spec: IndependentPanelSpec,
) -> tuple[_EvaluationControlPanel, WindowBudget]:
    """Build the OBSERVED ``gene``-perturbed basal panel at its independent-N budget.

    Args:
        context: A loaded ``BridgeAContext``.
        gene: The a-perturbed basal gene symbol.
        coverage: ``canonical_symbol -> observed cell count``
            (``bridge_a_panels.load_k562_gene_coverage``).
        spec: Shared window sizing (``cell_set_len``, ``max_windows``, ...).

    Returns:
        The built panel and the ``WindowBudget`` that sized it (using the
        RECONCILED, source-qualified cell count -- see module docstring and
        ``_eligible_perturbed_rows`` -- when the coverage table and the
        observed bag disagree, or the bag mixes Replogle and non-Replogle
        supplement cells).

    Raises:
        KeyError: If ``gene`` is not in the exp05 fixed pool or the coverage
            table.
        ValueError: If ``gene``'s merged bag is entirely non-Replogle source
            (no Replogle-source cells available for the a-perturbed basal --
            see ``_eligible_perturbed_rows``).
        RuntimeError: If the panel does not reach the computed window budget.
    """
    upper = str(gene).upper()
    if upper not in context.gene_to_index:
        raise KeyError(f"gene not in the exp05 fixed pool: {gene!r}")
    if upper not in coverage:
        raise KeyError(f"gene {upper} missing from the k562 gene coverage table")

    index = context.gene_to_index[upper]
    raw_bag = np.asarray(context.data.input_bags[index], dtype=np.float32)
    raw_batch = (
        context.data.batch_bags[index] if context.data.batch_bags is not None else None
    )
    eligible_bag, eligible_batch, n_cells = _eligible_perturbed_rows(
        upper, raw_bag, raw_batch, coverage[upper], spec.seed
    )
    budget = compute_window_budget(n_cells, spec.cell_set_len, spec.max_windows)
    basal_view = replace(
        context.data, control_input=eligible_bag, control_batch=eligible_batch
    )
    panel = _build_evaluation_control_panel(
        basal_view,
        spec.cell_set_len,
        budget.n_windows * spec.cell_set_len,
        spec.macro_batch_windows,
        spec.seed,
        context.device,
        context.batch_lookup,
    )
    if panel.n_windows != budget.n_windows:
        raise RuntimeError(
            f"{upper}: independent-N panel invariant violated: expected "
            f"{budget.n_windows} windows, got {panel.n_windows}"
        )
    return panel, budget


def control_panel_for_window_count(
    context: BridgeAContext,
    n_windows: int,
    spec: IndependentPanelSpec,
) -> _EvaluationControlPanel:
    """Build the pooled-control panel matched to ``n_windows`` (cache key).

    Drawn WITHOUT replacement from the (large) pooled non-targeting control
    bag -- ``n_windows * spec.cell_set_len`` is always far below the pool
    size in practice, so this never hits the bootstrap regime.

    Raises:
        RuntimeError: If the built panel does not reach ``n_windows``.
    """
    panel = _build_evaluation_control_panel(
        context.data,
        spec.cell_set_len,
        n_windows * spec.cell_set_len,
        spec.macro_batch_windows,
        spec.seed,
        context.device,
        context.batch_lookup,
    )
    if panel.n_windows != n_windows:
        raise RuntimeError(
            f"control panel invariant violated: expected {n_windows} windows, "
            f"got {panel.n_windows} (pooled control bag too small?)"
        )
    return panel


def compute_independent_c_hat_table(
    context: BridgeAContext,
    genes: Sequence[str],
    coverage: dict[str, int],
    spec: IndependentPanelSpec,
    reference_convention: str = "self",
) -> tuple[pd.DataFrame, dict[str, WindowBudget]]:
    """Compute ``c_hat[query | basal]`` for every (basal, query) pair over ``genes``.

    Basal states are each gene in ``genes`` as an OBSERVED a-perturbed basal
    (one independent-N-matched panel per gene, own window budget) and
    ``"control"`` at every window count actually required (cached by
    ``(query_gene, n_windows)`` -- see module docstring).

    Args:
        context: A loaded ``BridgeAContext``.
        genes: Query/basal gene symbols; each must be in the exp05 fixed pool
            and the coverage table.
        coverage: ``canonical_symbol -> observed cell count``.
        spec: Shared independent-N window sizing for this call.
        reference_convention: ``"self"`` or ``"control"`` -- which pooler
            reference latent the a-perturbed arm uses (Finding 1). The
            control arm always uses its own pooled-control latent.

    Returns:
        A tuple of the long-format frame (columns ``basal_state``,
        ``query_gene``, ``c_hat``, ``reference_convention``,
        ``forward_seconds``, ``n_windows``) and the per-gene
        ``WindowBudget`` map.

    Raises:
        ValueError: If ``reference_convention`` is not recognized.
    """
    if reference_convention not in REFERENCE_CONVENTIONS:
        raise ValueError(
            f"unknown reference_convention {reference_convention!r}; expected "
            f"one of {REFERENCE_CONVENTIONS}"
        )
    genes = [str(gene).upper() for gene in genes]
    budgets: dict[str, WindowBudget] = {}
    rows: list[dict[str, object]] = []
    control_panels: dict[int, tuple[_EvaluationControlPanel, torch.Tensor]] = {}
    control_cached: set[tuple[str, int]] = set()

    with torch.no_grad():
        for basal_gene in genes:
            panel, budget = independent_perturbed_panel(
                context, basal_gene, coverage, spec
            )
            budgets[basal_gene] = budget
            basal_latent = panel_latent(context.model, panel)
            n_windows = budget.n_windows

            if n_windows not in control_panels:
                c_panel = control_panel_for_window_count(context, n_windows, spec)
                c_latent = panel_latent(context.model, c_panel)
                control_panels[n_windows] = (c_panel, c_latent)
            c_panel, control_latent_w = control_panels[n_windows]

            ref_latent = reference_latent(
                reference_convention, control_latent_w, basal_latent
            )
            for query_gene in genes:
                if query_gene == basal_gene:
                    continue
                start = time.perf_counter()
                c_hat = predict_c_hat(context.model, panel, ref_latent, query_gene)
                rows.append(
                    {
                        "basal_state": basal_gene,
                        "query_gene": query_gene,
                        "c_hat": c_hat,
                        "reference_convention": reference_convention,
                        "forward_seconds": time.perf_counter() - start,
                        "n_windows": n_windows,
                    }
                )
                cache_key = (query_gene, n_windows)
                if cache_key not in control_cached:
                    control_cached.add(cache_key)
                    c_start = time.perf_counter()
                    control_c_hat = predict_c_hat(
                        context.model, c_panel, control_latent_w, query_gene
                    )
                    rows.append(
                        {
                            "basal_state": CONTROL_BASAL_STATE,
                            "query_gene": query_gene,
                            "c_hat": control_c_hat,
                            "reference_convention": reference_convention,
                            "forward_seconds": time.perf_counter() - c_start,
                            "n_windows": n_windows,
                        }
                    )
            _LOGGER.info(
                "independent-N c_hat[gene | %s-perturbed] for %d query genes "
                "(n_windows=%d, effective_n=%d, bootstrapped=%s, "
                "reference_convention=%s)",
                basal_gene,
                len(genes) - 1,
                n_windows,
                budget.effective_n,
                budget.bootstrapped,
                reference_convention,
            )
    return pd.DataFrame(rows), budgets


def bridge_a_pairs_from_independent_c_hat_table(
    c_hat_table: pd.DataFrame,
    budgets: dict[str, WindowBudget],
    cell_set_len: int,
) -> pd.DataFrame:
    """Fold an independent-N long ``c_hat`` table into per-pair ``s_A`` plus provenance.

    Args:
        c_hat_table: Output of ``compute_independent_c_hat_table``.
        budgets: The per-gene ``WindowBudget`` map from the same call.
        cell_set_len: Cells per STATE window (for the ``high_effective_n``
            flag: both genes need >= one independent window of observed
            cells).

    Returns:
        One row per unordered gene pair with both directed deltas, ``s_a``,
        and per-direction ``n_windows_*``, ``effective_n_*``,
        ``bootstrapped_*``, plus the pair-level ``high_effective_n`` flag
        (the PRIMARY analysis slice: both genes have
        ``n_cells >= cell_set_len``). Always contains EXACTLY
        ``C(len(budgets), 2)`` rows -- the complete clique over ``budgets``'
        genes -- never a partial pair set (FAILS CLOSED instead).

    Raises:
        RuntimeError: If either directed prediction (a->b or b->a) is missing
            for any pair, if a required control c_hat is missing from the
            table (the table was not produced by
            ``compute_independent_c_hat_table`` over the same gene set), or
            if the assembled pair table does not contain exactly
            ``C(len(budgets), 2)`` unique unordered pairs.
    """
    perturbed = c_hat_table.loc[c_hat_table["basal_state"] != CONTROL_BASAL_STATE]
    perturbed_lookup = {
        (str(row.basal_state), str(row.query_gene)): float(row.c_hat)
        for row in perturbed.itertuples(index=False)
    }
    control = c_hat_table.loc[c_hat_table["basal_state"] == CONTROL_BASAL_STATE]
    control_lookup = {
        (str(row.query_gene), int(row.n_windows)): float(row.c_hat)
        for row in control.itertuples(index=False)
    }

    genes = sorted(budgets)
    expected_pair_count = len(genes) * (len(genes) - 1) // 2
    rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, gene_a in enumerate(genes):
        for gene_b in genes[i + 1 :]:
            c_b_given_a = perturbed_lookup.get((gene_a, gene_b))
            c_a_given_b = perturbed_lookup.get((gene_b, gene_a))
            if c_b_given_a is None or c_a_given_b is None:
                missing = [
                    direction
                    for direction, value in (
                        (f"{gene_a}->{gene_b}", c_b_given_a),
                        (f"{gene_b}->{gene_a}", c_a_given_b),
                    )
                    if value is None
                ]
                raise RuntimeError(
                    f"pair ({gene_a}, {gene_b}) is missing directed prediction(s) "
                    f"{missing}; bridge_a_pairs_from_independent_c_hat_table "
                    "requires BOTH directions for every pair to guarantee the "
                    "complete-clique output -- compute_independent_c_hat_table "
                    "must be called with the SAME complete gene set"
                )
            seen_pairs.add((gene_a, gene_b))
            budget_a = budgets[gene_a]
            budget_b = budgets[gene_b]
            c_b_given_control = control_lookup.get((gene_b, budget_a.n_windows))
            c_a_given_control = control_lookup.get((gene_a, budget_b.n_windows))
            if c_b_given_control is None or c_a_given_control is None:
                raise RuntimeError(
                    f"missing cached control c_hat for pair ({gene_a}, {gene_b}); "
                    "compute_independent_c_hat_table must be called with both "
                    "genes present in the SAME run"
                )
            delta_ab = delta_a_to_b(c_b_given_control, c_b_given_a)
            delta_ba = delta_a_to_b(c_a_given_control, c_a_given_b)
            rows.append(
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "c_hat_b_given_control": c_b_given_control,
                    "c_hat_b_given_a_perturbed": c_b_given_a,
                    "c_hat_a_given_control": c_a_given_control,
                    "c_hat_a_given_b_perturbed": c_a_given_b,
                    "delta_a_to_b": delta_ab,
                    "delta_b_to_a": delta_ba,
                    "s_a": symmetrized_codependency(delta_ab, delta_ba),
                    "n_windows_a_to_b": budget_a.n_windows,
                    "n_windows_b_to_a": budget_b.n_windows,
                    "effective_n_a_to_b": budget_a.effective_n,
                    "effective_n_b_to_a": budget_b.effective_n,
                    "bootstrapped_a_to_b": budget_a.bootstrapped,
                    "bootstrapped_b_to_a": budget_b.bootstrapped,
                    "n_cells_a": budget_a.n_cells,
                    "n_cells_b": budget_b.n_cells,
                    "high_effective_n": (
                        budget_a.n_cells >= cell_set_len
                        and budget_b.n_cells >= cell_set_len
                    ),
                }
            )
    if len(seen_pairs) != expected_pair_count or len(rows) != expected_pair_count:
        raise RuntimeError(
            f"assembled {len(rows)} pair rows ({len(seen_pairs)} unique) but "
            f"expected exactly the complete clique C({len(genes)}, 2) = "
            f"{expected_pair_count} for {len(genes)} genes"
        )
    return pd.DataFrame(rows)


def validate_seed_replicate_tables(
    tables: Sequence[pd.DataFrame],
    seeds: Sequence[int],
    key_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    """Enforce the Finding-C seed-replicate cardinality/completeness guards.

    Shared by ``bridge_a_gates.aggregate_seed_replicate_pairs`` and
    ``bridge_a_gates.aggregate_seed_replicate_c_hat``: >= 2 DISTINCT requested
    seeds, exactly one table per requested seed carrying exactly that seed, no duplicate
    ``key_columns`` rows within one replicate, and an IDENTICAL
    ``key_columns`` key set across every replicate -- checked BEFORE any
    aggregation, so an incomplete or duplicated replicate set cannot silently
    read as a legitimate (or zero-variance) result.

    Args:
        tables: One table per requested seed, in the SAME order as ``seeds``.
        seeds: The seeds actually requested -- >= 2 DISTINCT values.
        key_columns: The columns that jointly define one aggregation unit.
        label: A short noun for error messages (e.g. ``"pair table"``).

    Returns:
        The seed-tagged ``tables`` concatenated into one frame.

    Raises:
        ValueError: On any cardinality or completeness violation.
    """
    requested_seeds = list(seeds)
    if len(requested_seeds) < 2:
        raise ValueError(
            f"{label} aggregation requires >= 2 distinct seeds, got {requested_seeds}"
        )
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError(
            f"{label} aggregation requires DISTINCT seeds, got duplicates in "
            f"{requested_seeds}"
        )
    if len(tables) != len(requested_seeds):
        raise ValueError(
            f"expected exactly one {label} per requested seed: {len(tables)} "
            f"tables for {len(requested_seeds)} seeds {requested_seeds}"
        )

    column_desc = ", ".join(key_columns)
    key_sets: list[frozenset[tuple[object, ...]]] = []
    for table, expected_seed in zip(tables, requested_seeds, strict=True):
        if "seed" not in table.columns:
            raise ValueError(f"each {label} must carry a 'seed' column")
        observed = table["seed"].unique()
        if len(observed) != 1 or int(observed[0]) != int(expected_seed):
            raise ValueError(
                f"{label} for requested seed {expected_seed} does not carry "
                f"exactly that single seed (observed {observed.tolist()})"
            )
        keys = list(zip(*(table[column] for column in key_columns), strict=True))
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"seed {expected_seed}: duplicate ({column_desc}) rows within "
                f"one {label} replicate"
            )
        key_sets.append(frozenset(keys))

    first_seed, first_keys = requested_seeds[0], key_sets[0]
    for seed_value, keys in zip(requested_seeds[1:], key_sets[1:], strict=True):
        if keys != first_keys:
            raise ValueError(
                f"seed {seed_value} covers a different set of ({column_desc}) "
                f"keys than seed {first_seed} -- replicates are not directly "
                "comparable"
            )
    return pd.concat(tables, ignore_index=True)

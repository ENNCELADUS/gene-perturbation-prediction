"""Tests for Bridge A: pure delta/symmetrization math, the independent-N
window-budget policy (Finding A), the strengthened perturbed-arm gate
(Finding B), and seed-replicate aggregation completeness guards (Finding C).
Exercised on a tiny, deterministic, CPU-only synthetic ``BridgeAContext`` --
no HPC checkpoint, GWPS data, or GPU required.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

import aivc_model.bridge_a_gates as bridge_a_gates
from aivc_model.bridge_a import (
    CONTROL_BASAL_STATE,
    BridgeAContext,
    delta_a_to_b,
    symmetrized_codependency,
)
from aivc_model.bridge_a_gates import (
    aggregate_seed_replicate_c_hat,
    aggregate_seed_replicate_pairs,
    assert_perturbed_arm_integrity,
    project_full_sweep,
)
from aivc_model.bridge_a_independent import (
    IndependentPanelSpec,
    _eligible_perturbed_rows,
    bridge_a_pairs_from_independent_c_hat_table,
    compute_independent_c_hat_table,
    control_panel_for_window_count,
    independent_perturbed_panel,
)
from aivc_model.bridge_a_panels import (
    WindowBudget,
    compute_window_budget,
    load_k562_gene_coverage,
)
from aivc_model.model import (
    AivcModel,
    MLPHead,
    PerturbationVectorAdapter,
    StateForwardAdapter,
    load_state_model,
)
from aivc_model.prepare import GeneBags
from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM


def test_delta_a_to_b_is_control_minus_perturbed() -> None:
    # b looks more essential (more negative GeneEffect) once a is lost:
    # a positive delta is a co-dependency spike.
    assert delta_a_to_b(-2.0, -3.0) == pytest.approx(1.0)


def test_delta_a_to_b_can_be_negative() -> None:
    assert delta_a_to_b(-2.0, -1.0) == pytest.approx(-1.0)


def test_symmetrized_codependency_is_the_mean_of_both_directions() -> None:
    assert symmetrized_codependency(1.0, 0.5) == pytest.approx(0.75)


def test_symmetrized_codependency_is_order_independent() -> None:
    assert symmetrized_codependency(1.0, 0.5) == symmetrized_codependency(0.5, 1.0)


# --- Finding A: pure window-budget policy (bridge_a_panels) -----------------


def test_compute_window_budget_without_replacement_regime() -> None:
    budget = compute_window_budget(n_cells=200, cell_set_len=64, max_windows=8)
    assert budget.n_windows == 3  # 200 // 64 == 3, under the cap
    assert budget.bootstrapped is False
    assert budget.effective_n == 192


def test_compute_window_budget_caps_at_max_windows() -> None:
    budget = compute_window_budget(n_cells=1963, cell_set_len=64, max_windows=8)
    assert budget.n_windows == 8
    assert budget.bootstrapped is False
    assert budget.effective_n == 512


def test_compute_window_budget_bootstrap_fallback_below_one_window() -> None:
    budget = compute_window_budget(n_cells=10, cell_set_len=64, max_windows=8)
    assert budget.n_windows == 1
    assert budget.bootstrapped is True
    assert budget.effective_n == 10


@pytest.mark.parametrize(
    ("n_cells", "cell_set_len", "max_windows"),
    [(-1, 64, 8), (10, 0, 8), (10, 64, 0)],
)
def test_compute_window_budget_rejects_invalid_inputs(
    n_cells: int, cell_set_len: int, max_windows: int
) -> None:
    with pytest.raises(ValueError):
        compute_window_budget(n_cells, cell_set_len, max_windows)


def test_load_k562_gene_coverage_filters_to_pool_and_covered(tmp_path) -> None:
    csv_path = tmp_path / "coverage.csv"
    pd.DataFrame(
        [
            {
                "canonical_symbol": "aars2",
                "replogle_n_cells": 125,
                "in_exp05_fixed_pool": True,
                "exp05_observed_covered": True,
            },
            {
                "canonical_symbol": "excl_not_covered",
                "replogle_n_cells": 50,
                "in_exp05_fixed_pool": True,
                "exp05_observed_covered": False,
            },
            {
                "canonical_symbol": "excl_not_in_pool",
                "replogle_n_cells": 50,
                "in_exp05_fixed_pool": False,
                "exp05_observed_covered": True,
            },
        ]
    ).to_csv(csv_path, index=False)
    coverage = load_k562_gene_coverage(csv_path)
    assert coverage == {"AARS2": 125}


# --- Finding A/1/B: tiny synthetic BridgeAContext (CPU, no HPC/GPU) ---------


def _toy_context(
    *,
    gene_cells: dict[str, int] | None = None,
    input_dim: int = 8,
    control_cells: int = 200,
    gene_source_labels: dict[str, list[str]] | None = None,
) -> BridgeAContext:
    """A tiny, deterministic ``BridgeAContext``.

    Default gene cell counts span both independent-N regimes (cell_set_len=2
    below): ``TINY`` (1 cell) triggers the sub-window bootstrap fallback;
    ``SMALL`` (6 cells) gets an UNCAPPED independent-N budget (3 windows);
    ``BIG``/``QUERY1``/``QUERY2`` (40 cells each) all get CAPPED at
    ``max_windows`` -- a shared window count that exercises the
    control-panel cache. ``latent_dim=4``: ``LayerNorm`` over a 2-element
    vector always collapses to exactly ``[-1, 1]`` regardless of its input,
    which would make ``ResponseEncoder`` output gene-INDEPENDENT and defeat
    the "non-constant across query genes" gate check -- an artifact of a
    too-small test fixture, not of the Bridge A code under test.

    Args:
        gene_source_labels: optional ``gene -> per-cell batch/source label``
            override (length must equal that gene's cell count), for
            exercising the ``merge_gene_bag_pool``-style
            ``"replogle:..."``/``"non_replogle:..."`` source tagging (Finding
            A's inverse-mismatch fix). When given, ``GeneBags.batch_bags`` is
            populated for every gene, defaulting an unlisted gene to a
            uniform ``"replogle"`` label (single-source, i.e. unrestricted).
            When ``None`` (default), ``batch_bags`` stays ``None`` -- no
            source tags at all, matching every other (pre-existing) test.
    """
    latent_dim = 4
    counts = gene_cells or {
        "TINY": 1,
        "SMALL": 6,
        "BIG": 40,
        "QUERY1": 40,
        "QUERY2": 40,
    }
    genes = list(counts)
    rng = np.random.default_rng(0)

    def _bag(n_cells: int, offset: float) -> np.ndarray:
        return (rng.standard_normal((n_cells, input_dim)) + offset).astype(np.float32)

    input_bags = tuple(
        _bag(counts[gene], 10.0 * (index + 1)) for index, gene in enumerate(genes)
    )
    control_input = _bag(control_cells, 0.0)
    batch_bags = None
    if gene_source_labels is not None:
        batch_bags = tuple(
            np.asarray(
                gene_source_labels.get(gene, ["replogle"] * counts[gene]),
                dtype=object,
            )
            for gene in genes
        )
    data = GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=np.asarray([-1.0] * len(genes), dtype=np.float32),
        input_bags=input_bags,
        latent_bags=tuple(bag[:, :latent_dim] for bag in input_bags),
        control_input=control_input,
        control_latent=control_input[:, :latent_dim],
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=batch_bags,
        control_batch=None,
        feature_names=None,
        metadata=pd.DataFrame(),
        input_dim=input_dim,
        latent_dim=latent_dim,
    )
    # Distinct, fixed known vectors per gene: PerturbationVectorAdapter gives
    # every gene WITHOUT a known vector the same shared initial value, which
    # would make the mock STATE model's output gene-INDEPENDENT at init and
    # defeat the "non-constant across query genes" gate check below.
    known_vectors = {gene: rng.standard_normal(2).astype(np.float32) for gene in genes}
    perturbations = PerturbationVectorAdapter(genes, known_vectors, pert_dim=2)
    state_model = load_state_model(
        backend="linear_mock",
        checkpoint_path=None,
        input_dim=input_dim,
        output_dim=input_dim,
        pert_dim=2,
    )
    response_pooler = TrainableDiagonalGMM(
        latent_dim=latent_dim, n_components=2, covariance_floor=1e-4, init_scale=0.02
    )
    model = AivcModel(
        state_adapter=StateForwardAdapter(state_model),
        perturbations=perturbations,
        response_encoder=ResponseEncoder(input_dim=input_dim, latent_dim=latent_dim),
        response_pooler=response_pooler,
        c_head=MLPHead(response_pooler.output_dim, (8,), 0.0),
        control_expression_mean=np.zeros(input_dim, dtype=np.float32),
    )
    model.eval()
    model.requires_grad_(False)
    return BridgeAContext(
        model=model,
        data=data,
        batch_lookup={},
        config=None,  # type: ignore[arg-type]  # unused by the functions under test
        eval_seed=41,
        device=torch.device("cpu"),
        gene_to_index={gene: index for index, gene in enumerate(genes)},
    )


def _toy_coverage(context: BridgeAContext) -> dict[str, int]:
    """A coverage dict matching the toy context's actual raw bag sizes exactly."""
    return {
        str(gene): int(np.asarray(context.data.input_bags[index]).shape[0])
        for index, gene in enumerate(context.data.genes)
    }


_INDEPENDENT_SPEC = IndependentPanelSpec(
    cell_set_len=2, macro_batch_windows=4, seed=7, max_windows=4
)


# --- Finding A: independent-N panel construction ----------------------------


def test_independent_perturbed_panel_bootstraps_when_below_one_window() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    panel, budget = independent_perturbed_panel(
        context, "TINY", coverage, _INDEPENDENT_SPEC
    )
    assert budget.bootstrapped is True
    assert budget.n_windows == 1
    assert budget.effective_n == 1
    assert panel.n_windows == 1
    raw = np.asarray(
        context.data.input_bags[context.gene_to_index["TINY"]], dtype=np.float32
    )
    observed = torch.cat(panel.control_chunks, dim=0).numpy()
    assert observed.shape[0] == 2  # one window == cell_set_len cells
    assert all(
        np.array_equal(row, raw[0]) for row in observed
    )  # only 1 cell to draw from


def test_independent_perturbed_panel_uncapped_is_without_replacement() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    panel, budget = independent_perturbed_panel(
        context, "SMALL", coverage, _INDEPENDENT_SPEC
    )
    assert budget.bootstrapped is False
    assert budget.n_windows == 3  # 6 // 2 == 3, under max_windows=4
    assert budget.effective_n == 6
    observed = torch.cat(panel.control_chunks, dim=0).numpy()
    assert observed.shape[0] == 6
    assert len({row.tobytes() for row in observed}) == 6  # no duplication


def test_independent_perturbed_panel_caps_at_max_windows() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    panel, budget = independent_perturbed_panel(
        context, "BIG", coverage, _INDEPENDENT_SPEC
    )
    assert budget.bootstrapped is False
    assert budget.n_windows == 4  # capped, not 40 // 2 == 20
    assert budget.effective_n == 8
    assert panel.n_windows == 4


def test_independent_perturbed_panel_only_reuses_this_genes_observed_rows() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    raw = np.asarray(
        context.data.input_bags[context.gene_to_index["SMALL"]], dtype=np.float32
    )
    raw_rows = {row.tobytes() for row in raw}
    panel, _budget = independent_perturbed_panel(
        context, "SMALL", coverage, _INDEPENDENT_SPEC
    )
    observed = torch.cat(panel.control_chunks, dim=0).numpy()
    assert all(row.tobytes() in raw_rows for row in observed)


def test_independent_perturbed_panel_is_deterministic_given_seed() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    first, budget_first = independent_perturbed_panel(
        context, "SMALL", coverage, _INDEPENDENT_SPEC
    )
    second, budget_second = independent_perturbed_panel(
        context, "SMALL", coverage, _INDEPENDENT_SPEC
    )
    assert budget_first == budget_second
    assert np.array_equal(first.selected_indices, second.selected_indices)
    torch.testing.assert_close(
        torch.cat(first.control_chunks), torch.cat(second.control_chunks)
    )


def test_independent_perturbed_panel_rejects_two_view_gene_bags() -> None:
    """Bridge A is paused and has no two-view config (docs/04-roadmap.md Sec
    1.1 T1). The panel below only overrides ``control_input``/``control_batch``
    with the gene-specific eligible rows; a populated target view would keep
    pointing at the ORIGINAL, unfiltered basal cells, silently describing two
    different cell sets. This must fail closed rather than guess which rows
    the target view should have followed.
    """
    context = _toy_context()
    coverage = _toy_coverage(context)
    two_view_data = replace(
        context.data,
        target_bags=context.data.input_bags,
        control_target=context.data.control_input,
        target_dim=context.data.input_dim,
    )
    two_view_context = replace(context, data=two_view_data)
    with pytest.raises(ValueError, match="two-space GeneBags"):
        independent_perturbed_panel(
            two_view_context, "SMALL", coverage, _INDEPENDENT_SPEC
        )


def test_independent_perturbed_panel_single_view_bag_still_works() -> None:
    """Regression guard for the two-view guard above: the ordinary
    single-view path (``target_bags is None``, exercised by every other test
    in this module) must be completely unaffected."""
    context = _toy_context()
    coverage = _toy_coverage(context)
    panel, budget = independent_perturbed_panel(
        context, "SMALL", coverage, _INDEPENDENT_SPEC
    )
    assert panel.n_windows == budget.n_windows


def test_independent_perturbed_panel_raises_keyerror_for_unknown_gene() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    with pytest.raises(KeyError):
        independent_perturbed_panel(context, "NOPE", coverage, _INDEPENDENT_SPEC)


def test_independent_perturbed_panel_raises_keyerror_when_missing_from_coverage() -> (
    None
):
    context = _toy_context()
    coverage = _toy_coverage(context)
    del coverage["SMALL"]
    with pytest.raises(KeyError):
        independent_perturbed_panel(context, "SMALL", coverage, _INDEPENDENT_SPEC)


def test_independent_perturbed_panel_reconciles_coverage_larger_than_observed_bag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The coverage table and the exp05 pool's observed bag disagree for a
    real slice of genes (see module docstring); the panel must use the
    SMALLER, conservative count -- never request more independent cells than
    are actually present -- and log a warning rather than crash or silently
    trust the (larger) coverage-table value.
    """
    context = _toy_context()
    coverage = _toy_coverage(context)
    coverage["SMALL"] = 100  # claims far more cells than the actual 6-cell raw bag
    with caplog.at_level("WARNING"):
        panel, budget = independent_perturbed_panel(
            context, "SMALL", coverage, _INDEPENDENT_SPEC
        )
    assert budget.n_cells == 6  # reconciled down to the observed bag size
    assert budget.n_windows == 3
    assert panel.n_windows == 3
    assert any(
        record.levelname == "WARNING" and "SMALL" in record.message
        for record in caplog.records
    )


def test_independent_perturbed_panel_reconciles_when_coverage_is_smaller() -> None:
    """The metadata (``bootstrapped=True``) must match the ACTUAL sampling, not
    just report a small ``n_cells`` while secretly drawing WITHOUT replacement
    from the full 40-cell raw bag (the bug FIX 1 addresses): every one of the
    ``cell_set_len`` cells in the single bootstrap window must be the SAME
    single reconciled-eligible row, proving with-replacement sampling from a
    pool of exactly 1, not from the full bag. Which single row that is is a
    FIX-B seeded choice (not necessarily the bag's first row), so this only
    asserts self-consistency (one row, drawn from the raw bag) rather than a
    specific row identity.
    """
    context = _toy_context()
    coverage = _toy_coverage(context)
    coverage["BIG"] = 1  # claims far fewer cells than the actual 40-cell raw bag
    panel, budget = independent_perturbed_panel(
        context, "BIG", coverage, _INDEPENDENT_SPEC
    )
    assert budget.n_cells == 1  # reconciled down to the (smaller) coverage value
    assert budget.bootstrapped is True
    assert panel.n_windows == 1
    raw = np.asarray(context.data.input_bags[context.gene_to_index["BIG"]])
    raw_rows = {row.tobytes() for row in raw}
    observed = torch.cat(panel.control_chunks, dim=0).numpy()
    observed_rows = {row.tobytes() for row in observed}
    assert len(observed_rows) == 1  # every draw is the SAME single eligible row
    assert observed_rows.issubset(raw_rows)  # drawn from the raw bag, not fabricated


def test_independent_perturbed_panel_restricts_to_replogle_source_when_mixed() -> None:
    """Inverse mismatch (coverage < bag_size), WITHOUT-replacement regime: a
    gene whose merged bag mixes 6 Replogle-source cells with 4 non-Replogle
    supplement cells (``merge_gene_bag_pool``'s source tagging) must draw its
    independent-N panel ONLY from the 6 Replogle-source cells, never the 4
    foreign ones -- even though the raw bag (10 cells) is well above one
    window and would otherwise pass the without-replacement path unrestricted.
    """
    labels = {"MIXED": ["replogle:a"] * 6 + ["non_replogle:adamson_pilot:b"] * 4}
    context = _toy_context(gene_cells={"MIXED": 10}, gene_source_labels=labels)
    coverage = {"MIXED": 6}  # matches the Replogle-only count exactly
    panel, budget = independent_perturbed_panel(
        context, "MIXED", coverage, _INDEPENDENT_SPEC
    )
    assert budget.n_cells == 6
    assert budget.bootstrapped is False  # 6 eligible cells >= cell_set_len=2
    assert budget.n_windows == 3  # 6 // 2, uncapped
    raw = np.asarray(context.data.input_bags[context.gene_to_index["MIXED"]])
    replogle_rows = {row.tobytes() for row in raw[:6]}
    non_replogle_rows = {row.tobytes() for row in raw[6:]}
    observed = torch.cat(panel.control_chunks, dim=0).numpy()
    assert observed.shape[0] == 6
    assert all(row.tobytes() in replogle_rows for row in observed)
    assert not any(row.tobytes() in non_replogle_rows for row in observed)


def test_independent_perturbed_panel_bootstrap_reflects_source_qualified_pool() -> None:
    """Inverse mismatch, WITH-replacement (bootstrap) regime -- the sharpest
    regression case: a gene has only 1 Replogle-source cell among 9
    non-Replogle supplement cells (10 total). The PRE-FIX code would have
    reconciled ``n_cells = min(coverage, raw_bag_size) = min(50, 10) = 10``
    (not bootstrapped, 4 windows) and sampled WITHOUT replacement from the
    full 10-cell bag -- drawing mostly foreign cells. The FIX must restrict
    to the single Replogle-source cell FIRST, so the reconciled pool is size
    1, correctly triggering the bootstrap (with-replacement) regime, and
    every sampled cell must equal that single eligible row.
    """
    labels = {"MIXEDSMALL": ["replogle:a"] * 1 + ["non_replogle:dixit_2016:b"] * 9}
    context = _toy_context(gene_cells={"MIXEDSMALL": 10}, gene_source_labels=labels)
    coverage = {"MIXEDSMALL": 50}  # far exceeds either the raw bag or the eligible pool
    panel, budget = independent_perturbed_panel(
        context, "MIXEDSMALL", coverage, _INDEPENDENT_SPEC
    )
    assert budget.n_cells == 1  # reconciled against the RESTRICTED pool, not raw_bag=10
    assert budget.bootstrapped is True
    assert budget.n_windows == 1
    raw = np.asarray(context.data.input_bags[context.gene_to_index["MIXEDSMALL"]])
    observed = torch.cat(panel.control_chunks, dim=0).numpy()
    non_replogle_rows = {row.tobytes() for row in raw[1:]}
    assert all(np.array_equal(row, raw[0]) for row in observed)  # only eligible cell
    assert not any(row.tobytes() in non_replogle_rows for row in observed)


def test_independent_perturbed_panel_raises_when_bag_is_entirely_non_replogle() -> None:
    """FIX A: a gene whose ENTIRE merged bag is non-Replogle source (zero
    Replogle rows among the recognized-prefix tags) must FAIL CLOSED --
    never silently build the a-perturbed panel from foreign (Adamson/Dixit)
    cells budgeted off the Replogle-only coverage count. The 408-gene sweep
    universe never hits this case (every universe gene has >= 8 Replogle
    cells), but it must still fail closed rather than substitute the wrong
    modality.
    """
    labels = {"ALLFOREIGN": ["non_replogle:adamson_pilot:x"] * 10}
    context = _toy_context(gene_cells={"ALLFOREIGN": 10}, gene_source_labels=labels)
    coverage = {"ALLFOREIGN": 6}
    with pytest.raises(ValueError, match="ALLFOREIGN") as excinfo:
        independent_perturbed_panel(context, "ALLFOREIGN", coverage, _INDEPENDENT_SPEC)
    assert "Replogle" in str(excinfo.value)


# --- Finding A (FIX B): seeded eligible-pool preselection -------------------


def test_eligible_perturbed_rows_seeded_preselection_varies_by_seed() -> None:
    """FIX B: when the source-qualified eligible pool is larger than the
    reconciled ``n_cells``, WHICH rows are kept must be a seeded random
    choice, not the pool's first ``n_cells`` rows -- different seeds must
    select different rows (bag row order is inherited from the h5ad and may
    be clustered by ``gem_group``, so a fixed truncation would bias which
    cells are used and hide genuine cross-seed variance).
    """
    context = _toy_context()
    raw_bag = np.asarray(
        context.data.input_bags[context.gene_to_index["BIG"]], dtype=np.float32
    )
    bag_seed7, _batch7, n_cells7 = _eligible_perturbed_rows(
        "BIG", raw_bag, None, coverage_n_cells=10, seed=7
    )
    bag_seed99, _batch99, n_cells99 = _eligible_perturbed_rows(
        "BIG", raw_bag, None, coverage_n_cells=10, seed=99
    )
    assert n_cells7 == n_cells99 == 10
    rows_seed7 = {row.tobytes() for row in bag_seed7}
    rows_seed99 = {row.tobytes() for row in bag_seed99}
    assert rows_seed7 != rows_seed99


def test_eligible_perturbed_rows_seeded_preselection_replays_deterministically() -> (
    None
):
    """FIX B: replaying the SAME seed must select the identical rows, in the
    identical order, every time.
    """
    context = _toy_context()
    raw_bag = np.asarray(
        context.data.input_bags[context.gene_to_index["BIG"]], dtype=np.float32
    )
    bag_first, batch_first, n_first = _eligible_perturbed_rows(
        "BIG", raw_bag, None, coverage_n_cells=10, seed=7
    )
    bag_second, batch_second, n_second = _eligible_perturbed_rows(
        "BIG", raw_bag, None, coverage_n_cells=10, seed=7
    )
    assert n_first == n_second == 10
    assert batch_first is None
    assert batch_second is None
    np.testing.assert_array_equal(bag_first, bag_second)


def test_eligible_perturbed_rows_no_truncation_when_coverage_equals_eligible() -> None:
    """When the reconciled ``n_cells`` equals the (already source-qualified)
    eligible pool size exactly, there is nothing to preselect -- the seeded
    choice must NOT fire, and the full bag comes back unchanged and in its
    original row order (this is the FIX-B-unaffected case).
    """
    context = _toy_context()
    raw_bag = np.asarray(
        context.data.input_bags[context.gene_to_index["BIG"]], dtype=np.float32
    )
    bag, batch, n_cells = _eligible_perturbed_rows(
        "BIG", raw_bag, None, coverage_n_cells=40, seed=7
    )
    assert n_cells == 40
    assert batch is None
    np.testing.assert_array_equal(bag, raw_bag)


def test_control_panel_for_window_count_matches_requested_windows() -> None:
    context = _toy_context()
    panel = control_panel_for_window_count(context, 3, _INDEPENDENT_SPEC)
    assert panel.n_windows == 3


# --- Finding 1: reference-latent convention (independent-N path) -----------


def test_control_arm_c_hat_is_unaffected_by_reference_convention() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    genes = ["SMALL", "BIG", "QUERY1"]
    table_self, _ = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, reference_convention="self"
    )
    table_control, _ = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, reference_convention="control"
    )
    control_self = table_self.loc[table_self["basal_state"] == CONTROL_BASAL_STATE]
    control_control = table_control.loc[
        table_control["basal_state"] == CONTROL_BASAL_STATE
    ]
    merged = control_self.merge(
        control_control, on=["query_gene", "n_windows"], suffixes=("_self", "_control")
    )
    assert len(merged) == len(control_self) == len(control_control)
    pd.testing.assert_series_equal(
        merged["c_hat_self"], merged["c_hat_control"], check_names=False
    )


def test_perturbed_arm_c_hat_differs_between_conventions() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    genes = ["SMALL", "BIG", "QUERY1"]
    table_self, _ = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, reference_convention="self"
    )
    table_control, _ = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, reference_convention="control"
    )
    perturbed_self = table_self.loc[table_self["basal_state"] != CONTROL_BASAL_STATE]
    perturbed_control = table_control.loc[
        table_control["basal_state"] != CONTROL_BASAL_STATE
    ]
    merged = perturbed_self.merge(
        perturbed_control,
        on=["basal_state", "query_gene"],
        suffixes=("_self", "_control"),
    )
    assert (merged["c_hat_self"] != merged["c_hat_control"]).any()


def test_compute_independent_c_hat_table_rejects_unknown_reference_convention() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    with pytest.raises(ValueError, match="reference_convention"):
        compute_independent_c_hat_table(
            context, ["SMALL", "BIG", "QUERY1"], coverage, _INDEPENDENT_SPEC, "bogus"
        )


def test_compute_independent_c_hat_table_covers_every_ordered_pair() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    genes = ["TINY", "SMALL", "BIG", "QUERY1"]
    table, budgets = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, "self"
    )
    perturbed = table.loc[table["basal_state"] != CONTROL_BASAL_STATE]
    assert len(perturbed) == len(genes) * (len(genes) - 1)
    assert set(budgets) == set(genes)
    assert table["c_hat"].apply(np.isfinite).all()


def test_compute_independent_c_hat_table_caches_control_by_query_and_window_count() -> (
    None
):
    context = _toy_context()
    coverage = _toy_coverage(context)
    genes = [
        "BIG",
        "QUERY1",
        "QUERY2",
    ]  # all capped at n_windows == 4 -> share the cache
    table, budgets = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, "self"
    )
    assert {budgets[gene].n_windows for gene in genes} == {4}
    control_rows = table.loc[table["basal_state"] == CONTROL_BASAL_STATE]
    # one control c_hat per query gene, reused across every basal gene sharing w=4
    assert len(control_rows) == len(genes)


# --- Finding A: pair assembly from an independent-N c_hat table ------------


def test_bridge_a_pairs_from_independent_table_is_internally_consistent() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    genes = ["TINY", "SMALL", "BIG"]
    table, budgets = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, "self"
    )
    pairs = bridge_a_pairs_from_independent_c_hat_table(table, budgets, cell_set_len=2)
    assert len(pairs) == 3

    for row in pairs.itertuples(index=False):
        expected_ab = delta_a_to_b(
            row.c_hat_b_given_control, row.c_hat_b_given_a_perturbed
        )
        expected_ba = delta_a_to_b(
            row.c_hat_a_given_control, row.c_hat_a_given_b_perturbed
        )
        assert row.delta_a_to_b == pytest.approx(expected_ab)
        assert row.delta_b_to_a == pytest.approx(expected_ba)
        assert row.s_a == pytest.approx(
            symmetrized_codependency(expected_ab, expected_ba)
        )
        # TINY has n_cells=1 < cell_set_len=2 -> any pair touching it is NOT
        # high_effective_n; SMALL/BIG both have n_cells >= cell_set_len.
        assert bool(row.high_effective_n) == ("TINY" not in (row.gene_a, row.gene_b))


def test_bridge_a_pairs_from_independent_table_raises_on_missing_direction() -> None:
    """FIX 3: pair assembly must FAIL CLOSED on a missing directed prediction,
    never silently emit fewer than the complete C(n, 2) clique.
    """
    context = _toy_context()
    coverage = _toy_coverage(context)
    genes = ["SMALL", "BIG", "QUERY1"]
    table, budgets = compute_independent_c_hat_table(
        context, genes, coverage, _INDEPENDENT_SPEC, "self"
    )
    # Drop c_hat[SMALL | BIG-perturbed]; the (BIG, SMALL) pair can no longer be scored
    # in both directions.
    table = table.loc[
        ~((table["basal_state"] == "BIG") & (table["query_gene"] == "SMALL"))
    ]
    with pytest.raises(RuntimeError, match="missing directed prediction"):
        bridge_a_pairs_from_independent_c_hat_table(table, budgets, cell_set_len=2)


def test_bridge_a_pairs_from_independent_table_raises_when_control_missing() -> None:
    budgets = {
        "A": WindowBudget(n_cells=10, n_windows=1, bootstrapped=False, effective_n=10),
        "B": WindowBudget(n_cells=10, n_windows=1, bootstrapped=False, effective_n=10),
    }
    table = pd.DataFrame(
        [
            {"basal_state": "A", "query_gene": "B", "c_hat": -1.0, "n_windows": 1},
            {"basal_state": "B", "query_gene": "A", "c_hat": -1.0, "n_windows": 1},
            # no "control" rows at all -- an incomplete table.
        ]
    )
    with pytest.raises(RuntimeError, match="missing cached control"):
        bridge_a_pairs_from_independent_c_hat_table(table, budgets, cell_set_len=2)


# --- Finding B: perturbed-arm integration gate ------------------------------


def test_assert_perturbed_arm_integrity_passes_on_toy_context() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    result = assert_perturbed_arm_integrity(
        context,
        ["SMALL", "BIG", "QUERY1", "QUERY2"],
        coverage,
        _INDEPENDENT_SPEC,
        "self",
    )
    assert result["passed"] is True
    assert len(result["genes_checked"]) == 4
    for entry in result["genes_checked"]:
        assert any(delta > 0 for delta in entry["basal_sensitivity_delta"])


def test_assert_perturbed_arm_integrity_requires_at_least_three_genes() -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)
    with pytest.raises(ValueError, match="3 distinct genes"):
        assert_perturbed_arm_integrity(
            context, ["SMALL", "BIG"], coverage, _INDEPENDENT_SPEC, "self"
        )


def test_assert_perturbed_arm_integrity_fails_when_perturbed_panel_equals_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _toy_context()
    coverage = _toy_coverage(context)

    def _degenerate_perturbed_panel(
        ctx: BridgeAContext, gene: str, cov: dict[str, int], spec: IndependentPanelSpec
    ) -> tuple[object, WindowBudget]:
        del gene, cov
        panel = control_panel_for_window_count(ctx, spec.max_windows, spec)
        budget = WindowBudget(
            n_cells=999,
            n_windows=spec.max_windows,
            bootstrapped=False,
            effective_n=spec.max_windows * spec.cell_set_len,
        )
        return panel, budget

    monkeypatch.setattr(
        bridge_a_gates, "independent_perturbed_panel", _degenerate_perturbed_panel
    )
    with pytest.raises(RuntimeError, match="IDENTICAL"):
        assert_perturbed_arm_integrity(
            context, ["SMALL", "BIG", "QUERY1"], coverage, _INDEPENDENT_SPEC, "self"
        )


def test_assert_perturbed_arm_integrity_fails_when_forward_ignores_basal_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding B (adversarial): a forward that substitutes a FIXED basal panel
    regardless of what panel is actually passed in (ignoring ``control_chunks``
    entirely) but still returns finite, deterministic, query-specific c_hat
    must fail the STRENGTHENED gate. The pre-Finding-B gate only checked that
    outputs varied by query gene off a single (perturbed) panel and would have
    missed this -- the panel itself is untouched (still gene-specific, still
    non-identical to control), only what the FORWARD does with it is broken.
    """
    context = _toy_context()
    coverage = _toy_coverage(context)
    fixed_chunks = control_panel_for_window_count(
        context, _INDEPENDENT_SPEC.max_windows, _INDEPENDENT_SPEC
    ).control_chunks
    fixed_batch_chunks = tuple(None for _ in fixed_chunks)
    original_forward = context.model.predict_response_chunks

    def _basal_input_bypass_forward(control_chunks, gene, batch_index_chunks):
        del control_chunks, batch_index_chunks
        return original_forward(fixed_chunks, gene, fixed_batch_chunks)

    monkeypatch.setattr(
        context.model, "predict_response_chunks", _basal_input_bypass_forward
    )
    with pytest.raises(RuntimeError, match="insensitive to the basal input"):
        assert_perturbed_arm_integrity(
            context,
            ["SMALL", "BIG", "QUERY1", "QUERY2"],
            coverage,
            _INDEPENDENT_SPEC,
            "self",
        )


# --- Finding C: seed-replicate aggregation completeness guards -------------


def _pair_row(
    gene_a: str,
    gene_b: str,
    delta_ab: float,
    delta_ba: float,
    s_a: float,
    seed: int,
    **provenance: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "gene_a": gene_a,
        "gene_b": gene_b,
        "delta_a_to_b": delta_ab,
        "delta_b_to_a": delta_ba,
        "s_a": s_a,
        "seed": seed,
    }
    row.update(provenance)
    return row


_PROVENANCE = {
    "n_windows_a_to_b": 2,
    "n_windows_b_to_a": 1,
    "effective_n_a_to_b": 128,
    "effective_n_b_to_a": 10,
    "bootstrapped_a_to_b": False,
    "bootstrapped_b_to_a": True,
    "high_effective_n": False,
}


def test_aggregate_seed_replicate_pairs_computes_mean_std_and_materiality_flag() -> (
    None
):
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41, **_PROVENANCE)])
    seed2 = pd.DataFrame([_pair_row("A", "B", 1.0, 3.0, 3.0, 42, **_PROVENANCE)])
    summary = aggregate_seed_replicate_pairs([seed1, seed2], [41, 42])
    row = summary.iloc[0]
    assert row["s_a_seed_mean"] == pytest.approx(2.0)
    assert row["s_a_seed_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert bool(row["seed_variance_material"]) is True
    assert int(row["n_seed_replicates"]) == 2
    assert int(row["n_windows_a_to_b"]) == 2
    assert bool(row["high_effective_n"]) is False


def test_aggregate_seed_replicate_pairs_flags_only_material_variance() -> None:
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41, **_PROVENANCE)])
    seed2 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.01, 1.01, 42, **_PROVENANCE)])
    summary = aggregate_seed_replicate_pairs([seed1, seed2], [41, 42])
    assert bool(summary.iloc[0]["seed_variance_material"]) is False


def test_aggregate_seed_replicate_pairs_requires_at_least_two_seeds() -> None:
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41)])
    with pytest.raises(ValueError, match=">= 2 distinct seeds"):
        aggregate_seed_replicate_pairs([seed1], [41])


def test_aggregate_seed_replicate_pairs_rejects_duplicate_requested_seeds() -> None:
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41)])
    seed2 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41)])
    with pytest.raises(ValueError, match="DISTINCT"):
        aggregate_seed_replicate_pairs([seed1, seed2], [41, 41])


def test_aggregate_seed_replicate_pairs_rejects_table_count_mismatch() -> None:
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41)])
    with pytest.raises(ValueError, match="one pair table per requested seed"):
        aggregate_seed_replicate_pairs([seed1], [41, 42])


def test_aggregate_seed_replicate_pairs_rejects_seed_column_mismatch() -> None:
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 999)])  # wrong seed value
    seed2 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 42)])
    with pytest.raises(ValueError, match="does not carry"):
        aggregate_seed_replicate_pairs([seed1, seed2], [41, 42])


def test_aggregate_seed_replicate_pairs_rejects_duplicate_pair_in_one_replicate() -> (
    None
):
    seed1 = pd.DataFrame(
        [
            _pair_row("A", "B", 1.0, 1.0, 1.0, 41),
            _pair_row("A", "B", 2.0, 2.0, 2.0, 41),
        ]
    )
    seed2 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 42)])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_seed_replicate_pairs([seed1, seed2], [41, 42])


def test_aggregate_seed_replicate_pairs_rejects_mismatched_pair_key_sets() -> None:
    seed1 = pd.DataFrame([_pair_row("A", "B", 1.0, 1.0, 1.0, 41)])
    seed2 = pd.DataFrame([_pair_row("A", "C", 1.0, 1.0, 1.0, 42)])
    with pytest.raises(ValueError, match="different set"):
        aggregate_seed_replicate_pairs([seed1, seed2], [41, 42])


def test_aggregate_seed_replicate_pairs_raises_when_provenance_varies() -> None:
    seed1 = pd.DataFrame(
        [
            _pair_row(
                "A", "B", 1.0, 1.0, 1.0, 41, n_windows_a_to_b=2, high_effective_n=True
            )
        ]
    )
    seed2 = pd.DataFrame(
        [
            _pair_row(
                "A", "B", 1.0, 1.0, 1.0, 42, n_windows_a_to_b=3, high_effective_n=True
            )
        ]
    )
    with pytest.raises(RuntimeError, match="vary across seed replicates"):
        aggregate_seed_replicate_pairs([seed1, seed2], [41, 42])


def _c_hat_row(
    basal_state: str,
    query_gene: str,
    c_hat: float,
    seed: int,
    *,
    n_windows: int = 4,
    reference_convention: str = "self",
) -> dict[str, object]:
    return {
        "basal_state": basal_state,
        "query_gene": query_gene,
        "c_hat": c_hat,
        "seed": seed,
        "n_windows": n_windows,
        "reference_convention": reference_convention,
    }


def test_aggregate_seed_replicate_c_hat_computes_mean_and_std() -> None:
    seed1 = pd.DataFrame([_c_hat_row("control", "A", -1.0, 41)])
    seed2 = pd.DataFrame([_c_hat_row("control", "A", -2.0, 42)])
    summary = aggregate_seed_replicate_c_hat([seed1, seed2], [41, 42])
    row = summary.iloc[0]
    assert row["c_hat_seed_mean"] == pytest.approx(-1.5)
    assert row["c_hat_seed_std"] == pytest.approx(np.std([-1.0, -2.0], ddof=1))
    assert int(row["n_seed_replicates"]) == 2


def test_aggregate_seed_replicate_c_hat_separates_window_budgets_and_conventions() -> (
    None
):
    """FIX 2: control@w=1 and control@w=2 (and a different reference
    convention) for the SAME (basal_state, query_gene) must NOT be pooled
    together -- each (n_windows, reference_convention) key gets its own row
    and its own seed mean/std, not one row mixing different estimands.
    """
    seed1 = pd.DataFrame(
        [
            _c_hat_row("control", "A", -1.0, 41, n_windows=1),
            _c_hat_row("control", "A", -1.0, 41, n_windows=2),
            _c_hat_row("control", "A", -1.0, 41, reference_convention="control"),
        ]
    )
    seed2 = pd.DataFrame(
        [
            _c_hat_row("control", "A", -3.0, 42, n_windows=1),
            _c_hat_row("control", "A", 5.0, 42, n_windows=2),
            _c_hat_row("control", "A", 9.0, 42, reference_convention="control"),
        ]
    )
    summary = aggregate_seed_replicate_c_hat([seed1, seed2], [41, 42])
    assert len(summary) == 3  # one row per distinct (n_windows, reference_convention)
    by_key = {
        (int(row.n_windows), row.reference_convention): row.c_hat_seed_mean
        for row in summary.itertuples(index=False)
    }
    assert by_key[(1, "self")] == pytest.approx(-2.0)  # mean(-1, -3), NOT mixed w/ w=2
    assert by_key[(2, "self")] == pytest.approx(2.0)  # mean(-1, 5)
    assert by_key[(4, "control")] == pytest.approx(4.0)  # mean(-1, 9)


def test_aggregate_seed_replicate_c_hat_never_fills_std_with_zero() -> None:
    """Every returned row already has exactly ``len(seeds) >= 2`` replicates by
    construction, so ``c_hat_seed_std`` is a genuinely computed std -- not the
    removed ``.fillna(0.0)`` that used to falsely report perfect stability.
    """
    seed1 = pd.DataFrame([_c_hat_row("control", "A", 1.0, 41)])
    seed2 = pd.DataFrame([_c_hat_row("control", "A", 1.0, 42)])  # identical c_hat
    summary = aggregate_seed_replicate_c_hat([seed1, seed2], [41, 42])
    # A genuine 0.0 here is np.std([1.0, 1.0], ddof=1) -- a real 2-observation
    # computation, not a fillna masking an undefined single-observation std.
    assert summary.iloc[0]["c_hat_seed_std"] == pytest.approx(0.0)
    assert int(summary.iloc[0]["n_seed_replicates"]) == 2


def test_aggregate_seed_replicate_c_hat_requires_at_least_two_seeds() -> None:
    seed1 = pd.DataFrame([_c_hat_row("control", "A", 1.0, 41)])
    with pytest.raises(ValueError, match=">= 2 distinct seeds"):
        aggregate_seed_replicate_c_hat([seed1], [41])


def test_aggregate_seed_replicate_c_hat_infers_seeds_when_omitted() -> None:
    """``scripts/bridge_a_forward.py``'s existing call site does not pass
    ``seeds`` explicitly; omitting it must infer one seed per table from each
    table's own ``seed`` column, with the SAME completeness guards applied.
    """
    seed1 = pd.DataFrame([_c_hat_row("control", "A", -1.0, 41)])
    seed2 = pd.DataFrame([_c_hat_row("control", "A", -2.0, 42)])
    summary = aggregate_seed_replicate_c_hat([seed1, seed2])  # no seeds arg
    row = summary.iloc[0]
    assert row["c_hat_seed_mean"] == pytest.approx(-1.5)
    assert int(row["n_seed_replicates"]) == 2


def test_aggregate_seed_replicate_c_hat_rejects_mismatched_key_sets() -> None:
    """Parity with ``aggregate_seed_replicate_pairs``: the shared
    ``validate_seed_replicate_tables`` guard applies identically here.
    """
    seed1 = pd.DataFrame(
        [_c_hat_row("control", "A", 1.0, 41), _c_hat_row("control", "B", 1.0, 41)]
    )
    seed2 = pd.DataFrame([_c_hat_row("control", "A", 1.0, 42)])
    with pytest.raises(ValueError, match="different set"):
        aggregate_seed_replicate_c_hat([seed1, seed2], [41, 42])


# --- FIX 4: project_full_sweep derives counts from the locked universe -----


def test_project_full_sweep_derives_pair_count_from_gene_count() -> None:
    """A reduced/alternate universe (e.g. ``--drop-nonqueryable``) must be
    projected using ITS OWN gene count, not the hardcoded 408/83,028 locked
    default.
    """
    result = project_full_sweep([0.1, 0.2], torch.device("cpu"), 8, gene_count=10)
    assert result["full_sweep_gene_count"] == 10
    assert result["full_sweep_pair_count"] == 45  # C(10, 2)
    assert result["full_sweep_a_perturbed_forwards"] == 90  # 10 * 9
    assert result["full_sweep_control_cache_forwards_upper_bound"] == 80  # 10 * 8


def test_project_full_sweep_defaults_to_the_locked_408_gene_universe() -> None:
    result = project_full_sweep([0.1, 0.2], torch.device("cpu"), 8)
    assert result["full_sweep_gene_count"] == 408
    assert result["full_sweep_pair_count"] == 408 * 407 // 2


def test_project_full_sweep_rejects_gene_count_below_two() -> None:
    with pytest.raises(ValueError, match="gene_count must be >= 2"):
        project_full_sweep([0.1], torch.device("cpu"), 8, gene_count=1)

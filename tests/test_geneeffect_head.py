"""Tests for :mod:`aivc_model.geneeffect_head` (Exp13 B2 + Phase 6 head).

Organized around the two pieces the module exists to provide:

1. An axis-aware training loss (:func:`per_gene_rank_variance_loss`) that
   correlates per gene along the context axis, unlike the retired
   ``tx1_geneeffect_head.rank_variance_loss`` (flattens to ``[B]``, one
   Pearson over mixed genes/contexts; inlined below as
   ``_flattened_rank_variance_loss`` since that module was deleted at
   ``873c99c``). The critical test
   (``test_per_gene_loss_disagrees_with_flattened_loss_when_mu_g_dominates``)
   constructs a batch where a shared per-gene offset makes the flattened loss
   look good while the true per-context signal is uncorrelated -- exactly
   the failure mode B2 exists to fix.
2. A five-block ``h_delta`` (:class:`GeneEffectResidualHead`) whose blocks
   are individually disableable and whose partial-coverage inputs are
   signalled by explicit masks, never a silent zero-fill.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from aivc_model.geneeffect_head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectResidualHead,
    macro_per_gene_spearman,
    per_gene_rank_variance_loss,
)

# ---------------------------------------------------------------------------
# per_gene_rank_variance_loss
# ---------------------------------------------------------------------------


def _flattened_rank_variance_loss(
    pred: torch.Tensor, target: torch.Tensor, lam: float = 1.0
) -> torch.Tensor:
    """Local copy of the retired ``tx1_geneeffect_head.rank_variance_loss``:
    flattens ``[n_genes, n_contexts]`` to ``[B]`` and computes ONE Pearson
    correlation over mixed genes/contexts -- exactly the failure mode B2
    (``per_gene_rank_variance_loss``) exists to fix. Inlined, not imported,
    because ``tx1_geneeffect_head.py`` was deleted at ``873c99c`` and this
    disagreement is the regression evidence for B2, not a tested unit."""
    std_eps = 1e-6
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    covariance = (pred_centered * target_centered).mean()
    pred_std = (pred_centered.square().mean() + std_eps).sqrt()
    target_std = (target_centered.square().mean() + std_eps).sqrt()
    corr_term = 1.0 - covariance / (pred_std * target_std)
    var_term = (pred.std(unbiased=False) - target.std(unbiased=False)).square()
    return corr_term + float(lam) * var_term


def _hand_reference(pred: torch.Tensor, target: torch.Tensor, lam: float) -> float:
    """Hand-computed per-gene loss: manual Python loop over rows, no torch
    vectorization, using plain float arithmetic -- an independent
    implementation of the same definition to check against."""
    n_genes, n_contexts = pred.shape
    losses = []
    for g in range(n_genes):
        p = [pred[g, c].item() for c in range(n_contexts)]
        t = [target[g, c].item() for c in range(n_contexts)]
        p_mean = sum(p) / n_contexts
        t_mean = sum(t) / n_contexts
        p_c = [x - p_mean for x in p]
        t_c = [x - t_mean for x in t]
        cov = sum(a * b for a, b in zip(p_c, t_c)) / n_contexts
        p_std = math.sqrt(sum(x * x for x in p_c) / n_contexts + 1e-6)
        t_std = math.sqrt(sum(x * x for x in t_c) / n_contexts + 1e-6)
        corr_loss = 1.0 - cov / (p_std * t_std)
        p_raw_std = math.sqrt(sum(x * x for x in p_c) / n_contexts)
        t_raw_std = math.sqrt(sum(x * x for x in t_c) / n_contexts)
        var_loss = (p_raw_std - t_raw_std) ** 2
        losses.append(corr_loss + lam * var_loss)
    return sum(losses) / len(losses)


def test_per_gene_loss_matches_hand_computed_reference() -> None:
    torch.manual_seed(0)
    pred = torch.randn(6, 9)
    target = torch.randn(6, 9)
    lam = 0.7
    result = per_gene_rank_variance_loss(pred, target, lam=lam)
    expected = _hand_reference(pred, target, lam)
    assert result.n_genes_scored == 6
    assert result.n_genes_excluded == 0
    assert result.loss.item() == pytest.approx(expected, abs=1e-5)


def test_per_gene_loss_disagrees_with_flattened_loss_when_mu_g_dominates() -> None:
    """The whole point of B2: construct a batch where a large shared
    per-gene offset dominates both pred and target's total variance (as
    real GeneEffect's mu_g dominates Var(delta), CLAUDE.md). The flattened
    rank_variance_loss sees near-perfect correlation (the offset explains
    almost everything), while the true per-context signal inside each gene
    is independent noise -- the per-gene axis correctly reports that as
    uninformative.
    """
    torch.manual_seed(1)
    n_genes, n_contexts = 24, 12
    gene_offset = torch.linspace(-6.0, 6.0, n_genes).unsqueeze(1)
    target = gene_offset + 0.02 * torch.randn(n_genes, n_contexts)
    pred = gene_offset + 0.02 * torch.randn(n_genes, n_contexts)

    flattened = _flattened_rank_variance_loss(
        pred.reshape(-1), target.reshape(-1), lam=1.0
    )
    per_gene = per_gene_rank_variance_loss(pred, target, lam=1.0)

    # Flattened: dominated by the shared offset -> near-perfect correlation.
    assert flattened.item() < 0.05
    # Per-gene: within a gene, pred/target vary only by independent noise
    # -> near-zero correlation -> loss near the "zero correlation" value (1).
    assert per_gene.loss.item() > 0.7
    # The disagreement itself is the point.
    assert per_gene.loss.item() - flattened.item() > 0.5


def test_constant_prediction_gets_worst_correlation_term() -> None:
    torch.manual_seed(2)
    target = torch.randn(5, 10)
    # Ensure no row is (near-)constant in target so none are excluded.
    target += torch.linspace(-1, 1, 10).unsqueeze(0)
    pred = torch.full((5, 10), 3.0)
    result = per_gene_rank_variance_loss(pred, target, lam=0.0)
    assert result.n_genes_excluded == 0
    # lam=0 isolates the correlation term; constant pred -> worst case (~1).
    assert result.loss.item() > 0.9


def test_variance_term_punishes_collapsed_prediction() -> None:
    torch.manual_seed(3)
    target = torch.randn(5, 10) * 3.0 + torch.linspace(-1, 1, 10).unsqueeze(0)
    collapsed_pred = torch.full((5, 10), 1.0)
    aligned_pred = target.clone() + 0.01 * torch.randn_like(target)

    collapsed = per_gene_rank_variance_loss(collapsed_pred, target, lam=1.0)
    aligned = per_gene_rank_variance_loss(aligned_pred, target, lam=1.0)
    assert collapsed.loss.item() > 5.0 * aligned.loss.item()


def test_constant_target_genes_excluded_not_zero_or_nan() -> None:
    torch.manual_seed(4)
    n_genes, n_contexts = 8, 6
    target = torch.randn(n_genes, n_contexts)
    # Force genes 0 and 3 to be exactly constant across contexts.
    target[0, :] = 2.5
    target[3, :] = -1.0
    pred = torch.randn(n_genes, n_contexts)

    result = per_gene_rank_variance_loss(pred, target, lam=1.0)
    assert result.n_genes_excluded == 2
    assert result.n_genes_scored == 6
    assert math.isfinite(result.loss.item())

    # Cross-check: excluding rows 0/3 by hand and averaging the rest must
    # match exactly (never silently folding the excluded rows in as 0).
    keep = [i for i in range(n_genes) if i not in (0, 3)]
    manual = _hand_reference(pred[keep], target[keep], 1.0)
    assert result.loss.item() == pytest.approx(manual, abs=1e-5)


def test_all_targets_constant_raises() -> None:
    pred = torch.randn(4, 5)
    target = torch.ones(4, 5) * 7.0
    with pytest.raises(ValueError, match="constant"):
        per_gene_rank_variance_loss(pred, target)


def test_per_gene_loss_rejects_mismatched_or_1d_shapes() -> None:
    with pytest.raises(ValueError):
        per_gene_rank_variance_loss(torch.randn(5), torch.randn(5))
    with pytest.raises(ValueError):
        per_gene_rank_variance_loss(torch.randn(4, 5), torch.randn(4, 6))


def test_per_gene_loss_gradient_flows_and_is_finite() -> None:
    torch.manual_seed(5)
    target = torch.randn(6, 8) + torch.linspace(-1, 1, 8).unsqueeze(0)
    pred = torch.randn(6, 8, requires_grad=True)
    result = per_gene_rank_variance_loss(pred, target, lam=1.0)
    result.loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_per_gene_loss_axis_invariance_to_per_gene_constant_shift() -> None:
    """Mirrors residual_ladder._check_axis_invariance: the per-gene loss is
    a function of within-gene structure only, so adding an arbitrary
    per-gene constant to the truth must not change it."""
    torch.manual_seed(6)
    n_genes, n_contexts = 7, 9
    target = torch.randn(n_genes, n_contexts) + torch.linspace(
        -1, 1, n_contexts
    ).unsqueeze(0)
    pred = torch.randn(n_genes, n_contexts)
    shift = torch.randn(n_genes, 1) * 100.0

    baseline = per_gene_rank_variance_loss(pred, target, lam=1.0)
    shifted = per_gene_rank_variance_loss(pred, target + shift, lam=1.0)
    assert shifted.loss.item() == pytest.approx(baseline.loss.item(), abs=1e-4)
    assert shifted.n_genes_excluded == baseline.n_genes_excluded


# ---------------------------------------------------------------------------
# macro_per_gene_spearman
# ---------------------------------------------------------------------------


def test_macro_per_gene_spearman_perfect_correlation() -> None:
    n_genes, n_contexts = 5, 6
    target = torch.randn(n_genes, n_contexts) + torch.linspace(
        -2, 2, n_contexts
    ).unsqueeze(0)
    pred = target.clone()
    result = macro_per_gene_spearman(pred, target)
    assert result.n_undefined == 0
    assert result.n_scored == n_genes
    assert result.macro == pytest.approx(1.0, abs=1e-6)
    assert isinstance(result.per_gene, pd.Series)


def test_macro_per_gene_spearman_undefined_for_constant_prediction() -> None:
    """A context-blind predictor (same value for every context, e.g. a raw
    gene-mean baseline) is undefined on this axis -- never scored as 0."""
    n_genes, n_contexts = 5, 6
    target = torch.randn(n_genes, n_contexts) + torch.linspace(
        -2, 2, n_contexts
    ).unsqueeze(0)
    pred = torch.zeros(n_genes, n_contexts)
    result = macro_per_gene_spearman(pred, target)
    assert result.n_scored == 0
    assert result.n_undefined == n_genes
    assert math.isnan(result.macro)
    assert result.per_gene.isna().all()


def test_macro_per_gene_spearman_uses_gene_ids_as_index() -> None:
    pred = torch.randn(3, 5)
    target = torch.randn(3, 5) + torch.linspace(-1, 1, 5).unsqueeze(0)
    gene_ids = ["TP53", "EGFR", "MYC"]
    result = macro_per_gene_spearman(pred, target, gene_ids=gene_ids)
    assert sorted(result.per_gene.index) == sorted(gene_ids)


def test_macro_per_gene_spearman_rejects_wrong_length_gene_ids() -> None:
    with pytest.raises(ValueError):
        macro_per_gene_spearman(torch.randn(3, 4), torch.randn(3, 4), gene_ids=["A"])


# ---------------------------------------------------------------------------
# GeneEffectResidualHead
# ---------------------------------------------------------------------------


def _full_dims() -> GeneEffectFeatureDims:
    return GeneEffectFeatureDims(delta_proj=6, s=4, q_sc=3, e_g=5, z_c=4)


def _full_inputs(batch: int, dims: GeneEffectFeatureDims) -> dict[str, torch.Tensor]:
    return {
        "delta_proj": torch.randn(batch, dims.delta_proj),
        "s": torch.randn(batch, dims.s),
        "q_sc": torch.randn(batch, dims.q_sc),
        "e_g": torch.randn(batch, dims.e_g),
        "z_c": torch.randn(batch, dims.z_c),
        "q_sc_mask": torch.ones(batch, dtype=torch.bool),
        "hvg_panel_mask": torch.ones(batch, dtype=torch.bool),
        "own_gene_shift_mask": torch.ones(batch, dtype=torch.bool),
    }


def test_head_forward_all_blocks_enabled_returns_finite_batch() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    inputs = _full_inputs(7, dims)
    out = head(**inputs)
    assert out.shape == (7,)
    assert torch.isfinite(out).all()


def test_head_input_width_accounts_for_blocks_and_mask_bits() -> None:
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    # delta_proj(6) + s(4)+2 + q_sc(3)+1 + e_g(5) + z_c(4) = 25
    assert head.input_width == 6 + (4 + 2) + (3 + 1) + 5 + 4


def test_disabling_a_block_changes_param_count_and_forward_result() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    full_head = GeneEffectResidualHead(dims=dims, hidden=16)
    torch.manual_seed(0)
    no_esm2_head = GeneEffectResidualHead(
        dims=dims,
        blocks=GeneEffectBlockConfig(use_e_g=False),
        hidden=16,
    )
    full_params = sum(p.numel() for p in full_head.parameters())
    reduced_params = sum(p.numel() for p in no_esm2_head.parameters())
    assert reduced_params < full_params
    assert no_esm2_head.input_width == full_head.input_width - dims.e_g

    torch.manual_seed(1)
    inputs = _full_inputs(5, dims)
    with torch.no_grad():
        full_out = full_head(**inputs)
    reduced_inputs = dict(inputs)
    del reduced_inputs["e_g"]
    with torch.no_grad():
        reduced_out = no_esm2_head(**reduced_inputs)
    # Different architectures (different Linear-in width) at minimum yield
    # a differently-shaped first layer; forward outputs need not be close.
    assert full_out.shape == reduced_out.shape
    assert not torch.allclose(full_out, reduced_out)


def test_virtual_cell_ablation_disables_delta_proj_and_s_together() -> None:
    dims = _full_dims()
    blocks = GeneEffectBlockConfig(use_delta_proj=False, use_s=False)
    head = GeneEffectResidualHead(dims=dims, blocks=blocks, hidden=16)
    assert head.input_width == dims.q_sc + 1 + dims.e_g + dims.z_c
    inputs = _full_inputs(4, dims)
    del inputs["delta_proj"], inputs["s"]
    del inputs["hvg_panel_mask"], inputs["own_gene_shift_mask"]
    out = head(**inputs)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_forward_rejects_tensor_for_disabled_block() -> None:
    dims = _full_dims()
    head = GeneEffectResidualHead(
        dims=dims, blocks=GeneEffectBlockConfig(use_e_g=False), hidden=16
    )
    inputs = _full_inputs(3, dims)
    with pytest.raises(ValueError, match="e_g"):
        head(**inputs)


def test_forward_requires_tensor_for_enabled_block() -> None:
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    inputs = _full_inputs(3, dims)
    del inputs["z_c"]
    with pytest.raises(ValueError, match="z_c"):
        head(**inputs)


def test_forward_requires_masks_for_enabled_blocks() -> None:
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    inputs = _full_inputs(3, dims)
    del inputs["q_sc_mask"]
    with pytest.raises(ValueError, match="q_sc_mask"):
        head(**inputs)


def test_no_block_enabled_config_rejected() -> None:
    with pytest.raises(ValueError):
        GeneEffectBlockConfig(
            use_delta_proj=False,
            use_s=False,
            use_q_sc=False,
            use_e_g=False,
            use_z_c=False,
        )


def test_masked_missing_q_sc_value_never_leaks_into_output() -> None:
    """The critical masking test: two forward passes with wildly different
    RAW q_sc values, but both marked missing (q_sc_mask=False), must yield
    identical output -- proving the raw value never enters as data, only
    the mask bit does. This is what distinguishes a masked-missing feature
    from a naive zero-fill (which a caller might do without a mask;
    zero-filling done correctly here is paired with an explicit bit, but
    the network must never see any information from the raw value itself)."""
    torch.manual_seed(0)
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    torch.manual_seed(0)
    inputs = _full_inputs(6, dims)
    inputs["q_sc_mask"] = torch.zeros(6, dtype=torch.bool)

    inputs_a = dict(inputs)
    inputs_a["q_sc"] = torch.zeros(6, dims.q_sc)
    inputs_b = dict(inputs)
    inputs_b["q_sc"] = torch.full((6, dims.q_sc), 1e6)

    with torch.no_grad():
        out_a = head(**inputs_a)
        out_b = head(**inputs_b)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_masked_missing_own_gene_shift_never_leaks_into_output() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    inputs = _full_inputs(6, dims)
    inputs["own_gene_shift_mask"] = torch.zeros(6, dtype=torch.bool)

    inputs_a = dict(inputs)
    s_a = inputs["s"].clone()
    s_a[:, -1] = 0.0
    inputs_a["s"] = s_a

    inputs_b = dict(inputs)
    s_b = inputs["s"].clone()
    s_b[:, -1] = 999.0
    inputs_b["s"] = s_b

    with torch.no_grad():
        out_a = head(**inputs_a)
        out_b = head(**inputs_b)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_mask_bit_itself_changes_output_between_present_and_missing() -> None:
    """Sanity check that the mask channel is actually wired into the net
    (not merely present-but-ignored): with the same (zero) value, toggling
    q_sc_mask must generally change the output."""
    torch.manual_seed(0)
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    inputs = _full_inputs(6, dims)
    inputs["q_sc"] = torch.zeros(6, dims.q_sc)

    inputs_present = dict(inputs)
    inputs_present["q_sc_mask"] = torch.ones(6, dtype=torch.bool)
    inputs_missing = dict(inputs)
    inputs_missing["q_sc_mask"] = torch.zeros(6, dtype=torch.bool)

    with torch.no_grad():
        out_present = head(**inputs_present)
        out_missing = head(**inputs_missing)
    assert not torch.allclose(out_present, out_missing)


def test_head_has_no_cell_line_identity_parameter() -> None:
    """Structural check mirroring Tx1GeneEffectHead's: no per-line embedding
    table anywhere, so the head accepts any batch size without a fixed
    line vocabulary and never sees a line-ID tensor."""
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    for module in head.modules():
        assert not isinstance(module, torch.nn.Embedding)
    with torch.no_grad():
        out_small = head(**_full_inputs(2, dims))
        out_large = head(**_full_inputs(50, dims))
    assert torch.isfinite(out_small).all()
    assert torch.isfinite(out_large).all()


def test_head_backward_populates_finite_grads() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = GeneEffectResidualHead(dims=dims, hidden=16)
    inputs = _full_inputs(5, dims)
    out = head(**inputs)
    out.sum().backward()
    for p in head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

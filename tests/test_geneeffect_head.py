"""Five-block head, explicit missingness and per-gene reporting metrics."""

import math
import pandas as pd
import pytest
import torch
from src.model.head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectResidualHead,
)
from src.eval.metrics import macro_per_gene_spearman


def test_formal_feature_dimension_defaults() -> None:
    dims = GeneEffectFeatureDims()
    assert dims == GeneEffectFeatureDims(
        delta_proj=256, s=6, q_sc=3, e_g=1280, z_c=5120
    )


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

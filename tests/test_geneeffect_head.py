"""Nested low-rank head, explicit missingness, the joint model's STATE use, context
and residual scale, and per-gene reporting metrics."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from src.model.head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectMLP,
    GeneEffectNestedHead,
    SwiGLU,
    masked_blocks,
)
from src.eval.metrics import macro_per_gene_spearman


def test_formal_feature_dimension_defaults() -> None:
    dims = GeneEffectFeatureDims()
    assert dims == GeneEffectFeatureDims(delta_proj=256, s=6, q_sc=3, e_g=1280, z_c=128)


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


N_GENES = 4
RANK = 3
NO_STATE = GeneEffectBlockConfig(use_delta_proj=False, use_s=False)
CONTEXT_ONLY = GeneEffectBlockConfig(
    use_delta_proj=False, use_s=False, use_q_sc=False, use_e_g=False
)


def _full_dims() -> GeneEffectFeatureDims:
    return GeneEffectFeatureDims(delta_proj=6, s=4, q_sc=3, e_g=5, z_c=4)


def _head(
    dims: GeneEffectFeatureDims,
    blocks: GeneEffectBlockConfig = GeneEffectBlockConfig(),
    *,
    dropout: float = 0.0,
) -> GeneEffectNestedHead:
    return GeneEffectNestedHead(
        dims, blocks, n_genes=N_GENES, factor_rank=RANK, dropout=dropout
    )


def _live(head: GeneEffectNestedHead) -> GeneEffectNestedHead:
    """Replace the zero-initialised SwiGLU output layers by random weights."""
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, SwiGLU):
                module.out.weight.normal_()
                module.out.bias.normal_()
    return head


def _full_inputs(batch: int, dims: GeneEffectFeatureDims) -> dict[str, torch.Tensor]:
    return {
        "gene_index": torch.arange(batch) % N_GENES,
        "delta_proj": torch.randn(batch, dims.delta_proj),
        "s": torch.randn(batch, dims.s),
        "q_sc": torch.randn(batch, dims.q_sc),
        "e_g": torch.randn(batch, dims.e_g),
        "z_c": torch.randn(batch, dims.z_c),
        "q_sc_mask": torch.ones(batch, dtype=torch.bool),
        "hvg_panel_mask": torch.ones(batch, dtype=torch.bool),
        "own_gene_shift_mask": torch.ones(batch, dtype=torch.bool),
    }


def _without_state(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in inputs.items()
        if name not in {"delta_proj", "s", "hvg_panel_mask", "own_gene_shift_mask"}
    }


def _blocks_only(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in inputs.items() if name != "gene_index"}


def test_head_forward_all_blocks_enabled_returns_finite_batch() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    out = _live(_head(dims))(**_full_inputs(7, dims))
    assert out.shape == (7,)
    assert torch.isfinite(out).all()


def test_zero_initialised_branches_leave_the_reduced_rank_ridge() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _head(dims)
    inputs = _full_inputs(6, dims)
    z = inputs["z_c"]
    gene = head.gene_embedding(inputs["gene_index"]) + head.gene_projection(
        inputs["e_g"]
    )
    expected = (gene * head.context_linear(z)).sum(-1) / math.sqrt(RANK)
    with torch.no_grad():
        assert torch.equal(head.context_residual(z), torch.zeros(6, RANK))
        parts = masked_blocks(dims, head.blocks, **_blocks_only(inputs))
        encoded = torch.cat(
            [encoder(parts[name]) for name, encoder in head.encoders.items()], -1
        )
        assert torch.equal(head.correction(encoded), torch.zeros(6, 1))
        torch.testing.assert_close(head(**inputs), expected, rtol=0, atol=0)


def test_head_is_gene_context_product_plus_correction() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims))
    inputs = _full_inputs(6, dims)
    parts = masked_blocks(dims, head.blocks, **_blocks_only(inputs))
    z = inputs["z_c"]
    gene = head.gene_embedding(inputs["gene_index"]) + head.gene_projection(
        inputs["e_g"]
    )
    context = head.context_linear(z) + head.context_residual(z)
    correction = head.correction(
        torch.cat([head.encoders[name](parts[name]) for name in head.encoders], -1)
    ).squeeze(-1)
    expected = (gene * context).sum(-1) / math.sqrt(RANK) + correction
    assert list(head.encoders) == ["delta_proj", "s", "q_sc", "e_g"]
    with torch.no_grad():
        torch.testing.assert_close(head(**inputs), expected)


def test_correction_never_receives_the_context() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims))
    assert "z_c" not in head.encoders
    seen = []
    hook = head.correction.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].clone())
    )
    inputs = _full_inputs(5, dims)
    with torch.no_grad():
        first = head(**inputs)
        second = head(**dict(inputs, z_c=inputs["z_c"] + 3.0))
    hook.remove()
    assert seen[0].shape == (5, 32 + 16 + 16 + 32)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    assert not torch.allclose(first, second)  # z_c still reaches C


def test_context_only_head_has_no_correction() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims, CONTEXT_ONLY))
    assert head.correction is None and len(head.encoders) == 0
    assert head.gene_projection is None
    inputs = {"gene_index": torch.arange(3) % N_GENES, "z_c": torch.randn(3, 4)}
    z = inputs["z_c"]
    expected = (
        head.gene_embedding(inputs["gene_index"])
        * (head.context_linear(z) + head.context_residual(z))
    ).sum(-1) / math.sqrt(RANK)
    with torch.no_grad():
        torch.testing.assert_close(head(**inputs), expected)


def test_prediction_depends_on_gene_index() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _head(dims)
    inputs = _full_inputs(5, dims)
    other = dict(inputs, gene_index=(inputs["gene_index"] + 1) % N_GENES)
    with torch.no_grad():
        assert not torch.allclose(head(**inputs), head(**other))
    # Without ESM2 the free embedding alone still separates genes.
    no_esm2 = _head(dims, GeneEffectBlockConfig(use_e_g=False))
    assert no_esm2.gene_projection is None
    del inputs["e_g"], other["e_g"]
    with torch.no_grad():
        assert not torch.allclose(no_esm2(**inputs), no_esm2(**other))


def test_gene_index_must_be_integer_per_row() -> None:
    dims = _full_dims()
    head = _head(dims)
    inputs = _full_inputs(3, dims)
    with pytest.raises(ValueError, match="gene_index"):
        head(**dict(inputs, gene_index=inputs["gene_index"].float()))
    with pytest.raises(ValueError, match="gene_index"):
        head(**dict(inputs, gene_index=torch.zeros(2, dtype=torch.long)))


def test_nested_head_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="use_z_c"):
        _head(_full_dims(), GeneEffectBlockConfig(use_z_c=False))
    with pytest.raises(ValueError, match="factor_rank"):
        GeneEffectNestedHead(
            _full_dims(), GeneEffectBlockConfig(), n_genes=4, factor_rank=0, dropout=0
        )
    with pytest.raises(ValueError, match="dropout"):
        _head(_full_dims(), dropout=1.0)


def test_disabling_a_block_changes_param_count_and_forward_result() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    full_head = _live(_head(dims))
    torch.manual_seed(0)
    no_esm2_head = _live(_head(dims, GeneEffectBlockConfig(use_e_g=False)))
    full_params = sum(p.numel() for p in full_head.parameters())
    reduced_params = sum(p.numel() for p in no_esm2_head.parameters())
    assert reduced_params < full_params
    assert list(no_esm2_head.encoders) == ["delta_proj", "s", "q_sc"]

    torch.manual_seed(1)
    inputs = _full_inputs(5, dims)
    reduced_inputs = dict(inputs)
    del reduced_inputs["e_g"]
    with torch.no_grad():
        full_out = full_head(**inputs)
        reduced_out = no_esm2_head(**reduced_inputs)
    assert full_out.shape == reduced_out.shape
    assert not torch.allclose(full_out, reduced_out)


def test_virtual_cell_ablation_disables_delta_proj_and_s_together() -> None:
    dims = _full_dims()
    head = _live(_head(dims, NO_STATE))
    assert list(head.encoders) == ["q_sc", "e_g"]
    out = head(**_without_state(_full_inputs(4, dims)))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "name", ["delta_proj", "s", "hvg_panel_mask", "own_gene_shift_mask"]
)
def test_head_without_state_rejects_every_state_tensor(name) -> None:
    dims = _full_dims()
    head = _head(dims, NO_STATE)
    full = _full_inputs(3, dims)
    inputs = _without_state(full)
    inputs[name] = full[name]
    with pytest.raises(ValueError, match=name):
        head(**inputs)


def test_forward_rejects_tensor_for_disabled_block() -> None:
    dims = _full_dims()
    head = _head(dims, GeneEffectBlockConfig(use_e_g=False))
    inputs = _full_inputs(3, dims)
    with pytest.raises(ValueError, match="e_g"):
        head(**inputs)


def test_forward_requires_tensor_for_enabled_block() -> None:
    dims = _full_dims()
    head = _head(dims)
    inputs = _full_inputs(3, dims)
    del inputs["z_c"]
    with pytest.raises(ValueError, match="z_c"):
        head(**inputs)


def test_forward_requires_masks_for_enabled_blocks() -> None:
    dims = _full_dims()
    head = _head(dims)
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
    from a naive zero-fill."""
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims))
    inputs = _full_inputs(6, dims)
    inputs["q_sc_mask"] = torch.zeros(6, dtype=torch.bool)
    with torch.no_grad():
        out_a = head(**dict(inputs, q_sc=torch.zeros(6, dims.q_sc)))
        out_b = head(**dict(inputs, q_sc=torch.full((6, dims.q_sc), 1e6)))
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_masked_missing_own_gene_shift_never_leaks_into_output() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims))
    inputs = _full_inputs(6, dims)
    inputs["own_gene_shift_mask"] = torch.zeros(6, dtype=torch.bool)
    s_a, s_b = inputs["s"].clone(), inputs["s"].clone()
    s_a[:, -1] = 0.0
    s_b[:, -1] = 999.0
    with torch.no_grad():
        out_a = head(**dict(inputs, s=s_a))
        out_b = head(**dict(inputs, s=s_b))
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_mask_bit_itself_changes_output_between_present_and_missing() -> None:
    """The mask channel is wired into the correction, not present-but-ignored."""
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims))
    inputs = _full_inputs(6, dims)
    inputs["q_sc"] = torch.zeros(6, dims.q_sc)
    with torch.no_grad():
        out_present = head(**dict(inputs, q_sc_mask=torch.ones(6, dtype=torch.bool)))
        out_missing = head(**dict(inputs, q_sc_mask=torch.zeros(6, dtype=torch.bool)))
    assert not torch.allclose(out_present, out_missing)


def test_dropout_is_off_in_eval() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _live(_head(dims, dropout=0.5))
    inputs = _full_inputs(64, dims)
    with torch.no_grad():
        head.eval()
        torch.testing.assert_close(head(**inputs), head(**inputs), rtol=0, atol=0)
        head.train()
        assert not torch.allclose(head(**inputs), head(**inputs))


def test_head_has_no_cell_line_identity_parameter() -> None:
    """No per-line embedding table anywhere: the only lookup is the gene
    table, so the head accepts any batch size without a line vocabulary."""
    dims = _full_dims()
    head = _head(dims)
    tables = [m for m in head.modules() if isinstance(m, torch.nn.Embedding)]
    assert tables == [head.gene_embedding]
    assert head.gene_embedding.num_embeddings == N_GENES
    with torch.no_grad():
        out_small = head(**_full_inputs(2, dims))
        out_large = head(**_full_inputs(50, dims))
    assert torch.isfinite(out_small).all()
    assert torch.isfinite(out_large).all()


def test_head_backward_populates_finite_grads() -> None:
    torch.manual_seed(0)
    dims = _full_dims()
    head = _head(dims)
    head(**_full_inputs(5, dims)).sum().backward()
    for p in head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    # The zero-initialised output layers still learn from the first step.
    assert head.context_residual.out.weight.grad.abs().sum() > 0
    assert head.correction[1].out.weight.grad.abs().sum() > 0


def test_mlp_keeps_its_gene_free_interface() -> None:
    dims = _full_dims()
    mlp = GeneEffectMLP(dims, hidden=16)
    assert mlp.input_width == 6 + (4 + 2) + (3 + 1) + 5 + 4
    inputs = _blocks_only(_full_inputs(3, dims))
    assert mlp(**inputs).shape == (3,)


# The joint model on a prepared fixture with a tiny STATE (2000 HVGs, as STATE's
# response features require).

HVG = 2000


def _released_state(path: Path, hvg_order) -> Path:
    from src.model.state import build_state

    hparams = dict(
        input_dim=HVG,
        hidden_dim=8,
        output_dim=HVG,
        pert_dim=6,
        batch_dim=3,
        batch_encoder=True,
        cell_set_len=4,
        predict_residual=True,
        output_space="gene",
        embed_key="X_hvg",
        gene_names=list(hvg_order),
        transformer_backbone_kwargs={
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 2,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        },
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(1)
    torch.save(
        {"hyper_parameters": hparams, "state_dict": build_state(hparams).state_dict()},
        path,
    )
    return path


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    pytest.importorskip("state.tx.models.state_transition")
    from src.data.prepared import load_inputs
    from tests.test_joint_data import make_prepared_fixture

    root = tmp_path_factory.mktemp("joint_model")
    config = make_prepared_fixture(root, hvg_width=HVG)
    inputs = load_inputs(config)
    config["paths"]["state_checkpoint"] = str(
        _released_state(root / "released.ckpt", inputs.hvg_order)
    )
    config["model"] = {
        "cell_sentence_len": 4,
        "esm2_adapter_hidden": 4,
        "head_blocks": dataclasses.asdict(GeneEffectBlockConfig()),
        "factor_rank": RANK,
        "context_components": config["model"]["context_components"],
        "dropout": 0.1,
    }
    return config, inputs


def _model(prepared, blocks: GeneEffectBlockConfig = GeneEffectBlockConfig()):
    from src.model.initialization import build_joint_model
    from src.model.normalization import fit_startup_standardizer

    config, inputs = prepared
    config = dict(
        config, model=dict(config["model"], head_blocks=dataclasses.asdict(blocks))
    )
    torch.manual_seed(0)
    model = build_joint_model(config, inputs)
    fit_startup_standardizer(model, inputs, batch_size=4)
    return model.eval()


def _train_batch(inputs):
    from src.data.datasets import DependencyDataset

    dataset = DependencyDataset(inputs, "train")
    return dataset.collate(range(len(dataset)))


BLOCK_SETTINGS = {
    "all blocks": GeneEffectBlockConfig(),
    "no STATE": NO_STATE,
    "delta_proj without s": GeneEffectBlockConfig(use_s=False),
    "context only": CONTEXT_ONLY,
}


@pytest.mark.parametrize("setting", BLOCK_SETTINGS)
def test_startup_standardizer_sees_exactly_the_enabled_blocks(prepared, setting):
    blocks = BLOCK_SETTINGS[setting]
    model = _model(prepared, blocks)
    enabled = {
        name
        for name in ("delta_proj", "s", "q_sc", "e_g")
        if getattr(blocks, f"use_{name}")
    }
    assert set(model.standardizer.to_state()["blocks"]) == enabled
    assert model.uses_state == (blocks.use_delta_proj or blocks.use_s)
    with torch.no_grad():
        delta_hat = model(_train_batch(prepared[1]).conditions).delta_hat
    assert torch.isfinite(delta_hat).all()


def test_model_without_state_never_calls_state(prepared, monkeypatch):
    import src.model.geneeffect as geneeffect
    from src.data.batches import ResponseForwardBatch

    calls = []
    real = geneeffect.predict_bags

    def spy(*args, **kwargs):
        calls.append(args[2])
        return real(*args, **kwargs)

    monkeypatch.setattr(geneeffect, "predict_bags", spy)
    _model(prepared)
    assert calls, "the spy must see the STATE model's calls"

    calls.clear()
    model = _model(prepared, NO_STATE)
    hook = model.backbone.register_forward_pre_hook(
        lambda *args: calls.append("backbone")
    )
    conditions = _train_batch(prepared[1]).conditions
    features = model.condition_features(conditions)
    output = model(conditions)
    hook.remove()
    assert not calls
    assert not model.uses_state
    assert features.delta_proj is None and features.s is None
    assert features.hvg_panel_mask is None and features.own_gene_shift_mask is None
    features.validate()
    assert output.delta_hat.shape == (conditions.batch_size,)
    with pytest.raises(ValueError, match="STATE"):
        model(conditions, ResponseForwardBatch(conditions.basal_hvg, conditions.genes))


def test_residual_scale_multiplies_the_head_per_gene(prepared):
    config, inputs = prepared
    scale = pd.Series([0.5, 2.0, 4.0], index=list(inputs.genes))
    scaled = dataclasses.replace(inputs, residual_scale=scale)
    model = _model((config, scaled), NO_STATE)
    np.testing.assert_array_equal(
        model.residual_scale.numpy(), np.array([0.5, 2.0, 4.0], dtype=np.float32)
    )
    conditions = _train_batch(scaled).conditions
    with torch.no_grad():
        head = model.forward_features(
            model.condition_features(conditions), conditions.gene_index
        )
        delta_hat = model(conditions).delta_hat
    row_scale = torch.tensor(
        [scale[gene] for gene in conditions.genes], dtype=torch.float32
    )
    torch.testing.assert_close(delta_hat, row_scale * head)


@pytest.mark.parametrize("setting", ["all blocks", "no STATE"])
def test_build_save_restore_round_trip(prepared, setting):
    from src.model.initialization import restore_joint_model

    _, inputs = prepared
    model = _model(prepared, BLOCK_SETTINGS[setting])
    architecture = model.architecture
    assert architecture["genes"] == list(inputs.genes)
    assert architecture["head"]["n_genes"] == len(inputs.genes)
    assert architecture["head"]["factor_rank"] == RANK
    assert architecture["head"]["dropout"] == 0.1
    assert architecture["head"]["dims"]["z_c"] == inputs.context_pca.n_components
    saved = {
        "architecture": architecture,
        "projection_state": model.projection.to_state(),
        "normalization_state": model.standardizer.to_state(),
        "model_state": model.state_dict(),
    }
    restored = restore_joint_model(saved, inputs).eval()
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    conditions = _train_batch(inputs).conditions
    with torch.no_grad():
        torch.testing.assert_close(
            restored(conditions).delta_hat, model(conditions).delta_hat
        )

    reordered = dict(
        saved, architecture=dict(architecture, genes=architecture["genes"][::-1])
    )
    with pytest.raises(ValueError, match="gene order"):
        restore_joint_model(reordered, inputs)
    head = dict(architecture["head"], dims=dict(architecture["head"]["dims"], z_c=2))
    wider = dict(saved, architecture=dict(architecture, head=head))
    with pytest.raises(ValueError, match="context components"):
        restore_joint_model(wider, inputs)


def test_context_reaches_the_head_as_unstandardized_pca_scores(prepared):
    from src.data.context_pca import pooled_context
    from src.data.datasets import DependencyDataset

    _, inputs = prepared
    model = _model(prepared)
    assert "z_c" not in model.standardizer.to_state()["blocks"]
    dataset = DependencyDataset(inputs, "train")
    conditions = dataset.collate(range(len(dataset))).conditions
    expected = inputs.context_pca.transform(
        np.stack(
            [pooled_context(inputs.lines[m].controls_tx1) for m in dataset.model_ids]
        )
    )
    np.testing.assert_allclose(conditions.z_c.numpy(), expected, rtol=1e-6, atol=1e-6)
    seen = {}
    hook = model.head.register_forward_pre_hook(
        lambda module, args, kwargs: seen.update(kwargs), with_kwargs=True
    )
    with torch.no_grad():
        model(conditions)
    hook.remove()
    torch.testing.assert_close(seen["z_c"], conditions.z_c, rtol=0, atol=0)
    standardized = model.standardizer.transform("q_sc", conditions.q_sc)
    torch.testing.assert_close(seen["q_sc"], standardized, rtol=0, atol=0)

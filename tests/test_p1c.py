import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


def test_bias_decomposition_splits_shared_from_condition_specific():
    from src.eval.p1c_tier0 import bias_decomposition

    observed = np.zeros((4, 3))
    predicted = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0], [1.0, -2.0, 0]])
    result = bias_decomposition(predicted, observed)
    # mean_g ||e||^2 = (1 + 1 + 5 + 5)/4 = 3 ; ||ē||^2 = 1 ; residual = 2
    assert result["total_mse"] == pytest.approx(3.0 / 3)  # per-coordinate mean
    assert result["shared_bias"] == pytest.approx(1.0 / 3)
    assert result["condition_specific"] == pytest.approx(2.0 / 3)
    assert result["shared_bias_fraction"] == pytest.approx(1 / 3)
    assert result["n"] == 4


def test_decompose_effects_reads_npz_by_method_and_anchor(tmp_path):
    from src.eval.p1c_tier0 import decompose_effects

    keys = np.array([("model", "A", "G"), ("model", "A", "H"), ("no_change", "A", "G")])
    predicted = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    observed = np.zeros((3, 2))
    np.savez(
        tmp_path / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    frame = decompose_effects(tmp_path / "effects.npz", method="model")
    assert frame.model_id.tolist() == ["A"] and frame.n.tolist() == [2]
    assert frame.shared_bias_fraction.iloc[0] == pytest.approx(1.0)


def test_matched_coverage_restricts_references_to_perturbation_mean_support():
    from src.eval.p1c_tier0 import matched_coverage

    rows = []
    for gene, covered in (("G", True), ("H", False)):
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="no_change",
                response_loss=1.0,
            )
        )
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="global_mean",
                response_loss=2.0,
            )
        )
        if covered:
            rows.append(
                dict(
                    role="val",
                    model_id="A",
                    panel="all",
                    gene=gene,
                    method="perturbation_mean",
                    response_loss=3.0,
                )
            )
    frame = matched_coverage(pd.DataFrame(rows))
    row = frame[(frame.model_id == "A") & (frame.reference == "no_change")].iloc[0]
    assert row.covered == 1 and row.total == 2
    assert row.perturbation_mean_loss == 3.0 and row.reference_loss_on_covered == 1.0


def test_cross_context_pearson_is_undefined_for_constant_effect_references():
    from src.eval.p1b import cross_context

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
    }
    # Identical predicted effects in both anchors -> zero difference up to float noise.
    noise = np.array([1e-7, -1e-7, 0.0])
    effects = {
        ("global_mean", "a", "G"): (
            np.array([1.0, 2.0, 3.0]) + noise,
            np.array([1.0, 0.0, 0.0]),
        ),
        ("global_mean", "b", "G"): (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
    }
    frame = pd.DataFrame({"method": ["global_mean"]})
    result = cross_context(
        frame,
        effects,
        {**bundle, "splits": {"train": [], "val": [0, 1], "external": [2]}},
    )
    assert len(result) == 1 and np.isnan(result.effect_difference_pearson.iloc[0])


def _build_tier0_fixture(tmp_path):
    from src.data.gene_splits import sha256_file

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    response_cache_dir = tmp_path / "cache"
    (response_cache_dir / "response_targets").mkdir(parents=True)
    pd.DataFrame({"model_id": ["a", "a", "b"], "n_cells": [100, 120, 90]}).to_parquet(
        response_cache_dir / "response_targets" / "metadata.parquet"
    )

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
        "controls": {
            "a": {"hvg": np.zeros((5, 3))},
            "b": {"hvg": np.zeros((5, 3))},
        },
        "response_cache": str(response_cache_dir),
    }
    torch.save(bundle, prepared / "bundle.pt")
    manifest = {
        "bundle_sha256": sha256_file(prepared / "bundle.pt"),
        "source_missing_genes": {"a": [], "b": []},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))

    runs = tmp_path / "runs"
    export_dir = runs / "evaluation" / "B-init" / "internal"
    export_dir.mkdir(parents=True)
    methods = ("model", "no_change", "global_mean", "perturbation_mean")
    conditions = pd.DataFrame(
        [
            dict(
                role="val",
                model_id=anchor,
                panel="all",
                gene="G",
                method=method,
                response_loss=1.0,
            )
            for anchor in ("a", "b")
            for method in methods
        ]
    )
    conditions.to_parquet(export_dir / "conditions.parquet", index=False)
    keys = np.array(
        [(method, anchor, "G") for anchor in ("a", "b") for method in methods]
    )
    predicted = np.tile(np.array([1.0, 0.0]), (len(keys), 1))
    observed = np.zeros((len(keys), 2))
    np.savez(
        export_dir / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    (export_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "state": "B-init",
                "bundle": manifest["bundle_sha256"],
            }
        )
    )
    return prepared, runs, export_dir, manifest


def test_run_tier0_verifies_bundle_and_evaluation_identity_and_writes_outputs(tmp_path):
    from src.eval.p1c_tier0 import run_tier0

    prepared, runs, export_dir, manifest = _build_tier0_fixture(tmp_path)
    original_bundle_bytes = (prepared / "bundle.pt").read_bytes()

    out_dir = tmp_path / "out"
    result = run_tier0(runs, prepared, out_dir)
    assert result == out_dir
    for name in (
        "bias_decomposition.csv",
        "matched_coverage.csv",
        "anchor_audit.csv",
        "cross_context_guarded.csv",
        "tier0.md",
    ):
        assert (out_dir / name).exists()

    # Tampering with the prepared bundle after preparation must be caught.
    (prepared / "bundle.pt").write_bytes(original_bundle_bytes + b"\x00")
    with pytest.raises(ValueError, match="identity changed"):
        run_tier0(runs, prepared, out_dir)
    (prepared / "bundle.pt").write_bytes(original_bundle_bytes)

    # An export recorded against a different bundle must also be rejected.
    (export_dir / "evaluation.json").write_text(
        json.dumps({"status": "completed", "state": "B-init", "bundle": "stale-hash"})
    )
    with pytest.raises(ValueError):
        run_tier0(runs, prepared, out_dir)


def test_fold_membership_rotates_one_anchor_out():
    from src.data.p1c import FOLDS, fold_membership, ALL_ANCHORS

    for fold, held in FOLDS.items():
        sources, external = fold_membership(fold)
        assert external == held and held not in sources
        assert tuple(a for a in ALL_ANCHORS if a != held) == sources
    with pytest.raises(KeyError):
        fold_membership("mcf7")


def test_response_view_input_layouts_concatenate_hvg_then_tx1():
    from types import SimpleNamespace
    from src.data.p1b import build_snapshot, ResponseView
    from src.data.response_cache import ResponseTargetsCache

    keys = [(a, g) for a in ("a", "b", "c", "j") for g in ("G", "H")]
    cache = ResponseTargetsCache(
        tuple(a for a, g in keys),
        tuple(g for a, g in keys),
        np.ones((16, 3), dtype=np.float32),
        np.arange(0, 17, 2),
        pd.DataFrame(),
    )
    inputs = SimpleNamespace(
        response_targets=cache,
        response_holdout=frozenset((a, "H") for a in ("a", "b", "c")),
        hvg_order=("G", "H", "I"),
        lines={
            a: SimpleNamespace(
                controls_tx1=np.full((2, 5), 2.0, dtype=np.float32),
                basal_hvg=np.full((2, 3), 1.0, dtype=np.float32),
            )
            for a in ("a", "b", "c", "j")
        },
    )
    bundle = build_snapshot(
        inputs, [0, 1], {"G"}, anchors=("a", "b", "c"), external="j"
    )
    for layout, width, first in (("tx1", 5, 2.0), ("hvg", 3, 1.0), ("hvg_tx1", 8, 1.0)):
        batch = ResponseView(bundle, cache, input_layout=layout).batch([0])
        assert batch.controls_tx1[0].shape == (2, width)
        assert float(batch.controls_tx1[0][0, 0]) == first
        assert batch.control_hvg[0].shape == (2, 3)
    with pytest.raises(ValueError):
        ResponseView(bundle, cache, input_layout="bogus")


def _bundle_matching_p1b_expectations(anchors, external):
    """Synthetic bundle whose check_membership() result equals P1B_EXPECTATIONS."""
    train_counts = (16399, 5481, 5481)  # sums to 27361; largest_pool == 16399
    keys = []
    for anchor, n in zip(anchors, train_counts):
        keys.extend((anchor, f"g{i}") for i in range(n))
    keys.extend((anchors[0], f"v{i}") for i in range(3047))
    keys.extend((external, f"e{i}") for i in range(2377))
    n_train = sum(train_counts)
    splits = {
        "train": list(range(0, n_train)),
        "val": list(range(n_train, n_train + 3047)),
        "external": list(range(n_train + 3047, n_train + 3047 + 2377)),
    }
    panels = {
        f"external/{external}/seen": {"indices": list(range(2373))},
        f"external/{external}/unseen": {"indices": list(range(4))},
        f"external/{external}/native_common": {"indices": list(range(2006))},
        f"external/{external}/native_all": {"indices": list(range(2009))},
    }
    return {"keys": keys, "splits": splits, "panels": panels}


def test_check_membership_none_expectations_checks_total_and_anchor_set():
    from src.experiments.p1b_preparation import check_membership
    from src.data.p1b import SOURCE_ANCHORS, EXTERNAL_ANCHOR

    def bundle_with(total):
        return {
            "keys": [(SOURCE_ANCHORS[0], "g")] * total,
            "splits": {
                "train": list(range(total - 2)),
                "val": [total - 2],
                "external": [total - 1],
            },
            "panels": {},
        }

    good = bundle_with(32785)
    result = check_membership(good, SOURCE_ANCHORS, EXTERNAL_ANCHOR, None)
    assert result["counts"] == {"train": 32783, "val": 1, "external": 1}
    assert result["external_panels"] == {
        "seen": 0,
        "unseen": 0,
        "native_common": 0,
        "native_all": 0,
    }

    with pytest.raises(ValueError, match="unexpected total condition count"):
        check_membership(bundle_with(100), SOURCE_ANCHORS, EXTERNAL_ANCHOR, None)

    with pytest.raises(ValueError, match="approved"):
        check_membership(good, ("x", "y", "z"), "w", None)

    empty_external = bundle_with(32785)
    empty_external["splits"]["train"] = list(range(32785))
    empty_external["splits"]["val"] = []
    empty_external["splits"]["external"] = []
    with pytest.raises(ValueError, match="external"):
        check_membership(empty_external, SOURCE_ANCHORS, EXTERNAL_ANCHOR, None)


def test_check_membership_matches_expectations_exactly_or_raises_on_first_mismatch():
    from src.experiments.p1b_preparation import check_membership, P1B_EXPECTATIONS
    from src.data.p1b import SOURCE_ANCHORS, EXTERNAL_ANCHOR

    tiny = {
        "keys": [(SOURCE_ANCHORS[0], "g")] * 5,
        "splits": {"train": [0, 1, 2], "val": [3], "external": [4]},
        "panels": {},
    }
    with pytest.raises(ValueError, match="counts"):
        check_membership(tiny, SOURCE_ANCHORS, EXTERNAL_ANCHOR, P1B_EXPECTATIONS)

    matching = _bundle_matching_p1b_expectations(SOURCE_ANCHORS, EXTERNAL_ANCHOR)
    result = check_membership(
        matching, SOURCE_ANCHORS, EXTERNAL_ANCHOR, P1B_EXPECTATIONS
    )
    assert result == P1B_EXPECTATIONS


def test_prepare_fold_writes_fold_json_and_checks_reference_coordinates(
    tmp_path, monkeypatch
):
    from pathlib import Path
    import src.experiments.p1c_preparation as p1c_prep
    from src.data.p1c import fold_membership
    from src.experiments.p1b_preparation import P1B_EXPECTATIONS

    calls = []

    def fake_prepare_bundle(checkpoint, directory, *, anchors, external, expectations):
        calls.append((checkpoint, anchors, external, expectations))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"coordinates": [0, 1, 2]}))

    monkeypatch.setattr(p1c_prep, "prepare_bundle", fake_prepare_bundle)

    jurkat_dir = tmp_path / "jurkat"
    p1c_prep.prepare_fold("ckpt", "jurkat", jurkat_dir)
    sources, external = fold_membership("jurkat")
    assert json.loads((jurkat_dir / "fold.json").read_text()) == {
        "fold": "jurkat",
        "external": external,
        "sources": list(sources),
    }
    assert calls[0] == ("ckpt", sources, external, P1B_EXPECTATIONS)

    k562_dir = tmp_path / "k562"
    p1c_prep.prepare_fold("ckpt", "k562", k562_dir)
    assert calls[1][3] is None

    # A reference manifest with matching coordinates is accepted.
    p1c_prep.prepare_fold(
        "ckpt",
        "hepg2",
        tmp_path / "hepg2",
        reference_manifest=jurkat_dir / "manifest.json",
    )

    # A reference manifest with different coordinates is rejected.
    def fake_prepare_bundle_mismatch(
        checkpoint, directory, *, anchors, external, expectations
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"coordinates": [9, 9, 9]}))

    monkeypatch.setattr(p1c_prep, "prepare_bundle", fake_prepare_bundle_mismatch)
    with pytest.raises(ValueError, match="fold coordinates differ"):
        p1c_prep.prepare_fold(
            "ckpt",
            "hct116",
            tmp_path / "hct116",
            reference_manifest=jurkat_dir / "manifest.json",
        )


def _build_p1c_variant_fixture(tmp_path):
    """Tiny real-STATE template plus a native basal-encoder state for P1-C variants.

    Mirrors ``tests/test_p1b.py``'s real-STATE fixture: Tx1 width 2560, HVG width
    2000, hidden 8, pert_dim 2, cell_set_len 4. Builds the native model FIRST
    (``input_dim=2000``, standing in for the released checkpoint a native
    evaluation would use), then derives the template's own state-model weights
    from it: every key copied verbatim except ``basal_encoder.0.weight``, which
    is freshly (randomly) initialised at the wider 2560 Tx1 width -- mirroring
    how a real P1-B ``B-init.pt`` actually originates (warm-started, shape
    filtered, from the same released checkpoint a native evaluation uses). This
    makes the template's transformer backbone / pert_encoder / project_out /
    basal_encoder bias bit-identical to the native model's own, so a
    native-basal variant's output can be checked against an independently
    reconstructed native forward, not merely against another P1-C variant.
    """
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    from src.model.initialization import _suppress_checkpoint_output
    from test_joint_training import tiny_training_config, fresh_model

    config = tiny_training_config(tmp_path / "inputs")
    joint, inputs = fresh_model(config)

    native_hparams = copy.deepcopy(joint.architecture["state_hparams"])
    native_hparams["input_dim"] = 2000
    torch.manual_seed(1)
    with _suppress_checkpoint_output():
        native_model = StateTransitionPerturbationModel(**copy.deepcopy(native_hparams))
    native_state = dict(native_model.state_dict())

    hidden_dim = int(joint.architecture["state_hparams"]["hidden_dim"])
    torch.manual_seed(2)
    template_state_model = dict(native_state)
    template_state_model["basal_encoder.0.weight"] = torch.randn(hidden_dim, 2560)
    model_state = {
        f"state_adapter.state_model.{name}": tensor
        for name, tensor in template_state_model.items()
    }
    model_state.update(
        {
            f"perturbations.{name}": tensor
            for name, tensor in joint.backbone.perturbations.state_dict().items()
        }
    )
    template = {
        "config": config,
        "preprocessing": inputs.preprocessing_state(),
        "architecture": joint.architecture,
        "model_state": model_state,
    }
    return template, inputs, native_state


def _rebuild_native_model(template, native_state):
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    from src.model.initialization import _suppress_checkpoint_output

    hparams = copy.deepcopy(template["architecture"]["state_hparams"])
    hparams["input_dim"] = 2000
    with _suppress_checkpoint_output():
        model = StateTransitionPerturbationModel(**copy.deepcopy(hparams))
    model.load_state_dict(native_state, strict=True)
    model.eval()
    return model


def test_null_subtracted_variant_starts_at_exact_no_change_and_dedupes_null(tmp_path):
    from src.model.p1c import build_variant
    from src.model.response import predict_bags

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    backbone, report = build_variant(template, "V1", expected_count=None)
    assert report["input_layout"] == "hvg_tx1"

    gene_a, gene_b = inputs.esm2_symbols[0], inputs.esm2_symbols[1]
    control = torch.randn(4, 2000 + 2560)
    control_hvg = control[:, :2000]

    (pred,) = predict_bags(backbone, [control], [gene_a], seed=0)
    torch.testing.assert_close(pred, control_hvg, atol=1e-6, rtol=0)

    calls: list[int] = []
    original = backbone.state_adapter.forward_condition_chunks

    def counting(control_chunks, perturbations, genes, batch_index_chunks):
        calls.append(len(control_chunks))
        return original(control_chunks, perturbations, genes, batch_index_chunks)

    backbone.state_adapter.forward_condition_chunks = counting
    try:
        predict_bags(backbone, [control, control], [gene_a, gene_b], seed=0)
    finally:
        backbone.state_adapter.forward_condition_chunks = original
    # The perturbed forward processes both conditions; the null forward
    # dedupes the two identical bags into a single representative.
    assert calls == [2, 1]


def test_null_outputs_dedup_respects_batch_index_identity(tmp_path):
    from src.model.p1c import build_variant

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    backbone, _ = build_variant(template, "V1", expected_count=None)

    calls: list[int] = []
    original = backbone.state_adapter.forward_condition_chunks

    def counting(control_chunks, perturbations, genes, batch_index_chunks):
        calls.append(len(control_chunks))
        return original(control_chunks, perturbations, genes, batch_index_chunks)

    backbone.state_adapter.forward_condition_chunks = counting
    tx1_chunk = torch.randn(4, 2560)
    batch_a = torch.zeros(4, dtype=torch.long)
    batch_b = torch.ones(4, dtype=torch.long)
    try:
        # Same underlying tensor object (identical data_ptr/shape/stride) for
        # both conditions, but different batch-index chunks: must NOT dedupe.
        backbone.null_outputs((tx1_chunk, tx1_chunk), (batch_a, batch_b))
    finally:
        backbone.state_adapter.forward_condition_chunks = original
    assert calls == [2]


def test_null_subtracted_backbone_forbids_train_mode(tmp_path):
    from src.model.p1c import build_variant
    from src.model.response import predict_bags

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    backbone, _ = build_variant(template, "V1", expected_count=None)
    backbone.train()
    control = torch.randn(4, 2000 + 2560)
    with pytest.raises(RuntimeError, match="eval mode"):
        predict_bags(backbone, [control], [inputs.esm2_symbols[0]], seed=0)


def test_native_basal_variants_load_released_encoder_and_zero_context(tmp_path):
    from src.model.p1c import build_variant
    from src.model.response import predict_bags

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    backbone, report = build_variant(
        template, "V2", native_state=native_state, expected_count=None
    )
    state = backbone.state_adapter.state_model
    assert state.input_dim == 2000 + 2560
    assert report["state_input_dim"] == 2000 + 2560
    assert torch.equal(
        state.basal_encoder.native[0].weight, native_state["basal_encoder.0.weight"]
    )
    assert torch.equal(
        state.basal_encoder.native[0].bias, native_state["basal_encoder.0.bias"]
    )
    assert torch.equal(
        state.basal_encoder.context.weight,
        torch.zeros_like(state.basal_encoder.context.weight),
    )

    gene = inputs.esm2_symbols[0]
    n_cells = 4  # == cell_set_len, so predict_bags needs no padding.
    hvg_cells = torch.randn(n_cells, 2000)
    tx1_cells = torch.randn(n_cells, 2560)
    control = torch.cat([hvg_cells, tx1_cells], dim=1)

    # Independent reference: a separately reconstructed native STATE model
    # (same weights as native_state, no relation to build_variant's own
    # construction path) run directly with a zero perturbation and batch 0.
    # This is only bit-comparable because the fixture derives the template's
    # transformer/pert_encoder/project_out/basal-bias from this same native
    # model (see _build_p1c_variant_fixture) -- V2's other-than-basal-weight
    # parameters are therefore identical to the native model's, not merely
    # structurally similar.
    native_model = _rebuild_native_model(template, native_state)
    with torch.no_grad():
        reference = native_model(
            {
                "ctrl_cell_emb": hvg_cells,
                "pert_emb": torch.zeros(n_cells, 2),
                "pert_name": [gene] * n_cells,
                "batch": torch.zeros(n_cells, dtype=torch.long),
            },
            padded=True,
        )

    (v2_pred,) = predict_bags(backbone, [control], [gene], seed=0)
    torch.testing.assert_close(v2_pred, reference, atol=1e-6, rtol=0)

    null_backbone, null_report = build_variant(
        template, "V2-null", native_state=native_state, expected_count=None
    )
    assert null_report["state_input_dim"] == 2000
    (null_pred,) = predict_bags(null_backbone, [hvg_cells], [gene], seed=0)
    torch.testing.assert_close(null_pred, reference, atol=1e-6, rtol=0)


def test_variant_parameter_ownership_matches_table(tmp_path):
    from src.model.p1c import build_variant

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    native_map = {
        "G": torch.tensor([1.0, 0.0]),
        "non-targeting": torch.tensor([0.0, 1.0]),
    }

    prefix = "state_adapter.state_model."
    adapter_names = {
        "perturbations.adapter.net.0.weight",
        "perturbations.adapter.net.0.bias",
        "perturbations.adapter.net.2.weight",
        "perturbations.adapter.net.2.bias",
    }
    expected_trainable = {
        "V0": adapter_names | {prefix + "basal_encoder.0.weight"},
        "V1": adapter_names | {prefix + "basal_encoder.0.weight"},
        "V2-null": adapter_names,
        "V2": adapter_names | {prefix + "basal_encoder.context.weight"},
        "V3": adapter_names | {prefix + "basal_encoder.context.weight"},
    }

    for variant, expected in expected_trainable.items():
        kwargs = (
            {"native_state": native_state}
            if variant != "V0" and variant != "V1"
            else {}
        )
        backbone, report = build_variant(
            template, variant, expected_count=None, **kwargs
        )
        trainable = {name for name, p in backbone.named_parameters() if p.requires_grad}
        assert trainable == expected, variant
        all_names = {name for name, _ in backbone.named_parameters()}
        assert set(report["parameters"]) == all_names, variant
        assert all(
            entry["origin"]
            in {
                "inherited-template",
                "inherited-native",
                "new-template",
                "new-zero",
            }
            for entry in report["parameters"].values()
        ), variant

    backbone, report = build_variant(
        template,
        "N-native",
        native_state=native_state,
        native_map=native_map,
        expected_count=None,
    )
    assert report["trainable_count"] == 0
    assert not any(p.requires_grad for _, p in backbone.named_parameters())
    from src.model.p1c import parameter_groups

    with pytest.raises(ValueError):
        parameter_groups(backbone, lr=1e-4)


def test_build_variant_raises_on_trainable_count_mismatch(tmp_path):
    from src.model.p1c import build_variant

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    with pytest.raises(ValueError, match="trainable"):
        build_variant(template, "V0", expected_count=999)


def test_native_onehot_perturbations_and_batch_index():
    from src.model.p1c import NativeOneHotPerturbations, NativeBackbone
    from src.model.state import StateForwardAdapter

    onehot = {"G": torch.tensor([1.0, 0.0]), "non-targeting": torch.tensor([0.0, 1.0])}
    perturbations = NativeOneHotPerturbations(onehot)
    stacked = perturbations.forward_many(["non-targeting", "G"])
    torch.testing.assert_close(
        stacked, torch.stack([onehot["non-targeting"], onehot["G"]])
    )
    assert perturbations.has_embedding("G") and not perturbations.has_embedding("H")
    with pytest.raises(KeyError):
        perturbations.forward_many(["H"])

    captured: dict[str, torch.Tensor | None] = {}

    class DummyState(nn.Module):
        pert_dim = 2
        cell_sentence_len = 4
        batch_encoder = None

        def forward(self, batch, padded=True):
            del padded
            captured["batch"] = batch.get("batch")
            return torch.zeros(batch["ctrl_cell_emb"].shape[0], 3)

    state_model = DummyState()
    adapter = StateForwardAdapter(state_model)
    backbone = NativeBackbone(adapter, perturbations, batch_index=3)
    control = torch.zeros(4, 2000)
    backbone((control,), ("G",), (None,))
    assert captured["batch"] is not None
    assert torch.equal(captured["batch"], torch.full((4,), 3, dtype=torch.long))


def test_gradients_reach_interface_through_frozen_variants(tmp_path):
    from src.model.p1c import build_variant, parameter_groups
    from src.model.response import predict_bags

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    gene = inputs.esm2_symbols[0]
    prefix = "state_adapter.state_model."
    net0 = {"perturbations.adapter.net.0.weight", "perturbations.adapter.net.0.bias"}
    net2 = ("perturbations.adapter.net.2.weight", "perturbations.adapter.net.2.bias")
    control_width = {"tx1": 2560, "hvg": 2000, "hvg_tx1": 2000 + 2560}

    # At init the adapter's final layer is zeroed, so every gene's raw
    # perturbation vector is identically zero. Two exact cancellations follow:
    # (a) net.0 is "dead": its only path to the loss is
    # net.2(GELU(net.0(x))), and d(that)/d(net.0.*) is multiplied by
    # net.2.weight, which is the zero matrix, so the gradient is exactly zero
    # regardless of net.0's own value; (b) in the null-subtracted variants
    # (V1/V3) the perturbed and null branches evaluate the identical STATE
    # sub-computation (same inputs, same zero perturbation) through
    # basal_encoder/context, so their derivatives w.r.t. any parameter shared
    # by both branches cancel exactly in p - n.
    exactly_zero_at_init = {
        "V0": set(),
        "V1": net0 | {prefix + "basal_encoder.0.weight"},
        "V2-null": set(net0),
        "V2": set(net0),
        "V3": net0 | {prefix + "basal_encoder.context.weight"},
    }

    for variant, kwargs in (
        ("V0", {}),
        ("V1", {}),
        ("V2-null", {"native_state": native_state}),
        ("V2", {"native_state": native_state}),
        ("V3", {"native_state": native_state}),
    ):
        backbone, report = build_variant(
            template, variant, expected_count=None, **kwargs
        )
        width = control_width[report["input_layout"]]
        control = torch.randn(4, width)
        (pred,) = predict_bags(backbone, [control], [gene], seed=0)
        pred.sum().backward()

        zero_names = exactly_zero_at_init[variant]
        actual = dict(backbone.named_parameters())
        for name, p in actual.items():
            if not report["parameters"][name]["trainable"]:
                assert p.grad is None, (variant, name)
                continue
            assert p.grad is not None, (variant, name)
            assert torch.isfinite(p.grad).all(), (variant, name)
            if name in zero_names:
                assert torch.equal(p.grad, torch.zeros_like(p.grad)), (variant, name)
            else:
                assert p.grad.abs().max() > 0, (variant, name)

        # The adapter's final layer must always carry signal: it is the only
        # trainable tensor never cancelled by the null branch or deadened by
        # a zeroed downstream layer.
        for name in net2:
            assert actual[name].grad.abs().max() > 0, (variant, name)

        if not zero_names:
            continue

        # Confirm the zero gradients above are a degenerate artifact of the
        # exact-zero init, not permanently dead parameters: one optimizer
        # step moves net.2 away from zero, so a second backward picks up a
        # non-zero gradient on every previously-zero trainable tensor.
        optimizer = torch.optim.SGD(parameter_groups(backbone, lr=1.0))
        optimizer.step()
        backbone.zero_grad(set_to_none=True)
        control2 = torch.randn(4, width)
        (pred2,) = predict_bags(backbone, [control2], [gene], seed=0)
        pred2.sum().backward()
        for name in zero_names:
            grad = actual[name].grad
            assert grad is not None, (variant, name)
            assert grad.abs().max() > 0, (variant, name)


def test_zero_final_layer_requires_linear_final_layer():
    from src.model.p1c import zero_final_layer

    class NotAnAdapter:
        class _Inner:
            net = nn.Sequential(nn.Linear(2, 2), nn.GELU())  # final layer is GELU

        adapter = _Inner()

    with pytest.raises(ValueError, match="Linear"):
        zero_final_layer(NotAnAdapter())


def test_native_one_hot_perturbations_rejects_empty_map():
    from src.model.p1c import NativeOneHotPerturbations

    with pytest.raises(ValueError, match="non-empty"):
        NativeOneHotPerturbations({})


def test_build_variant_rejects_native_map_vector_with_wrong_pert_dim(tmp_path):
    from src.model.p1c import build_variant

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    bad_map = {"G": torch.tensor([1.0, 0.0, 0.0])}  # width 3, pert_dim is 2
    with pytest.raises(ValueError, match="width"):
        build_variant(
            template,
            "N-native",
            native_state=native_state,
            native_map=bad_map,
            expected_count=None,
        )


def _build_p1c_fold_fixture(tmp_path, *, fold="k562"):
    """Tiny prepared P1-C fold directory: real-STATE bundle/template plus a
    native checkpoint and a two-gene (+ non-targeting) one-hot map, mirroring
    ``tests/test_p1b.py::test_real_cli_export_retry_preserves_checkpoint``.
    """
    from src.data.p1b import build_snapshot
    from src.experiments.p1b_preparation import digest

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    # G0/G1 are real conditions on every anchor in this fixture (the response
    # holdout puts G0 in val, G1 in train); non-targeting is not a real
    # condition but is needed for the native-null substitution. Mirroring
    # ``prepare_bundle``, the bundle's own native_common/native_all panels are
    # built from this same (small) native vocabulary, not the full ESM2 set,
    # so every condition inside those panels is one the native model can
    # actually score.
    native_map = {
        "G0": torch.tensor([1.0, 0.0]),
        "G1": torch.tensor([0.0, 1.0]),
        "non-targeting": torch.tensor([0.5, 0.5]),
    }
    bundle = build_snapshot(
        inputs,
        list(range(1957)),
        set(native_map),
        anchors=inputs.response_anchors[:3],
        external=inputs.response_anchors[3],
    )
    bundle["response_cache"] = str(inputs.response_cache)
    root = inputs.response_cache / "response_targets"
    target_stat = (root / "target_cells.npy").stat()
    bundle["target_stat"] = {
        "size": target_stat.st_size,
        "mtime_ns": target_stat.st_mtime_ns,
    }
    bundle["cache_identity"] = {
        n: digest(root / n)
        for n in ("manifest.json", "metadata.parquet", "offsets.npy")
    }
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    torch.save(bundle, prepared / "bundle.pt")
    torch.save(template, prepared / "B-init.pt")

    native_checkpoint_path = tmp_path / "native.pt"
    torch.save({"state_dict": native_state}, native_checkpoint_path)
    native_map_path = tmp_path / "native_map.pt"
    torch.save(native_map, native_map_path)

    manifest = {
        "bundle_sha256": digest(prepared / "bundle.pt"),
        "model_files": {"B-init": digest(prepared / "B-init.pt")},
        "native_state_checkpoint": str(native_checkpoint_path),
        "native_map": str(native_map_path),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    (prepared / "status.json").write_text(json.dumps({"status": "completed"}))
    (prepared / "fold.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "external": inputs.response_anchors[3],
                "sources": list(inputs.response_anchors[:3]),
            }
        )
    )
    return prepared, inputs, native_state


def test_p1c_cli_parses_all_commands():
    from src.experiments.p1c import parser
    from src.model.p1c import VARIANTS

    p = parser()
    for stage in (
        "prepare",
        "tier0",
        "evaluate-native",
        "train",
        "evaluate",
        "compare",
    ):
        with pytest.raises(SystemExit) as exc:
            p.parse_args([stage, "--help"])
        assert exc.value.code == 0

    args = p.parse_args(["evaluate-native", "--prepared", "x", "--runs", "y"])
    assert args.batch_indices == [0]

    with pytest.raises(SystemExit):
        p.parse_args(
            [
                "train",
                "--prepared",
                "x",
                "--runs",
                "y",
                "--variant",
                "bogus",
                "--lr",
                "1e-4",
            ]
        )

    args = p.parse_args(
        ["train", "--prepared", "x", "--runs", "y", "--variant", "V1", "--lr", "1e-4"]
    )
    assert args.variant in VARIANTS


def test_train_variant_v1_on_tiny_real_state_records_init_check_and_blocks_after_external(  # noqa: E501
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    import src.experiments.p1c as p1c
    import src.model.p1c as p1c_model

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(p1c, "MAX_EPOCHS", 1)
    # The tiny fixture's real STATE model has far fewer parameters than the
    # production architecture; skip the fixed trainable-count table for it.
    monkeypatch.setitem(p1c_model._TRAINABLE_COUNTS, "V1", 20506)

    train_args = [
        "train",
        "--prepared",
        str(prepared),
        "--runs",
        str(runs),
        "--variant",
        "V1",
        "--lr",
        "1e-4",
        "--device",
        "cpu",
    ]
    p1c.main(train_args)

    training = json.loads((runs / "training.json").read_text())
    assert training["status"] == "completed"
    assert training["variant"] == "V1"
    assert training["fold"] == "k562"
    assert training["lr"] == 1e-4
    assert training["input_layout"] == "hvg_tx1"
    assert training["init_check"]["status"] == "passed"
    assert training["init_check"]["tolerance"] == 1e-6

    p1c.main(
        [
            "evaluate",
            "--prepared",
            str(prepared),
            "--runs",
            str(runs),
            "--external",
            "--device",
            "cpu",
        ]
    )
    assert (runs / "external_evaluation.json").exists()
    external_status = json.loads(
        (runs / "evaluation" / "external" / "evaluation.json").read_text()
    )
    assert external_status["status"] == "completed"

    with pytest.raises(ValueError, match="external evaluation has started"):
        p1c.main(train_args)


def test_evaluate_native_restricts_to_native_vocabulary(tmp_path):
    torch.set_num_threads(1)
    from src.experiments.p1c import main, restrict_to_native
    from src.experiments.p1b_preparation import open_bundle

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    runs = tmp_path / "runs"

    bundle, view, manifest = open_bundle(prepared, input_layout="hvg")
    native_genes = {"G0", "G1"}
    restricted = restrict_to_native(bundle, native_genes)
    for indices in restricted["splits"].values():
        assert all(bundle["keys"][i][1] in native_genes for i in indices)
    assert restricted["panels"]
    assert all(
        key.endswith("/native_common") or key.endswith("/native_all")
        for key in restricted["panels"]
    )

    main(
        [
            "evaluate-native",
            "--prepared",
            str(prepared),
            "--runs",
            str(runs),
            "--batch-indices",
            "0",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert (runs / "evaluation" / "N-native" / "internal" / "summary.csv").exists()
    assert (runs / "evaluation" / "N-native" / "external" / "summary.csv").exists()
    assert (runs / "evaluation" / "N-native-b1" / "internal" / "summary.csv").exists()
    assert (runs / "evaluation" / "N-native-b1" / "external" / "summary.csv").exists()
    assert (
        runs / "evaluation" / "N-native" / "internal" / "native_null.parquet"
    ).exists()
    assert (
        runs / "evaluation" / "N-native" / "external" / "native_null.parquet"
    ).exists()
    for name in ("N-native", "N-native-b1"):
        for role in ("internal", "external"):
            status = json.loads(
                (runs / "evaluation" / name / role / "evaluation.json").read_text()
            )
            assert status["status"] == "completed"
            assert status["variant"] == "N-native"
            assert status["fold"] == "k562"

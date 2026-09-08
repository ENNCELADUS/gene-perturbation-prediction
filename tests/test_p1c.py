import copy
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert "role" not in frame.columns


def test_decompose_effects_groups_by_role_when_conditions_are_given(tmp_path):
    from src.eval.p1c_tier0 import decompose_effects

    keys = np.array(
        [("model", "A", "G"), ("model", "A", "H"), ("model", "J", "G")],
    )
    predicted = np.array([[1.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    observed = np.zeros((3, 2))
    np.savez(
        tmp_path / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    conditions = pd.DataFrame(
        [
            dict(role="val", model_id="A", gene="G", panel="all", method="model"),
            dict(role="val", model_id="A", gene="H", panel="all", method="model"),
            dict(role="val", model_id="A", gene="H", panel="seen", method="model"),
            dict(role="external", model_id="J", gene="G", panel="all", method="model"),
        ]
    )
    frame = decompose_effects(
        tmp_path / "effects.npz", method="model", conditions=conditions
    )
    assert list(zip(frame.model_id, frame.role, frame.n)) == [
        ("A", "val", 2),
        ("J", "external", 1),
    ]
    assert frame.total_mse.tolist() == pytest.approx([0.5, 8.0])

    with pytest.raises(ValueError, match="no role"):
        decompose_effects(
            tmp_path / "effects.npz",
            method="model",
            conditions=conditions[conditions.model_id == "A"],
        )


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
    losses = {
        "model": 0.8,
        "no_change": 1.0,
        "global_mean": 2.0,
        "perturbation_mean": 1.5,  # worse than no_change on the covered gene
    }
    # Two roles in one export, the shape every P1-B export has: the held-out
    # anchor's error must not be averaged into the source anchors'.
    roles = {"a": "val", "b": "val", "j": "external"}
    conditions = pd.DataFrame(
        [
            dict(
                role=roles[anchor],
                model_id=anchor,
                panel="all",
                gene="G",
                method=method,
                response_loss=losses[method],
            )
            for anchor in ("a", "b", "j")
            for method in methods
        ]
    )
    conditions.to_parquet(export_dir / "conditions.parquet", index=False)
    keys = np.array(
        [(method, anchor, "G") for anchor in ("a", "b", "j") for method in methods]
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

    bias = pd.read_csv(out_dir / "bias_decomposition.csv")
    assert set(zip(bias.model_id, bias.role)) == {
        ("a", "val"),
        ("b", "val"),
        ("j", "external"),
    }
    coverage = pd.read_csv(out_dir / "matched_coverage.csv")
    assert (coverage.model_loss_on_covered == 0.8).all()
    tier0 = (out_dir / "tier0.md").read_text()
    assert "train-side perturbation-mean prior versus no-change" in tier0
    # One line per anchor for the prior, one for the state's own model ratio.
    assert tier0.count("perturbation_mean_loss=1.5 > no_change") == 3
    assert tier0.count("model/no_change on the covered set") == 3
    assert "model_loss=0.8, ratio=0.8" in tier0

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


def _build_p1c_variant_fixture(tmp_path, *, n_encoder_layers=1):
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

    architecture = copy.deepcopy(joint.architecture)
    # A released checkpoint may carry a multi-layer basal encoder; the whole
    # state model (native and template alike) is then built at that depth.
    architecture["state_hparams"]["n_encoder_layers"] = int(n_encoder_layers)
    native_hparams = copy.deepcopy(architecture["state_hparams"])
    native_hparams["input_dim"] = 2000
    torch.manual_seed(1)
    with _suppress_checkpoint_output():
        native_model = StateTransitionPerturbationModel(**copy.deepcopy(native_hparams))
    native_state = dict(native_model.state_dict())

    hidden_dim = int(architecture["state_hparams"]["hidden_dim"])
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
        "architecture": architecture,
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
    assert report["null_perturbation"] == "zero raw vector, same forward call"

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
    # Exactly one STATE call carries both branches (bf16 kernel selection must
    # not differ between them): two perturbed chunks plus one null
    # representative, the two identical bags having deduped into one.
    assert calls == [3]


def test_null_dedup_respects_batch_index_identity_in_one_call(tmp_path):
    from src.model.p1c import build_variant
    from src.model.response import predict_bags

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    backbone, _ = build_variant(template, "V1", expected_count=None)

    calls: list[int] = []
    original = backbone.state_adapter.forward_condition_chunks

    def counting(control_chunks, perturbations, genes, batch_index_chunks):
        calls.append(len(control_chunks))
        return original(control_chunks, perturbations, genes, batch_index_chunks)

    control = torch.randn(4, 2000 + 2560)
    batch_a = torch.zeros(4, dtype=torch.long)
    batch_b = torch.ones(4, dtype=torch.long)
    gene = inputs.esm2_symbols[0]

    backbone.state_adapter.forward_condition_chunks = counting
    try:
        # Same underlying tensor object (identical data_ptr/shape/stride) for
        # both conditions, but different batch-index chunks: must NOT dedupe,
        # so the single call carries 2 perturbed + 2 null chunks.
        backbone((control, control), (gene, gene), (batch_a, batch_b))
        # Identical batch indices dedupe to a single null representative.
        backbone((control, control), (gene, gene), (batch_a, batch_a))
    finally:
        backbone.state_adapter.forward_condition_chunks = original
    assert calls == [4, 3]

    # A str gene (the single-condition entry point) takes the same one call.
    backbone.state_adapter.forward_condition_chunks = counting
    calls.clear()
    try:
        predict_bags(backbone, [torch.randn(4, 2000 + 2560)], [gene], seed=0)
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


def test_native_basal_variants_classify_every_encoder_layer(tmp_path):
    """A released checkpoint with a two-layer basal encoder must build.

    Wrapping renames ``basal_encoder.<k>.*`` to ``basal_encoder.native.<k>.*``;
    every one of those has to keep a provenance entry, or ``build_variant``
    raises on the deeper layers as unclassified.
    """
    from src.model.p1c import build_variant

    template, inputs, native_state = _build_p1c_variant_fixture(
        tmp_path, n_encoder_layers=2
    )
    assert "basal_encoder.3.weight" in native_state

    prefix = "state_adapter.state_model."
    adapter_names = {
        "perturbations.adapter.net.0.weight",
        "perturbations.adapter.net.0.bias",
        "perturbations.adapter.net.2.weight",
        "perturbations.adapter.net.2.bias",
    }
    for variant in ("V2", "V3"):
        backbone, report = build_variant(
            template, variant, native_state=native_state, expected_count=None
        )
        origins = {
            name: entry["origin"] for name, entry in report["parameters"].items()
        }
        assert origins[prefix + "basal_encoder.native.0.weight"] == "inherited-native"
        assert origins[prefix + "basal_encoder.native.0.bias"] == "inherited-native"
        assert origins[prefix + "basal_encoder.native.3.weight"] == "inherited-template"
        assert origins[prefix + "basal_encoder.native.3.bias"] == "inherited-template"
        assert origins[prefix + "basal_encoder.context.weight"] == "new-zero"
        # Trainable set is unchanged by the extra layer: adapter + context.
        trainable = {name for name, p in backbone.named_parameters() if p.requires_grad}
        assert trainable == adapter_names | {prefix + "basal_encoder.context.weight"}
        state = backbone.state_adapter.state_model
        assert torch.equal(
            state.basal_encoder.native[0].weight, native_state["basal_encoder.0.weight"]
        )


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
    # by both branches cancel in p - n. (b) cancels the *forward* bitwise
    # (predictions are exactly control_hvg), but both branches now ride in one
    # batched STATE call, so their gradient contributions are summed in
    # different reduction orders: the cancellation is exact to float rounding,
    # ~1e-7 of the surviving gradients' own scale, not bitwise.
    cancelled_at_init = {
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

        zero_names = cancelled_at_init[variant]
        actual = dict(backbone.named_parameters())
        # The adapter's final layer must always carry signal: it is the only
        # trainable tensor never cancelled by the null branch or deadened by
        # a zeroed downstream layer. Its magnitude also sets the scale a
        # cancelled gradient's rounding residual is judged against.
        for name in net2:
            assert actual[name].grad.abs().max() > 0, (variant, name)
        scale = max(float(actual[name].grad.abs().max()) for name in net2)
        tolerance = 1e-5 * max(scale, 1.0)

        for name, p in actual.items():
            if not report["parameters"][name]["trainable"]:
                assert p.grad is None, (variant, name)
                continue
            assert p.grad is not None, (variant, name)
            assert torch.isfinite(p.grad).all(), (variant, name)
            if name in zero_names:
                assert float(p.grad.abs().max()) <= tolerance, (variant, name)
            else:
                assert p.grad.abs().max() > 0, (variant, name)

        if not zero_names:
            continue

        # Confirm the cancelled gradients above are a degenerate artifact of
        # the exact-zero init, not permanently dead parameters: one optimizer
        # step moves net.2 away from zero, so a second backward picks up a
        # gradient far above the rounding residual on every previously
        # cancelled trainable tensor.
        optimizer = torch.optim.SGD(parameter_groups(backbone, lr=1.0))
        optimizer.step()
        backbone.zero_grad(set_to_none=True)
        control2 = torch.randn(4, width)
        (pred2,) = predict_bags(backbone, [control2], [gene], seed=0)
        pred2.sum().backward()
        for name in zero_names:
            grad = actual[name].grad
            assert grad is not None, (variant, name)
            assert float(grad.abs().max()) > tolerance, (variant, name)


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
    # A released checkpoint carries its own hyper_parameters; cell_set_len
    # deliberately differs from the template's, which N-native is built from.
    released_hparams = copy.deepcopy(template["architecture"]["state_hparams"])
    released_hparams.update(input_dim=2000, cell_set_len=8)
    torch.save(
        {"state_dict": native_state, "hyper_parameters": released_hparams},
        native_checkpoint_path,
    )
    native_map_path = tmp_path / "native_map.pt"
    torch.save(native_map, native_map_path)

    manifest = {
        "bundle_sha256": digest(prepared / "bundle.pt"),
        "model_files": {"B-init": digest(prepared / "B-init.pt")},
        "native_state_checkpoint": str(native_checkpoint_path),
        "native_map": str(native_map_path),
        "native_checkpoint_sha256": digest(native_checkpoint_path),
        "native_vocabulary_sha256": digest(native_map_path),
        "external": inputs.response_anchors[3],
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

    # The light (batch index != 0) summary and the native-null diagnostic both
    # tag every condition with the role it came from.
    light_internal = pd.read_csv(
        runs / "evaluation" / "N-native-b1" / "internal" / "summary.csv"
    )
    assert set(light_internal["role"]) <= {"train", "val"}
    light_external = pd.read_csv(
        runs / "evaluation" / "N-native-b1" / "external" / "summary.csv"
    )
    assert set(light_external["role"]) <= {"val", "external"}
    null_internal = pd.read_parquet(
        runs / "evaluation" / "N-native" / "internal" / "native_null.parquet"
    )
    assert set(null_internal["role"]) <= {"train", "val"}
    null_external = pd.read_parquet(
        runs / "evaluation" / "N-native" / "external" / "native_null.parquet"
    )
    assert set(null_external["role"]) <= {"val", "external"}


def test_evaluate_native_records_the_hparams_it_built_with(tmp_path):
    torch.set_num_threads(1)
    from src.experiments.p1c import evaluate_native

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    runs = tmp_path / "runs"
    evaluate_native(prepared, runs, [0], device="cpu")

    record = json.loads((runs / "native_hparams.json").read_text())
    assert record["hparams"]["input_dim"] == 2000
    assert record["input_dim_override"] == 2000
    assert record["checkpoint_hyper_parameters"] == "present"
    # The template's cell_set_len (4) is what N-native is actually built with;
    # the released checkpoint says 8, and that disagreement is recorded.
    assert record["differs_from_checkpoint"] == {
        "cell_set_len": {"used": 4, "checkpoint": 8}
    }


def test_native_null_reference_reports_without_gating(tmp_path):
    torch.set_num_threads(1)
    import src.experiments.p1c as p1c
    from src.experiments.p1b_preparation import digest, open_bundle
    from src.model.p1c import build_variant

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    manifest = json.loads((prepared / "manifest.json").read_text())
    template = torch.load(
        prepared / "B-init.pt", map_location="cpu", weights_only=False
    )

    for variant in ("V2", "V2-null"):
        bundle, view, _ = open_bundle(prepared, input_layout=p1c.INPUT_LAYOUT[variant])
        backbone, _ = build_variant(
            template, variant, native_state=native_state, expected_count=None
        )
        reference = p1c.native_null_reference(
            backbone, view, bundle, template, manifest, "cpu", cap=2
        )
        assert reference["status"] == "reported"
        assert reference["conditions_cap_per_anchor"] == 2
        assert 0 < reference["conditions"] <= 2 * len(bundle["anchors"])
        assert reference["variant_loss"] > 0 and reference["native_null_loss"] > 0
        assert reference["relative_difference"] == pytest.approx(
            (reference["variant_loss"] - reference["native_null_loss"])
            / reference["native_null_loss"]
        )

    # Without a non-targeting token there is nothing to compare against, and
    # the reference records that instead of failing the arm.
    without = {
        gene: vector
        for gene, vector in torch.load(
            manifest["native_map"], map_location="cpu", weights_only=False
        ).items()
        if gene != "non-targeting"
    }
    map_path = tmp_path / "native_map_no_control.pt"
    torch.save(without, map_path)
    manifest = dict(
        manifest, native_map=str(map_path), native_vocabulary_sha256=digest(map_path)
    )
    reference = p1c.native_null_reference(
        backbone, view, bundle, template, manifest, "cpu", cap=2
    )
    assert reference == {
        "status": "unavailable",
        "reason": "non-targeting absent from native_map",
    }


def test_train_variant_v2_records_the_native_null_reference(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    import src.experiments.p1c as p1c
    import src.model.p1c as p1c_model

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(p1c, "MAX_EPOCHS", 1)
    monkeypatch.setattr(p1c, "NATIVE_REFERENCE_CAP", 2)
    monkeypatch.setitem(p1c_model._TRAINABLE_COUNTS, "V2", 20506)

    p1c.main(
        [
            "train",
            "--prepared",
            str(prepared),
            "--runs",
            str(runs),
            "--variant",
            "V2",
            "--lr",
            "1e-4",
            "--device",
            "cpu",
        ]
    )
    training = json.loads((runs / "training.json").read_text())
    check = training["init_check"]
    # The structural check is what gates; the native comparison is recorded.
    assert check["status"] == "passed"
    assert check["check"] == "adapter_final_layer_zero"
    reference = check["native_null_reference"]
    assert reference["status"] == "reported"
    assert reference["conditions_cap_per_anchor"] == 2
    assert set(reference) == {
        "status",
        "conditions",
        "conditions_cap_per_anchor",
        "variant_loss",
        "native_null_loss",
        "relative_difference",
    }


def test_native_inputs_rejects_tampered_native_map(tmp_path):
    from src.experiments.p1c import native_inputs

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    manifest = json.loads((prepared / "manifest.json").read_text())

    # Sanity: the untampered fixture verifies cleanly.
    native_inputs(manifest)

    native_map_path = Path(manifest["native_map"])
    original_bytes = native_map_path.read_bytes()
    native_map_path.write_bytes(original_bytes + b"\x00")
    with pytest.raises(ValueError, match="native vocabulary identity changed"):
        native_inputs(manifest)
    native_map_path.write_bytes(original_bytes)

    native_checkpoint_path = Path(manifest["native_state_checkpoint"])
    original_checkpoint_bytes = native_checkpoint_path.read_bytes()
    native_checkpoint_path.write_bytes(original_checkpoint_bytes + b"\x00")
    with pytest.raises(ValueError, match="native checkpoint identity changed"):
        native_inputs(manifest)
    native_checkpoint_path.write_bytes(original_checkpoint_bytes)


def test_train_variant_rejects_rerun_without_resume_before_writing_parameters(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    import src.experiments.p1c as p1c
    import src.model.p1c as p1c_model

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(p1c, "MAX_EPOCHS", 1)
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
    before = (runs / "parameters.json").read_bytes()

    with pytest.raises(FileExistsError, match="run exists; use resume"):
        p1c.main(train_args)

    after = (runs / "parameters.json").read_bytes()
    assert before == after


def test_init_check_rejects_nonzero_effect_prediction(tmp_path, monkeypatch):
    import src.experiments.p1c as p1c
    from src.model.p1c import build_variant

    template, inputs, native_state = _build_p1c_variant_fixture(tmp_path)
    backbone, _ = build_variant(template, "V1", expected_count=None)

    bundle = {
        "anchors": ["a"],
        "splits": {"val": [0, 1, 2, 3]},
        "keys": [("a", "g0"), ("a", "g1"), ("a", "g2"), ("a", "g3")],
    }

    class FakeView:
        def batch(self, indices, device):
            from src.data.batches import ResponseBatch

            width = 2000 + 2560
            control = torch.randn(len(indices), 4, width)
            hvg = control[:, :, :2000]
            return ResponseBatch(
                tuple("a" for _ in indices),
                tuple(f"g{i}" for i in indices),
                tuple(control[i] for i in range(len(indices))),
                tuple(hvg[i] for i in range(len(indices))),
                tuple(hvg[i] for i in range(len(indices))),
            )

    def fake_predictions(model, batch, device, genes=None):
        return tuple(c + 1.0 for c in batch.control_hvg)

    monkeypatch.setattr(p1c, "predictions", fake_predictions)

    with pytest.raises(ValueError, match="zero-effect"):
        p1c.init_check(backbone, FakeView(), bundle, "V1", "cpu")


def _condition_rows(anchor, role, better, *, n_genes=3, panel="all", deltas=None):
    """Paired no_change/model rows. ``deltas`` overrides ``better``, giving each
    gene's model loss as ``no_change + deltas[gi]`` -- the way to build a fold
    whose paired interval straddles zero."""
    rows = []
    for gi in range(n_genes):
        gene = f"g{gi}"
        no_change_loss = 1.0 + 0.1 * gi
        model_loss = (
            no_change_loss + deltas[gi]
            if deltas is not None
            else (0.3 if better else 1.6) + 0.05 * gi
        )
        rows.append(
            dict(
                role=role,
                model_id=anchor,
                gene=gene,
                panel=panel,
                method="no_change",
                response_loss=no_change_loss,
            )
        )
        rows.append(
            dict(
                role=role,
                model_id=anchor,
                gene=gene,
                panel=panel,
                method="model",
                response_loss=model_loss,
            )
        )
    return rows


def _with_identity_column(rows, positive):
    out = []
    for row in rows:
        row = dict(row)
        if row["method"] == "model":
            gi = int(row["gene"][1:])
            delta = (0.3 if positive else -0.3) + 0.02 * gi
            row["wrong_response_loss"] = row["response_loss"] + delta
        out.append(row)
    return out


def _write_p1c_run(
    root,
    label,
    fold,
    *,
    lr=1e-4,
    n_genes=3,
    better_source_count=2,
    external_better=True,
    external_deltas=None,
    identity_advantage_positive=True,
):
    from src.data.p1c import fold_membership

    sources, external = fold_membership(fold)
    run_dir = root / "runs" / label / fold
    run_dir.mkdir(parents=True)

    variant = label.split("-lr")[0]
    training = {
        "status": "completed",
        "epoch": 3,
        "step": 30,
        "best_epoch": 3,
        "bad_epochs": 0,
        "identity": f"{label}:{fold}",
        "best_loss": 0.5,
        "variant": variant,
        "fold": fold,
        "lr": lr,
        "init_check": {"status": "passed"},
        "input_layout": "hvg_tx1",
    }
    (run_dir / "training.json").write_text(json.dumps(training))
    history = [
        {"epoch": e, "step": e * 10, "val": {"response_loss": 1.0 / (e + 1)}}
        for e in range(4)
    ]
    (run_dir / "history.json").write_text(json.dumps(history))

    internal_dir = run_dir / "evaluation" / "internal"
    internal_dir.mkdir(parents=True)
    internal_rows = []
    for idx, anchor in enumerate(sources):
        internal_rows.extend(
            _condition_rows(anchor, "val", idx < better_source_count, n_genes=n_genes)
        )
    pd.DataFrame(internal_rows).to_parquet(
        internal_dir / "conditions.parquet", index=False
    )
    pd.DataFrame(
        columns=[
            "method",
            "anchor_a",
            "anchor_b",
            "gene",
            "effect_difference_mse",
            "effect_difference_pearson",
        ]
    ).to_parquet(internal_dir / "cross_context.parquet", index=False)
    np.savez(
        internal_dir / "effects.npz",
        keys=np.array([("model", "dummy", "g0")]),
        predicted=np.array([[0.0]]),
        observed=np.array([[0.0]]),
    )
    (internal_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle": "b",
                "checkpoint_sha256": "c",
                "variant": variant,
                "fold": fold,
            }
        )
    )

    external_dir = run_dir / "evaluation" / "external"
    external_dir.mkdir(parents=True)
    external_rows = []
    for anchor in sources:
        external_rows.extend(_condition_rows(anchor, "val", True, n_genes=n_genes))
    ext_anchor_rows = _condition_rows(
        external,
        "external",
        external_better,
        n_genes=n_genes,
        deltas=external_deltas,
    )
    ext_anchor_rows = _with_identity_column(
        ext_anchor_rows, identity_advantage_positive
    )
    external_rows.extend(ext_anchor_rows)
    pd.DataFrame(external_rows).to_parquet(
        external_dir / "conditions.parquet", index=False
    )

    cross_rows = []
    for other in sources:
        for gi in range(n_genes):
            gene = f"g{gi}"
            cross_rows.append(
                dict(
                    method="model",
                    anchor_a=external,
                    anchor_b=other,
                    gene=gene,
                    effect_difference_mse=0.2 + 0.01 * gi,
                    effect_difference_pearson=0.5,
                )
            )
            cross_rows.append(
                dict(
                    method="no_change",
                    anchor_a=external,
                    anchor_b=other,
                    gene=gene,
                    effect_difference_mse=0.5 + 0.01 * gi,
                    effect_difference_pearson=0.4,
                )
            )
    pd.DataFrame(cross_rows).to_parquet(
        external_dir / "cross_context.parquet", index=False
    )

    keys, predicted, observed = [], [], []
    for method in ("model", "no_change"):
        for gi in range(n_genes):
            keys.append((method, external, f"g{gi}"))
            predicted.append([0.1, 0.0])
            observed.append([0.0, 0.0])
    np.savez(
        external_dir / "effects.npz",
        keys=np.array(keys),
        predicted=np.array(predicted),
        observed=np.array(observed),
    )
    (external_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle": "b",
                "checkpoint_sha256": "c",
                "variant": variant,
                "fold": fold,
            }
        )
    )
    return run_dir


def test_equal_fold_difference_pools_folds_equally_regardless_of_size():
    from src.eval.p1c import equal_fold_difference

    def frame(deltas, genes):
        left = pd.DataFrame({"gene": genes, "loss": deltas})
        right = pd.DataFrame({"gene": genes, "loss": [0.0] * len(genes)})
        return left, right

    # A shared gene ("gShared") appears in both folds with its own per-fold
    # value, exercising the union index: a replicate that draws "gShared"
    # must feed BOTH folds' own (different) data for it simultaneously.
    fold_a = frame([-2.0, -1.0, 0.0], ["g0", "g1", "gShared"])
    fold_b = frame([-4.0, -3.0, -2.0], ["g2", "g3", "gShared"])
    frames = [fold_a, fold_b]

    result = equal_fold_difference(frames, "loss", repeats=2000)
    # Equal-fold mean of per-fold means: mean(-1.0, -3.0) == -2.0.
    assert result["delta"] == pytest.approx(-2.0)
    # Per-gene values vary within each fold, so the bootstrap has genuine
    # spread -- a strict bracket, not a degenerate point interval.
    assert result["ci_low"] < result["delta"] < result["ci_high"]
    assert result["folds"] == 2
    assert result["pairs"] == 6

    repeat = equal_fold_difference(frames, "loss", repeats=2000)
    assert repeat == result  # seed-0 determinism: identical intervals


def test_equal_fold_ratio_pools_ratios_not_differences():
    from src.eval.p1c import equal_fold_ratio

    def frame(model_losses, no_change_losses, genes):
        return (
            pd.DataFrame({"gene": genes, "response_loss": model_losses}),
            pd.DataFrame({"gene": genes, "response_loss": no_change_losses}),
        )

    # Fold A is on a small scale (ratio 0.5), fold B on a large one (ratio
    # 0.25): the equal-fold mean ratio is 0.375, while a pooled *difference*
    # would be dominated by fold B's magnitude.
    fold_a = frame([0.4, 0.5, 0.6], [0.8, 1.0, 1.2], ["g0", "g1", "gShared"])
    fold_b = frame([20.0, 25.0, 30.0], [80.0, 100.0, 120.0], ["g2", "g3", "gShared"])
    result = equal_fold_ratio([fold_a, fold_b], repeats=2000)
    assert result["ratio"] == pytest.approx(0.375)
    assert result["ci_high"] < 1
    assert result["folds"] == 2
    assert result["pairs"] == 6
    assert equal_fold_ratio([fold_a, fold_b], repeats=2000) == result

    assert equal_fold_ratio([], repeats=10) == {
        "ratio": None,
        "ci_low": None,
        "ci_high": None,
        "folds": 0,
        "pairs": 0,
    }


def test_summarize_keeps_a_variant_whose_pooled_ratio_clears_one_fold_wide_interval(
    tmp_path,
):
    """(b) is pooled, not per-fold: one fold's external interval may straddle
    zero and the variant is still kept when the pooled ratio interval is below
    1 and (a) and (c) hold everywhere."""
    from src.eval.p1c import summarize
    from src.data.p1c import FOLDS

    root = tmp_path / "root"
    for fold in FOLDS:
        # hct116's external losses straddle no-change gene by gene (mean
        # difference 0), so that fold's own kept_b is false.
        _write_p1c_run(
            root,
            "V1",
            fold,
            external_deltas=[0.3, -0.35, 0.05] if fold == "hct116" else None,
        )

    out_dir = summarize(root, tmp_path / "out")
    summary = pd.read_csv(out_dir / "summary.csv")
    straddling = summary[(summary.label == "V1") & (summary.fold == "hct116")].iloc[0]
    assert straddling.kept_a == True  # noqa: E712
    assert straddling.kept_b == False  # noqa: E712
    assert straddling.kept_c == True  # noqa: E712
    assert straddling.kept == False  # noqa: E712 -- per-fold kept is descriptive

    variants = pd.read_csv(out_dir / "variants.csv")
    v1_row = variants[variants.label == "V1"].iloc[0]
    assert v1_row.folds == 4
    assert v1_row.kept_folds == 3
    assert v1_row.pooled_ratio < 1
    assert v1_row.pooled_ratio_ci_high < 1
    assert v1_row.kept_all == True  # noqa: E712

    kept = json.loads((out_dir / "kept.json").read_text())
    assert kept["V1"] is True
    assert kept["V3_eligible"] is True


def test_summarize_reports_epoch_zero_beside_the_selected_checkpoint(tmp_path):
    from src.eval.p1c import summarize

    root = tmp_path / "root"
    _write_p1c_run(root, "V1", "k562")
    summary = pd.read_csv(summarize(root, tmp_path / "out") / "summary.csv")
    row = summary.iloc[0]
    # history[0] is fit()'s untrained validation; the no-change equal-anchor
    # mean over the three source anchors' val rows is 1.1.
    assert row.epoch0_val_loss == pytest.approx(1.0)
    assert row.epoch0_internal_ratio_equal == pytest.approx(1.0 / 1.1)


def test_fold_summary_reports_a_completed_record_missing_its_fields(tmp_path):
    from src.eval.p1c import fold_summary

    root = tmp_path / "root"
    run_dir = _write_p1c_run(root, "V2", "hepg2")
    training = json.loads((run_dir / "training.json").read_text())
    del training["variant"], training["lr"], training["fold"]
    (run_dir / "training.json").write_text(json.dumps(training))

    row = fold_summary(run_dir)
    assert row["status"] == "incomplete-record"
    assert row["kept"] is None
    assert row["variant"] == "V2"  # recovered from the label
    assert row["missing_fields"] == "variant,fold,lr"


def test_summarize_applies_keep_predicate_per_fold_and_reports_two_of_four_folds(
    tmp_path,
):
    from src.eval.p1c import summarize

    root = tmp_path / "root"
    for fold in ("jurkat", "k562"):
        _write_p1c_run(root, "V1", fold)
        _write_p1c_run(
            root,
            "V0",
            fold,
            better_source_count=0,
            external_better=False,
            identity_advantage_positive=False,
        )

    out_dir = summarize(root, tmp_path / "out")
    assert out_dir == tmp_path / "out"
    summary = pd.read_csv(out_dir / "summary.csv")

    good = summary[summary.label == "V1"]
    assert len(good) == 2
    assert good.kept.all()

    bad = summary[summary.label == "V0"]
    assert len(bad) == 2
    assert not bad.kept.any()

    variants = pd.read_csv(out_dir / "variants.csv")
    v1_row = variants[variants.label == "V1"].iloc[0]
    assert v1_row.folds == 2
    assert v1_row.kept_folds == 2
    assert v1_row.kept_all == False  # noqa: E712 -- only 2 of 4 folds present

    v0_row = variants[variants.label == "V0"].iloc[0]
    assert v0_row.kept_folds == 0

    kept = json.loads((out_dir / "kept.json").read_text())
    assert kept["V1"] is False  # kept_all requires all four folds
    assert kept["V0"] is False
    assert kept["V3_eligible"] == (kept.get("V1", False) or kept.get("V2", False))
    assert (out_dir / "learning_curves.png").exists()
    assert (out_dir / "analysis.md").exists()


def test_summarize_kept_all_true_requires_all_four_folds_kept(tmp_path):
    from src.eval.p1c import summarize
    from src.data.p1c import FOLDS

    root = tmp_path / "root"
    for fold in FOLDS:
        _write_p1c_run(root, "V1", fold)

    out_dir = summarize(root, tmp_path / "out")
    variants = pd.read_csv(out_dir / "variants.csv")
    v1_row = variants[variants.label == "V1"].iloc[0]
    assert v1_row.folds == 4
    assert v1_row.kept_folds == 4
    assert v1_row.kept_all == True  # noqa: E712

    kept = json.loads((out_dir / "kept.json").read_text())
    assert kept["V1"] is True
    assert kept["V3_eligible"] is True


def test_fold_summary_marks_incomplete_training_as_not_kept_without_touching_exports(
    tmp_path,
):
    from src.eval.p1c import fold_summary

    root = tmp_path / "root"
    run_dir = _write_p1c_run(root, "V2", "hepg2")
    training = json.loads((run_dir / "training.json").read_text())
    training["status"] = "running"
    del training["variant"], training["lr"], training["fold"]
    (run_dir / "training.json").write_text(json.dumps(training))

    row = fold_summary(run_dir)
    assert row["kept"] is None
    assert row["status"] == "running"


def test_fold_summary_raises_on_missing_export_when_training_completed(tmp_path):
    from src.eval.p1c import fold_summary

    root = tmp_path / "root"
    run_dir = _write_p1c_run(root, "V3", "hct116")
    import shutil

    shutil.rmtree(run_dir / "evaluation" / "external")
    with pytest.raises(ValueError, match="external"):
        fold_summary(run_dir)


def test_fold_summary_raises_on_zero_pairs_instead_of_evaluating_kept_false(tmp_path):
    from src.eval.p1c import fold_summary

    root = tmp_path / "root"
    run_dir = _write_p1c_run(root, "V0", "jurkat")
    internal_path = run_dir / "evaluation" / "internal" / "conditions.parquet"
    conditions = pd.read_parquet(internal_path)
    # Drop every no_change row for one source anchor: that anchor's paired
    # difference now has zero pairs. This must raise, not silently count as
    # "not below zero" toward anchors_ci_below.
    from src.data.p1c import fold_membership

    sources, _ = fold_membership("jurkat")
    first_anchor = sources[0]
    conditions = conditions[
        ~((conditions.model_id == first_anchor) & (conditions.method == "no_change"))
    ]
    conditions.to_parquet(internal_path, index=False)

    with pytest.raises(ValueError, match=first_anchor):
        fold_summary(run_dir)


def test_fold_summary_kept_false_when_only_anchors_ci_below_fails(tmp_path):
    from src.eval.p1c import fold_summary

    root = tmp_path / "root"
    run_dir = _write_p1c_run(
        root,
        "V1",
        "jurkat",
        better_source_count=1,  # only 1 of 3 source anchors ci_high < 0
        external_better=True,
        identity_advantage_positive=True,
    )
    row = fold_summary(run_dir)
    assert row["anchors_ci_below"] == 1
    assert row["kept_a"] is False
    assert row["kept_b"] is True
    assert row["kept_c"] is True
    assert row["kept"] is False


def test_native_export_rows_handle_light_batches_without_conditions_parquet(tmp_path):
    from src.eval.p1c import summarize
    from src.data.p1c import fold_membership

    root = tmp_path / "root"
    fold = "k562"
    sources, external = fold_membership(fold)
    native_dir = root / "runs" / "N-native" / fold / "evaluation"

    # Batch index 0: full export_predictions output (conditions.parquet with
    # method/panel columns), mirroring evaluate_native's real shape -- whose
    # panels are the native ones only, never "all".
    zero_dir = native_dir / "N-native"
    for side, role, anchors in (
        ("internal", "val", sources),
        ("external", "external", (external,)),
    ):
        side_dir = zero_dir / side
        side_dir.mkdir(parents=True)
        rows = []
        for anchor in anchors:
            rows.extend(
                _condition_rows(anchor, role, True, n_genes=3, panel="native_all")
            )
        pd.DataFrame(rows).to_parquet(side_dir / "conditions.parquet", index=False)
        pd.DataFrame(
            columns=[
                "method",
                "anchor_a",
                "anchor_b",
                "gene",
                "effect_difference_mse",
                "effect_difference_pearson",
            ]
        ).to_parquet(side_dir / "cross_context.parquet", index=False)
        np.savez(
            side_dir / "effects.npz",
            keys=np.array([("model", "dummy", "g0")]),
            predicted=np.array([[0.0]]),
            observed=np.array([[0.0]]),
        )
        (side_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "bundle": "b",
                    "variant": "N-native",
                    "batch_index": 0,
                    "fold": fold,
                }
            )
        )

    # Batch index 1: light export -- raw evaluate_rows summary.csv only, no
    # conditions.parquet, no method/panel columns.
    light_dir = native_dir / "N-native-b1"
    for side, role, anchors in (
        ("internal", "val", sources),
        ("external", "external", (external,)),
    ):
        side_dir = light_dir / side
        side_dir.mkdir(parents=True)
        rows = [
            dict(
                role=role, model_id=anchor, gene=f"g{gi}", response_loss=0.4 + 0.01 * gi
            )
            for anchor in anchors
            for gi in range(3)
        ]
        pd.DataFrame(rows).to_csv(side_dir / "summary.csv", index=False)
        (side_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "bundle": "b",
                    "variant": "N-native",
                    "batch_index": 1,
                    "fold": fold,
                }
            )
        )

    out_dir = summarize(root, tmp_path / "out")  # must not raise
    summary = pd.read_csv(out_dir / "summary.csv")
    native = summary[summary.label == "N-native"]
    assert set(native.export) == {"N-native", "N-native-b1"}

    zero_row = native[native.export == "N-native"].iloc[0]
    assert zero_row.batch_index == 0
    assert zero_row.internal_ratio_equal == pytest.approx(0.35 / 1.1, rel=1e-6)
    assert zero_row.external_ratio == pytest.approx(0.35 / 1.1, rel=1e-6)

    light_row = native[native.export == "N-native-b1"].iloc[0]
    assert light_row.batch_index == 1
    assert pd.isna(light_row.internal_ratio_equal)
    assert pd.isna(light_row.external_ratio)
    assert light_row.native_val_loss_equal == pytest.approx(0.41, rel=1e-6)
    assert light_row.native_external_loss == pytest.approx(0.41, rel=1e-6)


def test_real_native_exports_summarize_on_the_native_panel(tmp_path, monkeypatch):
    """End to end: evaluate_native's own exports must be readable by summarize.

    The native exports carry no ``all`` panel (``restrict_to_native`` drops
    it), so a summary that looked for one produced silent NaN ratios.
    """
    torch.set_num_threads(1)
    import src.eval.p1c as p1c_eval
    from src.experiments.p1c import evaluate_native

    prepared, inputs, native_state = _build_p1c_fold_fixture(tmp_path)
    anchors = tuple(inputs.response_anchors[:3])
    external = inputs.response_anchors[3]
    # The fixture's anchors are synthetic ModelIDs, not the real four.
    monkeypatch.setattr(p1c_eval, "fold_membership", lambda fold: (anchors, external))

    root = tmp_path / "root"
    evaluate_native(prepared, root / "runs" / "N-native" / "k562", [0, 1], device="cpu")

    summary = pd.read_csv(p1c_eval.summarize(root, tmp_path / "out") / "summary.csv")
    native = summary[summary.label == "N-native"]
    assert set(native.export) == {"N-native", "N-native-b1"}

    zero_row = native[native.export == "N-native"].iloc[0]
    assert zero_row.panel == "native_all"
    assert np.isfinite(zero_row.internal_ratio_equal)
    assert np.isfinite(zero_row.external_ratio)

    light_row = native[native.export == "N-native-b1"].iloc[0]
    assert np.isfinite(light_row.native_val_loss_equal)
    assert np.isfinite(light_row.native_external_loss)


PIPELINE = "hpc/p1c_pipeline.sh"
PIPELINE_FOLDS = ("jurkat", "k562", "hepg2", "hct116")


def test_pipeline_script_is_syntactically_valid_bash():
    result = subprocess.run(
        ["bash", "-n", PIPELINE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_sh_p1c_help_names_train_and_compare():
    env = dict(os.environ)
    env["PYTHON_BIN"] = sys.executable
    result = subprocess.run(
        ["bash", "hpc/run.sh", "p1c", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout and "compare" in result.stdout


def _pipeline_env(run, kept, gpus="0"):
    env = dict(os.environ)
    env.update(
        {
            "PIPELINE_DRY_RUN": "1",
            "PIPELINE_DRY_RUN_KEPT": str(kept),
            "PIPELINE_POLL_SECONDS": "1",
            "GPUS": gpus,
            "RUN": str(run),
            "P0_CHECKPOINT": "dummy/p0/best.pt",
            "P1B_PREPARED": "dummy/p1b/prepared",
            "P1B_RUNS": "dummy/p1b/runs",
            "P1A_FEATURES": "dummy/p1a/features",
            "P1A_REFERENCE_P0": "dummy/p1a/p0/predictions.parquet",
            "P1A_REFERENCE_PCA": "dummy/p1a/pca/predictions.parquet",
            "PYTHON_BIN": sys.executable,
        }
    )
    return env


def _dry_run_pipeline(tmp_path, v3_eligible, name="run"):
    """Run the whole pipeline with every hpc/run.sh call replaced by an echo."""
    run = tmp_path / name
    kept = tmp_path / f"kept-{name}.json"
    kept.write_text(json.dumps({"V3_eligible": v3_eligible}))
    env = _pipeline_env(run, kept)
    result = subprocess.run(
        ["bash", PIPELINE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return run, result


def _pipeline_commands(run, result):
    """Every echoed hpc/run.sh invocation, from the job logs and stdout."""
    text = result.stdout + "".join(
        path.read_text() for path in sorted(run.glob("*.log"))
    )
    return [
        tokens
        for tokens in (line.split() for line in text.splitlines())
        if len(tokens) >= 3 and tokens[0] == "hpc/run.sh"
    ]


def test_pipeline_dry_run_issues_every_wave_command_with_v3_eligible(tmp_path):
    run, result = _dry_run_pipeline(tmp_path, True)
    assert result.returncode == 0, result.stderr
    assert (run / "phase.txt").read_text().strip() == "completed"
    assert not (run / "v3_skipped.txt").exists()

    commands = _pipeline_commands(run, result)
    counts = Counter((tokens[1], tokens[2]) for tokens in commands)
    assert counts[("p1c", "prepare")] == 4
    assert counts[("p1c", "tier0")] == 1
    assert counts[("p1c", "evaluate-native")] == 1
    assert counts[("p1a", "train")] == 2
    assert counts[("p1a", "compare")] == 2
    # V0/V1/V2-null/V2 on four folds, two learning-rate arms, four V3 folds.
    assert counts[("p1c", "train")] == 16 + 2 + 4
    assert counts[("p1c", "evaluate")] == 2 * (16 + 2 + 4)
    assert counts[("p1c", "compare")] == 2

    expected = [
        f"{run}/runs/{label}/{fold}"
        for label in ("V0", "V1", "V2-null", "V2", "V3")
        for fold in PIPELINE_FOLDS
    ]
    expected += [f"{run}/runs/V0-lr1e-6/jurkat", f"{run}/runs/V0-lr1e-5/jurkat"]
    actual = [
        tokens[tokens.index("--runs") + 1]
        for tokens in commands
        if (tokens[1], tokens[2]) == ("p1c", "train")
    ]
    assert sorted(actual) == sorted(expected)


def test_pipeline_dry_run_skips_v3_when_the_predicate_is_not_met(tmp_path):
    run, result = _dry_run_pipeline(tmp_path, False, name="run-ineligible")
    assert result.returncode == 0, result.stderr
    assert (run / "phase.txt").read_text().strip() == "completed"
    assert (run / "v3_skipped.txt").exists()

    counts = Counter(
        (tokens[1], tokens[2]) for tokens in _pipeline_commands(run, result)
    )
    assert counts[("p1c", "train")] == 16 + 2
    assert counts[("p1c", "evaluate")] == 2 * (16 + 2)


def _wait_until(predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def test_pipeline_sigterm_marks_interrupted_and_kills_running_jobs(tmp_path):
    run = tmp_path / "run-signal"
    kept = tmp_path / "kept-signal.json"
    kept.write_text(json.dumps({"V3_eligible": True}))
    env = _pipeline_env(run, kept, gpus="0 1")
    env["PIPELINE_DRY_RUN_SLEEP"] = "5"  # keep each queued job alive to be signalled

    process = subprocess.Popen(
        ["bash", PIPELINE],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Every started job has both its own pid file and its child's.
        assert _wait_until(lambda: any(run.glob("*.child.pid")), 30), "no job started"
        jobs = {
            path.stem: int(path.read_text())
            for path in sorted(run.glob("*.pid"))
            if not path.name.endswith(".child.pid")
        }
        children = {
            path.name[: -len(".child.pid")]: int(path.read_text())
            for path in sorted(run.glob("*.child.pid"))
        }
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=60)
    finally:
        if process.poll() is None:  # pragma: no cover -- only on a stuck pipeline
            process.kill()
            process.communicate()

    assert process.returncode == 143
    assert (run / "phase.txt").read_text().strip() == "interrupted"
    assert (run / "exit_code").read_text().strip() == "143"
    assert jobs and set(children) <= set(jobs)

    def gone(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        return False

    for name, pid in jobs.items():
        assert _wait_until(lambda: gone(pid), 10), f"job {name} ({pid}) survived"
        assert (run / f"{name}.exit").read_text().strip() == "143"
    # The job's own child -- standing in for the Python worker hpc/run.sh execs
    # -- must not be reparented to PID 1 and left holding a GPU.
    for name, pid in children.items():
        assert _wait_until(lambda: gone(pid), 10), f"{name} child {pid} survived"

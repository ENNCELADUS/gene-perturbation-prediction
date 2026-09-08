"""Fixed-representation diagnostic behavior through its public interfaces."""

from dataclasses import replace

import numpy as np
import torch
import json

from src.data.batches import FeatureBatch
from src.model.head import GeneEffectFeatureDims


def features():
    generator = torch.Generator().manual_seed(41)
    return FeatureBatch(
        **{
            name: torch.randn(4, width, generator=generator)
            for name, width in {
                "delta_proj": 3,
                "s": 6,
                "q_sc": 3,
                "e_g": 2,
                "z_c": 10,
            }.items()
        },
        q_sc_mask=torch.ones(4, dtype=torch.bool),
        hvg_panel_mask=torch.ones(4, dtype=torch.bool),
        own_gene_shift_mask=torch.ones(4, dtype=torch.bool),
        gene_symbols=("G0", "G1", "G0", "G1"),
        model_ids=("C0", "C0", "C1", "C1"),
    )


def test_explicit_arms_start_at_paired_predictions_and_no_response_arms_ignore_r():
    from src.model.readout import make_readout

    dims = GeneEffectFeatureDims(delta_proj=3, e_g=2, z_c=10)
    heads = {arm: make_readout(arm, dims, 2) for arm in ("A0", "A1", "A2", "A3")}
    batch = features()
    genes = torch.tensor([0, 1, 0, 1])
    context = torch.randn(4, 8)
    predictions = {arm: head(batch, genes, context) for arm, head in heads.items()}
    torch.testing.assert_close(predictions["A0"], predictions["A2"], rtol=0, atol=0)
    torch.testing.assert_close(predictions["A1"], predictions["A3"], rtol=0, atol=0)
    changed = replace(
        batch,
        delta_proj=batch.delta_proj + 100,
        s=batch.s - 50,
        hvg_panel_mask=~batch.hvg_panel_mask,
        own_gene_shift_mask=~batch.own_gene_shift_mask,
    )
    for arm in ("A0", "A2"):
        torch.testing.assert_close(
            heads[arm](changed, genes, context), predictions[arm]
        )
    assert not torch.allclose(heads["A1"](changed, genes, context), predictions["A1"])
    predictions["A2"].square().mean().backward()
    assert heads["A2"].slopes.grad.abs().sum() > 0


def test_make_readout_seed_controls_canonical_mlp_initialization():
    from src.model.readout import make_readout

    dims = GeneEffectFeatureDims(delta_proj=3, e_g=2, z_c=10)
    seed0_first = make_readout("A2", dims, 3, seed=0)
    seed0_second = make_readout("A2", dims, 3, seed=0)
    seed1 = make_readout("A2", dims, 3, seed=1)
    torch.testing.assert_close(
        seed0_first.mlp.net[0].weight,
        seed0_second.mlp.net[0].weight,
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(seed0_first.mlp.net[0].weight, seed1.mlp.net[0].weight)
    a0_seed1 = make_readout("A0", dims, 3, seed=1)
    torch.testing.assert_close(
        a0_seed1.mlp.net[0].weight, seed1.mlp.net[0].weight, rtol=0, atol=0
    )


def cache_fixture(root, val_shift=0, response_dtype=torch.float32):
    from src.data.readout_cache import write_feature_cache

    rng = np.random.default_rng(8)
    lines = {"train": [f"C{i}" for i in range(12)], "val": [f"V{i}" for i in range(4)]}
    batches = {}
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    for split, ids in lines.items():
        contexts = rng.normal(size=(len(ids), 10)).astype("float32")
        if split == "val":
            contexts += val_shift
        n = len(ids) * 2
        batch = FeatureBatch(
            delta_proj=torch.tensor(rng.normal(size=(n, 3)), dtype=response_dtype),
            s=torch.tensor(rng.normal(size=(n, 6)), dtype=response_dtype),
            q_sc=torch.zeros(n, 3),
            e_g=embeddings.repeat(len(ids), 1),
            z_c=torch.from_numpy(contexts).repeat_interleave(2, dim=0),
            q_sc_mask=torch.ones(n, dtype=torch.bool),
            hvg_panel_mask=torch.ones(n, dtype=torch.bool),
            own_gene_shift_mask=torch.ones(n, dtype=torch.bool),
            gene_symbols=tuple(["G0", "G1"] * len(ids)),
            model_ids=tuple(line for line in ids for _ in range(2)),
        )
        # Gene-specific slope: opposite response to the first context dimension.
        target = batch.z_c[:, 0] * torch.tensor([0.05, -0.05] * len(ids))
        batches[split] = [(batch, target, torch.zeros(n))]
    return write_feature_cache(
        root,
        batches,
        row_counts={k: len(v) * 2 for k, v in lines.items()},
        split_lines=lines,
        genes=["G0", "G1"],
        variable_genes=["G0", "G1"],
        provenance={"checkpoint_sha256": "fixture"},
    )


def test_cache_preserves_bf16_response_values_in_fp32_storage(tmp_path):
    original = cache_fixture(tmp_path / "fp32")
    autocast = cache_fixture(tmp_path / "bf16", response_dtype=torch.bfloat16)
    for split in ("train", "val"):
        for name in ("delta_proj", "s"):
            expected = torch.tensor(original.splits[split][name]).bfloat16().float()
            actual = torch.tensor(autocast.splits[split][name])
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cache_roundtrip_uses_unique_context_pca_and_train_only_scaling(tmp_path):
    first = cache_fixture(tmp_path / "first")
    shifted = cache_fixture(tmp_path / "shifted", val_shift=100)
    np.testing.assert_allclose(
        first.context_scores[:12].std(axis=0), np.ones(8), atol=1e-6
    )
    np.testing.assert_allclose(first.context_scores[:12], shifted.context_scores[:12])
    assert first.standardizer.to_state() == shifted.standardizer.to_state()
    batch, gene_indices, contexts, residual = first.batch("train", [3, 0, 5])
    assert batch.gene_symbols == ("G1", "G0", "G1")
    assert batch.model_ids == ("C1", "C0", "C2")
    assert gene_indices.tolist() == [1, 0, 1]
    assert contexts.shape == (3, 8)
    assert residual.shape == (3,)
    assert first.arrays["e_g"].shape == (2, 2)
    assert first.arrays["z_c"].shape == (16, 10)


def test_training_stops_on_huber_ties_and_exports_selected_checkpoint(tmp_path):
    from src.training.readout import ReadoutSettings, fit_readout
    from src.eval.readout import evaluate_readout

    cache = cache_fixture(tmp_path / "cache")
    # Zero LR is a controlled tie, through real optimization and evaluation.
    settings = ReadoutSettings(learning_rate=0, batch_size=10)
    run = tmp_path / "A2"
    result = fit_readout(cache, "A2", run, settings=settings)
    assert result["best_epoch"] == 1
    assert result["stopped_epoch"] == 6
    assert result["global_step"] == 18  # three batches, including the four-row tail
    saved = torch.load(run / "best.pt", weights_only=True)
    evaluated = evaluate_readout(cache, saved, "val")
    assert evaluated.metrics == json.loads(
        (run / "evaluation/best/val/metrics.json").read_text()
    )
    assert len(evaluated.predictions) == 8
    assert set(evaluated.predictions.model_id) == {"V0", "V1", "V2", "V3"}
    records = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 6
    assert all(r["train_rows"] == 24 for r in records)
    assert all(
        r["val_geneeffect_loss"] == records[0]["val_geneeffect_loss"] for r in records
    )


def test_real_state_extraction_is_frozen_and_never_calls_old_head(tmp_path):
    from src.experiments.p1a import iter_features
    from src.data.datasets import DependencyDataset
    from test_joint_training import tiny_training_config, fresh_model

    torch.set_num_threads(1)
    model, inputs = fresh_model(tiny_training_config(tmp_path / "inputs"))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []
    hook = model.head.register_forward_hook(lambda *args: calls.append(True))
    data = DependencyDataset(inputs, "train")
    batches = list(iter_features(model, data, batch_size=3, precision="no"))
    hook.remove()
    assert not calls
    assert sum(batch.batch_size for batch, _, _ in batches) == len(data)
    assert all(
        not value.requires_grad
        for batch, _, _ in batches
        for value in (batch.delta_proj, batch.s)
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_paired_context_bootstrap_aligns_keys_and_preserves_undefined_genes(tmp_path):
    from src.eval.readout_comparison import paired_diagnostic
    import pytest

    cache = cache_fixture(tmp_path / "cache")
    left = cache.labels("val")
    right = left.copy()
    left["residual_prediction"] = 0.0
    right["residual_prediction"] = right.residual
    groups = {"V0": "P0", "V1": "P0", "V2": "P1", "V3": "P2"}
    result = paired_diagnostic(
        left,
        right.sample(frac=1, random_state=2),
        cache.variable_genes,
        groups=groups,
        repeats=20,
    )
    assert result["clusters"] == 3
    assert result["point"]["huber_delta"] < 0
    assert result["point"]["pearson_common_genes"] == 0
    assert result["point"]["pearson_delta"] is None
    assert result["point"]["sd_ratio_delta"] == 1.0
    right.loc[0, "residual"] += 1
    with pytest.raises(ValueError, match="targets"):
        paired_diagnostic(left, right, cache.variable_genes, groups=groups, repeats=20)


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    from src.training.readout import ReadoutSettings, fit_readout
    import pytest

    cache = cache_fixture(tmp_path / "cache")
    settings = ReadoutSettings(max_epochs=3, batch_size=10)
    whole, interrupted = tmp_path / "whole", tmp_path / "interrupted"
    fit_readout(cache, "A3", whole, settings=settings)
    save = torch.save
    calls = 0

    def failing_disk_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("disk interruption")
        return save(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", failing_disk_save)
        with pytest.raises(OSError, match="disk interruption"):
            fit_readout(cache, "A3", interrupted, settings=settings)
    fit_readout(
        cache, "A3", interrupted, settings=settings, resume=interrupted / "last.pt"
    )
    expected = torch.load(whole / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    for name in expected["model_state"]:
        torch.testing.assert_close(
            actual["model_state"][name], expected["model_state"][name], rtol=0, atol=0
        )
    assert actual["train_state"] == expected["train_state"]
    assert (interrupted / "metrics.jsonl").read_text() == (
        whole / "metrics.jsonl"
    ).read_text()


def test_head_seed_is_recorded_and_guards_resume(tmp_path):
    from src.training.readout import ReadoutSettings, fit_readout
    import pytest

    cache = cache_fixture(tmp_path / "cache")
    settings = ReadoutSettings(max_epochs=1, batch_size=10)
    run = tmp_path / "A0"
    fit_readout(cache, "A0", run, settings=settings, head_seed=1)
    record = json.loads((run / "run.json").read_text())
    assert record["head_seed"] == 1
    saved = torch.load(run / "best.pt", weights_only=True)
    assert saved["head_seed"] == 1
    with pytest.raises(ValueError, match="head seed"):
        fit_readout(
            cache, "A0", run, settings=settings, resume=run / "last.pt", head_seed=0
        )


def test_four_arm_comparison_cli_exports_common_updates_and_gene_changes(tmp_path):
    from src.training.readout import ReadoutSettings, fit_readout
    from src.experiments.p1a import main
    import pandas as pd

    cache = cache_fixture(tmp_path / "cache")
    for arm in ("A0", "A1", "A2", "A3"):
        fit_readout(
            cache,
            arm,
            tmp_path / "runs" / arm,
            settings=ReadoutSettings(max_epochs=1, batch_size=10),
        )
    mapping = tmp_path / "contexts.csv"
    pd.DataFrame(
        {
            "model_id": [f"V{i}" for i in range(4)],
            "patient_id": ["P0", "P0", "P1", "P2"],
        }
    ).to_csv(mapping, index=False)
    main(
        [
            "compare",
            "--cache",
            str(cache.root),
            "--runs",
            str(tmp_path / "runs"),
            "--out-dir",
            str(tmp_path / "comparison"),
            "--context-map",
            str(mapping),
            "--bootstrap-repeats",
            "10",
        ]
    )
    selected = pd.read_csv(tmp_path / "comparison/selected.csv")
    assert len(selected) == 8
    paired = json.loads((tmp_path / "comparison/paired.json").read_text())
    assert set(paired) == {"A2-minus-A0", "A3-minus-A1", "A1-minus-A0", "A3-minus-A2"}
    assert all(value["clusters"] == 3 for value in paired.values())
    common = pd.read_csv(tmp_path / "comparison/A2-minus-A0-common-updates.csv")
    assert common.global_step.tolist() == [3]
    per_gene = pd.read_csv(tmp_path / "comparison/A2-minus-A0-per-gene.csv")
    assert set(per_gene.gene_symbol) == {"G0", "G1"}


def test_branch_penalty_covers_genes_absent_from_batch_and_is_not_validation_loss(
    tmp_path,
):
    from src.model.readout import make_readout
    from src.eval.readout import evaluate_head
    from torch.nn import functional as F

    cache = cache_fixture(tmp_path / "cache")
    head = make_readout("A2", cache.dims, len(cache.genes))
    with torch.no_grad():
        head.slopes.fill_(1.0)
    assert head.regularization().item() == np.float32(0.08)
    batch, genes, context, target = cache.batch("train", [0, 2])
    objective = (
        F.huber_loss(head(batch, genes, context), target) + head.regularization()
    )
    objective.backward()
    # Only G0 was sampled; G1 still receives the full-gene regularizer gradient.
    torch.testing.assert_close(head.slopes.grad[1], torch.full((8,), 0.01))
    result = evaluate_head(head, cache, "val")
    pred = torch.tensor(result.predictions.residual_prediction.to_numpy())
    truth = torch.tensor(result.predictions.residual.to_numpy())
    assert result.metrics["val_geneeffect_loss"] == F.huber_loss(pred, truth).item()


def test_bootstrap_point_matches_production_metrics_with_missing_pairs(tmp_path):
    from src.eval.readout_comparison import paired_diagnostic
    from src.eval.geneeffect import aggregate_geneeffect
    import pytest

    cache = cache_fixture(tmp_path / "cache")
    left = cache.labels("val").drop(index=0).reset_index(drop=True)
    right = left.copy()
    rng = np.random.default_rng(3)
    left["residual_prediction"] = rng.normal(size=len(left))
    right["residual_prediction"] = rng.normal(size=len(right))
    groups = {line: line for line in left.model_id.unique()}
    paired = paired_diagnostic(
        left, right, cache.variable_genes, groups=groups, repeats=5
    )
    details = []
    for frame in (left, right):
        frame["geneeffect_prediction"] = frame.residual_prediction
        _, _, genes = aggregate_geneeffect(
            frame,
            model_ids=list(groups),
            genes=cache.genes,
            variable_genes=cache.variable_genes,
        )
        details.append(genes)
    for metric in ("pearson", "spearman", "sd_ratio"):
        difference = (details[1][metric] - details[0][metric]).mean()
        assert paired["point"][f"{metric}_delta"] == pytest.approx(
            difference, abs=1e-10
        )


def test_export_retry_updates_completion_without_retraining(tmp_path, monkeypatch):
    from src.training.readout import ReadoutSettings, fit_readout
    from src.experiments.p1a import main
    import pytest
    import pandas as pd

    cache = cache_fixture(tmp_path / "cache")
    run = tmp_path / "A0"

    def failed_export(*args, **kwargs):
        raise OSError("export unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_parquet", failed_export)
        with pytest.raises(OSError, match="export unavailable"):
            fit_readout(cache, "A0", run, settings=ReadoutSettings(max_epochs=1))
    before = (run / "best.pt").read_bytes()
    record = json.loads((run / "run.json").read_text())
    assert record["training"] == "completed"
    assert record["evaluation"] == "failed"
    main(
        [
            "evaluate",
            "--cache",
            str(cache.root),
            "--checkpoint",
            str(run / "best.pt"),
            "--device",
            "cpu",
        ]
    )
    assert json.loads((run / "run.json").read_text())["evaluation"] == "completed"
    assert (run / "best.pt").read_bytes() == before


def test_constant_nonzero_predictions_remain_undefined_with_missing_pairs(tmp_path):
    from src.eval.readout_comparison import paired_diagnostic

    cache = cache_fixture(tmp_path / "cache")
    left = cache.labels("val").drop(index=0)
    right = left.copy()
    left["residual_prediction"] = 0.1
    right["residual_prediction"] = right.residual
    result = paired_diagnostic(
        left,
        right,
        cache.variable_genes,
        groups={f"V{i}": f"V{i}" for i in range(4)},
        repeats=5,
    )
    assert result["point"]["pearson_common_genes"] == 0
    assert result["point"]["pearson_delta"] is None

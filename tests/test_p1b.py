import numpy as np
import pytest

from src.data.p1b import split_conditions, balanced_epoch


def test_holdout_is_never_sampled_and_epoch_is_anchor_balanced():
    keys = [(a, str(i)) for a in ("a", "b", "c", "j") for i in range(5)]
    holdout = {(a, "4") for a in ("a", "b", "c")}
    splits = split_conditions(keys, holdout, ("a", "b", "c"), "j")
    batches = list(
        balanced_epoch(keys, splits["train"], ("a", "b", "c"), 0, per_anchor=2)
    )
    assert len(batches) == 2
    assert batches == list(
        balanced_epoch(keys, splits["train"], ("a", "b", "c"), 0, per_anchor=2)
    )
    for batch in batches:
        assert [sum(keys[i][0] == a for i in batch) for a in ("a", "b", "c")] == [
            2,
            2,
            2,
        ]
        assert set(batch) <= set(splits["train"])
    assert len(splits["external"]) == 5


def test_measured_mask_and_derangements_preserve_exact_coordinate_semantics():
    from src.data.p1b import common_coordinates, derangements

    assert common_coordinates(
        ["A", "B", "c"], [{"A", "B", "c"}, {"A", "B", "C"}], expected=2
    ) == [0, 1]
    with pytest.raises(ValueError, match="measured"):
        common_coordinates(["A", "B"], [{"A"}], expected=2)
    maps = derangements(["A", "B", "C"])
    assert len(maps) == 10
    assert maps == derangements(["A", "B", "C"])
    assert all(
        set(m.values()) == {"A", "B", "C"} and all(k != v for k, v in m.items())
        for m in maps
    )


def test_interface_gradients_cross_frozen_eval_backbone():
    import torch
    from torch import nn
    from src.model.p1b import configure_parameters
    from src.model.state import ForwardOnlyStateModel, StateForwardAdapter
    from src.model.perturbation import Esm2PerturbationAdapter
    from src.data.embeddings import Esm2EmbeddingTable

    class State(nn.Module):
        def __init__(self):
            super().__init__()
            self.basal_encoder = nn.Sequential(nn.Linear(2560, 328))
            self.project_out = nn.Linear(328, 2)

    backbone = ForwardOnlyStateModel(
        StateForwardAdapter(State()),
        Esm2PerturbationAdapter(
            ["G"],
            Esm2EmbeddingTable(1280, {"G": np.ones(1280, dtype=np.float32)}),
            512,
            2024,
        ),
    )
    inherited = [
        n
        for n, _ in backbone.state_adapter.state_model.named_parameters()
        if n != "basal_encoder.0.weight"
    ]
    groups, report = configure_parameters(backbone, inherited, unfreeze=False)
    assert sum(p.numel() for g in groups for p in g["params"]) == 2533864
    assert not backbone.training
    state = backbone.state_adapter.state_model
    state.project_out(state.basal_encoder(torch.ones(1, 2560))).sum().backward()
    assert state.basal_encoder[0].weight.grad is not None
    assert state.project_out.weight.grad is None
    groups, _ = configure_parameters(backbone, inherited, unfreeze=True)
    assert len(groups) == 2 and groups[1]["lr"] == 1e-6
    assert not state.training and state.project_out.weight.requires_grad


def test_response_scoring_masks_own_gene_and_baselines_fit_train_only():
    import torch
    from src.eval.p1b import score_bag
    from src.data.p1b import fit_baselines

    pred, obs, basal = (
        torch.tensor([[9.0, 1.0, 100.0]]),
        torch.tensor([[1.0, 1.0, -100.0]]),
        torch.zeros(1, 3),
    )
    score = score_bag(pred, obs, basal, [0, 1], own=0)
    assert score["mean_delta_mse"] == 32
    assert score["without_own_mean_delta_mse"] == 0
    rows = [
        ("a", "G", np.array([1.0, 2.0])),
        ("b", "G", np.array([3.0, 4.0])),
        ("b", "H", np.array([5.0, 6.0])),
    ]
    baseline = fit_baselines(rows)
    np.testing.assert_allclose(baseline["global"], [2.5, 3.5])
    np.testing.assert_allclose(baseline["genes"]["G"], [2.0, 3.0])
    assert "unseen" not in baseline["genes"]


def test_epoch_zero_early_stop_resume_and_stage_gate(tmp_path):
    import torch
    from src.training.p1b import fit, stage2_eligible

    model = torch.nn.Linear(1, 1)

    def train_epoch(epoch):
        return [torch.ones(1, 1)]

    def objective(batch):
        return model(batch).square().mean()

    def evaluate():
        return {"response_loss": 2.0}

    groups = [{"params": list(model.parameters()), "lr": 1e-4, "name": "interface"}]
    fit(model, groups, train_epoch, objective, evaluate, tmp_path, identity="test")
    saved = torch.load(tmp_path / "last.pt", weights_only=False)
    assert saved["epoch"] == 5 and saved["best_epoch"] == 0
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["epoch"] == 0
    fit(
        model,
        groups,
        train_epoch,
        objective,
        evaluate,
        tmp_path,
        identity="test",
        resume=True,
    )
    assert torch.load(tmp_path / "last.pt", weights_only=False)["epoch"] == 5
    assert stage2_eligible(2.0, 1.97) and not stage2_eligible(2.0, 1.99)


def test_preparation_snapshot_and_real_prediction_path(tmp_path):
    from types import SimpleNamespace
    import torch
    from src.experiments.p1b import loss_for_indices
    from src.data.p1b import build_snapshot, ResponseView
    from src.data.response_cache import ResponseTargetsCache
    import pandas as pd

    anchors = ("a", "b", "c")
    keys = [(a, g) for a in (*anchors, "j") for g in ("G", "H")]
    cache = ResponseTargetsCache(
        tuple(a for a, g in keys),
        tuple(g for a, g in keys),
        np.ones((16, 3), dtype=np.float32),
        np.arange(0, 17, 2),
        pd.DataFrame(),
    )
    inputs = SimpleNamespace(
        response_targets=cache,
        response_holdout=frozenset((a, "H") for a in anchors),
        hvg_order=("G", "H", "I"),
        lines={
            a: SimpleNamespace(
                controls_tx1=np.zeros((2, 3), dtype=np.float32),
                basal_hvg=np.zeros((2, 3), dtype=np.float32),
            )
            for a in (*anchors, "j")
        },
    )
    bundle = build_snapshot(inputs, [0, 1], {"G"}, anchors=anchors, external="j")
    assert bundle["splits"]["external"] == [6, 7]
    assert all(keys[i][0] != "j" for i in bundle["splits"]["train"])
    view = ResponseView(bundle, cache)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, controls, genes, batch_indices):
            return tuple(c + self.bias for c in controls)

    model = Model().eval()
    loss = loss_for_indices(model, view, bundle["splits"]["train"], device="cpu")
    loss.backward()
    assert model.bias.grad[:2].abs().sum() > 0
    assert model.bias.grad[2] == 0


def test_export_scores_identity_controls_and_cross_context(tmp_path):
    import torch
    from types import SimpleNamespace
    import pandas as pd
    from src.data.p1b import build_snapshot, ResponseView
    from src.data.response_cache import ResponseTargetsCache
    from src.eval.p1b import export_predictions, cross_context

    keys = [(a, g) for a in ("a", "b", "c", "j") for g in ("G", "H", "I")]
    targets = np.concatenate(
        [
            np.full((2, 3), {"G": 1, "H": 2, "I": 3}[g], dtype=np.float32)
            for a, g in keys
        ]
    )
    cache = ResponseTargetsCache(
        tuple(a for a, g in keys),
        tuple(g for a, g in keys),
        targets,
        np.arange(0, 25, 2),
        pd.DataFrame(),
    )
    inputs = SimpleNamespace(
        response_targets=cache,
        response_holdout=frozenset((a, g) for a in ("a", "b", "c") for g in ("H", "I")),
        hvg_order=("G", "H", "I"),
        lines={
            a: SimpleNamespace(
                controls_tx1=np.zeros((2, 3), dtype=np.float32),
                basal_hvg=np.zeros((2, 3), dtype=np.float32),
            )
            for a in ("a", "b", "c", "j")
        },
    )
    bundle = build_snapshot(
        inputs, [0, 1, 2], {"G", "H", "I"}, anchors=("a", "b", "c"), external="j"
    )

    class Model(torch.nn.Module):
        def forward(self, controls, genes, batches):
            return tuple(
                c + {"G": 1, "H": 2, "I": 3}[g] for c, g in zip(controls, genes)
            )

    frame, effects = export_predictions(
        Model().eval(),
        ResponseView(bundle, cache),
        ["val", "external"],
        device="cpu",
        directory=tmp_path,
    )
    correct = frame[(frame.method == "model") & (frame.panel == "all")]
    assert correct.response_loss.max() == 0
    assert correct.identity_advantage.min() > 0
    assert (tmp_path / "effects.npz").is_file()
    cross = cross_context(correct, effects, bundle)
    assert not cross.empty and cross.effect_difference_mse.max() == 0


def test_cli_has_all_stages_and_requires_explicit_external_access():
    from src.experiments.p1b import parser

    p = parser()
    args = p.parse_args(
        ["evaluate", "--prepared", "p", "--runs", "r", "--state", "B-init"]
    )
    assert not args.external
    for stage in ("prepare", "train-interface", "train-stage2", "compare"):
        with pytest.raises(SystemExit) as exc:
            p.parse_args([stage, "--help"])
        assert exc.value.code == 0


def test_paired_gene_bootstrap_and_relative_improvement():
    import pandas as pd
    from src.eval.p1b_comparison import paired_difference

    left = pd.DataFrame({"gene": ["a", "b"], "response_loss": [1.0, 2.0]})
    right = pd.DataFrame({"gene": ["b", "a"], "response_loss": [4.0, 2.0]})
    report = paired_difference(left, right, "response_loss")
    assert report["delta"] == -1.5
    assert report["relative_improvement_pct"] == 50.0
    assert report["ci_low"] <= report["delta"] <= report["ci_high"]


def test_real_state_resume_matches_uninterrupted_and_restores_without_upstream(
    tmp_path,
):
    import torch
    from test_joint_training import tiny_training_config, fresh_model, assert_tree_equal
    from src.experiments.p1b_preparation import restore_backbone
    from src.experiments.p1b import loss_for_indices
    from src.data.p1b import build_snapshot, ResponseView, balanced_epoch
    from src.eval.p1b import aggregate, evaluate_rows
    from src.training.p1b import fit

    torch.set_num_threads(1)
    config = tiny_training_config(tmp_path / "inputs")
    joint, inputs = fresh_model(config)
    anchors = inputs.response_anchors[:3]
    external = inputs.response_anchors[3]
    bundle = build_snapshot(
        inputs,
        list(range(1957)),
        set(inputs.esm2_symbols),
        anchors=anchors,
        external=external,
    )
    view = ResponseView(bundle, inputs.response_targets)
    template = {
        "config": config,
        "preprocessing": inputs.preprocessing_state(),
        "architecture": joint.architecture,
        "model_state": joint.backbone.state_dict(),
    }
    from pathlib import Path

    Path(config["paths"]["state_checkpoint"]).unlink()

    def run(path, epochs, resume=False):
        backbone = restore_backbone(template)
        backbone.eval()
        # Tiny fixture dimensions differ; exercise actual installed STATE through
        # the same training API while the full-size grouping has its own test.
        backbone.requires_grad_(False)
        params = list(backbone.perturbations.parameters())
        for p in params:
            p.requires_grad_(True)
        groups = [{"name": "interface", "params": params, "lr": 1e-4}]
        fit(
            backbone,
            groups,
            lambda e: balanced_epoch(
                bundle["keys"], bundle["splits"]["train"], anchors, e, per_anchor=1
            ),
            lambda i: loss_for_indices(backbone, view, i, device="cpu"),
            lambda: aggregate(
                evaluate_rows(backbone, view, bundle["splits"]["val"], device="cpu")
            ),
            path,
            identity="tiny",
            max_epochs=epochs,
            resume=resume,
        )
        return torch.load(path / "last.pt", weights_only=False)

    uninterrupted = run(tmp_path / "full", 2)
    run(tmp_path / "split", 1)
    resumed = run(tmp_path / "split", 2, resume=True)
    assert_tree_equal(uninterrupted["model_state"], resumed["model_state"])
    assert_tree_equal(uninterrupted["optimizer"], resumed["optimizer"])
    assert uninterrupted["history"] == resumed["history"]


def test_common_updates_compare_only_matching_stage_two_steps():
    import pandas as pd
    from src.eval.p1b_comparison import matched_updates

    rows = pd.DataFrame(
        [
            {"arm": "B-interface", "step": 1, "response_loss": 99},
            {"arm": "B-continue", "step": 1, "response_loss": 3},
            {"arm": "B-continue", "step": 2, "response_loss": 2},
            {"arm": "B-unfreeze", "step": 1, "response_loss": 1},
        ]
    )
    result = matched_updates(rows)
    assert result.step.tolist() == [1]
    assert result.response_loss_delta.tolist() == [-2]


def test_real_cli_export_retry_preserves_checkpoint(tmp_path, monkeypatch):
    import json
    import torch
    import pandas as pd
    from pathlib import Path
    from test_joint_training import tiny_training_config, fresh_model
    from src.data.p1b import build_snapshot
    from src.experiments.p1b import main
    from src.experiments.p1b_preparation import digest

    torch.set_num_threads(1)
    config = tiny_training_config(tmp_path / "inputs")
    model, inputs = fresh_model(config)
    bundle = build_snapshot(
        inputs,
        list(range(1957)),
        set(inputs.esm2_symbols),
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
    template = {
        "config": config,
        "preprocessing": inputs.preprocessing_state(),
        "architecture": model.architecture,
        "model_state": model.backbone.state_dict(),
    }
    torch.save(template, prepared / "B-init.pt")
    manifest = {
        "bundle_sha256": digest(prepared / "bundle.pt"),
        "model_files": {"B-init": digest(prepared / "B-init.pt")},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    (prepared / "status.json").write_text(json.dumps({"status": "completed"}))
    Path(config["paths"]["state_checkpoint"]).unlink()
    arguments = [
        "evaluate",
        "--prepared",
        str(prepared),
        "--runs",
        str(tmp_path / "runs"),
        "--state",
        "B-init",
        "--device",
        "cpu",
    ]
    original = pd.DataFrame.to_parquet

    def fail(*args, **kwargs):
        raise OSError("export interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="export interrupted"):
        main(arguments)
    status = tmp_path / "runs/evaluation/B-init/internal/evaluation.json"
    assert json.loads(status.read_text())["status"] == "failed"
    monkeypatch.setattr(pd.DataFrame, "to_parquet", original)
    main(arguments)
    assert json.loads(status.read_text())["status"] == "completed"
    assert digest(prepared / "B-init.pt") == manifest["model_files"]["B-init"]
    main(["compare", "--prepared", str(prepared), "--runs", str(tmp_path / "runs")])
    assert (tmp_path / "runs/comparison/paired_differences.csv").is_file()


def test_external_selection_requires_both_stage_two_completions(tmp_path):
    import json
    from src.experiments.p1b import freeze_external_selection
    from src.experiments.p1b_preparation import digest

    manifest = {"bundle_sha256": "bundle"}
    for arm in ("B-interface", "B-continue"):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "best.pt").write_bytes(arm.encode())
        (directory / "training.json").write_text(
            json.dumps({"status": "completed", "identity": f"bundle:{arm}:source"})
        )
    decision = {
        "bundle": "bundle",
        "eligible": True,
        "start_sha256": digest(tmp_path / "B-interface/best.pt"),
    }
    (tmp_path / "stage2.json").write_text(json.dumps(decision))
    with pytest.raises(FileNotFoundError):
        freeze_external_selection(None, tmp_path, manifest)
    assert not (tmp_path / "external_evaluation.json").exists()
    directory = tmp_path / "B-unfreeze"
    directory.mkdir()
    (directory / "best.pt").write_bytes(b"weights")
    (directory / "training.json").write_text(
        json.dumps({"status": "completed", "identity": "bundle:B-unfreeze:source"})
    )
    freeze_external_selection(None, tmp_path, manifest)
    assert (tmp_path / "external_evaluation.json").exists()
    (directory / "best.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="selected checkpoints changed"):
        freeze_external_selection(None, tmp_path, manifest)


def test_comparison_rejects_stale_exports_before_reading_metrics(tmp_path):
    import json
    from src.eval.p1b_comparison import compare

    prepared, runs = tmp_path / "prepared", tmp_path / "runs"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(json.dumps({"bundle_sha256": "bundle"}))
    checkpoint = runs / "B-interface/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"new selected weights")
    export = runs / "evaluation/B-interface/internal"
    export.mkdir(parents=True)
    (export / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "bundle": "bundle",
                "state": "B-interface",
                "checkpoint_sha256": "old",
            }
        )
    )
    with pytest.raises(ValueError, match="stale export"):
        compare(prepared, runs)


def test_descriptive_sensitivity_has_no_relative_improvement():
    import pandas as pd
    from src.eval.p1b_comparison import paired_difference

    a = pd.DataFrame({"gene": ["G"], "output_change_mse": [2.0]})
    b = pd.DataFrame({"gene": ["G"], "output_change_mse": [1.0]})
    assert (
        paired_difference(a, b, "output_change_mse")["relative_improvement_pct"] is None
    )


def test_baseline_preparation_never_reads_external_targets():
    from types import SimpleNamespace
    from src.data.p1b import build_snapshot

    keys = tuple((a, g) for a in ("a", "b", "c", "j") for g in ("G", "H"))

    class Cache:
        def __init__(self):
            self.keys = keys

        def target_bag(self, index):
            assert keys[index][0] != "j", "external targets leaked into fitting"
            assert keys[index][1] != "H", (
                "internal validation targets leaked into fitting"
            )
            return np.ones((2, 3), dtype=np.float32)

    inputs = SimpleNamespace(
        response_targets=Cache(),
        response_holdout={(a, "H") for a in ("a", "b", "c")},
        hvg_order=["G", "H", "I"],
        lines={
            a: SimpleNamespace(
                controls_tx1=np.zeros((2, 3)), basal_hvg=np.zeros((2, 3))
            )
            for a in ("a", "b", "c", "j")
        },
    )
    result = build_snapshot(
        inputs, [0, 1], {"G", "H"}, anchors=("a", "b", "c"), external="j"
    )
    assert set(result["baselines"]["genes"]) == {"G"}


def test_source_vocabularies_reads_only_var_without_observation_table(tmp_path):
    import h5py
    import pandas as pd
    from anndata.io import write_elem
    from src.data.p1b import source_vocabularies

    path = tmp_path / "var_only.h5ad"
    with h5py.File(path, "w") as handle:
        write_elem(handle, "var", pd.DataFrame({"gene_name": ["A", "b"]}))
    result = source_vocabularies(
        {
            "anchor": {
                "source_type": "h5ad",
                "h5ad_path": str(path),
                "target_gene_symbol_col": "gene_name",
            }
        }
    )
    assert result == {"anchor": {"A", "b"}}

"""Joint GeneEffect model: STATE on its own HVG basal path, training and evaluation.

Every test builds ``PreparedInputs`` directly (2000 HVGs, 8-wide Tx1 cells, 4-wide
ESM2 vectors) and a tiny released-STATE checkpoint from the installed arc-state
class, so nothing reads raw data, Tx1 weights or the real prepared root.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

from src.data.batches import DependencyBatch  # noqa: E402
from src.data.datasets import DependencyDataset, ResponseDataset  # noqa: E402
from src.data.prepared import PreparedInputs, PreparedLine  # noqa: E402
from src.data.q_sc import QScFeatures  # noqa: E402
from src.data.splits import FixedSplit  # noqa: E402
from src.eval.geneeffect import aggregate_geneeffect, evaluate_model  # noqa: E402
from src.experiments import geneeffect  # noqa: E402
from src.experiments.config import load_config, validate_config  # noqa: E402
from src.model.initialization import (  # noqa: E402
    build_joint_model,
    warm_start_state_dict,
)
from src.model.normalization import fit_startup_standardizer  # noqa: E402
from src.model.state import build_state, load_released_state, released_hparams  # noqa: E402
from src.training import trainer  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.sampling import balanced_responses, make_training_loaders  # noqa: E402

CONFIG = Path("configs/geneeffect_joint.yaml")
RELEASED = Path(
    "model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/checkpoints/final.ckpt"
)
HVG = 2000
TX1 = 8
ESM2 = 4
CELLS = 8
SENTENCE = 4
ANCHORS = ("ACH-A0", "ACH-A1", "ACH-A2", "ACH-A3")
TRAIN = (*ANCHORS, "ACH-T0", "ACH-T1")
VAL = ("ACH-V0", "ACH-V1")
TEST = ("ACH-X0",)
GENES = ("G0", "G1", "G2", "G3", "G4", "G5")
HVG_ORDER = ("G0", "G1", *(f"H{i}" for i in range(HVG - 2)))
RESPONSE_GENES = ("G0", "G2", "G4")


class FakeResponseTargets:
    """The prepared response cache's interface: ordered keys and one bag per key."""

    def __init__(self, bags: dict[tuple[str, str], np.ndarray]) -> None:
        self.keys = tuple(bags)
        self._bags = list(bags.values())

    def target_bag(self, index: int) -> np.ndarray:
        return self._bags[index]


def make_inputs(seed: int = 0) -> PreparedInputs:
    rng = np.random.default_rng(seed)
    lines = (*TRAIN, *VAL, *TEST)
    labels = pd.DataFrame(
        [(m, g, float(rng.normal(-0.3, 0.5))) for m in lines for g in GENES],
        columns=["model_id", "gene_symbol", "gene_effect"],
    )
    means = (
        labels[labels.model_id.isin(TRAIN)].groupby("gene_symbol").gene_effect.mean()
    )
    labels["residual"] = labels.gene_effect - labels.gene_symbol.map(means)
    prepared_lines = {
        m: PreparedLine(
            controls_tx1=rng.normal(size=(CELLS, TX1)).astype(np.float32),
            basal_hvg=np.log1p(rng.poisson(1.0, size=(CELLS, HVG))).astype(np.float32),
            q_sc=QScFeatures(
                symbols=GENES,
                values=rng.random((len(GENES), 3)),
                available=np.array([True, True, True, False, True, True]),
            ),
        )
        for m in lines
    }
    bags = {
        (anchor, gene): np.log1p(rng.poisson(1.5, size=(5, HVG))).astype(np.float32)
        for anchor in ANCHORS
        for gene in RESPONSE_GENES
    }
    return PreparedInputs(
        split=FixedSplit(train=TRAIN, val=VAL, test=TEST),
        labels=labels,
        genes=GENES,
        train_gene_means=means.reindex(list(GENES)),
        variable_genes=frozenset(GENES),
        hvg_order=HVG_ORDER,
        esm2_symbols=GENES,
        esm2_vectors=rng.normal(size=(len(GENES), ESM2)).astype(np.float32),
        lines=prepared_lines,
        response_targets=FakeResponseTargets(bags),
        response_anchors=ANCHORS,
        target_sum=1000.0,
    )


def write_released_state(path: Path, *, dropout: float = 0.0) -> Path:
    """A STATE-shaped released checkpoint: 2000-wide basal encoder, batch encoder."""
    hparams = dict(
        input_dim=HVG,
        hidden_dim=8,
        output_dim=HVG,
        pert_dim=6,
        batch_dim=3,
        batch_encoder=True,
        cell_set_len=SENTENCE,
        predict_residual=True,
        output_space="gene",
        embed_key="X_hvg",
        gene_names=list(HVG_ORDER),
        transformer_backbone_kwargs={
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 2,
            "resid_pdrop": dropout,
            "embd_pdrop": dropout,
            "attn_pdrop": dropout,
        },
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(1)
    state = build_state(hparams)
    torch.save({"hyper_parameters": hparams, "state_dict": state.state_dict()}, path)
    return path


def make_config(root: Path, *, dropout: float = 0.0) -> dict:
    config = load_config(CONFIG)
    config["precision"] = "no"
    config["output_root"] = str(root / "runs")
    config["paths"]["state_checkpoint"] = str(
        write_released_state(root / "released.ckpt", dropout=dropout)
    )
    config["model"].update(
        cell_sentence_len=SENTENCE, esm2_adapter_hidden=4, head_hidden=8, head_layers=1
    )
    config["train"].update(
        max_epochs=2, patience=5, dependency_batch_size=4, response_batch_size=8
    )
    return config


@pytest.fixture
def cpu(monkeypatch):
    # Keep Accelerate off a visible MPS device for the whole process.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    torch.set_num_threads(1)


@pytest.fixture
def world(tmp_path, cpu):
    config = make_config(tmp_path)
    inputs = make_inputs()
    torch.manual_seed(0)
    return SimpleNamespace(
        config=config, inputs=inputs, model=build_joint_model(config, inputs)
    )


def test_state_own_basal_path_loads_every_released_weight(world):
    released = torch.load(world.config["paths"]["state_checkpoint"], weights_only=False)
    fresh = build_state(
        released_hparams(world.config["paths"]["state_checkpoint"], cell_set_len=4)
    )
    report = warm_start_state_dict(fresh, world.config["paths"]["state_checkpoint"])
    assert not (report.missing_keys or report.unexpected_keys)
    assert not report.shape_skipped_keys
    state = world.model.backbone.state
    assert state.basal_encoder[0].weight.shape == (8, HVG)  # not the Tx1 width
    loaded = state.state_dict()
    assert loaded.keys() == released["state_dict"].keys()
    for name, value in released["state_dict"].items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)


@pytest.mark.skipif(not RELEASED.is_file(), reason="released STATE checkpoint absent")
def test_real_released_checkpoint_matches_its_own_architecture_at_2000_inputs():
    hparams = released_hparams(RELEASED, cell_set_len=64)
    assert hparams["input_dim"] == HVG and hparams["pert_dim"] == 2024
    expected = {
        name: tuple(value.shape)
        for name, value in build_state(hparams).state_dict().items()
    }
    checkpoint = torch.load(RELEASED, map_location="cpu", weights_only=False, mmap=True)
    actual = {name: tuple(v.shape) for name, v in checkpoint["state_dict"].items()}
    assert actual == expected
    assert actual["basal_encoder.0.weight"] == (328, 2000)


def test_tx1_does_not_reach_state(world):
    inputs, model = world.inputs, world.model
    fit_startup_standardizer(model, inputs, batch_size=8)
    seen = []
    hook = model.backbone.state.register_forward_pre_hook(
        lambda module, args: seen.append(args[0]["ctrl_cell_emb"].clone())
    )
    batch = DependencyDataset(inputs, "train").collate(range(len(GENES)))
    with torch.no_grad():
        features = model.condition_features(batch.conditions)
    hook.remove()
    basal = torch.from_numpy(inputs.lines[TRAIN[0]].basal_hvg)
    torch.testing.assert_close(seen[0], basal.repeat(len(GENES), 1))

    shifted = copy.copy(inputs)
    object.__setattr__(
        shifted,
        "lines",
        {
            m: PreparedLine(line.controls_tx1 + 5.0, line.basal_hvg, line.q_sc)
            for m, line in inputs.lines.items()
        },
    )
    other = DependencyDataset(shifted, "train").collate(range(len(GENES)))
    with torch.no_grad():
        moved = model.condition_features(other.conditions)
    torch.testing.assert_close(moved.delta_proj, features.delta_proj)
    torch.testing.assert_close(moved.s, features.s)
    assert not torch.allclose(moved.z_c, features.z_c)


def test_zero_key_checkpoint_load_raises(tmp_path):
    path = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"not.a.state.key": torch.zeros(2)}}, path)
    with pytest.raises(ValueError, match="zero keys"):
        warm_start_state_dict(torch.nn.Linear(2, 2), path)


def test_released_load_raises_on_any_incomplete_key(tmp_path, cpu):
    path = write_released_state(tmp_path / "released.ckpt")
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["state_dict"]["extra.weight"] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="does not load completely"):
        load_released_state(path, cell_set_len=SENTENCE)


def test_response_batches_cover_all_conditions():
    inputs = make_inputs()
    dataset = ResponseDataset(inputs)
    stream = balanced_responses(dataset, batch_size=8, epoch=0, rank=0)
    seen = []
    for _ in range(3):  # 3 batches x 2 per anchor = every anchor's 3 genes twice
        batch = next(stream)
        assert sorted(batch.model_ids) == sorted(ANCHORS * 2)
        seen.extend(zip(batch.model_ids, batch.genes, strict=True))
    assert set(seen) == set(inputs.response_targets.keys)
    assert len(set(inputs.response_targets.keys)) == len(ANCHORS) * len(RESPONSE_GENES)
    index = dataset.keys.index(("ACH-A1", "G2"))
    collated = dataset.collate([index])
    torch.testing.assert_close(
        collated.observed_hvg[0],
        torch.from_numpy(inputs.response_targets.target_bag(index)),
    )
    torch.testing.assert_close(
        collated.control_hvg[0], torch.from_numpy(inputs.lines["ACH-A1"].basal_hvg)
    )


def test_training_batches_shard_disjointly_across_ranks():
    inputs = make_inputs()
    config = {"train": {"dependency_batch_size": 2, "response_batch_size": 4}}
    keys, steps = [], []
    for rank in range(3):
        accelerator = SimpleNamespace(process_index=rank, num_processes=3, device="cpu")
        loader, _ = make_training_loaders(inputs, config, 0, accelerator)
        steps.append(len(loader))
        keys += [
            key
            for batch in loader
            for key in zip(batch.conditions.model_ids, batch.conditions.genes)
        ]
    assert len(set(steps)) == 1 and len(keys) == len(set(keys))
    assert len(keys) == (len(TRAIN) * len(GENES) // 6) * 6


def test_optimizer_rates(world):
    config = load_config(CONFIG)
    optimizer = trainer.make_optimizer(world.model, config)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert {name: group["lr"] for name, group in groups.items()} == {
        "state": 1e-5,
        "adapter": 1e-4,
        "head": 1e-4,
    }
    assigned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(assigned) == sorted(id(p) for p in world.model.parameters())
    assert {id(p) for p in groups["state"]["params"]} == {
        id(p) for p in world.model.backbone.state.parameters()
    }


def test_selection_uses_val_geneeffect_loss(world, tmp_path, monkeypatch):
    from accelerate import Accelerator

    config = copy.deepcopy(world.config)
    config["train"].update(max_epochs=5, patience=2)
    losses = iter([2.0, 1.0, 1.5, 3.0, 0.1])
    correlations = iter([0.9, -0.9, 0.0, 0.95, 0.0])

    def evaluate(model, inputs, config, *, split, accelerator, lines=None):
        if split == "train":
            return SimpleNamespace(metrics={"train_eval_geneeffect_loss": 0.0})
        return SimpleNamespace(
            metrics={
                "val_geneeffect_loss": next(losses),
                "val_residual_pearson_macro_per_gene": next(correlations),
            }
        )

    monkeypatch.setattr(trainer, "evaluate_model", evaluate)
    state = trainer.fit(
        world.model, world.inputs, config, tmp_path / "run", Accelerator(cpu=True)
    )
    # Epoch 1 has the lowest loss; epochs 2 and 3 exhaust patience before 0.1.
    assert (state.best_epoch, state.next_epoch, state.bad_epochs) == (1, 4, 2)
    assert (
        load_checkpoint(tmp_path / "run" / "best.pt")["train_state"]["best_epoch"] == 1
    )
    assert (
        load_checkpoint(tmp_path / "run" / "last.pt")["train_state"]["next_epoch"] == 4
    )


def test_training_diagnostic_uses_fixed_line_subset(world, tmp_path, monkeypatch):
    from accelerate import Accelerator

    many = SimpleNamespace(
        split=SimpleNamespace(supervised_train=tuple(f"ACH-{i:03d}" for i in range(60)))
    )
    chosen = trainer.training_diagnostic_lines(many)
    assert len(chosen) == trainer.TRAINING_DIAGNOSTIC_LINES == 27
    assert list(chosen) == sorted(chosen) and set(chosen) < set(
        many.split.supervised_train
    )
    reversed_order = SimpleNamespace(
        split=SimpleNamespace(supervised_train=many.split.supervised_train[::-1])
    )
    assert trainer.training_diagnostic_lines(reversed_order) == chosen

    # Three of the six synthetic training lines, every epoch the same three.
    monkeypatch.setattr(trainer, "TRAINING_DIAGNOSTIC_LINES", 3)
    expected = trainer.training_diagnostic_lines(world.inputs)
    assert len(expected) == 3
    scored = []

    def evaluate(model, inputs, config, *, split, accelerator, lines=None):
        result = evaluate_model(
            model, inputs, config, split=split, accelerator=accelerator, lines=lines
        )
        if split == "train":
            scored.append(set(result.predictions.model_id))
        else:
            assert lines is None and set(result.predictions.model_id) == set(VAL)
        return result

    monkeypatch.setattr(trainer, "evaluate_model", evaluate)
    trainer.fit(
        world.model, world.inputs, world.config, tmp_path / "run", Accelerator(cpu=True)
    )
    assert scored == [set(expected)] * world.config["train"]["max_epochs"]
    records = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").open()]
    epochs = [r for r in records if "val_geneeffect_loss" in r]
    assert all(
        r["train_eval_geneeffect_possible_pairs"] == 3 * len(GENES) for r in epochs
    )


def test_cpu_single_process_training_writes_best_last_and_done(
    tmp_path, cpu, monkeypatch
):
    import yaml

    from src import evaluate, train

    config = make_config(tmp_path)
    inputs = make_inputs()
    calls = []

    def load_inputs(config, *, preprocessing=None, include_test=False):
        calls.append(preprocessing is None)
        return inputs

    monkeypatch.setattr("src.data.prepared.load_inputs", load_inputs)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"
    assert train.main(["--config", str(path), "--run-dir", str(run_dir)]) == 0
    for name in ("best.pt", "last.pt", "done.json", "config.yaml", "metrics.jsonl"):
        assert (run_dir / name).is_file()
    done = json.loads((run_dir / "done.json").read_text())
    assert done["next_epoch"] == 2
    run_record = json.loads((run_dir / "run.json").read_text())
    assert run_record["training_diagnostic_lines"] == sorted(TRAIN)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    epochs = [r for r in records if "val_geneeffect_loss" in r]
    updates = [r for r in records if "train_geneeffect_loss" in r]
    assert len(epochs) == 2 and not any("val_response" in k for k in epochs[0])
    assert [r["train_response_loss"] is not None for r in updates[:5]] == [
        True,
        False,
        False,
        False,
        True,
    ]
    # A finished run directory is not retrained.
    assert geneeffect.run_training(config, run_dir) == run_dir / "best.pt"
    assert calls == [True]

    assert (
        evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--split", "val"]) == 0
    )
    out = run_dir / "evaluation" / "best" / "val"
    for name in ("predictions.parquet", "metrics.json", "per_line.csv", "per_gene.csv"):
        assert (out / name).is_file()
    metrics = json.loads((out / "metrics.json").read_text())
    saved = load_checkpoint(run_dir / "best.pt")
    assert metrics["val_geneeffect_loss"] == pytest.approx(
        saved["train_state"]["best_loss"], rel=1e-5
    )


def test_resume_continues_from_last(tmp_path, cpu, monkeypatch):
    config = make_config(tmp_path, dropout=0.1)
    reference = geneeffect.run_training(
        config, tmp_path / "whole", inputs=make_inputs()
    )
    expected = load_checkpoint(reference.parent / "last.pt")

    original = trainer.save_checkpoint

    def interrupt_after_first_last(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path.name == "last.pt":
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "save_checkpoint", interrupt_after_first_last)
    run_dir = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        geneeffect.run_training(config, run_dir, inputs=make_inputs())
    assert load_checkpoint(run_dir / "last.pt")["train_state"]["next_epoch"] == 1
    assert not (run_dir / "done.json").exists()
    monkeypatch.setattr(trainer, "save_checkpoint", original)

    changed = copy.deepcopy(config)
    changed["train"]["head_learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="config differs"):
        geneeffect.run_training(changed, run_dir, inputs=make_inputs())

    geneeffect.run_training(config, run_dir, inputs=make_inputs())
    resumed = load_checkpoint(run_dir / "last.pt")
    assert resumed["train_state"] == expected["train_state"]
    for name, value in expected["model_state"].items():
        torch.testing.assert_close(resumed["model_state"][name], value, rtol=0, atol=0)


def test_residual_targets_use_fold_fit_mean_predictions_use_fixed_mean(world):
    inputs = world.inputs
    labels = inputs.labels.copy()
    # A leave-one-out style fold mean differs from the fixed training mean.
    fold_mean = labels.gene_symbol.map(inputs.train_gene_means) + 0.25
    labels["residual"] = labels.gene_effect - fold_mean
    object.__setattr__(inputs, "labels", labels)

    batch: DependencyBatch = DependencyDataset(inputs, "val").collate(range(3))
    rows = DependencyDataset(inputs, "val").rows.iloc[:3]
    torch.testing.assert_close(
        batch.residual, torch.tensor(rows.residual.to_numpy(), dtype=torch.float32)
    )
    fit_startup_standardizer(world.model, inputs, batch_size=8)
    result = evaluate_model(world.model, inputs, world.config, split="val")
    frame = result.predictions.merge(
        labels, on=["model_id", "gene_symbol"], suffixes=("", "_label")
    )
    np.testing.assert_allclose(frame.residual, frame.residual_label, rtol=1e-6)
    np.testing.assert_allclose(frame.gene_effect, frame.gene_effect_label, rtol=1e-6)
    fixed = frame.gene_symbol.map(inputs.train_gene_means)
    np.testing.assert_allclose(
        frame.geneeffect_prediction, frame.residual_prediction + fixed, rtol=1e-5
    )
    assert set(frame.model_id) == set(VAL)


def test_constant_prediction_residual_correlation_is_nan():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        [(m, g) for m in ("L0", "L1", "L2", "L3") for g in GENES],
        columns=["model_id", "gene_symbol"],
    )
    frame["gene_effect"] = rng.normal(size=len(frame))
    frame["residual"] = rng.normal(size=len(frame))
    frame["residual_prediction"] = frame.gene_symbol.map(
        {g: float(i) for i, g in enumerate(GENES)}
    )  # constant within every gene, as a gene-mean predictor is
    frame["geneeffect_prediction"] = frame.residual_prediction + 1.0
    metrics, _, per_gene = aggregate_geneeffect(
        frame, model_ids=("L0", "L1", "L2", "L3"), genes=GENES, variable_genes=GENES
    )
    assert per_gene.pearson.isna().all() and per_gene.spearman.isna().all()
    assert metrics["residual_pearson_macro_per_gene"] is None
    assert metrics["residual_spearman_macro_per_gene"] is None
    assert metrics["residual_pearson_per_gene_undefined"] == len(GENES)
    assert metrics["residual_pearson_per_gene_scored"] == 0


def test_config_rejects_unknown_key():
    config = load_config(CONFIG)
    assert "selection" not in config
    unknown = copy.deepcopy(config)
    unknown["train"]["selection_metric"] = "val_geneeffect_loss"
    with pytest.raises(ValueError, match="unknown=\\['selection_metric'\\]"):
        validate_config(unknown)
    missing = copy.deepcopy(config)
    del missing["model"]["head_blocks"]["use_z_c"]
    with pytest.raises(ValueError, match="missing=\\['use_z_c'\\]"):
        validate_config(missing)
    top = copy.deepcopy(config)
    top["selection"] = {"metric": "val_geneeffect_loss"}
    with pytest.raises(ValueError, match="unknown"):
        validate_config(top)


def test_response_loss_is_mean_shift_mse_plus_energy_distance():
    from src.model.response import energy_distance, mean_delta_mse, response_loss

    torch.manual_seed(0)
    control, observed = torch.randn(6, 5), torch.randn(7, 5) + 1.0
    predicted = torch.randn(6, 5)
    torch.testing.assert_close(
        response_loss(predicted, observed, control),
        mean_delta_mse(predicted, observed, control.mean(0))
        + energy_distance(predicted, observed),
    )
    # A collapsed bag at the right mean has zero mean-shift error, positive energy.
    collapsed = observed.mean(0, keepdim=True).expand(7, -1)
    assert mean_delta_mse(collapsed, observed, control.mean(0)) < 1e-10
    assert energy_distance(collapsed, observed) > 0.1


def _two_rank_worker(rank, port, config, run_dir):
    import os

    os.environ.update(
        ACCELERATE_USE_CPU="true",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
    )
    torch.set_num_threads(1)
    geneeffect.run_training(config, Path(run_dir), inputs=make_inputs())


def test_two_rank_cpu_training_under_distributed_launch(tmp_path, cpu):
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = make_config(tmp_path)
    run_dir = tmp_path / "run"
    torch.multiprocessing.spawn(
        _two_rank_worker, args=(port, config, str(run_dir)), nprocs=2, join=True
    )
    saved = load_checkpoint(run_dir / "last.pt")
    assert saved["world_size"] == 2 and len(saved["rng_states"]) == 2
    assert json.loads((run_dir / "done.json").read_text())["next_epoch"] == 2
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    # 36 training rows over 2 ranks x batch 4 -> 4 updates per epoch.
    assert max(r["global_step"] for r in records) == 8


def _rank_zero_metrics_worker(rank, port, config, out_dir):
    import os

    from accelerate import Accelerator

    from src.eval import geneeffect as evaluation

    os.environ.update(
        ACCELERATE_USE_CPU="true",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
    )
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True)
    inputs = make_inputs()
    torch.manual_seed(0)
    model = build_joint_model(config, inputs)
    fit_startup_standardizer(model, inputs, batch_size=8)
    calls = []
    aggregate = evaluation.aggregate_geneeffect

    def spy(*args, **kwargs):
        calls.append(rank)
        return aggregate(*args, **kwargs)

    evaluation.aggregate_geneeffect = spy
    result = evaluation.evaluate_model(
        model, inputs, config, split="val", accelerator=accelerator
    )
    Path(out_dir, f"rank{rank}.json").write_text(
        json.dumps(
            {
                "aggregations": len(calls),
                "rows": len(result.predictions),
                "metrics": result.metrics,
            }
        )
    )


def test_metrics_aggregated_once_on_rank_zero(tmp_path, cpu):
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = make_config(tmp_path)  # batch 4: 12 validation rows -> 3 batches
    torch.multiprocessing.spawn(
        _rank_zero_metrics_worker,
        args=(port, config, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    ranks = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
    assert [r["aggregations"] for r in ranks] == [1, 0]
    assert [r["rows"] for r in ranks] == [len(VAL) * len(GENES), 0]
    assert ranks[0]["metrics"] == ranks[1]["metrics"]

    inputs = make_inputs()
    torch.manual_seed(0)
    model = build_joint_model(config, inputs)
    fit_startup_standardizer(model, inputs, batch_size=8)
    single = evaluate_model(model, inputs, config, split="val").metrics
    assert single.keys() == ranks[0]["metrics"].keys()
    for key, value in single.items():
        if isinstance(value, float):
            assert ranks[0]["metrics"][key] == pytest.approx(value, rel=0, abs=1e-6)
        else:
            assert ranks[0]["metrics"][key] == value

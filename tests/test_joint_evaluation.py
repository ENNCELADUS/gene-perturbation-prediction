"""Row weighting, reporting axes and real distributed tail removal."""

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import conftest  # noqa: F401 -- preserve OpenMP and xgboost-before-torch in spawned ranks
import numpy as np
import pandas as pd
import pytest
import torch
from test_joint_data import make_prepared_fixture
from torch import nn

from src.data.prepared import load_inputs
from src.eval.geneeffect import aggregate_geneeffect, compose_metrics, evaluate_model
from src.eval.metrics import _unit_pearson
from src.eval.response import aggregate_response
from src.training.checkpoint import TrainState, record_validation


class IdentityResponse(nn.Module):
    def forward(self, controls, genes, metadata):
        assert not self.training and not torch.is_grad_enabled()
        return tuple(control[:, :2] for control in controls)


class TinyEvaluatorModel(nn.Module):
    def __init__(self, predictions=(0.0, 0.0, 2.0)):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(predictions))
        self.backbone = IdentityResponse()
        self.collator_seed = 0

    def forward(self, batch):
        assert not self.training and not torch.is_grad_enabled()
        return SimpleNamespace(
            delta_hat=torch.stack([self.values[int(gene[1:])] for gene in batch.genes])
        )


def evaluation_inputs(config):
    inputs = load_inputs(config, include_test=True)
    labels = inputs.labels.copy()
    labels.loc[:, "residual"] = 0.0
    return replace(inputs, labels=labels, train_gene_means=inputs.train_gene_means * 0)


def test_unequal_batch_huber_is_pair_weighted_and_modes_restore(tmp_path):
    config = make_prepared_fixture(tmp_path)
    config["train"]["response_weight"] = 0.5
    inputs = evaluation_inputs(config)
    model = TinyEvaluatorModel()
    model.train()
    model.backbone.eval()
    result = evaluate_model(model, inputs, config, split="val")
    assert result.metrics["val_geneeffect_loss"] == 0.5
    assert result.metrics["val_geneeffect_valid_pairs"] == 3
    assert result.metrics["val_geneeffect_rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert result.metrics["val_geneeffect_mae"] == pytest.approx(2 / 3)
    assert result.metrics["val_geneeffect_coverage"] == 1
    assert set(result.predictions.columns) == {
        "model_id",
        "gene_symbol",
        "gene_effect",
        "residual",
        "geneeffect_prediction",
        "residual_prediction",
    }
    assert len(result.response) == len(inputs.response_holdout)
    assert model.training and not model.backbone.training
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(result.metrics, allow_nan=False)


def test_constant_prediction_all_undefined_still_selects_and_test_prefix(tmp_path):
    config = make_prepared_fixture(tmp_path)
    config["train"]["response_weight"] = 0.5
    result = evaluate_model(
        TinyEvaluatorModel((0.0, 0.0, 0.0)),
        evaluation_inputs(config),
        config,
        split="val",
    )
    assert result.metrics["val_geneeffect_pearson_macro_per_line"] is None
    assert result.metrics["val_geneeffect_spearman_per_line_undefined"] == 1
    assert record_validation(TrainState(), result.metrics, 0)
    test = evaluate_model(
        TinyEvaluatorModel(), evaluation_inputs(config), config, split="test"
    )
    assert all(key.startswith("test_") for key in test.metrics)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scored_prediction_is_error_and_restores_modes(tmp_path, bad):
    config = make_prepared_fixture(tmp_path)
    config["train"]["response_weight"] = 1.0
    model = TinyEvaluatorModel((0.0, bad, 2.0))
    model.train()
    with pytest.raises(ValueError, match="non-finite GeneEffect prediction"):
        evaluate_model(model, evaluation_inputs(config), config, split="val")
    assert model.training and model.backbone.training


def test_response_anchor_means_are_equal_weight_despite_unequal_counts():
    frame = pd.DataFrame(
        [
            {
                "model_id": f"A{i}",
                "gene_symbol": f"G{j}",
                "mean_delta_mse": mean,
                "energy_distance": 2 * mean,
            }
            for i, mean in enumerate([1, 3, 5, 7])
            for j in range(i + 1)
        ]
    )
    result = aggregate_response(frame, ["A0", "A1", "A2", "A3"])
    assert result["response_mean_delta_mse"] == 4
    assert result["response_energy_distance"] == 8
    assert result["response_loss"] == 12
    assert result["response_conditions"] == 10
    with pytest.raises(ValueError, match="four anchors"):
        aggregate_response(frame[frame.model_id != "A0"], ["A0", "A1", "A2", "A3"])


def test_total_objective_has_explicit_weight_without_replay_interval():
    result = compose_metrics(
        {"geneeffect_loss": 0.2},
        {"response_loss": 0.8},
        response_weight=0.5,
        prefix="val",
    )
    assert result["val_total_loss"] == pytest.approx(0.6)


def test_axes_use_absolute_per_line_and_only_variable_residual_per_gene():
    frame = pd.DataFrame(
        [
            {
                "model_id": line,
                "gene_symbol": gene,
                "residual": float(i),
                "residual_prediction": float(i if j == 0 else -i),
                "gene_effect": float(100 * j + i),
                "geneeffect_prediction": float(100 * j + (i if j == 0 else -i)),
            }
            for i, line in enumerate(["A", "B", "C"])
            for j, gene in enumerate(["G1", "G2", "G3"])
        ]
    )
    metrics, lines, genes = aggregate_geneeffect(
        frame,
        model_ids=["A", "B", "C", "D"],
        genes=["G1", "G2", "G3"],
        variable_genes=["G1", "ABSENT"],
    )
    assert metrics["residual_pearson_macro_per_gene"] == pytest.approx(1)
    assert metrics["residual_spearman_per_gene_undefined"] == 1
    assert metrics["geneeffect_spearman_macro_per_line"] == 1
    assert metrics["geneeffect_pearson_per_line_scored"] == 3
    assert metrics["geneeffect_pearson_per_line_undefined"] == 1
    assert metrics["geneeffect_coverage"] == 0.75
    assert metrics["geneeffect_missing_pairs"] == 3
    assert genes.gene_symbol.tolist() == ["G1", "ABSENT"]
    assert lines.model_id.tolist() == ["A", "B", "C", "D"]
    with pytest.raises(ValueError, match="no valid pairs"):
        aggregate_geneeffect(
            frame.iloc[:0], model_ids=["A"], genes=["G1"], variable_genes=[]
        )


def test_pearson_reuses_finite_minimum_and_constant_policy():
    assert _unit_pearson(
        np.array([0, 1, 2, np.nan]), np.array([0, 2, 4, 8])
    ) == pytest.approx(1)
    assert np.isnan(_unit_pearson(np.array([0, 1, np.inf]), np.array([0, 1, 2])))
    assert np.isnan(_unit_pearson(np.arange(3), np.ones(3)))


def _distributed_evaluation_worker(rank, port, config, output, fail_rank):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
        }
    )
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 2
    try:
        model = accelerator.prepare(TinyEvaluatorModel())
        if rank == fail_rank:
            with torch.no_grad():
                accelerator.unwrap_model(model).values[2] = float("nan")
        inputs = evaluation_inputs(config)
        # Correlations require all three rows; neither rank alone has enough.
        labels = inputs.labels.copy()
        labels.loc[:, "residual"] = labels.gene_symbol.map(
            {"G0": 0.0, "G1": 1.0, "G2": 2.0}
        )
        inputs = replace(inputs, labels=labels)
        try:
            result = evaluate_model(
                model, inputs, config, split="val", accelerator=accelerator
            )
        except RuntimeError as exc:
            if fail_rank is None:
                raise
            Path(output, f"rank-{rank}.json").write_text(
                json.dumps({"error": str(exc)})
            )
            return
        payload = {
            "metrics": result.metrics,
            "predictions": result.predictions.to_dict("records"),
            "response": result.response.to_dict("records"),
        }
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(payload, allow_nan=False)
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("fail_rank", [None, 1])
def test_two_rank_accelerate_removes_dependency_and_response_tail_padding(
    tmp_path, fail_rank
):
    config = make_prepared_fixture(tmp_path)
    config["train"].update(
        response_weight=0.5, dependency_batch_size=2, response_batch_size=3
    )
    inputs = evaluation_inputs(config)
    labels = inputs.labels.copy()
    labels.loc[:, "residual"] = labels.gene_symbol.map(
        {"G0": 0.0, "G1": 1.0, "G2": 2.0}
    )
    inputs = replace(inputs, labels=labels)
    expected = evaluate_model(TinyEvaluatorModel(), inputs, config, split="val")
    assert expected.metrics["val_geneeffect_pearson_macro_per_line"] is not None
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(
        _distributed_evaluation_worker,
        args=(port, config, str(tmp_path), fail_rank),
        nprocs=2,
        join=False,
    )
    try:
        # Bounded joins ensure a regression cannot hang the whole suite indefinitely.
        import time

        deadline = time.monotonic() + 60
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank evaluator did not finish within 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        actual = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        if fail_rank is not None:
            assert "evaluation failed on a rank" in actual["error"]
            assert "non-finite GeneEffect prediction" in actual["error"]
            continue
        assert actual["metrics"] == expected.metrics
        assert len(actual["predictions"]) == 3
        assert {
            (row["model_id"], row["gene_symbol"]) for row in actual["predictions"]
        } == {("ACH-VAL", gene) for gene in inputs.genes}
        assert len(actual["response"]) == len(inputs.response_holdout)
        assert {
            (row["model_id"], row["gene_symbol"]) for row in actual["response"]
        } == inputs.response_holdout

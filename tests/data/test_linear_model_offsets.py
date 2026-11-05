"""The linear model offset: export checks, identity, and the stack in loss and
evaluation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datasets import DependencyDataset
from src.data.prepared import load_inputs
from src.data.linear_model_offsets import (
    LinearModelOffsets,
    check_linear_model_identity,
    checked_linear_model,
    read_linear_model_offsets,
)
from tests.fixtures.joint import (
    GENES,
    TRAIN,
    VAL,
    make_config,
    make_inputs,
    with_linear_model,
)
from tests.fixtures.prepared import make_prepared_fixture


def write_export(path, lines, genes, scale, *, run_id="linear_model_run", value=0.5):
    path.mkdir(parents=True)
    np.savez(
        path / "linear_model.npz",
        values=np.full((len(lines), len(genes)), value, dtype=np.float32),
        lines=np.asarray(lines, dtype=str),
        genes=np.asarray(genes, dtype=str),
        residual_scale=np.asarray(scale, dtype=np.float64),
    )
    (path / "linear_model.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "reference": {"experiment": "affine", "block_set": "selected"},
                "units": "residual SD",
                "lines": {"train": list(lines), "val": [], "test": []},
                "scores": {},
            }
        )
    )
    return path


def test_read_and_check_an_export(tmp_path):
    inputs = make_inputs()
    lines = [*TRAIN, *VAL]
    path = write_export(tmp_path / "export", lines, GENES, inputs.residual_scale)
    linear_model = read_linear_model_offsets(path)
    assert linear_model.identity == {
        "run_id": "linear_model_run",
        "reference": {"experiment": "affine", "block_set": "selected"},
    }
    checked_linear_model(
        linear_model, genes=GENES, residual_scale=inputs.residual_scale, lines=lines
    )
    with pytest.raises(ValueError, match="lacks lines"):
        checked_linear_model(
            linear_model,
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=["ACH-X0"],
        )


def test_load_inputs_refuses_a_linear_model_with_another_gene_order_or_scale(tmp_path):
    inputs = make_inputs()
    lines = [*TRAIN, *VAL]
    reordered = write_export(
        tmp_path / "a", lines, GENES[::-1], inputs.residual_scale.loc[list(GENES[::-1])]
    )
    with pytest.raises(ValueError, match="gene order"):
        checked_linear_model(
            read_linear_model_offsets(reordered),
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=lines,
        )
    rescaled = write_export(tmp_path / "b", lines, GENES, inputs.residual_scale * 1.01)
    with pytest.raises(ValueError, match="residual scale"):
        checked_linear_model(
            read_linear_model_offsets(rescaled),
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=lines,
        )


def test_restore_refuses_a_checkpoint_trained_on_another_linear_model():
    recorded = {"run_id": "old", "reference": {}}
    current = LinearModelOffsets(
        {"run_id": "new", "reference": {}}, pd.DataFrame(), pd.Series()
    )
    with pytest.raises(ValueError, match="old.*new"):
        check_linear_model_identity(recorded, current)
    with pytest.raises(ValueError, match="without a linear model"):
        check_linear_model_identity(None, current)
    with pytest.raises(ValueError, match="no linear_model is configured"):
        check_linear_model_identity(recorded, None)
    check_linear_model_identity(None, None)
    check_linear_model_identity(dict(current.identity), current)


def test_load_inputs_attaches_the_export_and_records_its_identity(tmp_path):
    config = make_prepared_fixture(tmp_path)
    fitted = load_inputs(config)
    assert (
        fitted.linear_model is None
        and fitted.preprocessing_state()["linear_model"] is None
    )
    lines = [*fitted.split.supervised_train, *fitted.split.val]
    config["paths"]["linear_model"] = str(
        write_export(tmp_path / "export", lines, fitted.genes, fitted.residual_scale)
    )
    attached = load_inputs(config)
    assert attached.linear_model.identity["run_id"] == "linear_model_run"
    state = attached.preprocessing_state()
    assert state["linear_model"] == attached.linear_model.identity
    load_inputs(config, preprocessing=state)  # same linear model: accepted
    with pytest.raises(ValueError, match="lacks lines"):
        load_inputs(config, include_test=True)  # the export has no test lines


def test_dataset_rows_carry_the_offset_in_residual_units():
    inputs = with_linear_model(make_inputs(), {m: 0.25 for m in (*TRAIN, *VAL)})
    dataset = DependencyDataset(inputs, "val")
    expected = 0.25 * inputs.residual_scale.loc[dataset.genes].to_numpy()
    np.testing.assert_allclose(dataset.linear_model.numpy(), expected, rtol=1e-6)
    batch = dataset.collate(range(4))
    np.testing.assert_allclose(batch.linear_model.numpy(), expected[:4], rtol=1e-6)
    plain = DependencyDataset(make_inputs(), "val")
    assert torch.equal(plain.linear_model, torch.zeros(len(plain)))


@pytest.mark.usefixtures("cpu")
def test_training_loss_and_evaluation_score_the_stack(tmp_path):
    from accelerate import Accelerator

    from src.eval.accuracy import evaluate_model
    from src.model.initialization import build_joint_model
    from src.model.losses import geneeffect_loss
    from src.model.normalization import fit_startup_standardizer
    from src.training.trainer import make_optimizer, make_scheduler, train_update

    config = make_config(tmp_path, state=False)
    config["model"]["dropout"] = 0.0
    plain = make_inputs()
    offsets = {m: 0.1 * i for i, m in enumerate((*TRAIN, *VAL))}
    inputs = with_linear_model(plain, offsets)
    torch.manual_seed(0)
    model = build_joint_model(config, inputs)
    fit_startup_standardizer(model, inputs, batch_size=8)

    alone = evaluate_model(model, plain, config, split="val").predictions
    stacked = evaluate_model(model, inputs, config, split="val").predictions
    shift = stacked.model_id.map(offsets) * stacked.gene_symbol.map(
        plain.residual_scale
    )
    np.testing.assert_allclose(
        stacked.residual_prediction, alone.residual_prediction + shift, atol=1e-5
    )

    batch = DependencyDataset(inputs, "train").collate(range(8))
    with torch.no_grad():
        head = model(batch.conditions)
    expected = geneeffect_loss(
        head + batch.linear_model,
        batch.residual,
        batch.residual_scale,
        objective=config["train"]["objective"],
        gene_index=batch.conditions.gene_index,
        selective=batch.selective,
        gene_mean=batch.gene_mean,
    )
    optimizer = make_optimizer(model, config)
    metrics = train_update(
        model,
        optimizer,
        make_scheduler(optimizer, config, 1),
        batch,
        config,
        Accelerator(cpu=True),
    )
    assert metrics["train_geneeffect_loss"] == pytest.approx(expected.item(), rel=1e-5)

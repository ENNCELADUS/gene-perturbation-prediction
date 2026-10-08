"""The prior offset: export checks, identity, and the stack in loss and evaluation."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datasets import DependencyDataset
from src.data.prepared import load_inputs
from src.data.prior_offsets import (
    PriorOffsets,
    check_prior_identity,
    checked_prior,
    read_prior_offsets,
)
from tests.test_joint import (  # noqa: F401
    GENES,
    TRAIN,
    VAL,
    cpu,
    make_config,
    make_inputs,
)
from tests.test_joint_data import make_prepared_fixture


def write_export(path, lines, genes, scale, *, run_id="prior_run", value=0.5):
    path.mkdir(parents=True)
    np.savez(
        path / "prior.npz",
        values=np.full((len(lines), len(genes)), value, dtype=np.float32),
        lines=np.asarray(lines, dtype=str),
        genes=np.asarray(genes, dtype=str),
        residual_scale=np.asarray(scale, dtype=np.float64),
    )
    (path / "prior.json").write_text(
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
    prior = read_prior_offsets(path)
    assert prior.identity == {
        "run_id": "prior_run",
        "reference": {"experiment": "affine", "block_set": "selected"},
    }
    checked_prior(prior, genes=GENES, residual_scale=inputs.residual_scale, lines=lines)
    with pytest.raises(ValueError, match="lacks lines"):
        checked_prior(
            prior, genes=GENES, residual_scale=inputs.residual_scale, lines=["ACH-X0"]
        )


def test_load_inputs_refuses_a_prior_with_another_gene_order_or_scale(tmp_path):
    inputs = make_inputs()
    lines = [*TRAIN, *VAL]
    reordered = write_export(
        tmp_path / "a", lines, GENES[::-1], inputs.residual_scale.loc[list(GENES[::-1])]
    )
    with pytest.raises(ValueError, match="gene order"):
        checked_prior(
            read_prior_offsets(reordered),
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=lines,
        )
    rescaled = write_export(tmp_path / "b", lines, GENES, inputs.residual_scale * 1.01)
    with pytest.raises(ValueError, match="residual scale"):
        checked_prior(
            read_prior_offsets(rescaled),
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=lines,
        )


def test_restore_refuses_a_checkpoint_trained_on_another_prior():
    recorded = {"run_id": "old", "reference": {}}
    current = PriorOffsets(
        {"run_id": "new", "reference": {}}, pd.DataFrame(), pd.Series()
    )
    with pytest.raises(ValueError, match="old.*new"):
        check_prior_identity(recorded, current)
    with pytest.raises(ValueError, match="without a prior"):
        check_prior_identity(None, current)
    with pytest.raises(ValueError, match="no prior is configured"):
        check_prior_identity(recorded, None)
    check_prior_identity(None, None)
    check_prior_identity(dict(current.identity), current)


def test_load_inputs_attaches_the_export_and_records_its_identity(tmp_path):
    config = make_prepared_fixture(tmp_path)
    fitted = load_inputs(config)
    assert fitted.prior is None and fitted.preprocessing_state()["prior"] is None
    lines = [*fitted.split.supervised_train, *fitted.split.val]
    config["paths"]["prior"] = str(
        write_export(tmp_path / "export", lines, fitted.genes, fitted.residual_scale)
    )
    attached = load_inputs(config)
    assert attached.prior.identity["run_id"] == "prior_run"
    state = attached.preprocessing_state()
    assert state["prior"] == attached.prior.identity
    load_inputs(config, preprocessing=state)  # same prior: accepted
    with pytest.raises(ValueError, match="lacks lines"):
        load_inputs(config, include_test=True)  # the export has no test lines


def with_prior(inputs, per_line: dict[str, float]):
    frame = pd.DataFrame({gene: pd.Series(per_line) for gene in inputs.genes}).loc[
        :, list(inputs.genes)
    ]
    prior = PriorOffsets({"run_id": "r", "reference": {}}, frame, inputs.residual_scale)
    return dataclasses.replace(inputs, prior=prior)


def test_dataset_rows_carry_the_offset_in_residual_units():
    inputs = with_prior(make_inputs(), {m: 0.25 for m in (*TRAIN, *VAL)})
    dataset = DependencyDataset(inputs, "val")
    expected = 0.25 * inputs.residual_scale.loc[dataset.genes].to_numpy()
    np.testing.assert_allclose(dataset.prior.numpy(), expected, rtol=1e-6)
    batch = dataset.collate(range(4))
    np.testing.assert_allclose(batch.prior.numpy(), expected[:4], rtol=1e-6)
    plain = DependencyDataset(make_inputs(), "val")
    assert torch.equal(plain.prior, torch.zeros(len(plain)))


@pytest.mark.usefixtures("cpu")
def test_training_loss_and_evaluation_score_the_stack(tmp_path):
    from accelerate import Accelerator

    from src.eval.geneeffect import evaluate_model
    from src.model.initialization import build_joint_model
    from src.model.losses import geneeffect_loss
    from src.model.normalization import fit_startup_standardizer
    from src.training.trainer import make_optimizer, make_scheduler, train_update

    config = make_config(tmp_path, state=False)
    config["model"]["dropout"] = 0.0
    plain = make_inputs()
    offsets = {m: 0.1 * i for i, m in enumerate((*TRAIN, *VAL))}
    inputs = with_prior(plain, offsets)
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
        head = model(batch.conditions).delta_hat
    expected = geneeffect_loss(
        head + batch.prior,
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
        None,
        config,
        Accelerator(cpu=True),
    )
    assert metrics["train_geneeffect_loss"] == pytest.approx(expected.item(), rel=1e-5)

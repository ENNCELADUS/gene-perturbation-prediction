"""The stack equals its prior before the first update, which is a best.pt candidate."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

from src.eval.geneeffect import aggregate_geneeffect, evaluate_model  # noqa: E402
from src.experiments import geneeffect  # noqa: E402
from src.model.initialization import build_joint_model  # noqa: E402
from src.model.normalization import fit_startup_standardizer  # noqa: E402
from tests.test_joint import (  # noqa: E402, F401
    TEST,
    TRAIN,
    VAL,
    cpu,
    make_config,
    make_inputs,
)
from tests.test_prior_offsets import with_prior  # noqa: E402

PRIOR = {m: float(v) for m, v in zip((*TRAIN, *VAL, *TEST), np.linspace(-1, 1, 10))}


@pytest.mark.usefixtures("cpu")
def test_context_linear_map_starts_at_zero(tmp_path):
    model = build_joint_model(make_config(tmp_path, state=False), make_inputs())
    assert torch.count_nonzero(model.head.context_linear.weight) == 0
    assert torch.count_nonzero(model.head.context_linear.bias) == 0


@pytest.mark.usefixtures("cpu")
def test_stack_before_training_scores_exactly_the_prior(tmp_path):
    config = make_config(tmp_path, state=False)
    inputs = with_prior(make_inputs(), PRIOR)
    model = build_joint_model(config, inputs)
    fit_startup_standardizer(model, inputs, batch_size=8)
    result = evaluate_model(model, inputs, config, split="val")
    rows = inputs.labels.loc[inputs.labels.model_id.isin(VAL)].copy()
    scale = rows.gene_symbol.map(inputs.residual_scale)
    rows["residual_prediction"] = rows.model_id.map(PRIOR) * scale
    rows["geneeffect_prediction"] = rows.residual_prediction + rows.gene_symbol.map(
        inputs.train_gene_means
    )
    expected, _, _ = aggregate_geneeffect(
        rows,
        model_ids=VAL,
        genes=inputs.genes,
        variable_genes=list(inputs.genes),
        selective_genes=sorted(inputs.selective_genes),
    )
    assert result.metrics["val_selective_spearman"] == pytest.approx(
        expected["selective_spearman"], abs=1e-6
    )


def _validation_epochs(run_dir):
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    return [r["epoch"] for r in records if "val_selective_spearman" in r]


@pytest.mark.usefixtures("cpu")
def test_pre_update_validation_runs_with_a_prior(tmp_path):
    config = make_config(tmp_path, state=False)
    config["train"]["max_epochs"] = 1
    inputs = with_prior(make_inputs(), PRIOR)
    geneeffect.run_training(config, tmp_path / "train", inputs=inputs)
    assert _validation_epochs(tmp_path / "train") == [-1, 0]
    assert (tmp_path / "train" / "best.pt").is_file()


@pytest.mark.usefixtures("cpu")
def test_no_prior_run_has_no_pre_update_validation(tmp_path):
    config = make_config(tmp_path, state=False)
    config["train"]["max_epochs"] = 1
    geneeffect.run_training(config, tmp_path / "train", inputs=make_inputs())
    assert _validation_epochs(tmp_path / "train") == [0]

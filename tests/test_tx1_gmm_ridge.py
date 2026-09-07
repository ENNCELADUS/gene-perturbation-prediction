"""Train isolation, distribution semantics and saved-baseline evaluation."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml

from src.baselines.tx1_gmm import Tx1GMMRidge
from src.data.splits import FixedSplit
from src.experiments import tx1_gmm_ridge as experiment


def synthetic_inputs():
    rng = np.random.default_rng(4)
    split = FixedSplit(tuple(f"T{i}" for i in range(6)), ("V0", "V1", "V2"), ("TEST",))
    lines, rows = {}, []
    for i, model_id in enumerate(split.all_model_ids):
        lines[model_id] = SimpleNamespace(
            controls_tx1=rng.normal(size=(11 + i, 4)) + (i - 3)
        )
        for gene, value in (("UP", float(i)), ("DOWN", -float(i)), ("MISSING", i / 2)):
            if gene == "MISSING" and i in (1, 4):
                value = np.nan
            rows.append(dict(model_id=model_id, gene_symbol=gene, gene_effect=value))
    labels = pd.DataFrame(rows)
    means = (
        labels.loc[labels.model_id.isin(split.supervised_train)]
        .groupby("gene_symbol")
        .gene_effect.mean()
    )
    return SimpleNamespace(
        split=split,
        lines=lines,
        labels=labels,
        genes=("UP", "DOWN", "MISSING"),
        train_gene_means=means,
        variable_genes=frozenset(means.index),
    )


def test_balanced_training_and_heldout_mutation_cannot_change_fit():
    inputs = synthetic_inputs()
    first = Tx1GMMRidge.fit(inputs, n_components=2)
    selected = first.diagnostics["fit_cell_positions"]
    assert set(selected) == set(inputs.split.supervised_train)
    assert all(len(pos) == len(set(pos)) == 11 for pos in selected.values())
    balanced = np.concatenate(
        [inputs.lines[line].controls_tx1[pos] for line, pos in selected.items()]
    )
    np.testing.assert_allclose(first.cell_scaler.mean_, balanced.mean(0))
    assert first.diagnostics["fit_cell_rows"] == 66
    assert first.diagnostics["gene_train_observations"]["MISSING"] == 4
    poisoned = copy.deepcopy(inputs)
    for model_id in (*inputs.split.val, *inputs.split.test):
        poisoned.lines[model_id].controls_tx1[:] = np.nan
    poisoned.labels.loc[
        ~poisoned.labels.model_id.isin(inputs.split.train), "gene_effect"
    ] = 1e9
    second = Tx1GMMRidge.fit(poisoned, n_components=2)
    for left, right in (
        (first.cell_scaler.mean_, second.cell_scaler.mean_),
        (first.gmm.means_, second.gmm.means_),
        (first.context_scaler.mean_, second.context_scaler.mean_),
        (first.coefficients, second.coefficients),
        (first.intercepts, second.intercepts),
    ):
        np.testing.assert_array_equal(left, right)


def test_occupancy_statistics_and_bag_permutation():
    inputs = synthetic_inputs()
    model = Tx1GMMRidge.fit(inputs, n_components=2)
    bag = inputs.lines["V0"].controls_tx1
    features = model.context_features({"V0": bag})
    p = model.gmm.predict_proba(model.cell_scaler.transform(bag))
    occupancy = p.mean(0)
    entropy = -np.sum(occupancy * np.log(np.clip(occupancy, 1e-300, None)))
    np.testing.assert_allclose(features.iloc[0, :2], occupancy)
    assert features.iloc[0, :2].sum() == pytest.approx(1)
    assert features.loc["V0", "occupancy_entropy"] == pytest.approx(entropy)
    assert features.loc["V0", "effective_components"] == pytest.approx(np.exp(entropy))
    assert features.loc["V0", "assignment_confidence"] == pytest.approx(p.max(1).mean())
    assert features.loc["V0", "nll"] == pytest.approx(
        -model.gmm.score_samples(model.cell_scaler.transform(bag)).mean()
    )
    np.testing.assert_allclose(features, model.context_features({"V0": bag[::-1]}))


def test_gene_slopes_missing_labels_and_serialization(tmp_path):
    inputs = synthetic_inputs()
    model = Tx1GMMRidge.fit(inputs, n_components=2)
    features = model.context_features(
        {line: inputs.lines[line].controls_tx1 for line in inputs.split.train}
    )
    x = model.context_scaler.transform(features.to_numpy())
    predicted = model.predict(features)
    np.testing.assert_allclose(predicted.UP, -predicted.DOWN, atol=1e-12)
    assert np.std(predicted.UP) > 0.5
    y = (
        inputs.labels.query("gene_symbol == 'MISSING'")
        .set_index("model_id")
        .gene_effect.reindex(inputs.split.train)
    )
    valid = y.notna()
    reference = Ridge(alpha=1).fit(
        x[valid], y[valid] - inputs.train_gene_means["MISSING"]
    )
    np.testing.assert_allclose(predicted.MISSING, reference.predict(x))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(
        restored.predict(
            restored.context_features(
                {line: inputs.lines[line].controls_tx1 for line in inputs.split.train}
            )
        ),
        predicted,
        check_exact=True,
    )
    result, _ = experiment.evaluate_model(restored, inputs, split="train")
    assert all(key.startswith("train_eval_") for key in result.metrics)
    assert result.metrics["train_eval_geneeffect_valid_pairs"] == 16
    assert result.metrics["train_eval_geneeffect_missing_pairs"] == 2
    json.dumps(result.metrics, allow_nan=False)


@pytest.mark.parametrize("bad", ["nan", "width", "labels", "overlap", "unlabeled"])
def test_malformed_training_inputs_fail(bad):
    inputs = synthetic_inputs()
    if bad == "nan":
        inputs.lines["T0"].controls_tx1[0, 0] = np.nan
    elif bad == "width":
        inputs.lines["T0"].controls_tx1 = np.ones((11, 3))
    elif bad == "labels":
        inputs.labels.loc[inputs.labels.gene_symbol == "UP", "gene_effect"] = np.nan
    elif bad == "overlap":
        inputs.split = replace(inputs.split, val=("T0",))
    else:
        inputs.split = replace(inputs.split, unlabeled_train=inputs.split.train)
    with pytest.raises(ValueError):
        Tx1GMMRidge.fit(inputs, n_components=2)


def test_production_k64_without_pca():
    model = Tx1GMMRidge.fit(synthetic_inputs())
    assert model.gmm.n_components == 64
    assert model.gmm.means_.shape == (64, 4)
    assert model.gmm.covariance_type == "diag"
    assert model.coefficients.shape == (3, 68)


def test_inference_rejects_wrong_width_and_permuted_feature_axis():
    inputs = synthetic_inputs()
    model = Tx1GMMRidge.fit(inputs, n_components=2)
    with pytest.raises(ValueError, match="equal width"):
        model.context_features({"V0": np.ones((5, 3))})
    features = model.context_features({"V0": inputs.lines["V0"].controls_tx1})
    with pytest.raises(ValueError, match="feature order"):
        model.predict(features[features.columns[::-1]])


def test_failed_export_is_retryable_without_changing_fit_status(tmp_path, monkeypatch):
    from src.experiments import geneeffect

    inputs = synthetic_inputs()
    model = Tx1GMMRidge.fit(inputs, n_components=2)
    (tmp_path / "run.json").write_text(json.dumps({"fitting": {"status": "completed"}}))
    export = geneeffect.export_evaluation

    def fail(*args, **kwargs):
        raise OSError("simulated export failure")

    monkeypatch.setattr(geneeffect, "export_evaluation", fail)
    with pytest.raises(OSError, match="simulated"):
        experiment._export(model, inputs, tmp_path, "val")
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["fitting"]["status"] == "completed"
    assert record["evaluation"]["status"] == "failed"
    monkeypatch.setattr(geneeffect, "export_evaluation", export)
    experiment._export(model, inputs, tmp_path, "val")
    assert (
        json.loads((tmp_path / "run.json").read_text())["evaluation"]["status"]
        == "completed"
    )


def test_prepared_fit_and_evaluation_restore_without_refitting(tmp_path, monkeypatch):
    from test_joint_integration import full_tiny_config

    config = full_tiny_config(tmp_path / "inputs")
    config["features"]["cells_per_context"] = 16  # 5 training bags support K64.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out_dir = tmp_path / "run"
    # No STATE construction or prediction is part of the baseline command.
    Path(config["paths"]["state_checkpoint"]).unlink()
    path = experiment.fit_baseline(config_path, out_dir)
    record = json.loads((out_dir / "run.json").read_text())
    assert record["fitting"]["status"] == "completed"
    assert (out_dir / "evaluation/train/metrics.json").is_file()
    assert not (out_dir / "evaluation/test").exists()
    original = pd.read_parquet(out_dir / "evaluation/val/predictions.parquet")
    before = path.read_bytes()
    Path(config["paths"]["esm2_embeddings"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("saved-model evaluation refitted an estimator")

    monkeypatch.setattr(Tx1GMMRidge, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(Ridge, "fit", forbidden)
    result = experiment.evaluate_checkpoint(path, split="val")
    pd.testing.assert_frame_equal(original, result.predictions, check_exact=True)
    assert before == path.read_bytes()
    with pytest.raises(FileExistsError):
        experiment.fit_baseline(config_path, out_dir)

"""Opening a prepared root: train-only fitting, restore, and test exposure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepared import PreparedLine, load_inputs, write_prepared_line
from tests.fixtures.prepared import ANCHORS, TARGET_SUM, make_prepared_fixture


def test_fixture_opens_in_log_space(tmp_path):
    inputs = load_inputs(make_prepared_fixture(tmp_path, hvg_width=3))
    assert inputs.target_sum == TARGET_SUM
    assert inputs.genes == ("G1", "G0", "G2")
    assert inputs.hvg_order == ("G0", "G1", "HVG2")
    assert inputs.response_anchors == tuple(ANCHORS)
    assert len(inputs.response_targets.keys) == 2 * 4 + 10
    assert inputs.response_targets.target_bag(0).shape == (2, 3)
    line = inputs.lines["ACH-A0"]
    assert line.controls_tx1.shape == (5, 2560)
    assert line.basal_hvg.shape == (5, 3)
    assert np.isnan(line.q_sc.values[0]).all() and not line.q_sc.available[0]
    assert inputs.esm2_symbols == ("G2", "G0", "G1", "R0", "R1", "R2", "R3")


def test_train_only_fit_ignores_val_and_test_labels(tmp_path):
    config = make_prepared_fixture(tmp_path)
    before = load_inputs(config)
    assert before.train_gene_means["G0"] == 2
    assert set(before.labels.model_id) == {*before.split.supervised_train, "ACH-VAL"}
    wide = pd.read_csv(config["paths"]["gene_effect"], index_col=0)
    wide.loc[["ACH-VAL", "ACH-TEST"]] = 9999
    wide.to_csv(config["paths"]["gene_effect"])
    after = load_inputs(config)
    pd.testing.assert_series_equal(before.train_gene_means, after.train_gene_means)
    assert before.variable_genes == after.variable_genes


def test_test_lines_and_labels_need_include_test(tmp_path):
    config = make_prepared_fixture(tmp_path)
    assert "ACH-TEST" not in load_inputs(config).lines
    opened = load_inputs(config, include_test=True)
    assert "ACH-TEST" in opened.lines
    assert "ACH-TEST" in set(opened.labels.model_id)
    assert "ACH-UNLABELED" not in opened.lines


def test_restore_skips_fitting_and_external_esm2(tmp_path, monkeypatch):
    import src.data.prepared as prepared

    config = make_prepared_fixture(tmp_path)
    fitted = load_inputs(config)
    state = fitted.preprocessing_state()
    Path(config["paths"]["esm2_embeddings"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("restore must not refit")

    monkeypatch.setattr(prepared, "fit_gene_means", forbidden)
    monkeypatch.setattr(prepared, "fit_variable_gene_membership", forbidden)
    restored = load_inputs(config, preprocessing=state)
    pd.testing.assert_series_equal(restored.train_gene_means, fitted.train_gene_means)
    np.testing.assert_array_equal(restored.esm2_vectors, fitted.esm2_vectors)
    state["target_sum"] = TARGET_SUM * 2
    with pytest.raises(ValueError, match="target_sum"):
        load_inputs(config, preprocessing=state)


def test_restore_rejects_state_fitted_under_another_split_membership(tmp_path):
    config = make_prepared_fixture(tmp_path)
    state = load_inputs(config).preprocessing_state()
    assert load_inputs(config, preprocessing=state).split.train
    moved = dict(state, split=dict(state["split"]))
    line = moved["split"]["train"][0]
    moved["split"]["train"] = moved["split"]["train"][1:]
    moved["split"]["test"] = sorted([*moved["split"]["test"], line])
    with pytest.raises(ValueError, match="different split membership"):
        load_inputs(config, preprocessing=moved)
    legacy = {key: value for key, value in state.items() if key != "split"}
    with pytest.raises(ValueError, match="predates the 2026-10-11 in-place re-split"):
        load_inputs(config, preprocessing=legacy)


def test_dependency_batch_carries_gene_index_scale_and_selectivity(tmp_path):
    import dataclasses

    import torch

    from src.data.datasets import DependencyDataset

    opened = load_inputs(make_prepared_fixture(tmp_path))
    scale = pd.Series([0.5, 2.0, 4.0], index=list(opened.genes))
    inputs = dataclasses.replace(
        opened, residual_scale=scale, selective_genes=frozenset({"G0"})
    )
    dataset = DependencyDataset(inputs, "train")
    batch = dataset.collate(range(len(dataset)))
    genes = batch.conditions.genes
    gene_index = batch.conditions.gene_index
    assert gene_index.dtype == torch.long
    assert [inputs.genes[i] for i in gene_index.tolist()] == list(genes)
    assert batch.residual_scale.dtype == torch.float32
    assert batch.residual_scale.tolist() == [scale[gene] for gene in genes]
    assert batch.selective.dtype == torch.bool
    assert batch.selective.tolist() == [gene == "G0" for gene in genes]
    moved = batch.to("meta")
    for value in (
        moved.conditions.gene_index,
        moved.residual_scale,
        moved.selective,
        moved.residual,
    ):
        assert value.device.type == "meta"


def test_rows_by_gene_partitions_rows_in_gene_order(tmp_path):
    import dataclasses

    from src.data.datasets import DependencyDataset

    opened = load_inputs(make_prepared_fixture(tmp_path))
    assert opened.genes == ("G1", "G0", "G2")
    inputs = dataclasses.replace(
        opened, labels=opened.labels.loc[opened.labels.gene_symbol != "G0"]
    )
    dataset = DependencyDataset(inputs, "train")
    groups = dataset.rows_by_gene()
    assert len(groups) == len(inputs.genes)
    assert len(groups[1]) == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate(groups)), np.arange(len(dataset))
    )
    for gene, rows in zip(inputs.genes, groups, strict=True):
        assert {dataset.genes[i] for i in rows} <= {gene}
        assert len(rows) == sum(g == gene for g in dataset.genes)


def _rewrite_tx1(config: dict, model_id: str, controls: np.ndarray) -> None:
    from src.data.prepared import read_prepared_line

    path = Path(config["prepared_root"]) / "lines" / f"{model_id}.npz"
    line = read_prepared_line(path, ("G1", "G0", "G2"))
    write_prepared_line(
        path, PreparedLine(controls.astype(np.float32), line.basal_hvg, line.q_sc)
    )


def _same_pca(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, key), getattr(right, key))
        for key in ("mean", "scale", "components", "score_scale")
    )


def test_context_pca_fits_on_labelled_training_lines_only(tmp_path):
    config = make_prepared_fixture(tmp_path)
    before = load_inputs(config).context_pca
    assert before.n_components == 1 and before.width == 2 * 2560
    rng = np.random.default_rng(0)
    for model_id in ("ACH-VAL", "ACH-TEST"):
        _rewrite_tx1(config, model_id, rng.normal(size=(5, 2560)))
    assert _same_pca(before, load_inputs(config).context_pca)
    assert _same_pca(before, load_inputs(config, include_test=True).context_pca)
    _rewrite_tx1(config, "ACH-TRAIN", rng.normal(size=(5, 2560)))
    assert not _same_pca(before, load_inputs(config).context_pca)


def test_context_pca_restores_without_refitting(tmp_path, monkeypatch):
    import src.data.prepared as prepared
    from src.data.datasets import DependencyDataset

    config = make_prepared_fixture(tmp_path)
    fitted = load_inputs(config)
    state = fitted.preprocessing_state()

    def forbidden(*args, **kwargs):
        raise AssertionError("restore must not refit")

    monkeypatch.setattr(prepared, "fit_context_pca", forbidden)
    restored = load_inputs(config, preprocessing=state)
    assert _same_pca(fitted.context_pca, restored.context_pca)
    for inputs in (fitted, restored):
        assert DependencyDataset(inputs, "val").collate([0]).conditions.z_c.shape == (
            1,
            1,
        )
    broken = {key: value for key, value in state.items() if key != "context_pca"}
    with pytest.raises(ValueError, match="context_pca"):
        load_inputs(config, preprocessing=broken)
    partial = dict(state, context_pca=dict(state["context_pca"]))
    del partial["context_pca"]["score_scale"]
    with pytest.raises(ValueError, match="score_scale"):
        load_inputs(config, preprocessing=partial)


def test_context_pca_is_eigen_scaled_and_signed():
    from src.data.context_pca import ContextPCA, fit_context_pca

    rng = np.random.default_rng(1)
    latent = rng.normal(size=(40, 6)) * np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    contexts = latent @ rng.normal(size=(6, 30)) + rng.normal(size=30)
    contexts[:, 7] = 4.0  # a constant dimension is guarded, not divided by zero
    pca = fit_context_pca(contexts, 5)
    scores = pca.transform(contexts)
    sd = scores.std(axis=0)
    assert sd[0] == pytest.approx(1.0)
    assert np.all(np.diff(sd) < 0) and sd[-1] < 0.5
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(5), atol=1e-10)
    peak = np.abs(pca.components).argmax(axis=1)
    assert (pca.components[np.arange(5), peak] > 0).all()
    assert np.isfinite(scores).all() and pca.scale[7] == 1.0
    restored = ContextPCA.from_state(pca.to_state())
    np.testing.assert_array_equal(restored.transform(contexts), scores)
    with pytest.raises(ValueError, match="rank 3"):
        fit_context_pca(contexts[:4], 4)

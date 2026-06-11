from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aivc_model.model import StateForwardAdapter
from aivc_model.state_feature_ablation import (
    EXTERNAL_ENSEMBLE_SCOPE,
    PRIMARY_EXTERNAL_SCOPE,
    FeatureArmData,
    _control_panel,
    adamson_heldout_ensemble_predictions,
    fit_fold_ridge,
    load_state_feature_ablation_config,
    same_path_token_hidden_bags,
    token_hidden_bag,
    validate_result_tables,
)
from aivc_model.prepare import GeneBags


class _FakeTokenState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[float, ...]]] = []

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        padded: bool = False,
    ) -> torch.Tensor:
        del padded
        pert = batch["pert_emb"]
        self.calls.append((batch["pert_name"][0], tuple(pert[0].tolist())))
        self._token_features = batch["ctrl_cell_emb"] + pert[:, :2]
        return batch["ctrl_cell_emb"] - pert[:, :2]


class _NoTokenState(torch.nn.Module):
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        padded: bool = False,
    ) -> torch.Tensor:
        del padded
        return batch["ctrl_cell_emb"]


class _Bad3DTokenState(torch.nn.Module):
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        padded: bool = False,
    ) -> torch.Tensor:
        del padded
        self._token_features = torch.ones((2, 2, 3))
        return batch["ctrl_cell_emb"]


def test_state_adapter_exposes_token_hidden_features() -> None:
    state = _FakeTokenState()
    adapter = StateForwardAdapter(state)
    control = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perturbation = torch.tensor([10.0, 20.0, 30.0])

    output = adapter(control, perturbation, "GENE1")

    assert torch.equal(output, torch.tensor([[-9.0, -18.0], [-7.0, -16.0]]))
    assert adapter.last_token_features is not None
    assert torch.equal(
        adapter.last_token_features,
        torch.tensor([[11.0, 22.0], [13.0, 24.0]]),
    )


def test_token_hidden_uses_same_path_non_targeting_control_embedding() -> None:
    state = _FakeTokenState()
    adapter = StateForwardAdapter(state)
    control = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    non_targeting = torch.tensor([0.0, 0.0, 1.0])
    target = torch.tensor([5.0, 7.0, 1.0])

    pair = same_path_token_hidden_bags(
        adapter,
        control,
        non_targeting,
        target,
        "TARGET1",
    )

    assert np.array_equal(pair.control_bag, control.numpy())
    assert np.array_equal(pair.perturbed_bag, np.asarray([[6.0, 9.0], [8.0, 11.0]]))
    assert state.calls == [
        ("non-targeting", (0.0, 0.0, 1.0)),
        ("TARGET1", (5.0, 7.0, 1.0)),
    ]


def test_token_hidden_requires_state_token_features() -> None:
    adapter = StateForwardAdapter(_NoTokenState())
    control = torch.ones((2, 2))
    perturbation = torch.ones(2)

    with pytest.raises(ValueError, match="_token_features"):
        token_hidden_bag(adapter, control, perturbation, "GENE1")


def test_token_hidden_rejects_unexpected_3d_batch_shape() -> None:
    adapter = StateForwardAdapter(_Bad3DTokenState())
    control = torch.ones((2, 2))
    perturbation = torch.ones(2)

    with pytest.raises(ValueError, match="batch dimension 1"):
        token_hidden_bag(adapter, control, perturbation, "GENE1")


def test_fold_local_gmm_and_ridge_exclude_test_genes() -> None:
    data = _synthetic_arm_data()
    train_idx = np.asarray([0, 1], dtype=np.int64)
    test_idx = np.asarray([2, 3], dtype=np.int64)

    fit = fit_fold_ridge(
        data,
        train_idx,
        test_idx,
        fold=0,
        alpha=30.0,
        n_components=2,
        view="centered",
        random_state=42,
    )

    assert fit.train_genes == ("A", "B")
    assert fit.gmm_row["fit_gene_names"] == "A,B"
    assert set(fit.predictions["perturbation_gene"]) == {"C", "D"}
    assert fit.qa_row["gmm_fit_source"] == "B_hat_train_plus_controls"


def test_adamson_target_heldout_ensemble_excludes_train_target_models() -> None:
    predictions = pd.DataFrame(
        {
            "evaluation_scope": ["external:adamson_k562"] * 4,
            "fold": [0, 1, 0, 1],
            "feature_set": ["state_token_hidden"] * 4,
            "arm": ["state_token_hidden_gmm_ridge"] * 4,
            "model": ["ridge_alpha30"] * 4,
            "weighting": ["unweighted"] * 4,
            "perturbation_gene": ["A", "A", "C", "C"],
            "y_true": [-1.0, -1.0, -0.2, -0.2],
            "y_pred": [-0.8, -0.6, -0.1, -0.3],
        }
    )
    splits = pd.DataFrame(
        {
            "evaluation_scope": ["internal_cv_all"] * 4,
            "fold": [0, 0, 1, 1],
            "split": ["train", "train", "train", "train"],
            "perturbation_gene": ["A", "B", "B", "D"],
        }
    )

    ensemble = adamson_heldout_ensemble_predictions(predictions, splits)

    primary_a = ensemble.loc[
        (ensemble["evaluation_scope"] == EXTERNAL_ENSEMBLE_SCOPE)
        & (ensemble["perturbation_gene"] == "A")
    ].iloc[0]
    heldout_a = ensemble.loc[
        (ensemble["evaluation_scope"] == PRIMARY_EXTERNAL_SCOPE)
        & (ensemble["perturbation_gene"] == "A")
    ].iloc[0]
    heldout_c = ensemble.loc[
        (ensemble["evaluation_scope"] == PRIMARY_EXTERNAL_SCOPE)
        & (ensemble["perturbation_gene"] == "C")
    ].iloc[0]

    assert primary_a["ensemble_size"] == 2
    assert heldout_a["ensemble_size"] == 1
    assert heldout_a["y_pred"] == -0.6
    assert heldout_c["ensemble_size"] == 2
    assert heldout_a["arm"] == "state_token_hidden_gmm_ridge"


def test_result_tables_include_scope_and_qa_fields() -> None:
    data = _synthetic_arm_data()
    fit = fit_fold_ridge(
        data,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
        fold=0,
        alpha=300.0,
        n_components=2,
        view="centered",
        random_state=42,
    )
    fold_metrics = pd.DataFrame([fit.metric_row])
    feature_qa = pd.DataFrame([fit.qa_row])

    validate_result_tables(fold_metrics, fit.predictions, feature_qa)
    assert set(fold_metrics["primary_scope"]) == {PRIMARY_EXTERNAL_SCOPE}
    assert set(fit.predictions["secondary_scope"]) == {EXTERNAL_ENSEMBLE_SCOPE}
    assert feature_qa.iloc[0]["control_embedding_path"] == "same_path_non_targeting"


def test_observed_anchor_qa_uses_observed_fit_source() -> None:
    data = _synthetic_arm_data()
    data = FeatureArmData(
        feature_set=data.feature_set,
        arm="observed_scvi128_gmm_ridge_anchor",
        genes=data.genes,
        y=data.y,
        bags=data.bags,
        control_bag=data.control_bag,
        embedding_space="scvi_latent",
        gmm_fit_source="observed_train_plus_controls",
    )
    fit = fit_fold_ridge(
        data,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
        fold=0,
        alpha=300.0,
        n_components=2,
        view="centered",
        random_state=42,
    )

    assert fit.qa_row["gmm_fit_source"] == "observed_train_plus_controls"


def test_control_panel_applies_configured_cell_cap() -> None:
    data = GeneBags(
        genes=np.asarray(["A"], dtype=object),
        y=np.asarray([-1.0], dtype=np.float32),
        input_bags=(np.ones((2, 2), dtype=np.float32),),
        latent_bags=(np.ones((2, 2), dtype=np.float32),),
        control_input=np.arange(20, dtype=np.float32).reshape(10, 2),
        control_latent=np.arange(20, dtype=np.float32).reshape(10, 2),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=np.asarray(["b0"] * 10, dtype=object),
        feature_names=np.asarray(["g1", "g2"], dtype=object),
        metadata=pd.DataFrame({"perturbation_gene": ["A"]}),
        input_dim=2,
        latent_dim=2,
    )
    config = load_state_feature_ablation_config(
        Path(
            "configs/experiments/05_aivc_a_to_b_to_c/state_frozen_feature_ablation.yaml"
        )
    )

    cells, batch = _control_panel(
        data,
        config,
        fold_seed=0,
        device=torch.device("cpu"),
        batch_lookup={"b0": 3},
    )

    assert cells.shape == (10, 2)
    assert batch is not None
    assert set(batch.tolist()) == {3}


def test_state_feature_ablation_config_records_primary_contract() -> None:
    config = load_state_feature_ablation_config(
        Path(
            "configs/experiments/05_aivc_a_to_b_to_c/state_frozen_feature_ablation.yaml"
        )
    )

    assert config.seed == 42
    assert config.n_splits == 5
    assert config.max_control_cells_per_gene == 512
    assert config.gmm_components == 64
    assert config.ridge_alphas == (30.0, 300.0)
    assert config.primary_scope == PRIMARY_EXTERNAL_SCOPE
    assert config.interpretation == "adamson_guided_validation_sweep"


def _synthetic_arm_data() -> FeatureArmData:
    genes = np.asarray(["A", "B", "C", "D"], dtype=object)
    bags = tuple(
        np.asarray(
            [
                [float(index), 0.0],
                [float(index), 0.2],
                [float(index) + 0.1, 0.1],
            ],
            dtype=np.float32,
        )
        for index in range(4)
    )
    return FeatureArmData(
        feature_set="state_token_hidden",
        arm="state_token_hidden_gmm_ridge",
        genes=genes,
        y=np.asarray([-1.0, -0.8, -0.2, 0.1], dtype=np.float64),
        bags=bags,
        control_bag=np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
        embedding_space="state_token_hidden",
    )

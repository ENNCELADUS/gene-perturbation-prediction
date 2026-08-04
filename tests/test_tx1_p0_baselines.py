"""Tests for the development-only Tx1 P0 source-line baseline ladder."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aivc_model.tx1_p0_baselines as p0_module
from aivc_model.tx1_p0_baselines import (
    GENE_MEAN_METHOD,
    LINEAGE_METHOD,
    NEAREST_METHOD,
    PCA_RIDGE_METHOD,
    P0BaselineConfig,
    run_p0_baseline_ladder,
    write_p0_baseline_artifacts,
)
from aivc_model.tx1_p0_validation import (
    PROTOCOL_ID,
    ValidationPolicy,
    generate_nested_validation,
)
from aivc_model.tx1_geneeffect_eval import EvaluationContractError
from conftest import tx1_manifest_row, write_tx1_line_manifest

_REAL_PHASE_A_DIR = Path(__file__).parents[1] / "results" / "phase_a_tx1_20260724"


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    rows = [
        tx1_manifest_row(
            model_id=f"ACH-H{index:04d}",
            lineage="L1" if index < 15 else "L2",
            dmso_cells=100 * (index + 1),
            role="train_head",
        )
        for index in range(29)
    ]
    rows.extend(
        tx1_manifest_row(
            model_id=f"ACH-A{index:04d}",
            lineage="Anchor",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        )
        for index in range(4)
    )
    rows.extend(
        tx1_manifest_row(model_id=f"ACH-T{index:04d}", lineage="Opened", role="test")
        for index in range(9)
    )
    write_tx1_line_manifest(manifest_path, rows)

    genes = (
        pd.read_csv(_REAL_PHASE_A_DIR / "differentially_essential_slice.csv")[
            "gene_symbol"
        ]
        .astype(str)
        .tolist()
    )
    labels_path = tmp_path / "gene_effect.csv"
    pd.DataFrame(
        [
            {
                "model_id": f"ACH-H{line_index:04d}",
                "gene_symbol": gene,
                "gene_effect": float(
                    ((gene_index * 37 + line_index * 11) % 1009) / 100.0
                    + line_index * 0.01
                ),
            }
            for line_index in range(29)
            for gene_index, gene in enumerate(genes)
        ]
    ).to_csv(labels_path, index=False)
    context_path = tmp_path / "context.csv"
    pd.DataFrame(
        {
            "model_id": [f"ACH-H{index:04d}" for index in range(29)],
            "f1": [float(index) for index in range(29)],
            "f2": [float(index * index + 1) for index in range(29)],
        }
    ).to_csv(context_path, index=False)
    prior_path = tmp_path / "copy.csv"
    pd.DataFrame(
        {
            "gene_symbol": genes,
            "gene_effect": [float(index) / 100.0 for index in range(len(genes))],
        }
    ).to_csv(prior_path, index=False)

    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    policy_payload = {
        "protocol_id": PROTOCOL_ID,
        "version": 1,
        "seed": 17,
        "expected_manifest_sha256": manifest_sha,
        "expected_role_counts": {
            "train_head": 29,
            "train_response_and_head": 4,
            "test": 9,
        },
        "inner_fold_count": 5,
        "dmso_quantile_bins": 4,
    }
    policy = ValidationPolicy.from_mapping(policy_payload)
    policy_path = tmp_path / "validation_policy.json"
    policy_path.write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan_path = tmp_path / "validation_plan.json"
    plan_path.write_text(
        json.dumps(
            generate_nested_validation(manifest_path, policy=policy),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, policy_path, plan_path, labels_path, context_path, prior_path


def _prediction(result: object, model_id: str, method: str) -> np.ndarray:
    frame = result.per_prediction
    return (
        frame.loc[(frame["model_id"] == model_id) & (frame["method"] == method)]
        .sort_values("gene_symbol")["y_pred"]
        .to_numpy(dtype=float)
    )


def _run_kwargs(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    manifest, policy, plan, labels, context, prior = _fixtures(tmp_path)
    return (
        {
            "manifest_path": manifest,
            "phase_a_dir": _REAL_PHASE_A_DIR,
            "validation_policy_path": policy,
            "validation_plan_path": plan,
            "gene_effect_path": labels,
        },
        context,
        prior,
    )


def test_hand_calculated_oof_means_and_lineage_shrinkage(tmp_path: Path) -> None:
    kwargs, _, _ = _run_kwargs(tmp_path)
    result = run_p0_baseline_ladder(
        **kwargs, config=P0BaselineConfig(lineage_alpha=0.5)
    )
    frame = pd.read_csv(kwargs["gene_effect_path"])
    source = (
        frame[frame["model_id"] != "ACH-H0000"]
        .pivot(index="model_id", columns="gene_symbol", values="gene_effect")
        .sort_index(axis=1)
    )
    global_mean = source.to_numpy(dtype=float).mean(axis=0)
    lineage_ids = [f"ACH-H{index:04d}" for index in range(1, 15)]
    lineage_mean = source.loc[lineage_ids].to_numpy(dtype=float).mean(axis=0)

    np.testing.assert_allclose(
        _prediction(result, "ACH-H0000", GENE_MEAN_METHOD), global_mean
    )
    np.testing.assert_allclose(
        _prediction(result, "ACH-H0000", LINEAGE_METHOD),
        0.5 * lineage_mean + 0.5 * global_mean,
    )
    assert len(set(result.per_prediction["model_id"])) == 29


def test_held_line_label_cannot_affect_its_predictions(tmp_path: Path) -> None:
    kwargs, context, _ = _run_kwargs(tmp_path)
    first = run_p0_baseline_ladder(**kwargs, context_path=context)
    labels = kwargs["gene_effect_path"]
    changed = pd.read_csv(labels)
    changed.loc[changed["model_id"] == "ACH-H0000", "gene_effect"] += 10000.0
    changed.to_csv(labels, index=False)
    second = run_p0_baseline_ladder(**kwargs, context_path=context)

    for method in (GENE_MEAN_METHOD, LINEAGE_METHOD, NEAREST_METHOD, PCA_RIDGE_METHOD):
        np.testing.assert_allclose(
            _prediction(first, "ACH-H0000", method),
            _prediction(second, "ACH-H0000", method),
        )


def test_context_transforms_are_outer_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, context, _ = _run_kwargs(tmp_path)
    scaler_fit_inputs: list[np.ndarray] = []
    pca_fit_inputs: list[np.ndarray] = []
    original_scaler_fit = p0_module.StandardScaler.fit
    original_pca_fit = p0_module.PCA.fit

    def record_scaler_fit(
        scaler: object, values: np.ndarray, *args: object, **kwargs: object
    ) -> object:
        scaler_fit_inputs.append(np.asarray(values).copy())
        return original_scaler_fit(scaler, values, *args, **kwargs)

    def record_pca_fit(
        pca: object, values: np.ndarray, *args: object, **kwargs: object
    ) -> object:
        pca_fit_inputs.append(np.asarray(values).copy())
        return original_pca_fit(pca, values, *args, **kwargs)

    monkeypatch.setattr(p0_module.StandardScaler, "fit", record_scaler_fit)
    monkeypatch.setattr(p0_module.PCA, "fit", record_pca_fit)
    first = run_p0_baseline_ladder(
        **kwargs, context_path=context, config=P0BaselineConfig(pca_components=2)
    )

    expected = np.array(
        [[float(index), float(index * index + 1)] for index in range(1, 29)]
    )
    np.testing.assert_allclose(scaler_fit_inputs[0], expected)
    assert pca_fit_inputs[0].shape == (28, 2)
    np.testing.assert_allclose(pca_fit_inputs[0].mean(axis=0), 0.0, atol=1e-12)

    monkeypatch.setattr(p0_module.StandardScaler, "fit", original_scaler_fit)
    monkeypatch.setattr(p0_module.PCA, "fit", original_pca_fit)
    changed = pd.read_csv(context)
    changed.loc[changed["model_id"] == "ACH-H0000", ["f1", "f2"]] = [1e9, -1e9]
    changed.to_csv(context, index=False)
    second = run_p0_baseline_ladder(
        **kwargs, context_path=context, config=P0BaselineConfig(pca_components=2)
    )
    assert not np.allclose(
        _prediction(first, "ACH-H0001", PCA_RIDGE_METHOD),
        _prediction(second, "ACH-H0001", PCA_RIDGE_METHOD),
    )
    assert first.summary["config"]["pca_components_effective_by_fold"] == [2]


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "nonfinite", "uniform_truncation"]
)
def test_gene_effect_matrix_rejects_corruption(tmp_path: Path, corruption: str) -> None:
    kwargs, _, _ = _run_kwargs(tmp_path)
    labels = kwargs["gene_effect_path"]
    frame = pd.read_csv(labels)
    if corruption == "missing":
        frame = frame.iloc[1:]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif corruption == "uniform_truncation":
        frame = frame[frame["gene_symbol"] != frame["gene_symbol"].iloc[0]]
    else:
        frame.loc[0, "gene_effect"] = np.nan
    frame.to_csv(labels, index=False)
    with pytest.raises(ValueError):
        run_p0_baseline_ladder(**kwargs)


def test_role_test_labels_and_context_are_rejected(tmp_path: Path) -> None:
    kwargs, context, _ = _run_kwargs(tmp_path)
    labels = kwargs["gene_effect_path"]
    label_frame = pd.read_csv(labels)
    test_rows = label_frame[label_frame["model_id"] == "ACH-H0000"].copy()
    test_rows["model_id"] = "ACH-T0000"
    pd.concat([label_frame, test_rows], ignore_index=True).to_csv(labels, index=False)
    with pytest.raises(ValueError, match="role=test"):
        run_p0_baseline_ladder(**kwargs)

    kwargs, context, _ = _run_kwargs(tmp_path)
    context_frame = pd.read_csv(context)
    context_frame.loc[len(context_frame)] = ["ACH-T0000", 2.0, 3.0]
    context_frame.to_csv(context, index=False)
    with pytest.raises(ValueError, match="role=test"):
        run_p0_baseline_ladder(**kwargs, context_path=context)


def test_stale_or_tampered_validation_plan_is_rejected(tmp_path: Path) -> None:
    kwargs, _, _ = _run_kwargs(tmp_path)
    plan_path = kwargs["validation_plan_path"]
    plan = json.loads(plan_path.read_text())
    plan["outer_folds"][0]["outer_train_model_ids"][0] = "ACH-T0000"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage|role"):
        run_p0_baseline_ladder(**kwargs)


def test_external_policy_rejects_coherently_altered_manifest_and_plan(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _run_kwargs(tmp_path)
    manifest_path = kwargs["manifest_path"]
    altered_manifest = pd.read_csv(manifest_path)
    altered_manifest.loc[0, "dmso_cells"] += 1
    altered_manifest.to_csv(manifest_path, index=False)
    altered_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    altered_policy = ValidationPolicy.from_mapping(
        {
            "protocol_id": PROTOCOL_ID,
            "version": 1,
            "seed": 17,
            "expected_manifest_sha256": altered_sha,
            "expected_role_counts": {
                "train_head": 29,
                "train_response_and_head": 4,
                "test": 9,
            },
            "inner_fold_count": 5,
            "dmso_quantile_bins": 4,
        }
    )
    coherent_plan = generate_nested_validation(manifest_path, policy=altered_policy)
    kwargs["validation_plan_path"].write_text(
        json.dumps(coherent_plan, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_p0_baseline_ladder(**kwargs)


def test_out_of_band_anchor_rejects_coherently_replaced_slice_and_registration(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _run_kwargs(tmp_path)
    altered_dir = tmp_path / "altered_phase_a"
    altered_dir.mkdir()
    for filename in (
        "cell_line_manifest.csv",
        "differentially_essential_slice.csv",
        "k_label_panels.csv",
        "phase_a_registration.json",
    ):
        shutil.copyfile(_REAL_PHASE_A_DIR / filename, altered_dir / filename)
    slice_path = altered_dir / "differentially_essential_slice.csv"
    pd.read_csv(slice_path).iloc[:-1].to_csv(slice_path, index=False)
    registration_path = altered_dir / "phase_a_registration.json"
    registration = json.loads(registration_path.read_text())
    registration["artifacts"]["differential_slice_sha256"] = hashlib.sha256(
        slice_path.read_bytes()
    ).hexdigest()
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    kwargs["phase_a_dir"] = altered_dir

    with pytest.raises(EvaluationContractError, match="SHA-256 mismatch"):
        run_p0_baseline_ladder(**kwargs)


def test_deterministic_atomic_artifacts_and_copy_prior_metrics(tmp_path: Path) -> None:
    kwargs, context, prior = _run_kwargs(tmp_path)
    first = run_p0_baseline_ladder(**kwargs, context_path=context, copy_k562_path=prior)
    second = run_p0_baseline_ladder(
        **kwargs, context_path=context, copy_k562_path=prior
    )
    pd.testing.assert_frame_equal(first.per_prediction, second.per_prediction)
    pd.testing.assert_frame_equal(first.per_line, second.per_line)
    assert first.summary == second.summary
    assert first.summary["protocol_id"] == PROTOCOL_ID
    assert first.summary["formal"] is False
    assert first.summary["test_lines_excluded"] is True
    assert "delta_rho" in first.summary["methods"][PCA_RIDGE_METHOD]

    output = tmp_path / "output"
    write_p0_baseline_artifacts(first, output)
    assert json.loads((output / "summary.json").read_text()) == first.summary
    assert (output / "per_prediction.csv").is_file()
    assert (output / "per_line.csv").is_file()
    with pytest.raises(FileExistsError):
        write_p0_baseline_artifacts(first, output)

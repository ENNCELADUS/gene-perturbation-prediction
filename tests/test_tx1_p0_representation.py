"""Tests for the feature-file-driven diagnostic Tx1 P0 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aivc_model.tx1_p0_representation as p0
from aivc_model.tx1_p0_representation import (
    audit_representation,
    fit_outer_fold,
    load_gene_effect,
    load_representation,
    load_shared_prior,
    load_train_head_manifest,
    run_audit,
)
from aivc_model.tx1_p0_validation import ValidationPolicy, generate_nested_validation


def _write_inputs(
    tmp_path: Path,
    *,
    line_count: int = 5,
    anchor_count: int = 1,
    test_count: int = 1,
    gene_count: int = 4,
) -> tuple[Path, Path, Path]:
    model_ids = [f"ACH-{index:06d}" for index in range(line_count)]
    manifest_rows = [
        {
            "model_id": model_id,
            "role": "train_head",
            "basal_source": "Tahoe-100M DMSO",
            "lineage": f"Lineage_{index % 4}",
            "dmso_cells": 1000 + index,
        }
        for index, model_id in enumerate(model_ids)
    ]
    for index in range(anchor_count):
        manifest_rows.append(
            {
                "model_id": f"ACH-ANCHOR-{index}",
                "role": "train_response_and_head",
                "basal_source": "Perturb-seq non-targeting control",
                "lineage": "Anchor",
                "dmso_cells": 100,
            }
        )
    for index in range(test_count):
        manifest_rows.append(
            {
                "model_id": f"ACH-TEST-{index}",
                "role": "test",
                "basal_source": "Tahoe-100M DMSO",
                "lineage": "Test",
                "dmso_cells": 500,
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    representation_path = tmp_path / "representation.csv"
    pd.DataFrame(
        {
            "model_id": model_ids,
            "feature_a": np.arange(line_count, dtype=float),
            "feature_b": np.arange(line_count, dtype=float) ** 2,
            "feature_c": np.sin(np.arange(line_count, dtype=float)) + 0.1,
        }
    ).to_csv(representation_path, index=False)

    genes = [f"GENE_{index}" for index in range(gene_count)]
    base = np.linspace(-1.2, 1.1, gene_count)
    slopes = np.sin(np.arange(gene_count) + 0.5) * 0.2
    rows: list[dict[str, object]] = []
    for line_index, model_id in enumerate(model_ids):
        values = base + line_index * slopes + (line_index**2) * slopes[::-1] * 0.03
        for gene, value in zip(genes, values, strict=True):
            rows.append(
                {
                    "model_id": model_id,
                    "gene_symbol": gene,
                    "gene_effect": value,
                }
            )
    gene_effect_path = tmp_path / "gene_effect.csv"
    pd.DataFrame(rows).to_csv(gene_effect_path, index=False)
    return manifest_path, representation_path, gene_effect_path


def _load_synthetic(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    manifest_path, representation_path, gene_effect_path = _write_inputs(tmp_path)
    manifest = load_train_head_manifest(manifest_path, expected_lines=5)
    model_ids = manifest["model_id"].tolist()
    features = load_representation(representation_path, model_ids)
    labels = load_gene_effect(gene_effect_path, model_ids, expected_genes=4)
    return features, labels, manifest_path, representation_path, gene_effect_path


def _write_validation_plan(
    manifest_path: Path, output_path: Path, policy_path: Path
) -> tuple[Path, Path]:
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    policy_payload = {
        "protocol_id": "tx1_geneeffect_p0_v1",
        "version": 1,
        "seed": 20260804,
        "expected_manifest_sha256": digest,
        "expected_role_counts": {
            "train_head": 29,
            "train_response_and_head": 4,
            "test": 9,
        },
        "inner_fold_count": 5,
        "dmso_quantile_bins": 4,
    }
    policy_path.write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload = generate_nested_validation(
        manifest_path, policy=ValidationPolicy.from_mapping(policy_payload)
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path, policy_path


def _write_phase_a_dir(tmp_path: Path, genes: list[str]) -> Path:
    phase_a_dir = tmp_path / "phase_a"
    phase_a_dir.mkdir()
    (phase_a_dir / "phase_a_registration.json").write_text("{}\n", encoding="utf-8")
    pd.DataFrame({"gene_symbol": genes}).to_csv(
        phase_a_dir / "differentially_essential_slice.csv", index=False
    )
    return phase_a_dir


def test_run_audit_marks_diagnostic_and_excludes_test_and_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, representation_path, gene_effect_path = _write_inputs(
        tmp_path, line_count=29, anchor_count=4, test_count=9, gene_count=587
    )
    phase_a_dir = _write_phase_a_dir(
        tmp_path, [f"GENE_{index}" for index in range(587)]
    )
    monkeypatch.setattr(p0, "verify_artifact_hashes", lambda _: None)
    validation_plan_path, validation_policy_path = _write_validation_plan(
        manifest_path,
        tmp_path / "validation_plan.json",
        tmp_path / "validation_policy.json",
    )

    result = run_audit(
        manifest_path,
        validation_plan_path,
        validation_policy_path,
        phase_a_dir,
        {"candidate": representation_path},
        gene_effect_path,
    )

    assert result["protocol_id"] == "tx1_geneeffect_p0_v1"
    assert result["formal"] is False
    assert result["test_lines_excluded"] is True
    assert result["metadata"]["tx1_frozen_cache_accessed"] is False
    assert result["metadata"]["representation_fit_provenance_verified"] is False
    assert result["input_sha256"]["manifest"]
    assert result["input_sha256"]["validation_plan"]
    assert result["input_sha256"]["validation_policy"]
    assert result["input_sha256"]["phase_a_registration"]
    assert result["input_sha256"]["phase_a_slice"]
    assert result["config"]["pca_components_requested"] == 8
    per_line = result["representations"]["candidate"]["per_line"]
    assert len(per_line) == 29
    assert {row["model_id"] for row in per_line}.isdisjoint(
        {"ACH-ANCHOR-0", "ACH-TEST-0"}
    )
    assert {row["k"] for row in per_line} == {0}


def test_every_fit_uses_outer_train_only_and_never_held_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, labels, _, _, _ = _load_synthetic(tmp_path)
    original = p0.fit_outer_fold
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_fit(
        train_context: np.ndarray,
        train_gene_effect: np.ndarray,
        held_context: np.ndarray,
        prior: np.ndarray,
        **kwargs: object,
    ) -> p0.OuterFoldPredictions:
        calls.append((train_context.copy(), train_gene_effect.copy()))
        return original(
            train_context,
            train_gene_effect,
            held_context,
            prior,
            **kwargs,
        )

    monkeypatch.setattr(p0, "fit_outer_fold", recording_fit)
    audit_representation(features, labels)

    assert len(calls) == len(features)
    x = features.to_numpy()
    y = labels.to_numpy()
    for held_index, (fit_x, fit_y) in enumerate(calls):
        expected_mask = np.arange(len(features)) != held_index
        np.testing.assert_array_equal(fit_x, x[expected_mask])
        np.testing.assert_array_equal(fit_y, y[expected_mask])


@pytest.mark.parametrize("corruption", ["truncated", "replaced"])
def test_run_audit_rejects_wrong_frozen_gene_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    manifest_path, representation_path, gene_effect_path = _write_inputs(
        tmp_path, line_count=29, anchor_count=4, test_count=9, gene_count=587
    )
    validation_plan_path, validation_policy_path = _write_validation_plan(
        manifest_path,
        tmp_path / "validation_plan.json",
        tmp_path / "validation_policy.json",
    )
    genes = [f"GENE_{index}" for index in range(587)]
    genes = genes[:-1] if corruption == "truncated" else [*genes[:-1], "OTHER"]
    phase_a_dir = _write_phase_a_dir(tmp_path, genes)
    monkeypatch.setattr(p0, "verify_artifact_hashes", lambda _: None)

    with pytest.raises(ValueError, match="587 unique|differs"):
        run_audit(
            manifest_path,
            validation_plan_path,
            validation_policy_path,
            phase_a_dir,
            {"candidate": representation_path},
            gene_effect_path,
        )


def test_fit_outer_fold_caps_pca_and_has_no_held_label_argument() -> None:
    train_x = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])
    train_y = np.asarray([[-1.0, 0.0, 1.0], [-0.5, 0.4, 0.8], [-0.2, 0.9, 0.1]])

    result = fit_outer_fold(
        train_x,
        train_y,
        np.asarray([1.5, 2.0]),
        np.asarray([-0.6, 0.3, 0.5]),
        pca_components=8,
    )

    assert result.pca_components == 2
    assert result.ridge.shape == (3,)


def test_shuffled_context_control_is_deterministic(tmp_path: Path) -> None:
    features, labels, _, _, _ = _load_synthetic(tmp_path)

    first = audit_representation(features, labels, shuffle_seed=41)
    second = audit_representation(features, labels, shuffle_seed=41)

    assert first == second
    control = first["negative_control"]
    assert "shuffled_macro_delta_rho" in control
    if first["summary"]["ridge"]["macro_delta_rho"] <= 0:
        assert control["retained_gain_ratio"] is None
        assert control["retained_gain_ratio_reason"]


def test_shared_prior_uses_common_p0_gene_schema(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.csv"
    pd.DataFrame(
        {
            "gene_symbol": ["GENE_1", "GENE_0"],
            "gene_effect": [-0.2, -1.1],
        }
    ).to_csv(prior_path, index=False)

    prior = load_shared_prior(prior_path, ["GENE_0", "GENE_1"])

    np.testing.assert_allclose(prior, [-1.1, -0.2])


@pytest.mark.parametrize("bad_role", ["invalid", "anchor", "validation"])
def test_manifest_rejects_invalid_role(tmp_path: Path, bad_role: str) -> None:
    manifest_path, _, _ = _write_inputs(tmp_path)
    frame = pd.read_csv(manifest_path)
    frame.loc[0, "role"] = bad_role
    frame.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="invalid role"):
        load_train_head_manifest(manifest_path, expected_lines=4)


def test_run_audit_rejects_stale_validation_plan(tmp_path: Path) -> None:
    manifest_path, representation_path, gene_effect_path = _write_inputs(
        tmp_path, line_count=29, anchor_count=4, test_count=9
    )
    validation_plan_path, validation_policy_path = _write_validation_plan(
        manifest_path,
        tmp_path / "validation_plan.json",
        tmp_path / "validation_policy.json",
    )
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "role"] = "test"
    manifest.loc[manifest["model_id"] == "ACH-TEST-0", "role"] = "train_head"
    manifest.to_csv(manifest_path, index=False)
    altered_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    plan_payload = json.loads(validation_plan_path.read_text(encoding="utf-8"))
    plan_payload["input"]["cell_line_manifest_sha256"] = altered_digest
    validation_plan_path.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_audit(
            manifest_path,
            validation_plan_path,
            validation_policy_path,
            tmp_path,
            {"candidate": representation_path},
            gene_effect_path,
            expected_genes=4,
        )


def test_representation_rejects_test_or_anchor_id(tmp_path: Path) -> None:
    _, representation_path, _ = _write_inputs(tmp_path)
    frame = pd.read_csv(representation_path)
    frame.loc[0, "model_id"] = "ACH-TEST-0"
    frame.to_csv(representation_path, index=False)

    with pytest.raises(ValueError, match="exactly equal.*train_head"):
        load_representation(
            representation_path, [f"ACH-{index:06d}" for index in range(5)]
        )


def test_representation_rejects_missing_nonfinite_and_constant_features(
    tmp_path: Path,
) -> None:
    _, representation_path, _ = _write_inputs(tmp_path)
    model_ids = [f"ACH-{index:06d}" for index in range(5)]
    frame = pd.read_csv(representation_path)
    frame.loc[0, "feature_a"] = np.nan
    frame["constant"] = 1.0
    frame.to_csv(representation_path, index=False)

    with pytest.raises(ValueError, match="finite numeric"):
        load_representation(representation_path, model_ids)

    frame["feature_a"] = np.arange(5, dtype=float)
    frame.to_csv(representation_path, index=False)
    with pytest.raises(ValueError, match="constant feature"):
        load_representation(representation_path, model_ids)


def test_gene_effect_fails_closed_on_missing_or_different_gene_universe(
    tmp_path: Path,
) -> None:
    _, _, gene_effect_path = _write_inputs(tmp_path)
    model_ids = [f"ACH-{index:06d}" for index in range(5)]
    frame = pd.read_csv(gene_effect_path)
    frame = frame.loc[
        ~((frame["model_id"] == model_ids[-1]) & (frame["gene_symbol"] == "GENE_3"))
    ]
    frame.to_csv(gene_effect_path, index=False)

    with pytest.raises(ValueError, match="gene universe must be identical"):
        load_gene_effect(gene_effect_path, model_ids, expected_genes=4)

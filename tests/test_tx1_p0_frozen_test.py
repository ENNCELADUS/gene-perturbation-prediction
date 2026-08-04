"""Tests for the prediction-first P0 frozen-test diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aivc_model.tx1_p0_frozen_test import (
    EXPECTED_CACHE_MANIFEST_SHA256,
    EXPECTED_COMPARATORS,
    EXPECTED_TX1_MODEL_SHA256,
    build_frozen_predictions,
    evaluate_predictions,
    sha256_file,
)
from aivc_model.tx1_p0_representation import OuterFoldPredictions


def test_prediction_phase_uses_only_train_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_ids = [f"TRAIN-{index:02d}" for index in range(29)]
    test_ids = [f"TEST-{index:02d}" for index in range(9)]
    genes = ["A", "B", "C"]
    slice_frame = pd.DataFrame(
        {
            "gene_symbol": genes,
            "depmap_column": ["A (1)", "B (2)", "C (3)"],
        }
    )
    train = pd.DataFrame(
        {"model_id": train_ids, "role": "train_head", "basal_source": "Tahoe"}
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._load_authorities",
        lambda phase_a_dir, manifest_path: (train, slice_frame, {"sources": {}}),
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._test_ids", lambda manifest_path: test_ids
    )
    observed_ids: list[str] = []

    def fake_gene_effect(path, registration, requested_ids, frozen_slice):
        observed_ids.extend(requested_ids)
        long = pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "gene_symbol": gene,
                    "gene_effect": float(line_index + gene_index),
                }
                for line_index, model_id in enumerate(requested_ids)
                for gene_index, gene in enumerate(genes)
            ]
        )
        prior = pd.DataFrame({"gene_symbol": genes, "gene_effect": [0.0, 1.0, 2.0]})
        return long, prior

    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._load_gene_effect", fake_gene_effect
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test.verify_cache",
        lambda cache_root, frozen_manifest_path: {"status": "verified"},
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._validate_exposure_ledger",
        lambda path, ids: {"sha256": "ledger"},
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._pooled_test_context",
        lambda cache_root, ids, array_filename, prefix: pd.DataFrame(
            [[float(i), float(i + 1)] for i in range(9)],
            index=ids,
            columns=["feature_0", "feature_1"],
        ),
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test.fit_outer_fold",
        lambda *args, **kwargs: OuterFoldPredictions(
            nearest_neighbor=np.asarray([0.0, 1.0, 2.0]),
            ridge=np.asarray([0.1, 1.1, 2.1]),
            shuffled_ridge=np.asarray([0.2, 1.2, 2.2]),
            nearest_neighbor_index=0,
            pca_components=2,
            dropped_constant_feature_count=0,
        ),
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._load_comparator",
        lambda path, manifest_path, output_method, **kwargs: pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "depmap_column": depmap_column,
                    "gene_symbol": gene,
                    "method": output_method,
                    "base_pred": float(gene_index),
                }
                for model_id in test_ids
                for gene_index, (gene, depmap_column) in enumerate(
                    zip(genes, slice_frame["depmap_column"], strict=True)
                )
            ]
        ),
    )
    representation = tmp_path / "train_context.csv"
    pd.DataFrame(
        {
            "model_id": train_ids,
            "feature_0": np.arange(29),
            "feature_1": np.arange(29) + 1,
        }
    ).to_csv(representation, index=False)
    for filename in ("manifest.csv", "gene_effect.csv"):
        (tmp_path / filename).write_text("placeholder\n", encoding="utf-8")
    phase_a = tmp_path / "phase_a"
    phase_a.mkdir()
    (phase_a / "phase_a_registration.json").write_text("{}\n", encoding="utf-8")
    slice_frame.to_csv(phase_a / "differentially_essential_slice.csv", index=False)
    cache_root = tmp_path / "cache"
    for model_id in test_ids:
        line_dir = cache_root / model_id
        line_dir.mkdir(parents=True)
        np.save(line_dir / "embeddings.npy", np.ones((1, 2), dtype=np.float32))
    (cache_root / "manifest.json").write_text(
        '{"tx1_source_manifest": {}}\n', encoding="utf-8"
    )
    comparator_paths = {
        "previous_hvg": tmp_path / "previous_hvg.csv",
        "previous_tx1": tmp_path / "previous_tx1.csv",
    }
    for path in comparator_paths.values():
        path.write_text("placeholder\n", encoding="utf-8")
    comparator_manifest_paths = {
        name: tmp_path / f"{name}_manifest.json" for name in comparator_paths
    }
    for path in comparator_manifest_paths.values():
        path.write_text("{}\n", encoding="utf-8")

    predictions, metadata = build_frozen_predictions(
        phase_a_dir=phase_a,
        manifest_path=tmp_path / "manifest.csv",
        raw_gene_effect_path=tmp_path / "gene_effect.csv",
        cache_root=cache_root,
        exposure_ledger_path=tmp_path / "exposure.csv",
        representation_paths={"tx1": representation},
        cache_arrays={"tx1": ("embeddings.npy", "feature")},
        comparator_paths=comparator_paths,
        comparator_manifest_paths=comparator_manifest_paths,
    )

    assert observed_ids == train_ids
    assert not set(test_ids).intersection(observed_ids)
    assert metadata["test_labels_accessed"] is False
    assert len(predictions) == 9 * 3 * 7


def test_evaluation_rejects_invalid_predictions_before_opening_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_a = tmp_path / "phase_a"
    phase_a.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    gene_effect_path = tmp_path / "gene_effect.csv"
    exposure_path = tmp_path / "exposure.csv"
    registration_path = phase_a / "phase_a_registration.json"
    slice_path = phase_a / "differentially_essential_slice.csv"
    for path in (manifest_path, gene_effect_path, exposure_path, registration_path):
        path.write_text("placeholder\n", encoding="utf-8")
    pd.DataFrame({"gene_symbol": ["A"], "depmap_column": ["A (1)"]}).to_csv(
        slice_path, index=False
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_path = prediction_dir / "predictions.csv"
    prediction_path.write_text("model_id,method\nTEST,copy_k562\n", encoding="utf-8")
    prediction_manifest = prediction_dir / "prediction_manifest.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "protocol_id": "tx1_geneeffect_p0_frozen_test_v1",
                "test_labels_accessed": False,
                "formal": False,
                "post_hoc": True,
                "prediction_first": True,
                "n_train_lines": 29,
                "n_test_lines": 9,
                "n_genes": 587,
                "config": {
                    "pca_components": 8,
                    "ridge_alpha": 1.0,
                    "shuffle_seed": 20260804,
                },
                "predictions_sha256": sha256_file(prediction_path),
                "input_sha256": {
                    "manifest": sha256_file(manifest_path),
                    "phase_a_registration": sha256_file(registration_path),
                    "phase_a_slice": sha256_file(slice_path),
                    "raw_gene_effect": sha256_file(gene_effect_path),
                },
                "exposure": {"sha256": "ledger"},
                "comparators": {
                    name: {
                        "predictions_sha256": contract["predictions_sha256"],
                        "manifest_sha256": contract["manifest_sha256"],
                        "head_checkpoint_sha256": contract["head_sha256"],
                        "reason": contract["reason"],
                    }
                    for name, contract in EXPECTED_COMPARATORS.items()
                },
                "cache_manifest": {
                    "sha256": EXPECTED_CACHE_MANIFEST_SHA256,
                    "tx1_source_manifest": {
                        "model_label": "tahoe_x1_3b",
                        "status": "verified",
                        "files": {
                            "model.safetensors": {"sha256": EXPECTED_TX1_MODEL_SHA256}
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._load_authorities",
        lambda *args: (
            pd.DataFrame(),
            pd.DataFrame({"gene_symbol": ["A"], "depmap_column": ["A (1)"]}),
            {
                "sources": {
                    "depmap_gene_effect": {"sha256": sha256_file(gene_effect_path)}
                }
            },
        ),
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._test_ids",
        lambda *args: [f"TEST-{index:02d}" for index in range(9)],
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._load_test_truth",
        lambda *args: pytest.fail("test labels opened before structure checks"),
    )
    monkeypatch.setattr(
        "aivc_model.tx1_p0_frozen_test._validate_exposure_ledger",
        lambda path, ids: {"sha256": "ledger"},
    )

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_predictions(
            prediction_dir=prediction_dir,
            phase_a_dir=phase_a,
            manifest_path=manifest_path,
            raw_gene_effect_path=gene_effect_path,
            exposure_ledger_path=exposure_path,
            expected_prediction_manifest_sha256=sha256_file(prediction_manifest),
        )

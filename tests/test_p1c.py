import json
import numpy as np
import pandas as pd
import pytest
import torch


def test_bias_decomposition_splits_shared_from_condition_specific():
    from src.eval.p1c_tier0 import bias_decomposition

    observed = np.zeros((4, 3))
    predicted = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0], [1.0, -2.0, 0]])
    result = bias_decomposition(predicted, observed)
    # mean_g ||e||^2 = (1 + 1 + 5 + 5)/4 = 3 ; ||ē||^2 = 1 ; residual = 2
    assert result["total_mse"] == pytest.approx(3.0 / 3)  # per-coordinate mean
    assert result["shared_bias"] == pytest.approx(1.0 / 3)
    assert result["condition_specific"] == pytest.approx(2.0 / 3)
    assert result["shared_bias_fraction"] == pytest.approx(1 / 3)
    assert result["n"] == 4


def test_decompose_effects_reads_npz_by_method_and_anchor(tmp_path):
    from src.eval.p1c_tier0 import decompose_effects

    keys = np.array([("model", "A", "G"), ("model", "A", "H"), ("no_change", "A", "G")])
    predicted = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    observed = np.zeros((3, 2))
    np.savez(
        tmp_path / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    frame = decompose_effects(tmp_path / "effects.npz", method="model")
    assert frame.model_id.tolist() == ["A"] and frame.n.tolist() == [2]
    assert frame.shared_bias_fraction.iloc[0] == pytest.approx(1.0)


def test_matched_coverage_restricts_references_to_perturbation_mean_support():
    from src.eval.p1c_tier0 import matched_coverage

    rows = []
    for gene, covered in (("G", True), ("H", False)):
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="no_change",
                response_loss=1.0,
            )
        )
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="global_mean",
                response_loss=2.0,
            )
        )
        if covered:
            rows.append(
                dict(
                    role="val",
                    model_id="A",
                    panel="all",
                    gene=gene,
                    method="perturbation_mean",
                    response_loss=3.0,
                )
            )
    frame = matched_coverage(pd.DataFrame(rows))
    row = frame[(frame.model_id == "A") & (frame.reference == "no_change")].iloc[0]
    assert row.covered == 1 and row.total == 2
    assert row.perturbation_mean_loss == 3.0 and row.reference_loss_on_covered == 1.0


def test_cross_context_pearson_is_undefined_for_constant_effect_references():
    from src.eval.p1b import cross_context

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
    }
    # Identical predicted effects in both anchors -> zero difference up to float noise.
    noise = np.array([1e-7, -1e-7, 0.0])
    effects = {
        ("global_mean", "a", "G"): (
            np.array([1.0, 2.0, 3.0]) + noise,
            np.array([1.0, 0.0, 0.0]),
        ),
        ("global_mean", "b", "G"): (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
    }
    frame = pd.DataFrame({"method": ["global_mean"]})
    result = cross_context(
        frame,
        effects,
        {**bundle, "splits": {"train": [], "val": [0, 1], "external": [2]}},
    )
    assert len(result) == 1 and np.isnan(result.effect_difference_pearson.iloc[0])


def _build_tier0_fixture(tmp_path):
    from src.data.gene_splits import sha256_file

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    response_cache_dir = tmp_path / "cache"
    (response_cache_dir / "response_targets").mkdir(parents=True)
    pd.DataFrame({"model_id": ["a", "a", "b"], "n_cells": [100, 120, 90]}).to_parquet(
        response_cache_dir / "response_targets" / "metadata.parquet"
    )

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
        "controls": {
            "a": {"hvg": np.zeros((5, 3))},
            "b": {"hvg": np.zeros((5, 3))},
        },
        "response_cache": str(response_cache_dir),
    }
    torch.save(bundle, prepared / "bundle.pt")
    manifest = {
        "bundle_sha256": sha256_file(prepared / "bundle.pt"),
        "source_missing_genes": {"a": [], "b": []},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))

    runs = tmp_path / "runs"
    export_dir = runs / "evaluation" / "B-init" / "internal"
    export_dir.mkdir(parents=True)
    methods = ("model", "no_change", "global_mean", "perturbation_mean")
    conditions = pd.DataFrame(
        [
            dict(
                role="val",
                model_id=anchor,
                panel="all",
                gene="G",
                method=method,
                response_loss=1.0,
            )
            for anchor in ("a", "b")
            for method in methods
        ]
    )
    conditions.to_parquet(export_dir / "conditions.parquet", index=False)
    keys = np.array(
        [(method, anchor, "G") for anchor in ("a", "b") for method in methods]
    )
    predicted = np.tile(np.array([1.0, 0.0]), (len(keys), 1))
    observed = np.zeros((len(keys), 2))
    np.savez(
        export_dir / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    (export_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "state": "B-init",
                "bundle": manifest["bundle_sha256"],
            }
        )
    )
    return prepared, runs, export_dir, manifest


def test_run_tier0_verifies_bundle_and_evaluation_identity_and_writes_outputs(tmp_path):
    from src.eval.p1c_tier0 import run_tier0

    prepared, runs, export_dir, manifest = _build_tier0_fixture(tmp_path)
    original_bundle_bytes = (prepared / "bundle.pt").read_bytes()

    out_dir = tmp_path / "out"
    result = run_tier0(runs, prepared, out_dir)
    assert result == out_dir
    for name in (
        "bias_decomposition.csv",
        "matched_coverage.csv",
        "anchor_audit.csv",
        "cross_context_guarded.csv",
        "tier0.md",
    ):
        assert (out_dir / name).exists()

    # Tampering with the prepared bundle after preparation must be caught.
    (prepared / "bundle.pt").write_bytes(original_bundle_bytes + b"\x00")
    with pytest.raises(ValueError, match="identity changed"):
        run_tier0(runs, prepared, out_dir)
    (prepared / "bundle.pt").write_bytes(original_bundle_bytes)

    # An export recorded against a different bundle must also be rejected.
    (export_dir / "evaluation.json").write_text(
        json.dumps({"status": "completed", "state": "B-init", "bundle": "stale-hash"})
    )
    with pytest.raises(ValueError):
        run_tier0(runs, prepared, out_dir)

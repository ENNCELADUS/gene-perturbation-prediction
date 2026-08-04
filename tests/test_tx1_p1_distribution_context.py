"""Tests for the train-only P1 distribution-context ablation driver."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_tx1_p1_distribution_context as p1_module
from scripts.run_tx1_p1_distribution_context import (
    EXPECTED_REPRESENTATIONS,
    PROTOCOL_ID,
    _load_policy,
    _paired_comparisons,
)


def _policy() -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "formal": False,
        "ccle_bulk_control": {
            "status": "coverage_incomplete_not_run",
            "excluded_missing_model_id": "ACH-001039",
        },
        "test_lines_excluded": True,
        "selection": "none_fixed_ablation",
        "pca_components": 8,
        "ridge_alpha": 1.0,
        "shuffle_seed": 20260804,
        "representations": EXPECTED_REPRESENTATIONS,
    }


def test_policy_is_exact_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy()), encoding="utf-8")
    assert _load_policy(path)["formal"] is False

    changed = _policy()
    changed["pca_components"] = 9
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen implementation contract"):
        _load_policy(path)


def test_paired_comparisons_use_matching_lines() -> None:
    representations = {}
    offsets = {
        "tx1_mean": 0.0,
        "tx1_moments": 0.1,
        "hvg_mean": 0.2,
        "hvg_moments": 0.25,
        "multiview": 0.3,
    }
    for name, offset in offsets.items():
        representations[name] = {
            "per_line": [
                {"model_id": f"LINE-{index}", "ridge_rho": index / 10 + offset}
                for index in range(4)
            ]
        }

    comparisons = _paired_comparisons({"representations": representations})

    by_pair = {(row["candidate"], row["reference"]): row for row in comparisons}
    assert by_pair[("tx1_moments", "tx1_mean")]["delta_rho"] == pytest.approx(0.1)
    assert by_pair[("hvg_moments", "hvg_mean")]["delta_rho"] == pytest.approx(0.05)


def test_paired_comparisons_reject_mismatched_lines() -> None:
    audit = {
        "representations": {
            "tx1_mean": {"per_line": [{"model_id": "A", "ridge_rho": 0.0}]},
            "tx1_moments": {"per_line": [{"model_id": "B", "ridge_rho": 0.1}]},
            "hvg_mean": {"per_line": [{"model_id": "A", "ridge_rho": 0.0}]},
            "hvg_moments": {"per_line": [{"model_id": "A", "ridge_rho": 0.1}]},
            "multiview": {"per_line": [{"model_id": "A", "ridge_rho": 0.1}]},
        }
    }
    with pytest.raises(ValueError, match="coverage differs"):
        _paired_comparisons(audit)


def _fake_audit() -> dict[str, object]:
    representations = {}
    for name in EXPECTED_REPRESENTATIONS:
        representations[name] = {
            "per_line": [
                {"model_id": f"LINE-{index}", "ridge_rho": index / 10}
                for index in range(4)
            ]
        }
    return {
        "formal": False,
        "test_lines_excluded": True,
        "metadata": {},
        "representations": representations,
    }


def test_driver_writes_atomic_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    output = tmp_path / "run"
    args = SimpleNamespace(
        phase_a_dir=tmp_path,
        manifest=tmp_path / "manifest.csv",
        gene_effect=tmp_path / "gene_effect.csv",
        cache_root=tmp_path / "cache",
        expression=tmp_path / "expression.csv",
        validation_plan=tmp_path / "plan.json",
        validation_policy=tmp_path / "validation_policy.json",
        p1_policy=policy_path,
        output_dir=output,
    )
    monkeypatch.setattr(p1_module, "parse_args", lambda: args)
    monkeypatch.setattr(p1_module, "build_p0_inputs", lambda **kwargs: object())

    def fake_write(_result, inputs_dir: Path) -> None:
        inputs_dir.mkdir(parents=True)
        (inputs_dir / "provenance.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(p1_module, "write_p0_inputs", fake_write)
    monkeypatch.setattr(p1_module, "run_audit", lambda **kwargs: _fake_audit())

    assert p1_module.main() == 0

    payload = json.loads((output / "representation_audit.json").read_text())
    assert payload["formal"] is False
    assert payload["test_lines_excluded"] is True
    assert payload["metadata"]["test_context_integrity_read"] is True
    assert payload["metadata"]["test_context_used_for_representation_fit"] is False
    assert payload["metadata"]["test_labels_accessed"] is False
    assert payload["metadata"]["shared_prior_fit_provenance_verified"] is True
    assert payload["metadata"]["shared_prior_model_id"] == "ACH-000551"
    assert payload["input_provenance_sha256"]
    assert not list(tmp_path.glob(".run.tmp-*"))


def test_driver_failure_leaves_no_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    output = tmp_path / "failed_run"
    args = SimpleNamespace(
        phase_a_dir=tmp_path,
        manifest=tmp_path / "manifest.csv",
        gene_effect=tmp_path / "gene_effect.csv",
        cache_root=tmp_path / "cache",
        expression=tmp_path / "expression.csv",
        validation_plan=tmp_path / "plan.json",
        validation_policy=tmp_path / "validation_policy.json",
        p1_policy=policy_path,
        output_dir=output,
    )
    monkeypatch.setattr(p1_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        p1_module,
        "build_p0_inputs",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        p1_module.main()

    assert not output.exists()
    assert not list(tmp_path.glob(".failed_run.tmp-*"))

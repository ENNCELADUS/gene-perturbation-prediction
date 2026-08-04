"""Tests for the train-only P1 distribution-context ablation driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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

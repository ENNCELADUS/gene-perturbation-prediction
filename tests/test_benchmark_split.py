"""Tests for aivc_model.benchmark_split -- the Exp13 226-line fit/train guard.

Replaces ``tx1_geneeffect_data.assert_training_role``'s D6 guard, which
checked the retired Phase-A ``train_head``/``test`` role column. This
module's authority is the ``cell_line_geneeffect_226_split`` JSON instead
(``train``/``val``/``test``/``unlabeled_train``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aivc_model.benchmark_split import assert_fit_eligible, load_geneeffect_226_split
from aivc_model.residual_ladder import FixedSplit

_SPLIT = FixedSplit(
    train=("ACH-TRAIN-1", "ACH-TRAIN-2", "ACH-PC9", "ACH-HELA"),
    val=("ACH-VAL-1",),
    test=("ACH-TEST-1",),
    unlabeled_train=("ACH-PC9", "ACH-HELA"),
)


# --- assert_fit_eligible: the hard guard -------------------------------------


def test_assert_fit_eligible_accepts_labeled_train_member() -> None:
    assert_fit_eligible("ACH-TRAIN-1", _SPLIT)  # must not raise


def test_assert_fit_eligible_rejects_unlabeled_train_member() -> None:
    """PC9/HeLa are train members but have no GeneEffect label -- excluded
    from every supervised fit (the Exp13 spec under docs/specs/, §2)."""
    with pytest.raises(ValueError, match="unlabeled_train"):
        assert_fit_eligible("ACH-PC9", _SPLIT)


def test_assert_fit_eligible_rejects_val_line_leakage() -> None:
    """The leakage case: a val line must never reach a fit."""
    with pytest.raises(ValueError, match="not a labeled train member"):
        assert_fit_eligible("ACH-VAL-1", _SPLIT)


def test_assert_fit_eligible_rejects_test_line_leakage() -> None:
    """The leakage case: a test line must never reach a fit."""
    with pytest.raises(ValueError, match="not a labeled train member"):
        assert_fit_eligible("ACH-TEST-1", _SPLIT)


def test_assert_fit_eligible_rejects_unknown_line() -> None:
    with pytest.raises(ValueError, match="not a labeled train member"):
        assert_fit_eligible("ACH-NOT-IN-SPLIT", _SPLIT)


# --- load_geneeffect_226_split: shape validation -----------------------------


def _write_split(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_load_geneeffect_226_split_round_trips_the_real_split(tmp_path: Path) -> None:
    split_path = _write_split(
        tmp_path / "split.json",
        {
            "train": ["ACH-TRAIN-1", "ACH-PC9"],
            "val": ["ACH-VAL-1"],
            "test": ["ACH-TEST-1"],
            "unlabeled_train": ["ACH-PC9"],
        },
    )
    split = load_geneeffect_226_split(split_path)
    assert split.train == ("ACH-TRAIN-1", "ACH-PC9")
    assert split.val == ("ACH-VAL-1",)
    assert split.test == ("ACH-TEST-1",)
    assert split.unlabeled_train == ("ACH-PC9",)


def test_load_geneeffect_226_split_defaults_unlabeled_train_to_empty(
    tmp_path: Path,
) -> None:
    split_path = _write_split(
        tmp_path / "split.json",
        {"train": ["ACH-1"], "val": [], "test": []},
    )
    split = load_geneeffect_226_split(split_path)
    assert split.unlabeled_train == ()


def test_load_geneeffect_226_split_against_the_real_tracked_file() -> None:
    """The actual tracked split file must load and satisfy the documented shape
    (172 train / 27 val / 27 test, PC9 + HeLa unlabeled) -- docs/specs/2026-08-17-exp13-
    geneeffect-residual-protocol.md §2."""
    repo_root = Path(__file__).resolve().parents[1]
    split_path = (
        repo_root / "configs" / "benchmarks" / "cell_line_geneeffect_226_split.json"
    )
    if not split_path.exists():
        pytest.skip("cell_line_geneeffect_226_split.json not present")
    split = load_geneeffect_226_split(split_path)
    assert len(split.train) == 172
    assert len(split.val) == 27
    assert len(split.test) == 27
    assert set(split.unlabeled_train) == {"ACH-000779", "ACH-001086"}
    for model_id in split.unlabeled_train:
        with pytest.raises(ValueError):
            assert_fit_eligible(model_id, split)


def test_load_geneeffect_226_split_rejects_missing_key(tmp_path: Path) -> None:
    split_path = _write_split(tmp_path / "split.json", {"train": [], "val": []})
    with pytest.raises(ValueError, match="missing key"):
        load_geneeffect_226_split(split_path)


def test_load_geneeffect_226_split_rejects_non_string_list(tmp_path: Path) -> None:
    split_path = _write_split(
        tmp_path / "split.json",
        {"train": [1, 2], "val": [], "test": []},
    )
    with pytest.raises(ValueError, match="list of strings"):
        load_geneeffect_226_split(split_path)


def test_load_geneeffect_226_split_rejects_non_object_json(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_geneeffect_226_split(split_path)

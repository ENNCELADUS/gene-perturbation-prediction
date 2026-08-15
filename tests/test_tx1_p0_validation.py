"""Tests for Tx1 P0 nested validation folds."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_p0_validation import (
    MISSING_CONTEXT,
    PROTOCOL_ID,
    ValidationPolicy,
    generate_nested_validation,
    load_manifest,
    validate_nested_validation,
)
from conftest import tx1_manifest_row as _manifest_row


def _policy(expected_sha256: str) -> ValidationPolicy:
    return ValidationPolicy.from_mapping(
        {
            "protocol_id": PROTOCOL_ID,
            "version": 1,
            "seed": 17,
            "expected_manifest_sha256": expected_sha256,
            "expected_role_counts": {
                "train_head": 29,
                "train_response_and_head": 4,
                "test": 9,
            },
            "inner_fold_count": 5,
            "dmso_quantile_bins": 4,
        }
    )


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lineages: list[object] = ["Lung"] * 10 + ["Bowel"] * 7 + ["Skin"] * 6
    lineages += ["Pancreas"] * 5 + [None]
    for index, lineage in enumerate(lineages):
        rows.append(
            _manifest_row(
                model_id=f"ACH-H{index:04d}",
                lineage=lineage,
                dmso_cells=None if index == 28 else 100 * (index + 1),
                role="train_head",
            )
        )
    for index in range(4):
        rows.append(
            _manifest_row(
                model_id=f"ACH-A{index:04d}",
                lineage="Anchor",
                dmso_cells=None,
                basal_source="Perturb-seq non-targeting control",
                role="train_response_and_head",
            )
        )
    for index in range(9):
        rows.append(
            _manifest_row(model_id=f"ACH-T{index:04d}", lineage="Test", role="test")
        )
    return pd.DataFrame(rows)


def _generate(
    tmp_path: Path,
    manifest: pd.DataFrame | None = None,
    *,
    filename: str = "manifest.csv",
) -> tuple[dict[str, object], pd.DataFrame, ValidationPolicy]:
    path = tmp_path / filename
    (_manifest() if manifest is None else manifest).to_csv(path, index=False)
    policy = _policy(sha256_file(path))
    payload = generate_nested_validation(path, policy=policy)
    loaded, _ = load_manifest(path, policy)
    return payload, loaded, policy


def test_generation_is_deterministic_and_records_contract(
    tmp_path: Path,
) -> None:
    first, _, _ = _generate(tmp_path)
    second = generate_nested_validation(
        tmp_path / "manifest.csv",
        policy=_policy(sha256_file(tmp_path / "manifest.csv")),
    )

    assert first == second
    assert first["protocol_id"] == PROTOCOL_ID
    assert first["manifest_sha256"] == sha256_file(tmp_path / "manifest.csv")
    first_fold = first["outer_folds"][0]  # type: ignore[index]
    assert first_fold["held_out"] == first_fold["outer_validation_model_ids"][0]
    assert first_fold["outer_train"] == first_fold["outer_train_model_ids"]


def test_each_of_29_train_head_lines_is_outer_validation_exactly_once(
    tmp_path: Path,
) -> None:
    payload, _, _ = _generate(tmp_path)
    held_out = [
        model_id
        for fold in payload["outer_folds"]  # type: ignore[union-attr]
        for model_id in fold["outer_validation_model_ids"]
    ]
    assert len(held_out) == 29
    assert len(set(held_out)) == 29
    assert set(held_out) == {f"ACH-H{index:04d}" for index in range(29)}


def test_each_outer_fold_fits_context_only_on_outer_train(tmp_path: Path) -> None:
    payload, _, _ = _generate(tmp_path)
    for fold in payload["outer_folds"]:  # type: ignore[union-attr]
        outer_train = set(fold["outer_train_model_ids"])
        outer_validation = set(fold["outer_validation_model_ids"])
        preprocessing = fold["preprocessing"]
        assert set(preprocessing["fit_model_ids"]) == outer_train
        assert set(preprocessing["context_by_model_id"]) == outer_train
        assert outer_validation.isdisjoint(preprocessing["fit_model_ids"])
        assert preprocessing["dmso_cells"]["binning"] == "outer_train_quantiles"
        assert preprocessing["dmso_cells"]["transform"] == "log1p"

    missing_fold = next(
        fold
        for fold in payload["outer_folds"]  # type: ignore[union-attr]
        if fold["outer_validation_model_ids"] != ["ACH-H0028"]
    )
    missing_context = missing_fold["preprocessing"]["context_by_model_id"]["ACH-H0028"]
    assert missing_context == {
        "lineage": MISSING_CONTEXT,
        "log_dmso_cells": None,
        "dmso_bin": MISSING_CONTEXT,
    }


def test_inner_validation_partitions_outer_train_exactly_once(tmp_path: Path) -> None:
    payload, _, _ = _generate(tmp_path)
    for fold in payload["outer_folds"]:  # type: ignore[union-attr]
        expected = set(fold["outer_train_model_ids"])
        observed = [
            model_id
            for inner in fold["inner_folds"]
            for model_id in inner["validation_model_ids"]
        ]
        assert len(observed) == len(expected)
        assert set(observed) == expected
        sizes = [len(inner["validation_model_ids"]) for inner in fold["inner_folds"]]
        assert max(sizes) - min(sizes) <= 1


def test_manifest_row_order_does_not_change_fold_content(tmp_path: Path) -> None:
    manifest = _manifest()
    shuffled = manifest.sample(frac=1.0, random_state=91).reset_index(drop=True)
    first, _, _ = _generate(tmp_path, manifest, filename="first.csv")
    second, _, _ = _generate(tmp_path, shuffled, filename="second.csv")

    assert first["outer_folds"] == second["outer_folds"]


@pytest.mark.parametrize("leaked_role", ["test", "train_response_and_head"])
def test_role_leakage_in_fold_is_rejected(tmp_path: Path, leaked_role: str) -> None:
    payload, manifest, policy = _generate(tmp_path)
    leaked_id = str(manifest.loc[manifest["role"] == leaked_role, "model_id"].iloc[0])
    corrupted = copy.deepcopy(payload)
    fold = corrupted["outer_folds"][0]  # type: ignore[index]
    fold["outer_train_model_ids"][0] = leaked_id

    with pytest.raises(ValueError, match="coverage|role"):
        validate_nested_validation(corrupted, manifest, policy)


def test_duplicate_id_and_wrong_role_count_are_rejected(tmp_path: Path) -> None:
    duplicate = _manifest()
    duplicate.loc[1, "model_id"] = duplicate.loc[0, "model_id"]
    with pytest.raises(ValueError, match="duplicate model_id"):
        _generate(tmp_path, duplicate, filename="duplicate.csv")

    wrong_role = _manifest()
    wrong_role.loc[0, "role"] = "test"
    with pytest.raises(ValueError, match="role counts"):
        _generate(tmp_path, wrong_role, filename="wrong-role.csv")


def test_non_tahoe_train_head_and_stale_hash_are_rejected(tmp_path: Path) -> None:
    bad_source = _manifest()
    bad_source.loc[0, "basal_source"] = "Perturb-seq non-targeting control"
    with pytest.raises(ValueError, match="non-Tahoe"):
        _generate(tmp_path, bad_source, filename="bad-source.csv")

    path = tmp_path / "stale.csv"
    _manifest().to_csv(path, index=False)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        generate_nested_validation(path, policy=_policy("0" * 64))


def test_tampered_input_coverage_and_preprocessing_are_rejected(
    tmp_path: Path,
) -> None:
    payload, manifest, policy = _generate(tmp_path)
    wrong_coverage = copy.deepcopy(payload)
    wrong_coverage["input"]["row_count"] = 41  # type: ignore[index]
    with pytest.raises(ValueError, match="input SHA256"):
        validate_nested_validation(wrong_coverage, manifest, policy)

    wrong_preprocessing = copy.deepcopy(payload)
    preprocessing = wrong_preprocessing["outer_folds"][0]["preprocessing"]  # type: ignore[index]
    preprocessing["dmso_cells"]["fitted_bin_edges"][0] = -1.0
    with pytest.raises(ValueError, match="outer-train-only fit"):
        validate_nested_validation(wrong_preprocessing, manifest, policy)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("version", 1.5), ("seed", "17"), ("inner_fold_count", True)],
)
def test_policy_rejects_non_integer_json_values(field: str, bad_value: object) -> None:
    payload: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "version": 1,
        "seed": 17,
        "expected_manifest_sha256": "0" * 64,
        "expected_role_counts": {
            "train_head": 29,
            "train_response_and_head": 4,
            "test": 9,
        },
        "inner_fold_count": 5,
        "dmso_quantile_bins": 4,
    }
    payload[field] = bad_value

    with pytest.raises(ValueError, match="JSON integer"):
        ValidationPolicy.from_mapping(payload)

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aivc_model.tx1_p0_exposure import (
    PROTOCOL_ID,
    build_exposure_ledger,
    sha256_file,
    write_exposure_ledger,
)
from aivc_model.tx1_p0_validation import (
    ValidationPolicy,
    generate_nested_validation,
)


def _manifest_rows() -> list[dict[str, str]]:
    specifications = [
        ("ACH-T", "test", "known_present", "Tahoe-100M DMSO"),
        *[
            (f"ACH-T{index:02d}", "test", "known_present", "Tahoe-100M DMSO")
            for index in range(1, 9)
        ],
        ("ACH-H", "train_head", "verified_absent", "Tahoe-100M DMSO"),
        *[
            (
                f"ACH-H{index:02d}",
                "train_head",
                "known_present",
                "Tahoe-100M DMSO",
            )
            for index in range(1, 29)
        ],
        (
            "ACH-A",
            "train_response_and_head",
            "declared_separately",
            "Perturb-seq non-targeting control",
        ),
        *[
            (
                f"ACH-A{index:02d}",
                "train_response_and_head",
                "declared_separately",
                "Perturb-seq non-targeting control",
            )
            for index in range(1, 4)
        ],
    ]
    return [
        {
            "model_id": model_id,
            "cellosaurus_id": f"CVCL_{model_id.removeprefix('ACH-')}",
            "cell_line_name": model_id,
            "lineage": f"Lineage-{index % 5}",
            "dmso_cells": str(1000 + index),
            "role": role,
            "basal_source": basal_source,
            "tx1_pretraining_exposure": exposure,
        }
        for index, (model_id, role, exposure, basal_source) in enumerate(specifications)
    ]


def _write_manifest(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows if rows is not None else _manifest_rows()).to_csv(
        path, index=False
    )
    return path


def _write_evidence(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "evidence.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_validation_policy(tmp_path: Path, manifest_path: Path) -> Path:
    payload: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "version": 1,
        "seed": 20260804,
        "expected_manifest_sha256": sha256_file(manifest_path),
        "expected_role_counts": {
            "test": 9,
            "train_head": 29,
            "train_response_and_head": 4,
        },
        "inner_fold_count": 5,
        "dmso_quantile_bins": 4,
    }
    path = tmp_path / "validation_policy.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _read_policy(path: Path) -> ValidationPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ValidationPolicy.from_mapping(payload)


def _write_validation_plan(
    tmp_path: Path,
    manifest_path: Path,
    policy_path: Path,
    **overrides: object,
) -> Path:
    payload = generate_nested_validation(
        manifest_path, policy=_read_policy(policy_path)
    )
    payload.update(overrides)
    path = tmp_path / "validation_plan.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _build(
    tmp_path: Path,
    *,
    manifest_path: Path | None = None,
    opened_test_ids: set[str] | list[str] | None = None,
    evidence_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = manifest_path or _write_manifest(tmp_path)
    policy = _write_validation_policy(tmp_path, manifest)
    return build_exposure_ledger(
        manifest,
        validation_plan_path=_write_validation_plan(tmp_path, manifest, policy),
        validation_policy_path=policy,
        opened_test_ids=_opened_test_ids()
        if opened_test_ids is None
        else opened_test_ids,
        evidence_path=evidence_path,
    )


def _opened_test_ids() -> set[str]:
    return {row["model_id"] for row in _manifest_rows() if row["role"] == "test"}


def _evidence_row(**overrides: str) -> dict[str, str]:
    row = {
        "model_id": "ACH-A",
        "cellosaurus_id": "CVCL_A",
        "pretraining_exact_context_status": "unknown",
        "geneeffect_label_status": "label_source_present",
        "model_selection_exposure_status": "known_present",
    }
    row.update(overrides)
    return row


def test_conservative_manifest_mapping_and_roles(tmp_path: Path) -> None:
    ledger, summary = _build(tmp_path)
    indexed = ledger.set_index("model_id")

    assert indexed.loc["ACH-T", "pretraining_exact_context_status"] == "known_present"
    assert indexed.loc["ACH-H", "pretraining_exact_context_status"] == "verified_absent"
    assert indexed.loc["ACH-A", "pretraining_exact_context_status"] == "unknown"
    assert indexed.loc["ACH-T", "geneeffect_label_role"] == "opened_binding_historical"
    assert indexed.loc["ACH-H", "geneeffect_label_role"] == "development_head"
    assert (
        indexed.loc["ACH-A", "geneeffect_label_role"]
        == "development_response_and_head"
    )
    assert set(ledger["geneeffect_label_status"]) == {"label_source_present"}
    assert set(ledger["formal_eligibility"]) == {"ineligible"}
    assert summary["protocol_id"] == PROTOCOL_ID
    assert summary["formal"] is False
    assert summary["test_lines_excluded"] is True


def test_unknown_stays_unknown_without_explicit_evidence(tmp_path: Path) -> None:
    ledger, _ = _build(tmp_path)
    anchor = ledger.set_index("model_id").loc["ACH-A"]
    assert anchor["pretraining_exact_context_status"] == "unknown"


def test_explicit_evidence_may_resolve_unknown(tmp_path: Path) -> None:
    evidence = _write_evidence(
        tmp_path,
        [_evidence_row(pretraining_exact_context_status="verified_absent")],
    )
    ledger, _ = _build(tmp_path, evidence_path=evidence)
    anchor = ledger.set_index("model_id").loc["ACH-A"]
    assert anchor["pretraining_exact_context_status"] == "verified_absent"


def test_every_test_line_requires_explicit_opened_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit opened-test evidence"):
        _build(tmp_path, opened_test_ids=set())


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"protocol_id": "wrong"}, "contract metadata"),
        ({"formal": True}, "contract metadata"),
        ({"test_lines_excluded": False}, "exclude opened test lines"),
        (
            {"input": {"cell_line_manifest_sha256": "0" * 64}},
            "input SHA256",
        ),
    ],
)
def test_validation_plan_is_strictly_bound_to_manifest_and_protocol(
    tmp_path: Path, override: dict[str, object], error: str
) -> None:
    manifest = _write_manifest(tmp_path)
    policy = _write_validation_policy(tmp_path, manifest)
    validation_plan = _write_validation_plan(
        tmp_path, manifest, policy, **override
    )
    with pytest.raises(ValueError, match=error):
        build_exposure_ledger(
            manifest,
            validation_plan_path=validation_plan,
            validation_policy_path=policy,
            opened_test_ids=_opened_test_ids(),
        )


def test_policy_rejects_coherently_altered_manifest_and_plan(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    policy = _write_validation_policy(tmp_path, manifest)
    plan = _write_validation_plan(tmp_path, manifest, policy)

    altered = pd.read_csv(manifest)
    altered.loc[altered["model_id"] == "ACH-T", "cell_line_name"] = "Altered"
    altered.to_csv(manifest, index=False)
    altered_sha256 = sha256_file(manifest)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["manifest_sha256"] = altered_sha256
    plan_payload["input"]["cell_line_manifest_sha256"] = altered_sha256
    plan.write_text(json.dumps(plan_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        build_exposure_ledger(
            manifest,
            validation_plan_path=plan,
            validation_policy_path=policy,
            opened_test_ids=_opened_test_ids(),
        )


def test_opened_id_must_belong_to_test_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not role=test"):
        _build(tmp_path, opened_test_ids={"ACH-T", "ACH-H"})


@pytest.mark.parametrize(
    "field",
    [
        "cellosaurus_id",
        "cell_line_name",
        "basal_source",
        "tx1_pretraining_exposure",
    ],
)
def test_manifest_rejects_csv_empty_critical_fields(
    tmp_path: Path, field: str
) -> None:
    rows = _manifest_rows()
    anchor = next(row for row in rows if row["model_id"] == "ACH-A")
    anchor[field] = ""
    with pytest.raises(ValueError, match=rf"empty critical fields.*{field}"):
        _build(tmp_path, manifest_path=_write_manifest(tmp_path, rows))


@pytest.mark.parametrize("duplicate_field", ["model_id", "cellosaurus_id"])
def test_manifest_rejects_duplicate_identity(
    tmp_path: Path, duplicate_field: str
) -> None:
    rows = _manifest_rows()
    rows[1][duplicate_field] = rows[0][duplicate_field]
    pattern = "duplicate model_id" if duplicate_field == "model_id" else "Cellosaurus"
    with pytest.raises(ValueError, match=pattern):
        _build(tmp_path, manifest_path=_write_manifest(tmp_path, rows))


def test_evidence_rejects_unknown_column_enum_and_identity_conflict(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    bad_column = _evidence_row(extra="no")
    with pytest.raises(ValueError, match="unknown columns"):
        _build(
            tmp_path,
            manifest_path=manifest,
            evidence_path=_write_evidence(tmp_path, [bad_column]),
        )

    bad_enum = _evidence_row(pretraining_exact_context_status="probably_absent")
    with pytest.raises(ValueError, match="invalid pretraining"):
        _build(
            tmp_path,
            manifest_path=manifest,
            evidence_path=_write_evidence(tmp_path, [bad_enum]),
        )

    conflict = _evidence_row(cellosaurus_id="CVCL_WRONG")
    with pytest.raises(ValueError, match="Cellosaurus conflict"):
        _build(
            tmp_path,
            manifest_path=manifest,
            evidence_path=_write_evidence(tmp_path, [conflict]),
        )


def test_known_manifest_status_cannot_be_overridden(tmp_path: Path) -> None:
    evidence = _write_evidence(
        tmp_path,
        [
            _evidence_row(
                model_id="ACH-T",
                cellosaurus_id="CVCL_T",
                pretraining_exact_context_status="verified_absent",
                model_selection_exposure_status="unknown",
            )
        ],
    )
    with pytest.raises(ValueError, match="contradicts known pretraining"):
        _build(tmp_path, evidence_path=evidence)


def test_outputs_are_deterministic_and_summary_counts_statuses(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, list(reversed(_manifest_rows())))
    first_ledger, first_summary = _build(tmp_path, manifest_path=manifest)
    second_ledger, second_summary = _build(
        tmp_path,
        manifest_path=manifest,
        opened_test_ids=sorted(_opened_test_ids()),
    )
    first_csv, first_json = tmp_path / "first.csv", tmp_path / "first.json"
    second_csv, second_json = tmp_path / "second.csv", tmp_path / "second.json"
    write_exposure_ledger(
        first_ledger,
        first_summary,
        ledger_path=first_csv,
        summary_path=first_json,
    )
    write_exposure_ledger(
        second_ledger,
        second_summary,
        ledger_path=second_csv,
        summary_path=second_json,
    )

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    assert len(first_ledger) == 42
    assert first_ledger["model_id"].tolist() == sorted(
        first_ledger["model_id"].tolist()
    )
    parsed = json.loads(first_json.read_text())
    assert parsed["input_sha256"]["manifest"]
    assert parsed["input_sha256"]["opened_test_ids"]
    assert parsed["status_counts"]["pretraining_exact_context_status"] == {
        "known_present": 37,
        "unknown": 4,
        "verified_absent": 1,
    }

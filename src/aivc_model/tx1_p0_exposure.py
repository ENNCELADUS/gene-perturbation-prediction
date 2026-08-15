"""Build the Tx1 P0 context-exposure ledger from local evidence only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Final, Mapping

import pandas as pd

from aivc_model.tx1_p0_validation import (
    ValidationPolicy,
    load_manifest,
    validate_nested_validation,
)

PROTOCOL_ID: Final[str] = "tx1_geneeffect_p0_v1"

_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "cellosaurus_id",
    "cell_line_name",
    "role",
    "basal_source",
    "tx1_pretraining_exposure",
)
_ROLES: Final[frozenset[str]] = frozenset(
    {"test", "train_head", "train_response_and_head"}
)
PRETRAINING_STATUSES: Final[frozenset[str]] = frozenset(
    {"verified_absent", "known_present", "unknown"}
)
GENEEFFECT_LABEL_STATUSES: Final[frozenset[str]] = frozenset(
    {"label_source_present", "known_absent", "unknown"}
)
MODEL_SELECTION_STATUSES: Final[frozenset[str]] = frozenset(
    {"known_present", "known_absent", "unknown"}
)
_EVIDENCE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "model_id",
        "cellosaurus_id",
        "pretraining_exact_context_status",
        "geneeffect_label_status",
        "model_selection_exposure_status",
    }
)
_MANIFEST_PRETRAINING_MAP: Final[Mapping[str, str]] = {
    "known_present": "known_present",
    "verified_absent": "verified_absent",
    "unknown": "unknown",
    "declared_separately": "unknown",
}
_LEDGER_COLUMNS: Final[tuple[str, ...]] = (
    "protocol_id",
    "model_id",
    "cellosaurus_id",
    "name",
    "role",
    "basal_source",
    "pretraining_exact_context_status",
    "geneeffect_label_status",
    "model_selection_exposure_status",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_exposure_ledger(
    manifest_path: Path,
    *,
    validation_plan_path: Path,
    validation_policy_path: Path,
    evidence_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a deterministic exposure ledger and summary.

    The manifest is treated conservatively: ``declared_separately`` does not
    establish absence or presence and therefore maps to ``unknown``. Optional
    evidence may resolve unknown statuses, but may not contradict a known
    manifest-derived status.

    Args:
        manifest_path: Frozen Tx1 cell-line manifest CSV.
        validation_plan_path: Nested-validation JSON bound to the manifest.
        validation_policy_path: Registered authority for manifest and folds.
        evidence_path: Optional CSV containing per-line status evidence.

    Returns:
        A ModelID-sorted ledger and its machine-readable summary.

    Raises:
        ValueError: Inputs violate the fail-closed ledger contract.
    """
    policy = _read_validation_policy(validation_policy_path)
    manifest, _ = load_manifest(manifest_path, policy)
    _validate_manifest(manifest)
    _validate_validation_plan(validation_plan_path, manifest, policy)
    evidence = _read_evidence(evidence_path, manifest)

    records: list[dict[str, object]] = []
    for row in manifest.sort_values("model_id", kind="stable").itertuples(index=False):
        model_id = str(row.model_id)
        role = str(row.role)
        pretraining = _MANIFEST_PRETRAINING_MAP[str(row.tx1_pretraining_exposure)]
        label_status = "label_source_present"
        model_selection = "unknown" if role == "test" else "known_present"
        if model_id in evidence.index:
            evidence_row = evidence.loc[model_id]
            pretraining = _merge_status(
                pretraining,
                str(evidence_row["pretraining_exact_context_status"]),
                model_id=model_id,
                field="pretraining_exact_context_status",
            )
            label_status = _merge_status(
                label_status,
                str(evidence_row["geneeffect_label_status"]),
                model_id=model_id,
                field="geneeffect_label_status",
            )
            model_selection = _merge_status(
                model_selection,
                str(evidence_row["model_selection_exposure_status"]),
                model_id=model_id,
                field="model_selection_exposure_status",
            )

        records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "model_id": model_id,
                "cellosaurus_id": str(row.cellosaurus_id),
                "name": str(row.cell_line_name),
                "role": role,
                "basal_source": str(row.basal_source),
                "pretraining_exact_context_status": pretraining,
                "geneeffect_label_status": label_status,
                "model_selection_exposure_status": model_selection,
            }
        )

    ledger = pd.DataFrame.from_records(records, columns=_LEDGER_COLUMNS)
    input_sha256: dict[str, str] = {
        "manifest": sha256_file(manifest_path),
        "validation_plan": sha256_file(validation_plan_path),
        "validation_policy": sha256_file(validation_policy_path),
    }
    if evidence_path is not None:
        input_sha256["evidence"] = sha256_file(evidence_path)
    summary: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "n_lines": len(ledger),
        "n_test_lines": int((ledger["role"] == "test").sum()),
        "input_sha256": input_sha256,
        "status_counts": {
            field: dict(sorted(Counter(ledger[field].astype(str)).items()))
            for field in (
                "pretraining_exact_context_status",
                "geneeffect_label_status",
                "model_selection_exposure_status",
            )
        },
    }
    return ledger, summary


def write_exposure_ledger(
    ledger: pd.DataFrame,
    summary: Mapping[str, object],
    *,
    ledger_path: Path,
    summary_path: Path,
) -> None:
    """Write deterministic CSV and stable-key JSON outputs."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False, lineterminator="\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_manifest(manifest: pd.DataFrame) -> None:
    missing = [column for column in _MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    empty = [
        column
        for column in _MANIFEST_COLUMNS
        if manifest[column].isna().any()
        or manifest[column].astype(str).str.strip().eq("").any()
    ]
    if empty:
        raise ValueError(f"manifest has empty critical fields: {empty}")
    duplicate_ids = sorted(
        manifest.loc[manifest["model_id"].duplicated(keep=False), "model_id"].unique()
    )
    if duplicate_ids:
        raise ValueError(f"manifest has duplicate model_id values: {duplicate_ids}")
    cellosaurus_conflicts = sorted(
        manifest.loc[
            manifest["cellosaurus_id"].duplicated(keep=False), "cellosaurus_id"
        ].unique()
    )
    if cellosaurus_conflicts:
        raise ValueError(
            "manifest has Cellosaurus IDs assigned more than once: "
            f"{cellosaurus_conflicts}"
        )
    bad_roles = sorted(set(manifest["role"]) - _ROLES)
    if bad_roles:
        raise ValueError(f"manifest has invalid role values: {bad_roles}")
    bad_pretraining = sorted(
        set(manifest["tx1_pretraining_exposure"]) - set(_MANIFEST_PRETRAINING_MAP)
    )
    if bad_pretraining:
        raise ValueError(
            "manifest has invalid tx1_pretraining_exposure values: "
            f"{bad_pretraining}"
        )


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _read_validation_policy(path: Path) -> ValidationPolicy:
    payload = _read_json_object(path, label="validation policy")
    return ValidationPolicy.from_mapping(payload)


def _validate_validation_plan(
    path: Path, manifest: pd.DataFrame, policy: ValidationPolicy
) -> None:
    payload = _read_json_object(path, label="validation plan")
    validate_nested_validation(payload, manifest, policy)


def _read_evidence(path: Path | None, manifest: pd.DataFrame) -> pd.DataFrame:
    columns = list(_EVIDENCE_COLUMNS)
    if path is None:
        return pd.DataFrame(columns=columns).set_index("model_id")
    evidence = pd.read_csv(path, dtype=str, keep_default_na=False)
    unknown_columns = sorted(set(evidence.columns) - _EVIDENCE_COLUMNS)
    if unknown_columns:
        raise ValueError(f"evidence has unknown columns: {unknown_columns}")
    missing_columns = sorted(_EVIDENCE_COLUMNS - set(evidence.columns))
    if missing_columns:
        raise ValueError(f"evidence is missing required columns: {missing_columns}")
    has_empty = evidence[list(_EVIDENCE_COLUMNS)].apply(
        lambda column: column.str.strip().eq("")
    )
    if has_empty.any().any():
        raise ValueError("evidence has empty critical fields")
    duplicate_ids = sorted(
        evidence.loc[evidence["model_id"].duplicated(keep=False), "model_id"].unique()
    )
    if duplicate_ids:
        raise ValueError(f"evidence has duplicate model_id values: {duplicate_ids}")
    manifest_by_id = manifest.set_index("model_id")
    extra_ids = sorted(set(evidence["model_id"]) - set(manifest_by_id.index))
    if extra_ids:
        raise ValueError(f"evidence contains unknown model_id values: {extra_ids}")
    for row in evidence.itertuples(index=False):
        expected = str(manifest_by_id.loc[str(row.model_id), "cellosaurus_id"])
        if str(row.cellosaurus_id) != expected:
            raise ValueError(
                f"evidence Cellosaurus conflict for {row.model_id}: "
                f"expected {expected!r}, got {row.cellosaurus_id!r}"
            )
    _validate_enum(
        evidence, "pretraining_exact_context_status", PRETRAINING_STATUSES
    )
    _validate_enum(evidence, "geneeffect_label_status", GENEEFFECT_LABEL_STATUSES)
    _validate_enum(
        evidence, "model_selection_exposure_status", MODEL_SELECTION_STATUSES
    )
    return evidence.set_index("model_id")


def _validate_enum(frame: pd.DataFrame, field: str, allowed: frozenset[str]) -> None:
    bad = sorted(set(frame[field]) - allowed)
    if bad:
        raise ValueError(f"evidence has invalid {field} values: {bad}")


def _merge_status(default: str, supplied: str, *, model_id: str, field: str) -> str:
    if default == "unknown":
        return supplied
    if supplied != default:
        raise ValueError(
            f"evidence contradicts known {field} for {model_id}: "
            f"expected {default!r}, got {supplied!r}"
        )
    return default

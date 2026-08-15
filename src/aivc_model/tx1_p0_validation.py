"""Nested validation folds for Tx1 GeneEffect P0.

The outer loop is leave-one-line-out over the frozen manifest's 29 Tahoe
``train_head`` lines.  Each outer-training set is independently featurized and
partitioned into deterministic inner folds; ``test`` lines and the four
``train_response_and_head`` anchors are excluded from every fold.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from aivc_model.gene_splits import sha256_file

PROTOCOL_ID: Final[str] = "tx1_geneeffect_p0_v1"
CONTRACT: Final[str] = "nested_leave_one_line_out_context_balanced"
TRAIN_HEAD_ROLE: Final[str] = "train_head"
ANCHOR_ROLE: Final[str] = "train_response_and_head"
TEST_ROLE: Final[str] = "test"
MISSING_CONTEXT: Final[str] = "__MISSING__"

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "lineage",
    "dmso_cells",
    "basal_source",
    "role",
)
_TAHOE_SOURCE: Final[str] = "Tahoe-100M DMSO"
_EXPECTED_POLICY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "protocol_id",
        "version",
        "seed",
        "expected_manifest_sha256",
        "expected_role_counts",
        "inner_fold_count",
        "dmso_quantile_bins",
    }
)


def _require_json_int(value: object, field: str) -> int:
    """Return a JSON integer without accepting booleans, strings, or floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a JSON integer")
    return value


@dataclass(frozen=True)
class ValidationPolicy:
    """Fail-closed generation policy loaded from JSON."""

    protocol_id: str
    version: int
    seed: int
    expected_manifest_sha256: str
    expected_role_counts: Mapping[str, int]
    inner_fold_count: int
    dmso_quantile_bins: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ValidationPolicy:
        """Parse a policy mapping without accepting unknown or missing keys.

        Args:
            payload: Decoded policy JSON.

        Returns:
            Strictly validated policy.

        Raises:
            ValueError: The policy schema or a value is invalid.
        """
        keys = frozenset(payload)
        if keys != _EXPECTED_POLICY_KEYS:
            raise ValueError(
                "validation policy keys differ from contract: "
                f"missing={sorted(_EXPECTED_POLICY_KEYS - keys)}, "
                f"unknown={sorted(keys - _EXPECTED_POLICY_KEYS)}"
            )
        role_counts_raw = payload["expected_role_counts"]
        if not isinstance(role_counts_raw, Mapping):
            raise ValueError("expected_role_counts must be an object")
        expected_roles = {TRAIN_HEAD_ROLE, ANCHOR_ROLE, TEST_ROLE}
        if set(role_counts_raw) != expected_roles:
            raise ValueError(
                "expected_role_counts must contain exactly all three roles"
            )
        role_counts = {
            str(key): _require_json_int(value, f"expected_role_counts.{key}")
            for key, value in role_counts_raw.items()
        }
        policy = cls(
            protocol_id=str(payload["protocol_id"]),
            version=_require_json_int(payload["version"], "version"),
            seed=_require_json_int(payload["seed"], "seed"),
            expected_manifest_sha256=str(payload["expected_manifest_sha256"]),
            expected_role_counts=role_counts,
            inner_fold_count=_require_json_int(
                payload["inner_fold_count"], "inner_fold_count"
            ),
            dmso_quantile_bins=_require_json_int(
                payload["dmso_quantile_bins"], "dmso_quantile_bins"
            ),
        )
        if policy.protocol_id != PROTOCOL_ID:
            raise ValueError(f"protocol_id must equal {PROTOCOL_ID!r}")
        if policy.version < 1:
            raise ValueError("version must be positive")
        if len(policy.expected_manifest_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in policy.expected_manifest_sha256
        ):
            raise ValueError("expected_manifest_sha256 must be lowercase SHA256")
        if policy.expected_role_counts[TRAIN_HEAD_ROLE] != 29:
            raise ValueError("P0 requires exactly 29 train_head lines")
        if policy.expected_role_counts[ANCHOR_ROLE] != 4:
            raise ValueError("P0 requires exactly four train_response_and_head anchors")
        if policy.expected_role_counts[TEST_ROLE] != 9:
            raise ValueError("P0 requires exactly nine test lines")
        if policy.inner_fold_count < 2:
            raise ValueError("inner_fold_count must be at least two")
        if policy.dmso_quantile_bins < 2:
            raise ValueError("dmso_quantile_bins must be at least two")
        return policy


def load_manifest(path: Path, policy: ValidationPolicy) -> tuple[pd.DataFrame, str]:
    """Load the manifest only when its bytes match the registered SHA256.

    Args:
        path: Frozen ``cell_line_manifest.csv``.
        policy: P0 generation policy containing the expected byte hash.

    Returns:
        Validated manifest and its byte SHA256.

    Raises:
        ValueError: Hash, columns, identifiers, roles, or source coverage fail.
    """
    observed_sha256 = sha256_file(path)
    if observed_sha256 != policy.expected_manifest_sha256:
        raise ValueError(
            "cell line manifest SHA256 mismatch: "
            f"expected {policy.expected_manifest_sha256}, observed {observed_sha256}"
        )
    manifest = pd.read_csv(path)
    _validate_manifest(manifest, policy)
    return manifest, observed_sha256


def _validate_manifest(manifest: pd.DataFrame, policy: ValidationPolicy) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in manifest]
    if missing:
        raise ValueError(f"cell line manifest is missing columns: {missing}")
    ids = manifest["model_id"]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError("cell line manifest has missing or blank model_id")
    duplicated = sorted(ids.loc[ids.duplicated(keep=False)].astype(str).unique())
    if duplicated:
        raise ValueError(
            f"cell line manifest has duplicate model_id values: {duplicated}"
        )
    observed_counts = {
        str(key): int(value) for key, value in manifest["role"].value_counts().items()
    }
    if observed_counts != dict(policy.expected_role_counts):
        raise ValueError(
            "cell line manifest role counts differ from policy: "
            f"expected {dict(policy.expected_role_counts)}, observed {observed_counts}"
        )
    train_head = manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE]
    bad_sources = sorted(
        train_head.loc[train_head["basal_source"] != _TAHOE_SOURCE, "model_id"].astype(
            str
        )
    )
    if bad_sources:
        raise ValueError(f"train_head contains non-Tahoe lines: {bad_sources}")


def _stable_tie(seed: int, *parts: object) -> str:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fit_context(
    outer_train: pd.DataFrame, requested_bins: int
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    ordered = outer_train.sort_values("model_id").reset_index(drop=True)
    lineage_by_id: dict[str, str] = {}
    log_dmso_by_id: dict[str, float | None] = {}
    for row in ordered.itertuples(index=False):
        model_id = str(row.model_id)
        lineage = MISSING_CONTEXT if pd.isna(row.lineage) else str(row.lineage).strip()
        lineage_by_id[model_id] = lineage or MISSING_CONTEXT
        if pd.isna(row.dmso_cells):
            log_dmso_by_id[model_id] = None
            continue
        dmso_cells = float(row.dmso_cells)
        if not math.isfinite(dmso_cells) or dmso_cells < 0:
            raise ValueError(
                f"line {model_id} has invalid dmso_cells={row.dmso_cells!r}"
            )
        log_dmso_by_id[model_id] = float(math.log1p(dmso_cells))

    nonmissing = sorted(value for value in log_dmso_by_id.values() if value is not None)
    if not nonmissing:
        edges: list[float] = []
    else:
        bin_count = min(requested_bins, len(set(nonmissing)))
        quantiles = np.linspace(0.0, 1.0, bin_count + 1)
        edges = [
            float(value) for value in np.unique(np.quantile(nonmissing, quantiles))
        ]

    context_by_id: dict[str, dict[str, object]] = {}
    for model_id in sorted(lineage_by_id):
        value = log_dmso_by_id[model_id]
        if value is None:
            dmso_bin = MISSING_CONTEXT
        else:
            interior = np.asarray(edges[1:-1], dtype=np.float64)
            dmso_bin = f"bin_{int(np.searchsorted(interior, value, side='right'))}"
        context_by_id[model_id] = {
            "lineage": lineage_by_id[model_id],
            "log_dmso_cells": value,
            "dmso_bin": dmso_bin,
        }

    metadata: dict[str, object] = {
        "fit_model_ids": sorted(lineage_by_id),
        "lineage": {
            "categories": sorted(set(lineage_by_id.values())),
            "missing_value": MISSING_CONTEXT,
        },
        "dmso_cells": {
            "transform": "log1p",
            "binning": "outer_train_quantiles",
            "requested_bins": requested_bins,
            "fitted_bin_edges": edges,
            "missing_value": MISSING_CONTEXT,
        },
        "context_by_model_id": context_by_id,
    }
    return metadata, context_by_id


def _build_inner_folds(
    outer_train_ids: Sequence[str],
    context_by_id: Mapping[str, Mapping[str, object]],
    *,
    fold_count: int,
    seed: int,
    outer_validation_id: str,
) -> list[dict[str, object]]:
    lineage_frequency = Counter(
        str(context_by_id[mid]["lineage"]) for mid in outer_train_ids
    )
    dmso_frequency = Counter(
        str(context_by_id[mid]["dmso_bin"]) for mid in outer_train_ids
    )
    ordered_ids = sorted(
        outer_train_ids,
        key=lambda mid: (
            -lineage_frequency[str(context_by_id[mid]["lineage"])],
            -dmso_frequency[str(context_by_id[mid]["dmso_bin"])],
            _stable_tie(seed, outer_validation_id, mid),
            mid,
        ),
    )
    assignments: list[list[str]] = [[] for _ in range(fold_count)]
    lineage_counts: list[Counter[str]] = [Counter() for _ in range(fold_count)]
    dmso_counts: list[Counter[str]] = [Counter() for _ in range(fold_count)]
    for model_id in ordered_ids:
        lineage = str(context_by_id[model_id]["lineage"])
        dmso_bin = str(context_by_id[model_id]["dmso_bin"])
        fold_index = min(
            range(fold_count),
            key=lambda index: (
                len(assignments[index]),
                lineage_counts[index][lineage],
                dmso_counts[index][dmso_bin],
                _stable_tie(seed, outer_validation_id, model_id, index),
            ),
        )
        assignments[fold_index].append(model_id)
        lineage_counts[fold_index][lineage] += 1
        dmso_counts[fold_index][dmso_bin] += 1

    universe = set(outer_train_ids)
    return [
        {
            "inner_fold_index": index,
            "train_model_ids": sorted(universe - set(validation_ids)),
            "validation_model_ids": sorted(validation_ids),
        }
        for index, validation_ids in enumerate(assignments)
    ]


def generate_nested_validation(
    manifest_path: Path, *, policy: ValidationPolicy
) -> dict[str, object]:
    """Generate and self-validate deterministic P0 nested folds.

    Args:
        manifest_path: Frozen cell-line manifest whose bytes must match policy.
        policy: Strict P0 generation policy.

    Returns:
        JSON-serializable nested validation contract.

    Raises:
        ValueError: Any role, coverage, preprocessing, or fold invariant fails.
    """
    manifest, manifest_sha256 = load_manifest(manifest_path, policy)
    train_head = manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE].copy()
    train_head["model_id"] = train_head["model_id"].astype(str)
    eligible_ids = sorted(train_head["model_id"])
    outer_folds: list[dict[str, object]] = []
    for outer_index, held_out_id in enumerate(eligible_ids):
        outer_train = train_head.loc[train_head["model_id"] != held_out_id].copy()
        outer_train_ids = sorted(outer_train["model_id"].astype(str))
        preprocessing, context_by_id = _fit_context(
            outer_train, policy.dmso_quantile_bins
        )
        outer_folds.append(
            {
                "outer_fold_index": outer_index,
                "held_out": held_out_id,
                "outer_train": outer_train_ids,
                "outer_train_model_ids": outer_train_ids,
                "outer_validation_model_ids": [held_out_id],
                "preprocessing": preprocessing,
                "inner_folds": _build_inner_folds(
                    outer_train_ids,
                    context_by_id,
                    fold_count=policy.inner_fold_count,
                    seed=policy.seed,
                    outer_validation_id=held_out_id,
                ),
            }
        )
    payload: dict[str, object] = {
        "protocol_id": policy.protocol_id,
        "contract": CONTRACT,
        "version": policy.version,
        "seed": policy.seed,
        "manifest_sha256": manifest_sha256,
        "excluded_roles": [ANCHOR_ROLE, TEST_ROLE],
        "input": {
            "cell_line_manifest_sha256": manifest_sha256,
            "row_count": len(manifest),
            "role_counts": dict(sorted(policy.expected_role_counts.items())),
        },
        "outer_folds": outer_folds,
    }
    validate_nested_validation(payload, manifest, policy)
    return payload


def validate_nested_validation(
    payload: Mapping[str, object], manifest: pd.DataFrame, policy: ValidationPolicy
) -> None:
    """Fail closed on leakage, stale input, or incomplete nested coverage.

    Args:
        payload: Generated nested validation mapping.
        manifest: Manifest against which coverage is checked.
        policy: Strict P0 policy.

    Raises:
        ValueError: Any contract or coverage invariant fails.
    """
    _validate_manifest(manifest, policy)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("contract") != CONTRACT
        or payload.get("version") != policy.version
        or payload.get("seed") != policy.seed
        or payload.get("manifest_sha256") != policy.expected_manifest_sha256
    ):
        raise ValueError("nested validation contract metadata is invalid")
    if payload.get("excluded_roles") != [ANCHOR_ROLE, TEST_ROLE]:
        raise ValueError("nested validation excluded-role declaration is invalid")
    input_metadata = payload.get("input")
    expected_role_counts = dict(sorted(policy.expected_role_counts.items()))
    if (
        not isinstance(input_metadata, Mapping)
        or input_metadata.get("cell_line_manifest_sha256")
        != policy.expected_manifest_sha256
        or input_metadata.get("row_count") != len(manifest)
        or input_metadata.get("role_counts") != expected_role_counts
    ):
        raise ValueError("nested validation input SHA256 is missing or stale")
    eligible = set(
        manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE, "model_id"].astype(str)
    )
    forbidden = set(
        manifest.loc[
            manifest["role"].isin({ANCHOR_ROLE, TEST_ROLE}), "model_id"
        ].astype(str)
    )
    outer_folds = payload.get("outer_folds")
    if not isinstance(outer_folds, list) or len(outer_folds) != len(eligible):
        raise ValueError("outer fold count does not cover all train_head lines")
    outer_validation_seen: list[str] = []
    manifest_by_id = manifest.assign(
        model_id=manifest["model_id"].astype(str)
    ).set_index("model_id", drop=False)
    ordered_eligible = sorted(eligible)
    for expected_outer_index, fold in enumerate(outer_folds):
        if not isinstance(fold, Mapping):
            raise ValueError("outer fold must be an object")
        if fold.get("outer_fold_index") != expected_outer_index:
            raise ValueError("outer fold indices are incomplete or out of order")
        outer_train = _string_set(fold.get("outer_train_model_ids"), "outer train")
        outer_validation = _string_set(
            fold.get("outer_validation_model_ids"), "outer validation"
        )
        if len(outer_validation) != 1 or outer_train != eligible - outer_validation:
            raise ValueError("outer fold train/validation coverage is invalid")
        if fold.get("held_out") != next(iter(outer_validation)):
            raise ValueError("outer held_out alias differs from validation model ID")
        if fold.get("outer_train") != fold.get("outer_train_model_ids"):
            raise ValueError("outer_train alias differs from outer train model IDs")
        if outer_validation != {ordered_eligible[expected_outer_index]}:
            raise ValueError("outer folds are not in deterministic model_id order")
        if (outer_train | outer_validation) & forbidden:
            raise ValueError("test or anchor role entered an outer fold")
        outer_validation_seen.extend(outer_validation)
        preprocessing = fold.get("preprocessing")
        if not isinstance(preprocessing, Mapping):
            raise ValueError("outer fold preprocessing metadata is missing")
        fit_ids = _string_set(preprocessing.get("fit_model_ids"), "preprocessing fit")
        context = preprocessing.get("context_by_model_id")
        if (
            fit_ids != outer_train
            or not isinstance(context, Mapping)
            or set(context) != outer_train
        ):
            raise ValueError("preprocessing was not fitted exclusively on outer train")
        outer_train_frame = manifest_by_id.loc[sorted(outer_train)].reset_index(
            drop=True
        )
        expected_preprocessing, expected_context = _fit_context(
            outer_train_frame, policy.dmso_quantile_bins
        )
        if dict(preprocessing) != expected_preprocessing:
            raise ValueError("preprocessing metadata differs from outer-train-only fit")
        inner_folds = fold.get("inner_folds")
        if (
            not isinstance(inner_folds, list)
            or len(inner_folds) != policy.inner_fold_count
        ):
            raise ValueError("inner fold count differs from policy")
        inner_validation_seen: list[str] = []
        for inner in inner_folds:
            if not isinstance(inner, Mapping):
                raise ValueError("inner fold must be an object")
            inner_train = _string_set(inner.get("train_model_ids"), "inner train")
            inner_validation = _string_set(
                inner.get("validation_model_ids"), "inner validation"
            )
            if not inner_validation or inner_train != outer_train - inner_validation:
                raise ValueError("inner fold train/validation coverage is invalid")
            if (inner_train | inner_validation) & forbidden:
                raise ValueError("test or anchor role entered an inner fold")
            inner_validation_seen.extend(inner_validation)
        if (
            len(inner_validation_seen) != len(outer_train)
            or set(inner_validation_seen) != outer_train
        ):
            raise ValueError(
                "inner validation folds do not partition outer train exactly once"
            )
        expected_inner_folds = _build_inner_folds(
            sorted(outer_train),
            expected_context,
            fold_count=policy.inner_fold_count,
            seed=policy.seed,
            outer_validation_id=next(iter(outer_validation)),
        )
        if inner_folds != expected_inner_folds:
            raise ValueError("inner folds differ from deterministic context balance")
    if (
        len(outer_validation_seen) != len(eligible)
        or set(outer_validation_seen) != eligible
    ):
        raise ValueError(
            "outer validation folds do not cover each train_head line exactly once"
        )


def _string_set(value: object, label: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} model IDs must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} model IDs contain duplicates")
    return set(value)

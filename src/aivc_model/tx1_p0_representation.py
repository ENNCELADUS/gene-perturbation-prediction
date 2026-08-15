"""P0 outer-LOO audit of precomputed cell-line representations.
The audit never accesses the Tx1 cache and is not synthetic-lethality evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_geneeffect_eval import load_slice, verify_artifact_hashes
from aivc_model.tx1_p0_validation import (
    ValidationPolicy,
    load_manifest as load_registered_manifest,
    validate_nested_validation,
)

PROTOCOL_ID: Final[str] = "tx1_geneeffect_p0_v1"
TRAIN_HEAD_ROLE: Final[str] = "train_head"
TAHOE_SOURCE: Final[str] = "Tahoe-100M DMSO"
EXPECTED_TRAIN_HEAD_LINES: Final[int] = 29
EXPECTED_GENES: Final[int] = 587
_MANIFEST_ROLES: Final[frozenset[str]] = frozenset(
    {"train_head", "train_response_and_head", "test"}
)


@dataclass(frozen=True)
class OuterFoldPredictions:
    """Predictions fitted without access to the held line's labels."""

    nearest_neighbor: np.ndarray
    ridge: np.ndarray
    shuffled_ridge: np.ndarray
    nearest_neighbor_index: int
    pca_components: int
    dropped_constant_feature_count: int


@dataclass(frozen=True)
class OuterFold:
    """One manifest-bound fold from the P0 nested validation plan."""

    index: int
    train_model_ids: tuple[str, ...]
    held_model_id: str


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_train_head_manifest(
    path: Path,
    *,
    expected_lines: int = EXPECTED_TRAIN_HEAD_LINES,
) -> pd.DataFrame:
    """Load and strictly select the Tahoe ``train_head`` audit population.

    Args:
        path: Frozen-manifest-shaped CSV.
        expected_lines: Required number of selected lines.

    Returns:
        Selected rows sorted by ``model_id``.

    Raises:
        ValueError: The manifest has invalid roles, IDs, source, or count.
    """
    frame = pd.read_csv(path, dtype={"model_id": str, "role": str})
    _require_columns(frame, {"model_id", "role", "basal_source"}, "manifest")
    if frame["model_id"].isna().any() or frame["model_id"].duplicated().any():
        raise ValueError("manifest model_id values must be non-missing and unique")
    invalid_roles = sorted(set(frame["role"]) - _MANIFEST_ROLES)
    if invalid_roles:
        raise ValueError(f"manifest has invalid role values: {invalid_roles}")
    selected = frame.loc[frame["role"] == TRAIN_HEAD_ROLE].copy()
    if len(selected) != expected_lines:
        raise ValueError(
            f"manifest must contain exactly {expected_lines} train_head lines; "
            f"found {len(selected)}"
        )
    invalid_sources = selected.loc[
        selected["basal_source"] != TAHOE_SOURCE, ["model_id", "basal_source"]
    ]
    if not invalid_sources.empty:
        raise ValueError(
            "train_head audit population must be Tahoe-only; invalid rows: "
            f"{invalid_sources.to_dict(orient='records')}"
        )
    return selected.sort_values("model_id").reset_index(drop=True)


def load_validation_plan(
    path: Path,
    registered_manifest: pd.DataFrame,
    policy: ValidationPolicy,
) -> tuple[OuterFold, ...]:
    """Load a manifest-bound P0 plan and validate its outer LOO folds.

    Args:
        path: Stable JSON emitted by ``build_tx1_p0_validation.py``.
        registered_manifest: Full manifest validated against ``policy``.
        policy: Registered P0 validation policy.

    Returns:
        Outer folds in their registered order.

    Raises:
        ValueError: Metadata, manifest binding, or fold coverage is invalid.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("validation plan root must be a JSON object")
    validate_nested_validation(payload, registered_manifest, policy)
    raw_folds = payload["outer_folds"]
    folds: list[OuterFold] = []
    for raw_fold in raw_folds:
        folds.append(
            OuterFold(
                index=int(raw_fold["outer_fold_index"]),
                train_model_ids=tuple(raw_fold["outer_train_model_ids"]),
                held_model_id=str(raw_fold["outer_validation_model_ids"][0]),
            )
        )
    return tuple(folds)


def load_phase_a_gene_universe(phase_a_dir: Path) -> tuple[str, ...]:
    """Verify Phase A and return its exact frozen 587-gene symbol universe."""
    verify_artifact_hashes(phase_a_dir)
    frozen_slice = load_slice(phase_a_dir / "differentially_essential_slice.csv")
    _require_columns(frozen_slice, {"gene_symbol"}, "frozen Phase-A slice")
    symbols = frozen_slice["gene_symbol"]
    if (
        symbols.isna().any()
        or symbols.duplicated().any()
        or len(symbols) != EXPECTED_GENES
    ):
        raise ValueError(
            "frozen Phase-A slice must contain 587 unique gene_symbol rows"
        )
    return tuple(sorted(symbols.astype(str)))


def load_representation(path: Path, model_ids: Sequence[str]) -> pd.DataFrame:
    """Load one finite, complete line-representation matrix.

    Args:
        path: CSV with ``model_id`` and numeric feature columns.
        model_ids: Exact ordered audit population.

    Returns:
        Numeric features indexed in ``model_ids`` order.

    Raises:
        ValueError: IDs or feature values violate the closed input contract.
    """
    frame = pd.read_csv(path, dtype={"model_id": str})
    _require_columns(frame, {"model_id"}, "representation")
    if frame["model_id"].isna().any() or frame["model_id"].duplicated().any():
        raise ValueError(
            "representation model_id values must be non-missing and unique"
        )
    wanted = [str(value) for value in model_ids]
    observed = set(frame["model_id"])
    missing = sorted(set(wanted) - observed)
    extra = sorted(observed - set(wanted))
    if missing or extra:
        raise ValueError(
            "representation IDs must exactly equal the train_head population; "
            f"missing={missing}, extra={extra}"
        )
    feature_columns = [column for column in frame.columns if column != "model_id"]
    if not feature_columns:
        raise ValueError("representation must contain at least one feature column")
    features = frame.set_index("model_id").loc[wanted, feature_columns]
    numeric = features.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("representation features must all be finite numeric values")
    return numeric


def load_gene_effect(
    path: Path,
    model_ids: Sequence[str],
    *,
    expected_genes: int = EXPECTED_GENES,
) -> pd.DataFrame:
    """Load a complete long-form GeneEffect matrix with one shared universe.

    Args:
        path: CSV with ``model_id``, ``gene_symbol``, and ``gene_effect``.
        model_ids: Exact ordered audit population.
        expected_genes: Required gene-universe size.

    Returns:
        Finite line-by-gene matrix.

    Raises:
        ValueError: Keys, values, coverage, or universes are invalid.
    """
    frame = pd.read_csv(path, dtype={"model_id": str, "gene_symbol": str})
    required = {"model_id", "gene_symbol", "gene_effect"}
    _require_columns(frame, required, "GeneEffect")
    if frame[list(required)].isna().any().any():
        raise ValueError("GeneEffect keys and values must be non-missing")
    if frame.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError("GeneEffect has duplicate (model_id, gene_symbol) keys")
    wanted = [str(value) for value in model_ids]
    observed_ids = set(frame["model_id"])
    missing_ids = sorted(set(wanted) - observed_ids)
    extra_ids = sorted(observed_ids - set(wanted))
    if missing_ids or extra_ids:
        raise ValueError(
            "GeneEffect IDs must exactly equal the train_head population; "
            f"missing={missing_ids}, extra={extra_ids}"
        )
    universes = {
        model_id: frozenset(rows["gene_symbol"])
        for model_id, rows in frame.groupby("model_id", sort=False)
    }
    reference = universes[wanted[0]]
    if len(reference) != expected_genes:
        raise ValueError(
            f"GeneEffect must contain exactly {expected_genes} genes per line; "
            f"found {len(reference)}"
        )
    mismatched = sorted(
        model_id for model_id, universe in universes.items() if universe != reference
    )
    if mismatched:
        raise ValueError(
            "GeneEffect gene universe must be identical for every line; "
            f"mismatched={mismatched}"
        )
    numeric = pd.to_numeric(frame["gene_effect"], errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("GeneEffect values must all be finite numeric values")
    normalized = frame.assign(gene_effect=numeric)
    matrix = normalized.pivot(
        index="model_id", columns="gene_symbol", values="gene_effect"
    )
    return matrix.loc[wanted, sorted(reference)]


def load_shared_prior(path: Path, genes: Sequence[str]) -> np.ndarray:
    """Load a finite, exactly gene-matched shared prior profile."""
    frame = pd.read_csv(path, dtype={"gene_symbol": str})
    _require_columns(frame, {"gene_symbol", "gene_effect"}, "shared prior")
    if frame["gene_symbol"].isna().any() or frame["gene_symbol"].duplicated().any():
        raise ValueError("shared prior gene IDs must be non-missing and unique")
    wanted = [str(value) for value in genes]
    observed = set(frame["gene_symbol"])
    missing = sorted(set(wanted) - observed)
    extra = sorted(observed - set(wanted))
    if missing or extra:
        raise ValueError(
            f"shared prior gene universe mismatch; missing={missing}, extra={extra}"
        )
    prior = pd.to_numeric(
        frame.set_index("gene_symbol").loc[wanted, "gene_effect"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(prior).all():
        raise ValueError("shared prior values must all be finite numeric values")
    return prior


def fit_outer_fold(
    train_context: np.ndarray,
    train_gene_effect: np.ndarray,
    held_context: np.ndarray,
    prior: np.ndarray,
    *,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
    shuffle_seed: int = 0,
) -> OuterFoldPredictions:
    """Fit one strict outer fold; held-line labels are not an argument.

    Args:
        train_context: Outer-training representation matrix.
        train_gene_effect: Outer-training line-by-gene labels.
        held_context: One held line's representation.
        prior: Shared or outer-training-only gene profile.
        pca_components: Requested PCA width, capped to legal dimensions.
        ridge_alpha: Ridge regularization strength.
        shuffle_seed: Deterministic negative-control permutation seed.

    Returns:
        Nearest-neighbor, ridge, and shuffled-context predictions.
    """
    if pca_components < 1:
        raise ValueError("pca_components must be positive")
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("ridge_alpha must be finite and non-negative")
    if len(train_context) < 2:
        raise ValueError("outer training requires at least two source lines")
    keep = np.var(train_context, axis=0) > 0.0
    if not np.any(keep):
        raise ValueError("outer training representation has no non-constant features")
    train_context = train_context[:, keep]
    held_context = np.asarray(held_context)[keep]
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_context)
    scaled_held = scaler.transform(np.asarray(held_context).reshape(1, -1))
    width = min(pca_components, scaled_train.shape[0] - 1, scaled_train.shape[1])
    pca = PCA(n_components=width, svd_solver="full")
    reduced_train = pca.fit_transform(scaled_train)
    reduced_held = pca.transform(scaled_held)
    residuals = train_gene_effect - prior.reshape(1, -1)

    distances = np.linalg.norm(reduced_train - reduced_held, axis=1)
    nearest_index = int(np.argmin(distances))
    nearest = prior + residuals[nearest_index]

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(reduced_train, residuals)
    ridge_prediction = prior + np.asarray(ridge.predict(reduced_held)[0], dtype=float)

    permutation = np.random.default_rng(shuffle_seed).permutation(len(residuals))
    shuffled = Ridge(alpha=ridge_alpha)
    shuffled.fit(reduced_train, residuals[permutation])
    shuffled_prediction = prior + np.asarray(
        shuffled.predict(reduced_held)[0], dtype=float
    )
    return OuterFoldPredictions(
        nearest_neighbor=np.asarray(nearest, dtype=float),
        ridge=ridge_prediction,
        shuffled_ridge=shuffled_prediction,
        nearest_neighbor_index=nearest_index,
        pca_components=width,
        dropped_constant_feature_count=int(np.count_nonzero(~keep)),
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("Spearman correlation is undefined for a constant profile")
    return value


def _method_summary(
    rows: Sequence[Mapping[str, object]], prefix: str
) -> dict[str, float]:
    rhos = np.asarray([float(row[f"{prefix}_rho"]) for row in rows])
    deltas = np.asarray([float(row[f"{prefix}_delta_rho"]) for row in rows])
    tail_count = max(1, int(np.ceil(0.25 * len(deltas))))
    return {
        "macro_rho": float(np.mean(rhos)),
        "macro_delta_rho": float(np.mean(deltas)),
        "negative_transfer_rate": float(np.mean(deltas < 0.0)),
        "cvar25_delta_rho": float(np.mean(np.sort(deltas)[:tail_count])),
    }


def audit_representation(
    features: pd.DataFrame,
    gene_effect: pd.DataFrame,
    *,
    outer_folds: Sequence[OuterFold] | None = None,
    shared_prior: np.ndarray | None = None,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
    shuffle_seed: int = 20260804,
) -> dict[str, object]:
    """Run strict outer-LOO transfer and label-associated geometry analysis."""
    if list(features.index) != list(gene_effect.index):
        raise ValueError("representation and GeneEffect model_id order must match")
    x = features.to_numpy(dtype=float)
    y = gene_effect.to_numpy(dtype=float)
    folds = (
        tuple(outer_folds)
        if outer_folds is not None
        else tuple(
            OuterFold(
                index=index,
                train_model_ids=tuple(
                    str(value) for value in features.index if value != model_id
                ),
                held_model_id=str(model_id),
            )
            for index, model_id in enumerate(features.index)
        )
    )
    if {fold.held_model_id for fold in folds} != set(features.index) or len(
        folds
    ) != len(features):
        raise ValueError("outer folds do not exactly cover representation IDs")
    rows: list[dict[str, object]] = []
    for fold in folds:
        model_id = fold.held_model_id
        held_index = int(features.index.get_loc(model_id))
        train_indices = np.asarray(
            [
                int(features.index.get_loc(source_id))
                for source_id in fold.train_model_ids
            ]
        )
        if set(fold.train_model_ids) != set(features.index) - {model_id}:
            raise ValueError(f"outer fold {fold.index} has invalid training coverage")
        fold_prior = (
            np.asarray(shared_prior, dtype=float)
            if shared_prior is not None
            else np.mean(y[train_indices], axis=0)
        )
        predictions = fit_outer_fold(
            x[train_indices],
            y[train_indices],
            x[held_index],
            fold_prior,
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
            shuffle_seed=shuffle_seed + fold.index,
        )
        baseline_rho = _spearman(fold_prior, y[held_index])
        train_ids = np.asarray(fold.train_model_ids, dtype=object)
        dropped_count = predictions.dropped_constant_feature_count
        row: dict[str, object] = {
            "model_id": str(model_id),
            "k": 0,
            "baseline_rho": baseline_rho,
            "nearest_neighbor_model_id": str(
                train_ids[predictions.nearest_neighbor_index]
            ),
            "pca_components": predictions.pca_components,
            "dropped_constant_feature_count": dropped_count,
        }
        for prefix, prediction in (
            ("nearest_neighbor", predictions.nearest_neighbor),
            ("ridge", predictions.ridge),
            ("shuffled_ridge", predictions.shuffled_ridge),
        ):
            rho = _spearman(prediction, y[held_index])
            row[f"{prefix}_rho"] = rho
            row[f"{prefix}_delta_rho"] = rho - baseline_rho
        rows.append(row)

    summaries = {
        prefix: _method_summary(rows, prefix)
        for prefix in ("nearest_neighbor", "ridge", "shuffled_ridge")
    }
    actual_gain = summaries["ridge"]["macro_delta_rho"]
    shuffled_gain = summaries["shuffled_ridge"]["macro_delta_rho"]
    retained_ratio: float | None
    retained_reason: str | None
    if actual_gain > 0.0:
        retained_ratio = shuffled_gain / actual_gain
        retained_reason = None
    else:
        retained_ratio = None
        retained_reason = "actual_macro_delta_rho_not_positive"

    representation_distances = pdist(x, metric="euclidean")
    gene_effect_distances = pdist(y, metric="euclidean")
    geometry_rho = _spearman(representation_distances, gene_effect_distances)
    return {
        "per_line": rows,
        "summary": summaries,
        "negative_control": {
            "shuffle_seed": shuffle_seed,
            "shuffled_macro_delta_rho": shuffled_gain,
            "retained_gain_ratio": retained_ratio,
            "retained_gain_ratio_reason": retained_reason,
        },
        "geometry_analysis": {
            "representation_space": "raw",
            "label_distance_space": (
                "GeneEffect residual profiles; pairwise distances are invariant "
                "to the shared prior"
            ),
            "pair_scope": "Tahoe train_head source-only pairs",
            "pair_count": int(len(representation_distances)),
            "distance_spearman": geometry_rho,
        },
    }


def run_audit(
    manifest_path: Path,
    validation_plan_path: Path,
    validation_policy_path: Path,
    phase_a_dir: Path,
    representation_paths: Mapping[str, Path],
    gene_effect_path: Path,
    *,
    shared_prior_path: Path | None = None,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
    shuffle_seed: int = 20260804,
    expected_lines: int = EXPECTED_TRAIN_HEAD_LINES,
    expected_genes: int = EXPECTED_GENES,
) -> dict[str, object]:
    """Load validated files and run every named representation audit."""
    if not representation_paths:
        raise ValueError("at least one representation is required")
    policy_payload = json.loads(validation_policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy_payload, Mapping):
        raise ValueError("validation policy root must be a JSON object")
    policy = ValidationPolicy.from_mapping(policy_payload)
    registered_manifest, _ = load_registered_manifest(manifest_path, policy)
    manifest = load_train_head_manifest(manifest_path, expected_lines=expected_lines)
    frozen_genes = load_phase_a_gene_universe(phase_a_dir)
    outer_folds = load_validation_plan(
        validation_plan_path, registered_manifest, policy
    )
    model_ids = manifest["model_id"].astype(str).tolist()
    labels = load_gene_effect(
        gene_effect_path, model_ids, expected_genes=expected_genes
    )
    if tuple(labels.columns) != frozen_genes:
        raise ValueError("GeneEffect gene universe differs from frozen Phase-A slice")
    prior = (
        load_shared_prior(shared_prior_path, labels.columns)
        if shared_prior_path is not None
        else None
    )
    results: dict[str, object] = {}
    for name in sorted(representation_paths):
        if not name:
            raise ValueError("representation names must be non-empty")
        features = load_representation(representation_paths[name], model_ids)
        results[name] = audit_representation(
            features,
            labels,
            outer_folds=outer_folds,
            shared_prior=prior,
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
            shuffle_seed=shuffle_seed,
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "anchors_excluded": True,
        "outer_validation": "strict leave-one-source-line-out",
        "target_role": TRAIN_HEAD_ROLE,
        "metadata": {
            "audit_population": "Tahoe role=train_head only",
            "line_count": len(model_ids),
            "gene_count": len(labels.columns),
            "prior": "shared_file" if prior is not None else "outer_train_gene_mean",
            "representation_input": "precomputed feature CSVs only",
            "tx1_frozen_cache_accessed": False,
            "tx1_layer_extraction_performed": False,
            "phase_a_contract_modified": False,
            "representation_fit_provenance_verified": False,
            "shared_prior_fit_provenance_verified": prior is None,
            "provenance_limit": (
                "CSV content cannot prove that upstream representation or prior "
                "construction excluded held labels"
            ),
        },
        "config": {
            "pca_components_requested": pca_components,
            "ridge_alpha": ridge_alpha,
            "shuffle_seed": shuffle_seed,
        },
        "input_sha256": {
            "manifest": sha256_file(manifest_path),
            "validation_plan": sha256_file(validation_plan_path),
            "validation_policy": sha256_file(validation_policy_path),
            "phase_a_registration": sha256_file(
                phase_a_dir / "phase_a_registration.json"
            ),
            "phase_a_slice": sha256_file(
                phase_a_dir / "differentially_essential_slice.csv"
            ),
            "gene_effect": sha256_file(gene_effect_path),
            "shared_prior": (
                sha256_file(shared_prior_path)
                if shared_prior_path is not None
                else None
            ),
            "representations": {
                name: sha256_file(path)
                for name, path in sorted(representation_paths.items())
            },
        },
        "representations": results,
    }

"""Development-only, Tahoe-only source-line OOF GeneEffect baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from aivc_model.tx1_p0_validation import (
    CONTRACT as VALIDATION_CONTRACT,
    ValidationPolicy,
    load_manifest,
    validate_nested_validation,
)
from aivc_model.tx1_geneeffect_eval import verify_artifact_hashes

PROTOCOL_ID: Final[str] = "tx1_geneeffect_p0_v1"
TRAIN_HEAD_ROLE: Final[str] = "train_head"
TEST_ROLE: Final[str] = "test"

GENE_MEAN_METHOD: Final[str] = "train_line_gene_mean"
LINEAGE_METHOD: Final[str] = "lineage_shrunk_mean"
NEAREST_METHOD: Final[str] = "nearest_source_line"
PCA_RIDGE_METHOD: Final[str] = "context_pca_ridge"
COPY_K562_METHOD: Final[str] = "copy_k562"

_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "gene_symbol",
    "gene_effect",
)


@dataclass(frozen=True)
class P0BaselineConfig:
    """Fixed hyperparameters for the development-only baseline ladder."""

    lineage_alpha: float = 0.5
    pca_components: int = 8
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lineage_alpha <= 1.0:
            raise ValueError("lineage_alpha must be in [0, 1]")
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0.0:
            raise ValueError("ridge_alpha must be finite and > 0")


@dataclass(frozen=True)
class P0BaselineResult:
    """In-memory artifacts produced by :func:`run_p0_baseline_ladder`."""

    per_prediction: pd.DataFrame
    per_line: pd.DataFrame
    summary: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _load_validation_plan(
    path: Path, manifest_path: Path, policy_path: Path
) -> tuple[pd.DataFrame, list[Mapping[str, object]], str]:
    """Load and fully validate a shared P0 nested-validation plan.

    The policy is an external authority and is never reconstructed from the
    plan, because a manifest and plan could otherwise be coherently tampered.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("validation plan root must be a JSON object")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("contract") != VALIDATION_CONTRACT
        or payload.get("formal") is not False
        or payload.get("development_only") is not True
        or payload.get("test_lines_excluded") is not True
    ):
        raise ValueError("validation plan metadata is incompatible with P0")
    outer_folds = payload.get("outer_folds")
    if not isinstance(outer_folds, list) or not outer_folds:
        raise ValueError("validation plan must contain outer folds")
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy_payload, dict):
        raise ValueError("validation policy root must be a JSON object")
    policy = ValidationPolicy.from_mapping(policy_payload)
    manifest, manifest_sha256 = load_manifest(manifest_path, policy)
    validate_nested_validation(payload, manifest, policy)
    return manifest, outer_folds, manifest_sha256


def _load_frozen_slice(phase_a_dir: Path) -> tuple[list[str], str, str]:
    """Verify Phase A against its out-of-band anchor and load 587 genes."""
    verify_artifact_hashes(phase_a_dir)
    registration = phase_a_dir / "phase_a_registration.json"
    slice_path = phase_a_dir / "differentially_essential_slice.csv"
    frame = pd.read_csv(slice_path)
    _require_columns(frame, ("gene_symbol",), "differential slice")
    symbols = frame["gene_symbol"]
    if symbols.isna().any() or (symbols.astype(str).str.strip() == "").any():
        raise ValueError("differential slice has missing or blank gene_symbol")
    genes = symbols.astype(str).tolist()
    if len(genes) != 587 or len(set(genes)) != 587:
        raise ValueError("differential slice must contain exactly 587 unique genes")
    return sorted(genes), _sha256_file(registration), _sha256_file(slice_path)


def _load_labels(path: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_csv(path)
    _require_columns(labels, _LABEL_COLUMNS, "GeneEffect CSV")
    labels = labels.loc[:, list(_LABEL_COLUMNS)].copy()
    for column in ("model_id", "gene_symbol"):
        if (
            labels[column].isna().any()
            or (labels[column].astype(str).str.strip() == "").any()
        ):
            raise ValueError(f"GeneEffect CSV has missing or empty {column} values")
        labels[column] = labels[column].astype(str)
    if labels.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError("GeneEffect CSV has duplicate (model_id, gene_symbol) rows")
    labels["gene_effect"] = pd.to_numeric(labels["gene_effect"], errors="coerce")
    if not np.isfinite(labels["gene_effect"].to_numpy(dtype=float)).all():
        raise ValueError("GeneEffect CSV contains non-finite gene_effect values")

    role_by_id = manifest.set_index("model_id")["role"].astype(str).to_dict()
    unknown = sorted(set(labels["model_id"]) - set(role_by_id))
    if unknown:
        raise ValueError(
            f"GeneEffect CSV has model_id values absent from manifest: {unknown}"
        )
    test_ids = sorted(
        model_id
        for model_id in set(labels["model_id"])
        if role_by_id[model_id] == TEST_ROLE
    )
    if test_ids:
        raise ValueError(f"GeneEffect CSV must not contain role=test lines: {test_ids}")
    non_tahoe = sorted(
        model_id
        for model_id in set(labels["model_id"])
        if role_by_id[model_id] != TRAIN_HEAD_ROLE
    )
    if non_tahoe:
        raise ValueError(
            "P0 is Tahoe-only; GeneEffect CSV contains non-train_head lines: "
            f"{non_tahoe}"
        )

    target_ids = sorted(
        manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE, "model_id"].astype(str)
    )
    present_ids = sorted(labels["model_id"].unique())
    if present_ids != target_ids:
        missing = sorted(set(target_ids) - set(present_ids))
        raise ValueError(f"GeneEffect CSV is missing train_head lines: {missing}")
    genes = sorted(labels["gene_symbol"].unique())
    if len(target_ids) < 3:
        raise ValueError("P0 requires at least three train_head lines")
    if len(genes) < 2:
        raise ValueError("P0 requires at least two genes")
    expected_rows = len(target_ids) * len(genes)
    if len(labels) != expected_rows:
        counts = labels.groupby("model_id")["gene_symbol"].nunique()
        incomplete = sorted(counts[counts != len(genes)].index.astype(str))
        raise ValueError(
            "GeneEffect CSV is not a complete line-by-gene matrix; "
            f"incomplete lines: {incomplete}"
        )
    return labels, genes


def _load_context(
    path: Path, manifest: pd.DataFrame, target_ids: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    context = pd.read_csv(path)
    _require_columns(context, ("model_id",), "line-context CSV")
    if context["model_id"].isna().any():
        raise ValueError("line-context CSV contains missing model_id values")
    context["model_id"] = context["model_id"].astype(str)
    if context["model_id"].duplicated().any():
        raise ValueError("line-context CSV has duplicate model_id rows")
    role_by_id = manifest.set_index("model_id")["role"].astype(str).to_dict()
    unknown = sorted(set(context["model_id"]) - set(role_by_id))
    if unknown:
        raise ValueError(
            f"line-context CSV has model_id values absent from manifest: {unknown}"
        )
    test_ids = sorted(
        model_id
        for model_id in context["model_id"]
        if role_by_id[model_id] == TEST_ROLE
    )
    if test_ids:
        raise ValueError(
            f"line-context CSV must not contain role=test lines: {test_ids}"
        )
    feature_columns = [column for column in context.columns if column != "model_id"]
    if not feature_columns:
        raise ValueError("line-context CSV has no feature columns")
    context = context.loc[context["model_id"].isin(target_ids), :].copy()
    if sorted(context["model_id"]) != target_ids:
        missing = sorted(set(target_ids) - set(context["model_id"]))
        raise ValueError(f"line-context CSV is missing train_head lines: {missing}")
    for column in feature_columns:
        converted = pd.to_numeric(context[column], errors="coerce")
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(
                f"line-context feature {column!r} must be numeric and finite"
            )
        context[column] = converted
    return context.set_index("model_id").sort_index(), feature_columns


def _load_copy_prior(path: Path, genes: list[str]) -> pd.Series:
    prior = pd.read_csv(path)
    _require_columns(prior, ("gene_symbol", "gene_effect"), "copy-K562 prior CSV")
    prior = prior.loc[:, ["gene_symbol", "gene_effect"]].copy()
    if prior["gene_symbol"].isna().any():
        raise ValueError("copy-K562 prior CSV contains missing gene_symbol values")
    prior["gene_symbol"] = prior["gene_symbol"].astype(str)
    if prior["gene_symbol"].duplicated().any():
        raise ValueError("copy-K562 prior CSV has duplicate gene_symbol rows")
    prior["gene_effect"] = pd.to_numeric(prior["gene_effect"], errors="coerce")
    if not np.isfinite(prior["gene_effect"].to_numpy(dtype=float)).all():
        raise ValueError("copy-K562 prior CSV contains non-finite gene_effect values")
    if sorted(prior["gene_symbol"]) != genes:
        missing = sorted(set(genes) - set(prior["gene_symbol"]))
        extra = sorted(set(prior["gene_symbol"]) - set(genes))
        raise ValueError(
            "copy-K562 prior gene set must exactly match GeneEffect genes; "
            f"missing={missing}, extra={extra}"
        )
    return prior.set_index("gene_symbol")["gene_effect"].reindex(genes)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    statistic = float(spearmanr(y_true, y_pred).statistic)
    if not np.isfinite(statistic):
        raise ValueError("Spearman correlation is undefined; check constant vectors")
    return statistic


def _cvar25(values: np.ndarray) -> float:
    count = max(1, math.ceil(0.25 * len(values)))
    return float(np.mean(np.sort(values)[:count]))


def _prediction_rows(
    *,
    model_id: str,
    lineage: str,
    genes: list[str],
    y_true: np.ndarray,
    method: str,
    y_pred: np.ndarray,
    source_model_id: str | None = None,
    effective_pca_components: int | None = None,
) -> list[dict[str, object]]:
    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        raise ValueError(f"method {method} produced invalid predictions for {model_id}")
    return [
        {
            "model_id": model_id,
            "lineage": lineage,
            "gene_symbol": gene,
            "y_true": float(truth),
            "method": method,
            "y_pred": float(prediction),
            "source_model_id": source_model_id,
            "effective_pca_components": effective_pca_components,
        }
        for gene, truth, prediction in zip(genes, y_true, y_pred, strict=True)
    ]


def run_p0_baseline_ladder(
    *,
    manifest_path: Path,
    phase_a_dir: Path,
    validation_plan_path: Path,
    validation_policy_path: Path,
    gene_effect_path: Path,
    context_path: Path | None = None,
    copy_k562_path: Path | None = None,
    config: P0BaselineConfig = P0BaselineConfig(),
) -> P0BaselineResult:
    """Run strict source-line leave-one-out GeneEffect baselines.

    Args:
        manifest_path: Frozen cell-line manifest.
        phase_a_dir: Frozen Phase-A registration and 587-gene slice directory.
        validation_plan_path: Shared plan generated by
            ``build_tx1_p0_validation.py``.
        validation_policy_path: Independent policy JSON providing the
            authoritative manifest SHA and fold-generation settings.
        gene_effect_path: Long-form complete Tahoe-only GeneEffect matrix.
        context_path: Optional line-level numeric context features.
        copy_k562_path: Optional complete gene-level K562 prior.
        config: Fixed development-only hyperparameters.

    Returns:
        Prediction, per-line metric, and summary artifacts.
    """
    frozen_genes, registration_sha, slice_sha = _load_frozen_slice(phase_a_dir)
    manifest, outer_folds, manifest_sha256 = _load_validation_plan(
        validation_plan_path, manifest_path, validation_policy_path
    )
    labels, genes = _load_labels(gene_effect_path, manifest)
    if genes != frozen_genes:
        raise ValueError(
            "GeneEffect gene_symbol set must exactly match the frozen 587-gene slice"
        )
    target_ids = sorted(labels["model_id"].unique())
    lineages = manifest.set_index("model_id")["lineage"].astype(str).reindex(target_ids)
    matrix = (
        labels.pivot(index="model_id", columns="gene_symbol", values="gene_effect")
        .reindex(index=target_ids, columns=genes)
        .astype(float)
    )
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("GeneEffect matrix is incomplete or non-finite")

    context: pd.DataFrame | None = None
    feature_columns: list[str] = []
    if context_path is not None:
        context, feature_columns = _load_context(context_path, manifest, target_ids)
    copy_prior = (
        _load_copy_prior(copy_k562_path, genes) if copy_k562_path is not None else None
    )

    rows: list[dict[str, object]] = []
    effective_components: list[int] = []
    plan_target_ids: list[str] = []
    for fold in outer_folds:
        raw_targets = fold["outer_validation_model_ids"]
        raw_sources = fold["outer_train_model_ids"]
        if not isinstance(raw_targets, list) or not isinstance(raw_sources, list):
            raise ValueError("validation plan outer fold IDs must be lists")
        target_id = str(raw_targets[0])
        source_ids = [str(model_id) for model_id in raw_sources]
        plan_target_ids.append(target_id)
        source_y = matrix.loc[source_ids].to_numpy(dtype=float)
        y_true = matrix.loc[target_id].to_numpy(dtype=float)
        global_mean = source_y.mean(axis=0)
        lineage = str(lineages.loc[target_id])
        rows.extend(
            _prediction_rows(
                model_id=target_id,
                lineage=lineage,
                genes=genes,
                y_true=y_true,
                method=GENE_MEAN_METHOD,
                y_pred=global_mean,
            )
        )

        same_lineage_ids = [
            model_id
            for model_id in source_ids
            if str(lineages.loc[model_id]) == lineage
        ]
        lineage_mean = (
            matrix.loc[same_lineage_ids].to_numpy(dtype=float).mean(axis=0)
            if same_lineage_ids
            else global_mean
        )
        shrunk = (
            config.lineage_alpha * lineage_mean
            + (1.0 - config.lineage_alpha) * global_mean
        )
        rows.extend(
            _prediction_rows(
                model_id=target_id,
                lineage=lineage,
                genes=genes,
                y_true=y_true,
                method=LINEAGE_METHOD,
                y_pred=shrunk,
            )
        )

        if copy_prior is not None:
            rows.extend(
                _prediction_rows(
                    model_id=target_id,
                    lineage=lineage,
                    genes=genes,
                    y_true=y_true,
                    method=COPY_K562_METHOD,
                    y_pred=copy_prior.to_numpy(dtype=float),
                )
            )

        if context is None:
            continue
        source_context = context.loc[source_ids, feature_columns].to_numpy(dtype=float)
        target_context = context.loc[[target_id], feature_columns].to_numpy(dtype=float)
        scaler = StandardScaler().fit(source_context)
        source_scaled = scaler.transform(source_context)
        target_scaled = scaler.transform(target_context)
        if not np.isfinite(source_scaled).all() or not np.isfinite(target_scaled).all():
            raise ValueError(
                f"context standardization produced non-finite values for {target_id}"
            )
        distances = np.linalg.norm(source_scaled - target_scaled, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_id = source_ids[nearest_index]
        rows.extend(
            _prediction_rows(
                model_id=target_id,
                lineage=lineage,
                genes=genes,
                y_true=y_true,
                method=NEAREST_METHOD,
                y_pred=matrix.loc[nearest_id].to_numpy(dtype=float),
                source_model_id=nearest_id,
            )
        )

        n_components = min(
            config.pca_components,
            len(source_ids) - 1,
            len(feature_columns),
        )
        if n_components < 1:
            raise ValueError(f"cannot fit context PCA for outer target {target_id}")
        pca = PCA(n_components=n_components, svd_solver="full").fit(source_scaled)
        source_pca = pca.transform(source_scaled)
        target_pca = pca.transform(target_scaled)
        if (
            not np.isfinite(pca.explained_variance_).all()
            or not np.isfinite(source_pca).all()
            or not np.isfinite(target_pca).all()
        ):
            raise ValueError(f"context PCA is degenerate for outer target {target_id}")
        ridge = Ridge(alpha=config.ridge_alpha).fit(source_pca, source_y)
        ridge_prediction = np.asarray(ridge.predict(target_pca)[0], dtype=float)
        effective_components.append(n_components)
        rows.extend(
            _prediction_rows(
                model_id=target_id,
                lineage=lineage,
                genes=genes,
                y_true=y_true,
                method=PCA_RIDGE_METHOD,
                y_pred=ridge_prediction,
                effective_pca_components=n_components,
            )
        )
    if plan_target_ids != target_ids:
        raise ValueError("validation plan target order differs from GeneEffect matrix")

    per_prediction = (
        pd.DataFrame(rows)
        .sort_values(["method", "model_id", "gene_symbol"], kind="stable")
        .reset_index(drop=True)
    )
    if per_prediction.duplicated(["model_id", "gene_symbol", "method"]).any():
        raise ValueError("internal error: duplicate prediction keys")
    if not np.isfinite(per_prediction[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("internal error: non-finite output predictions")

    line_rows: list[dict[str, object]] = []
    for (method, model_id), group in per_prediction.groupby(
        ["method", "model_id"], sort=True
    ):
        rho = _spearman(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        )
        line_rows.append(
            {
                "model_id": model_id,
                "lineage": str(group["lineage"].iloc[0]),
                "method": method,
                "n_genes": len(group),
                "spearman": rho,
                "coverage": float(np.isfinite(group["y_pred"]).mean()),
            }
        )
    per_line = pd.DataFrame(line_rows)
    reference_method = COPY_K562_METHOD if copy_prior is not None else GENE_MEAN_METHOD
    reference = per_line.loc[per_line["method"] == reference_method].set_index(
        "model_id"
    )["spearman"]
    per_line["reference_method"] = reference_method
    per_line["reference_spearman"] = per_line["model_id"].map(reference)
    per_line["delta_rho"] = per_line["spearman"] - per_line["reference_spearman"]
    per_line["negative_transfer"] = per_line["delta_rho"] < 0.0
    per_line = per_line.sort_values(["method", "model_id"], kind="stable").reset_index(
        drop=True
    )

    method_summaries: dict[str, dict[str, object]] = {}
    for method, group in per_line.groupby("method", sort=True):
        deltas = group["delta_rho"].to_numpy(dtype=float)
        entry: dict[str, object] = {
            "macro_spearman": float(group["spearman"].mean()),
            "negative_transfer_rate": float(group["negative_transfer"].mean()),
            "cvar25_spearman": _cvar25(group["spearman"].to_numpy(dtype=float)),
            "cvar25_delta_rho": _cvar25(deltas),
            "coverage": float(group["coverage"].mean()),
            "n_lines": int(len(group)),
        }
        entry["delta_rho"] = float(deltas.mean())
        method_summaries[str(method)] = entry

    input_paths: dict[str, Path] = {
        "cell_line_manifest": manifest_path,
        "gene_effect": gene_effect_path,
    }
    if context_path is not None:
        input_paths["line_context"] = context_path
    if copy_k562_path is not None:
        input_paths["copy_k562_prior"] = copy_k562_path
    summary: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "formal": False,
        "development_only": True,
        "test_lines_excluded": True,
        "target_role": TRAIN_HEAD_ROLE,
        "source_role": TRAIN_HEAD_ROLE,
        "anchors_excluded": True,
        "outer_validation": "source-line leave-one-out",
        "reference_method": reference_method,
        "n_lines": len(target_ids),
        "n_genes": len(genes),
        "methods": method_summaries,
        "config": {
            "lineage_alpha": config.lineage_alpha,
            "pca_components_requested": config.pca_components,
            "ridge_alpha": config.ridge_alpha,
        },
        "input_sha256": {
            name: _sha256_file(path) for name, path in sorted(input_paths.items())
        },
    }
    summary["input_sha256"]["validation_plan"] = _sha256_file(validation_plan_path)
    summary["input_sha256"]["validation_policy"] = _sha256_file(validation_policy_path)
    summary["input_sha256"]["cell_line_manifest"] = manifest_sha256
    summary["input_sha256"]["phase_a_registration"] = registration_sha
    summary["input_sha256"]["differentially_essential_slice"] = slice_sha
    if effective_components:
        summary["config"]["pca_components_effective_by_fold"] = sorted(
            set(effective_components)
        )
    return P0BaselineResult(per_prediction, per_line, summary)


def write_p0_baseline_artifacts(result: P0BaselineResult, output_dir: Path) -> None:
    """Atomically write the three deterministic P0 output artifacts."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P0 output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        result.per_prediction.to_csv(temporary / "per_prediction.csv", index=False)
        result.per_line.to_csv(temporary / "per_line.csv", index=False)
        (temporary / "summary.json").write_text(
            json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

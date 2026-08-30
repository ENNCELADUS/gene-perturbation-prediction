"""Fail-closed lifecycle and integrity checks for Exp13 Stage 2 runs.

This module deliberately knows nothing about model architecture or training.  It
owns the small but load-bearing filesystem contract shared by the Stage 2 runner:
fresh run directories, atomic JSON writes, terminal sentinels, and verification
that every required output belongs to the same completed run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from aivc_model.esm2_provenance import (
    ISOFORM_POLICY,
    authenticate_uniprot_mapping,
)


REQUIRED_STAGE2_OUTPUTS: tuple[str, ...] = (
    "config_snapshot.json",
    "cell_line_geneeffect_226_split.json",
    "stage1_model_manifest.json",
    "stage1_objective.json",
    "esm2_gene_universe_manifest.json",
    "esm2_provenance_manifest.json",
    "esm2_uniprot_mapping.json",
    "esm2_uniprot_mapping.csv",
    "g_var_manifest.json",
    "feature_schema.json",
    "projection.npz",
    "standardizer.npz",
    "residual_targets.npz",
    "response_targets/lineage.json",
    "backbone_load_report.json",
    "lambda_calibration.json",
    "feature_generation.json",
    "checkpoint_selection.json",
    "geneeffect_residual_predictions.csv",
    "geneeffect_residual_metrics.json",
    "run_manifest.json",
    "condition_features/stage1_frozen/manifest.json",
    "condition_features/stage2_selected/manifest.json",
    "warmup/training/best/head.pt",
    "warmup/training/best/metadata.json",
    "warmup/training/train_log.csv",
    "joint/training/best/e2e_state.pt",
    "joint/training/best/metadata.json",
    "joint/training/train_log.csv",
    "model_package/e2e_state.pt",
    "model_package/model_manifest.json",
)

_STAGE2_METHODS = frozenset(
    {
        "e2e_full",
        "gene_mean",
        "copy_prior",
        "nearest_line[z_c]",
        "context_pca_ridge[z_c]",
    }
)
_BASELINE_METHODS = _STAGE2_METHODS - {"e2e_full"}
_STAGE1_CODE_PROVENANCE = {
    "status": "unavailable",
    "reason": "historical_run_has_no_immutable_training_code_identity",
}
_STAGE1_DATA_PROVENANCE = {
    "status": "incomplete",
    "missing_identities": [
        "cell_line_manifest",
        "tx1_basal_cache",
        "response_cache",
        "perturbseq_source_content",
    ],
    "reason": "historical_run_manifest_does_not_hash_all_training_data_inputs",
}

# Audited static-import closure from the formal Stage 2 entrypoint.  This list is
# intentionally explicit: adding a new local runtime dependency requires an
# equally explicit provenance-contract update.
STAGE2_RUNTIME_CODE_PATHS: tuple[str, ...] = (
    "scripts/train_geneeffect_e2e.py",
    "src/aivc_model/__init__.py",
    "src/aivc_model/benchmark_split.py",
    "src/aivc_model/distributed.py",
    "src/aivc_model/esm2_provenance.py",
    "src/aivc_model/gene_embeddings.py",
    "src/aivc_model/gene_splits.py",
    "src/aivc_model/geneeffect_data.py",
    "src/aivc_model/geneeffect_e2e.py",
    "src/aivc_model/geneeffect_feature_store.py",
    "src/aivc_model/geneeffect_features.py",
    "src/aivc_model/geneeffect_head.py",
    "src/aivc_model/geneeffect_sampler.py",
    "src/aivc_model/geneeffect_stage2_runner.py",
    "src/aivc_model/geneeffect_training.py",
    "src/aivc_model/geneeffect_training_loop.py",
    "src/aivc_model/residual_ladder.py",
    "src/aivc_model/residual_metrics.py",
    "src/aivc_model/residual_target.py",
    "src/aivc_model/response_training.py",
    "src/aivc_model/stage1_artifact.py",
    "src/aivc_model/stage1_config.py",
    "src/aivc_model/stage2_artifacts.py",
    "src/aivc_model/stage2_config.py",
    "src/aivc_model/state_core.py",
    "src/aivc_model/state_warm_start.py",
    "src/aivc_model/tx1_basal.py",
    "src/aivc_model/tx1_embed_cache.py",
    "src/aivc_model/tx1_predicted_response.py",
    "src/aivc_model/tx1_response_data.py",
    "src/aivc_model/tx1_response_gene_bags_cache.py",
    "src/aivc_model/tx1_response_streaming.py",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage2_runtime_code_sha256() -> dict[str, str]:
    """Hash the complete audited local runtime closure for formal Stage 2."""
    root = Path(__file__).resolve().parents[2]
    return {
        relative: sha256_file(root / relative) for relative in STAGE2_RUNTIME_CODE_PATHS
    }


def verify_stage2_runtime_code_sha256(recorded: object) -> dict[str, str]:
    """Require exact path membership and hashes for the current runtime tree."""
    if not isinstance(recorded, dict):
        raise ValueError("stage2_code_sha256 must be an object")
    if set(recorded) != set(STAGE2_RUNTIME_CODE_PATHS):
        missing = sorted(set(STAGE2_RUNTIME_CODE_PATHS) - set(recorded))
        extra = sorted(set(recorded) - set(STAGE2_RUNTIME_CODE_PATHS))
        raise ValueError(
            "Stage 2 runtime code path closure mismatch: "
            f"missing={missing}, extra={extra}"
        )
    current = stage2_runtime_code_sha256()
    mismatched = sorted(
        path
        for path in STAGE2_RUNTIME_CODE_PATHS
        if recorded.get(path) != current[path]
    )
    if mismatched:
        raise ValueError(f"Stage 2 runtime code SHA256 mismatch: {mismatched}")
    return current


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically replace *path* with canonical, newline-terminated JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Stage2RunLayout:
    """Resolved paths for one immutable Stage 2 run directory."""

    root: Path

    @property
    def warmup(self) -> Path:
        return self.root / "warmup"

    @property
    def joint(self) -> Path:
        return self.root / "joint"

    @property
    def condition_features(self) -> Path:
        return self.root / "condition_features"

    @property
    def model_package(self) -> Path:
        return self.root / "model_package"

    @property
    def complete(self) -> Path:
        return self.root / "complete.json"

    @property
    def failure(self) -> Path:
        return self.root / "failure.json"


def prepare_run_dir(path: Path, *, resume: bool = False) -> Stage2RunLayout:
    """Create a fresh run directory or validate an explicitly resumed one.

    A terminal run is never resumable: reusing it could place new checkpoints
    beside old predictions.  An incomplete run may be resumed only when it has
    a run manifest and no failure marker; failures require a new run id so the
    evidence from the failed attempt stays intact.
    """
    root = Path(path)
    if not root.exists():
        root.mkdir(parents=True)
    elif not resume:
        raise FileExistsError(
            f"run directory {root} already exists; choose a fresh run id or use "
            "resume=True for a verified incomplete run"
        )
    elif (root / "complete.json").exists():
        raise ValueError(f"completed run {root} cannot be resumed")
    elif (root / "failure.json").exists():
        raise ValueError(f"failed run {root} cannot be resumed; choose a fresh run id")
    elif not (root / "run_manifest.json").is_file():
        raise ValueError(
            f"cannot resume {root}: run_manifest.json is absent, so run identity "
            "cannot be authenticated"
        )
    layout = Stage2RunLayout(root=root)
    for directory in (
        layout.warmup,
        layout.joint,
        layout.condition_features,
        layout.model_package,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def artifact_digests(root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    """Hash required files, rejecting missing files and non-files."""
    root = Path(root)
    digests: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Stage 2 artifact is missing: {path}")
        digests[str(relative)] = sha256_file(path)
    if any(
        str(relative).startswith("condition_features/") for relative in relative_paths
    ):
        feature_root = root / "condition_features"
        for path in sorted(feature_root.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(root))
                digests[relative] = sha256_file(path)
    return digests


def _read_run_id(root: Path) -> str:
    manifest_path = Path(root) / "run_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"run manifest is missing or unreadable: {exc}") from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_manifest.json must contain a non-empty run_id")
    return run_id


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"required JSON artifact is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"required JSON artifact must contain an object: {path}")
    return payload


def _require_sha256(payload: Mapping[str, object], field: str, path: Path) -> str:
    value = payload.get(field)
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must contain lowercase SHA-256 field {field!r}")
    return value


def _verify_selected_checkpoint(
    root: Path,
    *,
    training_dir: str,
    filename: str,
    selection: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    checkpoint = root / training_dir / "best" / filename
    metadata_path = checkpoint.parent / "metadata.json"
    metadata = _read_json_object(metadata_path)
    selection_name = "validation_macro_per_gene_spearman"
    if (
        metadata.get("selection_name") != selection_name
        or metadata.get("selection_direction") != "maximize"
    ):
        raise ValueError(
            f"selected checkpoint metric contract mismatch: {metadata_path}"
        )
    digest = sha256_file(checkpoint)
    if _require_sha256(metadata, "checkpoint_sha256", metadata_path) != digest:
        raise ValueError(f"selected checkpoint SHA256 mismatch: {checkpoint}")
    if metadata.get("epoch") != selection.get("best_epoch"):
        raise ValueError(f"selected checkpoint epoch mismatch: {metadata_path}")
    if metadata.get("metric_value") != selection.get("best_metric"):
        raise ValueError(f"selected checkpoint metric mismatch: {metadata_path}")
    history_path = root / training_dir / "train_log.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = [
        row for row in rows if row.get("epoch") == str(metadata.get("epoch"))
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            f"selected checkpoint epoch is absent from training history: {history_path}"
        )
    try:
        history_metric = float(selected_rows[0][selection_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"training history has invalid selection metric: {history_path}"
        ) from exc
    if not math.isfinite(history_metric) or not math.isclose(
        history_metric,
        float(metadata["metric_value"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"training history selected metric mismatch: {history_path}")
    return digest, metadata


def _verify_lambda_calibration(payload: Mapping[str, object]) -> None:
    lambda_dep = payload.get("lambda_dep")
    if (
        isinstance(lambda_dep, bool)
        or not isinstance(lambda_dep, (int, float))
        or not math.isfinite(float(lambda_dep))
        or not 1e-3 <= float(lambda_dep) <= 1e3
    ):
        raise ValueError("lambda_dep must be finite and clipped to [1e-3, 1e3]")
    calibration_values: dict[str, list[float]] = {}
    for name in (
        "raw_ratios",
        "response_gradient_norms",
        "dependency_gradient_norms",
    ):
        values = payload.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 8
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in values
            )
        ):
            raise ValueError(
                f"lambda calibration {name} must contain 8 positive finite values"
            )
        calibration_values[name] = [float(value) for value in values]
    ratios = calibration_values["raw_ratios"]
    response_norms = calibration_values["response_gradient_norms"]
    dependency_norms = calibration_values["dependency_gradient_norms"]
    if any(
        not math.isclose(ratio, response / dependency, rel_tol=1e-12, abs_tol=1e-12)
        for ratio, response, dependency in zip(
            ratios, response_norms, dependency_norms, strict=True
        )
    ):
        raise ValueError("lambda calibration ratios do not match gradient norms")
    ordered = sorted(ratios)
    median_ratio = 0.5 * (ordered[3] + ordered[4])
    expected_lambda = min(1e3, max(1e-3, median_ratio))
    if not math.isclose(
        float(lambda_dep), expected_lambda, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("lambda_dep does not match the clipped median ratio")


def _verify_stage1_provenance_claims(
    stage1: Mapping[str, object],
    run: Mapping[str, object],
    model_package: Mapping[str, object],
) -> None:
    expected_stage1 = {
        "training_code_provenance_status": _STAGE1_CODE_PROVENANCE["status"],
        "training_code_provenance_reason": _STAGE1_CODE_PROVENANCE["reason"],
        "training_data_provenance_status": _STAGE1_DATA_PROVENANCE["status"],
        "training_data_provenance_missing_identities": _STAGE1_DATA_PROVENANCE[
            "missing_identities"
        ],
        "training_data_provenance_reason": _STAGE1_DATA_PROVENANCE["reason"],
    }
    for field, expected in expected_stage1.items():
        if stage1.get(field) != expected:
            raise ValueError(f"Stage-1 manifest provenance claim mismatch: {field}")
        propagated_field = f"stage1_{field}"
        for label, payload in (("run manifest", run), ("model package", model_package)):
            if payload.get(propagated_field) != expected:
                raise ValueError(
                    f"{label} Stage-1 provenance claim mismatch: {propagated_field}"
                )


def _verify_distributed_runtime(value: object) -> Mapping[str, object]:
    return _require_mapping(value, "distributed_runtime")


def _string_vector(array: object, label: str) -> tuple[str, ...]:
    import numpy as np

    value = np.asarray(array)
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"residual_targets.npz {label} must be a string vector")
    strings = tuple(value.astype(str).tolist())
    if (
        not strings
        or any(not item for item in strings)
        or len(set(strings)) != len(strings)
    ):
        raise ValueError(f"residual_targets.npz {label} must be nonempty and unique")
    return strings


def _verify_residual_target_artifact(
    root: Path,
    split_payload: Mapping[str, object],
    universe_payload: Mapping[str, object],
    run: Mapping[str, object],
) -> tuple[
    dict[str, set[tuple[str, str]]],
    dict[tuple[str, str], float],
    dict[str, float],
]:
    import numpy as np

    from aivc_model.geneeffect_data import PINNED_SPLIT_SHA256

    split_path = root / "cell_line_geneeffect_226_split.json"
    if sha256_file(split_path) != PINNED_SPLIT_SHA256:
        raise ValueError("copied split does not match the pinned Exp13 authority")
    if run.get("split_sha256") != PINNED_SPLIT_SHA256:
        raise ValueError("run manifest split identity mismatch")
    artifact_path = root / "residual_targets.npz"
    if _require_sha256(
        run, "residual_targets_artifact_sha256", root / "run_manifest.json"
    ) != sha256_file(artifact_path):
        raise ValueError("residual-target artifact SHA256 mismatch")
    expected_arrays = {
        "gene_symbols",
        "model_ids",
        "residual_targets",
        "label_mask",
        "mu_train",
        "centering_model_ids",
    }
    try:
        with np.load(artifact_path, allow_pickle=False) as loaded:
            if set(loaded.files) != expected_arrays:
                raise ValueError("array names mismatch")
            genes = _string_vector(loaded["gene_symbols"], "gene_symbols")
            model_ids = _string_vector(loaded["model_ids"], "model_ids")
            centering_ids = _string_vector(
                loaded["centering_model_ids"], "centering_model_ids"
            )
            targets = np.asarray(loaded["residual_targets"])
            label_mask = np.asarray(loaded["label_mask"])
            mu_train = np.asarray(loaded["mu_train"])
    except Exception as exc:
        raise ValueError(f"residual_targets.npz is invalid: {exc}") from exc

    split_parts: dict[str, tuple[str, ...]] = {}
    for name in ("train", "val", "test"):
        values = split_payload.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"copied split must contain unique nonempty {name} IDs")
        split_parts[name] = tuple(values)
    expected_model_ids = (
        *split_parts["train"],
        *split_parts["val"],
        *split_parts["test"],
    )
    if model_ids != expected_model_ids:
        raise ValueError("residual-target ModelID order differs from the copied split")
    unlabeled = split_payload.get("unlabeled_train")
    if not isinstance(unlabeled, list) or any(
        item not in split_parts["train"] for item in unlabeled
    ):
        raise ValueError("copied split unlabeled_train is malformed")
    expected_centering = tuple(
        model_id for model_id in split_parts["train"] if model_id not in set(unlabeled)
    )
    if centering_ids != expected_centering:
        raise ValueError("residual-target centering ModelIDs mismatch")
    if universe_payload.get("scored_symbols") != list(genes):
        raise ValueError("residual-target genes differ from the scored universe")
    shape = (len(genes), len(model_ids))
    if targets.dtype != np.dtype(np.float32) or targets.shape != shape:
        raise ValueError("residual-target matrix must be float32 genes-by-ModelIDs")
    if label_mask.dtype != np.dtype(bool) or label_mask.shape != shape:
        raise ValueError("residual-target label mask must be bool genes-by-ModelIDs")
    if mu_train.dtype != np.dtype(np.float64) or mu_train.shape != (len(genes),):
        raise ValueError("residual-target mu_train must be ordered float64")
    if not np.isfinite(targets).all() or not np.isfinite(mu_train).all():
        raise ValueError("residual-target arrays must be finite")

    target_digest = hashlib.sha256()
    target_digest.update("\n".join(genes).encode())
    target_digest.update("\n".join(model_ids).encode())
    target_digest.update(np.ascontiguousarray(targets).tobytes())
    target_digest.update(np.ascontiguousarray(label_mask).tobytes())
    mu_digest = hashlib.sha256()
    mu_digest.update("\n".join(genes).encode())
    mu_digest.update(np.ascontiguousarray(mu_train).tobytes())
    recomputed = {
        "residual_target_sha256": target_digest.hexdigest(),
        "mu_train_sha256": mu_digest.hexdigest(),
        "centering_fit_model_ids_sha256": hashlib.sha256(
            "\n".join(centering_ids).encode()
        ).hexdigest(),
    }
    for field, digest in recomputed.items():
        if _require_sha256(run, field, root / "run_manifest.json") != digest:
            raise ValueError(f"run manifest {field} does not match residual targets")

    column_by_model = {model_id: index for index, model_id in enumerate(model_ids)}
    evaluation_keys = {
        split_name: {
            (model_id, gene)
            for model_id in split_parts[split_name]
            for gene_index, gene in enumerate(genes)
            if bool(label_mask[gene_index, column_by_model[model_id]])
        }
        for split_name in ("val", "test")
    }
    authoritative_residual = {
        (model_id, gene): float(targets[gene_index, column_by_model[model_id]])
        for split_name in ("val", "test")
        for model_id in split_parts[split_name]
        for gene_index, gene in enumerate(genes)
        if bool(label_mask[gene_index, column_by_model[model_id]])
    }
    authoritative_mu = {
        gene: float(mu_train[gene_index]) for gene_index, gene in enumerate(genes)
    }
    return evaluation_keys, authoritative_residual, authoritative_mu


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_metric_scalar(recorded: object, actual: float, label: str) -> None:
    if math.isnan(actual):
        if recorded is not None:
            raise ValueError(f"{label} must be null because the metric is undefined")
        return
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, (int, float))
        or not math.isfinite(float(recorded))
        or not math.isclose(float(recorded), actual, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(f"{label} does not match recomputed predictions")


def _verify_metric_count(recorded: object, actual: int, label: str) -> None:
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, int)
        or recorded != actual
    ):
        raise ValueError(f"{label} does not match recomputed predictions")


def _verify_score_entry(
    entry: Mapping[str, object], score: object, *, label: str, e2e: bool
) -> None:
    if e2e:
        fields = {
            "macro_per_line_spearman": score.macro_per_line,
            "macro_per_gene_spearman": score.macro_per_gene,
        }
        counts = {
            "per_line_defined": score.n_lines,
            "per_gene_defined": score.n_genes,
            "per_line_undefined": score.n_line_undefined,
            "per_gene_undefined": score.n_gene_undefined,
        }
    else:
        fields = {
            "macro_per_line": score.macro_per_line,
            "macro_per_gene": score.macro_per_gene,
        }
        counts = {
            "n_lines": score.n_lines,
            "n_genes": score.n_genes,
            "n_line_undefined": score.n_line_undefined,
            "n_gene_undefined": score.n_gene_undefined,
        }
    for name, value in fields.items():
        _verify_metric_scalar(entry.get(name), value, f"{label}.{name}")
    for name, value in counts.items():
        _verify_metric_count(entry.get(name), value, f"{label}.{name}")
    if not e2e:
        return
    per_line = _require_mapping(entry.get("per_line_spearman"), f"{label}.per_line")
    expected_lines = {str(index) for index in score.per_line.index}
    if set(per_line) != expected_lines:
        raise ValueError(f"{label}.per_line_spearman membership mismatch")
    for model_id, value in score.per_line.items():
        _verify_metric_scalar(
            per_line[str(model_id)], float(value), f"{label}.per_line[{model_id}]"
        )


def _verify_predictions_and_metrics(
    root: Path,
    split_payload: Mapping[str, object],
    universe_payload: Mapping[str, object],
    expected_keys_by_split: Mapping[str, set[tuple[str, str]]],
    authoritative_residual: Mapping[tuple[str, str], float],
    authoritative_mu: Mapping[str, float],
    metrics: Mapping[str, object],
) -> float:
    import numpy as np
    import pandas as pd

    from aivc_model.residual_metrics import score_predictions

    split_members: dict[str, tuple[str, ...]] = {}
    for split_name in ("val", "test"):
        raw_members = split_payload.get(split_name)
        if (
            not isinstance(raw_members, list)
            or not raw_members
            or any(not isinstance(item, str) or not item for item in raw_members)
            or len(set(raw_members)) != len(raw_members)
        ):
            raise ValueError(
                f"copied split must contain unique nonempty {split_name} IDs"
            )
        split_members[split_name] = tuple(raw_members)
    if set(split_members["val"]) & set(split_members["test"]):
        raise ValueError("copied split val/test membership overlaps")
    raw_genes = universe_payload.get("scored_symbols")
    if (
        not isinstance(raw_genes, list)
        or not raw_genes
        or any(not isinstance(gene, str) or not gene for gene in raw_genes)
        or len(set(raw_genes)) != len(raw_genes)
        or universe_payload.get("scored_gene_count") != len(raw_genes)
    ):
        raise ValueError("scored-gene universe manifest is malformed")
    scored_genes = set(raw_genes)

    prediction_path = root / "geneeffect_residual_predictions.csv"
    try:
        frame = pd.read_csv(prediction_path)
    except Exception as exc:
        raise ValueError(
            f"GeneEffect predictions artifact is unreadable: {exc}"
        ) from exc
    required_columns = {
        "split",
        "method",
        "model_id",
        "gene_symbol",
        "gene_effect",
        "residual",
        "residual_prediction",
    }
    if set(frame.columns) != required_columns or frame.empty:
        raise ValueError(
            "GeneEffect predictions artifact has invalid columns or is empty"
        )
    identity_columns = ["split", "method", "model_id", "gene_symbol"]
    if frame[identity_columns].isna().any().any():
        raise ValueError("GeneEffect predictions contain missing identities")
    if frame.duplicated(identity_columns).any():
        raise ValueError("GeneEffect predictions contain duplicate evaluation keys")
    if set(frame["split"]) != {"val", "test"}:
        raise ValueError("GeneEffect predictions must contain exactly val and test")
    if set(frame["method"]) != _STAGE2_METHODS:
        raise ValueError("GeneEffect predictions do not contain the five fixed methods")
    numeric_columns = ["gene_effect", "residual", "residual_prediction"]
    try:
        numeric = frame[numeric_columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("GeneEffect truth/prediction columns must be numeric") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("GeneEffect truth/prediction columns must be finite")
    frame[numeric_columns] = numeric
    for row in frame.itertuples(index=False):
        key = (str(row.model_id), str(row.gene_symbol))
        expected_residual = authoritative_residual.get(key)
        expected_mu = authoritative_mu.get(str(row.gene_symbol))
        if expected_residual is None or expected_mu is None:
            raise ValueError(
                f"prediction truth key is absent from residual targets: {key}"
            )
        if not math.isclose(
            float(row.residual),
            expected_residual,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"prediction residual differs from authoritative target: {key}"
            )
        if not math.isclose(
            float(row.gene_effect),
            expected_residual + expected_mu,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"prediction gene_effect differs from residual + mu_train: {key}"
            )

    baselines = _require_mapping(metrics.get("baselines"), "metrics.baselines")
    baseline_split = _require_mapping(baselines.get("split"), "metrics.baselines.split")
    slices = _require_mapping(baselines.get("slices"), "metrics.baselines.slices")
    validation_primary: float | None = None
    for split_name, expected_members in split_members.items():
        if baseline_split.get(split_name) != list(expected_members):
            raise ValueError(f"baseline metrics {split_name} membership mismatch")
        split_frame = frame.loc[frame["split"] == split_name]
        if set(split_frame["model_id"]) != set(expected_members):
            raise ValueError(
                f"prediction {split_name} membership differs from copied split"
            )
        if set(split_frame["gene_symbol"]) != scored_genes:
            raise ValueError(
                f"prediction {split_name} genes differ from the scored universe"
            )
        method_keys: dict[str, set[tuple[str, str]]] = {}
        reference_truth = None
        for method in sorted(_STAGE2_METHODS):
            method_frame = split_frame.loc[split_frame["method"] == method].copy()
            method_frame = method_frame.sort_values(
                ["model_id", "gene_symbol"], kind="stable"
            ).reset_index(drop=True)
            keys = set(
                method_frame[["model_id", "gene_symbol"]].itertuples(
                    index=False, name=None
                )
            )
            if not keys:
                raise ValueError(f"{split_name}/{method} has no evaluation rows")
            method_keys[method] = keys
            truth = method_frame[["model_id", "gene_symbol", "gene_effect", "residual"]]
            if reference_truth is None:
                reference_truth = truth
            elif not truth[["model_id", "gene_symbol"]].equals(
                reference_truth[["model_id", "gene_symbol"]]
            ) or not np.allclose(
                truth[["gene_effect", "residual"]].to_numpy(dtype=float),
                reference_truth[["gene_effect", "residual"]].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(f"truth values differ across methods in {split_name}")
        reference_keys = method_keys["e2e_full"]
        if any(keys != reference_keys for keys in method_keys.values()):
            raise ValueError(f"method evaluation keys differ in {split_name}")
        if reference_keys != expected_keys_by_split.get(split_name):
            raise ValueError(
                "method evaluation keys differ from residual label mask in "
                f"{split_name}"
            )

        e2e_frame = split_frame.loc[split_frame["method"] == "e2e_full"]
        e2e_score = score_predictions(
            e2e_frame, truth_col="residual", pred_col="residual_prediction"
        )
        metric_name = "validation" if split_name == "val" else "test"
        e2e_entry = _require_mapping(metrics.get(metric_name), f"metrics.{metric_name}")
        _verify_score_entry(
            e2e_entry, e2e_score, label=f"metrics.{metric_name}", e2e=True
        )
        if split_name == "val":
            validation_primary = float(e2e_score.macro_per_gene)

        slice_entry = _require_mapping(
            slices.get(split_name), f"metrics.baselines.slices.{split_name}"
        )
        method_entries = _require_mapping(
            slice_entry.get("methods"),
            f"metrics.baselines.slices.{split_name}.methods",
        )
        if set(method_entries) != _BASELINE_METHODS:
            raise ValueError(f"baseline metric methods mismatch in {split_name}")
        for method in sorted(_BASELINE_METHODS):
            method_frame = split_frame.loc[split_frame["method"] == method]
            score = score_predictions(
                method_frame, truth_col="residual", pred_col="residual_prediction"
            )
            entry = _require_mapping(
                method_entries[method], f"metrics.{split_name}.{method}"
            )
            _verify_score_entry(
                entry, score, label=f"metrics.{split_name}.{method}", e2e=False
            )
            if method == "gene_mean":
                if score.n_genes != 0 or entry.get("macro_per_gene") is not None:
                    raise ValueError("gene_mean per-gene metric must be undefined")
                coverage = _require_mapping(
                    entry.get("coverage"), f"metrics.{split_name}.gene_mean.coverage"
                )
                if coverage != {
                    "observed_rows": len(reference_keys),
                    "expected_rows": len(reference_keys),
                    "complete": True,
                }:
                    raise ValueError("gene_mean coverage metadata mismatch")
    if validation_primary is None or not math.isfinite(validation_primary):
        raise ValueError("validation primary metric must be finite")
    return validation_primary


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_response_array_claim(value: object, label: str) -> Mapping[str, object]:
    claim = _require_mapping(value, label)
    dtype = claim.get("dtype")
    shape = claim.get("shape")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{label}.dtype is invalid")
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape
    ):
        raise ValueError(f"{label}.shape is invalid")
    _require_sha256(claim, "content_sha256", Path(label))
    return claim


def _verify_response_lineage(root: Path) -> tuple[str, str]:
    path = root / "response_targets/lineage.json"
    payload = _read_json_object(path)
    expected_fields = {
        "schema_version",
        "response_cache_fingerprint",
        "response_cache_files",
        "source_identities",
        "train_records",
        "heldout_records",
        "record_membership_sha256",
        "target_tensors_sha256",
        "objective_weights_sha256",
        "lineage_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("response lineage fields mismatch")
    if payload.get("schema_version") != "exp13-response-lineage-v1":
        raise ValueError("response lineage schema_version mismatch")
    for field in (
        "response_cache_fingerprint",
        "record_membership_sha256",
        "target_tensors_sha256",
        "objective_weights_sha256",
        "lineage_sha256",
    ):
        _require_sha256(payload, field, path)
    cache_files = _require_mapping(
        payload.get("response_cache_files"), "response lineage cache files"
    )
    if set(cache_files) != {
        "genes.npy",
        "manifest.json",
        "metadata.parquet",
        "offsets.npy",
        "target_cells.npy",
    }:
        raise ValueError("response lineage cache-file membership mismatch")
    for field in cache_files:
        _require_sha256(cache_files, field, path)
    sources = _require_mapping(
        payload.get("source_identities"), "response lineage source identities"
    )
    if set(sources) != {
        "cell_line_manifest_sha256",
        "perturbseq_sources_sha256",
        "referenced_source_sha256",
        "tx1_cache_manifest_sha256",
        "state_var_dims_sha256",
        "stage1_run_manifest_sha256",
        "stage1_heldout_metrics_sha256",
    }:
        raise ValueError("response lineage source-identity fields mismatch")
    for field in set(sources) - {"referenced_source_sha256"}:
        _require_sha256(sources, field, path)
    referenced = _require_mapping(
        sources.get("referenced_source_sha256"),
        "response lineage referenced sources",
    )
    if not referenced:
        raise ValueError("response lineage referenced sources are empty")
    for field in referenced:
        _require_sha256(referenced, field, path)

    memberships = []
    targets = []
    weights = []
    record_ids: set[str] = set()
    for membership, field in (
        ("train", "train_records"),
        ("heldout", "heldout_records"),
    ):
        records = payload.get(field)
        if not isinstance(records, list) or not records:
            raise ValueError(f"response lineage {field} must be non-empty")
        by_model: dict[str, list[Mapping[str, object]]] = {}
        for index, value in enumerate(records):
            record = _require_mapping(value, f"response lineage {field}[{index}]")
            required_record_fields = {
                "record_id",
                "gene",
                "model_id",
                "membership",
                "anchor_weight",
                "objective_weight",
                "control_tx1",
                "observed_hvg",
                "observed_hvg_mask",
                "control_hvg",
            }
            if not required_record_fields.issubset(record):
                raise ValueError(f"response lineage {field}[{index}] is incomplete")
            record_id = record.get("record_id")
            gene = record.get("gene")
            model_id = record.get("model_id")
            if (
                not isinstance(record_id, str)
                or not isinstance(gene, str)
                or not isinstance(model_id, str)
                or record_id != f"{gene}@{model_id}"
                or record_id in record_ids
                or record.get("membership") != membership
            ):
                raise ValueError("response lineage record identity/membership mismatch")
            record_ids.add(record_id)
            anchor_weight = record.get("anchor_weight")
            objective_weight = record.get("objective_weight")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) <= 0
                for item in (anchor_weight, objective_weight)
            ):
                raise ValueError("response lineage record weights are invalid")
            for array_field in (
                "control_tx1",
                "observed_hvg",
                "observed_hvg_mask",
                "control_hvg",
            ):
                _verify_response_array_claim(
                    record.get(array_field),
                    f"response lineage {field}[{index}].{array_field}",
                )
            by_model.setdefault(model_id, []).append(record)
            memberships.append({"record_id": record_id, "membership": membership})
            targets.append(
                {
                    "record_id": record_id,
                    "observed_hvg": record["observed_hvg"],
                    "observed_hvg_mask": record["observed_hvg_mask"],
                }
            )
            weights.append(
                {
                    "record_id": record_id,
                    "anchor_weight": anchor_weight,
                    "objective_weight": objective_weight,
                }
            )
        for model_id, model_records in by_model.items():
            anchor_weight = float(model_records[0]["anchor_weight"])
            if any(
                float(record["anchor_weight"]) != anchor_weight
                for record in model_records
            ) or not math.isclose(
                sum(float(record["objective_weight"]) for record in model_records),
                anchor_weight,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"response lineage {membership} weights do not preserve "
                    f"{model_id} anchor mass"
                )
    for recorded_field, canonical_payload in (
        ("record_membership_sha256", memberships),
        ("target_tensors_sha256", targets),
        ("objective_weights_sha256", weights),
    ):
        if payload.get(recorded_field) != _canonical_json_sha256(canonical_payload):
            raise ValueError(f"response lineage {recorded_field} mismatch")
    lineage = dict(payload)
    recorded_lineage = lineage.pop("lineage_sha256")
    if recorded_lineage != _canonical_json_sha256(lineage):
        raise ValueError("response lineage canonical digest mismatch")
    return str(recorded_lineage), sha256_file(path)


def _verify_preprocessing_artifacts(
    root: Path,
    *,
    run: Mapping[str, object],
    universe: Mapping[str, object],
    feature_generation: Mapping[str, object],
    model_package: Mapping[str, object],
    warmup_metadata: Mapping[str, object],
    joint_metadata: Mapping[str, object],
    stage1_digest: str,
    joint_digest: str,
) -> None:
    """Authenticate deployable preprocessing state and its feature stores."""
    import numpy as np

    from aivc_model.geneeffect_feature_store import (
        CONTEXT_WIDTH,
        DELTA_PROJ_WIDTH,
        GENE_WIDTH,
        Q_SC_WIDTH,
        SUMMARY_WIDTH,
        verify_geneeffect_feature_store,
    )
    from aivc_model.geneeffect_features import (
        FEATURE_SCHEMA,
        BlockStandardizer,
        FixedSparseProjection,
    )

    schema = _read_json_object(root / "feature_schema.json")
    if schema != FEATURE_SCHEMA.to_dict():
        raise ValueError("feature_schema.json does not match the runtime schema")
    schema_sha256 = FEATURE_SCHEMA.schema_hash

    projection_path = root / "projection.npz"
    try:
        with np.load(projection_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"components", "metadata"}:
                raise ValueError("projection.npz keys mismatch")
            components = np.asarray(loaded["components"])
            metadata_value = np.asarray(loaded["metadata"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"projection.npz is invalid: {exc}") from exc
    if components.dtype != np.dtype(np.float32):
        raise ValueError("projection.npz components must be float32")
    if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
        raise ValueError("projection.npz metadata must be a scalar JSON string")
    try:
        projection_metadata = json.loads(str(metadata_value.item()))
    except json.JSONDecodeError as exc:
        raise ValueError("projection.npz metadata is invalid JSON") from exc
    if not isinstance(projection_metadata, dict):
        raise ValueError("projection.npz metadata must decode to an object")
    projection = FixedSparseProjection.from_state(
        {"components": components, "metadata": projection_metadata}
    )

    standardizer_path = root / "standardizer.npz"
    try:
        with np.load(standardizer_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"state"}:
                raise ValueError("standardizer.npz keys mismatch")
            state_value = np.asarray(loaded["state"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"standardizer.npz is invalid: {exc}") from exc
    if state_value.shape != () or state_value.dtype.kind not in {"U", "S"}:
        raise ValueError("standardizer.npz state must be a scalar JSON string")
    try:
        standardizer_state = json.loads(str(state_value.item()))
    except json.JSONDecodeError as exc:
        raise ValueError("standardizer.npz state is invalid JSON") from exc
    if not isinstance(standardizer_state, dict):
        raise ValueError("standardizer.npz state must decode to an object")
    standardizer = BlockStandardizer.from_state(standardizer_state)
    if standardizer.to_state() != standardizer_state:
        raise ValueError("standardizer.npz state is not canonical")
    expected_widths = {
        "delta_proj": DELTA_PROJ_WIDTH,
        "s": SUMMARY_WIDTH,
        "q_sc": Q_SC_WIDTH,
        "e_g": GENE_WIDTH,
        "z_c": CONTEXT_WIDTH,
    }
    blocks = standardizer_state.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != set(expected_widths):
        raise ValueError("standardizer.npz must contain the five deployable blocks")
    for name, width in expected_widths.items():
        block = blocks.get(name)
        if not isinstance(block, dict) or len(block.get("mean", ())) != width:
            raise ValueError(f"standardizer.npz {name} width mismatch")

    if feature_generation.get("projection") != projection.metadata:
        raise ValueError("feature_generation projection identity mismatch")
    if feature_generation.get("standardizer") != standardizer_state:
        raise ValueError("feature_generation standardizer identity mismatch")

    target_esm2_sha256 = _require_sha256(
        run, "target_esm2_sha256", root / "run_manifest.json"
    )
    expected_package_fields = {
        "projection_sha256": projection.components_hash,
        "projection_artifact_sha256": sha256_file(projection_path),
        "standardizer_sha256": standardizer.state_hash,
        "standardizer_artifact_sha256": sha256_file(standardizer_path),
        "feature_schema_sha256": schema_sha256,
        "gene_embedding_source_sha256": target_esm2_sha256,
    }
    for field, expected in expected_package_fields.items():
        if _require_sha256(
            model_package, field, root / "model_package/model_manifest.json"
        ) != expected:
            raise ValueError(f"model package {field} mismatch")

    with np.load(root / "residual_targets.npz", allow_pickle=False) as loaded:
        model_ids = _string_vector(loaded["model_ids"], "model_ids")
    genes_value = universe.get("scored_symbols")
    if not isinstance(genes_value, list):
        raise ValueError("ESM2 universe scored_symbols must be a list")
    gene_symbols = tuple(genes_value)
    if any(not isinstance(gene, str) or not gene for gene in gene_symbols):
        raise ValueError("ESM2 universe scored_symbols are invalid")

    frozen_path = root / "condition_features/stage1_frozen/manifest.json"
    selected_path = root / "condition_features/stage2_selected/manifest.json"
    frozen_manifest = _read_json_object(frozen_path)
    selected_manifest = _read_json_object(selected_path)
    if feature_generation.get("feature_manifest") != frozen_manifest:
        raise ValueError("feature_generation frozen manifest identity mismatch")
    if feature_generation.get("final_feature_manifest") != selected_manifest:
        raise ValueError("feature_generation selected manifest identity mismatch")
    frozen_sha256 = sha256_file(frozen_path)
    selected_sha256 = sha256_file(selected_path)
    if _require_sha256(
        model_package,
        "frozen_feature_manifest_sha256",
        root / "model_package/model_manifest.json",
    ) != frozen_sha256:
        raise ValueError("model package frozen feature-manifest identity mismatch")
    if _require_sha256(
        model_package,
        "feature_manifest_sha256",
        root / "model_package/model_manifest.json",
    ) != selected_sha256:
        raise ValueError("model package selected feature-manifest identity mismatch")

    frozen_shards = frozen_manifest.get("shards")
    if not isinstance(frozen_shards, dict) or set(frozen_shards) != set(model_ids):
        raise ValueError("frozen feature-store model membership mismatch")
    source_sha256 = {
        model_id: _require_sha256(
            _require_mapping(frozen_shards[model_id], f"frozen shard {model_id}"),
            "source_sha256",
            frozen_path,
        )
        for model_id in model_ids
    }
    for stage, checkpoint_digest, manifest in (
        ("stage1_frozen", stage1_digest, frozen_manifest),
        ("stage2_selected", joint_digest, selected_manifest),
    ):
        report = verify_geneeffect_feature_store(
            root / "condition_features" / stage,
            expected_stage=stage,
            expected_checkpoint_sha256=checkpoint_digest,
            expected_feature_schema_sha256=schema_sha256,
            expected_projection_sha256=projection.components_hash,
            expected_source_sha256=source_sha256,
            expected_gene_embedding_source_sha256=target_esm2_sha256,
            expected_model_ids=model_ids,
            expected_gene_symbols=gene_symbols,
        )
        if report.get("status") != "passed":
            raise ValueError(
                f"{stage} feature store verification failed: "
                f"{report.get('discrepancies')}"
            )
        if report.get("manifest") != manifest:
            raise ValueError(f"{stage} feature-store manifest read mismatch")

    checkpoint_feature_fields = {
        "manifest": frozen_sha256,
        "projection": projection.components_hash,
        "standardizer": standardizer.state_hash,
        "feature_schema": schema_sha256,
        "gene_embedding_source": target_esm2_sha256,
    }
    for phase, metadata in (
        ("warmup", warmup_metadata),
        ("joint", joint_metadata),
    ):
        provenance = _require_mapping(
            metadata.get("provenance"), f"{phase} checkpoint provenance"
        )
        feature_sha256 = _require_mapping(
            provenance.get("feature_sha256"), f"{phase} checkpoint feature hashes"
        )
        for field, expected in checkpoint_feature_fields.items():
            if feature_sha256.get(field) != expected:
                raise ValueError(f"{phase} checkpoint/{field} identity mismatch")


def _verify_runner_contract(root: Path) -> None:
    """Verify cross-artifact identities emitted by the production runner."""
    json_names = (
        "config_snapshot.json",
        "cell_line_geneeffect_226_split.json",
        "stage1_model_manifest.json",
        "stage1_objective.json",
        "esm2_gene_universe_manifest.json",
        "esm2_provenance_manifest.json",
        "esm2_uniprot_mapping.json",
        "g_var_manifest.json",
        "feature_schema.json",
        "backbone_load_report.json",
        "lambda_calibration.json",
        "feature_generation.json",
        "checkpoint_selection.json",
        "geneeffect_residual_metrics.json",
        "run_manifest.json",
        "model_package/model_manifest.json",
    )
    documents = {name: _read_json_object(root / name) for name in json_names}
    response_lineage_sha256, response_lineage_artifact_sha256 = (
        _verify_response_lineage(root)
    )
    run = documents["run_manifest.json"]
    config = documents["config_snapshot.json"]
    stage1 = documents["stage1_model_manifest.json"]
    universe = documents["esm2_gene_universe_manifest.json"]
    esm2_provenance = documents["esm2_provenance_manifest.json"]
    uniprot_mapping = documents["esm2_uniprot_mapping.json"]
    model_package = documents["model_package/model_manifest.json"]
    selection = documents["checkpoint_selection.json"]
    runtime_code_sha256 = verify_stage2_runtime_code_sha256(
        run.get("stage2_code_sha256")
    )
    if model_package.get("stage2_code_sha256") != runtime_code_sha256:
        raise ValueError("model package/run Stage 2 runtime code identity mismatch")
    embedding_union = _require_mapping(
        universe.get("embedding_union"), "ESM2 universe embedding_union"
    )
    provenance_record = _require_mapping(
        embedding_union.get("provenance_manifest"),
        "ESM2 universe provenance_manifest",
    )
    if provenance_record.get("payload") != esm2_provenance:
        raise ValueError(
            "copied ESM2 provenance payload differs from universe manifest"
        )
    if provenance_record.get("sha256") != sha256_file(
        root / "esm2_provenance_manifest.json"
    ):
        raise ValueError("copied ESM2 provenance SHA256 mismatch")
    embedding_artifact = _require_mapping(
        esm2_provenance.get("embedding_artifact"),
        "ESM2 provenance embedding_artifact",
    )
    union_symbols = embedding_union.get("symbols")
    if not isinstance(union_symbols, list) or any(
        not isinstance(symbol, str) for symbol in union_symbols
    ):
        raise ValueError("ESM2 universe embedding symbols are invalid")
    union_symbols_sha256 = hashlib.sha256(
        "".join(f"{symbol}\n" for symbol in union_symbols).encode("utf-8")
    ).hexdigest()
    if (
        union_symbols != embedding_artifact.get("symbols")
        or embedding_union.get("count") != len(union_symbols)
        or embedding_union.get("symbols_sha256") != union_symbols_sha256
    ):
        raise ValueError("ESM2 builder union differs from embedding provenance")
    if run.get("target_esm2_sha256") != embedding_artifact.get("sha256"):
        raise ValueError("run target ESM2 identity differs from copied provenance")
    sequence_source = _require_mapping(
        esm2_provenance.get("sequence_source"), "ESM2 provenance sequence_source"
    )
    mapping_json_sha256 = sha256_file(root / "esm2_uniprot_mapping.json")
    mapping_csv_sha256 = sha256_file(root / "esm2_uniprot_mapping.csv")
    if embedding_union.get("uniprot_mapping") != {
        "isoform_policy": ISOFORM_POLICY,
        "json_sha256": mapping_json_sha256,
        "csv_sha256": mapping_csv_sha256,
    }:
        raise ValueError("ESM2 universe UniProt mapping contract mismatch")
    authenticated_mapping = authenticate_uniprot_mapping(
        sequence_source,
        embedding_artifact,
        root / "esm2_uniprot_mapping.json",
        root / "esm2_uniprot_mapping.csv",
    )
    if authenticated_mapping != uniprot_mapping:
        raise ValueError("copied UniProt mapping authentication mismatch")
    resolved_mapping_symbols = {
        record["gene_symbol"]
        for record in authenticated_mapping["records"]
        if record["resolved"] is True
    }
    final_symbols = universe.get("scored_symbols")
    if not isinstance(final_symbols, list) or not set(final_symbols).issubset(
        resolved_mapping_symbols
    ):
        raise ValueError(
            "final universe contains genes without resolved UniProt mapping"
        )
    feature_generation = documents["feature_generation.json"]
    metrics = documents["geneeffect_residual_metrics.json"]
    _verify_stage1_provenance_claims(stage1, run, model_package)
    (
        expected_evaluation_keys,
        authoritative_residual,
        authoritative_mu,
    ) = _verify_residual_target_artifact(
        root,
        documents["cell_line_geneeffect_226_split.json"],
        documents["esm2_gene_universe_manifest.json"],
        run,
    )
    residual_artifact_sha256 = sha256_file(root / "residual_targets.npz")

    for phase in ("warmup", "joint"):
        if not isinstance(selection.get(phase), dict):
            raise ValueError(f"checkpoint_selection.json has invalid {phase!r} outcome")
    _, warmup_metadata = _verify_selected_checkpoint(
        root,
        training_dir="warmup/training",
        filename="head.pt",
        selection=selection["warmup"],
    )
    joint_digest, joint_metadata = _verify_selected_checkpoint(
        root,
        training_dir="joint/training",
        filename="e2e_state.pt",
        selection=selection["joint"],
    )
    distributed_runtime = _verify_distributed_runtime(run.get("distributed_runtime"))
    if (
        _verify_distributed_runtime(model_package.get("distributed_runtime"))
        != distributed_runtime
    ):
        raise ValueError("model package/run distributed_runtime mismatch")
    for phase, metadata in (
        ("warmup", warmup_metadata),
        ("joint", joint_metadata),
    ):
        provenance = _require_mapping(
            metadata.get("provenance"), f"{phase} checkpoint provenance"
        )
        if provenance.get("stage2_code_sha256") != runtime_code_sha256:
            raise ValueError(f"{phase} checkpoint/run Stage 2 code identity mismatch")
        if (
            _verify_distributed_runtime(provenance.get("distributed_runtime"))
            != distributed_runtime
        ):
            raise ValueError(f"{phase} checkpoint/run distributed_runtime mismatch")
    packaged_digest = sha256_file(root / "model_package/e2e_state.pt")
    if packaged_digest != joint_digest:
        raise ValueError("packaged model is not the selected joint checkpoint")
    if (
        _require_sha256(
            model_package,
            "checkpoint_sha256",
            root / "model_package/model_manifest.json",
        )
        != joint_digest
    ):
        raise ValueError("model package checkpoint identity mismatch")
    if (
        _require_sha256(run, "selected_checkpoint_sha256", root / "run_manifest.json")
        != joint_digest
    ):
        raise ValueError("run manifest selected-checkpoint identity mismatch")

    stage1_digest = _require_sha256(
        stage1, "checkpoint_sha256", root / "stage1_model_manifest.json"
    )
    if (
        _require_sha256(run, "stage1_checkpoint_sha256", root / "run_manifest.json")
        != stage1_digest
    ):
        raise ValueError("run manifest Stage-1 checkpoint identity mismatch")
    config_digest = _require_sha256(
        config, "source_sha256", root / "config_snapshot.json"
    )
    if (
        _require_sha256(run, "config_sha256", root / "run_manifest.json")
        != config_digest
    ):
        raise ValueError("run manifest config identity mismatch")
    if (
        _require_sha256(
            model_package, "config_sha256", root / "model_package/model_manifest.json"
        )
        != config_digest
    ):
        raise ValueError("model package config identity mismatch")
    for field in (
        "tx1_registration_sha256",
        "tx1_source_manifest_sha256",
        "tx1_cache_manifest_sha256",
        "q_sc_cache_manifest_sha256",
    ):
        run_digest = _require_sha256(run, field, root / "run_manifest.json")
        if (
            _require_sha256(
                model_package, field, root / "model_package/model_manifest.json"
            )
            != run_digest
        ):
            raise ValueError(f"model package/run {field} mismatch")
        for phase, metadata in (
            ("warmup", warmup_metadata),
            ("joint", joint_metadata),
        ):
            provenance = _require_mapping(
                metadata.get("provenance"), f"{phase} checkpoint provenance"
            )
            feature_hashes = _require_mapping(
                provenance.get("feature_sha256"),
                f"{phase} checkpoint feature hashes",
            )
            if feature_hashes.get(field) != run_digest:
                raise ValueError(f"{phase} checkpoint/run {field} mismatch")

    split_digest = sha256_file(root / "cell_line_geneeffect_226_split.json")
    for phase, metadata in (
        ("warmup", warmup_metadata),
        ("joint", joint_metadata),
    ):
        provenance = _require_mapping(
            metadata.get("provenance"), f"{phase} checkpoint provenance"
        )
        if provenance.get("split_sha256") != split_digest:
            raise ValueError(f"{phase} checkpoint/copy split identity mismatch")
        for field in (
            "residual_target_sha256",
            "centering_fit_model_ids_sha256",
            "mu_train_sha256",
        ):
            recorded = _require_sha256(run, field, root / "run_manifest.json")
            if provenance.get(field) != recorded:
                raise ValueError(f"{phase} checkpoint/run {field} mismatch")
        feature_sha256 = _require_mapping(
            provenance.get("feature_sha256"), f"{phase} checkpoint feature hashes"
        )
        if feature_sha256.get("residual_targets") != residual_artifact_sha256:
            raise ValueError(
                f"{phase} checkpoint/residual-target artifact identity mismatch"
            )
        if feature_sha256.get("response_lineage") != (
            response_lineage_artifact_sha256
        ) or feature_sha256.get("response_lineage_semantic") != (
            response_lineage_sha256
        ):
            raise ValueError(f"{phase} checkpoint/response lineage mismatch")
    joint_provenance = _require_mapping(
        joint_metadata.get("provenance"), "joint checkpoint provenance"
    )
    if (
        joint_provenance.get("lambda_calibration_report")
        != documents["lambda_calibration.json"]
    ):
        raise ValueError("joint checkpoint lambda-calibration identity mismatch")
    _verify_lambda_calibration(documents["lambda_calibration.json"])
    if (
        _require_sha256(
            model_package,
            "residual_targets_artifact_sha256",
            root / "model_package/model_manifest.json",
        )
        != residual_artifact_sha256
    ):
        raise ValueError("model package residual-target identity mismatch")
    if model_package.get("split_sha256") != split_digest:
        raise ValueError("model package split identity mismatch")
    for label, payload, path in (
        ("run", run, root / "run_manifest.json"),
        (
            "model package",
            model_package,
            root / "model_package/model_manifest.json",
        ),
    ):
        if (
            _require_sha256(payload, "response_lineage_sha256", path)
            != (response_lineage_sha256)
            or _require_sha256(payload, "response_lineage_artifact_sha256", path)
            != response_lineage_artifact_sha256
        ):
            raise ValueError(f"{label}/response lineage mismatch")

    _verify_preprocessing_artifacts(
        root,
        run=run,
        universe=universe,
        feature_generation=feature_generation,
        model_package=model_package,
        warmup_metadata=warmup_metadata,
        joint_metadata=joint_metadata,
        stage1_digest=stage1_digest,
        joint_digest=joint_digest,
    )

    required_metric_sections = {"validation", "test", "baselines", "response"}
    if not required_metric_sections.issubset(metrics):
        raise ValueError("GeneEffect metrics artifact is missing required sections")
    response = _require_mapping(metrics.get("response"), "metrics.response")
    before_response = _require_mapping(
        response.get("before_stage2"), "metrics.response.before_stage2"
    )
    after_response = _require_mapping(
        response.get("after_stage2"), "metrics.response.after_stage2"
    )
    if before_response.get("input_lineage_status") != "historical_unverified_inputs":
        raise ValueError("before-stage2 response lineage status mismatch")
    if after_response.get("input_lineage_status") != "current_authenticated_inputs":
        raise ValueError("after-stage2 response lineage status mismatch")
    if after_response.get("response_lineage_sha256") != (
        response_lineage_sha256
    ) or after_response.get("response_lineage_artifact_sha256") != (
        response_lineage_artifact_sha256
    ):
        raise ValueError("after-stage2 response lineage identity mismatch")
    before_metrics = _require_mapping(
        before_response.get("metrics"), "metrics.response.before_stage2.metrics"
    )
    after_metrics = _require_mapping(
        after_response.get("metrics"), "metrics.response.after_stage2.metrics"
    )
    for label, value in (
        ("before_stage2.metrics.model_loss", before_metrics.get("model_loss")),
        ("after_stage2.metrics.model_loss", after_metrics.get("model_loss")),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"metrics.response.{label} must be finite")
    if response.get("comparison_status") != (
        "not_comparable_historical_input_lineage_incomplete"
    ):
        raise ValueError("metrics.response comparison status mismatch")
    if response.get("delta_reported") is not False:
        raise ValueError("metrics.response.delta_reported must be false")
    if response.get("hard_guard_applied") is not False:
        raise ValueError("metrics.response.hard_guard_applied must be false")
    validation_primary = _verify_predictions_and_metrics(
        root,
        documents["cell_line_geneeffect_226_split.json"],
        documents["esm2_gene_universe_manifest.json"],
        expected_evaluation_keys,
        authoritative_residual,
        authoritative_mu,
        metrics,
    )
    if not math.isclose(
        float(joint_metadata["metric_value"]),
        validation_primary,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "selected joint checkpoint metric does not match recomputed validation"
        )


def mark_failure(layout: Stage2RunLayout, error: BaseException, *, phase: str) -> None:
    """Write one recoverable failure marker without deleting partial evidence."""
    if layout.complete.exists():
        raise ValueError("cannot mark an already completed run as failed")
    atomic_write_json(
        layout.failure,
        {
            "phase": str(phase),
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def mark_complete(
    layout: Stage2RunLayout,
    *,
    run_id: str,
    required_outputs: Sequence[str] = REQUIRED_STAGE2_OUTPUTS,
) -> Mapping[str, object]:
    """Verify and seal all outputs, then emit the sole completion sentinel."""
    if layout.failure.exists():
        raise ValueError(f"refusing to complete failed run {layout.root}")
    if layout.complete.exists():
        raise FileExistsError(f"completion sentinel already exists: {layout.complete}")
    manifest_run_id = _read_run_id(layout.root)
    if str(run_id) != manifest_run_id:
        raise ValueError(
            f"completion run_id {run_id!r} != run manifest {manifest_run_id!r}"
        )
    digests = artifact_digests(layout.root, required_outputs)
    if tuple(required_outputs) == REQUIRED_STAGE2_OUTPUTS:
        _verify_runner_contract(layout.root)
    payload: dict[str, object] = {
        "status": "complete",
        "run_id": str(run_id),
        "artifact_sha256": digests,
    }
    if tuple(required_outputs) == REQUIRED_STAGE2_OUTPUTS:
        run_manifest = _read_json_object(layout.root / "run_manifest.json")
        payload["stage2_code_sha256"] = verify_stage2_runtime_code_sha256(
            run_manifest.get("stage2_code_sha256")
        )
        for field in (
            "tx1_registration_sha256",
            "tx1_source_manifest_sha256",
            "tx1_cache_manifest_sha256",
            "q_sc_cache_manifest_sha256",
            "response_lineage_sha256",
            "response_lineage_artifact_sha256",
        ):
            payload[field] = _require_sha256(
                run_manifest, field, layout.root / "run_manifest.json"
            )
    atomic_write_json(layout.complete, payload)
    return payload


def verify_complete_run(
    path: Path,
    *,
    required_outputs: Sequence[str] = REQUIRED_STAGE2_OUTPUTS,
) -> Mapping[str, object]:
    """Authenticate a terminal run against the hashes in ``complete.json``."""
    root = Path(path)
    if (root / "failure.json").exists():
        raise ValueError(f"run {root} carries failure.json")
    complete_path = root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"run {root} has no complete.json")
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not payload.get("run_id"):
        raise ValueError(f"invalid completion sentinel: {complete_path}")
    manifest_run_id = _read_run_id(root)
    if payload["run_id"] != manifest_run_id:
        raise ValueError(
            "completion sentinel run_id does not match run_manifest.json: "
            f"{payload['run_id']!r} != {manifest_run_id!r}"
        )
    if tuple(required_outputs) == REQUIRED_STAGE2_OUTPUTS:
        run_manifest = _read_json_object(root / "run_manifest.json")
        runtime_code_sha256 = verify_stage2_runtime_code_sha256(
            run_manifest.get("stage2_code_sha256")
        )
        if payload.get("stage2_code_sha256") != runtime_code_sha256:
            raise ValueError("completion sentinel/run Stage 2 code identity mismatch")
        for field in (
            "tx1_registration_sha256",
            "tx1_source_manifest_sha256",
            "tx1_cache_manifest_sha256",
            "q_sc_cache_manifest_sha256",
            "response_lineage_sha256",
            "response_lineage_artifact_sha256",
        ):
            if payload.get(field) != _require_sha256(
                run_manifest, field, root / "run_manifest.json"
            ):
                raise ValueError(f"completion sentinel/run {field} mismatch")
    expected = payload.get("artifact_sha256")
    if not isinstance(expected, dict):
        raise ValueError("complete.json artifact_sha256 must be an object")
    actual = artifact_digests(root, required_outputs)
    if tuple(required_outputs) == REQUIRED_STAGE2_OUTPUTS:
        _verify_runner_contract(root)
    if expected != actual:
        mismatched = sorted(
            name
            for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        raise ValueError(f"completed run artifact digest mismatch: {mismatched}")
    return payload

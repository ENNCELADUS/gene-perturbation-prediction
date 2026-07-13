"""Strict frozen-manifest cross-validation orchestration for exp05."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Callable

from accelerate import Accelerator
import anndata as ad
import numpy as np
import pandas as pd
import torch

from aivc_model.gene_splits import (
    CANONICAL_GENE_COUNT,
    FINAL_RESPONSE_STAGES,
    FIT_STAGES,
    FINAL_LABEL_STAGES,
    ORACLE_FIT_STAGES,
    ORACLE_SELECTION_STAGES,
    SELECTION_STAGES,
    FoldSpec,
    GeneAccessRecorder,
    assert_gene_access,
    attach_gene_provenance,
    load_canonical_outer_manifest,
    make_inner_fold_spec,
)
from aivc_model.gwps_cache import load_gwps_cache
from aivc_model.prepare import (
    AivcConfig,
    ExternalGeneBags,
    GeneBags,
    SealedGeneBags,
    load_config,
    load_external_gene_bags,
    load_gene_bags,
)
from aivc_model.model import load_state_model
from aivc_model.train import _make_accelerator, run_training

STATE_FEATURE_COUNT = 2000
EXPECTED_STATE_PERT_DIM = 2024
EXPECTED_GWPS_SHAPE = (1_989_578, 8_248)
EXPECTED_GWPS_NONCONTROL_GENES = 9_866
EXPECTED_ADAMSON_MATCHES = {
    "adamson_pilot": 1_876,
    "adamson_upr_epistasis": 1_874,
    "adamson_upr_perturb_seq": 1_874,
}
_LOGGER = logging.getLogger(__name__)


def run_preflight(config_path: Path) -> dict[str, object]:
    """Validate every frozen exp05 asset without fitting or writing artifacts."""
    config = load_config(config_path)
    _assert_locked_preflight_config(config)
    labels = _load_preflight_labels(config)
    manifest, split_sha256 = _load_preflight_manifest(config, labels)
    canonical_genes = manifest["perturbation_gene"].to_numpy(dtype=str)
    cache_features = _validate_prepared_cache(config, manifest)
    esm_resolved = _validate_esm2_asset(config, canonical_genes)
    gwps_shape, noncontrol_count = _validate_gwps_source(config, canonical_genes)
    state_dims = _validate_state_assets(config, cache_features)
    adamson_matches = _validate_adamson_sources(config, cache_features)
    report: dict[str, object] = {
        "gwps_shape": f"{gwps_shape[0]}x{gwps_shape[1]}",
        "gwps_noncontrol_genes": noncontrol_count,
        "gwps_depmap_overlap": len(canonical_genes),
        "canonical_split_genes": len(canonical_genes),
        "canonical_split_folds": int(manifest["outer_fold"].nunique()),
        "canonical_split_sha256_length": len(split_sha256),
        "esm2_resolved": f"{esm_resolved}/{len(canonical_genes)}",
        "state_expression_matches": (f"{len(cache_features)}/{STATE_FEATURE_COUNT}"),
        "state_input_dim": state_dims[0],
        "state_output_dim": state_dims[1],
        "state_pert_dim": state_dims[2],
    }
    report.update(
        {
            f"{name}_matches": f"{count}/{STATE_FEATURE_COUNT}"
            for name, count in adamson_matches.items()
        }
    )
    return report


def _assert_locked_preflight_config(config: AivcConfig) -> None:
    if config.cv.expected_gene_count != CANONICAL_GENE_COUNT:
        raise ValueError(
            f"expected_gene_count must be {CANONICAL_GENE_COUNT} for exp05"
        )
    if config.cv.n_splits != 5:
        raise ValueError("exp05 preflight requires exactly five outer folds")
    if config.data.state_hvg_n_top_genes is not None:
        raise ValueError("variance-HVG fallback is forbidden in repaired exp05")
    if not config.train.freeze_state:
        raise ValueError("repaired exp05 requires frozen STATE")
    if config.state.gene_tokenizer != "esm2" or not config.state.require_resolved_esm2:
        raise ValueError("repaired exp05 requires strict ESM-2 coverage")
    if config.state.representation_layer != "output":
        raise ValueError("repaired exp05 fixes the STATE representation to output")


def _load_preflight_labels(config: AivcConfig) -> pd.DataFrame:
    labels = pd.read_csv(config.data.overlap_csv)
    required = {
        "perturbation_gene",
        config.data.depmap_label_col,
        config.data.matched_label_col,
    }
    if not required <= set(labels):
        missing = sorted(required - set(labels))
        raise ValueError(f"label table is missing columns {missing}")
    labels = labels.copy()
    labels["perturbation_gene"] = labels["perturbation_gene"].astype(str).str.upper()
    values = labels[config.data.depmap_label_col].to_numpy(dtype=float)
    exact = (
        len(labels) == CANONICAL_GENE_COUNT
        and labels["perturbation_gene"].nunique() == CANONICAL_GENE_COUNT
        and np.isfinite(values).all()
        and labels[config.data.matched_label_col].eq(True).all()
    )
    if not exact:
        raise ValueError("label table must contain exactly 9338 finite matched genes")
    return labels


def _load_preflight_manifest(
    config: AivcConfig,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    manifest_path, expected_sha256 = _manifest_authority(config)
    observed_sha256 = _file_sha256(manifest_path)
    if observed_sha256 != expected_sha256:
        raise ValueError("canonical manifest SHA-256 mismatch")
    manifest = pd.read_csv(manifest_path)
    if manifest.columns.tolist() != ["perturbation_gene", "outer_fold"]:
        raise ValueError("canonical manifest columns are invalid")
    manifest["perturbation_gene"] = (
        manifest["perturbation_gene"].astype(str).str.upper()
    )
    expected_folds = set(range(config.cv.n_splits))
    if (
        len(manifest) != CANONICAL_GENE_COUNT
        or manifest["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
        or set(manifest["perturbation_gene"]) != set(labels["perturbation_gene"])
        or set(manifest["outer_fold"]) != expected_folds
    ):
        raise ValueError("canonical manifest does not match labels and five folds")
    return manifest, expected_sha256


def _validate_prepared_cache(
    config: AivcConfig,
    manifest: pd.DataFrame,
) -> np.ndarray:
    cache_dir = config.data.prepared_cache_dir
    if cache_dir is None or not (cache_dir / "manifest.json").is_file():
        raise ValueError("prepared GWPS cache manifest is missing")
    genes = np.load(cache_dir / "genes.npy", allow_pickle=True).astype(str)
    folds = np.load(cache_dir / "gene_outer_folds.npy").astype(np.int64)
    features = np.load(cache_dir / "feature_names.npy", allow_pickle=True).astype(str)
    expected_genes = manifest["perturbation_gene"].to_numpy(dtype=str)
    expected_folds = manifest["outer_fold"].to_numpy(dtype=np.int64)
    if not np.array_equal(genes, expected_genes):
        raise ValueError("GWPS cache gene set/order differs from canonical manifest")
    if not np.array_equal(folds, expected_folds):
        raise ValueError("GWPS cache fold assignments differ from canonical manifest")
    if len(features) != STATE_FEATURE_COUNT or len(set(features)) != len(features):
        raise ValueError("GWPS cache must contain exactly 2000 unique STATE features")
    return features


def _validate_esm2_asset(config: AivcConfig, canonical_genes: np.ndarray) -> int:
    path = config.state.esm2_npz
    if path is None:
        raise ValueError("state.esm2_npz is required")
    with np.load(path, allow_pickle=True) as payload:
        symbols = payload["symbols"].astype(str)
        resolved = payload["resolved"].astype(bool)
        vectors = payload["vectors"]
    if (
        len(symbols) != len(canonical_genes)
        or len(set(symbols)) != len(symbols)
        or set(symbols) != set(canonical_genes)
    ):
        raise ValueError("ESM-2 gene set must exactly match the canonical manifest")
    if resolved.shape != (len(symbols),) or vectors.shape[0] != len(symbols):
        raise ValueError("ESM-2 asset arrays have inconsistent row counts")
    resolved_count = int(resolved.sum())
    if resolved_count != CANONICAL_GENE_COUNT:
        raise ValueError("ESM-2 coverage must be complete over all canonical genes")
    return resolved_count


def _validate_gwps_source(
    config: AivcConfig,
    canonical_genes: np.ndarray,
) -> tuple[tuple[int, int], int]:
    adata = ad.read_h5ad(config.data.h5ad_path, backed="r")
    try:
        shape = (int(adata.n_obs), int(adata.n_vars))
        labels = adata.obs[config.data.obs_perturbation_col].astype(str).str.upper()
    finally:
        adata.file.close()
    noncontrol = set(labels[labels != config.data.control_label.upper()])
    if shape != EXPECTED_GWPS_SHAPE:
        raise ValueError(f"unexpected frozen GWPS shape: {shape}")
    if len(noncontrol) != EXPECTED_GWPS_NONCONTROL_GENES:
        raise ValueError("unexpected GWPS non-control gene count")
    if not set(canonical_genes) <= noncontrol:
        raise ValueError("GWPS response genes are missing canonical label genes")
    return shape, len(noncontrol)


def _validate_state_assets(
    config: AivcConfig,
    cache_features: np.ndarray,
) -> tuple[int, int, int]:
    model_dir = config.state.model_dir
    if model_dir is None:
        raise ValueError("state.model_dir is required")
    with (model_dir / "var_dims.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "gene_names" not in payload:
        raise ValueError("STATE var_dims.pkl is missing gene_names")
    checkpoint_features = np.asarray(payload["gene_names"]).astype(str)
    if not np.array_equal(cache_features, checkpoint_features):
        raise ValueError("STATE checkpoint and GWPS cache feature orders differ")
    input_dim = int(payload.get("input_dim", len(checkpoint_features)))
    output_dim = int(payload.get("output_dim", len(checkpoint_features)))
    pert_dim = _checkpoint_pert_dim(config)
    expected = (STATE_FEATURE_COUNT, STATE_FEATURE_COUNT, EXPECTED_STATE_PERT_DIM)
    if (input_dim, output_dim, pert_dim) != expected:
        raise ValueError("STATE checkpoint dimensions do not match 2000/2000/2024")
    configured = (
        config.state.input_dim,
        config.state.output_dim,
        config.state.pert_dim,
    )
    if configured != expected:
        raise ValueError("configured STATE dimensions do not match the checkpoint")
    return input_dim, output_dim, pert_dim


def _checkpoint_pert_dim(config: AivcConfig) -> int:
    state = load_state_model(
        backend=config.state.backend,
        checkpoint_path=config.state.checkpoint_path,
        input_dim=STATE_FEATURE_COUNT,
        output_dim=STATE_FEATURE_COUNT,
        pert_dim=EXPECTED_STATE_PERT_DIM,
        emit_checkpoint_output=False,
    )
    raw_dim = getattr(state, "pert_dim", None)
    if raw_dim is None:
        raise ValueError("loaded STATE checkpoint does not expose pert_dim")
    return int(raw_dim)


def _validate_adamson_sources(
    config: AivcConfig,
    state_features: np.ndarray,
) -> dict[str, int]:
    if config.external_test is None:
        raise ValueError("three Adamson sources are required")
    reference = set(state_features.astype(str))
    observed: dict[str, int] = {}
    for source in config.external_test.sources:
        if source.var_gene_symbol_col is not None:
            raise ValueError("Adamson var_gene_symbol_col must be null")
        adata = ad.read_h5ad(source.h5ad_path, backed="r")
        try:
            symbols = set(adata.var_names.astype(str))
        finally:
            adata.file.close()
        count = len(reference.intersection(symbols))
        expected = EXPECTED_ADAMSON_MATCHES.get(source.name)
        if expected is None or count != expected:
            raise ValueError(f"unexpected STATE feature matches for {source.name}")
        observed[source.name] = count
    if set(observed) != set(EXPECTED_ADAMSON_MATCHES):
        raise ValueError("Adamson source set is incomplete")
    return observed


def main(argv: list[str] | None = None) -> None:
    """Run the repaired exp05 preflight or frozen five-fold CV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.preflight_only:
        for key, value in run_preflight(args.config).items():
            _LOGGER.info("%s=%s", key, value)
        return
    run_cross_validation(args.config)


def run_training_fold(
    config: AivcConfig,
    data: GeneBags,
    external: ExternalGeneBags | None,
    fold_spec: FoldSpec,
    run_dir: Path,
    source_fingerprint: str,
    accelerator: Accelerator | None = None,
    canonical_gene_order: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Run one outer fold through role-limited GeneBags views."""
    accelerator = accelerator or _make_accelerator(config)
    _prepare_fresh_run_dir(run_dir, accelerator)
    tokenizer = getattr(getattr(config, "state", None), "gene_tokenizer", None)
    if tokenizer == "esm2" and canonical_gene_order is None:
        raise ValueError("audited ESM-2 training requires canonical manifest order")
    recorder = GeneAccessRecorder(fold_spec)
    data = replace(data, access_recorder=recorder)
    train_data = data.for_genes(fold_spec.train_genes, stage="fine_tuning")
    val_data = data.for_genes(fold_spec.val_genes, stage="early_stopping")
    manifest = _manifest_from_bags(data)
    train_data = replace(
        train_data,
        metadata=attach_gene_provenance(
            train_data.metadata,
            manifest,
            "perturbation_gene",
            "fine_tuning",
        ),
    )
    val_data = replace(
        val_data,
        metadata=attach_gene_provenance(
            val_data.metadata,
            manifest,
            "perturbation_gene",
            "gene_effect",
        ),
    )
    sealed_test = SealedGeneBags(data, fold_spec.test_genes)
    return run_training(
        config,
        accelerator=accelerator,
        train_data=train_data,
        val_data=val_data,
        sealed_test=sealed_test,
        external_override=external,
        fold_spec=fold_spec,
        run_dir_override=run_dir,
        source_fingerprint=source_fingerprint,
        canonical_gene_order=(
            canonical_gene_order
            if canonical_gene_order is not None
            else tuple(str(gene).upper() for gene in data.genes)
        ),
    )


def run_cross_validation(
    config_path: Path,
    accelerator: Accelerator | None = None,
) -> Path:
    """Execute all frozen outer folds and aggregate audited final artifacts."""
    config = load_config(config_path)
    accelerator = accelerator or _make_accelerator(config)
    _run_distributed_preflight(config_path, accelerator)
    _wait_for_everyone(accelerator)
    manifest_path, expected_sha256 = _manifest_authority(config)
    if _file_sha256(manifest_path) != expected_sha256:
        raise ValueError("canonical manifest SHA-256 mismatch")
    data = _load_primary_bags(config)
    labels = pd.DataFrame(
        {
            "perturbation_gene": [str(gene).upper() for gene in data.genes],
            "depmap_gene_effect": np.asarray(data.y, dtype=float),
        }
    )
    manifest = load_canonical_outer_manifest(
        manifest_path,
        labels,
        expected_sha256,
    )
    _assert_canonical_universe(manifest)
    if isinstance(data, GeneBags):
        data = replace(
            data,
            metadata=attach_gene_provenance(
                data.metadata,
                manifest,
                "perturbation_gene",
                "gwps_response",
            ),
        )
    run_id = config.train.run_id or "state_aivc_cv"
    run_dir = config.data.output_dir / "runs" / run_id
    _prepare_fresh_run_dir(run_dir, accelerator)
    artifacts_dir = run_dir / "artifacts"
    if accelerator.is_main_process:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    _wait_for_everyone(accelerator)
    external = load_external_gene_bags(
        config,
        data,
        artifacts_dir,
        project_scvi=False,
    )
    fingerprint = experiment_source_fingerprint(config)

    folds: list[FoldSpec] = []
    outputs: list[dict[str, Path]] = []
    for outer_fold in sorted(manifest["outer_fold"].unique()):
        fold = make_inner_fold_spec(
            manifest,
            labels,
            int(outer_fold),
            config.cv.inner_val_fraction,
            config.cv.random_state,
        )
        fold_dir = run_dir / f"fold_{fold.outer_fold}"
        folds.append(fold)
        outputs.append(
            run_training_fold(
                config=config,
                data=data,
                external=external,
                fold_spec=fold,
                run_dir=fold_dir,
                source_fingerprint=fingerprint,
                accelerator=accelerator,
                canonical_gene_order=tuple(manifest["perturbation_gene"]),
            )
        )
    _run_main_process(
        accelerator,
        lambda: _aggregate_outputs(
            run_dir,
            manifest,
            folds,
            outputs,
            manifest_path,
            expected_sha256,
            fingerprint,
            config,
        ),
    )
    _wait_for_everyone(accelerator)
    return run_dir


def _load_primary_bags(config: AivcConfig) -> GeneBags:
    cache_dir = getattr(config.data, "prepared_cache_dir", None)
    if cache_dir is not None:
        return load_gwps_cache(config, cache_dir)
    return load_gene_bags(config)


def _require_fresh_run_dir(run_dir: Path) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"fresh run directory required: {run_dir}")


def _prepare_fresh_run_dir(run_dir: Path, accelerator: Accelerator) -> None:
    """Let rank zero validate/create one shared directory, then synchronize."""
    error = _main_process_error(
        accelerator,
        lambda: (
            _require_fresh_run_dir(run_dir),
            run_dir.mkdir(parents=True, exist_ok=True),
        ),
    )
    if error is not None:
        raise error
    _wait_for_everyone(accelerator)


def _wait_for_everyone(accelerator: Accelerator) -> None:
    """Synchronize ranks without passing unsupported MPS device ids to Gloo."""
    if (
        accelerator.num_processes > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and accelerator.device.type in {"cpu", "mps"}
    ):
        torch.distributed.barrier()
        return
    accelerator.wait_for_everyone()


def _run_main_process(
    accelerator: Accelerator,
    action: Callable[[], None],
) -> None:
    """Execute a shared-filesystem mutation exactly once."""
    error = _main_process_error(accelerator, action)
    if error is not None:
        raise RuntimeError(f"rank-zero shared operation failed: {error}") from error


def _run_distributed_preflight(
    config_path: Path,
    accelerator: Accelerator,
) -> None:
    """Run locked preflight once and propagate any failure to every rank."""
    if accelerator.num_processes == 1:
        run_preflight(config_path)
        return
    error = _main_process_error(accelerator, lambda: run_preflight(config_path))
    if error is not None:
        raise RuntimeError(f"locked preflight failed: {error}") from error


def _main_process_error(
    accelerator: Accelerator,
    action: Callable[[], object],
) -> Exception | None:
    """Run an action on rank zero and broadcast its original exception."""
    error: Exception | None = None
    if accelerator.is_main_process:
        try:
            action()
        except Exception as caught:
            error = caught
    if accelerator.num_processes > 1:
        values = [error]
        torch.distributed.broadcast_object_list(
            values,
            src=0,
            device=_object_broadcast_device(accelerator),
        )
        error = values[0]
    return error


def _object_broadcast_device(accelerator: Accelerator) -> torch.device:
    """Select a tensor device compatible with the active process-group backend."""
    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend:
        return accelerator.device
    return torch.device("cpu")


def _manifest_authority(config: AivcConfig) -> tuple[Path, str]:
    manifest_path = config.cv.outer_split_manifest
    sha_path = config.cv.outer_split_sha256_file
    if manifest_path is None or sha_path is None:
        raise ValueError("canonical outer manifest and SHA-256 file are required")
    text = sha_path.read_text(encoding="utf-8")
    if re.fullmatch(r"[0-9a-f]{64}\n", text) is None:
        raise ValueError("outer split SHA-256 file must contain one digest and newline")
    return manifest_path, text[:-1]


def _manifest_from_bags(data: GeneBags) -> pd.DataFrame:
    if data.gene_outer_folds is not None:
        folds = np.asarray(data.gene_outer_folds, dtype=np.int64)
    elif "outer_fold" in data.metadata:
        folds = data.metadata["outer_fold"].to_numpy(dtype=np.int64)
    else:
        raise ValueError("GeneBags must carry canonical outer_fold provenance")
    return pd.DataFrame(
        {
            "perturbation_gene": [str(gene).upper() for gene in data.genes],
            "outer_fold": folds,
        }
    )


def _assert_canonical_universe(manifest: pd.DataFrame) -> None:
    if (
        len(manifest) != CANONICAL_GENE_COUNT
        or manifest["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
    ):
        raise ValueError("canonical universe must contain exactly 9338 unique genes")


def experiment_source_fingerprint(config: AivcConfig) -> str:
    """Fingerprint every source that can alter an exp05 fold fit."""
    cache_manifest = (
        config.data.prepared_cache_dir / "manifest.json"
        if config.data.prepared_cache_dir is not None
        else None
    )
    payload = {
        "schema_version": 2,
        "gwps_cache_manifest_sha256": _file_sha256_or_none(cache_manifest),
        "label_csv_sha256": _file_sha256(config.data.overlap_csv),
        "canonical_outer_manifest_sha256": _file_sha256(
            _required_path(config.cv.outer_split_manifest, "outer split manifest")
        ),
        "canonical_sha256_file_sha256": _file_sha256(
            _required_path(
                config.cv.outer_split_sha256_file,
                "outer split SHA-256 file",
            )
        ),
        "esm2_npz": _file_stat_signature(config.state.esm2_npz),
        "checkpoint": _file_stat_signature(config.state.checkpoint_path),
        "state_sidecars": _small_state_sidecars(
            config.state.model_dir,
            excluded=(config.state.checkpoint_path, config.state.esm2_npz),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_path(path: Path | None, description: str) -> Path:
    if path is None:
        raise ValueError(f"{description} is required for source fingerprint")
    return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_sha256_or_none(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return _file_sha256(path)


def _file_stat_signature(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _small_state_sidecars(
    model_dir: Path | None,
    *,
    excluded: tuple[Path | None, ...],
) -> list[dict[str, str]]:
    if model_dir is None or not model_dir.exists():
        return []
    excluded_paths = {
        path.resolve() for path in excluded if path is not None and path.exists()
    }
    sidecars = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        if path.resolve() in excluded_paths or path.stat().st_size > 16 * 1024 * 1024:
            continue
        sidecars.append(
            {
                "path": str(path.relative_to(model_dir)),
                "sha256": _file_sha256(path),
            }
        )
    return sidecars


def _aggregate_outputs(
    run_dir: Path,
    manifest: pd.DataFrame,
    folds: list[FoldSpec],
    outputs: list[dict[str, Path]],
    manifest_path: Path,
    split_sha256: str,
    source_fingerprint: str,
    config: AivcConfig,
) -> None:
    artifacts_dir = run_dir / "artifacts"
    predictions = _concat_output(outputs, "predictions")
    metrics = _concat_output(outputs, "fold_metrics")
    access_audit = _concat_output(outputs, "fit_access_audit")
    external_qa = _concat_output(outputs, "external_alignment_qa")
    _assert_predictions(predictions, manifest)
    _assert_access_audit(access_audit, folds, config)
    _write_frame(metrics, artifacts_dir / "fold_metrics.csv")
    _write_frame(predictions, artifacts_dir / "predictions.csv")
    _write_frame(access_audit, artifacts_dir / "fit_access_audit.csv")
    _write_frame(external_qa, artifacts_dir / "external_alignment_qa.csv")
    canonical_output = artifacts_dir / "gene_splits.csv"
    canonical_output.write_bytes(manifest_path.read_bytes())
    observed_canonical_sha = hashlib.sha256(canonical_output.read_bytes()).hexdigest()
    if observed_canonical_sha != split_sha256:
        raise ValueError("emitted canonical gene split SHA-256 mismatch")
    _write_frame(_fold_role_rows(manifest, folds), artifacts_dir / "fold_roles.csv")
    _write_frame(_summarize_metrics(metrics), run_dir / "summary.csv")
    runtime_evidence = _verified_runtime_evidence(outputs)
    fold_seeds = {
        str(fold.outer_fold): int(config.cv.random_state + fold.outer_fold + 1)
        for fold in folds
    }
    payload: dict[str, Any] = {
        "canonical_split_path": str(manifest_path),
        "canonical_split_sha256": split_sha256,
        "canonical_gene_count": CANONICAL_GENE_COUNT,
        "esm_resolved_count": runtime_evidence["esm_resolved_count"],
        "esm_total_count": runtime_evidence["esm_total_count"],
        "esm_gene_order_sha256": runtime_evidence["esm_gene_order_sha256"],
        "state_input_dim": runtime_evidence["state_input_dim"],
        "state_output_dim": runtime_evidence["state_output_dim"],
        "state_pert_dim": runtime_evidence["state_pert_dim"],
        "state_feature_match_count": runtime_evidence["state_feature_match_count"],
        "source_fingerprint": source_fingerprint,
        "fold_seeds": fold_seeds,
        "checkpoint_dimensions": {
            "input_dim": runtime_evidence["state_input_dim"],
            "output_dim": runtime_evidence["state_output_dim"],
            "pert_dim": runtime_evidence["state_pert_dim"],
        },
        "exact_feature_matches": runtime_evidence["state_feature_match_count"],
        "fold_fit_audit_summaries": [
            str(paths["fit_audit_summary"]) for paths in outputs
        ],
        "artifacts": {
            "fold_metrics": str(artifacts_dir / "fold_metrics.csv"),
            "predictions": str(artifacts_dir / "predictions.csv"),
            "gene_splits": str(artifacts_dir / "gene_splits.csv"),
            "fold_roles": str(artifacts_dir / "fold_roles.csv"),
            "fit_access_audit": str(artifacts_dir / "fit_access_audit.csv"),
            "external_alignment_qa": str(artifacts_dir / "external_alignment_qa.csv"),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _concat_output(outputs: list[dict[str, Path]], key: str) -> pd.DataFrame:
    frames = [pd.read_csv(paths[key]) for paths in outputs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _verified_runtime_evidence(outputs: list[dict[str, Path]]) -> dict[str, object]:
    evidence = [
        json.loads(paths["runtime_evidence"].read_text(encoding="utf-8"))
        for paths in outputs
    ]
    if not evidence:
        raise ValueError("runtime evidence is missing")
    first = evidence[0]
    if any(item != first for item in evidence[1:]):
        raise ValueError("runtime evidence differs across outer folds")
    required = {
        "esm_resolved_count",
        "esm_total_count",
        "esm_gene_order_sha256",
        "state_input_dim",
        "state_output_dim",
        "state_pert_dim",
        "state_feature_match_count",
    }
    if not required <= set(first):
        raise ValueError("runtime evidence is incomplete")
    return first


def _assert_predictions(predictions: pd.DataFrame, manifest: pd.DataFrame) -> None:
    required = {"perturbation_gene", "outer_fold", "inner_role", "evaluation_scope"}
    if not required <= set(predictions.columns):
        raise ValueError(
            f"predictions are missing columns {sorted(required - set(predictions))}"
        )
    internal_scopes = {
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    }
    internal = predictions.loc[predictions["evaluation_scope"].isin(internal_scopes)]
    canonical = manifest.set_index("perturbation_gene")["outer_fold"]
    observed = internal["perturbation_gene"].map(canonical)
    if observed.isna().any() or not np.array_equal(
        observed.to_numpy(dtype=np.int64),
        internal["outer_fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "derived prediction outer_fold conflicts with canonical manifest"
        )
    for scope in internal_scopes:
        rows = predictions.loc[predictions["evaluation_scope"] == scope]
        if (
            rows["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
            or rows.duplicated(["perturbation_gene", "evaluation_scope"]).any()
        ):
            raise ValueError(f"each canonical gene must appear once in {scope}")


def _assert_access_audit(
    audit: pd.DataFrame,
    folds: list[FoldSpec],
    config: AivcConfig,
) -> None:
    if audit.empty:
        raise ValueError("fit access audit is empty")
    fold_by_id = {fold.outer_fold: fold for fold in folds}
    mandatory = _mandatory_audit_stages(config)
    for row in audit.to_dict("records"):
        stage = str(row["stage"])
        fold = fold_by_id[int(row["outer_fold"])]
        checkpoint_frozen = bool(row.get("checkpoint_frozen", False))
        expected_genes = _stage_genes(stage, fold)
        expected_hash = _gene_set_sha256(expected_genes)
        if (
            int(row.get("gene_count", -1)) != len(expected_genes)
            or str(row.get("gene_set_sha256")) != expected_hash
        ):
            raise ValueError(
                f"{stage} audit gene set does not match its authorized role"
            )
        assert_gene_access(stage, expected_genes, fold, checkpoint_frozen)
    for fold in folds:
        observed = set(
            audit.loc[audit["outer_fold"] == fold.outer_fold, "stage"].astype(str)
        )
        missing = sorted(mandatory - observed)
        if missing:
            raise ValueError(
                f"fold {fold.outer_fold} missing mandatory audit stages: {missing}"
            )
        unexpected = sorted(observed - mandatory)
        if unexpected:
            raise ValueError(
                f"fold {fold.outer_fold} emitted disabled audit stages: {unexpected}"
            )


def _mandatory_audit_stages(config: AivcConfig) -> frozenset[str]:
    """Derive the exact executed stage contract from the repaired config."""
    stages = {
        "adapter_fit",
        "gmm_fit",
        "normalizer_fit",
        "projector_fit",
        "transition_supervision",
        "gene_prompt_fit",
        "fine_tuning",
        "early_stopping",
        "observed_b_oracle_fit",
        "observed_b_oracle_selection",
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    }
    if config.projector.teacher == "scvi":
        stages.add("scvi_fit")
    return frozenset(stages)


def _stage_genes(stage: str, fold: FoldSpec) -> tuple[str, ...]:
    if stage == "normalizer_fit":
        return ()
    if stage in FIT_STAGES | ORACLE_FIT_STAGES:
        return fold.train_genes
    if stage in SELECTION_STAGES | ORACLE_SELECTION_STAGES:
        return fold.val_genes
    if stage in FINAL_RESPONSE_STAGES | FINAL_LABEL_STAGES:
        return fold.test_genes
    raise ValueError(f"unknown gene-access stage {stage!r}")


def _gene_set_sha256(genes: tuple[str, ...]) -> str:
    value = "\n".join(sorted(str(gene).upper() for gene in genes))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fold_role_rows(manifest: pd.DataFrame, folds: list[FoldSpec]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        roles = {
            **dict.fromkeys(fold.train_genes, "inner_train"),
            **dict.fromkeys(fold.val_genes, "inner_validation"),
            **dict.fromkeys(fold.test_genes, "outer_test"),
        }
        for row in manifest.itertuples(index=False):
            rows.append(
                {
                    "perturbation_gene": row.perturbation_gene,
                    "outer_fold": int(row.outer_fold),
                    "evaluation_outer_fold": fold.outer_fold,
                    "inner_role": roles[str(row.perturbation_gene)],
                }
            )
    return pd.DataFrame(rows)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "evaluation_scope" not in metrics:
        return pd.DataFrame()
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column != "outer_fold"
    ]
    rows = []
    for scope, frame in metrics.groupby("evaluation_scope", sort=True):
        for metric in numeric:
            values = frame[metric].dropna().to_numpy(dtype=float)
            if values.size:
                rows.append(
                    {
                        "evaluation_scope": scope,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                        "n_folds": int(values.size),
                    }
                )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()

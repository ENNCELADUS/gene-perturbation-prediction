"""Prediction-only Phase-D checkpoint reuse for post-hoc diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_embed_cache import verify_cache
from aivc_model.tx1_fewshot_calibration import (
    CALIBRATION_DIMENSION_RULE,
    CALIBRATION_SCHEMA,
    DEFAULT_MAX_COMPONENTS,
)
from aivc_model.tx1_geneeffect_eval import (
    load_slice,
    verify_artifact_hashes,
)
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead
from aivc_model.tx1_geneeffect_pipeline import (
    Tx1GeneEffectHeadConfig,
    validate_config_shape,
    validate_gene_coverage,
    validate_roles,
    validate_widths,
    verify_gene_vocabulary_authenticity,
)
from aivc_model.tx1_geneeffect_pipeline_run import (
    _ensure_fresh_run_dir,
    build_forward_only_model,
    emit_test_predictions,
)
from aivc_model.tx1_geneeffect_train_io import load_checkpoint
from aivc_model.tx1_geneeffect_train import SELECTION_METRIC_NAME
from aivc_model.tx1_predicted_response import (
    load_forward_only_checkpoint,
    resolve_device,
)


def verify_expected_head_sha256(
    head_checkpoint_path: Path, expected_head_sha256: str
) -> str:
    """Fail closed unless the existing Phase-D head has the expected digest."""
    actual = sha256_file(head_checkpoint_path)
    if actual != expected_head_sha256:
        raise ValueError(
            "Phase-D head SHA-256 mismatch: "
            f"actual={actual}, expected={expected_head_sha256}"
        )
    return actual


def load_and_verify_head_provenance(
    config: Tx1GeneEffectHeadConfig,
    head_checkpoint_path: Path,
    depmap_gene_effect_path: Path,
    head: Tx1GeneEffectHead,
) -> tuple[Path, dict[str, object]]:
    """Authenticate a Phase-D head against its sibling provenance record."""
    provenance_path = Path(head_checkpoint_path).parent.parent / "provenance.json"
    if not provenance_path.is_file():
        raise ValueError(
            f"Phase-D head companion provenance not found: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text())
    provenance_config = provenance.get("config")
    if not isinstance(provenance_config, dict):
        raise ValueError("Phase-D provenance config must be a JSON object")
    if config.state.st_checkpoint_path is None:
        raise ValueError("prediction-only provenance check requires an ST checkpoint")
    if config.objective.selection_metric != SELECTION_METRIC_NAME:
        raise ValueError(
            "prediction-only config selection_metric is not the registered constant"
        )

    expected = {
        "arm": config.arm,
        "moments": config.training.moments,
        "hidden": config.training.hidden,
        "lam": config.objective.lam,
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.epochs,
        "seed": config.training.seed,
        "selection_metric": config.objective.selection_metric,
        "st_checkpoint_sha256": sha256_file(config.state.st_checkpoint_path),
        "depmap_gene_effect_sha256": sha256_file(depmap_gene_effect_path),
    }
    actual = {
        "arm": provenance.get("arm"),
        "moments": provenance_config.get("moments"),
        "hidden": provenance_config.get("hidden"),
        "lam": provenance_config.get("lam"),
        "learning_rate": provenance_config.get("learning_rate"),
        "epochs": provenance_config.get("epochs"),
        "seed": provenance_config.get("seed"),
        "selection_metric": provenance.get("selection_metric"),
        "st_checkpoint_sha256": provenance.get("st_checkpoint_sha256"),
        "depmap_gene_effect_sha256": provenance.get(
            "depmap_gene_effect_sha256"
        ),
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ValueError(
                f"Phase-D provenance mismatch for {key}: "
                f"{actual[key]!r} vs {expected_value!r}"
            )

    actual_head_hidden = int(head.net[0].out_features)
    if head.moments != expected["moments"] or actual_head_hidden != expected["hidden"]:
        raise ValueError(
            "Phase-D head architecture does not match provenance/config: "
            f"moments={head.moments}, hidden={actual_head_hidden}"
        )
    return provenance_path, provenance


def run_prediction_pipeline(
    config: Tx1GeneEffectHeadConfig,
    *,
    head_checkpoint_path: Path,
    expected_head_sha256: str,
    line_manifest_path: Path,
    phase_a_dir: Path,
    tx1_cache_dir: Path,
    depmap_gene_effect_path: Path,
    run_dir: Path,
) -> dict[str, Path]:
    """Run diagnostic inference with existing Phase-C/D checkpoints only."""
    resolved_run_dir = Path(run_dir)
    _ensure_fresh_run_dir(resolved_run_dir)
    actual_head_sha256 = verify_expected_head_sha256(
        head_checkpoint_path, expected_head_sha256
    )
    validate_config_shape(config, check_paths=True)
    validate_widths(config, tx1_cache_dir)
    manifest, _registration, split = validate_roles(
        line_manifest_path, config.validation_lines_path
    )
    verify_artifact_hashes(Path(phase_a_dir))
    cache_report = verify_cache(
        Path(tx1_cache_dir), frozen_manifest_path=Path(line_manifest_path)
    )
    if cache_report["status"] != "verified":
        raise ValueError(
            f"Phase B basal-embedding cache at {tx1_cache_dir} failed "
            f"verification: {cache_report['discrepancies']}"
        )

    phase_a_registration = json.loads(
        (Path(phase_a_dir) / "phase_a_registration.json").read_text()
    )
    slice_df = load_slice(Path(phase_a_dir) / "differentially_essential_slice.csv")
    validate_gene_coverage(slice_df, config.state.gene_vocabulary_path)
    vocabulary_genes = json.loads(Path(config.state.gene_vocabulary_path).read_text())
    verify_gene_vocabulary_authenticity(
        config.state.gene_vocabulary_path,
        config.state.st_checkpoint_path,
        config.state.backend,
    )

    device = resolve_device(config.state.device)
    model = build_forward_only_model(config, vocabulary_genes)
    load_forward_only_checkpoint(model, config.state.st_checkpoint_path)
    model.to(device)
    head = load_checkpoint(head_checkpoint_path).to(device)
    provenance_path, _head_provenance = load_and_verify_head_provenance(
        config, head_checkpoint_path, depmap_gene_effect_path, head
    )
    expected_head_shape = (
        config.state.output_dim,
        config.state.input_dim,
        config.training.moments,
    )
    actual_head_shape = (head.response_dim, head.basal_dim, head.moments)
    if actual_head_shape != expected_head_shape:
        raise ValueError(
            "head checkpoint dimensions do not match config: "
            f"{actual_head_shape} vs {expected_head_shape}"
        )

    predictions_path = emit_test_predictions(
        config,
        model,
        head,
        manifest,
        split,
        slice_df,
        phase_a_registration,
        phase_a_dir=phase_a_dir,
        depmap_gene_effect_path=depmap_gene_effect_path,
        tx1_cache_dir=tx1_cache_dir,
        run_dir=resolved_run_dir,
        device=device,
    )
    manifest_path = resolved_run_dir / "prediction_only_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "formal": False,
                "reason": "post_hoc_adapter_after_test_opening",
                "transductive_features": True,
                "calibration": {
                    "schema": CALIBRATION_SCHEMA,
                    "max_components": DEFAULT_MAX_COMPONENTS,
                    "dimension_rule": CALIBRATION_DIMENSION_RULE,
                    "reducer_scope": "all_genes_within_each_held_out_line",
                },
                "mode": "diagnostic_prediction_only",
                "head_checkpoint": str(head_checkpoint_path),
                "head_checkpoint_sha256": actual_head_sha256,
                "expected_head_checkpoint_sha256": expected_head_sha256,
                "st_checkpoint": str(config.state.st_checkpoint_path),
                "st_checkpoint_sha256": sha256_file(config.state.st_checkpoint_path),
                "predictions_sha256": sha256_file(predictions_path),
                "phase_d_provenance": str(provenance_path),
                "phase_d_provenance_sha256": sha256_file(provenance_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "run_dir": resolved_run_dir,
        "predictions": predictions_path,
        "manifest": manifest_path,
    }

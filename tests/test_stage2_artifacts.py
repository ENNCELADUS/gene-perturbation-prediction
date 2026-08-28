from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import aivc_model.stage2_artifacts as stage2_artifacts
from aivc_model.geneeffect_feature_store import GeneEffectFeatureStoreWriter
from aivc_model.geneeffect_features import (
    FEATURE_SCHEMA,
    BlockStandardizer,
    FixedSparseProjection,
)
from aivc_model.stage2_artifacts import (
    REQUIRED_STAGE2_OUTPUTS,
    STAGE2_RUNTIME_CODE_PATHS,
    Stage2RunLayout,
    atomic_write_json,
    mark_complete,
    mark_failure,
    prepare_run_dir,
    stage2_runtime_code_sha256,
    verify_complete_run,
)


def _stage2_static_import_closure() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    pending = [
        Path("scripts/train_geneeffect_e2e.py"),
        Path("src/aivc_model/__init__.py"),
    ]
    closure: set[str] = set()
    while pending:
        relative = pending.pop()
        normalized = relative.as_posix()
        if normalized in closure:
            continue
        closure.add(normalized)
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("aivc_model"):
                continue
            imported = Path("src") / Path(node.module.replace(".", "/") + ".py")
            if (root / imported).is_file():
                pending.append(imported)
    return closure


def test_runtime_code_manifest_matches_stage2_static_import_closure() -> None:
    assert set(STAGE2_RUNTIME_CODE_PATHS) == _stage2_static_import_closure()


def _write_required(root: Path, required: tuple[str, ...]) -> None:
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())


def _build_feature_store(
    root: Path,
    *,
    stage: str,
    checkpoint_sha256: str,
    model_ids: tuple[str, ...],
    gene_symbols: tuple[str, ...],
    projection_sha256: str,
    gene_embedding_source_sha256: str,
) -> dict[str, object]:
    writer = GeneEffectFeatureStoreWriter(
        root,
        stage=stage,
        model_ids=model_ids,
        gene_symbols=gene_symbols,
        e_g=np.zeros((len(gene_symbols), 1_280), dtype=np.float32),
        z_c=np.zeros((len(model_ids), 5_120), dtype=np.float32),
        gene_embedding_source_sha256=gene_embedding_source_sha256,
        feature_schema_sha256=FEATURE_SCHEMA.schema_hash,
        projection_sha256=projection_sha256,
    )
    for model_id in model_ids:
        writer.write_shard(
            model_id,
            delta_proj=np.zeros((len(gene_symbols), 256), dtype=np.float32),
            s=np.zeros((len(gene_symbols), 6), dtype=np.float32),
            q_sc=np.zeros((len(gene_symbols), 3), dtype=np.float32),
            q_sc_mask=np.ones(len(gene_symbols), dtype=bool),
            hvg_panel_mask=np.ones(len(gene_symbols), dtype=bool),
            own_gene_shift_mask=np.ones(len(gene_symbols), dtype=bool),
            source_sha256="d" * 64,
            model_checkpoint_sha256=checkpoint_sha256,
        )
    return dict(writer.finalize())


def _write_response_lineage(root: Path) -> tuple[str, str]:
    array_claim = {
        "dtype": "<f4",
        "shape": [1, 2],
        "content_sha256": "a" * 64,
    }

    def record(gene: str, membership: str) -> dict[str, object]:
        return {
            "record_id": f"{gene}@ACH-1",
            "gene": gene,
            "model_id": "ACH-1",
            "membership": membership,
            "anchor_weight": 1.0,
            "objective_weight": 1.0,
            "control_tx1": dict(array_claim),
            "observed_hvg": dict(array_claim),
            "observed_hvg_mask": {
                "dtype": "|b1",
                "shape": [1, 2],
                "content_sha256": "b" * 64,
            },
            "control_hvg": dict(array_claim),
        }

    train = [record("G0", "train")]
    heldout = [record("G1", "heldout")]
    records = [*train, *heldout]
    memberships = [
        {"record_id": item["record_id"], "membership": item["membership"]}
        for item in records
    ]
    targets = [
        {
            "record_id": item["record_id"],
            "observed_hvg": item["observed_hvg"],
            "observed_hvg_mask": item["observed_hvg_mask"],
        }
        for item in records
    ]
    weights = [
        {
            "record_id": item["record_id"],
            "anchor_weight": item["anchor_weight"],
            "objective_weight": item["objective_weight"],
        }
        for item in records
    ]
    payload: dict[str, object] = {
        "schema_version": "exp13-response-lineage-v1",
        "response_cache_fingerprint": "c" * 64,
        "response_cache_files": {
            name: str(index) * 64
            for index, name in enumerate(
                (
                    "genes.npy",
                    "manifest.json",
                    "metadata.parquet",
                    "offsets.npy",
                    "target_cells.npy",
                ),
                1,
            )
        },
        "source_identities": {
            "cell_line_manifest_sha256": "1" * 64,
            "perturbseq_sources_sha256": "2" * 64,
            "referenced_source_sha256": {"source.h5ad": "3" * 64},
            "tx1_cache_manifest_sha256": "4" * 64,
            "state_var_dims_sha256": "5" * 64,
            "stage1_run_manifest_sha256": "6" * 64,
            "stage1_heldout_metrics_sha256": "7" * 64,
        },
        "train_records": train,
        "heldout_records": heldout,
        "record_membership_sha256": stage2_artifacts._canonical_json_sha256(
            memberships
        ),
        "target_tensors_sha256": stage2_artifacts._canonical_json_sha256(targets),
        "objective_weights_sha256": stage2_artifacts._canonical_json_sha256(weights),
    }
    payload["lineage_sha256"] = stage2_artifacts._canonical_json_sha256(payload)
    path = root / "response_targets/lineage.json"
    atomic_write_json(path, payload)
    return str(payload["lineage_sha256"]), stage2_artifacts.sha256_file(path)


def _write_full_runner_artifacts(
    layout: Stage2RunLayout, *, world_size: int = 4
) -> None:
    root = layout.root
    warmup_checkpoint = root / "warmup/training/best/head.pt"
    joint_checkpoint = root / "joint/training/best/e2e_state.pt"
    warmup_checkpoint.parent.mkdir(parents=True)
    joint_checkpoint.parent.mkdir(parents=True)
    warmup_checkpoint.write_bytes(b"warmup")
    joint_checkpoint.write_bytes(b"joint")
    from aivc_model.stage2_artifacts import sha256_file

    warmup_sha = sha256_file(warmup_checkpoint)
    joint_sha = sha256_file(joint_checkpoint)
    stage1_sha = "1" * 64
    config_sha = "2" * 64
    stage1_claims = {
        "training_code_provenance_status": "unavailable",
        "training_code_provenance_reason": (
            "historical_run_has_no_immutable_training_code_identity"
        ),
        "training_data_provenance_status": "incomplete",
        "training_data_provenance_missing_identities": [
            "cell_line_manifest",
            "tx1_basal_cache",
            "response_cache",
            "perturbseq_source_content",
        ],
        "training_data_provenance_reason": (
            "historical_run_manifest_does_not_hash_all_training_data_inputs"
        ),
    }
    propagated_stage1_claims = {
        f"stage1_{field}": value for field, value in stage1_claims.items()
    }
    authoritative_split = (
        Path(__file__).resolve().parents[1]
        / "configs/benchmarks/cell_line_geneeffect_226_split.json"
    )
    split_path = root / "cell_line_geneeffect_226_split.json"
    shutil.copy2(authoritative_split, split_path)
    split_payload = json.loads(split_path.read_text())
    split_members = {
        "val": split_payload["val"],
        "test": split_payload["test"],
    }
    genes = ("G0", "G1", "G2")
    model_ids = tuple(
        [*split_payload["train"], *split_payload["val"], *split_payload["test"]]
    )
    centering_ids = tuple(
        model_id
        for model_id in split_payload["train"]
        if model_id not in set(split_payload["unlabeled_train"])
    )
    targets = np.zeros((len(genes), len(model_ids)), dtype=np.float32)
    label_mask = np.ones_like(targets, dtype=bool)
    model_column = {model_id: index for index, model_id in enumerate(model_ids)}
    for split_name in ("train", "val", "test"):
        for model_index, model_id in enumerate(split_payload[split_name]):
            targets[:, model_column[model_id]] = model_index + np.arange(len(genes))
    for model_id in split_payload["unlabeled_train"]:
        label_mask[:, model_column[model_id]] = False
        targets[:, model_column[model_id]] = 0.0
    mu_train = np.zeros(len(genes), dtype=np.float64)
    residual_path = root / "residual_targets.npz"
    np.savez(
        residual_path,
        gene_symbols=np.asarray(genes),
        model_ids=np.asarray(model_ids),
        residual_targets=targets,
        label_mask=label_mask,
        mu_train=mu_train,
        centering_model_ids=np.asarray(centering_ids),
    )
    target_digest = hashlib.sha256()
    target_digest.update("\n".join(genes).encode())
    target_digest.update("\n".join(model_ids).encode())
    target_digest.update(targets.tobytes())
    target_digest.update(label_mask.tobytes())
    mu_digest = hashlib.sha256()
    mu_digest.update("\n".join(genes).encode())
    mu_digest.update(mu_train.tobytes())
    residual_sha = target_digest.hexdigest()
    centering_sha = hashlib.sha256("\n".join(centering_ids).encode()).hexdigest()
    mu_sha = mu_digest.hexdigest()
    response_lineage_sha, response_lineage_artifact_sha = _write_response_lineage(root)
    distributed_runtime = {
        "world_size": world_size,
        "mixed_precision": "bf16",
        "conditions_per_rank": 256,
        "global_conditions_per_step": 256 * world_size,
        "rank_topology": [
            {
                "rank": rank,
                "local_rank": rank,
                "device": f"cuda:{rank}",
                "device_name": "NVIDIA H20",
                "hostname": "hpc",
            }
            for rank in range(world_size)
        ],
    }
    cache_identities = {
        "tx1_registration_sha256": "1" * 64,
        "tx1_source_manifest_sha256": "2" * 64,
        "tx1_cache_manifest_sha256": "3" * 64,
        "q_sc_cache_manifest_sha256": "4" * 64,
    }
    target_esm2_sha = "e" * 64
    projection = FixedSparseProjection()
    standardizer = BlockStandardizer().fit(
        {
            "delta_proj": np.zeros((2, 256), dtype=np.float32),
            "s": np.zeros((2, 6), dtype=np.float32),
            "q_sc": np.zeros((2, 3), dtype=np.float32),
            "e_g": np.zeros((2, 1_280), dtype=np.float32),
            "z_c": np.zeros((2, 5_120), dtype=np.float32),
        }
    )
    frozen = _build_feature_store(
        root / "condition_features/stage1_frozen",
        stage="stage1_frozen",
        checkpoint_sha256=stage1_sha,
        model_ids=model_ids,
        gene_symbols=genes,
        projection_sha256=projection.components_hash,
        gene_embedding_source_sha256=target_esm2_sha,
    )
    selected = _build_feature_store(
        root / "condition_features/stage2_selected",
        stage="stage2_selected",
        checkpoint_sha256=joint_sha,
        model_ids=model_ids,
        gene_symbols=genes,
        projection_sha256=projection.components_hash,
        gene_embedding_source_sha256=target_esm2_sha,
    )
    provenance = {
        "stage2_code_sha256": stage2_runtime_code_sha256(),
        "split_sha256": sha256_file(split_path),
        "residual_target_sha256": residual_sha,
        "centering_fit_model_ids_sha256": centering_sha,
        "mu_train_sha256": mu_sha,
        "feature_sha256": {
            "manifest": sha256_file(
                root / "condition_features/stage1_frozen/manifest.json"
            ),
            "projection": projection.components_hash,
            "standardizer": standardizer.state_hash,
            "feature_schema": FEATURE_SCHEMA.schema_hash,
            "gene_embedding_source": target_esm2_sha,
            "residual_targets": sha256_file(residual_path),
            "response_lineage": response_lineage_artifact_sha,
            "response_lineage_semantic": response_lineage_sha,
            **cache_identities,
        },
        "distributed_runtime": distributed_runtime,
    }
    calibration = {
        "lambda_dep": 1.0,
        "raw_ratios": [1.0] * 8,
        "response_gradient_norms": [1.0] * 8,
        "dependency_gradient_norms": [1.0] * 8,
    }
    atomic_write_json(
        warmup_checkpoint.parent / "metadata.json",
        {
            "epoch": 0,
            "metric_value": 0.5,
            "checkpoint_sha256": warmup_sha,
            "selection_name": "validation_macro_per_gene_spearman",
            "selection_direction": "maximize",
            "provenance": {**provenance, "lambda_calibration_report": None},
        },
    )
    atomic_write_json(
        joint_checkpoint.parent / "metadata.json",
        {
            "epoch": 0,
            "metric_value": 1.0,
            "checkpoint_sha256": joint_sha,
            "selection_name": "validation_macro_per_gene_spearman",
            "selection_direction": "maximize",
            "provenance": {
                **provenance,
                "lambda_calibration_report": calibration,
            },
        },
    )
    selection_column = "validation_macro_per_gene_spearman"
    (root / "warmup/training/train_log.csv").write_text(
        f"epoch,{selection_column}\n0,0.5\n"
    )
    (root / "joint/training/train_log.csv").write_text(
        f"epoch,{selection_column}\n0,1.0\n"
    )
    e2e_metrics = {
        split_name: {
            "macro_per_line_spearman": 1.0,
            "macro_per_gene_spearman": 1.0,
            "per_line_defined": 27,
            "per_gene_defined": 3,
            "per_line_undefined": 0,
            "per_gene_undefined": 0,
            "per_line_spearman": {model_id: 1.0 for model_id in model_ids},
        }
        for split_name, model_ids in split_members.items()
    }
    baseline_metrics: dict[str, object] = {
        "outer": "fixed",
        "split": split_members,
        "slices": {},
    }
    for split_name in split_members:
        methods: dict[str, object] = {}
        for method in (
            "gene_mean",
            "copy_prior",
            "nearest_line[z_c]",
            "context_pca_ridge[z_c]",
        ):
            constant = method in {"gene_mean", "copy_prior"}
            entry: dict[str, object] = {
                "macro_per_line": 1.0,
                "macro_per_gene": None if constant else 1.0,
                "n_lines": 27,
                "n_genes": 0 if constant else 3,
                "n_line_undefined": 0,
                "n_gene_undefined": 3 if constant else 0,
            }
            if method == "gene_mean":
                entry.update(
                    {
                        "evaluation_status": "not_evaluable_constant_prediction",
                        "coverage": {
                            "observed_rows": 81,
                            "expected_rows": 81,
                            "complete": True,
                        },
                    }
                )
            methods[method] = entry
        baseline_metrics["slices"][split_name] = {"methods": methods}
    uniprot_records = [
        {
            "gene_symbol": f"G{index}",
            "resolved": True,
            "primary_accession": f"P{index:05d}",
            "entry_id": f"G{index}_HUMAN",
            "isoform_identifier": f"P{index:05d}",
            "isoform_policy": "canonical_reviewed_top_hit",
            "sequence_sha256": str(index) * 64,
        }
        for index in range(3)
    ]
    uniprot_mapping = {
        "schema_version": "esm2-uniprot-mapping-v1",
        "records": uniprot_records,
    }
    uniprot_mapping_json = (
        json.dumps(uniprot_mapping, indent=2, sort_keys=True) + "\n"
    ).encode()
    uniprot_mapping_csv = (
        "gene_symbol,resolved,primary_accession,entry_id,isoform_identifier,"
        "isoform_policy,sequence_sha256\n"
        + "".join(
            f"G{index},True,P{index:05d},G{index}_HUMAN,P{index:05d},"
            f"canonical_reviewed_top_hit,{str(index) * 64}\n"
            for index in range(3)
        )
    ).encode()
    esm2_provenance = {
        "schema_version": "esm2-embedding-provenance-v1",
        "embedding_artifact": {
            "sha256": target_esm2_sha,
            "symbols": ["G0", "G1", "G2"],
            "resolved_symbols": ["G0", "G1", "G2"],
        },
        "sequence_source": {
            "sequence_sha256_by_symbol": {
                f"G{index}": str(index) * 64 for index in range(3)
            },
            "uniprot_mapping_json_sha256": hashlib.sha256(
                uniprot_mapping_json
            ).hexdigest(),
            "uniprot_mapping_csv_sha256": hashlib.sha256(
                uniprot_mapping_csv
            ).hexdigest(),
        },
    }
    esm2_provenance_sha = hashlib.sha256(
        (json.dumps(esm2_provenance, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    for name, payload in {
        "config_snapshot.json": {
            "source_sha256": config_sha,
            "distributed": {"mixed_precision": "bf16"},
            "joint": {"conditions_per_rank": 256},
        },
        "stage1_model_manifest.json": {
            "checkpoint_sha256": stage1_sha,
            **stage1_claims,
        },
        "stage1_objective.json": {"objective": "response"},
        "esm2_gene_universe_manifest.json": {
            "scored_gene_count": 3,
            "scored_symbols": ["G0", "G1", "G2"],
            "embedding_union": {
                "symbols": ["G0", "G1", "G2"],
                "count": 3,
                "symbols_sha256": hashlib.sha256(b"G0\nG1\nG2\n").hexdigest(),
                "uniprot_mapping": {
                    "isoform_policy": "canonical_reviewed_top_hit",
                    "json_sha256": hashlib.sha256(uniprot_mapping_json).hexdigest(),
                    "csv_sha256": hashlib.sha256(uniprot_mapping_csv).hexdigest(),
                },
                "provenance_manifest": {
                    "sha256": esm2_provenance_sha,
                    "payload": esm2_provenance,
                },
            },
        },
        "esm2_provenance_manifest.json": esm2_provenance,
        "esm2_uniprot_mapping.json": uniprot_mapping,
        "g_var_manifest.json": {"version": 1},
        "feature_schema.json": FEATURE_SCHEMA.to_dict(),
        "backbone_load_report.json": {"loaded_keys": ["state_adapter.weight"]},
        "lambda_calibration.json": calibration,
        "feature_generation.json": {
            "feature_manifest": frozen,
            "final_feature_manifest": selected,
            "projection": projection.metadata,
            "standardizer": standardizer.to_state(),
        },
        "checkpoint_selection.json": {
            "warmup": {"best_epoch": 0, "best_metric": 0.5},
            "joint": {"best_epoch": 0, "best_metric": 1.0},
        },
        "geneeffect_residual_metrics.json": {
            "validation": e2e_metrics["val"],
            "test": e2e_metrics["test"],
            "baselines": baseline_metrics,
            "response": {
                "before_stage2": {
                    "input_lineage_status": "historical_unverified_inputs",
                    "metrics": {"model_loss": 0.5},
                },
                "after_stage2": {
                    "input_lineage_status": "current_authenticated_inputs",
                    "metrics": {"model_loss": 0.6},
                    "response_lineage_sha256": response_lineage_sha,
                    "response_lineage_artifact_sha256": (response_lineage_artifact_sha),
                },
                "comparison_status": (
                    "not_comparable_historical_input_lineage_incomplete"
                ),
                "delta_reported": False,
                "hard_guard_applied": False,
            },
        },
    }.items():
        atomic_write_json(root / name, payload)
    (root / "esm2_uniprot_mapping.csv").write_bytes(uniprot_mapping_csv)
    np.savez(
        root / "projection.npz",
        components=projection.components,
        metadata=np.asarray(json.dumps(projection.metadata, sort_keys=True)),
    )
    np.savez(
        root / "standardizer.npz",
        state=np.asarray(json.dumps(standardizer.to_state(), sort_keys=True)),
    )
    prediction_rows = [
        "split,method,model_id,gene_symbol,gene_effect,residual,residual_prediction"
    ]
    for split_name, model_ids in split_members.items():
        for method in (
            "e2e_full",
            "gene_mean",
            "copy_prior",
            "nearest_line[z_c]",
            "context_pca_ridge[z_c]",
        ):
            for model_index, model_id in enumerate(model_ids):
                for gene_index in range(3):
                    truth = float(model_index + gene_index)
                    prediction = (
                        float(gene_index)
                        if method in {"gene_mean", "copy_prior"}
                        else truth
                    )
                    prediction_rows.append(
                        f"{split_name},{method},{model_id},G{gene_index},"
                        f"{truth},{truth},{prediction}"
                    )
    (root / "geneeffect_residual_predictions.csv").write_text(
        "\n".join(prediction_rows) + "\n"
    )
    packaged = root / "model_package/e2e_state.pt"
    packaged.parent.mkdir(exist_ok=True)
    packaged.write_bytes(joint_checkpoint.read_bytes())
    frozen_manifest = root / "condition_features/stage1_frozen/manifest.json"
    selected_manifest = root / "condition_features/stage2_selected/manifest.json"
    atomic_write_json(
        root / "model_package/model_manifest.json",
        {
            "checkpoint_sha256": joint_sha,
            "config_sha256": config_sha,
            "split_sha256": sha256_file(split_path),
            "projection_sha256": projection.components_hash,
            "projection_artifact_sha256": sha256_file(root / "projection.npz"),
            "standardizer_sha256": standardizer.state_hash,
            "standardizer_artifact_sha256": sha256_file(
                root / "standardizer.npz"
            ),
            "feature_schema_sha256": FEATURE_SCHEMA.schema_hash,
            "gene_embedding_source_sha256": target_esm2_sha,
            "frozen_feature_manifest_sha256": sha256_file(frozen_manifest),
            "feature_manifest_sha256": sha256_file(selected_manifest),
            "residual_targets_artifact_sha256": sha256_file(residual_path),
            "response_lineage_sha256": response_lineage_sha,
            "response_lineage_artifact_sha256": response_lineage_artifact_sha,
            "distributed_runtime": distributed_runtime,
            "stage2_code_sha256": stage2_runtime_code_sha256(),
            **cache_identities,
            **propagated_stage1_claims,
        },
    )
    atomic_write_json(
        root / "run_manifest.json",
        {
            "run_id": "formal",
            "config_sha256": config_sha,
            "split_sha256": sha256_file(split_path),
            "stage1_checkpoint_sha256": stage1_sha,
            "selected_checkpoint_sha256": joint_sha,
            "target_esm2_sha256": target_esm2_sha,
            "residual_targets_artifact_sha256": sha256_file(residual_path),
            "residual_target_sha256": residual_sha,
            "centering_fit_model_ids_sha256": centering_sha,
            "mu_train_sha256": mu_sha,
            "response_lineage_sha256": response_lineage_sha,
            "response_lineage_artifact_sha256": response_lineage_artifact_sha,
            "distributed_runtime": distributed_runtime,
            "stage2_code_sha256": stage2_runtime_code_sha256(),
            **cache_identities,
            **propagated_stage1_claims,
        },
    )


def test_atomic_write_json_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"z": 1, "a": 2})
    assert path.read_text() == '{\n  "a": 2,\n  "z": 1\n}\n'
    assert not (path.parent / ".value.json.tmp").exists()


def test_prepare_refuses_existing_without_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    prepare_run_dir(run)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_run_dir(run)


def test_resume_requires_authenticated_incomplete_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="run_manifest"):
        prepare_run_dir(run, resume=True)
    atomic_write_json(run / "run_manifest.json", {"run_id": "r"})
    layout = prepare_run_dir(run, resume=True)
    assert layout.root == run


def test_failed_or_complete_run_cannot_resume(tmp_path: Path) -> None:
    failed = Stage2RunLayout(tmp_path / "failed")
    failed.root.mkdir()
    atomic_write_json(failed.root / "run_manifest.json", {"run_id": "r"})
    mark_failure(failed, RuntimeError("boom"), phase="joint")
    with pytest.raises(ValueError, match="failed run"):
        prepare_run_dir(failed.root, resume=True)

    complete = tmp_path / "complete"
    complete.mkdir()
    atomic_write_json(complete / "run_manifest.json", {"run_id": "r"})
    atomic_write_json(complete / "complete.json", {"status": "complete"})
    with pytest.raises(ValueError, match="completed run"):
        prepare_run_dir(complete, resume=True)


def test_complete_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    required = ("a.json", "nested/b.bin")
    layout = prepare_run_dir(tmp_path / "run")
    atomic_write_json(layout.root / "run_manifest.json", {"run_id": "formal"})
    _write_required(layout.root, required)
    payload = mark_complete(layout, run_id="formal", required_outputs=required)
    assert payload["status"] == "complete"
    assert (
        verify_complete_run(layout.root, required_outputs=required)["run_id"]
        == "formal"
    )

    (layout.root / "nested/b.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_complete_run(layout.root, required_outputs=required)


def test_completion_rejects_run_id_mismatch(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    atomic_write_json(layout.root / "run_manifest.json", {"run_id": "manifest-id"})
    with pytest.raises(ValueError, match="run_id"):
        mark_complete(layout, run_id="sentinel-id", required_outputs=())


def test_condition_feature_shards_are_sealed_recursively(tmp_path: Path) -> None:
    required = (
        "run_manifest.json",
        "condition_features/stage1_frozen/manifest.json",
    )
    layout = prepare_run_dir(tmp_path / "run")
    atomic_write_json(layout.root / "run_manifest.json", {"run_id": "r"})
    atomic_write_json(
        layout.condition_features / "stage1_frozen" / "manifest.json",
        {"version": 1},
    )
    shard = layout.condition_features / "shards" / "ACH-1.npz"
    shard.parent.mkdir()
    shard.write_bytes(b"feature shard")
    payload = mark_complete(layout, run_id="r", required_outputs=required)
    assert "condition_features/shards/ACH-1.npz" in payload["artifact_sha256"]
    shard.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_complete_run(layout.root, required_outputs=required)


def test_failure_marker_preserves_partial_outputs(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    partial = layout.joint / "train_log.csv"
    partial.write_text("epoch,loss\n")
    mark_failure(layout, ValueError("bad batch"), phase="joint")
    failure = json.loads(layout.failure.read_text())
    assert failure == {
        "error": "bad batch",
        "error_type": "ValueError",
        "phase": "joint",
    }
    assert partial.is_file()
    with pytest.raises(ValueError, match="failed run"):
        mark_complete(layout, run_id="r", required_outputs=())


@pytest.mark.parametrize("world_size", [2, 4])
def test_default_completion_verifies_full_runner_contract(
    tmp_path: Path, world_size: int
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout, world_size=world_size)
    complete = mark_complete(layout, run_id="formal")
    assert complete["stage2_code_sha256"] == stage2_runtime_code_sha256()
    assert verify_complete_run(layout.root)["status"] == "complete"


def test_terminal_verifier_rejects_changed_response_training_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    mark_complete(layout, run_id="formal")
    changed = stage2_runtime_code_sha256()
    changed["src/aivc_model/response_training.py"] = "0" * 64
    monkeypatch.setattr(
        stage2_artifacts,
        "stage2_runtime_code_sha256",
        lambda: changed,
    )
    with pytest.raises(ValueError, match="response_training.py"):
        verify_complete_run(layout.root)


@pytest.mark.parametrize(
    "artifact",
    (
        "model_package/model_manifest.json",
        "warmup/training/best/metadata.json",
        "joint/training/best/metadata.json",
    ),
)
def test_completion_cross_checks_runtime_code_identity(
    tmp_path: Path, artifact: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    path = layout.root / artifact
    payload = json.loads(path.read_text())
    code = (
        payload["provenance"]["stage2_code_sha256"]
        if "metadata.json" in artifact
        else payload["stage2_code_sha256"]
    )
    code["src/aivc_model/response_training.py"] = "0" * 64
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="Stage 2 .*code identity mismatch"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_builder_uniprot_mapping_mismatch(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    universe_path = layout.root / "esm2_gene_universe_manifest.json"
    universe = json.loads(universe_path.read_text())
    universe["embedding_union"]["uniprot_mapping"]["json_sha256"] = "0" * 64
    atomic_write_json(universe_path, universe)
    with pytest.raises(ValueError, match="UniProt mapping contract mismatch"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_builder_embedding_union_mismatch(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    universe_path = layout.root / "esm2_gene_universe_manifest.json"
    universe = json.loads(universe_path.read_text())
    universe["embedding_union"]["symbols"] = ["G0", "G1", "WRONG"]
    atomic_write_json(universe_path, universe)
    with pytest.raises(ValueError, match="builder union differs"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_uniprot_json_csv_identity_mismatch(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    csv_path = layout.root / "esm2_uniprot_mapping.csv"
    csv_path.write_text(
        csv_path.read_text().replace("G0_HUMAN", "WRONG_HUMAN"),
        encoding="utf-8",
    )
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    provenance_path = layout.root / "esm2_provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["sequence_source"]["uniprot_mapping_csv_sha256"] = csv_sha256
    atomic_write_json(provenance_path, provenance)
    provenance = json.loads(provenance_path.read_text())
    universe_path = layout.root / "esm2_gene_universe_manifest.json"
    universe = json.loads(universe_path.read_text())
    universe["embedding_union"]["uniprot_mapping"]["csv_sha256"] = csv_sha256
    universe["embedding_union"]["provenance_manifest"] = {
        "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "payload": provenance,
    }
    atomic_write_json(universe_path, universe)
    with pytest.raises(ValueError, match="JSON/CSV mapping mismatch"):
        mark_complete(layout, run_id="formal")


def test_default_completion_binds_cache_and_encoder_identities(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    model_manifest_path = layout.root / "model_package/model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    model_manifest["tx1_source_manifest_sha256"] = "0" * 64
    atomic_write_json(model_manifest_path, model_manifest)
    with pytest.raises(ValueError, match="tx1_source_manifest_sha256 mismatch"):
        mark_complete(layout, run_id="formal")

    layout = prepare_run_dir(tmp_path / "run-valid")
    _write_full_runner_artifacts(layout)
    mark_complete(layout, run_id="formal")
    complete = json.loads(layout.complete.read_text())
    complete["q_sc_cache_manifest_sha256"] = "0" * 64
    atomic_write_json(layout.complete, complete)
    with pytest.raises(ValueError, match="completion sentinel/run"):
        verify_complete_run(layout.root)


@pytest.mark.parametrize(
    "relative",
    (
        "lambda_calibration.json",
        "warmup/training/best/head.pt",
        "joint/training/train_log.csv",
        "condition_features/stage2_selected/manifest.json",
        "residual_targets.npz",
        "model_package/e2e_state.pt",
    ),
)
def test_default_completion_rejects_missing_runner_artifact(
    tmp_path: Path, relative: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    (layout.root / relative).unlink()
    with pytest.raises(FileNotFoundError, match="required Stage 2 artifact"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_false_checkpoint_identity(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metadata_path = layout.root / "joint/training/best/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_sha256"] = "0" * 64
    atomic_write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="selected checkpoint SHA256 mismatch"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_false_feature_shard_identity(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    manifest_path = layout.root / "condition_features/stage2_selected/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first_model_id = manifest["model_ids"][0]
    manifest["shards"][first_model_id]["sha256"] = "0" * 64
    atomic_write_json(manifest_path, manifest)
    feature_generation_path = layout.root / "feature_generation.json"
    generation = json.loads(feature_generation_path.read_text())
    generation["final_feature_manifest"] = manifest
    atomic_write_json(feature_generation_path, generation)
    model_manifest_path = layout.root / "model_package/model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    from aivc_model.stage2_artifacts import sha256_file

    model_manifest["feature_manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(model_manifest_path, model_manifest)
    with pytest.raises(ValueError, match="feature store verification failed"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_corrupt_projection_artifact(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    (layout.root / "projection.npz").write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="projection.npz is invalid"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_wrong_projection_state(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    wrong = FixedSparseProjection(seed=20_260_829)
    np.savez(
        layout.root / "projection.npz",
        components=wrong.components,
        metadata=np.asarray(json.dumps(wrong.metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="feature_generation projection"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_wrong_standardizer_state(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    wrong = BlockStandardizer().fit(
        {
            "delta_proj": np.ones((2, 256), dtype=np.float32),
            "s": np.ones((2, 6), dtype=np.float32),
            "q_sc": np.ones((2, 3), dtype=np.float32),
            "e_g": np.ones((2, 1_280), dtype=np.float32),
            "z_c": np.ones((2, 5_120), dtype=np.float32),
        }
    )
    np.savez(
        layout.root / "standardizer.npz",
        state=np.asarray(json.dumps(wrong.to_state(), sort_keys=True)),
    )
    with pytest.raises(ValueError, match="feature_generation standardizer"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_wrong_feature_schema(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    schema = FEATURE_SCHEMA.to_dict()
    schema["summary_fields"] = [*schema["summary_fields"], "wrong"]
    atomic_write_json(layout.root / "feature_schema.json", schema)
    with pytest.raises(ValueError, match="runtime schema"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_feature_store_swap(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    frozen = layout.root / "condition_features/stage1_frozen"
    selected = layout.root / "condition_features/stage2_selected"
    temporary = layout.root / "condition_features/swap"
    frozen.rename(temporary)
    selected.rename(frozen)
    temporary.rename(selected)
    with pytest.raises(ValueError, match="feature_generation frozen manifest"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_wrong_split_membership(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    split_path = layout.root / "cell_line_geneeffect_226_split.json"
    split = json.loads(split_path.read_text())
    split["val"].pop()
    atomic_write_json(split_path, split)
    with pytest.raises(ValueError, match="split"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_valid_sized_split_substitution(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    split_path = layout.root / "cell_line_geneeffect_226_split.json"
    split = json.loads(split_path.read_text())
    split["val"] = [f"OTHER-{index:02d}" for index in range(27)]
    atomic_write_json(split_path, split)
    with pytest.raises(ValueError, match="pinned Exp13 authority"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_duplicate_prediction_key(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    lines = predictions.read_text().splitlines()
    predictions.write_text("\n".join([*lines, lines[1]]) + "\n")
    with pytest.raises(ValueError, match="duplicate evaluation keys"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_wrong_method_set(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    predictions.write_text(
        predictions.read_text().replace("context_pca_ridge[z_c]", "unregistered")
    )
    with pytest.raises(ValueError, match="five fixed methods"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_uniform_gene_omission(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    rows = predictions.read_text().splitlines()
    predictions.write_text(
        "\n".join([rows[0], *(row for row in rows[1:] if ",G2," not in row)]) + "\n"
    )
    with pytest.raises(ValueError, match="genes differ from the scored universe"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_single_finite_label_omission(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    split = json.loads(
        (layout.root / "cell_line_geneeffect_226_split.json").read_text()
    )
    missing_model = split["val"][0]
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    rows = predictions.read_text().splitlines()
    predictions.write_text(
        "\n".join(
            [
                rows[0],
                *(
                    row
                    for row in rows[1:]
                    if not (f",{missing_model},G2," in row and row.startswith("val,"))
                ),
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="residual label mask"):
        mark_complete(layout, run_id="formal")


def test_default_completion_recomputes_residual_target_digest(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    run_path = layout.root / "run_manifest.json"
    run = json.loads(run_path.read_text())
    run["residual_target_sha256"] = "0" * 64
    atomic_write_json(run_path, run)
    for phase in ("warmup", "joint"):
        metadata_path = layout.root / phase / "training/best/metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["provenance"]["residual_target_sha256"] = "0" * 64
        atomic_write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="does not match residual targets"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_consistently_tampered_csv_truth(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    rows = predictions.read_text().splitlines()
    tampered = [rows[0]]
    for row in rows[1:]:
        fields = row.split(",")
        fields[4] = str(float(fields[4]) + 10.0)
        fields[5] = str(float(fields[5]) + 10.0)
        tampered.append(",".join(fields))
    predictions.write_text("\n".join(tampered) + "\n")
    with pytest.raises(ValueError, match="authoritative target"):
        mark_complete(layout, run_id="formal")


def test_default_completion_reconstructs_gene_effect_from_mu(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    rows = predictions.read_text().splitlines()
    fields = rows[1].split(",")
    fields[4] = str(float(fields[4]) + 1.0)
    rows[1] = ",".join(fields)
    predictions.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=r"residual \+ mu_train"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_nonfinite_prediction(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    predictions = layout.root / "geneeffect_residual_predictions.csv"
    rows = predictions.read_text().splitlines()
    fields = rows[1].split(",")
    fields[-1] = "nan"
    rows[1] = ",".join(fields)
    predictions.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="must be finite"):
        mark_complete(layout, run_id="formal")


def test_default_completion_recomputes_metrics(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metrics_path = layout.root / "geneeffect_residual_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["validation"]["macro_per_gene_spearman"] = 0.0
    atomic_write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="does not match recomputed predictions"):
        mark_complete(layout, run_id="formal")


def test_default_completion_binds_selected_metric_to_validation(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metadata_path = layout.root / "joint/training/best/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metric_value"] = 0.5
    atomic_write_json(metadata_path, metadata)
    selection_path = layout.root / "checkpoint_selection.json"
    selection = json.loads(selection_path.read_text())
    selection["joint"]["best_metric"] = 0.5
    atomic_write_json(selection_path, selection)
    (layout.root / "joint/training/train_log.csv").write_text(
        "epoch,validation_macro_per_gene_spearman\n0,0.5\n"
    )
    with pytest.raises(ValueError, match="does not match recomputed validation"):
        mark_complete(layout, run_id="formal")


def test_default_completion_validates_lambda_calibration_schema(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    calibration_path = layout.root / "lambda_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    calibration["raw_ratios"].pop()
    atomic_write_json(calibration_path, calibration)
    metadata_path = layout.root / "joint/training/best/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance"]["lambda_calibration_report"] = calibration
    atomic_write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="must contain 8"):
        mark_complete(layout, run_id="formal")


def test_default_completion_requires_explicit_gene_mean_undefined_status(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metrics_path = layout.root / "geneeffect_residual_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["baselines"]["slices"]["test"]["methods"]["gene_mean"]["macro_per_gene"] = (
        0.0
    )
    atomic_write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="must be null"):
        mark_complete(layout, run_id="formal")


def test_default_completion_requires_finite_response_metrics(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metrics_path = layout.root / "geneeffect_residual_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["response"]["after_stage2"]["metrics"]["model_loss"] = None
    atomic_write_json(metrics_path, metrics)
    with pytest.raises(
        ValueError, match="after_stage2.metrics.model_loss must be finite"
    ):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_comparable_response_claim(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metrics_path = layout.root / "geneeffect_residual_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["response"]["comparison_status"] = "comparable"
    metrics["response"]["delta_reported"] = True
    atomic_write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="comparison status mismatch"):
        mark_complete(layout, run_id="formal")


@pytest.mark.parametrize("tamper", ["cache", "target", "membership"])
def test_terminal_verifier_rejects_response_lineage_tamper(
    tmp_path: Path, tamper: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    mark_complete(layout, run_id="formal")
    path = layout.root / "response_targets/lineage.json"
    payload = json.loads(path.read_text())
    if tamper == "cache":
        payload["response_cache_files"]["manifest.json"] = "f" * 64
    elif tamper == "target":
        payload["train_records"][0]["observed_hvg"]["content_sha256"] = "f" * 64
    else:
        payload["train_records"][0]["membership"] = "heldout"
    atomic_write_json(path, payload)

    with pytest.raises(ValueError, match="response lineage"):
        verify_complete_run(layout.root)


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    (
        ("run_manifest.json", ("world_size", 1)),
        ("model_package/model_manifest.json", ("mixed_precision", "fp16")),
        (
            "warmup/training/best/metadata.json",
            ("global_conditions_per_step", 256),
        ),
        ("joint/training/best/metadata.json", ("conditions_per_rank", 64)),
    ),
)
def test_default_completion_rejects_distributed_runtime_drift(
    tmp_path: Path, artifact: str, mutation: tuple[str, object]
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    path = layout.root / artifact
    payload = json.loads(path.read_text())
    runtime = (
        payload["provenance"]["distributed_runtime"]
        if artifact.endswith("metadata.json")
        else payload["distributed_runtime"]
    )
    runtime[mutation[0]] = mutation[1]
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="distributed_runtime"):
        mark_complete(layout, run_id="formal")


@pytest.mark.parametrize(
    ("artifact", "field"),
    (
        ("stage1_model_manifest.json", "training_code_provenance_status"),
        ("run_manifest.json", "stage1_training_data_provenance_status"),
        (
            "model_package/model_manifest.json",
            "stage1_training_data_provenance_missing_identities",
        ),
    ),
)
def test_default_completion_rejects_upgraded_stage1_provenance_claim(
    tmp_path: Path, artifact: str, field: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    path = layout.root / artifact
    payload = json.loads(path.read_text())
    payload[field] = "complete" if field.endswith("status") else []
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="Stage-1 .*provenance claim mismatch"):
        mark_complete(layout, run_id="formal")


def test_required_outputs_cover_every_production_training_phase() -> None:
    assert {
        "backbone_load_report.json",
        "lambda_calibration.json",
        "feature_generation.json",
        "warmup/training/best/head.pt",
        "warmup/training/best/metadata.json",
        "warmup/training/train_log.csv",
        "joint/training/best/e2e_state.pt",
        "joint/training/best/metadata.json",
        "joint/training/train_log.csv",
    }.issubset(REQUIRED_STAGE2_OUTPUTS)

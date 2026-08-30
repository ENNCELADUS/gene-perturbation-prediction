from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

import aivc_model.stage2_artifacts as stage2_artifacts
from aivc_model.geneeffect_feature_store import GeneEffectFeatureStoreWriter
from aivc_model.geneeffect_features import (
    FEATURE_SCHEMA,
    BlockStandardizer,
    FixedSparseProjection,
)
from aivc_model.stage2_artifacts import (
    REQUIRED_STAGE2_OUTPUTS,
    Stage2RunLayout,
    atomic_write_json,
    mark_complete,
    mark_failure,
    prepare_run_dir,
    verify_complete_run,
)


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
    path = root / "response_targets/lineage.json"
    atomic_write_json(path, {"status": "available"})
    return "1" * 64, "2" * 64


def _write_full_runner_artifacts(
    layout: Stage2RunLayout, *, world_size: int = 4
) -> None:
    root = layout.root
    warmup_checkpoint = root / "warmup/training/best/head.pt"
    joint_checkpoint = root / "joint/training/best/e2e_state.pt"
    warmup_checkpoint.parent.mkdir(parents=True)
    joint_checkpoint.parent.mkdir(parents=True)
    torch.save({"head.weight": torch.arange(6, dtype=torch.float32)}, warmup_checkpoint)
    torch.save(
        {"model.weight": torch.arange(12, dtype=torch.float32)}, joint_checkpoint
    )
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
    warmup_runtime = {
        "world_size": world_size,
        "conditions_per_rank": 256,
        "global_conditions_per_step": 256 * world_size,
        "optimizer_steps_per_epoch": 7,
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
        "warmup_runtime": warmup_runtime,
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
        f"epoch,{selection_column},optimizer_steps\n0,0.5,7\n"
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
    atomic_write_json(
        root / "model_package/model_manifest.json",
        {
            "checkpoint": "e2e_state.pt",
            "projection": "../projection.npz",
            "standardizer": "../standardizer.npz",
            "feature_schema": "../feature_schema.json",
            "frozen_features": "../condition_features/stage1_frozen",
            "selected_features": "../condition_features/stage2_selected",
            "distributed_runtime": distributed_runtime,
        },
    )
    atomic_write_json(
        root / "run_manifest.json",
        {
            "run_id": "formal",
            "status": "artifacts_written",
            "git_commit": "test",
            "distributed_runtime": distributed_runtime,
            "seeds": {"train": 1, "collator": 2, "projection": 3},
            "cells_per_context": 128,
            "cell_set_len": 128,
            "projection_seed": 3,
            "gene_universe": {"count": len(genes), "symbols": list(genes)},
            "preflight": {},
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


def test_complete_round_trip_requires_nonempty_outputs(tmp_path: Path) -> None:
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

    (layout.root / "nested/b.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="artifact is empty"):
        verify_complete_run(layout.root, required_outputs=required)


def test_completion_rejects_run_id_mismatch(tmp_path: Path) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    atomic_write_json(layout.root / "run_manifest.json", {"run_id": "manifest-id"})
    with pytest.raises(ValueError, match="run_id"):
        mark_complete(layout, run_id="sentinel-id", required_outputs=())


def test_completion_does_not_authenticate_unlisted_feature_shards(
    tmp_path: Path,
) -> None:
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
    assert payload == {"status": "complete", "run_id": "r"}
    shard.write_bytes(b"tampered")
    assert verify_complete_run(layout.root, required_outputs=required) == payload


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
    assert complete == {"status": "complete", "run_id": "formal"}
    assert verify_complete_run(layout.root)["status"] == "complete"


@pytest.mark.parametrize(
    "relative",
    (
        "warmup/training/best/head.pt",
        "joint/training/best/e2e_state.pt",
        "model_package/e2e_state.pt",
    ),
)
def test_default_completion_rejects_unloadable_checkpoint(
    tmp_path: Path, relative: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    (layout.root / relative).write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="checkpoint cannot be loaded"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_mismatched_packaged_checkpoint(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    torch.save(
        {"model.weight": torch.arange(12, dtype=torch.float32) + 1},
        layout.root / "model_package/e2e_state.pt",
    )
    with pytest.raises(ValueError, match="tensor differs from selected joint"):
        mark_complete(layout, run_id="formal")


@pytest.mark.parametrize("artifact", ("projection.npz", "standardizer.npz"))
def test_default_completion_rejects_unloadable_preprocessing_payload(
    tmp_path: Path, artifact: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    (layout.root / artifact).write_bytes(b"not an npz")
    with pytest.raises(ValueError, match=rf"{artifact} is invalid"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_mismatched_projection_payload(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    path = layout.root / "projection.npz"
    with np.load(path, allow_pickle=False) as loaded:
        components = loaded["components"].copy()
        metadata = loaded["metadata"].copy()
    components[0, 0] += 1
    np.savez(path, components=components, metadata=metadata)
    with pytest.raises(ValueError, match="declared generator"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_mismatched_standardizer_payload(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    path = layout.root / "standardizer.npz"
    with np.load(path, allow_pickle=False) as loaded:
        state = json.loads(str(loaded["state"].item()))
    state["blocks"]["s"]["mean"][0] = 1.0
    np.savez(path, state=np.asarray(json.dumps(state, sort_keys=True)))
    with pytest.raises(ValueError, match="differs from feature_generation"):
        mark_complete(layout, run_id="formal")


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selection_name", "loss"),
        ("selection_direction", "minimize"),
    ),
)
def test_default_completion_rejects_nonprotocol_checkpoint_selection(
    tmp_path: Path, field: str, value: str
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metadata_path = layout.root / "joint/training/best/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    atomic_write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="metric contract mismatch"):
        mark_complete(layout, run_id="formal")


def test_default_completion_requires_eight_lambda_calibration_batches(
    tmp_path: Path,
) -> None:
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


def test_lambda_calibration_rejects_false_ratio_or_lambda() -> None:
    calibration = {
        "lambda_dep": 1.0,
        "raw_ratios": [1.0] * 8,
        "response_gradient_norms": [2.0] * 8,
        "dependency_gradient_norms": [2.0] * 8,
    }
    calibration["raw_ratios"][0] = 2.0
    with pytest.raises(ValueError, match="ratios do not match"):
        stage2_artifacts._verify_lambda_calibration(calibration)

    calibration["raw_ratios"][0] = 1.0
    calibration["lambda_dep"] = 2.0
    with pytest.raises(ValueError, match="clipped median ratio"):
        stage2_artifacts._verify_lambda_calibration(calibration)


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


def test_default_completion_rejects_comparable_response_claim(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    metrics_path = layout.root / "geneeffect_residual_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["response"]["comparison_status"] = "comparable"
    metrics["response"]["delta_reported"] = True
    atomic_write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="comparison status mismatch"):
        mark_complete(layout, run_id="formal")


def test_default_completion_rejects_nonformal_distributed_runtime(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout, world_size=3)
    with pytest.raises(ValueError, match="world_size must be 2 or 4"):
        mark_complete(layout, run_id="formal")


def test_distributed_runtime_accepts_repeated_local_devices_across_hosts() -> None:
    runtime = {
        "world_size": 2,
        "mixed_precision": "bf16",
        "conditions_per_rank": 64,
        "global_conditions_per_step": 128,
        "rank_topology": [
            {
                "rank": rank,
                "local_rank": 0,
                "device": "cuda:0",
                "device_name": "NVIDIA H20",
                "hostname": f"host-{rank}",
            }
            for rank in range(2)
        ],
    }
    assert stage2_artifacts._verify_distributed_runtime(runtime) == runtime


def test_default_completion_rejects_consistent_optimizer_step_tamper(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    for phase in ("warmup", "joint"):
        path = layout.root / phase / "training/best/metadata.json"
        payload = json.loads(path.read_text())
        payload["provenance"]["warmup_runtime"]["optimizer_steps_per_epoch"] = 8
        atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="train_log optimizer_steps"):
        mark_complete(layout, run_id="formal")


def test_default_completion_checks_optimizer_steps_in_every_warmup_row(
    tmp_path: Path,
) -> None:
    layout = prepare_run_dir(tmp_path / "run")
    _write_full_runner_artifacts(layout)
    history = layout.root / "warmup/training/train_log.csv"
    history.write_text(
        "epoch,validation_macro_per_gene_spearman,optimizer_steps\n0,0.5,7\n1,0.4,6\n"
    )
    with pytest.raises(ValueError, match="train_log optimizer_steps"):
        mark_complete(layout, run_id="formal")


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

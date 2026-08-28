"""Pure tests for sealing and restoring the Exp13 Stage-1 artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aivc_model.gene_embeddings import Esm2EmbeddingTable
from aivc_model.stage1_artifact import (
    Stage1ArtifactManifest,
    load_stage1_artifact,
    seal_stage1_artifact,
    sha256_file,
)
from aivc_model.stage1_config import load_stage1_config
from aivc_model.state_core import (
    Esm2PerturbationAdapter,
    LinearMockStateModel,
    StateForwardAdapter,
)
from aivc_model.tx1_predicted_response import ForwardOnlyStateModel

_STAGE1_GENES = ("G1", "G2")
_TARGET_GENES = ("G1", "G2", "G3")


def _vectors() -> dict[str, np.ndarray]:
    return {
        "G1": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "G2": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        "G3": np.asarray([7.0, 8.0, 9.0], dtype=np.float32),
    }


def _write_esm(
    path: Path, genes: tuple[str, ...], *, g2_offset: float = 0.0
) -> None:
    vectors = _vectors()
    vectors["G2"] = vectors["G2"].copy()
    vectors["G2"][0] += g2_offset
    np.savez(
        path,
        symbols=np.asarray(genes, dtype=object),
        vectors=np.vstack([vectors[gene] for gene in genes]),
        resolved=np.ones(len(genes), dtype=bool),
    )


def _model(genes: tuple[str, ...]) -> ForwardOnlyStateModel:
    table = Esm2EmbeddingTable(dim=3, vectors_by_symbol=_vectors())
    perturbations = Esm2PerturbationAdapter(
        list(genes), table, adapter_hidden=4, pert_dim=2
    )
    return ForwardOnlyStateModel(
        StateForwardAdapter(LinearMockStateModel(3, 3, 2)), perturbations
    )


def _write_text(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, object]:
    run = tmp_path / "run"
    best = run / "best"
    best.mkdir(parents=True)
    esm = tmp_path / "stage1_esm2.npz"
    target_esm = tmp_path / "target_esm2.npz"
    _write_esm(esm, _STAGE1_GENES)
    _write_esm(target_esm, _TARGET_GENES)
    state_hparams = _write_text(tmp_path / "state.ckpt", "state-hparams")
    code = {"state_core.py": _write_text(tmp_path / "code.py", "code")}
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(
        Path(
            "configs/experiments/13_geneeffect_226/stage1_response.yaml"
        ).read_bytes()
    )
    config = {"stage1.yaml": config_path}
    source = {
        "split_json": _write_text(tmp_path / "split.json", "split"),
        "perturbseq_sources": _write_text(tmp_path / "sources.yaml", "source"),
    }
    source_model = _model(_STAGE1_GENES)
    checkpoint = dict(source_model.state_dict())
    checkpoint["perturbations.esm_matrix"] = source_model.perturbations.esm_matrix
    checkpoint["response_encoder.linear.weight"] = torch.zeros(1)
    checkpoint_path = best / "pytorch_model.bin"
    torch.save(checkpoint, checkpoint_path)
    manifest_path = tmp_path / "stage1_artifact.json"
    metadata = best / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint_kind": "best",
                "epoch": 4,
                "selection_metric": "heldout_loss",
                "metric_value": 1.25,
            }
        )
    )
    run_manifest = run / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "config_sha256": sha256_file(config["stage1.yaml"]),
                "best_epoch": 4,
                "selection_metric": "heldout_loss",
                "best_metric_value": 1.25,
                "input_sha256": {
                    "state_checkpoint": sha256_file(state_hparams),
                    "esm2_embeddings": sha256_file(esm),
                    "split_json": sha256_file(source["split_json"]),
                    "perturbseq_sources": sha256_file(
                        source["perturbseq_sources"]
                    ),
                },
            }
        )
    )
    objective = run / "stage1_objective.json"
    objective.write_text(json.dumps(load_stage1_config(config_path).objective_payload()))
    return {
        "esm": esm,
        "target_esm": target_esm,
        "state_hparams": state_hparams,
        "code": code,
        "config": config,
        "source": source,
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "run_manifest": run_manifest,
        "metadata": metadata,
        "objective": objective,
        "source_model": source_model,
    }


def _seal(fixture: dict[str, object]) -> Stage1ArtifactManifest:
    return seal_stage1_artifact(
        checkpoint_path=fixture["checkpoint"],
        manifest_path=fixture["manifest"],
        stage1_genes=_STAGE1_GENES,
        esm2_embeddings_path=fixture["esm"],
        state_hparams_path=fixture["state_hparams"],
        run_manifest_path=fixture["run_manifest"],
        checkpoint_metadata_path=fixture["metadata"],
        stage1_objective_path=fixture["objective"],
        compatibility_code_paths=fixture["code"],
        config_paths=fixture["config"],
        source_paths=fixture["source"],
    )


def _load(fixture: dict[str, object], model: ForwardOnlyStateModel, *, trainable: bool):
    return load_stage1_artifact(
        model,
        checkpoint_path=fixture["checkpoint"],
        manifest_path=fixture["manifest"],
        esm2_embeddings_path=fixture["esm"],
        target_esm_embeddings_path=fixture["target_esm"],
        target_esm_artifact_sha256=sha256_file(fixture["target_esm"]),
        state_hparams_path=fixture["state_hparams"],
        run_manifest_path=fixture["run_manifest"],
        checkpoint_metadata_path=fixture["metadata"],
        stage1_objective_path=fixture["objective"],
        compatibility_code_paths=fixture["code"],
        config_paths=fixture["config"],
        source_paths=fixture["source"],
        trainable=trainable,
    )


def test_manifest_separates_compatibility_code_from_training_provenance(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    sealed = _seal(fixture)

    loaded = Stage1ArtifactManifest.read(fixture["manifest"])

    assert loaded == sealed
    assert loaded.stage1_genes == _STAGE1_GENES
    assert loaded.legacy_esm_matrix_sha256 is not None
    payload = json.loads(fixture["manifest"].read_text())
    assert set(loaded.compatibility_code_sha256) == {"state_core.py"}
    assert loaded.training_code_provenance_status == "unavailable"
    assert (
        loaded.training_code_provenance_reason
        == "historical_run_has_no_immutable_training_code_identity"
    )
    assert loaded.training_data_provenance_status == "incomplete"
    assert loaded.training_data_provenance_missing_identities == (
        "cell_line_manifest",
        "tx1_basal_cache",
        "response_cache",
        "perturbseq_source_content",
    )
    assert (
        loaded.training_data_provenance_reason
        == "historical_run_manifest_does_not_hash_all_training_data_inputs"
    )
    assert "code_sha256" not in payload
    assert set(loaded.config_sha256) == {"stage1.yaml"}
    assert set(loaded.source_sha256) == {"split_json", "perturbseq_sources"}


def test_seal_rejects_same_genes_in_different_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(ValueError, match="gene-vocabulary SHA256 mismatch"):
        seal_stage1_artifact(
            checkpoint_path=fixture["checkpoint"],
            manifest_path=fixture["manifest"],
            stage1_genes=("G2", "G1"),
            esm2_embeddings_path=fixture["esm"],
            state_hparams_path=fixture["state_hparams"],
            run_manifest_path=fixture["run_manifest"],
            checkpoint_metadata_path=fixture["metadata"],
            stage1_objective_path=fixture["objective"],
            compatibility_code_paths=fixture["code"],
            config_paths=fixture["config"],
            source_paths=fixture["source"],
        )


def test_seal_rejects_legacy_matrix_shape_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = torch.load(fixture["checkpoint"], weights_only=True)
    checkpoint["perturbations.esm_matrix"] = checkpoint["perturbations.esm_matrix"][:1]
    torch.save(checkpoint, fixture["checkpoint"])

    with pytest.raises(ValueError, match="shape mismatch"):
        _seal(fixture)


def test_load_rejects_changed_esm_artifact_even_if_vectors_are_valid(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    _write_esm(fixture["esm"], _STAGE1_GENES, g2_offset=0.5)

    with pytest.raises(ValueError, match="ESM-2 SHA256 mismatch"):
        _load(fixture, _model(_TARGET_GENES), trainable=False)


def test_load_rejects_changed_compatibility_code(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    fixture["code"]["state_core.py"].write_text("changed loader code")

    with pytest.raises(ValueError, match="seal/load-time compatibility code"):
        _load(fixture, _model(_TARGET_GENES), trainable=False)


def test_seal_rejects_objective_not_derived_from_authenticated_config(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["objective"].write_text("{}")

    with pytest.raises(ValueError, match="objective does not match"):
        _seal(fixture)


def test_load_rejects_partial_learned_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    manifest_payload = json.loads(fixture["manifest"].read_text())
    checkpoint = torch.load(fixture["checkpoint"], weights_only=True)
    del checkpoint["perturbations.adapter.net.0.weight"]
    torch.save(checkpoint, fixture["checkpoint"])
    manifest_payload["checkpoint_sha256"] = sha256_file(fixture["checkpoint"])
    fixture["manifest"].write_text(json.dumps(manifest_payload))

    with pytest.raises(ValueError, match="missing=.*adapter.net.0.weight"):
        _load(fixture, _model(_TARGET_GENES), trainable=False)


@pytest.mark.parametrize("trainable", [False, True])
def test_load_supports_larger_target_universe_and_explicit_mode(
    tmp_path: Path, trainable: bool
) -> None:
    fixture = _fixture(tmp_path)
    source_model = fixture["source_model"]
    _seal(fixture)
    destination = _model(_TARGET_GENES)

    report = _load(fixture, destination, trainable=trainable)

    assert destination.perturbations.genes == list(_TARGET_GENES)
    assert destination.perturbations.esm_matrix.shape == (3, 3)
    assert report.legacy_esm_matrix_authenticated is True
    assert report.trainable is trainable
    assert destination.training is trainable
    assert all(
        parameter.requires_grad is trainable for parameter in destination.parameters()
    )
    for key, value in source_model.state_dict().items():
        if key == "perturbations.gene_vocabulary_sha256":
            continue
        assert torch.equal(destination.state_dict()[key], value)


def test_buffer_free_checkpoint_authenticates_persisted_vocabulary_hash(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = torch.load(fixture["checkpoint"], weights_only=True)
    del checkpoint["perturbations.esm_matrix"]
    torch.save(checkpoint, fixture["checkpoint"])

    manifest = _seal(fixture)
    report = _load(fixture, _model(_TARGET_GENES), trainable=False)

    assert manifest.legacy_esm_matrix_sha256 is None
    assert report.legacy_esm_matrix_authenticated is False


def test_manifest_rejects_mutated_gene_vocabulary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    payload = json.loads(fixture["manifest"].read_text())
    payload["stage1_genes"] = ["G2", "G1"]
    fixture["manifest"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="vocabulary SHA256 mismatch"):
        Stage1ArtifactManifest.read(fixture["manifest"])


def test_manifest_rejects_unsubstantiated_training_code_claim(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    payload = json.loads(fixture["manifest"].read_text())
    payload["training_code_provenance_status"] = "available"
    fixture["manifest"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="training-code provenance.*unavailable"):
        Stage1ArtifactManifest.read(fixture["manifest"])


def test_manifest_rejects_unsubstantiated_complete_training_data_claim(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    payload = json.loads(fixture["manifest"].read_text())
    payload["training_data_provenance_status"] = "complete"
    payload["training_data_provenance_missing_identities"] = []
    fixture["manifest"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="training-data provenance.*incomplete"):
        Stage1ArtifactManifest.read(fixture["manifest"])


def test_manifest_rejects_omitted_training_data_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _seal(fixture)
    payload = json.loads(fixture["manifest"].read_text())
    payload["training_data_provenance_missing_identities"].remove("response_cache")
    fixture["manifest"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="every missing identity recorded"):
        Stage1ArtifactManifest.read(fixture["manifest"])

"""Tests for the one-shot historical Stage-1 seal command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aivc_model.stage1_artifact import Stage1ArtifactManifest, sha256_file
from aivc_model.stage1_config import load_stage1_config
from scripts import seal_stage1_response_artifact as seal_cli
from scripts.seal_stage1_response_artifact import _seal


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _REPOSITORY_ROOT / (
    "configs/experiments/13_geneeffect_226/stage1_response.yaml"
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, matrix: np.ndarray | None = None) -> dict[str, Path]:
    run = tmp_path / "run"
    best = run / "best"
    best.mkdir(parents=True)
    esm = tmp_path / "stage1_esm.npz"
    symbols = np.asarray(["G2", "G1", "G3"], dtype=object)
    vectors = np.asarray([[4.0, 5.0], [1.0, 2.0], [7.0, 8.0]], dtype=np.float32)
    np.savez(
        esm,
        symbols=symbols,
        vectors=vectors,
        resolved=np.ones(3, dtype=bool),
    )
    if matrix is None:
        matrix = np.asarray([[1.0, 2.0], [7.0, 8.0]], dtype=np.float32)
    checkpoint = best / "pytorch_model.bin"
    torch.save({"perturbations.esm_matrix": torch.as_tensor(matrix)}, checkpoint)
    state_hparams = _write(tmp_path / "state.ckpt", "state")
    split = _write(tmp_path / "split.json", "split")
    sources = _write(tmp_path / "sources.json", "sources")
    _write(
        best / "metadata.json",
        json.dumps(
            {
                "checkpoint_kind": "best",
                "epoch": 32,
                "selection_metric": "heldout_anchor_weighted_L_resp",
                "metric_value": 0.25,
            }
        ),
    )
    _write(
        run / "run_manifest.json",
        json.dumps(
            {
                "config_sha256": sha256_file(_CONFIG),
                "best_epoch": 32,
                "selection_metric": "heldout_anchor_weighted_L_resp",
                "best_metric_value": 0.25,
                "input_sha256": {
                    "state_checkpoint": sha256_file(state_hparams),
                    "esm2_embeddings": sha256_file(esm),
                    "split_json": sha256_file(split),
                    "perturbseq_sources": sha256_file(sources),
                },
            }
        ),
    )
    _write(
        run / "stage1_objective.json",
        json.dumps(load_stage1_config(_CONFIG).objective_payload()),
    )
    return {
        "run": run,
        "esm": esm,
        "state": state_hparams,
        "split": split,
        "sources": sources,
    }


def _run(fixture: dict[str, Path], *, dry_run: bool) -> dict[str, object]:
    return _seal(
        run_dir=fixture["run"],
        stage1_config=_CONFIG,
        esm2_embeddings=fixture["esm"],
        state_hparams=fixture["state"],
        split_json=fixture["split"],
        perturbseq_sources=fixture["sources"],
        repository_root=_REPOSITORY_ROOT,
        dry_run=dry_run,
    )


def test_dry_run_authenticates_order_without_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = _run(fixture, dry_run=True)

    assert report["status"] == "compatibility_inputs_validated"
    assert report["training_data_provenance_status"] == "incomplete"
    assert report["stage1_genes"] == ["G1", "G3"]
    assert report["writes_planned"] == [
        "stage1_model_manifest.json",
        "stage2_bundle.json",
    ]
    assert not (fixture["run"] / "stage1_model_manifest.json").exists()
    assert not (fixture["run"] / "stage2_bundle.json").exists()


def test_seal_writes_manifest_and_runner_bundle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = _run(fixture, dry_run=False)

    manifest = Stage1ArtifactManifest.read(
        fixture["run"] / "stage1_model_manifest.json"
    )
    bundle = json.loads((fixture["run"] / "stage2_bundle.json").read_text())
    assert report["status"] == "compatibility_inputs_sealed"
    assert manifest.stage1_genes == ("G1", "G3")
    assert bundle["schema_version"] == "exp13-stage2-bundle-v1"
    assert set(bundle) == {
        "schema_version",
        "compatibility_code_paths",
        "config_paths",
        "source_paths",
    }
    assert set(bundle["config_paths"]) == {"stage1_response"}
    assert set(bundle["source_paths"]) == {"split_json", "perturbseq_sources"}
    assert set(bundle["compatibility_code_paths"]) == set(
        manifest.compatibility_code_sha256
    )
    assert manifest.training_code_provenance_status == "unavailable"
    assert report["training_data_provenance_missing_identities"] == [
        "cell_line_manifest",
        "tx1_basal_cache",
        "response_cache",
        "perturbseq_source_content",
    ]


def test_seal_refuses_to_overwrite_either_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write(fixture["run"] / "stage2_bundle.json", "existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _run(fixture, dry_run=False)

    assert not (fixture["run"] / "stage1_model_manifest.json").exists()
    assert (fixture["run"] / "stage2_bundle.json").read_text() == "existing"


def test_seal_rejects_row_that_only_matches_by_shape(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, matrix=np.asarray([[1.0, 2.001]], dtype=np.float32))

    with pytest.raises(ValueError, match="match at least one"):
        _run(fixture, dry_run=True)


def test_seal_rejects_globally_ambiguous_exact_vector_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with np.load(fixture["esm"], allow_pickle=True) as payload:
        symbols = payload["symbols"]
        vectors = payload["vectors"].copy()
        resolved = payload["resolved"]
    vectors[0] = vectors[1]
    np.savez(fixture["esm"], symbols=symbols, vectors=vectors, resolved=resolved)
    run_manifest = fixture["run"] / "run_manifest.json"
    manifest = json.loads(run_manifest.read_text())
    manifest["input_sha256"]["esm2_embeddings"] = sha256_file(fixture["esm"])
    run_manifest.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unique strictly sorted"):
        _run(fixture, dry_run=True)


def test_seal_resolves_aliases_when_sorted_vocabulary_is_unique(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    np.savez(
        fixture["esm"],
        symbols=np.asarray(["AARS1", "ZZZ", "AARS"], dtype=object),
        vectors=np.asarray(
            [[1.0, 2.0], [7.0, 8.0], [1.0, 2.0]], dtype=np.float32
        ),
        resolved=np.ones(3, dtype=bool),
    )
    torch.save(
        {
            "perturbations.esm_matrix": torch.as_tensor(
                [[1.0, 2.0], [1.0, 2.0], [7.0, 8.0]], dtype=torch.float32
            )
        },
        fixture["run"] / "best" / "pytorch_model.bin",
    )
    run_manifest = fixture["run"] / "run_manifest.json"
    manifest = json.loads(run_manifest.read_text())
    manifest["input_sha256"]["esm2_embeddings"] = sha256_file(fixture["esm"])
    run_manifest.write_text(json.dumps(manifest))

    report = _run(fixture, dry_run=True)

    assert report["stage1_genes"] == ["AARS", "AARS1", "ZZZ"]


def test_digest_collision_still_requires_exact_vector_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(seal_cli, "_vector_digest", lambda vector: "collision")

    report = _run(fixture, dry_run=True)

    assert report["stage1_genes"] == ["G1", "G3"]


def test_digest_collision_without_exact_match_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(
        tmp_path, matrix=np.asarray([[9.0, 10.0]], dtype=np.float32)
    )
    monkeypatch.setattr(seal_cli, "_vector_digest", lambda vector: "collision")

    with pytest.raises(ValueError, match="match at least one"):
        _run(fixture, dry_run=True)

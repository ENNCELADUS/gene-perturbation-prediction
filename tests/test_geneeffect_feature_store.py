from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aivc_model.geneeffect_feature_store import (
    CONTEXT_WIDTH,
    DELTA_PROJ_WIDTH,
    GENE_WIDTH,
    Q_SC_WIDTH,
    SUMMARY_WIDTH,
    GeneEffectFeatureStoreWriter,
    load_geneeffect_feature_batch,
    verify_geneeffect_feature_store,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _arrays(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "delta_proj": np.full((3, DELTA_PROJ_WIDTH), offset, dtype=np.float32),
        "s": np.full((3, SUMMARY_WIDTH), offset + 1, dtype=np.float32),
        "q_sc": np.full((3, Q_SC_WIDTH), offset + 2, dtype=np.float32),
        "q_sc_mask": np.array([True, False, True]),
        "hvg_panel_mask": np.array([True, True, False]),
        "own_gene_shift_mask": np.array([True, False, False]),
    }


def _build(root: Path, *, stage: str = "stage1_frozen", resume: bool = False):
    writer = GeneEffectFeatureStoreWriter(
        root,
        stage=stage,
        model_ids=("B", "A"),
        gene_symbols=("G2", "G1", "G3"),
        e_g=np.arange(3 * GENE_WIDTH, dtype=np.float32).reshape(3, GENE_WIDTH),
        z_c=np.arange(2 * CONTEXT_WIDTH, dtype=np.float32).reshape(2, CONTEXT_WIDTH),
        gene_embedding_source_sha256=_digest("esm2"),
        feature_schema_sha256=_digest("schema"),
        projection_sha256=_digest("projection"),
        resume=resume,
    )
    written = []
    for index, model_id in enumerate(("B", "A")):
        written.append(
            writer.write_shard(
                model_id,
                **_arrays(float(index)),
                source_sha256=_digest(f"source-{model_id}"),
                model_checkpoint_sha256=_digest("checkpoint"),
            )
        )
    return writer.finalize(), written


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def _load_identity() -> dict[str, object]:
    return {
        "expected_stage": "stage1_frozen",
        "expected_checkpoint_sha256": _digest("checkpoint"),
        "expected_feature_schema_sha256": _digest("schema"),
        "expected_projection_sha256": _digest("projection"),
        "expected_source_sha256": {
            "A": _digest("source-A"),
            "B": _digest("source-B"),
        },
        "expected_gene_embedding_source_sha256": _digest("esm2"),
        "expected_model_ids": ("B", "A"),
    }


def test_round_trip_preserves_frozen_orders_and_builds_precomputed_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    manifest, written = _build(root)
    assert written == [True, True]
    assert manifest["model_ids"] == ["B", "A"]
    assert manifest["gene_symbols"] == ["G2", "G1", "G3"]
    assert verify_geneeffect_feature_store(root)["status"] == "passed"

    loaded = load_geneeffect_feature_batch(
        root,
        "A",
        expected_gene_symbols=("G2", "G1", "G3"),
        **_load_identity(),
    )
    assert loaded.gene_symbols == ("G2", "G1", "G3")
    assert loaded.features.delta_proj.shape == (3, DELTA_PROJ_WIDTH)
    assert loaded.features.e_g.shape == (3, GENE_WIDTH)
    assert loaded.features.z_c.shape == (3, CONTEXT_WIDTH)
    assert loaded.features.delta_proj.dtype == torch.float32
    assert loaded.features.q_sc_mask.dtype == torch.bool
    assert torch.equal(loaded.features.z_c[0], loaded.features.z_c[2])
    assert loaded.features.gene_symbols == ("G2", "G1", "G3")
    assert loaded.features.model_ids == ("A", "A", "A")


def test_nonresume_refuses_nonempty_and_resume_skips_only_verified_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    with pytest.raises(FileExistsError, match="nonempty"):
        _build(root)
    before = (root / "shards" / "A.npz").stat().st_mtime_ns
    _, written = _build(root, resume=True)
    assert written == [False, False]
    assert (root / "shards" / "A.npz").stat().st_mtime_ns == before

    (root / "shards" / "A.npz").write_bytes(b"corrupt")
    _, written = _build(root, resume=True)
    assert written == [False, True]
    assert verify_geneeffect_feature_store(root)["status"] == "passed"

    writer = GeneEffectFeatureStoreWriter(
        root,
        stage="stage1_frozen",
        model_ids=("B", "A"),
        gene_symbols=("G2", "G1", "G3"),
        e_g=np.full((3, GENE_WIDTH), 777, dtype=np.float32),
        z_c=np.full((2, CONTEXT_WIDTH), 888, dtype=np.float32),
        gene_embedding_source_sha256=_digest("esm2"),
        feature_schema_sha256=_digest("schema"),
        projection_sha256=_digest("projection"),
        resume=True,
    )
    rewritten = [
        writer.write_shard(
            model_id,
            **_arrays(float(index)),
            source_sha256=_digest(f"source-{model_id}"),
            model_checkpoint_sha256=_digest("checkpoint"),
        )
        for index, model_id in enumerate(("B", "A"))
    ]
    writer.finalize()
    assert rewritten == [True, True]


def test_verifier_detects_tampering_reordered_genes_and_stale_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    stale = verify_geneeffect_feature_store(
        root, expected_checkpoint_sha256=_digest("new-checkpoint")
    )
    assert stale["status"] == "failed"
    assert any("stale model checkpoint" in item for item in stale["discrepancies"])
    stale = verify_geneeffect_feature_store(
        root,
        expected_feature_schema_sha256=_digest("new-schema"),
        expected_projection_sha256=_digest("new-projection"),
    )
    assert "stale feature schema hash" in stale["discrepancies"]
    assert "stale projection hash" in stale["discrepancies"]
    stale = verify_geneeffect_feature_store(
        root, expected_gene_embedding_source_sha256=_digest("other-esm2")
    )
    assert "stale gene embedding source hash" in stale["discrepancies"]
    membership = verify_geneeffect_feature_store(
        root,
        expected_model_ids=("A", "B"),
        expected_gene_symbols=("G1", "G2", "G3"),
    )
    assert any("expected membership" in item for item in membership["discrepancies"])
    assert any("expected universe" in item for item in membership["discrepancies"])

    path = root / "shards" / "A.npz"
    _rewrite_npz(path, gene_symbols=np.array(["G1", "G2", "G3"]))
    report = verify_geneeffect_feature_store(root)
    assert any("gene order mismatch" in item for item in report["discrepancies"])
    assert any("shard SHA-256 mismatch" in item for item in report["discrepancies"])


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"s": np.ones((3, 5), dtype=np.float32)}, "s must have shape"),
        (
            {"delta_proj": np.ones((3, DELTA_PROJ_WIDTH), dtype=np.float64)},
            "dtype float32",
        ),
        ({"q_sc_mask": np.ones(3, dtype=np.uint8)}, "dtype bool"),
        (
            {"q_sc": np.full((3, Q_SC_WIDTH), np.nan, dtype=np.float32)},
            "non-finite",
        ),
    ],
)
def test_verifier_reports_malformed_shard_without_crashing(
    tmp_path: Path, updates: dict[str, np.ndarray], message: str
) -> None:
    root = tmp_path / "features"
    _build(root)
    _rewrite_npz(root / "shards" / "A.npz", **updates)
    report = verify_geneeffect_feature_store(root)
    assert report["status"] == "failed"
    assert any(message in item for item in report["discrepancies"])


def test_verifier_rejects_own_shift_outside_hvg_panel(tmp_path: Path) -> None:
    root = tmp_path / "features"
    _build(root)
    _rewrite_npz(
        root / "shards" / "A.npz",
        hvg_panel_mask=np.array([False, True, False]),
        own_gene_shift_mask=np.array([True, False, False]),
    )
    report = verify_geneeffect_feature_store(root)
    assert report["status"] == "failed"
    assert any(
        "own_gene_shift_mask requires hvg_panel_mask" in item
        for item in report["discrepancies"]
    )


def test_verifier_rejects_context_dependent_hvg_panel_membership(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    _rewrite_npz(
        root / "shards" / "A.npz",
        hvg_panel_mask=np.array([False, True, False]),
        own_gene_shift_mask=np.array([False, False, False]),
    )
    report = verify_geneeffect_feature_store(root)
    assert report["status"] == "failed"
    assert any(
        "hvg_panel_mask differs from checkpoint-fixed mask" in item
        for item in report["discrepancies"]
    )


def test_verifier_reports_malformed_manifest_missing_and_extra_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_ids"] = ["A", "B"]
    manifest_path.write_text(json.dumps(manifest))
    report = verify_geneeffect_feature_store(root)
    assert report["status"] == "failed"
    assert any("model order mismatch" in item for item in report["discrepancies"])
    with pytest.raises(ValueError, match="authenticate resume"):
        _build(root, resume=True)

    paths_root = tmp_path / "path-errors"
    _build(paths_root)
    (paths_root / "shards" / "B.npz").unlink()
    (paths_root / "shards" / "EXTRA.npz").write_bytes(b"extra")
    (paths_root / "unexpected_dir").mkdir()
    report = verify_geneeffect_feature_store(paths_root)
    assert "missing shard: B.npz" in report["discrepancies"]
    assert "extra shard: EXTRA.npz" in report["discrepancies"]
    assert "extra path: unexpected_dir" in report["discrepancies"]

    paths_manifest = paths_root / "manifest.json"
    paths_manifest.write_text("not json")
    report = verify_geneeffect_feature_store(paths_root)
    assert report["status"] == "failed"
    assert "manifest unreadable" in report["discrepancies"][0]
    with pytest.raises(ValueError, match="authenticate resume"):
        _build(paths_root, resume=True)


def test_verifier_rejects_expected_shard_replaced_by_directory(tmp_path: Path) -> None:
    root = tmp_path / "features"
    _build(root)
    (root / "shards" / "A.npz").unlink()
    (root / "shards" / "A.npz").mkdir()
    report = verify_geneeffect_feature_store(root)
    assert report["status"] == "failed"
    assert "A: shard path is not a file" in report["discrepancies"]


def test_stage_identity_is_explicit_and_loader_never_reorders(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    selected = tmp_path / "selected"
    _build(frozen, stage="stage1_frozen")
    _build(selected, stage="stage2_selected")
    assert (
        verify_geneeffect_feature_store(frozen, expected_stage="stage2_selected")[
            "status"
        ]
        == "failed"
    )
    assert (
        verify_geneeffect_feature_store(selected, expected_stage="stage2_selected")[
            "status"
        ]
        == "passed"
    )
    with pytest.raises(ValueError, match="gene order"):
        load_geneeffect_feature_batch(
            selected,
            "A",
            expected_gene_symbols=("G1", "G2", "G3"),
            **{**_load_identity(), "expected_stage": "stage2_selected"},
        )


def test_verifier_rejects_mixed_checkpoint_identities(tmp_path: Path) -> None:
    mixed_root = tmp_path / "mixed-writer"
    writer = GeneEffectFeatureStoreWriter(
        mixed_root,
        stage="stage1_frozen",
        model_ids=("B", "A"),
        gene_symbols=("G2", "G1", "G3"),
        e_g=np.zeros((3, GENE_WIDTH), dtype=np.float32),
        z_c=np.zeros((2, CONTEXT_WIDTH), dtype=np.float32),
        gene_embedding_source_sha256=_digest("esm2"),
        feature_schema_sha256=_digest("schema"),
        projection_sha256=_digest("projection"),
    )
    writer.write_shard(
        "B",
        **_arrays(),
        source_sha256=_digest("source-B"),
        model_checkpoint_sha256=_digest("checkpoint-B"),
    )
    with pytest.raises(ValueError, match="one model checkpoint"):
        writer.write_shard(
            "A",
            **_arrays(1),
            source_sha256=_digest("source-A"),
            model_checkpoint_sha256=_digest("checkpoint-A"),
        )

    root = tmp_path / "features"
    _build(root)
    shard_path = root / "shards" / "A.npz"
    new_hash = _digest("other-checkpoint")
    _rewrite_npz(shard_path, model_checkpoint_sha256=np.asarray(new_hash))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"]["A"]["model_checkpoint_sha256"] = new_hash
    manifest["shards"]["A"]["sha256"] = hashlib.sha256(
        shard_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    report = verify_geneeffect_feature_store(root)
    assert report["status"] == "failed"
    assert "store mixes model checkpoint hashes" in report["discrepancies"]

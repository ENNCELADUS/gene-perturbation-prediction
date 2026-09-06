from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.experiments.exp13_legacy.geneeffect_feature_store as feature_store
from src.experiments.exp13_legacy.geneeffect_feature_store import (
    CONTEXT_WIDTH,
    DELTA_PROJ_WIDTH,
    GENE_WIDTH,
    Q_SC_WIDTH,
    SUMMARY_WIDTH,
    GeneEffectFeatureStoreWriter,
    GeneEffectFrozenFeatureCache,
    load_geneeffect_feature_batch,
)


GENES = ("G2", "G1", "G3")
MODELS = ("B", "A", "C")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _arrays(offset: float) -> dict[str, np.ndarray]:
    rows = np.arange(len(GENES), dtype=np.float32)[:, None] + offset
    return {
        "delta_proj": np.broadcast_to(rows, (len(GENES), DELTA_PROJ_WIDTH)).copy(),
        "s": np.broadcast_to(rows + 10, (len(GENES), SUMMARY_WIDTH)).copy(),
        "q_sc": np.broadcast_to(rows + 20, (len(GENES), Q_SC_WIDTH)).copy(),
        "q_sc_mask": np.array([True, False, True]),
        "hvg_panel_mask": np.array([True, False, True]),
        "own_gene_shift_mask": np.array([True, False, False]),
    }


def _build(root: Path) -> None:
    e_g = np.broadcast_to(
        np.arange(len(GENES), dtype=np.float32)[:, None],
        (len(GENES), GENE_WIDTH),
    ).copy()
    z_c = np.broadcast_to(
        (100 * np.arange(len(MODELS), dtype=np.float32))[:, None],
        (len(MODELS), CONTEXT_WIDTH),
    ).copy()
    writer = GeneEffectFeatureStoreWriter(
        root,
        stage="stage1_frozen",
        model_ids=MODELS,
        gene_symbols=GENES,
        e_g=e_g,
        z_c=z_c,
        gene_embedding_source_sha256=_digest("esm2"),
        feature_schema_sha256=_digest("schema"),
        projection_sha256=_digest("projection"),
    )
    for offset, model_id in enumerate(MODELS):
        writer.write_shard(
            model_id,
            **_arrays(1000 * offset),
            source_sha256=_digest(f"source-{model_id}"),
            model_checkpoint_sha256=_digest("checkpoint"),
        )
    writer.finalize()


def _identity() -> dict[str, object]:
    return {
        "expected_gene_symbols": GENES,
        "expected_model_ids": MODELS,
        "expected_stage": "stage1_frozen",
        "expected_checkpoint_sha256": _digest("checkpoint"),
        "expected_feature_schema_sha256": _digest("schema"),
        "expected_projection_sha256": _digest("projection"),
        "expected_source_sha256": {
            model_id: _digest(f"source-{model_id}") for model_id in MODELS
        },
        "expected_gene_embedding_source_sha256": _digest("esm2"),
    }


def _cache_identity() -> dict[str, object]:
    return {
        "expected_gene_symbols": GENES,
        "expected_model_ids": MODELS,
        "expected_stage": "stage1_frozen",
    }


def _load_cache(root: Path, selected: tuple[str, ...] = ("B", "A")):
    return GeneEffectFrozenFeatureCache.load(
        root,
        selected_model_ids=selected,
        device="cpu",
        **_cache_identity(),
    )


def _rewrite_shard(root: Path, model_id: str, **updates: np.ndarray) -> None:
    path = root / "shards" / f"{model_id}.npz"
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays.update(updates)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_cache_gather_matches_store_loader_and_preserves_repeated_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    loaded_b = load_geneeffect_feature_batch(root, "B", **_identity()).features
    loaded_a = load_geneeffect_feature_batch(root, "A", **_identity()).features
    pairs = ((2, 1), (0, 0), (2, 1), (1, 0))

    cache = _load_cache(root)
    gathered = cache.gather(pairs)
    expected_rows = (
        (loaded_a, 2),
        (loaded_b, 0),
        (loaded_a, 2),
        (loaded_b, 1),
    )
    for name in (
        "delta_proj",
        "s",
        "q_sc",
        "e_g",
        "z_c",
        "q_sc_mask",
        "hvg_panel_mask",
        "own_gene_shift_mask",
    ):
        expected = torch.stack(
            [getattr(batch, name)[row] for batch, row in expected_rows]
        )
        assert torch.equal(getattr(gathered, name), expected)
    assert gathered.gene_symbols == ("G3", "G2", "G3", "G1")
    assert gathered.model_ids == ("A", "B", "A", "B")
    assert cache.selected_model_ids == ("B", "A")
    assert cache.tensor_nbytes > 0


def test_cache_opens_each_selected_shard_once_and_never_opens_unselected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "features"
    _build(root)
    real_np_load = feature_store.np.load
    loaded_paths: list[str] = []

    def tracked_np_load(source, *args, **kwargs):
        loaded_paths.append(Path(source).name)
        return real_np_load(source, *args, **kwargs)

    monkeypatch.setattr(feature_store.np, "load", tracked_np_load)
    _load_cache(root)
    assert loaded_paths == ["genes.npz", "contexts.npz", "B.npz", "A.npz"]


def test_cache_context_mapping_supports_selected_order_permutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    cache = _load_cache(root, ("A", "B"))
    gathered = cache.gather(((0, 0), (0, 1), (0, 0)))
    assert gathered.model_ids == ("B", "A", "B")
    assert gathered.delta_proj[:, 0].tolist() == [0.0, 1000.0, 0.0]


def test_cache_rejects_duplicate_scope_out_of_scope_gather_and_use_after_close(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    with pytest.raises(ValueError, match="selected_model_ids must be unique"):
        _load_cache(root, ("B", "B"))
    with pytest.raises(ValueError, match="outside expected_model_ids"):
        _load_cache(root, ("B", "UNKNOWN"))

    cache = _load_cache(root, ("B",))
    with pytest.raises(ValueError, match="outside the frozen cache scope: A"):
        cache.gather(((0, 1),))
    with pytest.raises(IndexError, match="gene index is out of range"):
        cache.gather(((3, 0),))
    cache.close()
    with pytest.raises(RuntimeError, match="cache is closed"):
        cache.gather(((0, 0),))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"s": np.zeros((len(GENES), SUMMARY_WIDTH - 1), np.float32)}, "shape"),
        (
            {"delta_proj": np.zeros((len(GENES), DELTA_PROJ_WIDTH), np.float64)},
            "dtype float32",
        ),
        (
            {"q_sc": np.full((len(GENES), Q_SC_WIDTH), np.nan, np.float32)},
            "non-finite",
        ),
        ({"gene_symbols": np.array(["G1", "G2", "G3"])}, "gene order"),
    ],
)
def test_cache_fails_on_corrupt_selected_shard(
    tmp_path: Path, updates: dict[str, np.ndarray], message: str
) -> None:
    root = tmp_path / "features"
    _build(root)
    _rewrite_shard(root, "B", **updates)
    with pytest.raises(ValueError, match=message):
        _load_cache(root)


def test_cache_fails_on_duplicate_or_reordered_manifest_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    _build(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_ids"] = ["B", "B", "C"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest model_ids must be unique"):
        _load_cache(root)

    root = tmp_path / "reordered"
    _build(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["gene_symbols"] = ["G1", "G2", "G3"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gene order mismatch"):
        _load_cache(root)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    fold_artifact_dir,
    generator_manifest_path,
    generator_weights_path,
    load_embedding_cache,
    load_generator_manifest,
    save_embedding_cache,
    write_generator_manifest,
)
from sl_dl_model.exp08b_config import Exp08bConfig, load_exp08b_config


def test_exp08b_defaults_keep_distill_full_weight() -> None:
    cfg = Exp08bConfig()
    assert cfg.lambda_distill == 1.0
    assert cfg.lambda_distill_after_warmup == 1.0
    assert cfg.lambda_bag == 1.0
    assert cfg.lambda_sl == 0.0
    assert cfg.generator_kind == "state_adapter"
    assert cfg.generator_val_fraction == 0.2
    assert cfg.bag_scale_mode == "fixed_warmup"
    assert cfg.embedding_method == "exp08b_state_adapter_meanstd"


def test_load_exp08b_config_coerces_paths_and_tuples(tmp_path: Path) -> None:
    path = tmp_path / "exp08b.yaml"
    payload = {
        "input_csv": "data/pairs.csv",
        "output_dir": "results/exp08b/run",
        "split_types": ["CV2", "CV3"],
        "folds": [0, 4],
        "ranking_k": [10, 50],
        "esm2_npz": "data/esm2.npz",
        "bags_npz": "data/bags.npz",
        "generator_kind": "direct_mlp",
        "direct_mlp_hidden": 32,
    }
    path.write_text(yaml.safe_dump(payload))

    cfg = load_exp08b_config(path)

    assert cfg.input_csv == Path("data/pairs.csv")
    assert cfg.output_dir == Path("results/exp08b/run")
    assert cfg.split_types == ("CV2", "CV3")
    assert cfg.folds == (0, 4)
    assert cfg.ranking_k == (10, 50)
    assert cfg.esm2_npz == Path("data/esm2.npz")
    assert cfg.bags_npz == Path("data/bags.npz")
    assert cfg.generator_kind == "direct_mlp"
    assert cfg.direct_mlp_hidden == 32


def test_load_exp08b_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"unknown_field": 1}))

    try:
        load_exp08b_config(path)
    except ValueError as exc:
        assert "unknown config keys" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown_field")


def test_artifact_paths_are_fold_local(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")

    fold_dir = fold_artifact_dir(cfg, "CV2", 3)

    assert fold_dir == tmp_path / "run" / "step1_generator" / "CV2_fold3"
    assert embedding_cache_path(cfg, "CV2", 3).name == "predicted_embeddings.npz"
    assert generator_manifest_path(cfg, "CV2", 3).name == "generator_manifest.json"
    assert generator_weights_path(cfg, "CV2", 3).name == "generator_weights.pt"


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")
    symbols = np.array(["A", "B"], dtype=object)
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    coverage = np.array([1, 0], dtype=np.int64)
    path = embedding_cache_path(cfg, "CV2", 0)

    save_embedding_cache(
        path,
        symbols=symbols,
        embeddings=embeddings,
        coverage_mask=coverage,
        embedding_method="exp08b_state_adapter_meanstd",
    )
    loaded = load_embedding_cache(path)

    assert loaded["embedding_method"] == "exp08b_state_adapter_meanstd"
    assert loaded["symbols"].tolist() == ["A", "B"]
    np.testing.assert_allclose(loaded["embeddings"], embeddings)
    np.testing.assert_array_equal(loaded["coverage_mask"], coverage)


def test_generator_manifest_roundtrip(tmp_path: Path) -> None:
    cfg = Exp08bConfig(output_dir=tmp_path / "run")
    path = generator_manifest_path(cfg, "CV3", 4)
    payload = {
        "split_type": "CV3",
        "fold_id": 4,
        "bag_scale": 3.5,
        "train_bag_gene_count": 8,
        "val_bag_gene_count": 2,
    }

    write_generator_manifest(path, payload)
    loaded = load_generator_manifest(path)

    assert json.dumps(loaded, sort_keys=True) == json.dumps(payload, sort_keys=True)

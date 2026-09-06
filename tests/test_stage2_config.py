from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.experiments.exp13_legacy.stage2_config import load_stage2_config


TRACKED_YAML = Path("configs/experiments/13_geneeffect_226/stage2_e2e.yaml")


def _raw() -> dict[str, object]:
    return yaml.safe_load(TRACKED_YAML.read_text())


def _write(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "stage2.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_tracked_config_loads_frozen_plan() -> None:
    config = load_stage2_config(TRACKED_YAML)
    assert dataclasses.is_dataclass(config)
    assert config.features.context_dim == 5120
    assert config.seeds.projection == 20260828
    assert config.joint.genes_per_batch * config.joint.contexts_per_gene == 256
    assert config.distributed.mixed_precision == "bf16"
    assert config.run_scope.ablations == ()


@pytest.mark.parametrize(
    "section",
    [
        "seeds",
        "features",
        "loss",
        "warmup",
        "joint",
        "distributed",
        "lambda_calibration",
        "selection",
        "run_scope",
        "paths",
    ],
)
def test_unknown_nested_key_raises(tmp_path: Path, section: str) -> None:
    raw = _raw()
    raw[section]["misspelled"] = 1
    with pytest.raises(ValueError, match="misspelled"):
        load_stage2_config(_write(tmp_path, raw))


def test_missing_top_level_section_raises(tmp_path: Path) -> None:
    raw = _raw()
    del raw["loss"]
    with pytest.raises(ValueError, match="loss"):
        load_stage2_config(_write(tmp_path, raw))


def test_missing_nested_key_raises(tmp_path: Path) -> None:
    raw = _raw()
    del raw["joint"]["response_batch_size"]
    with pytest.raises(ValueError, match="response_batch_size"):
        load_stage2_config(_write(tmp_path, raw))


def test_batch_product_invariant_raises(tmp_path: Path) -> None:
    raw = _raw()
    raw["joint"]["contexts_per_gene"] = 31
    with pytest.raises(ValueError, match=r"genes_per_batch \* contexts_per_gene"):
        load_stage2_config(_write(tmp_path, raw))


def test_formal_distributed_precision_is_frozen(tmp_path: Path) -> None:
    raw = _raw()
    raw["distributed"]["mixed_precision"] = "fp16"
    with pytest.raises(ValueError, match="mixed_precision"):
        load_stage2_config(_write(tmp_path, raw))


def test_world_size_is_runtime_detected_not_configured(tmp_path: Path) -> None:
    raw = _raw()
    raw["distributed"]["world_size"] = 4
    with pytest.raises(ValueError, match="world_size"):
        load_stage2_config(_write(tmp_path, raw))


def test_fixed_width_invariant_raises(tmp_path: Path) -> None:
    raw = _raw()
    raw["features"]["context_dim"] = 128
    with pytest.raises(ValueError, match="feature dimensions"):
        load_stage2_config(_write(tmp_path, raw))


@pytest.mark.parametrize(
    ("key", "value"), [("model", "no_esm2"), ("num_seeds", 2), ("ablations", ["q_sc"])]
)
def test_full_model_only_invariant_raises(
    tmp_path: Path, key: str, value: object
) -> None:
    raw = _raw()
    raw["run_scope"][key] = value
    with pytest.raises(ValueError, match="full model, one seed, no ablations"):
        load_stage2_config(_write(tmp_path, raw))


def test_source_hash_and_snapshot_are_json_ready() -> None:
    config = load_stage2_config(TRACKED_YAML)
    assert config.source_sha256 == hashlib.sha256(TRACKED_YAML.read_bytes()).hexdigest()
    snapshot = config.snapshot()
    assert json.loads(json.dumps(snapshot))["paths"]["split"].endswith("split.json")
    assert snapshot["source_sha256"] == config.source_sha256


def test_path_validation_is_opt_in(tmp_path: Path) -> None:
    raw = _raw()
    raw["paths"] = {key: str(tmp_path / key) for key in raw["paths"]}
    path = _write(tmp_path, raw)
    load_stage2_config(path)
    with pytest.raises(ValueError, match="configured paths do not exist"):
        load_stage2_config(path, validate_paths=True)


def test_all_dataclasses_are_frozen() -> None:
    config = load_stage2_config(TRACKED_YAML)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seeds.train = 1

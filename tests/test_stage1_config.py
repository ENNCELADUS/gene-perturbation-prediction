from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

TRACKED_YAML = Path("configs/experiments/13_geneeffect_226/stage1_response.yaml")

VALID_TRAIN_BLOCK = """\
train:
  max_epochs: 50
  patience: 5
  learning_rate: 0.0001
  weight_decay: 0.01
  max_bag: 128
  gene_batch_size: 32
  validation_gene_batch_size: 1
  grad_clip: 1.0
  train_seed: 20260818
  collator_seed: 20260818
  data_seed: 42
  heldout_seed: 13
  heldout_fraction: 0.1
  log_every: 50
  float32_matmul_precision: high
  ddp_static_graph: true
  ddp_find_unused_parameters: false
  max_cells_per_gene: 128
  total_cells_per_line: null
  pert_dim: 2024
  max_esm2_drop_fraction: 0.10
  w_mean_delta: 1.0
  w_energy: 1.0
"""

VALID_OBJECTIVE_BLOCK = """\
objective:
  anchor_weights:
    ACH-000551: 0.25
    ACH-000995: 0.25
    ACH-000739: 0.25
    ACH-000971: 0.25
  required_anchor_metrics: [mean_delta_mse, energy_distance]
"""

VALID_YAML = VALID_TRAIN_BLOCK + VALID_OBJECTIVE_BLOCK


def _write(tmp_path: Path, text: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_tracked_yaml_loads_registered_objective() -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    cfg = load_stage1_config(TRACKED_YAML)
    assert cfg.train.max_epochs >= 1
    assert dict(cfg.objective.anchor_weights)["ACH-000551"] == 0.25


def test_valid_config_loads(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML)
    cfg = load_stage1_config(p)
    assert cfg.train.max_epochs == 50
    assert cfg.train.patience == 5
    assert cfg.train.float32_matmul_precision == "high"
    assert cfg.train.gene_batch_size == 32
    assert cfg.train.validation_gene_batch_size == 1
    assert cfg.train.total_cells_per_line is None
    assert dict(cfg.objective.anchor_weights) == {
        "ACH-000551": 0.25,
        "ACH-000995": 0.25,
        "ACH-000739": 0.25,
        "ACH-000971": 0.25,
    }
    assert cfg.objective.required_anchor_metrics == (
        "mean_delta_mse",
        "energy_distance",
    )


def test_required_anchor_metrics_must_include_both_metrics(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    incomplete = VALID_YAML.replace(
        "[mean_delta_mse, energy_distance]", "[mean_delta_mse]"
    )
    with pytest.raises(ValueError, match="must contain exactly"):
        load_stage1_config(_write(tmp_path, incomplete))


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML + "bogus_top_level: 1\n")
    with pytest.raises(ValueError, match="bogus_top_level"):
        load_stage1_config(p)


def test_misspelled_key_inside_train_raises_instead_of_taking_default(
    tmp_path: Path,
) -> None:
    """The CLAUDE.md failure mode this loader must not have: a typo'd YAML key
    silently falling back to a dataclass default instead of raising."""
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("max_epochs: 50", "max_epocsh: 50")
    p = _write(tmp_path, bad_train + VALID_OBJECTIVE_BLOCK)
    with pytest.raises(ValueError, match="max_epocsh"):
        load_stage1_config(p)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("  patience: 5\n", "")
    p = _write(tmp_path, bad_train + VALID_OBJECTIVE_BLOCK)
    with pytest.raises(ValueError, match="patience"):
        load_stage1_config(p)


def test_anchor_weights_not_summing_to_one_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_objective = VALID_OBJECTIVE_BLOCK.replace("ACH-000551: 0.25", "ACH-000551: 0.5")
    p = _write(tmp_path, VALID_TRAIN_BLOCK + bad_objective)
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_stage1_config(p)


def test_empty_anchor_weights_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_objective = """\
objective:
  anchor_weights: {}
  required_anchor_metrics: [mean_delta_mse, energy_distance]
"""
    p = _write(tmp_path, VALID_TRAIN_BLOCK + bad_objective)
    with pytest.raises(ValueError, match="non-empty"):
        load_stage1_config(p)


def test_negative_anchor_weight_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_objective = VALID_OBJECTIVE_BLOCK.replace(
        "ACH-000551: 0.25", "ACH-000551: -0.25"
    ).replace("ACH-000995: 0.25", "ACH-000995: 0.75")
    p = _write(tmp_path, VALID_TRAIN_BLOCK + bad_objective)
    with pytest.raises(ValueError, match="> 0"):
        load_stage1_config(p)


def test_max_bag_of_one_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("max_bag: 128", "max_bag: 1")
    p = _write(tmp_path, bad_train + VALID_OBJECTIVE_BLOCK)
    with pytest.raises(ValueError, match="max_bag"):
        load_stage1_config(p)


def test_gene_batch_size_of_zero_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("gene_batch_size: 32", "gene_batch_size: 0")
    p = _write(tmp_path, bad_train + VALID_OBJECTIVE_BLOCK)
    with pytest.raises(ValueError, match="gene_batch_size"):
        load_stage1_config(p)


def test_validation_gene_batch_size_of_zero_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace(
        "validation_gene_batch_size: 1", "validation_gene_batch_size: 0"
    )
    p = _write(tmp_path, bad_train + VALID_OBJECTIVE_BLOCK)
    with pytest.raises(ValueError, match="validation_gene_batch_size"):
        load_stage1_config(p)


def test_both_loss_weights_zero_raises(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace(
        "w_mean_delta: 1.0", "w_mean_delta: 0.0"
    ).replace("w_energy: 1.0", "w_energy: 0.0")
    p = _write(tmp_path, bad_train + VALID_OBJECTIVE_BLOCK)
    with pytest.raises(ValueError, match="weight"):
        load_stage1_config(p)


def test_source_sha256_matches_independently_computed_hash(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML)
    cfg = load_stage1_config(p)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert cfg.source_sha256 == expected
    assert cfg.source_path == p


def test_objective_payload_is_json_serializable(tmp_path: Path) -> None:
    from src.experiments.exp13_legacy.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML)
    cfg = load_stage1_config(p)
    payload = cfg.objective_payload()
    serialized = json.dumps(payload)
    round_tripped = json.loads(serialized)
    assert round_tripped["anchor_weights"]["ACH-000551"] == 0.25
    assert round_tripped["source_sha256"] == cfg.source_sha256

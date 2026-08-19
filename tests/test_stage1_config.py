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

VALID_THRESHOLDS_BLOCK = """\
freeze_thresholds:
  anchor_weights:
    ACH-000551: 0.25
    ACH-000995: 0.25
    ACH-000739: 0.25
    ACH-000971: 0.25
  required_anchor_metrics: [mean_delta_mse, energy_distance]
  min_improvement_over_basal_copy: 0.05
  min_improvement_over_null_shuffle: 0.1
"""

VALID_YAML = VALID_TRAIN_BLOCK + VALID_THRESHOLDS_BLOCK


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_tracked_yaml_train_block_is_well_formed_but_null_margins_raise() -> None:
    from aivc_model.stage1_config import load_stage1_config

    with pytest.raises(ValueError) as exc_info:
        load_stage1_config(TRACKED_YAML)
    message = str(exc_info.value)
    assert "min_improvement_over_basal_copy" in message or (
        "min_improvement_over_null_shuffle" in message
    )
    assert "§7" in message


def test_valid_config_loads(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML)
    cfg = load_stage1_config(p)
    assert cfg.train.max_epochs == 50
    assert cfg.train.patience == 5
    assert cfg.train.float32_matmul_precision == "high"
    assert cfg.train.total_cells_per_line is None
    assert dict(cfg.thresholds.anchor_weights) == {
        "ACH-000551": 0.25,
        "ACH-000995": 0.25,
        "ACH-000739": 0.25,
        "ACH-000971": 0.25,
    }
    assert cfg.thresholds.required_anchor_metrics == (
        "mean_delta_mse",
        "energy_distance",
    )
    assert cfg.thresholds.min_improvement_over_basal_copy == 0.05
    assert cfg.thresholds.min_improvement_over_null_shuffle == 0.1


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML + "bogus_top_level: 1\n")
    with pytest.raises(ValueError, match="bogus_top_level"):
        load_stage1_config(p)


def test_misspelled_key_inside_train_raises_instead_of_taking_default(
    tmp_path: Path,
) -> None:
    """The CLAUDE.md failure mode this loader must not have: a typo'd YAML key
    silently falling back to a dataclass default instead of raising."""
    from aivc_model.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("max_epochs: 50", "max_epocsh: 50")
    p = _write(tmp_path, bad_train + VALID_THRESHOLDS_BLOCK)
    with pytest.raises(ValueError, match="max_epocsh"):
        load_stage1_config(p)


def test_missing_required_key_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("  patience: 5\n", "")
    p = _write(tmp_path, bad_train + VALID_THRESHOLDS_BLOCK)
    with pytest.raises(ValueError, match="patience"):
        load_stage1_config(p)


def test_anchor_weights_not_summing_to_one_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    bad_thresholds = VALID_THRESHOLDS_BLOCK.replace(
        "ACH-000551: 0.25", "ACH-000551: 0.5"
    )
    p = _write(tmp_path, VALID_TRAIN_BLOCK + bad_thresholds)
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_stage1_config(p)


def test_empty_anchor_weights_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    bad_thresholds = """\
freeze_thresholds:
  anchor_weights: {}
  required_anchor_metrics: [mean_delta_mse, energy_distance]
  min_improvement_over_basal_copy: 0.05
  min_improvement_over_null_shuffle: 0.1
"""
    p = _write(tmp_path, VALID_TRAIN_BLOCK + bad_thresholds)
    with pytest.raises(ValueError, match="non-empty"):
        load_stage1_config(p)


def test_negative_anchor_weight_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    bad_thresholds = VALID_THRESHOLDS_BLOCK.replace(
        "ACH-000551: 0.25", "ACH-000551: -0.25"
    ).replace("ACH-000995: 0.25", "ACH-000995: 0.75")
    p = _write(tmp_path, VALID_TRAIN_BLOCK + bad_thresholds)
    with pytest.raises(ValueError, match="> 0"):
        load_stage1_config(p)


def test_max_bag_of_one_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace("max_bag: 128", "max_bag: 1")
    p = _write(tmp_path, bad_train + VALID_THRESHOLDS_BLOCK)
    with pytest.raises(ValueError, match="max_bag"):
        load_stage1_config(p)


def test_both_loss_weights_zero_raises(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    bad_train = VALID_TRAIN_BLOCK.replace(
        "w_mean_delta: 1.0", "w_mean_delta: 0.0"
    ).replace("w_energy: 1.0", "w_energy: 0.0")
    p = _write(tmp_path, bad_train + VALID_THRESHOLDS_BLOCK)
    with pytest.raises(ValueError, match="weight"):
        load_stage1_config(p)


def test_source_sha256_matches_independently_computed_hash(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML)
    cfg = load_stage1_config(p)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert cfg.source_sha256 == expected
    assert cfg.source_path == p


def test_freeze_thresholds_payload_is_json_serializable(tmp_path: Path) -> None:
    from aivc_model.stage1_config import load_stage1_config

    p = _write(tmp_path, VALID_YAML)
    cfg = load_stage1_config(p)
    payload = cfg.freeze_thresholds_payload()
    serialized = json.dumps(payload)
    round_tripped = json.loads(serialized)
    assert round_tripped["min_improvement_over_basal_copy"] == 0.05
    assert round_tripped["anchor_weights"]["ACH-000551"] == 0.25
    assert round_tripped["source_sha256"] == cfg.source_sha256

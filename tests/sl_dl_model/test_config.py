# tests/sl_dl_model/test_config.py
from pathlib import Path

import yaml

from sl_dl_model.config import SLDLConfig, load_config


def test_defaults_match_exp06_protocol():
    cfg = SLDLConfig()
    assert cfg.seed == 17
    assert cfg.ranking_k == (10, 20, 50)
    assert cfg.folds == (0, 1, 2, 3, 4)
    assert cfg.esm2_model == "facebook/esm2_t33_650M_UR50D"


def test_load_config_roundtrip(tmp_path: Path):
    payload = {
        "input_csv": "data/x.csv",
        "output_dir": "results/exp08/run",
        "split_types": ["CV2", "CV3"],
        "esm2_npz": "data/esm2.npz",
        "warmup_epochs": 3,
        "lambda_sl": 1.0,
        "lambda_distill": 0.5,
        "lambda_bag": 1.0,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    cfg = load_config(path)
    assert cfg.split_types == ("CV2", "CV3")
    assert cfg.esm2_npz == Path("data/esm2.npz")
    assert cfg.lambda_distill == 0.5


def test_load_config_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"nonsense_key": 1}))
    try:
        load_config(path)
    except ValueError as exc:
        assert "unknown config keys" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_early_stop_patience_default():
    cfg = SLDLConfig()
    assert cfg.early_stop_patience == 5
    assert cfg.batch_pairs == 1024


def test_load_config_accepts_early_stop_patience(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"early_stop_patience": 3, "batch_pairs": 256}))
    cfg = load_config(path)
    assert cfg.early_stop_patience == 3
    assert cfg.batch_pairs == 256

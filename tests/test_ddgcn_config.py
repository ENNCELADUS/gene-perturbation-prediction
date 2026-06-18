from __future__ import annotations

from pathlib import Path


def test_defaults_match_official_ddgcn() -> None:
    from ddgcn.config import DdgcnConfig

    c = DdgcnConfig()
    assert c.dropout == 0.5
    assert c.lr == 0.01
    assert c.hidden1 == 512
    assert c.hidden2 == 256
    assert c.init_type == "Kaiming"
    assert c.use_bias is False
    assert c.rho == 1.0
    assert c.max_epochs == 2000
    assert c.tolerance_epoch == 1000
    assert c.stop_threshold == 1e-5
    assert c.eval_interval == 50
    assert c.seed == 456
    assert c.ranking_k == (10, 20, 50)
    assert c.folds == (0, 1, 2, 3, 4)
    assert c.split_types is None


def test_load_config_coerces_paths_and_tuples(tmp_path: Path) -> None:
    from ddgcn.config import load_config

    yaml_text = (
        "input_csv: data/x.csv\n"
        "output_dir: results/y\n"
        "split_types: [CV1, CV2]\n"
        "folds: [0, 1]\n"
        "ranking_k: [10, 20]\n"
        "dropout: 0.5\n"
        "lr: 0.01\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    c = load_config(p)
    assert c.input_csv == Path("data/x.csv")
    assert c.output_dir == Path("results/y")
    assert c.split_types == ("CV1", "CV2")
    assert c.folds == (0, 1)
    assert c.ranking_k == (10, 20)


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from ddgcn.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("bogus_key: 1\n")
    try:
        load_config(p)
    except ValueError as exc:
        assert "bogus_key" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown key")

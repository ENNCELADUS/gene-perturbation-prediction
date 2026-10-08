"""The wave-one correction configs: one objective each, STATE absent, one prior."""

from pathlib import Path

import pytest

from src.experiments.config import load_config

CONFIGS = sorted(Path("configs/correction").glob("*.yaml"))
SCREEN = sorted(Path("configs/correction").glob("*_no_state.yaml"))
EXPORT = "outputs/context_prior/default_prior_export/export"


def test_four_objective_screen_configs_differ_only_in_objective():
    assert [p.stem for p in SCREEN] == [
        "dependency_classification_no_state",
        "huber_no_state",
        "line_ranking_no_state",
        "standardized_mse_no_state",
    ]
    configs = [load_config(p) for p in SCREEN]
    for path, config in zip(SCREEN, configs):
        assert config["train"]["objective"] == path.stem.removesuffix("_no_state")
        assert config["paths"]["prior"] == EXPORT
        assert config["output_root"] == "outputs/geneeffect_correction"
        blocks = config["model"]["head_blocks"]
        assert not blocks["use_delta_proj"] and not blocks["use_s"]
    for config in configs:
        config["train"]["objective"] = None
    assert all(config == configs[0] for config in configs)


def test_state_configs_differ_from_the_screen_choice_only_in_state():
    chosen = load_config(Path("configs/correction/standardized_mse_no_state.yaml"))
    for mode in ("frozen", "trainable"):
        config = load_config(
            Path(f"configs/correction/standardized_mse_{mode}_state.yaml")
        )
        assert config["train"]["state_mode"] == mode
        blocks = config["model"]["head_blocks"]
        assert blocks["use_delta_proj"] and blocks["use_s"]
        assert config["train"]["response_weight"] == 0.0
        config["train"]["state_mode"] = chosen["train"]["state_mode"]
        config["model"]["head_blocks"] = chosen["model"]["head_blocks"]
        assert config == chosen


@pytest.mark.parametrize("path", CONFIGS)
def test_correction_configs_validate(path):
    load_config(path)

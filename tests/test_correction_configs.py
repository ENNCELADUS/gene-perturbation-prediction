"""The wave-one correction configs: one objective each, STATE absent, one prior."""

from pathlib import Path

import pytest

from src.experiments.config import load_config

CONFIGS = sorted(Path("configs/correction").glob("*.yaml"))
EXPORT = "outputs/context_prior/default_prior_export/export"


def test_four_objective_screen_configs_differ_only_in_objective():
    assert [p.stem for p in CONFIGS] == [
        "dependency_classification_no_state",
        "huber_no_state",
        "line_ranking_no_state",
        "standardized_mse_no_state",
    ]
    configs = [load_config(p) for p in CONFIGS]
    for path, config in zip(CONFIGS, configs):
        assert config["train"]["objective"] == path.stem.removesuffix("_no_state")
        assert config["paths"]["prior"] == EXPORT
        assert config["output_root"] == "outputs/geneeffect_correction"
        blocks = config["model"]["head_blocks"]
        assert not blocks["use_delta_proj"] and not blocks["use_s"]
    for config in configs:
        config["train"]["objective"] = None
    assert all(config == configs[0] for config in configs)


@pytest.mark.parametrize("path", CONFIGS)
def test_correction_configs_validate(path):
    load_config(path)

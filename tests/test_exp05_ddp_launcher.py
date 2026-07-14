from pathlib import Path

import yaml


CONFIG_PATH = Path(
    "configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_5fold.yaml"
)


def test_exp05_config_is_e2e_response_gmm_ddp() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert "projector" not in config
    assert "scvi_obsm_key" not in config["data"]
    assert config["response_encoder"] == {"input_dim": 2000, "latent_dim": 128}
    assert config["gmm"] == {
        "n_components": 64,
        "covariance_floor": 0.0001,
        "init_scale": 0.02,
        "trainable": True,
    }
    assert config["data"]["prepared_cache_dir"].endswith(
        "k562_gwps_state2000_v2"
    )
    assert config["train"]["required_world_size"] == 4
    assert config["train"]["gene_batch_size"] == 1
    assert config["train"]["learning_rate"] == 0.000025
    assert config["train"]["state_learning_rate"] == 0.0000025
    assert config["train"]["max_grad_norm"] == 1.0
    assert config["train"]["run_id"] == "state_esm2_response_gmm_ddp_outer5"
    assert "freeze_state" not in config["train"]
    assert config["loss"]["occupancy_weight"] == 0.1
    assert config["loss"]["gmm_nll_weight"] == 0.01


def test_launcher_uses_exactly_four_processes() -> None:
    script = Path("scripts/run_exp05_ddp.sh").read_text()

    assert "--num_processes 4" in script
    assert "--mixed_precision bf16" in script
    assert "aivc_model.cross_validate" in script


def test_state_slurm_script_delegates_to_shared_launcher() -> None:
    script = Path("scripts/state.sh").read_text()

    assert script.rstrip().endswith('srun scripts/run_exp05_ddp.sh "$CONFIG_PATH"')
    assert "accelerate launch" not in script

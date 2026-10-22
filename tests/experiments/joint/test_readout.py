"""The readout entry point on a tiny synthetic joint checkpoint."""

import json

import pytest

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

from tests.fixtures import joint as joint_world  # noqa: E402
from src.experiments.joint.readout import run as readout  # noqa: E402
from src.experiments.joint.train import run_training  # noqa: E402


@pytest.fixture
def nine_training_lines(monkeypatch):
    # The readout's context PCA needs at least nine training contexts and eight
    # context components.
    monkeypatch.setattr(
        joint_world, "TRAIN", (*joint_world.ANCHORS, *(f"ACH-T{i}" for i in range(6)))
    )
    monkeypatch.setattr(joint_world, "CONTEXT_COMPONENTS", 8)


def test_run_readout_writes_val_metrics_and_skips_when_done(
    tmp_path, monkeypatch, nine_training_lines
):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    config = joint_world.make_config(tmp_path)
    config["train"]["max_epochs"] = 1
    inputs = joint_world.make_inputs()
    best = run_training(config, tmp_path / "train", inputs=inputs)

    out_dir = tmp_path / "readout"
    metrics = readout.run_readout(best, out_dir, device="cpu", inputs=inputs)

    assert json.loads((out_dir / "metrics.json").read_text()) == metrics
    assert (out_dir / "features" / "metadata.json").is_file()
    assert metrics["val_geneeffect_loss"] > 0
    for key in (
        "val_residual_pearson_macro_per_gene",
        "val_residual_spearman_macro_per_gene",
        "val_residual_sd_ratio_macro_per_gene",
        "val_geneeffect_pearson_macro_per_line",
        "val_selective_spearman",
        "val_selective_aupr_lift",
    ):
        assert key in metrics
    cache_metadata = json.loads((out_dir / "features" / "metadata.json").read_text())
    assert cache_metadata["selective_genes"] == [
        gene for gene in inputs.genes if gene in inputs.selective_genes
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("a finished readout was recomputed")

    monkeypatch.setattr(readout, "extract_cache", forbidden)
    assert readout.run_readout(best, out_dir, device="cpu") == metrics

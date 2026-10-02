"""The readout entry point on a tiny synthetic joint checkpoint."""

import json

import pytest

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

import test_joint  # noqa: E402
from src.experiments import readout  # noqa: E402
from src.experiments.geneeffect import run_training  # noqa: E402


@pytest.fixture
def nine_training_lines(monkeypatch):
    # The readout's context PCA needs at least nine training contexts.
    monkeypatch.setattr(
        test_joint, "TRAIN", (*test_joint.ANCHORS, *(f"ACH-T{i}" for i in range(6)))
    )


def test_run_readout_writes_val_metrics_and_skips_when_done(
    tmp_path, monkeypatch, nine_training_lines
):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    config = test_joint.make_config(tmp_path)
    config["train"]["max_epochs"] = 1
    inputs = test_joint.make_inputs()
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
    ):
        assert key in metrics

    def forbidden(*args, **kwargs):
        raise AssertionError("a finished readout was recomputed")

    monkeypatch.setattr(readout, "extract_cache", forbidden)
    assert readout.run_readout(best, out_dir, device="cpu") == metrics

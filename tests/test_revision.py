"""The revision route: summary and revision.json from synthetic run directories,
argument handling, step skipping and config binding. No GPU, no real data.

``paired_line_bootstrap`` belongs to ``src.eval.metrics`` (added with the revision's
metric changes); the tests stub it so they do not depend on its numerics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.experiments import all as pipeline
from src.experiments import revision

SELECTIVE = ["G1", "G2"]
CONFIG = {
    "output_root": "unused",
    "precision": "bf16",
    "prepared_root": "unused",
    "train": {"patience": 5, "max_epochs": 30},
}


def metric_row(spearman, *, split="val", aupr=0.1, pearson=0.2, huber=0.3, sd=0.9):
    return {
        f"{split}_selective_spearman": spearman,
        f"{split}_selective_aupr_lift": aupr,
        f"{split}_residual_pearson_macro_per_gene": pearson,
        f"{split}_geneeffect_loss": huber,
        f"{split}_residual_sd_ratio_macro_per_gene": sd,
        f"{split}_unrelated": 99.0,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def frame(method=None, value=0.0):
    rows = pd.DataFrame(
        {
            "model_id": ["L1", "L2", "L1", "L2"],
            "gene_symbol": ["G1", "G1", "G2", "G2"],
            "residual": [0.1, 0.2, 0.3, 0.4],
            "residual_prediction": [value] * 4,
        }
    )
    return rows if method is None else rows.assign(method=method)


@pytest.fixture
def finished(tmp_path) -> Path:
    """A finished run directory: training record, evaluation and baselines on
    validation and test (test scores are validation's minus 0.01)."""
    run = tmp_path / "run"
    write_json(run / "train" / "done.json", {"best_epoch": 1, "next_epoch": 4})
    records = [
        {"epoch": 0, "global_step": 1, "train_geneeffect_loss": 1.0},
        {
            "epoch": 0,
            "global_step": 2,
            "train_eval_selective_spearman": 0.5,
            "val_selective_spearman": 0.01,
        },
        {"epoch": 1, "global_step": 3, "train_geneeffect_loss": 0.9},
        {
            "epoch": 1,
            "global_step": 4,
            "train_eval_selective_spearman": 0.6,
            "val_selective_spearman": 0.07,
        },
        {
            "epoch": 2,
            "global_step": 6,
            "train_eval_selective_spearman": 0.8,
            "val_selective_spearman": 0.03,
        },
    ]
    (run / "train" / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    torch.save({"preprocessing": {"selective_genes": SELECTIVE}}, run / "train/best.pt")
    for split, shift in (("val", 0.0), ("test", 0.01)):
        write_json(
            run / f"evaluation/{split}/metrics.json",
            metric_row(0.07 - shift, split=split),
        )
        frame(value=0.2).to_parquet(run / f"evaluation/{split}/predictions.parquet")
        write_json(
            run / f"baselines/{split}/metrics.json",
            {
                "gene_mean": metric_row(None, split=split, pearson=None),
                "context_pca_ridge[tx1]": metric_row(0.05 - shift, split=split),
                "custom_method": metric_row(0.02 - shift, split=split),
                "nearest_line[tx1]": metric_row(0.04 - shift, split=split),
            },
        )
        pd.concat(
            [frame("context_pca_ridge[tx1]", 0.1), frame("gene_mean", 0.0)]
        ).to_parquet(run / f"baselines/{split}/predictions.parquet")
    return run


@pytest.fixture
def bootstrap(monkeypatch):
    calls = []

    def fake(left, right, selective_genes, *, repeats, seed):
        calls.append((left, right, selective_genes, repeats, seed))
        return {"difference": 0.02, "interval": [-0.01, 0.05]}

    monkeypatch.setattr("src.eval.metrics.paired_line_bootstrap", fake, raising=False)
    return calls


def test_revision_json_and_summary(finished, bootstrap, monkeypatch):
    monkeypatch.setattr(revision, "_git_revision", lambda: "abc123")
    summary = revision.write_outputs(Path("configs/revision/x.yaml"), finished, "rid")
    assert summary == finished / "summary.md"

    record = json.loads((finished / "revision.json").read_text())
    assert record["run_id"] == "rid"
    assert record["config"] == "configs/revision/x.yaml"
    assert record["git_revision"] == "abc123"
    assert list(record) == [
        "run_id",
        "config",
        "git_revision",
        "val",
        "test",
        "training",
    ]
    validation = record["val"]["models"]
    # Joint model first, then the baselines in the documented order, unknown last.
    assert list(validation) == [
        "Joint model",
        "Gene mean",
        "Nearest line (Tx1)",
        "Context-PCA ridge (Tx1)",
        "custom_method",
    ]
    assert validation["Joint model"] == {
        "selective_spearman": 0.07,
        "selective_aupr_lift": 0.1,
        "residual_pearson": 0.2,
        "huber": 0.3,
        "sd_ratio": 0.9,
    }
    # Undefined stays null, never 0.
    assert validation["Gene mean"]["selective_spearman"] is None
    assert validation["Gene mean"]["residual_pearson"] is None
    assert record["test"]["models"]["Joint model"][
        "selective_spearman"
    ] == pytest.approx(0.06)
    assert record["test"]["models"]["Context-PCA ridge (Tx1)"][
        "selective_spearman"
    ] == pytest.approx(0.04)
    for split in ("val", "test"):
        assert record[split]["bootstrap"] == {
            "comparison": "Joint model minus Context-PCA ridge (Tx1)",
            "repeats": 1000,
            "seed": 0,
            "difference": 0.02,
            "interval": [-0.01, 0.05],
        }
    # Best epoch 1 (0-based) is the second epoch; its own record, not the last.
    assert record["training"] == {
        "best_epoch": 2,
        "epochs_trained": 4,
        "train_eval_selective_spearman": 0.6,
        "val_selective_spearman": 0.07,
    }

    text = summary.read_text()
    assert text.startswith("# Revision run rid")
    assert "`abc123`" in text and "configs/revision/x.yaml" in text
    validation_text, test_text = text.split("## Test lines")
    assert "## Validation lines" in validation_text

    def table(section):
        return {
            line.split(" | ")[0].removeprefix("| "): line.split(" | ")[1:]
            for line in section.splitlines()
            if line.startswith("| ")
        }

    assert table(validation_text)["Model"][0] == "Selective Spearman"
    assert table(validation_text)["Joint model"][:3] == ["0.0700", "0.1000", "0.2000"]
    assert table(validation_text)["Gene mean"][0] == "undefined"
    assert table(test_text)["Joint model"][0] == "0.0600"
    assert "| Context-PCA ridge (Tx1) |" in test_text
    assert "0.0200 [-0.0100, 0.0500], paired bootstrap over validation" in text
    assert "0.0200 [-0.0100, 0.0500], paired bootstrap over test" in text
    assert "epoch 2; 4 epochs trained" in text
    assert "0.6000 on the training diagnostic" in text
    assert "0.0700 on the validation lines" in text


def test_best_before_the_first_update_is_the_prior_alone(
    finished, bootstrap, monkeypatch
):
    monkeypatch.setattr(revision, "_git_revision", lambda: "abc123")
    write_json(finished / "train" / "done.json", {"best_epoch": -1, "next_epoch": 3})
    before = {
        "epoch": -1,
        "global_step": 0,
        "train_eval_selective_spearman": 0.4,
        "val_selective_spearman": 0.09,
    }
    path = finished / "train" / "metrics.jsonl"
    path.write_text(json.dumps(before) + "\n" + path.read_text())
    summary = revision.write_outputs(Path("configs/revision/x.yaml"), finished, "rid")
    record = json.loads((finished / "revision.json").read_text())
    assert record["training"]["best_epoch"] == 0
    assert record["training"]["val_selective_spearman"] == 0.09
    text = summary.read_text()
    assert "the model before its first update (the prior alone)" in text
    assert "3 epochs trained" in text


def test_bootstrap_inputs_are_the_joint_and_tx1_ridge_rows(finished, bootstrap):
    revision.build_record(Path("c.yaml"), finished, "rid", "rev")
    assert len(bootstrap) == 2
    for left, right, selective, repeats, seed in bootstrap:
        assert set(left.residual_prediction) == {0.2}
        assert set(right.method) == {"context_pca_ridge[tx1]"}
        assert set(right.residual_prediction) == {0.1} and len(right) == 4
        assert selective == frozenset(SELECTIVE)
        assert (repeats, seed) == (1000, 0)


def test_summary_requires_the_tx1_ridge_and_a_best_epoch_record(finished, bootstrap):
    metrics = json.loads((finished / "baselines/test/metrics.json").read_text())
    del metrics["context_pca_ridge[tx1]"]
    write_json(finished / "baselines/test/metrics.json", metrics)
    with pytest.raises(ValueError, match="baselines/test/metrics.json has no"):
        revision.build_record(Path("c.yaml"), finished, "rid", "rev")
    write_json(finished / "train/done.json", {"best_epoch": 7, "next_epoch": 8})
    with pytest.raises(ValueError, match="no epoch record for epoch 7"):
        revision._training_record(finished)


def test_nan_difference_is_written_as_null(finished, monkeypatch):
    monkeypatch.setattr(
        "src.eval.metrics.paired_line_bootstrap",
        lambda *args, **kwargs: {"difference": math.nan, "interval": [math.nan, 1.0]},
        raising=False,
    )
    record = revision.build_record(Path("c.yaml"), finished, "rid", "rev")
    assert record["val"]["bootstrap"]["difference"] is None
    assert record["val"]["bootstrap"]["interval"] == [None, 1.0]
    json.dumps(record, allow_nan=False)


def test_main_passes_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        revision,
        "run_revision",
        lambda config, **kwargs: calls.append((config, kwargs)),
    )
    assert revision.main(["c.yaml", "--run-id", "r", "--gpus", "1, 3"]) == 0
    assert revision.main(["c.yaml"]) == 0
    assert calls == [
        (Path("c.yaml"), {"run_id": "r", "gpus": ("1", "3")}),
        (Path("c.yaml"), {"run_id": None, "gpus": None}),
    ]


class FakeProcess:
    def poll(self):
        return 0


@pytest.fixture
def world(tmp_path, monkeypatch):
    """run_revision with config, GPUs and preparation replaced; returns call logs."""
    config = {**CONFIG, "output_root": str(tmp_path / "runs")}
    prepared = []
    monkeypatch.setattr(revision, "load_config", lambda path: config)
    monkeypatch.setattr(revision, "visible_gpus", lambda: ())
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0)
    monkeypatch.setattr(
        "src.experiments.prepare.prepare_inputs", lambda cfg: prepared.append(cfg)
    )
    return config, prepared


def test_finished_steps_start_nothing_and_summary_is_rewritten(
    world, finished, bootstrap, monkeypatch
):
    config, prepared = world
    run = Path(config["output_root"]) / "done"
    run.parent.mkdir(parents=True)
    finished.rename(run)

    def refuse(*args, **kwargs):
        raise AssertionError("a finished step ran again")

    monkeypatch.setattr("src.experiments.geneeffect.run_training", refuse)
    monkeypatch.setattr("src.experiments.geneeffect.evaluate_checkpoint", refuse)
    monkeypatch.setattr("src.experiments.baselines.run_baselines", refuse)
    monkeypatch.setattr(revision, "start_process", refuse)
    assert revision.run_revision(Path("c.yaml"), run_id="done") == run
    assert prepared == [config]
    assert (run / "revision.json").is_file() and (run / "summary.md").is_file()


def test_unfinished_run_trains_then_evaluates_val_then_test(
    world, finished, bootstrap, monkeypatch
):
    config, _ = world
    run = Path(config["output_root"]) / "fresh"
    calls = []
    # Keep the evaluation and baseline outputs the fakes will rewrite, per split.
    outputs = {
        (kind, split): {
            p.name: p.read_bytes() for p in (finished / kind / split).iterdir()
        }
        for kind in ("evaluation", "baselines")
        for split in ("val", "test")
    }
    train_files = {
        p.name: p.read_bytes() for p in (finished / "train").iterdir() if p.is_file()
    }

    def fake_training(cfg, train):
        calls.append(("train", train))
        train.mkdir(parents=True)
        for name, content in train_files.items():
            (train / name).write_bytes(content)

    def fake_evaluate(checkpoint, *, split):
        calls.append(("evaluate", checkpoint, split))
        return split

    def fake_export(split, out_dir):
        assert out_dir == run / "evaluation" / split
        out_dir.mkdir(parents=True)
        for name, content in outputs["evaluation", split].items():
            (out_dir / name).write_bytes(content)

    def fake_baselines(cfg, *, split, out_dir):
        calls.append(("baselines", split))
        out_dir.mkdir(parents=True)
        for name, content in outputs["baselines", split].items():
            (out_dir / name).write_bytes(content)

    monkeypatch.setattr("src.experiments.geneeffect.run_training", fake_training)
    monkeypatch.setattr("src.experiments.geneeffect.evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr("src.experiments.geneeffect.export_evaluation", fake_export)
    monkeypatch.setattr("src.experiments.baselines.run_baselines", fake_baselines)
    assert revision.run_revision(Path("c.yaml"), run_id="fresh") == run
    assert calls == [
        ("train", run / "train"),
        ("evaluate", run / "train" / "best.pt", "val"),
        ("baselines", "val"),
        ("evaluate", run / "train" / "best.pt", "test"),
        ("baselines", "test"),
    ]
    assert (run / "summary.md").is_file()


def test_default_run_id_is_revision_timestamp(world, monkeypatch, capsys):
    config, _ = world

    def stop(*args, **kwargs):
        raise RuntimeError("stop after the run id")

    monkeypatch.setattr(revision, "train_joint", stop)
    with pytest.raises(RuntimeError, match="stop after"):
        revision.run_revision(Path("c.yaml"), run_id=None)
    (run,) = Path(config["output_root"]).iterdir()
    assert run.name.startswith("revision_")
    assert pd.Timestamp(run.name.removeprefix("revision_")).tzinfo is not None
    assert f"run id: {run.name}" in capsys.readouterr().out


def test_run_directory_refuses_a_different_config(world, monkeypatch):
    config, _ = world
    monkeypatch.setattr(revision, "train_joint", lambda *args, **kwargs: None)
    monkeypatch.setattr(revision, "_evaluate", lambda *args: None)
    monkeypatch.setattr(revision, "write_outputs", lambda *args: Path("s.md"))
    revision.run_revision(Path("c.yaml"), run_id="bound")
    revision.run_revision(Path("c.yaml"), run_id="bound")
    monkeypatch.setattr(
        revision, "load_config", lambda path: {**config, "precision": "no"}
    )
    with pytest.raises(ValueError, match="different config"):
        revision.run_revision(Path("c.yaml"), run_id="bound")


def test_training_uses_every_chosen_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0)
    started = []

    def launch(argv, gpus, log):
        started.append((tuple(argv), tuple(gpus), log))
        return FakeProcess()

    def refuse(*args, **kwargs):
        raise AssertionError("a GPU run trained in the orchestrating process")

    monkeypatch.setattr("src.experiments.geneeffect.run_training", refuse)
    run = tmp_path / "run"
    revision.train_joint(Path("c.yaml"), CONFIG, run, ("1", "3"), start=launch)
    ((argv, gpus, log),) = started
    assert gpus == ("1", "3")
    assert log == run / "logs" / "train.log"
    assert argv[argv.index("--num_processes") + 1] == "2"
    assert "--multi_gpu" in argv
    assert argv[-6:] == (
        "--module",
        "src.train",
        "--config",
        "c.yaml",
        "--run-dir",
        str(run / "train"),
    )


def test_training_resume_needs_the_started_gpu_count(tmp_path):
    train = tmp_path / "run" / "train"
    train.mkdir(parents=True)
    (train / "last.pt").write_bytes(b"")
    write_json(train / "run.json", {"world_size": 2})
    with pytest.raises(ValueError, match="started on 2 GPU.*not 4.*new --run-id"):
        revision.train_joint(
            Path("c.yaml"), CONFIG, tmp_path / "run", ("0", "1", "2", "3")
        )


def test_cpu_training_runs_in_process_and_finished_training_is_skipped(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "src.experiments.geneeffect.run_training",
        lambda config, train: calls.append(train),
    )
    run = tmp_path / "run"
    revision.train_joint(Path("c.yaml"), CONFIG, run, ())
    assert calls == [run / "train"]
    write_json(run / "train" / "done.json", {})
    revision.train_joint(Path("c.yaml"), CONFIG, run, ())
    assert calls == [run / "train"]

"""The ``all`` run end to end on CPU, from a synthetic raw world to summary.md.

The raw world is ``test_prepare.build_world`` widened to STATE's 2000 HVGs (the
joint model's width) and ten labeled training lines (the readout needs at least
nine), so the real ``prepare_inputs`` runs. STATE is a tiny released-shaped
checkpoint with a three-entry one-hot vocabulary covering one of the two
response genes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

import test_prepare  # noqa: E402
from src.experiments import all as pipeline  # noqa: E402
from src.model.state import build_state  # noqa: E402

VOCABULARY = ("GA", "GC", "non-targeting")


def write_state(state_dir: Path) -> Path:
    hparams = dict(
        input_dim=len(test_prepare.HVG),
        hidden_dim=8,
        output_dim=len(test_prepare.HVG),
        pert_dim=len(VOCABULARY),
        batch_dim=3,
        batch_encoder=True,
        cell_set_len=4,
        predict_residual=True,
        output_space="gene",
        embed_key="X_hvg",
        gene_names=list(test_prepare.HVG),
        transformer_backbone_kwargs={
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 2,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
        },
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(1)
    path = state_dir / "final.ckpt"
    torch.save(
        {"hyper_parameters": hparams, "state_dict": build_state(hparams).state_dict()},
        path,
    )
    onehot = torch.eye(len(VOCABULARY))
    torch.save(
        dict(zip(VOCABULARY, onehot, strict=True)), state_dir / "pert_onehot_map.pt"
    )
    return path


def make_world(root: Path, monkeypatch, *, max_epochs: int = 1) -> Path:
    """Write the raw world and its config; return the config path."""
    hvg = tuple(f"H{i}" for i in range(2000))
    monkeypatch.setattr(test_prepare, "HVG", hvg)
    monkeypatch.setattr(
        test_prepare,
        "SOURCE_GENES",
        (*hvg, *test_prepare.PANEL_CANDIDATES, "X1", "X2"),
    )
    monkeypatch.setattr(
        test_prepare,
        "TRAIN",
        (*test_prepare.ANCHORS, *(f"ACH-T{i}" for i in range(6)), "ACH-U1"),
    )
    config = test_prepare.build_world(root)
    config["precision"] = "no"
    config["output_root"] = str(root / "runs")
    config["paths"]["state_checkpoint"] = str(write_state(root / "state"))
    config["model"].update(
        cell_sentence_len=4, esm2_adapter_hidden=4, head_hidden=8, head_layers=1
    )
    config["train"].update(
        max_epochs=max_epochs,
        patience=5,
        dependency_batch_size=8,
        response_batch_size=4,
    )
    config["comparison"].update(
        epochs=1, hidden=8, batch_size=4, shuffles=2, bootstrap=20
    )
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """One complete run with spies on every route to the test split."""
    from src.data import prepared
    from src.experiments import baselines, geneeffect, response_comparison

    calls = {"evaluate": [], "baselines": [], "load_inputs": []}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ACCELERATE_USE_CPU", "true")
        torch.set_num_threads(1)
        config_path = make_world(tmp_path_factory.mktemp("world"), patch)

        def spy(name, original):
            def wrapped(*args, **kwargs):
                calls[name].append(kwargs)
                return original(*args, **kwargs)

            return wrapped

        patch.setattr(
            geneeffect,
            "evaluate_checkpoint",
            spy("evaluate", geneeffect.evaluate_checkpoint),
        )
        patch.setattr(
            baselines, "run_baselines", spy("baselines", baselines.run_baselines)
        )
        load = spy("load_inputs", prepared.load_inputs)
        patch.setattr(prepared, "load_inputs", load)
        patch.setattr(response_comparison, "load_inputs", load)
        run = pipeline.run_all(config_path, run_id="smoke")
    return run, calls


def test_all_end_to_end_writes_summary(finished):
    run, _ = finished
    assert run.name == "smoke"
    for path in (
        "comparison/verdicts.json",
        "train/best.pt",
        "train/done.json",
        "evaluation/val/metrics.json",
        "baselines/val/metrics.json",
        "readout/metrics.json",
    ):
        assert (run / path).is_file(), path
    summary = (run / "summary.md").read_text()
    manifest = json.loads(
        (run.parent.parent / "prepared" / "prepared_inputs.json").read_text()
    )
    assert f"T = {manifest['expression_space']['target_sum']:.1f}" in summary
    assert "Jurkat (ACH-000995) and HepG2 (ACH-000739)" in summary
    assert "STATE sanity line, released checkpoint untrained" in summary
    assert "(1 of 2 genes in its vocabulary)" in summary
    for arm in pipeline.ARM_NAMES.values():
        assert f"| {arm} |" in summary
    assert "- STATE as in the joint model vs MLP on HVG: all folds:" in summary
    assert "- MLP on Tx1 vs MLP on HVG: all folds:" in summary
    rows = [
        "Joint model",
        "Readout head with the explicit gene-specific context slope",
        *pipeline.BASELINE_NAMES.values(),
    ]
    table = {
        line.split(" | ")[0].removeprefix("| "): line.split(" | ")[1:]
        for line in summary.splitlines()
        if line.startswith("| ")
    }
    for row in rows:
        assert row in table and "not produced" not in table[row][-1], row
    # Gene mean is constant per gene across lines: undefined, never 0.
    assert table["Gene mean"][2] == "undefined"
    assert "training ran to the maximum of 1 epochs" in summary
    assert "A2" not in summary


def test_all_never_touches_test_split(finished):
    run, calls = finished
    assert [c["split"] for c in calls["evaluate"]] == ["val"]
    assert [c["split"] for c in calls["baselines"]] == ["val"]
    assert calls["load_inputs"]
    assert not any(c.get("include_test") for c in calls["load_inputs"])
    assert not [p for p in run.rglob("*") if p.name == "test"]


def test_all_resumes_and_skips(tmp_path, monkeypatch, cpu):
    from src.experiments import response_comparison
    from src.training import trainer

    config_path = make_world(tmp_path, monkeypatch, max_epochs=2)
    original = trainer.save_checkpoint

    def interrupt_after_first_last(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if Path(path).name == "last.pt":
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "save_checkpoint", interrupt_after_first_last)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_all(config_path, run_id="resume")
    run = tmp_path / "runs" / "resume"
    assert (run / "train" / "last.pt").is_file()
    assert not (run / "train" / "done.json").exists()
    assert (run / "comparison" / "verdicts.json").is_file()
    monkeypatch.setattr(trainer, "save_checkpoint", original)

    finished = [
        *(run / "comparison").rglob("*.json"),
        *(tmp_path / "prepared").rglob("*.*"),
    ]
    stamps = {p: p.stat().st_mtime_ns for p in finished}

    def refuse(*args, **kwargs):
        raise AssertionError("a finished comparison was recomputed")

    monkeypatch.setattr(response_comparison, "run_comparison", refuse)
    resumed = []
    fit = trainer.fit

    def spy_fit(*args, restored=None, **kwargs):
        resumed.append(None if restored is None else restored["train_state"])
        return fit(*args, restored=restored, **kwargs)

    monkeypatch.setattr(trainer, "fit", spy_fit)
    assert pipeline.run_all(config_path, run_id="resume") == run
    assert [state["next_epoch"] for state in resumed] == [1]
    assert json.loads((run / "train" / "done.json").read_text())["next_epoch"] == 2
    assert {p: p.stat().st_mtime_ns for p in finished} == stamps
    assert (run / "summary.md").is_file()

    # A finished run only rewrites its summary.
    monkeypatch.setattr(trainer, "fit", refuse)
    monkeypatch.setattr("src.experiments.readout.extract_cache", refuse)
    pipeline.run_all(config_path, run_id="resume")

    # The run directory refuses a different config instead of mixing experiments.
    changed = yaml.safe_load(config_path.read_text())
    changed["comparison"]["epochs"] += 1
    config_path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="different config"):
        pipeline.run_all(config_path, run_id="resume")


def test_gpu_schedule(tmp_path):
    config = {"precision": "bf16"}
    run = tmp_path / "run"
    config_path = Path("c.yaml")
    [[comparison, train]] = pipeline.gpu_schedule(
        config_path, config, run, ("3", "5", "7")
    )
    assert comparison.step == "comparison" and comparison.gpus == ("7",)
    assert comparison.argv[1:3] == ("-m", "src.experiments.response_comparison")
    assert train.step == "train" and train.gpus == ("3", "5")
    assert train.argv[train.argv.index("--num_processes") + 1] == "2"
    assert "--multi_gpu" in train.argv
    assert train.argv[-6:] == (
        "--module",
        "src.train",
        "--config",
        "c.yaml",
        "--run-dir",
        str(run / "train"),
    )

    two = pipeline.gpu_schedule(config_path, config, run, ("3", "7"))
    assert [[job.gpus for job in batch] for batch in two] == [[("7",), ("3",)]]
    assert "--multi_gpu" not in two[0][1].argv

    single = pipeline.gpu_schedule(config_path, config, run, ("0",))
    assert [[job.step for job in batch] for batch in single] == [
        ["comparison"],
        ["train"],
    ]
    assert single[1][0].gpus == ("0",)

    (run / "comparison").mkdir(parents=True)
    (run / "comparison" / "verdicts.json").write_text("{}")
    assert [
        [job.step for job in batch]
        for batch in pipeline.gpu_schedule(config_path, config, run, ("3", "7"))
    ] == [["train"]]


def test_failed_job_raises_with_step_and_log(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0.01)
    ok = pipeline.Job("comparison", (sys.executable, "-c", "print('fine')"), ("7",))
    bad = pipeline.Job(
        "train",
        (
            sys.executable,
            "-c",
            "import os, sys; print(os.environ['CUDA_VISIBLE_DEVICES']); sys.exit(3)",
        ),
        ("3", "5"),
    )
    with pytest.raises(RuntimeError) as error:
        pipeline.run_jobs([ok, bad], tmp_path)
    log = tmp_path / "logs" / "train.log"
    assert "train failed with exit code 3" in str(error.value)
    assert str(log) in str(error.value)
    assert "comparison" not in str(error.value)
    assert log.read_text().strip() == "3,5"
    assert (tmp_path / "logs" / "comparison.log").read_text().strip() == "fine"


def test_summary_marks_missing_baseline(finished, tmp_path):
    run, _ = finished
    copied = tmp_path / "copy"
    for name in (
        "train/done.json",
        "evaluation/val/metrics.json",
        "readout/metrics.json",
        "baselines/val/metrics.json",
    ):
        (copied / name).parent.mkdir(parents=True, exist_ok=True)
        (copied / name).write_text((run / name).read_text())
    metrics = json.loads((copied / "baselines/val/metrics.json").read_text())
    del metrics["nearest_line[hvg]"]
    (copied / "baselines/val/metrics.json").write_text(json.dumps(metrics))
    config = copy.deepcopy(yaml.safe_load((run / "train" / "config.yaml").read_text()))
    lines = pipeline._validation_section(config, copied)
    assert any(line.startswith("| Nearest line (HVG) | not produced") for line in lines)

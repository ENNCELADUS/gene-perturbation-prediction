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
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

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
    # The untrained arms run before training, the trained jobs after it.
    assert (run / "comparison" / "sanity.json").is_file()
    assert not list((run / "comparison" / "folds").glob("mlp_*"))
    assert not (run / "comparison" / "verdicts.json").exists()
    monkeypatch.setattr(trainer, "save_checkpoint", original)

    finished = [
        *(run / "comparison").rglob("*.json"),
        *(tmp_path / "prepared").rglob("*.*"),
    ]
    stamps = {p: p.stat().st_mtime_ns for p in finished}

    def refuse(*args, **kwargs):
        raise AssertionError("a finished comparison was recomputed")

    monkeypatch.setattr(response_comparison, "_run_untrained", refuse)
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
    assert (run / "comparison" / "verdicts.json").is_file()
    assert (run / "summary.md").is_file()

    # A finished run only rewrites its summary.
    monkeypatch.setattr(trainer, "fit", refuse)
    monkeypatch.setattr(response_comparison, "_run_trained", refuse)
    monkeypatch.setattr("src.experiments.readout.extract_cache", refuse)
    pipeline.run_all(config_path, run_id="resume")

    # The run directory refuses a different config instead of mixing experiments.
    changed = yaml.safe_load(config_path.read_text())
    changed["comparison"]["epochs"] += 1
    config_path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="different config"):
        pipeline.run_all(config_path, run_id="resume")


class FakeProcess:
    """Finishes with ``code`` on its ``polls + 1``-th poll."""

    def __init__(self, polls: int, code: int):
        self.left, self.code, self.done = polls, code, False

    def poll(self):
        if self.left:
            self.left -= 1
            return None
        self.done = True
        return self.code


class FakeLauncher:
    """Records every start and fails if a GPU would hold two processes at once."""

    def __init__(self, polls=lambda step: 1, codes=lambda step: 0):
        self.polls, self.codes = polls, codes
        self.started = []  # (step, argv, gpus, process)
        self.most_at_once = 0

    def __call__(self, argv, gpus, log):
        live = [entry for entry in self.started if not entry[3].done]
        busy = {gpu for entry in live for gpu in entry[2]}
        assert not busy & set(gpus), f"{log.stem} on busy GPU {gpus}"
        self.most_at_once = max(self.most_at_once, len(live) + 1)
        step = log.stem
        process = FakeProcess(self.polls(step), self.codes(step))
        self.started.append((step, tuple(argv), tuple(gpus), process))
        return process


TWELVE_JOBS = [
    (arm, anchor)
    for anchor in test_prepare.ANCHORS
    for arm in ("state_joint", "mlp_hvg", "mlp_tx1")
]


@pytest.fixture
def fake_comparison(monkeypatch):
    """Twelve pending trained jobs; summarise and in-process training refuse."""
    from src.experiments import geneeffect, response_comparison

    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0)
    monkeypatch.setattr(
        response_comparison,
        "pending_trained_jobs",
        lambda config, out_dir: list(TWELVE_JOBS),
    )
    summarised = []
    monkeypatch.setattr(
        response_comparison,
        "summarise",
        lambda config, out_dir: summarised.append(out_dir),
    )

    def refuse(*args, **kwargs):
        raise AssertionError("a GPU run trained in the orchestrating process")

    monkeypatch.setattr(geneeffect, "run_training", refuse)
    return summarised


def _plan(tmp_path, gpus, polls=lambda step: 1):
    launcher = FakeLauncher(polls)
    run = tmp_path / "run"
    pipeline.comparison_and_training(
        Path("c.yaml"), {"precision": "bf16"}, run, gpus, start=launcher
    )
    return run, launcher


def test_every_gpu_step_uses_every_chosen_gpu(tmp_path, fake_comparison):
    # Job lengths vary so that GPUs free out of order.
    run, launcher = _plan(
        tmp_path, ("0", "1", "2", "3"), polls=lambda step: len(step) % 4
    )
    steps = [entry[0] for entry in launcher.started]
    assert steps[:2] == ["comparison_untrained", "train"]
    untrained, train = launcher.started[0], launcher.started[1]
    assert untrained[2] == ("0",)
    assert untrained[1][1:3] == ("-m", "src.experiments.response_comparison")
    assert untrained[1][-1] == "--untrained"
    assert train[2] == ("0", "1", "2", "3")
    argv = train[1]
    assert argv[argv.index("--num_processes") + 1] == "4"
    assert argv[argv.index("--mixed_precision") + 1] == "bf16"
    assert "--multi_gpu" in argv
    assert argv[-6:] == (
        "--module",
        "src.train",
        "--config",
        "c.yaml",
        "--run-dir",
        str(run / "train"),
    )
    jobs = launcher.started[2:]
    assert sorted(entry[1][-2:] for entry in jobs) == sorted(TWELVE_JOBS)
    assert [entry[0] for entry in jobs] == [
        f"comparison_{arm}__{anchor}" for arm, anchor in TWELVE_JOBS
    ]
    assert all(len(entry[2]) == 1 for entry in jobs)
    assert {entry[2][0] for entry in jobs} == {"0", "1", "2", "3"}
    assert launcher.most_at_once == 4
    assert all(entry[3].done for entry in launcher.started)
    assert fake_comparison == [run / "comparison"]


def test_chosen_gpus_restrict_every_step(tmp_path, fake_comparison):
    gpus = pipeline.choose_gpus(("1", "3"), ("0", "1", "2", "3"))
    assert gpus == ("1", "3")
    _, launcher = _plan(tmp_path, gpus)
    untrained, train, *jobs = launcher.started
    assert untrained[2] == ("1",)
    assert train[2] == ("1", "3")
    assert train[1][train[1].index("--num_processes") + 1] == "2"
    assert "--multi_gpu" in train[1]
    assert len(jobs) == 12 and {entry[2] for entry in jobs} == {("1",), ("3",)}


def test_choose_gpus_defaults_to_every_visible_and_rejects_others():
    visible = ("4", "5", "6")
    assert pipeline.choose_gpus(None, visible) == visible
    assert pipeline.choose_gpus(None, ()) == ()
    for wrong in (("7",), ("4", "4"), ("",)):
        with pytest.raises(ValueError, match="visible ones 4,5,6"):
            pipeline.choose_gpus(wrong, visible)
    with pytest.raises(ValueError, match="no CUDA GPU"):
        pipeline.choose_gpus(("0",), ())


def test_one_gpu_trains_in_one_process(tmp_path, fake_comparison):
    _, launcher = _plan(tmp_path, ("0",))
    train = launcher.started[1]
    assert train[1][train[1].index("--num_processes") + 1] == "1"
    assert "--multi_gpu" not in train[1]
    assert {entry[2] for entry in launcher.started} == {("0",)}
    assert launcher.most_at_once == 1


def test_finished_steps_start_nothing(tmp_path, fake_comparison):
    run = tmp_path / "run"
    for name in ("comparison/verdicts.json", "train/done.json"):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("{}")
    _, launcher = _plan(tmp_path, ("0", "1"))
    assert launcher.started == [] and fake_comparison == []


def test_resume_on_another_gpu_count_is_refused(tmp_path, fake_comparison):
    train = tmp_path / "run" / "train"
    train.mkdir(parents=True)
    (train / "last.pt").write_bytes(b"")
    (train / "run.json").write_text(json.dumps({"world_size": 2}))
    (tmp_path / "run" / "comparison").mkdir()
    (tmp_path / "run" / "comparison" / "verdicts.json").write_text("{}")
    with pytest.raises(ValueError, match="started on 2 GPU.*not 4.*new --run-id"):
        _plan(tmp_path, ("0", "1", "2", "3"))
    _, launcher = _plan(tmp_path, ("0", "2"))
    assert [entry[0] for entry in launcher.started] == ["train"]


def test_failed_pool_job_lets_running_jobs_finish_and_starts_no_more(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0)
    launcher = FakeLauncher(
        polls=lambda step: {"slow": 3, "broken": 0}.get(step, 0),
        codes=lambda step: 3 if step == "broken" else 0,
    )
    jobs = [
        pipeline.Job(step, ("true",)) for step in ("slow", "broken", "next", "last")
    ]
    with pytest.raises(RuntimeError) as error:
        pipeline.run_pool(jobs, [("0",), ("1",)], tmp_path / "logs", start=launcher)
    assert [entry[0] for entry in launcher.started] == ["slow", "broken"]
    assert launcher.started[0][3].done
    message = str(error.value)
    assert message == (
        f"broken failed with exit code 3; see {tmp_path / 'logs' / 'broken.log'}"
    )


def test_processes_see_their_gpus_and_log_output(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "POLL_SECONDS", 0.01)
    show = "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
    ok = pipeline.Job("comparison_untrained", (sys.executable, "-c", show))
    bad = pipeline.Job("train", (sys.executable, "-c", show + "; exit(3)"))
    with pytest.raises(RuntimeError, match="train failed with exit code 3") as error:
        pipeline.run_pool([ok, bad], [("7",), ("0", "1", "2", "3")], tmp_path)
    assert "comparison_untrained" not in str(error.value)
    assert (tmp_path / "train.log").read_text().strip() == "0,1,2,3"
    assert (tmp_path / "comparison_untrained.log").read_text().strip() == "7"


def test_sigterm_terminates_child_process_groups(tmp_path):
    """A run killed with SIGTERM leaves no worker behind, grandchildren included."""
    group_file = tmp_path / "group"
    script = f"""
from pathlib import Path
from src.experiments import all as pipeline
pipeline.POLL_SECONDS = 0.05
child = pipeline.Job("sleeper", ("sh", "-c", "sleep 60 & echo $$ > {group_file}; wait"))
with pipeline.sigterm_raises():
    pipeline.run_pool([child], [("0",)], Path({str(tmp_path)!r}))
"""
    runner = subprocess.Popen(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1]
    )
    deadline = time.monotonic() + 30
    while not group_file.is_file() or not group_file.read_text().strip():
        assert runner.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    group = int(group_file.read_text())
    os.killpg(group, 0)  # the child group is alive
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=30) == 128 + signal.SIGTERM
    while True:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "the sleeping child survived SIGTERM"
        time.sleep(0.05)


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


def test_main_passes_gpus_outside_the_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "run_all", lambda config, **kwargs: calls.append((config, kwargs))
    )
    assert pipeline.main(["c.yaml", "--run-id", "r", "--gpus", "1, 3"]) == 0
    assert pipeline.main(["c.yaml"]) == 0
    assert calls == [
        (Path("c.yaml"), {"run_id": "r", "gpus": ("1", "3")}),
        (Path("c.yaml"), {"run_id": None, "gpus": None}),
    ]


def test_in_process_gpu_work_uses_first_chosen_gpu(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "visible_gpus", lambda: ("0", "1", "2", "3"))
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)

    def stop(config):
        raise RuntimeError("stop after the device is set")

    monkeypatch.setattr("src.experiments.prepare.prepare_inputs", stop)
    config_path = tmp_path / "config.yaml"
    config = test_prepare.build_world(tmp_path)
    config["output_root"] = str(tmp_path / "runs")
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(RuntimeError, match="stop after"):
        pipeline.run_all(config_path, run_id="device", gpus=("1", "3"))
    assert calls == [1]

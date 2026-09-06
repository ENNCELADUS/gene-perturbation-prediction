"""Fake environment executables inspect launch arguments without GPUs."""

import json
import os
import subprocess

import pytest


@pytest.fixture
def executable(tmp_path):
    binary = tmp_path / "python with spaces"
    binary.write_text("""#!/usr/bin/env python3
import json,os,sys
with open(os.environ["ARG_LOG"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:2] == ["-c"]:
    print(os.environ.get("FAKE_GPUS", "2"))
""")
    binary.chmod(0o755)
    log = tmp_path / "arguments.jsonl"
    env = dict(
        os.environ, PYTHON_BIN=str(binary), ARG_LOG=str(log), CUDA_VISIBLE_DEVICES="3,7"
    )
    return env, log


@pytest.mark.parametrize(
    "command, expected",
    [
        (["prepare", "a b.yaml"], ["-m", "src.experiments.prepare", "a b.yaml"]),
        (
            ["test", "checkpoint x.pt"],
            [
                "-m",
                "src.evaluate",
                "--checkpoint",
                "checkpoint x.pt",
                "--split",
                "test",
            ],
        ),
    ],
)
def test_single_process_dispatch(executable, command, expected):
    env, log = executable
    subprocess.run(["hpc/run.sh", *command], env=env, check=True)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [expected]


def test_visible_gpu_workers_and_resume_preserved(executable):
    env, log = executable
    subprocess.run(
        ["hpc/run.sh", "train", "a b.yaml", "--resume", "last x.pt"],
        env=env,
        check=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["-c", "import torch; print(torch.cuda.device_count())"]
    assert calls[1] == [
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "2",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--multi_gpu",
        "--module",
        "src.train",
        "a b.yaml",
        "--resume",
        "last x.pt",
    ]


def test_help_and_no_gpu(executable):
    env, log = executable
    subprocess.run(["hpc/run.sh", "--help"], env=env, check=True, capture_output=True)
    assert not log.exists()
    env["FAKE_GPUS"] = "0"
    completed = subprocess.run(
        ["hpc/run.sh", "train", "config", "--run-id", "run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "No visible GPUs" in completed.stderr
    assert len(log.read_text().splitlines()) == 1

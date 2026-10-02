"""A fake environment executable inspects the launcher's arguments without GPUs."""

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
""")
    binary.chmod(0o755)
    log = tmp_path / "arguments.jsonl"
    env = dict(os.environ, PYTHON_BIN=str(binary), ARG_LOG=str(log))
    return env, log


@pytest.mark.parametrize(
    "command, expected",
    [
        (["all", "a b.yaml"], ["-m", "src.experiments.all", "a b.yaml"]),
        (
            ["all", "a b.yaml", "--run-id", "run x"],
            ["-m", "src.experiments.all", "a b.yaml", "--run-id", "run x"],
        ),
        (
            ["all", "a b.yaml", "--run-id", "r", "--gpus", "1,3"],
            ["-m", "src.experiments.all", "a b.yaml", "--run-id", "r", "--gpus", "1,3"],
        ),
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
def test_dispatch(executable, command, expected):
    env, log = executable
    subprocess.run(["hpc/run.sh", *command], env=env, check=True)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [expected]


@pytest.mark.parametrize("command", ["prepare", "train"])
def test_retired_commands_rejected(executable, command):
    env, log = executable
    completed = subprocess.run(
        ["hpc/run.sh", command, "config.yaml"], env=env, capture_output=True, text=True
    )
    assert completed.returncode == 2
    assert "all CONFIG [--run-id ID]" in completed.stderr
    assert not log.exists()


def test_help(executable):
    env, log = executable
    completed = subprocess.run(
        ["hpc/run.sh", "--help"], env=env, check=True, capture_output=True, text=True
    )
    assert "hpc/run.sh all CONFIG [--run-id ID] [--gpus 0,1,2,3]" in completed.stdout
    assert "hpc/run.sh test CHECKPOINT" in completed.stdout
    assert "prepare" not in completed.stdout and "train" not in completed.stdout
    assert not log.exists()

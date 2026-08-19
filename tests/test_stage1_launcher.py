import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path("scripts/run_stage1_response_ddp.sh")


def _script_text() -> str:
    return SCRIPT_PATH.read_text()


def test_launcher_pins_accelerate_flags() -> None:
    script = _script_text()

    assert "--mixed_precision bf16" in script
    assert "--num_machines 1" in script


def test_launcher_is_strict_bash() -> None:
    script = _script_text()

    assert "set -euo pipefail" in script


def test_pythonpath_includes_src_and_repo_root() -> None:
    script = _script_text()

    assert 'export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"' in script


def test_launcher_never_hardcodes_num_processes() -> None:
    script = _script_text()

    assert "--num_processes 4" not in script
    assert '--num_processes "$NUM_PROCESSES"' in script


def test_launcher_references_tx1_venv() -> None:
    script = _script_text()

    assert ".venv-tx1" in script


def test_assemble_phase_runs_before_accelerate_launch() -> None:
    script = _script_text()

    assemble_index = script.index("--assemble-only")
    accelerate_launch_index = script.index('"$ACCELERATE_BIN" launch')

    assert assemble_index < accelerate_launch_index


def test_fails_fast_on_unset_response_cache_dir() -> None:
    env = dict(os.environ)
    env.pop("RESPONSE_CACHE_DIR", None)
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "RESPONSE_CACHE_DIR" in result.stderr

"""The cell-state telemetry CLI (protocol §17): the ``fit``, ``project``, ``inputs``
and ``report`` steps under ``<ladder_root>/states/``, each skipping what exists, so
rerunning resumes. It reports and decides nothing."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.experiments.run_dir import absolute, open_manifest, record_step
from src.experiments.states.inputs import step_inputs
from src.experiments.states.report import step_report
from src.experiments.states.telemetry import MANIFEST, Telemetry, settings
from src.experiments.states.vocabularies import step_fit, step_project

_LOGGER = logging.getLogger(__name__)

RUNNERS = {
    "fit": step_fit,
    "project": step_project,
    "inputs": step_inputs,
    "report": step_report,
}


def run(config: Mapping[str, Any], name: str, steps: Sequence[str]) -> None:
    t = Telemetry(config, name)
    t.layout.root.mkdir(parents=True, exist_ok=True)
    open_manifest(t.layout.root / MANIFEST, settings(t), keep_nan=True)
    for step in steps:
        _LOGGER.info("%s: start", step)
        record_step(t.layout.root / MANIFEST, step, RUNNERS[step](t), keep_nan=True)
        _LOGGER.info("%s: done", step)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in RUNNERS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {tuple(RUNNERS)}")
    from src.experiments.states.config import load_states_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(load_states_config(absolute(args.config)), args.config.stem, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

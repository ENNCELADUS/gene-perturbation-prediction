"""Config-driven entry point for inverse perturbation retrieval pipelines."""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

from src.utils.config import load_config, set_seed, validate_config

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the minimal CLI contract."""
    parser = argparse.ArgumentParser(description="Reverse perturbation retrieval")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    return parser.parse_args(argv)


def import_stage_runner(model_name: str, stage_name: str) -> Callable[[dict], dict]:
    """Import a model stage runner."""
    module = importlib.import_module(f"src.{model_name}.{stage_name}")
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise ValueError(f"src.{model_name}.{stage_name} must expose run(config)")
    return runner


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate config and execute configured stages in order."""
    validate_config(config)
    run_config = config["run_config"]
    model_name = config["model_config"]["model"]
    set_seed(run_config.get("seed"))

    results: list[dict[str, Any]] = []
    for stage_name in run_config["stages"]:
        LOGGER.info("Running %s.%s", model_name, stage_name)
        runner = import_stage_runner(model_name, stage_name)
        stage_result = runner(config)
        if not isinstance(stage_result, dict):
            stage_result = {"result": stage_result}
        results.append({"stage": stage_name, **stage_result})
    return {"model": model_name, "stages": results}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    config = load_config(args.config)
    results = run_from_config(config)
    LOGGER.info("Completed %s stage(s)", len(results["stages"]))
    return results


if __name__ == "__main__":
    main()

"""CLI for the exp10 DDGCN reproduction.

Usage::

    uv run python -m ddgcn run-cv \\
        --config configs/experiments/10_k562_sl_pair_ddgcn/ddgcn_cv.yaml

Add ``--split-type CV2`` to run a single split (partial reruns). ``--help``
works without importing torch (imports are deferred into ``main``).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ddgcn",
        description="exp10 DDGCN reproduction on the K562 SL-pair benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run-cv",
        help="Run CV and write official metrics to the configured output dir.",
    )
    run.add_argument(
        "--config", type=Path, required=True, help="Path to a DdgcnConfig YAML file."
    )
    run.add_argument(
        "--split-type",
        choices=["CV1", "CV2", "CV3"],
        default=None,
        help="Run only this CV split (overrides config.split_types).",
    )
    run.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write log output (appended).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m ddgcn``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    args = _build_parser().parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    from ddgcn.config import load_config
    from ddgcn.evaluate import run_cv

    config = load_config(args.config)
    if args.split_type is not None:
        config = dataclasses.replace(config, split_types=(args.split_type,))
    summary = run_cv(config)
    logging.getLogger(__name__).info(
        "wrote %d summary rows to %s", len(summary), config.output_dir
    )


if __name__ == "__main__":
    main()

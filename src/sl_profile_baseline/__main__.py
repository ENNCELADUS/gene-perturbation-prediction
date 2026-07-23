"""CLI for ``python -m sl_profile_baseline``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sl_profile_baseline.config import load_config
from sl_profile_baseline.evaluate import run_cv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate two DepMap GeneEffect profiles on Feng2024 splits."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the configured profile-only benchmark."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    summary = run_cv(config)
    logging.getLogger(__name__).info(
        "wrote %d summary rows to %s", len(summary), config.output_dir
    )


if __name__ == "__main__":
    main()


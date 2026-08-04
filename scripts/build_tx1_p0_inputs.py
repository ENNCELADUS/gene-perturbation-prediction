#!/usr/bin/env python3
"""Materialize authenticated development-only Tx1 P0 baseline inputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aivc_model.tx1_p0_inputs import build_p0_inputs, write_p0_inputs

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expression", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and write P0 inputs."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = build_p0_inputs(
        phase_a_dir=args.phase_a_dir,
        manifest_path=args.line_manifest,
        gene_effect_path=args.gene_effect,
        cache_root=args.cache_root,
        expression_path=args.expression,
    )
    write_p0_inputs(result, args.out_dir)
    _LOGGER.info("Wrote P0 inputs to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the feature-file-driven Tx1 P0 representation audit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aivc_model.tx1_p0_representation import run_audit

_LOGGER = logging.getLogger(__name__)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("representation must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("representation must use non-empty NAME=PATH")
    return name, Path(raw_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-plan",
        type=Path,
        required=True,
        help="Stable JSON emitted by build_tx1_p0_validation.py.",
    )
    parser.add_argument(
        "--validation-policy",
        type=Path,
        required=True,
        help="Registered policy used to generate and validate the plan.",
    )
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        type=_named_path,
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--gene-effect",
        type=Path,
        required=True,
        help="Long CSV: model_id,gene_symbol,gene_effect.",
    )
    parser.add_argument(
        "--shared-prior",
        type=Path,
        help="Optional shared prior CSV: gene_symbol,gene_effect.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=20260804)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the audit and write one stable-key JSON artifact."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    named_paths = dict(args.representation)
    if len(named_paths) != len(args.representation):
        raise ValueError("representation names must be unique")
    result = run_audit(
        args.manifest,
        args.validation_plan,
        args.validation_policy,
        args.phase_a_dir,
        named_paths,
        args.gene_effect,
        shared_prior_path=args.shared_prior,
        pca_components=args.pca_components,
        ridge_alpha=args.ridge_alpha,
        shuffle_seed=args.shuffle_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info("Wrote diagnostic P0 audit to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preflight or start the fail-closed Exp13 Stage 2 E2E runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aivc_model.geneeffect_stage2_runner import preflight_stage2, run_full_stage2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage1-provenance",
        type=Path,
        required=True,
        help=(
            "strict JSON maps for compatibility-code/config/source paths in the "
            "Stage 1 seal"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="authenticate every configured input without writing a run directory",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "run the same non-mutating preflight and report that training did not start"
        ),
    )
    mode.add_argument(
        "--run-id",
        help="create a fresh formal run and enter the full training path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight or args.dry_run:
        state = preflight_stage2(args.config, args.stage1_provenance)
        payload = dict(state.report)
        if args.dry_run:
            payload.update(
                {
                    "status": "dry_run_passed",
                    "training_started": False,
                    "completion_written": False,
                }
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    run_full_stage2(args.config, args.stage1_provenance, run_id=args.run_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

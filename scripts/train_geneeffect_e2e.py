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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="validate every configured input without writing a run directory",
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
        help="create a fresh run and enter the full training path",
    )
    parser.add_argument(
        "--reuse-frozen-feature-store",
        type=Path,
        help="import an authenticated stage1_frozen store into a fresh run",
    )
    args = parser.parse_args(argv)
    if args.reuse_frozen_feature_store is not None and args.run_id is None:
        parser.error("--reuse-frozen-feature-store requires --run-id")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight or args.dry_run:
        state = preflight_stage2(args.config)
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
    run_full_stage2(
        args.config,
        run_id=args.run_id,
        reuse_frozen_feature_store=args.reuse_frozen_feature_store,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

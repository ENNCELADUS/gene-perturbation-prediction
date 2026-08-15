#!/usr/bin/env python3
"""Build the local-evidence-only Tx1 P0 exposure ledger."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from aivc_model.tx1_p0_exposure import (
    build_exposure_ledger,
    write_exposure_ledger,
)

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--validation-policy", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write the exposure ledger."""
    args = parse_args(argv)
    ledger, summary = build_exposure_ledger(
        args.manifest,
        validation_plan_path=args.validation_plan,
        validation_policy_path=args.validation_policy,
        evidence_path=args.evidence,
    )
    write_exposure_ledger(
        ledger,
        summary,
        ledger_path=args.output_csv,
        summary_path=args.summary_json,
    )
    _LOGGER.info("wrote %d exposure-ledger rows", len(ledger))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

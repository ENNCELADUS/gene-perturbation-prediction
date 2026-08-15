#!/usr/bin/env python3
"""Build the Tx1 GeneEffect P0 nested validation contract."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from aivc_model.tx1_p0_validation import (
    ValidationPolicy,
    generate_nested_validation,
)

_LOGGER = logging.getLogger(__name__)
_DEFAULT_MANIFEST = Path("results/phase_a_tx1_20260724/cell_line_manifest.csv")
_DEFAULT_POLICY = Path(
    "configs/experiments/12_tx1_st_geneeffect/p0/validation_policy.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a stable JSON artifact after all fail-closed checks pass."""
    args = _parse_args(argv)
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    if not isinstance(policy_payload, dict):
        raise ValueError("validation policy root must be a JSON object")
    policy = ValidationPolicy.from_mapping(policy_payload)
    payload = generate_nested_validation(args.manifest, policy=policy)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _LOGGER.info("Wrote %d outer folds to %s", len(payload["outer_folds"]), args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

#!/usr/bin/env python3
"""CLI: train one Phase D GeneEffect head arm (Task 5).

Wires Phase D Tasks 1-4 into one runnable chain for a single arm
(``tx1_arm`` or ``hvg_arm``, D7 -- both run the identical head and data,
differing only in which basal/ST-input view a config selects): line
selection -> forward-only ST + predicted-response generation/caching -> head
training -> (optionally) the Phase F ``tx1_3b_st`` predictions table for the
9 frozen held-out lines.

All substantive logic lives in the library, not here (this file stays a thin
argparse shell, mirroring ``scripts/evaluate_tx1_backbone.py``'s split from
``aivc_model.tx1_geneeffect_eval``):
``aivc_model.tx1_geneeffect_pipeline`` owns the config schema and every
``--dry-run`` check; ``aivc_model.tx1_geneeffect_pipeline_run`` owns the real
orchestration.

Runnable directly -- no ``PYTHONPATH`` needed. This file prepends the repo
root and ``src/`` to ``sys.path`` itself (Codex P1-a from Phase B: a plain
``python scripts/train_tx1_geneeffect_head.py`` otherwise fails on
``from aivc_model...`` because the repo is not pip-installed into the HPC's
dedicated venvs), and never imports from another ``scripts/*`` module (that
is what forced ``PYTHONPATH=src:.`` in the first place).

``--dry-run`` validates config keys, path existence, widths, roles, and gene
coverage WITHOUT materializing a single cell matrix -- Phase C's Task 7
dry-run once hung for 58 minutes doing exactly that. Success (dry-run or
real run) exits 0; any validation failure raises and the process exits
non-zero via Python's own uncaught-exception handling, naming the problem in
the exception message (Codex P1-c from Phase B: a shard that actually
succeeded must never exit non-zero, so nothing here maps a success outcome
onto a non-zero code).

Invocation (dry run -- validates paths/widths/roles/gene-coverage, trains
nothing)::

    .venv-tx1/bin/python scripts/train_tx1_geneeffect_head.py \\
        --config configs/experiments/12_tx1_st_geneeffect/phase_d/tx1_arm.yaml \\
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv \\
        --phase-a-dir results/phase_a_tx1_20260724 \\
        --tx1-cache-dir data/tx1_basal_embeddings/v1 \\
        --depmap-gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \\
        --dry-run

Invocation (real training; single-process -- head training itself is small
enough for one GPU, unlike Phase C's ST training)::

    .venv-tx1/bin/python scripts/train_tx1_geneeffect_head.py \\
        --config configs/experiments/12_tx1_st_geneeffect/phase_d/tx1_arm.yaml \\
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv \\
        --phase-a-dir results/phase_a_tx1_20260724 \\
        --tx1-cache-dir data/tx1_basal_embeddings/v1 \\
        --predicted-response-cache-dir data/tx1_predicted_response_cache \\
        --depmap-gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Codex P1-a (Phase B): executed directly, this file's own directory lands on
# sys.path, not the repo root, so ``from aivc_model...`` fails unless the
# packages are pip-installed (they are not, on the HPC's dedicated venvs).
# Prepend the repo root and ``src/`` here, before any local import, exactly
# as scripts/build_tx1_basal_embeddings.py and
# scripts/train_tx1_st_response.py already do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from aivc_model.tx1_geneeffect_pipeline import (  # noqa: E402
    dry_run_summary,
    load_pipeline_config,
)
from aivc_model.tx1_geneeffect_pipeline_run import run_training_pipeline  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PHASE_A_DIR = Path("results/phase_a_tx1_20260724")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (mirrors train_tx1_st_response.py's conventions).

    Args:
        argv: Argument list, or ``None`` to use ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--phase-a-dir", type=Path, default=_DEFAULT_PHASE_A_DIR)
    parser.add_argument("--tx1-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--predicted-response-cache-dir",
        type=Path,
        default=None,
        help="Required for a real run; unused by --dry-run.",
    )
    parser.add_argument("--depmap-gene-effect", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output dir; defaults to <run.output_dir>/runs/<run.run_id>.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-test-predictions",
        action="store_true",
        help=(
            "Train the head but skip emitting the Phase F tx1_3b_st "
            "predictions table for the 9 held-out lines."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate (``--dry-run``) or run the Phase D Task 5 chain for one arm.

    Returns:
        0 on success. Any failure raises (see module docstring); this
        function never catches a validation error and maps it onto a
        misleadingly-successful 0, nor onto a non-zero code for a path that
        actually succeeded.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_pipeline_config(args.config)

    if args.dry_run:
        summary = dry_run_summary(
            config,
            line_manifest_path=args.line_manifest,
            phase_a_dir=args.phase_a_dir,
            tx1_cache_dir=args.tx1_cache_dir,
            depmap_gene_effect_path=args.depmap_gene_effect,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.predicted_response_cache_dir is None:
        raise ValueError("--predicted-response-cache-dir is required for a real run")

    outputs = run_training_pipeline(
        config,
        line_manifest_path=args.line_manifest,
        phase_a_dir=args.phase_a_dir,
        tx1_cache_dir=args.tx1_cache_dir,
        predicted_response_cache_dir=args.predicted_response_cache_dir,
        depmap_gene_effect_path=args.depmap_gene_effect,
        run_dir=args.run_dir,
        emit_test_predictions_flag=not args.skip_test_predictions,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

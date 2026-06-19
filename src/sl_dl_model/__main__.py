# src/sl_dl_model/__main__.py
"""CLI for the exp08 STATE-adapter DL SL-pair model.

Usage::

    uv run python -m sl_dl_model run-cv \\
        --config configs/experiments/08_k562_sl_pair_state_dl/phase0_parity.yaml \\
        --producer zero

The ``state_dl`` producer builds a per-fold ``StateDlProducer`` inside
:func:`~sl_dl_model.evaluate.run_cv`; the constructor (which needs ESM2 +
gwps bags + train pairs) is called there, not here, so this module stays
importable without torch/accelerate loaded.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sl_dl_model",
        description="exp08 STATE-adapter DL model for K562 SL-pair ranking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run-cv",
        help="Run CV and write official metrics to the configured output dir.",
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a SLDLConfig YAML file.",
    )
    run.add_argument(
        "--producer",
        choices=["zero", "state_dl"],
        default="state_dl",
        help=(
            "Embedding producer to use. "
            "'zero' = GeneEffect-only exp06-parity baseline; "
            "'state_dl' = frozen STATE + trainable ESM2 adapter (per-fold)."
        ),
    )
    run.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write log output (appended).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m sl_dl_model``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    args = _build_parser().parse_args(argv)

    # Lazy imports so that --help works without loading torch/accelerate.
    from sl_dl_model.config import load_config

    config = load_config(args.config)

    from accelerate import PartialState

    is_main = PartialState().is_main_process
    log_file = args.log_file
    if log_file is None:
        log_file = Path(config.output_dir) / "train.log"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    # Per-rank metric log captures this rank's folds' curves.
    rank = PartialState().process_index
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    handlers.append(
        logging.FileHandler(Path(config.output_dir) / f"train_rank{rank}.log", mode="a")
    )
    # The shared train.log is written by the main process only.
    if is_main:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    from sl_dl_model.evaluate import run_cv

    if args.producer == "zero":
        from sl_dl_model.evaluate import ZeroEmbeddingProducer

        run_cv(config, ZeroEmbeddingProducer())
    else:
        # state_dl: run_cv handles per-fold producer construction internally.
        run_cv(config, producer="state_dl")


if __name__ == "__main__":
    main()

"""Evaluate a saved checkpoint independently of training."""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from src.experiments.geneeffect import evaluate_checkpoint

    evaluate_checkpoint(args.checkpoint, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

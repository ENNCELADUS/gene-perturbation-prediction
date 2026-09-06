"""Train the joint GeneEffect model from prepared inputs."""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id")
    mode.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from src.experiments.geneeffect import run_training

    run_training(args.config, run_id=args.run_id, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the joint GeneEffect model into a run directory (resumes from last.pt).

Runs as one process (CPU or one GPU) or under ``accelerate launch``.
"""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from src.experiments.config import load_config
    from src.experiments.geneeffect import run_training

    print(run_training(load_config(args.config), args.run_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

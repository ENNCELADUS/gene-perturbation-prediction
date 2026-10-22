"""What limits the default prior: per-cell sampling, the number of training lines,
the deposit a line's cells come from, cell-state heterogeneity and the noise of the
label (protocol §14)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from src.data.io import write_json, write_text
from src.experiments.prior.config import load_noise_config
from src.experiments.prior.diagnosis import OUT_DIR, SECTIONS, Diagnosis, load_diagnosis
from src.experiments.prior.noise_report import results_markdown
from src.experiments.prior.noise_sections import (
    cell_sampling,
    cell_states,
    deposit_confound,
    deposit_swap,
    learning_curve,
)
from src.experiments.run_dir import bind

RUNNERS: dict[str, Callable[[Diagnosis, Path], dict]] = {
    "cell_sampling": cell_sampling,
    "learning_curve": learning_curve,
    "deposit_confound": deposit_confound,
    "deposit_swap": deposit_swap,
    "cell_states": cell_states,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--sections", help=f"comma-separated, of {', '.join(SECTIONS)} (default: all)"
    )
    args = parser.parse_args(argv)
    config = load_noise_config(args.config)
    names = (
        SECTIONS
        if args.sections is None
        else tuple(dict.fromkeys(args.sections.split(",")))
    )
    unknown = sorted(set(names) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown sections {unknown}; sections are {list(SECTIONS)}")
    run_dir = Path(config["output_root"]) / args.run_id
    out = run_dir / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    pending = [name for name in names if not (out / f"{name}.json").is_file()]
    if pending:
        diagnosis = load_diagnosis(config)
        for name in pending:
            print(f"section {name}", flush=True)
            write_json(out / f"{name}.json", RUNNERS[name](diagnosis, out))
    results = {
        name: json.loads((out / f"{name}.json").read_text())
        for name in SECTIONS
        if (out / f"{name}.json").is_file()
    }
    write_text(out / "results.md", results_markdown(args.run_id, config, results))
    print(f"noise diagnosis: {out / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What limits the default prior: per-cell sampling, the number of training lines,
the deposit a line's cells come from, cell-state heterogeneity and the noise of
the label (protocol §14; design ``docs/specs/2026-10-12-noise-diagnosis-design.md``).

``python -m src.experiments.prior.noise_diagnosis CONFIG --run-id ID [--sections A,B]``
reads the prior run named by ``prior_run`` (its bound config and ``export/``) and
writes ``<output_root>/<run id>/noise/<section>.json`` for each section, skipping
a section whose file exists, then ``results.md`` from every section present:

- ``cell_sampling``: split-half reliability of pseudo-bulk per gene against the
  bridge's agreement with bulk (``cell_sampling_genes.csv``), and the reference
  prior queried with half of each validation and test line's cells and with
  10 to 400 of them;
- ``learning_curve``: the reference row refitted on nested patient-grouped
  fractions of the training side, each beside its own gene mean, scored on the
  line-specific part (selective Spearman, within-gene R², AUPR lift, residual
  Pearson) and the whole prediction (per-line Pearson, RMSE); at the full side,
  Cilantro-SL's stage one on the pairs it covers;
- ``deposit_confound``: the share of the out-of-fold error a training line's
  deposit explains, against permutations within lineage;
- ``deposit_swap``: out-of-fold predictions of training lines read from a second
  deposit, against two halves of their own deposit;
- ``cell_states``: cell-state features stacked on the out-of-fold export.

Every score is on the metric of the prior run; gains carry 95% paired
line-bootstrap intervals. It reports and chooses nothing; nothing here is
synthetic-lethality evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path


from src.data.io import write_json, write_text
from src.experiments.run_dir import bind
from src.experiments.prior.config import load_noise_config
from src.experiments.prior.diagnosis import OUT_DIR, SECTIONS, Diagnosis, load_diagnosis
from src.experiments.prior.noise_sections import (
    cell_sampling,
    cell_states,
    deposit_confound,
    deposit_swap,
    learning_curve,
)
from src.experiments.prior.noise_report import results_markdown


# ----------------------------------------------------------------------------
# The prior run
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------


RUNNERS: dict[str, Callable[[Diagnosis, Path], dict]] = {
    "cell_sampling": cell_sampling,
    "learning_curve": learning_curve,
    "deposit_confound": deposit_confound,
    "deposit_swap": deposit_swap,
    "cell_states": cell_states,
}


# ----------------------------------------------------------------------------
# results.md
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


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

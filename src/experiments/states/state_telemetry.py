"""State telemetry (PIE stage-one design §3.5, Task 8; protocol §17), project env.

``python -m src.experiments.states.state_telemetry CONFIG --steps fit,project,inputs``
then ``--steps report`` (``hpc/run.sh cell-states``). The ladder config the states
config names gives the ladder root, whose ``states/`` this writes. Steps, each
skipping what exists, so rerunning resumes:

- ``fit``: per fitting seed, the cells each single-cell training line
  contributes (``fit/sample_seed<s>.parquet``: at most ``fit.cells_per_line``
  distinct cells per line, so smaller lines weigh less; each cell's chemistry
  from the reviewed table by its deposit, line and arm); ``fit/plan.json`` and, in
  ``.venv-states`` (``STATES_PYTHON``), ``src.experiments.states.programs``:
  consensus NMF programs and their starCAT-style reference, scVI, and program
  usages and scVI latents for every cell of the projected lines and their second
  deposits (``programs/``, ``scvi/``); then the mixtures (``fit/seed<s>/
  <vocabulary>.npz``): the cycle on S and G2/M, and K in ``fit.mixture_counts``
  on scVI's latent (line-centred and absolute), the line-centred HVG cells' and
  Tx1-3B embeddings' principal components, and the line-centred, standardised QC
  columns.
- ``project``: every vocabulary at every fitting seed on every cell of the lines
  in ``project.roles`` (``memberships/<ModelID>.npz``) and of their second
  deposits (``second/``; no Tx1 there).
- ``inputs``: one raw-h5ad pass per projected line: each set's state masses and
  inputs in PIE's input space at the first fitting seed, merged below
  ``inputs.merge_effective_count``, its permuted memberships at every
  permutation seed, the cycling-weighted input and the whole-line input
  (``lines/<ModelID>.npz``); per-cell reference-program scores and QC values
  weighted by each state (identity and confound telemetry); then per set
  ``<set>/inputs.parquet`` and ``<set>/masses.parquet``, the frozen PIE's
  contexts (``hpc/run.sh pie-precompute CONFIG --steps states``).
- ``report``: ``<output_root>/<config stem>/results.md`` and ``report.json``:
  states' identity against Kinker's, Gavish's, hallmark and Tirosh's programs;
  the mass explained by deposit and by QC; recurrence across lines; the
  centred-log-ratio SVD of the line x state masses; agreement between a line's
  two deposits; stability across fitting seeds (states matched by the
  Hungarian method); the frozen PIE's gap ``sum_k pi_k D(x_k) - D(x)`` for dH,
  dE and the own slots, against its ``|whole`` context and the main pass.

The frozen PIE's state outputs live under ``states/pie/``, so moving ``states/``
aside moves them too; each line's carries the fingerprint of the state inputs
it was computed from, and the report refuses outputs of other inputs.

It reports and decides nothing.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from src.experiments.run_dir import absolute, open_manifest, record_step
from src.experiments.states.inputs import step_inputs
from src.experiments.states.report import step_report
from src.experiments.states.telemetry import MANIFEST, Telemetry, settings
from src.experiments.states.vocabularies import step_fit, step_project

_LOGGER = logging.getLogger(__name__)

STEPS: Final[tuple[str, ...]] = ("fit", "project", "inputs", "report")


# --- the run --------------------------------------------------------------------


# --- per-cell representations -------------------------------------------------------


# --- fit ---------------------------------------------------------------------------


# --- project -------------------------------------------------------------------------


# --- inputs -------------------------------------------------------------------------


# --- report --------------------------------------------------------------------------


# --- results.md ---------------------------------------------------------------------


# --- entry point ------------------------------------------------------------------


RUNNERS = {
    "fit": step_fit,
    "project": step_project,
    "inputs": step_inputs,
    "report": step_report,
}


def run(config: Mapping[str, Any], name: str, steps: Sequence[str]) -> None:
    t = Telemetry(config, name)
    t.layout.root.mkdir(parents=True, exist_ok=True)
    open_manifest(t.layout.root / MANIFEST, settings(t), keep_nan=True)
    for step in steps:
        _LOGGER.info("%s: start", step)
        record_step(t.layout.root / MANIFEST, step, RUNNERS[step](t), keep_nan=True)
        _LOGGER.info("%s: done", step)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    from src.experiments.states.config import load_states_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(load_states_config(absolute(args.config)), args.config.stem, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

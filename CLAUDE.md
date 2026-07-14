# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and Codex when working with code in this repository.

## Quick Context

- **Active project**: cell-fate outcome dynamics. The same net fitness loss can
  arise from completely different cellular dynamics — division suppression, cell
  loss, or loss followed by survivor regrowth — and it is unknown whether early
  molecular state prospectively distinguishes those trajectories in genetic
  loss-of-function.
- **Status**: literature funnel (L0 -> Gate 4) complete; decision is
  `narrow-or-pivot` for both candidates. **No production modeling is
  authorized.** Next work is three public-data reanalyses.
- **Authority**: the vault under `docs/`, not this file. Start at
  [`docs/README.md`](docs/README.md).
- **The code predates the current direction.** `src/` implements a retired
  dependency-prediction / SL-ranking program. It still runs and the reanalyses
  reuse parts of it — but do not assume a task is about dependency prediction
  just because the code is.
- **Role**: careful junior engineer. Follow **Plan -> Confirm -> Code** for
  non-trivial changes.

## Research Direction

Two candidate research questions are carried **in parallel**; no unit has been
selected:

- **Candidate A (lineage/clone)** — does an early molecular state predict a
  linked lineage's division/persistence/extinction trajectory, beyond an
  independently measured net fitness? Evidence ceiling is **A2** (sibling/clone
  proxy) for anything pooled.
- **Candidate B (population)** — under comparable, independently measured net
  fitness, does the early single-cell state *distribution* carry incremental
  information about independently measured future population dynamics?

The contract, acceptance criteria, gate verdicts, and decision record live in the
vault. Do not restate them here — link to them.

## Research Vault (`docs/`)

1. **Authority ordering.** [`01-blueprint.md`](docs/01-blueprint.md)
   (contract) > [`02-acceptance-criteria.md`](docs/02-acceptance-criteria.md)
   (acceptance criteria) > [`03-literature-review.md`](docs/03-literature-review.md)
   (gate verdicts) > [`04-roadmap.md`](docs/04-roadmap.md)
   (decision) > `docs/results/`. **When two documents conflict, flag it — do not
   resolve it unilaterally.**
2. **Freeze rule.** `01` and `02` are frozen. Change them by editing **in place** —
   **never** by writing a new file. `01` §13 Locked Decisions are settled; changing
   one is a change of research program, not a refinement.
3. **The vault is a snapshot, not a changelog.** It states what is true now. Do not
   add revision histories, "what we got wrong" sections, or superseded-claim logs —
   git already holds that. Correct a wrong statement by replacing it.
4. **Results enter the docs only after the analysis actually runs.** A planned
   number is not a number.
5. **Status must agree** across `docs/README.md`, `docs/04-roadmap.md`, and root
   `README.md`.
6. **New analysis -> a new `docs/results/<slug>.md`.** New gate or re-run -> a
   new section in `03`.
7. **Style**: plain GitHub markdown, relative links, no YAML frontmatter, no
   wikilinks, status as `**Status:**` bold-key lines.
8. **The eleven review memos under `ideaspark_run/cell-fate-outcome-dynamics/`
   are the evidence record and are not edited.** They hold the full evidence
   tables and the `UNVERIFIED` registers.

## Commands

```bash
uv sync                                                    # Install/sync deps
uv run ruff check . && uv run ruff format .                # Lint + format
uv run python -m pytest                                    # Full suite (31 files)
uv run python -m pytest tests/test_dependency_baseline.py  # One file
uv run python -m pytest -k "test_build_features"           # One test by name
```

Prefix every Python/pytest/ruff invocation with `uv run` (project-local `.venv`).
Ruff is configured line-length 88, target py311.

## Codebase Map

Retired-program code, retained and runnable. Data files are gitignored, so most
entrypoints need assets that are not in the repo.

| Package | Purpose | Entrypoint |
|---|---|---|
| `src/dependency_baseline/` | Predict DepMap GeneEffect (`C`) from perturbation transcriptomes (`B`). Pseudobulk delta, single-cell Deep Sets, and distribution/GMM tracks. | `uv run vcc-dep-baseline <subcmd>` (14 subcommands; `--help`) |
| `src/aivc_model/` | A->B->C forward model wrapping Arc's STATE. Not on the CLI. | `uv run python src/aivc_model/train.py --config ...`; `scripts/state.sh` on Slurm |
| `src/sl_benchmark_baseline/` | Dependency-only SL-pair (`D`) baseline. | `uv run python -m sl_benchmark_baseline` |
| `src/sl_dl_model/` | STATE-adapter deep model for SL-pair ranking. | `uv run python -m sl_dl_model` |
| `src/ddgcn/` | DDGCN reproduction on the SL benchmark. | `uv run python -m ddgcn` |

**What the planned reanalyses reuse:** the residualization machinery in
`src/dependency_baseline/` — `NuisanceResidualizer` in `models.py` and the
burden / program-score / NAR feature sets in `features.py` + `datasets.py`. The
three reanalyses (Jost 2020 titration, Dixit 2016 panel, Nadal-Ribelles
mean-vs-variance on Replogle) are residualization audits in that same shape.
The SL packages are prior evidence, not part of the current direction.

Configs live in `configs/experiments/<NN>_<name>/`; `models:` defines the ladder,
`selection:` filters what actually runs. Experiment write-ups from the retired
program are under `docs/archive/` (untracked, gitignored).

## Data Rules

- K562 is the proof-of-concept line. Prioritize CRISPRi / knockout Perturb-seq.
- Norman is CRISPRa — auxiliary only, never aligned to knockout labels without a
  modality caveat.
- Replogle Perturb-seq is presumptively **late-state** and cannot support a
  prospective (T2) claim.
- Raw `*.h5ad`, `*.csv`, checkpoints, and large artifacts are gitignored.

## Code Style

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings.
- Small functions (<50 lines), small files (<600 lines). Composition over
  inheritance.
- No `print` in library code — use `logging`. No hardcoded paths or thresholds —
  use config. No bare `except`.
- Conventional Commits: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`,
  `chore`, `ci`.

## Terminology Guardrails

Binding on all writing, per the contract's claim boundaries
([`docs/01-blueprint.md`](docs/01-blueprint.md) §12).

- **DepMap GeneEffect is a relative growth-rate effect** under an explicit
  population-dynamics model. It is not a cell-death label and not a single-cell
  readout.
- Do not say population screens cannot separate death from arrest. Say: a single
  endpoint net-fitness readout does not uniquely determine the underlying
  dynamics.
- Do not write "loss" without disambiguating **biological extinction** from
  **assay attrition**.
- Do not equate high-mito / low-UMI cells with dying cells, and do not describe a
  QC-relaxation-induced cluster as a recovered dying population.
- **T2 (prospective) is not T3 (counterfactual).** Do not report a same-window
  `F_net` result (Analysis R) as prospective prediction.
- Do not upgrade a sibling/clone-proxy (A2), clone-average (A3), or
  population-level (Candidate B) result into a per-cell fate claim.
- Do not infer causation, fate commitment, mechanism, or manipulability from
  incremental predictive information.
- Do not treat absence of data as falsification.
- Do not cite Live-seq as "non-destructive" without the 85-89% post-biopsy
  viability caveat.
- Do not present a drug-derived wedge result as evidence for genetic
  perturbation.
- Do not treat `|ΔF_net| ≤ 1 SD` as fitness equivalence.
- Synthetic lethality is **out of scope** (contract §11) and requires an explicit
  combination null; never claim it from essentiality alone.

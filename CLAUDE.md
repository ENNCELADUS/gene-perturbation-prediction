# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and Codex when working with code in this repository.

## Quick Context

- **Active project**: synthetic-lethality discovery by virtual-cell composition.
  Compose the exp05 forward model (perturbation transcriptome -> DepMap
  GeneEffect) into a pairwise interaction score, and beat the strong graph/KG SOTA
  (**SLMGAE, KR4SL**; KG4SL is weak at cold-start) on the inductive cold-start
  splits where they break. The graph-free/inductive framing is shared with prior
  art (CILANTRO-SL); the contribution is the mechanism + measured-epistasis
  validation (see [`docs/03`](docs/03-literature-review.md)).
- **Status**: contract (`01`) and acceptance criteria (`02`) established;
  related-work review (`03`) and experiment roadmap (`04`) pending; forward model
  (exp05) in progress on branch `codex/exp05-k562-fixed-pool`. **No SL graph in
  the feature path.**
- **Authority**: the vault under `docs/`, not this file. Start at
  [`docs/README.md`](docs/README.md).
- **The code is now central, not retired.** `src/aivc_model/` (exp05) is the
  composition backbone and `src/sl_benchmark_baseline/` + `data/SL_benchmark/`
  are the benchmark, floor, and SOTA baselines. The dependency-prediction tracks
  in `src/dependency_baseline/` remain as baselines and feature machinery.
- **Role**: careful junior engineer. Follow **Plan -> Confirm -> Code** for
  non-trivial changes.

## Research Direction

The task is graph-free, inductive SL partner ranking in K562: learn `s(a,b)` with
**no SL graph in the feature path**. The mechanism composes the exp05 virtual cell
into a pairwise interaction, two ways, compared head-to-head:

- **Bridge A (counterfactual co-dependency)** — simulate loss of `a`, predict
  `b`'s GeneEffect in the `a`-lost state; SL = the dependency spike. Single-gene
  labels only.
- **Bridge B (virtual double-knockout)** — forward the joint `a+b` perturbation ->
  joint fitness; SL = interaction residual vs. an explicit additive/min null.
  Validated against measured epistasis (Horlbeck 2018 K562 GI, to acquire; Adamson
  UPR as a small local check). Caveat: foundation models underestimate synergy on
  double perturbations, so the explicit null + a GenePert-style linear ablation are
  guards (see [`docs/03`](docs/03-literature-review.md) §2).

Single-gene GeneEffect alone is the ~0.70-AUROC floor, not the method. The
contract, acceptance criteria, and roadmap live in the vault. Do not restate them
here — link to them.

## Research Vault (`docs/`)

1. **Authority ordering.** [`01-blueprint.md`](docs/01-blueprint.md)
   (contract) > [`02-acceptance-criteria.md`](docs/02-acceptance-criteria.md)
   (acceptance criteria) > [`03-literature-review.md`](docs/03-literature-review.md)
   (related work) > [`04-roadmap.md`](docs/04-roadmap.md)
   (roadmap) > `docs/results/`. **When two documents conflict, flag it — do not
   resolve it unilaterally.**
2. **Freeze rule.** `01` and `02` are frozen. Change them by editing **in place** —
   **never** by writing a new file. `01` §10 Locked Decisions are settled; changing
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
8. **The retired program's evidence memos (`ideaspark_run/`, `docs/archive/`) are
   not edited.** They are prior evidence for the current direction, not a roadmap
   to execute; they hold the full evidence tables and `UNVERIFIED` registers.

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

**What the current direction uses:** `src/aivc_model/` (exp05) is the composition
backbone; `src/sl_benchmark_baseline/` + `data/SL_benchmark/` (Feng2024 zoo incl.
`kg4sl`, `slmgae`) are the benchmark, dependency-only floor, and SOTA baselines;
`src/dependency_baseline/` supplies the swap-invariant GeneEffect features and
residualization machinery. The only local combinatorial-CRISPRi set is
`data/sl_dependency_v0/raw/adamson/adamson_2016_upr_epistasis.h5ad` (small,
qualitative); the fitness-GI anchor **Horlbeck 2018** is not yet local. (The
`jost_replogle_dual_sgrna` file is single-gene knockdown efficacy, **not** epistasis.)

Configs live in `configs/experiments/<NN>_<name>/`; `models:` defines the ladder,
`selection:` filters what actually runs. Experiment write-ups from the retired
program are under `docs/archive/` (untracked, gitignored).

## Data Rules

- K562 is the proof-of-concept line. Prioritize CRISPRi / knockout Perturb-seq.
- Norman is CRISPRa — auxiliary only, never aligned to knockout labels without a
  modality caveat.
- Measured epistasis for the virtual double-KO: **Horlbeck 2018** K562 GI (to
  acquire) + **Adamson UPR** (local, qualitative). Jost dual-sgRNA is single-gene
  efficacy, **not** epistasis.
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
([`docs/01-blueprint.md`](docs/01-blueprint.md) §9).

- **DepMap GeneEffect is a relative growth-rate effect** under an explicit
  population-dynamics model — single-gene, not a cell-death label, not a
  single-cell readout, and never a double-knockout quantity.
- **SL benchmark outputs are candidate prioritization, not validated targets.**
  Rand negatives are unconfirmed non-SL.
- **Never claim SL from single-gene essentiality.** An explicit interaction null
  is required (`interaction = joint - psi(singles)`; declare `psi`).
- **Generalization claims come only from CV2/CV3.** CV1 is degree-gameable and is
  a diagnostic, never evidence of cold-start generalization.
- **A pan-essentiality lift is not an SL result.** Report the non-pan-essential
  slice; a win that vanishes there is downgraded.
- **The virtual double-knockout is an extrapolation** of a single-perturbation
  backbone; its benchmark rank is not mechanistic proof until it clears the
  measured-epistasis bar.
- **A benchmark rank is not a mechanism.** Do not infer causation, fate
  commitment, mechanism, or manipulability from predictive ranking.
- **Norman CRISPRa is auxiliary only**, never aligned to knockout labels without
  the modality caveat.
- **A single-fold or test-fold-selected result is not a result.** Report 5-fold
  mean +/- spread; never select on the test fold.

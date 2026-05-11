# Repository Guidelines

## Quick Context

- **Project**: Cancer dependency prediction from perturbation-induced
  transcriptomes.
- **Goal**: Connect AIVC / virtual-cell perturbation modeling to cancer
  dependency and synthetic-lethality target prioritization.
- **Core task**: Predict DepMap-style CRISPR gene-effect / dependency scores from
  observed or predicted post-perturbation transcriptomic responses.
- **Current branch state**: `main` is a cleaned rebuild base. Legacy `src/`,
  `src_tahoe/`, `scripts/`, reports, and historical result folders were removed.
  Do not assume old pipeline entry points are runnable.
- **Role**: Act as a careful junior engineer. Follow **Plan -> Confirm -> Code**
  for non-trivial research or implementation changes.

## Research Framing

Use this framing when updating docs, designing data schemas, or rebuilding code:

```text
cell line + perturbation gene
    -> observed or predicted post-perturbation transcriptome
    -> dependency / essentiality score
    -> context-specific target ranking
```

The matched training key is **(cell line, perturbation gene)**. Preserve it in
all intermediate tables, filenames, and model inputs.

### Stage 1: Observed Transcriptome to Dependency

First build the supervised downstream task with real post-perturbation
transcriptomes. Inputs may be pseudobulk delta expression, top-DE signatures,
pathway scores, or foundation-model embeddings. Labels should default to
continuous DepMap/Achilles CRISPR gene-effect scores.

### Stage 2: Virtual-Cell Extension

After Stage 1 is validated, connect a forward perturbation model such as scGPT,
GEARS, STATE, or a simple additive / linear baseline:

```text
basal cell state + candidate perturbation
    -> predicted post-perturbation transcriptome
    -> downstream dependency predictor
```

Do not assume forward perturbation prediction is solved. Quantify how forward
model error affects downstream ranking.

### Stage 3: SL Candidate Prioritization

Synthetic lethality requires context specificity. A DepMap essential gene is not
automatically an SL target. Use mutation, copy-number, lineage, pathway, normal
cell, TCGA, CCLE, or DepMap context evidence before using SL language.

## Data Rules

- Prioritize CRISPRi or knockout Perturb-seq / CROP-seq data for alignment with
  DepMap CRISPR gene-effect labels.
- K562 is the natural proof-of-concept start because several Perturb-seq
  resources exist there. HCT116 and A549 are extension candidates only after
  identifier alignment is clear.
- Norman is CRISPRa. Treat it as a perturbation-response reference or auxiliary
  benchmark, not as a direct knockout-dependency label alignment source.
- DepMap labels are population-level fitness / proliferation readouts. They are
  not single-cell death labels.
- Required alignment fields should include cell-line ID, perturbation gene
  symbol/ID, perturbation modality, expression matrix or signature, and DepMap
  model/gene identifiers.
- Raw `*.h5ad`, `*.csv`, checkpoints, and other large data artifacts are
  gitignored. Keep only lightweight metadata and reproducible processing code in
  git.

## Current Project Structure

- `README.md`: human-facing project description, roadmap, and setup notes.
- `AGENTS.md`: Codex/OpenAI-agent instructions.
- `CLAUDE.md`: Claude-agent instructions.
- `docs/discussion/0408.md`: 2026-04-08 discussion notes.
- `docs/discussion/0429.md`: 2026-04-29 discussion notes.
- `docs/images/core.png`: triangular technical route diagram.
- `docs/images/roadmap.png`: staged roadmap diagram.
- `data/norman/splits/`: retained Norman split metadata.
- `scGPT/`: local scGPT reference code.
- `pyproject.toml` and `uv.lock`: Python dependency metadata.

If you reintroduce implementation code, create a clear package layout and update
`README.md`, `AGENTS.md`, `CLAUDE.md`, and any relevant docs in the same change.

## Environment Requirement

- This repository uses `uv` with the project-local `.venv` as the supported
  Python environment workflow.
- Sync the environment with `uv sync`.
- Prefix every Bash tool call that runs Python, pytest, ruff, mypy, or a Python
  script with `uv run`.
- If dependencies are missing or the lockfile changed, run `uv sync` before
  continuing.
- Do not document old `uv run vcc` or `uv run vcc-tahoe` commands as valid until
  the corresponding packages exist again.

## Code Style

When implementation code is present:

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings.
- Prefer composition and small functions. Target functions under 50 lines and
  files under 600 lines.
- No `print` statements in library or pipeline code; use logging.
- No hardcoded dataset paths, thresholds, or model settings; use config.
- Handle specific exceptions. Do not use bare `except`.

## Testing and Verification

- Use `pytest` for tests.
- Use `uv run python -m pytest` for the full test suite once tests exist.
- Use `uv run ruff check .` and `uv run ruff format .` for lint/format.
- If the repo is in a docs-only or rebuild-base state with no tests, say that
  explicitly in the final response.
- For data or modeling changes, include a focused verification plan even if the
  full experiment cannot run locally.

## Documentation Hygiene

- Keep project framing synchronized across `README.md`, `AGENTS.md`, and
  `CLAUDE.md`.
- Use absolute dates in status notes and discussion summaries.
- Do not leave stale references to removed paths such as `src/`, `src_tahoe/`,
  `scripts/`, or old result folders unless clearly labeled as removed legacy
  material.
- When the data route changes, update the role of each data source and the
  matched key assumptions.

## Commit and Pull Request Guidelines

- Commits use Conventional Commits: `feat`, `fix`, `perf`, `refactor`, `docs`,
  `test`, `chore`, or `ci`.
- PRs should summarize the research or code change, list touched data
  assumptions, and include verification results or an explicit reason tests were
  not run.

## Security and Data Practices

- Never commit API keys, `.env` files, credentials, private patient data, or
  large raw datasets.
- Use environment variables for secrets.
- Sanitize user-provided paths and config values at system boundaries.
- Review `git diff` before pushing.

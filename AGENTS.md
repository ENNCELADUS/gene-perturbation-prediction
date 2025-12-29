# Repository Guidelines

## Project Structure & Module Organization
Core automation sits in `scripts/` (data preview and visualization helpers) and the vendorized `scGPT/` modules/tests. Keep raw AnnData or CSV inputs inside `data/raw/` (gitignored) and write derived features to `data/processed/`. Docs live in `docs/`, and notebooks stay in `notebook/` with outputs cleared before commit.

## Build, Test, and Development Commands
- `source ~/.bashrc && conda activate vcc`: activate conda environment to run python.
- `ruff format . && ruff check . --fix`: applies PEP 8 formatting and autofixes lint issues using the repo `pyproject.toml`.

## Coding Style & Naming Conventions
Target Python 3.11 with 4-space indentation, snake_case functions, and UpperCamelCase classes. Keep modules cohesive, favor f-strings for logging, and document new entry points with concise docstrings that list required AnnData keys or files. Add type hints when touching tensors or sparse matrices. Run `ruff format` (configured for PEP 8 in `pyproject.toml`) before `ruff check --fix`, and reserve inline comments for non-obvious control flow.

## Testing Guidelines
No CI exists yet, so run the command list locally and note the data slice in the PR. Guard preprocessing code with quick assertions (row counts, sparsity bounds) and prefer deterministic sampling so reviewers can replicate outputs. Extend `scGPT/tests/` or add new pytest modules when altering tokenizer or model logic, and run `pytest scGPT/tests -k <feature>` before requesting review.

## Commit & Pull Request Guidelines
Commits stay short and imperative (`Add project proposal`, `Update README.md`), so group logical changes and avoid mixing environment bumps with analytical notebooks. A pull request should explain motivation, highlight key files, list the commands/tests you ran, and summarize any regenerated artifacts. Mention the data location used, link to the tracking issue, and wait for one reviewer plus a passing local check (pytest or data preview) before merging.

## Data Handling & Configuration
Keep raw `.h5ad` and large `.csv` files in `data/raw/` (not in Git history). Document download steps in `docs/`, expose configuration flags or env vars instead of hard-coded paths, scrub secrets from notebooks, and remove temporary configs when experiments finish. Version derived metadata (plots, metrics) for reproducibility.

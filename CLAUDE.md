# Repository Guidelines

## Quick Context
- **What**: A virtual cell framework that shifts from forward prediction (simulating perturbation effects) to "Reverse Perturbation Retrieval" (inferring causal genetic or pharmacological targets from observed transcriptomic phenotypes). Evaluates foundation models (e.g., scGPT, STATE) on datasets like Virtual Cell Challenge (VCC), Norman, and Tahoe-100M.
- **Goal**: Reproducible, config-driven pipelines for Pretraining, Finetuning, and Evaluation.
- **Role**: Act as a careful junior engineer. Follow **Plan → Confirm → Code**.

## Project Structure & Module Organization
Standard Deep Learning project layout:
- `src/model/`: Neural network architectures (`nn.Module` subclasses).
- `src/train/`: Training logic, including `Trainer` classes and strategies.
- `src/evaluate/`: Evaluation scripts and logic.
- `src/utils/`: Shared utilities (logging, config parsing, device management).
- `src/configs/`: YAML configuration files for experiments.
- `scripts/`: Shell scripts for orchestration (e.g., `scgpt.sh`).
- `tests/`: Project tests (Unit, Integration, E2E).

## Environment Requirement (Critical)
- This repository uses `uv` with the project-local `.venv` as the only supported Python environment workflow.
- Sync the environment with: `uv sync`
- Prefix every Bash tool call that runs Python, pytest, ruff, mypy, or any script with `uv run`.
- If dependencies are missing or the lockfile changed, run `uv sync` before continuing.

## Code Style
Act as a careful junior engineer with strong tooling.
- **Core**: Write clean, efficient Python 3.10+. Prefer composition. Be concise.
- **Structure**: Target 200-400 line files (max 600). Keep functions <50 lines. Organize by feature.
- **Naming**: `snake_case` (files/funcs), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- **Quality**:
  - No `print` statements (use logging).
  - No hardcoded values (use config).
  - Max nesting level: 4.
  - Strict type hints (avoid `Any`).
- **Best Practices**:
  - Absolute imports only (`from src.x import y`).
  - Google-style docstrings.
  - Handle specific exceptions (no bare `except`).


## Build, Test, and Development Commands
- **Environment (required before Python commands)**: `uv sync`
- **Linting**: `uv run ruff check --fix .` (Fixes lint errors)
- **Formatting**: `uv run ruff format .` (Formats code)
- **Testing**: `uv run python -m pytest` (Runs all tests)
- **Orchestration**: Use shell scripts in `scripts/` to run pipelines on hpc. Avoid running `python src/run.py` directly.

## Testing Guidelines
- **Framework**: `pytest`.
- **Coverage**: Aim for ≥80% coverage on touched modules.
- **Workflow**: Test-Driven Development (TDD) — Write failing tests -> Implement -> Refactor.
- **Types**:
  - **Unit**: Fast, isolated tests for functions/classes.
  - **Integration**: Database/API interactions.
  - **E2E**: Critical user flows.

## Commit & Pull Request Guidelines
- **Commits**: Conventional Commits format `<type>: <description>`
  - Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`.
- **Branches**: `short-description` (e.g., `new-scheduler`, `fix-nans`). Rebase on `main` before merging.
- **Pull Requests**:
  - Summarize changes and context.
  - Include a test plan and verification results.
  - Address all critical review feedback.

## Security & Best Practices
- **Secrets**: Never commit API keys or `.env` files. Use environment variables.
- **Validation**: Sanitize all user inputs.
- **Dependencies**: Periodically review and update dependencies.

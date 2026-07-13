# GRAND Codex Multiagent Guide

This directory contains project-local Codex subagent configuration for the GRAND
protein-interaction network reconstruction repository. It supplements the root
`AGENTS.md`; it does not replace the root project rules.

## Skills Discovery

Relevant globally installed skills from `~/.codex/skills/` include:

- `grill-with-docs` — Stress-test plans against `CONTEXT.md`, terminology, and
  documented decisions.
- `experiment-plan` — Design experiment plans, comparisons, and acceptance
  criteria.
- `ml-pipeline-architecture-review` — Review ML pipeline structure, data flow,
  and experiment boundaries.
- `python-testing` — Build focused pytest coverage and verification strategy.
- `python-patterns` — Keep Python implementation idiomatic and maintainable.
- `pytorch-patterns` — Apply PyTorch modeling and training patterns when model
  code changes.
- `train` — Run or structure model-training workflows.
- `train-debug` — Diagnose failed or stalled training runs.
- `benchmark` — Design and interpret benchmark comparisons.
- `interpret-curves` — Analyze training curves, metrics, and performance
  trends.
- `run-validation` — Execute validation checks before completion.
- `verification-loop` — Coordinate build, test, lint, and typecheck loops.
- `documentation-lookup` — Verify library and API behavior against docs.
- `paper-analysis` — Extract evidence and methods from papers.
- `deep-research-ingest` — Ingest research artifacts into the literature vault.
- `security-review` — Check security and data-handling risks.
- `git-workflow` — Keep commits, branches, and diffs clean.

Project-local skills, if added later, should live under `.agents/skills/`.
Each skill should provide:

- `SKILL.md` — Detailed instructions and workflow.
- `agents/openai.yaml` — Optional Codex interface metadata when the skill
  provides it.

Prefer skills that support research planning, Python testing, documentation
review, and verification. Do not invent or list unavailable skills.

## Default Workflow

- Use subagents only when the user explicitly asks for multiagent or parallel
  work.
- Keep `max_depth = 1`; child agents should not recursively fan out work.
- Each custom agent must read the root `AGENTS.md` and `CONTEXT.md` before
  making claims or edits.
- The parent agent owns orchestration, final decisions, and integration.
- Remote SSH, tmux, scheduler, and artifact-state operations stay with the
  parent agent.

## Roles

- `domain_mapper`: read-only mapping of real code, config, data, and docs paths.
- `experiment_planner`: read-only experiment and verification planning around
  PRING splits, pairwise scoring, topology fine-tuning, and topology-evaluation
  constraints.
- `implementation_worker`: scoped code or config edits after the parent agent
  assigns a specific ownership boundary.
- `reviewer`: read-only correctness, leakage, data-contract, security, and
  missing-test review.
- `docs_guard`: read-only documentation and terminology drift checks.

## Documentation Flow

`docs_guard` reports stale wording and suggested replacements. The parent agent
performs final documentation edits so `README.md`, root `AGENTS.md`,
`CLAUDE.md`, `CONTEXT.md`, and experiment docs stay synchronized.

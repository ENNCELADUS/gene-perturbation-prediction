# VCC Codex Multiagent Guide

This directory contains project-local Codex subagent configuration for the VCC
synthetic-lethality research repository.

## Default Workflow

- Use subagents only when the user explicitly requests multiagent, delegated, or
  parallel work and delegation reduces coordination cost.
- Give each editing agent an explicit ownership boundary. Agents share the same
  worktree and must preserve unrelated or concurrent changes.
- Keep remote SSH, scheduler, long-running experiment, and artifact state
  operations with the parent agent unless the user explicitly requests otherwise.
- The parent agent owns orchestration, final decisions, integration, and the
  evidence-backed final report.

## Agent Roles

- `domain_mapper`: read-only mapping of the active `aivc_model` backbone and the
  relevant code, config, data, artifact, and documentation paths.
- `experiment_planner`: read-only planning for the Exp13 GeneEffect residual
  benchmark and `context_screen_v2`, leakage controls, and verification.
- `implementation_worker`: narrowly scoped code, config, or test changes after the
  parent assigns ownership and success criteria.
- `reviewer`: read-only adversarial review for correctness, leakage, silent
  defaults, data contracts, claim boundaries, and missing tests.
- `docs_guard`: read-only review of documentation authority, dataset semantics,
  terminology, claim bars, and implementation-to-claim alignment.

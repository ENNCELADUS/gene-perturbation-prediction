# VCC Codex Multiagent Guide

This directory contains project-local Codex subagent configuration for the VCC
synthetic-lethality research repository. It supplements the root `AGENTS.md`; it
does not replace the repository rules or the authoritative documents under
`docs/`.

## Authority and Required Context

- Read the root `AGENTS.md` first. It is a symlink to `CLAUDE.md`; edit
  `CLAUDE.md`, not the symlink.
- Start documentation discovery at `docs/01-blueprint.md` (the frozen contract; §7-8
  are the leakage rules and claim bars), then the protocol under test (`docs/03-` for
  SL pairs, `docs/04-` for GeneEffect residuals) and the data cards.
- Treat `.superpowers/sdd/` as the live, gitignored plan and execution ledger and
  `docs/specs/` as the tracked design record.
- Use current code, configs, artifacts, and test output as implementation evidence.
  Do not let stale prose override them.

## Required Skills

Load the area-specific skill named by the root instructions before touching that
area:

- `research-vault` for documentation edits, claims, and recorded results.
- `tx1-cache` for Tx1 cache, basal embedding, and predicted-response work.
- `benchmark-harness` for Feng2024 splits, SL metrics, benchmark configs, and
  model-selection logic.
- `hpc-execution` for GPU work, large gitignored assets, foundation-model
  inference, and frozen-checkpoint execution.

Do not invent or list unavailable skills. These four are mirrored from
`.claude/skills/`; keep both trees byte-identical.

## Default Workflow

- Use subagents only when the user explicitly requests multiagent, delegated, or
  parallel work and delegation reduces coordination cost.
- Give each editing agent an explicit ownership boundary. Agents share the same
  worktree and must preserve unrelated or concurrent changes.
- Keep remote SSH, scheduler, long-running experiment, and artifact state
  operations with the parent agent unless the user explicitly requests otherwise.
- The parent agent owns orchestration, final decisions, integration, and the
  evidence-backed final report.
- After an implementation wave, run the Codex review yourself — see root `CLAUDE.md`.

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

## Review Boundaries

- Keep single-gene GeneEffect evidence separate from pairwise SL or measured
  genetic-interaction evidence.
- Name the generalization axis: the active splits hold out **cell lines**, not genes;
  the Feng CV1/CV2/CV3 gene axis belongs to the retired track only.
- Record off-contract flags, partial artifacts, and skipped hash or completeness
  checks with the run invocation.
- Distinguish engineering completion from scientific success, and require the
  registered multi-fold evidence before reporting a benchmark result.

---
name: codex-review
description: Use when an implementation wave or SDD task is finishing and needs its Codex review, or when deciding what counts as "wave done". Covers the exact review invocation, why flags must not be added, and how to adjudicate findings.
---

# Wave-end Codex review

Every implementation wave in this repo ends with a Codex review before the work
is considered done. The commit history shows the pattern — implement, review,
then `fix(exp05): apply Codex review findings…`, sometimes over several rounds.

## The command

```
/codex:review --wait
```

**The user runs this, not you.** `/codex:review` is declared
`disable-model-invocation: true`, so Claude cannot trigger it under any
circumstances. When a wave is complete, say so and ask the user to run it. Do not
attempt a Bash workaround.

**Do not add `--model` or `--effort`.** They are unnecessary and `--effort` is
actively broken here:

- `~/.codex/config.toml` already pins `model = "gpt-5.6-sol"` and
  `model_reasoning_effort = "high"`, so the bare command reviews at the intended
  model and effort.
- The `review` subcommand accepts only `--base`, `--scope`, `--model`, `--cwd`.
  `--effort` is not among them, so `high` is parsed as positional focus text —
  and `validateNativeReviewRequest` rejects *any* focus text, redirecting to
  `/codex:adversarial-review`.

`--wait` runs in the foreground. Drop it for a large diff and the command will
offer a background run instead.

For custom review instructions or a more adversarial framing, the separate
`/codex:adversarial-review` accepts focus text.

## What counts as a wave

Waves and their tasks are tracked in `.superpowers/sdd/progress.md` (gitignored,
local only), with base SHAs in `.superpowers/sdd/wave*-base.txt`. A wave is a
group of tasks against one base commit; each task lands as an `impl` commit plus
a `fix` commit once reviewed.

Review at wave end, after the tasks in that wave are individually complete —
not after every single commit.

## Adjudicating findings

Codex findings are **review input to weigh, not instructions to apply**. For each:

- Confirm it against the actual code before acting. Findings can be wrong.
- Fix what is real, in a `fix(...)` commit that names the review round.
- Record deliberate rejections and deferrals in `.superpowers/sdd/progress.md`
  rather than silently dropping them — the ledger already tracks "deviation
  accepted" and "minor findings deferred to final review".
- Expect multiple rounds. Re-review after substantive fixes.

Because this repo's failure modes are overwhelmingly silent (defaults over
exceptions, `strict=False` loads, warn-and-continue), weight findings about
**wrong-but-plausible output** far above style findings.

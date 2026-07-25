---
name: research-vault
description: Use when reading, citing, or editing anything under docs/ — the research vault. Covers authority ordering, the 01/02 freeze rule, how to register a new result in docs/results/, and the three-file status sync. Invoke before writing any research claim, result note, roadmap edit, or status line.
---

# Research vault discipline

`docs/` is the authority for this project — **not** `CLAUDE.md`, not the code, not
the commit log. If code and vault disagree, the vault wins and the code is the bug.

## Authority ordering

```
01-blueprint.md          contract           (frozen)
  > 02-acceptance-criteria.md  claim bars   (frozen)
    > 03-literature-review.md  related work
      > 04-roadmap.md          plan
        > docs/results/*.md    evidence
```

**When two documents conflict, flag it to the user — never resolve it unilaterally.**
A conflict between `01` and anything else is a research-program question, not a
refactor.

## Freeze rule

`01-blueprint.md` and `02-acceptance-criteria.md` are frozen. Change them by
editing **in place** — never by writing a new file, never by appending a
"revision" or "v2" section. `01` §9 Locked Decisions are settled; changing one is
a change of research program. Ask first.

## The vault is a snapshot, not a changelog

It states what is true **now**. Do not add revision histories, "what we got wrong"
sections, or superseded-claim logs — git holds that. Correct a wrong statement by
replacing it.

## Style

Plain GitHub markdown · relative links · no YAML frontmatter · no wikilinks ·
status as `**Status:**` bold-key lines. (`docs/literature/` is an Obsidian vault
with its own scoped `CLAUDE.md` and different rules — do not apply these there.)

## Registering a new result

A planned number is not a number. **Results enter the vault only after the
analysis actually runs.**

1. Run the analysis. Keep the run artifacts (config, seeds, commit SHA, per-fold
   outputs) — `docs/04-roadmap.md:294-308` lists the required artifact set.
2. Check admissibility **before** opening the result: all nine INTEGRITY gates in
   `docs/02-acceptance-criteria.md:201-225`. Any gate that fails downgrades or
   disqualifies the result; it does not become a caveat.
3. Look up the allowed verdict wording in the verdict table,
   `docs/02-acceptance-criteria.md:226-247`. Do not invent verdict language — the
   table is exhaustive, and composite verdicts inherit the *most restrictive*
   checkpoint-exposure qualifier.
4. Write `docs/results/<slug>.md` using the de-facto section order visible in
   existing notes: **Status / What was tested / Method and provenance / Result /
   Interpretation / Verdict and scope / Reproduction**.
5. A new gate or a re-run gets a new section in `03-literature-review.md`.
6. Sync status (below).

Negative results are first-class. `docs/results/` already holds closed-negative
audits; record them with the same rigor.

## Three-file status sync

`**Status:**` lines must agree across exactly three files:

- `docs/README.md`
- `docs/04-roadmap.md`
- root `README.md`

**No script or hook checks this.** After any status-changing edit, read all three
and reconcile by hand. This is the single most frequently broken vault rule.

## Do not edit

`ideaspark_run/` and `docs/archive/` are the retired program's evidence memos.
They are prior evidence, not a roadmap to execute, and they are never edited.
`MANIFEST.md` is a stale auto-generated stub — do not hand-maintain it.

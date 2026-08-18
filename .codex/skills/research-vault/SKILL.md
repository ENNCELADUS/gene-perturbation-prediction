---
name: research-vault
description: Use when reading, citing, or editing anything under docs/ — the research vault. Covers authority ordering, the 01 freeze rule, how to register a new result in docs/results/, and the status sync. Invoke before writing any research claim, result note, protocol edit, or status line.
---

# Research vault discipline

`docs/` is the authority for this project — **not** `CLAUDE.md`, not the code, not
the commit log. If code and vault disagree, the vault wins and the code is the bug.

## Authority ordering

```
01-blueprint.md              contract + claim bars   (frozen)
  > 02-literature-review.md  related work, novelty boundary
    > 03-experiment-protocol.md            SL-pair protocol
      04-exp13-geneeffect-residual-protocol.md   GeneEffect residual protocol
        > docs/results/*.md  evidence
```

`03` and `04` are peers: `03` §7 defers its dependency-residual metric to `04`, which is
**scope-closed** — a GeneEffect number is never SL evidence (`01` §8). A protocol makes
`01` executable; it never relaxes it.

**When two documents conflict, flag it to the user — never resolve it unilaterally.**
A conflict between `01` and anything else is a research-program question, not a
refactor.

## Freeze rule

`01-blueprint.md` is frozen. Change it **in place** — never a new file, never an
appended "revision" or "v2" section. Its §3 amendment is the model: one in-place,
dated paragraph restating the corrected quantity. Changing §7 (leakage) or §8 (claim
boundaries) is a change of research program. Ask first.

## The vault is a snapshot, not a changelog

It states what is true **now**. Do not add revision histories, "what we got wrong"
sections, or superseded-claim logs — git holds that. Correct a wrong statement by
replacing it. A retired design keeps one `**Status:** retired` line and its result link.

## Style

Plain GitHub markdown · relative links · no YAML frontmatter · no wikilinks ·
status as `**Status:**` bold-key lines. (`docs/literature/` is an Obsidian vault
with its own scoped `CLAUDE.md` and different rules — do not apply these there.)

## Registering a new result

A planned number is not a number. **Results enter the vault only after the
analysis actually runs.**

1. Run the analysis. Keep the artifacts the protocol's "Required Outputs" section
   lists (`03` §10, `04` §10): config, seeds, commit SHA, per-fold outputs, hashes.
2. Check admissibility **before** opening the result: the leakage and integrity
   rules in `01` §7, plus the protocol's own leakage section (`03` §8, `04` §9).
   A failed rule downgrades or disqualifies the result; it does not become a caveat.
3. State the verdict in the vocabulary the contract already uses — negative, paused,
   closed, scope-closed, reportable-but-not-claimable — and carry the most restrictive
   qualifier, including Tx1 Tahoe-100M pretraining exposure (`01` §7).
4. Write `docs/results/<slug>.md` in the de-facto section order: **Status / What was
   tested / Method and provenance / Result / Interpretation / Verdict and scope /
   Reproduction**. Name the commit the code ran at — several notes now cite deleted scripts.
5. A new gate or a re-run gets a new section in `02-literature-review.md` §8.
6. Sync status (below).

Negative results are first-class: `docs/results/` holds closed-negative audits already.

## Status sync

`**Status:**` lines must agree across the root `README.md` (status blockquote *and* news
list), `docs/01-blueprint.md` (header line and §9 Current Scientific State), and whichever
protocol changed (`03` or `04`).

**No script or hook checks this.** After any status-changing edit, read all of them
and reconcile by hand. This is the single most frequently broken vault rule.

## Do not edit

`ideaspark_run/` and `docs/archive/` are the retired program's evidence memos — prior
evidence, not a roadmap, never edited. `docs/superpowers/` is gitignored local scratch,
and `MANIFEST.md` is a stale auto-generated stub — do not hand-maintain it.

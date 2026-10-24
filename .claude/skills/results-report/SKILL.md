---
name: results-report
description: How to report experiment results in this repo, in a chat reply or a results/<run>/README.md record. Use whenever you present numbers from a run, write or revise a results record, or summarise a stage-one or stage-two report, even if the user only asks "how did it go" or "what are the results".
---

# Results report

Present results as an ML paper would. Each table has a caption that names the metric, the benchmark and split, and
whether higher is better. It has one row per named model, baseline or control (controls sit in the same table), one
column per benchmark, screen or held-out line plus a mean column, and the best value in bold. Explain the metric once
in one sentence. Give validation and test as separate tables or columns; validation only selects, test is reported
once. Contrasts with bootstrap intervals go in a second table. Never use bullet dumps or arm × scorer pivots as the
main presentation, and name every row by what it is, not by an internal label.

A `results/<run>/README.md` record opens with what ran and where. Then comes a table of steps with the host output
path, code SHA and finish time in UTC, and a link to the copied host `results.md`. Next come the result tables, then
a short **Reading** section of bold one-line findings, each backed by a table. A chat reply uses the same tables,
without the provenance table.

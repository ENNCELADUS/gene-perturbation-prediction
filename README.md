<div align="center">

  <h1>Generalizable Synthetic-Lethality Discovery by Virtual-Cell Composition</h1>

  <h2>Context-specific synthetic-lethality ranking in held-out cancer cell lines.</h2>

  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg"/></a>
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/managed%20with-uv-261230.svg"/></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-orange.svg"/></a>

</div>

> Can a dependency profile predicted from a cell line's basal single-cell profile rank the synthetic-lethal partners of a gene lost in that line, when the line was excluded from every fitting and selection step, beyond what pan-essentiality, the partner's predicted dependency alone, pair identity and a context-ablated model already achieve?

Stage one predicts DepMap GeneEffect for held-out cell lines; stage two ranks SL partners against experimental SL-database labels ([blueprint](docs/01-blueprint.md), §3). The generalization axis is the **cell line**; genes are not held out.

**Priority.** The goal is SL ranking in held-out lines. Stage-one accuracy is a means and is reported after the SL result. The work must differ from Cilantro-SL by an SL score that changes with the cell line and holds in unseen lines, not only by the foundation model.

**Status (2026-11-26).** The best stage-two model is the gene-knowledge, cell-set model with stage one trained on GeneEffect itself (SL protocol §19, [record](results/sl_line_term_seed0/summary.html)). Its recall in the top 10% is 0.553 validation / 0.496 test, above every published baseline that ran on the [SL benchmark](docs/data/sl-benchmark.html) (best: Kebabci on predicted GeneEffect, 0.355 / 0.395). Its score changes with the line's input (own input minus another line's input +0.045 [+0.007, +0.083] / +0.012 [−0.007, +0.030]). But no model yet orders one pair between two unseen lines beyond chance: a cross-line ranking loss and local features of the genes in the line fit the 44 training lines and do not carry to held-out lines (SL protocol §20–§21, [record](results/sl_cross_line_seed0/summary.html), [record](results/sl_context_rows_seed0/summary.html)). Since 2026-11-26 a variant is chosen by its validation line specificity (own input minus another line's input) among those within one standard error of the best validation recall (SL protocol §8); the chosen cross-line model orders test lines at 0.553 [0.510, 0.596] but validation lines at 0.468. Only a reference that reads the line's measured GeneEffect does (0.645). In stage one, no architecture beats the linear model refitted on the stage-one frame (per-line residual Spearman 0.226 / 0.227). The [model status page](model-status.html) gives the details; the GeneEffect track's results are in [protocol §7](docs/03-geneeffect-protocol.md#7-results).

## Quick start

```bash
git clone git@github.com:ENNCELADUS/gene-perturbation-prediction.git
cd gene-perturbation-prediction
uv sync                                                   # Python 3.11–3.12, project-local .venv
uv run python -m pytest tests -q > pytest.txt 2>&1        # synthetic fixtures, no external data
.venv/bin/ruff check .
```

Running the pipeline needs the raw single-cell data, DepMap labels and model weights, which live only on the GPU host and are not in git.

## Where things are

- [`docs/01-blueprint.md`](docs/01-blueprint.md): the research statement (start here). [`docs/02-literature-review.md`](docs/02-literature-review.md): related work.
- [`docs/03-geneeffect-protocol.md`](docs/03-geneeffect-protocol.md) and [`docs/04-sl-ranking-protocol.md`](docs/04-sl-ranking-protocol.md): the executable protocols and their rules. [`docs/specs/`](docs/specs/): live designs and plans.
- [`docs/data/`](docs/data/): one card per dataset; read it before using the data.
- [`src/README.md`](src/README.md): one line per module. [`hpc/README.md`](hpc/README.md): every command, run on the H20 host.
- [`results/`](results/) (local only, not tracked): one record per finished run (`summary.html`; older runs have a `README.md`).
- [`CLAUDE.md`](CLAUDE.md): rules and the current state, for coding agents.

## Data roles

| Source | Role |
| --- | --- |
| Basal single-cell atlases | Context input: the basal cells of every single-cell line ([split card](docs/data/cell-line-geneeffect-split.md), [atlas search](docs/data/single-cell-atlas-search.md)). |
| DepMap 26Q1 | GeneEffect labels and bulk expression, joined by ModelID ([card](docs/data/depmap-public-26q1.md)). |
| Perturb-seq anchors | Response supervision for the joint model (K562, HepG2, Jurkat, HCT116). |
| Multi-line SL screens | Multi-study SL benchmark: 58 studies, 310 screens, 135 lines (15 without a ModelID join by Cellosaurus ID), 76 lines in the genetic-screen headline, SL split ([card](docs/data/sl-benchmark.html), [design](docs/specs/2026-10-29-multi-study-sl-benchmark-design.md), [baselines](results/sl_study_benchmark/summary.html), [model status](model-status.html)). |

Single-gene GeneEffect predictions and DepMap-derived dependency patterns are not SL labels; a cross-line SL claim needs experimental SL labels in the named lines.

## Terminology guardrails

- Say **dependency / GeneEffect prediction** for the single-gene task and **SL candidate ranking** for the pairwise score, never "validated SL target" (benchmark negatives are unconfirmed).
- Do not claim SL from single-gene essentiality; a declared null baseline is required. A pan-essentiality lift is not an SL result: the gene-mean block must be ablated, not assumed away.
- **Never write "interaction" about a model result.** The pair head predicts no joint outcome; an interaction claim needs a joint or measured genetic-interaction quantity.
- Qualify every held-out-context result with pretraining exposure: task-label holdout is not representation-pretraining holdout (Tx1 saw Tahoe-100M; STATE's released checkpoint saw K562, HepG2 and Jurkat).
- No significance claim across contexts: one split with few test contexts admits no family-wise inference over baselines, arms and strata.
- DepMap GeneEffect is a single-gene relative growth-rate effect, not a single-cell death label.

<div align="center">

  <h1>Generalizable Synthetic-Lethality Discovery by Virtual-Cell Composition</h1>

  <h2>Context-specific synthetic-lethality ranking in held-out cancer cell lines.</h2>

  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg"/></a>
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/managed%20with-uv-261230.svg"/></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-orange.svg"/></a>

</div>

> Can a dependency profile predicted from a cell line's basal single-cell profile rank the synthetic-lethal partners of a gene lost in that line, when the line was excluded from every fitting and selection step, beyond what pan-essentiality, the partner's predicted dependency alone, pair identity and a context-ablated model already achieve?

Stage one predicts DepMap GeneEffect for held-out cell lines; stage two ranks SL partners against experimental SL-database labels ([blueprint](docs/01-blueprint.md), §3). The generalization axis is the **cell line**; genes are not held out.

**Status.** Stage one is under way: no architecture yet beats the linear context prior refitted on the stage-one frame ([stage-one record](results/stage_one_architecture/README.md), [prior-input record](results/stage_one_prior_input/README.md)). Stage two ran on the screen benchmark: pair identity wins, and predicted GeneEffect adds nothing over each gene's mean ([stage-two record](results/sl_stage_two_seed0/README.md)), so nothing here is SL context evidence. Both stages are summarised on the [status page](results/status.html); the GeneEffect track's results are in [protocol §7](docs/03-geneeffect-protocol.md#7-results) and [`results/`](results/).

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
- [`results/`](results/): one record per finished run (`summary.html`; older runs have a `README.md`), and the [status page](results/status.html).
- [`CLAUDE.md`](CLAUDE.md): rules and the current state, for coding agents.

## Data roles

| Source | Role |
| --- | --- |
| Basal single-cell atlases | Context input: the basal cells of every single-cell line ([split card](docs/data/cell-line-geneeffect-split.md), [atlas search](docs/data/single-cell-atlas-search.md)). |
| DepMap 26Q1 | GeneEffect labels and bulk expression, joined by ModelID ([card](docs/data/depmap-public-26q1.md)). |
| Perturb-seq anchors | Response supervision for the joint model (K562, HepG2, Jurkat, HCT116). |
| Multi-line SL screens | Stage-two labels ([benchmark card](docs/data/sl-screen-benchmark.md)); the multi-study benchmark that replaces it has labels, a split and baselines ([design](docs/specs/2026-10-29-multi-study-sl-benchmark-design.md), [card](docs/data/sl-study-inventory.md), [baselines](results/sl_study_benchmark/summary.html)). |

Single-gene GeneEffect predictions and DepMap-derived dependency patterns are not SL labels; a cross-line SL claim needs experimental SL labels in the named lines.

## Terminology guardrails

- Say **dependency / GeneEffect prediction** for the single-gene task and **SL candidate prioritization** for the pairwise score, never "validated SL target" (benchmark negatives are unconfirmed).
- Do not claim SL from single-gene essentiality; a declared null baseline is required. A pan-essentiality lift is not an SL result: the gene-mean block must be ablated, not assumed away.
- **Never write "interaction" about a model result.** The pair head predicts no joint outcome; an interaction claim needs a joint or measured genetic-interaction quantity.
- Qualify every held-out-context result with pretraining exposure: task-label holdout is not representation-pretraining holdout (Tx1 saw Tahoe-100M; STATE's released checkpoint saw K562, HepG2 and Jurkat).
- No significance claim across contexts: one split with few test contexts admits no family-wise inference over baselines, arms and strata.
- DepMap GeneEffect is a single-gene relative growth-rate effect, not a single-cell death label.

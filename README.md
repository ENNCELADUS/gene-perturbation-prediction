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

**Status.** Stage one is under way: no architecture yet beats the linear model refitted on the stage-one frame ([stage-one record](results/stage_one_architecture/README.md), [linear-model-input record](results/stage_one_linear_model_input/README.md)). Stage two ran on the screen benchmark: pair identity wins, and predicted GeneEffect adds nothing over each gene's mean ([stage-two record](results/sl_stage_two_seed0/README.md)), so nothing here is SL context evidence. Seven published SL methods ran as baselines on the SL benchmark (KG4SL, Cilantro-SL, PARIS, SLIdR, De Kegel 2021, Kebabci 2026 and MiT4SL; official code or Python ports, refitted on the training lines): the training SL fraction is still the best score. The best baseline, the Kebabci paralog classifier on stage one's predicted GeneEffect, comes near it only on its paralog rows. Cilantro-SL scores only pairs of genes inside a line's top 2,048 Geneformer tokens (1.5% of held-out rows); its limits are in [its record](results/cilantro_reproduction_seed0/README.md). Round zero of a modular SL backbone (SL protocol §11) ran: its conditional score (B's predicted change when A is lost) equals its context-ablated version, and its small head beats KG4SL but not the training SL fraction ([record](results/sl_backbone_round_zero_seed0/README.md)); each later round changes one module. The next module, a reader that knows each gene's buffer in the line (its paralogs and complex members, SL protocol §12), gave a null result: on the partner pairs its conditional score equals its context-ablated version ([record](results/sl_buffer_aware_reader_seed0/README.md)); the next module was the context editor. Three parts of the [PIE and single-cell plan](docs/specs/2026-11-12-pie-and-single-cell-plan.md) ran on 2026-11-13, and all three gave null results. PIE's predicted profile after A's loss as the context editor (SL protocol §14) is below the simple edit and equal to PIE's profile of another line, and on its own task PIE is not above the mean response of the other lines ([record](results/sl_pie_editor_seed0/README.md)). A stage-one reader with a weight vector for each gene on a shared line encoder (GeneEffect protocol §18) does not beat the linear model, 0.226 / 0.227 ([record](results/stage_one_gene_reader/README.md)). Each line read in its own genes, and the co-detection of A and B in the line's cells (SL protocol §13), do not change the SL ranking ([record](results/sl_single_cell_inputs_seed0/README.md)). Two studies of the labels alone (SL protocol §15, §16) find a line-specific signal that repeats weakly between two studies (0.04 to 0.12) and clearly inside one study (0.35 to 0.39 between two replicates of Harle 2025). Of four options for a less noisy training target, the top 10% of the rank as a continuous target repeats best; the logistic SL head trained on it equals the head trained on the calls (0.389 / 0.367 against 0.394 / 0.361, SL protocol §17), because that head has one weight for each input over all pairs. A deep model for stage two ran on 2026-11-19: the gene-knowledge, cell-set model (SL protocol §18, [record](results/sl_embedding_seed0/summary.html)) scores a recall in the top 10% of 0.500 validation / 0.485 test, above every published baseline (Kebabci 0.355 / 0.395, KG4SL 0.367 / 0.338). Its cell tokens add nothing (without cell tokens: 0.508 / 0.501), its early stop kept the untrained stage-one model, the recall falls from 0.782 on the training lines to 0.518 / 0.514 on seen pairs of unseen lines and 0.475 / 0.354 on new pairs, and the same weights with the line's input replaced by the fit lines' mean score the same (0.509 / 0.498), so the score is a score of the gene pair and does not yet read the line. On 2026-11-20 stage one trained on GeneEffect itself, so that stage two starts from pretrained weights (SL protocol §19, [record](results/sl_line_term_seed0/summary.html), [plan](docs/specs/2026-11-20-line-term-and-direct-target-plan.md)): the model scores 0.553 / 0.496 (0.545 / 0.505 without cell tokens), and its score now changes with the line (minus the same weights on the mean input +0.088 [+0.049, +0.128] / +0.067 [+0.015, +0.119]; minus another line's input +0.045 [+0.007, +0.083] / +0.012 [−0.007, +0.030]). The gain over the runs of 2026-11-19 holds on validation only (+0.052 [+0.019, +0.088] / +0.011 [−0.027, +0.055]). The cell tokens add nothing, and a line term on gene-in-line features of the cells (own expression, PIE's predicted change, Tx1-3B's gene embeddings) adds nothing. On 2026-11-23 the Tx1-3B caches of one SL training line (ACH-000696), which came from padding tokens because its raw file lists versioned Ensembl IDs, were encoded again; the runs trained again on the repaired caches change no conclusion (0.553 / 0.499; each difference within ±0.02). The next study is built and was launched on 2026-11-24, with no result yet (SL protocol §20, [design](docs/specs/2026-11-24-context-specific-sl-architecture-design.md)): a ranking loss between two training lines of one study for the same pair in this model, then local features of the two genes in the line (wide expression, paralog and complex buffers, DepMap mutations and copy number), one change for each run. Both stages are summarised on the [status page](results/status.html); the GeneEffect track's results are in [protocol §7](docs/03-geneeffect-protocol.md#7-results) and [`results/`](results/).

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
| Multi-line SL screens | Multi-study SL benchmark: 58 studies, 310 screens, 135 lines (15 without a ModelID join by Cellosaurus ID), 76 lines in the genetic-screen headline, SL split ([card](docs/data/sl-benchmark.html), [design](docs/specs/2026-10-29-multi-study-sl-benchmark-design.md), [baselines](results/sl_study_benchmark/summary.html), [model status](model-status.html)). |

Single-gene GeneEffect predictions and DepMap-derived dependency patterns are not SL labels; a cross-line SL claim needs experimental SL labels in the named lines.

## Terminology guardrails

- Say **dependency / GeneEffect prediction** for the single-gene task and **SL candidate ranking** for the pairwise score, never "validated SL target" (benchmark negatives are unconfirmed).
- Do not claim SL from single-gene essentiality; a declared null baseline is required. A pan-essentiality lift is not an SL result: the gene-mean block must be ablated, not assumed away.
- **Never write "interaction" about a model result.** The pair head predicts no joint outcome; an interaction claim needs a joint or measured genetic-interaction quantity.
- Qualify every held-out-context result with pretraining exposure: task-label holdout is not representation-pretraining holdout (Tx1 saw Tahoe-100M; STATE's released checkpoint saw K562, HepG2 and Jurkat).
- No significance claim across contexts: one split with few test contexts admits no family-wise inference over baselines, arms and strata.
- DepMap GeneEffect is a single-gene relative growth-rate effect, not a single-cell death label.

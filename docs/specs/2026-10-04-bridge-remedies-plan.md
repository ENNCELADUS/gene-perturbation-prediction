# Bridge Remedies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out which remedy lets gene-level features (own expression, paralog and complex partners, data-selected genes) carry their signal through the bridge from single-cell pseudo-bulk to bulk space. Run four remedies side by side in a minimal experiment runner that reports validation and test scores and decides nothing.

**Why (from the first runs, `results/context_prior_seed0/README.md`):** the chosen prior (expression components over the 953 labelled lines without haematopoietic extras) scores selective Spearman 0.223 / 0.223 on validation / test from bridged pseudo-bulk. The gene-level blocks, the most SL-shaped signal in the design, did not survive the bridge: own expression and partners lowered the score. The per-gene affine bridge's median per-gene correlation with bulk across out-of-fold training lines is 0.43.

**Architecture:**
- **Runner.** `src/experiments/context_prior.py` becomes a minimal experiment runner. A config names experiments; each experiment is a bridge remedy (`kind`), a list of settings and a list of block sets.
- **What it computes.** For every setting it builds the bridged inputs, fits the prior for every block set and penalty, and writes validation and test scores, the bulk-input oracle score on validation, paired-bootstrap gains over a reference row, and bridge diagnostics.
- **Tuning is the only choice in code.** The components penalty that a gene-level block set builds on is the one with the best validation score. Nothing is kept, dropped, passed or failed in code: a human or agent reads the table and decides.
- **Remedies.** Each remedy is a module `src/context_prior/remedies/<kind>.py` with `build(base, setting) -> BridgeInputs`, found by name, so the four remedies are disjoint files written in parallel.
- **Library hooks.** The prior gains two optional inputs, `gene_space` and `gene_rows`, that the remedies set.
- **Removed.** The runner's gates (learning curve, extra-lines decision and fallback, block-by-block selection with keep rules, view-weight trial) are deleted. `crossfit` stays in the library.

**Tech Stack:** numpy, pandas, scikit-learn (existing), the `src/context_prior` package, PyYAML, pytest.

**Spec:** `docs/specs/2026-10-04-context-generalization-design.md` (the prior, the bridge, the data); decisions for this plan were settled in conversation on 2026-10-04 and are listed under Global Constraints.

## Global Constraints

- `<scratch>` is the session scratchpad directory; `<date>` is the run date as `YYYYMMDD`.
- Every Python invocation is `uv run python -m ...` from the repo root; tests run as `uv run python -m pytest <files> -q -p no:cacheprovider > <scratch>/<name>.txt 2>&1; tail -5 <scratch>/<name>.txt`; lint with `.venv/bin/ruff check src tests`; `.venv/bin/ruff format` only on files you touched.
- Experiment code reports; it decides nothing. No pass/fail, keep/drop or fallback logic. The only automatic choice is hyperparameter tuning on validation (best point estimate). Validation and test are reported for every row.
- Training lines: the 953 labelled lines with bulk RNA after dropping the haematopoietic extras (`exclude_lineages: [Lymphoid, Myeloid]`); the shared space (9,711 genes), quantile normalisation, leakage guards and pseudo-bulk preparation are unchanged.
- Every fit reads training lines only; validation and test lines enter only as scored queries (and validation bulk as the oracle row).
- `src/context_prior/` imports `src.data` only. Configs are strict about keys they define.
- Branch `feat/bridge-remedies` from `main`; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- H20: `ssh -J richard@100.91.229.50 -p 30838 root@10.15.171.204`; code reaches the host as a git bundle; the host has no internet; read remote files in slices of a few kilobytes (larger outputs drop the session).
- Name things by what they are; no bare labels.

## Review Focus

- A remedy that changes only gene-level blocks (gating, noise-matched fitting) must leave the components-only predictions identical to the reference — pinned by a test that `build(...).expression` and `queries` equal the affine remedy's.
- `gene_rows` with a context block after a gene-level block would mix row sets — `fit_prior` refuses it; pinned in the runner task's prior-hook tests.
- A gating threshold that leaves no gene in `gene_space`: the gene-level blocks see every feature as undefined (zero) and predict only intercepts; nothing raises — pinned in the gating task.
- Two runner processes writing the same run directory with different `--experiments`: rows are per setting and never shared; `results.md` is rewritten from whatever rows exist — pinned in the runner task.
- A single-cell training line without bulk RNA: it has an out-of-fold bridged row (noise-matched fitting reads it) but no bulk row — pinned in the bridging test.

---

### Task 1: Minimal experiment runner and library hooks

**Model:** Opus. Blocks every other code task.

**Files:**
- Create: `src/context_prior/bridging.py`, `src/context_prior/remedies/__init__.py`, `src/context_prior/remedies/affine.py`, `tests/test_context_prior_bridging.py`, `configs/context_prior/bridge_remedies.yaml`
- Modify: `src/context_prior/prior.py` (`PriorInputs.gene_space`, `PriorInputs.gene_rows`, `fit_prior`), `src/experiments/config.py` (prior schema), `tests/test_context_prior_fit.py` (append hook tests)
- Rewrite: `src/experiments/context_prior.py`, `tests/test_context_prior_run.py`
- Delete: `src/context_prior/view_weights.py`, `tests/test_context_prior_view_weights.py` (orphaned once the runner stops calling them), `configs/context_prior/prior.yaml`, `configs/context_prior/prior_no_haematopoietic.yaml`

**Interfaces:**
- Produces:
  - `BridgeBase(bulk, oracle, pseudobulk, paired, single_cell_train, val, test, folds)`;
  - `BridgeInputs(expression, queries, oof_paired, gene_space=None, gene_rows=None)`;
  - `oof_bridged(pseudobulk, bulk, paired, lines, folds) -> pd.DataFrame`;
  - `bridge_diagnostics(oof_paired, bulk_paired, selective, paralogs) -> dict`;
  - `src.context_prior.remedies.affine.build(base: BridgeBase, setting: Mapping) -> BridgeInputs`;
  - `PriorInputs(..., gene_space: tuple[str, ...] | None = None, gene_rows: pd.DataFrame | None = None)`;
  - runner `main`, writing `rows/<experiment>__<n>.json` and `results.md`.

- [ ] **Step 1: Bridging module, test first.** `tests/test_context_prior_bridging.py`:

```python
"""Out-of-fold bridging, diagnostics and the affine remedy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.bridge import fit_bridge
from src.context_prior.bridging import BridgeBase, bridge_diagnostics, oof_bridged
from src.context_prior.remedies import affine

GENES = [f"G{i}" for i in range(12)]


def base(seed=0):
    rng = np.random.default_rng(seed)
    sc = [f"S{i}" for i in range(20)]  # S18, S19 have no bulk row
    paired = sc[:18]
    extras = [f"E{i}" for i in range(10)]
    signal = rng.normal(size=(len(sc) + 12, 12))
    pseudo = pd.DataFrame(
        signal + 0.3 * rng.normal(size=signal.shape),
        index=[*sc, *[f"V{i}" for i in range(6)], *[f"T{i}" for i in range(6)]],
        columns=GENES,
    )
    bulk = pd.DataFrame(
        np.vstack([2 * signal[:18] + 1, rng.normal(size=(10, 12))]),
        index=[*paired, *extras],
        columns=GENES,
    )
    oracle = pd.DataFrame(rng.normal(size=(6, 12)), index=[f"V{i}" for i in range(6)], columns=GENES)
    return BridgeBase(
        bulk=bulk,
        oracle=oracle,
        pseudobulk=pseudo,
        paired=tuple(paired),
        single_cell_train=tuple(sc),
        val=tuple(f"V{i}" for i in range(6)),
        test=tuple(f"T{i}" for i in range(6)),
        folds={m: i % 4 for i, m in enumerate(sc)},
    )


def test_oof_rows_come_from_bridges_fitted_without_their_fold():
    b = base()
    rows = oof_bridged(b.pseudobulk, b.bulk, b.paired, b.single_cell_train, b.folds)
    assert list(rows.index) == list(b.single_cell_train)  # S18, S19 included
    fold = b.folds["S3"]
    outside = [m for m in b.paired if b.folds[m] != fold]
    manual = fit_bridge(b.pseudobulk.loc[outside], b.bulk.loc[outside]).apply(
        b.pseudobulk.loc[["S3"]]
    )
    assert np.allclose(rows.loc[["S3"]], manual)


def test_affine_remedy_and_diagnostics():
    b = base()
    inputs = affine.build(b, {})
    assert inputs.expression is b.bulk
    assert set(inputs.queries) == {"val", "test", "oracle"}
    assert list(inputs.queries["test"].index) == list(b.test)
    assert list(inputs.oof_paired.index) == list(b.paired)
    paralogs = pd.DataFrame({"gene": ["G0"], "paralog": ["G1"], "identity": [50.0]})
    report = bridge_diagnostics(inputs.oof_paired, b.bulk.loc[list(b.paired)], ["G0", "G2"], paralogs)
    assert report["selective"]["total"] == 2 and report["selective_paralogs"]["total"] == 1
    assert 0.5 < report["median"] <= 1.0
```

Then `src/context_prior/bridging.py`:

```python
"""Shared inputs of the bridge remedies: the quantile-normalised sources, out-of-
fold bridging, and diagnostics of how well bridged pseudo-bulk tracks bulk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.context_prior.bridge import bridge_quality, fit_bridge

THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class BridgeBase:
    """Quantile-normalised sources every remedy starts from.

    Attributes:
        bulk: training-side bulk rows. oracle: validation bulk rows (off-contract).
        pseudobulk: pseudo-bulk of the 226 lines. paired: labelled single-cell
        training lines with bulk RNA (the bridge's fit lines). single_cell_train:
        every labelled single-cell training line. folds: patient-grouped folds over
        ``single_cell_train``.
    """

    bulk: pd.DataFrame
    oracle: pd.DataFrame
    pseudobulk: pd.DataFrame
    paired: tuple[str, ...]
    single_cell_train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    folds: Mapping[str, int]


@dataclass(frozen=True)
class BridgeInputs:
    """What a remedy hands the prior.

    Attributes:
        expression: rows the prior is fitted on (training-side bulk, possibly transformed).
        queries: ``val`` and ``test`` (bridged pseudo-bulk) and ``oracle`` (validation bulk).
        oof_paired: out-of-fold bridged rows of the paired lines, for diagnostics.
        gene_space: expression genes the gene-level blocks may read; None for all.
        gene_rows: rows the gene-level blocks are fitted on instead of ``expression``.
    """

    expression: pd.DataFrame
    queries: Mapping[str, pd.DataFrame]
    oof_paired: pd.DataFrame
    gene_space: tuple[str, ...] | None = None
    gene_rows: pd.DataFrame | None = None


def oof_bridged(
    pseudobulk: pd.DataFrame,
    bulk: pd.DataFrame,
    paired: Sequence[str],
    lines: Sequence[str],
    folds: Mapping[str, int],
) -> pd.DataFrame:
    """Bridged pseudo-bulk of ``lines``, each from a bridge fitted on the paired
    lines outside its fold."""
    parts = []
    for fold in sorted({folds[m] for m in lines}):
        fit = [m for m in paired if folds[m] != fold]
        held = [m for m in lines if folds[m] == fold]
        bridge = fit_bridge(pseudobulk.loc[fit], bulk.loc[fit])
        parts.append(bridge.apply(pseudobulk.loc[held]))
    return pd.concat(parts).loc[list(lines)]


def bridge_diagnostics(
    oof_paired: pd.DataFrame,
    bulk_paired: pd.DataFrame,
    selective: Sequence[str],
    paralogs: pd.DataFrame,
) -> dict:
    """Per-gene correlation across lines between out-of-fold bridged pseudo-bulk
    and bulk: quartiles, and counts at each threshold for all genes, the selective
    genes and the selective genes' paralogs present in the space."""
    quality = bridge_quality(oof_paired, bulk_paired.loc[oof_paired.index, oof_paired.columns])
    present = set(quality.index)
    selective_present = [g for g in selective if g in present]
    partner = paralogs.loc[paralogs["gene"].isin(set(selective)), "paralog"]
    paralogs_present = sorted(set(partner) & present)

    def counts(genes: Sequence[str]) -> dict:
        values = quality.loc[list(genes)]
        return {"total": len(genes), **{str(t): int((values >= t).sum()) for t in THRESHOLDS}}

    defined = quality.dropna()
    return {
        "median": float(defined.median()),
        "q25": float(defined.quantile(0.25)),
        "q75": float(defined.quantile(0.75)),
        "all": counts(list(quality.index)),
        "selective": counts(selective_present),
        "selective_paralogs": counts(paralogs_present),
    }
```

`src/context_prior/remedies/__init__.py`: one docstring line, `"""Bridge remedies: each module exposes build(base, setting) -> BridgeInputs."""`. `src/context_prior/remedies/affine.py`:

```python
"""The reference bridge: one affine map per gene, pseudo-bulk to bulk."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.context_prior.bridge import fit_bridge
from src.context_prior.bridging import BridgeBase, BridgeInputs, oof_bridged


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    if setting:
        raise ValueError(f"the affine bridge takes no settings, got {dict(setting)}")
    paired = list(base.paired)
    bridge = fit_bridge(base.pseudobulk.loc[paired], base.bulk.loc[paired])
    return BridgeInputs(
        expression=base.bulk,
        queries={
            "val": bridge.apply(base.pseudobulk.loc[list(base.val)]),
            "test": bridge.apply(base.pseudobulk.loc[list(base.test)]),
            "oracle": base.oracle,
        },
        oof_paired=oof_bridged(base.pseudobulk, base.bulk, paired, paired, base.folds),
    )
```

Run the test file; expect `2 passed`.

- [ ] **Step 2: Prior hooks, test first.** Append to `tests/test_context_prior_fit.py` (reuse its `synthetic()`):

```python
def test_gene_space_hides_unlisted_genes_from_gene_level_blocks():
    from dataclasses import replace

    inputs, ids = synthetic()
    spec = PriorSpec((Stage("expression_components", 0.1), Stage("own_expression", 1.0)))
    hidden = replace(inputs, gene_space=tuple(g for g in SPACE if g != "E1"))
    stage = fit_prior(spec, hidden, fit_lines=ids, encoder_lines=ids).predict(
        inputs.expression.loc[ids[:10]]
    )["own_expression"]
    assert np.allclose(stage["E1"], stage["E1"].iloc[0])  # intercept only
    visible = fit_prior(spec, inputs, fit_lines=ids, encoder_lines=ids).predict(
        inputs.expression.loc[ids[:10]]
    )["own_expression"]
    assert visible["E1"].std() > 0  # the same gene varies when its column is readable


def test_gene_rows_fit_gene_level_blocks_on_other_rows_and_order_is_enforced():
    from dataclasses import replace

    inputs, ids = synthetic()
    rng = np.random.default_rng(3)
    noisy = inputs.expression.loc[ids[:60]] + 3.0 * rng.normal(size=(60, 12))
    spec = PriorSpec((Stage("expression_components", 0.1), Stage("own_expression", np.inf)))
    clean = fit_prior(spec, inputs, fit_lines=ids, encoder_lines=ids)
    matched = fit_prior(
        spec, replace(inputs, gene_rows=noisy), fit_lines=ids, encoder_lines=ids
    )

    def weight(fitted):
        models = {block: model for block, _, model in fitted.stages}
        return abs(models["own_expression"].pooled[0])

    assert weight(matched) < weight(clean)  # attenuation learnt on noisy rows
    wrong = PriorSpec((Stage("own_expression", 1.0), Stage("expression_components", 0.1)))
    with pytest.raises(ValueError, match="before"):
        fit_prior(wrong, replace(inputs, gene_rows=noisy), fit_lines=ids, encoder_lines=ids)
```

Then change `src/context_prior/prior.py`:
- Add to `PriorInputs`, after `patients`: `gene_space: tuple[str, ...] | None = None` and `gene_rows: pd.DataFrame | None = None`, with docstring lines.
- Replace the body of `fit_prior` after `basis = ...` with:

```python
    gene_columns = (
        space
        if inputs.gene_space is None
        else tuple(g for g in space if g in set(inputs.gene_space))
    )
    gene_rows = inputs.gene_rows
    gene_fit_rows = fit_rows if gene_rows is None else gene_rows
    gene_remaining: np.ndarray | None = None
    index: PartnerIndex | None = None
    low: np.ndarray | None = None
    stages = []
    for stage in spec.stages:
        if stage.block in CONTEXT_BLOCKS:
            if gene_remaining is not None:
                raise ValueError("context blocks must come before gene-level blocks")
            features, train = _context_view(stage.block, inputs, fit_rows, encoder_rows)
            model = shared_ridge(train, remaining, [stage.penalty], basis=basis)[0]
            remaining -= model.predict(train)
            stages.append((stage.block, features, model))
            continue
        if gene_remaining is None:
            # Gene-level blocks fit on their own rows: what the context stages
            # leave on those rows (the same rows as the context stages by default).
            if gene_rows is None:
                gene_remaining = remaining
            else:
                residual = inputs.residual.loc[list(gene_rows.index)].to_numpy(dtype=np.float64)
                gene_remaining = np.where(np.isfinite(residual), residual, 0.0)
                for _, context_features, context_model in stages:
                    gene_remaining = gene_remaining - context_model.predict(
                        context_features(gene_rows)
                    )
        rows = gene_fit_rows.loc[:, list(gene_columns)]
        if stage.block in _KNOWLEDGE_FEATURES:
            if index is None:
                index = partner_index(genes, gene_columns, inputs.reference)
                source = encoder_rows if gene_rows is None else gene_rows
                low = np.percentile(
                    source.loc[:, list(gene_columns)].to_numpy(dtype=np.float64),
                    LOW_PERCENTILE,
                    axis=0,
                )
            inner, train = _knowledge_view(_KNOWLEDGE_FEATURES[stage.block], index, low, rows)
            model = gene_ridge(train, gene_remaining, [stage.penalty], pooled=True)[0]
        else:
            inner, train = _selected_view(rows)
            position = {gene: i for i, gene in enumerate(gene_columns)}
            own = np.array([position.get(gene, -1) for gene in genes], dtype=int)
            selection = select_genes(train, gene_remaining, stage.selected, own)
            model = selected_ridge(train, selection, gene_remaining, [stage.penalty])[0]
        gene_remaining = gene_remaining - model.predict(train)

        def features(rows_in: pd.DataFrame, inner=inner) -> np.ndarray:
            return inner(rows_in.loc[:, list(gene_columns)])

        stages.append((stage.block, features, model))
    return FittedPrior(genes, space, stages)
```

(`gene_remaining = remaining` aliases the array on purpose when there are no separate rows; gene stages reassign instead of updating in place.) Run `tests/test_context_prior_fit.py`; every existing test must still pass.

- [ ] **Step 3: Config schema and config.** In `src/experiments/config.py` replace the prior schema with:

```python
_PRIOR_GROUPS = {
    "paths": "extra_lines reference model bulk_expression",
    "training_side": "exclude_lineages",
    "prior": "components folds bootstrap_repeats selected_genes",
    "penalties": "components gene",
}
_PRIOR_TOP_LEVEL = "seed joint_config output_root reference block_sets experiments"
_EXPERIMENT_KEYS = "kind settings block_sets"
GENE_BLOCKS = ("own_expression", "partners", "data_selected")


def validate_prior_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys of a linear-context-prior experiment config; experiments and block
    sets are named by the config, their shape is checked."""
    _require_keys(config, {*_PRIOR_GROUPS, *_PRIOR_TOP_LEVEL.split()}, "config")
    for name, keys in _PRIOR_GROUPS.items():
        _require_keys(config[name], set(keys.split()), name)
    for name, blocks in config["block_sets"].items():
        if not blocks or blocks[0] != "expression_components" or any(
            b not in GENE_BLOCKS for b in blocks[1:]
        ):
            raise ValueError(
                f"block set {name}: expression_components first, then gene-level blocks"
            )
    for name, experiment in config["experiments"].items():
        _require_keys(experiment, set(_EXPERIMENT_KEYS.split()), f"experiments.{name}")
        unknown = set(experiment["block_sets"]) - set(config["block_sets"])
        if unknown:
            raise ValueError(f"experiments.{name}: unknown block sets {sorted(unknown)}")
    if config["reference"] not in config["experiments"]:
        raise ValueError("reference must name an experiment")
    return dict(config)
```

`configs/context_prior/bridge_remedies.yaml`:

```yaml
# Bridge remedies for the linear context prior: every experiment x setting x block
# set x penalty is scored on validation and test; the table is the result.
seed: 0
joint_config: configs/revision/frozen_huber.yaml
output_root: outputs/context_prior
paths:
  extra_lines: configs/benchmarks/extra_bulk_lines_26Q1.json
  reference: configs/context_prior/reference
  model: data/sl_dependency_v0/raw/depmap/Model.csv
  bulk_expression: data/sl_dependency_v0/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
training_side:
  exclude_lineages: [Lymphoid, Myeloid]
prior: {components: 128, folds: 5, bootstrap_repeats: 1000, selected_genes: 50}
penalties:
  components: [0.1, 1.0, 10.0, 100.0, 1000.0]
  gene: [0.1, 1.0, 10.0]
block_sets:
  components: [expression_components]
  own_and_partners: [expression_components, own_expression, partners]
  selected: [expression_components, data_selected]
  all: [expression_components, own_expression, partners, data_selected]
reference: affine
experiments:
  affine:
    kind: affine
    settings: [{}]
    block_sets: [components, own_and_partners, selected, all]
  contrastive:
    kind: contrastive
    settings:
      - {pseudo_components: 2, bulk_components: 0}
      - {pseudo_components: 4, bulk_components: 0}
      - {pseudo_components: 8, bulk_components: 0}
      - {pseudo_components: 16, bulk_components: 0}
      - {pseudo_components: 2, bulk_components: 4}
      - {pseudo_components: 4, bulk_components: 4}
      - {pseudo_components: 8, bulk_components: 4}
      - {pseudo_components: 16, bulk_components: 4}
    block_sets: [components, own_and_partners, selected, all]
  gating:
    kind: gating
    settings: [{threshold: 0.3}, {threshold: 0.5}, {threshold: 0.7}]
    block_sets: [own_and_partners, selected, all]
  noise_matched:
    kind: noise_matched
    settings: [{}]
    block_sets: [own_and_partners, selected, all]
  denoise:
    kind: denoise
    settings: [{rank: 16}, {rank: 32}, {rank: 64}]
    block_sets: [components, own_and_partners, selected, all]
```

- [ ] **Step 4: Rewrite the runner, test first.** Replace `tests/test_context_prior_run.py` with tests that build a synthetic `RunBase` (from the bridging test's `base()` plus a `Definitions` over 8 target genes, a `Reference` like `tests/test_context_prior_fit.py`'s, a `gene_effect` frame for every line, and a `models` frame) and check:
  - `validate_prior_config` accepts `bridge_remedies.yaml` and rejects an unknown experiment key, a block set not starting with `expression_components`, and a `reference` that names no experiment;
  - `run_setting` for the affine experiment returns one row per components penalty plus 3 rows per other block set, each with `val`, `test` and `oracle` metrics and `val`/`test` gains, and the gene-level rows use the best-validation components penalty;
  - the reference row's own gain is 0 on both splits;
  - `main` with a tiny config writes `rows/affine__0.json` and `results.md`, and a second call skips the existing row file (monkeypatch `load_base` to return the synthetic base and count calls to `run_setting`);
  - `--experiments` limits the run to the named experiments.

Then replace `src/experiments/context_prior.py` with this structure (keep the existing helpers `scaled_residual`, `long_frame`, `_jsonable`, `_write_json`, `_read_json`, `_bind` as they are, minus the `oracle_only` argument of `_bind`):

```python
"""Linear context prior experiments.

``python -m src.experiments.context_prior CONFIG [--run-id ID] [--experiments A,B]``
runs every listed experiment: for each bridge setting it builds the bridged inputs
(``src.context_prior.remedies.<kind>.build``), fits the prior for every block set
and penalty, and scores validation and test (and the bulk-input oracle on
validation). Each setting writes ``rows/<experiment>__<n>.json`` and is skipped when
it exists, so a rerun resumes; ``results.md`` tabulates every row present. The
components penalty under a gene-level block set is the one with the best validation
score (tuning); nothing else is chosen here.
"""

METRICS = (
    "selective_spearman",
    "selective_aupr_lift",
    "residual_pearson_macro_per_gene",
    "geneeffect_loss",
    "residual_sd_ratio_macro_per_gene",
)


@dataclass(frozen=True)
class RunBase:
    config: Mapping[str, Any]
    split: FixedSplit
    gene_effect: pd.DataFrame
    definitions: Definitions
    labelled: tuple[str, ...]  # labelled training-side lines with bulk RNA
    models: pd.DataFrame
    reference: Reference
    bridge: BridgeBase
    filled_lines: int

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        return scaled_residual(self.gene_effect, lines, self.definitions)


def load_base(config: Mapping[str, Any]) -> RunBase:
    """Read and normalise once: today's ``load_run_data`` up to and including the
    quantile normalisation of bulk (training side, then validation rows) and of
    pseudo-bulk, with the same drops, measured-gene space and fill; plus the
    patient-grouped folds over the labelled single-cell training lines."""


def score(prediction: pd.DataFrame, truth: pd.DataFrame, definitions: Definitions) -> dict:
    from src.eval.geneeffect import aggregate_geneeffect

    frame = long_frame(prediction, truth, definitions)
    metrics, _, _ = aggregate_geneeffect(
        frame,
        model_ids=list(prediction.index),
        genes=list(definitions.genes),
        variable_genes=list(definitions.variable),
        selective_genes=list(definitions.selective),
    )
    return {key: metrics[key] for key in METRICS}


def gain(better, reference, truth, definitions, repeats) -> dict:
    """Selective-Spearman difference and its 95% paired line-bootstrap interval."""


def prior_inputs(base: RunBase, inputs: BridgeInputs) -> PriorInputs:
    lines = list(dict.fromkeys([*base.labelled, *base.bridge.single_cell_train]))
    return PriorInputs(
        expression=inputs.expression,
        residual=base.truth(lines),
        components=fit_expression_components(
            inputs.expression, int(base.config["prior"]["components"])
        ),
        reference=base.reference,
        lineage=base.models.loc[list(inputs.expression.index), "lineage"],
        patients=base.models["patient_id"].to_dict(),
        gene_space=inputs.gene_space,
        gene_rows=inputs.gene_rows,
    )


def run_setting(base, experiment, setting, reference) -> dict:
    """Every block set and penalty of one bridge setting. ``reference`` maps
    ``val``/``test`` to the reference row's predictions (the gain baseline)."""
    # build -> prior_inputs -> components grid (all penalties, scored) -> best
    # validation penalty -> each other block set x gene penalty with
    # Stage(block, g, selected=prior.selected_genes if data_selected else 0)
    # -> rows {block_set, components_penalty, gene_penalty, val, test, oracle,
    # val_gain, test_gain}; components rows only when "components" is listed;
    # diagnostics = bridge_diagnostics(inputs.oof_paired,
    # inputs.expression.loc[paired], selective, reference.paralogs) plus
    # gene_space size when set.


def reference_predictions(base: RunBase) -> dict[str, pd.DataFrame]:
    """The reference experiment's first setting, components only, at its best
    validation penalty: its ``val`` and ``test`` predictions."""


def write_results(run_dir: Path) -> Path:
    """``results.md``: run facts (space size, filled lines, training lines), then per
    experiment a table with one row per setting x block set x penalty (validation and
    test selective Spearman, gain over the reference with interval, AUPR lift,
    residual Pearson, oracle validation selective Spearman) and a diagnostics table
    per setting."""


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CONFIG, --run-id (default ``prior_<UTC timestamp>``), --experiments;
    bind the run directory to the config; run every pending setting of the named
    experiments; write results.md."""
```

Fill in each function body. `load_base` reuses the body of today's `load_run_data` (read it before deleting it). `gain` reuses today's `paired_gain` body. `run_setting` imports the remedy with `importlib.import_module(f"src.context_prior.remedies.{experiment['kind']}")`. Delete everything else in the old runner: curve, decide, select_blocks, run_selection, run_crossfit, run_view_weights, write_oof, run_test, the old summary and the `--oracle-only` flag.

- [ ] **Step 5: Delete the orphans, then run the tests and lint.**
  - Delete: `git rm src/context_prior/view_weights.py tests/test_context_prior_view_weights.py configs/context_prior/prior.yaml configs/context_prior/prior_no_haematopoietic.yaml`.
  - Run: the full suite to a file, and `.venv/bin/ruff check src tests`.
  - Expected: all pass.

- [ ] **Step 6: Commit** with `refactor(context-prior): minimal experiment runner; remedies plug in by kind`.

---

### Task 2: Contrastive-PCA remedy

**Model:** Opus. Starts after Task 1; parallel with Tasks 3–6.

**Files:** create `src/context_prior/alignment.py`, `src/context_prior/remedies/contrastive.py`, `tests/test_context_prior_alignment.py`.

**Interfaces:**
- Consumes: `BridgeBase`, `BridgeInputs`, `affine.build`.
- Produces: `contrastive_directions(first, second, count) -> np.ndarray`; `Alignment(genes, directions, pseudo_mean, bulk_mean)` with `apply(frame, *, source)` and `count`; `fit_alignment(pseudobulk, bulk, *, pseudo_components, bulk_components) -> Alignment`; `remedies.contrastive.build(base, setting)` with setting keys `pseudo_components`, `bulk_components`.

- [ ] **Step 1: Alignment module, test first.**

```python
"""Contrastive directions between paired pseudo-bulk and bulk, and their removal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.alignment import contrastive_directions, fit_alignment

GENES = [f"G{i}" for i in range(30)]
RNG = np.random.default_rng(7)
#: The pseudo-bulk-only direction: spread over every gene, not one gene's axis.
NOISE_AXIS = RNG.normal(size=30) / np.linalg.norm(RNG.normal(size=30))


def paired(seed=0, lines=60):
    rng = np.random.default_rng(seed)
    axis = NOISE_AXIS / np.linalg.norm(NOISE_AXIS)
    shared = rng.normal(size=(lines, 3)) @ rng.normal(size=(3, 30))
    pseudo = shared + 5.0 * rng.normal(size=(lines, 1)) * axis + 0.1 * rng.normal(size=(lines, 30))
    bulk = shared + 0.1 * rng.normal(size=(lines, 30))
    index = [f"L{i}" for i in range(lines)]
    return (
        pd.DataFrame(pseudo, index=index, columns=GENES),
        pd.DataFrame(bulk, index=index, columns=GENES),
    )


def mean_gene_correlation(left, right):
    return np.mean([np.corrcoef(left[g], right[g])[0, 1] for g in GENES])


def test_directions_find_the_pseudo_bulk_only_axis():
    pseudo, bulk = paired()
    directions = contrastive_directions(pseudo.to_numpy(), bulk.to_numpy(), 1)
    axis = NOISE_AXIS / np.linalg.norm(NOISE_AXIS)
    assert directions.shape == (30, 1) and abs(directions[:, 0] @ axis) > 0.95


def test_only_positive_excess_variance_counts():
    x = np.random.default_rng(1).normal(size=(40, 10))
    assert contrastive_directions(x, 2 * x, 3).shape == (10, 0)


def test_zero_counts_are_the_identity():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=0, bulk_components=0)
    assert alignment.count == 0 and alignment.apply(pseudo, source="pseudo").equals(pseudo)


def test_removal_makes_the_sources_agree_and_is_affine_for_other_rows():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=1, bulk_components=0)
    before = mean_gene_correlation(pseudo, bulk)
    after = mean_gene_correlation(
        alignment.apply(pseudo, source="pseudo"), alignment.apply(bulk, source="bulk")
    )
    assert after > before + 0.05 and after > 0.95
    other = bulk.iloc[:5] + 1.0
    shifted = alignment.apply(other, source="bulk") - alignment.apply(bulk.iloc[:5], source="bulk")
    keep = np.eye(30) - alignment.directions @ alignment.directions.T
    assert np.allclose(shifted.to_numpy(), (other - bulk.iloc[:5]).to_numpy() @ keep)


def test_overlapping_direction_sets_are_removed_once():
    pseudo, bulk = paired()
    alignment = fit_alignment(pseudo, bulk, pseudo_components=3, bulk_components=3)
    assert np.allclose(alignment.directions.T @ alignment.directions, np.eye(alignment.count))
    assert alignment.count <= 6


def test_misaligned_frames_and_unknown_source_raise():
    pseudo, bulk = paired()
    with pytest.raises(ValueError, match="paired"):
        fit_alignment(pseudo, bulk.iloc[::-1], pseudo_components=1, bulk_components=0)
    alignment = fit_alignment(pseudo, bulk, pseudo_components=1, bulk_components=0)
    with pytest.raises(ValueError, match="source"):
        alignment.apply(pseudo, source="tumour")
```

`src/context_prior/alignment.py`:

```python
"""Contrastive alignment of pseudo-bulk and bulk before the bridge.

Celligner's contrastive PCA adapted to paired profiles: on lines measured both
ways, the directions with more variance in one source than in the other are the
top eigenvectors of the difference of their covariance matrices. Their union is
projected out of every row of both sources, each about its own paired mean. Both
covariances are over the same lines, so no cluster-mean removal or nearest-
neighbour matching is needed. Reads no label; fitted once on the paired lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Singular values below this fraction of the largest mark a dependent column.
_RANK_TOLERANCE = 1e-8


def contrastive_directions(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray:
    """Genes x k orthonormal directions (k <= ``count``) with the largest positive
    excess variance of ``first`` over ``second``, computed in the span of the data."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.zeros((first.shape[1], 0))
    a = first - first.mean(axis=0)
    b = second - second.mean(axis=0)
    basis, triangle = np.linalg.qr(np.vstack([a, b]).T)
    left, right = triangle[:, : len(a)], triangle[:, len(a) :]
    values, vectors = np.linalg.eigh(left @ left.T / len(a) - right @ right.T / len(b))
    order = np.argsort(values)[::-1][:count]
    order = order[values[order] > 0]
    return basis @ vectors[:, order]


@dataclass(frozen=True)
class Alignment:
    genes: tuple[str, ...]
    directions: np.ndarray
    pseudo_mean: np.ndarray
    bulk_mean: np.ndarray

    @property
    def count(self) -> int:
        return int(self.directions.shape[1])

    def apply(self, frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
        if source == "pseudo":
            mean = self.pseudo_mean
        elif source == "bulk":
            mean = self.bulk_mean
        else:
            raise ValueError(f"source must be 'pseudo' or 'bulk', not {source!r}")
        if self.count == 0:
            return frame
        values = frame.loc[:, list(self.genes)].to_numpy(dtype=np.float64)
        projected = ((values - mean) @ self.directions) @ self.directions.T
        return pd.DataFrame(values - projected, index=frame.index, columns=list(self.genes))


def fit_alignment(
    pseudobulk: pd.DataFrame,
    bulk: pd.DataFrame,
    *,
    pseudo_components: int,
    bulk_components: int,
) -> Alignment:
    if list(pseudobulk.index) != list(bulk.index) or list(pseudobulk.columns) != list(
        bulk.columns
    ):
        raise ValueError("pseudo-bulk and bulk must be paired on lines and genes")
    p, b = pseudobulk.to_numpy(dtype=np.float64), bulk.to_numpy(dtype=np.float64)
    union = np.hstack(
        [
            contrastive_directions(p, b, pseudo_components),
            contrastive_directions(b, p, bulk_components),
        ]
    )
    if union.shape[1]:
        left, singular, _ = np.linalg.svd(union, full_matrices=False)
        union = left[:, singular > _RANK_TOLERANCE * singular[0]]
    return Alignment(tuple(pseudobulk.columns), union, p.mean(axis=0), b.mean(axis=0))
```

(Its code and tests passed together in a scratch run while this plan was written.)

- [ ] **Step 2: The remedy.** `src/context_prior/remedies/contrastive.py`:

```python
"""Contrastive-PCA remedy: project the directions the two sources do not share out
of both, then bridge as the affine reference does."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.context_prior.alignment import fit_alignment
from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    paired = list(base.paired)
    alignment = fit_alignment(
        base.pseudobulk.loc[paired],
        base.bulk.loc[paired],
        pseudo_components=int(setting["pseudo_components"]),
        bulk_components=int(setting["bulk_components"]),
    )
    aligned = replace(
        base,
        bulk=alignment.apply(base.bulk, source="bulk"),
        oracle=alignment.apply(base.oracle, source="bulk"),
        pseudobulk=alignment.apply(base.pseudobulk, source="pseudo"),
    )
    return affine.build(aligned, {})
```

Add to the test file a test that `build` with `pseudo_components=0, bulk_components=0` returns the affine remedy's inputs unchanged. Use the bridging test's `base()`; copy it in, since tests don't import each other.

- [ ] **Step 3:** Run the tests and ruff, then commit with `feat(context-prior): contrastive-PCA bridge remedy`.

---

### Task 3: Reliability-gating remedy

**Model:** Opus. Parallel with Tasks 2, 4–6.

**Files:** create `src/context_prior/remedies/gating.py`, `tests/test_context_prior_gating.py`.

- [ ] **Step 1: Test first** (copy the bridging test's `base()`):
  - the inputs equal the affine remedy's except `gene_space`;
  - `gene_space` holds exactly the genes whose out-of-fold bridge quality on the paired lines is at or above the threshold (compute it independently with `bridge_quality(affine.build(...).oof_paired, base.bulk.loc[paired])`);
  - a threshold above every quality gives an empty `gene_space` without raising.
- [ ] **Step 2: Implement.**

```python
"""Reliability gating: gene-level blocks read only genes whose out-of-fold bridge
quality reaches the threshold; others count as undefined."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.context_prior.bridge import bridge_quality
from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    inputs = affine.build(base, {})
    quality = bridge_quality(inputs.oof_paired, base.bulk.loc[list(base.paired)])
    threshold = float(setting["threshold"])
    space = tuple(g for g in base.bulk.columns if quality[g] >= threshold)
    return replace(inputs, gene_space=space)
```

- [ ] **Step 3:** Run the tests and ruff, then commit with `feat(context-prior): reliability-gating bridge remedy`.

---

### Task 4: Noise-matched remedy

**Model:** Opus. Parallel with Tasks 2, 3, 5, 6.

**Files:** create `src/context_prior/remedies/noise_matched.py`, `tests/test_context_prior_noise_matched.py`.

- [ ] **Step 1: Test first** (copy the bridging test's `base()`):
  - the inputs equal the affine remedy's except `gene_rows`;
  - `gene_rows` is `oof_bridged(..., lines=base.single_cell_train, ...)`, including the lines without bulk;
  - fitting a prior with own expression on these inputs learns a smaller pooled own-expression weight than on the affine inputs, on synthetic data where pseudo-bulk is a noisy copy of bulk.
- [ ] **Step 2: Implement.**

```python
"""Noise-matched fitting: gene-level blocks are fitted on the single-cell training
lines' out-of-fold bridged pseudo-bulk, the inputs they meet at query time, so
their weights learn the bridge's noise."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.context_prior.bridging import BridgeBase, BridgeInputs, oof_bridged
from src.context_prior.remedies import affine


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    if setting:
        raise ValueError(f"noise-matched fitting takes no settings, got {dict(setting)}")
    inputs = affine.build(base, {})
    rows = oof_bridged(
        base.pseudobulk, base.bulk, base.paired, base.single_cell_train, base.folds
    )
    return replace(inputs, gene_rows=rows)
```

- [ ] **Step 3:** Run the tests and ruff, then commit with `feat(context-prior): noise-matched bridge remedy`.

---

### Task 5: Low-rank denoising remedy

**Model:** Sonnet. Parallel with Tasks 2–4, 6.

**Files:** create `src/context_prior/remedies/denoise.py`, `tests/test_context_prior_denoise.py`.

- [ ] **Step 1: Test first** (copy the bridging test's `base()`):
  - `expression` and the `oracle` query equal the affine remedy's;
  - the bridged `val`/`test` queries and `oof_paired` are reconstructions of rank `rank` in the bulk components' standardised space: projecting a denoised row again leaves it unchanged;
  - with `rank` equal to the number of bulk components it can hold, the queries equal the affine remedy's up to float tolerance.
- [ ] **Step 2: Implement.**

```python
"""Low-rank denoising: bridged pseudo-bulk queries are replaced by their
reconstruction from the leading bulk components; the prior is fitted on bulk as
in the reference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine
from src.data.context_pca import fit_context_pca


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    inputs = affine.build(base, {})
    pca = fit_context_pca(base.bulk.to_numpy(dtype=np.float64), int(setting["rank"]))

    def denoise(frame: pd.DataFrame) -> pd.DataFrame:
        z = (frame.to_numpy(dtype=np.float64) - pca.mean) / pca.scale
        rebuilt = (z @ pca.components.T) @ pca.components
        return pd.DataFrame(
            rebuilt * pca.scale + pca.mean, index=frame.index, columns=frame.columns
        )

    queries = dict(inputs.queries)
    queries["val"] = denoise(queries["val"])
    queries["test"] = denoise(queries["test"])
    return replace(inputs, queries=queries, oof_paired=denoise(inputs.oof_paired))
```

- [ ] **Step 3:** Run the tests and ruff, then commit with `feat(context-prior): low-rank denoising bridge remedy`.

---

### Task 6: Documents

**Model:** Sonnet. Parallel with Tasks 2–5. Edits docs only.

**Files:** modify `docs/01-blueprint.md` (rewrite), `docs/03-geneeffect-protocol.md`, `docs/04-sl-ranking-protocol.md`, `docs/specs/2026-10-04-context-generalization-design.md`, `CLAUDE.md`, `hpc/README.md`.

- [ ] **Step 1: Blueprint.** Rewrite `docs/01-blueprint.md` as research motivation and big picture only, without hedging language and without operative rules. It should cover:
  - the question: rank context-dependent synthetic-lethal partners in a cell line from its basal single cells;
  - why held-out cell lines;
  - the two-step approach (a GeneEffect context model, then SL ranking built on it);
  - where the work stands, with links to the protocols, specs and results.
- [ ] **Step 2: Rules.**
  - **GeneEffect.** Add a short "Rules" section to `docs/03-geneeffect-protocol.md` holding the blueprint's operative content, stated as standard ML practice:
    - fit on training lines; tune on validation; report validation and test;
    - a query line supplies basal single cells only (validation bulk RNA only in the labelled oracle row);
    - the split file and the extra-line membership file are the line authorities;
    - Tx1 and STATE pretraining exposure is noted where results are compared.
  - **SL.** Move the SL-specific rules (pair universe, out-of-fold SL vectors, label provenance) into `docs/04-sl-ranking-protocol.md` beside its existing rules.
  - Keep both sections minimal.
- [ ] **Step 3: Pointers.**
  - Retarget CLAUDE.md's references to "blueprint §1" and "§4" (and any in the protocols and specs) to the new sections.
  - In CLAUDE.md, replace the "follow the blueprint's claim boundaries (§4)" pitfall with a pointer to the rules sections.
  - Add one line under Project rules: "Experiment code reports results; decisions are made by reading them, not in code."
- [ ] **Step 4: The runner.**
  - Rewrite protocol §10's run description for the minimal runner: config with experiments, settings, block sets and penalties; rows per setting; `results.md`; tuning as the only automatic choice.
  - Update the design spec §7–§8 to say selection and the extra-lines decision are read from the reported rows, not applied in code.
  - Update `hpc/README.md`'s `prior` section: outputs `rows/`, `results.md` and `run_config.json`, plus the `--experiments` flag.
  - Keep `hpc/run.sh prior CONFIG [--run-id ID] [--experiments A,B]` in CLAUDE.md's commands.
- [ ] **Step 5: Commit** with `docs: blueprint as research statement; minimal rules in the protocols; minimal experiment runner`.

---

### Task 7: Run on H20

**Owner:** the coordinator.

- [ ] **Step 1: Ship the branch.**
  1. After Tasks 1–6 are merged on `feat/bridge-remedies` and the full suite passes, run `git bundle create <scratch>/br.bundle 4334eae..feat/bridge-remedies`.
  2. Stream the bundle over SSH and `git fetch` it into `/2023533015/VCC_Project` as `feat/bridge-remedies`.
  3. Run `git worktree add /2023533015/VCC_Project_bridge_remedies feat/bridge-remedies`.
  4. Inside the worktree, symlink `data`, `model`, `.venv-tx1` and `configs/experiments/13_geneeffect_226/basal_source_registry.csv` from `/2023533015/VCC_Project`.
- [ ] **Step 2: Launch two processes on one run directory, detached, 32 threads each:**

```bash
cd /2023533015/VCC_Project_bridge_remedies && mkdir -p outputs/launches
(OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 setsid nohup hpc/run.sh prior configs/context_prior/bridge_remedies.yaml --run-id bridge_remedies_<date> --experiments affine,contrastive > outputs/launches/bridge_remedies_<date>_a.log 2>&1 < /dev/null &)
(OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 setsid nohup hpc/run.sh prior configs/context_prior/bridge_remedies.yaml --run-id bridge_remedies_<date> --experiments gating,noise_matched,denoise > outputs/launches/bridge_remedies_<date>_b.log 2>&1 < /dev/null &)
```

- [ ] **Step 3: Confirm, then collect.**
  - Confirm the 16 row files (`rows/affine__0.json` through `rows/denoise__2.json`) and `results.md`.
  - Copy `results.md` and the row files to `<scratch>` in slices of a few kilobytes.

---

### Task 8: Report

**Model:** Opus. Reads results; writes no code.

- [ ] **Step 1:** Read `results.md` and the row files. Write `results/bridge_remedies_seed0/README.md` covering:
  - the run facts;
  - for each remedy, what it did to bridge quality and to each gene-level block set relative to the reference (validation and test side by side, with the bootstrap intervals);
  - the best row per remedy by validation, and its test score;
  - whether any remedy lets own expression, partners or data-selected genes add to the components;
  - a recommendation for what to integrate, stated as the author's reading of the numbers.

  Copy `results.md` beside it. Commit with `docs(results): bridge remedies seed-0 runs`. You decide what to integrate; then merge `feat/bridge-remedies` into `main`.

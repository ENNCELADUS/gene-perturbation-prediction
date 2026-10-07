# Single-Cell Correction, Wave One: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing nested low-rank head on what the default linear context prior leaves, so that the stack (prior plus head) is scored against the prior alone, the prior plus a Tx1 ridge, and the existing controls.

**Architecture:** A CPU step exports the default prior's predictions:
- out of fold for the 170 labelled single-cell training lines, with the bridge and every stage refitted without each line's fold;
- for validation and test, from the fit on the whole training side.

The joint config names that export (`paths.prior`). The dataset carries each row's prior offset, and both the loss and evaluation score `prior + head`. The head's output layers start at zero and validation runs before the first update, so "the prior alone" is a candidate for `best.pt`. Wave one changes no model input:
- an objective screen with STATE absent (Huber, standardised MSE, standardised MSE plus a listwise loss across lines, standardised MSE plus a dependency cross-entropy);
- then the winning objective with frozen and with trainable STATE.

**Tech Stack:** Python 3.11–3.12 via `uv`, PyTorch with Accelerate (bf16 on the H20 host), NumPy, pandas, scikit-learn, pytest, ruff.

**Spec:** `docs/specs/2026-10-04-context-generalization-design.md` §5.2, §7.2, §7.3, §8.2, as amended on 2026-10-07 (§12 of that file). Decisions come from the 2026-10-07 design interview; the evidence is in `results/default_prior_followups_seed0/` and `results/bridge_remedies_seed0/`.

## Decisions this plan implements

| Topic | Decision |
| --- | --- |
| Prior | The default prior, `configs/context_prior/default_prior.yaml`: the affine bridge, 128 expression components at penalty 1, 50 data-selected genes at gene penalty 10 (0.2290 validation / 0.2338 test). |
| Stack | `prediction = prior offset + head`, in residual units; the head trains on the full loss of the stack, so for Huber and standardised MSE this is the same as fitting what the prior leaves. |
| Start | `C`'s linear map starts at zero (weight and bias); with the zero-initialised SwiGLU and `h` outputs, the stack equals the prior before the first update. Validation runs then, recorded as epoch −1, and is eligible for `best.pt`. |
| Objective screen | STATE absent: `huber`, `standardized_mse`, `line_ranking` (standardised MSE + ListNet cross-entropy across a gene's lines, temperature 1 in σ units, gene-blocked batches), `dependency_classification` (standardised MSE + binary cross-entropy of GeneEffect < −0.5 on the stack's own output, logit `(−0.5 − μ_g − ŷ)/σ_g`, selective genes). Weights fixed at 1. |
| STATE | The screen's winner with frozen and with trainable STATE (no response replay). |
| Reading | The executor reads each result with the protocol §9.3 rule (highest validation selective Spearman at `best.pt`; within the 27-line paired bootstrap interval the simpler wins: a single-term loss before a two-term one, STATE absent before frozen before trainable) and launches the next runs. At the end of wave one it stops and reports. **If STATE absent wins, nothing else launches until the user has discussed the research plan.** |
| Controls | Every correction summary adds the prior alone and the prior plus the Tx1 context-PCA ridge fitted on what the prior leaves (same view, 8 components, ridge α 1 as the existing Tx1 ridge), with paired bootstraps of the stack minus each and minus the Tx1 ridge, and a per-lineage table. |
| Where | H20 container port 30838 only, runs one after another on its 4 GPUs; the export on its CPUs. |
| In parallel | The single-cell atlas search and the membership file for extra labelled single-cell lines (Task 8). Ingestion is a later, separate step. |

## Global Constraints

- Every Python call: `uv run python -m …` from the repo root; imports are `src.*`.
- Lint: `.venv/bin/ruff check src tests hpc` (the tracked `autoresearch/` scripts already fail `ruff check .` on `main`; do not touch them). Format only touched files: `.venv/bin/ruff format <files>`.
- Suite: `uv run python -m pytest tests -q > /tmp/pytest.txt 2>&1`, then read the tail; never foreground pytest (the `rtk` hook fakes collection errors). Baseline the full suite before the first change.
- `src/experiments/config.py` rejects unknown and missing keys; never add `.get(key, default)` config reads. The new key `paths.prior` goes into every joint YAML (`configs/geneeffect_joint.yaml`, `configs/revision/*.yaml`, `configs/correction/*.yaml`) and into the `make_prepared_fixture` config of `tests/test_joint_data.py`.
- `data` and `model` never import `training`, `eval` or `experiments`; `context_prior` imports none of them either.
- Experiment code reports and decides nothing. Guards against silently wrong artifacts fail closed (gene order, residual scale, missing lines, prior identity); no digests, no quality gates.
- Name things by what they are; no bare labels. One config is one run at seed 0: validation chooses, test is reported for every row.
- Docs that describe changed behaviour change in the same commit.
- Commits: Conventional Commits, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `feat/single-cell-correction` from `main`; merge into `main` once the suite and lint pass and the export plus one correction run have run on the H20 host; then delete the branch.
- H20 (port 30838): `ssh -o BatchMode=yes -J richard@100.91.229.50 -p 30838 root@10.15.171.204`; code arrives as a git bundle; output over ~1 KB drops the session (pull files as tar + base64 in paced 700-byte `dd` chunks, check md5); never `pkill -f` inside ssh; set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` for CPU jobs.

## Review Focus

1. **A prior export from another gene panel or residual scale.** If the export's gene order or residual SD differs from the joint model's, the offsets would be silently shifted. Expect `load_inputs` to raise. Test: `test_load_inputs_refuses_a_prior_with_another_gene_order_or_scale` (Task 2).
2. **A checkpoint evaluated against a different prior than it trained on.** `evaluate_checkpoint` re-reads `paths.prior` from the saved config, and the export at that path may have been replaced. Expect a refusal naming both run ids. Test: `test_restore_refuses_a_checkpoint_trained_on_another_prior` (Task 2).
3. **Training-line prior rows that saw their own labels.** An out-of-fold row fitted with its own label makes the head learn the wrong scale and gives no exception. Expect fold-held labels to leave that fold's rows unchanged. Test: `test_out_of_fold_rows_ignore_their_own_labels` (Task 1).
4. **Pre-update validation without a prior.** A zero model has undefined selective Spearman, and `record_validation` raises on it. Expect pre-update validation only when a prior is set. Test: `test_no_prior_run_has_no_pre_update_validation` (Task 3).
5. **The stack at its start not reproducing the prior's own score.** If the offset is lost in evaluation (`_predict`) or misaligned by row, the "prior alone" row and the stack at epoch −1 disagree. Expect identical validation metrics. Test: `test_stack_before_training_scores_exactly_the_prior` (Task 3).

---

## File structure

| File | Change | Responsibility |
| --- | --- | --- |
| `src/context_prior/folds.py` | modify | `training_side_folds`: folds over the whole training side for cross-fitting the single-cell lines |
| `src/experiments/prior_export.py` | create | Fit the reference row; write `export/prior.npz` and `export/prior.json` |
| `src/data/prior_offsets.py` | create | Read an export; fail-closed checks against the joint model's genes, scale, lines and recorded identity |
| `src/experiments/config.py` | modify | `paths.prior`; two new objectives |
| `src/data/prepared.py` | modify | `PreparedInputs.prior`; read and check in `load_inputs`; identity in `preprocessing_state` |
| `src/data/batches.py`, `src/data/datasets.py` | modify | `DependencyBatch.prior`: each row's prior offset in residual units (zero without a prior) |
| `src/training/trainer.py` | modify | Loss on `delta_hat + prior`; pre-update validation when a prior is set |
| `src/eval/geneeffect.py` | modify | Predictions are `delta_hat + prior` |
| `src/model/head.py` | modify | `C`'s linear map starts at zero |
| `src/model/losses.py` | modify | `line_ranking_loss`, `dependency_loss`, two objectives |
| `src/training/sampling.py` | modify | Gene-blocked batches for `line_ranking` too |
| `src/baselines/prior_controls.py` | create | The prior alone and the prior plus Tx1 ridge, as ladder-shaped prediction rows |
| `src/experiments/baselines.py` | modify | Adds the prior controls when a prior is set |
| `src/experiments/all.py` | modify | Display names of the two controls |
| `src/experiments/revision.py` | modify | Bootstraps against each control; per-lineage table; "before the first update" wording |
| `src/experiments/compare_runs.py` | create | Paired line bootstrap between two runs' predictions (the executor's reading tool) |
| `hpc/run.sh` | modify | `prior-export` mode |
| `configs/correction/*.yaml` | create | The four objective-screen configs (STATE absent) |
| `src/data/prepare/build_extra_single_cell_lines.py`, `configs/benchmarks/single_cell_atlas_candidates.csv`, `configs/benchmarks/extra_single_cell_lines_26Q1.json`, `docs/data/single-cell-atlas-search.md` | create | Atlas search track |
| Docs | modify | Protocol §10 and new §12, design spec §12, `CLAUDE.md`, `hpc/README.md`, `src/README.md` |

---

### Task 1: Export the default prior for the correction

**Files:**
- Modify: `src/context_prior/folds.py`
- Create: `src/experiments/prior_export.py`
- Modify: `hpc/run.sh`
- Test: `tests/test_prior_export.py`

**Interfaces:**
- Consumes: `src.experiments.context_prior.{RunBase, load_base, _build, prior_inputs, stages, score, _bind, _jsonable, _write_json, SCORED}`; `src.context_prior.prior.{PriorSpec, fit_prior, crossfit, total}`; `src.context_prior.bridging.oof_bridged`.
- Produces:
  - `training_side_folds(single_cell: Mapping[str, int], lines: Sequence[str], patients: Mapping[str, object]) -> dict[str, int]`;
  - `export_predictions(base: RunBase) -> dict[str, pd.DataFrame]` with keys `train`, `val`, `test` (lines × `base.definitions.genes`, residual-SD units);
  - `write_export(run_dir: Path, base: RunBase, run_id: str) -> Path` writing `export/prior.npz` (`values` float32 [lines, genes], `lines`, `genes`, `residual_scale` float64) and `export/prior.json` (`run_id`, `reference`, `units`, `lines` per split, `scores` per split);
  - CLI `python -m src.experiments.prior_export CONFIG --run-id ID`.

- [ ] **Step 1: Write the failing tests**

```python
"""The prior export: reference row on validation and test, out of fold on training."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from src.context_prior.folds import training_side_folds
from src.experiments import context_prior as run
from src.experiments import prior_export as export
from src.experiments.config import validate_prior_config
from tests.test_context_prior_run import synthetic_base, tiny_config


def selected_config(tmp_path):
    config = tiny_config(tmp_path)
    config["reference"] = {
        "experiment": "affine",
        "block_set": "selected",
        "components_penalty": 1.0,
        "gene_penalty": 1.0,
    }
    return validate_prior_config(config)


def test_training_side_folds_follow_patients():
    folds = training_side_folds(
        {"S0": 0, "S1": 1},
        ["S0", "S1", "E0", "E1", "E2"],
        {"S0": "P0", "S1": "P1", "E0": "P1", "E1": "P9", "E2": float("nan")},
    )
    assert folds == {"S0": 0, "S1": 1, "E0": 1, "E1": -1, "E2": -1}


def test_validation_and_test_rows_are_the_reference_row(tmp_path):
    base = synthetic_base(selected_config(tmp_path))
    predictions = export.export_predictions(base)
    reference = run.reference_predictions(base)
    for split in ("val", "test"):
        np.testing.assert_allclose(predictions[split], reference[split])
    assert list(predictions["train"].index) == list(base.bridge.single_cell_train)
    assert list(predictions["train"].columns) == list(base.definitions.genes)


def test_out_of_fold_rows_ignore_their_own_labels(tmp_path):
    base = synthetic_base(selected_config(tmp_path))
    before = export.export_predictions(base)
    held = [m for m, fold in base.bridge.folds.items() if fold == 0]
    base.gene_effect.loc[held] = base.gene_effect.loc[held] + 3.0
    after = export.export_predictions(base)
    np.testing.assert_allclose(after["train"].loc[held], before["train"].loc[held])
    others = [m for m in base.bridge.single_cell_train if m not in held]
    assert not np.allclose(after["train"].loc[others], before["train"].loc[others])
    assert not np.allclose(after["val"], before["val"])


def test_export_refuses_a_bridge_it_cannot_refit(tmp_path):
    config = selected_config(tmp_path)
    config["experiments"]["affine"]["kind"] = "gating"
    with pytest.raises(ValueError, match="affine bridge only"):
        export.export_predictions(synthetic_base(config))


def test_main_writes_the_export_once(tmp_path, monkeypatch):
    config = selected_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(export, "load_base", synthetic_base)
    assert export.main([str(path), "--run-id", "r"]) == 0
    out = tmp_path / "out" / "r" / "export"
    record = json.loads((out / "prior.json").read_text())
    assert record["run_id"] == "r"
    assert record["reference"]["block_set"] == "selected"
    assert record["units"] == "residual SD"
    with np.load(out / "prior.npz") as payload:
        lines = [str(m) for m in payload["lines"]]
        assert payload["values"].shape == (len(lines), len(payload["genes"]))
        assert payload["values"].dtype == np.float32
    assert lines == [*record["lines"]["train"], *record["lines"]["val"],
                     *record["lines"]["test"]]
    assert set(record["scores"]) == {"train", "val", "test"}
    with pytest.raises(FileExistsError, match="new --run-id"):
        export.main([str(path), "--run-id", "r"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/test_prior_export.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: FAIL, `ImportError: cannot import name 'training_side_folds'`.

- [ ] **Step 3: Add `training_side_folds` to `src/context_prior/folds.py`**

```python
def training_side_folds(
    single_cell: Mapping[str, int],
    lines: Sequence[str],
    patients: Mapping[str, object],
) -> dict[str, int]:
    """Folds over the whole training side for cross-fitting the single-cell lines.

    A single-cell training line keeps its fold; any other line takes the fold of
    the single-cell training lines that share its patient, and -1 when none does
    (or its patient is unknown), so cross-fitting always fits on it and never
    holds it out.
    """
    by_patient = {
        patients[m]: fold for m, fold in single_cell.items() if isinstance(patients[m], str)
    }
    folds = dict(single_cell)
    for model_id in lines:
        if model_id not in folds:
            patient = patients[model_id]
            folds[model_id] = by_patient.get(patient, -1) if isinstance(patient, str) else -1
    return folds
```

(`by_patient.get` is a data lookup with an explicit "no such patient" meaning, not a config read.)

- [ ] **Step 4: Create `src/experiments/prior_export.py`**

```python
"""Export the reference row of a prior config for the single-cell correction.

``python -m src.experiments.prior_export CONFIG --run-id ID`` writes
``<output_root>/<run id>/export/{prior.npz, prior.json}``: the prior's residual
prediction in residual-SD units for every labelled single-cell training line out of
fold (the bridge and every stage refitted without the line's fold; extras sharing a
patient with a line of the fold are held with it) and for the validation and test
lines from the fit on every labelled training-side line. ``prior.json`` records the
run id and reference row (the identity a correction checkpoint records) and the
export's own selective-Spearman scores. Nothing is chosen here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.context_prior.bridging import oof_bridged
from src.context_prior.folds import training_side_folds
from src.context_prior.prior import PriorSpec, crossfit, fit_prior, total
from src.experiments.config import load_prior_config
from src.experiments.context_prior import (
    SCORED,
    RunBase,
    _bind,
    _build,
    _jsonable,
    _write_json,
    load_base,
    prior_inputs,
    score,
    stages,
)

EXPORT_DIR = "export"
SPLITS = ("train", *SCORED)


def export_predictions(base: RunBase) -> dict[str, pd.DataFrame]:
    """The reference row's predictions in residual-SD units: ``train`` out of fold
    for every labelled single-cell training line, ``val`` and ``test`` from the fit
    on every labelled training-side line."""
    config = base.config
    reference = config["reference"]
    experiment = config["experiments"][reference["experiment"]]
    if experiment["kind"] != "affine":
        raise ValueError(
            "the export refits the bridge out of fold for the affine bridge only, "
            f"not {experiment['kind']!r}"
        )
    inputs = _build(base, "affine", experiment["settings"][0])
    prior = prior_inputs(base, inputs)
    spec = PriorSpec(
        tuple(
            stages(
                config,
                reference["block_set"],
                reference["components_penalty"],
                reference["gene_penalty"],
            )
        )
    )
    encoders = list(prior.expression.index)
    full = fit_prior(spec, prior, fit_lines=list(base.labelled), encoder_lines=encoders)
    bridge = base.bridge
    train = list(bridge.single_cell_train)
    folds = training_side_folds(
        bridge.folds, encoders, base.models["patient_id"].to_dict()
    )
    rows = oof_bridged(
        bridge.pseudobulk, bridge.bulk, list(bridge.paired), train, bridge.folds
    )
    queries = {
        fold: rows.loc[[m for m in train if bridge.folds[m] == fold]]
        for fold in sorted(set(bridge.folds.values()))
    }
    held = total(
        crossfit(
            spec,
            prior,
            folds=folds,
            queries=queries,
            labelled=list(base.labelled),
            encoder_lines=encoders,
        )
    )
    return {
        "train": held.loc[train],
        **{name: total(full.predict(inputs.queries[name])) for name in SCORED},
    }


def write_export(run_dir: Path, base: RunBase, run_id: str) -> Path:
    """Write ``export/`` once; an existing export is never replaced, because a
    correction checkpoint may have recorded it."""
    out = Path(run_dir) / EXPORT_DIR
    if out.exists():
        raise FileExistsError(
            f"{out} exists and a correction may have recorded it; use a new --run-id"
        )
    predictions = export_predictions(base)
    genes = list(base.definitions.genes)
    frame = pd.concat([predictions[split] for split in SPLITS]).loc[:, genes]
    scores = {
        split: score(
            predictions[split],
            base.truth(list(predictions[split].index)),
            base.definitions,
        )
        for split in SPLITS
    }
    staging = out.with_name(f".{EXPORT_DIR}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    np.savez(
        staging / "prior.npz",
        values=frame.to_numpy(dtype=np.float32),
        lines=np.asarray(frame.index, dtype=str),
        genes=np.asarray(genes, dtype=str),
        residual_scale=base.definitions.residual_scale.loc[genes].to_numpy(np.float64),
    )
    _write_json(
        staging / "prior.json",
        _jsonable(
            {
                "run_id": run_id,
                "reference": dict(base.config["reference"]),
                "units": "residual SD",
                "lines": {split: list(predictions[split].index) for split in SPLITS},
                "scores": scores,
            }
        ),
    )
    os.replace(staging, out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    config = load_prior_config(args.config)
    run_dir = Path(config["output_root"]) / args.run_id
    if (run_dir / EXPORT_DIR).exists():
        raise FileExistsError(
            f"{run_dir / EXPORT_DIR} exists and a correction may have recorded it; "
            "use a new --run-id"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _bind(run_dir, config)
    out = write_export(run_dir, load_base(config), args.run_id)
    print(f"prior export: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add the `prior-export` mode to `hpc/run.sh`**

In the usage text, after the `prior-selected` line:

```
       hpc/run.sh prior-export CONFIG --run-id ID   (the reference prior's predictions for the correction, CPU)
```

and after the `prior-selected` paragraph:

```
`prior-export` writes the config's reference row, out of fold for the
single-cell training lines, into outputs/context_prior/<id>/export/.
```

Change the command check to `case "$command" in all|revision|test|prior|prior-selected|prior-export) ;;` and add the dispatch line:

```bash
  prior-export) exec "$python_bin" -m src.experiments.prior_export "$@" ;;
```

- [ ] **Step 6: Run the tests**

Run: `uv run python -m pytest tests/test_prior_export.py tests/test_context_prior_run.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: all pass. If `test_out_of_fold_rows_ignore_their_own_labels` fails on the training rows, the fold of an extra was not carried. Check `training_side_folds` against `synthetic_base`'s patients, which equal the line ids.

- [ ] **Step 7: Lint, format, commit**

```bash
.venv/bin/ruff format src/context_prior/folds.py src/experiments/prior_export.py tests/test_prior_export.py
.venv/bin/ruff check src tests hpc
git add src/context_prior/folds.py src/experiments/prior_export.py tests/test_prior_export.py hpc/run.sh
git commit -m "feat(context-prior): export the reference prior out of fold for the single-cell correction" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The prior offset in the joint data path

**Files:**
- Create: `src/data/prior_offsets.py`
- Modify: `src/experiments/config.py`, `src/data/prepared.py`, `src/data/batches.py`, `src/data/datasets.py`, `src/training/trainer.py:116-136`, `src/eval/geneeffect.py:201-227`
- Modify: `configs/geneeffect_joint.yaml`, `configs/revision/frozen_huber.yaml`, `configs/revision/frozen_standardized_mse.yaml`, `configs/revision/frozen_pearson_blocks.yaml` (add `prior: null` under `paths`)
- Modify: `tests/test_joint_data.py` (fixture `paths` gains `"prior": None`)
- Test: `tests/test_prior_offsets.py`

**Interfaces:**
- Consumes: the export layout of Task 1.
- Produces:
  - `PriorOffsets(identity: dict, values: pd.DataFrame, residual_scale: pd.Series)`;
  - `read_prior_offsets(path: Path) -> PriorOffsets`;
  - `checked_prior(prior, *, genes, residual_scale, lines) -> PriorOffsets`;
  - `check_prior_identity(recorded: dict | None, prior: PriorOffsets | None) -> None`;
  - `PreparedInputs.prior: PriorOffsets | None` (last field, default `None`);
  - `preprocessing_state()["prior"]` (the identity, or `None`);
  - `DependencyDataset.prior` and `DependencyBatch.prior` (float32 per row, residual units, zeros without a prior);
  - `geneeffect_loss(..., gene_mean=...)` is called with the stack prediction.

- [ ] **Step 1: Write the failing tests** (`tests/test_prior_offsets.py`)

```python
"""The prior offset: export checks, identity, and the stack in loss and evaluation."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datasets import DependencyDataset
from src.data.prepared import load_inputs
from src.data.prior_offsets import (
    PriorOffsets,
    check_prior_identity,
    checked_prior,
    read_prior_offsets,
)
from tests.test_joint import GENES, TRAIN, VAL, make_inputs
from tests.test_joint_data import make_prepared_fixture


def write_export(path, lines, genes, scale, *, run_id="prior_run", value=0.5):
    path.mkdir(parents=True)
    np.savez(
        path / "prior.npz",
        values=np.full((len(lines), len(genes)), value, dtype=np.float32),
        lines=np.asarray(lines, dtype=str),
        genes=np.asarray(genes, dtype=str),
        residual_scale=np.asarray(scale, dtype=np.float64),
    )
    (path / "prior.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "reference": {"experiment": "affine", "block_set": "selected"},
                "units": "residual SD",
                "lines": {"train": list(lines), "val": [], "test": []},
                "scores": {},
            }
        )
    )
    return path


def with_prior(inputs, per_line: dict[str, float]):
    frame = pd.DataFrame(
        {gene: pd.Series(per_line) for gene in inputs.genes}
    ).loc[:, list(inputs.genes)]
    prior = PriorOffsets({"run_id": "r", "reference": {}}, frame, inputs.residual_scale)
    return dataclasses.replace(inputs, prior=prior)


def test_read_and_check_an_export(tmp_path):
    inputs = make_inputs()
    lines = [*TRAIN, *VAL]
    path = write_export(tmp_path / "export", lines, GENES, inputs.residual_scale)
    prior = read_prior_offsets(path)
    assert prior.identity == {
        "run_id": "prior_run",
        "reference": {"experiment": "affine", "block_set": "selected"},
    }
    checked_prior(prior, genes=GENES, residual_scale=inputs.residual_scale, lines=lines)
    with pytest.raises(ValueError, match="lacks lines"):
        checked_prior(
            prior, genes=GENES, residual_scale=inputs.residual_scale, lines=["ACH-X0"]
        )


def test_load_inputs_refuses_a_prior_with_another_gene_order_or_scale(tmp_path):
    inputs = make_inputs()
    lines = [*TRAIN, *VAL]
    reordered = write_export(
        tmp_path / "a", lines, GENES[::-1], inputs.residual_scale.loc[list(GENES[::-1])]
    )
    with pytest.raises(ValueError, match="gene order"):
        checked_prior(
            read_prior_offsets(reordered),
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=lines,
        )
    rescaled = write_export(tmp_path / "b", lines, GENES, inputs.residual_scale * 1.01)
    with pytest.raises(ValueError, match="residual scale"):
        checked_prior(
            read_prior_offsets(rescaled),
            genes=GENES,
            residual_scale=inputs.residual_scale,
            lines=lines,
        )


def test_restore_refuses_a_checkpoint_trained_on_another_prior():
    recorded = {"run_id": "old", "reference": {}}
    current = PriorOffsets({"run_id": "new", "reference": {}}, pd.DataFrame(), pd.Series())
    with pytest.raises(ValueError, match="old.*new"):
        check_prior_identity(recorded, current)
    with pytest.raises(ValueError, match="without a prior"):
        check_prior_identity(None, current)
    with pytest.raises(ValueError, match="no prior is configured"):
        check_prior_identity(recorded, None)
    check_prior_identity(None, None)
    check_prior_identity(dict(current.identity), current)


def test_load_inputs_attaches_the_export_and_records_its_identity(tmp_path):
    config = make_prepared_fixture(tmp_path)
    fitted = load_inputs(config)
    assert fitted.prior is None and fitted.preprocessing_state()["prior"] is None
    lines = [*fitted.split.supervised_train, *fitted.split.val]
    config["paths"]["prior"] = str(
        write_export(tmp_path / "export", lines, fitted.genes, fitted.residual_scale)
    )
    attached = load_inputs(config)
    assert attached.prior.identity["run_id"] == "prior_run"
    state = attached.preprocessing_state()
    assert state["prior"] == attached.prior.identity
    load_inputs(config, preprocessing=state)  # same prior: accepted
    with pytest.raises(ValueError, match="lacks lines"):
        load_inputs(config, include_test=True)  # the export has no test lines


def test_dataset_rows_carry_the_offset_in_residual_units():
    inputs = with_prior(make_inputs(), {m: 0.25 for m in (*TRAIN, *VAL)})
    dataset = DependencyDataset(inputs, "val")
    expected = 0.25 * inputs.residual_scale.loc[dataset.genes].to_numpy()
    np.testing.assert_allclose(dataset.prior.numpy(), expected, rtol=1e-6)
    batch = dataset.collate(range(4))
    np.testing.assert_allclose(batch.prior.numpy(), expected[:4], rtol=1e-6)
    plain = DependencyDataset(make_inputs(), "val")
    assert torch.equal(plain.prior, torch.zeros(len(plain)))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/test_prior_offsets.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.data.prior_offsets'`.

- [ ] **Step 3: Create `src/data/prior_offsets.py`**

```python
"""The linear context prior's exported predictions, as the joint model reads them.

An export (``src.experiments.prior_export``) holds per-line residual predictions in
residual-SD units. The joint model adds ``value x residual SD`` to its head's output,
so the export must be on the same gene order and residual SD; ``checked_prior``
refuses one that is not, and ``check_prior_identity`` refuses a checkpoint trained
on another export.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriorOffsets:
    """A prior export.

    Attributes:
        identity: The export's run id and reference row; checkpoints record it.
        values: Lines x genes, residual-SD units.
        residual_scale: The per-gene residual SD the export was fitted with.
    """

    identity: dict[str, Any]
    values: pd.DataFrame
    residual_scale: pd.Series


def read_prior_offsets(path: Path) -> PriorOffsets:
    """Read ``prior.npz`` and ``prior.json`` from an export directory."""
    path = Path(path)
    record = json.loads((path / "prior.json").read_text())
    with np.load(path / "prior.npz") as payload:
        genes = [str(gene) for gene in payload["genes"]]
        values = pd.DataFrame(
            payload["values"],
            index=[str(model_id) for model_id in payload["lines"]],
            columns=genes,
        )
        scale = pd.Series(payload["residual_scale"], index=genes)
    return PriorOffsets(
        {"run_id": record["run_id"], "reference": record["reference"]}, values, scale
    )


def checked_prior(
    prior: PriorOffsets,
    *,
    genes: Sequence[str],
    residual_scale: pd.Series,
    lines: Collection[str],
) -> PriorOffsets:
    """The export, if it matches the joint model's genes, residual SD and lines."""
    if tuple(prior.values.columns) != tuple(genes):
        raise ValueError("the prior export's gene order differs from the gene panel")
    if not np.allclose(
        prior.residual_scale.to_numpy(),
        residual_scale.loc[list(genes)].to_numpy(),
        rtol=1e-6,
        atol=0.0,
    ):
        raise ValueError(
            "the prior export's residual scale differs from the joint model's"
        )
    missing = sorted(set(lines) - set(prior.values.index))
    if missing:
        raise ValueError(f"the prior export lacks lines {missing[:10]}")
    return prior


def check_prior_identity(
    recorded: Mapping[str, Any] | None, prior: PriorOffsets | None
) -> None:
    """Refuse a checkpoint whose recorded prior is not the configured one."""
    if recorded is None and prior is None:
        return
    if recorded is None:
        raise ValueError(
            "the checkpoint was trained without a prior but the config names "
            f"prior run {prior.identity['run_id']}"
        )
    if prior is None:
        raise ValueError(
            f"the checkpoint was trained on prior run {recorded['run_id']} but no "
            "prior is configured"
        )
    if dict(recorded) != prior.identity:
        raise ValueError(
            f"the checkpoint was trained on prior run {recorded['run_id']}, the "
            f"config's export is prior run {prior.identity['run_id']}"
        )
```

- [ ] **Step 4: Add `prior` to the joint config schema**

In `src/experiments/config.py`, extend the `paths` group:

```python
    "paths": (
        "split gene_effect source_registry tx1_registration cell_line_manifest "
        "tx1_model_dir tx1_cache esm2_embeddings state_checkpoint state_model_dir "
        "perturbseq_sources prior"
    ),
```

Add `  prior: null` as the last `paths` entry of `configs/geneeffect_joint.yaml` and the three `configs/revision/*.yaml`, with this comment line above it in each:

```yaml
  # The linear context prior export the head stacks on (prior-export), or null.
```

In `tests/test_joint_data.py`, add `"prior": None,` to the fixture's `"paths"` dict.

- [ ] **Step 5: Read and check the export in `load_inputs` (`src/data/prepared.py`)**

Add the import `from src.data.prior_offsets import PriorOffsets, check_prior_identity, checked_prior, read_prior_offsets`. Add the field as the last field of `PreparedInputs`:

```python
    prior: PriorOffsets | None = field(default=None, repr=False, compare=False)
```

and to the docstring: "``prior`` is the linear context prior export the head stacks on, or None." In `preprocessing_state`, add the entry:

```python
            "prior": None if self.prior is None else dict(self.prior.identity),
```

In `_restored_keys`, add `"prior"` to the required keys:

```python
        for key in ("selective_genes", "residual_scale", "context_pca", "prior")
```

In `load_inputs`, after the `exposed` set is computed and before `labels` is filtered:

```python
    prior = None
    if config["paths"]["prior"] is not None:
        prior = checked_prior(
            read_prior_offsets(Path(config["paths"]["prior"])),
            genes=genes,
            residual_scale=residual_scale,
            lines=exposed,
        )
    if preprocessing is not None:
        check_prior_identity(preprocessing["prior"], prior)
```

and pass `prior=prior` to the `PreparedInputs(...)` constructor. Update the docstring: "``config["paths"]["prior"]`` names a prior export; it must match the gene panel, the residual SD and every exposed line, and a restored checkpoint must have been trained on it."

- [ ] **Step 6: Carry the offset in the dataset and batch**

In `src/data/batches.py`, add the field to `DependencyBatch` after `selective` and document it ("``prior`` is each row's prior prediction in residual units, zero without a prior; the stack predicts ``prior`` plus the head's output"):

```python
    prior: torch.Tensor
```

and in `DependencyBatch.to` add `prior=self.prior.to(device),`.

In `src/data/datasets.py`, after `self.selective = ...`:

```python
        if inputs.prior is None:
            prior = np.zeros(len(self.rows))
        else:
            table = inputs.prior.values
            prior = table.to_numpy()[
                table.index.get_indexer(self.model_ids),
                table.columns.get_indexer(self.genes),
            ] * inputs.residual_scale.loc[self.genes].to_numpy()
        self.prior = on_device(prior, torch.float32)
```

and in `collate` pass `prior=self.prior[rows],` to `DependencyBatch(...)`. Add to the class docstring: "Each row also carries its prior offset (``prior``), zero without a prior."

- [ ] **Step 7: Score the stack in training and evaluation**

In `src/training/trainer.py` `train_update`, replace the loss call:

```python
    # The stack: the prior's offset plus the head's output, in residual units.
    prediction = output.delta_hat + dependency_batch.prior
    dependency_loss = geneeffect_loss(
        prediction,
        dependency_batch.residual,
        dependency_batch.residual_scale,
        objective=config["train"]["objective"],
        gene_index=dependency_batch.conditions.gene_index,
        selective=dependency_batch.selective,
    )
```

(Task 4 adds `gene_mean=dependency_batch.gene_mean` to this call.)

In `src/eval/geneeffect.py` `_predict`:

```python
            predicted = model(batch.conditions).delta_hat.float() + batch.prior
```

and its docstring: "float32 residual predictions of the stack (prior offset plus head)".

- [ ] **Step 8: Run the tests**

Run: `uv run python -m pytest tests/test_prior_offsets.py tests/test_joint_data.py tests/test_joint.py tests/test_objectives.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: all pass.

- [ ] **Step 9: Run the full suite, lint and commit**

```bash
uv run python -m pytest tests -q > /tmp/pytest.txt 2>&1; tail -3 /tmp/pytest.txt
.venv/bin/ruff format src/data/prior_offsets.py src/data/prepared.py src/data/batches.py src/data/datasets.py src/training/trainer.py src/eval/geneeffect.py src/experiments/config.py tests/test_prior_offsets.py tests/test_joint_data.py
.venv/bin/ruff check src tests hpc
git add -A src/data src/training/trainer.py src/eval/geneeffect.py src/experiments/config.py configs/geneeffect_joint.yaml configs/revision tests
git commit -m "feat(correction): stack the joint head on a prior export, checked fail-closed" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The stack starts at the prior, and the prior alone can be `best.pt`

**Files:**
- Modify: `src/model/head.py:400-430` (zero `C`'s linear map), its class docstring
- Modify: `src/training/trainer.py:233-297` (pre-update validation)
- Modify: `src/experiments/revision.py:181-199`, `:258-283` (wording for epoch −1)
- Modify: `tests/test_geneeffect_head.py:136-153`
- Test: `tests/test_correction_start.py`

**Interfaces:**
- Consumes: `PreparedInputs.prior` and `DependencyBatch.prior` (Task 2).
- Produces:
  - `fit` runs validation before the first update when `inputs.prior is not None` and the run is fresh, recorded as epoch −1;
  - `train/metrics.jsonl` gains a record with `"epoch": -1`;
  - `done.json`'s `best_epoch` may be −1;
  - `revision.json`'s `training.best_epoch` is then 0, and the summary says "before the first update (the prior alone)".

- [ ] **Step 1: Write the failing tests** (`tests/test_correction_start.py`)

```python
"""The stack equals its prior before the first update, which is a best.pt candidate."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

from src.eval.geneeffect import aggregate_geneeffect, evaluate_model  # noqa: E402
from src.experiments import geneeffect  # noqa: E402
from src.model.initialization import build_joint_model  # noqa: E402
from tests.test_joint import (  # noqa: E402, F401
    TEST,
    TRAIN,
    VAL,
    cpu,
    make_config,
    make_inputs,
)
from tests.test_prior_offsets import with_prior  # noqa: E402

PRIOR = {m: float(v) for m, v in zip((*TRAIN, *VAL, *TEST), np.linspace(-1, 1, 10))}


def test_context_linear_map_starts_at_zero(cpu, tmp_path):
    config = make_config(tmp_path, state=False)
    inputs = make_inputs()
    model = build_joint_model(config, inputs)
    head = model.head
    assert torch.count_nonzero(head.context_linear.weight) == 0
    assert torch.count_nonzero(head.context_linear.bias) == 0


def test_stack_before_training_scores_exactly_the_prior(cpu, tmp_path):
    config = make_config(tmp_path, state=False)
    inputs = with_prior(make_inputs(), PRIOR)
    model = build_joint_model(config, inputs)
    from src.model.normalization import fit_startup_standardizer

    fit_startup_standardizer(model, inputs, batch_size=8)
    result = evaluate_model(model, inputs, config, split="val")
    rows = inputs.labels.loc[inputs.labels.model_id.isin(VAL)].copy()
    scale = rows.gene_symbol.map(inputs.residual_scale)
    rows["residual_prediction"] = rows.model_id.map(PRIOR) * scale
    rows["geneeffect_prediction"] = rows.residual_prediction + rows.gene_symbol.map(
        inputs.train_gene_means
    )
    expected, _, _ = aggregate_geneeffect(
        rows,
        model_ids=VAL,
        genes=inputs.genes,
        variable_genes=list(inputs.genes),
        selective_genes=sorted(inputs.selective_genes),
    )
    assert result.metrics["val_selective_spearman"] == pytest.approx(
        expected["selective_spearman"], abs=1e-6
    )


def test_pre_update_validation_runs_with_a_prior(cpu, tmp_path):
    config = make_config(tmp_path, state=False, max_epochs=1)
    inputs = with_prior(make_inputs(), PRIOR)
    geneeffect.run_training(config, tmp_path / "train", inputs=inputs)
    records = [
        json.loads(line)
        for line in (tmp_path / "train" / "metrics.jsonl").read_text().splitlines()
    ]
    epochs = [r["epoch"] for r in records if "val_selective_spearman" in r]
    assert epochs == [-1, 0]
    assert (tmp_path / "train" / "best.pt").is_file()


def test_no_prior_run_has_no_pre_update_validation(cpu, tmp_path):
    config = make_config(tmp_path, state=False, max_epochs=1)
    geneeffect.run_training(config, tmp_path / "train", inputs=make_inputs())
    records = [
        json.loads(line)
        for line in (tmp_path / "train" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["epoch"] for r in records if "val_selective_spearman" in r] == [0]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/test_correction_start.py -q > /tmp/pytest.txt 2>&1; tail -8 /tmp/pytest.txt`
Expected: FAIL. `test_context_linear_map_starts_at_zero` finds a non-zero weight; `test_pre_update_validation_runs_with_a_prior` gets epochs `[0]`.

- [ ] **Step 3: Zero `C`'s linear map in `GeneEffectNestedHead.__init__`**

```python
        self.context_linear = nn.Linear(dims.z_c, self.factor_rank)
        # C starts at zero: with h's zero output layer the head predicts zero before
        # the first update, so a stack equals its prior there.
        nn.init.zeros_(self.context_linear.weight)
        nn.init.zeros_(self.context_linear.bias)
```

In the class docstring, replace "The SwiGLU branch's output layer is zero-initialised, so training starts at a reduced-rank context ridge." with: "``W`` and the SwiGLU branch's output layer start at zero, so the head predicts zero before the first update (its gradient reaches ``W`` first through ``G``)."

Update `tests/test_geneeffect_head.py::test_zero_initialised_branches_leave_the_reduced_rank_ridge`: rename it `test_head_predicts_zero_before_the_first_update`, and replace its final assertion with:

```python
        torch.testing.assert_close(head(**inputs), torch.zeros(6), rtol=0, atol=0)
        assert torch.equal(head.context_linear(z), torch.zeros(6, RANK))
```

Also remove the now-unused `expected` computation. Check that `_live` randomises `context_linear` too: it must re-initialise every zero-initialised layer so that `test_head_is_gene_context_product_plus_correction` stays meaningful. If it only touches the SwiGLU outputs, add `nn.init.normal_(head.context_linear.weight, std=0.1)` there.

- [ ] **Step 4: Validate before the first update in `fit`**

Refactor the end-of-epoch block of `fit` (`src/training/trainer.py`, the code after the batch loop) into a closure defined after `diagnostic_lines`, and call it before the loop when a fresh run has a prior:

```python
    def close_epoch(epoch: int) -> None:
        """Score, record and checkpoint; epoch -1 is the model before any update."""
        train_metrics = evaluate_model(
            model,
            inputs,
            config,
            split="train",
            accelerator=accelerator,
            lines=diagnostic_lines,
        ).metrics
        validation = evaluate_model(
            model, inputs, config, split="val", accelerator=accelerator
        ).metrics
        improved = record_validation(state, validation, epoch)
        _log(
            run_dir,
            {
                "epoch": epoch,
                "global_step": state.global_step,
                **train_metrics,
                **validation,
            },
            accelerator,
        )
        for name, write in (("best.pt", improved), ("last.pt", True)):
            if write:
                save_checkpoint(
                    run_dir / name,
                    model,
                    optimizer,
                    scheduler,
                    state,
                    config,
                    preprocessing,
                    accelerator,
                )

    if restored is None and inputs.prior is not None:
        # The head's output layers start at zero, so the model is its prior here:
        # "the prior alone" is scored and eligible for best.pt.
        close_epoch(-1)
    for epoch in range(state.next_epoch, train["max_epochs"]):
        if state.bad_epochs >= train["patience"]:
            break
        model.train()
        if not trains_state(network, config):
            network.backbone.state.eval()  # a frozen STATE runs without dropout
        loader = dependency_loader(dependency, config, epoch, accelerator)
        responses = response_stream(inputs, config, epoch, accelerator)
        for batch in loader:
            replay = (
                responses is not None
                and state.global_step % train["response_interval"] == 0
            )
            metrics = train_update(
                model,
                optimizer,
                scheduler,
                batch,
                next(responses) if replay else None,
                config,
                accelerator,
            )
            state.global_step += 1
            _log(
                run_dir,
                {"epoch": epoch, "global_step": state.global_step, **metrics},
                accelerator,
            )
        close_epoch(epoch)
    return state
```

Update the module docstring: "Validation runs once per epoch and, when a prior is set, once before the first update (epoch −1), so the prior alone can be ``best.pt``." The `-1` default of `TrainState.best_epoch` is never read before `record_validation` has run, so it stays.

- [ ] **Step 5: Report epoch −1 in the revision summary**

In `src/experiments/revision.py` `_summary_lines`, replace the training sentence with:

```python
    when = (
        "the model before its first update (the prior alone)"
        if training["best_epoch"] == 0
        else f"epoch {training['best_epoch']}"
    )
    return lines + [
        "## Training",
        "",
        f"`train/best.pt` is {when}; {training['epochs_trained']} epochs trained. "
        f"At that point the selective Spearman is "
        f"{_number(training[TRAIN_DIAGNOSTIC_KEY])} on the training diagnostic "
        f"lines and {_number(training[VALIDATION_KEY])} on the validation lines.",
    ]
```

Update the `_training_record` docstring: "Best epoch (1-based; 0 is the pre-update validation of a stack)."

- [ ] **Step 6: Run the tests**

Run: `uv run python -m pytest tests/test_correction_start.py tests/test_geneeffect_head.py tests/test_joint.py tests/test_revision.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: all pass. A resume test in `tests/test_joint.py` (`test_resume_continues_from_last`) uses no prior, so its epochs are unchanged.

- [ ] **Step 7: Commit**

```bash
.venv/bin/ruff format src/model/head.py src/training/trainer.py src/experiments/revision.py tests/test_correction_start.py tests/test_geneeffect_head.py
.venv/bin/ruff check src tests hpc
git add src/model/head.py src/training/trainer.py src/experiments/revision.py tests/test_correction_start.py tests/test_geneeffect_head.py
git commit -m "feat(correction): the stack starts at its prior and validates before the first update" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Line-ranking and dependency objectives

**Files:**
- Modify: `src/model/losses.py`, `src/experiments/config.py:42`, `src/training/sampling.py:96-130`
- Test: `tests/test_objectives.py`

**Interfaces:**
- Consumes: `DEPENDENCY_THRESHOLD` from `src.data.geneeffect` (−0.5).
- Produces:
  - `OBJECTIVES = ("huber", "standardized_mse", "pearson_blocks", "line_ranking", "dependency_classification")`;
  - `line_ranking_loss(prediction, target, gene_index, selective) -> Tensor`;
  - `dependency_loss(prediction, target, scale, gene_mean, selective) -> Tensor`;
  - `geneeffect_loss(prediction, target, scale, *, objective, gene_index, selective, gene_mean)`;
  - `GENE_BLOCK_OBJECTIVES = ("pearson_blocks", "line_ranking")` in `src.training.sampling`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_objectives.py`; extend the `loss` helper with `gene_mean=torch.zeros(len(prediction))`)

```python
from src.model.losses import dependency_loss, line_ranking_loss  # add to the imports


def test_line_ranking_is_listnet_cross_entropy_over_a_genes_lines():
    target = torch.tensor([-2.0, 0.0, 1.0, 0.5, 0.0, -0.5])
    prediction = torch.tensor([-1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    gene_index = torch.tensor([3, 3, 3, 8, 8, 8])
    selective = torch.ones(6, dtype=torch.bool)
    value = line_ranking_loss(prediction, target, gene_index, selective)
    expected = []
    for rows in (slice(0, 3), slice(3, 6)):
        p = torch.softmax(-target[rows], 0)
        log_q = torch.log_softmax(-prediction[rows], 0)
        expected.append(-(p * log_q).sum())
    assert value.item() == pytest.approx(torch.stack(expected).mean().item(), rel=1e-6)


def test_line_ranking_is_smallest_at_the_target_and_ignores_constant_genes():
    target = torch.tensor([-2.0, 0.0, 1.0, 0.4, 0.4, 0.4])
    gene_index = torch.tensor([0, 0, 0, 1, 1, 1])
    selective = torch.ones(6, dtype=torch.bool)
    exact = line_ranking_loss(target + 3.0, target, gene_index, selective)
    worse = line_ranking_loss(-target, target, gene_index, selective)
    assert exact < worse
    alone = line_ranking_loss(target[:3], target[:3], gene_index[:3], selective[:3])
    assert exact.item() == pytest.approx(alone.item(), rel=1e-6)


def test_dependency_loss_reads_the_measured_threshold():
    gene_mean = torch.tensor([-0.4, -0.4, 0.0])
    target = torch.tensor([-0.3, 0.2, 0.1])  # GeneEffect -0.7, -0.2, 0.1
    prediction = torch.tensor([-0.3, 0.2, 0.1])
    scale = torch.tensor([0.5, 0.5, 1.0])
    selective = torch.tensor([True, True, False])
    value = dependency_loss(prediction, target, scale, gene_mean, selective)
    logit = (-0.5 - gene_mean[:2] - prediction[:2]) / scale[:2]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, torch.tensor([1.0, 0.0])
    )
    assert value.item() == pytest.approx(expected.item(), rel=1e-6)
    none = dependency_loss(prediction, target, scale, gene_mean, torch.zeros(3, dtype=torch.bool))
    assert none.item() == 0.0


def test_two_term_objectives_add_to_standardized_mse():
    prediction = torch.tensor([0.1, -0.3, 0.2, 0.0])
    target = torch.tensor([0.0, -0.5, 0.4, 0.1])
    scale = torch.tensor([0.5, 0.5, 0.5, 0.5])
    gene_index = torch.tensor([0, 0, 0, 0])
    selective = torch.ones(4, dtype=torch.bool)
    gene_mean = torch.full((4,), -0.2)
    mse = ((prediction - target) / scale).square().mean()

    def value(objective):
        return geneeffect_loss(
            prediction, target, scale, objective=objective, gene_index=gene_index,
            selective=selective, gene_mean=gene_mean,
        )

    ranking = line_ranking_loss(prediction / scale, target / scale, gene_index, selective)
    assert value("line_ranking").item() == pytest.approx((mse + ranking).item(), rel=1e-6)
    dependency = dependency_loss(prediction, target, scale, gene_mean, selective)
    assert value("dependency_classification").item() == pytest.approx(
        (mse + dependency).item(), rel=1e-6
    )
```

Also add `"line_ranking", "dependency_classification"` to the objective tuple of `test_objectives_compute_in_fp32` (pass `gene_mean=torch.zeros(4)`), and add a sampler test:

```python
def test_line_ranking_uses_gene_blocks(monkeypatch):
    from types import SimpleNamespace

    from src.training import sampling

    rows = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    dataset = SimpleNamespace(rows_by_gene=lambda: rows, collate=lambda batch: batch)
    config = {"train": {"objective": "line_ranking", "genes_per_block": 1}}
    accelerator = SimpleNamespace(process_index=0, num_processes=1)
    loader = sampling.dependency_loader(dataset, config, 0, accelerator)
    assert isinstance(loader.batch_sampler, sampling.GeneBlockSampler)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/test_objectives.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: FAIL, `ImportError: cannot import name 'dependency_loss'`.

- [ ] **Step 3: Implement the losses (`src/model/losses.py`)**

```python
from src.data.geneeffect import DEPENDENCY_THRESHOLD

# Softmax temperature of the line-ranking term, in residual-SD units.
LINE_RANKING_TEMPERATURE = 1.0


def _gene_groups(gene_index: torch.Tensor):
    return torch.unique(gene_index, return_inverse=True, return_counts=True)


def line_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gene_index: torch.Tensor,
    selective: torch.Tensor,
) -> torch.Tensor:
    """Mean over eligible genes of the ListNet cross-entropy across a gene's lines.

    Inputs are standardised residuals. For each gene, the target distribution is the
    softmax over its rows of ``-target / LINE_RANKING_TEMPERATURE`` (the most
    dependent lines carry the most mass) and the loss is its cross-entropy against
    the same softmax of the prediction. Eligibility as :func:`blocked_pearson_loss`;
    without an eligible gene the term is zero.
    """
    rows = selective.bool()
    prediction, target = prediction[rows], target[rows]
    genes, group, counts = _gene_groups(gene_index[rows])
    if not len(genes):
        return prediction.sum() * 0.0

    def log_softmax(scores: torch.Tensor) -> torch.Tensor:
        top = scores.new_full((len(genes),), -torch.inf).scatter_reduce(
            0, group, scores, reduce="amax"
        )
        shifted = scores - top[group]
        total = shifted.new_zeros(len(genes)).index_add_(0, group, shifted.exp())
        return shifted - total.log()[group]

    target_log = log_softmax(-target / LINE_RANKING_TEMPERATURE)
    predicted_log = log_softmax(-prediction / LINE_RANKING_TEMPERATURE)
    per_gene = prediction.new_zeros(len(genes)).index_add_(
        0, group, -target_log.exp() * predicted_log
    )
    means = target.new_zeros(len(genes)).index_add_(0, group, target) / counts
    spread = target.new_zeros(len(genes)).index_add_(
        0, group, (target - means[group]).square()
    )
    eligible = (counts >= MIN_PEARSON_ROWS) & (spread > 0)
    if not bool(eligible.any()):
        return per_gene.sum() * 0.0
    return per_gene[eligible].mean()


def dependency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    gene_mean: torch.Tensor,
    selective: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy of ``GeneEffect < DEPENDENCY_THRESHOLD`` on selective rows.

    ``prediction`` and ``target`` are residuals; ``gene_mean + target`` is the
    measured GeneEffect. The logit is ``(threshold - gene_mean - prediction) /
    scale``, so a more negative predicted GeneEffect means a likelier dependency.
    Without a selective row the term is zero.
    """
    rows = selective.bool()
    if not bool(rows.any()):
        return prediction.sum() * 0.0
    absolute = gene_mean[rows] + prediction[rows]
    logit = (DEPENDENCY_THRESHOLD - absolute) / scale[rows]
    label = ((gene_mean[rows] + target[rows]) < DEPENDENCY_THRESHOLD).float()
    return F.binary_cross_entropy_with_logits(logit, label)
```

Replace `geneeffect_loss` with:

```python
def geneeffect_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    objective: str,
    gene_index: torch.Tensor,
    selective: torch.Tensor,
    gene_mean: torch.Tensor,
) -> torch.Tensor:
    """One FP32 training objective over a batch of GeneEffect residual rows.

    ``prediction`` and ``target`` are in residual units, ``scale`` is each row's
    training-line residual SD and ``gene_mean`` its gene's training mean. ``huber``
    is Huber (delta 1) on residuals; ``standardized_mse`` the mean squared error of
    ``(prediction - target) / scale``; ``pearson_blocks``, ``line_ranking`` and
    ``dependency_classification`` add to it, with weight 1, the gene-blocked Pearson
    term, the ListNet term across lines (both on standardised residuals) or the
    dependency cross-entropy.
    """
    prediction, target = prediction.float(), target.float()
    if objective == "huber":
        return F.huber_loss(prediction, target, delta=1.0)
    scale = scale.float()
    standardized, standardized_target = prediction / scale, target / scale
    loss = (standardized - standardized_target).square().mean()
    if objective == "standardized_mse":
        return loss
    if objective == "pearson_blocks":
        return loss + blocked_pearson_loss(
            standardized, standardized_target, gene_index, selective
        )
    if objective == "line_ranking":
        return loss + line_ranking_loss(
            standardized, standardized_target, gene_index, selective
        )
    if objective == "dependency_classification":
        return loss + dependency_loss(
            prediction, target, scale, gene_mean.float(), selective
        )
    raise ValueError(f"unknown GeneEffect objective {objective!r}")
```

In `src/training/trainer.py` `train_update`, pass the gene mean to the loss:

```python
    dependency_loss = geneeffect_loss(
        prediction,
        dependency_batch.residual,
        dependency_batch.residual_scale,
        objective=config["train"]["objective"],
        gene_index=dependency_batch.conditions.gene_index,
        selective=dependency_batch.selective,
        gene_mean=dependency_batch.gene_mean,
    )
```

- [ ] **Step 4: Config choices and gene blocks**

`src/experiments/config.py`:

```python
OBJECTIVES = (
    "huber",
    "standardized_mse",
    "pearson_blocks",
    "line_ranking",
    "dependency_classification",
)
```

`src/training/sampling.py`: add near the top

```python
# Objectives whose batches hold every training row of a few genes.
GENE_BLOCK_OBJECTIVES = ("pearson_blocks", "line_ranking")
```

change `if train["objective"] == "pearson_blocks":` to `if train["objective"] in GENE_BLOCK_OBJECTIVES:`, and in the `dependency_loader` docstring replace "``pearson_blocks`` takes gene blocks" with "``pearson_blocks`` and ``line_ranking`` take gene blocks".

- [ ] **Step 5: Run the tests**

Run: `uv run python -m pytest tests/test_objectives.py tests/test_joint.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff format src/model/losses.py src/experiments/config.py src/training/sampling.py src/training/trainer.py tests/test_objectives.py
.venv/bin/ruff check src tests hpc
git add src/model/losses.py src/experiments/config.py src/training/sampling.py src/training/trainer.py tests/test_objectives.py
git commit -m "feat(objectives): listwise ranking across lines and a dependency cross-entropy" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Prior controls, bootstraps, per-lineage table and run comparison

**Files:**
- Create: `src/baselines/prior_controls.py`, `src/experiments/compare_runs.py`
- Modify: `src/experiments/baselines.py`, `src/experiments/all.py:52-59`, `src/experiments/revision.py`
- Test: `tests/test_prior_controls.py`, `tests/test_revision.py`

**Interfaces:**
- Consumes: `PreparedInputs.prior` (Task 2); the ladder's prediction columns `slice model_id gene_symbol method gene_effect residual residual_prediction`.
- Produces:
  - `PRIOR = "context_prior"` and `PRIOR_PLUS_TX1_RIDGE = "context_prior+context_pca_ridge[tx1]"`;
  - `prior_control_rows(inputs, split, *, pca_components=8, ridge_alpha=1.0) -> pd.DataFrame`;
  - `revision.json` per split: `bootstraps` (a list of `{comparison, repeats, seed, difference, interval}`) replacing `bootstrap`, and `lineages`;
  - CLI `python -m src.experiments.compare_runs RUN_A RUN_B --split val` printing the paired bootstrap of A minus B.

- [ ] **Step 1: Write the failing tests** (`tests/test_prior_controls.py`)

```python
"""Controls built on the prior: the prior alone and the prior plus the Tx1 ridge."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.prior_controls import (
    PRIOR,
    PRIOR_PLUS_TX1_RIDGE,
    prior_control_rows,
)
from tests.test_joint import TRAIN, VAL, make_inputs
from tests.test_prior_offsets import with_prior

PRIOR_VALUES = {m: 0.1 * i for i, m in enumerate((*TRAIN, *VAL))}


def test_prior_rows_are_the_offsets_on_the_splits_labels():
    inputs = with_prior(make_inputs(), PRIOR_VALUES)
    rows = prior_control_rows(inputs, "val")
    alone = rows.loc[rows.method == PRIOR].set_index(["model_id", "gene_symbol"])
    labels = inputs.labels.loc[inputs.labels.model_id.isin(VAL)]
    assert len(alone) == len(labels)
    for _, row in labels.iterrows():
        expected = PRIOR_VALUES[row.model_id] * inputs.residual_scale[row.gene_symbol]
        assert alone.loc[(row.model_id, row.gene_symbol), "residual_prediction"] == (
            pytest.approx(expected, rel=1e-6)
        )
    assert set(rows.slice) == {"val"}
    assert list(rows.columns) == [
        "slice", "model_id", "gene_symbol", "method", "gene_effect", "residual",
        "residual_prediction",
    ]


def test_ridge_on_what_the_prior_leaves_reproduces_a_prior_that_explains_nothing():
    plain = make_inputs()
    zero = with_prior(plain, {m: 0.0 for m in (*TRAIN, *VAL)})
    rows = prior_control_rows(zero, "val")
    stacked = rows.loc[rows.method == PRIOR_PLUS_TX1_RIDGE]
    shifted = with_prior(plain, {m: 0.0 for m in TRAIN} | {m: 1.0 for m in VAL})
    moved = prior_control_rows(shifted, "val")
    moved = moved.loc[moved.method == PRIOR_PLUS_TX1_RIDGE]
    scale = stacked.gene_symbol.map(plain.residual_scale).to_numpy()
    np.testing.assert_allclose(
        moved.residual_prediction.to_numpy(),
        stacked.residual_prediction.to_numpy() + scale,
        rtol=1e-6,
    )


def test_prior_controls_need_a_prior():
    with pytest.raises(ValueError, match="no prior"):
        prior_control_rows(make_inputs(), "val")
```

In `tests/test_revision.py`:
- the `finished` fixture also writes `tmp_path / "split.csv"` (`pd.DataFrame({"model_id": ["L1", "L2"], "lineage": ["Lung", "Skin"]}).to_csv(..., index=False)`);
- every existing `revision.write_outputs(Path("configs/revision/x.yaml"), finished, "rid")` call gains the argument `finished.parent / "split.csv"`;
- the existing assertion on `record[split]["bootstrap"]` becomes one on `record[split]["bootstraps"] == [{...the same dict...}]`;
- `list(record)` is unchanged.

Add:

```python
def test_summary_bootstraps_the_stack_against_every_prior_control(
    finished, bootstrap, monkeypatch
):
    monkeypatch.setattr(revision, "_git_revision", lambda: "abc123")
    for split in ("val", "test"):
        path = finished / f"baselines/{split}/metrics.json"
        metrics = json.loads(path.read_text())
        metrics["context_prior"] = metric_row(0.06, split=split)
        metrics["context_prior+context_pca_ridge[tx1]"] = metric_row(0.065, split=split)
        path.write_text(json.dumps(metrics))
        pd.concat(
            [
                frame("context_pca_ridge[tx1]", 0.1),
                frame("gene_mean", 0.0),
                frame("context_prior", 0.15),
                frame("context_prior+context_pca_ridge[tx1]", 0.12),
            ]
        ).to_parquet(finished / f"baselines/{split}/predictions.parquet")
    revision.write_outputs(
        Path("configs/revision/x.yaml"), finished, "rid", finished.parent / "split.csv"
    )
    record = json.loads((finished / "revision.json").read_text())
    comparisons = [b["comparison"] for b in record["val"]["bootstraps"]]
    assert comparisons == [
        "Joint model minus Context-PCA ridge (Tx1)",
        "Joint model minus Linear context prior",
        "Joint model minus Linear context prior + Tx1 context-PCA ridge",
    ]
    assert len(bootstrap) == 6  # three comparisons on each split
    lineages = record["val"]["lineages"]
    assert [row["lineage"] for row in lineages] == ["Lung", "Skin"]
    assert {"Joint model", "Linear context prior"} <= set(lineages[0])
    summary = (finished / "summary.md").read_text()
    assert "Joint model minus Linear context prior:" in summary
    assert "| Lung | 1 |" in summary
```

Without prior-control rows, `test_revision_json_and_summary` shows only the Tx1 comparison.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run python -m pytest tests/test_prior_controls.py tests/test_revision.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.baselines.prior_controls'`.

- [ ] **Step 3: Create `src/baselines/prior_controls.py`**

```python
"""Controls built on the linear context prior.

``context_prior`` is the prior export alone. ``context_prior+context_pca_ridge[tx1]``
adds the Tx1 context-PCA ridge of the control ladder (mean and variance of the
line's Tx1 cell embeddings, standardised on the training lines, 8 principal
components, a ridge per gene at alpha 1) fitted on what the prior leaves on the
labelled training lines: the linear special case of the head on the same target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.prepared import PreparedInputs

PRIOR = "context_prior"
PRIOR_PLUS_TX1_RIDGE = "context_prior+context_pca_ridge[tx1]"
COLUMNS = [
    "slice",
    "model_id",
    "gene_symbol",
    "method",
    "gene_effect",
    "residual",
    "residual_prediction",
]


def _tx1_view(inputs: PreparedInputs, lines: list[str]) -> np.ndarray:
    return np.stack(
        [
            np.concatenate(
                (
                    inputs.lines[m].controls_tx1.mean(0),
                    inputs.lines[m].controls_tx1.var(0),
                )
            )
            for m in lines
        ]
    ).astype(np.float64)


def _per_gene_ridge(
    x_train: np.ndarray, targets: pd.DataFrame, x_eval: np.ndarray, alpha: float
) -> np.ndarray:
    """A ridge per gene column on the rows where that gene has a label; genes with
    the same labelled rows share one multi-output fit (identical per-gene fits)."""
    values = targets.to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    out = np.empty((len(x_eval), values.shape[1]))
    patterns: dict[bytes, list[int]] = {}
    for column in range(values.shape[1]):
        patterns.setdefault(observed[:, column].tobytes(), []).append(column)
    for key, columns in patterns.items():
        rows = np.frombuffer(key, dtype=bool)
        model = Ridge(alpha=alpha).fit(x_train[rows], values[rows][:, columns])
        out[:, columns] = model.predict(x_eval).reshape(len(x_eval), len(columns))
    return out


def prior_control_rows(
    inputs: PreparedInputs,
    split: str,
    *,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Both controls' prediction rows for ``split``'s labelled lines."""
    if inputs.prior is None:
        raise ValueError("prior controls need a prior export, and no prior is set")
    train = list(inputs.split.supervised_train)
    lines = list(getattr(inputs.split, split))
    genes = list(inputs.genes)
    scale = inputs.residual_scale.loc[genes].to_numpy()
    offsets = inputs.prior.values.loc[[*train, *lines], genes] * scale
    residual = inputs.labels.pivot(
        index="model_id", columns="gene_symbol", values="residual"
    ).reindex(index=train, columns=genes)
    view = _tx1_view(inputs, [*train, *lines])
    spread = view[: len(train)].max(axis=0) - view[: len(train)].min(axis=0)
    view = view[:, spread > 1e-12]
    scaler = StandardScaler().fit(view[: len(train)])
    scaled = scaler.transform(view)
    components = min(pca_components, len(train) - 1, scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="full").fit(scaled[: len(train)])
    scores = pca.transform(scaled)
    ridge = _per_gene_ridge(
        scores[: len(train)],
        residual - offsets.loc[train],
        scores[len(train) :],
        ridge_alpha,
    )
    stacked = offsets.loc[lines] + ridge
    rows = inputs.labels.loc[inputs.labels.model_id.isin(lines)]

    def frame(method: str, matrix: pd.DataFrame) -> pd.DataFrame:
        values = matrix.to_numpy()[
            matrix.index.get_indexer(rows.model_id),
            matrix.columns.get_indexer(rows.gene_symbol),
        ]
        return rows.assign(slice=split, method=method, residual_prediction=values)[
            COLUMNS
        ]

    return pd.concat(
        [frame(PRIOR, offsets.loc[lines]), frame(PRIOR_PLUS_TX1_RIDGE, stacked)],
        ignore_index=True,
    )
```

- [ ] **Step 4: Add the controls to `run_baselines` and name them**

In `src/experiments/baselines.py`, after `predictions = result.predictions.copy()`:

```python
    if inputs.prior is not None:
        from src.baselines.prior_controls import prior_control_rows

        predictions = pd.concat(
            [predictions, prior_control_rows(inputs, split)], ignore_index=True
        )
```

and update its docstring: "With a prior export (``paths.prior``) the prior alone and the prior plus the Tx1 ridge join the ladder."

In `src/experiments/all.py` `BASELINE_NAMES`, add:

```python
    "context_prior": "Linear context prior",
    "context_prior+context_pca_ridge[tx1]": "Linear context prior + Tx1 context-PCA ridge",
```

- [ ] **Step 5: Bootstrap against every control and add the per-lineage table (`src/experiments/revision.py`)**

Replace `_bootstrap` with a version keyed by method, and add the comparisons list and the lineage table:

```python
# Controls the joint model is bootstrapped against, in order, when present.
COMPARED = (TX1_RIDGE, "context_prior", "context_prior+context_pca_ridge[tx1]")


def _bootstrap(
    run: Path, split: str, method: str, selective: frozenset[str]
) -> dict[str, Any]:
    """Paired line bootstrap of selective Spearman, joint model minus ``method``."""
    import pandas as pd

    from src.eval import metrics

    joint = pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
    control = pd.read_parquet(run / "baselines" / split / "predictions.parquet")
    control = control.loc[control["method"] == method]
    result = metrics.paired_line_bootstrap(
        joint, control, selective, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED
    )
    low, high = result["interval"]
    return {
        "comparison": f"{JOINT_NAME} minus {BASELINE_NAMES[method]}",
        "repeats": BOOTSTRAP_REPEATS,
        "seed": BOOTSTRAP_SEED,
        "difference": _finite(result["difference"]),
        "interval": [_finite(low), _finite(high)],
    }


def _lineages(
    run: Path, split: str, selective: frozenset[str], split_table: Path
) -> list[dict[str, Any]]:
    """Per lineage: lines, and the mean over its lines of each model's per-line
    residual Spearman across the selective genes (descriptive; 1-5 lines each)."""
    import pandas as pd
    from scipy.stats import spearmanr

    lineage = pd.read_csv(split_table).set_index("model_id")["lineage"]
    frames = {
        JOINT_NAME: pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
    }
    controls = pd.read_parquet(run / "baselines" / split / "predictions.parquet")
    for method in COMPARED:
        if method in set(controls["method"]):
            frames[BASELINE_NAMES[method]] = controls.loc[controls["method"] == method]
    per_line = {}
    for name, frame in frames.items():
        rows = frame.loc[frame["gene_symbol"].isin(selective)]
        per_line[name] = (
            rows.groupby("model_id")[["residual", "residual_prediction"]]
            .apply(lambda g: spearmanr(g["residual"], g["residual_prediction"]).statistic)
        )
    table = pd.DataFrame(per_line)
    table["lineage"] = table.index.map(lineage)
    out = []
    for name, group in table.groupby("lineage", sort=True):
        out.append(
            {
                "lineage": name,
                "lines": len(group),
                **{model: _finite(group[model].mean()) for model in per_line},
            }
        )
    return out
```

In `_split_record`, take `config_path` and build the bootstraps and lineages:

```python
def _split_record(
    run: Path, split: str, selective: frozenset[str], split_table: Path
) -> dict[str, Any]:
    """The split's table (joint model, then baselines), its bootstraps and lineages."""
    baselines = _read_json(run / "baselines" / split / "metrics.json")
    joint = _read_json(run / "evaluation" / split / "metrics.json")
    if TX1_RIDGE not in baselines:
        raise ValueError(f"baselines/{split}/metrics.json has no {TX1_RIDGE} method")
    return {
        "models": {
            JOINT_NAME: _row(joint, split),
            **{
                BASELINE_NAMES.get(method, method): _row(baselines[method], split)
                for method in _baseline_order(baselines)
            },
        },
        "bootstraps": [
            _bootstrap(run, split, method, selective)
            for method in COMPARED
            if method in baselines
        ],
        "lineages": _lineages(run, split, selective, split_table),
    }
```

`build_record` and `write_outputs` take a new last argument `split_table: Path` and pass it to `_split_record`. `run_revision` computes it from the bound config, since the table sits next to the split JSON:

```python
        split_table = Path(config["paths"]["split"]).with_suffix(".csv")
        if not split_table.is_file():
            raise FileNotFoundError(f"{split_table}: the split table next to the split file")
        summary = write_outputs(config_path, run, run_id, split_table)
```

In `tests/test_revision.py`, the `finished` fixture writes a `split.csv` with `model_id,lineage` rows for its lines. The existing `write_outputs(...)` calls pass it as the new argument.

In `_split_lines`, print one sentence per bootstrap and a lineage table:

```python
def _split_lines(title: str, split: dict[str, Any]) -> list[str]:
    header = ["Model", *(name for name, _, _ in COLUMNS)]
    lines = [
        f"## {title} lines",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, row in split["models"].items():
        cells = [_number(row[key]) for _, key, _ in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for bootstrap in split["bootstraps"]:
        low, high = bootstrap["interval"]
        lines.append(
            f"Selective Spearman, {bootstrap['comparison']}: "
            f"{_number(bootstrap['difference'])} [{_number(low)}, {_number(high)}], "
            f"paired bootstrap over {title.lower()} lines ({bootstrap['repeats']} "
            f"resamples, seed {bootstrap['seed']})."
        )
    models = [key for key in split["lineages"][0] if key not in {"lineage", "lines"}]
    lines += [
        "",
        f"Per lineage, descriptive: mean over the lineage's {title.lower()} lines of "
        "the per-line residual Spearman across the selective genes.",
        "",
        "| Lineage | Lines | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    for row in split["lineages"]:
        cells = [_number(row[model]) for model in models]
        lines.append(f"| {row['lineage']} | {row['lines']} | " + " | ".join(cells) + " |")
    return lines + [""]
```

Update the module docstring: "the paired line bootstrap of selective-gene Spearman against the Tx1 context-PCA ridge and, with a prior, against the prior alone and the prior plus the Tx1 ridge; a per-lineage table".

- [ ] **Step 6: Create `src/experiments/compare_runs.py`**

```python
"""Paired line bootstrap of selective Spearman between two runs of one split.

``python -m src.experiments.compare_runs RUN_A RUN_B --split val`` reads
``evaluation/<split>/predictions.parquet`` of two revision run directories and prints
A minus B with its 95% interval (1,000 resamples, seed 0) over the selective genes
of A's ``best.pt``. It reports; the reader decides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval import metrics
from src.experiments.revision import BOOTSTRAP_REPEATS, BOOTSTRAP_SEED
from src.training.checkpoint import load_checkpoint


def compare(run_a: Path, run_b: Path, split: str) -> dict:
    selective = load_checkpoint(run_a / "train" / "best.pt")["preprocessing"][
        "selective_genes"
    ]
    frames = [
        pd.read_parquet(run / "evaluation" / split / "predictions.parquet")
        for run in (run_a, run_b)
    ]
    return metrics.paired_line_bootstrap(
        *frames, selective, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    args = parser.parse_args(argv)
    result = compare(args.run_a, args.run_b, args.split)
    low, high = result["interval"]
    print(
        f"{args.run_a.name} minus {args.run_b.name}, {args.split}: "
        f"{result['difference']:.4f} [{low:.4f}, {high:.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `tests/test_revision.py`:

```python
def test_compare_runs_bootstraps_a_minus_b(finished, bootstrap, tmp_path):
    from src.experiments.compare_runs import compare

    result = compare(finished, finished, "val")
    assert result == {"difference": 0.02, "interval": [-0.01, 0.05]}
    left, right, selective, repeats, seed = bootstrap[-1]
    assert list(selective) == list(SELECTIVE) and (repeats, seed) == (1000, 0)
```

`compare` calls the bootstrap through the module (`metrics.paired_line_bootstrap`), as `revision._bootstrap` does, so the fixture's monkeypatch reaches it.

- [ ] **Step 7: Run the tests, the suite, lint; commit**

```bash
uv run python -m pytest tests/test_prior_controls.py tests/test_revision.py tests/test_baselines.py -q > /tmp/pytest.txt 2>&1; tail -5 /tmp/pytest.txt
uv run python -m pytest tests -q > /tmp/pytest.txt 2>&1; tail -3 /tmp/pytest.txt
.venv/bin/ruff format src/baselines/prior_controls.py src/experiments/compare_runs.py src/experiments/baselines.py src/experiments/all.py src/experiments/revision.py tests/test_prior_controls.py tests/test_revision.py
.venv/bin/ruff check src tests hpc
git add src/baselines/prior_controls.py src/experiments/compare_runs.py src/experiments/baselines.py src/experiments/all.py src/experiments/revision.py tests/test_prior_controls.py tests/test_revision.py
git commit -m "feat(correction): prior controls, bootstraps against each, per-lineage table" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(If `tests/test_baselines.py` does not exist, drop it from the command; `grep -l run_baselines tests/*.py` finds the baseline tests.)

---

### Task 6: Correction configs and documents

**Files:**
- Create: `configs/correction/huber_no_state.yaml`, `standardized_mse_no_state.yaml`, `line_ranking_no_state.yaml`, `dependency_classification_no_state.yaml`
- Modify: `docs/03-geneeffect-protocol.md` (§4–§6 objectives, §10, new §12), `CLAUDE.md`, `hpc/README.md`, `src/README.md`
- Test: `tests/test_correction_configs.py`

- [ ] **Step 1: Write the failing test**

```python
"""The wave-one correction configs: one objective each, STATE absent, one prior."""

from pathlib import Path

import pytest

from src.experiments.config import load_config

CONFIGS = sorted(Path("configs/correction").glob("*.yaml"))
EXPORT = "outputs/context_prior/default_prior_export/export"


def test_four_objective_screen_configs_differ_only_in_objective():
    assert [p.stem for p in CONFIGS] == [
        "dependency_classification_no_state",
        "huber_no_state",
        "line_ranking_no_state",
        "standardized_mse_no_state",
    ]
    configs = [load_config(p) for p in CONFIGS]
    for path, config in zip(CONFIGS, configs):
        assert config["train"]["objective"] == path.stem.removesuffix("_no_state")
        assert config["paths"]["prior"] == EXPORT
        assert config["output_root"] == "outputs/geneeffect_correction"
        blocks = config["model"]["head_blocks"]
        assert not blocks["use_delta_proj"] and not blocks["use_s"]
    for config in configs:
        config["train"]["objective"] = None
    assert all(config == configs[0] for config in configs)


@pytest.mark.parametrize("path", CONFIGS)
def test_correction_configs_validate(path):
    load_config(path)
```

- [ ] **Step 2: Create the configs**

`configs/correction/huber_no_state.yaml` is a copy of `configs/revision/frozen_huber.yaml` (after Task 2 added `prior: null`) with these changes:

```yaml
# Single-cell correction, wave one, objective screen: the nested head on what the
# default prior leaves (paths.prior), STATE absent, objective huber.
...
train:
  ...
  objective: huber
  state_mode: frozen   # unused: STATE is absent (use_delta_proj and use_s false)
...
output_root: outputs/geneeffect_correction
...
model:
  ...
  head_blocks: {use_delta_proj: false, use_s: false, use_q_sc: true, use_e_g: true, use_z_c: true}
...
paths:
  ...
  # The default prior's export (hpc/run.sh prior-export configs/context_prior/default_prior.yaml --run-id default_prior_export).
  prior: outputs/context_prior/default_prior_export/export
```

The other three configs differ only in the header comment and `objective` (`standardized_mse`, `line_ranking`, `dependency_classification`).

- [ ] **Step 3: Check the design amendments.** §12 of the design spec was written with this plan; change it only if implementation changed a decision, and say so in the commit message.

- [ ] **Step 4: Update the protocol, `CLAUDE.md`, `hpc/README.md` and `src/README.md`**

- `docs/03-geneeffect-protocol.md`:
  - **§5 (objectives):** add `line_ranking` and `dependency_classification` with their definitions from Task 4.
  - **§10:** add a paragraph on `hpc/run.sh prior-export CONFIG --run-id ID`, describing the export and its out-of-fold rule.
  - **New §12 "Single-cell correction":**
    - `paths.prior`;
    - the stack and its start at the prior (validation at epoch −1);
    - the fail-closed checks;
    - the two prior controls and their bootstraps;
    - the per-lineage table;
    - the wave-one configs and the reading rule;
    - the claim boundary: single-gene dependency evidence, not SL.
- `CLAUDE.md`: in "Current state", one sentence that the correction (wave one) is implemented with its configs in `configs/correction/`. In "Commands", add `hpc/run.sh prior-export CONFIG --run-id ID` and `uv run python -m src.experiments.compare_runs RUN_A RUN_B --split val`.
- `hpc/README.md`: a "`prior-export`" paragraph after the `prior` command section. Also a sentence in the `revision` section: with `paths.prior` set, the run is a correction, its baselines add the two prior controls, and validation runs before the first update.
- `src/README.md`:
  - `data`: add the prior offsets;
  - `model`: add the two objectives;
  - `baselines`: add the prior controls;
  - `experiments`: add `prior_export` and `compare_runs`.

- [ ] **Step 5: Run the tests; commit**

```bash
uv run python -m pytest tests/test_correction_configs.py -q > /tmp/pytest.txt 2>&1; tail -3 /tmp/pytest.txt
git add configs/correction tests/test_correction_configs.py docs CLAUDE.md hpc/README.md src/README.md
git commit -m "docs(correction): wave-one configs, protocol §12, design amendments" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Run wave one on the H20 host (port 30838)

No code. Every step reports evidence (paths and numbers), never just a PID.

- [ ] **Step 1: Full suite and lint locally.** Both must pass. Push is through Git only. The host cannot reach GitHub, so send a bundle and create the worktree:

```bash
git bundle create /tmp/correction.bundle main..feat/single-cell-correction
H='ssh -o BatchMode=yes -J richard@100.91.229.50 -p 30838 root@10.15.171.204'
$H 'cat > /tmp/correction.bundle' < /tmp/correction.bundle
$H 'cd /2023533015/VCC_Project && git status --short | head -3 && git fetch /tmp/correction.bundle feat/single-cell-correction:feat/single-cell-correction && git worktree add /2023533015/VCC_Project_correction feat/single-cell-correction && cd /2023533015/VCC_Project_correction && for d in data model outputs .venv-tx1; do ln -s /2023533015/VCC_Project/$d $d; done && git log -1 --oneline'
```

The host's `main` (at 1694f5c) does not contain the merge base `a5d95df`. In that case, bundle `feat/single-cell-correction` in full (`git bundle create /tmp/correction.bundle feat/single-cell-correction`).

- [ ] **Step 2: Check the hardware and that nothing else runs:**

```bash
$H 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; ps -eo args | grep -c "[s]rc\."'
```

- [ ] **Step 3: Export the prior** (CPU, disconnect-safe):

```bash
$H 'cd /2023533015/VCC_Project_correction && mkdir -p outputs/launches && OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 nohup hpc/run.sh prior-export configs/context_prior/default_prior.yaml --run-id default_prior_export > outputs/launches/default_prior_export.log 2>&1 &'
```

Confirm from a fresh session that the export is finished: `tail -c 300 outputs/launches/default_prior_export.log` shows `prior export: …`. Then read `scores` in `outputs/context_prior/default_prior_export/export/prior.json`. The `val` and `test` selective Spearman must reproduce the default prior's row (0.2290 and 0.2338). If they differ, stop and investigate before any training. The `train` score is the out-of-fold prior on training lines; it is telemetry.

- [ ] **Step 4: Objective screen, one run at a time, each on all four GPUs.** Run the four configs in the order Huber, standardised MSE, line ranking, dependency classification, chained in one disconnect-safe shell:

```bash
$H 'cd /2023533015/VCC_Project_correction && nohup bash -c "for o in huber standardized_mse line_ranking dependency_classification; do hpc/run.sh revision configs/correction/\${o}_no_state.yaml --run-id correction_\${o}_no_state_seed0 || exit 1; done" > outputs/launches/correction_objective_screen.log 2>&1 &'
```

Confirm each run from `outputs/geneeffect_correction/<id>/`:
- `train/done.json`;
- `evaluation/{val,test}/metrics.json`;
- `baselines/{val,test}/metrics.json`, holding the two prior controls;
- `summary.md` last.

The first run's `train/metrics.jsonl` must start with an epoch −1 record whose validation selective Spearman is the export's `val` score within 1e-3. If it isn't, stop.

- [ ] **Step 5: Read the screen.** Pull each `summary.md` (under 1 KB per chunk; use the paced `dd` pull for anything larger).
  1. Take the highest validation selective Spearman at `best.pt`.
  2. For the runner-up and every run within reach, run on the host: `uv run python -m src.experiments.compare_runs outputs/geneeffect_correction/<best> outputs/geneeffect_correction/<other> --split val`.
  3. If the interval contains zero and the other run is simpler, the simpler wins:
     - Huber and standardised MSE are single-term;
     - line ranking and dependency classification are two-term;
     - between the two single-term losses, the higher point estimate wins.
  4. Write the reading into `results/correction_wave_one_seed0/README.md`: every run's validation and test rows, the prior controls, the bootstraps, and the choice with its reason.

- [ ] **Step 6: STATE runs.**
  1. Create `configs/correction/<winner>_frozen_state.yaml` and `<winner>_trainable_state.yaml` from the winner's config:
     - `use_delta_proj: true` and `use_s: true`;
     - `state_mode: frozen` or `state_mode: trainable`;
     - a header comment naming the screen's choice.
  2. Commit them, bundle and fast-forward the worktree.
  3. Launch both runs in sequence, as in Step 4.
  4. Read them the same way. Order of simplicity: STATE absent, then frozen, then trainable.

- [ ] **Step 7: Stop and report.** Complete `results/correction_wave_one_seed0/README.md`:
  - every row, with validation and test;
  - the stack minus the prior, minus the prior plus Tx1 ridge, and minus the Tx1 ridge;
  - the per-lineage tables;
  - which setting wins and why.

  Commit it. Then merge `feat/single-cell-correction` into `main`: the suite and lint pass, and the export and the runs have run on the host. Remove the host worktree after checking that no process runs in it, delete the branch locally and on the host, and ask the user to push `main`.

  **Report to the user and launch nothing more.** If STATE absent won, say so first, because the research plan is to be discussed before the second wave.

---

### Task 8: Single-cell atlas search and the extra single-cell membership (in parallel with Tasks 1–7)

**Files:**
- Create: `docs/data/single-cell-atlas-search.md`, `configs/benchmarks/single_cell_atlas_candidates.csv`
- Create: `src/data/prepare/build_extra_single_cell_lines.py`, `configs/benchmarks/extra_single_cell_lines_26Q1.json`
- Test: `tests/test_extra_single_cell_lines.py`

**Interfaces:**
- Consumes:
  - `src.data.depmap.read_models`, `read_model_ids`;
  - `src.data.splits.load_geneeffect_226_split`;
  - the local 26Q1 `Model.csv` and `CRISPRGeneEffect.csv` (on the Mac under `data/sl_dependency_v0/raw/depmap/`).
- Produces: `build_extra_single_cell_lines(candidates: pd.DataFrame, models: pd.DataFrame, *, labelled_ids, split) -> dict` with keys `schema_version`, `policy`, `labelled`, `unlabelled`, `excluded`, `sources`.

- [ ] **Step 1: Search** (web; a research agent is appropriate). Find public single-cell RNA-seq of untreated or vehicle-treated cancer cell lines outside the 226 that DepMap has screened. Leads:
  - MIX-seq (McFarland et al. 2020, pooled lines with DMSO controls);
  - the Tahoe-100M lines not in the split (38 of its 50 are in train);
  - the Kinker lines outside the 226 (152 of 198 are used; their raw UMI is on the host as SCP542);
  - any 2023–2026 multi-line atlas.

  For each source record:
  - the accession or URL;
  - raw UMI counts or not;
  - the untreated/vehicle condition;
  - cells per line;
  - how lines are identified (DepMap ModelID, RRID, CCLE name);
  - total size;
  - the licence.

  Resolve lines to ModelIDs only through identifiers the source gives (ModelID, RRID or CCLE name matched against `Model.csv` columns), never by informal names. Write `configs/benchmarks/single_cell_atlas_candidates.csv` with columns `source, accession, model_id, identifier, identifier_kind, cells, raw_counts, condition, size_gb`, and the narrative in `docs/data/single-cell-atlas-search.md`.

- [ ] **Step 2: Write the failing test**

```python
"""Extra single-cell lines: labelled, outside the 226, never sharing a held-out patient."""

import pandas as pd

from src.data.prepare.build_extra_single_cell_lines import build_extra_single_cell_lines
from src.data.splits import FixedSplit


def test_membership_excludes_held_patients_and_our_lines():
    models = pd.DataFrame(
        {"patient_id": ["P1", "P2", "P3", "P4", "P5"]},
        index=["ACH-T", "ACH-V", "ACH-A", "ACH-B", "ACH-C"],
    )
    models.loc["ACH-D"] = {"patient_id": "P2"}
    split = FixedSplit(train=("ACH-T",), val=("ACH-V",), test=())
    candidates = pd.DataFrame(
        {
            "source": ["mixseq"] * 4 + ["tahoe"],
            "model_id": ["ACH-T", "ACH-A", "ACH-B", "ACH-D", "ACH-A"],
        }
    )
    payload = build_extra_single_cell_lines(
        candidates, models, labelled_ids={"ACH-A", "ACH-D"}, split=split
    )
    assert payload["labelled"] == ["ACH-A"]
    assert payload["unlabelled"] == ["ACH-B"]
    assert "ACH-T" in payload["excluded"] and "226" in payload["excluded"]["ACH-T"]
    assert "ACH-V" in payload["excluded"]["ACH-D"]
    assert payload["sources"]["ACH-A"] == ["mixseq", "tahoe"]
```

- [ ] **Step 3: Implement the builder** (modelled on `build_extra_bulk_lines.py`)

```python
"""Build configs/benchmarks/extra_single_cell_lines_26Q1.json from the pinned
single-cell atlas candidates and the 26Q1 files.

uv run python -m src.data.prepare.build_extra_single_cell_lines \
    --candidates configs/benchmarks/single_cell_atlas_candidates.csv \
    --model data/sl_dependency_v0/raw/depmap/Model.csv \
    --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv \
    --split configs/benchmarks/cell_line_geneeffect_226_split.json \
    --out configs/benchmarks/extra_single_cell_lines_26Q1.json
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from src.data.depmap import read_model_ids, read_models
from src.data.splits import FixedSplit, load_geneeffect_226_split

SCHEMA_VERSION = 1
POLICY = (
    "Lines outside the 226 with public basal single-cell RNA; labelled ones have 26Q1 "
    "GeneEffect. A line in the 226 or sharing a PatientID with a validation or test "
    "line is excluded. Membership is pinned; ingesting the lines is a separate step."
)


def build_extra_single_cell_lines(
    candidates: pd.DataFrame,
    models: pd.DataFrame,
    *,
    labelled_ids: Collection[str],
    split: FixedSplit,
) -> dict:
    missing = sorted(set(candidates["model_id"]) - set(models.index))
    if missing:
        raise ValueError(f"ModelIDs absent from Model.csv: {missing[:10]}")
    ours = set(split.all_model_ids)
    held = {models.at[m, "patient_id"]: m for m in (*split.val, *split.test)}
    excluded = {}
    for model_id in sorted(set(candidates["model_id"])):
        if model_id in ours:
            excluded[model_id] = "already one of the 226"
        elif models.at[model_id, "patient_id"] in held:
            excluded[model_id] = (
                f"shares PatientID {models.at[model_id, 'patient_id']} with held-out "
                f"line {held[models.at[model_id, 'patient_id']]}"
            )
    usable = set(candidates["model_id"]) - set(excluded)
    sources = candidates.groupby("model_id")["source"].apply(
        lambda s: sorted(set(s))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "labelled": sorted(usable & set(labelled_ids)),
        "unlabelled": sorted(usable - set(labelled_ids)),
        "excluded": excluded,
        "sources": {m: sources[m] for m in sorted(usable)},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "model", "gene-effect", "split", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_extra_single_cell_lines(
        pd.read_csv(args.candidates),
        read_models(args.model),
        labelled_ids=read_model_ids(args.gene_effect),
        split=load_geneeffect_226_split(args.split),
    )
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{len(payload['labelled'])} labelled, {len(payload['unlabelled'])} "
        f"unlabelled, {len(payload['excluded'])} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The test builds `models` with a `patient_id` column only. `read_models` returns more columns, and the builder reads only `patient_id`.

- [ ] **Step 4: Run the test and the builder; commit**

```bash
uv run python -m pytest tests/test_extra_single_cell_lines.py -q > /tmp/pytest.txt 2>&1; tail -3 /tmp/pytest.txt
uv run python -m src.data.prepare.build_extra_single_cell_lines --candidates configs/benchmarks/single_cell_atlas_candidates.csv --model data/sl_dependency_v0/raw/depmap/Model.csv --gene-effect data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv --split configs/benchmarks/cell_line_geneeffect_226_split.json --out configs/benchmarks/extra_single_cell_lines_26Q1.json
git add docs/data/single-cell-atlas-search.md configs/benchmarks/single_cell_atlas_candidates.csv configs/benchmarks/extra_single_cell_lines_26Q1.json src/data/prepare/build_extra_single_cell_lines.py tests/test_extra_single_cell_lines.py
git commit -m "feat(data): single-cell atlas search and extra labelled single-cell membership" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Report** how many labelled single-cell lines the search adds, per source, with sizes and the download path. Ingesting them is the user's decision: it changes the training set of every later run. Anything over 5 GB downloads on the host only.

---

## Self-review notes

- **Spec coverage.**

  | Spec section | What it asks | Task |
  | --- | --- | --- |
  | §5.2 | the head on `r − r̂_prior` | 2, 3 |
  | §5.2 | output layers at zero | 3 |
  | §5.2 | validation before the first update | 3 |
  | §7.2 | export and the identity guard | 1, 2 |
  | §7.3 | objective screen, as amended | 4, 6, 7 |
  | §7.3 | STATE comparison | 7 |
  | §8.2 | the prior as a control | 5 |
  | §8.2 | stack-minus-prior and stack-minus-Tx1 bootstraps | 5 |
  | §8.2 | per-lineage table | 5 |
  | §10.1 | atlas search | 8 |

  Bridge quality is already in the prior's `results.md` and is not repeated in the correction summary. The second-wave inputs are in the second-wave plan.
- **Units.**
  - The export is in residual-SD units.
  - `DependencyDataset` multiplies by `inputs.residual_scale`, so the offset is in residual units.
  - Loss and evaluation add it to `delta_hat`, which is also in residual units.
  - The prior control multiplies by the same scale.
- **Names used across tasks:**
  - `PriorOffsets`, `read_prior_offsets`, `checked_prior`, `check_prior_identity` (Task 2);
  - `with_prior` (test helper, Task 2, reused in Tasks 3 and 5);
  - `PRIOR`, `PRIOR_PLUS_TX1_RIDGE` (Task 5);
  - `GENE_BLOCK_OBJECTIVES` (Task 4);
  - `export_predictions`, `write_export`, `training_side_folds` (Task 1).

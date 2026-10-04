"""Prior config schema, block selection, long frames, and the runner's steps on
synthetic lines: curve and decision, cross-fitting, view weights, test scoring,
JSON and summary."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.context_prior.bridge import fit_bridge
from src.context_prior.prior import BLOCKS, PriorInputs, PriorSpec, Stage, fit_prior
from src.context_prior.reference import Reference
from src.context_prior.targets import Definitions
from src.context_prior.views import fit_expression_components
from src.data.splits import FixedSplit
from src.experiments import context_prior as run
from src.experiments.config import validate_prior_config

CONFIG = yaml.safe_load(Path("configs/context_prior/prior.yaml").read_text())


def test_prior_config_is_strict():
    validate_prior_config(CONFIG)
    missing = copy.deepcopy(CONFIG)
    del missing["curve"]["subsets"]
    with pytest.raises(ValueError, match="missing"):
        validate_prior_config(missing)
    unknown = copy.deepcopy(CONFIG)
    unknown["prior"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_prior_config(unknown)
    assert math.isinf(CONFIG["selection"]["shrinkages"][-1])


def test_long_frame_keeps_finite_labels_in_geneeffect_units():
    definitions = Definitions(
        genes=("A", "B"),
        gene_means=pd.Series({"A": -1.0, "B": 0.0}),
        selective=("A",),
        variable=("A", "B"),
        residual_scale=pd.Series({"A": 2.0, "B": 1.0}),
    )
    truth = pd.DataFrame({"A": [1.0, np.nan], "B": [0.5, 0.0]}, index=["L1", "L2"])
    prediction = pd.DataFrame({"A": [0.5, 0.1], "B": [0.0, 0.2]}, index=["L1", "L2"])
    frame = run.long_frame(prediction, truth, definitions)
    assert len(frame) == 3
    row = frame.set_index(["model_id", "gene_symbol"]).loc[("L1", "A")]
    assert row["residual"] == 2.0 and row["residual_prediction"] == 1.0
    assert row["gene_effect"] == 1.0 and row["geneeffect_prediction"] == 0.0


def test_select_blocks_keeps_only_blocks_whose_gain_interval_excludes_zero():
    gains = {
        "pathway_scores": [-0.01, 0.02],
        "predicted_genotype": [0.01, 0.03],
        "own_expression": [0.001, 0.01],
        "partners": [-0.02, 0.0],
        "data_selected": [0.02, 0.04],
        "rank": [-0.01, 0.01],
    }

    def evaluate(spec: PriorSpec):
        last = spec.stages[-1]
        score = last.penalty if last.block != "data_selected" else last.selected
        return float(score), pd.DataFrame({"A": [float(len(spec.stages))]})

    calls = []

    def interval(prediction, current, block):
        calls.append(block)
        return gains[block]

    selection = {
        "penalties": [0.1, 1.0],
        "shrinkages": [0.5, 2.0],
        "selected": [10, 50],
        "rank": 64,
    }
    spec, log = run.select_blocks(evaluate, interval, selection)
    assert [s.block for s in spec.stages] == [
        "expression_components",
        "predicted_genotype",
        "own_expression",
        "data_selected",
    ]
    assert spec.stages[2].penalty == 2.0  # best shrinkage by point estimate
    assert [s for s in log if s["block"] == "partners"][0]["penalty"] == 2.0
    assert spec.stages[-1].selected == 50 and spec.rank is None
    assert calls[0] == "pathway_scores"  # the first block is never tested


# ----------------------------------------------------------------------------
# The runner's steps on synthetic lines
# ----------------------------------------------------------------------------

SPACE = [f"E{i}" for i in range(16)]
GENES = ["E0", "E1", "E2", "E3", "Q"]  # Q has no expression column
TX1 = "context_pca_ridge[tx1]"


def synthetic_run(tmp_path: Path) -> run.RunData:
    """40 labelled single-cell training lines (3 without bulk RNA), one unlabelled
    one, 60 labelled and 20 unlabelled extra lines, 10 validation lines (one without
    bulk RNA) and 10 test lines; pseudo-bulk is a per-gene affine distortion of the
    expression with noise."""
    rng = np.random.default_rng(0)
    single = [f"S{i}" for i in range(40)]
    no_bulk = single[-3:]
    extras = [f"X{i}" for i in range(60)]
    unlabelled = [f"Y{i}" for i in range(20)]
    val, test = [f"V{i}" for i in range(10)], [f"T{i}" for i in range(10)]
    everyone = [*single, "U0", *extras, *unlabelled, *val, *test]
    expression = pd.DataFrame(
        rng.normal(size=(len(everyone), len(SPACE))), index=everyone, columns=SPACE
    )
    signal = pd.DataFrame(
        {
            "E0": expression["E3"],
            "E1": -expression["E1"],
            "E2": expression["E4"] + expression["E5"],
            "E3": expression["E6"],
            "Q": expression["E5"] - expression["E6"],
        }
    ) + 0.3 * rng.normal(size=(len(everyone), len(GENES)))
    gene_effect = signal.loc[[*single, *extras, *val, *test]].copy()
    gene_effect.iloc[0, 0] = np.nan  # a missing training label
    split = FixedSplit(
        train=(*single, "U0"), val=tuple(val), test=tuple(test), unlabeled_train=("U0",)
    )
    definitions = Definitions(
        genes=tuple(GENES),
        gene_means=pd.Series(-0.5, index=GENES),
        selective=tuple(GENES),
        variable=tuple(GENES),
        residual_scale=pd.Series(0.5, index=GENES),
    )
    side = [m for m in [*single, "U0", *extras, *unlabelled] if m not in no_bulk]
    labelled = tuple(m for m in [*single, *extras] if m not in no_bulk)
    single_cell_train = tuple(m for m in single if m not in no_bulk)
    single_cell = [*single, "U0", *val, *test]
    slope = rng.uniform(0.5, 2.0, len(SPACE))
    pseudobulk = (expression.loc[single_cell] - rng.normal(size=len(SPACE))) / slope
    pseudobulk += 0.1 * rng.normal(size=pseudobulk.shape)
    bridge = fit_bridge(
        pseudobulk.loc[list(single_cell_train)], expression.loc[list(single_cell_train)]
    )
    bulk = expression.loc[side]
    reference = Reference(
        paralogs=pd.DataFrame({"gene": ["E0"], "paralog": ["E2"], "identity": [50.0]}),
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["E1", "E4"]}),
        hallmark=pd.DataFrame({"gene_set": ["S"], "gene": ["E7"]}),
        progeny=pd.DataFrame({"pathway": ["P"], "gene": ["E8"], "weight": [1.0]}),
        drivers=pd.DataFrame(
            {"KRAS": (bulk["E9"] > 0).astype(int)},
            index=pd.Index(side, name="model_id"),
        ),
        msi=pd.Series(bulk["E10"].to_numpy(), index=pd.Index(side, name="model_id")),
    )
    lineage = pd.Series(rng.choice(["Lung", "Skin"], len(everyone)), index=everyone)
    inputs = PriorInputs(
        expression=bulk,
        residual=run.scaled_residual(gene_effect, labelled, definitions),
        components=fit_expression_components(bulk, 6),
        reference=reference,
        lineage=lineage.loc[side],
        patients={m: m for m in everyone},
    )
    vectors = rng.normal(size=(4, 3)).astype(np.float32)
    np.savez(
        tmp_path / "esm2.npz",
        symbols=np.array(GENES[:4], dtype=object),
        vectors=vectors,
        resolved=np.ones(4, dtype=bool),
    )
    config = {
        "seed": 0,
        "curve": {"sizes": [20, 40], "subsets": 2, "selected": 3},
        "selection": {
            "penalties": [0.01, 1.0],
            "shrinkages": [1.0, math.inf],
            "selected": [2, 3],
            "rank": 2,
            "view_weights": True,
        },
        "prior": {"components": 6, "folds": 3, "bootstrap_repeats": 50},
    }
    return run.RunData(
        config=config,
        joint={"paths": {"esm2_embeddings": str(tmp_path / "esm2.npz")}},
        split=split,
        gene_effect=gene_effect,
        definitions=definitions,
        inputs=inputs,
        labelled=labelled,
        single_cell_train=single_cell_train,
        lineage=lineage,
        pseudobulk=pseudobulk,
        bridge=bridge,
        queries={
            "oracle": expression.loc[val[:-1]],
            "val": bridge.apply(pseudobulk.loc[val]),
        },
    )


SPEC = PriorSpec(
    (
        Stage("expression_components", 0.1),
        Stage("pathway_scores", 1.0),
        Stage("data_selected", 0.1, selected=2),
    )
)


def test_curve_and_decision_score_both_inputs_and_round_trip_as_json(tmp_path):
    data = synthetic_run(tmp_path)
    points = run.curve_points(data)
    assert points[0] == ("single_cell_train", data.single_cell_train)
    assert points[-1] == ("all", data.labelled) and len(points) == 6
    rows, predictions = run.run_curve(data)
    assert len(rows) == len(points) * 2 * 2
    assert {row["input"] for row in rows} == {"oracle", "val"}
    assert all(p in (0.01, 1.0) for row in rows for p in row["penalties"])
    oracle = predictions[("all", "components_and_selected", "oracle")]
    assert list(oracle.index) == list(data.split.val[:-1])
    decision = run.decide(data, predictions)
    assert decision["input"] == "val" and decision["binding"]
    assert decision["passes"] == (decision["interval"][0] > 0)
    assert decision["bridge_failing"] == (
        decision["oracle_interval"][0] > 0 and not decision["passes"]
    )
    run._write_json(tmp_path / "curve.json", rows)
    run._write_json(tmp_path / "decision.json", decision)
    assert run._read_json(tmp_path / "decision.json")["passes"] == decision["passes"]


def test_crossfit_bridges_single_cell_lines_without_their_fold(tmp_path):
    data = synthetic_run(tmp_path)
    stages, quality, folds = run.run_crossfit(data, SPEC)
    extras = [m for m in data.labelled if not m.startswith("S")]
    assert list(stages) == [s.block for s in SPEC.stages]
    assert sorted(stages["pathway_scores"].index) == sorted(
        [*data.split.supervised_train, *extras]
    )  # the 3 lines without bulk RNA are queries only; unlabelled lines are not
    assert set(folds) == {*data.inputs.expression.index, *data.split.supervised_train}
    held = [m for m in data.split.supervised_train if folds[m] == 0]
    held_extras = [m for m in extras if folds[m] == 0]
    rest = [m for m in data.single_cell_train if folds[m] != 0]
    fitted = fit_prior(
        SPEC,
        data.inputs,
        fit_lines=[m for m in data.labelled if folds[m] != 0],
        encoder_lines=[m for m in data.inputs.expression.index if folds[m] != 0],
    )
    fold_bridge = fit_bridge(
        data.pseudobulk.loc[rest], data.inputs.expression.loc[rest]
    )
    direct = fitted.predict(fold_bridge.apply(data.pseudobulk.loc[held]))
    from_bulk = fitted.predict(data.inputs.expression.loc[held_extras])
    for block in stages:
        assert np.allclose(stages[block].loc[held], direct[block])
        assert np.allclose(stages[block].loc[held_extras], from_bulk[block])
    full_bridge = fitted.predict(data.bridge.apply(data.pseudobulk.loc[held]))
    assert not np.allclose(
        stages["data_selected"].loc[held], full_bridge["data_selected"]
    )
    assert list(quality.index) == SPACE and quality.notna().all()


def test_failed_decision_trains_on_single_cell_lines_only(tmp_path):
    data = run.single_cell_only(synthetic_run(tmp_path))
    assert data.labelled == data.single_cell_train
    assert list(data.inputs.residual.index) == list(data.single_cell_train)
    stages, _, _ = run.run_crossfit(data, SPEC)
    assert sorted(stages["expression_components"].index) == sorted(
        data.split.supervised_train
    )


def test_selection_logs_every_block_and_the_chosen_prior_round_trips(tmp_path):
    data = synthetic_run(tmp_path)
    spec, log = run.run_selection(data)
    assert spec.stages[0].block == "expression_components"
    assert [entry["block"] for entry in log] == [*BLOCKS, "reduced_rank"]
    run._write_json(
        tmp_path / "selection.json", {"spec": run._spec_json(spec), "log": log}
    )
    restored = run._spec_from_json(run._read_json(tmp_path / "selection.json")["spec"])
    assert restored == spec
    pooled = PriorSpec(
        (Stage("expression_components", 0.1), Stage("own_expression", math.inf)), 2
    )
    run._write_json(tmp_path / "pooled.json", run._spec_json(pooled))
    assert run._spec_from_json(run._read_json(tmp_path / "pooled.json")) == pooled


def test_view_weights_are_tried_with_two_context_blocks(tmp_path):
    data = synthetic_run(tmp_path)
    stages, _, _ = run.run_crossfit(data, SPEC)
    weights, record = run.run_view_weights(data, SPEC, stages)
    assert record["tried"] and record["blocks"] == [
        "expression_components",
        "pathway_scores",
    ]
    assert (weights is not None) == record["kept"]
    single = PriorSpec((Stage("expression_components", 0.1),))
    assert run.run_view_weights(data, single, stages) == (
        None,
        {"tried": False, "kept": False},
    )


def test_out_of_fold_parquet_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    lines, genes = ["L2", "L1", "L3"], ["G1", "G0"]
    stages = {
        block: pd.DataFrame(rng.normal(size=(3, 2)), index=lines, columns=genes)
        for block in ("expression_components", "data_selected")
    }
    run.write_oof(stages, tmp_path / "oof.parquet")
    back = pd.read_parquet(tmp_path / "oof.parquet").astype(
        {"block": str, "model_id": str, "gene_symbol": str}
    )
    assert len(back) == 12
    wide = (
        back.loc[back["block"] == "data_selected"]
        .set_index(["model_id", "gene_symbol"])["prediction_sigma"]
        .unstack()
    )
    assert np.allclose(wide.loc[lines, genes], stages["data_selected"])


def _write_controls(data: run.RunData, run_dir: Path, split: str) -> int:
    """Pre-fitted controls for ``split``: the Tx1 ridge lacks one scored pair and
    holds one gene the prior does not score. Returns the common pairs."""
    lines = list(getattr(data.split, split))
    truth = data.truth(lines) * 0.5
    frame = truth.stack().rename("residual").reset_index()
    frame.columns = ["model_id", "gene_symbol", "residual"]
    frame["residual_prediction"] = frame["residual"] + 0.1
    frame = pd.concat(
        [
            frame.iloc[1:],
            pd.DataFrame(
                {
                    "model_id": [lines[0]],
                    "gene_symbol": ["OTHER"],
                    "residual": [0.0],
                    "residual_prediction": [0.0],
                }
            ),
        ]
    ).assign(method=TX1)
    directory = run_dir / "baselines" / split
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    metrics = {f"{split}_selective_spearman": 0.1, f"{split}_geneeffect_loss": 0.2}
    (directory / "metrics.json").write_text(
        json.dumps({TX1: metrics, "gene_mean": metrics})
    )
    return len(frame) - 1  # the OTHER row


def test_test_scores_against_controls_and_the_summary_renders(tmp_path):
    data = synthetic_run(tmp_path)
    pairs = {split: _write_controls(data, tmp_path, split) for split in ("val", "test")}
    record = run.run_test(data, SPEC, None, tmp_path)
    assert list(record["splits"]) == ["val", "test", "oracle_val"]
    assert set(record["splits"]["oracle_val"]) == {"prior"}  # never a comparison
    for split in ("val", "test"):
        gain = record["splits"][split]["prior_minus_tx1_ridge"]
        assert gain["pairs"] == pairs[split] and len(gain["interval"]) == 2
    assert {row["split"] for row in record["lineages"]} == {"val", "test"}
    scored = pd.read_parquet(tmp_path / "predictions.parquet")
    assert set(scored["split"]) == {"val", "test", "oracle_val"}
    run._write_json(tmp_path / "metrics.json", record)
    run._write_json(
        tmp_path / "curve.json",
        [
            {
                "point": "all",
                "lines": 97,
                "config": "components",
                "input": "val",
                "penalties": [0.1],
                "score": float("nan"),
            },
        ],
    )
    run._write_json(
        tmp_path / "decision.json",
        {
            "input": "val",
            "binding": True,
            "config": "components_and_selected",
            "lines": {"all": 97, "single_cell_train": 37},
            "difference": 0.01,
            "interval": [-0.01, 0.03],
            "passes": False,
            "oracle_difference": 0.02,
            "oracle_interval": [0.01, 0.03],
            "bridge_failing": True,
        },
    )
    run._write_json(
        tmp_path / "selection.json",
        {
            "spec": run._spec_json(SPEC),
            "log": [
                {
                    "block": "expression_components",
                    "penalty": 0.1,
                    "selected": 0,
                    "rank": None,
                    "score": 0.2,
                    "interval": None,
                    "kept": True,
                },
                {
                    "block": "own_expression",
                    "penalty": math.inf,
                    "selected": 0,
                    "rank": None,
                    "score": 0.21,
                    "interval": [-0.01, 0.02],
                    "kept": False,
                },
                {
                    "block": "reduced_rank",
                    "penalty": None,
                    "selected": 0,
                    "rank": 2,
                    "score": float("nan"),
                    "interval": [float("nan"), float("nan")],
                    "kept": False,
                },
            ],
        },
    )
    run._write_json(
        tmp_path / "crossfit.json",
        {
            "bridge_quality": {"median": 0.5, "defined": 16, "undefined": 0},
            "view_weights": {"tried": False, "kept": False},
        },
    )
    summary = run.write_summary(tmp_path).read_text()
    for heading in (
        "Learning curve",
        "Extra-lines decision",
        "Block selection",
        "Cross-fitting",
        "## Validation",
        "## Test",
        "off-contract",
        "Per lineage",
        "bridge is failing",
        "shrinkage inf",
    ):
        assert heading in summary
    assert "| Context-PCA ridge (Tx1) |" in summary


def test_json_values_are_plain():
    payload = {
        "a": np.float64(0.5),
        "b": np.int64(3),
        "c": np.bool_(True),
        "d": [float("nan"), math.inf],
        ("e"): np.array([1.0, 2.0]),
    }
    assert run._jsonable(payload) == {
        "a": 0.5,
        "b": 3,
        "c": True,
        "d": [None, "inf"],
        "e": [1.0, 2.0],
    }
    with pytest.raises(TypeError):
        run._jsonable({"x": object()})


def test_a_run_directory_belongs_to_one_config_and_mode(tmp_path):
    run._bind(tmp_path, CONFIG, oracle_only=True)
    run._bind(tmp_path, copy.deepcopy(CONFIG), oracle_only=True)
    with pytest.raises(ValueError, match="different config"):
        run._bind(tmp_path, CONFIG, oracle_only=False)
    changed = copy.deepcopy(CONFIG)
    changed["curve"]["selected"] = 10
    with pytest.raises(ValueError, match="different config"):
        run._bind(tmp_path, changed, oracle_only=True)


def test_run_data_drops_test_excluded_and_dropped_lineage_lines(monkeypatch):
    from src.data.extra_lines import ExtraLines

    rng = np.random.default_rng(0)
    single = [f"S{i}" for i in range(30)]
    extras, unlabelled = [f"X{i}" for i in range(20)], [f"Y{i}" for i in range(10)]
    val, test = [f"V{i}" for i in range(6)], [f"T{i}" for i in range(6)]
    blood = ["XB0", "YB0"]  # a labelled and an unlabelled haematopoietic extra
    everyone = [*single, "U0", *extras, *unlabelled, *blood, *val, *test, "Z0"]
    bulk = pd.DataFrame(
        rng.normal(size=(len(everyone), len(SPACE))), index=everyone, columns=SPACE
    ).drop(index=["S29", "V5"])  # a training and a validation line without bulk
    gene_effect = pd.DataFrame(
        rng.normal(size=(len(everyone), 3)), index=everyone, columns=["E0", "E1", "Q"]
    ).drop(index=["U0", *unlabelled, "YB0"])
    models = pd.DataFrame(
        {
            "patient_id": everyone,
            "lineage": ["Myeloid" if m in blood else "Lung" for m in everyone],
        },
        index=everyone,
    )
    split = FixedSplit(
        train=(*single, "U0"), val=tuple(val), test=tuple(test), unlabeled_train=("U0",)
    )
    features = {
        "variable_gene_min_observations": 3,
        "variable_gene_percentile": 50,
        "selective_min_lines": 1,
        "selective_max_fraction": 0.9,
        "residual_sd_floor_percentile": 10,
    }
    joint = {"paths": {"split": "split", "gene_effect": "effect"}, "features": features}
    blocked = []
    monkeypatch.setattr(run, "load_config", lambda path: joint)
    monkeypatch.setattr(run, "load_geneeffect_226_split", lambda path: split)
    monkeypatch.setattr(
        run,
        "load_extra_lines",
        lambda path, split: ExtraLines(
            (*extras, "XB0"), (*unlabelled, "YB0"), {"Z0": "shares a patient"}
        ),
    )
    monkeypatch.setattr(run, "read_models", lambda path: models)
    monkeypatch.setattr(
        run,
        "read_depmap_matrix",
        lambda path: (bulk if str(path) == "bulk" else gene_effect).copy(),
    )
    monkeypatch.setattr(
        run, "load_reference", lambda path, *, blocked: blocked_seen(blocked)
    )

    def blocked_seen(lines):
        blocked.extend(lines)
        return None

    config = copy.deepcopy(CONFIG)
    config["paths"]["bulk_expression"] = "bulk"
    config["prior"]["components"] = 6
    config["training_side"]["exclude_lineages"] = ["Myeloid", "Lymphoid"]
    data = run.load_run_data(config, oracle_only=True)
    side = [*single[:-1], "U0", *extras, *unlabelled]
    assert list(data.inputs.expression.index) == side
    assert data.labelled == (*single[:-1], *extras)
    assert data.single_cell_train == tuple(single[:-1])
    assert list(data.inputs.residual.index) == list(data.labelled)
    assert list(data.queries) == ["oracle"] and data.pseudobulk is None
    assert list(data.queries["oracle"].index) == val[:-1]
    assert set(blocked) == {*val, *test, "Z0"}
    assert "Z0" not in data.gene_effect.index and "XB0" not in data.gene_effect.index
    assert data.definitions.genes == ("E0", "E1")  # Q has no bulk column

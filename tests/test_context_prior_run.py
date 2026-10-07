"""Prior config schema, long frames, JSON, the run data, and the experiment runner
on synthetic lines: rows per setting, the reference gain, resuming and
``--experiments``."""

from __future__ import annotations

import copy
import math
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.context_prior.reference import Reference
from src.context_prior.space import quantile_reference
from src.context_prior.targets import Definitions
from src.data.splits import FixedSplit
from src.experiments import context_prior as run
from src.experiments.config import validate_prior_config
from tests.test_context_prior_bridging import GENES, base

CONFIG = yaml.safe_load(Path("configs/context_prior/bridge_remedies.yaml").read_text())
TARGETS = GENES[:8]


def test_prior_config_is_strict():
    validate_prior_config(CONFIG)
    unknown = copy.deepcopy(CONFIG)
    unknown["experiments"]["affine"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_prior_config(unknown)
    missing = copy.deepcopy(CONFIG)
    del missing["prior"]["selected_genes"]
    with pytest.raises(ValueError, match="missing"):
        validate_prior_config(missing)
    unordered = copy.deepcopy(CONFIG)
    unordered["block_sets"]["selected"] = ["data_selected", "expression_components"]
    with pytest.raises(ValueError, match="expression_components first"):
        validate_prior_config(unordered)
    unnamed = copy.deepcopy(CONFIG)
    unnamed["reference"]["experiment"] = "nothing"
    with pytest.raises(ValueError, match="reference"):
        validate_prior_config(unnamed)
    # A gene penalty belongs to a gene-level reference row and only to one.
    gene_level = copy.deepcopy(CONFIG)
    gene_level["reference"]["block_set"] = "all"
    with pytest.raises(ValueError, match="gene_penalty"):
        validate_prior_config(gene_level)
    gene_level["reference"]["gene_penalty"] = 10.0
    validate_prior_config(gene_level)


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


def test_a_run_directory_belongs_to_one_config(tmp_path):
    run._bind(tmp_path, CONFIG)
    run._bind(tmp_path, copy.deepcopy(CONFIG))
    changed = copy.deepcopy(CONFIG)
    changed["prior"]["selected_genes"] = 10
    with pytest.raises(ValueError, match="different config"):
        run._bind(tmp_path, changed)


def test_shared_files_wait_for_the_run_directory_lock(tmp_path):
    """Two processes share a run directory: the config check and a results scan
    with its replacement each run under the lock, so neither interleaves."""
    run._bind(tmp_path, CONFIG)
    (tmp_path / "rows").mkdir()
    changed = copy.deepcopy(CONFIG)
    changed["prior"]["selected_genes"] = 10
    errors = []

    def bind():
        try:
            run._bind(tmp_path, changed)
        except ValueError as error:
            errors.append(error)

    with run._locked(tmp_path):
        writer = threading.Thread(target=run.write_results, args=(tmp_path,))
        binder = threading.Thread(target=bind)
        writer.start()
        binder.start()
        writer.join(0.3)
        binder.join(0.3)
        assert writer.is_alive() and binder.is_alive()
        assert not (tmp_path / "results.md").exists()
    writer.join(5)
    binder.join(5)
    assert (tmp_path / "results.md").is_file()
    assert len(errors) == 1


# ----------------------------------------------------------------------------
# The run data
# ----------------------------------------------------------------------------

SPACE = [f"E{i}" for i in range(16)]


def test_run_base_drops_test_excluded_and_dropped_lineage_lines(monkeypatch):
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
    scored = [*single, "U0", *val, *test]
    pseudo = pd.DataFrame(
        rng.normal(size=(len(scored), len(SPACE))), index=scored, columns=SPACE
    )
    pseudo.loc["S0", "E15"] = np.nan  # a training line's source lacks E15: filled
    pseudo.loc["V0", "E14"] = np.nan  # a validation line lacks E14: out of the space
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
    joint = {
        "paths": {"split": "split", "gene_effect": "effect"},
        "features": features,
        "prepared_root": "prepared",
    }
    seen, prepared = [], []
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

    def reference(path, *, blocked):
        seen.extend(blocked)

    monkeypatch.setattr(run, "load_reference", reference)
    monkeypatch.setattr(
        "src.data.prepared.read_manifest",
        lambda root: {"common_gene_panel": ["E0", "E1", "Q"]},
    )
    monkeypatch.setattr(
        "src.experiments.prepare.prepare_pseudobulk",
        lambda config, genes: prepared.append(list(genes)),
    )
    monkeypatch.setattr("src.data.pseudobulk.read_pseudobulk", lambda root: pseudo)

    config = copy.deepcopy(CONFIG)
    config["paths"]["bulk_expression"] = "bulk"
    config["training_side"]["exclude_lineages"] = ["Myeloid", "Lymphoid"]
    loaded = run.load_base(config)
    side = [*single[:-1], "U0", *extras, *unlabelled]
    space = [g for g in SPACE if g != "E14"]
    bridge = loaded.bridge
    assert list(bridge.bulk.index) == side
    assert list(bridge.bulk.columns) == space == list(bridge.pseudobulk.columns)
    assert list(bridge.oracle.index) == val[:-1]
    assert loaded.labelled == (*single[:-1], *extras)
    assert bridge.paired == tuple(single[:-1])
    assert bridge.single_cell_train == tuple(single)  # S29 without bulk included
    assert set(bridge.folds) == set(single)
    assert bridge.val == tuple(val) and bridge.test == tuple(test)
    assert loaded.filled_lines == 1
    # Every normalised row takes the training side's reference profile.
    profile = quantile_reference(bulk.loc[side, space])
    assert np.allclose(np.sort(bridge.oracle.to_numpy(), axis=1), profile)
    assert np.allclose(np.sort(bridge.pseudobulk.to_numpy(), axis=1), profile)
    assert set(seen) == {*val, *test, "Z0"}
    assert prepared == [list(bulk.columns)]
    assert "Z0" not in loaded.gene_effect.index
    assert "XB0" not in loaded.gene_effect.index
    assert loaded.definitions.genes == ("E0", "E1", "Q")


# ----------------------------------------------------------------------------
# The runner on synthetic lines
# ----------------------------------------------------------------------------


def tiny_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(CONFIG)
    config["output_root"] = str(tmp_path / "out")
    config["prior"] = {
        "components": 4,
        "folds": 4,
        "bootstrap_repeats": 20,
        "selected_genes": 2,
    }
    config["experiments"] = {
        "affine": copy.deepcopy(CONFIG["experiments"]["affine"]),
        "affine_again": {
            "kind": "affine",
            "settings": [{}],
            "block_sets": ["components", "selected"],
        },
    }
    return validate_prior_config(config)


def synthetic_base(config) -> run.RunBase:
    """The bridging test's lines; labels follow expression with noise."""
    bridge = base()
    extras = [m for m in bridge.bulk.index if m not in set(bridge.paired)]
    expression = pd.concat([bridge.pseudobulk, bridge.bulk.loc[extras]])
    lines = list(expression.index)
    rng = np.random.default_rng(1)
    effect = (
        -0.5
        + 0.5 * expression.loc[:, GENES[1:9]].to_numpy()
        + 0.2 * rng.normal(size=(len(lines), len(TARGETS)))
    )
    gene_effect = pd.DataFrame(effect, index=lines, columns=TARGETS)
    gene_effect.iloc[0, 0] = np.nan  # a missing training label
    definitions = Definitions(
        genes=tuple(TARGETS),
        gene_means=pd.Series(-0.5, index=TARGETS),
        selective=tuple(TARGETS[:6]),
        variable=tuple(TARGETS),
        residual_scale=pd.Series(0.5, index=TARGETS),
    )
    side = pd.Index(list(bridge.bulk.index), name="model_id")
    reference = Reference(
        paralogs=pd.DataFrame(
            {"gene": ["G0", "G2"], "paralog": ["G1", "G3"], "identity": [50.0, 40.0]}
        ),
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["G4", "G5"]}),
        hallmark=pd.DataFrame({"gene_set": ["S"], "gene": ["G7"]}),
        progeny=pd.DataFrame({"pathway": ["P"], "gene": ["G8"], "weight": [1.0]}),
        drivers=pd.DataFrame({"KRAS": 0}, index=side),
        msi=pd.Series(0.0, index=side),
    )
    return run.RunBase(
        config=config,
        split=FixedSplit(
            train=bridge.single_cell_train, val=bridge.val, test=bridge.test
        ),
        gene_effect=gene_effect,
        definitions=definitions,
        labelled=(*bridge.paired, *extras),
        models=pd.DataFrame({"patient_id": lines, "lineage": "Lung"}, index=lines),
        reference=reference,
        bridge=bridge,
        filled_lines=0,
    )


def test_run_setting_scores_every_block_set_and_penalty(tmp_path):
    config = tiny_config(tmp_path)
    synthetic = synthetic_base(config)
    reference = run.reference_predictions(synthetic)
    assert set(reference) == {"val", "test"}
    result = run.run_setting(synthetic, config["experiments"]["affine"], {}, reference)
    rows = result["rows"]
    components = [row for row in rows if row["block_set"] == "components"]
    assert [row["components_penalty"] for row in components] == [
        float(p) for p in config["penalties"]["components"]
    ]
    assert len(rows) == len(components) + 3 * len(components) * 3
    for row in rows:
        assert set(row["val"]) == set(run.METRICS) == set(row["oracle"])
        assert set(row["test"]) == set(run.METRICS)
        for split in ("val", "test"):
            assert len(row[f"{split}_gain"]["interval"]) == 2
    pinned = [row for row in components if row["components_penalty"] == 10.0]
    (best,) = pinned
    assert set(result) == {"rows", "diagnostics"}  # nothing chosen
    gene_level = [row for row in rows if row["block_set"] != "components"]
    assert {row["block_set"] for row in gene_level} == {
        "own_and_partners",
        "selected",
        "all",
    }
    # Every components penalty pairs with every gene penalty.
    pairs = [(row["components_penalty"], row["gene_penalty"]) for row in gene_level]
    grid = [
        (float(c), g)
        for c in config["penalties"]["components"]
        for g in (0.1, 1.0, 10.0)
    ]
    assert pairs == grid * 3
    # The reference row is the row the config pins.
    for split in ("val", "test"):
        assert best[f"{split}_gain"] == {"difference": 0.0, "interval": [0.0, 0.0]}
    # A gain is the row's selective Spearman minus the reference row's.
    for row in rows:
        for split in ("val", "test"):
            expected = (
                row[split]["selective_spearman"] - best[split]["selective_spearman"]
            )
            assert np.isclose(row[f"{split}_gain"]["difference"], expected)
    assert any(row["val_gain"]["difference"] != 0.0 for row in gene_level)
    diagnostics = result["diagnostics"]
    assert diagnostics["all"]["total"] == len(GENES)
    assert diagnostics["selective"]["total"] == 6
    assert "gene_space" not in diagnostics


def test_gains_are_measured_against_a_pinned_gene_level_row(tmp_path):
    config = tiny_config(tmp_path)
    config["reference"] = {
        "experiment": "affine",
        "block_set": "all",
        "components_penalty": 1.0,
        "gene_penalty": 1.0,
    }
    config = validate_prior_config(config)
    synthetic = synthetic_base(config)
    reference = run.reference_predictions(synthetic)
    rows = run.run_setting(synthetic, config["experiments"]["affine"], {}, reference)[
        "rows"
    ]
    (pinned,) = [
        row
        for row in rows
        if (row["block_set"], row["components_penalty"], row["gene_penalty"])
        == ("all", 1.0, 1.0)
    ]
    for split in ("val", "test"):
        assert pinned[f"{split}_gain"] == {"difference": 0.0, "interval": [0.0, 0.0]}
        for row in rows:
            expected = (
                row[split]["selective_spearman"] - pinned[split]["selective_spearman"]
            )
            assert np.isclose(row[f"{split}_gain"]["difference"], expected)


def _counted(monkeypatch, config):
    calls = {"load_base": 0, "run_setting": 0}
    original = run.run_setting

    def load(loaded_config):
        calls["load_base"] += 1
        return synthetic_base(loaded_config)

    def counted(*args):
        calls["run_setting"] += 1
        return original(*args)

    monkeypatch.setattr(run, "load_base", load)
    monkeypatch.setattr(run, "run_setting", counted)
    return calls


def test_main_writes_rows_and_results_and_skips_existing_rows(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = _counted(monkeypatch, config)
    assert run.main([str(path), "--run-id", "r"]) == 0
    run_dir = tmp_path / "out" / "r"
    assert sorted(p.name for p in (run_dir / "rows").iterdir()) == [
        "affine__0.json",
        "affine_again__0.json",
    ]
    record = run._read_json(run_dir / "rows" / "affine__0.json")
    assert record["experiment"] == "affine" and record["setting"] == {}
    assert len(record["rows"]) == 5 + 3 * 5 * 3
    results = (run_dir / "results.md").read_text()
    for text in ("## affine", "## affine_again", "own_and_partners", "Oracle"):
        assert text in results
    assert calls == {"load_base": 1, "run_setting": 2}
    assert run.main([str(path), "--run-id", "r"]) == 0
    assert calls == {"load_base": 1, "run_setting": 2}  # every row exists


def test_experiments_limit_the_run_and_share_a_run_directory(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = _counted(monkeypatch, config)
    rows = tmp_path / "out" / "r" / "rows"
    assert run.main([str(path), "--run-id", "r", "--experiments", "affine_again"]) == 0
    assert [p.name for p in rows.iterdir()] == ["affine_again__0.json"]
    assert calls["run_setting"] == 1
    results = (tmp_path / "out" / "r" / "results.md").read_text()
    assert "## affine_again" in results and "## affine\n" not in results
    # Another process on the same run directory with other experiments.
    assert run.main([str(path), "--run-id", "r", "--experiments", "affine"]) == 0
    assert sorted(p.name for p in rows.iterdir()) == [
        "affine__0.json",
        "affine_again__0.json",
    ]
    results = (tmp_path / "out" / "r" / "results.md").read_text()
    assert "## affine\n" in results and "## affine_again" in results
    with pytest.raises(ValueError, match="unknown experiments"):
        run.main([str(path), "--run-id", "r", "--experiments", "nothing"])

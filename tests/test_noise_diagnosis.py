"""The noise diagnosis: its statistics, its config, and every section on
synthetic lines read through the real prior and export code."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.context_prior import noise
from src.experiments import noise_diagnosis as run
from src.experiments.config import validate_noise_config, validate_prior_config
from src.experiments.prior_export import export_predictions
from tests.test_context_prior_run import synthetic_base, tiny_config

CONFIG = yaml.safe_load(Path("configs/context_prior/noise_diagnosis.yaml").read_text())


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------


def test_column_pearson_uses_rows_finite_in_both():
    left = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 1.0, 1.0, 2.0]})
    right = pd.DataFrame({"a": [2.0, 4.0, np.nan, 8.0], "b": [5.0, 5.0, 5.0, 5.0]})
    rho = noise.column_pearson(left, right)
    assert rho["a"] == pytest.approx(1.0)
    assert np.isnan(rho["b"])  # constant on one side: undefined, not zero
    assert noise.spearman_brown(0.5) == pytest.approx(2 / 3)


def test_fractions_nest_and_keep_patients_whole():
    lines = [f"L{i}" for i in range(20)]
    patients = {m: f"P{i // 2}" for i, m in enumerate(lines)}
    kept = noise.nested_fractions(lines, patients, [0.2, 0.5, 1.0], seed=0)
    assert [len(kept[f]) for f in (0.2, 0.5, 1.0)] == [4, 10, 20]
    assert set(kept[0.2]) <= set(kept[0.5]) <= set(kept[1.0])
    for chosen in kept.values():
        assert {patients[m] for m in chosen} == {
            patients[m] for m in lines if patients[m] in {patients[c] for c in chosen}
        }
        assert all(sum(patients[c] == patients[m] for c in chosen) == 2 for m in chosen)


def test_group_share_finds_a_planted_deposit_effect_and_not_a_null_one():
    rng = np.random.default_rng(1)
    lines = [f"L{i}" for i in range(60)]
    deposit = pd.Series(np.repeat(["d0", "d1", "d2"], 20), index=lines)
    lineage = pd.Series(np.tile(["x", "y"], 30), index=lines)
    frame = pd.DataFrame(rng.normal(size=(60, 30)), index=lines)
    plain = noise.group_share(frame, deposit, lineage, permutations=200, seed=0)
    shifted = frame.add(deposit.map({"d0": 0.0, "d1": 1.0, "d2": -1.0}), axis=0)
    shifted.iloc[3, 2] = np.nan  # a missing value is left out of its column
    planted = noise.group_share(shifted, deposit, lineage, permutations=200, seed=0)
    assert plain["p_value"] > 0.05
    assert planted["p_value"] < 0.01 and planted["observed"] > 0.3
    assert planted["groups"] == 3 and planted["lines"] == 60


def test_thresholds_weight_lines_equally_and_ignore_held_out_lines():
    rng = np.random.default_rng(0)
    scores = {
        "T0": rng.normal(size=(1000, 2)),
        "T1": rng.normal(loc=3.0, size=(10, 2)),  # few cells, equal weight
        "V0": rng.normal(size=(50, 2)),
    }
    fractions, means, thresholds = noise.program_features(
        scores, ["T0", "T1"], ["P", "Q"], 50.0
    )
    # Half the weight sits at 0 and half at 3: the median lies between them.
    assert 0.5 < thresholds[0] < 3.0
    moved = {**scores, "V0": scores["V0"] + 100.0}
    _, _, again = noise.program_features(moved, ["T0", "T1"], ["P", "Q"], 50.0)
    assert np.array_equal(again, thresholds)
    assert list(fractions.index) == ["T0", "T1", "V0"]
    assert np.allclose(means.loc["V0"], scores["V0"].mean(axis=0))


def test_cluster_proportions_ignore_line_identity_and_held_out_lines():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(3, 6)) * 4

    def line(offset: float, n: int = 90) -> np.ndarray:
        labels = np.arange(n) % 3
        return states[labels] + offset + 0.1 * rng.normal(size=(n, 6))

    cells = {"T0": line(0.0), "T1": line(5.0), "T2": line(-3.0), "V0": line(50.0)}
    result = noise.cluster_proportions(
        cells, ["T0", "T1", "T2"], components=3, counts=[3], seed=0
    )[3]
    assert np.allclose(result.sum(axis=1), 1.0)
    # Every line holds the three states equally, whatever its offset.
    assert np.allclose(result.to_numpy(), 1 / 3, atol=0.02)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


def test_noise_config_is_strict():
    validate_noise_config(CONFIG)
    unknown = copy.deepcopy(CONFIG)
    unknown["cell_states"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_noise_config(unknown)
    whole = copy.deepcopy(CONFIG)
    whole["learning_curve"]["fractions"] = [0.5, 1.0]  # the full prior is a row anyway
    with pytest.raises(ValueError, match="fractions"):
        validate_noise_config(whole)
    repeated = copy.deepcopy(CONFIG)
    repeated["cell_states"]["programs"] = ["S", "S"]
    with pytest.raises(ValueError, match="programs"):
        validate_noise_config(repeated)


def test_config_programs_are_pinned_and_none_has_a_cell_line_study():
    root = Path("configs/context_prior/reference")
    sources = pd.read_csv(root / "meta_program_sources.tsv", sep="\t")
    clean = set(sources.loc[sources["cell_line_studies"] == 0, "program"])
    programs = CONFIG["cell_states"]["programs"]
    assert programs[:2] == ["S", "G2M"]
    assert set(programs[2:]) == clean
    base = type("Base", (), {"config": {"paths": {"reference": str(root)}}})()
    members = run.program_members(base, CONFIG)
    assert list(members) == programs
    assert len(members["S"]) == 43 and len(members["G2M"]) == 54
    assert all(len(genes) >= 48 for name, genes in members.items() if name[:2] == "MP")


# ----------------------------------------------------------------------------
# Sections on synthetic lines
# ----------------------------------------------------------------------------


class SyntheticDiagnosis(run.Diagnosis):
    """A diagnosis whose prepared files are frames held in memory."""

    raw: pd.DataFrame
    draws: pd.DataFrame

    def raw_pseudobulk(self) -> pd.DataFrame:
        return self.raw

    def subsamples(self) -> pd.DataFrame:
        return self.draws


def _draws(raw: pd.DataFrame, lines, seed: int = 4) -> pd.DataFrame:
    """Halves and two cell draws: the line's pseudo-bulk plus sampling noise that
    shrinks with the draw's size; the last validation line has too few cells for
    the 30-cell draw."""
    rng = np.random.default_rng(seed)
    frames = []
    for name, scale in (
        ("half_a", 0.3),
        ("half_b", 0.3),
        ("cells_10", 0.6),
        ("cells_30", 0.2),
    ):
        rows = [m for m in lines if not (name == "cells_30" and m == "V5")]
        frame = raw.loc[rows] + scale * rng.normal(size=(len(rows), raw.shape[1]))
        frame.index = pd.MultiIndex.from_tuples(
            [(m, name) for m in rows], names=["model_id", "draw"]
        )
        frames.append(frame)
    return pd.concat(frames)


@pytest.fixture
def diagnosis(tmp_path) -> SyntheticDiagnosis:
    config = tiny_config(tmp_path)
    config["reference"] = {
        "experiment": "affine",
        "block_set": "selected",
        "components_penalty": 1.0,
        "gene_penalty": 1.0,
    }
    config = validate_prior_config(config)
    base = synthetic_base(config)
    raw = base.bridge.pseudobulk.copy()  # the synthetic prepared transform
    base = replace(
        base, bridge=replace(base.bridge, pseudobulk=base.normalize_pseudobulk(raw))
    )
    predictions = export_predictions(base)
    export = pd.concat([predictions[s] for s in ("train", "val", "test")])
    noise_config = copy.deepcopy(CONFIG)
    noise_config.update(bootstrap_repeats=20)
    noise_config["cell_sampling"]["cells"] = [10, 30]
    noise_config["learning_curve"]["fractions"] = [0.5, 0.75]
    noise_config["deposit"].update(permutations=50, min_lines=3)
    noise_config["cell_states"].update(
        programs=["P0", "P1", "P2"], cluster_components=3, cluster_counts=[2, 3]
    )
    found = SyntheticDiagnosis(validate_noise_config(noise_config), base, export, {})
    found.raw = raw
    found.draws = _draws(raw, list(base.split.all_model_ids))
    return found


def test_cell_sampling_reports_reliability_and_every_draw(diagnosis, tmp_path):
    result = run.cell_sampling(diagnosis, tmp_path)
    genes = pd.read_csv(tmp_path / "cell_sampling_genes.csv", index_col="gene")
    assert list(genes.index) == list(diagnosis.base.bridge.pseudobulk.columns)
    assert genes["reliability"].median() > 0.5  # halves differ by small noise
    assert set(result["draws"]) == {"half_a", "half_b", "cells_10", "cells_30"}
    assert result["draws"]["cells_30"]["val"]["capped_lines"] == 1
    assert result["draws"]["cells_10"]["val"]["capped_lines"] == 0
    for draw in result["draws"].values():
        for split in ("val", "test"):
            assert len(draw[split]["gain"]["interval"]) == 2
    assert set(result["prediction_agreement_between_halves"]) == {"val", "test"}


def test_the_reference_fit_must_reproduce_the_export(diagnosis):
    run.reference_fit(diagnosis)
    diagnosis.export.loc[list(diagnosis.base.bridge.val)] += 0.01
    with pytest.raises(ValueError, match="differ from the export"):
        run.reference_fit(diagnosis)


def test_learning_curve_ends_at_the_prior_itself(diagnosis, tmp_path):
    rows = run.learning_curve(diagnosis, tmp_path)["rows"]
    assert [row["fraction"] for row in rows] == [0.5, 0.75, 1.0]
    sizes = [row["training_side_lines"] for row in rows]
    assert sizes == sorted(sizes) and sizes[-1] == len(diagnosis.base.bridge.bulk)
    for split in ("val", "test"):
        assert rows[-1][split]["gain"]["difference"] == 0.0


def test_deposit_confound_reports_both_groupings(diagnosis, tmp_path, monkeypatch):
    train = diagnosis.train
    deposits = pd.Series(
        ["big"] * 10 + ["mid"] * 6 + ["tiny"] * 2 + ["mid"] * (len(train) - 18),
        index=train,
    )
    monkeypatch.setattr(run, "_deposits", lambda found: deposits)
    result = run.deposit_confound(diagnosis, tmp_path)
    assert result["pooled_as_other"] == ["tiny"]
    assert set(result["per_deposit"]) == {"big", "mid", "other"}
    for of in ("error", "residual"):
        assert set(result["shares"][of]) == {"deposit", "lineage"}


def test_deposit_swap_reads_each_line_four_ways(diagnosis, tmp_path, monkeypatch):
    lines = diagnosis.train[:12]
    second = diagnosis.raw.loc[lines] + 0.5
    (tmp_path / "second").mkdir()
    second.to_parquet(tmp_path / "second" / "pseudobulk.parquet")
    monkeypatch.setattr(
        "src.experiments.prepare.prepare_second_source",
        lambda joint, genes, root: tmp_path / "second" / "manifest.json",
    )
    monkeypatch.setattr(
        run, "_deposits", lambda found: pd.Series("d0", index=diagnosis.train)
    )
    monkeypatch.setattr(
        run.Diagnosis, "pseudobulk_genes", lambda self: list(self.raw.columns)
    )
    result = run.deposit_swap(diagnosis, tmp_path)
    assert result["lines"] == 12
    assert set(result["scores"]) == {"primary", "second", "half_a", "half_b"}
    assert result["scores"]["primary"]["gain"]["difference"] == 0.0
    # A constant shift survives quantile normalisation only as ranks: the inputs
    # of the second deposit read like the primary's.
    assert result["input_agreement"]["primary_second"]["median"] > 0.99


def test_cell_states_stack_every_arm_on_the_export(diagnosis, tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    every = [*diagnosis.train, *diagnosis.base.bridge.val, *diagnosis.base.bridge.test]
    monkeypatch.setattr(
        run,
        "_program_scores",
        lambda found, lines: {m: rng.normal(size=(40, 3)) for m in lines},
    )
    monkeypatch.setattr(
        run,
        "_context_cells",
        lambda found, lines: {
            m: rng.normal(size=(30, 8)).astype(np.float32) for m in lines
        },
    )
    result = run.cell_states(diagnosis, tmp_path)
    assert set(result["arms"]) == {
        "program_fractions",
        "program_means",
        "clusters_2",
        "clusters_3",
    }
    assert result["features"]["clusters_3"] == 3
    for arm, rows in result["arms"].items():
        assert [row["penalty"] for row in rows] == CONFIG["cell_states"]["penalties"]
        for row in rows:
            assert ("gain_over_program_means" in row["val"]) == (
                arm == "program_fractions"
            )
    assert len(every) == len(set(every))


def test_screen_replicates_pair_two_screens_outside_held_out_lines(diagnosis, tmp_path):
    rng = np.random.default_rng(6)
    genes = list(diagnosis.base.definitions.selective)
    models = [f"M{i}" for i in range(30)] + ["V0"]
    signal = rng.normal(size=(len(models), len(genes)))
    screens, rows = [], []
    for k, model in enumerate(models):
        for replicate in range(2 + (k == 0)):
            screen = f"SC{k:02d}{replicate}"
            screens.append(
                pd.Series(signal[k] + 0.3 * rng.normal(size=len(genes)), name=screen)
            )
            rows.append({"ScreenID": screen, "ModelID": model})
    frame = pd.DataFrame(screens)
    frame.columns = [f"{g} (1)" for g in genes]
    frame.to_csv(tmp_path / "screens.csv")
    pd.DataFrame(rows).to_csv(tmp_path / "map.csv", index=False)
    diagnosis.config["screen_replicates"] = {
        "screen_gene_effect": str(tmp_path / "screens.csv"),
        "screen_map": str(tmp_path / "map.csv"),
    }
    result = run.screen_replicates(diagnosis, tmp_path)
    assert result["models"] == 30  # V0 is a validation line
    assert result["agreement"]["median"] > 0.8
    assert result["ceiling_macro"] >= result["agreement_macro"]
    pd.DataFrame([{"ScreenID": "x"}]).to_csv(tmp_path / "map.csv", index=False)
    with pytest.raises(ValueError, match="ModelID"):
        run.screen_replicates(diagnosis, tmp_path)


def test_main_writes_sections_and_skips_finished_ones(diagnosis, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "load_diagnosis", lambda config: diagnosis)
    monkeypatch.setitem(
        run.RUNNERS,
        "learning_curve",
        lambda found, out: (
            calls.append("learning_curve") or run.learning_curve(found, out)
        ),
    )
    config = copy.deepcopy(diagnosis.config)
    config["output_root"] = str(tmp_path / "out")
    path = tmp_path / "noise.yaml"
    path.write_text(yaml.safe_dump(config))
    args = [str(path), "--run-id", "r", "--sections", "learning_curve"]
    assert run.main(args) == 0
    assert run.main(args) == 0
    assert calls == ["learning_curve"]
    out = tmp_path / "out" / "r" / "noise"
    assert "## Learning curve" in (out / "results.md").read_text()
    assert json.loads((out / "learning_curve.json").read_text())["rows"]
    with pytest.raises(ValueError, match="unknown sections"):
        run.main([str(path), "--run-id", "r", "--sections", "nothing"])

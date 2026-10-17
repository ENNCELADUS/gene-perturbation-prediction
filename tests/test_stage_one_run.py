"""Stage-one runs on a synthetic ladder: the reader's guards and targets, the
profile delta's states, a training run with its fold, the prior reference, and the
report.

The ladder is the stage-one inputs test's world (``build_world``, then the
``lines``, ``cells`` and ``mean`` steps), with synthetic frozen-PIE arrays, gene
sources and Cellosaurus text written beside them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data import ladder_reader as reader
from src.experiments import ladder_inputs as ladder
from src.experiments import stage_one
from src.experiments.config import load_stage_one_config
from tests.test_ladder_inputs import BULK_LABELLED, TEST, TRAIN, VAL, build_world

WIDTHS = {"dH": 6, "dL": 6, "own_h": 6, "own_dp": 1, "own_x": 1}
TEXT_DIM, GENE_DIM = 5, 6
CONFIGS = Path("configs/stage_one/architecture")


def _dense(path: Path, name: str, keys, vectors: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embeddings.npy", vectors.astype(np.float32))
    (path / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "layout": "dense",
                "dtype": "float32",
                "dim": int(vectors.shape[1]),
                "keys": list(keys),
            }
        )
    )


def _write_pie(root: Path, rng: np.random.Generator) -> np.ndarray:
    """Frozen-PIE arrays as ``assemble`` writes them; returns the full predicted
    changes (panel genes x lines x panel) behind ``dE``."""
    tables = reader.read_ladder(root)
    panel, lines = tables.panel, tables.line_ids
    frame = np.isin(lines, tables.frame)
    pie = root / "pie"
    pie.mkdir()
    arrays, means = {}, {}
    for name, width in WIDTHS.items():
        raw = rng.normal(size=(len(panel), len(lines), width)) + rng.normal(
            size=(len(panel), 1, width)
        )
        if name == "own_x":
            raw = reader.pie_input(tables).to_numpy().T[:, :, None]
        arrays[name] = raw
    changes = rng.normal(size=(len(panel), len(lines), len(panel)))
    components = np.linalg.qr(rng.normal(size=(len(panel), len(panel))))[0].T
    mean = changes[:, frame].reshape(-1, len(panel)).mean(axis=0)
    arrays["dE"] = (changes - mean) @ components.T
    shapes = {}
    for name, raw in arrays.items():
        centre = raw[:, frame].mean(axis=1)
        np.save(pie / f"{name}.npy", (raw - centre[:, None]).astype(np.float32))
        means[name] = centre.astype(np.float32)
        shapes[name] = list(raw.shape)
    np.savez(pie / "frame_means.npz", **means)
    np.savez(
        pie / "basis.npz",
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained=np.full(len(panel), 1 / len(panel)),
    )
    (pie / "index.json").write_text(
        json.dumps({"genes": panel, "lines": lines, "arrays": shapes})
    )
    # The full-panel predicted change, kept for the selective genes in the panel;
    # the last one has none (coverage is reported, not assumed).
    selective = [g for g in panel if bool(tables.genes.at[g, "selective"])]
    (pie / "full").mkdir()
    for gene in selective[:-1]:
        np.save(
            pie / "full" / f"{gene}.npy", changes[panel.index(gene)].astype(np.float32)
        )
    return changes


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage_one")
    config = build_world(root)
    rng = np.random.default_rng(3)
    text_dir = Path(config["paths"]["gene_text"])
    keys = json.loads((text_dir / "meta.json").read_text())["keys"]
    _dense(text_dir, "gene_text", keys, rng.normal(size=(len(keys), GENE_DIM)))
    ladder.run(config, ["lines", "cells", "mean"], (0, 1))
    ladder_root = Path(config["ladder_root"])
    changes = _write_pie(ladder_root, rng)
    genes = reader.read_ladder(ladder_root).genes
    present = genes["gene_text_key"].to_numpy() != ""
    vectors = rng.normal(size=(len(genes), GENE_DIM)).astype(np.float32)
    (ladder_root / "gene_sources").mkdir()
    np.savez(
        ladder_root / "gene_sources" / "gene_text.npz",
        axis=np.asarray(genes.index, dtype=str),
        vectors=np.where(present[:, None], vectors, np.nan),
        present=present,
    )
    lines = reader.read_ladder(ladder_root).line_ids
    with_text = [m for m in lines if m != "ACH-T2"]
    _dense(
        ladder_root / "context_text",
        "context_text",
        with_text,
        rng.normal(size=(len(with_text), TEXT_DIM)),
    )
    prior = yaml.safe_load(Path(config["prior_config"]).read_text())
    prior["prior"].update(components=3, selected_genes=2)
    Path(config["prior_config"]).write_text(yaml.safe_dump(prior))
    (root / "ladder.yaml").write_text(yaml.safe_dump(config))
    return root, ladder_root, changes


def _setup(world) -> stage_one.Setup:
    root = world[0]
    return stage_one.load_setup(root / "ladder.yaml")


# --- the reader --------------------------------------------------------------------


def test_a_mismatched_pie_index_fails(world):
    _, ladder_root, _ = world
    tables = reader.read_ladder(ladder_root)
    index = reader.check_pie_index(tables)
    assert reader.open_delta(tables, "dH", index).shape[2] == 6
    shuffled = reader.Ladder(ladder_root, tables.lines.iloc[::-1], tables.genes)
    with pytest.raises(ValueError, match="lines"):
        reader.check_pie_index(shuffled)
    other = reader.Ladder(ladder_root, tables.lines, tables.genes.iloc[::-1])
    with pytest.raises(ValueError, match="panel"):
        reader.check_pie_index(other)


def test_the_target_is_geneeffect_minus_the_frame_mean(world):
    setup = _setup(world)
    targets = setup.targets(None)
    genes = setup.ladder.genes
    effect = setup.gene_effect.loc[list(targets.lines), list(genes.index)]
    expected = (effect - genes["m_g"]) / genes["sigma_g"]
    assert np.allclose(targets.values.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert set(targets.lines) == {*TRAIN, *BULK_LABELLED[:5]}


def test_validation_and_test_lines_are_refused(world):
    setup = _setup(world)
    for line in (VAL[0], TEST[0]):
        with pytest.raises(ValueError, match="refused"):
            reader.training_targets(
                setup.ladder,
                setup.gene_effect,
                setup.split,
                setup.bulk_labelled,
                lines=[TRAIN[0], line],
            )


def test_excluding_a_fold_refits_the_base_without_it(world):
    setup = _setup(world)
    folds = setup.ladder.folds
    targets = setup.targets(1)
    kept = [m for m in setup.ladder.frame if folds[m] != 1]
    assert list(targets.lines) == kept
    expected = setup.gene_effect.loc[kept, list(setup.ladder.genes.index)].mean()
    assert np.allclose(targets.m_g.to_numpy(), expected.to_numpy())
    assert not np.allclose(targets.m_g, setup.ladder.genes["m_g"])


def test_profile_states_round_trip(world):
    _, ladder_root, changes = world
    tables = reader.read_ladder(ladder_root)
    index = reader.check_pie_index(tables)
    means = reader.frame_means(tables, index)
    basis = reader.basis(tables)
    pie_input = reader.pie_input(tables)
    states = reader.profile_states(tables, pie_input, means, basis)
    components = basis["components"].astype(np.float64)
    frame = pie_input.loc[tables.frame].to_numpy().mean(axis=0)
    assert np.allclose(
        states.before @ components + frame, pie_input.to_numpy(), atol=1e-4
    )
    assert states.variance_kept == pytest.approx(1.0)
    stored = reader.open_delta(tables, "dE", index)
    lines = list(range(len(tables.line_ids)))
    for gene in (0, 3):
        after = states.after(np.asarray(stored[gene]), gene, lines)
        expected = changes[gene] @ components.T
        assert np.allclose(after - states.before, expected, atol=1e-4)


def test_the_full_panel_profile_round_trips_and_reports_coverage(world):
    _, ladder_root, changes = world
    tables = reader.read_ladder(ladder_root)
    panel = tables.panel
    selective = [g for g in panel if bool(tables.genes.at[g, "selective"])]
    assert len(selective) >= 3
    pie_input = reader.pie_input(tables)
    profile = reader.full_profile(tables, pie_input, selective)
    assert profile.genes == tuple(selective[:-1])
    assert profile.missing == (selective[-1],)
    frame = pie_input.loc[tables.frame].to_numpy().mean(axis=0)
    assert np.allclose(profile.before, pie_input.to_numpy() - frame, atol=1e-5)
    lines = list(range(len(tables.line_ids)))
    for gene in profile.genes:
        after = profile.after(gene, lines)
        expected = changes[panel.index(gene)]
        assert np.allclose(after - profile.before, expected, atol=1e-5)
    rows = [tables.line_ids.index(m) for m in tables.frame]
    stream = reader.FullStream(profile, rows, torch.device("cpu"), torch.float32)
    block = stream.rows(np.array([0, 1]), np.array([2, 0, 5]))
    for i, gene in enumerate(profile.genes[:2]):
        full = changes[panel.index(gene)]
        assert np.allclose(block[i] + stream.means[i], full[[2, 0, 5]], atol=1e-5)
        assert np.allclose(stream.means[i], full[rows].mean(axis=0), atol=1e-5)


def test_the_full_panel_projects_onto_the_512_component_input(world):
    """Projected on the basis, the full-panel states are the 512-component states:
    a state encoder whose first layer is W C on the full panel computes exactly the
    change W reads from the components."""
    from src.model.stage_one import StateEncoder

    _, ladder_root, changes = world
    tables = reader.read_ladder(ladder_root)
    index = reader.check_pie_index(tables)
    basis = reader.basis(tables)
    components = basis["components"].astype(np.float64)
    pie_input = reader.pie_input(tables)
    states = reader.profile_states(
        tables, pie_input, reader.frame_means(tables, index), basis
    )
    selective = [g for g in tables.panel if bool(tables.genes.at[g, "selective"])]
    profile = reader.full_profile(tables, pie_input, selective)
    assert np.allclose(profile.before @ components.T, states.before, atol=1e-4)
    stored = reader.open_delta(tables, "dE", index)
    torch.manual_seed(0)
    small = StateEncoder(components.shape[0], 7)
    full = StateEncoder(components.shape[1], 7)
    with torch.no_grad():
        full.first.weight.copy_(small.first.weight @ torch.tensor(components).float())
        full.first.bias.copy_(small.first.bias)
        full.second.load_state_dict(small.second.state_dict())
    for gene in profile.genes:
        g = tables.panel.index(gene)
        step = np.asarray(profile.change(gene))
        projected = np.asarray(stored[g]) + states.shift[g]
        assert np.allclose(step @ components.T, projected, atol=1e-4)
        with torch.no_grad():
            reads_full = full.change(torch.tensor(profile.before), torch.tensor(step))
            reads_basis = small.change(
                torch.tensor(states.before), torch.tensor(projected)
            )
        assert torch.allclose(reads_full, reads_basis, atol=1e-4)


# --- runs --------------------------------------------------------------------------


def _config(root: Path, name: str, **data) -> Path:
    config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    config["output_root"] = str(root / "outputs")
    config["data"]["ladder_config"] = str(root / "ladder.yaml")
    config["data"].update(data)
    config["train"].update(
        max_epochs=2, patience=2, warmup_epochs=0, lines_per_step=8, genes_per_step=4
    )
    config["eval"].update(lines_per_chunk=5, genes_per_chunk=3)
    if config["model"]["context"] is not None:
        config["model"]["context"] = {"width": 8, "queries": 2}
    path = root / "configs" / f"{name}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_run_trains_selects_on_validation_and_scores_test_once(world):
    root = world[0]
    path = _config(root, "film_dH")
    torch.manual_seed(0)
    run_dir = stage_one.run(path, None, None, "cpu")
    assert run_dir == root / "outputs" / "architecture" / "film_dH"
    row = json.loads((run_dir / "row.json").read_text())
    assert set(row["scores"]) == {"val", "test", "train_in_sample"}
    assert row["scores"]["val"]["lines"] == len(VAL)
    assert row["history"][0]["epoch"] == 0 and not row["history"][0]["candidate"]
    assert row["best_epoch"] >= 1
    setup = _setup(world)
    base = reader.base_prediction(setup.ladder, setup.ladder.genes["m_g"], list(VAL))
    from src.eval.geneeffect import wide_accuracy

    expected = wide_accuracy(base, setup.truth(list(VAL)), setup.definitions)
    assert row["history"][0]["val_within_gene_r2"] == pytest.approx(
        expected["within_gene_r2"]
    )
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    lines = setup.ladder.line_ids
    assert list(predictions.index) == lines
    outside = [g for g in setup.ladder.genes.index if g not in setup.ladder.panel]
    assert np.allclose(
        predictions.loc[:, outside], base.iloc[:1].loc[:, outside].to_numpy()
    )
    viability = np.load(run_dir / "V.npy")
    assert viability.shape == (len(lines), len(setup.ladder.panel), 32)
    assert viability.dtype == np.float16
    record = json.loads((run_dir / "V.json").read_text())
    assert sum(record["in_sample"]) == len(setup.ladder.frame)
    # Done: a rerun skips, and the bound config refuses another.
    assert stage_one.run(path, None, None, "cpu") == run_dir
    changed = yaml.safe_load(path.read_text())
    changed["train"]["learning_rate"] = 0.5
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="different config"):
        stage_one.run(path, None, None, "cpu")


@pytest.mark.parametrize(
    "name",
    [
        "film_dE",
        "permuted_film_dH",
        "multi_dL",
        "nodelta_film",
        "lineage_only",
        "text_only",
        "own_x_only",
    ],
)
def test_every_film_family_row_runs(world, name):
    root = world[0]
    torch.manual_seed(0)
    run_dir = stage_one.run(_config(root, name), None, None, "cpu")
    row = json.loads((run_dir / "row.json").read_text())
    assert np.isfinite(row["scores"]["val"]["within_gene_r2"])


def test_a_fold_run_trains_without_its_fold(world):
    root = world[0]
    torch.manual_seed(0)
    run_dir = stage_one.run(_config(root, "film_dH"), None, 0, "cpu")
    assert run_dir.name == "film_dH__fold0"
    row = json.loads((run_dir / "row.json").read_text())
    setup = _setup(world)
    folds = setup.ladder.folds
    held = [m for m in setup.ladder.role_lines("train_cells") if folds[m] == 0]
    assert row["scores"]["oof"]["lines"] == len(held)
    assert "test" not in row["scores"]
    in_sample = json.loads((run_dir / "V.json").read_text())["in_sample"]
    lines = setup.ladder.line_ids
    assert not any(in_sample[lines.index(m)] for m in held)


def test_prior_reference_components_reader_and_report(world):
    from src.experiments.stage_one_prior import run as prior_run

    root = world[0]
    config = yaml.safe_load((CONFIGS / "prior_reference.yaml").read_text())
    config.update(
        output_root=str(root / "outputs"), ladder_config=str(root / "ladder.yaml")
    )
    prior_dir = prior_run(config, "prior_reference_seed0")
    setup = _setup(world)
    oof = pd.read_parquet(prior_dir / "predictions_oof.parquet")
    assert list(oof.index) == setup.ladder.frame
    for name in ("main", "fold0", "fold1"):
        components = pd.read_parquet(prior_dir / "components" / f"{name}.parquet")
        assert list(components.index) == setup.ladder.line_ids
        assert components.shape[1] == 3
    reader_path = _config(root, "components_reader", prior_run=str(prior_dir))
    torch.manual_seed(0)
    for fold in (None, 0, 1):
        stage_one.run(reader_path, None, fold, "cpu")
    report = {
        "kind": "report",
        "task": "architecture",
        "output_root": str(root / "outputs"),
        "ladder_config": str(root / "ladder.yaml"),
        "bootstrap_repeats": 20,
        "rows": [
            {"name": "base", "kind": "base"},
            {"name": "prior", "kind": "prior", "run": str(prior_dir)},
            {"name": "dH ridge", "kind": "dh_ridge", "penalty": 10.0},
            {"name": "components reader", "kind": "run", "config": str(reader_path)},
            {"name": "missing", "kind": "run", "config": str(_config(root, "film_dL"))},
        ],
        "contrasts": [
            ["components reader", "prior"],
            ["dH ridge", "base"],
            ["missing", "prior"],
        ],
    }
    path = root / "configs" / "report.yaml"
    path.write_text(yaml.safe_dump(report))
    out = stage_one.report(path, "cpu")
    results = json.loads((out / "results.json").read_text())
    assert results["scores"]["components reader"]["oof"]["lines"] == len(TRAIN)
    assert results["scores"]["missing"]["val"] is None
    assert results["scores"]["base"]["val"]["selective_spearman"] is None
    contrast = results["contrasts"][0]["val"]
    assert len(contrast["selective_spearman"]["interval"]) == 2
    assert results["contrasts"][2]["val"] is None
    text = (out / "results.md").read_text()
    assert "## Contrasts" in text and "| components reader |" in text
    assert load_stage_one_config(CONFIGS / "report_3a.yaml")["kind"] == "report"
    # Out of fold, each fold's base differs: rank metrics are scored within folds.
    base_oof = results["scores"]["base"]["oof"]
    assert base_oof["selective_spearman"] is None
    assert sum(f["lines"] for f in base_oof["per_fold"].values()) == len(TRAIN)
    # A contrast's Spearman covers the genes defined on both sides.
    assert (
        0 < contrast["selective_spearman"]["genes"] <= len(setup.definitions.selective)
    )
    against_base = results["contrasts"][1]["val"]["selective_spearman"]
    assert against_base["genes"] == 0 and against_base["difference"] is None
    _check_prior_out_of_fold(setup, prior_dir, root)
    _check_reads_are_bound(root, path, reader_path, prior_dir, out)


def _check_prior_out_of_fold(setup, prior_dir: Path, root: Path) -> None:
    """The prior's fold-0 lines are predicted with the components fitted without
    fold 0, the components the reader's fold-0 run reads."""
    from src.context_prior.prior import PriorInputs, PriorSpec, fit_prior, total
    from src.context_prior.reference import load_reference
    from src.context_prior.views import fit_expression_components
    from src.experiments.config import load_prior_config
    from src.experiments.context_prior import stages

    ladder = setup.ladder
    prior = load_prior_config(Path(setup.ladder_config["prior_config"]))
    reference = prior["reference"]
    expression = reader.mean_channel(ladder).astype(np.float64)
    kept = [m for m in ladder.frame if ladder.folds[m] != 0]
    held = [m for m in ladder.frame if ladder.folds[m] == 0]
    inputs = PriorInputs(
        expression=expression.loc[ladder.frame],
        residual=setup.truth(ladder.frame),
        components=fit_expression_components(expression.loc[kept], 3),
        reference=load_reference(
            Path(prior["paths"]["reference"]),
            blocked={*ladder.role_lines("val", "test"), *setup.excluded},
        ),
        lineage=ladder.lines.loc[ladder.frame, "lineage"],
        patients=ladder.lines["patient_id"].astype(str).to_dict(),
    )
    spec = PriorSpec(
        tuple(
            stages(
                prior,
                reference["block_set"],
                reference["components_penalty"],
                reference["gene_penalty"],
            )
        )
    )
    fitted = fit_prior(spec, inputs, fit_lines=kept, encoder_lines=kept)
    expected = total(fitted.predict(expression.loc[held]))
    written = pd.read_parquet(prior_dir / "predictions_oof.parquet").loc[held]
    assert np.allclose(written.to_numpy(), expected.to_numpy(), atol=1e-5)


def _check_reads_are_bound(
    root: Path, report_path: Path, reader_path: Path, prior_dir: Path, out: Path
) -> None:
    """The queue and the report read a finished run only under the config it was
    bound to, a prior only from the report's ladder and with its row.json, and the
    ridge's cache only from the inputs it was computed from."""
    run_dir = root / "outputs" / "architecture" / "components_reader"
    bound = run_dir / "run_config.json"
    original = bound.read_text()
    assert stage_one.queue([reader_path], ["0"], ["main", "0", "1"]) == 0  # all done
    changed = json.loads(original)
    changed["config"]["run"]["train"]["learning_rate"] = 0.5
    bound.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="another config"):
        stage_one.queue([reader_path], ["0"], ["main"])
    with pytest.raises(ValueError, match="another config"):
        stage_one.report(report_path, "cpu")
    bound.write_text(original)

    prior_bound = prior_dir / "run_config.json"
    original = prior_bound.read_text()
    changed = json.loads(original)
    changed["config"]["ladder_config"] = str(root / "elsewhere.yaml")
    prior_bound.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="the report"):
        stage_one.report(report_path, "cpu")
    prior_bound.write_text(original)
    (prior_dir / "row.json").rename(prior_dir / "row.json.aside")
    results = json.loads(
        (stage_one.report(report_path, "cpu") / "results.json").read_text()
    )
    assert results["scores"]["prior"]["val"] is None
    (prior_dir / "row.json.aside").rename(prior_dir / "row.json")

    fingerprint = out / "dh_ridge" / "fingerprint.json"
    original = fingerprint.read_text()
    changed = json.loads(original)
    changed["lines"] = {}
    fingerprint.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="other inputs"):
        stage_one.report(report_path, "cpu")
    fingerprint.write_text(original)


def test_matched_profile_rows_train_on_the_same_genes_and_are_contrasted(world):
    root = world[0]
    setup = _setup(world)
    selective = [
        g for g in setup.ladder.panel if bool(setup.ladder.genes.at[g, "selective"])
    ]
    runs = {}
    for name in ("film_dE_selective", "film_dE_full"):
        path = _config(root, name)
        torch.manual_seed(0)
        runs[name] = (path, stage_one.run(path, None, None, "cpu"))
        row = json.loads((runs[name][1] / "row.json").read_text())
        assert row["genes"]["count"] == len(selective) - 1
        assert row["genes"]["without_full_panel"] == [selective[-1]]
        record = json.loads((runs[name][1] / "V.json").read_text())
        assert record["genes"] == selective[:-1]
        assert np.load(runs[name][1] / "V.npy").shape[1] == len(selective) - 1
        assert all(e["epoch_seconds"] >= 0 for e in row["history"][1:])
        predictions = pd.read_parquet(runs[name][1] / "predictions.parquet")
        others = [g for g in setup.ladder.genes.index if g not in set(selective[:-1])]
        base = reader.base_prediction(setup.ladder, setup.ladder.genes["m_g"], ["x"])
        assert np.allclose(predictions[others], base[others].to_numpy())
    full_row = json.loads((runs["film_dE_full"][1] / "row.json").read_text())
    assert full_row["full_panel_stream"]["genes"] == len(selective) - 1
    report = {
        "kind": "report",
        "task": "architecture",
        "output_root": str(root / "outputs"),
        "ladder_config": str(root / "ladder.yaml"),
        "bootstrap_repeats": 20,
        "rows": [
            {"name": "512", "kind": "run", "config": str(runs["film_dE_selective"][0])},
            {"name": "full", "kind": "run", "config": str(runs["film_dE_full"][0])},
        ],
        "contrasts": [["full", "512"]],
    }
    path = root / "configs" / "report_profile.yaml"
    path.write_text(yaml.safe_dump(report))
    results = json.loads((stage_one.report(path, "cpu") / "results.json").read_text())
    contrast = results["contrasts"][0]["val"]["selective_spearman"]
    assert contrast["genes"] <= len(selective) - 1


def test_a_run_refuses_a_changed_gene_set(tmp_path):
    stage_one.bind_genes(tmp_path, ["GA", "GB"])
    stage_one.bind_genes(tmp_path, ["GA", "GB"])
    with pytest.raises(ValueError, match="another gene set"):
        stage_one.bind_genes(tmp_path, ["GA"])

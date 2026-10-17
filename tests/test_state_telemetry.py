"""State telemetry end to end on the synthetic ladder world: fit samples, the
mixtures, projection, state inputs that reproduce PIE's input, the frozen PIE's
gap and the report. The ``.venv-states`` fits (programs, scVI) are replaced by
stand-in files of the same format."""

from __future__ import annotations

import json
import shutil
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import cell_states as cs
from src.data.tx1_cache import EMBEDDING_WIDTH, write_line_cache
from src.experiments import ladder_inputs as ladder
from src.experiments import state_telemetry as st
from src.experiments.config import validate_states_config
from tests.test_ladder_inputs import TEST, TRAIN, VAL, build_world

SECOND = ("ACH-T0", "ACH-T1", "ACH-T2", "ACH-V0")
PROGRAM_KINDS = ("recurrent", "recurrent", "cycle", "background")


def _stand_in_state_programs(t: st.Telemetry) -> None:
    """Files of state_programs' format: a reference of two recurrent programs,
    a cycle phase and the background; usages and latents of every cell."""
    plan = json.loads(t.layout.plan.read_text())
    assert set(plan["encode"]) == set(t.projected)
    assert set(plan["second"]) == set(SECOND)
    for seed in t.seeds:
        folder = t.layout.programs(seed)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / "reference.npz",
            programs=np.asarray(["program0", "program1", "cycle_s", "background"]),
            kinds=np.asarray(PROGRAM_KINDS),
        )
        for second, sources in ((False, plan["encode"]), (True, plan["second"])):
            for model_id, path in sources.items():
                n = ad.read_h5ad(path).n_obs
                rng = np.random.default_rng(
                    zlib.crc32(f"{model_id}{seed}{second}".encode())
                )
                usage = st.usage_path(t.layout, seed, model_id, second)
                usage.parent.mkdir(parents=True, exist_ok=True)
                np.save(usage, rng.gamma(1.0, size=(n, 4)).astype(np.float32))
                latent = st.latent_path(t.layout, seed, model_id, second)
                latent.parent.mkdir(parents=True, exist_ok=True)
                offset = 3.0 * (zlib.crc32(model_id.encode()) % 3)
                np.save(latent, (rng.normal(size=(n, 3)) + offset).astype(np.float32))
        (t.layout.scvi(seed) / "manifest.json").write_text("{}")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("states")
    ladder_config = build_world(root)
    ladder.run(ladder_config, ["lines", "cells", "mean"], (0, 1))
    (root / "ladder.yaml").write_text(yaml.safe_dump(ladder_config))
    deposits = {m: "dep_a" if i < 5 else "dep_b" for i, m in enumerate(TRAIN)}
    deposits |= {m: "dep_c" for m in (*VAL, *TEST, "ACH-U0")}
    pd.DataFrame(
        {"model_id": list(deposits), "expression_source": list(deposits.values())}
    ).to_csv(root / "split.csv", index=False)
    pd.DataFrame(
        [
            ("dep_a", "", "", "10x 3' v2"),
            ("dep_b", "", "", "mixed"),
            ("dep_b", "", "expt_a", "10x 3' v2"),
            ("dep_b", "", "expt_b", "10x 3' v3"),
            ("dep_b", TRAIN[9], "", "10x 5' v2"),
            ("dep_c", "", "", "unknown"),
        ],
        columns=["source", "model_id", "arm", "chemistry"],
    ).assign(evidence="synthetic", note="").to_csv(root / "chemistry.csv", index=False)
    # dep_b's lines carry each cell's experiment in obs, as MIX-seq's do.
    for model_id in TRAIN[5:9]:
        path = root / "basal_raw_umi" / f"{model_id}.h5ad"
        source = ad.read_h5ad(path)
        source.obs["arm"] = [
            "expt_a" if i % 3 else "expt_b" for i in range(source.n_obs)
        ]
        source.write_h5ad(path)
    rng = np.random.default_rng(7)
    for model_id in (*TRAIN, *VAL, *TEST):
        cells = ladder.read_cells(root / "ladder", model_id)
        n = len(cells["cell_ids"])
        write_line_cache(
            root / "tx1" / "real",
            model_id,
            rng.normal(size=(n, EMBEDDING_WIDTH)).astype(np.float32),
            np.zeros((n, 4)),
            pd.DataFrame(index=pd.Index(cells["cell_ids"].astype(str))),
        )
    second = root / "second"
    second.mkdir()
    for model_id in SECOND:
        source = ad.read_h5ad(root / "basal_raw_umi" / f"{model_id}.h5ad")
        source = source[::-1].copy()
        source.obs_names = [f"{n}-second" for n in source.obs_names]
        source.write_h5ad(second / f"{model_id}.h5ad")
    noise = yaml.safe_load((root / "noise.yaml").read_text())
    noise["deposit"]["second_source_root"] = str(second)
    (root / "noise.yaml").write_text(yaml.safe_dump(noise))
    (root / "prepared" / "noise" / "manifest.json").write_text(
        json.dumps(
            {
                "settings": {
                    "genes": [f"G{i}" for i in range(8)],
                    "programs": {"S": ["G0", "G1"], "G2M": ["G2", "G3"]},
                    "target_sum": 5000.0,
                }
            }
        )
    )
    config = yaml.safe_load(Path("configs/stage_one/states/telemetry.yaml").read_text())
    config.update(
        ladder_config=str(root / "ladder.yaml"), output_root=str(root / "out")
    )
    config["fit"].update(
        cells_per_line=20,
        seeds=[0, 1],
        mixture_counts=[2, 3],
        mixture_inits=1,
        components=3,
    )
    config["inputs"].update(merge_effective_count=3, permutation_seeds=[0, 1])
    config["references"]["chemistry"] = str(root / "chemistry.csv")
    for group, key in (
        ("kinker", "program"),
        ("gavish", "program"),
        ("hallmarks", "gene_set"),
        ("tirosh", "phase"),
    ):
        names = ["S", "G2M"] if group == "tirosh" else [f"{group} one", f"{group} two"]
        path = root / f"{group}.tsv"
        pd.DataFrame(
            {
                key: [names[0]] * 2 + [names[1]] * 3,
                "gene": ["G0", "G1", "G4", "G5", "OLD"],
            }
        ).to_csv(path, sep="\t", index=False)
        config["references"][group] = str(path)
    config["report"].update(permutations=20, stability_cells=10)
    config = validate_states_config(config)
    t = st.Telemetry(config, "telemetry")
    original = st.run_state_programs
    st.run_state_programs = _stand_in_state_programs
    try:
        st.run(config, "telemetry", ["fit", "project", "inputs"])
    finally:
        st.run_state_programs = original
    return root, config, t


def test_fit_samples_read_training_lines_only(world):
    _, _, t = world
    sample = pd.read_parquet(t.layout.sample(0))
    assert set(sample["model_id"]) == set(TRAIN)
    assert (sample.groupby("model_id").size() == 20).all()
    assert not sample.duplicated(["model_id", "row"]).any()
    # Chemistry per cell: dep_a by source, dep_b by each cell's arm, one line by
    # its own row.
    by_line = sample.groupby("model_id")["chemistry"].agg(set)
    assert by_line[TRAIN[0]] == {"10x 3' v2"}
    assert by_line[TRAIN[9]] == {"10x 5' v2"}
    for model_id in TRAIN[5:9]:
        rows = sample.loc[sample["model_id"] == model_id]
        expected = np.where(rows["row"] % 3 == 0, "10x 3' v3", "10x 3' v2")
        assert rows["chemistry"].tolist() == expected.tolist()


def test_a_mixed_source_cell_without_a_row_fails_closed(world):
    root, config, _ = world
    table = pd.read_csv(root / "chemistry.csv", keep_default_na=False)
    t = st.Telemetry(config, "telemetry")
    t.chemistry_table = table.loc[table["arm"] != "expt_b"]  # expt_b cells uncovered
    ids = t.cell_arrays(TRAIN[5], "cell_ids")[0]
    with pytest.raises(ValueError, match="covers arm 'expt_b'"):
        t.cell_chemistry(TRAIN[5], ids)
    params = st.load_mixture(t, 0, "hvg_k3")
    assert params["pca_components"].shape == (3, 4) and bool(params["centred"])
    assert list(st.load_mixture(t, 0, "cycle")["states"]) != ["0", "1", "2"]


def test_memberships_cover_every_projected_line_and_project_identically(world):
    _, config, t = world
    assert t.projected == [*TRAIN, *VAL]
    keys = {cs.membership_key(v, s) for s in (0, 1) for v in cs.vocabularies(config)}
    for model_id in t.projected:
        stored = np.load(t.layout.memberships(model_id))
        assert set(stored.files) == keys
    again = st.project_line(t, "ACH-V1")
    with np.load(t.layout.memberships("ACH-V1")) as stored:
        for key in keys:
            assert np.array_equal(again[key], stored[key])
    second = np.load(t.layout.memberships("ACH-T0", second=True))
    assert not any(k.startswith("tx1") for k in second.files)
    assert "hvg_k2@seed0" in second.files


def test_state_inputs_reproduce_pies_input_in_every_set(world):
    root, config, t = world
    pie_input = pd.read_parquet(root / "ladder" / "mean" / "pie_input.parquet")
    for set_name in cs.state_sets(config):
        inputs = pd.read_parquet(t.layout.state_set(set_name) / "inputs.parquet")
        masses = pd.read_parquet(t.layout.state_set(set_name) / "masses.parquet")
        assert list(inputs.columns) == t.panel
        for model_id in t.projected:
            rows = masses["model_id"] == model_id
            states = rows & (masses["state"] != cs.WHOLE)
            whole = inputs.loc[rows & (masses["state"] == cs.WHOLE)].to_numpy()[0]
            mixed = (
                masses.loc[states, "mass"].to_numpy() @ inputs.loc[states].to_numpy()
            )
            assert np.allclose(whole, pie_input.loc[model_id], atol=1e-5)
            if set_name == cs.CYCLING_SET:  # reweighted cells: not the line's input
                assert not np.allclose(mixed, whole, atol=1e-4)
                continue
            assert np.allclose(mixed, whole, atol=1e-4), (set_name, model_id)
            assert (
                masses.loc[states, "effective_count"] >= 3
            ).all() or states.sum() == 1
    cycling = pd.read_parquet(t.layout.state_set(cs.CYCLING_SET) / "masses.parquet")
    assert len(cycling) == 2 * len(t.projected)


def test_permuted_sets_keep_the_real_states_masses_and_effective_counts(world):
    _, config, t = world
    merged_somewhere = False
    for set_name in cs.state_sets(config):
        vocabulary = cs.base_vocabulary(set_name)
        if set_name == vocabulary:
            continue
        real = pd.read_parquet(t.layout.state_set(vocabulary) / "masses.parquet")
        permuted = pd.read_parquet(t.layout.state_set(set_name) / "masses.parquet")
        assert real["state"].tolist() == permuted["state"].tolist()
        assert real["model_id"].tolist() == permuted["model_id"].tolist()
        assert np.allclose(real["mass"], permuted["mass"], equal_nan=True)
        assert np.allclose(real["effective_count"], permuted["effective_count"])
        assert real["merged"].tolist() == permuted["merged"].tolist()
        merged_somewhere |= (real["merged"] != "").any()
    assert merged_somewhere  # the world's settings do merge small states


def test_matched_states_count_unmatched_mass_against_zero():
    rng = np.random.default_rng(3)
    qa = rng.dirichlet([1.0, 1.0, 1.0], size=200)
    qb = qa[:, :2]  # the second fit lacks state 2
    ma = np.tile([0.5, 0.3, 0.2], (4, 1))
    mb = np.tile([0.6, 0.4], (4, 1))
    result = st.matched_states(qa, qb, ma, mb)
    assert result["states"] == [3, 2] and result["unmatched_states"] == [1, 0]
    assert np.isclose(result["unmatched_mass_mean"], 0.2)
    # Matched |0.5 - 0.6| + |0.3 - 0.4| plus the unmatched 0.2, halved.
    assert np.isclose(result["matched_mass_tvd_mean"], 0.2)
    same = st.matched_states(qa, qa, ma, ma)
    assert same["unmatched_mass_mean"] == 0 and same["matched_mass_tvd_mean"] == 0


def test_deposit_share_is_reported_per_state():
    rng = np.random.default_rng(4)
    lines = [f"ACH-{i}" for i in range(40)]
    deposit = pd.Series(["d1"] * 20 + ["d2"] * 20, index=lines)
    driven = np.where(deposit == "d1", 0.2, 0.6) + rng.normal(0, 0.02, 40)
    frame = pd.DataFrame({"driven": driven, "free": rng.uniform(0, 1, 40)}, index=lines)
    shares = st.per_state_group_share(
        frame, deposit, pd.Series("Lung", index=lines), permutations=200, seed=0
    )
    assert shares["driven"]["observed"] > 0.9 > shares["driven"]["null_q95"]
    assert shares["driven"]["p_value"] < 0.01
    assert shares["free"]["observed"] < 0.3


def test_a_state_with_constant_mass_gets_no_deposit_p_value():
    lines = [f"ACH-{i}" for i in range(10)]
    deposit = pd.Series(["d1"] * 5 + ["d2"] * 5, index=lines)
    frame = pd.DataFrame({"constant": np.full(10, 0.5)}, index=lines)
    shares = st.per_state_group_share(
        frame, deposit, pd.Series("Lung", index=lines), permutations=50, seed=0
    )
    assert np.isnan(shares["constant"]["observed"])
    assert np.isnan(shares["constant"]["p_value"])
    assert np.isnan(shares["constant"]["null_q95"])


def _write_pie(root: Path, t: st.Telemetry, set_name: str, bump: float) -> None:
    """A main pass and an assembled state pass on two genes where every context
    reads its line's main-pass deltas, one state of one line moved by ``bump``."""
    rng = np.random.default_rng(9)
    pie = root / "ladder" / "pie"
    pie.mkdir(parents=True, exist_ok=True)
    n_all, p = len(t.lines), len(t.panel)
    widths = {"dH": 4, "dE": 2, "own_h": 4, "own_dp": 1}
    np.savez(pie / "basis.npz", mean=np.zeros(p), components=rng.normal(size=(2, p)))
    means = {k: rng.normal(size=(p, w)) for k, w in widths.items()}
    np.savez(pie / "frame_means.npz", **means)
    genes = t.panel[:2]
    main = {}
    for gene in genes:
        main[gene] = {k: rng.normal(size=(n_all, w)) for k, w in widths.items()}
        (pie / "genes").mkdir(parents=True, exist_ok=True)
        np.savez(
            pie / "genes" / f"{gene}.npz",
            **{**main[gene], "own_dp": main[gene]["own_dp"][:, 0]},
        )
    folder = t.layout.pie_states / set_name
    folder.mkdir(parents=True, exist_ok=True)
    contexts = pd.read_parquet(t.layout.state_set(set_name) / "masses.parquet")
    rows = [t.lines.index.get_loc(m) for m in contexts["model_id"]]
    moved = np.flatnonzero(
        (contexts["model_id"] == "ACH-T3").to_numpy()
        & (contexts["state"] != cs.WHOLE).to_numpy()
    )[0]
    for name, width in widths.items():
        array = np.zeros((2, len(contexts), width))
        for g, gene in enumerate(genes):
            array[g] = main[gene][name][rows] - means[name][t.panel.index(gene)]
            array[g, moved] += bump
        np.save(
            folder / f"{name}.npy",
            array.astype(np.float16 if width > 1 else np.float32),
        )
    contexts.to_parquet(folder / "contexts.parquet")
    inputs = pd.read_parquet(t.layout.state_set(set_name) / "inputs.parquet")
    fingerprints = cs.set_fingerprints(inputs, contexts)
    (folder / "index.json").write_text(
        json.dumps({"genes": genes, "fingerprints": fingerprints})
    )


def test_gap_is_zero_when_states_read_the_line_and_grows_with_a_moved_state(world):
    root, _, t = world
    assert t.layout.pie_states == t.layout.root / "pie"  # moves with states/
    _write_pie(root, t, "cycle", 0.0)
    gap = st.gap_section(t, "cycle")
    for name in st.DELTAS:
        assert gap[name]["train_cells"]["gap_ratio"] < 1e-5
        assert gap[name]["train_cells"]["noise_ratio"] < 1e-5
    _write_pie(root, t, "cycle", 2.0)
    moved = st.gap_section(t, "cycle")
    assert moved["dH"]["train_cells"]["gap_ratio"] > 1e-3
    assert moved["dH"]["val"]["gap_ratio"] < 1e-5


def test_gap_refuses_a_pass_over_other_state_inputs(world):
    root, _, t = world
    _write_pie(root, t, "programs", 0.0)
    index = t.layout.pie_states / "programs" / "index.json"
    payload = json.loads(index.read_text())
    payload["fingerprints"][TRAIN[0]] = "an earlier state input"
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="other state inputs"):
        st.gap_section(t, "programs")
    shutil.rmtree(t.layout.pie_states / "programs")


def test_report_writes_every_section(world):
    root, config, t = world
    st.run(config, "telemetry", ["report"])
    # Stability reads training lines only: 10 cells from each of the 10.
    t_report = st.Telemetry(config, "telemetry")
    st.stability_section(t_report, "cycle")
    assert t_report.cache["stability_draw"]["cycle@seed0"].shape[0] == 10 * len(TRAIN)
    report = json.loads((root / "out" / "telemetry" / "report.json").read_text())
    assert set(report["vocabularies"]) == set(cs.vocabularies(config))
    entry = report["vocabularies"]["hvg_k2"]
    assert set(entry["identity"]["members_listed"]) == set(st.REFERENCE_GROUPS)
    assert entry["second_deposit"]["lines"] == len(SECOND)
    assert set(entry["stability"]) == {"0-1"}
    assert "unmatched_mass_mean" in entry["stability"]["0-1"]
    assert set(entry["confound"]["deposit_per_state"]) == set(entry["states"])
    assert report["pie_gap"]["cycle"] is not None
    assert report["pie_gap"]["hvg_k2"] is None
    assert set(report["determinism"]["lines"]) == set(VAL)
    assert all(report["determinism"]["lines"].values())
    assert (
        "## The frozen PIE's gap"
        in (root / "out" / "telemetry" / "results.md").read_text()
    )
    shutil.rmtree(t.layout.pie_states)

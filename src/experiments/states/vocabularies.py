"""The state vocabularies: per-cell representations, their mixtures fitted on the
training lines (the ``fit`` step, cNMF and scVI in ``programs``) and every projected
line's memberships (the ``project`` step)."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd
from src.data import cell_states as cs
from src.data.io import write_json, write_npz, write_parquet
from src.experiments.run_dir import REPO_ROOT, absolute
from src.experiments.states.telemetry import (
    MANIFEST,
    Telemetry,
    latent_path,
    load_npz,
    usage_path,
)


def _tx1(t: Telemetry, model_id: str, cell_ids: np.ndarray) -> np.ndarray:
    from src.data.tx1_cache import load_line_cache

    root = absolute(t.setup.config["paths"]["tx1_cache"]) / "real"
    embeddings, _, obs = load_line_cache(root, model_id)
    if list(obs.index.astype(str)) != [str(c) for c in cell_ids]:
        raise ValueError(f"{model_id}: Tx1 cells are not the line's cells in order")
    return np.asarray(embeddings, dtype=np.float32)


def representations(
    t: Telemetry, model_id: str, seeds: Sequence[int], second: bool = False
) -> dict[str, Any]:
    """A line's per-cell representations: ``cycle`` (S, G2/M), ``qc``, ``hvg``,
    ``tx1`` (primary deposit only) and ``scvi`` per seed."""
    if second:
        cells = load_npz(t.layout.second_cells(model_id))
    else:
        keys = ("qc", "hvg", "cell_ids")
        cells = dict(zip(keys, t.cell_arrays(model_id, *keys), strict=True))
    qc = np.asarray(cells["qc"], dtype=np.float64)
    out: dict[str, Any] = {
        "cycle": qc[:, list(cs.CYCLE_FEATURES)],
        "qc": qc[:, list(cs.QC_FEATURES)],
        "hvg": np.asarray(cells["hvg"], dtype=np.float32),
        "scvi": {s: np.load(latent_path(t.layout, s, model_id, second)) for s in seeds},
        "n": int(qc.shape[0]),
    }
    if not second:
        out["tx1"] = _tx1(t, model_id, cells["cell_ids"])
    return out


def _features(representation: Mapping[str, Any], name: str, seed: int) -> np.ndarray:
    value = representation[name]
    return value[seed] if name == "scvi" else value


def write_samples(t: Telemetry) -> None:
    if all(t.layout.sample(seed).is_file() for seed in t.seeds):
        return
    counts, scores = {}, {}
    for model_id in t.train:
        ids, qc = t.cell_arrays(model_id, "cell_ids", "qc")
        counts[model_id] = int(ids.shape[0])
        scores[model_id] = (
            ids,
            qc[:, list(cs.CYCLE_FEATURES)],
            t.cell_chemistry(model_id, ids),
        )
    roles = t.lines["role"].to_dict()
    for seed in t.seeds:
        path = t.layout.sample(seed)
        if path.is_file():
            continue
        sample = cs.fit_sample(
            counts,
            roles,
            t.split,
            cells_per_line=int(t.config["fit"]["cells_per_line"]),
            seed=seed,
        )
        ids = np.concatenate([scores[m][0][r] for m, r in _by_line(sample)])
        cycle = np.vstack([scores[m][1][r] for m, r in _by_line(sample)])
        chemistry = np.concatenate([scores[m][2][r] for m, r in _by_line(sample)])
        sample = sample.assign(
            cell_id=ids.astype(str),
            chemistry=chemistry,
            s_score=cycle[:, 0],
            g2m_score=cycle[:, 1],
        )
        write_parquet(path, sample)


def _by_line(sample: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
    for model_id, group in sample.groupby("model_id", sort=False):
        yield str(model_id), group["row"].to_numpy()


def write_plan(t: Telemetry) -> dict[str, Any]:
    """Everything ``state_programs`` reads, as files: it exchanges nothing else
    with this environment."""
    hvg_order = list(t.recorded["hvg_order"])
    plan = {
        "root": str(t.layout.root),
        "seeds": t.seeds,
        "train": t.train,
        "programs": dict(t.config["programs"]),
        "scvi": dict(t.config["scvi"]),
        "panel": t.panel,
        "hvg": hvg_order,
        "cycle_genes": t.tirosh(),
        "var_ensembl_col": t.preparation["var_ensembl_col"],
        "encode": {m: str(t.source(m)) for m in t.projected},
        "second": {m: str(p) for m, p in t.second_sources.items()},
        "samples": {str(s): str(t.layout.sample(s)) for s in t.seeds},
    }
    write_json(t.layout.plan, plan, keep_nan=True)
    return plan


def _states_outputs_complete(t: Telemetry) -> bool:
    for seed in t.seeds:
        if not (t.layout.programs(seed) / "reference.npz").is_file():
            return False
        if not (t.layout.scvi(seed) / MANIFEST).is_file():
            return False
        for second, lines in ((False, t.projected), (True, list(t.second_sources))):
            for model_id in lines:
                if not usage_path(t.layout, seed, model_id, second).is_file():
                    return False
                if not latent_path(t.layout, seed, model_id, second).is_file():
                    return False
    return True


def run_state_programs(t: Telemetry) -> None:
    """``src.experiments.states.programs`` in ``.venv-states`` (``STATES_PYTHON``)."""
    if _states_outputs_complete(t):
        return
    python = os.environ.get("STATES_PYTHON")
    if not python:
        raise RuntimeError(
            "STATES_PYTHON is unset: the program and scVI fits run in .venv-states "
            "(hpc/states/setup.sh); run through hpc/run.sh cell-states"
        )
    subprocess.run(
        [
            python,
            "-m",
            "src.experiments.states.programs",
            "--plan",
            str(t.layout.plan),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    if not _states_outputs_complete(t):
        raise RuntimeError("state_programs finished without every output")


def _sample_features(
    t: Telemetry, sample: pd.DataFrame, name: str, seed: int, centred: bool
) -> np.ndarray:
    """The sample's rows of a representation, each line centred on its mean over
    all of its cells when ``centred``."""
    parts = []
    for model_id, rows in _by_line(sample):
        if name == "scvi":
            values = np.load(latent_path(t.layout, seed, model_id, False))
        elif name == "tx1":
            values = _tx1(t, model_id, t.cell_arrays(model_id, "cell_ids")[0])
        elif name == "hvg":
            values = t.cell_arrays(model_id, "hvg")[0]
        else:
            columns = cs.QC_FEATURES if name == "qc" else cs.CYCLE_FEATURES
            values = t.cell_arrays(model_id, "qc")[0][:, list(columns)]
        wide = name in ("hvg", "tx1")
        values = np.asarray(values, dtype=np.float32 if wide else np.float64)
        chosen = values[rows]
        parts.append(chosen - values.mean(axis=0) if centred else chosen)
    return np.vstack(parts)


def fit_vocabularies(t: Telemetry, seed: int) -> dict[str, Any]:
    """Every mixture vocabulary at one fitting seed (``fit/seed<s>/``)."""
    fit = t.config["fit"]
    sample = pd.read_parquet(t.layout.sample(seed))
    inits, counts = int(fit["mixture_inits"]), list(fit["mixture_counts"])
    written = []
    path = t.layout.mixture(seed, cs.CYCLE)
    if not path.is_file():
        features = _sample_features(t, sample, "cycle", seed, centred=False)
        params = cs.fit_mixture(
            features, int(fit["cycle_states"]), seed=seed, inits=inits
        )
        labels = cs.cycle_labels(params["means"])
        write_npz(path, **params, centred=False, states=np.asarray(labels))
        written.append(cs.CYCLE)
    for family, (representation, centred) in cs.MIXTURE_FAMILIES.items():
        todo = [
            k for k in counts if not t.layout.mixture(seed, f"{family}_k{k}").is_file()
        ]
        if not todo:
            continue
        features = _sample_features(t, sample, representation, seed, centred)
        extra: dict[str, np.ndarray] = {}
        if representation == "qc":
            extra["scale"] = features.std(axis=0)
            features = features / extra["scale"]
        if representation in ("hvg", "tx1"):
            extra |= cs.fit_pca(features, int(fit["components"]), seed)
            features = (features - extra["pca_mean"]) @ extra["pca_components"].T
        for k in todo:
            params = cs.fit_mixture(features, k, seed=seed, inits=inits)
            write_npz(
                t.layout.mixture(seed, f"{family}_k{k}"),
                **params,
                **extra,
                centred=centred,
                states=np.asarray([str(i) for i in range(k)]),
            )
            written.append(f"{family}_k{k}")
    return {"seed": seed, "written": written, "sample_cells": len(sample)}


def step_fit(t: Telemetry) -> dict[str, Any]:
    write_samples(t)
    write_plan(t)
    run_state_programs(t)
    return {
        "seeds": [fit_vocabularies(t, seed) for seed in t.seeds],
        "train_lines": len(t.train),
    }


def load_mixture(t: Telemetry, seed: int, vocabulary: str) -> dict[str, np.ndarray]:
    path = t.layout.mixture(seed, vocabulary)
    return t.cached(("mixture", path), lambda: load_npz(path))


def load_reference(t: Telemetry, seed: int) -> dict[str, np.ndarray]:
    path = t.layout.programs(seed) / "reference.npz"
    return t.cached(("reference", path), lambda: load_npz(path))


def vocabulary_labels(t: Telemetry, seed: int, vocabulary: str) -> list[str]:
    if vocabulary == cs.PROGRAMS:
        reference = load_reference(t, seed)
        kinds = reference["kinds"]
        return [
            str(n)
            for n, k in zip(reference["programs"], kinds, strict=True)
            if k != "cycle"
        ]
    return [str(s) for s in load_mixture(t, seed, vocabulary)["states"]]


def project_line(
    t: Telemetry, model_id: str, second: bool = False
) -> dict[str, np.ndarray]:
    """Memberships of every cell of a line (or its second deposit) under every
    vocabulary at every fitting seed (``<vocabulary>@seed<s>``); a vocabulary
    whose representation the deposit lacks (Tx1 for a second deposit) is
    absent."""
    rep = representations(t, model_id, t.seeds, second)
    out: dict[str, np.ndarray] = {}
    for seed in t.seeds:
        for vocabulary in cs.vocabularies(t.config):
            key = cs.membership_key(vocabulary, seed)
            if vocabulary == cs.PROGRAMS:
                reference = load_reference(t, seed)
                usages = np.load(usage_path(t.layout, seed, model_id, second))
                out[key] = cs.program_memberships(usages, reference["kinds"])
                continue
            name = (
                "cycle"
                if vocabulary == cs.CYCLE
                else cs.MIXTURE_FAMILIES[cs.family(vocabulary)][0]
            )
            if name not in rep:
                continue
            params = load_mixture(t, seed, vocabulary)
            features = _features(rep, name, seed)
            centre = features.mean(axis=0) if bool(params["centred"]) else None
            out[key] = cs.project(params, features, centre)
    return {k: v.astype(np.float32) for k, v in out.items()}


def read_memberships(
    t: Telemetry, model_id: str, second: bool = False
) -> dict[str, np.ndarray]:
    """Memberships with rows renormalised to one in float64."""
    stored = load_npz(t.layout.memberships(model_id, second))
    out = {}
    for key, value in stored.items():
        q = value.astype(np.float64)
        out[key] = q / q.sum(axis=1, keepdims=True)
    return out


def second_cells(t: Telemetry, model_id: str) -> None:
    """A second deposit's per-cell QC and cycle scores and HVG cells, as Task 2
    computes them for the primary deposit (no doublet score)."""
    path = t.layout.second_cells(model_id)
    if path.is_file():
        return
    from src.data.basal import symbol_column
    from src.data.ladder_cells import cell_qc, hvg_cells, line_rng
    from src.data.noise_inputs import program_scores
    from src.data.tx1_cache import read_basal_source
    from src.experiments.prepare import NOISE_DIR

    noise = json.loads((t.setup.prepared / NOISE_DIR / "manifest.json").read_text())[
        "settings"
    ]
    cycle = {name: noise["programs"][name] for name in ("S", "G2M")}
    source = read_basal_source(
        t.second_sources[model_id],
        model_id=model_id,
        var_ensembl_col=t.preparation["var_ensembl_col"],
    )
    symbols = source.var[symbol_column(source.var)].astype(str).tolist()
    scores, _ = program_scores(
        source.X, symbols, noise["genes"], cycle, float(noise["target_sum"])
    )
    hvg = hvg_cells(
        source.X,
        symbols,
        t.recorded["hvg_order"],
        t.recorded["prepared_target_sum"],
        rng=line_rng(model_id, t.seed, "second_hvg"),
    )
    qc = np.column_stack(
        [cell_qc(source.X, symbols), scores, np.full(source.n_obs, np.nan)]
    ).astype(np.float32)
    write_npz(
        path, cell_ids=np.asarray(source.obs_names, dtype=str), qc=qc, hvg=hvg.cells
    )


def step_project(t: Telemetry) -> dict[str, Any]:
    keys = {cs.membership_key(v, s) for s in t.seeds for v in cs.vocabularies(t.config)}
    done = 0
    for model_id in t.projected:
        path = t.layout.memberships(model_id)
        if path.is_file():
            with np.load(path) as payload:
                if set(payload.files) == keys:
                    continue
        write_npz(path, **project_line(t, model_id))
        done += 1
    second = 0
    for model_id in t.second_sources:
        if t.layout.memberships(model_id, True).is_file():
            continue
        second_cells(t, model_id)
        write_npz(
            t.layout.memberships(model_id, True), **project_line(t, model_id, True)
        )
        second += 1
    return {
        "lines": len(t.projected),
        "projected": done,
        "second_deposits": len(t.second_sources),
        "second_projected": second,
        "roles": list(t.config["project"]["roles"]),
    }

"""State telemetry (PIE stage-one design §3.5, Task 8; protocol §17), project env.

``python -m src.experiments.state_telemetry CONFIG --steps fit,project,inputs``
then ``--steps report`` (``hpc/run.sh cell-states``). The ladder config the states
config names gives the ladder root, whose ``states/`` this writes. Steps, each
skipping what exists, so rerunning resumes:

- ``fit``: per fitting seed, the cells each single-cell training line
  contributes (``fit/sample_seed<s>.parquet``: at most ``fit.cells_per_line``
  distinct cells per line, so smaller lines weigh less; each cell's chemistry
  from the reviewed table by its deposit, line and arm); ``fit/plan.json`` and, in
  ``.venv-states`` (``STATES_PYTHON``), ``src.data.prepare.state_programs``:
  consensus NMF programs and their starCAT-style reference, scVI, and program
  usages and scVI latents for every cell of the projected lines and their second
  deposits (``programs/``, ``scvi/``); then the mixtures (``fit/seed<s>/
  <vocabulary>.npz``): the cycle on S and G2/M, and K in ``fit.mixture_counts``
  on scVI's latent (line-centred and absolute), the line-centred HVG cells' and
  Tx1-3B embeddings' principal components, and the line-centred, standardised QC
  columns.
- ``project``: every vocabulary at every fitting seed on every cell of the lines
  in ``project.roles`` (``memberships/<ModelID>.npz``) and of their second
  deposits (``second/``; no Tx1 there).
- ``inputs``: one raw-h5ad pass per projected line: each set's state masses and
  inputs in PIE's input space at the first fitting seed, merged below
  ``inputs.merge_effective_count``, its permuted memberships at every
  permutation seed, the cycling-weighted input and the whole-line input
  (``lines/<ModelID>.npz``); per-cell reference-program scores and QC values
  weighted by each state (identity and confound telemetry); then per set
  ``<set>/inputs.parquet`` and ``<set>/masses.parquet``, the frozen PIE's
  contexts (``hpc/run.sh pie-precompute CONFIG --steps states``).
- ``report``: ``<output_root>/<config stem>/results.md`` and ``report.json``:
  states' identity against Kinker's, Gavish's, hallmark and Tirosh's programs;
  the mass explained by deposit and by QC; recurrence across lines; the
  centred-log-ratio SVD of the line x state masses; agreement between a line's
  two deposits; stability across fitting seeds (states matched by the
  Hungarian method); the frozen PIE's gap ``sum_k pi_k D(x_k) - D(x)`` for dH,
  dE and the own slots, against its ``|whole`` context and the main pass.

The frozen PIE's state outputs live under ``states/pie/``, so moving ``states/``
aside moves them too; each line's carries the fingerprint of the state inputs
it was computed from, and the report refuses outputs of other inputs.

It reports and decides nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data import cell_states as cs

_LOGGER = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
STEPS: Final[tuple[str, ...]] = ("fit", "project", "inputs", "report")
MANIFEST: Final[str] = "manifest.json"
#: Largest difference allowed between the mass-weighted state inputs and the
#: line's PIE input (float32 storage of the latter).
IDENTITY_TOLERANCE: Final[float] = 1e-4
#: Groups of the reference-program table, in report order.
REFERENCE_GROUPS: Final[tuple[str, ...]] = ("kinker", "gavish", "hallmarks", "tirosh")
QC_NAMES: Final[tuple[str, ...]] = ("log_umi", "genes_detected", "mito_fraction")
DELTAS: Final[tuple[str, ...]] = ("dH", "dE", "own_h", "own_dp")


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _npz_keys(path: Path, *keys: str) -> dict[str, np.ndarray]:
    """Only these arrays of an npz (each member is read on its own)."""
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in keys}


# --- the run --------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """``<ladder_root>/states/``, the frozen PIE's state outputs (``pie/``) in it."""

    root: Path

    @property
    def fit(self) -> Path:
        return self.root / "fit"

    @property
    def plan(self) -> Path:
        return self.fit / "plan.json"

    def sample(self, seed: int) -> Path:
        return self.fit / f"sample_seed{seed}.parquet"

    def mixture(self, seed: int, vocabulary: str) -> Path:
        return self.fit / f"seed{seed}" / f"{vocabulary}.npz"

    def programs(self, seed: int) -> Path:
        return self.root / "programs" / f"seed{seed}"

    def scvi(self, seed: int) -> Path:
        return self.root / "scvi" / f"seed{seed}"

    def memberships(self, model_id: str, second: bool = False) -> Path:
        base = self.root / "second" if second else self.root
        return base / "memberships" / f"{model_id}.npz"

    def second_cells(self, model_id: str) -> Path:
        return self.root / "second" / "cells" / f"{model_id}.npz"

    def line(self, model_id: str) -> Path:
        return self.root / "lines" / f"{model_id}.npz"

    def state_set(self, name: str) -> Path:
        return self.root / name

    @property
    def pie_states(self) -> Path:
        return self.root / "pie"


def usage_path(layout: Layout, seed: int, model_id: str, second: bool) -> Path:
    folder = "usages_second" if second else "usages"
    return layout.programs(seed) / folder / f"{model_id}.npy"


def latent_path(layout: Layout, seed: int, model_id: str, second: bool) -> Path:
    folder = "latent_second" if second else "latent"
    return layout.scvi(seed) / folder / f"{model_id}.npy"


@dataclass
class Telemetry:
    """The states config, the ladder setup and tables it reads."""

    config: Mapping[str, Any]
    name: str
    #: Frozen fits and per-line summaries read more than once in a run.
    cache: dict[Any, Any] = field(default_factory=dict)

    def cached(self, key: Any, make: Any) -> Any:
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def cell_arrays(self, model_id: str, *keys: str) -> list[np.ndarray]:
        """Arrays of a line's ``cells/<ModelID>.npz``, reading only these keys."""
        path = self.ladder_root / "cells" / f"{model_id}.npz"
        with np.load(path, allow_pickle=False) as payload:
            return [payload[key] for key in keys]

    @cached_property
    def setup(self) -> Any:
        from src.experiments.config import load_ladder_config
        from src.experiments.ladder_inputs import load_setup

        return load_setup(load_ladder_config(_absolute(self.config["ladder_config"])))

    @property
    def ladder_root(self) -> Path:
        return _absolute(self.setup.config["ladder_root"])

    @property
    def layout(self) -> Layout:
        return Layout(self.ladder_root / "states")

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.config["fit"]["seeds"]]

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @cached_property
    def lines(self) -> pd.DataFrame:
        from src.experiments.ladder_inputs import read_lines

        return read_lines(self.ladder_root)

    @cached_property
    def split(self) -> Any:
        from src.data.splits import load_geneeffect_split

        return load_geneeffect_split(_absolute(self.setup.joint["paths"]["split"]))

    @property
    def train(self) -> list[str]:
        return list(self.lines.index[self.lines["role"] == "train_cells"])

    @property
    def projected(self) -> list[str]:
        roles = list(self.config["project"]["roles"])
        return list(self.lines.index[self.lines["role"].isin(roles)])

    @cached_property
    def panel(self) -> list[str]:
        genes = pd.read_parquet(self.ladder_root / "genes.parquet")
        return list(genes.index[genes["in_panel"]])

    @cached_property
    def cell_genes(self) -> list[str]:
        path = self.ladder_root / "cells" / "genes.json"
        return json.loads(path.read_text())["genes"]

    @cached_property
    def pie_input(self) -> pd.DataFrame:
        return pd.read_parquet(self.ladder_root / "mean" / "pie_input.parquet")

    @cached_property
    def pie_listed(self) -> pd.DataFrame:
        return pd.read_parquet(self.ladder_root / "mean" / "pie_listed.parquet")

    @cached_property
    def deposits(self) -> pd.Series:
        """Each split line's deposit (the noise diagnosis's ``_deposits``)."""
        from src.data.prepare.plan_second_sources import PRIMARY_KEYS

        path = _absolute(self.setup.joint["paths"]["split"]).with_suffix(".csv")
        table = pd.read_csv(path)
        sources = table.set_index("model_id")["expression_source"].astype(str)
        return sources.replace(PRIMARY_KEYS)

    @cached_property
    def chemistry_table(self) -> pd.DataFrame:
        path = _absolute(self.config["references"]["chemistry"])
        return pd.read_csv(path, keep_default_na=False, dtype=str)

    def cell_chemistry(self, model_id: str, cell_ids: np.ndarray) -> np.ndarray:
        """Each cell's chemistry from its line's primary deposit and the cell's
        arm in the h5ad's obs (none: the whole line), through the reviewed table
        (``cell_states.resolve_chemistry``, which fails closed)."""
        import anndata as ad

        obs = ad.read_h5ad(self.source(model_id), backed="r").obs
        if list(obs.index.astype(str)) != [str(c) for c in cell_ids]:
            raise ValueError(f"{model_id}: the h5ad's cells are not the line's cells")
        arms = obs["arm"].astype(str).to_numpy() if "arm" in obs else [""] * len(obs)
        return cs.resolve_chemistry(
            self.chemistry_table, str(self.deposits[model_id]), model_id, arms
        )

    @cached_property
    def second_sources(self) -> dict[str, Path]:
        """Second-deposit h5ads of the projected lines (noise diagnosis's
        ``deposit.second_source_root``)."""
        root = _absolute(self.setup.noise["deposit"]["second_source_root"])
        return {
            m: root / f"{m}.h5ad"
            for m in self.projected
            if (root / f"{m}.h5ad").is_file()
        }

    def source(self, model_id: str) -> Path:
        return (
            _absolute(self.setup.joint["paths"]["basal_raw_umi"]) / f"{model_id}.h5ad"
        )

    @cached_property
    def preparation(self) -> Mapping[str, Any]:
        return self.setup.joint["preparation"]

    @cached_property
    def recorded(self) -> Mapping[str, Any]:
        from src.experiments.ladder_inputs import settings

        return settings(self.setup)

    def references(self) -> dict[str, list[str]]:
        """Reference programs with a member on the panel (symbols matched
        upper-case; a program none of whose symbols is a panel gene, such as an
        outdated alias, cannot be scored), named ``<group>:<program>``."""
        refs = self.config["references"]
        panel = set(self.panel)
        out: dict[str, list[str]] = {}
        for group in REFERENCE_GROUPS:
            table = pd.read_csv(_absolute(refs[group]), sep="\t")
            key = table.columns[0]
            for name, members in table.groupby(key, sort=False):
                genes = list(members["gene"].astype(str))
                if {g.upper() for g in genes} & panel:
                    out[f"{group}:{name}"] = genes
        return out

    def tirosh(self) -> dict[str, list[str]]:
        table = pd.read_csv(_absolute(self.config["references"]["tirosh"]), sep="\t")
        return {p: list(g["gene"].astype(str)) for p, g in table.groupby("phase")}


def settings(t: Telemetry) -> dict[str, Any]:
    """What changes the states root's contents (the projected roles may grow)."""
    return {
        "seed": t.config["seed"],
        "ladder_config": t.config["ladder_config"],
        **{key: t.config[key] for key in ("fit", "programs", "scvi", "inputs")},
        "references": dict(t.config["references"]),
    }


def open_manifest(t: Telemetry) -> dict[str, Any]:
    path = t.layout.root / MANIFEST
    current = settings(t)
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != current:
            raise ValueError(
                f"{t.layout.root} was written under other settings; move it aside "
                "(it holds the frozen PIE's state outputs too, under pie/)"
            )
        return manifest
    manifest = {"settings": current, "steps": {}}
    _write_json(path, manifest)
    return manifest


def record_step(t: Telemetry, step: str, payload: Mapping[str, Any]) -> None:
    path = t.layout.root / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["steps"][step] = {
        **payload,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(path, manifest)


# --- per-cell representations -------------------------------------------------------


def _tx1(t: Telemetry, model_id: str, cell_ids: np.ndarray) -> np.ndarray:
    from src.data.tx1_cache import load_line_cache

    root = _absolute(t.setup.config["paths"]["tx1_cache"]) / "real"
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
        cells = _load_npz(t.layout.second_cells(model_id))
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


# --- fit ---------------------------------------------------------------------------


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
        _write_parquet(path, sample)


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
        "hvg_gene_symbol_col": t.preparation["hvg_gene_symbol_col"],
        "encode": {m: str(t.source(m)) for m in t.projected},
        "second": {m: str(p) for m, p in t.second_sources.items()},
        "samples": {str(s): str(t.layout.sample(s)) for s in t.seeds},
    }
    _write_json(t.layout.plan, plan)
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
    """``src.data.prepare.state_programs`` in ``.venv-states`` (``STATES_PYTHON``)."""
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
            "src.data.prepare.state_programs",
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
        _save_npz(path, **params, centred=False, states=np.asarray(labels))
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
            _save_npz(
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


# --- project -------------------------------------------------------------------------


def load_mixture(t: Telemetry, seed: int, vocabulary: str) -> dict[str, np.ndarray]:
    path = t.layout.mixture(seed, vocabulary)
    return t.cached(("mixture", path), lambda: _load_npz(path))


def load_reference(t: Telemetry, seed: int) -> dict[str, np.ndarray]:
    path = t.layout.programs(seed) / "reference.npz"
    return t.cached(("reference", path), lambda: _load_npz(path))


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
    stored = _load_npz(t.layout.memberships(model_id, second))
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
    symbols = (
        source.var[symbol_column(source.var, t.preparation["hvg_gene_symbol_col"])]
        .astype(str)
        .tolist()
    )
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
    _save_npz(
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
        _save_npz(path, **project_line(t, model_id))
        done += 1
    second = 0
    for model_id in t.second_sources:
        if t.layout.memberships(model_id, True).is_file():
            continue
        second_cells(t, model_id)
        _save_npz(
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


# --- inputs -------------------------------------------------------------------------


def line_cells(t: Telemetry, model_id: str) -> tuple[cs.LineCells, Any, list[str]]:
    """A line's per-cell PIE input, its raw counts and source symbols."""
    from src.data.basal import symbol_column
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        t.source(model_id),
        model_id=model_id,
        var_ensembl_col=t.preparation["var_ensembl_col"],
    )
    symbols = (
        source.var[symbol_column(source.var, t.preparation["hvg_gene_symbol_col"])]
        .astype(str)
        .tolist()
    )
    cells = cs.per_cell_log_cp10k(
        source.X,
        symbols,
        t.panel,
        t.pie_input.loc[model_id].to_numpy(dtype=np.float64),
        t.pie_listed.loc[model_id].to_numpy(dtype=bool),
    )
    return cells, source.X, symbols


def _check_identity(
    model_id: str, name: str, values: np.ndarray, line: np.ndarray
) -> None:
    """Fail closed when a set's states do not add up to the line's PIE input."""
    difference = float(np.max(np.abs(values - line)))
    if not difference <= IDENTITY_TOLERANCE:
        raise ValueError(
            f"{model_id} {name}: mass-weighted state inputs differ from the line's "
            f"PIE input by {difference:.3g}"
        )


def line_inputs(t: Telemetry, model_id: str) -> dict[str, Any]:
    """Every set's states of one line (``lines/<ModelID>.npz``)."""
    from src.data.noise_inputs import program_scores

    cells, counts, symbols = line_cells(t, model_id)
    line = t.pie_input.loc[model_id].to_numpy(dtype=np.float64)
    _check_identity(model_id, "whole", cells.mean(), line)
    memberships = read_memberships(t, model_id)
    minimum = float(t.config["inputs"]["merge_effective_count"])
    out: dict[str, Any] = {"whole": cells.mean().astype(np.float32)}
    states: dict[str, tuple[cs.StateInputs, dict[int, cs.StateInputs]]] = {}
    permutation_seeds = [int(s) for s in t.config["inputs"]["permutation_seeds"]]
    for set_name in cs.state_sets(t.config):
        if set_name == cs.CYCLING_SET:
            q = memberships[cs.membership_key(cs.CYCLE, t.seed)]
            labels = vocabulary_labels(t, t.seed, cs.CYCLE)
            weights = cs.cycling_weights(
                cs.cycling_probability(q, labels),
                float(t.config["inputs"]["cycling_floor"]),
            )
            inputs = cells.weighted_inputs(weights[:, None])
            state = cs.StateInputs(
                ["0"], np.ones(1), cs.effective_count(weights[:, None]), inputs, [[]]
            )
        else:
            vocabulary = cs.base_vocabulary(set_name)
            if vocabulary not in states:
                # Merged once on the real memberships; the permuted sets permute
                # the merged rows, so they keep the real states and masses.
                states[vocabulary] = cs.permuted_state_inputs(
                    memberships[cs.membership_key(vocabulary, t.seed)],
                    cells,
                    vocabulary_labels(t, t.seed, vocabulary),
                    minimum,
                    model_id=model_id,
                    vocabulary=vocabulary,
                    seeds=permutation_seeds
                    if cs.family(vocabulary) in cs.PERMUTED_FAMILIES
                    else [],
                )
            real, permuted = states[vocabulary]
            state = (
                real
                if set_name == vocabulary
                else permuted[int(set_name.rpartition("__perm")[2])]
            )
            _check_identity(model_id, set_name, state.mass @ state.inputs, line)
        out[f"inputs__{set_name}"] = state.inputs.astype(np.float32)
        out[f"mass__{set_name}"] = state.mass
        out[f"effective__{set_name}"] = state.effective
        out[f"labels__{set_name}"] = np.asarray(state.labels)
        out[f"merged__{set_name}"] = np.asarray([",".join(m) for m in state.merged])
    # Identity and confound telemetry on the unmerged states of every vocabulary.
    references = t.references()
    scores, listed = program_scores(
        counts, symbols, t.panel, references, float(t.recorded["prepared_target_sum"])
    )
    scores = scores.astype(np.float64)
    scores -= scores.mean(axis=0)
    qc = t.cell_arrays(model_id, "qc")[0][:, list(cs.QC_FEATURES)].astype(np.float64)
    qc -= qc.mean(axis=0)
    for vocabulary in cs.vocabularies(t.config):
        q = memberships[cs.membership_key(vocabulary, t.seed)]
        total = q.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"identity__{vocabulary}"] = (q.T @ scores) / total[:, None]
            out[f"qcshift__{vocabulary}"] = (q.T @ qc) / total[:, None]
    out["reference_programs"] = np.asarray(list(references))
    out["reference_listed"] = np.asarray([listed[name] for name in references])
    out["n_cells"] = np.asarray(cells.n_cells)
    return out


def assemble_sets(t: Telemetry) -> dict[str, Any]:
    """``<set>/inputs.parquet`` (contexts x panel, the line's ``|whole`` context
    after its states) and ``<set>/masses.parquet`` for every set."""
    counts = {}
    for set_name in cs.state_sets(t.config):
        names, rows, table = [], [], []
        for model_id in t.projected:
            line = _npz_keys(
                t.layout.line(model_id),
                *(f"{part}__{set_name}" for part in ("labels", "inputs", "mass")),
                *(f"{part}__{set_name}" for part in ("effective", "merged")),
                "whole",
                "n_cells",
            )
            labels = [str(s) for s in line[f"labels__{set_name}"]]
            for k, label in enumerate(labels):
                names.append(cs.context_name(model_id, set_name, label))
                rows.append(line[f"inputs__{set_name}"][k])
                table.append(
                    {
                        "model_id": model_id,
                        "state": label,
                        "mass": float(line[f"mass__{set_name}"][k]),
                        "effective_count": float(line[f"effective__{set_name}"][k]),
                        "merged": str(line[f"merged__{set_name}"][k]),
                    }
                )
            names.append(cs.context_name(model_id, set_name, cs.WHOLE))
            rows.append(line["whole"])
            table.append(
                {
                    "model_id": model_id,
                    "state": cs.WHOLE,
                    "mass": float("nan"),
                    "effective_count": float(line["n_cells"]),
                    "merged": "",
                }
            )
        index = pd.Index(names, name="context")
        inputs = pd.DataFrame(np.vstack(rows), index=index, columns=t.panel)
        _write_parquet(t.layout.state_set(set_name) / "inputs.parquet", inputs)
        _write_parquet(
            t.layout.state_set(set_name) / "masses.parquet",
            pd.DataFrame(table, index=index),
        )
        counts[set_name] = len(names)
    return counts


def step_inputs(t: Telemetry) -> dict[str, Any]:
    done = 0
    for model_id in t.projected:
        if t.layout.line(model_id).is_file():
            continue
        _save_npz(t.layout.line(model_id), **line_inputs(t, model_id))
        done += 1
    contexts = assemble_sets(t)
    return {
        "lines": len(t.projected),
        "computed": done,
        "contexts": contexts,
        "contexts_total": int(sum(contexts.values())),
    }


# --- report --------------------------------------------------------------------------


def membership_masses(t: Telemetry, second: bool = False) -> dict[str, dict[str, Any]]:
    """Per line (or second deposit), every membership key's masses and the cell
    count, each memberships file read once."""

    def make() -> dict[str, dict[str, Any]]:
        lines = list(t.second_sources) if second else t.projected
        out = {}
        for model_id in lines:
            memberships = read_memberships(t, model_id, second)
            out[model_id] = {key: cs.masses(q) for key, q in memberships.items()}
            out[model_id]["n"] = next(iter(memberships.values())).shape[0]
        return out

    return t.cached(("masses", second), make)


def line_masses(
    t: Telemetry, vocabulary: str, seed: int, lines: Sequence[str], second: bool = False
) -> pd.DataFrame:
    """Lines x states masses (unmerged) of one vocabulary at one seed."""
    key = cs.membership_key(vocabulary, seed)
    table = membership_masses(t, second)
    rows = {m: table[m][key] for m in lines}
    return pd.DataFrame(rows, index=vocabulary_labels(t, seed, vocabulary)).T


def identity_section(t: Telemetry, vocabulary: str) -> dict[str, Any]:
    """Each state's mass-weighted mean, over training lines, of its cells'
    line-centred reference-program scores; the three highest programs and every
    group's best."""
    lines = t.train
    labels = vocabulary_labels(t, t.seed, vocabulary)
    table = membership_masses(t)
    key = cs.membership_key(vocabulary, t.seed)
    stacked, weights, shares = [], [], []
    names: list[str] = []
    for model_id in lines:
        line = _npz_keys(
            t.layout.line(model_id),
            "reference_programs",
            "reference_listed",
            f"identity__{vocabulary}",
        )
        names = [str(n) for n in line["reference_programs"]]
        shares.append(line["reference_listed"])
        stacked.append(np.nan_to_num(line[f"identity__{vocabulary}"]))
        weights.append(table[model_id][key] * table[model_id]["n"])
    weights = np.asarray(weights)
    total = weights.sum(axis=0)
    mean = (
        np.einsum("lk,lkp->kp", weights, np.asarray(stacked))
        / np.maximum(total, 1e-12)[:, None]
    )
    frame = pd.DataFrame(mean, index=labels, columns=names)
    listed = dict(zip(names, np.mean(shares, axis=0).tolist(), strict=True))
    groups = {
        group: [n for n in names if n.startswith(f"{group}:")]
        for group in REFERENCE_GROUPS
    }
    states = {}
    for label in labels:
        row = frame.loc[label].sort_values(ascending=False)
        states[label] = {
            "top": [[name, float(value)] for name, value in row.head(3).items()],
            "best_per_group": {
                group: [best, float(row[best])]
                for group, members in groups.items()
                if members
                for best in [row[members].idxmax()]
            },
        }
    return {
        "states": states,
        "programs_scored": {group: len(members) for group, members in groups.items()},
        # The share of each group's members the training lines' sources list.
        "members_listed": {
            group: float(np.mean([listed[n] for n in members]))
            for group, members in groups.items()
            if members
        },
    }


def line_qc(t: Telemetry) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each training line's median and SD of log UMI, genes detected and
    mitochondrial fraction over its cells."""

    def make() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for model_id in t.train:
            qc = t.cell_arrays(model_id, "qc")[0][:, list(cs.QC_FEATURES)]
            qc = qc.astype(np.float64)
            out[model_id] = (np.median(qc, axis=0), qc.std(axis=0))
        return out

    return t.cached("line_qc", make)


def per_state_group_share(
    frame: pd.DataFrame,
    groups: pd.Series,
    strata: pd.Series,
    *,
    permutations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Per state (column of lines x states masses), the share of its mass's
    variance across lines explained by ``groups`` (eta squared), against
    ``groups`` permuted among the lines of each stratum (the noise diagnosis's
    null, state by state)."""
    from src.context_prior.noise import _between_share

    lines = list(frame.index)
    labels = groups.loc[lines].astype(str).to_numpy()
    levels = np.asarray(sorted(set(labels)))
    values = frame.to_numpy(dtype=np.float64)
    finite = np.isfinite(values).astype(np.float64)
    values = np.where(finite > 0, values, 0.0)
    stratum = strata.loc[lines].to_numpy()
    blocks = [np.flatnonzero(stratum == s) for s in sorted(set(stratum))]

    def share(assigned: np.ndarray) -> np.ndarray:
        onehot = (assigned[:, None] == levels[None, :]).astype(np.float64)
        return _between_share(values, finite, onehot)

    observed = share(labels)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, values.shape[1]))
    for k in range(permutations):
        shuffled = labels.copy()
        for block in blocks:
            shuffled[block] = labels[rng.permutation(block)]
        null[k] = share(shuffled)
    # A state whose mass does not vary has no defined share: NaN, not a p-value.
    return {
        str(state): {
            "observed": float(observed[j]),
            "null_q95": float(np.nanquantile(null[:, j], 0.95))
            if np.isfinite(observed[j])
            else float("nan"),
            "p_value": float(
                (1 + (null[:, j] >= observed[j]).sum()) / (1 + permutations)
            )
            if np.isfinite(observed[j])
            else float("nan"),
        }
        for j, state in enumerate(frame.columns)
    }


def confound_section(
    t: Telemetry, vocabulary: str, frame: pd.DataFrame
) -> dict[str, Any]:
    """Mass explained by deposit (eta squared against permutations within
    lineage) and by the lines' median QC (R squared), and each state's
    within-line QC shift in pooled within-line SD units."""
    from src.context_prior.noise import group_share

    train = list(frame.index)
    deposit = t.deposits.loc[train]
    counts = deposit.value_counts()
    small = set(counts.index[counts < int(t.setup.noise["deposit"]["min_lines"])])
    grouped = deposit.where(~deposit.isin(small), "other")
    lineage = t.lines.loc[train, "lineage"].astype(str)
    share = group_share(
        frame,
        grouped,
        lineage,
        permutations=int(t.config["report"]["permutations"]),
        seed=t.seed,
    )
    per_state = per_state_group_share(
        frame,
        grouped,
        lineage,
        permutations=int(t.config["report"]["permutations"]),
        seed=t.seed,
    )
    summaries = line_qc(t)
    medians, spreads, shifts, weights = [], [], [], []
    for model_id in train:
        medians.append(summaries[model_id][0])
        spreads.append(summaries[model_id][1])
        key = f"qcshift__{vocabulary}"
        shifts.append(np.nan_to_num(_npz_keys(t.layout.line(model_id), key)[key]))
        weights.append(frame.loc[model_id].to_numpy())
    design = np.column_stack([np.ones(len(train)), np.asarray(medians)])
    r2 = {}
    for state in frame.columns:
        y = frame[state].to_numpy()
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = ((y - y.mean()) ** 2).sum()
        r2[state] = (
            float(1 - ((y - fitted) ** 2).sum() / total) if total > 0 else float("nan")
        )
    pooled_sd = np.sqrt(np.mean(np.asarray(spreads) ** 2, axis=0))
    w = np.asarray(weights)
    shift = (
        np.einsum("lk,lkq->kq", w, np.asarray(shifts))
        / np.maximum(w.sum(axis=0), 1e-12)[:, None]
    )
    return {
        "deposit_per_state": per_state,
        # The mean over states of the same share (noise.group_share).
        "deposit": share,
        "pooled_as_other": sorted(small),
        "qc_r2": r2,
        "qc_shift_sd": {
            state: dict(zip(QC_NAMES, (shift[k] / pooled_sd).tolist(), strict=True))
            for k, state in enumerate(frame.columns)
        },
    }


def composition_section(
    frame: pd.DataFrame, mass: float, floor: float
) -> dict[str, Any]:
    """Recurrence (share of lines holding each state above ``mass``), each
    state's spread across lines, and the singular values of the centred
    log-ratio masses (its K - 1 contrasts)."""
    values = np.maximum(frame.to_numpy(dtype=np.float64), floor)
    values /= values.sum(axis=1, keepdims=True)
    clr = np.log(values) - np.log(values).mean(axis=1, keepdims=True)
    clr -= clr.mean(axis=0)
    singular = np.linalg.svd(clr, compute_uv=False)[: max(frame.shape[1] - 1, 1)]
    energy = singular**2
    held = (frame.to_numpy() >= mass).sum(axis=1)
    return {
        "recurrence": {s: float((frame[s] >= mass).mean()) for s in frame.columns},
        "mass_mean": {s: float(frame[s].mean()) for s in frame.columns},
        "mass_sd": {s: float(frame[s].std()) for s in frame.columns},
        "states_per_line_median": float(np.median(held)),
        "clr_variance_share": (energy / energy.sum()).tolist()
        if energy.sum() > 0
        else [],
    }


def second_section(t: Telemetry, vocabulary: str) -> dict[str, Any] | None:
    """Agreement of state masses between a line's two deposits: per state the
    Pearson correlation across lines, per line the total-variation distance,
    beside the distance between different lines' primary deposits."""
    key = cs.membership_key(vocabulary, t.seed)
    seconds = membership_masses(t, second=True)
    lines = [m for m in t.second_sources if key in seconds[m]]
    if len(lines) < 3:
        return None
    primary = line_masses(t, vocabulary, t.seed, lines)
    second = line_masses(t, vocabulary, t.seed, lines, second=True)
    distance = 0.5 * np.abs(primary.to_numpy() - second.to_numpy()).sum(axis=1)
    values = primary.to_numpy()
    other = [
        0.5 * np.abs(values[i] - values[j]).sum()
        for i in range(len(lines))
        for j in range(i + 1, len(lines))
    ]
    pearson = {}
    for state in primary.columns:
        a, b = primary[state].to_numpy(), second[state].to_numpy()
        pearson[state] = (
            float(np.corrcoef(a, b)[0, 1])
            if a.std() > 0 and b.std() > 0
            else float("nan")
        )
    return {
        "lines": len(lines),
        "pearson": pearson,
        "tvd_median": float(np.median(distance)),
        "tvd_between_lines_median": float(np.median(other)),
    }


def stability_section(t: Telemetry, vocabulary: str) -> dict[str, Any]:
    """States of each pair of fitting seeds matched by the Hungarian method on
    the correlation of their memberships over a fixed draw of cells per training
    line (``matched_states``)."""
    from src.data.ladder_cells import line_rng

    def draw() -> dict[str, np.ndarray]:
        """Every membership key's rows of a fixed draw of cells per line, each
        memberships file read once."""
        per_line = int(t.config["report"]["stability_cells"])
        parts: dict[str, list[np.ndarray]] = {}
        for model_id in t.train:
            memberships = read_memberships(t, model_id)
            n = next(iter(memberships.values())).shape[0]
            rows = line_rng(model_id, 0, "state_stability").permutation(n)[:per_line]
            for key, q in memberships.items():
                parts.setdefault(key, []).append(q[rows])
        return {key: np.vstack(values) for key, values in parts.items()}

    drawn = t.cached("stability_draw", draw)
    table = membership_masses(t)
    stacked = {s: drawn[cs.membership_key(vocabulary, s)] for s in t.seeds}
    line_masses_by_seed = {
        s: [table[m][cs.membership_key(vocabulary, s)] for m in t.train]
        for s in t.seeds
    }
    return {
        f"{a}-{b}": matched_states(
            stacked[a],
            stacked[b],
            np.asarray(line_masses_by_seed[a]),
            np.asarray(line_masses_by_seed[b]),
        )
        for i, a in enumerate(t.seeds)
        for b in t.seeds[i + 1 :]
    }


def matched_states(
    qa: np.ndarray, qb: np.ndarray, ma: np.ndarray, mb: np.ndarray
) -> dict[str, Any]:
    """Two fits' states matched by the Hungarian method on the correlation of
    their memberships (cells x states ``qa``, ``qb``); per line (lines x states
    masses ``ma``, ``mb``), the total-variation distance with every unmatched
    state set against a zero-mass counterpart, and the unmatched mass."""
    from scipy.optimize import linear_sum_assignment

    ka = qa.shape[1]
    corr = np.nan_to_num(np.corrcoef(qa.T, qb.T)[:ka, ka:])
    rows, cols = linear_sum_assignment(-corr)
    unmatched_a = np.setdiff1d(np.arange(ka), rows)
    unmatched_b = np.setdiff1d(np.arange(qb.shape[1]), cols)
    unmatched = ma[:, unmatched_a].sum(axis=1) + mb[:, unmatched_b].sum(axis=1)
    distance = 0.5 * (np.abs(ma[:, rows] - mb[:, cols]).sum(axis=1) + unmatched)
    return {
        "matched_correlation": corr[rows, cols].tolist(),
        "matched_mass_tvd_mean": float(distance.mean()),
        "unmatched_states": [int(unmatched_a.size), int(unmatched_b.size)],
        "unmatched_mass_mean": float(unmatched.mean()),
        "states": [int(ka), int(qb.shape[1])],
    }


def determinism_section(t: Telemetry) -> dict[str, Any]:
    """Validation lines from a deposit no training line comes from, projected
    again: whether their memberships are identical to the stored ones."""
    train_deposits = set(t.deposits.loc[t.train])
    unseen = [
        m
        for m in t.projected
        if t.lines.loc[m, "role"] != "train_cells"
        and t.deposits[m] not in train_deposits
    ]
    out = {}
    for model_id in unseen:
        again = project_line(t, model_id)
        stored = _load_npz(t.layout.memberships(model_id))
        out[model_id] = bool(
            set(again) == set(stored)
            and all(np.array_equal(again[k], stored[k]) for k in stored)
        )
    return {"lines": out, "deposits": {m: str(t.deposits[m]) for m in unseen}}


def _main_pass(
    pie_root: Path, gene: str, line_index: np.ndarray, basis: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The main pass's uncentred deltas of a gene on the given lines."""
    shard = _load_npz(pie_root / "genes" / f"{gene}.npz")
    if "dE" in shard:
        dE = shard["dE"]
    else:
        full = np.load(pie_root / "full" / f"{gene}.npy")
        dE = (full - basis["mean"]) @ basis["components"].T
    return {
        "dH": shard["dH"][line_index],
        "dE": np.asarray(dE)[line_index],
        "own_h": shard["own_h"][line_index],
        "own_dp": np.asarray(shard["own_dp"]).reshape(-1, 1)[line_index],
    }


def gap_section(t: Telemetry, set_name: str) -> dict[str, Any] | None:
    """The frozen PIE's gap ``sum_k pi_k D(x_k) - D(x)`` per (line, gene), for
    each delta, against the set's own ``|whole`` context; its size and its
    spread across lines within a gene, each relative to the spread across lines
    of the whole-line delta within the gene; and the noise floor, the
    ``|whole`` context against the main pass, on the same scale."""
    folder = t.layout.pie_states / set_name
    if not (folder / "index.json").is_file():
        return None
    index = json.loads((folder / "index.json").read_text())
    current = cs.set_fingerprints(
        pd.read_parquet(t.layout.state_set(set_name) / "inputs.parquet"),
        pd.read_parquet(t.layout.state_set(set_name) / "masses.parquet"),
    )
    if index.get("fingerprints") != current:
        raise ValueError(
            f"{folder} was computed from other state inputs than "
            f"{t.layout.state_set(set_name)}; move it aside and rerun the pass"
        )
    contexts = pd.read_parquet(folder / "contexts.parquet")
    genes = list(index["genes"])
    pie_root = t.ladder_root / "pie"
    basis = _load_npz(pie_root / "basis.npz")
    means = _load_npz(pie_root / "frame_means.npz")
    panel_position = {g: i for i, g in enumerate(t.panel)}
    lines = list(dict.fromkeys(contexts["model_id"]))
    line_index = np.asarray([t.lines.index.get_loc(m) for m in lines])
    roles = t.lines.loc[lines, "role"].to_numpy()
    owner = contexts["model_id"].to_numpy()
    is_whole = contexts["state"].to_numpy() == cs.WHOLE
    mass = np.nan_to_num(contexts["mass"].to_numpy(dtype=np.float64))
    # Rows of each line's whole context, and the lines x contexts mixing matrix.
    whole_rows = np.asarray([np.flatnonzero((owner == m) & is_whole)[0] for m in lines])
    mixing = np.zeros((len(lines), len(contexts)))
    for i, model_id in enumerate(lines):
        states = (owner == model_id) & ~is_whole
        mixing[i, states] = mass[states]
    out: dict[str, Any] = {"lines": len(lines), "genes": len(genes)}
    for name in DELTAS:
        values = np.load(folder / f"{name}.npy", mmap_mode="r")
        # Per role: gap energy, gap spread, whole-line spread, noise energy.
        sums = {role: np.zeros(4) for role in ("train_cells", "val")}
        for g, gene in enumerate(genes):
            array = np.asarray(values[g], dtype=np.float64).reshape(len(contexts), -1)
            array = array + means[name][panel_position[gene]]
            whole = array[whole_rows]
            gap = mixing @ array - whole
            main = _main_pass(pie_root, gene, line_index, basis)[name]
            noise = whole - np.asarray(main, dtype=np.float64).reshape(len(lines), -1)
            for role, total in sums.items():
                keep = roles == role
                if keep.sum() < 2:
                    continue
                total += [
                    (gap[keep] ** 2).sum(),
                    ((gap[keep] - gap[keep].mean(axis=0)) ** 2).sum(),
                    ((whole[keep] - whole[keep].mean(axis=0)) ** 2).sum(),
                    (noise[keep] ** 2).sum(),
                ]
        out[name] = {
            role: {
                "gap_ratio": float(total[0] / total[2]),
                "gap_spread_ratio": float(total[1] / total[2]),
                "noise_ratio": float(total[3] / total[2]),
            }
            for role, total in sums.items()
            if total[2] > 0
        }
    run = t.layout.pie_states / "run.json"
    if run.is_file():
        timing = json.loads(run.read_text()).get("sets", {}).get(set_name)
        out["seconds_per_context"] = timing
    return out


def step_report(t: Telemetry) -> dict[str, Any]:
    out_dir = _absolute(t.config["output_root"]) / t.name
    report: dict[str, Any] = {"vocabularies": {}, "pie_gap": {}}
    for vocabulary in cs.vocabularies(t.config):
        frame = line_masses(t, vocabulary, t.seed, t.train)
        report["vocabularies"][vocabulary] = {
            "states": list(frame.columns),
            "identity": identity_section(t, vocabulary),
            "confound": confound_section(t, vocabulary, frame),
            "composition": composition_section(
                frame,
                float(t.config["report"]["recurrence_mass"]),
                float(t.config["report"]["clr_floor"]),
            ),
            "second_deposit": second_section(t, vocabulary),
            "stability": stability_section(t, vocabulary),
        }
    report["determinism"] = determinism_section(t)
    for set_name in cs.state_sets(t.config):
        report["pie_gap"][set_name] = gap_section(t, set_name)
    _write_json(out_dir / "report.json", report)
    (out_dir / "results.md").write_text(results_markdown(t, report))
    return {"output": str(out_dir)}


# --- results.md ---------------------------------------------------------------------


def _f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}"


def results_markdown(t: Telemetry, report: Mapping[str, Any]) -> str:
    lines = [
        f"# State telemetry ({t.name})",
        "",
        "Frozen state vocabularies fitted on the single-cell training lines (at "
        "most 500 distinct cells each) and projected on the lines of "
        f"{', '.join(t.config['project']['roles'])}; the frozen PIE on their state "
        "inputs. Telemetry only: nothing here says whether states help predict "
        "GeneEffect, and nothing is chosen in code.",
        "",
        "## Composition, confound and stability (training lines, first fitting seed)",
        "",
        "Deposit η²: the mean over states of the share of each state's mass across "
        "lines explained by deposit (null q95), then the largest state's; per state "
        "in the identity tables. Seed match: the matched states' mean correlation "
        "(the mean unmatched mass per line, when the fits' state counts differ).",
        "",
        "| Vocabulary | States | States per line ≥ mass | Deposit η² mean (null q95); "
        "max state | QC R² max | Second deposit TVD (between lines) | "
        "Seed match (unmatched mass) |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for vocabulary, entry in report["vocabularies"].items():
        deposit = entry["confound"]["deposit"]
        per_state = entry["confound"]["deposit_per_state"]
        largest = max((v["observed"] for v in per_state.values()), default=None)
        r2 = [v for v in entry["confound"]["qc_r2"].values() if np.isfinite(v)]
        second = entry["second_deposit"]
        stability = list(entry["stability"].values())
        matched = [np.mean(p["matched_correlation"]) for p in stability]
        unmatched = [p["unmatched_mass_mean"] for p in stability]
        agreement = (
            f"{_f(second['tvd_median'])} ({_f(second['tvd_between_lines_median'])}), "
            f"{second['lines']} lines"
            if second
            else "–"
        )
        cells = [
            vocabulary,
            str(len(entry["states"])),
            _f(entry["composition"]["states_per_line_median"], 1),
            f"{_f(deposit['observed'])} ({_f(deposit['null_q95'])}); {_f(largest)}",
            _f(max(r2) if r2 else None),
            agreement,
            f"{_f(float(np.mean(matched)) if matched else None)} "
            f"({_f(float(np.mean(unmatched)) if unmatched else None)})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Identity (best reference program per state)", ""]
    for vocabulary, entry in report["vocabularies"].items():
        lines += [
            f"### {vocabulary}",
            "",
            "| State | Mass (mean) | Deposit η² (null q95) | Top programs |",
            "| --- | ---: | --- | --- |",
        ]
        composition = entry["composition"]
        per_state = entry["confound"]["deposit_per_state"]
        for state, identity in entry["identity"]["states"].items():
            top = "; ".join(f"{name} {value:+.3f}" for name, value in identity["top"])
            share = per_state[state]
            cells = [
                state,
                _f(composition["mass_mean"][state]),
                f"{_f(share['observed'])} ({_f(share['null_q95'])})",
                top,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## The frozen PIE's gap",
        "",
        "Pooled over the genes and the role's lines: the gap's energy, and its "
        "spread across lines within a gene, over the spread across lines of the "
        "whole-line delta within the gene; the noise floor is the set's `|whole` "
        "context against the main pass on the same scale.",
        "",
        "| Set | Delta | Gap / spread (train) | Gap spread (train) | Noise (train) | "
        "Gap / spread (val) | Noise (val) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for set_name, entry in report["pie_gap"].items():
        if entry is None:
            lines.append(f"| {set_name} | not computed | | | | | |")
            continue
        for name in DELTAS:
            train = entry[name].get("train_cells", {})
            val = entry[name].get("val", {})
            cells = [
                set_name,
                name,
                _f(train.get("gap_ratio"), 4),
                _f(train.get("gap_spread_ratio"), 4),
                _f(train.get("noise_ratio"), 4),
                _f(val.get("gap_ratio"), 4),
                _f(val.get("noise_ratio"), 4),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    determinism = report["determinism"]["lines"]
    lines += [
        "",
        "## Projection of lines from deposits no training line comes from",
        "",
        f"{sum(determinism.values())} of {len(determinism)} such lines project "
        "identically when projected again.",
        "",
    ]
    return "\n".join(lines)


# --- entry point ------------------------------------------------------------------


RUNNERS = {
    "fit": step_fit,
    "project": step_project,
    "inputs": step_inputs,
    "report": step_report,
}


def run(config: Mapping[str, Any], name: str, steps: Sequence[str]) -> None:
    t = Telemetry(config, name)
    t.layout.root.mkdir(parents=True, exist_ok=True)
    open_manifest(t)
    for step in steps:
        _LOGGER.info("%s: start", step)
        record_step(t, step, RUNNERS[step](t))
        _LOGGER.info("%s: done", step)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    from src.experiments.config import load_states_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(load_states_config(_absolute(args.config)), args.config.stem, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

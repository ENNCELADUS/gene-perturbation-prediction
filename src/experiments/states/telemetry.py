"""A cell-state telemetry run: where it reads and writes (``Layout``), what it loads
once (``Telemetry``: the configs, ladder, split, lines and references), and the
settings its manifest records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data import cell_states as cs
from src.experiments.run_dir import absolute

MANIFEST: Final[str] = "manifest.json"
#: Largest difference allowed between the mass-weighted state inputs and the
#: line's PIE input (float32 storage of the latter).
IDENTITY_TOLERANCE: Final[float] = 1e-4
#: Groups of the reference-program table, in report order.
REFERENCE_GROUPS: Final[tuple[str, ...]] = ("kinker", "gavish", "hallmarks", "tirosh")
QC_NAMES: Final[tuple[str, ...]] = ("log_umi", "genes_detected", "mito_fraction")
DELTAS: Final[tuple[str, ...]] = ("dH", "dE", "own_h", "own_dp")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def npz_keys(path: Path, *keys: str) -> dict[str, np.ndarray]:
    """Only these arrays of an npz (each member is read on its own)."""
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in keys}


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
        from src.experiments.stage_one.setup import load_setup

        return load_setup(absolute(self.config["ladder_config"]))

    @property
    def ladder_root(self) -> Path:
        return absolute(self.setup.config["ladder_root"])

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
        from src.experiments.stage_one.ladder_inputs import read_lines

        return read_lines(self.ladder_root)

    @cached_property
    def split(self) -> Any:
        from src.data.splits import load_geneeffect_split

        return load_geneeffect_split(absolute(self.setup.joint["paths"]["split"]))

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
        from src.data.builders.plan_second_sources import PRIMARY_KEYS

        path = absolute(self.setup.joint["paths"]["split"]).with_suffix(".csv")
        table = pd.read_csv(path)
        sources = table.set_index("model_id")["expression_source"].astype(str)
        return sources.replace(PRIMARY_KEYS)

    @cached_property
    def chemistry_table(self) -> pd.DataFrame:
        path = absolute(self.config["references"]["chemistry"])
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
        root = absolute(self.setup.noise["deposit"]["second_source_root"])
        return {
            m: root / f"{m}.h5ad"
            for m in self.projected
            if (root / f"{m}.h5ad").is_file()
        }

    def source(self, model_id: str) -> Path:
        return absolute(self.setup.joint["paths"]["basal_raw_umi"]) / f"{model_id}.h5ad"

    @cached_property
    def preparation(self) -> Mapping[str, Any]:
        return self.setup.joint["preparation"]

    @cached_property
    def recorded(self) -> Mapping[str, Any]:
        from src.experiments.stage_one.ladder_inputs import settings

        return settings(self.setup)

    def references(self) -> dict[str, list[str]]:
        """Reference programs with a member on the panel (symbols matched
        upper-case; a program none of whose symbols is a panel gene, such as an
        outdated alias, cannot be scored), named ``<group>:<program>``."""
        refs = self.config["references"]
        panel = set(self.panel)
        out: dict[str, list[str]] = {}
        for group in REFERENCE_GROUPS:
            table = pd.read_csv(absolute(refs[group]), sep="\t")
            key = table.columns[0]
            for name, members in table.groupby(key, sort=False):
                genes = list(members["gene"].astype(str))
                if {g.upper() for g in genes} & panel:
                    out[f"{group}:{name}"] = genes
        return out

    def tirosh(self) -> dict[str, list[str]]:
        table = pd.read_csv(absolute(self.config["references"]["tirosh"]), sep="\t")
        return {p: list(g["gene"].astype(str)) for p, g in table.groupby("phase")}


def settings(t: Telemetry) -> dict[str, Any]:
    """What changes the states root's contents (the projected roles may grow)."""
    return {
        "seed": t.config["seed"],
        "ladder_config": t.config["ladder_config"],
        **{key: t.config[key] for key in ("fit", "programs", "scvi", "inputs")},
        "references": dict(t.config["references"]),
    }

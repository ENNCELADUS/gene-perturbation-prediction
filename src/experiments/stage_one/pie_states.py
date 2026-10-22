"""The frozen PIE on cell-state pseudo-contexts (design §3.5, Task 8): each line's
states read as contexts, their passes, and the assembled per-set arrays."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data.io import write_json, write_npz
from src.experiments.run_dir import absolute, record_step
from src.experiments.stage_one.pie import (
    NO_PERTURBATION,
    Layout,
    Pie,
    Tables,
    autocast,
    build_batch,
    exclusive,
    gene_evidence,
    is_mixed,
    layout,
    load_pie,
    pie_environment,
    pilot_genes,
    read_tables,
    shard_genes,
    write_dataset,
)
from src.experiments.stage_one.pie_decode import (
    checked_agreement,
    float_tokens,
    forward_tensors,
)

log = logging.getLogger(__name__)


#: Arrays a state pass stores per (gene, context); the wide ones in float16.
STATE_ARRAYS: Final[tuple[str, ...]] = ("dH", "dE", "own_h", "own_dp", "own_x")
_WIDE_STATE_ARRAYS: Final[frozenset[str]] = frozenset({"dH", "dE", "own_h"})


@dataclass(frozen=True)
class StateTables:
    """One set's contexts (``states/<set>/{inputs,masses}.parquet``)."""

    name: str
    contexts: list[str]
    owners: list[str]
    masses: pd.DataFrame
    inputs: np.ndarray  # contexts x panel, float32
    #: Each line's ``cell_states.state_fingerprint`` of its contexts, inputs
    #: and masses: a pass or assembly made from other inputs is refused.
    fingerprints: dict[str, str]

    def lines(self) -> list[str]:
        return list(dict.fromkeys(self.owners))


def read_state_tables(states_root: Path, name: str, tables: Tables) -> StateTables:
    """A set's state inputs; refuses a table over another panel, with other
    contexts than its masses, or owned by a line PIE's dataset lacks."""
    from src.data.cell_states import context_owner, set_fingerprints

    inputs = pd.read_parquet(states_root / name / "inputs.parquet")
    masses = pd.read_parquet(states_root / name / "masses.parquet")
    if list(inputs.columns) != tables.panel:
        raise ValueError(f"{name}: state inputs are not on the decode panel")
    if list(masses.index) != list(inputs.index):
        raise ValueError(f"{name}: state inputs and masses list other contexts")
    owners = [context_owner(c) for c in inputs.index]
    if owners != list(masses["model_id"]):
        raise ValueError(f"{name}: context names and owners disagree")
    unknown = sorted(set(owners) - set(tables.lines.index))
    if unknown:
        raise ValueError(
            f"{name}: contexts of lines outside PIE's dataset: {unknown[:5]}"
        )
    return StateTables(
        name=name,
        contexts=list(inputs.index),
        owners=owners,
        masses=masses,
        inputs=inputs.to_numpy(dtype=np.float32),
        fingerprints=set_fingerprints(inputs, masses),
    )


def owner_memory(
    pie: Pie,
    owners: Sequence[str],
    perturbation: str | None,
    evidence: Mapping[str, np.ndarray],
    *,
    precise: bool = False,
) -> Any:
    """The encoded latents of a batch whose contexts belong to ``owners``,
    encoding each distinct owner once: the latents read the owner's text and the
    perturbation's sources, never expression."""
    import torch

    distinct = list(dict.fromkeys(owners))
    inverse = torch.tensor([distinct.index(o) for o in owners], device=pie.device)
    width = len(pie.dataset.gene_union_ids[0])
    batch = build_batch(
        pie,
        distinct,
        perturbation,
        np.zeros((len(distinct), width), dtype=np.float32),
        evidence,
    )
    if not is_mixed(pie, precise):
        batch = float_tokens(batch)
    with torch.inference_mode(), autocast(pie, precise):
        return pie.model._encode(batch)[inverse]


def evict_contexts(pie: Pie) -> None:
    """Drop every cached context-token and control encoding (a state batch's)."""
    for key in [k for k in pie.cache if k[0] in ("context", "control")]:
        del pie.cache[key]


def state_batch(
    pie: Pie,
    names: Sequence[str],
    owners: Sequence[str],
    ctrl: np.ndarray,
    genes: Sequence[tuple[str, int]],
    evidence: Mapping[str, Mapping[str, np.ndarray]],
    basis: Mapping[str, Any] | None,
    size: int,
    *,
    precise: bool = False,
) -> dict[str, np.ndarray]:
    """One batch of contexts x every gene (``(key, own_index)``): its reference
    pass, then per gene the knockdown minus reference of the pooled decoder
    features (``dH``), the predicted change on the basis (``dE``, when given),
    the own feature, own predicted change and own input. Arrays are
    (contexts, genes, width); the batch is padded to ``size`` and the padding
    dropped; its cache entries are evicted."""
    import torch

    n = len(names)
    if not 0 < n <= size:
        raise ValueError(f"a state batch holds 1 to {size} contexts, got {n}")
    pad = size - n
    names = [*names, *[names[-1]] * pad]
    owners = [*owners, *[owners[-1]] * pad]
    ctrl = np.vstack([ctrl, np.repeat(ctrl[-1:], pad, axis=0)]).astype(np.float32)
    blank = evidence[NO_PERTURBATION]
    reference = forward_tensors(
        pie,
        build_batch(pie, names, None, ctrl, blank, owners=owners),
        None,
        heads=False,
        precise=precise,
        memory=owner_memory(pie, owners, None, blank, precise=precise),
    )["features"]
    parts: dict[str, list[Any]] = {name: [] for name in STATE_ARRAYS}
    for key, own_index in genes:
        result = forward_tensors(
            pie,
            build_batch(pie, names, key, ctrl, evidence[key], owners=owners),
            own_index,
            heads=True,
            precise=precise,
            memory=owner_memory(pie, owners, key, evidence[key], precise=precise),
        )
        change = result["change"].float()
        parts["dH"].append(result["features"] - reference)
        parts["own_h"].append(result["own"])
        parts["own_dp"].append(change[:, own_index : own_index + 1])
        parts["own_x"].append(
            torch.from_numpy(ctrl[:, own_index : own_index + 1]).to(change.device)
        )
        if basis is not None:
            parts["dE"].append((change - basis["mean"]) @ basis["components"].T)
    evict_contexts(pie)
    return {
        name: torch.stack(values, dim=1)[:n].float().cpu().numpy()
        for name, values in parts.items()
        if values
    }


@dataclass(frozen=True)
class StateLayout:
    """``states/pie/`` under the state inputs' root, so moving ``states/`` aside
    moves these too: per set, the per-line pass outputs, then the assembled
    arrays."""

    root: Path

    def lines(self, name: str) -> Path:
        return self.root / name / "lines"

    def line(self, name: str, model_id: str) -> Path:
        return self.lines(name) / f"{model_id}.npz"

    def index(self, name: str) -> Path:
        return self.root / name / "index.json"


def state_layout(out: Layout) -> StateLayout:
    return StateLayout(out.root / "states" / "pie")


def stored_fingerprint(path: Path) -> str:
    """The state-input fingerprint a per-line pass file was computed from."""
    with np.load(path) as payload:
        return str(payload["fingerprint"]) if "fingerprint" in payload.files else ""


def assembled(layout: StateLayout, states: StateTables) -> bool:
    """Whether the set is assembled from these state inputs; an assembly from
    other inputs is refused."""
    path = layout.index(states.name)
    if not path.is_file():
        return False
    if json.loads(path.read_text()).get("fingerprints") != states.fingerprints:
        raise ValueError(
            f"{path.parent} was assembled from other state inputs; move it aside"
        )
    return True


def states_pass(
    pie: Pie,
    tables: Tables,
    states: StateTables,
    genes: Sequence[str],
    basis: Mapping[str, Any] | None,
    lines: Sequence[str],
    layout: StateLayout,
    size: int,
    evidence: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """The state pass over this shard's lines lacking an output: context-major
    batches of ``size``; a line's ``<set>/lines/<ModelID>.npz`` (arrays
    genes x the line's contexts x width, and the line's state-input
    fingerprint) is written once all its contexts ran. A line whose file holds
    another fingerprint is refused; an assembled set is skipped."""
    if assembled(layout, states):
        return {"contexts": 0, "assembled": True}
    todo = set()
    for model_id in lines:
        if model_id not in states.fingerprints:
            continue
        path = layout.line(states.name, model_id)
        if not path.is_file():
            todo.add(model_id)
        elif stored_fingerprint(path) != states.fingerprints[model_id]:
            raise ValueError(
                f"{path} was computed from other state inputs; move it aside"
            )
    order = [i for i, owner in enumerate(states.owners) if owner in todo]
    if not order:
        return {"contexts": 0}
    start = time.monotonic()
    keys = [(tables.genes.at[g, "gene_text_key"], tables.panel.index(g)) for g in genes]
    for key in [NO_PERTURBATION, *(k for k, _ in keys)]:
        if key not in evidence:
            evidence[key] = gene_evidence(pie, key)
    remaining = {m: states.owners.count(m) for m in todo}
    done: dict[int, dict[str, np.ndarray]] = {}
    for first in range(0, len(order), size):
        chunk = order[first : first + size]
        arrays = state_batch(
            pie,
            [states.contexts[i] for i in chunk],
            [states.owners[i] for i in chunk],
            states.inputs[chunk],
            keys,
            evidence,
            basis,
            size,
        )
        for row, i in enumerate(chunk):
            done[i] = {name: value[row] for name, value in arrays.items()}
            owner = states.owners[i]
            remaining[owner] -= 1
            if remaining[owner]:
                continue
            rows = [j for j, o in enumerate(states.owners) if o == owner]
            finished = [done.pop(j) for j in rows]
            payload = {
                name: np.stack([entry[name] for entry in finished], axis=1).astype(
                    np.float16 if name in _WIDE_STATE_ARRAYS else np.float32
                )
                for name in arrays
            }
            write_npz(
                layout.line(states.name, owner),
                contexts=np.asarray([states.contexts[j] for j in rows]),
                genes=np.asarray(list(genes)),
                fingerprint=np.asarray(states.fingerprints[owner]),
                **payload,
            )
    seconds = time.monotonic() - start
    return {"contexts": len(order), "seconds_per_context": seconds / len(order)}


def record_states(out: Layout, name: str, payload: Mapping[str, Any]) -> None:
    """Add a set's shard record to ``states/pie/run.json``."""
    path = state_layout(out).root / "run.json"
    with exclusive(out):
        run = json.loads(path.read_text()) if path.is_file() else {"sets": {}}
        run["sets"].setdefault(name, {})[payload["shard"]] = {
            **payload,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_json(path, run)


def state_genes(
    tables: Tables, states_config: Mapping[str, Any], seed: int
) -> list[str]:
    """The genes a state pass decodes: the Task 2 pilot's draw."""
    return pilot_genes(tables, int(states_config["pie"]["genes"]), seed)


def run_states(
    out: Layout,
    tables: Tables,
    states_config: Mapping[str, Any],
    sets: Sequence[str],
    shard: tuple[int, int],
    device: str,
) -> None:
    import torch

    genes = state_genes(tables, states_config, int(out.config["seed"]))
    layout = state_layout(out)
    lines = shard_genes(list(tables.lines.index), shard)
    stored = dict(np.load(out.pie / "basis.npz"))
    size = int(states_config["pie"]["contexts_per_batch"])
    with pie_environment(out):
        pie = load_pie(out, tables, device)
        basis = {
            key: torch.from_numpy(np.asarray(stored[key], dtype=np.float32)).to(
                pie.device
            )
            for key in ("mean", "components")
        }
        difference = checked_agreement(pie, tables, genes)
        evidence: dict[str, dict[str, np.ndarray]] = {}
        for name in sets:
            states = read_state_tables(out.root / "states", name, tables)
            result = states_pass(
                pie, tables, states, genes, basis, lines, layout, size, evidence
            )
            log.info("states %s: %s", name, result)
            record_states(
                out,
                name,
                {
                    **result,
                    "shard": f"{shard[0]}/{shard[1]}",
                    "genes": len(genes),
                    "agreement": difference,
                },
            )


def assemble_states(
    out: Layout, tables: Tables, states_config: Mapping[str, Any], name: str
) -> dict[str, Any]:
    """``states/pie/<set>/<array>.npy`` (genes x contexts x width, in the set's
    context order), each centred on the main pass's frame means of its gene
    (``pie/frame_means.npz``) as the assembled main arrays are; then the
    per-line files, which they replace, are removed. A set assembled from these
    state inputs is skipped; a line's pass or an assembly from other inputs is
    refused (``index.json`` records each line's fingerprint)."""
    import shutil

    layout = state_layout(out)
    folder = layout.root / name
    states = read_state_tables(out.root / "states", name, tables)
    genes = state_genes(tables, states_config, int(out.config["seed"]))
    if assembled(layout, states):
        return {"contexts": len(states.contexts), "genes": len(genes), "skipped": True}
    missing = [m for m in states.lines() if not layout.line(name, m).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} lines without a pass: {missing[:5]}"
        )
    means = dict(np.load(out.pie / "frame_means.npz"))
    position = [tables.panel.index(g) for g in genes]
    arrays: dict[str, np.ndarray] = {}
    column = 0
    for model_id in states.lines():
        with np.load(layout.line(name, model_id)) as payload:
            part = {k: payload[k] for k in payload.files}
        expected = [c for c, o in zip(states.contexts, states.owners) if o == model_id]
        if part["contexts"].tolist() != expected or part["genes"].tolist() != genes:
            raise ValueError(
                f"{name} {model_id}: the pass holds other contexts or genes"
            )
        if str(part.get("fingerprint", "")) != states.fingerprints[model_id]:
            raise ValueError(
                f"{name} {model_id}: the pass was computed from other state inputs"
            )
        width = len(expected)
        for array in STATE_ARRAYS:
            if array not in part:
                continue
            value = part[array].astype(np.float32) - means[array][position][:, None, :]
            if array not in arrays:
                arrays[array] = np.lib.format.open_memmap(
                    folder / f".{array}.npy.tmp",
                    mode="w+",
                    dtype=np.float16 if array in _WIDE_STATE_ARRAYS else np.float32,
                    shape=(len(genes), len(states.contexts), value.shape[2]),
                )
            arrays[array][:, column : column + width] = value
        column += width
    for array, values in arrays.items():
        values.flush()
        os.replace(folder / f".{array}.npy.tmp", folder / f"{array}.npy")
    states.masses.to_parquet(folder / "contexts.parquet")
    write_json(
        folder / "index.json",
        {
            "genes": genes,
            "contexts": len(states.contexts),
            "arrays": {k: list(v.shape) for k, v in arrays.items()},
            "centring": (
                "each gene's main-pass frame mean subtracted (pie/frame_means.npz)"
            ),
            "contexts_table": "contexts.parquet, the set's masses.parquet",
            "fingerprints": states.fingerprints,
        },
    )
    shutil.rmtree(layout.lines(name))
    return {"contexts": len(states.contexts), "genes": len(genes)}


def run_state_steps(
    states_config: Mapping[str, Any],
    steps: Sequence[str],
    sets: Sequence[str],
    shard: tuple[int, int],
    device: str,
) -> None:
    from src.experiments.stage_one.config import load_ladder_config

    config = load_ladder_config(absolute(states_config["ladder_config"]))
    out = layout(config)
    tables = read_tables(out.root)
    with exclusive(out):
        write_dataset(out, tables)
    for step in steps:
        if step == "states":
            run_states(out, tables, states_config, sets, shard, device)
        elif step == "assemble_states":
            for name in sets:
                record_step(
                    out.pie / "run.json",
                    f"assemble_states_{name}",
                    assemble_states(out, tables, states_config, name),
                )

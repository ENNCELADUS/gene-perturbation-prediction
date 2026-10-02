"""Six-arm response-model comparison in STATE's log expression space.

Leave-one-anchor-out over the four response anchors: each fold trains on every
condition of three anchors and scores every condition of the fourth. The arms
answer two questions: what the STATE transformer adds over a plain MLP on the
same log HVG cells, and what Tx1 adds as the MLP's cell representation.

- no-change: the anchor's basal control bag is the prediction;
- global mean effect: the source anchors' mean shift added to every basal cell,
  gene-blind;
- released STATE checkpoint, untrained, driven by its own one-hot perturbation
  vocabulary; conditions whose gene is outside the vocabulary are skipped and
  counted;
- STATE as in the joint model: released STATE plus a new ESM2 adapter, trained;
- MLP on log HVG cells and MLP on Tx1 cell embeddings, both zero-initialised so
  they start exactly at no-change, trained.

Every arm x fold result is written to ``folds/<arm>__<anchor>.json`` and is not
recomputed on rerun; ``verdicts.json`` is written last and marks the run done.

The work splits into independent jobs: the untrained arms (with ``sanity.json``),
one job per trained arm and held-out anchor, and the summary. A trained job
starts from a fixed seed and reads the no-change losses from the untrained fold
files, so jobs may run in any order, in one process or on separate GPUs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.embeddings import Esm2EmbeddingTable
from src.data.prepared import load_inputs, read_manifest
from src.experiments.config import load_config
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.response import predict_bags
from src.model.response_mlp import ResponseMLP
from src.model.state import StateResponse, load_released_state

HCT116 = "ACH-000971"
UNTRAINED_ARMS = ("no_change", "global_mean_effect", "released_state")
TRAINED_ARMS = ("state_joint", "mlp_hvg", "mlp_tx1")
ARMS = (*UNTRAINED_ARMS, *TRAINED_ARMS)
VERDICTS = (
    ("state_joint_vs_mlp_hvg", "state_joint", "mlp_hvg"),
    ("mlp_tx1_vs_mlp_hvg", "mlp_tx1", "mlp_hvg"),
)
SEED = 0
# Observed cells stay on a CUDA device when they take at most this share of its
# free memory; the rest is left for the model, activations and optimizer state.
RESIDENT_SHARE = 0.6
# Conditions read from the response cache per host-to-device copy while loading.
LOAD_CONDITIONS = 512

Predict = Callable[[Sequence[str], Sequence[str]], Sequence[torch.Tensor]]


class OneHotPerturbations(nn.Module):
    """STATE's released one-hot perturbation vocabulary, looked up by gene symbol."""

    def __init__(self, vectors: Mapping[Any, torch.Tensor]) -> None:
        super().__init__()
        symbols = [str(gene).upper() for gene in vectors]
        self._index = {gene: row for row, gene in enumerate(symbols)}
        self.register_buffer(
            "matrix",
            torch.stack(
                [torch.as_tensor(v, dtype=torch.float32) for v in vectors.values()]
            ),
            persistent=False,
        )

    def covers(self, gene: str) -> bool:
        return str(gene).upper() in self._index

    def forward_many(self, genes: Sequence[str]) -> torch.Tensor:
        rows = [self._index[str(gene).upper()] for gene in genes]
        return self.matrix[torch.as_tensor(rows, device=self.matrix.device)]


class _ObservedCells:
    """Every response condition's observed cells, read from the cache once.

    The cells stay on the comparison device when they take at most
    ``RESIDENT_SHARE`` of its free memory; otherwise they stay in host memory
    and each batch's cells move to the device in one pinned transfer. Each
    condition's observed mean and within-observed energy-distance term
    ``E|y-y'|`` do not depend on any prediction and are computed once.
    """

    def __init__(self, cache: Any, device: torch.device, batch_size: int) -> None:
        count = len(cache.keys)
        self.length = np.array(
            [np.asarray(cache.target_bag(i)).shape[0] for i in range(count)],
            dtype=np.int64,
        )
        self.start = np.concatenate([[0], np.cumsum(self.length)]).astype(np.int64)
        width = int(np.asarray(cache.target_bag(0)).shape[1])
        total = int(self.start[-1])
        resident = device.type != "cuda" or (
            total * width * 4 <= RESIDENT_SHARE * torch.cuda.mem_get_info(device)[0]
        )
        self.device = device
        self.resident = resident
        self._cells = torch.empty(
            (total, width),
            dtype=torch.float32,
            device=device if resident else torch.device("cpu"),
            pin_memory=not resident,
        )
        for first in range(0, count, LOAD_CONDITIONS):
            last = min(first + LOAD_CONDITIONS, count)
            host = np.concatenate(
                [
                    np.asarray(cache.target_bag(i), dtype=np.float32)
                    for i in range(first, last)
                ]
            )
            self._cells[self.start[first] : self.start[last]].copy_(
                torch.from_numpy(host)
            )
        self.mean = torch.empty((count, width), device=device)
        self.within = torch.empty(count, device=device)
        with torch.no_grad():
            for first in range(0, count, batch_size):
                part = np.arange(first, min(first + batch_size, count))
                for group in _groups(self.length[part]):
                    rows = part[group]
                    cells = self.bags(rows)
                    index = torch.from_numpy(rows).to(device)
                    self.mean[index] = cells.mean(dim=1)
                    self.within[index] = torch.cdist(cells, cells).mean(dim=(1, 2))

    def bags(self, indices: np.ndarray) -> torch.Tensor:
        """``[conditions, cells, genes]`` cells of conditions with equal cell counts."""
        cells = int(self.length[indices[0]])
        rows = (self.start[indices][:, None] + np.arange(cells)).reshape(-1)
        picked = self._cells[torch.from_numpy(rows).to(self._cells.device)]
        if not self.resident:
            picked = picked.pin_memory().to(self.device, non_blocking=True)
        return picked.view(len(indices), cells, -1)


def _groups(keys: Sequence[Any]) -> list[np.ndarray]:
    """Positions with equal keys, groups in order of first appearance."""
    groups: dict[Any, list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    return [np.asarray(group, dtype=np.int64) for group in groups.values()]


@dataclass
class _World:
    """Response conditions and per-anchor cells on the comparison device."""

    model_ids: tuple[str, ...]
    genes: tuple[str, ...]
    anchors: tuple[str, ...]
    by_anchor: dict[str, np.ndarray]
    basal: dict[str, torch.Tensor]
    tx1: dict[str, torch.Tensor]
    control_mean: torch.Tensor
    anchor_of: np.ndarray
    esm2: torch.Tensor
    esm2_row: dict[str, int]
    observed: _ObservedCells
    device: torch.device

    def gene_vectors(self, genes: Sequence[str]) -> torch.Tensor:
        rows = [self.esm2_row[str(gene).upper()] for gene in genes]
        return self.esm2[torch.as_tensor(rows, device=self.device)]


def _world(inputs: Any, device: torch.device, batch_size: int) -> _World:
    keys = tuple(inputs.response_targets.keys)
    anchors = tuple(inputs.response_anchors)
    model_ids = tuple(str(m) for m, _ in keys)
    position = {a: k for k, a in enumerate(anchors)}
    strays = sorted(set(model_ids) - set(anchors))
    if strays:
        raise ValueError(f"response conditions of lines that are not anchors: {strays}")
    symbols = [str(s).upper() for s in inputs.esm2_symbols]

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)

    basal = {a: tensor(inputs.lines[a].basal_hvg) for a in anchors}
    return _World(
        model_ids=model_ids,
        genes=tuple(str(g) for _, g in keys),
        anchors=anchors,
        by_anchor={
            a: np.flatnonzero(np.asarray(model_ids) == a).astype(np.int64)
            for a in anchors
        },
        basal=basal,
        tx1={a: tensor(inputs.lines[a].controls_tx1) for a in anchors},
        control_mean=torch.stack([basal[a].mean(dim=0) for a in anchors]),
        anchor_of=np.array([position[m] for m in model_ids], dtype=np.int64),
        esm2=tensor(inputs.esm2_vectors),
        esm2_row={s: row for row, s in enumerate(symbols)},
        observed=_ObservedCells(inputs.response_targets, device, batch_size),
        device=device,
    )


@dataclass
class _Arm:
    """A response predictor: ``predict(model_ids, genes) -> predicted bags``."""

    predict: Predict
    module: nn.Module | None = None
    param_groups: list[dict[str, Any]] = field(default_factory=list)
    covers: Callable[[str], bool] = lambda gene: True
    # The prediction ignores the gene, so a gene shuffle cannot change the loss.
    gene_blind: bool = False


def _no_change(world: _World) -> _Arm:
    return _Arm(
        lambda model_ids, genes: [world.basal[m] for m in model_ids], gene_blind=True
    )


def _global_mean_effect(world: _World, source: np.ndarray) -> _Arm:
    """Mean over source conditions of mean(target) - mean(that anchor's controls)."""
    index = torch.from_numpy(source).to(world.device)
    controls = world.control_mean[
        torch.from_numpy(world.anchor_of[source]).to(world.device)
    ]
    shift = (world.observed.mean[index] - controls).sum(dim=0) / len(source)
    return _Arm(
        lambda model_ids, genes: [world.basal[m] + shift for m in model_ids],
        gene_blind=True,
    )


def _state_arm(world: _World, response: StateResponse) -> Predict:
    def predict(
        model_ids: Sequence[str], genes: Sequence[str]
    ) -> Sequence[torch.Tensor]:
        bags = [world.basal[m] for m in model_ids]
        return predict_bags(response, bags, genes, seed=SEED)

    return predict


def _released_state(
    world: _World, state: nn.Module, onehot: OneHotPerturbations
) -> _Arm:
    response = StateResponse(state, onehot).to(world.device)
    return _Arm(_state_arm(world, response), response, covers=onehot.covers)


def _state_joint(
    world: _World, state: nn.Module, inputs: Any, config: Mapping[str, Any]
) -> _Arm:
    vectors = np.asarray(inputs.esm2_vectors, dtype=np.float32)
    table = Esm2EmbeddingTable(
        vectors.shape[1], dict(zip(inputs.esm2_symbols, vectors, strict=True))
    )
    adapter = Esm2PerturbationAdapter(
        list(inputs.esm2_symbols),
        table,
        int(config["model"]["esm2_adapter_hidden"]),
        int(state.pert_dim),
    )
    response = StateResponse(state, adapter).to(world.device)
    comparison = config["comparison"]
    groups = [
        {"params": list(state.parameters()), "lr": comparison["state_learning_rate"]},
        {"params": list(adapter.parameters()), "lr": comparison["learning_rate"]},
    ]
    return _Arm(_state_arm(world, response), response, groups)


def _mlp(
    world: _World, cells: Mapping[str, torch.Tensor], hidden: int, lr: float
) -> _Arm:
    """``basal + f([cell ; adapter(gene)])``, ``cells`` the cell representation."""
    width = world.basal[world.anchors[0]].shape[1]
    model = ResponseMLP(
        cell_dim=int(cells[world.anchors[0]].shape[1]),
        gene_dim=int(world.esm2.shape[1]),
        hidden=int(hidden),
        out_dim=int(width),
    ).to(world.device)

    def predict(
        model_ids: Sequence[str], genes: Sequence[str]
    ) -> Sequence[torch.Tensor]:
        # Every condition of the batch in one call: each cell row carries its
        # condition's ESM2 vector.
        counts = [int(cells[m].shape[0]) for m in model_ids]
        gene = world.gene_vectors(genes).repeat_interleave(
            torch.as_tensor(counts, device=world.device), dim=0, output_size=sum(counts)
        )
        predicted = model(
            torch.cat([cells[m] for m in model_ids]),
            torch.cat([world.basal[m] for m in model_ids]),
            gene,
        )
        return predicted.split(counts)

    return _Arm(predict, model, [{"params": list(model.parameters()), "lr": lr}])


def _response_losses(
    world: _World, predicted: Sequence[torch.Tensor], indices: Sequence[int]
) -> torch.Tensor:
    """``response_loss`` of each predicted bag against its observed condition.

    Mean-shift MSE from the anchor's control mean plus energy distance; bags
    with equal predicted and observed cell counts are computed as one batch.
    """
    indices = np.asarray(indices, dtype=np.int64)
    observed = world.observed
    shapes = [
        (int(p.shape[0]), int(observed.length[i]))
        for p, i in zip(predicted, indices, strict=True)
    ]
    parts, order = [], []
    for group in _groups(shapes):
        rows = indices[group]
        cells = torch.stack([predicted[k] for k in group])
        targets = observed.bags(rows)
        index = torch.from_numpy(rows).to(world.device)
        control = world.control_mean[
            torch.from_numpy(world.anchor_of[rows]).to(world.device)
        ]
        shift = (cells.mean(dim=1) - control) - (observed.mean[index] - control)
        energy = (
            2.0 * torch.cdist(cells, targets).mean(dim=(1, 2))
            - torch.cdist(cells, cells).mean(dim=(1, 2))
            - observed.within[index]
        )
        parts.append(shift.pow(2).mean(dim=1) + energy)
        order.extend(group)
    losses = torch.cat(parts)
    return losses[torch.from_numpy(np.argsort(order)).to(losses.device)]


def _batch_losses(
    arm: _Arm, world: _World, indices: Sequence[int], genes: Sequence[str]
) -> torch.Tensor:
    predicted = arm.predict([world.model_ids[i] for i in indices], genes)
    return _response_losses(world, predicted, indices)


def _losses(
    arm: _Arm,
    world: _World,
    indices: np.ndarray,
    batch_size: int,
    genes: Sequence[str] | None = None,
) -> np.ndarray:
    """Per-condition response loss, NaN where the arm does not cover the gene."""
    genes = [world.genes[i] for i in indices] if genes is None else list(genes)
    out = np.full(len(indices), np.nan)
    covered = [k for k, gene in enumerate(genes) if arm.covers(gene)]
    if arm.module is not None:
        arm.module.eval()
    with torch.no_grad():
        for start in range(0, len(covered), batch_size):
            part = covered[start : start + batch_size]
            losses = _batch_losses(
                arm, world, [int(indices[k]) for k in part], [genes[k] for k in part]
            )
            out[part] = losses.double().cpu().numpy()
    return out


def _ratio(loss: np.ndarray, no_change: np.ndarray) -> float:
    """Mean loss over covered conditions / no-change mean over the same conditions."""
    covered = np.isfinite(loss)
    if not covered.any():
        return math.nan
    return float(loss[covered].mean() / no_change[covered].mean())


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray | None:
    if n < 2:
        return None
    while True:
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            return order


def _gene_blind_shuffled(loss: np.ndarray, shuffles: int) -> np.ndarray:
    """Gene-shuffled losses of a gene-blind arm: its own losses, where scored."""
    shuffled = np.full(len(loss), np.nan)
    scored = np.isfinite(loss)
    if shuffles and scored.sum() >= 2:
        shuffled[scored] = loss[scored]
    return shuffled


def _score(
    arm: _Arm,
    world: _World,
    held_out: np.ndarray,
    source: np.ndarray,
    *,
    position: int,
    shuffles: int,
    batch_size: int,
) -> dict[str, list[float | None]]:
    """Held-out, gene-shuffled held-out and source per-condition losses."""
    loss = _losses(arm, world, held_out, batch_size)
    if arm.gene_blind:
        shuffled = _gene_blind_shuffled(loss, shuffles)
    else:
        covered = held_out[np.isfinite(loss)]
        genes = np.asarray([world.genes[i] for i in covered], dtype=object)
        shuffled = np.full(len(held_out), np.nan)
        totals = np.zeros(len(covered))
        for shuffle in range(shuffles):
            # Fixed per anchor and shuffle, so every arm sees the same derangements.
            order = _derangement(
                len(covered), np.random.default_rng([SEED, position, shuffle])
            )
            if order is None:
                break
            totals += _losses(arm, world, covered, batch_size, genes=list(genes[order]))
        else:
            if shuffles:
                shuffled[np.isfinite(loss)] = totals / shuffles
    return _result(loss, shuffled, _losses(arm, world, source, batch_size))


def _result(
    loss: np.ndarray, shuffled: np.ndarray, source: np.ndarray
) -> dict[str, list[float | None]]:
    return {
        "loss": _listed(loss),
        "shuffled_loss": _listed(shuffled),
        "source_loss": _listed(source),
    }


def _listed(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def _array(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _balanced_batches(
    pools: Sequence[np.ndarray], batch_size: int
) -> tuple[int, Callable[[], list[int]]]:
    """Steps per epoch and a batch draw with equal conditions from every pool.

    Each pool cycles through fresh seeded permutations; one epoch passes once
    over the largest pool, so smaller pools are revisited.
    """
    per_pool = max(1, batch_size // len(pools))
    steps = math.ceil(max(len(pool) for pool in pools) / per_pool)
    rngs = [np.random.default_rng([SEED, position]) for position in range(len(pools))]
    orders = [rng.permutation(pool) for rng, pool in zip(rngs, pools, strict=True)]
    positions = [0] * len(pools)

    def draw() -> list[int]:
        batch: list[int] = []
        for p, pool in enumerate(pools):
            for _ in range(per_pool):
                if positions[p] == len(orders[p]):
                    orders[p] = rngs[p].permutation(pool)
                    positions[p] = 0
                batch.append(int(orders[p][positions[p]]))
                positions[p] += 1
        return batch

    return steps, draw


def _train(
    arm: _Arm,
    world: _World,
    pools: Sequence[np.ndarray],
    held_out: np.ndarray,
    no_change: np.ndarray,
    comparison: Mapping[str, Any],
) -> list[float]:
    """Fixed-epoch AdamW training; the held-out ratio before and after each epoch."""
    batch_size = int(comparison["batch_size"])
    optimizer = torch.optim.AdamW(arm.param_groups)
    parameters = [p for group in arm.param_groups for p in group["params"]]
    steps, draw = _balanced_batches(pools, batch_size)
    curve = [_ratio(_losses(arm, world, held_out, batch_size), no_change)]
    for _ in range(int(comparison["epochs"])):
        arm.module.train()
        for _ in range(steps):
            batch = draw()
            loss = _batch_losses(
                arm, world, batch, [world.genes[i] for i in batch]
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite response loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        curve.append(_ratio(_losses(arm, world, held_out, batch_size), no_change))
    return curve


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _fold_path(out_dir: Path, arm: str, anchor: str) -> Path:
    return out_dir / "folds" / f"{arm}__{anchor}.json"


def _fold_indices(world: _World, anchor: str) -> tuple[np.ndarray, np.ndarray]:
    """Held-out conditions of ``anchor`` and the source conditions, anchor order."""
    source = np.concatenate([world.by_anchor[b] for b in world.anchors if b != anchor])
    return world.by_anchor[anchor], source


def _write_fold(
    out_dir: Path,
    world: _World,
    arm: str,
    anchor: str,
    result: Mapping[str, Any],
    curve: list[float | None] | None = None,
) -> None:
    payload = {
        "arm": arm,
        "anchor": anchor,
        "genes": [world.genes[i] for i in world.by_anchor[anchor]],
        **result,
        "curve": curve,
    }
    _write_json(_fold_path(out_dir, arm, anchor), payload)
    print(f"response comparison: {arm} held out {anchor} done", flush=True)


def _anchors(config: Mapping[str, Any], inputs: Any) -> tuple[str, ...]:
    if inputs is not None:
        return tuple(inputs.response_anchors)
    return tuple(read_manifest(Path(config["prepared_root"]))["response_anchors"])


def _released_state_factory(config: Mapping[str, Any]) -> Callable[[], nn.Module]:
    def build() -> nn.Module:
        return load_released_state(
            Path(config["paths"]["state_checkpoint"]),
            cell_set_len=int(config["model"]["cell_sentence_len"]),
        )

    return build


def _untrained_done(out_dir: Path, anchors: Sequence[str]) -> bool:
    return (out_dir / "sanity.json").is_file() and all(
        _fold_path(out_dir, arm, a).is_file() for arm in UNTRAINED_ARMS for a in anchors
    )


def _run_untrained(
    world: _World,
    config: Mapping[str, Any],
    out_dir: Path,
    state_factory: Callable[[], nn.Module],
) -> None:
    comparison = config["comparison"]
    batch_size, shuffles = int(comparison["batch_size"]), int(comparison["shuffles"])
    anchors = world.anchors
    out_dir.mkdir(parents=True, exist_ok=True)

    def pending(arm: str) -> list[str]:
        return [a for a in anchors if not _fold_path(out_dir, arm, a).is_file()]

    def score(arm: _Arm, a: str) -> dict[str, list[float | None]]:
        held_out, source = _fold_indices(world, a)
        return _score(
            arm,
            world,
            held_out,
            source,
            position=anchors.index(a),
            shuffles=shuffles,
            batch_size=batch_size,
        )

    if pending("no_change"):
        # The basal bag is the prediction whatever the arm, fold or gene, so one
        # pass over every condition gives every fold's held-out and source losses.
        everything = np.concatenate([world.by_anchor[a] for a in anchors])
        loss = np.full(len(world.model_ids), np.nan)
        loss[everything] = _losses(_no_change(world), world, everything, batch_size)
        for a in pending("no_change"):
            held_out, source = _fold_indices(world, a)
            shuffled = _gene_blind_shuffled(loss[held_out], shuffles)
            _write_fold(
                out_dir,
                world,
                "no_change",
                a,
                _result(loss[held_out], shuffled, loss[source]),
            )

    # The released checkpoint is untrained, so its held-out score per anchor is
    # also the sanity line.
    if pending("released_state"):
        vocabulary = torch.load(
            Path(config["paths"]["state_model_dir"]) / "pert_onehot_map.pt",
            map_location="cpu",
            weights_only=False,
        )
        released = _released_state(
            world, state_factory(), OneHotPerturbations(vocabulary)
        )
        for a in pending("released_state"):
            _write_fold(out_dir, world, "released_state", a, score(released, a))
    if not (out_dir / "sanity.json").is_file():
        sanity = {}
        for a in anchors:
            fold = json.loads(_fold_path(out_dir, "released_state", a).read_text())
            base = json.loads(_fold_path(out_dir, "no_change", a).read_text())
            loss = _array(fold["loss"])
            sanity[a] = {
                "ratio_to_no_change": _number(_ratio(loss, _array(base["loss"]))),
                "covered_genes": int(np.isfinite(loss).sum()),
                "total_genes": len(loss),
            }
        _write_json(
            out_dir / "sanity.json",
            {
                "checkpoint": str(config["paths"]["state_checkpoint"]),
                "statistic": "released STATE checkpoint, untrained: mean loss over the "
                "anchor's covered conditions / no-change mean over the same conditions",
                "anchors": sanity,
            },
        )

    for a in pending("global_mean_effect"):
        _, source = _fold_indices(world, a)
        arm = _global_mean_effect(world, source)
        _write_fold(out_dir, world, "global_mean_effect", a, score(arm, a))


def _run_trained(
    world: _World,
    inputs: Any,
    config: Mapping[str, Any],
    out_dir: Path,
    arm_name: str,
    anchor: str,
    state_factory: Callable[[], nn.Module],
) -> None:
    comparison = config["comparison"]
    held_out, source = _fold_indices(world, anchor)
    base = _array(
        json.loads(_fold_path(out_dir, "no_change", anchor).read_text())["loss"]
    )
    pools = [world.by_anchor[b] for b in world.anchors if b != anchor]
    # Every job starts from the same fixed seed, so its result does not depend on
    # which jobs ran before it, in this process or another.
    torch.manual_seed(SEED)
    if arm_name == "state_joint":
        arm = _state_joint(world, state_factory(), inputs, config)
    else:
        cells = world.basal if arm_name == "mlp_hvg" else world.tx1
        arm = _mlp(world, cells, comparison["hidden"], comparison["learning_rate"])
    curve = _train(arm, world, pools, held_out, base, comparison)
    result = _score(
        arm,
        world,
        held_out,
        source,
        position=world.anchors.index(anchor),
        shuffles=int(comparison["shuffles"]),
        batch_size=int(comparison["batch_size"]),
    )
    _write_fold(out_dir, world, arm_name, anchor, result, [_number(r) for r in curve])


def run_untrained(
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    device: str,
    inputs: Any = None,
    state_factory: Callable[[], nn.Module] | None = None,
) -> None:
    """Write every untrained arm's fold files and ``sanity.json``.

    Existing fold files are kept. ``state_factory`` returns a fresh released
    STATE model; by default it loads ``paths.state_checkpoint``.
    """
    out_dir = Path(out_dir)
    if _untrained_done(out_dir, _anchors(config, inputs)):
        return
    inputs = load_inputs(config) if inputs is None else inputs
    world = _world(
        inputs, torch.device(device), int(config["comparison"]["batch_size"])
    )
    _run_untrained(
        world, config, out_dir, state_factory or _released_state_factory(config)
    )


def pending_trained_jobs(
    config: Mapping[str, Any], out_dir: Path, *, inputs: Any = None
) -> list[tuple[str, str]]:
    """``(arm, anchor)`` of every trained fold without a fold file, anchors outer."""
    out_dir = Path(out_dir)
    return [
        (arm, a)
        for a in _anchors(config, inputs)
        for arm in TRAINED_ARMS
        if not _fold_path(out_dir, arm, a).is_file()
    ]


def run_trained_job(
    config: Mapping[str, Any],
    out_dir: Path,
    arm: str,
    anchor: str,
    *,
    device: str,
    inputs: Any = None,
    state_factory: Callable[[], nn.Module] | None = None,
) -> None:
    """Train ``arm`` on the other anchors, score it on ``anchor``, write its fold file.

    Does nothing when the fold file exists. Needs the no-change fold file of
    ``anchor`` (written by :func:`run_untrained`) for the held-out curve.
    """
    out_dir = Path(out_dir)
    anchors = _anchors(config, inputs)
    if arm not in TRAINED_ARMS:
        raise ValueError(f"{arm!r} is not a trained arm; choose from {TRAINED_ARMS}")
    if anchor not in anchors:
        raise ValueError(f"{anchor!r} is not a response anchor; choose from {anchors}")
    if _fold_path(out_dir, arm, anchor).is_file():
        return
    base = _fold_path(out_dir, "no_change", anchor)
    if not base.is_file():
        raise FileNotFoundError(f"{base} is missing: run the untrained arms first")
    inputs = load_inputs(config) if inputs is None else inputs
    world = _world(
        inputs, torch.device(device), int(config["comparison"]["batch_size"])
    )
    _run_trained(
        world,
        inputs,
        config,
        out_dir,
        arm,
        anchor,
        state_factory or _released_state_factory(config),
    )


def _bootstrap(
    folds: Mapping[str, Mapping[str, Mapping[str, Any]]],
    anchors: Sequence[str],
    variants: Mapping[str, Sequence[str]],
    resamples: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Pooled ratio per resample, variant and arm.

    The bootstrap unit is the perturbed gene: each resample draws gene
    identities with replacement once, from every held-out gene of every fold,
    and weights each fold's conditions by how often their gene was drawn. A
    gene perturbed in several anchors therefore moves all its folds together,
    and every arm sees the same draw.
    """
    rng = np.random.default_rng(SEED)
    losses = {arm: {a: _array(folds[arm][a]["loss"]) for a in anchors} for arm in ARMS}
    genes = {a: folds["no_change"][a]["genes"] for a in anchors}
    universe = sorted({gene for a in anchors for gene in genes[a]})
    index = {gene: i for i, gene in enumerate(universe)}
    positions = {a: np.array([index[g] for g in genes[a]], dtype=int) for a in anchors}
    samples = {v: {arm: np.empty(resamples) for arm in ARMS} for v in variants}
    for b in range(resamples):
        counts = np.bincount(
            rng.integers(0, len(universe), len(universe)), minlength=len(universe)
        )
        weights = {a: counts[positions[a]].astype(float) for a in anchors}
        for arm in ARMS:
            ratio = {
                a: _weighted_ratio(losses[arm][a], losses["no_change"][a], weights[a])
                for a in anchors
            }
            for variant, members in variants.items():
                samples[variant][arm][b] = np.mean([ratio[a] for a in members])
    return samples


def _weighted_ratio(
    loss: np.ndarray, no_change: np.ndarray, weights: np.ndarray
) -> float:
    """``_ratio`` with each condition counted ``weights`` times."""
    covered = np.isfinite(loss) & (weights > 0)
    if not covered.any():
        return math.nan
    w = weights[covered]
    return float((w * loss[covered]).sum() / (w * no_change[covered]).sum())


def _interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [float(x) for x in np.percentile(finite, [2.5, 97.5])]


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarise(
    config: Mapping[str, Any], out_dir: Path, *, inputs: Any = None
) -> pd.DataFrame:
    """Write ``comparison.csv``, ``curves.csv`` and, last, ``verdicts.json``.

    Reads every arm x fold file; raises naming the fold files that are missing.
    """
    out_dir = Path(out_dir)
    anchors = _anchors(config, inputs)
    comparison = config["comparison"]
    missing = [
        str(_fold_path(out_dir, arm, a))
        for arm in ARMS
        for a in anchors
        if not _fold_path(out_dir, arm, a).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"response comparison fold files missing: {missing}")
    folds = {
        arm: {a: json.loads(_fold_path(out_dir, arm, a).read_text()) for a in anchors}
        for arm in ARMS
    }
    rows, curves = [], []
    for arm in ARMS:
        for a in anchors:
            fold = folds[arm][a]
            base = folds["no_change"][a]
            loss, shuffled = _array(fold["loss"]), _array(fold["shuffled_loss"])
            scored = np.isfinite(loss) & np.isfinite(shuffled)
            identity = math.nan
            if scored.any():
                own = loss[scored].mean()
                identity = (shuffled[scored].mean() - own) / own
            rows.append(
                {
                    "arm": arm,
                    "fold": a,
                    "held_out_ratio": _ratio(loss, _array(base["loss"])),
                    "identity_share": float(identity),
                    "source_ratio": _ratio(
                        _array(fold["source_loss"]), _array(base["source_loss"])
                    ),
                    "n_conditions": len(loss),
                    "n_covered": int(np.isfinite(loss).sum()),
                }
            )
            for epoch, ratio in enumerate(fold["curve"] or ()):
                curves.append(
                    {"arm": arm, "fold": a, "epoch": epoch, "held_out_ratio": ratio}
                )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    pd.DataFrame(curves, columns=["arm", "fold", "epoch", "held_out_ratio"]).to_csv(
        out_dir / "curves.csv", index=False
    )

    variants = {
        "all_folds": list(anchors),
        "without_hct116": [a for a in anchors if a != HCT116],
    }
    samples = _bootstrap(folds, anchors, variants, int(comparison["bootstrap"]))
    by_fold = table.set_index(["arm", "fold"])["held_out_ratio"]
    pooled, verdicts = {}, {}
    for variant, members in variants.items():
        point = {
            arm: float(np.mean([by_fold[(arm, a)] for a in members])) for arm in ARMS
        }
        pooled[variant] = {
            arm: {
                "ratio": _number(point[arm]),
                "interval": _interval(samples[variant][arm]),
            }
            for arm in ARMS
        }
        verdicts[variant] = {}
        for name, arm, reference in VERDICTS:
            low, high = _interval(samples[variant][arm] - samples[variant][reference])
            separated = low is not None and (low > 0 or high < 0)
            verdicts[variant][name] = {
                "difference": _number(point[arm] - point[reference]),
                "interval": [low, high],
                "label": "separated" if separated else "overlap",
            }
    _write_json(
        out_dir / "verdicts.json",
        {
            "statistic": "held-out loss ratio to no-change, mean of per-fold ratios",
            "difference": "first arm minus second arm; below 0 favours the first",
            "folds": variants,
            "bootstrap": int(comparison["bootstrap"]),
            "seed": SEED,
            "pooled": pooled,
            "verdicts": verdicts,
        },
    )
    return table


def run_comparison(
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    device: str,
    inputs: Any = None,
    state_factory: Callable[[], nn.Module] | None = None,
) -> pd.DataFrame:
    """Run (or resume) the whole comparison in this process; the arm x fold table.

    The untrained arms, every pending trained job and the summary, in that
    order. ``state_factory`` returns a fresh released STATE model on each call;
    by default it loads ``paths.state_checkpoint``. Returns the existing table
    without recomputing when ``verdicts.json`` exists.
    """
    out_dir = Path(out_dir)
    if (out_dir / "verdicts.json").is_file():
        return pd.read_csv(out_dir / "comparison.csv")
    inputs = load_inputs(config) if inputs is None else inputs
    state_factory = state_factory or _released_state_factory(config)
    jobs = pending_trained_jobs(config, out_dir, inputs=inputs)
    if jobs or not _untrained_done(out_dir, _anchors(config, inputs)):
        # One world for every job: the observed cells are read from the cache once.
        world = _world(
            inputs, torch.device(device), int(config["comparison"]["batch_size"])
        )
        _run_untrained(world, config, out_dir, state_factory)
        for arm, anchor in jobs:
            _run_trained(world, inputs, config, out_dir, arm, anchor, state_factory)
    return summarise(config, out_dir, inputs=inputs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--untrained",
        action="store_true",
        help="only the untrained arms and sanity.json",
    )
    mode.add_argument(
        "--job",
        nargs=2,
        metavar=("ARM", "ANCHOR"),
        help="only one trained arm on one held-out anchor",
    )
    mode.add_argument(
        "--summarise",
        action="store_true",
        help="only the tables and verdicts from existing fold files",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.untrained:
        run_untrained(config, args.out_dir, device=args.device)
    elif args.job:
        run_trained_job(config, args.out_dir, *args.job, device=args.device)
    else:
        if args.summarise:
            table = summarise(config, args.out_dir)
        else:
            table = run_comparison(config, args.out_dir, device=args.device)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

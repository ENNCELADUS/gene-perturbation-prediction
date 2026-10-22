"""The six response-model arms the comparison scores in STATE's log expression space:
the world they share (anchors, observed cells, folds), each arm, their losses and
scores, training, and the per-fold result files."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.embeddings import Esm2EmbeddingTable
from src.data.io import write_json
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.response import predict_bags
from src.model.response_mlp import ResponseMLP
from src.model.state import StateResponse

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
class World:
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


def build_world(inputs: Any, device: torch.device, batch_size: int) -> World:
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
    return World(
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
class Arm:
    """A response predictor: ``predict(model_ids, genes) -> predicted bags``."""

    predict: Predict
    module: nn.Module | None = None
    param_groups: list[dict[str, Any]] = field(default_factory=list)
    covers: Callable[[str], bool] = lambda gene: True
    # The prediction ignores the gene, so a gene shuffle cannot change the loss.
    gene_blind: bool = False


def no_change(world: World) -> Arm:
    return Arm(
        lambda model_ids, genes: [world.basal[m] for m in model_ids], gene_blind=True
    )


def global_mean_effect(world: World, source: np.ndarray) -> Arm:
    """Mean over source conditions of mean(target) - mean(that anchor's controls)."""
    index = torch.from_numpy(source).to(world.device)
    controls = world.control_mean[
        torch.from_numpy(world.anchor_of[source]).to(world.device)
    ]
    shift = (world.observed.mean[index] - controls).sum(dim=0) / len(source)
    return Arm(
        lambda model_ids, genes: [world.basal[m] + shift for m in model_ids],
        gene_blind=True,
    )


def _state_arm(world: World, response: StateResponse) -> Predict:
    def predict(
        model_ids: Sequence[str], genes: Sequence[str]
    ) -> Sequence[torch.Tensor]:
        bags = [world.basal[m] for m in model_ids]
        return predict_bags(response, bags, genes, seed=SEED)

    return predict


def released_state(world: World, state: nn.Module, onehot: OneHotPerturbations) -> Arm:
    response = StateResponse(state, onehot).to(world.device)
    return Arm(_state_arm(world, response), response, covers=onehot.covers)


def state_joint(
    world: World, state: nn.Module, inputs: Any, config: Mapping[str, Any]
) -> Arm:
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
    return Arm(_state_arm(world, response), response, groups)


def mlp(world: World, cells: Mapping[str, torch.Tensor], hidden: int, lr: float) -> Arm:
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

    return Arm(predict, model, [{"params": list(model.parameters()), "lr": lr}])


def _response_losses(
    world: World, predicted: Sequence[torch.Tensor], indices: Sequence[int]
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
    arm: Arm, world: World, indices: Sequence[int], genes: Sequence[str]
) -> torch.Tensor:
    predicted = arm.predict([world.model_ids[i] for i in indices], genes)
    return _response_losses(world, predicted, indices)


def losses(
    arm: Arm,
    world: World,
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


def loss_ratio(loss: np.ndarray, no_change: np.ndarray) -> float:
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


def gene_blind_shuffled(loss: np.ndarray, shuffles: int) -> np.ndarray:
    """Gene-shuffled losses of a gene-blind arm: its own losses, where scored."""
    shuffled = np.full(len(loss), np.nan)
    scored = np.isfinite(loss)
    if shuffles and scored.sum() >= 2:
        shuffled[scored] = loss[scored]
    return shuffled


def score_arm(
    arm: Arm,
    world: World,
    held_out: np.ndarray,
    source: np.ndarray,
    *,
    position: int,
    shuffles: int,
    batch_size: int,
) -> dict[str, list[float | None]]:
    """Held-out, gene-shuffled held-out and source per-condition losses."""
    loss = losses(arm, world, held_out, batch_size)
    if arm.gene_blind:
        shuffled = gene_blind_shuffled(loss, shuffles)
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
            totals += losses(arm, world, covered, batch_size, genes=list(genes[order]))
        else:
            if shuffles:
                shuffled[np.isfinite(loss)] = totals / shuffles
    return result(loss, shuffled, losses(arm, world, source, batch_size))


def result(
    loss: np.ndarray, shuffled: np.ndarray, source: np.ndarray
) -> dict[str, list[float | None]]:
    return {
        "loss": _listed(loss),
        "shuffled_loss": _listed(shuffled),
        "source_loss": _listed(source),
    }


def _listed(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def array(values: Sequence[float | None]) -> np.ndarray:
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


def train(
    arm: Arm,
    world: World,
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
    curve = [loss_ratio(losses(arm, world, held_out, batch_size), no_change)]
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
        curve.append(loss_ratio(losses(arm, world, held_out, batch_size), no_change))
    return curve


def fold_path(out_dir: Path, arm: str, anchor: str) -> Path:
    return out_dir / "folds" / f"{arm}__{anchor}.json"


def fold_indices(world: World, anchor: str) -> tuple[np.ndarray, np.ndarray]:
    """Held-out conditions of ``anchor`` and the source conditions, anchor order."""
    source = np.concatenate([world.by_anchor[b] for b in world.anchors if b != anchor])
    return world.by_anchor[anchor], source


def write_fold(
    out_dir: Path,
    world: World,
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
    write_json(fold_path(out_dir, arm, anchor), payload)
    print(f"response comparison: {arm} held out {anchor} done", flush=True)

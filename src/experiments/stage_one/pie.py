"""The frozen PIE as the ladder runs it: the ladder root's layout and tables, PIE's
run root and checkpoint, its DepMap source masked, and the batches it reads."""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.experiments.run_dir import absolute
from src.experiments.stage_one.pie_reproduction import CHECKPOINT_FILES

DEPMAP_SOURCE: Final[str] = "depmap_gene_effect"
#: Our lines' controls-only dataset, as PIE names a preprocessed dir.
DATASET: Final[str] = "geneeffect_ladder"
EXPERIMENT: Final[str] = "replogle_xdataset"


ROW_SETS: Final[tuple[str, ...]] = ("test_seen", "test_unseen")

SHARD_ARRAYS: Final[tuple[str, ...]] = ("dH", "dL", "own_h", "own_dp", "own_x")


@dataclass(frozen=True)
class Layout:
    """Where the pass reads and writes."""

    root: Path  # the ladder root
    checkpoint: Path
    context_text: Path
    config: Mapping[str, Any]

    @property
    def pie(self) -> Path:
        return self.root / "pie"

    @property
    def dataset(self) -> Path:
        return self.pie / "dataset"

    @property
    def genes_dir(self) -> Path:
        return self.pie / "genes"

    @property
    def full_dir(self) -> Path:
        return self.pie / "full"

    @property
    def pilot(self) -> Path:
        return self.pie / "pilot"


def layout(config: Mapping[str, Any]) -> Layout:
    return Layout(
        root=absolute(config["ladder_root"]),
        checkpoint=absolute(config["paths"]["pie_checkpoint"]),
        context_text=absolute(config["paths"]["context_text"]),
        config=config,
    )


@contextlib.contextmanager
def exclusive(out: Layout) -> Iterator[None]:
    """Shards on one host take turns at what they share: the dataset, the run
    root's links, the reference pass and ``run.json``. Not reentrant."""
    out.pie.mkdir(parents=True, exist_ok=True)
    with (out.pie / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Tables:
    """Lines, the decode panel (gene_text keys, in genes.parquet order) and
    PIE's input."""

    lines: pd.DataFrame
    genes: pd.DataFrame
    pie_input: pd.DataFrame

    @property
    def panel(self) -> list[str]:
        """Upper-case symbols of the decode panel."""
        return list(self.genes.index[self.genes["in_panel"]])

    @property
    def keys(self) -> list[str]:
        """The panel's gene_text keys: PIE's names for its genes."""
        return list(self.genes.loc[self.panel, "gene_text_key"])

    @property
    def frame(self) -> list[str]:
        return list(
            self.lines.index[self.lines["role"].isin(["train_cells", "train_bulk"])]
        )

    @property
    def selective(self) -> set[str]:
        return set(self.genes.index[self.genes["selective"] & self.genes["in_panel"]])


def read_tables(root: Path) -> Tables:
    lines = pd.read_parquet(root / "lines.parquet")
    genes = pd.read_parquet(root / "genes.parquet")
    pie_input = pd.read_parquet(root / "mean" / "pie_input.parquet")
    panel = list(genes.index[genes["in_panel"]])
    if list(pie_input.columns) != panel or list(pie_input.index) != list(lines.index):
        raise ValueError(f"{root}: PIE's input does not match the lines and panel")
    return Tables(lines=lines, genes=genes, pie_input=pie_input)


def pilot_genes(tables: Tables, count: int, seed: int) -> list[str]:
    """``count`` panel genes drawn at random, in panel order."""
    panel = tables.panel
    chosen = np.random.default_rng(seed).choice(len(panel), size=count, replace=False)
    return [panel[i] for i in sorted(chosen)]


def shard_genes(genes: Sequence[str], shard: tuple[int, int]) -> list[str]:
    index, count = shard
    return list(genes)[index::count]


def write_dataset(out: Layout, tables: Tables) -> Path:
    """Our lines as PIE's controls-only preprocessed dir: one context per line
    (its ModelID), the decode panel as the gene axis, PIE's input as the control
    means. An existing dir is reused only when it holds these lines, genes and
    control means."""
    from pie.data.preprocessed import (
        CTRL_MEANS,
        FORMAT_VERSION,
        PreprocessedDir,
        PreprocessedMeta,
        write_preprocessed,
    )

    lines = list(tables.lines.index)
    if (out.dataset / "meta.json").is_file():
        existing = PreprocessedDir.open(out.dataset)
        if existing.contexts != lines or existing.genes != list(tables.keys):
            raise ValueError(
                f"{out.dataset} holds other lines or genes than these tables; "
                "move it aside before writing another dataset"
            )
        if not np.array_equal(
            existing.ctrl_means, tables.pie_input.to_numpy(dtype=np.float32)
        ):
            raise ValueError(
                f"{out.dataset} holds other control means than these tables"
            )
        return out.dataset
    meta = PreprocessedMeta(
        format_version=FORMAT_VERSION,
        dataset=DATASET,
        genes=tables.keys,
        context_to_id={m: i for i, m in enumerate(lines)},
        pert_to_id={},
        pert_kind="gene",
        control_label="basal",
        num_rows=0,
        num_genes=len(tables.keys),
        num_contexts=len(lines),
        num_perts=0,
        controls_only=True,
        tool_version="geneeffect-ladder",
        array_sha256={},
    )
    values = tables.pie_input.to_numpy(dtype=np.float32)
    return write_preprocessed(out.dataset, meta, {CTRL_MEANS: values})


@contextlib.contextmanager
def pie_environment(out: Layout) -> Iterator[Path]:
    """PIE's run root (``PIE_RUNS_ROOT``: links to the checkpoint's files) and
    its directory as the working directory, where the checkpoint's legacy
    relative paths resolve to PIE's copies."""
    for name in ("PIE_DATA_ROOT", "PIE_CACHE_DIR"):
        if not os.environ.get(name):
            raise RuntimeError(
                f"{name} is unset; run through hpc/run.sh pie-precompute"
            )
    runs = out.pie / "runs"
    experiment = runs / EXPERIMENT
    experiment.mkdir(parents=True, exist_ok=True)
    with exclusive(out):
        for name in CHECKPOINT_FILES:
            link = experiment / name
            target = out.checkpoint.parent / name
            if link.is_symlink() and link.resolve() != target.resolve():
                link.unlink()
            if not link.is_symlink():
                link.symlink_to(target)
    os.environ["PIE_RUNS_ROOT"] = str(runs)
    with contextlib.chdir(experiment):
        yield experiment


def mask_depmap(batch: Any) -> Any:
    """The batch without PIE's DepMap token: rows left with no source attend
    PIE's zero carrier, as a perturbation without a DepMap row does."""
    import dataclasses

    import torch
    from pie.data.dataset import NO_SOURCE

    tokens = {k: v for k, v in batch.source_tokens.items() if k != DEPMAP_SOURCE}
    masks = {k: v for k, v in batch.source_masks.items() if k != DEPMAP_SOURCE}
    if DEPMAP_SOURCE in batch.source_masks:
        left = [m for k, m in masks.items() if k != NO_SOURCE]
        b = batch.source_masks[DEPMAP_SOURCE].shape[0]
        any_left = (
            torch.stack([m.any(dim=1) for m in left], dim=1).any(dim=1)
            if left
            else torch.zeros(b, dtype=torch.bool)
        )
        empty = ~any_left
        if NO_SOURCE in masks:
            empty = empty | masks[NO_SOURCE][:, 0]
        if bool(empty.any()):
            tokens[NO_SOURCE] = torch.zeros(b, 1, 1)
            masks[NO_SOURCE] = empty[:, None]
        else:
            tokens.pop(NO_SOURCE, None)
            masks.pop(NO_SOURCE, None)
    return dataclasses.replace(batch, source_tokens=tokens, source_masks=masks)


def masked_collate(
    samples: list[dict[str, Any]], source_dims: Mapping[str, int]
) -> Any:
    """PIE's ``collate``, then the DepMap token removed."""
    from pie.data.dataset import collate

    return mask_depmap(collate(samples, source_dims))


@contextlib.contextmanager
def depmap_masked_prediction() -> Iterator[None]:
    """PIE's own prediction path (``pie.predict``) with every batch masked."""
    import pie.predict

    original = pie.predict.collate
    pie.predict.collate = masked_collate
    try:
        yield
    finally:
        pie.predict.collate = original


@dataclass
class Pie:
    """The frozen checkpoint and PIE's data path over our dataset."""

    model: Any
    module: Any  # PieDataModule
    dataset: Any  # PieDataset over our query rows
    config: Any  # the checkpoint's TrainConfig
    centres: Any  # (n_bins,) delta-p bin centres, on the device
    temperature: float
    device: Any
    pert_sources: tuple[str, ...]
    context_sources: tuple[str, ...]
    #: Run-level caches of the decode and the batches (see ``forward_tensors``).
    cache: dict[Any, Any] = field(default_factory=dict)


def load_pie(out: Layout, tables: Tables, device: str) -> Pie:
    """The checkpoint loaded strictly, over our dataset and context text; must run
    inside ``pie_environment``."""
    import torch
    from pie.assets import resolve_asset
    from pie.data.datamodule import PieDataModule
    from pie.data.dataset import RowRef
    from pie.data.delta_p import dequantize
    from pie.model import PieModel
    from pie.predict import load_checkpoint
    from pie.sources.contract import read_source
    from pie.utils import configure_determinism

    loaded = load_checkpoint(out.checkpoint)
    configure_determinism(loaded.config.seed)
    train = loaded.config.data
    sources = dict(train.source_dirs)
    sources["context_text"] = str(out.context_text)
    data = train.model_copy(
        update={
            "preprocessed_dirs": [str(out.dataset)],
            "dataset_weights": None,
            "source_dirs": sources,
        }
    )
    module = PieDataModule(
        data, None, loaded.stats, seed=loaded.config.seed, evidence_source=train
    )
    module.setup_stats(rank=0)
    if module.gene_axis != tables.keys:
        raise ValueError("PIE's gene axis is not the decode panel")
    rows = [RowRef(0, -1, m, tables.keys[0]) for m in tables.lines.index]
    dataset = module.make_dataset(rows, "none")
    index = {
        name: read_source(resolve_asset(path, kind="source")).meta.index
        for name, path in sources.items()
    }
    torch_device = torch.device(device)
    model = PieModel(
        loaded.config.model,
        loaded.stats,
        dict(module.source_dims),
        module.gene_query_text(),
    )
    model.load_state_dict(loaded.state_dict, strict=True)
    model.to(torch_device).eval()
    grid = loaded.stats.delta_p
    return Pie(
        model=model,
        module=module,
        dataset=dataset,
        config=loaded.config,
        centres=dequantize(torch.arange(grid.n_bins), grid).to(torch_device),
        temperature=float(loaded.config.model.temperature.delta_p),
        device=torch_device,
        pert_sources=tuple(
            n for n, i in index.items() if i != "context" and n != DEPMAP_SOURCE
        ),
        context_sources=tuple(n for n, i in index.items() if i == "context"),
    )


def is_mixed(pie: Pie, precise: bool) -> bool:
    return not precise and pie.config.trainer.precision == "bf16-mixed"


def autocast(pie: Pie, precise: bool = False):
    import torch

    if is_mixed(pie, precise):
        return torch.autocast(
            pie.device.type, dtype=torch.bfloat16, cache_enabled=False
        )
    return contextlib.nullcontext()


def gene_evidence(pie: Pie, perturbation: str) -> dict[str, np.ndarray]:
    """A perturbation's same-perturbation evidence on the panel (our dataset is
    not in PIE's evidence, so nothing is excluded and the same-context blocks are
    empty); a perturbation PIE's evidence does not know has none (all zero)."""
    from pie.data.evidence import serve_evidence

    return serve_evidence(
        pie.dataset.evidence,
        dataset=DATASET,
        context=pie.dataset.rows[0].context,
        perturbation=perturbation,
        gene_union_ids=pie.dataset.gene_union_ids[0],
    )


#: A perturbation no source and no evidence knows: the reference pass.
NO_PERTURBATION: Final[str] = "__no_perturbation__"


def _context_tokens(pie: Pie, lines: Sequence[str]) -> tuple[dict, dict, Any]:
    """The lines' context-indexed source tokens and masks on the device, and
    which lines have any (CPU)."""
    import torch

    device, b = pie.device, len(lines)
    lookup = pie.dataset.sources
    tokens: dict[str, Any] = {}
    masks: dict[str, Any] = {}
    present = torch.zeros(b, dtype=torch.bool)
    for name in pie.context_sources:
        rows = [lookup.tokens(name, context=m, perturbation="") for m in lines]
        if all(r is None for r in rows):
            continue
        width = max(r.shape[0] for r in rows if r is not None)
        dim = next(r.shape[1] for r in rows if r is not None)
        dtype = next(r.dtype for r in rows if r is not None)
        padded = np.zeros((b, width, dim), dtype=dtype)
        mask = np.zeros((b, width), dtype=bool)
        for i, r in enumerate(rows):
            if r is not None:
                padded[i, : r.shape[0]] = r
                mask[i, : r.shape[0]] = True
        tokens[name] = torch.from_numpy(padded).to(device)
        masks[name] = torch.from_numpy(mask).to(device)
        present |= torch.from_numpy(mask.any(axis=1))
    return tokens, masks, present


def build_batch(
    pie: Pie,
    lines: Sequence[str],
    perturbation: str | None,
    ctrl: np.ndarray,
    evidence: Mapping[str, np.ndarray],
    owners: Sequence[str] | None = None,
) -> Any:
    """The batch PIE's dataset and collate build for ``lines`` x one
    perturbation (None: the reference pass), the perturbation's tokens and
    evidence shared across rows, no DepMap token, on the device. ``lines`` name
    the contexts (the control cache's key); a context's text is its owner's
    (``owners``, default the context names themselves: a state pseudo-context
    reads its line's Cellosaurus text)."""
    import torch
    from pie.data.dataset import NO_SOURCE, Batch, GeneGroup

    owners = list(lines) if owners is None else list(owners)
    if len(owners) != len(lines):
        raise ValueError("one owner per context")
    device, b = pie.device, len(lines)
    lookup = pie.dataset.sources
    context_tokens, context_masks, context_present = cached(
        pie, ("context", tuple(owners)), lambda: _context_tokens(pie, owners)
    )
    tokens: dict[str, Any] = dict(context_tokens)
    masks: dict[str, Any] = dict(context_masks)
    present = context_present.clone()
    if perturbation is not None:
        for name in pie.pert_sources:
            row = lookup.tokens(name, context="", perturbation=perturbation)
            if row is None:
                continue
            single = torch.from_numpy(row).to(device)
            tokens[name] = single.unsqueeze(0).expand(b, -1, -1)
            masks[name] = torch.ones(b, row.shape[0], dtype=torch.bool, device=device)
            present[:] = True
    if not bool(present.all()):
        tokens[NO_SOURCE] = torch.zeros(b, 1, 1, device=device)
        masks[NO_SOURCE] = (~present).to(device)[:, None]
    gene_ids = torch.from_numpy(pie.dataset.gene_union_ids[0]).to(device)
    group = GeneGroup(
        rows=torch.arange(b, device=device),
        dir_index=0,
        gene_ids=gene_ids,
        ctrl_means=torch.from_numpy(np.ascontiguousarray(ctrl, dtype=np.float32)).to(
            device
        ),
        evidence={
            key: torch.from_numpy(value).to(device).unsqueeze(0).expand(b, -1, -1)
            for key, value in evidence.items()
        },
        fold_changes=None,
        de_mask=None,
        tested=None,
        delta_p=None,
        lfc_true=None,
    )
    name = perturbation if perturbation is not None else NO_PERTURBATION
    return Batch(
        source_tokens=tokens,
        source_masks=masks,
        dataset_ids=torch.zeros(b, dtype=torch.long, device=device),
        ctx_names=list(lines),
        pert_names=[name] * b,
        row_index=torch.full((b,), -1, dtype=torch.long, device=device),
        target_gene=torch.full((b,), -1, dtype=torch.long, device=device),
        target_gene_idx=torch.full((b,), -1, dtype=torch.long, device=device),
        groups=[group],
    )


@dataclass
class PassOutput:
    """One batch of lines x one perturbation (numpy, float32)."""

    latent: np.ndarray  # (b, d) mean over the latent array
    features: np.ndarray  # (b, d) decoder features pooled over the panel
    own: np.ndarray | None  # (b, d) the perturbed gene's own features
    change: np.ndarray | None  # (b, P) predicted change in mean expression
    dp_logits: np.ndarray | None  # (b, P, n_bins), only when asked (agreement)


def cached(pie: Pie, key: tuple, make: Any) -> Any:
    if key not in pie.cache:
        pie.cache[key] = make()
    return pie.cache[key]

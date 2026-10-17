"""The frozen PIE's deltas for every line x panel gene (PIE stage-one design §3.2,
Task 2), run in PIE's environment (``.venv-pie``).

``python -m src.experiments.pie_precompute CONFIG --steps STEPS [--shard I/N]
[--device cuda]`` (``hpc/run.sh pie-precompute``) reads the ladder root that
``src.experiments.ladder_inputs`` wrote (``lines.parquet``, ``genes.parquet``,
``mean/pie_input.parquet``) and the Cellosaurus ``context_text`` source. Steps,
each skipping what exists:

- ``rescore``: PIE_replogle_xdataset re-scored on Replogle ``test_seen`` and
  ``test_unseen`` with its DepMap source masked, beside Task 1's unmasked
  numbers (what masking costs PIE on its own task).
- ``pilot``: the gene pass (below) on ``pie.pilot_genes`` random panel genes,
  every line, keeping each pair's full predicted change.
- ``basis``: the ``pie.components`` principal components of the pilot's
  frame-line predicted changes (the profile delta's basis); the pilot report
  ``pilot.md`` (timing, variance across lines within a gene against variance
  between genes per delta, the basis's variance kept, and agreement with PIE's
  own data path). People read it before ``full`` runs.
- ``full``: the gene pass on every panel gene (``--shard I/N`` takes every
  N-th), projecting each predicted change on the basis.
- ``assemble``: every delta centred within its gene on the frame lines' mean
  (the means stored) into gene x line arrays under ``pie/``.

The gene pass, for gene g and every line c: PIE's inputs are the line's control
mean on the decode panel (``pie_input``), its Cellosaurus text where a record
exists, g's perturbation sources and g's same-perturbation evidence from the
checkpoint's training data (our dataset is not in that evidence, so nothing is
excluded and the same-context blocks are empty). PIE's ``depmap_gene_effect``
token is never served. The reference pass of each line has the same context and
no perturbation source or response evidence. Per pair:

- ``dH``: last decoder-layer gene features pooled over the panel, knockdown
  minus reference;
- ``dE``: the predicted change in mean expression (softmax expectation over the
  delta-p bin centres at PIE's temperature), as basis coordinates; the full
  panel is kept for the selective genes and the pilot;
- ``dL``: the mean-pooled latent array, knockdown minus reference;
- ``own``: g's own decoder feature, its own predicted change and its control
  mean.

It reports and decides nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS: Final[tuple[str, ...]] = ("rescore", "pilot", "basis", "full", "assemble")
DEPMAP_SOURCE: Final[str] = "depmap_gene_effect"
#: Our lines' controls-only dataset, as PIE names a preprocessed dir.
DATASET: Final[str] = "geneeffect_ladder"
EXPERIMENT: Final[str] = "replogle_xdataset"
CKPT: Final[str] = "best_auprc"
CHECKPOINT_FILES: Final[tuple[str, ...]] = (
    f"{CKPT}.ckpt",
    "config.yaml",
    "data_stats.json",
)
ROW_SETS: Final[tuple[str, ...]] = ("test_seen", "test_unseen")
#: Task 1's run, whose unmasked xdataset scores the masked ones sit beside.
TASK_ONE_RUN: Final[Path] = (
    REPO_ROOT / "outputs/pie_reproduction/pie_reproduction_20261016"
)
#: Our batches against PIE's own dataset and collate, on the same pairs.
AGREEMENT_PAIRS: Final[int] = 4
AGREEMENT_TOLERANCE: Final[float] = 1e-3
SHARD_ARRAYS: Final[tuple[str, ...]] = ("dH", "dL", "own_h", "own_dp", "own_x")


# --- paths ------------------------------------------------------------------------


def _absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


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
        root=_absolute(config["ladder_root"]),
        checkpoint=_absolute(config["paths"]["pie_checkpoint"]),
        context_text=_absolute(config["paths"]["context_text"]),
        config=config,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


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


def record(out: Layout, step: str, payload: Mapping[str, Any]) -> None:
    """Add a step's record to ``pie/run.json``, re-read so shards keep theirs."""
    path = out.pie / "run.json"
    with exclusive(out):
        run = json.loads(path.read_text()) if path.is_file() else {"steps": {}}
        run["steps"][step] = {
            **payload,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json(path, run)


# --- the ladder's tables ------------------------------------------------------------


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


# --- PIE --------------------------------------------------------------------------


def write_dataset(out: Layout, tables: Tables) -> Path:
    """Our lines as PIE's controls-only preprocessed dir: one context per line
    (its ModelID), the decode panel as the gene axis, PIE's input as the control
    means."""
    from pie.data.preprocessed import (
        CTRL_MEANS,
        FORMAT_VERSION,
        PreprocessedMeta,
        write_preprocessed,
    )

    if (out.dataset / "meta.json").is_file():
        return out.dataset
    lines = list(tables.lines.index)
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


def _mixed(pie: Pie, precise: bool) -> bool:
    return not precise and pie.config.trainer.precision == "bf16-mixed"


def _autocast(pie: Pie, precise: bool = False):
    import torch

    if _mixed(pie, precise):
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


def build_batch(
    pie: Pie,
    lines: Sequence[str],
    perturbation: str | None,
    ctrl: np.ndarray,
    evidence: Mapping[str, np.ndarray],
) -> Any:
    """The batch PIE's dataset and collate build for ``lines`` x one
    perturbation (None: the reference pass), the perturbation's tokens and
    evidence shared across rows, no DepMap token, on the device."""
    import torch
    from pie.data.dataset import NO_SOURCE, Batch, GeneGroup

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


def forward(
    pie: Pie,
    batch: Any,
    own_index: int | None,
    *,
    heads: bool,
    logits: bool = False,
    precise: bool = False,
) -> PassOutput:
    """PIE's forward on one batch, as ``PieModel._decode_group`` runs it, keeping
    the latents, the pooled and own decoder features and the delta-p expectation;
    ``precise`` runs it in float32 instead of the checkpoint's precision."""
    import torch

    model = pie.model
    if not _mixed(pie, precise):
        # As PIE's full-precision path: token sources in the parameter dtype.
        batch = _float_tokens(batch)
    with torch.inference_mode(), _autocast(pie, precise):
        memory = model._encode(batch)
        group = batch.groups[0]
        mem = memory[group.rows]
        total = int(group.gene_ids.shape[0])
        chunk = model.cfg.inference_chunk_size
        pooled = torch.zeros(mem.shape[0], mem.shape[2], device=mem.device)
        own = None
        changes, all_logits = [], []
        adapter = model.evidence_output_adapters["delta_p"]
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            feats, fused = model._decode_chunk(
                mem,
                group.gene_ids[start:end],
                group.ctrl_means[:, start:end],
                {key: value[:, start:end] for key, value in group.evidence.items()},
            )
            pooled += feats.float().sum(dim=1)
            if own_index is not None and start <= own_index < end:
                own = feats[:, own_index - start].float()
            if heads:
                dp = model.delta_p_head(
                    feats + adapter(torch.cat([fused, feats], dim=-1))
                )
                probabilities = torch.softmax(dp.float() / pie.temperature, dim=-1)
                # Not a matmul: autocast would run that in bf16.
                changes.append((probabilities * pie.centres).sum(dim=-1))
                if logits:
                    all_logits.append(dp.float())
        return PassOutput(
            latent=memory.float().mean(dim=1).cpu().numpy(),
            features=(pooled / total).cpu().numpy(),
            own=None if own is None else own.cpu().numpy(),
            change=torch.cat(changes, dim=1).cpu().numpy() if heads else None,
            dp_logits=torch.cat(all_logits, dim=1).cpu().numpy() if logits else None,
        )


def _batches(lines: Sequence[str], size: int) -> Iterator[slice]:
    for start in range(0, len(lines), size):
        yield slice(start, min(start + size, len(lines)))


def reference_pass(out: Layout, pie: Pie, tables: Tables) -> dict[str, np.ndarray]:
    """Every line's reference pass (``pie/reference.npz``), computed once: a
    shard that finds it reads it, so every shard subtracts the same one."""
    path = out.pie / "reference.npz"
    with exclusive(out):
        if path.is_file():
            with np.load(path) as payload:
                return {key: payload[key] for key in payload.files}
        lines = list(tables.lines.index)
        blank = gene_evidence(pie, NO_PERTURBATION)
        latent, features = [], []
        size = int(out.config["pie"]["lines_per_batch"])
        for part in _batches(lines, size):
            batch = build_batch(
                pie, lines[part], None, tables.pie_input.to_numpy()[part], blank
            )
            result = forward(pie, batch, None, heads=False)
            latent.append(result.latent)
            features.append(result.features)
        payload = {"latent": np.vstack(latent), "features": np.vstack(features)}
        _save_npz(path, **payload)
        return payload


def gene_pass(
    out: Layout,
    pie: Pie,
    tables: Tables,
    gene: str,
    reference: Mapping[str, np.ndarray],
    *,
    basis: Mapping[str, np.ndarray] | None,
    keep_full: bool,
) -> float:
    """One panel gene over every line: ``pie/genes/<gene>.npz`` (and
    ``pie/full/<gene>.npy`` when kept); returns the seconds it took."""
    start = time.monotonic()
    lines = list(tables.lines.index)
    key = tables.genes.at[gene, "gene_text_key"]
    own_index = tables.panel.index(gene)
    evidence = gene_evidence(pie, key)
    ctrl = tables.pie_input.to_numpy(dtype=np.float32)
    parts: dict[str, list[np.ndarray]] = {
        "latent": [],
        "features": [],
        "own": [],
        "change": [],
    }
    size = int(out.config["pie"]["lines_per_batch"])
    for part in _batches(lines, size):
        batch = build_batch(pie, lines[part], key, ctrl[part], evidence)
        result = forward(pie, batch, own_index, heads=True)
        for name in parts:
            parts[name].append(getattr(result, name))
    change = np.vstack(parts["change"]).astype(np.float32)
    arrays = {
        "dH": np.vstack(parts["features"]) - reference["features"],
        "dL": np.vstack(parts["latent"]) - reference["latent"],
        "own_h": np.vstack(parts["own"]),
        "own_dp": change[:, own_index],
        "own_x": ctrl[:, own_index],
    }
    if basis is not None:
        arrays["dE"] = (change - basis["mean"]) @ basis["components"].T
    if keep_full:
        _save_npy(out.full_dir / f"{gene}.npy", change)
    _save_npz(
        out.genes_dir / f"{gene}.npz",
        **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()},
    )
    return time.monotonic() - start


def run_genes(
    out: Layout,
    tables: Tables,
    genes: Sequence[str],
    device: str,
    *,
    keep_full: set[str],
    basis: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    """The gene pass over ``genes`` lacking a shard."""
    todo = [g for g in genes if not (out.genes_dir / f"{g}.npz").is_file()]
    if not todo:
        return {"genes": len(genes), "computed": 0}
    with pie_environment(out):
        pie = load_pie(out, tables, device)
        difference = agreement(pie, tables, todo)
        if not difference <= AGREEMENT_TOLERANCE:
            raise ValueError(
                f"our batches differ from PIE's own data path by {difference:.3g} in "
                "the float32 delta-p logits"
            )
        reference = reference_pass(out, pie, tables)
        seconds = []
        for number, gene in enumerate(todo, start=1):
            seconds.append(
                gene_pass(
                    out,
                    pie,
                    tables,
                    gene,
                    reference,
                    basis=basis,
                    keep_full=gene in keep_full,
                )
            )
            if number % 25 == 0 or number == len(todo):
                log.info(
                    "gene pass: %d of %d genes, %.1f s per gene",
                    number,
                    len(todo),
                    float(np.mean(seconds)),
                )
    return {
        "genes": len(genes),
        "computed": len(todo),
        "seconds_per_gene": float(np.mean(seconds)),
        "lines": len(tables.lines),
        "agreement": difference,
    }


# --- the masked re-score ----------------------------------------------------------


def rescore(out: Layout, device: str) -> dict[str, Any]:
    """PIE's evaluation of the checkpoint on Replogle with DepMap masked, beside
    Task 1's unmasked scores."""
    from pie.config import EvalConfig
    from pie.evaluate import run_eval

    from src.experiments.pie_reproduction import SPLITS_URI, compare_metrics, read_table

    tables = {}
    with pie_environment(out) as experiment:
        for row_set in ROW_SETS:
            metrics = experiment / "eval" / row_set / f"metrics_{CKPT}.csv"
            granular = experiment / "eval" / row_set / f"granular_{CKPT}.csv"
            if not granular.is_file():
                metrics.unlink(missing_ok=True)
                config = EvalConfig(
                    experiment_name=EXPERIMENT,
                    run_dir=str(experiment),
                    ckpt=CKPT,
                    split_path=f"{SPLITS_URI}/replogle_xdataset/{row_set}.json",
                    row_set=row_set,
                    preprocessed_dirs=None,
                    save_predictions=False,
                    overwrite=False,
                    device=device,
                    batch_size=16,
                )
                with depmap_masked_prediction():
                    run_eval(config)
            unmasked = read_table(
                TASK_ONE_RUN
                / "runs"
                / EXPERIMENT
                / "eval"
                / row_set
                / f"metrics_{CKPT}.csv"
            )
            table = compare_metrics(unmasked, read_table(metrics)).rename(
                columns={"released": "unmasked", "reproduced": "masked"}
            )
            tables[row_set] = table.assign(row_set=row_set)
    joined = pd.concat(tables.values(), ignore_index=True)
    joined.to_csv(out.pilot / "rescore_comparison.csv", index=False)
    overall = joined.loc[joined["context"] == "all"]
    return {
        row_set: {
            row["metric"]: [float(row["unmasked"]), float(row["masked"])]
            for row in overall.loc[overall["row_set"] == row_set].to_dict("records")
        }
        for row_set in ROW_SETS
    }


# --- the basis and the pilot report -----------------------------------------------


def fit_basis(
    changes: np.ndarray, components: int, device: str
) -> dict[str, np.ndarray]:
    """Leading principal components of predicted changes (pairs x panel): mean,
    components (k x panel), and the share of the centred variance each keeps."""
    import torch

    x = torch.from_numpy(np.asarray(changes, dtype=np.float32)).to(device)
    mean = x.mean(dim=0)
    x.sub_(mean)
    total = float(x.pow(2).sum(dtype=torch.float64))
    k = min(components, min(x.shape) - 1)
    _, s, v = torch.svd_lowrank(x, q=k + 10, niter=6)
    s, v = s[:k], v[:, :k]
    return {
        "mean": mean.cpu().numpy(),
        "components": v.T.contiguous().cpu().numpy(),
        "explained": ((s.double() ** 2) / total).cpu().numpy(),
    }


def variance_split(values: np.ndarray) -> dict[str, float]:
    """``values`` genes x lines x width: the mean over genes of the variance
    across lines within a gene, against the variance between the genes' means
    (both summed over the width)."""
    means = values.mean(axis=1)
    within = float(((values - means[:, None]) ** 2).sum(axis=2).mean())
    between = float(((means - means.mean(axis=0)) ** 2).sum(axis=1).mean())
    return {"within_gene": within, "between_genes": between, "ratio": within / between}


def _float_tokens(batch: Any) -> Any:
    """The batch with every floating token source in float32 (PIE's
    full-precision path casts them to the parameter dtype)."""
    import dataclasses

    import torch

    return dataclasses.replace(
        batch,
        source_tokens={
            name: t.to(torch.float32) if t.is_floating_point() else t
            for name, t in batch.source_tokens.items()
        },
    )


def agreement(pie: Pie, tables: Tables, genes: Sequence[str]) -> float:
    """Largest absolute difference of the delta-p logits, in float32, between our
    batches and PIE's own dataset and collate (DepMap masked) on a training line
    with cells, a bulk-only line and a test line, for two genes."""
    import torch
    from pie.data.dataset import RowRef, collate
    from pie.predict import _batch_to

    roles = tables.lines["role"]
    lines = list(tables.lines.index)
    picks = [
        next(m for m in lines if roles[m] == role)
        for role in ("train_cells", "train_bulk", "test")
    ]
    position = [lines.index(m) for m in picks]
    worst = 0.0
    for gene in genes[: AGREEMENT_PAIRS // 2]:
        key = tables.genes.at[gene, "gene_text_key"]
        rows = [RowRef(0, -1, m, key) for m in picks]
        dataset = pie.module.make_dataset(rows, "none")
        batch = mask_depmap(
            collate([dataset[i] for i in range(len(rows))], pie.module.source_dims)
        )
        theirs = _batch_to(batch, pie.device, torch.float32)
        with torch.inference_mode():
            reference = pie.model(theirs)[0].dp_logits.float().cpu().numpy()
        ours = build_batch(
            pie,
            picks,
            key,
            tables.pie_input.to_numpy(dtype=np.float32)[position],
            gene_evidence(pie, key),
        )
        mine = forward(pie, ours, None, heads=True, logits=True, precise=True).dp_logits
        worst = max(worst, float(np.abs(mine - reference).max()))
    return worst


def build_basis(out: Layout, tables: Tables, device: str, seed: int) -> dict[str, Any]:
    """The profile delta's basis (``pie/basis.npz``) and ``pilot/pilot.md``."""
    genes = pilot_genes(tables, int(out.config["pie"]["pilot_genes"]), seed)
    missing = [g for g in genes if not (out.genes_dir / f"{g}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"pilot genes without a shard: {missing[:5]}")
    frame = [tables.lines.index.get_loc(m) for m in tables.frame]
    changes = np.vstack([np.load(out.full_dir / f"{g}.npy")[frame] for g in genes])
    basis = fit_basis(changes, int(out.config["pie"]["components"]), device)
    _save_npz(out.pie / "basis.npz", **basis)
    shards = {}
    for gene in genes:
        with np.load(out.genes_dir / f"{gene}.npz") as payload:
            shards[gene] = {k: payload[k] for k in payload.files}
        full = np.load(out.full_dir / f"{gene}.npy")
        shards[gene]["dE"] = (full - basis["mean"]) @ basis["components"].T
    n_lines = len(tables.lines)
    split = {
        name: variance_split(
            np.stack([np.asarray(shards[g][name]).reshape(n_lines, -1) for g in genes])
        )
        for name in ("dH", "dE", "dL", "own_h", "own_dp")
    }
    run = json.loads((out.pie / "run.json").read_text())
    report = {
        "genes": len(genes),
        "selective": len(set(genes) & tables.selective),
        "lines": len(tables.lines),
        "frame_pairs_in_basis": int(changes.shape[0]),
        "components": int(basis["components"].shape[0]),
        "variance_kept": float(basis["explained"].sum()),
        "variance_split": split,
        "agreement_max_abs_logit_difference": max(
            v["agreement"] for k, v in run["steps"].items() if k.startswith("pilot")
        ),
        "pilot_seconds_per_gene": [
            v["seconds_per_gene"]
            for k, v in run["steps"].items()
            if k.startswith("pilot") and "seconds_per_gene" in v
        ],
        "rescore": run["steps"].get("rescore"),
    }
    write_pilot_report(out, report)
    return report


def write_pilot_report(out: Layout, report: Mapping[str, Any]) -> None:
    lines = [
        "# Frozen-PIE pilot",
        "",
        f"{report['genes']} random panel genes ({report['selective']} selective) x "
        f"{report['lines']} lines; DepMap source masked in every pass.",
        "",
        "## Masked re-score (Replogle, `all` row: unmasked -> masked)",
        "",
        "| Row set | Metric | Unmasked (Task 1) | Masked |",
        "| --- | --- | ---: | ---: |",
    ]
    for row_set, metrics in (report["rescore"] or {}).items():
        if row_set == "finished":
            continue
        for metric, (unmasked, masked) in metrics.items():
            lines.append(f"| {row_set} | {metric} | {unmasked:.4f} | {masked:.4f} |")
    lines += [
        "",
        "## Variance across lines within a gene against variance between genes",
        "",
        "| Delta | Within gene | Between genes | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, split in report["variance_split"].items():
        lines.append(
            f"| {name} | {split['within_gene']:.4g} | {split['between_genes']:.4g} | "
            f"{split['ratio']:.4g} |"
        )
    seconds = report["pilot_seconds_per_gene"]
    lines += [
        "",
        f"Profile-delta basis: {report['components']} components on "
        f"{report['frame_pairs_in_basis']} frame-line pairs keep "
        f"{report['variance_kept']:.4f} of the centred variance.",
        "",
        f"Seconds per gene (all lines, per process): {seconds}.",
        "",
        "Our batches against PIE's own dataset and collate: largest delta-p logit "
        f"difference {report['agreement_max_abs_logit_difference']:.3g}.",
        "",
    ]
    (out.pilot / "pilot.md").write_text("\n".join(lines))


# --- assembly ---------------------------------------------------------------------


def assemble(out: Layout, tables: Tables) -> dict[str, Any]:
    """Gene x line arrays under ``pie/``, each delta centred within its gene on
    the frame lines' mean (``<name>_frame_mean``); pilot genes, computed before
    the basis, are projected now."""
    basis = dict(np.load(out.pie / "basis.npz"))
    genes = tables.panel
    missing = [g for g in genes if not (out.genes_dir / f"{g}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} panel genes without a shard: {missing[:5]}"
        )
    frame = np.asarray([tables.lines.index.get_loc(m) for m in tables.frame])
    n_lines = len(tables.lines)
    arrays: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    for g, gene in enumerate(genes):
        with np.load(out.genes_dir / f"{gene}.npz") as payload:
            shard = {k: payload[k] for k in payload.files}
        if "dE" not in shard:
            full = np.load(out.full_dir / f"{gene}.npy")
            shard["dE"] = (full - basis["mean"]) @ basis["components"].T
        for name in (*SHARD_ARRAYS, "dE"):
            value = np.asarray(shard[name], dtype=np.float32).reshape(n_lines, -1)
            if name not in arrays:
                width = value.shape[1]
                arrays[name] = np.lib.format.open_memmap(
                    out.pie / f".{name}.npy.tmp",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(genes), n_lines, width),
                )
                means[name] = np.zeros((len(genes), width), dtype=np.float32)
            centre = value[frame].mean(axis=0)
            arrays[name][g] = value - centre
            means[name][g] = centre
    for name, array in arrays.items():
        array.flush()
        del array
        os.replace(out.pie / f".{name}.npy.tmp", out.pie / f"{name}.npy")
    _save_npz(out.pie / "frame_means.npz", **means)
    _write_json(
        out.pie / "index.json",
        {
            "genes": genes,
            "lines": list(tables.lines.index),
            "arrays": {
                name: [len(genes), n_lines, int(m.shape[1])]
                for name, m in means.items()
            },
            "centring": "each gene's frame-line mean subtracted; frame_means.npz",
            "full_change": "pie/full/<gene>.npy, lines x panel, for the selective "
            "and pilot genes",
        },
    )
    return {"genes": len(genes), "lines": n_lines}


# --- entry point ------------------------------------------------------------------


def run(
    config: Mapping[str, Any], steps: Sequence[str], shard: tuple[int, int], device: str
) -> None:
    out = layout(config)
    out.pilot.mkdir(parents=True, exist_ok=True)
    if set(steps) - {"rescore"}:
        # The re-score reads PIE's own data alone; every other step our lines.
        tables = read_tables(out.root)
        with exclusive(out):
            write_dataset(out, tables)
    seed = int(config["seed"])
    for step in steps:
        if step == "rescore":
            record(out, "rescore", rescore(out, device))
        elif step == "pilot":
            genes = shard_genes(
                pilot_genes(tables, int(config["pie"]["pilot_genes"]), seed), shard
            )
            result = run_genes(
                out, tables, genes, device, keep_full=set(genes), basis=None
            )
            record(out, f"pilot_shard_{shard[0]}_of_{shard[1]}", result)
        elif step == "basis":
            record(out, "basis", build_basis(out, tables, device, seed))
        elif step == "full":
            basis_path = out.pie / "basis.npz"
            if not basis_path.is_file():
                raise FileNotFoundError(f"{basis_path}: run the pilot and basis first")
            basis = dict(np.load(basis_path))
            genes = shard_genes(tables.panel, shard)
            result = run_genes(
                out, tables, genes, device, keep_full=tables.selective, basis=basis
            )
            record(out, f"full_shard_{shard[0]}_of_{shard[1]}", result)
        elif step == "assemble":
            record(out, "assemble", assemble(out, tables))


def _shard(value: str) -> tuple[int, int]:
    index, count = (int(part) for part in value.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"shard {value}: need 0 <= I < N")
    return index, count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--shard", type=_shard, default=(0, 1))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    from src.experiments.config import load_ladder_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(load_ladder_config(_absolute(args.config)), steps, args.shard, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

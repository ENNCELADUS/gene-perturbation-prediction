"""The frozen PIE's deltas for every line x panel gene (PIE stage-one design §3.2,
Task 2), run in PIE's environment (``.venv-pie``).

``python -m src.experiments.stage_one.pie_precompute CONFIG --steps STEPS [--shard I/N]
[--device cuda]`` (``hpc/run.sh pie-precompute``) reads the ladder root that
``src.experiments.stage_one.ladder_inputs`` wrote (``lines.parquet``, ``genes.parquet``,
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

On a states config (``configs/stage_one/states/``, which names the ladder
config), ``--steps states|assemble_states --set NAME[,NAME]|ALL`` (design §3.5,
Task 8; ``src.experiments.states.state_telemetry`` writes the inputs):

- ``states``: the frozen PIE on every (line, state) pseudo-context of the sets
  (``<ModelID>|<set>|<state>``, the line's Cellosaurus text, the state's input
  as control mean, the set's ``|whole`` context among them) for the pilot's
  genes; ``--shard I/N`` takes every N-th line. Per line
  ``states/pie/<set>/lines/<ModelID>.npz`` (with the line's state-input
  fingerprint), skipped when present with the same fingerprint, refused with
  another; an assembled set is skipped.
- ``assemble_states``: per set, genes x contexts arrays centred on the main
  pass's frame means, ``states/pie/<set>/``, with each line's fingerprint.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from src.data.io import write_json, write_npy, write_npz
from src.experiments.run_dir import REPO_ROOT, absolute, record_step, shard

log = logging.getLogger(__name__)

STEPS: Final[tuple[str, ...]] = ("rescore", "pilot", "basis", "full", "assemble")
#: Steps on a states config (``configs/stage_one/states/``), with ``--set``.
STATE_STEPS: Final[tuple[str, ...]] = ("states", "assemble_states")
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
#: Our (compiled) bf16 decode may be at most this many times as far from PIE's
#: float32 logits as PIE's own bf16 forward is, plus the tolerance.
BF16_ERROR_RATIO: Final[float] = 2.0
SHARD_ARRAYS: Final[tuple[str, ...]] = ("dH", "dL", "own_h", "own_dp", "own_x")


# --- paths ------------------------------------------------------------------------


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
    context_tokens, context_masks, context_present = _cached(
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


# --- the decode, as PIE runs it, with what does not change hoisted --------------
#
# PieModel._decode_group step for step (third_party/pie/src/pie/model/pie.py,
# trunk.py, evidence.py), with three exact hoists: the decoder's memory keys and
# values once per batch, not per gene chunk; the control-mean encoding once per
# line batch and chunk for the whole run (it depends on the line, not on the
# perturbation); the evidence-block encoders on one row, then expanded (the
# blocks are the perturbation's, the same on every line). Under bf16 the chunk
# runs through torch.compile, which fuses the norms, casts and the delta-p
# expectation; the agreement check holds it to PIE's own forward.


def _norm(norm: Any, x: Any) -> Any:
    """PIE's RMSNorm forward (``pie.model.layers.rms_norm``) without its
    autograd wrapper: the same ``F.rms_norm`` call, eps resolved by input dtype."""
    import torch.nn.functional as F

    return F.rms_norm(x, (x.shape[-1],), norm.weight, norm.eps)


def _sequential(layers: Any, x: Any) -> Any:
    from torch import nn

    for layer in layers:
        x = _norm(layer, x) if isinstance(layer, nn.RMSNorm) else layer(x)
    return x


def _heads(attention: Any, x: Any) -> Any:
    b, s, _ = x.shape
    return x.view(b, s, attention.num_heads, attention.head_dim).transpose(1, 2)


def memory_keys(model: Any, memory: Any) -> list[tuple[Any, Any]]:
    """Each decoder layer's normalised keys and values of the latents."""
    keys = []
    for block in model.decoder:
        attention = block.attn
        kv = _norm(block.norm_kv, memory)
        k = _heads(attention, attention.k_proj(kv))
        v = _heads(attention, attention.v_proj(kv))
        keys.append((_norm(attention.k_norm, k), v))
    return keys


def _decoder_block(block: Any, x: Any, k: Any, v: Any) -> Any:
    import torch.nn.functional as F

    attention = block.attn
    b, s, d = x.shape
    q = _heads(attention, attention.q_proj(_norm(block.norm_attn, x)))
    q = _norm(attention.q_norm, q) * attention.attn_scale.view(1, -1, 1, 1)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0
    )
    x = x + attention.out_proj(out.transpose(1, 2).contiguous().view(b, s, d))
    return x + block.ff(_norm(block.norm_ff, x))


def control_encoding(model: Any, ctrl_means: Any) -> Any:
    """(b, chunk) control means -> (b, chunk, width), as ``_fuse_evidence``."""
    from pie.model import evidence as ev

    return _sequential(model.evidence_ctrl_enc, ev.ctrl_query_features(ctrl_means))


def block_encodings(model: Any, evidence: Mapping[str, Any]) -> list[Any]:
    """The evidence blocks' gated encodings in ``_fuse_evidence``'s order, after
    the control one; ``evidence`` holds (1, chunk, k) float32 rows."""
    from pie.data.evidence import HAVE_COL

    def gated(encoder: Any, key: str) -> Any:
        x = evidence[key]
        return _sequential(encoder, x) * x[..., HAVE_COL : HAVE_COL + 1]

    parts = [
        gated(model.evidence_dp_enc, "evidence_dp"),
        gated(model.evidence_de_enc, "evidence_de"),
        gated(model.evidence_lfc_enc, "evidence_lfc"),
    ]
    if model.n_donor_datasets > 1:
        any_donor = evidence["evidence_dp"][..., HAVE_COL : HAVE_COL + 1]
        parts.append(
            _sequential(model.evidence_prov_enc, evidence["evidence_prov"]) * any_donor
        )
    parts.append(gated(model.evidence_ctx_dp_enc, "evidence_ctx_dp"))
    parts.append(gated(model.evidence_ctx_de_enc, "evidence_ctx_de"))
    parts.append(gated(model.evidence_ctx_lfc_enc, "evidence_ctx_lfc"))
    return parts


def decode_chunk(
    model: Any,
    centres: Any,
    temperature: float,
    query: Any,
    control: Any,
    blocks: Sequence[Any],
    keys: Sequence[tuple[Any, Any]],
    heads: bool,
) -> tuple[Any, Any, Any, Any]:
    """One gene chunk: decoded features (b, c, d), their float32 sum over the
    chunk, the delta-p expectation (b, c) and the delta-p logits (heads only)."""
    import torch

    b = control.shape[0]
    parts = [control, *(part.expand(b, -1, -1) for part in blocks)]
    fused = _sequential(model.evidence_fuse, torch.cat(parts, dim=-1))
    x = _norm(
        model.gene_norm,
        query.unsqueeze(0).expand(b, -1, -1)
        + _sequential(model.evidence_query_adapter, fused),
    )
    for block, (k, v) in zip(model.decoder, keys, strict=True):
        x = _decoder_block(block, x, k, v)
    pooled = x.float().sum(dim=1)
    if not heads:
        return x, pooled, None, None
    adapter = model.evidence_output_adapters["delta_p"]
    logits = model.delta_p_head(x + _sequential(adapter, torch.cat([fused, x], dim=-1)))
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return x, pooled, (probabilities * centres).sum(dim=-1), logits


def _chunk_function(pie: Pie, mixed: bool) -> Any:
    """``decode_chunk``, compiled for bf16 on CUDA (compiled once per process)."""
    import torch

    if not (mixed and pie.device.type == "cuda"):
        return decode_chunk
    if "compiled" not in pie.cache:
        pie.cache["compiled"] = torch.compile(decode_chunk, dynamic=False)
    return pie.cache["compiled"]


def _cached(pie: Pie, key: tuple, make: Any) -> Any:
    if key not in pie.cache:
        pie.cache[key] = make()
    return pie.cache[key]


def forward_tensors(
    pie: Pie,
    batch: Any,
    own_index: int | None,
    *,
    heads: bool,
    logits: bool = False,
    precise: bool = False,
    memory: Any | None = None,
) -> dict[str, Any]:
    """PIE's forward on one batch, on the device: the mean latent, the decoder
    features pooled over the panel and at ``own_index``, the delta-p expectation
    per panel gene (heads) and its logits (``logits``); ``precise`` runs in
    float32 instead of the checkpoint's precision. The control and query caches
    (``pie.cache``) are keyed by precision and the batch's lines. ``memory``, the
    batch's encoded latents when given (``owner_memory``), replaces encoding the
    batch."""
    import torch

    model = pie.model
    mixed = _mixed(pie, precise)
    if not mixed:
        # As PIE's full-precision path: token sources in the parameter dtype.
        batch = _float_tokens(batch)
    chunk_function = _chunk_function(pie, mixed)
    group = batch.groups[0]
    lines = tuple(batch.ctx_names)
    with torch.inference_mode(), _autocast(pie, precise):
        if memory is None:
            memory = model._encode(batch)[group.rows]
        keys = memory_keys(model, memory)
        total = int(group.gene_ids.shape[0])
        size = model.cfg.inference_chunk_size
        pooled = torch.zeros(memory.shape[0], memory.shape[2], device=memory.device)
        own = None
        changes, all_logits = [], []
        for start in range(0, total, size):
            end = min(start + size, total)
            ids = group.gene_ids[start:end]
            query = _cached(
                pie,
                ("query", mixed, start),
                lambda: _sequential(model.gene_query_mlp, model.gene_query_text[ids]),
            )
            control = _cached(
                pie,
                ("control", mixed, lines, start),
                lambda: control_encoding(model, group.ctrl_means[:, start:end]),
            )
            blocks = block_encodings(
                model,
                {key: value[:1, start:end] for key, value in group.evidence.items()},
            )
            x, chunk_sum, change, chunk_logits = chunk_function(
                model, pie.centres, pie.temperature, query, control, blocks, keys, heads
            )
            pooled += chunk_sum
            if own_index is not None and start <= own_index < end:
                own = x[:, own_index - start].float()
            if heads:
                changes.append(change)
                if logits:
                    all_logits.append(chunk_logits.float())
        return {
            "latent": memory.float().mean(dim=1),
            "features": pooled / total,
            "own": own,
            "change": torch.cat(changes, dim=1) if heads else None,
            "dp_logits": torch.cat(all_logits, dim=1) if logits else None,
        }


def _numpy(value: Any) -> np.ndarray | None:
    return None if value is None else value.cpu().numpy()


def forward(
    pie: Pie,
    batch: Any,
    own_index: int | None,
    *,
    heads: bool,
    logits: bool = False,
    precise: bool = False,
) -> PassOutput:
    """``forward_tensors`` as numpy."""
    result = forward_tensors(
        pie, batch, own_index, heads=heads, logits=logits, precise=precise
    )
    return PassOutput(**{name: _numpy(value) for name, value in result.items()})


def _batches(lines: Sequence[str], size: int) -> Iterator[slice]:
    for start in range(0, len(lines), size):
        yield slice(start, min(start + size, len(lines)))


def reference_pass(out: Layout, pie: Pie, tables: Tables) -> dict[str, np.ndarray]:
    """Every line's reference pass (``pie/reference.npz``), computed once: a
    shard that finds it reads it, so every shard subtracts the same one. A file
    for other lines is refused (one written before it recorded its lines is
    checked by its row count)."""
    path = out.pie / "reference.npz"
    lines = list(tables.lines.index)
    with exclusive(out):
        if path.is_file():
            with np.load(path) as payload:
                stored = payload["lines"].tolist() if "lines" in payload.files else None
                rows = payload["latent"].shape[0]
                if (stored is not None and stored != lines) or rows != len(lines):
                    raise ValueError(f"{path} holds the reference pass of other lines")
                return {key: payload[key] for key in ("latent", "features")}
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
        write_npz(path, lines=np.asarray(lines), **payload)
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
    parts: dict[str, list[Any]] = {
        "latent": [],
        "features": [],
        "own": [],
        "change": [],
    }
    size = int(out.config["pie"]["lines_per_batch"])
    for part in _batches(lines, size):
        batch = build_batch(pie, lines[part], key, ctrl[part], evidence)
        # On the device until the gene ends: no synchronisation per batch.
        result = forward_tensors(pie, batch, own_index, heads=True)
        for name in parts:
            parts[name].append(result[name])
    stacked = {
        name: np.vstack([_numpy(value) for value in values])
        for name, values in parts.items()
    }
    change = stacked["change"].astype(np.float32)
    arrays = {
        "dH": stacked["features"] - reference["features"],
        "dL": stacked["latent"] - reference["latent"],
        "own_h": stacked["own"],
        "own_dp": change[:, own_index],
        "own_x": ctrl[:, own_index],
    }
    if basis is not None:
        arrays["dE"] = (change - basis["mean"]) @ basis["components"].T
    if keep_full:
        write_npy(out.full_dir / f"{gene}.npy", change)
    write_npz(
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
        difference = checked_agreement(pie, tables, todo)
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

    from src.experiments.stage_one.pie_reproduction import (
        SPLITS_URI,
        compare_metrics,
        read_table,
    )

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


def agreement(pie: Pie, tables: Tables, genes: Sequence[str]) -> dict[str, float]:
    """Largest absolute differences of the delta-p logits on a training line with
    cells, a bulk-only line and a test line, for two genes, against PIE's own
    dataset, collate and forward (DepMap masked) in float32: ``float32`` for our
    batches and decode in float32; under bf16 also ``pie_bf16`` for PIE's own bf16
    forward and ``ours_bf16`` for our (compiled) bf16 decode."""
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
    mixed = _mixed(pie, False)
    worst: dict[str, float] = {"float32": 0.0}
    if mixed:
        worst |= {"pie_bf16": 0.0, "ours_bf16": 0.0}

    def update(name: str, value: np.ndarray, reference: np.ndarray) -> None:
        worst[name] = max(worst[name], float(np.abs(value - reference).max()))

    for gene in genes[: AGREEMENT_PAIRS // 2]:
        key = tables.genes.at[gene, "gene_text_key"]
        rows = [RowRef(0, -1, m, key) for m in picks]
        dataset = pie.module.make_dataset(rows, "none")
        batch = mask_depmap(
            collate([dataset[i] for i in range(len(rows))], pie.module.source_dims)
        )
        with torch.inference_mode():
            theirs = _batch_to(batch, pie.device, torch.float32)
            reference = pie.model(theirs)[0].dp_logits.float().cpu().numpy()
            if mixed:
                with _autocast(pie):
                    own_bf16 = pie.model(_batch_to(batch, pie.device, None))[0]
                update("pie_bf16", own_bf16.dp_logits.float().cpu().numpy(), reference)
        ours = build_batch(
            pie,
            picks,
            key,
            tables.pie_input.to_numpy(dtype=np.float32)[position],
            gene_evidence(pie, key),
        )
        mine = forward(pie, ours, None, heads=True, logits=True, precise=True)
        update("float32", mine.dp_logits, reference)
        if mixed:
            mine = forward(pie, ours, None, heads=True, logits=True)
            update("ours_bf16", mine.dp_logits, reference)
    return worst


def checked_agreement(
    pie: Pie, tables: Tables, genes: Sequence[str]
) -> dict[str, float]:
    """``agreement``, failing closed beyond its tolerances."""
    difference = agreement(pie, tables, genes)
    if not difference["float32"] <= AGREEMENT_TOLERANCE:
        raise ValueError(
            f"our batches differ from PIE's own data path by "
            f"{difference['float32']:.3g} in the float32 delta-p logits"
        )
    if "ours_bf16" in difference and not (
        difference["ours_bf16"]
        <= BF16_ERROR_RATIO * difference["pie_bf16"] + AGREEMENT_TOLERANCE
    ):
        raise ValueError(
            f"our bf16 decode is {difference['ours_bf16']:.3g} from PIE's float32 "
            f"delta-p logits, PIE's own bf16 forward {difference['pie_bf16']:.3g}"
        )
    return difference


def build_basis(out: Layout, tables: Tables, device: str, seed: int) -> dict[str, Any]:
    """The profile delta's basis (``pie/basis.npz``) and ``pilot/pilot.md``."""
    genes = pilot_genes(tables, int(out.config["pie"]["pilot_genes"]), seed)
    missing = [g for g in genes if not (out.genes_dir / f"{g}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"pilot genes without a shard: {missing[:5]}")
    frame = [tables.lines.index.get_loc(m) for m in tables.frame]
    changes = np.vstack([np.load(out.full_dir / f"{g}.npy")[frame] for g in genes])
    basis = fit_basis(changes, int(out.config["pie"]["components"]), device)
    write_npz(out.pie / "basis.npz", **basis)
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
        "agreement": {
            name: max(
                v["agreement"][name]
                for k, v in run["steps"].items()
                if k.startswith("pilot") and "agreement" in v
            )
            for name in ("float32", "pie_bf16", "ours_bf16")
        },
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
        "Largest delta-p logit difference from PIE's own dataset, collate and "
        f"float32 forward: our float32 decode {report['agreement']['float32']:.3g}; "
        f"PIE's own bf16 forward {report['agreement']['pie_bf16']:.3g}, our compiled "
        f"bf16 decode {report['agreement']['ours_bf16']:.3g}.",
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
    write_npz(out.pie / "frame_means.npz", **means)
    write_json(
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


# --- state pseudo-contexts (design §3.5, Task 8) ------------------------------------
#
# Each (line, state) is a context of its own, named ``<ModelID>|<set>|<state>``
# (``src.data.cell_states``), whose control mean is the state's input and whose
# Cellosaurus text is its owner line's. The main dataset and checkpoint are
# reused. The pass is context-major: a batch of contexts runs its own reference
# pass, then every gene, and its cached context tokens and control encodings
# are evicted after it. The encoder reads no expression, so it runs once per
# distinct owner in a batch (exact). The last batch is padded with copies of its
# last context so the compiled decode sees one batch size.

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
    if not _mixed(pie, precise):
        batch = _float_tokens(batch)
    with torch.inference_mode(), _autocast(pie, precise):
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
            record_step(out.pie / "run.json", "rescore", rescore(out, device))
        elif step == "pilot":
            genes = shard_genes(
                pilot_genes(tables, int(config["pie"]["pilot_genes"]), seed), shard
            )
            result = run_genes(
                out, tables, genes, device, keep_full=set(genes), basis=None
            )
            record_step(
                out.pie / "run.json", f"pilot_shard_{shard[0]}_of_{shard[1]}", result
            )
        elif step == "basis":
            record_step(
                out.pie / "run.json", "basis", build_basis(out, tables, device, seed)
            )
        elif step == "full":
            basis_path = out.pie / "basis.npz"
            if not basis_path.is_file():
                raise FileNotFoundError(f"{basis_path}: run the pilot and basis first")
            basis = dict(np.load(basis_path))
            genes = shard_genes(tables.panel, shard)
            result = run_genes(
                out, tables, genes, device, keep_full=tables.selective, basis=basis
            )
            record_step(
                out.pie / "run.json", f"full_shard_{shard[0]}_of_{shard[1]}", result
            )
        elif step == "assemble":
            record_step(out.pie / "run.json", "assemble", assemble(out, tables))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--shard", type=shard, default=(0, 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--set",
        dest="sets",
        default=None,
        help="states steps: the sets of state inputs (comma-separated, or ALL)",
    )
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in (*STEPS, *STATE_STEPS)]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS + STATE_STEPS}")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if set(steps) & set(STATE_STEPS):
        if set(steps) - set(STATE_STEPS) or args.sets is None:
            parser.error(f"{STATE_STEPS} run alone, with --set, on a states config")
        from src.data.cell_states import state_sets
        from src.experiments.states.config import load_states_config

        states_config = load_states_config(absolute(args.config))
        every = state_sets(states_config)
        sets = every if args.sets == "ALL" else args.sets.split(",")
        if set(sets) - set(every):
            parser.error(f"unknown sets {sorted(set(sets) - set(every))}")
        run_state_steps(states_config, steps, sets, args.shard, args.device)
        return 0
    if args.sets is not None:
        parser.error("--set belongs to the states steps")
    from src.experiments.stage_one.config import load_ladder_config

    run(load_ladder_config(absolute(args.config)), steps, args.shard, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

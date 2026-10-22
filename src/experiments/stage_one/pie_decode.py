"""PIE's decode as it runs it (``PieModel._decode_group``, step for step), with what
does not change across genes hoisted, the gene pass over every line, and the check
that it agrees with PIE's own data path."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

import numpy as np

from src.data.io import write_npy, write_npz
from src.experiments.stage_one.pie import (
    NO_PERTURBATION,
    Layout,
    PassOutput,
    Pie,
    Tables,
    autocast,
    build_batch,
    cached,
    exclusive,
    gene_evidence,
    is_mixed,
    load_pie,
    mask_depmap,
    pie_environment,
)

log = logging.getLogger(__name__)


#: Our batches against PIE's own dataset and collate, on the same pairs.
AGREEMENT_PAIRS: Final[int] = 4
AGREEMENT_TOLERANCE: Final[float] = 1e-3
#: Our (compiled) bf16 decode may be at most this many times as far from PIE's
#: float32 logits as PIE's own bf16 forward is, plus the tolerance.
BF16_ERROR_RATIO: Final[float] = 2.0


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


# The hoists are exact: the decoder's memory keys and values depend on the batch, not
# the gene chunk; the control encoding on the line, not the perturbation; the evidence
# blocks on the perturbation alone, so they are encoded on one row and expanded.
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
    mixed = is_mixed(pie, precise)
    if not mixed:
        # As PIE's full-precision path: token sources in the parameter dtype.
        batch = float_tokens(batch)
    chunk_function = _chunk_function(pie, mixed)
    group = batch.groups[0]
    lines = tuple(batch.ctx_names)
    with torch.inference_mode(), autocast(pie, precise):
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
            query = cached(
                pie,
                ("query", mixed, start),
                lambda: _sequential(model.gene_query_mlp, model.gene_query_text[ids]),
            )
            control = cached(
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


def float_tokens(batch: Any) -> Any:
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
    mixed = is_mixed(pie, False)
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
                with autocast(pie):
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

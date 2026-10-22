"""The direct-readout trunk of the stage-one GeneEffect model (an addition to PIE;
see ``VENDORED.md``; the stage-one design is the project's
``docs/specs/2026-10-16-pie-geneeffect-ladder-design.md`` §3.3 and §4.3).

PIE's architecture over one cell line's tokens instead of knowledge sources: the
latents cross-attend to each token channel with that channel's own attention,
normalised over its tokens only, and the channels' outputs are summed (duplicating
a channel's tokens leaves it unchanged; there is no sum pooling and no token-count
input); self-attention blocks process the latents; every gene is decoded
independently by its query (the caller's gene embedding plus the gene's own
evidence, PIE's control-mean features) cross-attending to the latents. Nothing is
per gene, so a gene never trained on decodes.

A long channel is attended in key chunks merged exactly by their log-sum-exp
(``attention_chunk`` keys at a time), equal to one unchunked pass; attention
dropout is off on every channel so the two paths agree in training too.

``warm_start`` loads a released PIE checkpoint's tensors through an explicit name
map, strictly: every mapped tensor must exist with the same shape on both sides;
whatever the map leaves out stays at its fresh initialisation and is returned.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pie.model import evidence as ev
from pie.model.layers import RMSNorm
from pie.model.trunk import FeedForward, MultiHeadAttention, TransformerBlock


def chunked_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, chunk: int) -> Tensor:
    """softmax(q k^T) v over the keys ``mask`` keeps, ``chunk`` keys at a time.

    q (b, h, s, e), k and v (b, h, t, e), mask (b, t) with True = attend; the
    scale is already in q. Each chunk's scores are merged by a running maximum and
    sum (exact); a row with no key outputs zero.
    """
    b, h, s, _ = q.shape
    out = torch.zeros(b, h, s, v.shape[-1], dtype=torch.float32, device=q.device)
    top = torch.full((b, h, s, 1), -torch.inf, dtype=torch.float32, device=q.device)
    total = torch.zeros((b, h, s, 1), dtype=torch.float32, device=q.device)
    for start in range(0, k.shape[2], chunk):
        end = start + chunk
        scores = (q @ k[:, :, start:end].transpose(-1, -2)).float()
        keep = mask[:, None, None, start:end]
        scores = scores.masked_fill(~keep, -torch.inf)
        new_top = torch.maximum(top, scores.amax(dim=-1, keepdim=True))
        finite = torch.where(torch.isfinite(new_top), new_top, torch.zeros_like(new_top))
        rescale = torch.exp(top - finite)
        weights = torch.exp(scores - finite)
        total = total * rescale + weights.sum(dim=-1, keepdim=True)
        out = out * rescale + weights @ v[:, :, start:end].float()
        top = new_top
    return torch.where(total > 0, out / total.clamp_min(1e-30), torch.zeros_like(out)).to(q.dtype)


class ChannelAttention(MultiHeadAttention):
    """PIE's ``MultiHeadAttention`` (same parameters, so it warm-starts by name) over
    one token channel: padding-masked, key-chunked past ``chunk`` keys, no
    attention dropout, a query whose channel is empty for its row gets zero."""

    def __init__(self, d_model: int, num_heads: int, chunk: int) -> None:
        super().__init__(d_model, num_heads, dropout=0.0)
        self.chunk = chunk

    def _heads(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        b, s, d = q.shape
        if kv_mask is None:
            kv_mask = torch.ones(kv.shape[:2], dtype=torch.bool, device=kv.device)
        q_heads = self.q_norm(self._heads(self.q_proj(q)))
        q_heads = q_heads * self.attn_scale.view(1, -1, 1, 1)
        k_heads = self.k_norm(self._heads(self.k_proj(kv)))
        v_heads = self._heads(self.v_proj(kv))
        any_key = kv_mask.any(dim=1)
        if kv.shape[1] <= self.chunk:
            safe = kv_mask | ~any_key[:, None]
            out = F.scaled_dot_product_attention(
                q_heads, k_heads, v_heads, attn_mask=safe[:, None, None, :], scale=1.0
            )
            out = out * any_key[:, None, None, None].to(out.dtype)
        else:
            out = chunked_attention(q_heads, k_heads, v_heads, kv_mask, self.chunk)
        return self.out_proj(out.transpose(1, 2).contiguous().view(b, s, d))


class Channel(nn.Module):
    """One token channel's key norm and attention inside an encoder block."""

    def __init__(self, d_model: int, num_heads: int, chunk: int) -> None:
        super().__init__()
        self.norm_kv = RMSNorm(d_model)
        self.attn = ChannelAttention(d_model, num_heads, chunk)


class ChannelBlock(nn.Module):
    """PIE's pre-norm cross block with one attention per channel, outputs summed:
    RMSNorm -> sum over channels of attention -> residual -> RMSNorm -> FFN ->
    residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        channels: list[str],
        chunk: int,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.channels = nn.ModuleDict(
            {name: Channel(d_model, num_heads, chunk) for name in channels}
        )
        self.norm_ff = RMSNorm(d_model)
        self.ff = FeedForward(d_model, ff_mult, dropout)

    def forward(self, x: Tensor, tokens: Mapping[str, tuple[Tensor, Tensor]]) -> Tensor:
        q = self.norm_attn(x)
        update = None
        for name, channel in self.channels.items():
            kv, mask = tokens[name]
            part = channel.attn(q, channel.norm_kv(kv), mask)
            update = part if update is None else update + part
        x = x + update
        return x + self.ff(self.norm_ff(x))


class GeneEffectTrunk(nn.Module):
    """Latents <- token channels -> processor blocks -> gene queries.

    ``channels`` maps a channel's name to its raw token width (projected to
    ``d_model`` as PIE projects a source: RMSNorm -> Linear) or None when its tokens
    arrive at ``d_model``; every channel's tokens are then RMS-normed (PIE's key
    norm).
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_latents: int,
        n_encoder_layers: int,
        n_processor_layers: int,
        n_decoder_layers: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        channels: Mapping[str, int | None],
        attention_chunk: int,
        evidence_width: int = 64,
        evidence_dim: int = 128,
    ) -> None:
        super().__init__()
        d = d_model
        self.d_model = d
        self.channel_names = list(channels)
        self.token_projections = nn.ModuleDict(
            {
                name: nn.Sequential(RMSNorm(dim), nn.Linear(dim, d, bias=False))
                for name, dim in channels.items()
                if dim is not None
            }
        )
        self.token_norms = nn.ModuleDict({name: RMSNorm(d) for name in channels})
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.latent_prenorm = RMSNorm(d)
        self.encoder = nn.ModuleList(
            [
                ChannelBlock(d, num_heads, ff_mult, dropout, self.channel_names, attention_chunk)
                for _ in range(n_encoder_layers)
            ]
        )
        self.processor = nn.ModuleList(
            [
                TransformerBlock(d, num_heads, ff_mult, dropout, is_cross=False)
                for _ in range(n_processor_layers)
            ]
        )
        self.gene_norm = RMSNorm(d)
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(d, num_heads, ff_mult, dropout, is_cross=True)
                for _ in range(n_decoder_layers)
            ]
        )
        self.evidence_ctrl_enc = ev.encoder(ev.N_CTRL_FEATURES, evidence_width)
        self.evidence_fuse = ev.fuse_block(evidence_width, evidence_dim, dropout)
        self.evidence_query_adapter = ev.query_adapter(evidence_dim, d)

    def encode(
        self, channels: Mapping[str, tuple[Tensor, Tensor]], *, token_dropout: float = 0.0
    ) -> Tensor:
        """``channels`` maps each channel to (tokens (b, t, width), mask (b, t));
        returns the latents (b, n_latents, d). In training a share
        ``token_dropout`` of each line's tokens is dropped from every channel."""
        tokens = {}
        for name in self.channel_names:
            x, mask = channels[name]
            if name in self.token_projections:
                x = self.token_projections[name](x)
            if self.training and token_dropout > 0:
                mask = mask & (torch.rand(mask.shape, device=mask.device) >= token_dropout)
            tokens[name] = (self.token_norms[name](x), mask)
        b = next(iter(tokens.values()))[0].shape[0]
        z = self.latent_prenorm(self.latents.unsqueeze(0).expand(b, -1, -1))
        for block in self.encoder:
            z = block(z, tokens)
        for block in self.processor:
            z = block(z)
        return z

    def decode(
        self,
        memory: Tensor,
        queries: Tensor,
        evidence: Tensor,
        have: Tensor,
        *,
        evidence_dropout: float = 0.0,
    ) -> Tensor:
        """Decoded features (b, g, d) of ``queries`` (g, d) on each line's latents
        ``memory`` (b, n_latents, d). ``evidence`` (b, g) is each gene's own control
        mean in the line, ``have`` (b, g) whether the line's source lists it: an
        unlisted gene's evidence is zero, and in training a share
        ``evidence_dropout`` of the pairs' evidence is zeroed too. Only the gene's
        own evidence moves its query."""
        if self.training and evidence_dropout > 0:
            have = have & (torch.rand(have.shape, device=have.device) >= evidence_dropout)
        features = ev.ctrl_query_features(evidence.float())
        encoded = self.evidence_ctrl_enc(features) * have.unsqueeze(-1).to(features.dtype)
        fused = self.evidence_fuse(encoded)
        b = memory.shape[0]
        x = self.gene_norm(
            queries.unsqueeze(0).expand(b, -1, -1) + self.evidence_query_adapter(fused)
        )
        for block in self.decoder:
            x = block(x, kv=memory, kv_mask=None)
        return x


def trunk_name_map(
    trunk: GeneEffectTrunk, pie_layers: Mapping[str, int], prefix: str = ""
) -> dict[str, str]:
    """Our parameter prefixes -> a released PIE model's, for the parts PIE has: the
    latent norm and (at PIE's latent count) the latents; per encoder block its norms
    and feed-forward, and the text channel's attention from PIE's one attention over
    its sources (the token channel has none in PIE and starts fresh); processor and
    decoder blocks in order; the gene norm; the control-mean evidence encoder; the
    text projection and key norm from PIE's ``context_text`` source. ``pie_layers``
    holds PIE's ``n_latents`` and layer counts; ours may be fewer, never more."""
    names = {f"{prefix}latent_prenorm": "latent_prenorm", f"{prefix}gene_norm": "gene_norm"}
    if trunk.latents.shape[0] == pie_layers["n_latents"]:
        names[f"{prefix}latents"] = "latents"
    for stage in ("encoder", "processor", "decoder"):
        ours = len(getattr(trunk, stage))
        if ours > pie_layers[f"n_{stage}_layers"]:
            raise ValueError(f"{ours} {stage} blocks, PIE has {pie_layers[f'n_{stage}_layers']}")
        for i in range(ours):
            if stage != "encoder":
                names[f"{prefix}{stage}.{i}"] = f"{stage}.{i}"
                continue
            for part in ("norm_attn", "norm_ff", "ff"):
                names[f"{prefix}encoder.{i}.{part}"] = f"encoder.{i}.{part}"
            if "text" in trunk.channel_names:
                names[f"{prefix}encoder.{i}.channels.text.norm_kv"] = f"encoder.{i}.norm_kv"
                names[f"{prefix}encoder.{i}.channels.text.attn"] = f"encoder.{i}.attn"
    names[f"{prefix}evidence_ctrl_enc"] = "evidence_ctrl_enc"
    if "text" in trunk.token_projections:
        names[f"{prefix}token_projections.text"] = "source_projections.context_text"
        names[f"{prefix}token_norms.text"] = "source_k_norms.context_text"
    return names


def warm_start(
    module: nn.Module, state_dict: Mapping[str, Tensor], name_map: Mapping[str, str]
) -> list[str]:
    """Copy ``state_dict``'s tensors into ``module`` through ``name_map`` (our
    prefix -> theirs), strictly: under each mapped prefix both sides must hold the
    same tensor names with the same shapes. Returns our tensors the map leaves
    fresh."""
    ours = module.state_dict()
    loaded: dict[str, Tensor] = {}
    for mine, theirs in name_map.items():
        here = {k[len(mine) :]: k for k in ours if k == mine or k.startswith(mine + ".")}
        there = {
            k[len(theirs) :]: k for k in state_dict if k == theirs or k.startswith(theirs + ".")
        }
        if not here or set(here) != set(there):
            raise ValueError(
                f"warm start {mine!r} <- {theirs!r}: tensors differ "
                f"(ours {sorted(here)[:5]}, PIE's {sorted(there)[:5]})"
            )
        for suffix, key in here.items():
            value = state_dict[there[suffix]]
            if tuple(value.shape) != tuple(ours[key].shape):
                raise ValueError(
                    f"warm start {key} <- {there[suffix]}: shape {tuple(ours[key].shape)} "
                    f"against {tuple(value.shape)}"
                )
            if key in loaded:
                raise ValueError(f"warm start maps {key} twice")
            loaded[key] = value
    module.load_state_dict({**ours, **loaded}, strict=True)
    return sorted(set(ours) - set(loaded))

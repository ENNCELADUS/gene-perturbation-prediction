"""Stage-one architecture arms (PIE stage-one design §3.3; protocol §16).

Every arm predicts ``z``, a (gene, line) pair's GeneEffect minus the base ``m_g`` in
units of the residual SD ``sigma_g``, and ends in Cilantro's tail
(``-> 64 -> act -> 32 -> act -> 1``, from ``FiLMNet128``) whose 32-d layer is the
viability embedding ``V(c, g)``. Cilantro's ReLUs are leaky (slope 0.01) here: on the
residual target, whose learnable part is a few per cent of its variance, the ReLU
tail's units died within an epoch on the H20 and the arms collapsed to the base
(protocol §16). The tail's output layer starts at zero, so an untrained arm predicts
the base. No module holds a per-gene parameter: a gene enters
only through its sources (``GeneEncoder``), so a gene never seen in training decodes.

Arms, by what comes before the tail:

- ``DeltaFiLM`` (virtual cell + FiLM, Cilantro): the pair input -> Linear 256 ->
  ReLU -> FiLM conditioned on the gene encoder. One class serves the frozen-PIE
  deltas (``[Delta centred ; frame mean Delta_g]`` plus own-gene slots), the
  multi-encoder (FiLM condition ``gene encoder (+) context encoder``), its no-delta
  control (a learned constant in place of the input) and the line-level reference
  rows (components, lineage, Cellosaurus text, own expression).
- ``DirectReadout``: a PIE-architecture trunk (``pie.model.geneeffect``) trained from
  fresh weights reads the line's tokens; gene g's query decodes ``h(c, g)`` ->
  Linear 256 -> ReLU.
- ``Hybrid``: ``[h(c, g) ; FiLM(delta)]`` -> Linear 256 -> ReLU.

A batch is a block of genes by a block of lines (``PairBatch``); every arm returns
``z`` (genes x lines) and ``V`` (genes x lines x 32).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: Cilantro's FiLMNet128: cell embedding 256, gene condition 128.
HIDDEN = 256
VIABILITY_WIDTH = 32
#: The tail's activation slope below zero; 0 is Cilantro's ReLU.
TAIL_SLOPE = 0.01


class FiLM(nn.Module):
    """Cilantro's ``FiLMLayer``: ``[gamma, beta] = Linear(condition)``,
    ``x = gamma * x + beta``."""

    def __init__(self, primary: int, condition: int) -> None:
        super().__init__()
        self.cond_gen = nn.Linear(condition, 2 * primary)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = torch.chunk(self.cond_gen(condition), 2, dim=-1)
        return x * gamma + beta


class CilantroTail(nn.Module):
    """``FiLMNet128``'s layers after FiLM: ``l2`` (-> 64) -> act -> ``l3`` (-> 32,
    the viability embedding V, taken before its activation as Cilantro saves it) ->
    act -> ``l4`` (-> 1), where act is a leaky ReLU of ``negative_slope`` (0 gives
    Cilantro's ReLU). ``l4`` starts at zero: an untrained tail outputs 0."""

    def __init__(self, width: int = HIDDEN, negative_slope: float = TAIL_SLOPE) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.l2 = nn.Linear(width, 64)
        self.l3 = nn.Linear(64, VIABILITY_WIDTH)
        self.l4 = nn.Linear(VIABILITY_WIDTH, 1)
        nn.init.zeros_(self.l4.weight)
        nn.init.zeros_(self.l4.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        slope = self.negative_slope
        viability = self.l3(F.leaky_relu(self.l2(x), slope))
        # The output layer runs in fp32 outside autocast, so a stored z is not
        # rounded to bf16.
        with torch.autocast(x.device.type, enabled=False):
            z = self.l4(F.leaky_relu(viability.float(), slope)).squeeze(-1)
        return z, viability


def source_projection(dim: int, width: int) -> nn.Sequential:
    """One gene source -> ``width``, as PIE's ``gene_query_mlp``: RMSNorm -> Linear
    -> GELU -> Linear (so ``gene_text``'s can warm-start from it)."""
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, width, bias=False),
        nn.GELU(),
        nn.Linear(width, width, bias=False),
    )


class GeneEncoder(nn.Module):
    """The masked sum of per-source projections: a gene without an entry in a
    source has that source masked; no per-gene parameter."""

    def __init__(self, source_dims: Mapping[str, int], width: int) -> None:
        super().__init__()
        self.width = width
        self.projections = nn.ModuleDict(
            {name: source_projection(dim, width) for name, dim in source_dims.items()}
        )

    def forward(
        self, vectors: Mapping[str, Tensor], present: Mapping[str, Tensor]
    ) -> Tensor:
        out = None
        for name, projection in self.projections.items():
            part = projection(vectors[name]) * present[name].unsqueeze(-1).to(
                vectors[name].dtype
            )
            out = part if out is None else out + part
        return out


def value_embedding(width: int) -> nn.Sequential:
    """A token's ``[pie value, listed, mean-channel value, in space]`` -> width."""
    return nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))


def masked_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """Single-head attention of (b, q, d) queries over (b, t, d) keys normalised
    over the unmasked keys only; a row with no key outputs zero."""
    any_key = mask.any(dim=1)
    safe = mask | ~any_key[:, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe[:, None, :])
    return out * any_key[:, None, None].to(out.dtype)


class ContextEncoder(nn.Module):
    """The multi-encoder's context: learned queries attend each channel (the
    line's tokens, its Cellosaurus text) with the channel's own normalisation; the
    channels' outputs are summed, flattened and projected to ``width``."""

    def __init__(self, gene_width: int, text_dim: int, width: int, queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.token_gene = nn.Linear(gene_width, width, bias=False)
        self.token_value = value_embedding(width)
        self.text = nn.Sequential(nn.RMSNorm(text_dim), nn.Linear(text_dim, width))
        self.keys = nn.ModuleDict(
            {name: nn.Linear(width, 2 * width) for name in ("tokens", "text")}
        )
        self.norm = nn.RMSNorm(width)
        self.out = nn.Linear(queries * width, width)

    def forward(
        self,
        token_genes: Tensor,  # (t, gene_width): the gene encoder on the token axis
        token_values: Tensor,  # (b, t, 4)
        token_present: Tensor,  # (b, t)
        text: Tensor,  # (b, text_dim)
        text_present: Tensor,  # (b,)
    ) -> Tensor:
        b = token_values.shape[0]
        query = self.queries.unsqueeze(0).expand(b, -1, -1)
        tokens = self.token_gene(token_genes).unsqueeze(0) + self.token_value(
            token_values
        )
        channels = (
            ("tokens", tokens, token_present),
            ("text", self.text(text).unsqueeze(1), text_present[:, None]),
        )
        pooled = 0
        for name, x, mask in channels:
            k, v = torch.chunk(self.keys[name](x), 2, dim=-1)
            pooled = pooled + masked_attention(query, k, v, mask)
        return self.out(self.norm(pooled).flatten(1))


class StateEncoder(nn.Module):
    """The profile delta's state encoder ``E(x) = second(relu(first(x)))``.

    ``change(state, step)`` is ``E(state + step) - E(state)`` computed without
    cancellation: the first layer is linear, so the after-state's pre-activation is
    the before-state's plus ``first.weight @ step``, the two activations are
    subtracted before the second layer, whose bias cancels exactly; all of it in
    fp32 outside autocast (a knockdown moves the state by ~1% of its size, which a
    bf16 difference of two encodings would lose)."""

    def __init__(self, components: int, width: int) -> None:
        super().__init__()
        self.first = nn.Linear(components, width)
        self.second = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.second(F.relu(self.first(x)))

    def change(self, state: Tensor, step: Tensor) -> Tensor:
        with torch.autocast(step.device.type, enabled=False):
            state, step = state.float(), step.float()
            before = self.first(state)
            after = before + F.linear(step, self.first.weight)
            return F.linear(F.relu(after) - F.relu(before), self.second.weight)


@dataclass
class PairBatch:
    """A block of panel genes by a block of lines, on the model's device.

    Gene-level tensors are (genes, ...), line-level (lines, ...), pair-level
    (genes, lines, ...). Only the fields an arm reads are set.
    """

    gene_vectors: Mapping[str, Tensor]  # source -> (genes, dim)
    gene_present: Mapping[str, Tensor]  # source -> (genes,)
    delta: Tensor | None = None  # (genes, lines, w), centred within gene
    delta_mean: Tensor | None = None  # (genes, w)
    before: Tensor | None = None  # (lines, k) or (genes, lines, k): before-state
    shift: Tensor | None = None  # (genes, k): after = before + delta + shift
    own: Tensor | None = None  # (genes, lines, s), centred within gene
    own_mean: Tensor | None = None  # (genes, s)
    line: Tensor | None = None  # (lines, w): a line-level reference input
    token_genes: Mapping[str, Tensor] | None = None  # source -> (tokens, dim)
    token_gene_present: Mapping[str, Tensor] | None = None  # source -> (tokens,)
    token_values: Tensor | None = None  # (lines, tokens, 4)
    token_present: Tensor | None = None  # (lines, tokens)
    text: Tensor | None = None  # (lines, text_dim)
    text_present: Tensor | None = None  # (lines,)
    evidence: Tensor | None = None  # (genes, lines): own value, PIE's input
    evidence_have: Tensor | None = None  # (genes, lines): listed by the source
    prior: Tensor | None = None  # (genes, lines): the prior's prediction, z units

    @property
    def shape(self) -> tuple[int, int]:
        genes = next(iter(self.gene_vectors.values())).shape[0]
        for lines_tensor in (self.delta, self.own, self.evidence, self.prior):
            if lines_tensor is not None:
                return genes, lines_tensor.shape[1]
        for lines_tensor in (self.line, self.token_values, self.before):
            if lines_tensor is not None:
                return genes, lines_tensor.shape[0]
        raise ValueError("a batch without any line input")


class DeltaFiLM(nn.Module):
    """Pair input -> Linear ``hidden`` -> ReLU -> FiLM(condition) -> tail.

    ``input`` names what the pair input is: a delta (``dH``, ``dL``: the stored
    delta beside its frame mean; ``dE`` (on the 512-component basis) and ``dE_full``
    (on the full decode panel): ``E(after) - E(before)`` beside its value
    at the frame-mean states, ``E(shift_g) - E(0)``), a line-level input
    (``components``, ``lineage``, ``text``), ``own_x`` (the gene's own expression
    beside its frame mean), ``none`` (a learned constant) or ``prior`` (the prior's
    prediction alone). Own-gene slots, when given, are appended to a delta as
    ``[slot centred ; frame mean]``; with ``prior`` set, the prior's prediction is
    appended last. The FiLM condition is the gene encoder, plus the context encoder
    when one is given.
    """

    def __init__(
        self,
        *,
        input: str,
        input_dim: int,
        own_dim: int,
        gene_encoder: GeneEncoder,
        context: ContextEncoder | None,
        context_width: int,
        state: StateEncoder | None,
        prior: bool = False,
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        self.input = input
        self.prior = prior or input == "prior"
        self.gene_encoder = gene_encoder
        self.context = context
        self.state = state
        # [value ; frame mean] for a delta (with its own slots) and own_x.
        width = {"own_x": 2 * input_dim}.get(input, input_dim)
        if input in ("dH", "dL", "dE", "dE_full"):
            width = 2 * input_dim + 2 * own_dim
        if input == "prior":
            width = 0
        width += int(self.prior)
        if input == "none":
            self.constant = nn.Parameter(torch.zeros(hidden))
            self.inflow = None
        else:
            self.inflow = nn.Linear(width, hidden)
        condition = gene_encoder.width + (context_width if context is not None else 0)
        self.film = FiLM(hidden, condition)
        self.tail = CilantroTail(hidden)

    def pair_input(self, batch: PairBatch) -> Tensor:
        genes, lines = batch.shape
        if self.input in ("dH", "dL"):
            parts = [batch.delta, _per_gene(batch.delta_mean, lines)]
        elif self.input in ("dE", "dE_full"):
            # The frame-mean term is E at the frame-mean states (before 0, after
            # shift_g): for a linear E, exactly the frame mean of the change.
            before = batch.before
            if before.dim() == 2:  # one state per line, shared by the genes
                before = before.unsqueeze(0)
            step = batch.delta.float() + batch.shift.float().unsqueeze(1)
            change = self.state.change(before, step)
            mean = self.state.change(torch.zeros_like(batch.shift), batch.shift)
            parts = [change - mean.unsqueeze(1), _per_gene(mean, lines)]
        elif self.input == "own_x":
            parts = [batch.own, _per_gene(batch.own_mean, lines)]
        elif self.input == "prior":
            parts = []
        else:  # a line-level input
            return batch.line.float().unsqueeze(0).expand(genes, -1, -1)
        if self.input != "own_x" and batch.own is not None:
            parts += [batch.own, _per_gene(batch.own_mean, lines)]
        if self.prior:
            parts.append(batch.prior.unsqueeze(-1))
        return torch.cat([part.float() for part in parts], dim=-1)

    def condition(self, batch: PairBatch) -> Tensor:
        genes, lines = batch.shape
        gene = self.gene_encoder(batch.gene_vectors, batch.gene_present)
        gene = gene.unsqueeze(1).expand(-1, lines, -1)
        if self.context is None:
            return gene
        token_genes = self.gene_encoder(batch.token_genes, batch.token_gene_present)
        context = self.context(
            token_genes,
            batch.token_values,
            batch.token_present,
            batch.text,
            batch.text_present,
        )
        return torch.cat([gene, context.unsqueeze(0).expand(genes, -1, -1)], dim=-1)

    def features(self, batch: PairBatch) -> Tensor:
        """The FiLM output (genes, lines, hidden), before the tail."""
        genes, lines = batch.shape
        if self.inflow is None:
            x = self.constant.expand(genes, lines, -1)
        else:
            x = F.relu(self.inflow(self.pair_input(batch)))
        return self.film(x, self.condition(batch).to(x.dtype))

    def forward(self, batch: PairBatch) -> tuple[Tensor, Tensor]:
        return self.tail(self.features(batch))


def _per_gene(values: Tensor, lines: int) -> Tensor:
    return values.unsqueeze(1).expand(-1, lines, -1)


class DirectReadout(nn.Module):
    """A PIE-architecture trunk from fresh weights: the line's tokens (gene encoder
    + embedded values) and its Cellosaurus text, one cross-attention per channel,
    into the latents; gene g's query (gene encoder + its own evidence, PIE's input
    for g, masked where the source does not list g) decodes ``h(c, g)`` -> Linear
    ``hidden`` -> ReLU -> tail."""

    def __init__(
        self,
        *,
        gene_encoder: GeneEncoder,
        text_dim: int,
        trunk: Mapping[str, object],
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        from pie.model.geneeffect import GeneEffectTrunk

        d = int(trunk["d_model"])
        if gene_encoder.width != d:
            raise ValueError("the direct readout's gene encoder must be d_model wide")
        self.gene_encoder = gene_encoder
        self.token_value = value_embedding(d)
        self.trunk = GeneEffectTrunk(
            d_model=d,
            n_latents=int(trunk["n_latents"]),
            n_encoder_layers=int(trunk["n_encoder_layers"]),
            n_processor_layers=int(trunk["n_processor_layers"]),
            n_decoder_layers=int(trunk["n_decoder_layers"]),
            num_heads=int(trunk["num_heads"]),
            ff_mult=int(trunk["ff_mult"]),
            dropout=float(trunk["dropout"]),
            channels={"tokens": None, "text": text_dim},
            attention_chunk=int(trunk["attention_chunk"]),
        )
        self.head = nn.Linear(d, hidden)
        self.tail = CilantroTail(hidden)
        self.token_dropout = 0.0
        self.evidence_dropout = 0.0

    def features(self, batch: PairBatch) -> Tensor:
        """``relu(Linear(h(c, g)))`` (genes, lines, hidden), before the tail."""
        token_genes = self.gene_encoder(batch.token_genes, batch.token_gene_present)
        tokens = token_genes.unsqueeze(0) + self.token_value(batch.token_values)
        memory = self.trunk.encode(
            {
                "tokens": (tokens, batch.token_present),
                "text": (batch.text.unsqueeze(1), batch.text_present[:, None]),
            },
            token_dropout=self.token_dropout,
        )
        queries = self.gene_encoder(batch.gene_vectors, batch.gene_present)
        h = self.trunk.decode(
            memory,
            queries,
            batch.evidence.transpose(0, 1),
            batch.evidence_have.transpose(0, 1),
            evidence_dropout=self.evidence_dropout,
        )  # (lines, genes, d)
        return F.relu(self.head(h.transpose(0, 1)))

    def forward(self, batch: PairBatch) -> tuple[Tensor, Tensor]:
        return self.tail(self.features(batch))


class Hybrid(nn.Module):
    """``[h(c, g) ; FiLM(delta)]`` -> Linear ``hidden`` -> ReLU -> tail."""

    def __init__(self, direct: DirectReadout, film: DeltaFiLM, hidden: int = HIDDEN):
        super().__init__()
        self.direct = direct
        self.film = film
        self.join = nn.Linear(2 * hidden, hidden)
        self.tail = CilantroTail(hidden)

    def forward(self, batch: PairBatch) -> tuple[Tensor, Tensor]:
        joined = torch.cat(
            [self.direct.features(batch), self.film.features(batch)], dim=-1
        )
        return self.tail(F.relu(self.join(joined)))


@dataclass(frozen=True)
class InputDims:
    """Widths the model is built for, read from the inputs."""

    gene_sources: Mapping[str, int]
    delta: int  # the delta's width (the basis's k for dE), or the line input's
    own: int  # summed own-slot widths
    text: int


def build_model(config: Mapping[str, object], dims: InputDims) -> nn.Module:
    """The arm a training config names (``validate_stage_one_config`` checked it)."""
    data, model = config["data"], config["model"]
    arm, kind = model["arm"], data["input"]
    if arm == "direct_readout" or arm == "hybrid":
        trunk = model["trunk"]
        direct = DirectReadout(
            gene_encoder=GeneEncoder(dims.gene_sources, int(trunk["d_model"])),
            text_dim=dims.text,
            trunk=trunk,
            hidden=int(model["hidden"]),
        )
        if arm == "direct_readout":
            return direct
    state = None
    if kind in ("dE", "dE_full"):
        state = StateEncoder(dims.delta, int(model["state_width"]))
    context, context_width = None, 0
    if model["context"] is not None:
        context_width = int(model["context"]["width"])
        context = ContextEncoder(
            int(model["gene_width"]),
            dims.text,
            context_width,
            int(model["context"]["queries"]),
        )
    film = DeltaFiLM(
        input=kind,
        input_dim=int(model["state_width"])
        if kind in ("dE", "dE_full")
        else dims.delta,
        own_dim=dims.own,
        gene_encoder=GeneEncoder(dims.gene_sources, int(model["gene_width"])),
        context=context,
        context_width=context_width,
        state=state,
        prior=data["prior_run"] is not None and kind != "components",
        hidden=int(model["hidden"]),
    )
    if arm == "hybrid":
        return Hybrid(direct, film, int(model["hidden"]))
    return film


def direct_readouts(model: nn.Module) -> Sequence[DirectReadout]:
    """The model's direct-readout parts (to set their dropout rates)."""
    return [m for m in model.modules() if isinstance(m, DirectReadout)]

"""The gene-knowledge, cell-set model (SL protocol §18): a gene vector from frozen gene
sources plus a free deviation, a line as its mean token plus a set of cell tokens (one
from each cell's frozen embedding), a reading in which each gene weights the line's
cells in its own way, and a pair head on the gene-in-line embeddings.

    u_g      = sum over sources of P_s(k_gs) + r_g
    alpha_g  = softmax over the line's cells i of q(u_g) . t_i / sqrt(width)
    h_g(c)   = t_0 + sum_i alpha_gi t_i          (t_0: the mean token)
    V(g, c)  = u_g * h_g(c)                      z(g, c) = sum_k V_k(g, c)
    s(c, A, B) = head([V_A + V_B ; |V_A - V_B| ; V_A * V_B ;
                       u_A * u_B * sum_i w_i t_i^2 ; combinatorial])
    w_i      ~ alpha_Ai alpha_Bi, normalised over the cells
    e_g      = sum over gene-in-line sources n of E_n(x_g[n])
    l(c, A, B) = line_head([phi(c, A, B) ; e_A + e_B ; |e_A - e_B| ; e_A * e_B])

``phi`` is the input of the pair head (``features``). The line term ``l`` reads the
gene-in-line features ``x_g`` of the two genes in the line; the line head ends in a
zero layer, so the untrained line term is 0.

``z`` is the scaled GeneEffect stage one trains on; the pair head reads ``V`` and the
cell tokens, never ``z``. The cells' weights are normalised over the cells alone, so
neither their order nor their number changes the output, and a line without cells reads
``h = t_0``. The token projections end in a zero layer, so the untrained model predicts
``z = 0`` (the one centre of every gene)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.model.stage_one import VIABILITY_WIDTH, GeneEncoder

WIDTH = VIABILITY_WIDTH


def _zero_last(module: nn.Sequential) -> nn.Sequential:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


class GeneLineModel(nn.Module):
    """``sources`` maps a gene source to its vectors (genes x dim, zero where absent)
    and presence (genes,), on the model's gene axis; they are fixed inputs, kept as
    buffers outside the checkpoint."""

    def __init__(
        self,
        sources: Mapping[str, tuple[Tensor, Tensor]],
        expression_dim: int,
        cell_dim: int,
        hidden: int,
        head_hidden: int,
        gene_in_line: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.source_names = tuple(sources)
        for name, (vectors, present) in sources.items():
            self.register_buffer(f"vectors_{name}", vectors.float(), persistent=False)
            self.register_buffer(f"present_{name}", present.bool(), persistent=False)
        genes = len(next(iter(sources.values()))[0])
        self.gene_encoder = GeneEncoder(
            {name: vectors.shape[1] for name, (vectors, _) in sources.items()}, WIDTH
        )
        # r_g: zero at the start, pulled back to zero by weight decay
        self.deviation = nn.Parameter(torch.zeros(genes, WIDTH))
        self.mean_token = _zero_last(
            nn.Sequential(
                nn.Linear(expression_dim, hidden), nn.GELU(), nn.Linear(hidden, WIDTH)
            )
        )
        self.cell_token = _zero_last(
            nn.Sequential(nn.RMSNorm(cell_dim), nn.Linear(cell_dim, WIDTH))
        )
        self.query = nn.Linear(WIDTH, WIDTH, bias=False)
        self.head = nn.Sequential(
            nn.Linear(4 * WIDTH + 1, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1)
        )
        self.line_head = _zero_last(
            nn.Sequential(
                nn.Linear(7 * WIDTH + 1, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
        )
        self.in_line = nn.ModuleDict(
            {name: nn.Linear(width, WIDTH) for name, width in gene_in_line.items()}
        )

    def backbone(self) -> list[nn.Parameter]:
        """Every parameter but those of the pair head, the line head and the
        gene-in-line maps: what stage one trains."""
        heads = {id(p) for p in self.head.parameters()}
        heads |= {id(p) for p in self.line_parameters()}
        return [p for p in self.parameters() if id(p) not in heads]

    def line_parameters(self) -> list[nn.Parameter]:
        """The parameters of the line head and the gene-in-line maps."""
        return [*self.line_head.parameters(), *self.in_line.parameters()]

    def gene_vectors(self) -> Tensor:
        """``u`` for every gene, then one zero row: index -1 is a gene without a
        vector (an anchor that is a drug or outside the gene axis)."""
        vectors = {n: getattr(self, f"vectors_{n}") for n in self.source_names}
        present = {n: getattr(self, f"present_{n}") for n in self.source_names}
        u = self.gene_encoder(vectors, present) + self.deviation
        return torch.cat([u, u.new_zeros(1, WIDTH)])

    def tokens(
        self, expression: Tensor, cells: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """``expression`` (lines, genes) -> the mean tokens (lines, width); padded
        ``cells`` (lines, k, cell_dim), when given, -> the cell tokens (lines, k,
        width)."""
        mean = self.mean_token(expression)
        return mean, None if cells is None else self.cell_token(cells.float())

    def weights(self, u: Tensor, cells: Tensor, mask: Tensor) -> Tensor:
        """Each gene's weights over each line's cells, (lines, genes, k); zero for a
        line without cells."""
        logits = torch.einsum("gw,lkw->lgk", self.query(u), cells) / math.sqrt(WIDTH)
        some = mask.any(dim=1)
        safe = mask | ~some[:, None]
        alpha = logits.masked_fill(~safe[:, None, :], -torch.inf).softmax(dim=-1)
        return alpha * some[:, None, None].to(alpha.dtype)

    def embed(
        self, u: Tensor, mean: Tensor, cells: Tensor | None, mask: Tensor | None
    ) -> Tensor:
        """``V`` (lines, genes, width) of the genes ``u`` in the lines."""
        h = mean.unsqueeze(1)
        if cells is not None:
            h = h + self.weights(u, cells, mask) @ cells
        return u.unsqueeze(0) * h

    def dependency(
        self, u: Tensor, mean: Tensor, cells: Tensor | None, mask: Tensor | None
    ) -> Tensor:
        """``z`` (lines, genes)."""
        return self.embed(u, mean, cells, mask).sum(-1)

    def features(
        self,
        u_a: Tensor,
        u_b: Tensor,
        combinatorial: Tensor,
        mean: Tensor,
        cells: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """The pair features ``phi`` (rows, 4 * width + 1) of rows (A, B) of one
        line: ``u_a``, ``u_b`` (rows, width), ``combinatorial`` (rows,), the line's
        ``mean`` token (width,), its ``cells`` (k, width) and their ``mask`` (k,).
        The same for (A, B) and (B, A); a zero ``u_a`` leaves gene B's embedding
        alone; without cells the cell term is 0."""
        h_a = h_b = mean.unsqueeze(0)
        cell = torch.zeros_like(u_a)
        if cells is not None and bool(mask.any()):
            alpha_a = self.weights(u_a, cells[None], mask[None])[0]  # (rows, k)
            alpha_b = self.weights(u_b, cells[None], mask[None])[0]
            h_a, h_b = h_a + alpha_a @ cells, h_b + alpha_b @ cells
            # alpha_A alpha_B normalised over the cells: the query is linear, so it
            # is the softmax of the summed logits (the product itself can underflow
            # to 0 / 0 when the two genes' weights sit on different cells)
            both = self.weights(u_a + u_b, cells[None], mask[None])[0]
            cell = u_a * u_b * (both @ cells.square())
        v_a, v_b = u_a * h_a, u_b * h_b
        phi = torch.cat(
            [v_a + v_b, (v_a - v_b).abs(), v_a * v_b, cell,
             combinatorial.unsqueeze(-1).to(v_a.dtype)],
            dim=-1,
        )  # fmt: skip
        return phi

    def pair(
        self,
        u_a: Tensor,
        u_b: Tensor,
        combinatorial: Tensor,
        mean: Tensor,
        cells: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """SL logits (rows,) of rows (A, B) of one line: the pair head on
        ``features``."""
        phi = self.features(u_a, u_b, combinatorial, mean, cells, mask)
        return self.head(phi).squeeze(-1)

    def line_term(
        self,
        u_a: Tensor,
        u_b: Tensor,
        combinatorial: Tensor,
        in_line_a: Mapping[str, Tensor],
        in_line_b: Mapping[str, Tensor],
        mean: Tensor,
        cells: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """The line term (rows,) of rows (A, B) of one line. ``in_line_a[name]``
        (rows, width of ``name``) holds the gene-in-line features of gene A; the
        other arguments are those of ``features``. The same for (A, B) and (B, A)."""
        e_a, e_b = torch.zeros_like(u_a), torch.zeros_like(u_b)
        for name, project in self.in_line.items():
            e_a = e_a + project(in_line_a[name])
            e_b = e_b + project(in_line_b[name])
        phi = self.features(u_a, u_b, combinatorial, mean, cells, mask)
        psi = torch.cat([phi, e_a + e_b, (e_a - e_b).abs(), e_a * e_b], dim=-1)
        return self.line_head(psi).squeeze(-1)

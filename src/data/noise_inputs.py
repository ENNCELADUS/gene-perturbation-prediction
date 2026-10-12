"""Per-line inputs of the noise diagnosis, computed from a line's raw basal cells.

- **Cell draws.** A line's cells are ranked by ``sha256(ModelID|cell)`` (the order
  of :func:`src.data.prepared.select_context_cells`); half A takes the even ranks,
  half B the odd ones, and the ``n``-cell draw the first ``n``, so draws nest. A
  draw needing at least as many cells as the line has is not made.
- **Program scores.** Per cell, the mean log-normalised expression
  (``log1p(x * T / L_cell)``, the prepared space) over a program's member genes,
  minus the cell's mean over every gene of the scoring set. The scoring set is the
  pseudo-bulk's genes (DepMap's protein-coding symbols), one set for every line;
  a member outside it is not part of the program here. A gene the source does not
  list counts as an unexpressed gene: some builders drop each line's zero-count
  genes, and scoring over each source's own list would move identical cells by
  build policy. The share of the scoring set and of each program the source lists
  is reported beside the scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

from src.data.basal import align_columns
from src.data.prepared import select_context_cells


def cell_draws(
    model_id: str, cell_ids: Sequence[object], cells: Sequence[int]
) -> dict[str, np.ndarray]:
    """Row positions of each draw: ``half_a`` and ``half_b`` (a line of two or
    more cells), then ``cells_<n>`` for every ``n`` below the line's cell count."""
    ranking = select_context_cells(model_id, cell_ids, len(cell_ids))
    draws = {}
    if len(ranking) >= 2:
        draws.update(half_a=ranking[0::2], half_b=ranking[1::2])
    for n in sorted(set(cells)):
        if n < len(ranking):
            draws[f"cells_{n}"] = ranking[:n]
    return draws


def program_weights(
    genes: Sequence[str], members: Mapping[str, Sequence[str]]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Genes x programs averaging weights over each program's members in
    ``genes`` (a program with none averages nothing), and each program's member
    positions. Symbols match case-insensitively."""
    position = {str(gene).upper(): i for i, gene in enumerate(genes)}
    weights = np.zeros((len(genes), len(members)))
    inside = []
    for k, program in enumerate(members.values()):
        rows = np.asarray(
            sorted({position[g.upper()] for g in program if g.upper() in position}),
            dtype=np.int64,
        )
        inside.append(rows)
        if rows.size:
            weights[rows, k] = 1.0 / rows.size
    return weights, inside


def program_scores(
    counts: object,
    symbols: Sequence[str],
    genes: Sequence[str],
    members: Mapping[str, Sequence[str]],
    target_sum: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Cells x programs scores of one line over the scoring set ``genes``, and the
    share of the scoring set (``"scoring_set"``) and of each program's members in
    it that the source lists.

    ``counts`` holds raw UMI over every gene of the source; library sizes are
    taken over all of them.
    """
    matrix = sparse.csr_matrix(counts)
    library = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    upper = [str(gene).upper() for gene in genes]
    aligned, listed = align_columns(
        matrix, [str(s).strip().upper() for s in symbols], upper
    )
    weights, inside = program_weights(upper, members)
    shares = {"scoring_set": float(listed.mean())}
    for name, rows in zip(members, inside, strict=True):
        shares[name] = float(listed[rows].mean()) if rows.size else 0.0
    scale = np.zeros_like(library)
    scale[library > 0] = target_sum / library[library > 0]
    logged = sparse.csr_matrix(sparse.diags(scale) @ aligned)
    logged.data = np.log1p(logged.data)
    background = np.asarray(logged.sum(axis=1)).ravel() / len(upper)
    scores = np.asarray(logged @ weights) - background[:, None]
    return scores.astype(np.float32), shares

"""Per-line inputs of the noise diagnosis, computed from a line's raw basal cells.

- **Cell draws.** A line's cells are ranked by ``sha256(ModelID|cell)`` (the order
  of :func:`src.data.prepared.select_context_cells`); half A takes the even ranks,
  half B the odd ones, and the ``n``-cell draw the first ``n``, so draws nest. A
  draw needing at least as many cells as the line has is not made.
- **Program scores.** Per cell, the mean log-normalised expression
  (``log1p(x * T / L_cell)``, the prepared space) over a program's member genes
  that the source measures, minus the cell's mean over every gene of the scoring
  set it measures (an unexpressed measured gene counts as zero). The scoring set is
  the pseudo-bulk's genes (DepMap's protein-coding symbols), one set for every
  line; a member outside it is not part of the program here, and a program's
  coverage in a line is the share of its in-set members the line's source
  measures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

from src.data.basal import align_columns
from src.data.prepared import select_context_cells

#: A program measured at fewer of its in-set members than this, in any line, stops
#: preparation: its score would describe a different gene set in that line.
MIN_PROGRAM_COVERAGE = 0.5


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
    genes: Sequence[str], available: np.ndarray, members: Mapping[str, Sequence[str]]
) -> tuple[np.ndarray, dict[str, float]]:
    """Genes x programs averaging weights over each program's measured members,
    and each program's coverage (measured members over its members in
    ``genes``; 0 when it has none). Symbols match case-insensitively."""
    position = {str(gene).upper(): i for i, gene in enumerate(genes)}
    weights = np.zeros((len(genes), len(members)))
    coverage = {}
    for k, (name, program) in enumerate(members.items()):
        inside = [position[g.upper()] for g in program if g.upper() in position]
        measured = [i for i in inside if available[i]]
        coverage[name] = len(measured) / len(inside) if inside else 0.0
        if measured:
            weights[measured, k] = 1.0 / len(measured)
    return weights, coverage


def program_scores(
    counts: object,
    symbols: Sequence[str],
    genes: Sequence[str],
    members: Mapping[str, Sequence[str]],
    target_sum: float,
    *,
    model_id: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Cells x programs scores of one line over the scoring set ``genes``, and
    each program's coverage.

    ``counts`` holds raw UMI over every gene of the source; library sizes are
    taken over all of them. Raises when a program's coverage is below
    ``MIN_PROGRAM_COVERAGE``, naming the line and the program.
    """
    matrix = sparse.csr_matrix(counts)
    library = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    upper = [str(gene).upper() for gene in genes]
    aligned, available = align_columns(
        matrix, [str(s).strip().upper() for s in symbols], upper
    )
    weights, coverage = program_weights(upper, available, members)
    thin = {
        name: value for name, value in coverage.items() if value < MIN_PROGRAM_COVERAGE
    }
    if thin:
        raise ValueError(
            f"{model_id}: programs measured at under {MIN_PROGRAM_COVERAGE:.0%} of "
            f"their members: {thin}"
        )
    scale = np.zeros_like(library)
    scale[library > 0] = target_sum / library[library > 0]
    logged = sparse.csr_matrix(sparse.diags(scale) @ aligned)
    logged.data = np.log1p(logged.data)
    background = np.asarray(logged.sum(axis=1)).ravel() / int(available.sum())
    scores = np.asarray(logged @ weights) - background[:, None]
    return scores.astype(np.float32), coverage

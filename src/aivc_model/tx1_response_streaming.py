"""Memory-bounded assembly of a per-gene cell reservoir into a CSR matrix.

Wave 2 Phase C, fix-round-3, Fix 1 -- see
``.superpowers/sdd/phase-c/progress.md``'s 2026-07-26 entry for the incident
this module fixes: both Phase C training arms were killed after climbing to
~621-625 GB RSS each, stuck at "X-Atlas-Orion response cell filtering" for
HCT116 (18,293 perturbed genes; a 44 GB source expanded ~14x in RAM).

**What did NOT cause the incident and is not touched here:** the per-gene
cell cap (``max_cells_per_gene``) already works exactly as implemented --
each gene's reservoir really is bounded to at most that many cells. The
incident was the multiplication the cap's own docstring never spelled out:
18,293 genes x 256 cells/gene = 4.68 million cells, ALL of them necessarily
resident at once (a gene's reservoir cannot be closed out early -- its cells
are scattered across shards in an order this module does not control) --
matching the observed RSS almost exactly at ~130 KB/cell. That per-gene x
gene-count product is bounded only by an explicit total-cell budget (see
``total_cells``/``seed`` below), which this module exposes as an optional
knob but does not default to any value -- choosing that value is a reserved
human decision (fix-round-3 brief), pending the peak-RSS measurement
:mod:`aivc_model.tx1_response_data` now logs per line (Fix 3).

**What this module DOES fix:** the old pipeline (formerly
``tx1_basal._stream_xatlas_response_cells`` + ``_assemble_token_matrix``)
flattened the finalized reservoir into a fresh list, then built
``rows``/``columns``/``values`` as three plain Python lists of BOXED
ints/floats -- one Python object per nonzero matrix entry -- before finally
calling ``scipy.sparse.csr_matrix`` once. That pays for the reservoir's real
data a SECOND time, at several times a raw numpy array's per-element
overhead (see this module's own peak-memory test). :func:`
drain_gene_reservoirs_to_matrix` instead computes the CSR component arrays
with vectorized numpy operations, and DRAINS the caller-owned reservoir as
it consumes it -- nulling each processed slot so its ``(genes, values)``
arrays are garbage-collected immediately rather than staying referenced
until the whole function returns -- so peak memory during assembly is
proportional to the finalized reservoir/output matrix, not to the assembly
method's own intermediate representation.

Deliberately generic over the caller's per-cell wrapper type: any 4-tuple-
or 4-field-NamedTuple-like object that unpacks to ``(genes, values, barcode,
sample)`` works (``tx1_basal._XatlasCell`` unpacks this way already), so
this module has no import-time dependency on ``tx1_basal``.
"""

from __future__ import annotations

import logging
from typing import Mapping, MutableSequence, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

_LOGGER = logging.getLogger(__name__)


def resolve_total_budget_keep_mask(
    n_cells: int, total_cells: int | None, seed: int
) -> np.ndarray:
    """Boolean keep-mask over ``n_cells`` gene-sorted, slot-ordered rows.

    ``total_cells is None`` (the default everywhere in this repo today) or
    ``total_cells >= n_cells`` keeps every row -- i.e. this is a no-op unless
    a caller opts in, so existing behavior is reproduced exactly (C1) when
    the reserved total-cell-budget knob is left unset.

    Args:
        n_cells: Total cells across every gene's reservoir, before any
            total-budget trim.
        total_cells: Optional cap on the combined cell count. ``None``
            leaves every cell.
        seed: Seed for the deterministic downsample when trimming.

    Returns:
        A length-``n_cells`` boolean array, ``True`` for kept rows.
    """
    if total_cells is None or total_cells >= n_cells:
        return np.ones(n_cells, dtype=bool)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_cells, size=total_cells, replace=False)
    mask = np.zeros(n_cells, dtype=bool)
    mask[chosen] = True
    return mask


def drain_gene_reservoirs_to_matrix(
    reservoirs: Mapping[str, MutableSequence[Sequence[object] | None]],
    metadata: pd.DataFrame,
    *,
    metadata_var_columns: Sequence[str],
    total_cells: int | None = None,
    seed: int = 0,
) -> tuple[csr_matrix, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build a CSR matrix from a finalized per-gene cell reservoir, draining
    it as it is consumed.

    Args:
        reservoirs: ``{gene: [cell, ...]}``, where each ``cell`` unpacks to
            ``(genes, values, barcode, sample)`` -- ``genes``/``values`` are
            parallel 1-D numpy arrays for one cell, already filtered to
            positive-value entries by the caller. **Mutated in place**:
            every slot is set to ``None`` once consumed, regardless of
            ``total_cells`` (an unselected cell's memory is freed exactly
            like a selected one's).
        metadata: Token-indexed metadata (``token_id`` -> Ensembl id/name
            columns), e.g. the X-Atlas-Orion ``gene_metadata.parquet``.
        metadata_var_columns: The ``metadata`` columns to carry into the
            returned ``var``; the first must be the Ensembl id column,
            which becomes ``var.index``.
        total_cells: Optional total-cell budget across every gene combined
            (fix-round-3, Fix 1's reserved knob -- see the module
            docstring). ``None`` keeps every reservoir-selected cell, i.e.
            today's behavior.
        seed: Seed for the total-budget downsample; unused when
            ``total_cells`` is ``None``.

    Returns:
        ``(matrix, var, perturbation_genes, barcodes, samples)``, every
        array ordered exactly as ``sorted(reservoirs)`` then each gene's own
        slot order ``0..len(bucket)-1`` -- the same final cell order the
        pre-fix implementation always produced.
    """
    genes_order = sorted(reservoirs)
    n_cells_before_budget = sum(len(reservoirs[gene]) for gene in genes_order)
    keep_mask = resolve_total_budget_keep_mask(n_cells_before_budget, total_cells, seed)

    lengths: list[int] = []
    perturbation_genes: list[str] = []
    barcodes: list[str] = []
    samples: list[str] = []
    valid_tokens: set[int] = set()
    global_index = 0
    for gene in genes_order:
        for cell in reservoirs[gene]:
            if keep_mask[global_index]:
                cell_genes = cell[0]
                perturbation_genes.append(gene)
                barcodes.append(cell[2])
                samples.append(cell[3])
                lengths.append(len(cell_genes))
                valid_tokens.update(int(token) for token in cell_genes.tolist())
            global_index += 1

    n_cells = len(lengths)
    tokens = sorted(token for token in valid_tokens if token in metadata.index)
    n_dropped = len(valid_tokens) - len(tokens)
    if n_dropped:
        _LOGGER.warning(
            "dropped %d/%d (%.1f%%) otherwise-valid gene tokens missing from "
            "gene metadata index",
            n_dropped,
            len(valid_tokens),
            100.0 * n_dropped / len(valid_tokens),
        )
    token_lookup = np.asarray(tokens, dtype=np.int64)
    total_nnz = int(np.asarray(lengths, dtype=np.int64).sum()) if lengths else 0
    out_rows = np.empty(total_nnz, dtype=np.int64)
    out_cols = np.empty(total_nnz, dtype=np.int64)
    out_values = np.empty(total_nnz, dtype=np.float32)

    offset = 0
    row_index = 0
    global_index = 0
    for gene in genes_order:
        bucket = reservoirs[gene]
        for slot in range(len(bucket)):
            cell = bucket[slot]
            # Drain unconditionally -- an unselected cell's memory is freed
            # exactly like a selected one's, since keep_mask only decides
            # what feeds the OUTPUT, not what stays resident.
            bucket[slot] = None
            keep = keep_mask[global_index]
            global_index += 1
            if not keep:
                continue
            cell_genes, cell_values = cell[0], cell[1]
            if token_lookup.size:
                positions = np.searchsorted(token_lookup, cell_genes)
                positions_clipped = np.clip(positions, 0, len(token_lookup) - 1)
                valid_entries = token_lookup[positions_clipped] == cell_genes
            else:
                valid_entries = np.zeros(len(cell_genes), dtype=bool)
            n_valid = int(valid_entries.sum())
            if n_valid:
                out_rows[offset : offset + n_valid] = row_index
                out_cols[offset : offset + n_valid] = positions_clipped[valid_entries]
                out_values[offset : offset + n_valid] = cell_values[valid_entries]
            offset += n_valid
            row_index += 1

    matrix = csr_matrix(
        (out_values[:offset], (out_rows[:offset], out_cols[:offset])),
        shape=(n_cells, len(tokens)),
    )
    var = metadata.loc[tokens, list(metadata_var_columns)].copy()
    var.index = var[metadata_var_columns[0]].astype(str)
    return (
        matrix,
        var,
        np.asarray(perturbation_genes, dtype=object),
        np.asarray(barcodes, dtype=object),
        np.asarray(samples, dtype=object),
    )


__all__ = [
    "drain_gene_reservoirs_to_matrix",
    "resolve_total_budget_keep_mask",
]

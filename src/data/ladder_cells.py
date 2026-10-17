"""Per-line cell inputs of the stage-one ladder, from a line's raw basal cells.

Every quantity reads all of a line's cells. Source symbols are matched to a gene
list upper-cased (DepMap's symbols are upper-case; a source that spells
``C1orf112`` lists ``C1ORF112``); columns sharing a symbol are summed. Library
sizes are each cell's UMI total over every gene of its source.

- **PIE's input** is PIE's control mean: each cell normalised to 10^4 UMI
  (CP10K), ``log1p``, then the mean over cells (``pie.process.normalize`` and
  ``pie.prep.labels.pseudobulk_h5ad``).
- **Pseudo-bulk** is ``log1p`` of the mean over cells of each cell's CP10K: PIE's
  per-cell normalisation, then the log of the mean (not a mean of logs, which
  moves with a gene's variance and dropout).
- **Per-gene spread** is each gene's detected fraction and SD of ``log1p`` CP10K
  over cells, beside what depth alone predicts: detection
  ``mean_i(1 - exp(-mu_g L_i / 10^4))`` with ``mu_g`` the gene's mean CP10K, and
  the SD of one Poisson replicate of the line's cells at ``lambda_ig = mu_g L_i /
  10^4``.
- **Per-cell QC**: log UMI, genes detected, mitochondrial (``MT-``) and ribosomal
  (``RPS``/``RPL``) fractions.
- **HVG cells**: STATE's HVGs in the prepared space ``log1p(x T / L)``; the
  shuffled cells permute each HVG's values across the line's cells
  independently, keeping every gene's within-line distribution and destroying
  co-expression within cells. The mean as one cell is ``log1p`` of the line's
  mean normalised HVG profile.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from src.data.basal import align_columns

#: PIE's per-cell target sum (CP10K).
PIE_TARGET_SUM = 1e4
QC_COLUMNS = ("log_umi", "genes_detected", "mito_fraction", "ribo_fraction")
_RIBOSOMAL = re.compile(r"^RP[SL]\d+[A-Z]?\d*$")
#: Genes per chunk of the dense Poisson replicate.
_POISSON_CHUNK = 512


def line_rng(model_id: str, seed: int, purpose: str) -> np.random.Generator:
    """A generator fixed by the seed, the line and what it draws."""
    digest = hashlib.sha256(f"{seed}|{model_id}|{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def upper_symbols(symbols: Sequence[str]) -> list[str]:
    return [str(symbol).strip().upper() for symbol in symbols]


def library_sizes(counts: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()


def _scaled(counts: sparse.csr_matrix, library: np.ndarray, target_sum: float):
    """``x * target_sum / L`` per cell; cells without UMI stay zero."""
    scale = np.zeros_like(library)
    scale[library > 0] = target_sum / library[library > 0]
    return sparse.csr_matrix(sparse.diags(scale) @ counts)


@dataclass(frozen=True)
class GeneSummaries:
    """Per gene of a list: PIE's input, pseudo-bulk and spread; NaN where the
    line's source does not list the gene (``listed`` False)."""

    pie_input: np.ndarray
    pseudobulk: np.ndarray
    detected: np.ndarray
    sd: np.ndarray
    detected_expected: np.ndarray
    sd_expected: np.ndarray
    listed: np.ndarray


def _log_moments(scaled: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD over cells of ``log1p`` of a scaled matrix."""
    logged = scaled.copy()
    logged.data = np.log1p(logged.data)
    n_cells = logged.shape[0]
    mean = np.asarray(logged.sum(axis=0)).ravel() / n_cells
    square = np.asarray(logged.multiply(logged).sum(axis=0)).ravel() / n_cells
    return mean, np.sqrt(np.maximum(square - mean**2, 0.0))


def gene_summaries(
    counts: object,
    symbols: Sequence[str],
    genes: Sequence[str],
    *,
    rng: np.random.Generator,
) -> GeneSummaries:
    """PIE's input, pseudo-bulk and spread of one line over ``genes``.

    ``counts`` holds raw UMI over every gene of the source (``symbols``
    columns); ``genes`` are upper-case symbols.
    """
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    library = library_sizes(matrix)
    aligned, listed = align_columns(matrix, upper_symbols(symbols), list(genes))
    aligned.eliminate_zeros()
    n_cells = aligned.shape[0]
    scaled = _scaled(aligned, library, PIE_TARGET_SUM)
    rate = np.asarray(scaled.sum(axis=0)).ravel() / n_cells
    pie_input, sd = _log_moments(scaled)
    detected = np.diff(aligned.tocsc().indptr) / n_cells
    depth = library / PIE_TARGET_SUM
    detected_expected = np.empty(len(genes))
    sd_expected = np.empty(len(genes))
    for start in range(0, len(genes), _POISSON_CHUNK):
        stop = min(start + _POISSON_CHUNK, len(genes))
        expected = np.outer(depth, rate[start:stop])
        detected_expected[start:stop] = 1.0 - np.exp(-expected).mean(axis=0)
        replicate = sparse.csr_matrix(rng.poisson(expected), dtype=np.float64)
        _, sd_expected[start:stop] = _log_moments(
            _scaled(replicate, library, PIE_TARGET_SUM)
        )
    values = [
        pie_input,
        np.log1p(rate),
        detected,
        sd,
        detected_expected,
        sd_expected,
    ]
    for value in values:
        value[~listed] = np.nan
    return GeneSummaries(*(v.astype(np.float32) for v in values), listed=listed)


def cell_qc(counts: object, symbols: Sequence[str]) -> np.ndarray:
    """Cells x ``QC_COLUMNS``: log UMI, genes detected, mitochondrial and
    ribosomal fractions, over every gene of the source."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    matrix.eliminate_zeros()
    library = library_sizes(matrix)
    upper = upper_symbols(symbols)
    mito = np.asarray([s.startswith("MT-") for s in upper])
    ribo = np.asarray([_RIBOSOMAL.fullmatch(s) is not None for s in upper])
    safe = np.where(library > 0, library, 1.0)
    return np.column_stack(
        [
            np.log(np.maximum(library, 1.0)),
            np.diff(matrix.indptr).astype(np.float64),
            np.asarray(matrix[:, mito].sum(axis=1)).ravel() / safe,
            np.asarray(matrix[:, ribo].sum(axis=1)).ravel() / safe,
        ]
    ).astype(np.float32)


def shuffle_within_columns(matrix: object, rng: np.random.Generator) -> np.ndarray:
    """Each column's values permuted across rows independently (dense)."""
    dense = matrix.toarray() if sparse.issparse(matrix) else np.array(matrix)
    out = np.empty_like(dense)
    for j in range(dense.shape[1]):
        out[:, j] = dense[rng.permutation(dense.shape[0]), j]
    return out


def shuffle_counts(counts: object, rng: np.random.Generator) -> sparse.csr_matrix:
    """Raw counts with each gene's values permuted across cells independently;
    sparse in, sparse out (a gene's nonzero entries move to random cells)."""
    csc = sparse.csc_matrix(counts)
    n_cells = csc.shape[0]
    indices = np.empty_like(csc.indices)
    for j in range(csc.shape[1]):
        start, stop = csc.indptr[j], csc.indptr[j + 1]
        if stop > start:
            indices[start:stop] = rng.choice(n_cells, stop - start, replace=False)
    shuffled = sparse.csc_matrix(
        (csc.data.copy(), indices, csc.indptr.copy()), shape=csc.shape
    )
    shuffled.sort_indices()
    return shuffled.tocsr()


@dataclass(frozen=True)
class HvgCells:
    """A line's cells over STATE's HVGs in the prepared space, their shuffle and
    the mean as one cell; ``listed`` marks the HVGs the source lists."""

    cells: np.ndarray
    shuffled: np.ndarray
    mean_cell: np.ndarray
    listed: np.ndarray


def hvg_cells(
    counts: object,
    symbols: Sequence[str],
    hvg_order: Sequence[str],
    target_sum: float,
    *,
    rng: np.random.Generator,
) -> HvgCells:
    """``log1p(x T / L)`` over the HVGs (upper-case symbols), every cell."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    library = library_sizes(matrix)
    aligned, listed = align_columns(
        matrix, upper_symbols(symbols), upper_symbols(hvg_order)
    )
    scaled = _scaled(aligned, library, target_sum)
    cells = np.log1p(scaled.toarray()).astype(np.float32)
    mean_cell = np.log1p(np.asarray(scaled.mean(axis=0)).ravel()).astype(np.float32)
    return HvgCells(
        cells=cells,
        shuffled=shuffle_within_columns(cells, rng),
        mean_cell=mean_cell,
        listed=listed,
    )


def bulk_to_pie_input(log2_tpm: np.ndarray) -> np.ndarray:
    """DepMap ``log2(TPM + 1)`` as PIE's input scale: ``log1p(TPM / 100)``, the
    CP10K of a profile that sums to 10^6 over its genes."""
    tpm = np.exp2(np.asarray(log2_tpm, dtype=np.float64)) - 1.0
    return np.log1p(np.maximum(tpm, 0.0) / 100.0).astype(np.float32)


def symbol_key(genes: Sequence[str], keys: Sequence[str]) -> dict[str, str]:
    """Each upper-case gene's key in a source whose keys are spelled any case;
    a gene matching no key, or two keys, has none."""
    by_upper: dict[str, list[str]] = {}
    for key in keys:
        by_upper.setdefault(str(key).upper(), []).append(str(key))
    return {g: by_upper[g][0] for g in genes if len(by_upper.get(g, ())) == 1}


__all__ = [
    "PIE_TARGET_SUM",
    "QC_COLUMNS",
    "GeneSummaries",
    "HvgCells",
    "bulk_to_pie_input",
    "cell_qc",
    "gene_summaries",
    "hvg_cells",
    "line_rng",
    "shuffle_counts",
    "shuffle_within_columns",
    "symbol_key",
    "upper_symbols",
]

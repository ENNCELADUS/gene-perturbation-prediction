"""Per-line cell inputs of the stage-one ladder (PIE's input, pseudo-bulk, per-gene
spread, per-cell QC, HVG cells), each over every raw basal cell of the line. Pseudo-bulk
is ``log1p`` of the mean CP10K, not a mean of logs, which moves with a gene's variance
and dropout."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from src.data.basal import align_columns
from src.data.expression import library_sizes

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


def scale_to_total(counts: sparse.csr_matrix, library: np.ndarray, target_sum: float):
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
    scaled = scale_to_total(aligned, library, PIE_TARGET_SUM)
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
            scale_to_total(replicate, library, PIE_TARGET_SUM)
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


#: Columns of ``codetection``, per pair of genes (A, B) of one line.
CODETECTION_COLUMNS = (
    "codetected",
    "expected",
    "expected_matched",
    "detected_a",
    "detected_b",
)
#: Cells, and pairs, per block of the co-detection sums.
_CELL_CHUNK = 2048
_PAIR_CHUNK = 8192


def matched_rates(depth: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Per gene, the rate ``r`` with ``mean_i(1 - exp(-r * depth_i)) = detected``:
    the Poisson rate that gives the gene its observed detection share at each
    cell's depth. Newton's method from 0: the share is concave and increasing in
    ``r``, so each step stays below the root."""
    values, weight = np.unique(depth, return_counts=True)
    weight = weight / weight.sum()
    reach = float(weight[values > 0].sum())  # a cell without UMI detects nothing
    target = np.minimum(np.asarray(detected, dtype=np.float64), reach * (1 - 1e-6))
    rate = np.zeros(len(target))
    for start in range(0, len(target), _POISSON_CHUNK):
        part = slice(start, start + _POISSON_CHUNK)
        r = rate[part]
        for _ in range(100):
            miss = np.exp(-np.outer(values, r))
            gap = target[part] - (1.0 - miss).T @ weight
            r = r + gap / ((values[:, None] * miss).T @ weight)
            if (gap <= 1e-9 * np.maximum(target[part], 1e-12)).all():
                break
        rate[part] = r
    return rate


def codetection(
    counts: object,
    symbols: Sequence[str],
    gene_a: Sequence[str],
    gene_b: Sequence[str],
) -> np.ndarray:
    """Pairs x ``CODETECTION_COLUMNS`` over every cell of one line: the share of
    cells that detect both genes (count > 0); the share expected under independence
    at each cell's depth, ``mean_i (1 - exp(-mu_A L_i / T)) (1 - exp(-mu_B L_i / T))``
    with ``mu`` the line's mean CP10K (``T`` = 10^4, as ``detected_expected``) and
    again with each gene's rate matched to its detection share (``matched_rates``);
    the two single-gene detection shares. NaN where the source does not list a
    gene (a drug anchor included)."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    library = library_sizes(matrix)
    genes = sorted(set(upper_symbols(gene_a)) | set(upper_symbols(gene_b)))
    aligned, listed = align_columns(matrix, upper_symbols(symbols), genes)
    aligned.eliminate_zeros()
    n_cells = aligned.shape[0]
    rate = np.asarray(scale_to_total(aligned, library, PIE_TARGET_SUM).sum(axis=0))
    rate = rate.ravel() / n_cells
    detected = np.diff(aligned.tocsc().indptr) / n_cells
    depth = library / PIE_TARGET_SUM
    matched = matched_rates(depth, detected)
    position = {g: i for i, g in enumerate(genes)}
    ia = np.array([position[g] for g in upper_symbols(gene_a)], dtype=np.int64)
    ib = np.array([position[g] for g in upper_symbols(gene_b)], dtype=np.int64)
    sums = np.zeros((len(ia), 3))
    for start in range(0, n_cells, _CELL_CHUNK):
        cells = slice(start, min(start + _CELL_CHUNK, n_cells))
        hit = (aligned[cells].toarray() > 0).astype(np.float32)
        chunk_depth = depth[cells]
        expected = (1.0 - np.exp(-np.outer(chunk_depth, rate))).astype(np.float32)
        fitted = (1.0 - np.exp(-np.outer(chunk_depth, matched))).astype(np.float32)
        for first in range(0, len(ia), _PAIR_CHUNK):
            pairs = slice(first, first + _PAIR_CHUNK)
            a, b = ia[pairs], ib[pairs]
            for k, values in enumerate((hit, expected, fitted)):
                sums[pairs, k] += np.einsum("ij,ij->j", values[:, a], values[:, b])
    out = np.column_stack([sums / n_cells, detected[ia], detected[ib]])
    out[~(listed[ia] & listed[ib])] = np.nan
    return out


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
    scaled = scale_to_total(aligned, library, target_sum)
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

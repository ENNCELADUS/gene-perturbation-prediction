"""Tests for src/aivc_model/tx1_response_streaming.py (fix-round-3, Fix 1).

Two things must be proven here, not just asserted:

1. **Correctness is non-negotiable.** The new streaming/draining assembly
   must produce EXACTLY the same AnnData (cells, order, values) as the
   pre-fix implementation, for a fixed seed. ``test_matches_legacy_*`` below
   runs a frozen, verbatim copy of the pre-fix pipeline
   (``_legacy_build_xatlas_orion_response_adata``, built only from
   still-unchanged private helpers -- ``_assemble_token_matrix`` itself was
   never touched by this fix) side by side with the real, current
   ``build_xatlas_orion_response_adata`` on the same fixture and compares
   every observable field exactly.
2. **The memory reduction is real, not assumed.** ``test_peak_rss_*`` runs
   each pipeline in its OWN freshly spawned process (so neither
   measurement's allocator state leaks into the other) on a fixture large
   enough that the pre-fix implementation's Python-boxed
   ``rows``/``columns``/``values`` triple-list construction dominates, and
   asserts the new pipeline's peak RSS is materially lower -- a concrete
   numeric bound, not "it runs".
"""

from __future__ import annotations

import multiprocessing
import resource
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from aivc_model.tx1_basal import (
    _XATLAS_CONTROL_LABEL,
    _XATLAS_GENE_METADATA_TOKEN_COL,
    _XATLAS_GENE_METADATA_VAR_COLUMNS,
    _XATLAS_PASS_GUIDE_FILTER_COL,
    _XATLAS_PASS_GUIDE_FILTER_VALUE,
    _XATLAS_PERTURBATION_COL,
    _XATLAS_READ_COLUMNS,
    _assemble_token_matrix,
    _row_to_xatlas_cell,
    assert_tx1_input_contract,
    build_xatlas_orion_response_adata,
)
from aivc_model.tx1_response_streaming import (
    drain_gene_reservoirs_to_matrix,
    resolve_total_budget_keep_mask,
)


# --- Frozen reference: the pre-fix-round-3 pipeline, verbatim -------------
#
# NOT production code. Kept only so the identity test below has a concrete,
# independently-executable ground truth to compare against, without relying
# on git history or a hand-maintained "expected" fixture that could silently
# drift out of sync with what the old code actually did. Built entirely from
# private helpers this fix did NOT change (`_row_to_xatlas_cell`,
# `_assemble_token_matrix`, the `_XATLAS_*` schema constants) -- the only
# thing reproduced here is the shape this fix REPLACED: streaming into one
# reservoir dict, then flattening it into fresh lists, then handing those to
# `_assemble_token_matrix`'s Python-boxed-list matrix construction.


def _legacy_stream_xatlas_response_cells(
    shard_dir: Path,
    shard_glob: str,
    *,
    control_label: str,
    pass_guide_filter_value: int,
    genes,
    max_cells_per_gene: int | None,
    seed: int,
):
    paths = sorted(Path(shard_dir).glob(shard_glob))
    allowed = {str(gene) for gene in genes} if genes is not None else None
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list] = {}
    seen: dict[str, int] = {}
    for path in paths:
        frame = pd.read_parquet(path, columns=list(_XATLAS_READ_COLUMNS))
        frame = frame[frame[_XATLAS_PERTURBATION_COL].astype(str) != control_label]
        if allowed is not None:
            frame = frame[frame[_XATLAS_PERTURBATION_COL].astype(str).isin(allowed)]
        frame = frame[
            frame[_XATLAS_PASS_GUIDE_FILTER_COL].astype(int) == pass_guide_filter_value
        ]
        for row in frame.itertuples(index=False):
            gene = str(getattr(row, _XATLAS_PERTURBATION_COL))
            cell = _row_to_xatlas_cell(row)
            bucket = reservoirs.setdefault(gene, [])
            seen[gene] = seen.get(gene, 0) + 1
            gene_seen = seen[gene]
            if max_cells_per_gene is None or len(bucket) < max_cells_per_gene:
                bucket.append(cell)
                continue
            replacement = int(rng.integers(0, gene_seen))
            if replacement < max_cells_per_gene:
                bucket[replacement] = cell
    cells: list = []
    perturbation_genes: list[str] = []
    for gene in sorted(reservoirs):
        for cell in reservoirs[gene]:
            cells.append(cell)
            perturbation_genes.append(gene)
    if not cells:
        raise ValueError("no perturbed cells found")
    return cells, np.asarray(perturbation_genes, dtype=object)


def _legacy_build_xatlas_orion_response_adata(
    shard_dir: Path,
    gene_metadata_path: Path,
    *,
    cell_line_name: str,
    model_id: str,
    cellosaurus_id: str,
    shard_glob: str = "*.parquet",
    control_label: str = _XATLAS_CONTROL_LABEL,
    pass_guide_filter_value: int = _XATLAS_PASS_GUIDE_FILTER_VALUE,
    genes=None,
    max_cells_per_gene: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    cells, perturbation_genes = _legacy_stream_xatlas_response_cells(
        shard_dir,
        shard_glob,
        control_label=control_label,
        pass_guide_filter_value=pass_guide_filter_value,
        genes=genes,
        max_cells_per_gene=max_cells_per_gene,
        seed=seed,
    )
    metadata = pd.read_parquet(gene_metadata_path).set_index(
        _XATLAS_GENE_METADATA_TOKEN_COL
    )
    matrix, var = _assemble_token_matrix(
        [(cell.genes, cell.values) for cell in cells],
        metadata,
        metadata_var_columns=_XATLAS_GENE_METADATA_VAR_COLUMNS,
    )
    n_cells = matrix.shape[0]
    obs = pd.DataFrame(
        {
            "cell_type": [cell_line_name] * n_cells,
            "cellosaurus_id": [cellosaurus_id] * n_cells,
            "model_id": [model_id] * n_cells,
            "perturbation_gene": perturbation_genes,
            "sample": [cell.sample for cell in cells],
        },
        index=[f"{cell.sample}:{cell.cell_barcode}" for cell in cells],
    )
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    assert_tx1_input_contract(adata)
    return adata


# --- Fixture builders -------------------------------------------------------


def _write_varied_response_shard(
    path: Path,
    *,
    gene_cells: dict[str, int],
    n_tokens_per_cell: int,
    n_vocab: int,
    seed: int,
    barcode_prefix: str = "",
) -> None:
    """One shard with several genes, each with its own candidate cell count
    (so some genes exercise reservoir eviction and others do not), and each
    cell carrying ``n_tokens_per_cell`` distinct detected genes -- large
    enough per cell that the legacy Python-boxed-list construction's
    per-element overhead actually dominates. ``barcode_prefix`` keeps
    barcodes unique across multiple shards written by separate calls (a
    fixture spanning >1 shard must not accidentally collide on
    ``obs_names``, which would mask a real ordering bug behind an unrelated
    anndata "make unique" warning).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gene, n_cells in gene_cells.items():
        for cell_index in range(n_cells):
            tokens = rng.choice(n_vocab, size=n_tokens_per_cell, replace=False)
            tokens.sort()
            values = rng.uniform(0.1, 20.0, size=n_tokens_per_cell).astype(np.float32)
            rows.append(
                {
                    "gene_token_id": tokens.astype(np.int64),
                    "gene_expression": values,
                    "cell_barcode": f"{barcode_prefix}{gene}_{cell_index}",
                    "sample": "batch0",
                    "gene_target": gene,
                    "pass_guide_filter": 1,
                }
            )
    pd.DataFrame(rows).to_parquet(path)


def _write_vocab_metadata(path: Path, n_vocab: int) -> None:
    tokens = list(range(n_vocab))
    pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{token:011d}" for token in tokens],
            "gene_name": [f"GENE{token}" for token in tokens],
            "gene_token_id": tokens,
        }
    ).to_parquet(path)


# --- 1. Identity: new streaming path == frozen legacy pipeline -------------


def test_new_pipeline_matches_legacy_pipeline_exactly(tmp_path: Path) -> None:
    """The most important test in this diff (see module docstring).

    A moderately large, multi-gene fixture: some genes are under the cap
    (kept in full), some are well over it (forcing Algorithm-R eviction),
    genes span more than one shard, and each cell has enough detected genes
    that a column-position or value-summing mistake in the vectorized
    rewrite would show up as a numeric mismatch, not just a shape mismatch.
    """
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    gene_cells = {
        "GENE_A": 37,  # over cap
        "GENE_B": 4,  # under cap
        "GENE_C": 20,  # exactly at cap
        "GENE_D": 1,  # single cell
    }
    n_vocab = 60
    _write_varied_response_shard(
        shard_dir / "batch0_part0.parquet",
        gene_cells={k: v // 2 + (v % 2) for k, v in gene_cells.items()},
        n_tokens_per_cell=15,
        n_vocab=n_vocab,
        seed=1,
        barcode_prefix="p0_",
    )
    _write_varied_response_shard(
        shard_dir / "batch0_part1.parquet",
        gene_cells={k: v // 2 for k, v in gene_cells.items()},
        n_tokens_per_cell=15,
        n_vocab=n_vocab,
        seed=2,
        barcode_prefix="p1_",
    )
    metadata_path = tmp_path / "genes.parquet"
    _write_vocab_metadata(metadata_path, n_vocab)

    kwargs = dict(
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells_per_gene=20,
        seed=7,
    )
    legacy = _legacy_build_xatlas_orion_response_adata(
        shard_dir, metadata_path, **kwargs
    )
    current = build_xatlas_orion_response_adata(shard_dir, metadata_path, **kwargs)

    assert current.n_obs == legacy.n_obs
    assert current.n_vars == legacy.n_vars
    assert current.obs_names.tolist() == legacy.obs_names.tolist()
    assert current.var.index.tolist() == legacy.var.index.tolist()
    np.testing.assert_array_equal(
        current.obs["perturbation_gene"].to_numpy(),
        legacy.obs["perturbation_gene"].to_numpy(),
    )
    np.testing.assert_array_equal(
        current.obs["sample"].to_numpy(), legacy.obs["sample"].to_numpy()
    )
    np.testing.assert_array_equal(current.X.toarray(), legacy.X.toarray())
    # A gene really was capped, and a gene really was left alone -- otherwise
    # this test would pass trivially without ever exercising eviction.
    counts = current.obs["perturbation_gene"].value_counts()
    assert counts["GENE_A"] == 20
    assert counts["GENE_B"] == 4
    assert counts["GENE_D"] == 1


def test_new_pipeline_matches_legacy_pipeline_uncapped(tmp_path: Path) -> None:
    """Same identity guarantee with ``max_cells_per_gene=None`` (every cell
    kept) -- the other branch of the reservoir-vs-no-cap logic."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    gene_cells = {"GENE_A": 9, "GENE_B": 5}
    n_vocab = 20
    _write_varied_response_shard(
        shard_dir / "part0.parquet",
        gene_cells=gene_cells,
        n_tokens_per_cell=8,
        n_vocab=n_vocab,
        seed=3,
    )
    metadata_path = tmp_path / "genes.parquet"
    _write_vocab_metadata(metadata_path, n_vocab)

    kwargs = dict(
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells_per_gene=None,
    )
    legacy = _legacy_build_xatlas_orion_response_adata(
        shard_dir, metadata_path, **kwargs
    )
    current = build_xatlas_orion_response_adata(shard_dir, metadata_path, **kwargs)
    assert current.n_obs == 14
    assert current.obs_names.tolist() == legacy.obs_names.tolist()
    np.testing.assert_array_equal(current.X.toarray(), legacy.X.toarray())


# --- 2. Peak memory: materially lower, not just "runs" ---------------------


def _run_current_pipeline(
    shard_dir: str, metadata_path: str, max_cells_per_gene: int, seed: int
) -> None:
    build_xatlas_orion_response_adata(
        Path(shard_dir),
        Path(metadata_path),
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells_per_gene=max_cells_per_gene,
        seed=seed,
    )


def _run_legacy_pipeline(
    shard_dir: str, metadata_path: str, max_cells_per_gene: int, seed: int
) -> None:
    _legacy_build_xatlas_orion_response_adata(
        Path(shard_dir),
        Path(metadata_path),
        cell_line_name="HCT116",
        model_id="ACH-000971",
        cellosaurus_id="CVCL_0291",
        max_cells_per_gene=max_cells_per_gene,
        seed=seed,
    )


def _rss_measuring_worker(target, args: tuple, queue: multiprocessing.Queue) -> None:
    """Run ``target(*args)`` then report this process's peak RSS.

    Module-level (not a closure) so ``multiprocessing``'s ``spawn`` start
    method -- required for a clean, non-copy-on-write baseline -- can pickle
    it by reference.
    """
    target(*args)
    queue.put(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _peak_rss_kib(target, args: tuple) -> int:
    """Peak RSS (KiB) of ``target(*args)`` run in a FRESH spawned process.

    A fresh ``spawn`` (not ``fork``) child avoids any copy-on-write pages or
    allocator arenas already resident in the test process leaking into
    either measurement, and avoids one measurement's peak contaminating the
    other's baseline. ``ru_maxrss`` is already KiB on Linux; macOS reports
    bytes, so normalize.
    """
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=_rss_measuring_worker, args=(target, args, queue))
    process.start()
    peak = queue.get(timeout=120)
    process.join(timeout=120)
    return int(peak) if sys.platform != "darwin" else int(peak) // 1024


def test_peak_rss_materially_lower_than_legacy_pipeline(tmp_path: Path) -> None:
    """Real, measured peak-RSS comparison on a fixture large enough that the
    legacy pipeline's Python-boxed ``rows``/``columns``/``values`` triple
    lists dominate: 120 genes x 80 cells/gene, 1200 detected genes per cell
    (~11.5M nonzero entries total) -- large enough that boxed-Python-object
    overhead, not numpy array headers or fixed process/import overhead, is
    what differs between the two pipelines. Measured on this machine across
    several runs: current ~600-645 MB, legacy ~1270-1340 MB peak RSS (a
    ~48-53% reduction); the assertion below only requires >=15% to stay
    robust to interpreter/allocator noise across machines rather than
    over-fit to one measured ratio -- see the fix-round-3 report for the
    exact numbers this test produced.
    """
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    n_genes = 120
    gene_cells = {f"GENE_{i}": 80 for i in range(n_genes)}
    n_vocab = 2000
    _write_varied_response_shard(
        shard_dir / "part0.parquet",
        gene_cells=gene_cells,
        n_tokens_per_cell=1200,
        n_vocab=n_vocab,
        seed=11,
    )
    metadata_path = tmp_path / "genes.parquet"
    _write_vocab_metadata(metadata_path, n_vocab)

    shard_dir_str, metadata_path_str = str(shard_dir), str(metadata_path)
    current_peak = _peak_rss_kib(
        _run_current_pipeline, (shard_dir_str, metadata_path_str, 80, 5)
    )
    legacy_peak = _peak_rss_kib(
        _run_legacy_pipeline, (shard_dir_str, metadata_path_str, 80, 5)
    )

    assert current_peak > 0
    assert legacy_peak > 0
    # A real, material reduction -- not "it runs". The legacy pipeline pays
    # for the reservoir's real data a SECOND time as three Python-boxed
    # lists (rows/columns/values) sized to total nnz; the new pipeline
    # builds the same CSR matrix with vectorized numpy ops and drains the
    # reservoir as it goes.
    assert current_peak < legacy_peak * 0.85, (
        f"expected the new pipeline's peak RSS ({current_peak} KiB) to be "
        f"materially lower than the legacy pipeline's ({legacy_peak} KiB)"
    )


# --- resolve_total_budget_keep_mask ----------------------------------------


def test_keep_mask_none_keeps_everything() -> None:
    mask = resolve_total_budget_keep_mask(100, None, seed=0)
    assert mask.dtype == bool
    assert mask.all()
    assert mask.sum() == 100


def test_keep_mask_budget_above_n_cells_keeps_everything() -> None:
    mask = resolve_total_budget_keep_mask(10, total_cells=50, seed=0)
    assert mask.all()


def test_keep_mask_trims_to_exact_budget() -> None:
    mask = resolve_total_budget_keep_mask(1000, total_cells=100, seed=0)
    assert mask.sum() == 100


def test_keep_mask_deterministic_for_same_seed() -> None:
    first = resolve_total_budget_keep_mask(500, total_cells=50, seed=3)
    second = resolve_total_budget_keep_mask(500, total_cells=50, seed=3)
    np.testing.assert_array_equal(first, second)


def test_keep_mask_different_seed_changes_selection() -> None:
    first = resolve_total_budget_keep_mask(500, total_cells=50, seed=3)
    second = resolve_total_budget_keep_mask(500, total_cells=50, seed=99)
    assert not np.array_equal(first, second)


# --- drain_gene_reservoirs_to_matrix: total-cell budget --------------------


def _reservoir_cell(value: float, barcode: str) -> tuple:
    return (
        np.array([0], dtype=np.int64),
        np.array([value], dtype=np.float32),
        barcode,
        "s",
    )


def test_drain_applies_total_cell_budget_after_per_gene_cap() -> None:
    """Direct unit coverage of the total-budget knob at the level this
    module owns it (the public builder's own coverage lives in
    test_tx1_basal.py alongside its existing per-gene-cap tests)."""
    reservoirs = {
        "GENE_A": [_reservoir_cell(1.0, f"a{i}") for i in range(6)],
        "GENE_B": [_reservoir_cell(2.0, f"b{i}") for i in range(6)],
    }
    metadata = pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0])
    matrix, var, genes, barcodes, samples = drain_gene_reservoirs_to_matrix(
        reservoirs,
        metadata,
        metadata_var_columns=("ensembl_id", "gene_name"),
        total_cells=5,
        seed=0,
    )
    assert matrix.shape[0] == 5
    assert len(genes) == 5
    assert len(barcodes) == 5
    assert len(samples) == 5


def test_drain_total_cells_none_keeps_all() -> None:
    reservoirs = {
        "GENE_A": [_reservoir_cell(1.0, f"a{i}") for i in range(4)],
    }
    metadata = pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0])
    matrix, _var, genes, _barcodes, _samples = drain_gene_reservoirs_to_matrix(
        reservoirs,
        metadata,
        metadata_var_columns=("ensembl_id", "gene_name"),
        total_cells=None,
    )
    assert matrix.shape[0] == 4
    assert len(genes) == 4


def test_drain_drains_reservoir_slots_to_none() -> None:
    """Every consumed slot is nulled, whether or not it survives the total
    budget -- proves the draining actually happens, not just that the
    output is correct."""
    reservoirs = {"GENE_A": [_reservoir_cell(1.0, "a")]}
    drain_gene_reservoirs_to_matrix(
        reservoirs,
        pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0]),
        metadata_var_columns=("ensembl_id", "gene_name"),
    )
    assert reservoirs["GENE_A"][0] is None


def test_drain_matches_dict_lookup_for_tokens_missing_from_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cell referencing a token absent from ``metadata`` drops that entry
    (not the whole cell), exactly like the legacy dict-based lookup, and
    logs the same warning shape."""
    reservoirs = {
        "GENE_A": [
            (
                np.array([0, 1, 2], dtype=np.int64),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "a",
                "s",
            )
        ],
    }
    metadata = pd.DataFrame({"ensembl_id": ["ENSG0"], "gene_name": ["G0"]}, index=[0])
    with caplog.at_level("WARNING", logger="aivc_model.tx1_response_streaming"):
        matrix, var, _genes, _barcodes, _samples = drain_gene_reservoirs_to_matrix(
            reservoirs, metadata, metadata_var_columns=("ensembl_id", "gene_name")
        )
    assert matrix.shape == (1, 1)
    assert matrix.toarray().tolist() == [[1.0]]
    assert var.index.tolist() == ["ENSG0"]
    warnings = [
        record.message
        for record in caplog.records
        if "missing from gene metadata index" in record.message
    ]
    assert len(warnings) == 1
    assert "2/3" in warnings[0]
